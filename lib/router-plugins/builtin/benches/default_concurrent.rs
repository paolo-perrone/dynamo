// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admission throughput of the shared core under concurrent callers.
//!
//! This is the bench for a change to `SelectionCore`, the scheduler queue
//! actor, or the default policy. Many tasks select, reserve, and release at
//! once through the production core, the default policy, and the frontend's
//! concurrent indexer, so the per-scheduler admission actor is the serial point,
//! as in the frontend. Criterion's elements per second is admitted requests per
//! second.
//!
//! Each worker count (256, 1024, 2048) runs with KV fetch credit off or 0.5,
//! and with queueing off (the default), on with requests rarely waiting
//! (threshold 1.0), or on with most requests parked (threshold 0.1).
//!
//! To A/B a change:
//! - Build base and branch with separate `CARGO_TARGET_DIR`s.
//! - Pin each run to the same performance cores, e.g.
//!   `taskset -c 0-7 cargo bench -p dynamo-custom-policy-builtin --bench default_concurrent`.
//! - Alternate base and branch processes for at least three rounds and compare
//!   medians per cell. Report the cells the change affects, plus the queue-off,
//!   credit-off cell as a control.
//!
//! Environment knobs: `BENCH_CALLERS`, `BENCH_HOLD_MS` with
//! `BENCH_MIN_PER_CALLER`, `BENCH_OVERLOADED`, and `BENCH_DYN_LOGGING`; see
//! each below. State the ones you set.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dynamo_custom_policy_builtin::default_factory;
use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{
    BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, StorageTier, compute_block_hash_for_seq,
    compute_seq_hash_for_block,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::selection::{
    HostLoad, PromptRequest, SelectAndReserveRequest, SelectionCacheConfig, SelectionCore,
    SelectionHost, WorkerRequest,
};

const BLOCK_SIZE: u32 = 16;
const SHARED_PREFIX_TOKENS: u32 = 2048;
const UNIQUE_SUFFIX_TOKENS: u32 = 2048;
const RUNTIME_THREADS: usize = 8;
/// The frontend's default `--router-event-threads`: a concurrent indexer whose
/// lookups run on the caller.
const INDEXER_THREADS: usize = 4;

/// Concurrent select-reserve-free loops; `BENCH_CALLERS` overrides the default of 64.
fn callers() -> u64 {
    std::env::var("BENCH_CALLERS")
        .ok()
        .and_then(|callers| callers.parse().ok())
        .unwrap_or(64)
}

/// Admissions run so far, printed at exit so profiles can be normalized per admission.
static ADMISSIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `BENCH_HOLD_MS` holds each reservation before freeing it, so ranks stay occupied.
fn hold() -> Duration {
    Duration::from_millis(
        std::env::var("BENCH_HOLD_MS")
            .ok()
            .and_then(|n| n.parse().ok())
            .unwrap_or(0),
    )
}

/// `BENCH_OVERLOADED=n` marks workers 1..=n overloaded, as the runtime client reports them.
fn overloaded() -> u64 {
    std::env::var("BENCH_OVERLOADED")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(0)
}

/// Every request shares the first 2k tokens and has a unique 2k-token suffix.
fn prompt(sequence: u64) -> Vec<u32> {
    let suffix_start =
        SHARED_PREFIX_TOKENS.wrapping_add((sequence as u32).wrapping_mul(UNIQUE_SUFFIX_TOKENS));
    (0..SHARED_PREFIX_TOKENS)
        .chain((0..UNIQUE_SUFFIX_TOKENS).map(|offset| suffix_start.wrapping_add(offset)))
        .collect()
}

fn reserve_request(selection_id: String, sequence: u64) -> SelectAndReserveRequest {
    SelectAndReserveRequest {
        model_name: "default".to_string(),
        routing_group: "default".to_string(),
        selection_id: Some(selection_id),
        prompt: PromptRequest {
            token_ids: Some(prompt(sequence)),
            ..PromptRequest::default()
        },
        router_config_override: None,
        expected_output_tokens: None,
        priority_jump: None,
        strict_priority: None,
        session_id: None,
        session_context: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: Default::default(),
    }
}

/// Store the first `blocks` blocks of the shared prefix on `worker_id`.
async fn seed_shared_prefix(core: &SelectionCore, worker_id: u64, blocks: usize) {
    let shared: Vec<u32> = (0..SHARED_PREFIX_TOKENS).collect();
    let local_hashes = compute_block_hash_for_seq(&shared, BLOCK_SIZE, BlockHashOptions::default());
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    let blocks = local_hashes
        .iter()
        .zip(sequence_hashes.iter())
        .take(blocks)
        .map(|(&tokens_hash, &sequence_hash)| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(sequence_hash),
            tokens_hash,
            mm_extra_info: None,
        })
        .collect();
    let key = RoutingPartitionId::new("default", "default");
    let indexer = core.partition(&key).expect("partition").indexer().clone();
    indexer
        .apply_event_routed(RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: 0,
            },
            StorageTier::Device,
        ))
        .await
        .expect("seed index");
    if let Indexer::Concurrent { primary, .. } = &indexer {
        primary.flush_and_wait().await.expect("flush index");
    }
}

/// Workers hold different-length prefixes of the shared prompt, so every row
/// has cache inputs. With `hints`, every worker is also a fetch source.
fn core_with_workers(
    runtime: &tokio::runtime::Runtime,
    workers: u64,
    hints: bool,
    shared_cache_multiplier: f64,
    router_queue_threshold: Option<f64>,
) -> Arc<SelectionCore> {
    let _guard = runtime.enter();
    let overloaded_workers = (overloaded() > 0).then(|| {
        let set: Arc<HashSet<u64>> = Arc::new((1..=overloaded()).collect());
        Arc::new(move || Some(Arc::clone(&set))) as _
    });
    let core = SelectionCore::bench_try_new_local_with_host(
        KvRouterConfig {
            use_kv_events: true,
            router_queue_threshold,
            shared_cache_multiplier: Some(shared_cache_multiplier),
            ..Default::default()
        },
        INDEXER_THREADS,
        Default::default(),
        SelectionCacheConfig::default(),
        default_factory(),
        SelectionHost {
            load: HostLoad {
                overloaded_workers,
                ..HostLoad::default()
            },
            ..SelectionHost::default()
        },
    )
    .expect("core");
    let prefix_blocks = (SHARED_PREFIX_TOKENS / BLOCK_SIZE) as usize;
    runtime.block_on(async {
        for worker_id in 1..=workers {
            let mut request = WorkerRequest {
                worker_id,
                endpoint: Some(format!("http://worker-{worker_id}:8000")),
                kv_events_endpoint: Some(format!("tcp://127.0.0.1:{}", 40_000 + worker_id)),
                block_size: Some(BLOCK_SIZE),
                max_num_batched_tokens: Some(8192),
                ..WorkerRequest::default()
            };
            if hints {
                request.router_hint_worker_type = Some("decode".to_string());
                request.router_hint_source_control_endpoints =
                    HashMap::from([(0, format!("tcp://worker-{worker_id}:9000"))]);
            }
            core.upsert_worker(request).await.expect("upsert");
        }
        for worker_id in 1..=workers {
            let blocks = worker_id as usize % prefix_blocks + 1;
            seed_shared_prefix(&core, worker_id, blocks).await;
        }
    });
    Arc::new(core)
}

/// Run `iters` select-reserve-free cycles split across `callers` tasks.
fn run_concurrent(
    runtime: &tokio::runtime::Runtime,
    core: &Arc<SelectionCore>,
    next_sequence: &mut u64,
    iters: u64,
    callers: u64,
) -> Duration {
    // `BENCH_MIN_PER_CALLER` keeps many callers looping long enough to reach a steady state.
    let min_per_caller = std::env::var("BENCH_MIN_PER_CALLER")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1);
    let per_caller = iters.div_ceil(callers).max(min_per_caller);
    let base = *next_sequence;
    *next_sequence += per_caller * callers;
    ADMISSIONS.fetch_add(per_caller * callers, std::sync::atomic::Ordering::Relaxed);
    let elapsed = runtime.block_on(async {
        let start = Instant::now();
        let tasks: Vec<_> = (0..callers)
            .map(|caller| {
                let core = Arc::clone(core);
                tokio::spawn(async move {
                    let hold = hold();
                    for index in 0..per_caller {
                        let sequence = base + caller * per_caller + index;
                        let id = format!("bench-{sequence}");
                        core.select_and_reserve(reserve_request(id.clone(), sequence))
                            .await
                            .expect("reserve");
                        if !hold.is_zero() {
                            tokio::time::sleep(hold).await;
                        }
                        core.free_reservation(&id).await.expect("free");
                    }
                })
            })
            .collect();
        for task in tasks {
            task.await.expect("caller");
        }
        start.elapsed()
    });
    // Callers run whole cycles; scale back to the `iters` criterion asked for.
    elapsed.mul_f64(iters as f64 / (per_caller * callers) as f64)
}

fn bench(c: &mut Criterion) {
    // `BENCH_DYN_LOGGING=1` installs the frontend's logging (`DYN_LOG` filters apply).
    if std::env::var("BENCH_DYN_LOGGING").is_ok_and(|value| value == "1") {
        dynamo_runtime::logging::init();
    }
    let callers = callers();
    let mut group = c.benchmark_group("default_concurrent/select_reserve_free");
    group
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sample_size(10)
        .throughput(Throughput::Elements(1));
    for &workers in &[256u64, 1024, 2048] {
        for (name, hints, multiplier, queue_threshold) in [
            ("credit0", false, 0.0, None),
            ("credit0.5", true, 0.5, None),
            // Every admission passes the busy gate, but workers rarely fill, so
            // requests rarely wait.
            ("queue1.0_credit0", false, 0.0, Some(1.0)),
            ("queue1.0_credit0.5", true, 0.5, Some(1.0)),
            // One active request fills a rank, so with more callers than ranks the
            // rest park and every gate check scans for a freed rank.
            ("queue0.1_credit0", false, 0.0, Some(0.1)),
        ] {
            // One runtime per core: dropping it stops the per-worker listener tasks.
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(RUNTIME_THREADS)
                .enable_all()
                .build()
                .expect("runtime");
            let core = core_with_workers(&runtime, workers, hints, multiplier, queue_threshold);
            let mut next_sequence = 0u64;
            group.bench_with_input(BenchmarkId::new(name, workers), &core, |b, core| {
                b.iter_custom(|iters| {
                    run_concurrent(&runtime, core, &mut next_sequence, iters, callers)
                });
            });
            core.shutdown();
        }
    }
    group.finish();
    eprintln!(
        "admissions={}",
        ADMISSIONS.load(std::sync::atomic::Ordering::Relaxed)
    );
}

criterion_group!(benches, bench);
criterion_main!(benches);
