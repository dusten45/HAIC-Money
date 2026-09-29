# Run Experiment Workflow

## Before Running

1. Read `talk/README.md` and the latest relevant messages before deciding whether
   prior work or a shared-file conflict affects this experiment.
2. Locate the active plan and relevant prior experiment records.
3. State the hypothesis, control, treatment, scope, expected failure signal, and
   acceptance/rejection/stop criteria.
4. Predeclare source revision, command, dependency/runtime, environment contract,
   training exclusions, budget, and evaluation partitions for the protocol.
5. Consult `docs/evaluation/generalization-policy.md`, then audit **all relevant**
   frozen protocols, run receipts, source training ledgers, and retired allocations
   across research lanes before declaring any training or evaluation cell unused.
   Its selective history table alone cannot establish freshness. For fresh
   TRAIN-only roads, decide from candidate-specific use, active reservations,
   exposure, and partition exclusions, not unrelated repository file changes;
   preserve source-specific hashes and candidate-relevant unknowns as blockers.
   Reserve audited TRAIN roads under the shared claim lock and coordinate disjoint
   seed namespaces with parallel lanes. An audit receipt alone is not a claim.
6. Freeze the protocol with the claimed cells, claim digest, source/runtime hashes,
   exclusions, and fixed gates. Recheck with an authenticated self-claim mode
   immediately before the first reset; without one, a prior claim correctly
   appears as an ordinary collision. Preserve
   partial-run ledgers; do not silently recycle retired allocations. Confirmation/
   blind rules remain stricter and unchanged.

## Local Training Resources

Official HAIC CPU inference limits apply to the **submitted Agent**, not to
free RAM, disk, GPU memory, CPU cores, or elapsed time required before local
training. The [current Participants README](https://github.com/2026-HAIC/Participants)
allows participants to configure their training environment. Numeric floors
in old experiment protocols are source-frozen decisions for those attempts,
not universal limits, official rules, or defaults to copy into a new run.
Never weaken a running study's pinned executable or protocol in place; use a
new source, protocol, run path, and tests for a future resource-policy change.

For a **new** study, record the measurement and forecast behind each hard
resource gate before the first reset:

- RAM: measure comparable peak process and checkpoint-serialization memory,
  then forecast *additional* memory needed at the next phase, concurrent-job
  growth, and a documented safety reserve. Record both raw cgroup
  `memory.max - memory.current` and host available memory. Clean inactive
  file cache may be reclaimable; record it separately from dirty/writeback
  pages, and never treat all cache or host availability as guaranteed cgroup
  headroom. Check OOM counters/pressure. Do not demand the full initial peak
  as free memory again after this job has already allocated part of it.
- Disk: estimate remaining immutable checkpoint writes, active replay/logs,
  temporary serialization and partial receipts, plus concurrent writers and
  reserve. Recompute **remaining** bytes after each sealed checkpoint rather
  than requiring the initial free-space floor forever. Preserve incomplete
  checkpoint evidence if a write fails; prefer atomic finalization where
  feasible.
- GPU: check the device's remaining capacity against measured peak reserved
  memory, CUDA context/driver use, upcoming batch demand and concurrent-job
  growth. `max_memory_allocated()` alone omits reservations. Another CUDA
  process is not by itself an insufficient-resource signal; exclusive use
  must have an explicit experiment-specific reason.
- CPU and wall time: bound threads and record throughput under actual
  contention. A busy CPU is a latency/cost warning, not a fabricated
  free-core quota. Choose a wall budget from measured throughput and study
  cost requirements; it is not an official local training limit.

Recheck near-term allocations before resets, update bursts and checkpoints.
Hard-stop only when the next operation or the remaining declared study cannot
fit its *documented* forecast with safety reserve, or when critical telemetry
is unavailable; otherwise warn and continue. Preserve reset intents, step
ledgers, completed checkpoint hashes and a conservative partial receipt if a
stop is needed. Never present a mid-episode partial as fresh or exactly
resumable. See the [resource-floor provenance note](../../talk/messages/20260929T022000Z-k3p7-resource-floor-origin-result.md)
for why earlier local TD-MPC2 attempts used fixed values.

## During Execution

- Change one core variable when a matched comparison is the goal.
- Preserve run IDs, checkpoint/model hashes, source/environment/protocol hashes,
  episode/seed records, and CPU operational receipts.
- Keep generated checkpoints, replay, and detailed logs in run artifacts; do not
  commit them merely to make a Markdown claim convenient.
- Stop on the plan's hard failures. Do not alter gates or select a different
  checkpoint after confirmation begins.
- Coordinate material overlap and post decision-relevant observations, results,
  or reproducible failures to `talk/` as separate, evidence-labeled messages.

## After Execution

1. Run the declared local/internal evaluation and compare the same candidate scope.
2. Record outcome counts, proxy metrics, operational results, limitations, and the
   distinction between observations and hypotheses in the result artifact.
3. Post a concise `result` or `failure` message to `talk/` when other agents can
   use or challenge the evidence; link the artifact and reproducibility details.
4. Add only decision-relevant studies to `docs/experiments/INDEX.md`.
5. Add a decision record when the evidence narrows or closes future work.
6. Move/close the plan only after the stated evidence exists.

An experiment result does not itself authorize official submission or model
confirmation.
