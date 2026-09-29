# DAgger recovery-timing V2 source capsule and adapter — design revision 2

## Status and approval boundary

- Cycle: `DISCOVER → HYPOTHESIZE → DESIGN_PENDING_APPROVAL`.
- Revision: `2`; predecessor: [revision 1](dagger-obstacle-recovery-timing-v2-adapter.md), SHA-256 `6D8F657C07E5C867E7EBBFE11B2F28867462286896DFAD644B38BCCD4E22EFE1`.
- The user approved the revision-1 design and then approved creating this scope-revision design. Revision-1 implementation did not occur. Its approvals do not authorize revision-2 implementation or execution.
- This document's SHA-256 is calculated over its exact saved bytes and recorded in the approval message; it is not embedded here.
- Scope is a local adapter for the preserved 144-cell DAgger tail5 confirmation. It is a separate internal diagnostic effort, not one of the four candidate directions in the current completion-first batch.
- No code, protocol, checkpoint, run plan, or experiment output is changed by this design document.

## Hypothesis registration

- `hypothesis_id`: `H-DAgger-tail5-v2-source-capsule`.
- `source_ref`: preserved snapshot commit `81a759bf8102d7fddbfcad46dba5c60d4dfef000`, ref `refs/codex/snapshots/3693b7378da3c32fedc142e117d8faeb74e8fd91`; predecessor design hash above; preserved confirmation protocol's expected Windows-byte SHA-256 `19b9da2789736f192104425c521c4c5c6567dd3fd0fce4024597b3679cc7c645`.
- `rule_or_requirement`: register the adapter as a typed, local V2 profile; default to plan-only; bind design, implementation, and execution approvals separately to the exact revision/hash; write run records only below `runs/haic-research-v2/<run-id>/` and experiment artifacts only below `artifacts/haic-research-v2/<run-id>/`. Preserve legacy runs, artifacts, checkpoints, and submissions in place.
- `observable_information`: the frozen protocol specifies six actors, eight geometries, three conditions (`control`, `teacher_steer`, `teacher_steer_tail5`), one episode per actor/geometry/condition, and 144 total cells with action/controller traces.
- `allowed_action_or_state_change`: in a future separately approved implementation, place versioned DAgger source modules at the repository-root `training/` level under V2-prefixed filenames, add an immutable source-protocol copy plus a new V2 protocol path, register one local diagnostic profile, and route freeze/cell/receipt output below the profile's V2 artifact directory. Keep all experimental conditions, order, denominators, checkpoint identities, and decision budgets fixed.
- `expected_success_endpoint`: the registered profile and V2 source capsule can be fingerprinted by the existing non-recursive V2 source inventory, and a later plan-only registration can bind the new V2 protocol bytes and output destination without launching an episode. This endpoint is infrastructure readiness, not a performance result.
- `eligible_state`: local diagnostic evidence only. It cannot enter candidate completion rates, promote SOTA, establish official score, or authorize packaging/submission.
- `control`: preserve the protocol's matched `control`, immediate `teacher_steer`, and `teacher_steer_tail5` arms across the same six actors and eight geometry seeds.
- `falsifier`: reject this design if copied modules retain imports that resolve to missing historical names, any condition/order/checkpoint/seed/denominator changes, the shared evaluator or official harness must be modified, source hashes cannot bind every declared input, the profile is not explicitly SOTA-ineligible, or any output can escape the V2 roots. Missing source inputs or checkpoint bytes leave execution blocked; they do not authorize substitutes or retries.
- `smallest_decisive_experiment`: after implementation approval and source recovery, perform only a plan-only profile registration and inspect its immutable plan for the V2 protocol hash, source-module inventory, and V2 output destination. No simulator episode is part of that check. Actual 144-cell evaluation requires its own fresh plan and execution approval.
- `resource_and_risk_gate`: this design consumes zero training steps, simulator episodes, checkpoints, or run budget. Any later evaluation remains limited to the protocol's CPU, one thread, deterministic inference, 2,000 decisions per episode, 4.5-second planner budget, 5,400-second maximum, and 144 cells. No cell replacement or retry. Current missing checkpoint files block any run.

## Evidence and unknowns

### Facts

- The preserved snapshot contains the 48,135-byte confirmation runner, six directly or transitively imported DAgger helper modules, and the 12,325-byte V1 protocol. Those seven Python files and the protocol are absent from the active checkout at their original paths.
- The six helper modules are `run_dagger_obstacle_override_confirmation`, `run_dagger_preppo_screen`, `run_dagger_probability_screen`, `run_dagger_obstacle_side_audit`, `run_dagger_obstacle_recovery_timing_screen_v2`, and `run_dagger_obstacle_override_screen`. The override-screen import was not included in the first adapter estimate; this revision includes it.
- The V2 command source inventory includes only non-recursive `.py` files directly inside `training/`. A new nested package would not be fully fingerprinted without changing the shared command layer. This design therefore uses root-level filenames prefixed `dagger_recovery_timing_v2_` and keeps `haic_research/commands.py` unchanged.
- The active `training/evaluate_closed_loop.py` is modified in the shared working tree (158 added and 21 removed lines); the preserved runner imports its `run_episode`. Its compatibility with the preserved run's expected behavior has not been established.
- The preserved snapshot metadata audit matched 14 of 16 declared hashes. `docs/context/current-state.md` and `docs/evaluation/generalization-policy.md` did not match; the latter is absent from the active checkout.
- All six protocol checkpoint paths are absent from the active checkout and the preserved source snapshot. The earlier exact-path check did not scan other legacy run directories.

### Inference

- A thin wrapper cannot run: it would import historical modules that are not present in the active checkout. At least seven versioned Python modules and a protocol path must be supplied before the profile can be planned.
- Restoring those files alone does not make the experiment runnable. Metadata identity, shared-evaluator compatibility, checkpoint hashes, and the exact V2 protocol still need resolution.
- The revision-1 plan's “five DAgger modules” count was incomplete. The archived import graph requires six helpers plus the runner.

### Unknown

- Whether the current shared `run_episode` behavior is semantically equivalent for this protocol.
- Whether the six checkpoint byte streams can be recovered from an authorized exact path, and whether their producer hashes match the protocol.
- Whether the 16-path metadata audit can be re-established from current authorized inputs without opening held-out or blind material.

### Recommendation

- Use an isolated V2 source capsule with version-prefixed root-level modules. Rewrite only imports among the copied modules to their V2-prefixed siblings. Keep shared lower-level dependencies read-only, hash-bound, and subject to a pre-execution compatibility gate.
- Copy the archived V1 protocol verbatim to `experiments/v2/dagger-recovery-timing/source-v1-protocol.json` for provenance. Create `experiments/v2/dagger-recovery-timing/protocol-v2.json` only after exact source, metadata, and checkpoint hashes are available; preserve every registered cell and intervention rule while changing only the V2 identity, source-hash manifest, and output/provenance paths.
- Never copy or rewrite checkpoint bytes or historical run outputs. Read them only through explicitly declared protocol references if they become available.

## Proposed implementation scope, pending fresh approval

### Add from the preserved snapshot, with only import-name and V2 path changes

- `training/dagger_recovery_timing_v2_adapter.py` — profile entry point and output-path adapter, based on the preserved 144-cell runner.
- `training/dagger_recovery_timing_v2_override_confirmation.py`.
- `training/dagger_recovery_timing_v2_preppo_screen.py`.
- `training/dagger_recovery_timing_v2_probability_screen.py`.
- `training/dagger_recovery_timing_v2_side_audit.py`.
- `training/dagger_recovery_timing_v2_timing_screen.py`.
- `training/dagger_recovery_timing_v2_override_screen.py`.
- `experiments/v2/dagger-recovery-timing/source-v1-protocol.json` — byte-preserving source copy only.
- `experiments/v2/dagger-recovery-timing/protocol-v2.json` — only after all explicit source/input identities are available and audited.

### Register and document

- Add profile `dagger_obstacle_recovery_timing_confirmation_v2`, module `training.dagger_recovery_timing_v2_adapter`, arguments `protocol` (required input file) and `output` (run-specific output directory), diagnostic-only and SOTA-ineligible.
- Update only the profile/schema allowlists, exact-profile count, and the diagnostic profile's explicit `sota_eligible=false` validation in `haic_research/config.py`, plus the matching profile entry in `harness.config.json`. Do not change workflow transitions or approval semantics.
- Update `README.md` and `PROJECT_INFO.md` with the profile's plan-only interface and V2 output layout.
- Keep the generic dispatcher, shared evaluator, official competition harness, policy, packaging path, checkpoint files, `runs/`, `artifacts/haic/`, and `submissions/` unchanged.

## Output and source-binding contract

- The V2 profile's `--output` is a directory derived under `artifacts/haic-research-v2/<run-id>/`. The adapter places the protocol freeze, 144 immutable cell files, and final receipt below that directory; it does not write to the historical V1 `runs/` path.
- The V2 CLI hashes the explicit protocol input and its top-level `training/*.py` executable-source inventory into the run plan. The V2 protocol must carry the expected SHA-256 values for its explicitly named nested inputs; the adapter verifies those exact paths before the first cell and rechecks frozen inputs during the attempt.
- The existing V1 JSON remains untouched. The V2 protocol's `source-v1-protocol` reference and digest preserve provenance. The expected source manifest must not claim a checkpoint identity until its exact bytes and producer record are available.
- The adapter's run command creates its freeze, then consumes the one permitted attempt. It may not silently turn a missing input into a replacement, retry, or partial ranking.

## Explicit non-goals and stop conditions

- No DAgger training, checkpoint regeneration, checkpoint copying, or training-data import.
- No change to the shared evaluator or `haic_research/commands.py`; no official competition code or external action.
- No modification of the four-direction completion-first batch design, its candidate ranking, or its approvals.
- No `run_manifest.json`, `events.jsonl`, integration report, protocol freeze, cell output, simulation, package, or submission is created in this design stage.
- Implementation begins only after this revision receives design approval and a separate implementation approval for this exact revision/hash. A future run plan needs its own fresh design/implementation/execution approvals after all input identities are recoverable.
- If shared-evaluator compatibility, metadata hashes, or any checkpoint identity cannot be resolved without widening scope again, stop and issue another design revision rather than patching around the blocker.
