# Battlecode-Style Local Completion Tournament (Draft)

**Status:** Design draft only. This is not an approved implementation plan or execution plan. No training or evaluation is authorized by this document.

## Format

Treat the four discovered directions as four independent candidate teams. Each candidate changes one registered mechanism and competes against the same control policy on the same fixed `(map, seed)` episode cells. This is a local multi-arm tournament; the driving environment remains single-agent, so candidates are ranked on matched episodes rather than placed as simultaneous opponents in one race.

The four candidate directions are:

1. `HZ-PERCEPT-01`: add a pixel-derived signed lateral offset between visible obstacle and vehicle.
2. `HZ-PERCEPT-02`: add a clipped temporal cue for visible obstacle approach rate.
3. `PPO-PEDAL-RANGE-COMPLETION-01`: expand the throttle action range after proving the control saturates in eligible training states.
4. `CEM-UNCERTAINTY-FALLBACK-01`: use the existing PPO action when a pre-registered high-uncertainty planning state occurs.

One shared control policy is required. The current proposal is a baseline PPO policy with a fixed dynamics checkpoint and fixed CEM settings. The exact control checkpoint, dynamics checkpoint, planner settings, and hashes have not been selected. Every arm must use the same starting weights where compatible, train maps, optimizer settings, training seeds, and step budget. If one candidate cannot meet its activation condition using the registered train-only material, replace or revise it before registering the batch; do not run a three-direction batch.

## Scoring

The primary score is completion rate:

`completed episodes / all registered attempts`

All five arms use the same registered episode cells and denominator. Timeouts, DNF, invalid actions, and evaluation failures count as non-completions; they are not removed from the denominator. Rank completion rate first. Use the registered tie-break order only when completion rates match: median completed lap time, incomplete progress, P90 lap time, collisions, damage, and p95 action latency.

The harness supports matched candidate/control identity through the comparison ID, split, map set, seed set, and episode count. Promotion requires at least two evaluation seeds. A proposed practical floor is 10 held-out episodes across at least two seeds; this is an execution floor, not a power calculation or guarantee of generalization. The baseline completion rate and target effect size are unknown, so sample size has not been justified statistically.

## Data-consumption order

1. **Train-only activation checks:** confirm each feature or fallback actually reaches policy behavior. Calibrate the CEM uncertainty threshold and check throttle saturation using train-only material.
2. **Tune tournament:** run the control and all four eligible candidates on the same registered tune cells. Do not adjust features, thresholds, or planner settings after seeing these results. If no candidate outranks control, do not open held-out data.
3. **Held-out confirmation:** compare only the tune winner and control, once, on the same registered held-out cells. Do not use confirmation or blind cells for this cycle.

Completion rate remains the first comparison criterion. `ADVANCE`, `REJECT`, `REVISE`, and `PIVOT` each enter their matching gate-review state. Only `ADVANCE` with all three gate results `PASS` can be released.

## Required implementation before the tournament

The current `evaluate_closed_loop` path chooses CEM settings from tune and then evaluates held-out in the same call. It does not expose a split-only mode with fixed planner settings. The current training profile also does not register all candidate-specific settings, and the CEM fallback candidate has no implementation or calibrated threshold. A tournament must not use the current combined evaluation call as if it were an isolated tune comparison.

The implementation plan must therefore register:

- split-specific evaluation profiles with fixed planner settings and a fixed candidate revision;
- the candidate feature or fallback settings as typed arguments or immutable source revisions;
- exact control and dynamics checkpoint identities;
- fixed train/tune/held-out map and seed lists, episode denominator, training seeds, steps, and resource limits;
- activation checks and the result-record mapping for completion rate and tie-break metrics;
- one plan hash for each control/candidate operation, with separate design, implementation, and execution approvals.

## Open fields before design approval

- Control PPO and dynamics checkpoint paths and SHA-256 identities.
- Exact train, tune, and held-out maps; episode seeds and episode count.
- Training seeds, optimizer settings, training budget, timeout, and compute limit.
- HZ1 offset formula and normalization; HZ2 frame matching, interval, and clipping rule.
- HZ3 throttle expansion value and train-only saturation threshold.
- HZ4 uncertainty measure and train-only calibration rule.
- The fixed CEM configuration and split-isolated evaluation implementation.

Until these fields are registered and reviewed, this remains a Battlecode-style tournament proposal, not a runnable batch. Historical results are not auto-imported or used as controls; prior data remain at their original paths.
