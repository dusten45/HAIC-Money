# Battlecode-Inspired Agent Research and Evaluation Harness (Draft)

**Status:** Tournament runner and harness path are implemented. This document is still not an approved experiment plan or execution approval. No training or evaluation has been run.

## Format

Use the Battlecode-inspired workflow to assign independent candidate directions to agent workers, have a central coordinator reconcile the handoffs, and rank eligible candidates under one fixed protocol. Each candidate changes one registered mechanism and is compared with the same control policy on the same fixed `(map, seed)` episode cells. HAIC's driving environment remains single-agent, so the evaluator runs policies sequentially; the Battlecode inspiration is the team-based research and repeatable ranking process, not simultaneous gameplay.

The following four directions are the current **proposed** batch, subject to design approval and activation checks:

1. `HZ-PERCEPT-01`: add a pixel-derived signed lateral offset between visible obstacle and vehicle.
2. `HZ-PERCEPT-02`: add a clipped temporal cue for visible obstacle approach rate.
3. `PPO-PEDAL-RANGE-COMPLETION-01`: expand the throttle action range after proving the control saturates in eligible training states.
4. `PPO-CURVE-BRAKE-01`: add the existing curve-conditioned brake reward; control and candidate use the same visual-feature architecture, with coefficient 0 in control.

`CEM-UNCERTAINTY-FALLBACK-01` is deferred: it needs a dynamics checkpoint, selected-plan uncertainty plumbing, and train-only calibration, and the current registered harness has no dynamics-training profile. Curve braking is a lower-friction alternative, not a proven stronger method. HZ1 and HZ2 remain separate hypotheses with distinct direct spatial and temporal cues; their shared visual-feature dependency must be held constant across arms.

One shared control policy is required. For the curve-brake comparison it must be trained with visual features enabled and the brake-reward coefficient set to zero, so the candidate measures the additional reward rather than an architecture change. Existing `CURVE_SPEED_RELIEF` already changes the target-speed shortfall shaping on curves; the experiment measures the extra brake-reward effect on top of that behavior. The exact checkpoint and hash have not been selected. Every arm must use the same starting weights where compatible, train maps, optimizer settings, training seeds, and step budget. If one candidate cannot meet its activation condition using the registered train-only material, replace or revise it before registering the batch; do not run a three-direction batch.

## Scoring

The primary score is completion rate:

`completed episodes / all registered attempts`

All five arms use the same registered episode cells and denominator. Candidate setup/reset/action failures, returned-call timeouts, invalid actions, and DNF results are non-completions; they are not removed from the denominator. Infrastructure errors invalidate the comparison rather than creating a candidate win. Rank completion rate first. Use the registered tie-break order only when completion rates match: median completed lap time, incomplete progress, P90 lap time, collisions, damage, and p95 action latency.

The harness supports matched candidate/control identity through the comparison ID, split, map set, seed set, and episode count. Promotion requires at least two evaluation seeds. A proposed practical floor is 10 held-out episodes across at least two seeds; this is an execution floor, not a power calculation or guarantee of generalization. The baseline completion rate and target effect size are unknown, so sample size has not been justified statistically.

## Data-consumption order

1. **Train-only activation checks:** confirm each feature or shaping mechanism actually reaches policy behavior. Check throttle saturation and curve-weighted brake shaping using train-only material.
2. **Tune tournament:** run the control and all four eligible candidates on the same registered tune cells. Do not adjust features, thresholds, or planner settings after seeing these results. If no candidate outranks control, do not open held-out data.
3. **Held-out confirmation:** compare only the tune winner and control, once, on the same registered held-out cells. Do not use confirmation or blind cells for this cycle.

Completion rate remains the first comparison criterion. `ADVANCE`, `REJECT`, `REVISE`, and `PIVOT` each enter their matching gate-review state. Only `ADVANCE` with all three gate results `PASS` can be released.

## Implemented tournament path

The registered `evaluate_closed_loop` profile now accepts `candidate-manifest` and `split-group`. Its intended path selects one tune or held-out group and runs control and candidates sequentially over the same ordered `(map_id, seed)` cells, persisting each episode row as it finishes. A Windows case-alias defect means a tune path and held-out path can still identify the same physical map; split isolation is not safe until canonical path comparison is fixed. The manifest fixes each candidate's mode, checkpoints, hypothesis document, and planner settings.

The runner ranks completion rate first, then the documented tie-breakers. Malformed, out-of-range, or non-finite actions are DNF in tournament mode. Candidate setup/reset/action failures remain attempts in the fixed denominator; missing measurements from a candidate failure receive a conservative tie-break penalty. Environment/evaluation faults, incomplete cell coverage, mismatched returned map/seed identity, invalid telemetry, or changed input bytes invalidate the comparison. A tune winner must strictly outrank control; ties do not open held-out. Current source calls `_verify_frozen_tune` before loading selected map payloads; it checks the approved completed tune run, plan and summary hashes, candidate/control/runtime identities, ranked status, and actual winner ID/rank. Argument-pair validation precedes split traversal, and split map paths are compared with Windows case normalization. These guards exist in the dirty working tree but were not behavior-tested in this review, so their source presence is not a test pass.

The harness plan hashes the tournament manifest, candidate and control checkpoints, hypothesis references, the full split-manifest bytes, selected map payload bytes, and current executable/harness sources. At runtime the evaluator requires the active approved harness plan and rechecks its inputs before and after scoring. Every map/seed cell is run as a round; arm order rotates across rounds, and each episode row is flushed immediately. A returned call over five seconds is recorded as DNF, but the runner cannot interrupt a call that never returns. The outer harness timeout then fails the whole run; earlier rows remain for diagnosis, but there is no final ranking.

Candidate manifests use schema version 2. A held-out manifest has a single candidate and `tune_source` fields `run_id`, `plan_hash`, `summary_ref`, and `summary_sha256`. Runtime checks these references against the approved tune summary and requires that candidate to be its recorded winner. The manifest does not create a run approval. The separate HAIC gates still decide whether any result may be released.

The internal local evaluator is not the HAIC official ranking system. UNSW Battlecode uses ranked five-game battles on randomly chosen maps and a team ELO ladder ([game format](https://game.battlecode.au/docs/game-format)); here the simulator has one driving agent, so the analogous useful unit is a matched multi-candidate series on fixed HAIC map/seed cells. The comparison is local research evidence, not an official HAIC score.

## Still required before the tournament

No runnable candidate batch has been selected. The proposed directions are research hypotheses, not four implemented and trained candidate checkpoints. The chosen control checkpoint, candidate checkpoints, exact maps/seeds, plan budget, decision limit, run IDs, resource limits, and approval sources are also not registered. The current source contains split and tune-winner guards, but their behavior remains unverified. Candidate train flags for visual/temporal features, initialization, throttle expansion, and curve-brake reward are not yet in the registered `train_policy` allowlist.

Before tune, finish a fixed batch design, register the manifest and exact split, and complete design, implementation, and execution approvals through the local harness. Run authorized verification of the current split and winner-binding guards before relying on them. After tune, only the verified strict winner can receive a held-out manifest, which needs its own plan and approval cycle. The held-out floor remains 10 attempts across at least two seeds; it is not a power calculation or a guarantee of generalization.

## Open fields before design approval

- Control PPO checkpoint path and SHA-256 identity.
- Exact train, tune, and held-out maps; episode seeds and episode count.
- Training seeds, optimizer settings, training budget, timeout, and compute limit.
- HZ1 offset formula and normalization; HZ2 frame matching, interval, and clipping rule.
- HZ3 throttle expansion value and train-only saturation threshold.
- Curve-brake reward coefficient and train-only activation rule; or, if CEM is retained, its uncertainty measure and calibration rule.
- Behavior verification of split-argument validation, canonical path separation, and tune-winner binding.

Historical results are not auto-imported or used as controls; prior data remain at their original paths. Candidate comparisons are not simultaneous races. A hung individual call is not safely converted to a DNF; it fails the whole run at the outer harness timeout.
