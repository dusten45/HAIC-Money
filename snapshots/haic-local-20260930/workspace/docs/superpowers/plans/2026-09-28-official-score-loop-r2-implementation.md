# Official Score Learning Loop r2 Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task after a separate implementation approval. Do not dispatch subagents without explicit user authorization. Checkboxes track work; no item below has been executed.

**Goal:** Make the four r2 pixel-policy mechanisms executable through approved, typed HAIC profiles, with a TRAIN activation gate before any TUNE evaluation.

**Architecture:** Keep official environment code unchanged. Add narrow candidate mechanisms to the existing pixel actor and TRAIN-only training path, then use one registered comparison adapter to measure each eligible candidate against a fresh control on identical cells. Teacher data remain TRAIN-only and cannot enter student training until the teacher preflight succeeds. Each run is registered and bound to source/input hashes by the v2 harness.

**Tech Stack:** Python 3.11, PyTorch CPU, existing `haic_research` v2 CLI and official `variables-6` local simulator.

**Spec:** `docs/plans/active/official-score-learning-loop-r2.md` at SHA-256 `eda9c0e0ab3584260d744182c4c7afa577c41837f1d9acf1b0be2f0030309f6c`.

## Global Constraints

- The official simulator and submission observation/action contract stay unchanged: `(4,84,84)` grayscale frames to finite `[steer,gas,brake]`.
- The first batch has exactly one candidate from each of four independent directions and one newly trained pixel-policy control. No old ZIP or checkpoint initializes the control.
- TRAIN/TUNE/HELD_OUT/CONFIRMATION/BLIND cell identities and denominators are the exact r2 table; audit collisions before registration. The first cycle is capped at 14 CPU hours and 2,000 decisions per episode.
- Candidate selection compares completion rate first on matched cells, then the ordered tie-breakers in `AGENTS.md`. Off-road time is diagnostic, not a separate selection penalty.
- No teacher/student imitation follows a failed or UNKNOWN TRAIN teacher preflight. No TUNE evaluation follows a failed mechanism activation.
- Every source edit needs its own implementation approval; every local run needs a registered plan and separate execution approval. Submission, model confirmation, and site upload each need immediate explicit authorization.
- Developer instruction for this task: do not add or run tests unless the user asks. Future implementation review can use source inspection; training/evaluation are separately approved operations, not tests in this plan.

## Review Focus

1. A split identity already consumed elsewhere must be rejected before a plan is registered.
2. A teacher trajectory with an invalid finish, less than 95% tile coverage, or fewer than 12 shared finishes must fail or remain UNKNOWN, never enter student imitation.
3. An activation failure must leave TUNE cells unopened.
4. A checkpoint architecture mismatch must stop the run without changing the original checkpoint.
5. A completed episode that cuts grass must be judged by valid finish and time; a 101-decision negative-reward retirement is a failure.

## Task 1: Freeze cell identities and register run inputs

**Files:** Create `docs/experiments/official-score-loop-r2-registry.md`; create a TRAIN-only site-map split under `training/maps/site/`; modify only the typed profile allowlist in `harness.config.json`, `haic_research/config.py`, and `haic_research/commands.py` as needed.

**Interface:** `(track_id, seed, variables-6)` is the cell identity. The registry lists every cell and its split, the common fresh control, four candidate IDs, model seeds, per-run CPU limits, denominator, activation rule, and the exact design SHA. The harness fingerprints the split file, referenced map files, executable sources, and all checkpoint inputs. No generic shell profile is added.

- [ ] Inspect explicit v2 experiment registry and current v2 run manifests for r2 cell collisions, without walking legacy roots.
- [ ] Materialize r2 TRAIN, TUNE, HELD_OUT, CONFIRMATION, and BLIND groups as exact map/seed identities; reject collisions and any cross-split duplicate.
- [ ] Add only the typed arguments needed by Tasks 2–6 to registered local profiles; retain plan-only default, source hashing, timeouts, and one-use execution approval.
- [ ] Check that plan preview names the correct split, denominator, source hashes, output root, and permission limits. This is a read-only review, not an execution.

## Task 2: Fresh control and speed-state candidate

**Files:** Modify `haic_agent/networks.py`, `training/train_policy.py`, and `training/ppo.py` only for the r2 speed-state mechanism; record candidate-specific options in the Task 1 profile.

**Interface:** `candidate_id=H-SCORE-R2-SPEED` adds a TRAIN-only supervised target for speed, yaw rate, and lateral slip to the pixel actor. Submission inference receives pixels only. `candidate_id=CONTROL-R2` uses the same architecture, initialization seed, action limits, TRAIN cells, and CPU cap with the mechanism disabled. Both record action gas/brake distributions, speed, slip, auxiliary loss, and policy-head change.

- [ ] Preserve the existing auxiliary target timing: each label must correspond to the pixels that selected its action.
- [ ] Add an explicit speed/slip supervision option with a fixed coefficient and source identity; do not tune a coefficient before activation.
- [ ] Record whether predicted state and actual pedal/brake timing change on TRAIN relative to the control.
- [ ] If the predictor improves but action timing does not change, mark activation FAIL and stop this candidate before TUNE.

## Task 3: Pixel-only short-horizon candidate

**Files:** Create `training/pixel_prediction.py`; modify `haic_agent/networks.py` and `training/train_policy.py` for an optional prediction head and action selector; extend the typed Task 1 profile.

**Interface:** `candidate_id=H-SCORE-R2-PREDICT` learns future pixel-derived road alignment and visible obstacle occupancy from TRAIN rollouts. At inference it uses only the last four grayscale frames and its learned state. The action selector uses a fixed short horizon; all choices and latency are logged.

- [ ] Define future targets from TRAIN pixels only and mask unavailable future frames at episode end.
- [ ] Register one fixed horizon and candidate budget; reject latency above the official `act()` limit.
- [ ] Compare TRAIN action timing with the control; no action change or no meaningful prediction signal is activation FAIL.
- [ ] Keep simulator geometry and obstacle truth out of inference input.

## Task 4: Teacher preflight and gated student

**Files:** Create `training/teacher_preflight_r2.py`; modify `training/imitation.py` and `training/train_policy.py` only if the preflight passes; add a typed TRAIN-only profile in Task 1.

**Interface:** `candidate_id=H-SCORE-R2-TEACHER` may use TRAIN simulator state to create demonstrations, but the student sees only pixels. Preflight reports all 48 TRAIN outcomes and paired lap differences against the fresh control on shared finishes. It must show at least 36/48 valid finishes, at least 12 shared finishes, and a strictly negative median teacher-minus-control finished time. The old V1–V5 teacher is evidence of failure, not automatically eligible demonstration data.

- [ ] Make the teacher preflight produce a frozen summary and individual cell records with the exact eligibility calculation.
- [ ] Block demonstration loading when any required threshold fails or official rule compliance remains UNKNOWN for a promotion decision.
- [ ] Only after a passing preflight, train one student from eligible TRAIN trajectories using pixel observations and the same CPU cap as control.
- [ ] If preflight fails, preserve records, stop this direction, and return to a new design revision before a four-direction comparison.

## Task 5: Completion-constrained time candidate

**Files:** Create `training/finish_time_objective.py`; modify `training/train_policy.py` and `training/ppo.py` for an optional episode-end objective; extend the typed Task 1 profile.

**Interface:** `candidate_id=H-SCORE-R2-TIME` changes TRAIN learning signal using valid finish and lap time. It never receives held-out, confirmation, blind, or official outcomes for learning. Episode failures retain a fixed completion constraint; road departure has no separate penalty beyond actual validity and time effects.

- [ ] Compute the objective from TRAIN episode terminal status, unique tile coverage, finish crossing, and simulated time.
- [ ] Log objective components separately from raw CarRacing reward and official score.
- [ ] Check TRAIN action and finish behavior against the fresh control; unchanged behavior is activation FAIL.
- [ ] Do not sweep objective weights before activation is observed.

## Task 6: Registered matched comparison and gate report

**Files:** Modify `training/evaluate_closed_loop.py`, `haic_research/commands.py`, `haic_research/config.py`, and `harness.config.json` for a pixel-policy-only comparison profile; create `docs/experiments/official-score-loop-r2-outcome.md` after approved runs.

**Interface:** The comparison consumes only frozen checkpoint hashes and the registered TUNE split. It writes one row per `(candidate, control, track_id, seed)` with valid completion, lap time, incomplete progress, collision/damage, off-road/negative streak diagnostics, action latency p50/p95/max, invalid actions, import/reset time, RSS, and run duration. It emits an `IntegrationReport` with exactly the three gates and evidence paths.

- [ ] Compare only activation-eligible candidates on all 12 registered TUNE cells; never select on a subset.
- [ ] Make the typed comparison profile accept explicit policy checkpoint hashes without requiring a dynamics checkpoint; the current `evaluate_closed_loop` profile requires one even for policy-only driving.
- [ ] Rank by completion rate first and the exact AGENTS tie-break order; report per-track official-style results separately.
- [ ] Freeze the TUNE winner before any separately approved HELD_OUT run. Use CONFIRMATION once on the frozen candidate/package; keep BLIND reserved.
- [ ] Record `ADVANCE`, `REJECT`, `REVISE`, or `PIVOT`, then the matching gate-review state. Allow release only for ADVANCE with all three gates PASS.

## Implementation and execution handoff

This plan does not authorize implementation. After the user approves this exact implementation plan, record that approval with the plan SHA and implement Tasks 1–6 within the approved scope. Then register exact local operation plans with the v2 CLI, review each hash and resource cap, and seek separate execution approval before any learning or simulator run. The teacher's official-rule uncertainty remains explicit and blocks its SOTA promotion unless resolved.
