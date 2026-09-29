# PPO Recovery-Clearance Reward Screen Implementation Plan

> Execute inline in the current root checkout. Keep PPO actor-only; do not request reviewers or subagents.

**Goal:** Test whether PPO learns to escape a post-impact obstacle stall when the reward follows actual reduction in visible hazard risk instead of merely rewarding gas or steering commands.

**Architecture:** Extend the existing transition reward with one optional coefficient. In damaged, low-speed, non-collision, on-track states with high current hazard risk, compare pixel-derived risk before and after the action and award only positive risk reduction. Thread next-observation features through rollout collection; all actor inputs and submitted inference remain unchanged.

**Tech Stack:** Python 3.11, PyTorch, NumPy, existing PPO trainer and custom-map screen runner.

**Spec:** `research/2026-09-23-lap13-strategy-lab.md` (2026-09-25 policy-mean LR result and registered recovery hypothesis)

## Global Constraints

- Submission runtime remains PPO actor-only; reward shaping runs only during training.
- Teacher warmup is zero; no privileged values are added to the actor observation or inference path.
- Starting actor remains `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`; fresh Adam per arm.
- Keep the registered custom split, seeds `8104/8105`, 8,192 decisions/U8 per arm, LR `2e-6`, and all existing reward terms fixed.
- Compare only `recovery_clearance_reward=0` versus `6`; do not load held-out or official maps during training or Tune selection.
- Tune seed labels repeat one custom geometry. Treat results as a mechanism screen, not map-diverse evidence.
- Do not update SOTA or replace/package the submission ZIP unless a separate official and map-diverse validation supports it.

## Files

- Modify `training/train_policy.py`: accept the coefficient and next-state visual features, compute positive hazard-risk reduction only in qualified recovery states, and pass/store the configuration.
- Modify `tests/test_train_policy.py`: exercise reward arithmetic, all recovery gates, validation, and rollout feature plumbing.
- Create a timestamped run directory under `artifacts/haic/`: freeze the run configuration, copied experiment runner, source manifest, per-arm checkpoints, TUNE episode traces, and summaries.
- Append the outcome and next hypothesis to `research/2026-09-23-lap13-strategy-lab.md`.

## Interfaces

- `shape_transition_reward(..., visual_features, next_visual_features, recovery_clearance_reward) -> float` receives features from the observation used for the action and the resulting observation.
- `collect_rollout(..., recovery_clearance_reward=0.0)` captures next-state visual features during the existing actor-critic pass over `transition.next_observation` and passes both feature vectors to the reward function.
- `train(..., recovery_clearance_reward=0.0)` validates and records the coefficient; the default preserves existing runs.

## Verification boundaries

- Positive risk drop earns reward only after existing damage, low-speed, non-collision, and on-track gates are met and current risk is at least `RECOVERY_MAX_HAZARD_RISK`.
- Risk increase or no change earns no clearance reward.
- Collision/off-track transitions earn no clearance reward, even if the next image has lower hazard risk.
- Missing visual features produce no clearance signal; malformed feature vectors follow the existing feature-shape validation.
- Negative, NaN, or infinite coefficients are rejected.

## Tasks

### Task 1: Specify the shaping behavior

**Files:** `tests/test_train_policy.py`

- [x] Add a unit test where a qualified damaged low-speed transition lowers risk from `1.0` to `0.5`; at coefficient `6`, assert the shaped reward exceeds the coefficient-0 reward by exactly `0.3` after `REWARD_SCALE`.
- [x] Add tests that no reduction, risk increase, collision, off-track, insufficient damage, high speed, and current risk below the threshold add zero clearance reward.
- [x] Add tests that missing next features add zero and an invalid coefficient is rejected.
- [x] Add a rollout test confirming the current and next feature vectors are passed to reward shaping in the correct order.
- [x] Run the focused new tests and confirm they fail because the new argument/behavior is missing.

### Task 2: Implement the one-variable reward path

**Files:** `training/train_policy.py`

- [x] Add optional `next_visual_features=None` and `recovery_clearance_reward=0.0` keyword arguments to `shape_transition_reward`; validate the coefficient as finite and non-negative.
- [x] Compute `current_risk = visible_hazard_risk(current_features)` and `next_risk = visible_hazard_risk(next_features)`; add `coefficient * max(0.0, current_risk - next_risk)` only when the verification-boundary gates hold.
- [x] Add the optional coefficient to `collect_rollout` and `train`; capture next-state features from the existing model pass on `transition.next_observation`, and pass the coefficient through.
- [x] Record the effective coefficient in result/checkpoint metadata and the runner's frozen reward config.
- [x] Run the focused tests and confirm they pass.

### Task 3: Freeze, train, and compare

**Files:** New timestamped `artifacts/haic/recovery-clearance-reward-ppo-2seed-u8-<timestamp>/`

- [ ] Copy the completed policy-mean screen runner as the starting protocol, then change only the arm variable to `recovery_clearance_reward=0/6` and preserve the same maps, actor, seeds, U8 budget, delayed TUNE gate, and integrity checks.
- [ ] Confirm preflight loads exactly 4 TRAIN and 2 TUNE episodes, 0 held-out/official maps, four unique seed/coefficient arms, strict actor initialization, and fresh optimizers.
- [ ] Freeze the config, runner, and source hashes before training.
- [ ] Train all four arms and run the fixed TUNE episodes only after every arm passes its integrity gate.
- [ ] Report completion, under-13 finishes, lap time, collision onsets, damage, off-track, final/max progress, post-impact stall duration, and urgent-state brake behavior for every arm.
- [ ] Reject the treatment if either seed loses its control completions, damage/collisions worsen without a completion or stall-recovery gain, or under-13 remains zero without a clear validated speed/completion improvement. Regardless, do not promote a single-geometry result to SOTA.

### Task 4: Record the result

**Files:** `research/2026-09-23-lap13-strategy-lab.md`

- [ ] Record the run ID, source/integrity status, exact counts and times, treatment decision, data limitations, and the next distinct hypothesis.
- [ ] Leave `SOTA.md` and the existing submission archive unchanged unless later independent validation establishes a candidate.
