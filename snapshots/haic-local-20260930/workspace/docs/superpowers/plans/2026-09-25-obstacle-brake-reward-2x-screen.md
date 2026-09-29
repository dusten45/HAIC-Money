# Obstacle-Brake Reward 6-vs-12 PPO Screen Implementation Plan

> **For agentic workers:** Execute inline in the current root checkout. Keep PPO actor-only. The user prohibited reviews and subagents; do not request either.

**Goal:** Test whether doubling the train-only visible-hazard brake reward improves obstacle avoidance and lap completion across PPO seeds after the curve-brake screen produced only a seed-specific finish.

**Architecture:** Add an optional per-training-run `obstacle_brake_reward` parameter through the existing reward, rollout, and PPO training path, defaulting to the current constant. Reuse the custom-only split loader and frozen runner protocol in a new timestamped experiment directory. The four arms are PPO seeds 8104/8105 crossed with obstacle-brake reward 6/12; curve-brake reward remains 6 in every arm. Train all arms before loading TUNE, then compare completion, under-13 finishes, collision onsets, damage, lap time, urgent-cue braking, and stalled recovery.

**Tech Stack:** Python 3.11, PyTorch 2.1 CPU, NumPy 1.26, Gymnasium 0.29.1, pytest.

**Spec:** `research/2026-09-23-lap13-strategy-lab.md`, the 2026-09-25 curve-brake screen result and follow-up decision.

## Global Constraints

- Use `C:\Users\koi\Coding\HAIC` as the source of truth; create a new experiment artifact directory and preserve prior runs.
- Strictly initialize only the actor weights from checkpoint SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`; each PPO arm uses a fresh Adam optimizer.
- Train PPO seeds `8104` and `8105`, each with obstacle-brake reward `6` and `12`, curve-brake reward fixed at `6`, speed target `70`, shortfall penalty `.6`, learning rate `2e-6`, U8/8,192 decisions, teacher warmup 0.
- Load only the current registered split's four TRAIN and two TUNE custom episodes. Do not materialize held-out or official maps.
- All arm fields, reward constants, source files, map hashes, and output paths must be recorded and integrity checked. No TUNE evaluation or selection may run during training.
- Evaluate TUNE only after all four final checkpoints pass the integrity gate; retain raw episode and decision traces.
- Keep the result TUNE-only: no SOTA change, package, submission, or official evaluation. Do not add runtime rules or teacher state to the actor.
- Wall-time limit: 7,200 seconds. Stop before training if frozen source/config/map hashes or split counts fail.

## Validation Focus

- The default obstacle-brake value 6 preserves all existing callers and reward outputs.
- An arm value 12 changes only `OBSTACLE_BRAKE_REWARD * visible_hazard_risk * normalized_brake_fraction * reward_scale`; verify the exact numeric delta in a deterministic reward test.
- The runner creates unique directories for same-seed arms and records both reward coefficients in run records, checkpoints, Tune summaries, and paired comparisons.
- The split loader materializes exactly 4 TRAIN and 2 TUNE custom episodes with zero held-out/official geometry loaded.
- Fresh optimizer state, strict actor-only loading, and deferred-TUNE behavior remain true for all four arms.
- The result summary reports completion, under-13, collision decisions/onsets, off-track, damage, progress, speed, lap times, urgent-visible braking, immediate pre-visible braking, and post-impact stall duration.

---

### Task 1: Parameterize the learned visible-hazard brake reward

**Files:**
- Modify: `training/train_policy.py`
- Test: `tests/test_train_policy.py`

**Interfaces:**
- Add optional `obstacle_brake_reward: float = OBSTACLE_BRAKE_REWARD` to `shape_transition_reward`, `collect_rollout`, and `train`.
- Validate the value is finite and non-negative; pass it unchanged through each call.
- Record the effective value in training metadata and checkpoint metadata. Existing callers that omit it must retain value 6.

- [ ] Add a failing test using equal transitions and visual features to assert the reward delta between coefficients 6 and 12 is `6 * risk * normalized_brake_fraction * reward_scale`; include the default-value equivalence and invalid-value case.
- [ ] Run the focused new tests and confirm failure occurs because the optional parameter is absent.
- [ ] Thread the validated argument through shaping, rollout collection, and training without changing the default.
- [ ] Run `tests/test_train_policy.py` and `tests/test_curve_brake_screen.py` with the project `.venv`.

### Task 2: Freeze and execute the paired four-arm PPO screen

**Files:**
- Create: `artifacts/haic/obstacle-brake-reward-ppo-2seed-u8-20260925/<run-id>/run_config.json`
- Create: matching `run_experiment.py`, `source-manifest.json`, and `preflight.json` in that directory.
- Create: four arm records/checkpoints, raw Tune traces, integrity-gate record, same-state action replay, and final summary under that directory.

**Interfaces:**
- Reuse `training/curve_brake_screen.py` for exact registered custom TRAIN/TUNE loading and per-decision traces.
- Arm configs carry both `curve_brake_reward=6` and `obstacle_brake_reward` in `{6,12}`; every other setting is identical within each matched PPO seed.
- Each arm starts from the same strict actor weights and a fresh empty Adam optimizer, trains 8 updates / 8,192 decisions, and disables Tune selection during updates.
- Final summary pairs reward 12 against 6 for each PPO seed and labels the evaluation as one custom TUNE geometry repeated over two registered seeds.

- [ ] Write the immutable config and runner to a unique output directory; pin the root checkpoint, split, three custom map geometries, runtime sources, and plan hashes.
- [ ] Run preflight and verify 4 TRAIN / 2 TUNE, held-out=0, official=0, actor load strict, exact fixed rewards, and unique arm output paths.
- [ ] Run all four PPO arms; capture the live process handle and preserve intermediate U2/U5/U8 snapshots.
- [ ] Verify every checkpoint and source hash before the first TUNE evaluation.
- [ ] Evaluate the same two TUNE episodes on every final actor and write raw traces; compute paired seed summaries and fixed-image action changes.
- [ ] Append the raw-evidence result and next falsifiable hypothesis to `research/2026-09-23-lap13-strategy-lab.md`; do not promote SOTA from TUNE.
