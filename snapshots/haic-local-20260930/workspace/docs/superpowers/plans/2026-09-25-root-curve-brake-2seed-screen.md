# Root Curve-Brake Two-Seed PPO Screen Implementation Plan

> **For agentic workers:** Execute this plan inline in the authorized root checkout. Do not create or use a Strategy-A worktree. User explicitly prohibited any review; no reviewer or review request is part of this plan.

**Goal:** Run a reproducible, four-arm custom-only PPO screen comparing curve-brake reward coefficient 0 versus 6 across seeds 8104 and 8105, with TUNE deferred until all training artifacts pass integrity checks.

**Architecture:** Extend the root trainer with actor-only checkpoint initialization, the single curve-magnitude × normalized-brake reward term, and a mode that skips all TUNE-based checkpoint selection during training. A small training helper loads only registered TRAIN/TUNE custom maps, collects a fixed SOTA-derived TRAIN observation bank, and records deterministic per-decision evaluation traces; a frozen experiment runner trains four arms from the same actor with fresh optimizers, verifies all inputs/outputs, then evaluates the two TUNE episodes for every arm and runs same-image action diagnostics.

**Tech Stack:** Python 3.11, PyTorch 2.1 CPU, NumPy 1.26, Gymnasium 0.29.1, pytest.

**Spec:** User-provided 2026-09-25 experiment protocol in the active task; no separate design document.

## Global Constraints

- Use only `C:\Users\koi\Coding\HAIC` as the authoritative source; do not use Strategy-A worktree, v1/v2 snapshots, or their checkpoints.
- Initial actor checkpoint SHA-256: `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`; load `model_state` strictly and do not restore its optimizer state or training step.
- Train four arms: PPO seeds 8104 and 8105, each with coefficient 0 and 6, fresh Adam, teacher warmup 0, U8/8192 steps per arm, and the same registered four TRAIN episodes.
- Read root `training/maps/site/site_map_split.json` only for its registered TRAIN/TUNE entries; materialize only its four TRAIN and two TUNE custom episodes. Do not load held-out or official geometry.
- The only reward difference between paired arms is `coefficient * road_curve_magnitude * normalized_brake_fraction * reward_scale`.
- Before PPO, collect and freeze the deterministic SOTA actor's pixel observations from the four TRAIN episodes. Compare SOTA and every arm's U2/U5/U8 snapshots on those exact images; this is diagnostic only and does not add an arm or alter training.
- Do not evaluate, select checkpoints, or otherwise use TUNE during any arm's updates. Run each final TUNE evaluation only after all four arm checkpoints and input/source hashes pass integrity checks.
- Preserve all pre-existing dirty root files; make no SOTA/RESULTS changes, held-out/official run, package, or submission.
- Wall-time limit is 7,200 seconds; final report includes completion, under-13, lap time, collisions/onsets, off-track, damage, progress, high-curvature speed/braking, pre-visible braking, and raw per-decision traces.

## Review Focus (verified by tests and preflight; no external review requested)

- Coefficient 0 versus 6 changes only the specified scaled reward term; test the literal expected delta.
- Actor initialization is strict and leaves a newly created Adam optimizer empty; never reuse saved optimizer moments or step.
- Deferred-TUNE mode calls no TUNE evaluator during training and saves the final U8 actor rather than a Tune-selected actor.
- TRAIN/TUNE loading never touches held-out/official map files, even if such entries exist in the split manifest.
- Evaluation records the action and pre-action visual features alongside collision, off-track, damage, progress, finish, and lap time for each decision.
- Same-state diagnostic actions use the exact frozen TRAIN image bank, with row counts for urgent-visible, high-curve, and their intersection; no diagnostic image is sampled from TUNE/held-out/official maps.

---

### Task 1: Add the controlled PPO training behavior

**Files:**
- Modify: `training/train_policy.py`
- Test: `tests/test_train_policy.py`

**Interfaces:**
- Consumes: Existing `shape_transition_reward`, `collect_rollout`, `train`, `VisualActorCritic`, and `PPOUpdater` behavior.
- Produces: `curve_brake_reward` parameter on reward/rollout/train; actor-only `initialize_from`; opt-in `tune_selection=False` mode that performs no TUNE evaluation and saves the final update.

- [ ] Add tests first for the exact coefficient-6 reward delta (`6 * 0.75 * 0.5 * 0.1 = 0.225`), strict actor-only load with fresh optimizer, no TUNE/selection calls when deferred mode is enabled, and actor-only snapshots at requested completed steps.
- [ ] Run each new test and confirm it fails for the missing behavior.
- [ ] Implement the minimal optional API while preserving current defaults for every existing caller.
- [ ] Run the focused trainer tests and the root trainer test file.

### Task 2: Create custom-only loading and trace evaluation

**Files:**
- Create: `training/curve_brake_screen.py`
- Create: `tests/test_curve_brake_screen.py`

**Interfaces:**
- Consumes: Root split JSON, `load_site_map`, `SiteMapEpisode`, `SiteMapSplit`, current actor inference, and `create_episode_environment`.
- Produces: A loader for only four TRAIN/two TUNE custom episodes; a deterministic evaluator returning one JSON-safe episode record with one trace row per policy decision; summaries for event counts, under-13, damage, progress, high-curvature actions, and pre-visible braking.

- [ ] Test the split loader against a fixture containing an intentionally missing held-out map and verify only TRAIN/TUNE maps are materialized.
- [ ] Test raw-trace capture and summaries with a deterministic fake environment/model, including one collision onset and one obstacle-visibility onset.
- [ ] Test that the fixed observation bank preserves float32 pixel inputs and source episode/decision labels and can be replayed against different actor weights without advancing the simulator.
- [ ] Run new tests to confirm expected RED failures, implement, then run them GREEN.
- [ ] Run focused custom-map and trace tests.

### Task 3: Freeze and run the four-arm root experiment

**Files:**
- Create: `artifacts/haic/curve-brake-reward-ppo-2seed-u8-20260925T0529KST/run_config.json`
- Create: `artifacts/haic/curve-brake-reward-ppo-2seed-u8-20260925T0529KST/source-manifest.json`
- Create: `artifacts/haic/curve-brake-reward-ppo-2seed-u8-20260925T0529KST/run_experiment.py`
- Create: per-arm `run_record.json`, `policy.pt`, raw TUNE episode JSONL, and comparison outputs under the same directory.

**Interfaces:**
- Consumes: Tasks 1–2 and frozen root source, map, split, config, and checkpoint hashes.
- Produces: `preflight.json`, a hash-pinned SOTA-derived TRAIN observation bank, four actor-only PPO arm outputs, integrity verification, per-episode/raw-decision TUNE traces, seed-8104 same-state action comparisons for U2/U5/U8, and a paired comparison by seed and coefficient.

- [ ] Preflight exact root checkpoint/split hashes, register hashes for all loaded sources and the 3 custom geometries, and confirm four TRAIN/two TUNE episodes, zero held-out episodes, and no official geometry.
- [ ] Before training, follow the SOTA actor on only those four TRAIN episodes and freeze its pixel observations plus visual features as the shared action-diagnostic bank.
- [ ] Train arms in paired order for each seed; record step/update counts and prove each Adam optimizer begins empty.
- [ ] Verify all four final actors and frozen input hashes before any TUNE inference.
- [ ] Evaluate the exact same two TUNE episodes for each actor, write per-episode/raw-action traces and summaries, then compute paired comparisons for seeds 8104 and 8105.
- [ ] Replay SOTA and all four arms' U2/U5/U8 checkpoints on the frozen observation bank; summarize urgent-visible, high-curve, and intersection action outputs.
- [ ] Verify the final evidence and report the outcome without promoting SOTA/RESULTS or packaging/submitting.
