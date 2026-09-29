# Collision-Budget PPO Screen: Execution Plan

> **Current status (2026-09-25):** trainer integration and per-episode TRAIN outcome logging are implemented; the focused suite passes 15/15 and the matching-thread preflight passes. Three clean-start arms completed (seed8104 off/adaptive; seed8105 off). They are diagnostics only: all use `use_hud=true/use_visual_features=false`, unlike the selected SOTA actor. Do not run seed8105 adaptive or TUNE. The next step is a non-executing preflight for SOTA-initialized fine-tuning with the checkpoint-matched `use_hud=false/use_visual_features=true` architecture; no new PPO run is authorized by this plan state.

## Goal

Compare standard PPO (`lagrangian-mode=off`) with the adaptive TRAIN-only safety-cost path. This is a method-level comparison: the adaptive arm adds the cost critic and routes safety penalties through its separate cost channel. Its multiplier starts at `λ=30` and changes only from completed TRAIN episode costs. Reward coefficients and all other command settings are held fixed.

This is a safety-objective screen. It does not directly teach the actor when to brake, which side to pass, or how to escape a visually unchanged post-impact state. It must not be presented as the answer to the completion or lap-time problem unless its measured action and driving outcomes support that conclusion.

## Current decision and correction

The original random-start comparison was too weak to answer whether fine-tuning the selected policy helps. Three of its four arms completed, but the actor configuration did not match the selected SOTA checkpoint; these runs are retained as clean-start diagnostics and must not be combined with SOTA-initialized results. The seed8105 adaptive arm and TUNE were not run.

The next experiment is a matched SOTA-initialized off/adaptive PPO fine-tuning design. Use strict actor-only loading from the checkpoint, fresh independent Adam optimizers, the same four registered TRAIN cells, paired model/rollout seeds 8104 and 8105, 8,192 decisions/eight updates per arm, throttle expansion 3.5, and four raw ticks per action. The only arm setting is `--lagrangian-mode`. This is an internal diagnostic only: the selected SOTA lineage has prior official-track exposure and TUNE-based selection, so no descendant from this experiment is eligible for submission or SOTA promotion.

## Diagnosis from existing evidence

The current evidence points to a learning-signal and initialization problem, not simply too few PPO updates:

- The seed-8105 continuation added 32,768 decisions and 32 PPO updates. TUNE completion stayed 0/2, mean progress changed `.7572 → .7531`, collision onsets rose `2 → 4`, damage remained `1.0`, and urgent-state brake mean changed only `.01801 → .01826`. The actor weights moved, but the safety response did not improve. Repeating the same continuation is therefore not a useful next experiment.
- In the recorded late obstacle approach, urgency rose `.81 → 1.0` while speed rose `44.15 → 44.59 m/s`; brake stayed at zero before contact. Earlier, the actor did brake before another obstacle but still collided across three collision-bearing decisions. This points to timing and clearance margin, not a universal “brake more” fix.
- `shape_transition_reward()` contains damaged-state gas/steer rewards that pay for commanded actions without checking whether the car actually advances. Its high-risk gate also withholds that signal in the pinned-in-front-of-obstacle state where recovery is needed. A scalar risk multiplier cannot repair absent or contradictory cue-to-action credit.
- The recorded DAgger warm-start did not optimize the active visual feature encoder. Its raw observation/action sequence was not retained, so left/right label conflict and collision-approach behavior cannot be reconstructed from that run.
- A separate clean-start teacher-warmup→PPO attempt also failed its TUNE screen: the seed-8105 teacher-BC actor completed one repeated-geometry TUNE lap in 18.96 s before PPO, but its PPO U8 continuation went off-track on both TUNE resets; another clean-start teacher-warmup attempt recorded 0/8 TUNE finishes. This is a reason to keep this screen PPO-only and not reuse teacher-warmed weights; it does not establish that every teacher signal is ineffective.
- The current actor is feed-forward. A fixed visual input can produce the same action repeatedly after the car stalls; PPO rollout storage has no recurrent state. A cost critic can price a bad outcome, but it cannot by itself add temporal memory or a missing perception cue.
- The audited SOTA checkpoint is not a clean starting point: its metadata records teacher demonstrations, official-track entries in its historical training split, and TUNE-based checkpoint selection. The audit found zero clean-proven checkpoints among six representatives. That does not prove why the actor collides, but it makes the existing SOTA lineage unsuitable as a compliant experiment initialization.

These observations do not establish one sole cause. The screen below is gated on obtaining a clean-proven actor and measuring actual TRAIN cue exposure first.

## Architecture

- Keep the submitted policy path actor-only and image-only. Training may add a cost-value head; the head and all cost labels are excluded from the submission artifact.
- Keep the ordinary reward critic and add a scalar cost critic. Compute raw reward and cost GAE separately, combine actor advantages as `A = A_reward - λ A_cost`, then normalize that combined advantage once, matching the existing PPO convention.
- Use `γ_reward=0.99`, `GAE_reward=0.95`; use `γ_cost=1.0`, `GAE_cost=0.95` for the episodic safety budget. This makes the cost return an undiscounted episode constraint. Therefore `λ=30` matches the old penalties' per-transition units algebraically, but does **not** make the full PPO objective identical to the old discounted reward objective.
- Both arms use matching actor initialization, fresh optimizer, reward coefficients, maps, seeded episode order, PPO settings, and random seeds. The off arm uses standard PPO with safety penalties in reward; adaptive adds a cost critic and moves those penalties to the cost channel. The CLI arm selector is the only changed command argument, while those mode-specific algorithm components are the treatment.

## Tech Stack

Python 3.11, PyTorch, the existing PPO trainer and custom TRAIN map loader. The plan references `training/train_policy.py`, `training/rollout.py`, `training/ppo.py`, `haic_agent/networks.py`, and focused tests; none are edited in this preparation task.

## Frozen experiment definition

### Starting actor and eligibility gate

The completed clean-start arms above used `use_hud=true/use_visual_features=false` and are not SOTA transfer evidence. For the next design:

1. Load `artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt` as the actor-only initializer. Its recorded file SHA-256 is `3ceb5ebbe2a4bd96944649a693b8ef924c0252fa620b0fd547c7fb494afff199`.
2. Instantiate both arms with `use_hud=false`, `use_visual_features=true`, and `use_temporal_features=false` to match the checkpoint. `load_actor_weights()` must strict-load the complete actor state dict. Record the checkpoint file hash, strict-load result, actor tensor hash, and architecture in the preflight.
3. Create a separate fresh Adam optimizer for each arm after model construction. Do not restore SOTA optimizer state, training step, cost critic, demonstrations, or warm-up state. `--resume` stays unset; both arms use the same `--initialize-from` path and hash.
4. Keep this warm-start experiment diagnostic only. The audited SOTA ancestry includes official-track training data and TUNE-based selection; improvements can help explain transfer behavior but cannot establish a compliant submission candidate or update `SOTA.md`.

### TRAIN-only diagnostic gate

Before PPO training, collect a frozen-actor, TRAIN-only behavior trace on the four registered cells. Log the actual actor pixels/features, urgency, speed, steer, gas, brake, collision decision, collision onset, damage before/after, progress delta, and episode boundary on every policy decision. This is a mechanism probe, not a policy-selection run.

For each collision, report whether a visible urgent cue existed one or more policy decisions before contact, whether brake/steer changed after that cue, and whether progress continued after contact. Do not use the SOTA's 1,040-image fixed observation bank as a substitute for on-policy exposure, and do not use TUNE, held-out, or official traces to set a TRAIN curriculum.

Interpretation gate:

- If the cue is absent until contact, the limiting mechanism is perception/latency; do not expect λ adaptation to solve it.
- If the cue is visible but the action mean does not respond before contact, the Lagrangian screen is eligible as a credit-assignment test.
- If the action responds but clearance still fails, record the likely braking-distance/action-repeat or route-clearance problem; this screen alone is insufficient.
- If no relevant hazard state occurs in a TRAIN cell, mark the cost-learning question underexposed and stop rather than treating zero cost as evidence of success.

### Paired arms

| Setting | Off arm | Adaptive arm |
|---|---:|---:|
| Safety cost actor objective | disabled; standard PPO reward penalties remain | enabled; safety penalties move to the cost channel |
| Initial λ | not used | 30 |
| λ update | none | `clip(λ + 5 × (mean_window_cost - 0.05), 0, 60)` |
| Update window | none | every 8 completed TRAIN episodes |
| Shared actor initialization, fresh optimizer, reward coefficients, cost definition, PPO recipe | identical | identical |

This compares the complete adaptive safety-cost method against standard PPO. It does not isolate dual adaptation from the added cost critic.

`0.05` is a proposed engineering budget in the cost units below. It is not an official competition threshold and is not estimated from TUNE. Freeze it in the run manifest before either arm starts; do not tune it after seeing results.

### Cost labels and reward separation

At each policy decision, let `D_prev` be pre-action damage and `D_next` be post-action damage. Verify those fields against `transition.observation_labels` and `transition.labels` before coding; if that mapping is not valid, establish another pre/post source and add a reset test.

Define the contact/damage-increment component once, so a collision decision that also increments damage is not counted twice:

```text
damage_increase = max(0, D_next - D_prev)
contact_cost = max(damage_increase, 0.2 * I(collision))
damage_stock_cost = D_next / 3000
off_track_cost = I(off_track) / 30
failure_cost = 0.2 * I(environment_terminated or environment_truncated or episode_time_limit, and not finished)
c_t = contact_cost + damage_stock_cost + off_track_cost + failure_cost
```

The collision floor preserves one `0.2` cost for every collision-bearing policy decision, including a collision at the damage cap; collision **onsets** remain a separate descriptive metric. A positive damage increment contributes positive cost. `failure_cost` applies at a true environment terminal or true environment time-cap, never at a collector cutoff.

At λ=30, the explicit `shape_transition_reward()` terms are recovered per transition: `30×0.2=6` for its collision penalty after `REWARD_SCALE=0.1`; `30×D/3000=0.01D` for its scaled damage-stock penalty; `30/30=1` for its scaled off-track penalty; and `30×0.2=6` for its scaled terminal-failure penalty. The adaptive trainer removes the simulator's raw out-of-playfield `-100` using the underlying termination cause, including when a collision label is also present. This is a unit check, not proof of equal policy gradients or equal episode returns.

In the adaptive arm, remove these explicit penalties from reward so they are not charged in both channels. The off arm retains them in standard PPO reward. Disable the damaged-state gas-only and steer×gas recovery bonuses in both arms because they pay for commands without requiring progress. Keep all remaining progress, time, route, speed, obstacle-brake, and curve-brake coefficients identical and frozen. The result is not a direct re-evaluation of the historical SOTA objective.

Before implementation, confirm the simulator's raw `transition.reward` does not already include any of the explicit penalties. If it does, define the decomposition to remove double charging before training.

### Episode and PPO budget

- Use model seeds `8104` and `8105`, paired across both arms.
- Per model seed and arm, collect exactly 8,192 policy decisions in 8 PPO updates of 1,024 decisions each.
- The four fixed TRAIN cells below are the only map episodes loaded. They are interleaved with the same seed-specific order in paired arms; record actual map/seed counts and transitions.
- A collector cutoff is not a completed episode. Discard that fragment when its environment is reset and never use it for a dual update. Therefore 8,192 decisions do not guarantee eight completed episodes or any λ update. If fewer than eight TRAIN episodes complete, report that adaptive λ did not update and treat adaptation as inconclusive.
- The collector carries separate `episode_time_limit` and `collector_truncated` flags. The registered per-episode cap counts as a failed episode; a 1,024-decision batch boundary leaves an incomplete fragment.
- Count true termination, underlying environment time limit, and the registered per-episode decision cap as episode completion for the dual window; unfinished endings incur failure cost. A collector-only cutoff does not complete an episode. Because the current collector closes its environment at a batch cutoff, integration must either preserve that environment/episode accumulator or discard the partial record on reset; it cannot silently join fragments from different resets.

| TRAIN cell | Map | Environment seed | Episodes per arm/model seed |
|---|---|---:|---:|
| 1 | `custom-track-haic-obstacles-20260920` | 20260920 | 8 |
| 2 | `custom-track-haic-obstacles-20260920` | 20260924 | 8 |
| 3 | `custom-track-haic-train-20260921` | 20260921 | 8 |
| 4 | `custom-track-haic-train-20260921` | 20260925 | 8 |

Use the same seed-specific cell order in paired arms. The TRAIN-only loader opens only the four registered map/seed cells. TUNE, held-out, and official map files are not loaded; TUNE is deferred until all four training arms finish.

### PPO settings

Freeze one source of truth before running: seeded random actor initialization with no parent checkpoint, fresh Adam state, PPO learning rate `2e-6`, `gamma_reward=.99`, `gae_reward=.95`, clip `.2`, entropy `.01`, value coefficient `.5`, auxiliary coefficient `.1`, max grad norm `.5`, 4 epochs, minibatch 32, throttle expansion `3.5`, and no teacher warm-up. Defer TUNE selection. The preflight records paired actor hashes, map/config hashes, and the exact arm commands.

## Metrics and decision rules

Per episode and per TRAIN cell, retain raw JSON for:

- completion, true terminal/time-cap reason, progress, lap time, mean/max speed;
- collision-bearing decisions and collision onsets separately, `D_prev`, `D_next`, damage increments, max/final damage, off-track events, and each cost component;
- pre-impact visible cue lead time in policy decisions, speed, steer, gas, brake, and progress delta;
- post-impact progress/speed at 1, 5, and 10 policy decisions where the episode continues;
- reward return, cost return, reward/cost critic losses, combined actor advantage scale, gradient norm, λ by window, and nonfinite/invalid outputs;
- actual transitions, PPO minibatches, wall time, checkpoint hashes, source/config hashes, and train-only manifest.

Stop before training on a provenance failure, ambiguous damage timing, a cost/reward double-charge, or an unverified time-cap/collector distinction. During training, stop the affected arm on nonfinite loss/gradient/cost value or a train-split violation. Stop adaptive training if λ reaches 60 and mean cost remains above `.05` for two consecutive 8-episode windows; label it budget-infeasible rather than successful safety learning. Complete the preregistered 32 episodes otherwise; do not select intermediate checkpoints using TUNE.

Compare arms by paired seed and TRAIN cell. Primary outcomes are complete-episode rate, per-episode cost distribution, damage cap, collision-bearing decisions, and whether brake/steer changes occur before contact. A lower cost caused only by early off-track/episode termination is not an improvement. Speed is secondary in this safety screen; faster speed with worse completion or damage is rejected.

## TUNE gate

After both arms, seeds, and integrity checks finish, run the same two registered TUNE episodes once on each final U8 actor as a measurement-only gate. Do not feed TUNE observations into training, cost estimation, λ updates, early stopping, or checkpoint choice. Do not run held-out or official maps from this screen. A TUNE improvement does not update SOTA or the submission ZIP; a candidate needs a separate provenance-clean, map-diverse independent validation.

## Implementation and verification status

### Task 1 — Close the evidence gaps

**Read-only sources:** `research/artifacts/checkpoint-provenance-audit-20260924.md`, `research/2026-09-23-lap13-strategy-lab.md`, the U32 raw JSON, `training/env_factory.py`, `training/train_policy.py`, `training/rollout.py`, `training/ppo.py`, and `haic_agent/networks.py`.

- [x] Use seeded random initialization with no parent checkpoint, teacher warm-up, demonstrations, or restored optimizer state.
- [x] Preflight both model seeds and verify byte-identical actor hashes across `off` and `adaptive`.
- [x] Verify pre/post damage labels, raw reward decomposition and source termination flag, true time-cap signaling, and collector truncation behavior in focused tests.
- [x] Freeze the four TRAIN cells, throttle expansion, PPO settings, cost budget, and exact arm commands. Preserve manifest/map hashes.
- [x] Use the existing TRAIN-only hazard exposure diagnostic as context; this paired screen does not open TUNE, held-out, or official maps before training completes.

### Task 2 — Add cost-aware rollout and update support

**Implemented files:** `training/lagrangian.py`, `training/lagrangian_ppo.py`, `training/train_policy.py`, `training/env_factory.py`, `env_wrapper.py`, `training/preflight_lagrangian_screen.py`, and focused tests.

- [x] Add a standalone validated cost helper, TRAIN-only completed-episode accumulator, partial-fragment discard record, and fixed/adaptive multiplier window.
- [x] Add focused tests for cost arithmetic, damage-cap contact, positive damage increments, TRAIN split enforcement, real episode/time-limit versus collector boundaries, and the eight-completed-episode λ update.
- [x] Wire the storage to the live trainer and keep per-episode time limits distinct from collector cutoffs. Incomplete collector cost fragments are discarded on reset.
- [x] Add an independent cost-value head using the actor's existing 128-dimensional pixel latent; add reward/cost GAE with terminal, episode time-cap, and collector-cut bootstrap tests.
- [x] Apply raw combined actor advantage `A_reward - λ A_cost` and normalize the combined vector once. The isolated updater reuses existing PPO optimizer-group construction and adds cost-value loss.
- [x] Provide a strict-loadable actor-only state export; no cost critic or privileged/map input is present in that export path.
- [x] Keep the explicit safety penalties out of the adaptive reward channel; preserve standard reward in the `off` arm. Disable behavior-only recovery gas/steer bonuses in both paired commands.
- [x] Test lambda-30 units, positive damage increments without double counting, repeated collisions, damage-cap collision, off-track, finish, terminal failure, true time-cap, and collector truncation.
- [x] Test that λ ignores collector fragments and updates only after eight complete TRAIN episodes.

### Task 3 — Freeze and run the paired U8 screen

**Files:** a new immutable run directory under `artifacts/haic/` with protocol, config, manifests, raw episode JSON, checkpoints, logs, and summary.

- [x] Confirm strategy C's fixed-actor action-repeat diagnostic completed before the first PPO arm.
- [x] Run seed8104 off/adaptive and seed8105 off at 8,192 decisions/eight updates each; retain every log and checkpoint. The first seed8104-off attempt failed before simulator actions because CPU thread settings produced different preflight/trainer initial actor hashes; the failure is preserved and excluded from results.
- [x] Verify completed episode counts, per-episode TRAIN outcomes, incomplete collector fragments, and λ from each completed arm.
- [x] Stop after the source audit found the clean-start actor architecture differed from SOTA. Do not run seed8105 adaptive or any TUNE evaluation.
- [ ] Add SOTA checkpoint initialization and architecture/hash validation to the non-executing preflight. Do not launch the resulting runner commands.

### Task 4 — Record without promotion

**Files:** append an evidence-backed note to `research/2026-09-23-lap13-strategy-lab.md`; add allowlisted per-arm TRAIN summaries under the run directory and rebuild `RESULTS.md` through `research_ops.cli sync-results`.

- [x] Preserve the failed gate attempt and all three completed clean-start arms; state that each used eight updates and 8,192 decisions.
- [x] Mark the clean-start results as diagnostic only and state that they do not test transfer from SOTA.
- [x] Leave `SOTA.md` and the submission ZIP unchanged. Do not treat SOTA descendants as promotion candidates because the recorded lineage is not clean.

## Source audit: confirmed environment facts

- `training/env_factory.py::CollectingEnvironment.step_transition()` calls `collect_labels()` before stepping and `labels_with_hud()` after stepping. `observation_labels.damage` is the pre-decision value and `labels.damage` is the post-decision value in this collector.
- `env_wrapper.py::CarEnvironment.step()` ORs collision info across up to four raw physics ticks and calls `CollisionDamage.update()` once per policy decision. `damage.py` increments by `0.2` per collision-bearing decision, caps at `1.0`, and the wrapper terminates when the cap is reached.
- The wrapper exposes post-step `collision`, `damage`, and `retire_reason`; `TrainingLabels.off_track` is true when that reason is `off_track`. The wrapper sets `terminated` for damage-cap crash and off-track retirement.
- `core/vendor/car_racing.py::CarRacing.step()` sets `truncated=True` when the finish line is crossed and `terminated=True` with raw reward `-100` when the car leaves the playfield. Thus `truncated` alone is not a failure label; pair it with `finished` and preserve the actual end cause.
- The raw contact callback sets the collision flag but has no collision/damage reward term. However, raw out-of-playfield failure reward `-100` remains in `transition.reward`; the standalone helper cannot move it. Full cost-channel integration must explicitly decide how to remove or retain this raw terminal contribution before claiming all failure penalties were symmetrically moved.
- The helper in `training/lagrangian.py` deliberately accepts four separate boundary facts: environment termination, environment truncation, the trainer's episode time cap, and PPO collector truncation. Only an actual episode end completes the accumulator; only TRAIN completions can enter λ windows.
- `haic_agent.networks.VisualActorCritic.forward()` already returns a differentiable `PolicyOutput.latent` of width 128. `training/lagrangian_ppo.py::LagrangianActorCritic` adds the cost head to that latent without editing A's model file; it also forwards `last_visual_features` and `last_temporal_features` for the existing TRAIN-only recorder interface.
- `LagrangianPPOUpdater` is called by `train_policy.train()` for fixed/adaptive modes. The first frozen screen confirmed the CLI preflight and trainer hash gate, but its starting actor configuration differed from SOTA.
- Focused command `python -m unittest discover -s tests -p 'test_lagrangian*.py' -v` passed 23 tests. No full test suite or simulator run was performed in this bounded implementation step.

## Next preflight requirements (no training)

- Accept an explicit `--initialize-from` checkpoint while continuing to reject `--resume`.
- Read its model metadata and construct the actor with `use_hud=false`, `use_visual_features=true`, `use_temporal_features=false`; strict-load its `model_state` through `load_actor_weights()`.
- Record the selected checkpoint SHA-256 and reject a changed file. Verify the same loaded actor hash in off/adaptive for each seed, with empty Adam state in both arms.
- Add the matched architecture flags and initializer path to all runner commands and shared-config validation.
- Keep the TRAIN-only loader and verify TUNE, held-out, and official map files remain unopened. This step ends at a non-executing preflight; no PPO or TUNE process starts.

## Self-audit before execution

- [x] The completed off/adaptive clean-start arms differed by `--lagrangian-mode`; they are not SOTA-transfer evidence because their actor feature configuration differed.
- [x] No TUNE, held-out, official map, or privileged state entered the three completed TRAIN arms.
- [x] Damage increments are positive cost and are not double-counted with collision decisions.
- [x] Environment episode ends and collector cutoffs remain distinct throughout storage, GAE, and dual updates.
- [x] Per-episode completion and actual decision counts are reported; each completed arm collected exactly 8,192 decisions.
- [ ] The next preflight verifies strict SOTA actor loading, shared actor hashes, fresh optimizers, and TRAIN-only file access; no execution follows in this task.
