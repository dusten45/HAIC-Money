# HAIC score-reset batch — design revision 2

## Status and intent

- State: `DESIGN_PENDING_APPROVAL`.
- Revision: `2`. Its SHA-256 is calculated from the saved file bytes and recorded with any approval; it is intentionally not embedded here.
- Purpose: create a clean, score-oriented V2 experiment cycle with newly trained actor checkpoints, comparable evaluation, and an explicit reference run against the recorded local SOTA.
- This is a design only. It does not authorize code changes, tests, training, simulation, packaging, SOTA updates, or submission.
- “Reset” means fresh policy initialization, fresh run IDs, fresh V2 plans, and V2-only outputs. It does not mean deleting or rewriting user files, old runs, artifacts, submissions, maps, or reports.

## Selected approach

| Approach | Decision | Reason |
|---|---|---|
| Delete old harness data and rebuild | Reject | It destroys provenance, conflicts with the preserve-in-place rule, and still would not make historical scores comparable. |
| Continue the existing local DAgger adapter task | Defer | That adapter is diagnostic-only, its preserved checkpoints are unavailable, and it cannot establish a competitive score improvement. |
| Start fresh in V2 and compare against both a new matched control and the explicitly named SOTA actor | Select | It keeps training fresh, tests each mechanism against a common control, and measures the current best on the same new cells without searching or importing legacy data. |

The SOTA checkpoint is a read-only evaluation reference, never a training initialization. It is permitted only at the exact path and digest recorded in `SOTA.md`; if the file is absent or its digest differs, the comparison stops rather than searching for a substitute.

## Performance target and score separation

There are two reported outcomes, with completion as a hard priority:

1. **HAIC-like episode ordering:** on a matched cell, every finisher ranks ahead of every non-finisher; finishers compare by simulated lap time; non-finishers compare by progress; execution failures are excluded from this official-like ordering. That makes completion the first gate; speed only separates finishers. The current official numeric score formula and leaderboard position are not known from the pinned local source, and the competition website was inaccessible on 2026-09-26. Recheck official sources before any external action. Do not call this local projection an official score.
2. **V2 candidate selection:** compare completion rate over the full registered denominator first. On a completion-rate tie, use median finished lap time, mean incomplete progress, P90 finished lap time, collisions, damage, then p95 action latency, in that order. Agent failures count as non-completions here; infrastructure-invalid batches have no winner.

The held-out advancement target is **16/16 completed episodes for the selected candidate** (both independently trained actors × all 8 held-out cells); a faster lap cannot offset any miss. If no candidate reaches that completion gate, stop without promoting speed and revise toward recovery/completion. After completion, report the user's `under-13-second finish fraction` as `completed laps under 13,000 ms / all registered attempts` and compare simulated lap-time distributions. Under-13 is a speed target, not a current official scoring rule. Also report incomplete progress, collision onsets, damage, invalid actions, action latency p50/p95/max, reset/import time, RSS, training/update/evaluation duration, and exact checkpoint/package identities.

## Direction-change rule

This batch tests four different causal mechanisms: signed obstacle offset, obstacle approach motion, pedal range, and curve-conditioned braking. It is not permission to keep tuning one mechanism's thresholds or weights. A candidate continues only when the registered activation check passes and matched completion evidence supports it. An inactive or non-improving mechanism is removed from the next cycle rather than rescued by an unregistered sweep.

Count a cycle as valid and comparable only when its registered control, map/seed cells, denominator, resource budget, and evaluation protocol match and all three evidence gates have usable evidence. Infrastructure-invalid cycles do not count. A cycle is non-improving when no candidate clears the 16/16 completion gate and shows a strict improvement under the registered completion-first ordering against the matched control/SOTA; speed cannot rescue a candidate that misses the completion gate. After three consecutive valid comparable non-improving cycles, the next effort must begin at `DISCOVER` with a new plan hash and at least one newly registered causal mechanism class; it must not repeat this same four-arm bundle under new parameter values. Preserve prior checkpoints and run records. A pivot mechanism is chosen only after the new discovery step, so this design does not claim an unmeasured replacement will work.

## Fresh control, reference, and candidate arms

- `sota_reference`: read-only actor checkpoint at `artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt`, expected SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199` as recorded in `SOTA.md`. It is evaluated, not resumed, copied, or rewritten.
- `fresh_control`: new actor initialization, no `--resume`, no legacy actor initialization, and no legacy optimizer state. Use teacher-only supervised warm-up on a frozen TRAIN-only pixel demonstration bank, then PPO. At inference the package contains the PPO actor only.
- Candidates each change exactly one registered mechanism relative to the same fresh-control protocol. All arms use one fixed observation schema and network shape; feature slots for inactive mechanisms are zeroed, not removed. Common HUD/visual features are enabled for all arms.

### Candidate 1 — obstacle/car signed lateral cue (`HZ-PERCEPT-01`)

- `source_ref`: `docs/handoffs/completion-first-discovery-20260926/1-0afbd31b55ea.md`; `haic_agent/pixel_features.py`; `haic_agent/networks.py`.
- `rule_or_requirement`: submitted actions must come from the pixel-observation policy; maps and simulator state are not runtime inputs.
- `observable_information`: the current 84×84 grayscale observation, the detected obstacle center, and the fixed image-space ego reference.
- `allowed_action_or_state_change`: add one signed pixel-derived cue `clip((obstacle_center_x - 42) / 12, -1, 1)` when the existing detector finds an obstacle; otherwise use zero. Positive means obstacle is right of image center. No map, collision label, or simulator state enters this feature.
- `expected_success_endpoint`: higher V2 completion rate than `fresh_control` on identical evaluation cells, without violating the actor-only runtime contract.
- `eligible_state`: visible-object detections from permitted TRAIN pixels only.
- `control`: same input schema and actor architecture, with this cue zeroed; same seed, data, optimizer, warm-up and PPO budget.
- `falsifier`: cue fixture sign/scale is wrong; cue never activates on TRAIN pixels; paired action response is indistinguishable on activated TRAIN states; or matched completion rate does not improve.
- `smallest_decisive_experiment`: a TRAIN-only feature/paired-action activation report followed, only if active, by the registered tune batch.
- `resource_and_risk_gate`: pixel-only at runtime; fixed cue; no threshold sweep; profile time remains below official 5-second act limit.

### Candidate 2 — obstacle approach-rate cue (`HZ-PERCEPT-02`)

- `source_ref`: `docs/handoffs/completion-first-discovery-20260926/2-ee0cf2563e29.md`; `haic_agent/pixel_features.py`; `env_wrapper.py`.
- `rule_or_requirement`: temporal sensing may use only the submitted agent's stacked grayscale frames.
- `observable_information`: the nearest bright obstacle candidate in the newest two frames of the `(4,84,84)` pixel stack.
- `allowed_action_or_state_change`: add one signed-free approach feature `clip((y_now - y_previous) / 8, 0, 1)`. Match the same nearest candidate only when both detections exist and their x-centers differ by at most 12 pixels; otherwise emit zero. This is an image-motion cue, not a time-to-collision estimate.
- `expected_success_endpoint`: higher V2 completion rate than `fresh_control` on identical evaluation cells.
- `eligible_state`: consecutive TRAIN pixel frames with a valid matched obstacle detection.
- `control`: same input schema/architecture with this slot zeroed; same seed, data, optimizer, warm-up and PPO budget.
- `falsifier`: fixture direction or clipping is wrong; the cue is inactive on TRAIN frames; the paired action response is indistinguishable on activated states; or matched completion rate does not improve.
- `smallest_decisive_experiment`: TRAIN-only motion fixtures and paired-action activation report, then one fixed tune comparison if active.
- `resource_and_risk_gate`: no tracking over multiple objects, no privileged state, no tune-set threshold adjustment, and runtime latency measured against the official limit.

### Candidate 3 — expanded throttle range (`PPO-PEDAL-RANGE-COMPLETION-01`)

- `source_ref`: `docs/handoffs/completion-first-discovery-20260926/3-4f55118bea37.md`; `haic_agent/networks.py`; `agent.py`; `training/train_policy.py`.
- `rule_or_requirement`: preserve finite bounded actions and compare completion before speed.
- `observable_information`: identical actor inputs and action means, with gas output after the registered transform.
- `allowed_action_or_state_change`: set throttle expansion to exactly `3.5`, keep brake expansion at `1.0`, and leave weights/features/reward unchanged. Do not sweep other values.
- `expected_success_endpoint`: completion rate is no lower than both fresh control and SOTA reference; on completion tie, registered lap-time tie-breakers improve.
- `eligible_state`: actor-only model with positive longitudinal demand on clear-road TRAIN observations.
- `control`: identical fresh actor training protocol at throttle expansion `1.0`.
- `falsifier`: the transform does not produce a measurable gas increase on eligible states; completion declines; or a completion tie has no lap-time improvement.
- `smallest_decisive_experiment`: replay fixed TRAIN-only pixels through both transforms to prove activation, then a single matched tune comparison.
- `resource_and_risk_gate`: fixed `3.5`; record action range, collisions and damage; no faster-lap result advances if completion falls.

Historical context: `RESULTS.md` records a small custom-map throttle screen where expansion `3.5` tied the baseline completion count and was faster on recorded finishes, with more collisions/damage. That row is historical summary evidence, not revalidated here and not a matched official-track result. This cycle changes initialization, TRAIN coverage, training budget, and registered map/seed evaluation; it will test whether the effect generalizes rather than claim a new optimum.

### Candidate 4 — curve-conditioned brake reward (`PPO-CURVE-BRAKE-01`)

- `source_ref`: `docs/handoffs/completion-first-discovery-20260926/5-curve-brake-alternative.md`; `haic_agent/pixel_features.py`; `training/train_policy.py`.
- `rule_or_requirement`: keep the shared actor architecture and training data fixed while isolating one training-only reward mechanism.
- `observable_information`: pixel-derived `road_curve_magnitude` and normalized brake action.
- `allowed_action_or_state_change`: add exactly `6.0 * road_curve_magnitude * normalized_brake_fraction * reward_scale` to PPO reward. Keep the existing curve-speed relief and all other reward terms fixed. The feature is not a runtime override.
- `expected_success_endpoint`: higher V2 completion rate than the architecture-matched control on identical cells.
- `eligible_state`: visual-feature actor and permitted TRAIN-only curve observations.
- `control`: same fresh initialization, architecture, feature inputs, optimizer, map/seed schedule, warm-up and PPO budget; curve-brake coefficient `0.0`.
- `falsifier`: reward contribution is always zero; TRAIN-only curve-bin brake behavior does not differ by the registered activation threshold; or matched completion rate does not improve.
- `smallest_decisive_experiment`: TRAIN-only reward/action activation report, then one matched tune comparison if active.
- `resource_and_risk_gate`: coefficient `6.0` is fixed before training; unnecessary braking and lap time are recorded; no coefficient sweep.

This replaces the more costly CEM uncertainty candidate for this batch. DAgger and CEM remain separate later-cycle ideas. The parallel chat comparison recommended curve braking as the fourth arm based on the current hypothesis handoffs; no performance gain is inferred from that recommendation.

## Shared training protocol

- Training seeds: `20261001` and `20261002`; each seed trains the fresh control and all four candidates from a newly initialized actor with the same per-seed initial weights. Seeds are independent training runs, not map-seed labels.
- TRAIN data: only the `train` group of `training/maps/site/full_site_map_split.json`. This includes the user-authored obstacle and training maps and their listed train-only augmentations. No `tune`, `held_out`, `confirmation`, `blind`, or official evaluation cells enter teacher labels, PPO rollouts, normalization, feature calibration, or hyperparameter choice.
- Teacher warm-up: the same frozen bank of exactly 8,192 teacher-labeled pixel/action examples per training seed, with 3 supervised epochs, batch size 32, learning rate `1e-3`. The teacher observes pixels only and exists only in training code.
- PPO: exactly 65,536 environment decisions per model, divided into 64 updates of 1,024 decisions each, learning rate `3e-4`; keep remaining PPO settings byte-identical in the run plan. No mid-run tune check or checkpoint selection; save the final requested update.
- Total: 10 fresh PPO actors (5 arms × 2 training seeds), 655,360 PPO decisions, plus at most 16,384 TRAIN-only teacher collection decisions for the two frozen banks (reused across the five arms for each seed). CPU-only, one training process at a time. Maximum 3,600 seconds per actor, 1,800 seconds per demonstration bank, and 39,600 seconds aggregate training/collection wall time. A timeout or incomplete budget invalidates the batch; do not silently extend or replace an arm.
- User-map smoke/activation is diagnostic only. It cannot enter candidate completion rates or SOTA promotion.

## Exact comparison sets and endpoints

- **Tune:** official track IDs `1` and `2`, each with deterministic environment seeds `132, 133, 134, 135` (8 matched cells). Evaluate all 10 fresh actors and the explicitly named SOTA reference on every cell. Select at most one strict winner under the registered internal ordering; a tie with control does not open held-out.
- **User-map diagnostic:** four cells from the two original authored TRAIN maps only: `custom-track-haic-obstacles-20260920.json` with seeds `20260920, 20260924`, and `custom-track-haic-train-20260921.json` with seeds `20260921, 20260925`. Evaluate all 10 fresh actors and the SOTA reference. Re-run only as in-sample diagnostics; never mix these cells into tune/held-out rates.
- **Held-out:** official track IDs `1` and `2`, each with deterministic seeds `136, 137, 138, 139` (8 matched cells). Evaluate only the tune winner's two independently trained actors, the two matching fresh-control actors, and the SOTA reference. No retuning or candidate replacement after opening this set.
- **Reserved:** seeds `140–143` remain confirmation and `144–147` remain blind; do not open them in this cycle. Any follow-on use needs a new revision and fresh approvals.
- `max_decisions=2,000` per episode. The evaluator records raw episode rows first and derives all summaries from those rows. Candidate action/reset/invalid failures remain in the V2 denominator as non-completions; environment/infrastructure faults invalidate the comparison. Record the official-like episode ordering separately from the V2 internal ranking.
- A candidate can advance only if it completes all 16 held-out episodes, is non-inferior to both the fresh control and SOTA reference on completion, and shows a strict improvement under the registered ordering with the same direction for both independent training seeds. Faster laps never compensate for a DNF. If SOTA also completes all matched cells, the candidate must improve the registered finisher time ordering; if SOTA has a DNF, a 16/16 candidate has already cleared the completion priority, and lap time is then compared among finishers. Under-13-second rate is reported after this gate and cannot override it.
- A SOTA pointer changes only after a valid `ADVANCE`, all three gates are `PASS`, and the exact historical SOTA reference was evaluated on the same registered cells. Otherwise preserve the existing historical entry.

The selected seeds are beyond the maximum official seed present in the currently registered local split manifests. The execution plan must recheck only the explicit V2 experiment registry and registered split identities for collision before approval; it must not walk old run or artifact directories. Tune, user-map diagnostic, and held-out evaluation caps are 3,600, 1,800, and 1,800 seconds respectively. Maximum evaluation wall time is 7,200 seconds; the full cycle is capped at 46,800 seconds (13 hours), one CPU process at a time.

## Harness and reporting changes proposed for the implementation stage

These changes remain unapproved until a separate implementation approval is recorded against this design hash:

1. Add typed V2 train/evaluate profile arguments for the four fixed interventions, the selected split groups, actor-only checkpoints, exact run IDs and resource caps. Keep the CLI plan-only by default; expose no arbitrary shell or free-form command.
2. Keep new checkpoints, maps generated for these registered cells, raw episodes, `run_manifest.json`, append-only `events.jsonl`, and `integration_report.json` inside their configured V2 run/artifact roots. Bind every source, split, map, teacher bank, checkpoint, and runtime file by SHA-256.
3. Allow an explicitly supplied read-only historical SOTA checkpoint reference at only its `SOTA.md` path and expected SHA-256; do not scan for artifacts, copy the checkpoint, or use it to initialize any model. A missing/mismatched reference blocks the score comparison.
4. Make actor-only evaluation accept a missing dynamics checkpoint when planning is disabled. Align local action handling with the pinned official participant contract: clip finite in-range-contract violations as the official wrapper does; handle malformed/non-finite output according to the official invalid-action rule; enforce reset/action deadlines and retire after the registered invalid-action streak. Measure lap time from the end of the 50-raw-frame warm-up through the 50-FPS simulated finish tick.
5. Derive metrics only from immutable per-episode rows. Keep the current internal completion-first ordering separate from the official-like finish-time/progress ordering. Correct or disable legacy routes that can create an alternate score summary.
6. Update `RULES.md` as a router to the V2 `plan → approve → run → report` path. Remove any instruction that automatically scans legacy runs, papers, PDFs, submissions, or rewrites `RESULTS.md`/`SOTA.md`. Preserve the file and all historical sources.

Do not modify `env_wrapper.py`, `damage.py`, `core/`, the official Participants environment, or official submission code except for a narrow local evaluator contract adapter approved in this cycle. Do not package, upload, confirm a model, or contact the competition site.

## Activation, outcome, and stop gates

- `mechanism_activation`: report `PASS/FAIL/UNKNOWN` independently for every candidate before opening tune. HZ1 and HZ2 each require at least 128 unique cue-positive TRAIN observations and a paired action difference of at least `0.01` in one action component on at least 5% of eligible same-state observations. Throttle requires gas to increase by at least `0.01` on at least 5% of eligible positive-demand clear-road TRAIN observations. Curve braking requires at least 128 non-zero reward deltas and a normalized-brake median shift of at least `0.01` in fixed TRAIN curve bins. Activation failure excludes that candidate before tune; it does not justify a parameter sweep or a three-direction batch.
- `rule_compliance`: check official action/observation/runtime behavior against the current official repository and local mirror, actor-only export, no map/state/teacher at inference, package contract, and run-path/hash protections. `UNKNOWN` is not a pass.
- `competitive_or_product_outcome`: derive from matched held-out raw rows and the registered control/SOTA identities; do not use papers, proxy reward, one checkpoint, or external rank claims as performance evidence. Each model's final 8 PPO updates are also checked against its preceding 8-update window. If either completion fraction increases by at least `0.05` or mean normalized progress increases by at least `0.05`, mark the training budget unresolved and do not promote from this cycle; make a revised budget plan.
- Every outcome transitions through its matching gate-review state and records exactly those three gates. Only `GATE_REVIEW_ADVANCE` with all three `PASS` may release; release does not authorize external submission.
- Any infrastructure-invalid comparison stops without a winner. Three consecutive valid comparable non-improving cycles, as defined above, trigger a pivot while preserving checkpoints and run records.

## Known evidence, unknowns, and source paths

### Facts

- The pinned Participants README describes `(4,84,84)` grayscale observations, `[steer,gas,brake]`, the 10-second init / 5-second reset / 5-second action / 1,024-MB limits, and official episode ordering by finish time then incomplete progress; execution failures are excluded. The website was inaccessible on 2026-09-26; the official source precedence remains in `AGENTS.md`.
- The repository currently has train-only user-authored maps and four already recorded independent directions. The V2 train allowlist does not expose every candidate flag; the design-stage draft records that no new plan hash, exact split, denominator, or approvals existed before this cycle.
- Existing SOTA is marked historical and unvalidated by V2 gates; its explicit checkpoint path and digest are listed above.
- The current candidate ideas have no result from this proposed matched official-seed batch. The earlier custom throttle summary is not revalidated and is not evidence of generalization.
- Four read-only parallel conversations completed and are idle: official score/harness audit (`01a0e28e-a7b1-7f33-9e18-12fdf3c118d8`), visual obstacle strategy (`01a0e28e-b0d0-71c0-b0c2-62f808d0bddd`), speed/action strategy (`01a0e28e-baca-7cc0-b758-bb0c7214d662`), and the fourth-candidate comparison (`01a0e2ae-8696-7762-ab24-f59bdb2dbd40`). They completed their read-only research tasks; they did not drive episodes, and no subagents edited files. Their outputs are design inputs, not performance evidence.

### Unknowns

- Whether the exact SOTA reference bytes are still present at their recorded path. Only an explicit path/hash check is allowed before a later run plan.
- Whether the added pixel cues activate reliably on the site obstacle maps, alter actor behavior, and improve matched completion.
- Whether 65,536 fresh PPO decisions after fixed teacher warm-up are sufficient for learning to stabilize. Record train progress across all 64 updates; if the final 8-update window improves completion fraction by at least `0.05` or mean normalized progress by at least `0.05` over the preceding 8-update window, mark the performance comparison inconclusive and revise the plan instead of declaring a winner.
- Whether local deadline/invalid-action behavior can exactly reproduce the current official runner; the website must be rechecked before any external action.

### Source paths

- `AGENTS.md`
- `PROJECT_INFO.md`
- `RESTRICTIONS.md`
- `docs/report.md`
- `docs/context/current-state.md`
- `docs/handoffs/completion-first-discovery-20260926/INDEX.md`
- `docs/handoffs/completion-first-discovery-20260926/coordinator-preflight-20260926.md`
- `docs/handoffs/completion-first-discovery-20260926/battlecode-style-tournament-draft.md`
- `docs/handoffs/completion-first-discovery-20260926/1-0afbd31b55ea.md`
- `docs/handoffs/completion-first-discovery-20260926/2-ee0cf2563e29.md`
- `docs/handoffs/completion-first-discovery-20260926/3-4f55118bea37.md`
- `docs/handoffs/completion-first-discovery-20260926/5-curve-brake-alternative.md`
- `docs/sources/INDEX.md`
- `docs/sources/official-participants/README.md`
- `docs/sources/legacy-path-map.md`
- `SOTA.md`
- `RESULTS.md`
- `training/maps/site/full_site_map_split.json`
- `training/train_policy.py`
- `training/evaluate_closed_loop.py`
- `training/evaluate_tournament.py`
- `training/site_maps.py`
- `haic_agent/pixel_features.py`
- `haic_agent/networks.py`
- `agent.py`
- [DAgger theory, Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html) — supports the learner-induced-state labeling hypothesis only; it is not HAIC performance evidence.

## Approval boundary

The exact saved SHA-256 of this document must receive **design approval** before writing the implementation plan or changing code. That approval does not approve implementation or execution. The implementation and the experiment batch each require a separate fresh approval tied to their exact immutable plan hashes. No package, official model confirmation, or website upload is in scope.
