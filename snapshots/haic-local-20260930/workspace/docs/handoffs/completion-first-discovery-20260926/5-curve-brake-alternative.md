# Agent handoff: PPO curve-conditioned brake reward

Assignment, owner, scope, hypothesis ID, and plan revision/hash:

- Assignment: inspect one lower-friction replacement for the CEM uncertainty candidate.
- Owner: `alternative_candidate_discovery`.
- Scope: read-only source inspection and hypothesis formation; no code, data, checkpoint, or run changes.
- Hypothesis ID: `PPO-CURVE-BRAKE-01`.
- Plan revision/hash: none; this is not a registered plan.

## fact

- The pixel feature extractor exposes `road_curve_magnitude`, and PPO has a `--curve-brake-reward` shaping coefficient. The coefficient defaults to zero.
- The training harness profile does not currently expose `--enable-visual-features` or `--curve-brake-reward`.
- Curve observations already affect target-speed shortfall shaping through `CURVE_SPEED_RELIEF = 0.4`. The proposed candidate therefore adds a brake reward on top of existing curve-related shaping.
- A visual-feature actor has a different architecture from the default actor. A valid control for this candidate must use the same visual-feature architecture and initialization, with the curve-brake coefficient set to zero.

## inference

Increasing brake behavior when visible road curvature is high could reduce off-track failures and improve completion. This is a distinct mechanism direction from obstacle spatial features, obstacle temporal features, throttle action range, and CEM runtime planning. The mechanism is untested; no performance gain is known. Additional reward may also cause unnecessary braking.

## unknown

- Whether visible curvature predicts an upcoming loss of control early enough to brake.
- Whether the additional shaping changes normalized brake actions across curve bins.
- Whether any action change improves completion on matched evaluation cells.
- The reward coefficient and numeric activation threshold; neither is selected.

## recommendation

**Hypothesis record**

- `hypothesis_id`: `PPO-CURVE-BRAKE-01`
- `source_ref`: `haic_agent/pixel_features.py:142,153`; `training/train_policy.py:105,248,319,1946`; `haic_agent/networks.py:181,194`
- `rule_or_requirement`: optimize internal episode completion rate first; preserve official environment and rule-compliance constraints.
- `observable_information`: `road_curve_magnitude` and the actor's normalized brake action.
- `allowed_action_or_state_change`: PPO may increase brake action in higher-curvature states through an additional curve-weighted braking reward.
- `expected_success_endpoint`: higher completion rate than the architecture-matched control on the same registered evaluation cells and fixed denominator. No gain is established.
- `eligible_state`: visual-feature PPO policy; numeric coefficient fixed before tuning; activation assessed only on authorized TRAIN data.
- `control`: same visual-feature architecture, initialization, train split, seed, optimizer, and step budget, with curve-brake coefficient `0.0`.
- `falsifier`: keep it out of the tournament if the shaping term is inactive or candidate/control normalized brake behavior does not differ in preregistered curve bins. Reject the mechanism if completion does not improve on its registered matched comparison.
- `smallest_decisive_experiment`: first an approved TRAIN-only activation check; if it passes, one preregistered matched tune comparison against the architecture-matched control. Held-out confirmation requires a verified tune-winner binding and its own approval.
- `resource_and_risk_gate`: expose the two training options only after implementation approval; register a fixed compute/time cap and coefficient before execution. Watch for unnecessary braking, preserve `CURVE_SPEED_RELIEF` settings in both arms, and do not use historical runs or checkpoints as implicit controls.

## source_paths

- `haic_agent/pixel_features.py:142,153` — curvature feature extraction.
- `training/train_policy.py:105,248,319,1946` — curve speed relief, reward coefficient, visual feature settings, and CLI option.
- `haic_agent/networks.py:181,194` — visual-feature actor architecture.
- `harness.config.json:116` — current train-policy allowlist.
