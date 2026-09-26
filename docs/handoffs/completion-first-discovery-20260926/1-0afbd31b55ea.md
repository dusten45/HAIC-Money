# Handoff — HZ-PERCEPT-01

Assignment: completion-first-discovery-20260926 / owner-1; scope: pixel perception / learned PPO candidate.
Agent: `discover_perception`. Edited paths: none. Plan revision: `DISCOVER-2026-09-26`; plan hash and approvals: not assigned.

## fact

The current policy receives separate visual features for near-road center offset and obstacle lateral position relative to road center. Submission inference must rely on permitted image observations, and matched completion rate is the primary internal metric.

## inference

A signed obstacle-to-car lateral offset may help the learned policy judge whether an obstacle overlaps its current trajectory, especially when the car is already off-center. This mechanism and any effect on completion are untested.

## unknown

The frequency of obstacle/carlane alignment states preceding incomplete PPO episodes and feature utility are unknown. Training-profile registration and eligible evaluation cells are not yet specified.

## recommendation

Hypothesis fields:

- `hypothesis_id`: `HZ-PERCEPT-01`
- `source_ref`: this handoff; active source files below
- `rule_or_requirement`: pixels only; completion first on an identical preregistered comparison
- `observable_information`: obstacle-to-road-center offset plus near-road car offset
- `allowed_action_or_state_change`: add a signed pixel-derived obstacle-to-car lateral feature to PPO observation encoding; keep action selection unchanged
- `expected_success_endpoint`: higher matched completion rate than the same PPO setup without the feature
- `eligible_state`: visible obstacle in the road corridor with nonzero car/road offset; candidate trained on allowed data
- `control`: same initial model, training protocol, maps, seeds, denominator and evaluation conditions without this feature
- `falsifier`: no feature activation in eligible states, or no completion gain after activation (including a decrease)
- `smallest_decisive_experiment`: verify feature activation on fixed pixel observations; only after separate approvals, run one preregistered matched tune comparison
- `resource_and_risk_gate`: protect reserved splits, measure action latency and invalid actions, register resource budget

No threshold sweep or runtime evaluation is authorized by this discovery report.

## source_paths

- `haic_agent/pixel_features.py:49`, `haic_agent/pixel_features.py:153`
- `haic_agent/networks.py:241`
- `training/train_policy.py:172`
- `RESTRICTIONS.md:14`, `PROJECT_INFO.md:3`
