# PPO budget continuation screen

## Question

Does additional on-policy PPO training help the failing actor brake, clear collisions, or gain pace when reward coefficients and policy architecture stay fixed?

## Evidence and hypothesis

- The selected SOTA checkpoint is saved at PPO update 4 of 8 because later TUNE updates did not improve its rank.
- The later recovery-clearance screen's seed 8105, coefficient 0 actor completed U8 but finished 0/2 repeated-geometry TUNE episodes. Both runs stalled near progress 0.753 after collision.
- Reward increases, a separate policy-mean learning-rate increase, and recovery-clearance shaping did not produce a safe TUNE gain. Fixed-image action changes remained small or unsafe.
- Hypothesis: the current actor has not received enough useful on-policy optimization to change its deterministic decisions. Continuing the exact failed U8 actor and Adam state for more PPO updates will either improve obstacle clearance and speed or show that budget alone is insufficient.

## Single-variable experiment

- Start from `artifacts/haic/recovery-clearance-reward-ppo-2seed-u8-20260925T095639KST/arm-seed8105-recovery-clearance-reward0e00/policy.pt` (SHA-256 `FF574E7CE75EEA3D473E078B04809EF7A87F9B03B9E43C717F45EFFB8BE3F9E3`). Resume both model and Adam state.
- Add 32,768 PPO decisions in 32 updates. Keep PPO seed 8105, learning rate 2e-6, gamma .99, GAE lambda .95, actor inputs and pedal scales, reward settings, current four TRAIN episodes, two TUNE episodes, and 600-decision episode cap unchanged.
- Keep TUNE deferred during training. Capture actor-only snapshots at each 8,192-decision boundary, then evaluate the exact two registered TUNE episodes once after training.
- Hold-out and official maps stay unopened for this screen. Do not update SOTA or package a submission from a TUNE-only result.

## Decision rules

- Compare against the same checkpoint's deterministic TUNE replay: finish rate, under-13 rate, collision onsets, damage, progress, lap time, and speed/action traces.
- A TUNE change that worsens collision, damage, or progress is rejected even if speed rises.
- Improvement to at least one TUNE finish with no safety regression justifies a separate held-out/official validation; it does not by itself qualify the actor as SOTA.
- If 32,768 additional decisions still produce no finish or useful action change, stop budget-only continuation and move to a distinct PPO representation/exposure hypothesis.

## Runtime contract

The submitted actor remains deterministic PPO actor-only. Teacher labels, simulator labels, map geometry, and privileged state remain TRAIN-only. No rule-based recovery or planner is added to inference.
