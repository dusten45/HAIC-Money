# Policy-Mean Learning-Rate PPO Screen

> Execute inline in the current root checkout. Keep PPO actor-only. Do not request reviews or subagents.

**Goal:** Test whether the PPO actor is failing to learn stronger deterministic actions because the policy-mean head barely moves at the current learning rate.

**Evidence:** In the obstacle-brake reward 6-vs-12 screen, four U8 actors moved the policy-mean head only 0.078%-0.114% relative L2 from the shared initialization. On the same 74 urgent-visible TRAIN images, brake-active fractions stayed between 0.405 and 0.432. Reward 12 completed 0/4 Tune episodes versus 2/4 for reward 6, with no under-13 lap. The prior global LR 1e-5 experiment underperformed 2e-6, so this test raises LR only for `policy_mean.weight` and `policy_mean.bias`.

**Single variable:** Policy-mean head learning rate: 2e-6 control vs 1e-5 treatment. Encoder, value and auxiliary heads, `policy_log_std`, and all other optimizer groups stay at 2e-6.

## Fixed Protocol

- Starting actor: strict model-state-only load from SHA-256 `3CEB5EBBE2A4BD96944649A693B8EF924C0252FA620B0FD547C7FB494AFFF199`; fresh Adam for every arm.
- PPO seeds: 8104 and 8105; crossed with the two policy-mean learning rates.
- Reward coefficients fixed in every arm: obstacle brake 6, curve brake 6; all other reward values unchanged.
- 8,192 decisions / 8 updates per arm, base learning rate 2e-6, teacher warmup 0, speed target 70, shortfall penalty .6.
- Load exactly 4 custom TRAIN and 2 custom TUNE episodes from the registered split; do not load held-out or official maps. Defer all TUNE until every arm completes and passes the checkpoint/source integrity gate.
- Preserve the same 1,040-image deterministic TRAIN observation bank and U2/U5/U8 actor snapshots. Record policy-mean and full-actor parameter drift from initialization.
- TUNE has two seed labels on one geometry. Report it as a repeated-geometry mechanism screen, not independent-map generalization. Do not update SOTA, RESULTS, package, or submit from this screen.
- Wall-time limit: 7,200 seconds. Stop before training if hashes, split counts, arm uniqueness, or optimizer-group rates do not match.

## Acceptance and Rejection

- Confirm control behavior matches the existing global 2e-6 default and treatment changes only the two policy-mean parameter group rates.
- Treat increased fixed-bank urgent-state brake response as a mechanism signal only. Promote the treatment to a broader allowed TUNE screen only if completion/damage do not regress and the treatment produces a measurable policy-mean/action change across both PPO seeds.
- Reject the treatment if it increases collisions, damage, off-track failures, or if policy/action movement remains within control variation. Under-13 completion remains the competition target; no result on this single TUNE geometry proves that target.

## Implementation and Verification

1. Add failing tests for optimizer parameter grouping, fallback/default equivalence, invalid head learning rates, and `train()` metadata/pass-through.
2. Add the optional policy-mean learning-rate path to `PPOConfig`, `PPOUpdater`, and `train()`, preserving existing callers.
3. Run focused PPO/trainer/screen tests.
4. Freeze a unique runner/config/source manifest, run preflight, then execute the paired four-arm PPO screen.
5. After the integrity gate, evaluate TUNE, inspect raw collision/brake traces and same-image policy actions, and record the result and next hypothesis in the strategy lab.
