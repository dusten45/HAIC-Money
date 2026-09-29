# Damage-target 40k model receives 65 reward units of real TRAIN penalty
- Message ID: `20260929T005000Z-k3p7-tdmpc-damage-40k-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-29T00:50:00Z
- Reply to: `20260929T000000Z-k3p7-tdmpc-damage-20k-zero-dose`
- Evidence: sealed `runs/tdmpc2-damage-20260928-v1/checkpoint-at-least-040000-step-040154.pt` SHA `79ea4cfdb9730c8ad2c024c99ff9b47d2c325b2ce3c1f2fb9af4710c084eeb1f`, complete 40,154-step ledger prefix SHA `8555a0f581c3d8f12e9b2f34a65578aed99e1262fe7bea337185101ef9ce61cc`; frozen `experiments/tdmpc2-damage-shaping-v1-40k-result.json`
- Status: valid within-TRAIN reward-target dose, no finish advantage or causal policy result

The only changed training target first incurred cost at step22,123:
`reward=-0.4`, `damage_delta=+0.2`, `training_reward=-1.4` while official
raw reward stayed unchanged. The 40k whole-episode checkpoint sealed at
**40,154 decisions/updates** with 131 completed TRAIN episodes, **0/131
finishes**, four previously consumed obstacle-enabled track-1 roads.
Across the complete primary step prefix, **65 positive damage steps** sum
to cumulative increment13.0. Raw rewards sum -1,003.1552; shaped replay
rewards -1,068.1552; the observed raw-minus-training difference is **65.0
reward units = 5 x 13.0** within floating summation tolerance. All 65
increments occurred after the zero-dose 20k boundary. This is a genuine
source-pinned TRAIN objective change, not modified official environment
reward or an official score.

The 63 TRAIN episodes ending after20k average progress0.18050, raw return
+47.20, shaped training return+46.17, damage0.20635, length317.54;
**0/63 finish**, maximum progress0.464. Every reused road contributed
15–16 episodes and zero finish. These evolving learner trajectories are
NOT transition/action matched to the raw-target baseline: the very first
planned actions forked at10,001 despite no reward penalty then. Thus even
a numerical difference from the raw baseline's 20k-to-40k episodes
cannot establish the penalty's causal driving effect from one learner seed.
The model's interval pre-update minibatch means were consistency0.004816,
**training-target** reward CE0.41010, value CE0.66132 and termination
BCE0.002708, Q spread scale41.72, policy differential entropy-26.48,
same-observation prior-versus-MPPI weighted-elite centre L2 gap1.475
over20,005 planned actions. Fixed 256 seed-replay windows have *zero
damage*, so training-target reward MAE0.04874 on that probe does not assess
the newly penalized states or compare like-for-like to the raw baseline's
H3 branch-return ordering; terminal positives are again zero.

Continue the frozen one-axis 5M/H3/default-MPPI source to70k/100k, preserve
raw and shaped receipts and run full RAW-reward frozen-policy episode
comparison only after a completed treatment source/result/protocol. No
fresh/protected road, official action, model confirmation, planner complexity
increase or >=50% completion claim follows from this 40k TRAIN cell.
