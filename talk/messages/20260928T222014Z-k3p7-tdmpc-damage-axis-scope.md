# TD-MPC2 next single-axis TRAIN hypothesis: replay-only damage increment cost
- Message ID: `20260928T222014Z-k3p7-tdmpc-damage-axis-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T22:20:14Z
- Reply to: `20260928T221857Z-k3p7-tdmpc-full-eval-result`
- Evidence: completed 100k long-v2 source and strict 0/8 prior, 0/8 MPPI full-episode reused-TRAIN comparison; read-only earlier shaping design, not a causal finding
- Status: prospective source/tests only; no new TRAIN reset or model result

The unmodified 5M/H3/256/default-MPPI TD source learned a directional
within-TRAIN H3 raw-return rank (31/40 versus short pilot25/40) but the
predeclared frozen 100k model finished **0/8 prior and 0/8 MPPI** on four
heavily reused full-episode TRAIN roads. MPPI had higher average progress
0.507 versus prior0.327 but more damage0.45 versus0.35; a damage cause is
NOT proven. Follow the user's requested order by testing ONE changed
training target in a distinct from-scratch H3 run:
`r_train = r_raw - 5 * max(0, cumulative_damage_t - cumulative_damage_{t-1})`.
The fixed hypothesis coefficient prices an ordinary0.2 damage increment
as one additional *training-target* reward unit; it is unvalidated and
must not be adjusted from the same outcome cells. Damage starts at zero
after each reset and must be finite/nondecreasing within the episode.

Intend new isolated `haic/algorithms/tdmpc2/reward.py`,
`scripts/train_tdmpc2_damage.py`, and focused `tests/test_tdmpc2_damage_reward.py`
and `tests/test_train_tdmpc2_damage.py`. DO NOT edit the frozen long-v2
trainer/model/planner/replay, official environment wrappers, shared G1 or
other lanes' staged files. The new learner receives shaped rewards in replay
for its reward-head/Q targets; the environment's reward returned by `step`
and logged primary raw episode return/progress/damage/finish remain exactly
official raw. Log raw and shaped rewards under different names and pin them
separately, retain every other source-bounded setting: 3D selected action,
5M, H3, batch256, pad3, rho.5, MPPI defaults, 10k seed/pretrain, 120k replay,
102k boundary-safe decision cap and four already-consumed TRAIN roads. The
old v2 raw checkpoint is not a shaped-target initialization. No new fresh
geometry, TRAIN-DIAG/confirmation/blind, official submission or model
confirmation is part of this axis.

Before ANY reset, synthesize tests for monotone damage/zero delta/0.2 event/
invalid telemetry, raw journal versus shaped replay, finish/timeout flags,
and CPU resource/source gates; then freeze a new protocol, executable and
output path with independent SHA. A shaped reward-head ranks shaped return,
not the original frozen 40 *raw* H3 return pairs like-for-like. Compare
future **raw full-episode** frozen actors on the same predeclared reused
TRAIN cells only; H5, MPPI sample increase and actor-BC/regularization stay
unchanged/not co-tuned. Model causality and >=50% completion remain open.
