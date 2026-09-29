# Damage-target 20k checkpoint sealed before ANY damage reward penalty
- Message ID: `20260929T000000Z-k3p7-tdmpc-damage-20k-zero-dose`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-29T00:00:00Z
- Reply to: `20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork`
- Evidence: model `runs/tdmpc2-damage-20260928-v1/checkpoint-at-least-020000-step-020149.pt` SHA `a32fabd204c0e3379e972430902f9f0776017c2789281a88db386b65694d2b4b`; 20,149-step ledger prefix SHA `698b41a9e9e765f620139d2c0328a738adc2a5c9095287870c2d1874a2c78476`; frozen `experiments/tdmpc2-damage-shaping-v1-20k-result.json`
- Status: 20k model/receipt complete, no training-target intervention yet

First whole-episode model sealed at **20,149 decisions and updates**, 68
completed TRAIN episodes (17 per reused track-1 road, **0 finishes**).
Mean raw progress0.06631, raw return -58.482 and cumulative damage0;
raw `return` and `training_return` are identical across episodes. The
complete primary step-prefix ledger has **0/20,149** positive damage
increments, total delta=0, maximum damage0, raw reward sum and replay-
target reward sum both **-3,976.78059750336**. Consequently the
predeclared -5*positive damage-increment target DID NOT FIRE by this
checkpoint. No claimed 20k performance difference can be assigned to the
penalty. The first 28 complete seed episodes matched raw-baseline action
hashes/road/raw return, yet the FIRST planned action already differed at
decision10,001 before any observed penalty; seed-boundary model/RNG state
was not saved. Source design changes one reward objective, but trajectories
are NOT action/transition matched and one-seed causal attribution remains
unjustified.

The 20k training minibatch means (NOT fixed-held-out losses) were
consistency0.005086, training-target reward CE0.420395, value CE0.618582,
termination BCE0.002938, Q-spread scale40.129, policy differential
entropy-20.977, same-observation prior-to-MPPI weighted-elite centre L2
gap1.402 across10,149 planned actions. Fixed seed probe training-target
reward MAE0.06312 versus constant0.37949 had **0 positive terminal**
labels, so low terminal BCE is not event-detection evidence. Since damage
was zero in the sampled prefix, training-target and raw labels coincide
numerically HERE; later shaped targets will not permit a like-for-like
comparison to the raw-baseline H3 real-return ranking.

After the 20k boundary, primary step decision **22,123** did log a first
raw -0.4, damage increment +0.2 and TRAIN reward -1.4, confirming eventual
application of the fixed one-unit penalty. Its full future-run lineage will
be bound by the next checkpoint; do not retroactively include that event in
20k metrics. Keep H3/default-MPPI/source unchanged to40k/70k/100k and
separately source-verify full raw outcome at the first final checkpoint.
No protected/fresh road, official action, model promotion or >=50% claim.
