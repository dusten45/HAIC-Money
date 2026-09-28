# Frozen 100k TD-MPC2 prior and MPPI each finish 0/8 reused TRAIN episodes
- Message ID: `20260928T221857Z-k3p7-tdmpc-full-eval-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T22:18:57Z
- Reply to: `20260928T212858Z-k3p7-tdmpc-full-eval-preflight`
- Evidence: completed `runs/tdmpc2-full-train-20260928-v1/result.json` SHA `09b54dd849438f3a37bbfdf5c25f8448164e04f559056b57e5f8fb3ae641978c`, `episodes.jsonl` SHA `519bcd0b51aa13336128ae733be37a229e0c7b969fdb7e817db00f037395dfcf`; frozen `experiments/tdmpc2-full-consumed-train-v1-result.json`
- Status: primary four-reused-road frozen-policy gate complete, 50% not reached

The predeclared SHA-bound final 100k model and CPU Torch2.1 evaluator
completed **all 16/16 full 2,000-decision-scope TRAIN episodes**, four
already-consumed obstacle-enabled track-1 road geometries x2 canonical
repeats x2 modes `[prior,mppi]`. All **16/16 terminated naturally** before
the limit; 0 censored, no missing episodes, complete result/ledger hashes
and valid full-episode finish comparison. Prior mean finished **0/8**;
default MPPI finished **0/8**. Raw mean progress/return/damage were prior
0.327/+237.73/0.35 and MPPI 0.507/+385.93/0.45, respectively. Per-road
progress averages for MPPI exceeded prior on all four matched road/reset
*cells*, but actions immediately diverged; it is NOT a state/action matched
intervention, replicated learner-seed improvement or a finish advantage.
CPU mean action inference latency was 0.00193s prior versus 0.739s MPPI
(max 0.839s), internal runtime only, not official package compliance.

The separate from-scratch long-v2 TRAIN collector had **14/87 on-policy
post70k finishes** as its actor/model and planner exploration noise changed
online; that numerator must NEVER be substituted for a frozen final actor's
0/8 per mode. The final model's SAME in-development H3 reward-only ranking
31/40 versus pilot25/40 is a *model-learning directional signal*, not a
policy-level completion result. Zero frozen full-episode finishes on only
four repeatedly trained roads does NOT prove an intrinsic TD-MPC2 failure
or any fresh-road/generalization rate; it does fail the user's local >=50%
full-episode gate for BOTH frozen modes in this cohort. Do not select a
70k checkpoint post hoc, add H5 or MPPI samples together, tune on protected
cells, or claim an official/private-track score.

Next permitted axis, conditional on explicit new protocol/source/tests, is
ONE internal *training-target* reward-shaping hypothesis (damage increment
penalty) with official raw env reward, progress, damage and finish unchanged.
This is not a demonstrated cause or guaranteed fix; compare a separately
frozen treatment on the SAME consumed TRAIN roads, source-bound whole-
episode CPU checks and preserve the unmodified 100k baseline. Old
fresh-grid collision auditor remains BLOCKED and its newer uncommitted
extension has safety HOLD; no new training/evaluation road is opened here.
