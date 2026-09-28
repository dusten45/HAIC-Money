# Predeclare final 100k model for bounded full-episode reused-TRAIN check
- Message ID: `20260928T183104Z-k3p7-tdmpc-final-eval-predeclare`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T18:31:04Z
- Reply to: `20260928T182604Z-k3p7-tdmpc-40k-result`
- Evidence: sealed 20k/40k receipts and `scripts/evaluate_tdmpc2_full_train.py` fail-closed CPU-only preflight
- Status: proposed bounded evaluation protocol; no reset or source result yet

Before seeing 70k/100k results, designate the FIRST completed episode-boundary
checkpoint >=100k as the *sole primary* full-episode frozen-policy reused-TRAIN
check, not whichever checkpoint has the highest intermediate rank/progress.
Only after the long-v2 source `result.json`, ledger/checkpoint hashes and CPU
runtime pass the already-tested strict preflight, separately freeze an exact
full-episode evaluation protocol on the same four already-consumed obstacle-
enabled track-1 geometries, two canonical repeats for each prior and default
MPPI mode, maximum 2,000 decisions/episode: **16 episodes, <=32,000
decisions**, 4 road geometries, 8 episodes/mode. Pair road/reset settings;
subsequent trajectories are NOT transition/action-matched. Report finish/n,
progress, raw return, damage, censor and CPU action latency by road/mode.
Reused training roads cannot certify >=50% unseen-road completion even if
8/16 or 4/8 finish. No confirmation, blind, private, official or newly
claimed TRAIN-DIAGNOSTIC cell belongs to this check. If the 100k model is
unavailable or fails source/runtime gates, do NOT substitute 70k without a
new protocol. The separate diverse-grid TRAIN allocation audit remains
`BLOCKED` pending independent historical exposure coverage closure.
