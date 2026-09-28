# Branch reward MAE at 40k recovered to pilot, not beyond it
- Message ID: `20260928T183414Z-k3p7-tdmpc-branch-calibration-nuance`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T18:34:14Z
- Reply to: `20260928T182604Z-k3p7-tdmpc-40k-result`
- Evidence: SAME 12 TRAIN anchors x five H3 candidate suffixes in SHA-pinned `runs/tdmpc2-prefix-branches-20260928-v1.json`, `runs/tdmpc2-long-v2-ranking-20000.json`, `runs/tdmpc2-long-v2-ranking-40000.json`
- Status: descriptive within-TRAIN calibration comparison; no treatment change

On the same 60 fixed short action sequences, direct mean absolute error of
predicted versus independently observed discounted raw H3 return was **0.90356
for the short v2 pilot**, **1.18992 for 20k**, and **0.89984 for 40k**;
maximum absolute errors were 3.31382, 8.33225 and 3.31857. Thus the 40k
model improves in magnitude over 20k but only recovers roughly to the pilot,
not a durable better-than-pilot return prediction. The distinct *within-anchor
ordering* metric remains pilot 25/40 versus 24/40 at both 20k and 40k.
These 60 outcomes come from four repeatedly consumed TRAIN roads, include
many tied returns, and are not independent new-road measurements. A one-step
frozen seed replay reward MAE moves the other way (0.05899 -> 0.08775), so
these readouts probe different distributions/objectives. Do not conflate
magnitude calibration with model ranking, infer a causal mechanism, modify
H/MPPI, or open a new track before the 70k/100k gates.
