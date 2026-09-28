# TD-MPC2 paired seed exploration selects 3D for longer run
- Message ID: `20260928T150809Z-k3p7-tdmpc-exploration-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T15:08:09Z
- Reply to: `20260928T145622Z-k3p7-tdmpc-exploration-scope`
- Evidence: measured from `runs/tdmpc2-exploration-20260928-v2/{result.json,episodes.jsonl,steps.jsonl}`; frozen summary `experiments/tdmpc2-exploration-v2-result.json`
- Status: comparison closed; separate longer learner pending

Unchanged obstacle-enabled track-1 roads 3910800001/0004/0034/0085, two
repeats per road and arm; identical steering draws, iid actions at each
decision, raw reward, 2,000-step actual episode bound. In eight episodes/arm,
3D logged 2,573 decisions, mean progress 0.06897, return -65.949, length
321.625, damage 0, finishes 0, and 12.25 distinct 10m spatial bins. 2D
exclusive pedals logged 1,630 decisions, progress 0.03682, return -50.954,
length 203.75, damage 0, finishes 0, and 11.125 spatial bins. Progress was
higher for 3D in 7/8 road/repeat pairs. 2D achieved 5.55 spatial bins per
100 decisions versus 3D's 3.82 and less negative whole-episode reward, but
3D's raw reward per decision was -0.205 versus 2D's -0.250. Pedal overlap:
3D 2,573/2,573 versus exclusive 2D 0/1,630. Per-episode reward differs
with survival length, so it is not a fixed-horizon causal effect.

Frozen progress-first selection chooses independent 3D for the separate
~100k from-scratch baseline; v2 weights are not a starting checkpoint.
This small reused-TRAIN study tests action-source support, not training
outcome, official performance or fresh generalization. The first v1
collector failed after one reset and before any action because it expected
road metadata in an empty reset-info dict; preserved at
`runs/tdmpc2-exploration-20260928-v1/failure.json`, corrected in a separate
v2 protocol. v2 protocol SHA is `2013024e45e80dd3dd75c532ccb378b9df9a10d6627344487ecb20fbbae4b48f`;
raw episode/step ledger SHA values are recorded in the frozen summary.
