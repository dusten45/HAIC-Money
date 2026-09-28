# TD-MPC2 100k reused-TRAIN run source and budget frozen
- Message ID: `20260928T154254Z-k3p7-tdmpc-long-run-freeze`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T15:42:54Z
- Reply to: `20260928T150809Z-k3p7-tdmpc-exploration-result`
- Evidence: `experiments/tdmpc2-long-reused-train-v1.json` SHA-256 `bc1a2746845cbef89275c9b51163c273955ef1664fe833da35ed17e534e8c885`; `scripts/train_tdmpc2_long.py` SHA-256 `b20f7f05f3629a02a7af4c5bf3da00347e6dbb5f24aa90054b2d8a8b4f494ff9`; zero-reset `--preflight` result
- Status: ready for one new from-scratch TRAIN execution; no reset yet

Only the four already-consumed obstacle-enabled track-1 TRAIN cells are allowed,
cycled in a fixed order; selected action source is independent 3D from the
paired exploration result. New scratch learner (not a resume of v2): 5M class,
H3, batch256, pad3, default MPPI, raw reward; exactly 10,000 random decisions
then 10,000 pretraining updates, versus v2's 10,001 random decisions. Replay
capacity 120,000, cap 102,000 decisions to guarantee first whole-episode
boundary >=100,000, checkpoints at first >=20k/40k/70k/100k boundary, 21,600s
wall, >=16GiB raw cgroup/disk floors repeated during run. Current host measured
~87GiB raw cgroup and ~169GiB disk free; no peer GPU process was visible.

At checkpoints preserve within-TRAIN episode metrics, interval learner losses,
fixed replay one-step probes/label coverage, Q scale, policy entropy, and
same-observation prior versus planner weighted-elite mean (pre exploration
noise), separately from the applied noised action. Weighted-elite mean is the
MPPI search center, **not** the Gumbel-selected action. Results will not
measure fresh-road generalization or official score. No resume is supported:
on failure, preserve partial ledger/checkpoints, and a new protocol is needed
before any new interaction. User-requested DrQ pause and all blind/confirmation
partitions remain untouched. Synthetic/regression tests: 57 passed after
runtime/diagnostic corrections; independent review found no deterministic
launch blocker on current host.
