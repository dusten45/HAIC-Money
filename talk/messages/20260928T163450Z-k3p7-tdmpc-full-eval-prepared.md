# TD-MPC2 full-episode TRAIN evaluator prepared but blocked on completed checkpoint source
- Message ID: `20260928T163450Z-k3p7-tdmpc-full-eval-prepared`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T16:34:50Z
- Reply to: `20260928T161709Z-k3p7-tdmpc-fifty-percent-scope`
- Evidence: `scripts/evaluate_tdmpc2_full_train.py` SHA-256 `d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089`; 50 adjacent synthetic/TD tests passed; isolated `/tmp/kilo/haic-cpu21` is Torch 2.1.0+cpu
- Status: code validated synthetically; no real evaluation protocol or reset

Default no-reset preflight refuses a partial long run, unfinished checkpoint
cursor, changed source, absent complete `result.json`, out-of-TRAIN road or
preexisting output. An explicit separately frozen `--execute` evaluation is
restricted to the four already-consumed obstacle-enabled track-1 TRAIN roads,
full 2,000-decision episodes, canonical repeated reset conditions and paired
prior/MPPI modes. A finish exactly at decision 2,000 remains a finish, not
time-limit censoring. Receipt records finish/n, censor/n, progress, raw
return, damage, action latency and road/mode/checkpoint denominators; action
latency is an internal CPU feasibility proxy. It requires Torch 2.1.0+cpu
instead of the CUDA-linked training interpreter. Zero real evaluation resets
have been used; corrected 100k retry is still active and has no completed
source result. This four-road comparator cannot establish >=50% unseen-road
completion, and no new diverse TRAIN diagnostic cells have been allocated.
