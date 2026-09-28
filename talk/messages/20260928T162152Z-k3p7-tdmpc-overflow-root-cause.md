# TD-MPC2 100k partial was diagnostic mean overflow, not applied action
- Message ID: `20260928T162152Z-k3p7-tdmpc-overflow-root-cause`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T16:21:52Z
- Reply to: `20260928T161418Z-k3p7-tdmpc-long-v1-partial-failure`
- Evidence: original run ledgers; source adapter/planner; source-and-synthetic forensic; `experiments/tdmpc2-long-reused-train-v1-failure.json`
- Status: likely mechanism reproduced; new run not yet started

Independent forensic audit decoded all 10,020 applied model/native actions in
the original partial step ledger: they are finite, in bounds, and consistent
with the official adapter. The first unrecorded next action cannot be
reconstructed from the non-resumable run. The *only* unchecked input to that
adapter in the runner is the diagnostic copy of `planner.prev_mean[0]`; both
the applied planner action and prior mean are bounded. Synthetic float32
weighted-elite normalization reproduced numerator 1.0 / denominator
0.9999999403953552 = 1.0000001192092896 and the exact observed ValueError.
The specific failing mean is unobserved, so root cause is a strongly
supported mechanism, not a direct measurement of that unlogged value.

Corrected runner SHA-256 `a7d8a20171edf50ff97a7dd04a82164590f79205e8f150ed73d766c9576d5236`
rejects nonfinite or >1+1e-6 weighted means, clips **only its diagnostic
copy** by one-ULP tolerance, and logs raw-max/clipped flags; planner state and
actual applied environment action are unchanged. Injected nextafter(1,2)
and material 1.01 overflows are covered, and 60 TD-MPC2 regression tests pass
without any environment reset. The original 10,020-step source/ledger and
protocol remain frozen at old source SHA `b20f7f05...`; a separately frozen
retry protocol/path and pre-reset gates are still mandatory. The 28 completed
episodes were seed-exploration episodes (0 finishes), not trained-policy
evidence. Four reused TRAIN roads cannot prove the user's >=50% goal.
