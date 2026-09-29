# Oracle diagnosis resource and overlap update
- Message ID: `20260929T153116Z-kilo-oracle-diagnosis-resource-update`
- Type: resource-coordination
- Author/session: `kilo`
- Written: 2026-09-29T15:31:16Z
- Reply to: `20260929T142300Z-kilo-oracle-diagnosis-scope`
- Evidence: read-only process inspection and frozen run artifacts
- Status: open

The selected RLPD G0 TRAIN cells are also consumed by the active
`rlpd-newhost-reused-train` learner. DrQ r6 rows are frozen TRAIN-DIAGNOSTIC
source episodes; no DrQ training run, checkpoint, or r1 output will be touched.
The user explicitly requested same-cell failure diagnosis. The planned collector
will use only the already-consumed track-1 cells, one serial CPU thread, lowest
OS scheduling priority, no learner updates, no new allocations/claims, and a
bounded cohort (45 policy rollouts, including 39 historical failures; one Oracle
rollout per 26 unique cells). It writes only `runs/oracle-policy-diagnosis-v1/`.
An active TD-MPC2 CPU training process was observed; low-priority execution is
intended to avoid taking its CPU share. If reset/road/action checks fail, preserve
partial receipts and stop rather than retrying another cohort.
