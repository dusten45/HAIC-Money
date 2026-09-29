# Coupled recovery resumed with isolated code paths
- Message ID: `20260929T212450Z-q5n2-rlpd-recovery-resumed-design`
- Type: coordination
- Author/session: `q5n2`
- Written: 2026-09-29T21:24:50Z
- Reply to: `20260929T210806Z-q5n2-rlpd-coupled-recovery-scope`
- Evidence: hypothesis; source inspection
- Status: open

The user resumed after restarting Kilo only. Three isolated implementations are
underway: `scripts/collect_rlpd_coupled_recovery.py`,
`haic/algorithms/rlpd/recovery.py` plus `scripts/train_rlpd_recovery.py`, and
`scripts/evaluate_rlpd_recovery.py`, with separate tests. Existing source-bound
RLPD/Oracle runners and immutable prior datasets remain untouched. I will update
the existing RLPD plan, experiment index and current-state section after primary
results exist, preserving unrelated staged/unstaged changes.

Collection uses ten already measured curve-entry precursors and four finished
parent controls from consumed G0 TRAIN roads only. Two fixed full Oracle-vector
closed-loop durations (12/25 decisions) precede return to the original actor.
Qualification follows at least 63 actor decisions after handoff, not four logged
suffix actions; full episode outcomes and all failed branches are retained.
Best-of-menu selection is a training-data upper bound, never a deployment claim.

The intended first learning comparison imports the same V5 seed50 learner state
into two explicitly new fine-tuning runs with fresh online replay/RNG. Control
uses 32 online/32 ordinary prior; treatment uses 32 online/16 ordinary prior/16
validated recovery. Raw reward and SAC actor loss stay unchanged. Initial data
sufficiency will be assessed before any learner run; recovery transition counts,
distinct geometries and full-finish rescue/harm must remain separate metrics.
The original actor and both new exports require contemporary consumed-TRAIN
full episodes. Curve-entry counts on successful controls and retained old
finishes must be reported, not just aggregate completion. No protected cells,
new roads, official action or model promotion is authorized by this design.
