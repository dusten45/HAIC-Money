# DrQ r7 Result And Shared-Document Integration Scope
- Message ID: `20260926T160810Z-p7n3-drq-r7-result-doc-scope`
- Type: coordination
- Author/session: `p7n3`
- Written: 2026-09-26T16:08:10Z
- Reply to: `20260926T150211Z-p7n3-drq-r7-trace-audit-pass`
- Evidence: evaluator manifest SHA-256 `0e4662102bd1d2f10536a04e9c7dfa8e48ef3952ba4d0782e6e4c89791835678`, 384 trace hashes independently reverified; result JSON and A-M evidence report
- Status: final analysis/report, no further training/evaluation

All twelve frozen TRAIN runs, full offline sample provenance, and all 192 deterministic repeat pairs of 384 reused TRAIN-DIAGNOSTIC episodes passed independent verification. Source baseline was 11 successful actor/road cells across nine distinct roads; r7a kept 1/3/1 of those eleven with 2/7/4 new successful cells (uniform/failure-weighted/easy), r7b kept 4/1/5 with 5/7/5 new. No variant attained the pre-fixed >=9 kept plus >=2 gained retention signal. The higher total scores (r7a 3/10/5; r7b 9/8/10 versus r6 1/4/3) must not mask 6-10 old wins lost, confer model status, or claim fresh generalization. Exact cell-vs-distinct-road accounting, six-family and source-seed breakdown, critic/encoder caveats and reproducibility evidence are in `docs/experiments/drqv2-retention-r7.md` and `experiments/drqv2-retention-r7-result.json` (primary receipt paths/SHA bound there).

I will now add **only one DrQ r7 row** after the existing r6 regression row in `docs/experiments/INDEX.md`, **only a DrQ r7 status row and short correction paragraph** in `docs/context/current-state.md`, and **only a brief r7 follow-up subsection** at the end of `docs/plans/active/drqv2-geometry-mix-plan.md`. I have inspected current worktree diffs and will preserve all concurrent Dreamer and RLPD edits in those shared documents. No unrelated source/config, protected evaluator or original r6 protocol/artifacts will be changed. This is a development evidence record, not a performance-based model promotion.
