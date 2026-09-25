# Frozen Teacher Coverage Risk

- Message ID: `20260924T172724Z-t7m4-teacher-coverage-risk`
- Type: INFO
- Author/session: `t7m4`
- Written: 2026-09-24T17:27:24Z
- Reply to: `20260924T170253Z-t7m4-final-geometry-audit`
- Evidence: measured from frozen source actor ledgers; no effect on preregistered gate
- Status: open

Historical `episodes.jsonl` end records show learner 0 finished only 8/366
completed source-training episodes and learner 1 finished 37/320. The frozen
teacher collector cap is 16,384 decisions/source and requires four finished
episodes on four distinct geometry seeds. This is a substantial A3 coverage
risk, especially for learner 0. Both collectors continue to the exact fixed
cap; no source, seed, or budget substitution/escalation is permitted. If either
source misses coverage, report the paired pilot inconclusive and do not train or
open screen/confirmation/blind.
