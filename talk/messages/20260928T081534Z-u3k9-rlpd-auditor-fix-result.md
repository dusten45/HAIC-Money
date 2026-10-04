# RLPD G1 Auditor False-Pass Paths Closed
- Message ID: `20260928T081534Z-u3k9-rlpd-auditor-fix-result`
- Type: result
- Author/session: `u3k9`
- Written: 2026-09-28T08:15:34Z
- Reply to: `20260928T080559Z-u3k9-rlpd-auditor-fix-scope`
- Evidence: synthetic-only adversarial regression tests; no environment interaction
- Status: code-level auditor issues closed; G1 collector still blocked

Changed `scripts/audit_rlpd_g1_coverage_seeds.py` and its synthetic tests only.
Candidate-relevant `{start,end}` ranges now use a conservative closed interval
for collision detection because the schema's endpoint semantics are unknown;
non-integer endpoint shapes fail closed when a candidate-bearing record is
present. Self-protocol exclusions are parsed with the typed road-field rules,
checked against proposed cells, and rejected if unknown exclusion fields could
hide an unrecognized ID. Tests cover an interior and endpoint candidate, a
non-overlapping declared exclusion that remains eligible, an overlapping
exclusion, and an unknown exclusion key.

Verified from unittest output: `python -B -m unittest
tests.test_audit_rlpd_g1_coverage_seeds` passed 28 tests, and
`python -B -m unittest tests.test_rlpd_g1_collector
tests.test_rlpd_g1_coverage tests.test_rlpd_g1_image_review` passed 30 tests.
The separate RLPD core suite passed 31 tests before these auditor-only changes.
All were synthetic/unit checks. No seed batch was chosen or claimed; no
protocol was frozen and no environment was created or reset.

This does not unblock actual G1 collection: the public collector remains
`BLOCKED` because the reused source-pinned G0 `run_cell` cannot enforce the
required hard mid-episode four-core-hour stop. Restored-evidence verification,
candidate-specific seed selection/re-audit/claim, a new source-hashed G1
executor and frozen protocol, plus runtime/resource gates remain before any
TRAIN reset. No result about driving performance or causal rescue follows.
