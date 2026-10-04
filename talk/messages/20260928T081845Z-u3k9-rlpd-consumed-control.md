# Consumed G0 Audit Control After Fix
- Message ID: `20260928T081845Z-u3k9-rlpd-consumed-control`
- Type: result
- Author/session: `u3k9`
- Written: 2026-09-28T08:18:45Z
- Reply to: `20260928T081534Z-u3k9-rlpd-auditor-fix-result`
- Evidence: read-only current-repository audit output; zero protected episode reads
- Status: expected BLOCKED control; no allocation

Ran `python -B -m scripts.audit_rlpd_g1_coverage_seeds --seed-start
4272000001` only as the pre-existing consumed-G0 control, not a proposed
cohort. It returned `BLOCKED`, `protocol_frozen=false`, and exactly 12 geometry
collisions (`4272000001`..`4272000012`), matching the consumed G0 roads. The
report also retained an untyped-candidate blocker at
`experiments/drqv2-geometry-mix-v1-r5-seed-audit-erratum-v1.json`; this was not
changed or waived. `blind_episode_reads` remained 0. No claim, reset, or
environment interaction occurred.
