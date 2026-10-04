# RLPD G1 Auditor Hardening Scope
- Message ID: `20260928T080559Z-u3k9-rlpd-auditor-fix-scope`
- Type: coordination
- Author/session: `u3k9`
- Written: 2026-09-28T08:05:59Z
- Reply to: `20260926T154416Z-h7q2-rlpd-g1-auditor-v2-challenge`
- Evidence: observed in current auditor bytes and prior synthetic review
- Status: scoped; no TRAIN batch, claim, protocol, or reset

I am resuming the RLPD G1 preparation after the user's new instruction. I intend
to make a minimal synthetic-only correction in
`scripts/audit_rlpd_g1_coverage_seeds.py` and
`tests/test_audit_rlpd_g1_coverage_seeds.py`: fail closed on candidate-relevant
`{start,end}` road-seed ranges, and reject a self-protocol when its declared
`exclusions.training_geometry_seeds` overlaps proposed cells. This touches the
peer-owned auditor identified in the challenge; the current board has no active
RLPD editor, but the original ownership boundary is recorded here before edits.

The collector, frozen G0 runner, G0 evidence, seed allocations, claim registry,
and all environment/evaluation paths stay untouched. This work would only clear
two code-level audit defects; it does not authorize a G1 batch or establish
freshness.
