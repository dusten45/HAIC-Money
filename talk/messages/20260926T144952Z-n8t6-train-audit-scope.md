# Parallel TRAIN Freshness Audit Scope
- Message ID: `20260926T144952Z-n8t6-train-audit-scope`
- Type: coordination
- Author/session: `n8t6`
- Written: 2026-09-26T14:49:52Z
- Reply to: none
- Evidence: observed user request, current G1 BLOCKED description and dirty working tree; hypothesis on global inventory false blockers
- Status: investigation and implementation planned, no G1 allocation or reset

I am inspecting RLPD G0/G1 seed auditors, DrQ/Dreamer evidence and partition rules to narrow TRAIN-only freshness from repository-global content hashes to candidate-specific collision, consumption and reservation, without weakening confirmation/blind. Likely edits: `scripts/audit_rlpd_g0_seeds.py`, `scripts/audit_rlpd_g1_coverage_seeds.py`, relevant audit tests, `AGENTS.md`, `docs/evaluation/protocol.md`, `docs/workflows/run-experiment.md`, a narrowly scoped RLPD G1/current-state note and possibly a minimal reservation mechanism. I will not alter historical frozen protocols/results/audits or run G1. Shared `docs/context/current-state.md` and `docs/plans/rlpd-completion-first-research-2026-09-26.md` already have uncommitted changes; I will preserve them and limit edits to new scoped paragraphs. Other lanes' new protocols and ledgers are not themselves candidate collisions; unknown candidate-relevant evidence must remain fail-closed. Please avoid overlapping auditor edits while this work is in progress; this note is coordination, not a lock.
