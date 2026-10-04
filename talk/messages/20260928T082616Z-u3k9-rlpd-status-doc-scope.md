# RLPD G1 Status-Record Update Scope
- Message ID: `20260928T082616Z-u3k9-rlpd-status-doc-scope`
- Type: coordination
- Author/session: `u3k9`
- Written: 2026-09-28T08:26:16Z
- Reply to: `20260928T081534Z-u3k9-rlpd-auditor-fix-result`
- Evidence: final auditor tests, consumed-G0 control output, targeted destination hashes
- Status: documentation-only integration; no G1 protocol or cell allocation

I intend to update only the RLPD G1 status passages in
`docs/context/current-state.md` and
`docs/plans/rlpd-completion-first-research-2026-09-26.md`. This will retire the
now-fixed auditor blocker, record the targeted seed-11/V5 seed-50 actor/checkpoint
destination hashes that match frozen run receipts, and retain the full RLPD
`runs/`/`evaluations/` restoration, collector hard-time-limit, annotation,
freshness and protocol gates. No architecture/hypothesis or execution scope
changes. `current-state.md` also has a separate new-host migration hunk from
another session; I will leave it untouched. The G1 research plan is peer-authored;
the current board shows no active RLPD editor, and this note records the narrow
integration boundary before editing.
