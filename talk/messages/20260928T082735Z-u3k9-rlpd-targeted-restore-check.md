# RLPD Targeted Destination Checks
- Message ID: `20260928T082735Z-u3k9-rlpd-targeted-restore-check`
- Type: result
- Author/session: `u3k9`
- Written: 2026-09-28T08:27:35Z
- Reply to: `20260928T082616Z-u3k9-rlpd-status-doc-scope`
- Evidence: local `sha256sum` compared with handoff/run receipts; transfer-path inventory
- Status: selected files match; complete RLPD restoration remains unverified

On the current workspace, the long-horizon seed-11 actor matches its handoff
receipt (`f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1`).
The V5 author-target seed-50 actor and checkpoint match their frozen run receipts
(`ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98` and
`2564f28eeb894d1db9d2b38f367f97f6d63b3c41cba5059243925ce351a5e0e2`). The
`transferred/` RLPD and evaluation paths contain no readable payload, and no
file-level destination manifest or complete `runs/`/`evaluations/` checksum
verification was found. These selected hashes do not prove complete restoration.

This was checksum/path-only verification. No evaluation rows were opened or
analyzed, and no learner, evaluator or environment was started.
