# Root CLI cleanup boundary
- Message ID: `20260929T105602Z-k2f7-root-cli-cleanup-scope`
- Type: coordination
- Author/session: `k2f7`
- Written: 2026-09-29T10:56:02Z
- Reply to: none
- Evidence: current imports, tests, architecture map, frozen protocol/source paths; no environment reset
- Status: planned

The user requested fewer physical files at repository root. I plan to move only `local_runner.py`, `export_policy.py`, and `diagnose_drqv2.py` into the existing `scripts/` package (no root shims), update their current test imports, the live README/architecture entry points, and the prospective DrQ diagnostic self-hash path. This intentionally changes the three development CLI invocations to `python -m scripts.<name>` from repository root. Root `agent.py`, the official mirror, native trainers, shared evaluator, packager, helpers, model and project configuration remain unchanged; active RLPD/TD/DrQ source-bound processes should not be affected. Frozen protocols, archived instructions, run/source copies, and previous diagnostic reports remain untouched. Please flag any current caller or source-bound overlap before integration.
