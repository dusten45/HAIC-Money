# Root CLI cleanup complete
- Message ID: `20260929T105951Z-k2f7-root-cli-cleanup-result`
- Type: result
- Author/session: `k2f7`
- Written: 2026-09-29T10:59:51Z
- Reply to: `20260929T105602Z-k2f7-root-cli-cleanup-scope`
- Evidence: working-tree paths, focused pytest results, module `--help`, `git diff --check`
- Status: closed

Moved only the three independent development tools `local_runner.py`, `export_policy.py`, and `diagnose_drqv2.py` to `scripts/`, reducing physical root entries from 52 to 49. Updated current test imports and the live README/architecture map; prospective DrQ diagnostic reports hash `scripts/diagnose_drqv2.py` from the repository root. Historic protocols, old reports, archived commands and immutable run/evaluation copies were not changed. New commands run from repository root as `python -m scripts.local_runner`, `python -m scripts.export_policy`, and `python -m scripts.diagnose_drqv2`; the old root CLI/import paths intentionally no longer exist (no root shims). Official Agent, packager, environment mirror, trainers, evaluator and root config were untouched.

`python -B -m pytest -q tests/test_local_contract.py tests/test_agent_inference.py tests/test_submission_policy.py tests/test_submission_package.py tests/test_drqv2_diagnostics.py`: 60 passed, 36 subtests passed. `python -B -m pytest -q tests/test_submission_layout.py tests/test_server_parity.py`: 9 passed, 1 skipped, 7 subtests passed. All three `python -B -m scripts.<tool> --help` calls exited successfully; `git diff --check` passed. No driving/evaluation cell, training run, external submission or official action was launched. Existing concurrent worktree changes were preserved; no commit or push was made.
