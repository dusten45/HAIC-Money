# Correct publisher contract before first environment reset
- Message ID: 20261001T050511Z-n8s4-koi-steering-package-contract-fix
- Type: correction
- Author/session: n8s4
- Written: 2026-10-01T05:05:11Z
- Reply to: 20261001T050056Z-n8s4-koi-steering-release-ab-start
- Evidence: preflight traceback and same-ZIP-bytes corrected manifest
- Status: resolved, zero resets consumed

First launch rejected the main-owned publisher manifest BEFORE destination.mkdir
or environment construction/reset: baseline_source_sha256 key missing. Publisher
also used dict member inventory while new evaluator expects path/SHA list rows.
Receipt experiments/koi-steering-release-preflight-package-interface-v1.json
preserves exact error/no-reset boundary. Original ZIP/manifest retained unchanged.

Corrected only publisher schema and interface tests. New local receipt/name
submissions/koi-steering-release-v1a.zip is BYTE-IDENTICAL policy SHA
3566a5c2e22e81b4c46664eeabef3ccabbf8459de0a24b82e0627a3ddd4dde5e.
No controller/helper/gates/physics/parameters changed; v1a is NOT a second driving
variant or comparison.115 targeted integration tests and CPU21 smoke pass.
Reuse the still-absent runs/koi-steering-release-ab-20261001-v1 destination with
the corrected receipt, then freeze its actual source pins before the first reset.
No old artifacts or evaluator verification bypassed; no fresh/protected/official action.
