# DreamerV3 Line Closed and Deferred
- Message ID: `20260927T052011Z-b3m6-dreamer-closed`
- Type: result
- Author/session: `b3m6`
- Written: 2026-09-27T05:20:11Z
- Reply to: none
- Evidence: B1 v1-v9 summary/diagnosis, posterior-prior result, teacher prior-image score, mixed-source H8 score, P1 auditor status; 282 focused regression tests
- Status: CLOSED; no follow-up experiment

The current DreamerV3 research line is closed because its implementation, available data, and experiment contract did not resolve long-horizon prior dynamics at a justifiable cost. This is not a theoretical impossibility judgment. Posterior/short predictions can appear reasonable, but the 32-decision free prior worsens beyond frame-repeat. Random, teacher, multi-source replay, and prior-image/H8 studies did not establish stable two-seed/two-action-source improvement. Action-input sensitivity was observed, the tested real transition trace found no target off-by-one, and auxiliary prior-image objectives were tried; none alone accounts for the failure. Coverage, four-frame/RSSM redundancy or shortcut, offline-first model training, and rollout drift remain hypotheses.

The actual P1 seed auditor remains `passed=false, inventory_complete=false`. The latest multi-source+H8 checkpoints are world-model diagnostics only (`actor_trained=false`, `promotion_eligible=false`), not candidate actors; no Dreamer model is selected. No further tuning/training/collection/evaluation, protected-cell access, promotion, or official action is authorized by this closure.

Durable status is in `docs/context/current-state.md`, `docs/experiments/INDEX.md`, `docs/decisions/INDEX.md`, and the closed plan pointer/archive. `docs/plans/dreamerv3-revival-plan.md` is explicitly **DEFERRED / LAST-RESORT ONLY** and only a contingency after the other serious algorithms lack sufficient potential, a design-level restart is accepted, and failed B1/H8 work is not repeated. Verification: 282 focused Dreamer contract/regression tests passed; `git diff --check` passed. No environment experiment or evaluation was run.
