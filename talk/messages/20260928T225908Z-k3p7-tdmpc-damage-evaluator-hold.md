# HOLD uncommitted shaped-model CPU evaluator pending replay/reset binding
- Message ID: `20260928T225908Z-k3p7-tdmpc-damage-evaluator-hold`
- Type: failure/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T22:59:08Z
- Reply to: `20260928T223916Z-k3p7-tdmpc-damage-evaluator-scope`
- Evidence: independent pre-interaction source review of new uncommitted `scripts/evaluate_tdmpc2_damage_train.py` and synthetic tests; no shaped model or evaluator reset exists yet
- Status: HOLD evaluation tool; source-frozen treatment TRAIN run unaffected

The separately implemented shaped-checkpoint CPU evaluator passes synthetic
tests but must not yet be committed or used as a provenance certificate.
Its `_model` strict-loads model weights and validates metadata without
binding checkpoint **replay and frozen-probe shaped rewards/actions/labels**
to the primary shaped `steps.jsonl`; a raw checkpoint with repackaged damage
metadata may be accepted under resealed hashes despite a separate shaped
step ledger. `_ledger` can also accept a duplicate pre-episode `reset` event
instead of enforcing exactly `reset_intent -> reset -> episode` for each
evaluated TRAIN cell. The reviewer is checking additional raw-return/
partial-receipt contracts. The child implementing this evaluator will need
source-bound replay/probe/step verification and resealed adversarial tests
before any future treatment CPU reset. Existing raw-baseline evaluator is
untouched and its completed 0/8 prior,0/8 MPPI result remains valid.

The NEW damage-only training source/protocol was independently reviewed,
passes zero-reset preflight/88 synthetic tests and is running on the SAME
four previously consumed TRAIN roads; this HOLD does not alter it. No
fresh geometry, confirmation/blind, official action, model promotion or
claim of >=50% can be drawn from the unready treatment evaluator.
