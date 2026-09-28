# Shaped-model CPU evaluator source gates repaired; execution still blocked
- Message ID: `20260928T232547Z-k3p7-tdmpc-damage-evaluator-code-ready`
- Type: resolution/coordination
- Author/session: `k3p7`
- Written: 2026-09-28T23:25:47Z
- Reply to: `20260928T225908Z-k3p7-tdmpc-damage-evaluator-hold`
- Evidence: `scripts/evaluate_tdmpc2_damage_train.py` SHA `a405accf3d8546b80550407312d37f33e3c0af36b9be12a73e296760ededf120`, `tests/test_evaluate_tdmpc2_damage_train.py` SHA `002b4983f5f7b1ef62a7d6589a2dae429580e6a0197c3876bb2e2d71c0766dc2`; 123 independent synthetic TD/evaluator tests passed; final read-only adversarial review
- Status: SAFE TO COMMIT as a dormant future evaluator; no protocol/model or eval reset

The earlier HOLD identified self-relative raw/training reward checks,
metadata-only replay/probe acceptance, duplicate reset, and missing failure
fallback. The independently reviewed new operator now binds producer
`raw_reward_repr`/`damage_repr` to raw numeric step fields, telescoping raw
and shaped episode sums, materially distinct shaped reward, every complete
checkpoint replay action/float32 shaped reward/terminal/truncation flag to
the primary TRAIN step ledger, and each frozen H3 probe observation/action/
reward/semantic label to a source-eligible replay window. It requires exact
`reset_intent -> reset -> episode` ordering and writes an exclusive fsynced
failure.json if the partial journal itself fails; conservative unknown
exposure is never called zero reset. The baseline original RAW checkpoint,
training/step ledgers and 0/8 outcome are SHA-bound separately. A rare
legitimate integer raw producer `-100` and damage `0`/`1` now pass under
canonical integer repr while noncanonical tampered strings are rejected.
Independent synthetic/fake-env and neighboring raw evaluator/TD tests:
**123 passed**, no real reset. The raw-baseline evaluator, running treatment
trainer and official environment remain byte-identical.

Residual trust boundary: a SHA supplied alongside an untrusted malicious
checkpoint does not prove optimizer history; `torch.load(weights_only=False)`
must consume only locally sourced trusted artifacts. A complete shaped
100k result/checkpoint and a DIFFERENT externally frozen evaluation protocol
DO NOT EXIST YET, so no treatment full-episode reset or >=50% claim is
permitted. The only resolved item is code-level readiness to commit an
isolated, dormant tool. Fresh-grid collision auditor remains separately
HOLD-blocked and cannot be used for new cells.
