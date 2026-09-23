# Experiment Evidence Index

The JSON protocol/result artifacts under repository-root `experiments/` are the
source of truth. This index is intentionally selective: it promotes only evidence
that changes future research decisions. Timestamped `runs/` and `evaluations/`
artifacts remain the detailed execution receipts.

| Study | Question and conclusion | Primary evidence | Status |
|---|---|---|---|
| DrQ-v2 steering-logit L2 | Does `steering_logit_l2=0.001` improve repeated completion over a matched control? It scored 3 vs 6 and 0 vs 7 confirmation finishes for training seeds 0 and 1. | [`protocol`](../../experiments/drqv2-steering-logit-v1.json), [`result`](../../experiments/drqv2-steering-logit-v1-result.json), [`execution`](../../experiments/drqv2-l2-execution.json), [`diagnostics`](../../experiments/drqv2-steering-logit-v1-diagnostics.json) | Rejected. |
| DrQ-v2 padding | Does reducing random-shift padding from 4 to 1 improve generalization? Pad 1 had zero confirmation finishes versus 4 and 7 for controls. | [`protocol`](../../experiments/drqv2-augmentation-pad-v1.json), [`result`](../../experiments/drqv2-augmentation-pad-v1-result.json), [`execution`](../../experiments/drqv2-pad-execution.json), [`diagnostics`](../../experiments/drqv2-augmentation-pad-v1-diagnostics.json) | Rejected; final authorized narrow DrQ follow-up. |
| DreamerV3 feasibility | Is the current native DreamerV3 implementation ready for 131k matched training? Interface/export and short GPU smoke passed, but the pilot policy collapsed into steering saturation. | [`feasibility record`](../../experiments/dreamerv3-feasibility-gate.json) | Current formulation rejected for scale-up; fidelity-recovery plan remains active. |

## Companion Evidence

- The historical DrQ-v2 promotion record is a caveat, not a standalone research
  direction: its undocumented historical result was 4/24, its fresh screen was
  0/24, and its predeclared diagnostic confirmation was 4/32. It was never
  promoted. See [`protocol`](../../experiments/drqv2-promotion-v1.json),
  [`result`](../../experiments/drqv2-promotion-v1-result.json), and
  [`diagnostic decision`](../../experiments/drqv2-confirmation-diagnostic-v1.json).
- [`drqv2-pre-l2-diagnostics.json`](../../experiments/drqv2-pre-l2-diagnostics.json)
  supplied the bounded L2 hypothesis; it did not itself establish a causal failure
  mechanism.
- [`drqv2-steering-logit-proposal-v1.json`](../../experiments/drqv2-steering-logit-proposal-v1.json)
  is the preregistered L2 proposal and remains historical after rejection.
- The Dreamer feasibility JSON originally marks its Gate 3 diagnostics `PASSED`.
  The later active recovery analysis at source revision `6feb60b` supersedes that
  interpretation: the recorded diagnostics are preserved, but they are insufficient
  proof of a faithful, decision-useful world model. Gate 1 and Gate 2 remain valid;
  Gate 4's pilot failure remains valid.

See [`docs/decisions/INDEX.md`](../decisions/INDEX.md) for the consequences of this
evidence and `docs/plans/active/` for work still authorized.
