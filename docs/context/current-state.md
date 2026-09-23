# Current Research State

Last refreshed: 2026-09-23. This is the current-state source of truth, not an
experiment changelog. Evidence and historical decisions are linked below.

## Current Position

| Area | Current status |
|---|---|
| Validated internal baseline | Native DrQ-v2 control with augmentation pad 4. Two unique control actors, one per training seed, were evaluated on two fresh internal confirmation cohorts; the recorded control outcomes range from 4 to 7 finishes in 32 cells. |
| Active research direction | Restore DreamerV3 fidelity before deciding whether it merits another training pilot. |
| Current blocker | The current DreamerV3 formulation failed its small policy-learning gate through steering saturation; its reported reconstruction/continuation diagnostics do not alone establish a faithful, decision-useful world model. |
| DrQ-v2 narrow tuning | Closed. The predeclared steering-logit L2 and pad=1 follow-ups both regressed; do not launch a third narrow tuning axis. |
| Best single-model designation | No official or confirmed model is designated. See `docs/results/MODEL_STATUS.md`. |
| Official external state | This repository has no committed official server submission identifier, public-track result, or model-confirmation receipt. A local package archive is not proof of an official submission. |

## Active Work

The active plan is
[`docs/plans/active/dreamerv3-recovery-strategy.md`](../plans/active/dreamerv3-recovery-strategy.md).
It is a correctness-and-feasibility plan, not authorization to start a 131,072-
decision run. Its first work is to repair and test the stated objective/transition
fidelity defects, then repeat the pretraining and world-model gates before another
small pilot.

## Closest Known Competition Gate

The closest verified competition gate is the first mock competition at 2026-09-24
18:00 KST. The migration request's 15:00 KST confirmation cutoff was not present in
the accessible official sources and remains unverified. The authoritative schedule,
verification status, and timezone details are in
[`docs/competition/info.md`](../competition/info.md). Recheck the competition site
before acting on any gate.

## Read Next

- Current candidate provenance and confirmation status:
  [`docs/results/MODEL_STATUS.md`](../results/MODEL_STATUS.md)
- Evaluation and partition discipline:
  [`docs/evaluation/protocol.md`](../evaluation/protocol.md) and
  [`docs/evaluation/generalization-policy.md`](../evaluation/generalization-policy.md)
- Durable evidence and prior decisions:
  [`docs/experiments/INDEX.md`](../experiments/INDEX.md) and
  [`docs/decisions/INDEX.md`](../decisions/INDEX.md)
- External actions and official rules:
  [`docs/competition/`](../competition/)

## Evidence Boundary

The DrQ-v2 and DreamerV3 statements above summarize frozen experiment records, not
new measurements. Consult the linked JSON protocol/result artifacts before making a
new comparison or changing a candidate status.
