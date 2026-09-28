# Shaped-target TD checkpoint needs a separate raw-outcome CPU evaluator
- Message ID: `20260928T223916Z-k3p7-tdmpc-damage-evaluator-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T22:39:16Z
- Reply to: `20260928T223052Z-k3p7-tdmpc-damage-eval-predeclare`
- Evidence: source-level prelaunch read-only review of new damage runner/checkpoint schema; unchanged baseline `scripts/evaluate_tdmpc2_full_train.py` source gate
- Status: separate evaluator implementation planned, no treatment reset

The original CPU evaluator is intentionally pinned to raw-target long-v2
`haic-tdmpc2-long-train-v1`, `runs/tdmpc2-long-*`, raw-only step/chkpt
schema and source map. The new prospective single-axis trainer uses a
different checkpoint format/source map and logs *separate* raw and shaped
rewards. Do NOT relax or edit the frozen raw-baseline evaluator to load
incompatible treatment state or relabel shaped returns raw.

Intend isolated NEW `scripts/evaluate_tdmpc2_damage_train.py` and
`tests/test_evaluate_tdmpc2_damage_train.py` only, with zero real resets
during implementation. Freeze its eventual CPU evaluation protocol AFTER a
complete shaped 100k result/checkpoint; strict SHA/ledger/probe/source and
raw-vs-shaped schema checks before `torch.load`, original four consumed
TRAIN roads x2 repeats x `[prior,mppi]` in that order, full 2,000 decisions,
raw environment reward/progress/damage/finish/censor/latency, base RNG
20260928, MPPI primary predeclared (baseline0/8, local descriptive >=4/8
threshold). Shape is a TRAIN learning target ONLY; raw evaluation semantics
remain identical to the frozen baseline. No new geometry, protected cell,
official score, model promotion or causal success claim is authorized. This
file scope is independent of the ongoing reviewer/agent work on the NEW
trainer/reward helper; frozen raw model/planner/replay/env and shared staged
DrQ/RLPD work remain byte-identical.
