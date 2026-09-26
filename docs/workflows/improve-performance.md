# Improve Performance Workflow

Use this workflow for requests such as "improve performance", "raise the score",
"continue", or "find a better algorithm". It performs research work; it does not
submit or confirm a model on the competition site.

## Read Minimum Context

1. `AGENTS.md`
2. `docs/context/current-state.md`
3. `talk/README.md` and the latest relevant messages in `talk/messages/`
4. The active plan in `docs/plans/active/`
5. `docs/evaluation/protocol.md` and `docs/evaluation/metrics.md`
6. Only the relevant entries in `docs/experiments/INDEX.md` and
   `docs/decisions/INDEX.md`

Keep an independent hypothesis and research lane. Before a substantial change,
share intended files if another agent may be touching the same model component;
use `talk/` to exchange evidence and counterarguments, not to force convergence.

## Execution Loop

1. Identify the current bottleneck from code, tests, and artifacts rather than a
   stale prose claim.
2. Exclude a local bug, environment mismatch, checkpoint mismatch, selection
   leakage, or evaluation mismatch before changing an algorithm.
3. Do not repeat a rejected approach without a stated revisit condition and new
   evidence.
4. Search official implementations or literature when a new algorithmic direction
   or missing evidence justifies it; do not turn routine tuning into a mandatory
   literature review.
5. Write or update an active plan with a falsifiable hypothesis, scope, one primary
   changed variable where practical, matched evaluation method, acceptance,
   rejection, and stop conditions.
6. Implement the smallest correct change and run relevant unit/integration tests.
7. Freeze and run the internal evaluation protocol. Preserve result artifacts and
   compare against the baseline on the same declared conditions.
8. Classify the outcome as accepted, rejected, or inconclusive. Record durable
   evidence and a decision when it changes future search space.
9. Update `docs/context/current-state.md`; update `docs/results/MODEL_STATUS.md` only when an
   actual candidate status changes; set the next highest-value active plan.

## External Boundary

A promising result is an internal candidate only. Do not launch an official
submission, model confirmation, or browser-side competition action from this
workflow without the user explicitly authorizing that external action immediately
before it occurs.
