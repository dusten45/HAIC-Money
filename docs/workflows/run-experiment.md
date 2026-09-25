# Run Experiment Workflow

## Before Running

1. Read `talk/README.md` and the latest relevant messages before deciding whether
   prior work or a shared-file conflict affects this experiment.
2. Locate the active plan and relevant prior experiment records.
3. State the hypothesis, control, treatment, scope, expected failure signal, and
   acceptance/rejection/stop criteria.
4. Freeze the source revision, command, dependency/runtime, environment contract,
   training exclusions, budget, and evaluation partitions in a protocol artifact.
5. Consult `docs/evaluation/generalization-policy.md`, then audit **all relevant**
   frozen protocols, run receipts, source training ledgers, and retired allocations
   across research lanes before declaring any training or evaluation cell unused.
   Its selective history table alone cannot establish freshness.

## During Execution

- Change one core variable when a matched comparison is the goal.
- Preserve run IDs, checkpoint/model hashes, source/environment/protocol hashes,
  episode/seed records, and CPU operational receipts.
- Keep generated checkpoints, replay, and detailed logs in run artifacts; do not
  commit them merely to make a Markdown claim convenient.
- Stop on the plan's hard failures. Do not alter gates or select a different
  checkpoint after confirmation begins.
- Coordinate material overlap and post decision-relevant observations, results,
  or reproducible failures to `talk/` as separate, evidence-labeled messages.

## After Execution

1. Run the declared local/internal evaluation and compare the same candidate scope.
2. Record outcome counts, proxy metrics, operational results, limitations, and the
   distinction between observations and hypotheses in the result artifact.
3. Post a concise `result` or `failure` message to `talk/` when other agents can
   use or challenge the evidence; link the artifact and reproducibility details.
4. Add only decision-relevant studies to `docs/experiments/INDEX.md`.
5. Add a decision record when the evidence narrows or closes future work.
6. Move/close the plan only after the stated evidence exists.

An experiment result does not itself authorize official submission or model
confirmation.
