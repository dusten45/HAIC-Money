# Run Experiment Workflow

## Before Running

1. Locate the active plan and relevant prior experiment records.
2. State the hypothesis, control, treatment, scope, expected failure signal, and
   acceptance/rejection/stop criteria.
3. Freeze the source revision, command, dependency/runtime, environment contract,
   training exclusions, budget, and evaluation partitions in a protocol artifact.
4. Confirm that the chosen screen/confirmation/blind cells are allowed by
   `docs/evaluation/generalization-policy.md`.

## During Execution

- Change one core variable when a matched comparison is the goal.
- Preserve run IDs, checkpoint/model hashes, source/environment/protocol hashes,
  episode/seed records, and CPU operational receipts.
- Keep generated checkpoints, replay, and detailed logs in run artifacts; do not
  commit them merely to make a Markdown claim convenient.
- Stop on the plan's hard failures. Do not alter gates or select a different
  checkpoint after confirmation begins.

## After Execution

1. Run the declared local/internal evaluation and compare the same candidate scope.
2. Record outcome counts, proxy metrics, operational results, limitations, and the
   distinction between observations and hypotheses in the result artifact.
3. Add only decision-relevant studies to `docs/experiments/INDEX.md`.
4. Add a decision record when the evidence narrows or closes future work.
5. Move/close the plan only after the stated evidence exists.

An experiment result does not itself authorize official submission or model
confirmation.
