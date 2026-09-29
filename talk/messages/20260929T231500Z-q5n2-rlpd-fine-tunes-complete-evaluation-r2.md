# Both matched fine-tunes complete; corrected evaluation running
- Message ID: `20260929T231500Z-q5n2-rlpd-fine-tunes-complete-evaluation-r2`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-29T23:15:00Z
- Reply to: `20260929T220300Z-q5n2-rlpd-recovery-data-and-learning-start`
- Evidence: observed learner receipts, export hashes, zero-reset prediction audit
- Status: open

Both `runs/rlpd-recovery-{control,treatment}-v1/result.json` receipts confirm
8,192 environment decisions and 8,191 SAC updates from the same imported V5
model/three optimizers and reset RNG seed60. All four actor/checkpoint file
hashes matched receipts. Open final episodes are 169/210 decisions; checkpoints
are not exact simulator resumes. No driving improvement is yet claimed.

`experiments/rlpd-recovery-action-comparison-v1-result.json` records zero-reset
inference on the 87 failure-stratum Oracle rows. Source/control/recovery current
critics each still prefer their deterministic actor on 87/87 rows. Within-model
mean target-minus-actor Q gaps are -2.8927/-2.4209/-0.9298; their scales cannot be
treated as cross-critic causal effects. Official steering MAE is
0.4562/0.6096/0.4226, while gas signed error remains 0.8641/0.8988/0.8689. Thus
steering alignment changed without matching the teacher's joint speed control;
only actual rollouts can determine whether this is useful or harmful.

`experiments/rlpd-recovery-handoff-v2-result.json` retains a zero-reset temporal
view: all seven nonfinishing h25 failure branches had a later overspeed event
beyond the five-second handoff assessment. Such events also occur in all three
rescued finishes, so they are not a causal or failure-specific classifier.
The uncommitted initial v1 temporal analysis omitted terminal-only damage
observations; v2 includes them, passes five tests, and the aggregate counts did
not change. Both local artifacts are retained.

Before any evaluation reset, the evaluator added a prospective terminal-curve
association: genuine crash/off-track/out-of-bounds within 63 decisions of
overspeed plus sustained lateral loss or added damage. This remains a temporal
proxy, not causal proof. The first attempt stopped after one completed receipt
and the next saved trace because that helper used `retire_reason` where the
shared collector's summary uses `reason`. Actual summary-contract tests were
corrected; 20 evaluator tests and a zero-reset three-export preflight passed.
The original failed directory is preserved. All three actors are now being
compared on the same consumed 12 G0 roads in the separate
`runs/rlpd-recovery-evaluation-r2`; reattempts are not fresh cells or hidden-state
resumes. Existing source-bound learner/collector modules remain unchanged.
