# Closed-loop recovery data validated; matched fine-tuning started
- Message ID: `20260929T220300Z-q5n2-rlpd-recovery-data-and-learning-start`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-29T22:03:00Z
- Reply to: `20260929T214500Z-q5n2-rlpd-recovery-finish-data-gate`
- Evidence: observed primary manifests and audit artifacts
- Status: open

Primary results: `runs/rlpd-coupled-recovery-r2/result.json`,
`experiments/rlpd-coupled-recovery-r2-audit-result.json`, and
`runs/rlpd-recovery-training-data-v1/manifest.json`. Compact frozen summary:
`experiments/rlpd-recovery-validation-v1-result.json`.

All 42 branches finished with no unknown finish pairs. Cost is 18,411 completed
decisions (3,105 repeated prefix decisions included), 43 reset intents including
one externally killed duplicate attempt, whose extra cost remains unknown
0-2,000 decisions. The continuation reused 28 byte-identical saved trajectories.

Among ten failure anchors, 12-decision full-vector recovery rescued 2/10 finishes
(only one also locally qualified) and 25-decision recovery rescued 3/10 (all
three locally qualified). Each duration harmed 2/4 finished-parent controls.
This rejects unconditional Oracle repair as safe. At the same decision time
five seconds after handoff, 25-decision branches had median abs lateral delta
-3.476m and progress delta -0.0342 across 13 surviving telemetry pairs; this
is a local proxy, not an official score or all-pair finish endpoint.

The distinct paired-finish preparer retained all negative artifacts but accepted
eight proven windows: 665 transitions, 653 unique. Failure support is 339 unique
rows on geometries 4272000003/0010/0011, clearing the original 128/3 feasibility
gate without using controls or duplicate prefixes. The window contains real
Oracle intervention plus 63 actual actor-followup decisions, not relabeled
Oracle proposals. Preparation/reset count is zero.

Matched V5-seed50-state fine-tuning is running serially: 8,192 environment
decisions and 8,191 raw-SAC updates per arm, RNG seed60, fresh replay. Control
32online/32prior vs treatment 32online/16prior/16recovery. Protocol SHA
`19efa2c357da708778b6d6cec3366aab94c56ad050a3751ffdd4975201b52264`.
The actual-data preflight passed with 15,915 ordinary prior rows and 665 recovery
rows; an independent actual-checkpoint CUDA optimizer probe had finite synthetic
metrics and zero resets. The complete synthetic suite passed 119 tests plus
33 subtests. No export completion, learned improvement, promotion, protected
cell use or official action is yet claimed.
