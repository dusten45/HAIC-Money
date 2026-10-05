# Interval road comparison and bounded TRAIN pilot
- Message ID: 20261005T111000Z-r6t9-joint-interval-pilot-scope
- Type: coordination
- Author/session: r6t9
- Written: 2026-10-05T11:10:00Z
- Reply to: 20261005T101609Z-j8f3-temporal-pairs-complete
- Evidence: user instruction 2026-10-05T10:59:00Z; prior frozen source and diagnosis
- Status: implementation and passive revalidation, before new resets

New separate joint-temporal interval study preserves all old frozen results,
champion sources/ZIP, path/speed distillation and unrelated RLPD edits. Physics
1/1/1, first-hold steer .04/pedal .05 and H4/.32s remain fixed. Costs retain the
original weights, never selected to favor a desired suffix. New files under
haic/algorithms/joint_control/, scripts/, tests/ and joint-temporal-interval-v1
artifacts isolate changes from the old comparator and analysis.

Represent observed interior, bracketed uncertain boundary and unobserved pixels
separately; missing adjacent edge classification must not delete a known span.
Use common full-horizon support for candidate comparisons. Calibrate empirical
trajectory residuals including dynamics with geometry-grouped held-out checks;
do not call a small-sample range a safety guarantee. Reuse eight TRAIN pairs for
development comparison, not fresh validation. Any additional paired collection
is small and separately declared; no blanket four-regime prerequisite.

Predeclare three consumed representative TRAIN roads before a six-episode
baseline/successor pilot. The isolated successor issues baseline on unsupported
or uncertain comparisons and commits only the issued action once. Measure full
act CPU time, actual failures and prediction-range misses separately. No official
action/model replacement, broad audit rerun or controller adoption. Shared
current-state and experiment index receive only this study's appended result.
