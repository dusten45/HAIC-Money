# Same-speed pursuit implementation repair

This is a correction of one arm within the existing four-direction TRAIN batch, not a new independent one-direction search or a parameter sweep. The six consumed cells remain TRAIN diagnostics. Preserve the original 30-episode report and original runtime.

Review found that subtracting an unclipped road term from clipped inherited steering introduces an unwanted residual. Implement a subclass that computes desired pursuit road steering plus the exact inherited obstacle term, then clips once. It retains the same governor, target 60, first ten actions, obstacle logic, speed and road gates, and six cells. No lowering speed or widening qualification thresholds.

Run six corrected pursuit episodes only. Compare their first-ten action/state hashes to the original control hashes; mismatch invalidates comparison. Resource maximum 1200 decisions/episode, 2 CPU, 2 GiB, 600 seconds. No independent score, selected-model change, package or external action. Outcome must enter three-gate review.

The old `pixel_physical_speed_mae` field measures lagged pre-action pixel/post-action physical discrepancy; it is not a calibrated estimator-error measurement. Preserve this field in old reports and explicitly qualify its meaning. Qualification separately checks both speed sequences; it does not depend on this MAE.

Authorization: 2026-09-29 standing local authorization and user's explicit setup-and-proceed instruction. The original batch is STOPPED/REVISE before repair. Design and execution events attach to a new exact hash.

Tasks: regression assertion for saturation; subclass repair; narrow registered replay profile; six same-cell replays; hash verification and integration report.
