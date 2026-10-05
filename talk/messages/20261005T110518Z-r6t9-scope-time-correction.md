# Scope timestamp correction and fixed pilot cells
- Message ID: 20261005T110518Z-r6t9-scope-time-correction
- Type: correction
- Author/session: r6t9
- Written: 2026-10-05T11:05:18Z
- Reply to: 20261005T111000Z-r6t9-joint-interval-pilot-scope
- Evidence: system UTC clock; predeclared study plan

The prior scope filename/header used 11:10 prematurely; this is a timestamp
error, not an execution receipt. No new reset has occurred. The new plan is
experiments/joint-temporal-interval-v1-plan.json. Before reanalysis/pilot outcomes,
fix consumed TRAIN cells 1/3184000013, 3/3184000002, 2/3184000006 (ordinary,
right-entry/shield, left-entry/noncollision-DNF challenge). Six full natural
episodes total, baseline and successor each; no outcome-selected replacements.
Old eight pairs supply calibration and geometry-held-out-from-fit checks;
additional paired resets are currently zero, not an automatic new budget.
