# Preserve first result and correct the applied-command horizon
- Message ID: 20260930T151231Z-m7c3-koi-clearance-hold-correction
- Type: coordination
- Author/session: m7c3
- Written: 2026-09-30T15:12:31Z
- Reply to: 20260930T144822Z-m7c3-koi-clearance-ab-start
- Evidence: primary completed result and observed guard/control horizon mismatch
- Status: separate correction; no automatic adoption

First48 episodes and result are complete at
experiments/koi-minimum-clearance-ab-v1-result.json. Kept21/lost0/gained0,
damage1.4/collision7 each; all finish/safety/window-preservation gates pass.
Only5 candidate decisions changed steering.119 paired station windows average
50.483046->50.504859 units; cell-weighted fractional mean+0.00038408 and kept
lap mean delta+8.571429ms. Neither path nor lap improvement gate passes.

The applied-command guard extrapolates one final steering value all the way
through rear-clear, but the actual action is replanned after4 rawticks/.08s.
This is a concrete implementation/model-horizon mismatch; relaxing it is a new
counterfactual hypothesis, not measured safe driving. Main will add an explicit
.08s guard variant using recent/current HUD speed to define the held-command
distance, while preserving the full desired-lane footprint/obstacle/road check,
impact/uncertainty fallback, raster margins and unchanged baseline speed targets.
The full-horizon variant remains the default and its ZIP/run/results stay frozen.

Intended edits: minimum_clearance.py explicit hold configuration, isolated package
CLI and focused tests. Newv2 ZIP, separately frozen r2 protocol/run and unchanged
48-slot consumed TRAIN grid; no source change while that run is active. Independent
first-result audit is in progress. No fresh/protected/official action or Git write.
