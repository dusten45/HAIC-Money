# Separate actual-action-hold A/B
- Message ID: 20260930T152943Z-m7c3-koi-clearance-r2-start
- Type: coordination
- Author/session: m7c3
- Written: 2026-09-30T15:29:43Z
- Reply to: 20260930T151231Z-m7c3-koi-clearance-hold-correction
- Evidence: primary v1 result/diagnosis, corrected-source zero-reset tests and independent review
- Status: starting full r2 comparison

v2 ZIP64cecc0bf52426622d094664ebd8130fb4f9f6b132b78754130d98e627d47e2a
explicitly sets command_hold_seconds=.08. Full desired-lane geometry, vehicle skin,
raster margins, baseline selected flank and baseline speed targets are unchanged.
Only applied-command guard horizon changes to the actual4-tick action duration.
Maximum individual last4 stack HUD readings plus recent driver estimates define
the distance. Every input is checked before max; any saturation/nonfinite value
falls back. Independent raw-saturation-onset and non-first-NaN reproductions pass.
Final integrated suite331 tests+28 subtests and CPU21 real-source package smoke pass.

New run runs/koi-minimum-clearance-ab-20260930-r2 uses the same48-slot schedule and
unchanged measurement/gates asv1; source/runtime/claims/resources are checked again.
Mode label minimum_clearance_v1 is the evaluator's generic treatment key; candidate
manifestv2/model/member hashes and study koi-minimum-clearance-ab-r2 disambiguate it.
All original v1 run artifacts/source copies/package/results remain frozen.

This is a counterfactual test of a diagnosed horizon overconstraint, not prior proof
that the shorter hold is physically safe or faster. Do not alter imported sources
or model ZIP while running. No fresh/protected/official action or Git writes.
