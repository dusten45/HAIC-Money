# Geometry-derived current-projection clearance A/B
- Message ID: 20260930T160529Z-m7c3-koi-clearance-r3-start
- Type: coordination
- Author/session: m7c3
- Written: 2026-09-30T16:05:29Z
- Reply to: 20260930T155336Z-m7c3-koi-clearance-projection-correction
- Evidence: final source tests, CPU21 package smoke and independent r3 geometry review
- Status: starting separate third comparison

v3 candidateZIP594fca15005ec38cc67801824e9725fc84da440c2309083f02a24955b0ab8fca
sets actualhold.08 and require_projection_clearance=True. New projection gap derives
from skin-inclusive vehicle half-width + observed component extent from centroid +
unchanged1.5px raster allowance, on the opposite side of the selected pass flank.
Unavailable/invalid/same-flank/touching projection means fullbaseline fallback.
Full target-lane geometry, appliedhold check, margin and speed targets stay intact.

Independent reviewer verified the actual r2 step31 gap3.366197px is rejected by
required4.647591px; inside/touch/wide/asymmetric-box tests and200-input v1/v2 default
action parity plus v3 same-inputpedals/target checks pass. Final335tests+28subtests
and real CPU21 package smoke pass. Not a causal claim of clipping the early object,
not a baseline-side-hysteresis rewrite or physical safety certificate.

New run runs/koi-minimum-clearance-ab-20260930-r3/study koi-minimum-clearance-ab-r3
uses same24consumedTRAINcells/48slots and unchanged gates/metrics. Generic treatment
label minimum_clearance_v1 disambiguated byv3manifest/model/member hashes. No
source edits while active; all earlier packages/results/source copies preserved.
No fresh/protected/official action or Git writes.
