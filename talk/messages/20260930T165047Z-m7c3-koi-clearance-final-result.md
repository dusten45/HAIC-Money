# Minimum-clearance comparisons complete; keep crossing baseline
- Message ID: 20260930T165047Z-m7c3-koi-clearance-final-result
- Type: result
- Author/session: m7c3
- Written: 2026-09-30T16:50:47Z
- Reply to: 20260930T160529Z-m7c3-koi-clearance-r3-start
- Evidence: immutable three full results and independent primary audits
- Status: complete, candidates not adopted

Adaptive-v1 remains failed/nonadopted; speed-target investigation is closed by
user direction. Crossing_projection source reconstruction remains the fixed KOI
baseline, not restoration of the missing original ZIP. Other lanes/root Agent/
official model status remain untouched. No fresh/protected/official/Git action.

Implemented separate minimum supported lateral target from obstacle boundaries,
skin-inclusive full hull/wheel footprint and asphalt, .05px displacement grid,
not global steering scaling. Existing speed targets/pedal computation unchanged.
Same24 consumed TRAIN layouts/8 geometry seeds used in3 independent48-slot runs
(144 episodes); frozen packages/source copies/negative results all preserved.

v1 preserves21 finishes/damage1.4/collision-positive decisions7, but only5 steering
changes and path+0.038408%; full-current-command guard extrapolated beyond actual
.08s hold. Separatev2 corrected hold horizon, but loses3/38301 with one new obstacle
hit and3 lost baseline windows. Diagnosis verifies clean initial changed-object
passage, then a later inherited flank flip and on-road obstacle stall, not premature
return. Desired-lane clearance did not validate current-motion projection.

Separatev3 adds geometry-derived projection footprint gap/opposite-flank requirement,
preserving full target-lane/actualhold/road/margin checks. ZIP594fca15005ec38cc67801824e9725fc84da440c2309083f02a24955b0ab8fca.
Final r3 keeps21/lost0/gained0, all119 windows and144-object safety, damage1.4/7
collision-positive decisions,2 existing hit objects each/no newhit. Lost r2cell
now repeats ALL221 recorded actions/world decisions+881 raw records,17.6sfinish.

Primary final comparison B->C:138 complete matched passages mean-of-min clearance
4.667251->4.485511 units; minimum1.371783->.662154 (not collision-free whole inventory).
119 windows pathmean50.483046->50.412679, meanmaxlat6.207286->6.151952, globalmaxlat
9.334206->9.608379. Meanabssteer.141869->.142343/integral.148389->.148885 command-s.
Cellweighted21 path decrease only.138079%, no qualifying2% geometry;21matchedlap
mean19.054286->19.057143s (+2.857143ms). All3improvement gatesFAIL. No real
return_centerline calls;144 returns EACH1returned/118windowcensored/6unpassed/
19invalid. Sole both-returned pair.24->.28s is descriptive n1, not cohort benefit.

All3 candidates NOT ADOPTED; no meaningful safe shorter route/earlier return
established. Final335 tests+28subtests and independent source reviews pass; CPU21
v3 package smoke/deterministic rebuild samehash. Independentaudits1/2/3 verify every
episode/raw/source/model/ledger/gate, with raw/hidden-state/process-hash limitations
explicit. Final audit experiments/koi-minimum-clearance-ab-r3-audit.json SHA
45fd860befe916ad9284a386b99ccb2c384f8d2f4a69c6130b9322dcfab53e22.
Current-state, experiment index, closure decision and active KOI analysis updated.
