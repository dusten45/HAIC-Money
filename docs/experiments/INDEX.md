# Experiment Evidence Index

## H4 Successor Research Closure (2026-10-06)

**Closed by user direction; no new experiment.** Keep the champion for this
submission and stop current H4-successor tuning and driving. The
[closure decision](../decisions/INDEX.md#close-the-current-h4-cost-successor)
preserves the [ABC evidence](../../experiments/joint-single-branch-v1-result.json),
observer, compact physics model and validation assets. Short benefit survives
actual feedback but does not persist to the lap; terminal-state/long-term utility
is a separate, unapproved research problem. Historical plans and results below
remain frozen and do not authorize another run or outcome-fitted weight change.

## Single-Intervention Feedback Branch (2026-10-05)

- [Fixed plan](../../experiments/joint-single-branch-v1-plan.json), [A/C/fixedH4 reuse proof](../../experiments/joint-single-branch-v1-reuse.json), [one-reset claim](../../experiments/joint-single-branch-v1-claim.json), [admission](../../experiments/joint-single-branch-v1-admission.json), [result/reproduction](../../experiments/joint-single-branch-v1-result.json).
- [ABC summary](../../runs/joint-single-branch-v1/summary.json) SHA`cc51986f...`: only B NEW, A/C REUSED. Same3/3184000005, exactlyoneBintervention34 followed by genuinechampionfeedback onnewBobservations, actualhistorycorrected. Allfinish: A21.720s/B21.860s/C22.100s; B-A+140ms,C-B+240ms. All-wheel-road-loss spells0/1/2; contacts/collision/damage0. One consumedroad, not replicatedgeneralization.
- [Same-scene actualfeedback H4 costs](../../runs/joint-single-branch-v1/feedback-cost.json) SHA`5004b48a...`: B-A[-.742509,-.624026], C-A[-1.113553,-.938447], all17ticks supported. Reusedexactonline-state fixedtail gain also reproduces. B-A GTprogress+.626208 at.32s becomes-.331300 at1s/-.699620 at2s. Thus immediatecostbenefit exists even underfeedback, but doesnotpersisttolap; no proof ofaunique long-term mechanism.
- [One-shot adapter](../../haic/algorithms/joint_control/single_intervention.py), [one B operator](../../scripts/run_joint_single_branch.py), [actualfeedbackcost operator](../../scripts/score_joint_branch_feedback.py). No futureAaction replay, nooldsourcechange oroutcome fitting.233tests+61subtests then24targetedtests(1new) pass;234distinctcases. One newreset,274decisions,1093drivingraw+51warmup; no retry. CurrentC excludedfromsubmissioncandidates; Bdiagnosticnegative, champion/distillation preserved.

## Joint Diagnosis And Conditional Pilot (2026-10-05)

- [Fixed new-data plan](../../experiments/joint-temporal-diagnosis-v1-plan.json), [diagnosis and residual derivation](../../experiments/joint-temporal-diagnosis-v1-diagnosis.json), [scoped consumed-road exposure](../../experiments/joint-temporal-diagnosis-v1-exposure.json).
- [931-decision funnel](../../runs/joint-temporal-diagnosis-v1/funnel.json) SHA`00d5f270...`:856pre exclusions+75comparisons;59full-cost-supported,19risk-pass. Of19,2 fail shared stress and17 fail original allowance; all4 original robust gains fail road margin. Exclusive order and nonexclusive overlaps are separate; no observer-only explanation.
- [Separate calibration](../../runs/joint-temporal-diagnosis-v1/envelope-calibration.json) SHA`a0ceed50...`, [old-pair retrospective](../../runs/joint-temporal-diagnosis-v1/envelope-retrospective.json) SHA`db712808...`: correctly paired oldq=.449619 is a stronger every-prediction target, not duplicate numeric addition. New per-reference envelope excess plus NEW residual floor .05 leaves all physical guards unchanged. Old-pilot17would-pass classifications are posthoc, not executed interventions or efficacy.
- [New-pair result](../../experiments/joint-temporal-diagnosis-v1-paired-result.json), [primary analysis](../../runs/joint-temporal-diagnosis-v1/pairs/analysis.json) SHA`1043a3c6...`: three complete pairs,9resets/243actual decisions/1431raw including459warmup, no retries, exactrepeat/prefix parity. Actual cost3alternative wins, newpredicted2wins, onlycell3/3184000005 passesunchangedphysicalguard and actualbenefit/tube/harm gate. Other2abstentions retained; nofit onnewdata. Observed-reference costprogress is not GTprogress orlap utility; rawrewarddelta0each.15correlatedreference labels/6pairedpaths covered, not a safetyguarantee.
- [Final result/reproduction](../../experiments/joint-temporal-diagnosis-v1-result.json), [conditional full summary](../../runs/joint-temporal-diagnosis-v1/full-pilot/summary.json) SHA`72dad221...`: ONLY3/3184000005, matchedprefix33/firstchanged34,6interventions/277decisions. Bothfinish,21.720s baseline versus22.100s successor,+380ms;0contact/collision/damage, but2all-wheel-road-loss spells versus0. This is actual negative ONE-road effect, not another no-op and not promotion evidence. All16forecasts future-control-mismatched; H4coverageN/A, not predictedaccidents. Wholeact CPU p99=173.189ms/max178.955ms onthisroad, no general CPUoptimizationclaim.
- [Funnel CLI](../../scripts/diagnose_joint_temporal_funnel.py), [calibration CLI](../../scripts/calibrate_joint_paired_envelope.py), [new performance comparator](../../haic/algorithms/joint_control/paired_residual.py), [paired operator](../../scripts/collect_joint_envelope_pairs.py), [isolated envelope successor](../../haic/algorithms/joint_control/envelope_successor.py), [conditional full operator](../../scripts/run_joint_envelope_pilot.py). Combined244tests+72subtests, then29hardened-gate tests(6new) pass;250distinct cases covered. Fivezero-reset imports pass. Total11resets(9paired+2full), no retries/replacements/retuning/extraepisode. No champion, distillation, oldartifact, officialmodel or safety-guarantee change; source/manualrecords published, generated evidence local.

## Interval Road And Temporal Pilot (2026-10-05)

**Cost support improved; zero-intervention pilot does not evaluate candidate
effect.** Separate consumed-TRAIN study, preserving the champion and old evidence.

- [Plan](../../experiments/joint-temporal-interval-v1-plan.json), [scoped exposure](../../experiments/joint-temporal-interval-v1-exposure.json), [claim](../../experiments/joint-temporal-interval-v1-claim.json), [admission](../../experiments/joint-temporal-interval-v1-admission.json), [result and reproduction](../../experiments/joint-temporal-interval-v1-result.json).
- [Interval scorer](../../haic/algorithms/joint_control/interval_comparison.py), [isolated successor](../../haic/algorithms/joint_control/successor.py), [paired operator](../../scripts/analyze_joint_temporal_intervals.py), [pilot operator](../../scripts/run_joint_temporal_pilot.py). Unknown space is never filled; shared full-H4 support, state/reference stresses, unchanged weights/physics and honest one-action commit.
- [Eight-pair replay](../../runs/joint-temporal-interval-v1/paired-analysis.json) SHA`19e2fc4e...`: support2/8->6/8; actual3 baseline/3 alternative/2 unsupported. Five reference-only material predicted orders agree; after CAL cost-error allowance .449619, all orders abstain. Components and progress/clearance tradeoffs remain explicit, not tuned to desired winners.
- [Empirical calibration](../../runs/joint-temporal-interval-v1/calibration.json) SHA`5c8fe665...`: three CAL roads, two roads excluded from fitting; total state+dynamics H4 position1.095715/yaw.092495 envelopes, same-scenario whole-path misses0/6 versus central-only2/6. Mapping calibration separate. Consumed TRAIN, not safety or statistical coverage.
- [Frozen pilot protocol](../../runs/joint-temporal-interval-v1/pilot/protocol.json) SHA`11fac3c4...`, [six-episode receipt](../../runs/joint-temporal-interval-v1/pilot/episode-report.json), [summary](../../runs/joint-temporal-interval-v1/pilot/summary.json) SHA`91518e06...`: three predeclared roads, each arm2/3 finishes,19.960s/17.860s and one identical off_track DNF.931 decisions per arm; all actions/raw records equal; damage/contact/collision0.
- Successor compares75 decisions, intervenes0/931, abstains100%; candidate effect and post-intervention failure risk NOT EVALUATED. All75 forecast continuations differ from later issued actions, leaving0 qualified pilot range windows. Whole-act CPU p99=253.613ms/max308.017ms, not the old component timing estimate.
- 197 tests+65 subtests and zero-reset preflights pass. No extra paired resets, retries, broad audit rerun, champion/distillation edit, official action or model adoption. Raw/frozen/generated artifacts remain local; Git alone does not include the external CPU21 environment or all historical primary data.

## Temporal Observer And Diverse Pairs (2026-10-05)

**Relative-effect signal beyond launch; no qualifying pilot domain.** Persistent
original-physics/HUD/image state estimation, same-state uncertainty stresses and
progress-aware costs were frozen before eight diverse paired outcomes. Absolute
trajectory error was not used as a blanket ranking stop.

- [Plan](../../experiments/joint-temporal-v1-plan.json), [protocol](../../experiments/joint-temporal-v1-protocol.json) SHA`61683b20...`, [result](../../experiments/joint-temporal-v1-result.json), [primary analysis](../../runs/joint-temporal-v1/analysis.json), [audit](../../experiments/joint-temporal-v1-audit.json), [road diagnosis](../../experiments/joint-temporal-v1-road-support-diagnosis.json), [compact statistics](../../experiments/joint-temporal-v1-statistics.json).
- Eight starts/two per high-speed straight,left/right entry,braking turn; six cells/five consumed road geometries.24 baseline/repeat/alternative resets,768decisions,4296raw including1224warmup,232champion queries. No retries or fresh-confirmation claim; all prefixes and repeats verify.
- Temporal observer143/174 unique baseline states valid; anchors7/8, current-image motion flags3/8. Inferred mapping yaw bounds miss25/44, so heuristic intervals are not guarantees. Only original1/1/1 coefficients; prior4x correction excluded.
- H4 physical effect centre signs8/8 per component, median error/effect right/forward/yaw/speed `.09170/.02088/.11998/.00515`; same19-coordinate-state stresses preserve material signs7/8,8/8,7/8,8/8. Dependent samples, not32independent successes.
- Cost/reference support2/8, both baseline wins, no confidently wrong supported ranking, sixunknowns. Allfour alternative GT-progress wins are amongunknowncosts. Coverage2/25,0/2,2/5,2/13; cost-bias absence and usable intervention coverage are unproven.
- Five unknowns come from a single ambiguous gray edgepixel plus strict contiguous-reference requirements, not bad mean historywarps or missing physicalroad. First7 fullyrepair ego; causalpose-oracle diagnostic doesnot improve2/8support. Brake2 is separately state/mappinginvalid. No frozen source/gate retuning.
- All16 actual suffixes physicallysafe, but8/16 miss `.5world/.05rad` same-scenario tube; all5predicted-safe paths misscontainment. These are uncertified envelopes, not five collision events. Independent regionalgate0/4; no successor pilot or official action.
- [Observer](../../haic/algorithms/joint_control/observer.py), [comparator](../../haic/algorithms/joint_control/comparison.py), [collector](../../scripts/collect_joint_temporal_pairs.py), [analyzer](../../scripts/analyze_joint_temporal_pairs.py).200tests pass; independent856pin/108499assertion audit agrees within1.707e-15. Protected champion source/ZIP unchanged.
- [Development](../../experiments/joint-temporal-v1-development.json), [raster equivalence](../../experiments/joint-temporal-v1-cost-equivalence.json), [optimized timing](../../experiments/joint-temporal-v1-latency-optimized.json), [exact source archive](../../experiments/joint-temporal-v1-source-archive.json). New measured segment-sum p99~169.7ms is not integratedpilotact. Collection259.60s is not inference latency.

## Joint Relative-Effect TRAIN Probe (2026-10-05)

**Local pairwise signal, but observer coverage is only1/3 starts. No controller
promotion.** This distinct predeclared measurement follows positive posthoc
directional-response evidence, preserving the failed archive gate0/3 unchanged.
It addresses relative comparison rather than requiring an exact simulator clone.

- [Plan](../../experiments/joint-prediction-probe-v1-plan.json), [protocol](../../experiments/joint-prediction-probe-v1-protocol.json), [scoped exposure](../../experiments/joint-prediction-probe-v1-exposure.json), [immutable consumed-use claim](../../experiments/joint-prediction-probe-v1-claim.json), [result](../../experiments/joint-prediction-probe-v1-result.json), [analysis](../../runs/joint-prediction-probe-v1/analysis.json) and [independent audit](../../experiments/joint-prediction-probe-v1-audit.json).
- Three consumed Track1 cells3184000002/0013/0015, anchor10, baseline/repeat/one joint alternative with common four-hold continuation. Exactly9 resets/126 decisions/963 raw ticks; strict prefix and complete repeat parity. No retries, new controller, full-lap A/B or official evaluation.
-57 unique instantaneous frame-label records give HUD speed/joint/yaw p95 `1.0993/.0029509/.0290693` in respective units; wheel-omega errors/abstentions remain. Only seed0015 passes motion validity at the anchor; seeds0002/0013 are retained as flagged diagnostics.
- Source-default.32s six-endpoint median position/yaw/speed-change error: full oracle `.008409/.002454/.002735`, runtime `.647272/.037612/1.362050`. FIT4x correction gives runtime `1.720888/.025944/8.419092` and does not transfer well. No new fitting or coefficient selection on these probe cells.
- H4 source-default runtime effect signs match3/3 per component; median relative errors right/forward/yaw/speed `.0814/.0403/.0328/.00420`. Fixed image-derived local tracking-proxy ordering3/3 is descriptive, only1/1 within runtime-valid support; not lap, broad action-ranking or safety evidence.
- [Collector](../../scripts/collect_joint_prediction_probe.py), [analyzer](../../scripts/analyze_joint_prediction_probe.py);80 focused tests pass. Audit checks499 hashes,1161 exact arrays,6885 numeric values with zero discrepancy. Raw physical/observation evidence is retained; no stronger hidden-solver cloning claim.
- Collection464.50s includes durable capture/fsync; sampled RSS max324,120,576B, not a continuous peak or inference-time measurement. Champion ZIPc9e376a0.../all11 sources and142 environment pins unchanged. No further resets or larger search follow automatically.

## Joint HUD And Physics Prediction (2026-10-05)

**Observer signal confirmed; prediction gate failed0/3 TEST roads.** Separate
joint-control research, not a champion patch or full controller. Missing saved
joint branches is a data gap, not a technical stop; the new user instruction
permits small TRAIN collection only after promising prediction validation.

- [Predeclared plan](../../experiments/joint-prediction-v1-plan.json), [study summary](../../experiments/joint-prediction-v1-result.json), [frozen protocol](../../runs/joint-prediction-v1/protocol.json), [primary result](../../runs/joint-prediction-v1/result.json) and [independent audit](../../experiments/joint-prediction-v1-audit.json).
- 42 saved TRAIN episodes/18,453 frames/9 geometry seeds, split 4 FIT/2 CAL/3 TEST. No recovery-mask filtering; repeated frames and exact input/action/outcome windows are deduplicated within whole-road splits.
- HUD speed on TEST has p95 error1.170-1.194 units/s. Joint/yaw/wheel decoders use source-based geometry, but exact instantaneous labels are absent. Yaw integral consistency p95.0189-.0236rad is not direct yaw-rate accuracy. Causal displacement p95.343-.348 units on3070/5168 supported intervals.
- [FIT omega ablation](../../runs/joint-prediction-fit-ablation-v1/result.json) rejects omission as a fix. [Exactly9 FIT corrections](../../runs/joint-prediction-fit-calibration-v1/result.json) select effective mass/yaw inertia scales4/4, tire1; no grid expansion. These are approximation corrections, not actual physical constants.
- TEST.32s mean of three geometry medians, position/yaw/speed-change: estimated`.7289/.1835/1.6164`; same-state persistence`.8718/.2663/.7007`; partial oracle`.6922/.1717/1.5921`. Retrospective executed-action prediction is not counterfactual action ranking.
- Runtime-valid.32s windows2689/5145. All3 roads have relative position/yaw signal, but yaw p95`.439-.489rad` exceeds fixed`.25rad`, and median exceeds`.08rad`. This archive-only phase had no simulator/Agent execution; the later separate probe above does not change this failure. No official action or champion modification.
- [Independent audit](../../experiments/joint-prediction-v1-audit.json), SHA`2faf77e5...`, reconstructs 1512 error arrays/580608 values exactly and matches all 1296 prediction-summary groups/gate. Motion vectors and omitted-window initial-state dictionaries were not separately saved, limiting passive revalidation of those quantities.
- [Components](../../haic/algorithms/joint_control/), [isolated offline operator](../../scripts/validate_joint_prediction.py), [FIT diagnosis](../../scripts/diagnose_joint_prediction_fit.py);45 focused tests pass. Frozen-source full run30.37s/150,081,536B peak RSS. All raw/frozen evidence retained.

## KOI Fixed Sprint72 Handback Relief (2026-10-04)

**REJECTED / NOT ADOPTED; exact specification closed, no retuning.** One stateless
pre-arrival pedal substitution, gas0/brake`clip(.02*(v-72),0,.15)`, only under
the exact diagnosed T60/spatial/no-hazard/no-intervention guards. Existing sprint,
floating-point boundaries, target, arrival, steering and shield code stay fixed.

- [Protocol](../../runs/koi-sprint72-relief-v1/protocol.json), SHA`7fca6b24...`: all8 consumed pairs/16 contemporary natural episodes/5 roads; exact champion`c9e376a0...` versus separate candidate`e7062c66...`.
- [Primary result](../../experiments/koi-sprint72-relief-v1-result.json), SHA`b20da6b3...`: complete and matched,7/8->7/8 finishes but lost1/gained1; all48 objects/arm, damage/collision/contact/hits0->0.
- [Independent passive audit](../../experiments/koi-sprint72-relief-v1-audit.json), SHA`cfe02a62...`: no discrepancy across116 run artifacts,4094 history/law decisions,8 prefixes,96 object records, native finish-line timing and exact frozen-analyzer output bytes; confirms REJECTED, not adoption.
- Lost finish1/0013 has345 all-wheel-offroad ticks and6.88s longest spell; gained2/0006 finishes21.8s. Overall all-wheel-offroad64->366 ticks. Common6 laps slow26.667ms on average, with1/0015+180ms and2/0015+80ms.
- Local cap undershoot improvements do not establish complete-cohort efficiency: only4/13 baseline station windows covered, with6 earlier restrictions/2 unreached reverse starts/1 unreached end. Own-window cycles3->0 and all-episode brake-to-gas215->198 are descriptive; common6 transitions135->136.
- Historical reverse steps194/195/447/448 remain eligible and evaluated. Candidate has no reverse phase in that cell because earlier driving changes and it finishes, not because the policy uses hidden heading or those rows were filtered out.
- [Runtime](../../haic/algorithms/koi/sprint72_relief.py), [operator](../../scripts/evaluate_koi_sprint72_relief.py), [passive analyzer](../../scripts/analyze_koi_sprint72_relief.py) and [CPU21 preflight](../../runs/koi-sprint72-relief-v1/preflight.json). Integrated159 tests+132 subtests PASS; exact actual-brake history and subsequent impact/steering behavior retained.

See [full metrics, limitations and closure](../architecture/koi-baseline-analysis-2026-09-30.md#32-fixed-sprint72-handback-relief-2026-10-04).
Champion ZIP/all11 sources/root Agent/prior candidates remain unchanged. No second
candidate, coefficient change, FP repair, timer/hysteresis, repeat, fresh/protected
generalization or official action. Existing staged/unrelated work remains intact.

## KOI Outside Entry Pre-Positioning (2026-10-04)

**CLOSED AT DIAGNOSIS; no candidate/A-B.** Separate from the closed direct-apex
pull, this study checks outside setup10/20/30 decisions before a sharp corner.
Only existing eight champion episodes/five geometries were read. Natural shorter
path plus lower maximum wheel-angle contrasts are confined to one geometry and
have approach/layout confounding; no replicated causal effect is established.

- [Natural entry histories, actual outcomes and geometry-pair comparisons](../../experiments/koi-corner-entry-natural-v1.json).
- [288 full-cost preparation/apex/return paths and sensitivities](../../experiments/koi-corner-preposition-diagnosis-v1.json).
- [Natural analysis script](../../scripts/diagnose_koi_corner_entry_natural.py) and [geometric recalculation](../../scripts/diagnose_koi_corner_preposition.py).
- [Scope, methodology, tables and closure](../architecture/koi-baseline-analysis-2026-09-30.md#31-outside-entry-pre-positioning-diagnosis-2026-10-04).

All55 complete observations, one censored and six unreached remain. All288 paths
include setup and return cost, preserve approach shape and actually reach outside
entry. Nineteen shorten distance, but none saves>=2 world units; maximum1.459.
An any-positive/demand-relaxed sensitivity leaves only two same-case paths,
saving.187/.228. No short-offroad hard rejection, physical-impossibility proof or
measured lap/safety gain is claimed. Eleven focused tests pass; frozen policy and
all prior closed directions remain unchanged. Zero new simulator resets.

## KOI Soft-Boundary Corner Cutting (2026-10-04)

**CLOSED AT DIAGNOSIS; no runtime candidate or A/B.** Eight existing champion
episodes/five geometries give62 sharp-corner observations,55 complete;44 complete
actual paths are already shorter than corresponding centerline arcs. Of15 complete
windows without recorded obstacle/recovery interference, four on two geometries
have >=2-unit/3% geometric savings, but none passes the combined fixed-speed
steering/lateral-demand/reentry screen. This is not absence of geometric savings
or a proof that every possible cutting controller is unsafe.

- [Result, primary hashes, all220 path alternatives and limits](../../experiments/koi-corner-cutting-diagnosis-v1.json).
- [Passive geometric analyzer](../../scripts/diagnose_koi_corner_cutting.py).
- [Definitions, example table and closure](../architecture/koi-baseline-analysis-2026-09-30.md#30-soft-boundary-corner-cutting-diagnosis-2026-10-04).

Short offroad was allowed, not rejected categorically. Unconstrained chords are
lower bounds, not feasible routes. Time/grass/reentry/steering are offline proxies;
the unchanged-speed analysis is optimistic about traction. Four focused synthetic
checks pass. Frozen champion and all previous closed directions remain unchanged.

## KOI Corner Target-Speed Diagnosis (2026-10-04)

**NO CLEAR HEADROOM / direction CLOSED; no candidate or A/B.** Passive analysis
of eight existing frozen champion TRAIN episodes/five roads/2,130 decisions;
832 decisions isolate the actual target-to-pedal calculation. Geometric windows
include68 complete/one censored/nine unreached observations, not78 independent
corners. Among23 unconfounded complete passages,12 clean above-target entries
yield only one sustained same-spread-band headroom observation on one road.
No band demonstrates the repeated cross-road safety margin required to proceed.

- [Result, input hashes, shape/heading tables and all corner windows](../../experiments/koi-corner-target-diagnosis-v1.json).
- [Passive reproduction script](../../scripts/diagnose_koi_corner_target.py).
- [Scope, definitions, limitations and outcome](../architecture/koi-baseline-analysis-2026-09-30.md#29-corner-target-speed-headroom-diagnosis-2026-10-04).

HUD targets and physical speeds are separate measurements; local projected road
margin is not exact curved-road clearance. Existing safety limitations and
censored observations are retained. Four focused synthetic checks pass; no broad
audit, policy execution or new environment interaction. Champion bytes and prior
exit-throttle/overavoidance closure are preserved.

## KOI Corner-Exit Throttle Diagnosis (2026-10-04)

**HYPOTHESIS NOT CONFIRMED / direction CLOSED.** Existing frozen champion logs
only: eight consumed TRAIN episodes/five roads/2,130 decisions. Of36 putative
alignment onsets,30 already have gas>=.5. All six low-gas onsets fail the existing
projected-space gate and retain19.62-70.38deg of upcoming road heading change
within28.8m; four also exceed their current speed target. Current-frame target
and current-speed pedal formula checks find no unexplained retained deceleration.

- [Primary diagnosis, definitions, event contexts and hashes](../../experiments/koi-corner-exit-diagnosis-v1.json).
- [Passive reproduction script](../../scripts/diagnose_koi_corner_exit.py).
- [Narrow plan/outcome](../architecture/koi-baseline-analysis-2026-09-30.md#28-corner-exit-throttle-diagnosis-2026-10-04).

No candidate, A/B, new reset, generalization or official action. Frozen champion
and shield remain byte-identical; overavoidance remains closed. No safety or lap
improvement is claimed; archived control finishes remain7/8, damage/collisions0.

## KOI Stateless Avoidance Magnitude (2026-10-02)

**REJECTED / NOT ADOPTED; overavoidance optimization CLOSED by the user's stop
criterion.** One current-observation clearance/risk-dependent replacement of
nominal `.34/.55`, followed by byte-identical submitted v1 shield. No retained
plan, feedback/recovery/release state, margin reduction or steering-release.

- [Frozen protocol](../../runs/koi-avoidance-magnitude-v1/protocol.json),
  SHA`1456126e...`:4 consumed ordinary TRAIN pairs/8 contemporary episodes/2
  actual road geometries; exact champion`c9e376a0...` and shield`ad772bde...`.
- [Primary result](../../experiments/koi-avoidance-magnitude-v1-result.json),
  SHA`55d489f9...`: all8 valid with exact logged pre-divergence prefixes;
  finishes4->3, lost1/gained0, damage0->1.4, collision decisions0->7, physical
  contact events0->4 and hit objects0->3 across all24 objects/arm. New hits:
  1/0013 object4,1/0015 object1 (lost finish),3/0015 object1.
- Baseline window and return coverage each falls24->19. Three-retained-cell
  lateral-48.188963%, path-0.526375%, steering integral-12.230624% are conditional,
  NOT a valid4-cell efficiency gain. Road0013 path+2.813968%, integral+14.947015%,
  variation+15.514245% and kept lap+80ms regress. Kept3 mean lap-120ms omits one
  lost baseline finish; no adoption credit. Common19 return bounds/censors improve
  conditionally, but5 missing followups violate coverage.
- [Zero-reset runtime review](../../experiments/koi-avoidance-magnitude-v1-runtime-review.json)
  and [exact CPU21 preflight](../../runs/koi-avoidance-magnitude-v1/preflight.json):
  integrated277 tests+133 subtests PASS. Pre-freeze independent review resolved
  endpoint-only swept risk and source-specific raw/wrapper counter/damage bindings.
- Candidate nominal98/820 actions changed; same shield9->25 interventions,
  including20 nominal/shield overlaps. Source, root Agent, shield and all11
  frozen champion members remain unchanged. Preserve only as negative evidence;
  no tuning/repeat, protected/fresh/Track4 or official action.
- [Independent primary audit](../../experiments/koi-avoidance-magnitude-v1-audit.json),
  SHA`7bc1297f...`: all63 study-root files/frozen inventories/preservation pins
  verify; pure re-analysis exactly matches the primary result and all17 gates.
  Independent raw reductions confirm safety,19/24 window/return coverage,
  conditional-only efficiency and kept-lap bias. No discrepancy or driving replay.

See [architecture section27](../architecture/koi-baseline-analysis-2026-09-30.md#27-stateless-bounded-avoidance-magnitude-2026-10-02)
and the [closure decision](../decisions/INDEX.md#close-koi-overavoidance-optimization).

## KOI Nominal Trajectory Selection (2026-10-02)

Separate bilateral approach/pass/rejoin generation and actual lagged pursuit
rollouts before byte-identical collision-shield v1. No margin/release retuning,
root Agent change, baseline replacement or official action.

- [Frozen protocol](../../runs/koi-nominal-trajectory-v1/protocol.json),
  SHA`77bda836...`:16 contemporary episodes/eight consumed TRAIN layouts/five
  roads; ordinary four layouts/two roads. Both arms load exact submitted
  ZIP`c9e376a0...` and shield`ad772bde...`;48 safety objects per arm. No old24,
  fresh/protected/Track4 geometry. Source/environment/runtime/metric/gate pins
  are frozen before resets.
- [Primary result](../../experiments/koi-nominal-trajectory-v1-result.json),
  SHA`b020c3cb...`: **REJECTED**. All16 episodes/eight pairs are valid and fully
  matched. Finishes7/8->7/8, damage/collision decisions/hit objects0->0, no new
  hit; all38 baseline windows and43 common return followups remain covered.
- Ordinary equal-cell window changes: max lateral+0.068713%, path-0.014782%,
  issued steering integral+4.193204%, steering variation+4.125609%. Road0013 is
  an exact no-op, so improvement on both ordinary geometries is not established.
  Common-return bound and steering variation worsen; the seven kept laps average
  +20ms, with3/3184000015+140ms violating the predeclared per-cell+20ms ceiling.
  Preserve this efficiency-regressing candidate as negative evidence, not deploy
  or repeat it. No gate relaxation or baseline edit follows.
- [Zero-reset preflight](../../runs/koi-nominal-trajectory-v1/preflight.json):
  integrated75 tests+30 subtests PASS and both exact CPU21 arms import/construct
  successfully. Focused independent pre-run review found and resolved feedback-
  speed and short-terminal-hold correctness gaps before the source freeze.
- [Independent primary audit](../../experiments/koi-nominal-trajectory-v1-audit.json),
  SHA`9ab10494...`: all64 episode/raw/decision/process hashes and frozen inventories
  verify; full passive analyzer output equals the primary result. Independent
  reductions reproduce all15 gates, ordinary24 windows and43 common return rows.
  Return statuses remain33 returned/10 next-entry censors per arm; descriptive
  bound sum31.555812->31.675812s, not an unbiased return-time mean. Only4/2132
  candidate decisions change nominal actions; six entire pairs are exact no-ops.
  No additional simulator resets, policy replay or artifact discrepancy.

Source-backed diagnosis, implementation contract and rejection are in
[architecture section26](../architecture/koi-baseline-analysis-2026-09-30.md#26-separate-nominal-trajectory-selection-2026-10-02).
These are outcome-selected consumed-TRAIN internal proxies, not fresh or official
generalization evidence. The user-designated submission baseline stays immutable.

## KOI Minimum-Intervention Collision Shield (2026-10-01)

Separate `collision_shield.py` directly wraps fixed crossing_projection, with no
collision_recovery/steering-release dependency or pedal changes. Recovery-state-
machine development is closed by user direction; all old source/evidence remains.

- [Frozen protocol](../../runs/koi-collision-shield-v1/protocol.json), SHA`afeeb356...`:
  same six outcome-selected consumed TRAIN cells/five road seeds,12 serial CPU21
  episodes; baseline ZIP`a4b35c56...`, shield source`ad772bde...`. No Track4 geometry,
  old24/fresh/protected cells, official action or model change.
- [Primary result](../../experiments/koi-collision-shield-v1-result.json): all six
  pairs valid,3/6->5/6 finishes, kept3/lost0/gained2, damage1.4->0, collision-positive
  decisions7->0, hit objects2->0 and no new hit. Physical clearance minimum
  -.012577->.990591m.17 changed actions in16 bursts; maximum2 consecutive/.16s,
  maximum5 per encounter, against a structural6-action cap.
- **NOT ADOPTED under the frozen efficiency gate:** retained3/3184000002 lap
  +180ms violates the predeclared+20ms limit. Other gates pass, including safety,
  finish preservation, exact no-op/pedals/budget, and bounded lateral/heading/offroad.
  Three kept laps average+40ms/path+.714544m. Ordinary controls remain2/2 clean,
  but neither max lateral nor lateral integral/path establishes reduced avoidance.
  This is a promising narrow safety observation, not proof of both user objectives
  or fresh/generalized performance. No post-result threshold change or repeat.
- [Independent primary audit](../../experiments/koi-collision-shield-v1-audit.json),
  SHA`24bbe060...`: all48 primary files and205 frozen inventory pins verify;
  independent raw/finish-tracker reductions reproduce all metrics and9 gates.
  The17/1639 changed actions total1.36s. Administrative encounter lifetime can
  include long baseline-control intervals; max active burst remains.16s, not the
  22.24s maximum first-to-last changed-action span. All six end rearm-blocked.

Details and per-cell caveats are in
[architecture section25](../architecture/koi-baseline-analysis-2026-09-30.md#25-minimum-intervention-collision-shield).
Focused integrated regression:51 tests+31 subtests PASS; actual frozen CPU21 arms
import/construct with zero simulator resets in preflight. Root/crossing/v2/recovery
preservation hashes remain bound and unchanged.

## KOI Collision And Heading Recovery (2026-10-01)

Current comparator is frozen crossing_projection; v2 and all original artifacts
remain unchanged. No Track4 geometry, protected cells, fresh-road claim or official
action. User requested focused checks rather than another broad audit framework.

- [Archived conflict diagnosis](../../experiments/koi-collision-priority-diagnosis-v1.json):
  47 consumed TRAIN pairs, source-exact road/avoidance cancellation before hits,
  plus clean corner controls showing cancellation alone is insufficient.
- [Post-avoidance diagnosis](../../experiments/koi-collision-recovery-diagnosis-v1.json):
  temporary off-road, heading/lateral transitions, backward on-road recovery and
  the rejected priority-only trajectory. Pixel secant heading is curvature-biased.
- [Initial v2-priority result](../../experiments/koi-collision-priority-v1-result.json):
  six pairs,4/6->3/6 finishes; NOT ADOPTED. Frozen run/source preserved at
  `runs/koi-collision-priority-v1/` and superseded by the user's baseline pivot.
- [Crossing-based recovery protocol](../../runs/koi-collision-recovery-v1/protocol.json)
  and [result](../../experiments/koi-collision-recovery-v1-result.json): six pairs,
  12 valid episodes,3/6->2/6 finishes (kept1/lost2/gained1), damage1.4->1.0,
  collision-positive decisions7->5 but one new baseline-clean hit. Longest
  all-wheels-offroad1.14->7.96s, unreacquired departures0->3, sole matched lap
  +3.88s. **NOT ADOPTED; no positive-result repeat or baseline promotion.**

The runtime is separate `haic/algorithms/koi/collision_recovery.py`, source SHA
`eff68d75c1cf737f73ba953d860be42d20b5984ba3501e1418684fa40ea5bfe1`.
Five focused runtime tests and seven evaluator tests passed. Official reward-streak
retirement is measured separately from physical off-road occupancy; see architecture
section24. The later user direction closes further long-lived recovery development;
this is not a theoretical impossibility claim about all heading recovery.

## Frozen KOI v2 TRAIN Generalization (2026-10-01)

User freezesv2, crossing_projection and root Agent unchanged; old24 consumed TRAIN
conditions are forbidden for further tuning/evaluation. One newly audited,
nonprotected TRAIN/dev A/B is authorized, with no protected/blind/official action
or result-drivenv2 correction. **Generalized adoption is rejected by observed
matched safety/completion counterexamples; the original planned matrix is incomplete.**

- [Exposure audit](../../experiments/koi-steering-generalization-v1-exposure.json),
  SHA`b2efd664347d841b595e3362d1b2bc387ac2b36139475fe038129690c5867fe4`:
  24 geometry-global claims for3184000001-3184000024 x tracks1/2/3,72 TRAIN
  obstacle-layout cells/144 slots;1005 evidence pins,0 collisions/blockers and
  zero environment construction/reset/protected observation reads. Candidate-
  lineage-unseen only;15 unrelated legacy PPO uncertainties remain recorded.
- Separate audit/evaluate/analyze adapters preserve all historical sources and
  artifacts, exactv2 ZIP`b1911d7d...`/baseline`a4b35c56...`, passive measurement and
  dynamic baseline-window/finish denominators. Predeclared geometry consistency
  and coverage prevent survivor-only promotion. See the
  [generalization gate](../architecture/koi-baseline-analysis-2026-09-30.md#23-frozen-v2-train-generalization-gate).
- [Frozen protocol](../../experiments/koi-steering-generalization-v1.json) SHA
  `ccc6720f676814eb705e853792eceb31c6c5fb2b2aa695b1144641c2befcd7dd` and
  [separate forensic guard](../../experiments/koi-steering-generalization-v1-evidence-guard.json):
  CPU21 source/runtime/claims/resource preflight passed without resets. Guarded
  finalization must verify original failure-time partial hashes/bytes; the frozen
  raw loader's omission is documented, not patched or silently waived. No model,
  measurement or promotion gate changed. [Independent review](../../experiments/koi-steering-generalization-v1-preflight-review.json)
  SHA`67157ff69ce5e5e038255a5a0820dce392b53a71c8670b9a2ff6eeb56978fa5f`
  passed for mandatory guarded finalization. Serial CPU21 evaluation ended
  naturally at the original fixed budget with95 valid completions, one bootstrap
  timeout and48 unrun slots. No budget change, model update or consumed24 reset.
- [Guarded original result](../../experiments/koi-steering-generalization-v1-result.json)
  SHA`2572e09dfc8dd4a1399a1168ea15692b6be946cbb20a72b18d0532ea00d5843d`
  and [matched diagnostic result](../../experiments/koi-steering-generalization-v1-diagnosis-result.json):
  47 fully matched pairs/16 roads, B41/47 vsC40/47 finishes; kept36/lost5/gained4/
  neither2. Five natural losses across four geometries and two new baseline-clean
  hits irreversibly violate predeclared gates regardless of unexecuted cells.
  The incomplete original attempt is not relabeled complete/passed.
- Matched aggregate damage4.4->3.4/collision-positive decisions22->17/hit objects
  5->4 (282 objects EACH) coexist with two per-cell safety regressions and two new
  hits. Conditional preserved173-window lateral-10.4390%/avoidance integral-15.9312%
  cannot offset17 lost baseline windows,20 lost return followups, common censors
  37->43 or five lost baseline finishes. Whole-episode max lateral29.9527->260.0025m;
  kept36 lap timing-53.889ms is survivor-only, not population improvement.
- Boundary manager [conditional review](../../experiments/koi-steering-generalization-pause-review.json)
  passed, but actual invocation refused before any lock/signal because the parent
  had already finalized. Prepared continuation/composite code was not frozen or
  executed; pending intent/bootstrap evidence is preserved, not called fresh.
  No furtherv2 interaction is needed to establish non-adoption.
- [Independent postrun audit](../../experiments/koi-steering-generalization-v1-postrun-audit.json)
  SHA`c4bd5b112cc9ea414a0e1dd0b22626324257e0607e863d0cbe4c323132875b78`
  exactly reproduces the summary and rehashes source/raw/stream/process/ledger/
  model/copies. Outcome receipt publication is conservatively visible as a frozen
  metadata-scanner `HOLD`; it is preserved, not hidden/waived/reclassified fresh.
- [Independent five-failure audit](../../experiments/koi-steering-generalization-v1-failure-audit-result.json)
  SHA`55897acfaf394f2ff932b44517b4bb7bc2b9f5c9ef8c504a6c1e339cdab49ed8`:
  exact actions/poses/raw prefixes before firstnear_release, no generation
  suppression in all five. All17 released holds and selected-object passages remain
  sampled collision-free; two contacts are different later unreleased objects and
  three losses have road/heading-recovery failures. Conditional improvements in
  every failed geometry do not transfer to whole-episode completion/safety.
  Only a separate unimplemented road-recovery H1 hypothesis is recorded; no v2 fix
  or new candidate/evaluation. See architecture section23 for per-geometry and
  failure timelines, source-boundary and untested-mediation caveats.

## KOI Steering Lifecycle (2026-10-01)

**Final r2: INTERNAL ADOPTION CANDIDATE.** Mechanism-led steering generation/hold/
release correction passes the consumed-TRAIN gates, not an official/fresh
generalization gate. Crossing_projection stays FIXED; root Agent is unchanged.
All minimum-clearance variants remain FAILED / NOT ADOPTED by user direction;
speed-target and margin reduction remain closed. Same24 consumed TRAIN layouts/
eight geometry seeds only; no fresh, protected or official evaluation. See the
[final analysis](../architecture/koi-baseline-analysis-2026-09-30.md#22-final-r2-internal-adoption-candidate).

- [Final r2 result](../../experiments/koi-steering-release-ab-r2-result.json) and
  [r2 protocol](../../runs/koi-steering-release-ab-20261001-r2/protocol.json): all48
  slots and all17 gates PASS. B21/C24 finishes, kept21/lost0/gained3; damage1.4->0,
  collision decisions7->0, whole-episode hit objects2->0/no new hits. All119 baseline
  windows across21 cells retained. Primary equal-cell relative changes:
  max lateral-12.544446%, avoidance duration-3.312396% (NOT5%), avoidance integral
  -17.805737%, path-0.169631%, actual steer integral-5.367767%. Integral satisfies
  the avoidance OR branch; lateral/integral qualify on seven seed IDs each, with
  38300 seam cells excluded from window means. Matched21 lap means including38300
  are19.054285714->19.016190476s/-38.095238ms; own24 candidate mean is unmatched.
  All136 common return followups retained, censors23->19. Full144 prospective
  statuses B113 returned/23 next-entry censored/6 unpassed/2 invalid versus C125
  returned/17 next-entry censored/1 fixed-time censored/1 invalid are kept distinct
  from common-horizon statuses. Common onset-delay/censored-bound mixture
  0.726180453->0.666090505s is descriptive, not an unbiased population mean or
  KM/RMST. [Final independent audit](../../experiments/koi-steering-release-ab-r2-audit.json)
  rehashes48 episode/raw/bound-original-process sets,158 frozen copies/1046 pins,
  10/12 model members and96 ledger rows; all17 gates independently PASS, formula
  and return mismatches0. Both-confirmed109 conditional onset means
  0.724220183->0.646972477s are not136/144 cohort means. CPU21 deterministic
  rebuild reproduces ZIPb1911d7d....
- [Baseline diagnosis](../../experiments/koi-steering-release-baseline-diagnosis-v1.json)
  and [decision/event rows](../../runs/koi-steering-release-diagnosis-20261001-v1/):
  all5703 issued/sum reconstructions exact, nonlinear clipping/float32/replacement
  preserved; all144 objects,42 active-to-measured-off transitions and36 persistent
  separated/outward cases retained. Conservative yaw-envelope timing is not
  archived exact transverse extents or counterfactual safety proof.
- [Consumed provenance](../../experiments/koi-steering-release-consumed-audit-v1.json):
  zero-reset independent rehash of old48 episodes/raw/source inventories and exact
  24 layouts/88 claims, no allocation/protection blocker. Not a freshness claim.
- [First run protocol](../../runs/koi-steering-release-ab-20261001-v1/protocol.json)
  binds candidate3566a5c2.../source0850c521...,12 members with9 frozen dependencies,
  same48 slots; completed. [First result](../../experiments/koi-steering-release-ab-v1-result.json)
  and [independent audit](../../experiments/koi-steering-release-ab-v1-audit.json)
  show23 vs21 finishes but kept20/lost1/gained3 and new hits1:38301/object1 and
  2:50301/object4,117 of119 windows/134 of136 return followups retained. NOT ADOPTED;
  conditional survivor gains are not an accepted improvement. Original6px band/targets/margins unchanged,
  near-only staged release with current reentry rearm and hypothetical motion
  guards, not steering-wide scaling.521 tests+28subtests and CPU21 smoke pass.
- [Zero-reset publisher preflight failure](../../experiments/koi-steering-release-preflight-package-interface-v1.json):
  oldv1 manifest shape did not match new operator contract; preserve original
  ZIP/manifest. V1a receipt fixes only schema, with byte-identical policy and no
  prior run directory/reset/episode; not a second driving comparison.
- [First-hit diagnosis](../../experiments/koi-steering-release-v1-new-hit-diagnosis.json)
  and [lost-finish diagnosis](../../experiments/koi-steering-release-v1-lost-finish-diagnosis.json)
  trace later original-generator motion-only flank reversals; released earlier
  objects pass without contact. No unique wheel-lag/slip explanation is claimed.
  Separate [r2 protocol](../../runs/koi-steering-release-ab-20261001-r2/protocol.json)
  freezes narrow ambiguous-generation correction ZIPb1911d7d... (flagTrue), same
  48slots/gates/cells,525 tests+28subtests and actual CPU21 generation smoke.
  Original unambiguous/sign-crossing choices retained; no blanket flank freeze,
  speed-target or margin tuning. DefaultFalse preserves oldv1; source52f54099.../
  helpercb1149b0... and all9 byte-identical baseline dependencies are bound in the
  [v2 manifest](../../submissions/koi-steering-release-v2.manifest.json). Its
  UNEVALUATED status is the frozen preflight snapshot, not the final r2 verdict.
  Corrected full A/B completed; firstv1 negative proof is not relabeled.

No automatic next evaluation or official/root-model promotion. Future confirmation
requires separate explicit user authorization and provenance/exposure gates;
no consumed cell is relabeled fresh and no fresh-cell availability is promised.

## KOI Minimum Clearance (Closed, 2026-09-30)

**Closed by user direction, 2026-10-01:** all variants FAILED / NOT ADOPTED;
do not continue safety-margin reduction. Preserved evidence follows, not an
active candidate or an authorization for further margin tuning.

**Completed, NOT ADOPTED:** three48-episode comparisons (144 episodes total) reuse
only the same24 consumed TRAIN layouts. Fixed crossing baseline remains unchanged;
no speed-target tuning, fresh/protected/official action or root Agent replacement.

- [Final r3 result](../../experiments/koi-minimum-clearance-ab-r3-result.json),
  [independent final audit](../../experiments/koi-minimum-clearance-ab-r3-audit.json),
  [r3 protocol](../../runs/koi-minimum-clearance-ab-20260930-r3/protocol.json) and
  [v3 manifest](../../submissions/koi-minimum-clearance-v3.manifest.json):
  current-motion full-footprint projection gate plus minimum supported lane and
  applied.08s command checks restores21 finishes, damage1.4/collision7 and all119
  paired windows/no new hits. Cell-weighted path-0.138079% does not meet2% on two
  geometries; matched21-lap mean+2.857143ms, max lateral9.334206->9.608379 units,
  steering mean/integral slightly increase. Not adopted; no early-return claim.
  Final335 tests+28 subtests, independent source review and CPU21 package smoke pass.
  All48 episodes/raw pairs and150 frozen source copies rehash correctly.138 matched
  complete passages have mean-of-min-clearance4.667251->4.485511 units, minimum
  1.371783->0.662154; existing144-object hit count stays2/2. All119 windows retained.
  3/38301 restores all221 recorded decisions/881 raw records exactly.22 changed
  calls across14 cells, no return_centerline calls;144 return statuses per arm
  retain118 window-exit censors/6 unpassed/19 invalid/1 returned. The n1 paired
  return0.24->0.28s is descriptive only.
- [r2 result](../../experiments/koi-minimum-clearance-ab-r2-result.json) and
  [independent r2 audit](../../experiments/koi-minimum-clearance-ab-r2-audit.json):
  correcting full-command horizon to actual.08s increases intervention27 calls
  but loses3/38301 finish, adds obstacle3 hit, damage1.4->1.6/collision7->8 and loses
  three of119 baseline windows. Survivor-only path reduction is not an improvement
  verdict. [Single-cell diagnosis](../../experiments/koi-minimum-clearance-r2-lost-finish-diagnosis.json)
  verifies sole early override at active crossing, clean initial passage, later
  inherited flank switching and on-road obstacle stall; no premature-return claim.
- [First minimum-clearance A/B protocol](../../runs/koi-minimum-clearance-ab-20260930-v1/protocol.json):
  frozen crossing reconstruction versus isolated geometric steering candidate
  `ddadbc512...`,48 episodes on the same24 consumed TRAIN layouts. The
  [completed first result](../../experiments/koi-minimum-clearance-ab-v1-result.json)
  and [independent primary audit](../../experiments/koi-minimum-clearance-ab-v1-audit.json)
  preserve21 finishes, damage1.4/collision7, no new hit and119 paired windows,
  but change only5 steering decisions; cell-weighted path+0.038408% and kept-lap
  mean+8.571429ms fail improvement gates. Not adopted. Full footprint clearance, station±25 path windows,
  lateral/steering/return/lap and all finish/safety denominators are retained.
  Speed targets remain unchanged. Final pre-driving integration suite passed323
  tests+28 subtests; actual CPU21 package smoke recreated identical ZIP bytes.
- [Frozen first-30-slot diagnosis](../../experiments/koi-minimum-clearance-v1-prefix-diagnosis.json)
  finds geometrically supported proposals suppressed by a command guard assuming
  a full passage rather than the actual.08s action hold. Counterfactual short-hold
  asphalt support/physics is not proven by those traces. Separatev2 uses the
  actual hold guard while retaining full lane geometry and margins. Its
  [r2 protocol](../../runs/koi-minimum-clearance-ab-20260930-r2/protocol.json)
  binds ZIP64cecc0b... and331-test/28-subtest preflight; completed negative result
  is preserved above, followed by independently reviewed projection-footprintv3.
  First result, ZIP and source copies stay unchanged. No protected/official cells.
- [Independent consumed-cell provenance audit](../../experiments/koi-minimum-clearance-consumed-audit-v1.json):
  rehashed all48 prior episode/raw pairs and142 immutable/restored environment
  pins, confirmed the reconstructed10-member crossing baseline, and found no
  exact-cell allocation/protected conflict for the authorized24 TRAIN layouts.
  This is zero-reset preflight evidence only, not a driving result, freshness
  claim or clearance of candidate/measurement review findings. Original missing
  crossing ZIP/report and38300 reset-seam caveats remain. The
  [historical iteration](../architecture/koi-baseline-analysis-2026-09-30.md#15-minimum-clearance-가설과-adaptive-v1-종료)
  preserves the fixed crossing baseline and closed adaptive speed-target line.

## KOI Adaptive Avoidance (2026-09-30)

**Closed by user direction:** adaptive-v1 is a failed, nonadopted experiment.
No further speed-target tuning is queued. Crossing_projection remains the fixed
KOI improvement baseline; the separately authorized minimum-clearance path
hypothesis is not an adaptive-v1 continuation or promotion.

- [First unchanged A/B result](../../experiments/koi-adaptive-ab-v1-result.json),
  [frozen protocol](../../runs/koi-adaptive-ab-20260930-v1/protocol.json) and
  [48-episode report](../../runs/koi-adaptive-ab-20260930-v1/episode-report.json):
  crossing source reconstruction versus adaptive-v1 on24 paired consumed-TRAIN
  track/layout cells (eight geometry seeds). Both finish21/24, kept21/lost0/gained0,
  damage1.4 and seven collision decisions each. Mean mutually finished lap time
  is19.05429->19.07905s; 120 valid matched obstacle segments across21 cells average
  0.649009->0.649890s, with cell-weighted change+0.000839s (below20ms sampling
  resolution, not an improvement). Adaptive changes only7/5710 pedals decisions.
  One clean sustained physical min>44 passage is observed, but the predeclared
  two-geometry/valid-common-entry gate fails; reset-seam38300 cells are excluded
  from common-entry comparisons. **Not adopted.** Frozen models and all failure
  evidence remain unchanged; no fresh, held-out, blind, official or second run.
  [Existing KOI analysis and measurement plan](../architecture/koi-baseline-analysis-2026-09-30.md#13-사용자-지정-consumed-train-ab-검증)
  distinguish original missing ZIP restoration, detector timestamps, physical
  speed, complete-car passage and whole-episode safety.
  The [independent raw audit](../../experiments/koi-adaptive-ab-v1-audit.json)
  confirms all48 episode/raw hashes and142 environment Git-blob pins. Same138
  completed-passage mean speeds are42.94373->43.00255; the sole sustained actual
  >44 case is linked to adaptive but has identical passage timing and a60ms
  slower lap. Baseline already has114/138 completed-passage peaks above44.

## RLPD Coupled Recovery (2026-09-29)

- [Pixel-local gate protocol](../../experiments/rlpd-local-recovery-gate-v1.json):
  unchanged source actor when inactive, learned joint correction held12 decisions
  only after pixel-feature support triggers. Build calibration covered87/87
  prototypes and excluded16,023 reference triggers, not a full-episode harm
  guarantee. [First verified outcome](../../experiments/rlpd-local-recovery-gate-evaluation-v1-result.json)
  is original5/12 versus gated6/12, kept5/lost0/gained1, damage0.2->0.05 and
  progress0.6944->0.7566. Sparse curve-terminal count remains1/1; the
  [unchanged-model repeat](../../experiments/rlpd-local-recovery-gate-evaluation-repeat-v1-result.json)
  matches and the [trajectory audit](../../experiments/rlpd-local-recovery-gate-trajectory-review-v1-result.json)
  confirms24 correction decisions followed by383 source-only decisions on the
  gained road. Consumed TRAIN repeatability, not independent training/fresh evidence.
- [Frozen-encoder actor-only protocol](../../experiments/rlpd-recovery-actor-only-v1.json)
  and [verified evaluation](../../experiments/rlpd-recovery-actor-only-evaluation-v1-result.json):
  zero-reset offline 2,048-update fit preserves frozen components but finishes
  5/12 versus original5/12 with four original finishes lost and four gains. Global
  correction is not a retained improvement. Pixel-only local support gating is
  the next separately frozen hypothesis, not an outcome-selected road router.
- [Closed-loop validation and learning status](../../experiments/rlpd-recovery-validation-v1-result.json):
  42 complete branches; 12/25-decision feedback rescued 2/10 and 3/10 failure
  finishes but each harmed 2/4 finished controls. Strict paired-finish preparation
  yielded 339 unique failure-support rows on three geometries, plus qualified
  retained-finish rows. Both matched fine-tunes completed; the Q-only intervention
  was rejected after full evaluation.
- [Verified full evaluation](../../experiments/rlpd-recovery-evaluation-v1-result.json):
  36 uncensored consumed-TRAIN episodes, finishes V5 5/12, control 4/12, recovery
  1/12; all five original finishes lost. Prospective curve-terminal proxy counts
  are 1/0/1 respectively and do not explain every failure. Guided joint-mean
  recovery plus source-mean retention is the next isolated intervention.
- [Guided child protocol](../../experiments/rlpd-recovery-guided-learning-v1.json)
  and [actual-visit regression review](../../experiments/rlpd-recovery-regression-review-v1-result.json):
  joint mean guidance on qualified failure Oracle rows with original-mean prior/
  handoff retention; 8,192 decisions, 8,191 SAC updates plus separately reported
  8,191 extra actor updates. Learning completed, not matched-compute with v1.
- [Guided checkpoint](../../experiments/rlpd-recovery-guided-v1-result.json):
  exported hashes verified; official-coordinate joint MSE on 87 positive failure Oracle
  inputs is 0.0287 versus original 0.3760. Contemporary closed-loop outcome
  evaluation rejected it: [verified guided outcomes](../../experiments/rlpd-recovery-guided-evaluation-v1-result.json)
  show original5/12 versus guided2/12, four original finishes lost. Reduced sparse
  curve proxy and action error do not offset lost completion. Frozen-encoder
  actor-only correction with initial/successful-state protection is next.
- [Post-learning action/Q comparison](../../experiments/rlpd-recovery-action-comparison-v1-result.json)
  and [policy-handoff timeline](../../experiments/rlpd-recovery-handoff-v2-result.json):
  zero-reset diagnostics separate actual trajectory outcomes, action alignment,
  within-critic rankings and recurring warnings on successful controls.
- [Same-time multi-second audit](../../experiments/rlpd-coupled-recovery-r2-audit-result.json)
  and [source-bound continuation](../../experiments/rlpd-coupled-recovery-r2.json):
  preserve the interrupted v1 attempt, reuse 28 completed traces and finish 14
  missing branches without new cells or historical replay repair.
- [Coupled branch protocol](../../experiments/rlpd-coupled-recovery-v1.json):
  consumed G0 TRAIN failure precursors and finished-parent controls; full-vector
  12/25-decision recovery followed by real closed-loop actor continuation. The
  v1 execution is partial after an external shell timeout; no completed cohort
  or learner effect is established by this protocol.
- [Archived precursor review](../../experiments/rlpd-recovery-precursor-review-v1-result.json):
  zero resets, 24 existing actor-road traces; heading/lateral thresholds also
  appear on successes. Curve overspeed/opposition remain unassessed for these
  archives because the necessary telemetry is absent.
- [Foundation checks](../../experiments/rlpd-recovery-foundation-check-v1-result.json):
  unchanged RLPD regressions, isolated coupled collection/evaluation audits and
  V5 learning-checkpoint/export action parity. Synthetic engineering evidence,
  not a recovery or model performance result.
- [Iteration and data gate](../plans/rlpd-completion-first-research-2026-09-26.md):
  training requires paired full-finish rescue plus local qualification, not just
  a short Oracle imitation or five-second safety pass.

The JSON protocol/result artifacts under repository-root `experiments/` are the
source of truth. This index is intentionally selective: it promotes only evidence
that changes future research decisions. Timestamped `runs/` and `evaluations/`
artifacts remain the detailed execution receipts.

## DreamerV3 Final Closure (2026-09-27)

**CLOSED: no further DreamerV3 experiment, training, data collection, evaluation,
or hyperparameter search under the current research line.** The current design
and budget do not justify additional work: local implementation/data/contracts
have not resolved long-horizon prior dynamics, and further progress would require
design-level rework. This is not a theoretical impossibility judgment.

- Posterior reconstruction and short logged-action prediction can look reasonable,
  but multi-step free-prior rollout is unstable; the 32-decision error worsened
  beyond simple frame-repeat. See the [posterior/prior gap](../../experiments/dreamerv3-reused-posterior-gap-v1-result.json).
- Random, teacher, and multi-source replay plus prior-image/H8 auxiliaries did not
  show stable improvement across both learner seeds and both action-source strata.
  The strict H8 paired-improvement-and-repeat gate failed on consumed TRAIN data.
  See the [teacher prior-image score](../../experiments/dreamerv3-reused-train-prior-image-score-v1-result.json)
  and [multi-source H8 score](../../experiments/dreamerv3-reused-train-multisource-prior-score-v1-result.json).
- Action-input sensitivity was observed; the tested real transition trace found
  no target off-by-one; prior-image and H8 objectives were tried but missed their
  declared gates. None alone explains the remaining failure.
- Residual explanations remain hypotheses: narrow state/action coverage, possible
  redundancy or reconstruction shortcut from four stacked frames plus RSSM memory,
  offline-world-model-first training, and long-rollout dynamics drift.
- No Dreamer actor/candidate is designated. The latest H8 world-model checkpoints
  are diagnostic-only (`actor_trained=false`, `promotion_eligible=false`). The
  P1 freshness audit remains non-passing. No official submission, promotion, or
  protected evaluation cell is used. Stop local weight, horizon, and update-count
  searches.

The [archived recovery plan](../plans/archived/dreamerv3-recovery-strategy-2026-09-27.md)
and the separate
[`DEFERRED / LAST-RESORT ONLY revival plan`](../plans/dreamerv3-revival-plan.md)
are status/history references, not active work.

**DrQ-v2 research line CLOSED by user direction on 2026-09-29.** The
[closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line)
supersedes historical DrQ row statuses that describe running learners or
conditional follow-ups. Preserve their artifacts; no further DrQ research,
evaluation, packaging or official action is planned. The
[speed-only final result](../../experiments/drqv2-speed-reused-development-v1-result.json)
is negative and does not alter the original actor or local ZIP.

| Study | Question and conclusion | Primary evidence | Status |
|---|---|---|---|
| DrQ-v2 brake-only speed pilot | Immutable pad-4 seed1 actor versus itself with light-brake attenuation on 32 previously consumed development cells, twice each (128 verified episodes). Track 1 finished 5/16 control versus 3/16 treatment, losing three successes; track 2 was 1/8 versus 0/8, losing one; track 3 was 0/8 in both arms. Two mutually finished track-1 laps improved by 1.74 and 1.28 seconds; no speed comparison exists for tracks 2/3. | [`source-bound protocol`](../../experiments/drqv2-speed-reused-development-v1.json), [`candidate cell audit`](../../experiments/drqv2-speed-reused-development-v1-audit.json), [`frozen result`](../../experiments/drqv2-speed-reused-development-v1-result.json), [`primary run`](../../runs/20260929-drqv2-speed-reused-development-v1/result.json) | **REJECTED:** no-loss/per-track-speed gate failed. Mixed TRAIN-DIAGNOSTIC and residual-option development roads are heavily reused, not fresh evidence. No threshold retune, actor/package change, official score or promotion. |
| DrQ-v2 steering-logit L2 | Does `steering_logit_l2=0.001` improve repeated completion over a matched control? It scored 3 vs 6 and 0 vs 7 confirmation finishes for training seeds 0 and 1. | [`protocol`](../../experiments/drqv2-steering-logit-v1.json), [`result`](../../experiments/drqv2-steering-logit-v1-result.json), [`execution`](../../experiments/drqv2-l2-execution.json), [`diagnostics`](../../experiments/drqv2-steering-logit-v1-diagnostics.json) | Rejected. |
| DrQ-v2 padding | Does reducing random-shift padding from 4 to 1 improve generalization? Pad 1 had zero confirmation finishes versus 4 and 7 for controls. | [`protocol`](../../experiments/drqv2-augmentation-pad-v1.json), [`result`](../../experiments/drqv2-augmentation-pad-v1-result.json), [`execution`](../../experiments/drqv2-pad-execution.json), [`diagnostics`](../../experiments/drqv2-augmentation-pad-v1-diagnostics.json) | Rejected; final authorized narrow DrQ follow-up. |
| DrQ-v2 teacher replay | Does a fixed 48:16 online/teacher batch improve matched pad-4 fine-tuning? The r3 implementation reached both 16,384-decision teacher caps, but source learner 1 finished on only 3 distinct geometries versus the preregistered 4. The A3 gate stopped the pair before training. | [`r3 protocol`](../../experiments/drqv2-teacher-replay-v1-r3.json), [`A3 result`](../../experiments/drqv2-teacher-replay-v1-r3-result.json), [`A3 stop receipt`](../../runs/20260924-drqv2-teacher-replay-v1-r3/a3-coverage-stop.json), [`teacher seed 0 receipt`](../../runs/20260924-drqv2-teacher-replay-v1-r3/teacher-data/learner-0/collection-result.json), [`teacher seed 1 receipt`](../../runs/20260924-drqv2-teacher-replay-v1-r3/teacher-data/learner-1/collection-result.json) | Inconclusive; no learner trained or evaluation partition opened. Do not reuse either dataset for fresh tuning. |
| DrQ-v2 training-only road-shape augmentation | Do independently generated seeded roads near consumed early/late failure-feature neighborhoods expose useful variation without leaking validation or blind cells? Six measured families yielded 120 static/smoke-valid TRAIN roads plus 16 disjoint TRAIN-DIAGNOSTIC roads, selected before any frozen actor rollout. On the same 136 roads the two source actors finished 37 and 28 single attempts; 10 roads were completed by both, 74 are provisional boundary, 50 difficult with progress, and 2 unresolved, with no selected malformed roads. | [`consumed failure analysis`](../../experiments/drqv2-geometry-augmentation-v1-analysis.json), [`frozen catalog protocol`](../../experiments/drqv2-geometry-augmentation-v1.json), [`catalog result`](../../experiments/drqv2-geometry-augmentation-v1-catalog-result.json), [`CPU diagnostic protocol`](../../experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json), [`272-cell summary`](../../experiments/drqv2-geometry-augmentation-v1-training-summary.json), [`per-road final set`](../../experiments/drqv2-geometry-augmentation-v1-final-set.json), [`report`](drqv2-geometry-augmentation-v1.md) | Catalog and actor diagnostics completed without learner training at that stage. Subsequent r6 learner runs on these TRAIN roads are indexed in the next row; neither stage opened held-out or blind cells. First-bend and finish-line mechanisms remain hypotheses. |
| DrQ-v2 geometry-mix fine-tuning | Does uniform, failure-weighted, or easy-retention sampling across the same 120 TRAIN roads change matched pad-4 fine-tuning on six source-seed/mixture runs? All six online-only r6 runs completed 32,768 additional decisions and 22,768 updates. | [`r6 protocol`](../../experiments/drqv2-geometry-mix-v1-r6.json), [`learner-0 uniform`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform/result.json), [`learner-0 failure-weighted`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-failure_weighted/result.json), [`learner-0 easy-retention`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-easy_retention/result.json), [`learner-1 uniform`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-uniform/result.json), [`learner-1 failure-weighted`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-failure_weighted/result.json), [`learner-1 easy-retention`](../../runs/20260925-drqv2-geometry-mix-v1-r6/learner-1-easy_retention/result.json), [`TRAIN-DIAGNOSTIC manifest`](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json), [`family summary`](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/summary.json), [`active plan and checkpoint/replay hashes`](../plans/active/drqv2-geometry-mix-plan.md), [`r4 partial-run abort`](../../runs/20260925-drqv2-geometry-mix-v1-r4/learner-0-uniform/precheckpoint-abort.json), [`r5 partial-run abort`](../../runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/precheckpoint-abort.json), [`TRAIN catalog`](../../runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json) | The canonical repeat-0 TRAIN-DIAGNOSTIC outcomes on the same 16 development roads were uniform 1/32, failure-weighted 4/32, and easy-retention 3/32 finishes across two learner seeds; unchanged source actors finished 11/32. This descriptive 256-episode diagnostic is unranked and not fresh generalization evidence; no screen, confirmation, blind, or official evaluation was opened and no model was selected or promoted. r4/r5 partial TRAIN exposure is preserved and not resumed. |
| DrQ-v2 geometry-mix r6 regression diagnosis | Why did source actors at 11/32 finishes drop to uniform 1/32, failure-weighted 4/32, easy-retention 3/32? Paired source-success losses were 11, 10, 10, compared with 1, 3, 2 gained; 5,998 exact source-action replay decisions yielded 666 sampled source-state instances (664 unique observation hashes) for actor/twin-Q/encoder checks. | [`A-K evidence report`](drqv2-geometry-mix-r6-regression.md), [`paired roads and trace hashes`](../../runs/20260925-drqv2-geometry-mix-v1-r6/paired-regression-v1.json), [`replay distribution`](../../runs/20260925-drqv2-geometry-mix-v1-r6/replay-distribution-v1.json), [`same-state Q/encoder checks`](../../runs/20260925-drqv2-geometry-mix-v1-r6/source-state-replay-v2/result.json), [`noncausal failure markers`](../../runs/20260925-drqv2-geometry-mix-v1-r6/regression-attribution-v1.json) | Early actor/encoder drift is observed; minority Q1 reversals and imperfect source-state coverage are possible contributors, not proven single causes. All evidence reuses development roads; no training, held-out/confirmation/blind, promotion or official score. |
| DrQ-v2 r7 source-behavior retention | Does enforced 32:32 original-source/online replay, with or without one prespecified source-action MSE, preserve old DrQ finishes while adding new geometry successes? All twelve r7 TRAIN-only arms completed 32,768 decisions and 22,768 updates; all 384 reused TRAIN-DIAGNOSTIC episodes passed repeat parity. In uniform/failure-weighted/easy-retention order, r7a kept 1/3/1 of eleven original source-success actor/road cells and gained 2/7/4; r7b kept 4/1/5 and gained 5/7/5. Best retained only five of eleven, below the predeclared nine-plus-two retention signal. | [`frozen protocol`](../../experiments/drqv2-retention-r7.json), [`frozen result`](../../experiments/drqv2-retention-r7-result.json), [`A-M report`](drqv2-retention-r7.md), [`offline hybrid`](../../runs/20260926-drqv2-retention-r7/hybrid/first20-v2/result.json), [`TRAIN-only gradient weight`](../../runs/20260926-drqv2-retention-r7/gradient-scale-v1.json), [`all-arm sample audit`](../../runs/20260926-drqv2-retention-r7/pre-evaluation-trace-audit-v1.json), [`384-episode manifest`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/manifest.json), [`paired cells`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/paired.json) | No established source-success retention contract. All six r7 variants gained some previously source-failed cells but lost six to ten original successful cells. Original source replay was collected by an evolving actor; 32:32 also halves online sample exposure versus r6. This is reused-TRAIN development evidence only, not a model promotion, held-out/confirmation/blind result, or official score. |
| DrQ-v2 final-source replay retention, migration stop/reconstruction r1 rev2 | Did r7's evolving-source replay underrepresent later source-success states, and can a final frozen source actor's TRAIN-only replay restore retention at the same 32:32, lambda=0.5 budget? The original five learners completed; seed1/easy-retention stopped after step 16,384 with ledger through step 21,037. Exact resume is impossible. Revision 2 preserves a first seed0/uniform launcher failure before environment creation/reset (zero exposure); the retried seed0/uniform arm completed 32,768 decisions/22,768 updates and 67 reused TRAIN cell episodes, with checkpoint/actor/sample hashes independently verified. | [`revision-2 protocol`](../../experiments/drqv2-final-source-replay-reconstruction-r1.json), [`final exposure receipt`](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/preflight-exposure-audit.json), [`setup-failure receipt`](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/preflight-failure-20260928T104204Z-seed0-uniform.json), [`arm0 result`](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/learner-0-uniform-final_source/result.json), [`report`](drqv2-final-source-replay-v1.md), [`active plan`](../plans/active/drqv2-geometry-mix-plan.md), [`exposure auditor`](../../scripts/audit_drq_final_source_reconstruction.py), [`sample adapter`](../../scripts/audit_drq_final_source_reconstruction_samples.py), [`diagnostic adapter`](../../scripts/diagnose_drq_final_source_reconstruction.py), [`parent protocol`](../../experiments/drqv2-final-source-replay-v1.json) | **SERIAL TRAINING IN PROGRESS; 1/6 arms complete, seed0/failure-weighted next.** All 120 parent-catalog TRAIN seeds are reused; none are fresh. Complete all six arms before the independent sample audit, then 192-episode TRAIN-DIAGNOSTIC; >=9/11 retained AND >=2/21 gained remains UNEVALUATED. No confirmation/blind/official action or promotion. |
| Pixel RLPD off-policy pilot v2 | Does a frozen DrQ prior dataset, mixed 32:32 with online replay, improve a matched ten-Q pixel SAC at 16,384 student decisions? Teacher coverage passed (8,192 decisions; 3 distinct-geometry finishes); all four student runs completed. The eight-actor CPU21 screen completed 192 episodes but found only one canonical finish, for RLPD seed 1 at 8,192 steps. RLPD seed 0 had none, so the frozen two-seed gate failed. | [`protocol`](../../experiments/pixel-rlpd-offpolicy-pilot-v2.json), [`result`](../../experiments/pixel-rlpd-offpolicy-pilot-v2-result.json), [`teacher data`](../../runs/20260924-pixel-rlpd-offpolicy-pilot-v2/prior-data/manifest.json), [`screen`](../../evaluations/20260924T184536689584Z_pixel-rlpd-offpolicy-pilot-v2-screen/report.md) | Stop/hold; no candidate promotion, confirmation/blind/full-stage execution, or official score claim. |
| Pixel RLPD long-horizon follow-up v1 | Separate fresh 16,384-decision teacher dataset (16,305 stored transitions; 6 distinct-geometry finishes) plus matched 131,072-decision students. The expanded data/horizon jointly change two factors, so this tests a fuller recipe, not a causal mechanism. Screen: RLPD/SAC finishes 7/24 vs 3/24 at seed 10 and 12/24 vs 3/24 at seed 11. Confirmation: RLPD/SAC 3/32 vs 2/32 and 12/32 vs 7/32. One screen-preselected RLPD seed-11, 131,072-step actor passed internal blind at 9/24 canonical finishes (mean progress 0.679). All screen/confirmation/blind CPU-reload, determinism, and operational checks passed. | [`protocol`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json), [`result`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json), [`run result`](../../runs/20260924-pixel-rlpd-long-horizon-followup-v1/result.json), [`screen`](../../evaluations/20260925T002348999661Z_pixel-rlpd-long-horizon-followup-v1-screen/report.md), [`confirmation recovery`](../../runs/20260924-pixel-rlpd-long-horizon-followup-v1/confirmation-recovery-execution.json), [`blind`](../../evaluations/20260925T030439692341Z_pixel-rlpd-long-horizon-followup-v1-blind/report.md) | Internal candidate only; v3 partitions consumed and sealed. No official score claim, model confirmation, or submission. Later entropy-target studies are listed separately below; their retired allocations stay excluded. |
| Pixel RLPD entropy-target ablation v1 | A fresh-data one-factor comparison of the pinned `-1.5` and `+1.5` targets was protocol-frozen but its helper rejected descriptive teacher-budget metadata before the collector ran. | [`protocol`](../../experiments/pixel-rlpd-entropy-target-ablation-v1.json), [`preflight`](../../experiments/pixel-rlpd-entropy-target-ablation-v1-preflight.json); the source snapshot is retained locally, not versioned | Zero interactions/cells. Allocation retired; not reusable. |
| Pixel RLPD entropy-target ablation v2 | The second file had a V2 filename/run template but a V1 internal name, so protocol-name validation stopped before collection. | [`protocol`](../../experiments/pixel-rlpd-entropy-target-ablation-v2.json), [`preflight`](../../experiments/pixel-rlpd-entropy-target-ablation-v2-preflight.json); the source snapshot is retained locally, not versioned | Zero interactions/cells. Allocation retired; not reusable. V3 likewise stopped before interaction; V4 and V5 outcomes are listed below. |
| Pixel RLPD entropy-target ablation v3 | The protocol requested learner seeds 30/31 while the helper/hashed reader still required 20/21; post-freeze source bytes also differed. | [`protocol`](../../experiments/pixel-rlpd-entropy-target-ablation-v3.json), [`preflight`](../../experiments/pixel-rlpd-entropy-target-ablation-v3-preflight.json) | Zero interactions/cells. Allocation retired; not reusable. |
| Pixel RLPD entropy-target ablation v4 | New one-factor RLPD target comparison (`-1.5` author target vs `+1.5` alternative) with the same prior data, online mixing, architecture, optimizer, and 131,072-decision horizon per arm/seed. Teacher/prior data and four fresh student runs completed; evaluation source recheck then found `generalization-policy.md` changed after freeze. | [`frozen protocol`](../../experiments/pixel-rlpd-entropy-target-ablation-v4.json), [`pre-screen result`](../../experiments/pixel-rlpd-entropy-target-ablation-v4-result.json), [`run`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v4/execution.json) | No screen/confirmation/blind cell ran. V4 training/data experience is consumed and cannot be relabeled fresh; the entire V4 evaluation allocation is retired. V5 subsequently used separate fresh data and cells (next row). |
| Pixel RLPD entropy-target ablation v5 | One-factor comparison of the author `-1.5` target and `+1.5` alternative, using the same fresh 16,384-decision prior dataset, RLPD architecture, 32:32 mix, and matched 131,072-decision initializations. Screen selected actor finishes were author/+1.5 8/2 at seed 50 and 7/7 at seed 51; 29/192 screen canonical episodes finished. Strict confirmations were author 17/32 and 7/32 versus +1.5 6/32 and 6/32; all four actors were eligible, deterministic, CPU-reload identical, and operationally clean. Author-target strictly dominated both matched seeds. Its selected seed-50 actor passed the one internal blind with 7/24 finishes (mean progress 0.657). | [`protocol`](../../experiments/pixel-rlpd-entropy-target-ablation-v5.json), [`result`](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json), [`execution`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v5/execution.json), [`attempt-4 recovery`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v5/entropy-recovery-attempt4.json), [`screen`](../../evaluations/20260925T162014319049Z_pixel-rlpd-entropy-target-ablation-v5-screen/report.md), [`screen selection`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v5/screen-selection-projections.json), [`confirmation`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v5/confirmation-rlpd-author-target-seed50.json), [`blind`](../../runs/20260925-pixel-rlpd-entropy-target-ablation-v5/blind-finalist.json) | All V5 cells are consumed and sealed. This is a two-seed internal CarRacing comparison, not an official score, model confirmation, or submission. |
| Pixel RLPD completion-first G0 | Two frozen local RLPD actors were diagnosed on the same 12 TRAIN-only geometries/one obstacle variant. Each finished 3/12, retired `off_track` 8/12, and crashed 1/12; 10,049 decisions under a 48,000-decision cap and zero learner updates. First observed events also occur on successful controls, so a causal bottleneck or intervention is not yet selected. The narrow r5 receipt erratum does not establish original-ledger byte attestation. | [`seed audit`](../../experiments/rlpd-g0-completion-v1-seed-audit.json), [`frozen protocol`](../../experiments/rlpd-g0-completion-v1.json), [`result`](../../experiments/rlpd-g0-completion-v1-result.json), [`manifest`](../../runs/20260926-rlpd-g0-completion-v1/manifest.json) | Complete observational TRAIN diagnostic, not a matched training treatment, held-out evaluation, ranking or official result. All 12 geometry IDs consumed; do not relabel them fresh. |
| Pixel RLPD same-cell new-host repeat | A separate source-pinned, uncapped (no four-core-hour CPU budget) V5 seed-50 actor drove the previously consumed G0 track-1/4272000001 TRAIN road once. On the same hashed road the old episode ended `off_track` in 456 decisions; the new one finished in 531, with equal initial pixels but first differing next frame at zero-based decision 33. The first native action differed by only 1.19e-7; the cause of the later fork is unestablished. | [`new protocol`](../../experiments/rlpd-reused-train-immediate-v1.json), [`result and old/new trace SHA`](../../experiments/rlpd-reused-train-immediate-v1-result.json), [`attempt`](../../runs/20260929-rlpd-reused-train-immediate-v1/attempt.json), [`new cell`](../../runs/20260929-rlpd-reused-train-immediate-v1/result.json) | Reused TRAIN engineering repeat, not independent-road coverage, an updated learner, a matched intervention, G1, or performance improvement. |
| Pixel RLPD new-host G1 source-actor collection | A source-bound uncapped protocol collected 48/48 one-shot episodes on 24 claimed TRAIN roads: source V5 seed-50 finished 12/24, seed-11 11/24; paired outcomes were 7 both, 5 primary only, 4 comparator only, 8 neither. All trace/cell SHAs verified. At least five short episodes cannot supply the frozen ordinary image window; no image labels were sealed and the zero-unknown coverage pass is unreachable. | [`protocol`](../../experiments/rlpd-g1-newhost-20260929-v1.json), [`pre-reset audit`](../../experiments/rlpd-g1-newhost-20260929-v1-preflight.json), [`result`](../../experiments/rlpd-g1-newhost-20260929-v1-result.json), [`manifest`](../../runs/20260929-rlpd-g1-newhost-v1/manifest.json) | Collected and internally audited, but postrun coverage INCONCLUSIVE under frozen rubric; NOT a learner treatment or official score. Do not top up or alter labels. Rejected zero-reset preflight bytes remain under `.invalid-frozen` names. |
| Pixel RLPD new-host actor-critic learner | Authentic V5 prior transitions were mixed 32:32 with online replay from 12 previously consumed G0 TRAIN roads. Completed 131,072 online decisions and 130,072 actor/critic updates with two SHA-checked exported actors/checkpoints on RTX 4060 Ti; final actor SHA `28d82208...` strict-loads on CPU. Evolving training trajectories are NOT saved actor performance. | [`protocol`](../../experiments/rlpd-newhost-reused-train-20260929-v1.json), [`result`](../../experiments/rlpd-newhost-reused-train-20260929-v1-result.json), [`primary run`](../../runs/20260929-rlpd-newhost-reused-train-v1/result.json), [`final candidate`](../../runs/20260929-rlpd-newhost-reused-train-v1/checkpoints/step-000131072/candidate.json) | Complete TRAIN learning, but post-training in-sample paired screen below is negative. No independent screen, confirmation/blind/official score or model promotion. |
| Pixel RLPD new-host final-actor paired TRAIN screen | Both exported actors passed zero-reset CPU double-reload/adapter parity, then source-pinned final seed-52 and V5 seed-50 drove identical hashed roads on 12 ALREADY CONSUMED G0 TRAIN geometries, one episode per actor/road (24/24 complete, all trace/cell SHAs checked). Final seed-52 finished 1/12, V5 5/12; 1 both, 0 new-only, 4 V5-only, 7 neither. Only seed-52 trained online on these exact roads; V5 trained on a different pool. Historical G0 V5 3/12 had four outcome flips versus current V5 5/12. | [`frozen protocol`](../../experiments/rlpd-newhost-train-screen-20260929-v1.json), [`result with exposure erratum`](../../experiments/rlpd-newhost-train-screen-20260929-v1-result.json), [`primary manifest`](../../runs/20260929-rlpd-newhost-train-screen-v1/manifest.json) | Complete NEGATIVE IN-SAMPLE TRAIN sanity; do not promote new actor. Not a matched training intervention, fresh geometry test, generalization, confirmation/blind/official result or causal diagnosis. |
| Pixel RLPD G0 fixed-window review | Offline reanalysis of those same 24 consumed TRAIN traces, using six predeclared +/-20-decision anchors for every episode and successful-parent control. The hashed extractor retained 144 slots, including 101 pixel/telemetry windows and 43 explicit missing anchors; 24 source trace hashes and 101 derived tile hashes were verified. No causal or manual labels assigned. | [`freeze`](../../experiments/rlpd-g0-fixed-windows-v1.json), [`result`](../../experiments/rlpd-g0-fixed-windows-v1-result.json), [`manifest`](../../runs/20260926-rlpd-g0-fixed-windows-v1/manifest.json) | Exploratory reused-TRAIN diagnostic only; not new data, a matched intervention, or evidence to promote a learner. |
| Pixel RLPD G0 pixel-motion feasibility | A frozen threshold-free four-frame grayscale change score was computed for every decision of the same 24 consumed TRAIN episodes. A retrospective contact-stall label marks 74/10,049 correlated decisions in only four failed episodes/three geometries; all six finishes (2,828 decisions) remain in the control distribution. Scores are lower on many labelled decisions but overlap finished controls, so no global trigger or policy was selected. Speed/contact/tile information was diagnostic-only. | [`rule`](../../experiments/rlpd-g0-pixel-motion-v1.json), [`summary`](../../experiments/rlpd-g0-pixel-motion-v1-result.json), [`gate decision`](../../experiments/rlpd-g0-pixel-motion-v1-decision.json), [`per-decision result`](../../runs/20260926-rlpd-g0-pixel-motion-v1.json) | Exploratory reused-TRAIN representation proxy. Next gate is a separately frozen geometry-grouped visual representation probe, not a learner treatment, fresh validation or official result. |
| Pixel RLPD frozen-encoder speed probe | A CPU-only linear diagnostic head was fitted for 256 updates on an unchanged seed-11 actor encoder, after excluding each episode's first 20 decisions. V1 blocked preflight on concurrent Git revision drift before data scan; source-byte-pinned V2 used 8 fit and 4 retrospective diagnostic geometries of the 12 already consumed TRAIN roads. The diagnostic cutoff recorded TP/FN/FP/TN 188/2/9/2,760, but every TP came from the two correlated episodes of one road and no successful parent was in the diagnostic split. The actor parameter digest was identical before/after; six fitting-parent finishes had one negative-row false positive. | [`v1 blocked protocol`](../../experiments/rlpd-visual-representation-probe-v1.json), [`v2 protocol`](../../experiments/rlpd-visual-representation-probe-v2.json), [`result`](../../experiments/rlpd-visual-representation-probe-v2-result.json), [`attempt`](../../runs/20260926-rlpd-visual-representation-probe-v2/attempt.json), [`receipt`](../../runs/20260926-rlpd-visual-representation-probe-v2/receipt.json) | HUD-inclusive speed decoding proxy, not optical-flow proof, policy update, G1 rescue, fresh generalization or official score. Independent positive and successful-parent diagnostic coverage are still required. |
| Pixel RLPD visual-head feasibility | A frozen RLPD pixel encoder was probed with 256 CPU updates to a separate linear head, using the already inspected G0 TRAIN roads. On eight fitting geometries 275/6,610 eligible decisions had the retrospective low-speed target; the four previously viewed diagnostic geometries had 190/2,959 positives. Fixed-cutoff TP/FN/FP/TN were 188/2/9/2,760, but all 188 true positives occurred in two correlated episodes on one geometry; no finished-parent episode was in the diagnostic group. The policy pixels contain a speed HUD, so this cannot establish road-motion perception. | [`v1 failed preflight`](../../experiments/rlpd-visual-representation-probe-v1.json), [`v2 protocol`](../../experiments/rlpd-visual-representation-probe-v2.json), [`result`](../../experiments/rlpd-visual-representation-probe-v2-result.json), [`attempt`](../../runs/20260926-rlpd-visual-representation-probe-v2/attempt.json), [`receipt`](../../runs/20260926-rlpd-visual-representation-probe-v2/receipt.json) | Completed head-only, reused-TRAIN internal proxy. No environment reset, actor update, G1 trigger, promotion or official claim; geometry-level positive and finished-parent controls are still required. |
| DrQ-v2 residual options pilot | Can a short-horizon Q controller over a frozen DrQ driver preserve or improve finishes? One development iteration tied at 5/32 with 4 paired wins/4 losses; a training-replay-calibrated KEEP margin reduced interventions but scored 3/32. Offline replay audit found v2's applied margin differs from its own-head training-replay q75 and substantially changes the next-action selector. Training-only paired prefix branches passed exact replay parity; v1 COAST both rescued and derailed baseline trajectories, while the two v2 BRAKE branches reduced progress. | [`v1 protocol`](../../experiments/drqv2-residual-options-pilot-v1.json), [`v2 protocol`](../../experiments/drqv2-residual-options-pilot-v2.json), [`result`](../../experiments/drqv2-residual-options-pilot-result.json), [`offline training audit`](../../evaluations/drqv2-residual-options-training-offline-audit.json), [`prefix branch protocol`](../../experiments/drqv2-residual-options-prefix-branch-v1.json), [`prefix branch result`](../../experiments/drqv2-residual-options-prefix-branch-v1-result.json) | Prefix parity and local branch diagnostic complete; this does not promote a policy or authorize reuse as confirmation evidence. |
| TD-MPC2 5M-class pixel online baseline, completed v2 pilot | Is the original stochastic-prior/latent-dynamics/MPPI baseline operational and helpful in HAIC? V1 stopped at 10,061 decisions/10,060 updates after a planner-reset inference-tensor error; its 29 episodes had no finishes and its checkpoint is provenance-only. User-directed v2 from scratch ended cleanly at a whole-episode boundary: 12,058 decisions, 37 episodes, 12,057 updates (10,000 pretraining), and 1,629.3 seconds; all 37 TRAIN episodes had no finish. The new RTX 4060 Ti/Torch 2.1.0+cu121 runtime is not an exact reproduction of the prior stack. A source-bound Torch 2.1.0+cpu export passed with zero resets. The paired 16-episode, two-repeat diagnostic on the same four consumed TRAIN cells observed prior 0/8 finishes, 0 censored, mean progress 0.05252/reward -46.904; MPPI 0/8 finishes, 1/8 censored at 500 steps, mean progress 0.03487/reward -50.155. The evaluator marks `capped_finish_comparison_valid: false`, so these small reused-TRAIN metrics establish no MPPI advantage/failure or generalization. | [`active plan`](../plans/active/tdmpc2-pixel-online-baseline.md), [`v1 failure`](../../experiments/tdmpc2-reused-train-pilot-v1-failure.json), [`v2 protocol`](../../experiments/tdmpc2-reused-train-pilot-v2.json), [`v2 exposure`](../../experiments/tdmpc2-reused-train-pilot-v2-exposure.json), [`v2 training result`](../../runs/tdmpc2-reused-train-20260927-v2/result.json), [`CPU export receipt`](../../runs/tdmpc2-reused-train-20260927-v2/cpu-model-export.json), [`paired evaluation`](../../runs/tdmpc2-reused-train-20260927-v2/evaluation-result.json), [`evaluation ledger`](../../runs/tdmpc2-reused-train-20260927-v2/train-evaluation.jsonl) | Internal, consumed-TRAIN engineering result only; all four roads had prior exposure. The successful reset run resolves the observed integration defect in this run but does not prove broader correctness or driving benefit. No new generalization, matched r6/r7 comparison, candidate promotion, protected-cell or official action. |
| DreamerV3 feasibility | Is the current native DreamerV3 implementation ready for 131k matched training? Interface/export and short GPU smoke passed, but the pilot policy collapsed into steering saturation. | [`feasibility record`](../../experiments/dreamerv3-feasibility-gate.json) | Historical feasibility result only. The fidelity-recovery line is CLOSED; no scale-up or follow-up study. |
| DreamerV3 B1 failure diagnosis | All nine static-random-data B1 iterations failed for two seeds; v8's small image improvement did not carry task-relevant termination/reward signals. The v9 BCE `pos_weight=56` multiplied majority `continue=1`, not terminal errors; the sampled-window BCE baseline and episode-average reward rank also need reinterpretation. Four-step real environment trace found no target offset in the tested path. | [`v1-v9 summary`](../../experiments/dreamerv3-b1-iteration-summary-v1-v9.json), [`diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json), [`same-anchor seed 0`](../../experiments/dreamerv3-b1-context-audit-v9-seed0.json), [`seed 1`](../../experiments/dreamerv3-b1-context-audit-v9-seed1.json) | Both seeds failed all nine B1 studies. Historical diagnostic only; no actor/candidate or official claim. Final research-line status is CLOSED, not pending another decision. |
| DreamerV3 reused-TRAIN engineering loop | Does a bounded archive-to-replay-to-model-only pipeline operate on previously allocated TRAIN roads? On four fixed roads per arm, random collected 1,306 decisions with 0/4 finishes and unchanged DrQ-source teacher collected 2,548 decisions with 1/4 finishes. Two offline learner seeds per arm completed 64 model-only updates each without new environment interactions or actor/critic training. Different data sizes and training distributions preclude a matched policy/prediction benefit claim. | [`collection protocol`](../../experiments/dreamerv3-reused-train-diagnostic-v1.json), [`random collection`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/random/collection-result.json), [`teacher collection`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/teacher/collection-result.json), [`offline protocol`](../../experiments/dreamerv3-reused-train-offline-v1.json), [`frozen result`](../../experiments/dreamerv3-reused-train-diagnostic-v1-result.json), [`random seed 0`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline/random-seed-0/training-result.json), [`random seed 1`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline/random-seed-1/training-result.json), [`teacher seed 0`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline/teacher-seed-0/training-result.json), [`teacher seed 1`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline/teacher-seed-1/training-result.json) | Non-promoting reused-TRAIN engineering diagnostic only. No fresh P1/P1b, training-excluded development score, held-out/blind, or official result. B1 remains no-go; a separately frozen scorer and excluded development data are prerequisites for prediction claims. |
| DreamerV3 reused-TRAIN development and update-budget probe | On four different, training-excluded but already allocated r6 TRAIN roads per action-source stratum, both collectors completed four episodes and neither finished. All four 64-update models failed the shifted-repeat image baseline on teacher-action development; updating the same four initialized models for 256 steps instead of 64 worsened both teacher-trained seeds' image error on both development strata. Each stratum has only four independent terminal-positive episodes, no finish positives, and 256 scored labels. | [`development collection`](../../experiments/dreamerv3-reused-train-development-v1.json), [`64-update score`](../../experiments/dreamerv3-reused-train-score-v1.json), [`64-update receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/scoring-v1/score-result.json), [`256-update offline`](../../experiments/dreamerv3-reused-train-offline-v2.json), [`paired probe`](../../experiments/dreamerv3-reused-train-budget-probe-v1.json), [`paired receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/scoring-v2/score-result.json), [`frozen result`](../../experiments/dreamerv3-reused-train-development-v1-result.json) | Reused TRAIN and consumed tuning data, not fresh validation or a matched P1b claim. Loss reduction did not establish forward prediction, and longer training did not rescue this local image baseline. No Dreamer actor update, held-out/blind, promotion, or official result. |

| DreamerV3 reused-TRAIN action, sampling and prior-gap diagnostics | On the same four already-inspected training-excluded but r6-allocated TRAIN roads per source stratum, the teacher-data world models respond to future action changes without consistent logged-action superiority. Eight-sample predictive-mean images retain a late 9-32-decision error above frozen-anchor repeat. Target-conditioned posterior reconstruction and one-step logged-action prior are near the true-previous-frame repeat, while the 32-decision free-running prior worsens with longer training despite lower same-transition KL. | [`action-input result`](../../experiments/dreamerv3-reused-train-action-input-v1-result.json), [`action receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/action-sensitivity-v1/action-result.json), [`K=8 sampling result`](../../experiments/dreamerv3-reused-train-sampling-v1-result.json), [`sampling receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/sampling-v1/sampling-result.json), [`prior-gap protocol`](../../experiments/dreamerv3-reused-posterior-gap-v1.json), [`prior-gap result`](../../experiments/dreamerv3-reused-posterior-gap-v1-result.json), [`gap receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/posterior-gap-v1/posterior-gap-result.json) | Read-only, consumed iterative tuning. Posterior reconstruction sees its target, shuffled actions have no counterfactual outcomes, and eight latent paths are not eight roads. Four road episodes per stratum and no development finishes cannot authorize actor training, fresh P1b, model promotion or official claims. |

| DreamerV3 reused-TRAIN prior-image auxiliary | A separate world-model objective trained from the same small sealed random/teacher archives with 256 ordinary plus 64 additional eight-decision free-prior image steps. Both teacher-data seeds improved image MSE versus their paired pure-256 models on both previously scored source strata, but at 0.015952/0.016135 failed the predeclared teacher-action frozen-anchor repeat 0.014884 gate. No third-group roads were opened. | [`training protocol`](../../experiments/dreamerv3-reused-train-prior-image-v1.json), [`training result`](../../experiments/dreamerv3-reused-train-prior-image-v1-result.json), [`score protocol`](../../experiments/dreamerv3-reused-train-prior-image-score-v1.json), [`score receipt`](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/scoring-prior-v1/score-result.json), [`frozen score result`](../../experiments/dreamerv3-reused-train-prior-image-score-v1-result.json) | Failed local teacher-data gate on four consumed reused-TRAIN development roads, no model selection or fresh P1b. The extra 64 optimizer steps confound loss-shape attribution. Actor/critic/env unchanged; no official result. |
| DreamerV3 source0 reused-TRAIN diversity feasibility | On a separately frozen six-family schedule of twelve previously r6-allocated TRAIN roads, both arms attempted every road once. Random recorded 4,137 decisions and 0/12 finishes; DrQ source0 recorded 5,420 decisions and two finishes across opening-delayed and easy-curvature families. The predeclared >=3 distinct-finish teacher-support gate failed. | [`collection protocol`](../../experiments/dreamerv3-reused-train-diversity-v1.json), [`random receipt`](../../runs/20260926-dreamerv3-reused-train-diversity-v1/collection/random/collection-result.json), [`teacher receipt`](../../runs/20260926-dreamerv3-reused-train-diversity-v1/collection/teacher/collection-result.json), [`root receipt`](../../runs/20260926-dreamerv3-reused-train-diversity-v1/collection/collection-result.json), [`frozen result`](../../experiments/dreamerv3-reused-train-diversity-v1-result.json) | Both sealed archives are retained, but this study authorizes **zero** student-model updates. The catalog digest in its result is a post-collection read-only snapshot, not collector-enforced pinning. No fresh P1/P1b, Dreamer actor, protected or official score. |

| DreamerV3 source1 same-road TRAIN replication | A new, independently pinned DrQ source1 actor repeated exactly the same twelve r6-allocated TRAIN roads as the failed source0 diversity collection. Source1 recorded 5,192 decisions, 12 complete episodes and finishes on two roads, one not finished by the current source0 receipt. The source-union is three distinct roads across three shape families, meeting a preregistered **design-only** complement gate. A previous DrQ summary had already observed source1 finish four of these roads once, so this is variation/replication rather than new geometry generalization. | [`source1 protocol`](../../experiments/dreamerv3-reused-train-source1-v1.json), [`source1 receipt`](../../runs/20260926-dreamerv3-reused-train-source1-v1/collection/collection-result.json), [`sealed teacher archive`](../../runs/20260926-dreamerv3-reused-train-source1-v1/collection/teacher/support-dataset.npz), [`frozen result`](../../experiments/dreamerv3-reused-train-source1-v1-result.json), [`previous paired actor summary`](../../experiments/drqv2-geometry-augmentation-v1-training-summary.json) | The union passed design only, not the source0 learner gate or fresh-road support. The resulting multi-source diagnostic studies are complete and failed their prediction gates; no follow-up study is active. |

| DreamerV3 lineage-safe mixed-source model-only learner | After source1's separate descriptive complement gate, a new SHA-pinned TRAIN-reuse study independently validated source0 and source1 whole-episode archives and interleaved 24 actor-tagged episodes over only twelve distinct r6-allocated TRAIN roads, with 10,612 combined decisions in a 16,384-capacity replay. Both learner seeds completed 256 model-only steps, environment_steps=0, unchanged actor/critic, and identical hashed lineage sidecars covering sequence IDs 0..10611. | [`training protocol`](../../experiments/dreamerv3-reused-train-multisource-v1.json), [`seed 0 result`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/learner-0/training-result.json), [`seed 1 result`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/learner-1/training-result.json), [`lineage`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/learner-0/lineage.json), [`frozen training result`](../../experiments/dreamerv3-reused-train-multisource-v1-result.json) | Source0's original 2/12 standalone gate remains failed. The two sources share road IDs, so 24 actor episodes are not 24 independent geometries. Training loss is not prediction or student driving; no fresh P1b or matched random-vs-teacher learner comparison. |
| DreamerV3 mixed-source training-excluded development | A separately frozen static four-road r6 TRAIN-reuse development collection recorded four complete episodes per source0/random action stratum, no finished roads. Two 256-update world models scored the same 8+32 logged-action windows; the preregistered BOTH-seed/BOTH-stratum image-below-frozen-anchor-repeat gate failed. Random-action MSE was 0.001544/0.003862 vs repeat 0.001494; source0-action was 0.009116/0.013987 vs repeat 0.009230. | [`dev collection protocol`](../../experiments/dreamerv3-reused-train-multisource-development-v1.json), [`dev root receipt`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/development/collection-result.json), [`score protocol`](../../experiments/dreamerv3-reused-train-multisource-score-v1.json), [`score receipt`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/open-loop-v1/score-result.json), [`frozen score result`](../../experiments/dreamerv3-reused-train-multisource-score-v1-result.json) | Scorer executable was completed after collection (metric/window/gate were predeclared), and roads were already allocated r6 TRAIN. A single small seed0/source0 advantage cannot replace the failed two-seed gate. No finish-event evidence, actor, protected cell, model promotion or official result. |

| DreamerV3 mixed-source H8 prior-image training | After independent source and memory preflight, the SAME SHA-lineaged 10,612-decision source0/source1 previously allocated r6 TRAIN replay was used for two small world-model seeds. Each completed 256 ordinary plus 64 additional H8/weight0.25 prior-image optimizer steps, environment_steps=0 and unchanged actor/critic/target. The sealed lineage is identical per seed; own-replay auxiliary image MSE changes differ by seed. | [`training protocol`](../../experiments/dreamerv3-reused-train-multisource-prior-v1.json), [`seed0 receipt`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-0/training-result.json), [`seed1 receipt`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-1/training-result.json), [`source lineage`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-0/lineage.json), [`frozen training result`](../../experiments/dreamerv3-reused-train-multisource-prior-v1-result.json) | Historical training-only evidence. Development scoring is complete in the next row and failed; checkpoints remain diagnostic-only. No actor, promotion, or protected/official action. |

| DreamerV3 mixed-source H8 consumed-development score | Both frozen 256-base+64-extra model-only seeds were scored on the same four reused r6 TRAIN development roads per source0/random action stratum with paired pure256 windows/baselines. The strict BOTH-seed/BOTH-stratum improve-paired-AND-beat-frozen-anchor image gate failed: seed0 worsened versus own pure256 on both strata; seed1 improved paired MSE but random-action `0.001509` stayed above repeat `0.001494`. Four independent road episodes and zero finishes per stratum prevent driving conclusions. | [`H8 training protocol`](../../experiments/dreamerv3-reused-train-multisource-prior-v1.json), [`score protocol`](../../experiments/dreamerv3-reused-train-multisource-prior-score-v1.json), [`primary score`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-score-v1/score-result.json), [`frozen scored result`](../../experiments/dreamerv3-reused-train-multisource-prior-score-v1-result.json), [`pre-score cap failure`](../../talk/messages/20260927T025027Z-k9r4-dreamer-score-catalog-cap-abort.md) | One metadata-cap preflight failed before ZIP/output, was corrected/catalog-bounded and source-refrozen before the first actual score. Same development was consumed tuning, final scorer postdated collection, and extra64 optimizer steps are not compute matched. Seventh static roads remain unopened; no actor, fresh P1b, protected or official score. |

**TD-MPC2 seed-exploration comparison (2026-09-28):** The same four already
consumed obstacle-enabled track-1 TRAIN roads, twice per action source, compared
iid independent 3D gas/brake to 2D exclusive signed longitudinal control.
The [frozen comparison result](../../experiments/tdmpc2-exploration-v2-result.json)
and [v2 protocol](../../experiments/tdmpc2-exploration-v2.json) bind the
[primary result](../../runs/tdmpc2-exploration-20260928-v2/result.json) and
step/episode hashes. Mean progress was 0.06897 for 3D (2,573 decisions,
8 episodes) versus 0.03682 for 2D (1,630 decisions, 8 episodes), and the
predeclared progress-first rule selects 3D for a **separate from-scratch**
longer-budget run. 3D episodes lasted longer and had more negative whole-episode
raw return (-65.95 versus -50.95); both arms had zero finishes and damage.
The first collector's one-reset/zero-decision metadata-check failure remains
preserved [separately](../../runs/tdmpc2-exploration-20260928-v1/failure.json).
This is reused-TRAIN action-source evidence, not trained-policy generalization,
a learner comparison, or an official result.

**TD-MPC2 v2 logged-sequence ranking limitation:** A read-only, source-bound
H=3 open-loop reward scorer checked v2's replay/ledger actions, rewards and
terminal boundaries across 37 already-consumed TRAIN episodes; no reset was
performed. All 666 cross-episode first-three-step real-return pairs are tied
at 1e-6 tolerance (range below 1e-15), so there is no identifiable ranking
signal. The [frozen diagnostic](../../experiments/tdmpc2-v2-logged-ranking-diagnostic.json)
is neither a same-state prefix branch nor evidence of world-model quality.

**TD-MPC2 100k TRAIN follow-up (partial attempt, not a learner result):** The
[frozen protocol](../../experiments/tdmpc2-long-reused-train-v1.json) at
SHA-256 `bc1a2746845cbef89275c9b51163c273955ef1664fe833da35ed17e534e8c885`
passed zero-reset preflight and began new 3D from-scratch training on the four
already-consumed roads at 2026-09-28 15:44 UTC. It stopped with an
action-bounds `ValueError` at 10,020 decisions/updates, after the 10,000-update
pretraining boundary, before the first 20k checkpoint. The partial primary
ledger at `runs/tdmpc2-long-20260928-v1/` has no exact resume; its
[frozen failure receipt](../../experiments/tdmpc2-long-reused-train-v1-failure.json)
pins ledger hashes and the strongly inferred diagnostic-only float32
weighted-elite mean overflow (the unrecorded failing value was not measured).
All 10,020 applied actions were valid; no longer-budget benefit or ranking
improvement is established by this attempt.

**TD-MPC2 corrected 100k retry v2 (running):** A
[separate pinned protocol](../../experiments/tdmpc2-long-reused-train-v2.json)
at SHA-256 `d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`
retains v1's four consumed TRAIN roads and H3/10k seed/pretrain/100k targets,
changing only the diagnostic weighted-elite mean's one-ULP overflow tolerance.
Regression tests (88 passed) and zero-reset preflight passed; the new
`runs/tdmpc2-long-20260928-v2/` ledger began at 16:27 UTC. The original
partial v1 run is preserved. Its first completed-episode checkpoint at 20,099
decisions/updates now has a [source-bound frozen result](../../experiments/tdmpc2-long-reused-train-v2-20k-result.json):
0/67 on-policy reused-TRAIN episode finishes, mean progress 0.06746, fixed
seed-probe positive terminals 0/768, and same-anchor model-reward H3 ranking
24/40 concordant pairs versus the pilot's 25/40. The second sealed model at
40,024 decisions/updates has a [frozen 40k result](../../experiments/tdmpc2-long-reused-train-v2-40k-result.json):
0/124 cumulative on-policy finishes on four reused roads; 20k-to-40k
episode progress and raw return rose with longer episodes and more damage,
fixed-probe reward MAE worsened 0.059 to 0.088, and same-anchor reward ranking
remained 24/40 (pilot 25/40). This is not a matched independent policy test
or a world-model ranking improvement. 70k/100k and any full-episode frozen
policy/road-diverse assessment remain pending.
An [additional fixed-branch magnitude check](../../experiments/tdmpc2-long-v2-branch-calibration-v1.json)
on the same 60 short sequences found H3 raw-return MAE 0.904 (pilot), 1.190
(20k) and 0.900 (40k): 40k recovers roughly to the pilot but does **not**
improve the primary within-anchor ranking.

**TD-MPC2 70k TRAIN checkpoint (run continues):** The
[frozen 70k result](../../experiments/tdmpc2-long-reused-train-v2-70k-result.json)
binds the completed-episode model at 70,361 decisions/updates, 0/220
cumulative TRAIN finishes on four repeatedly trained roads, and three
separately hashed read-only loss, reward-ranking and balanced-terminal
receipts. The 96 episodes since 40k averaged progress 0.3463 and damage
0.3979; 19/96 exceeded half progress, but none finished. On the SAME
40 informative short action-sequence return pairs, world-model reward-only
ranking improved pilot 25/40, 20k 24/40, 40k 24/40, to **70k 33/40**.
The balanced predicted-latent terminal path recalled 53/67 raw endings
(true-next-image path 0/67) on fixed 20k replay, without finish/timeout
labels. This is a promising dependent **within-TRAIN** model signal, not
frozen-policy driving success, planner-Q validation, fresh generalization
or a >=50% completion result. H3/default MPPI were preserved to 100k.
The continuing run later recorded its
[first single on-policy reused-TRAIN finish](../../talk/messages/20260928T200236Z-k3p7-tdmpc-first-training-finish.md)
at decision 74,216 (one 646-decision episode, 1/232 training episodes at
that boundary). This is after the 70k snapshot, not attributable to the
frozen 70k actor or evidence of an independently replicated finish rate.

**TD-MPC2 completed 100k model and frozen full-episode reused-TRAIN check:** The
[source-bound final model result](../../experiments/tdmpc2-long-reused-train-v2-100k-result.json)
seals 100,354 decisions/updates, 307 TRAIN episodes, complete checkpoint and
step/episode ledger SHAs in 17,459 wall seconds. On-policy repeated training
episodes finished 14/307, including 14/87 after70k across four consumed
roads; these are NOT the frozen model's finish denominator. On the SAME fixed
40 informative H3 return-order pairs the final model scored **31/40**,
versus pilot25/40,20k/40k24/40,70k33/40: directional in-TRAIN model
signal, not a matched independent/generalization or planner-Q benefit.
The fixed replay has zero positive terminal labels; a separate balanced
in-replay probe's predicted-latent raw-ending recall reached 59/67, with
no finish/timeout examples. The predeclared
[four-road, 16-episode full-episode CPU protocol](../../experiments/tdmpc2-full-consumed-train-v1.json)
SHA-verified completed result/model/ledger/source and passed zero-reset
preflight. Its [frozen result](../../experiments/tdmpc2-full-consumed-train-v1-result.json)
contains 16/16 uncensored 2,000-decision-scope episodes on the SAME four
training roads x2 repeats/mode: **prior 0/8 and MPPI 0/8 finishes**.
MPPI mean progress 0.507, raw return +385.93, damage 0.45 versus prior
0.327/+237.73/0.35; paired reset conditions do not mean matched actions.
Neither the late *training* 14/87 finish count nor model ranking31/40 is a
frozen-policy completion result. This is a valid negative on four reused
roads, not a fresh multi-track diagnostic, official score, generalization
or intrinsic TD-MPC2 failure. A separate one-axis training-target damage
penalty is an untested TRAIN follow-up; H=5, MPPI samples and BC are not
co-tuned or authorized by this result.
The separate [damage-only training protocol](../../experiments/tdmpc2-damage-shaping-v1.json)
SHA `4f037f39ac7ab4f56961723478048a72d604972c86be17b9b0b9dc5786cbd2c8`
has now passed zero-reset preflight and 88 synthetic TD regressions and
started a new TRAIN run under `runs/tdmpc2-damage-20260928-v1/`; the first
episode is random seed exploration, **not** a learned-policy or completion
result. Replay reward penalizes only positive cumulative damage increments;
primary raw environment metrics and all other H3/model/planner settings
stay unchanged. The [first treatment 20k checkpoint](../../experiments/tdmpc2-damage-shaping-v1-20k-result.json)
sealed at 20,149 decisions/updates (0/68 TRAIN finishes), but **0/20,149**
step damage increments means raw and training reward sums are identical:
the penalty had no exposure before this model. Its first actual +0.2
damage/one-unit training-target cost was logged later at decision 22,123.
The [40k treatment result](../../experiments/tdmpc2-damage-shaping-v1-40k-result.json)
seals 40,154 decisions/updates with 0/131 cumulative TRAIN finishes;
65 positive step damage increments summed to13.0 and deducted exactly
65 units ONLY from replay target reward. The 63 episodes since20k averaged
raw progress0.1805, raw return+47.20, damage0.206 but finished 0/63.
The [sealed 70k treatment](../../experiments/tdmpc2-damage-shaping-v1-70k-result.json)
has 70,481 decisions/updates, 2/203 cumulative evolving-TRAIN finishes
(2/72 since40k, both on ONE reused road) and a SHA-bound cumulative
replay-only damage cost of207 units from207 step increments. This is not a
frozen-policy or causal shaping improvement. The
[completed 100k treatment source](../../experiments/tdmpc2-damage-shaping-v1-100k-result.json)
then sealed 100,186 decisions/updates and 286 evolving-TRAIN episodes,
10 on-policy finishes (8/83 since70k) and 391 raw-minus-replay-only
damage penalty units. The separately
[frozen RAW-outcome full-episode result](../../experiments/tdmpc2-damage-full-consumed-train-v1-result.json)
completed all16 planned, uncensored episodes on four heavily used roads:
prior0/8, MPPI **2/8**, both MPPI finishes on two repeats of ONE geometry.
The predeclared local MPPI >=4/8 gate failed; baseline RAW-target MPPI
finished0/8 under the same road/reset conditions, but actions and learner
trajectories were not matched and no causal shaping benefit is established.
The damaged-target
reward-head fit is not comparable to the baseline's raw H3 reward ranking.
The [seed-boundary control note](../../talk/messages/20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork.md)
records all 28 complete seed episodes matching the raw baseline, but first
planned action10,001 differing before observed damage. Same source settings
are not matched transitions, and no10k model/RNG snapshot exists; future
one-seed outcome differences cannot be causally ascribed to shaping alone.
The [SHA-bound seed replay-byte audit](../../experiments/tdmpc2-raw-vs-damage-seed-parity-v1.json)
confirms 0/10,000 differences in seed observation pixels/actions/raw
rewards/terminal flags and identical 256 frozen-probe windows, but does not
recover the missing 10k model/RNG state or explain that first planned fork.
The separate shaped checkpoint-specific CPU evaluator's source-level
[HOLD was repaired](../../talk/messages/20260928T232547Z-k3p7-tdmpc-damage-evaluator-code-ready.md):
strict replay/probe-to-step rewards, producer raw telemetry and reset-intent
provenance passed 123 synthetic/adjacent tests and an independent read-only
review. The later completed treatment result and a distinct SHA-bound CPU
protocol enabled the separate16-episode result above; the frozen RAW
baseline evaluator and official environment remain unchanged.
The [no-reset frozen-model device probe](../../experiments/tdmpc2-final-100k-freeze-parity-v1.json)
found close CPU/GPU forward agreement under identical replacement image
shifts on four archived TRAIN H3 windows; an otherwise identical CPU
planner-mode pair differed in its final Gaussian action noise by L2 0.0823.
Neither probes historical full-episode failure actions or proves the cause
of prior/MPPI 0/8; treatment source remains a separate hypothesis.

**TD-MPC2 balanced terminal-loss diagnosis:** The
[frozen source-bound result](../../experiments/tdmpc2-long-v2-balanced-terminal-20k40k.json)
adds 67 raw-terminated and 335 ordinary-negative transitions from identical
20k replay windows, after the original 256-window probe had 0/768 positives.
The true-next-image latent path recalls 0/67 at both 20k and 40k; the
model-predicted rollout path recalls 42/67 and 43/67, with 335/335
ordinary-negative specificity. There are no finish or time-limit examples;
this is only an in-replay TRAIN path comparison, not validated terminal
prediction on unseen roads or a diagnosed source bug.

**TD-MPC2 v2 within-anchor H3 reward ranking:** The
[frozen branch result](../../experiments/tdmpc2-v2-prefix-branches-v1-result.json)
binds the [primary receipt](../../runs/tdmpc2-prefix-branches-20260928-v1.json)
from 12 predeclared, full-prefix-parity-checked anchors and 60 fixed candidate
suffixes on four consumed TRAIN roads (72 resets). Of 120 within-anchor pairs,
80 had tied actual returns; among 40 informative pairs, v2 model-predicted
reward-only rankings matched 25 and disagreed on 15 (62.5%). This is a
dependent-pair development diagnostic, not generalization or an official
score. Historical hidden Box2D identity remains unproven; 20k and 40k each
scored 24/40, 70k scored 33/40 and final100k scored 31/40 on those anchors.

**TD-MPC2 H5 real five-action decision gate (2026-09-29):** A weak
[256-window logged H5 quality screen](../../experiments/tdmpc2-h5-logged-v1-result.json)
passed without resets, then the separate [frozen five-action branch protocol](../../experiments/tdmpc2-h5-branches-v1.json)
and [source-bound result](../../experiments/tdmpc2-h5-branches-v1-result.json)
recorded 72 prefix/branch resets on four **already-consumed** TRAIN roads.
Forty-four passing synthetic branch tests check source/parity/receipt code
safety, not driving performance. The measured real run comprised
12 anchors, 60 complete H5 candidate suffixes, 120 same-anchor real-H5
pairs, 29 real ties. The predeclared reward-only gate passed narrowly at
56/91 concordant pairs across four roads, but on the SAME candidates and
actual H5 outcomes H3 reward scored 63/91 versus H5 56/91, and the
terminal/Q sampled-planner score fell from H3 65/91 to H5 51/91. H3/H5
planner top-choice tied-best H5 outcome on 8/12 versus 7/12 anchors;
the two score horizons randomly sampled different Q-head pairs at 10/12
anchors, so the 65/91 versus 51/91 contrast does not isolate horizon or
prove full-MPPI inferiority. Four roads and one Q sample preclude an
independent-road policy claim.
The separately [SHA-bound read-only all-ten-pair Q control](../../experiments/tdmpc2-h5-fixed-q-v1-result.json)
held critic-head pairs and bootstrap-policy RNG identical across H3/H5:
H3 ordered 63-72/91 actual H5 pairs versus H5 42-53/91 on every fixed
pair, with zero new resets/updates. This strengthens the fixed-candidate
H5 HOLD but cannot isolate depth from model/terminal errors or predict
full-MPPI-policy finishes.
The previous historical H3 31/40 refers to a different three-action
branch set, not these H5 outcomes. Despite passing a necessary reward
screen, actual H5 action ordering is not sufficiently positive to release
full-episode H5 MPPI. No H5 policy episode, protected cell, fresh-road,
model promotion or official score exists.

**TD-MPC2 frozen H3 action-mode intervention (2026-09-29):** The
[separate source-bound CPU21 eight-episode result](../../experiments/tdmpc2-h3-noise-full-consumed-train-v1-result.json)
held RAW100k weights/H3/512-sample MPPI/raw environment/four consumed
TRAIN roads fixed and toggled ONLY final-action training Gaussian
`eval_mode=False`. All 8 episodes completed uncensored, with **2/8 finishes
on two roads**, below the predeclared >=4/8 and >=2-road local gate;
the original frozen no-Gaussian H3 MPPI completed 0/8 on the same reset
IDs. Different stochastic actions and subsequent planner RNG prevent a
trajectory-matched causal attribution, and neither result is fresh or
official. The 147 combined synthetic TD diagnostic/evaluator tests and an
independent source review support code safety, not driving benefit.
A follow-up [read-only consumed-TRAIN reuse inventory](../../scripts/audit_tdmpc2_consumed_train_reuse.py)
and [synthetic tests](../../tests/test_audit_tdmpc2_consumed_train_reuse.py)
verify inputs but **always BLOCK** prospective 24-road training: legacy
result-only exposures and protected-ID metadata lack candidate-specific
closure. The explicit [24-cell catalog-order TRAIN proposal](../../experiments/tdmpc2-consumed-train-24-proposal-v1.json)
was inventoried read-only: 24 IDs/four per family and 921 typed exposure
rows, but many known TD/DrQ prior TRAIN records remain unclassified and
the tool unconditionally blocks; this is NOT evidence all proposed roads
collide. No claim, actual training reset or new learner exists. Road
coverage is untested rather than disproved. The next feasible one-axis
TRAIN diagnostic is an all-five-Q-head H3 ranking screen on the already
consumed branches; launch full episodes only if its fixed choice-quality
gate passes under a new source-bound protocol.
That [read-only all-five-Q screen](../../experiments/tdmpc2-h3-all-q-v1-result.json)
subsequently **FAILED** its predeclared >=66/91 real-H5 pair requirement:
65/91 concordant (same as stochastic H3), despite tied-best 9/12 and
slightly lower regret 0.79416 versus 0.79535. No new policy reset or
training update followed. A next single-axis model target intervention
must use source-bound evidence on already consumed TRAIN cells; H5 rollout
reward error worsened with depth but no causal head/dynamics attribution
exists.
The separately [SHA-bound no-reset overshoot replay audit](../../experiments/tdmpc2-reward-overshoot-target-audit-v1-result.json)
subsequently found 99,126 same-episode H5-extendable windows among 99,740
eligible H3 starts (99.38% >95%) across 307 completed episodes, with
nonconstant step-4/5 reward on all four consumed TRAIN roads. The
isolated new `haic/algorithms/tdmpc2/reward_overshoot.py` adds only
masked depth-4/5 reward CE while preserving old base modules; 38 focused
synthetic/base tests passed. This is *target availability and code safety*,
not itself learned performance. A candidate-specific consumed-TRAIN
claim audit was completed before the first separate variant reset.
The NEW isolated reward-overshoot trainer's source-bound/partial-run
contracts passed 63 CPU/fake-environment tests and an independent
read-only review. Both container/host PID identity and own-GPU-utilization
false alarms were fixed BEFORE the first
[synthetic CUDA receipt](../../experiments/tdmpc2-reward-overshoot-throughput-v1-result.json):
the original 128 B256 full updates averaged 0.07749s and reward-overshoot
updates 0.08978s, forecasting 18,693s for the same 100,354 updates,
below the predeclared <20,600s gate/unchanged 21,600s wall cap. The
generated fixture has only 434 transitions; this is resource evidence,
NOT actual 100k replay throughput or driving benefit. The
[variant protocol](../../experiments/tdmpc2-reward-overshoot-train-v1.json)
passed strict zero-reset source/runtime/cell/benchmark/resource preflight.
The original four consumed TRAIN cells received a contemporaneous
claims/protected-ID metadata review before the first training reset;
the read-only 4/4 candidate-specific review found no current exclusive
or protected conflict, and a [non-exclusive reuse intent](../../talk/messages/20260929T122427Z-p6c4-tdmpc-overshoot-consumed-train-reuse-intent.md)
was posted before reset. The [100k learner](../../experiments/tdmpc2-reward-overshoot-train-v1-result.json)
completed at 100,159 decisions/updates (309 episodes, 19,902 seconds
under its unchanged 21,600-second wall cap). Its [zero-reset score](../../experiments/tdmpc2-overshoot-old-branch-score-v1-result.json)
of the SAME 12 archived pixel anchors, five action bytes and actual H5
outcomes per anchor improved H5 reward order **56/91 -> 74/91**, H3
prefix **31/40 -> 34/40**, and all 4/4 roads' H5 ordering
nonregressed. However normalized H5 absolute reward error worsened
**0.39668 -> 0.40691**, so the predeclared four-way gate **FAILED**.
The eight-full-episode H3 CPU evaluator is dormant; no frozen-policy,
fresh-road or official benefit can be inferred. Its evolving TRAIN
collector's episodes are not a matched policy comparison. A new
single-axis hypothesis must account for the rank/magnitude split
without retuning the same failed gate.

**TD-MPC2 Q and positive-reward mechanism controls (2026-09-29):** The
[source-bound zero-reset planner-term receipt](../../experiments/tdmpc2-overshoot-planner-terms-v1-result.json)
found 0 predicted terminal-threshold crossings on 12 old TRAIN anchors:
new H5 reward-only 74/91 actual return pairs became **31-36/91**
with every fixed two-of-five Q-head pair, and selected-action mean
real H5 regret changed 0.502 -> 2.076 (five candidates, not full
MPPI). The separately [reward-blind 8,192-window replay probe](../../experiments/tdmpc2-overshoot-positive-events-v1-result.json)
found new-model positive raw reward underprediction even at true
observation-encoded latents on 4/4 old roads (new signed bias
-1.15..-1.37/step); its **head-only mechanism screen PASSED**, but
old model's in-sample/variant's off-policy replay and overlap prevent
causal attribution. The first probe's RAW307/new309 episode-identity
bug was stopped by its zero-load preflight and fixed before any real
checkpoint load. Neither diagnostic changes the failed four-way reward
gate or releases H3/H5 full episodes. A new **single-axis H5 Q0 planner**
was then evaluated on disjoint old RAW episodes4..7 on the same
already-consumed TRAIN roads. The [source-bound 72-reset real
branch result](../../experiments/tdmpc2-overshoot-q0-branches-v1-result.json)
returned all 60 complete H5 suffixes, 93 informative actual pairs
(27 ties) across four old TRAIN roads: H5 Q0 **68/93 (73.1%)** beat
each H5 fixed Q1 score **39-41/93**, but H3 Q0 reached **70/93** on
the SAME actual H5 outcomes. H5 Q0 tied-best **7/12** versus the
predeclared >=8/12, mean regret 1.279 versus H3 Q0 1.028. The
multi-condition Q0 action-choice gate **FAILED**; no H5 Q0 full
episodes, protected/official run or model promotion followed.
The reward-head-only positive-event mechanism PASS is a separate
hypothesis under read-only investigation, not a retrospective escape
from either failed gate.
The original planned head-only adaptation-excluded RAW ep12..19 split
then FAILED its prespecified >=100 positive labels per road (38/87/38/39)
before any optimizer step. A separately declared eight-episode-per-road
ep12..43 v2 split has 144/199/137/151 positives, passing only its
TRAIN source-count gate; ep44..306 would be the fit pool and ep8..11
stays excluded for a possible third branch study. Neither split is a
fresh model holdout. The [frozen v2 pilot result](../../experiments/tdmpc2-head-only-adaptation-v2-result.json)
subsequently completed512 head-only updates in367.66s, with0 resets
and non-head bitwise parity, but **FAILED** its excluded logged error
screen (0/4 qualifying roads). Positive-event MAE increased on all4
roads, and nonpositive MAE deterioration exceeded0.05 on3; training
CE reduction is not model or policy benefit. No ep4..7 preservation
score, new ep8..11 real branch or full policy run follows this FAIL.
Its CUDA budget and synthetic tests were operational safety only.
The [no-update frozen-head fit/transfer diagnosis](../../experiments/tdmpc2-head-fit-transfer-v1-result.json)
scored96,510 unique RAW transitions. Fit naturalCE1.5066->1.2208
but positive rawMAE worsened onALL4 roads: fitgate0/4 rejects
transfer-only attribution. Next prepare isolated raw-MSE head-loss
control from original overshoot parent with unchanged split/sampling
and512-step budget; no branch/policy release from this diagnosis.
MSE implementation passed 91 synthetic/parent tests and a targeted
independent result-finalization review. Before execution it was held
on a conservative memory gate (35.39 GB observed <42.67 GB configured).
On **2026-09-30 the user temporarily paused TD-MPC2 to focus on other
ideas**, cancelling the hourly follow-up. This is now a user-requested
focus pause, not an automatic resource-wait queue or MSE performance FAIL.
Generated v1/v2 benchmark PASS receipts are superseded-source evidence;
final-source v3 benchmark, actual protocol and MSE run remain absent,
with zero actual MSE loads/updates/resets. The [detailed paused plan](../plans/active/tdmpc2-pixel-online-baseline.md#user-requested-focus-pause-2026-09-30)
records the exact parent, objective-only difference, data/optimizer
constants, hashes, unresolved budget rationale and conditional gates.
No actual branch or policy outcome exists for MSE; explicit user
reopening is required before any further benchmark or resource check.

**Latest DrQ final-source status (2026-09-29 09:43 UTC; supersedes the 1/6
table row):** All six revision-2 learners completed their fixed TRAIN budgets.
The original CPU21 sample audit **failed** historical warmup-byte parity;
its receipt is absent and its matched replay-only causal question remains
unevaluated. A separately pinned [postrun r2 protocol](../../experiments/drqv2-final-source-replay-reconstruction-postrun-r2.json)
passed its [distinct zero-reset sample audit](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/pre-evaluation-postrun-r2-sample-audit.json),
then the [r2 diagnostic manifest](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/train-diagnostic-postrun-r2/manifest.json)
sealed 192 episodes on 16 already-consumed TRAIN-DIAGNOSTIC roads. Of the
same 11 source-success/21 source-failure actor/road cells per mixture, r2
uniform kept 2/gained 4, failure-weighted kept 5/gained 7 and easy-retention
kept 6/gained 6. All three **FAIL** >=9/11 kept AND >=2/21 gained even as a
descriptive signal; no DrQ candidate, protected/official evaluation or
promotion follows. The [report](drqv2-final-source-replay-v1.md) preserves
the original failure, cross-GPU/child-replay-parity limitations and primary
artifact hashes. A separate hypothesis requires a new protocol and user
decision, not tuning on these consumed cells.

## DrQ-v2 Geometry-Mix r6 TRAIN-DIAGNOSTIC

Canonical repeat-0 descriptive outcomes pool both learner seeds over the same
previously designated 16 TRAIN-DIAGNOSTIC roads. Each `n` is the episode
denominator across those two seeds; mean maximum progress is visited-tile
fraction, not physical road distance. Repeat 1 reproduced repeat 0 exactly and
is not an independent sample. These rows are development diagnostics only.

| Geometry family | Uniform finishes / n (mean max progress) | Failure-weighted finishes / n (mean max progress) | Easy-retention finishes / n (mean max progress) |
|---|---:|---:|---:|
| `easy-curvature-anchor` | 1/6 (0.476) | 1/6 (0.470) | 0/6 (0.456) |
| `finish-approach-turn` | 0/4 (0.755) | 0/4 (0.706) | 0/4 (0.610) |
| `mid-road-left-right-reversal` | 0/6 (0.468) | 2/6 (0.766) | 1/6 (0.640) |
| `mid-road-sustained-or-same-turn` | 0/4 (0.331) | 0/4 (0.414) | 2/4 (0.795) |
| `opening-delayed-high-turn` | 0/6 (0.337) | 1/6 (0.525) | 0/6 (0.431) |
| `opening-short-entry-left-turn` | 0/6 (0.455) | 0/6 (0.551) | 0/6 (0.727) |

The [full diagnostic summary](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/summary.json),
[manifest and trace inventory](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json),
and [active plan with per-run checkpoint/replay hashes](../plans/active/drqv2-geometry-mix-plan.md)
retain the complete 256-episode record. No generalization, promotion, or official
performance conclusion follows from these reused development cells.

## Companion Evidence

- The historical DrQ-v2 promotion record is a caveat, not a standalone research
  direction: its undocumented historical result was 4/24, its fresh screen was
  0/24, and its predeclared diagnostic confirmation was 4/32. It was never
  promoted. See [`protocol`](../../experiments/drqv2-promotion-v1.json),
  [`result`](../../experiments/drqv2-promotion-v1-result.json), and
  [`diagnostic decision`](../../experiments/drqv2-confirmation-diagnostic-v1.json).
- The teacher-replay experiment has two preserved pre-collection/pretraining
  failures before r3: the original r1 sampler-seam protocol had zero resets, and
  r2's trainer stopped on a Python binding error before any online decision. The
  r3 protocol used fresh cells excluding both frozen allocations and their run
  artifacts; its A3 coverage stop is not a model-quality comparison. See
  [`r1 abort`](../../runs/20260924-drqv2-teacher-replay-v1/pre-collection-abort.json)
  and [`r2 abort`](../../runs/20260924-drqv2-teacher-replay-v1-r2/training-preflight-abort.json).
- Pixel RLPD v1 was frozen but aborted before environment interaction because the
  collector import graph changed the effective installed-distribution inventory.
  Its zero-cell preflight is preserved at
  [`v1 preflight`](../../experiments/pixel-rlpd-offpolicy-pilot-v1-preflight.json);
  v2 uses a corrected, post-import lock and entirely new cells. V2's first evaluator
  invocation also failed before candidate discovery and consumed zero screen cells;
  the corrected run-root config and successful same-protocol retry are detailed in
  [`v2 execution`](../../runs/20260924-pixel-rlpd-offpolicy-pilot-v2/execution.json).
- The immutable v2 protocol's collector/trainer command templates name
  `runs/20260924-pixel-rlpd-pilot-v2/`, while the actual teacher collection and
  four training commands recorded in the execution receipt above used
  `runs/20260924-pixel-rlpd-offpolicy-pilot-v2/`. This is an execution-path
  deviation, not a second teacher dataset or a retroactive protocol amendment;
  retain the original protocol hash and use the recorded command/receipt paths
  when reproducing its outcome.
- [`drqv2-pre-l2-diagnostics.json`](../../experiments/drqv2-pre-l2-diagnostics.json)
  supplied the bounded L2 hypothesis; it did not itself establish a causal failure
  mechanism.
- [`drqv2-steering-logit-proposal-v1.json`](../../experiments/drqv2-steering-logit-proposal-v1.json)
  is the preregistered L2 proposal and remains historical after rejection.
- The Dreamer feasibility JSON originally marks its Gate 3 diagnostics `PASSED`.
  The later B1 failure diagnosis and DreamerV3 closure supersede that
  interpretation: the recorded diagnostics are preserved, but they are insufficient
  proof of a faithful, decision-useful world model. Gate 1 and Gate 2 remain valid;
  Gate 4's pilot failure remains valid.

See [`docs/decisions/INDEX.md`](../decisions/INDEX.md) for the consequences of this
evidence and `docs/plans/active/` for work still authorized.
