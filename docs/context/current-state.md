# Current Research State

## Single-Intervention Branch Completed (2026-10-05)

**Current joint successor excluded from submission candidates.** The first
intervention alone does not retain a lap-time benefit under champion feedback;
later interventions cannot be the sole explanation for C's regression. This is
one consumed TRAIN road, not a claim that joint-control is generally impossible.

- [Plan](../../experiments/joint-single-branch-v1-plan.json), [reuse certificate](../../experiments/joint-single-branch-v1-reuse.json), [result/reproduction](../../experiments/joint-single-branch-v1-result.json), [ABC summary](../../runs/joint-single-branch-v1/summary.json) SHA`cc51986f...`, [actual-feedback H4 costs](../../runs/joint-single-branch-v1/feedback-cost.json) SHA`5004b48a...`.
- Only ONE new reset: B on3/3184000005,274decisions/1093driving rawticks plus51warmup, no retry/error/replacement. A/C and priorfixedH4 records REUSED after exact online34 observation/state/hypothesis/prediction/action checks. All33prefix actions/pre34states match; B34action andfirstH1match C. TimeLimit64vs1200 difference inshortrecords is explicitlyinactive, not a hidden-solverclone claim.
- B intervenes only34, then all273otherissued actions equal freshchampion proposals from Bobservations with actualbrake/wheel/box/observer history. No replay of A's futurecontrols. A/B/C allfinish: **21.720/21.860/22.100s**, B-A=+140ms,C-A=+380ms,C-B=+240ms. Interventions0/1/6, all-wheel-road-loss spells0/1/2; Bspell.22s at2.70s afterits soleintervention. Collision/contact/damage0all. B also regresses; no one-shot fix promoted.
- Same-online-state fixedtail: predictedcostdelta[-.731880,-.263843], actual[-.568279,-.457101]. Actual FEEDBACK H4 also favors B overA[-.742509,-.624026] andC overA[-1.113553,-.938447], common17/17support. Cost order C/B/A isopposite laporder A/B/C. The benefit isnot merely a failed fixedtail reproduction or erased immediately byreplanning.
- B-A signedGTprogress:+.072832 atH1,+.626208 atH4, then-.331300 at1s and-.699620 at2s. At1s Bspeed is7.1924world/s lower, gasintegral .036action-seconds lower, brakeintegral .015347 higher andsteeringvariation .162 higher. Observedlater response, not a uniquelyisolated braking/steering mechanism. Short-horizon benefit is not wholefeedback utility; no new weight/horizon/physics fitting follows.
- Allactual35..37continuations differ fromfixeda0tail. Bhas1forecast/1continuationmismatch/0qualifiedH4 prediction windows; physicalroadloss isnot a claimedpredictionrange miss. Measuredactualfeedbackcosts are separate fromforecast-error labels. One-shotfullact CPU p99=38.927ms,max305.144ms (274calls,onlyonecomparison); lowp99 isnot a planningoptimization claim.
- 233combined tests+61subtests passed, then24targeted tests including anadditional pre34 byteguard case;234distinctcases covered. Zero-reset preflight passes. Review tightened liveearlyfork rejection and report-bound costprovenance before B. OriginalC/calibration/1/1/1/.04/.05/H4/weights/champion/distillation unchanged; no official action. Researchcode/sourcecommit`f80eae9` pushed; raw/frozen/generated data local.

The requested narrow branch is complete. Current C has no actual-feedback lap
improvement and is excluded in the candidate ledger. B is a fixed-step diagnostic,
not a deployable generalized policy. Keep the official champion unchanged; no
further run or outcome-fitted correction under this request.

## Joint Diagnosis And Negative Pilot (2026-10-05)

**The correction produces real interventions, but this one-road pilot regresses;
do not promote it.** Diagnosis, three new paired starts and the predeclared
conditional full pair are complete. No observer-only explanation, arbitrary
winner-tuning, safety guarantee or generalized improvement is established.

- [Fixed plan](../../experiments/joint-temporal-diagnosis-v1-plan.json), [diagnosis](../../experiments/joint-temporal-diagnosis-v1-diagnosis.json), [final result and reproduction](../../experiments/joint-temporal-diagnosis-v1-result.json), [primary funnel](../../runs/joint-temporal-diagnosis-v1/funnel.json) SHA`00d5f270...`.
- 931 decisions=856 pre exclusions+75 comparisons. Fixed first-failure pre counts: observer/forward287,mapping1,pilot envelope464,shield/recovery/feedback104. Nonexclusive counts are also retained; mapping217 overlaps observer216 times. Do not equate ordering-dependent exclusive counts with causal importance or invent costs for856 uncomputed decisions.
- 75 comparisons:59 common full-cost support,19 also pass unchanged risk gates. Those19 split into2 shared-stress ambiguities and17 old-allowance ambiguities. Four gains survive old allowance, but all fail road margin. Observer alone and allowance alone are both incomplete explanations.
- Oldq=.449619 is correctly paired, not added twice. Its maximum is a negative `brake1` residual from an extreme state/reference. Bounding error to EVERY prediction already incorporates state spread, then the cost interval retains that spread again; abs(q) also penalizes the upper tail. This is a conservative target mismatch relative to envelope containment, not an indexing/arithmetic bug.
- Separate [empirical paired-envelope target](../../haic/algorithms/joint_control/paired_residual.py) fits only error outside each shared-reference state interval. Raw old-CAL lower/upper=.03751048/.03997636; a **NEW** conservative residual floor .05 gives .05/.05. The original .05 was the material-order threshold, not a residual floor. [Calibration](../../runs/joint-temporal-diagnosis-v1/envelope-calibration.json) SHA`a0ceed50...` preserves the complete old physical/mapping calibration exactly, as well as1/1/1,weights,.04/.05,H4 and risk gates.
- Posthoc old-pilot state classifications would pass17/931 with the new target versus0 before (9/5/3 bycell). These are NOT17 achievable interventions, benefits or valid continuation labels. Old8paired starts and45natural rows still have zero fullyeligible alternatives under unchanged physical guards.
- New paired starts on consumed roads outside the old joint split completed: cells1/3184000003,2/3184000004,3/3184000005, first causal eligible/common-supported prefixes18/18/33, without winner/risk/future-label selection. [Result](../../experiments/joint-temporal-diagnosis-v1-paired-result.json), [primary analysis](../../runs/joint-temporal-diagnosis-v1/pairs/analysis.json) SHA`1043a3c6...`:9resets,243actual decisions,1431raw incl459warmup; allrepeat/prefix parity, nooperator error/retry/newdatafit. Declared caps64decisions/307raw perarm are NOT actual usage.
- Actual observed-scene cost intervals favor alternative in3/3. New performance bounds favor2/3 but unchanged physical guards permitonly3/3184000005. Firstcell remains unknown/road-margin-vetoed; secondjoint state/reference stress already fails material ordering before allowance and also failsphysical guards. Third has19/19absolute support/minroadgap1.293127, predicted[-.731880,-.263843] versusactual[-.568279,-.457101]. Oldbounds abstained onall3. All15correlatedreference labels and6correlatedpaths lie within frozenempiricalbounds; actualall6paths have4road wheels/0contacts/damage. Not a safetycoverage guarantee.
- Observed-reference progress gains1.09554/.97859/.42608 differ from separate GTcenterline gains .05381/.03535/.42479, with rawrewarddelta0each. Cost improvement is not lap utility. Onlythe third newpair qualifies the predeclared conditional fullstage; no all-regime prerequisite or selection by eventual lap outcome.
- [Conditional full summary](../../runs/joint-temporal-diagnosis-v1/full-pilot/summary.json) SHA`72dad221...`, protocol`a2f07616...`: exact first33decisions/preaction34/rawprefix match, then6actual interventions/277successor decisions(2.166%),271abstentions(97.834%),16comparisons. Baseline272decisions. Both finish:21.720s baseline versus22.100s successor, **+.380s/about1.75% slower**. Completion1/1each, progress1.0/.987988 separately;0collision/contact/damage both. Actualeffect evaluated on ONEroad, not replicated improvement.
- Successor has2all-wheel-road-loss spells versusbaseline0 (18rawticks/.36s total,max.28s; any-wheel loss91ticks versus75). They occur1.02s and14.72s afterlastintervention55; temporal association, not a per-intervention causal failure rate. All16onlineH4forecasts have different later actual controls, so qualifiedcoverage0 andmiss-rateN/A, not16misses orsuccesses. Do not confuse these road-loss events with measured.32s model-range departures.
- Whole successoract CPU n277:p50=20.056ms,p95=167.169ms,p99=173.189ms,max178.955ms;0calls>1s; max sampledRSS305127424B. Differentroad/decision mix from old253.6ms result, not a demonstrated CPUoptimization.244combined tests+72subtests passed, then29hardened-gate tests including6new cases;250distinct cases covered.5zero-reset imports pass. Gate checks report-bound masks onall3anchors before admission, not justPASS labels.
- [Exposure](../../experiments/joint-temporal-diagnosis-v1-exposure.json), [paired claim](../../experiments/joint-temporal-diagnosis-v1-claim.json), [full claim](../../experiments/joint-temporal-diagnosis-v1-full-claim.json), [full admission](../../experiments/joint-temporal-diagnosis-v1-full-admission.json) bind this study. Total11resets spent(9paired+2full), noerror/retry/replacement/post-outcome tuning or extraepisode. Champion/distillation, original physicalcalibration/evidence and unrelated RLPD work preserved; no official action ormodel replacement. LBMPC separation is context, not a transferred theorem.

The remaining question is the connection between isolated fixed-tail cost benefit
and actual feedback/later lap utility. Objective/progress proxy, continuation and
later response interactions are hypotheses, not isolated mechanisms. Preserve
the negative one-road result; it does not prove joint-control impossible or
justify a blanket observer-first recommendation. No automatic next run.

## Interval Road Pilot Completed (2026-10-05)

**Road-cost support improved; successor effect NOT EVALUATED because intervention
was zero.** The separately authorized interval study and all six full TRAIN
episodes completed. Champion, path/speed distillation, original physics1/1/1,
steering/pedal increments .04/.05 and H4/.32s remain unchanged. No official action,
model replacement, desired-winner weighting or post-pilot gate relaxation.

- [Plan](../../experiments/joint-temporal-interval-v1-plan.json), [result](../../experiments/joint-temporal-interval-v1-result.json), [paired analysis](../../runs/joint-temporal-interval-v1/paired-analysis.json) SHA`19e2fc4e...`, [calibration](../../runs/joint-temporal-interval-v1/calibration.json) SHA`5c8fe665...`, [pilot summary](../../runs/joint-temporal-interval-v1/pilot/summary.json) SHA`91518e06...`.
- Road interior, bracketed uncertain boundary and unobserved pixels are separate. Boundary ambiguity can support an interval center reference but never fills free space. All candidates retain the same full17 endpoints and shared19 state/five reference stresses; the latter are finite empirical stresses, not exhaustive bounds.
- Same eight consumed pairs: common cost support2/8->6/8, four new/two retained/zero lost. Actual supported suffix orders are3 baseline and3 alternative. Reference-only predicted orders are3 baseline/2 alternative/3 ambiguous. The CAL cost-difference error allowance .449619 makes all eight prediction orders ambiguous; no retuning follows.
- Natural cost support straight2/25->9/25, left0/2->1/2, right2/5->3/5, braking2/13->9/13. Left2's road reaches the visible left edge on rows36..38, leaving no observed nonroad bracket, not the repaired one-pixel ambiguity; brake2 also lacks valid state/history. Unsupported suffixes are not truncated or normalized away. No extra paired resets were necessary.
- CAL three roads/five starts fit time-index total runtime errors including dynamics, H4 position1.095715/yaw.092495. On two roads/three starts excluded from this fit, same-scenario whole-path misses0/6, central-only misses2/6. Held-out mapping misses0/38 measured and0/15 valid inferred intervals. All are small consumed-TRAIN empirical ranges, not fresh validation, probability coverage or safety guarantees.
- Fixed pilot cells1/3184000013,3/3184000002,2/3184000006: baseline and successor each finish2/3; laps19.960s/17.860s and the same off_track DNF. Progress .987179/.992509/1.0 is not finish status. Every action and raw physical record matches across arms; six natural episodes,1862 decisions, no retry or omitted failure.
- Successor compares75/931 decisions (35/14/26 per cell), intervenes0/931 and abstains931/931. No candidate effect or post-intervention failure rate is estimable. Collision decisions/contact events/damage are0 in both arms; both retain the same three all-wheel-offroad spells and one DNF. This is not a safety improvement.
- All75 online forecasts have a different actually issued future continuation; pilot range coverage is0 qualified windows, NOT0/75 misses. The geometry-held-out paired check above supplies the separately measured containment result. Prediction-range departures and physical accidents remain distinct.
- Actual whole successor act CPU on931 calls: p50=24.062ms,p95=196.306ms,p99=253.613ms,max308.017ms; no call over1s. Baseline p99=31.135ms. This exceeds the earlier .2s internal target, though it is not an official inference failure. Maximum sampled successor RSS313683968B; total execution wall215.555s includes simulator/logging and is not inference latency.
- Integrated197 tests+65 subtests and both zero-reset CPU21 imports pass. Reviews fixed support diagnostics, actual-action feedback test gaps, JSON initial-parity and partial/provenance reporting before resets. Only targeted source/claim/champion checks were repeated, not the old large audits. Source and concise records are committed; raw/frozen run artifacts remain local.

The six-reset budget is exhausted. Preserve this coverage/no-op result without
claiming candidate efficacy, universal safety or promotion; no additional run or
weight/uncertainty reduction follows automatically. Historical failed gates below
remain unchanged and refer to their own earlier protocols.

## Temporal Joint Validation Completed (2026-10-05)

**Relative physical effects transfer to the eight tested starts, but usable cost
references and uncertainty coverage are not ready for a closed-loop pilot.** The
temporal observer and progress/road/clearance comparator are implemented with
original physics1/1/1. No4x inertia correction, post-outcome tuning, champion edit,
official action or successor pilot was used. All prior evidence remains frozen.

- [Plan](../../experiments/joint-temporal-v1-plan.json), [result](../../experiments/joint-temporal-v1-result.json), [primary analysis](../../runs/joint-temporal-v1/analysis.json) SHA`f24fa32f...`, [passive audit](../../experiments/joint-temporal-v1-audit.json) SHA`0fd38b86...`, [road-support diagnosis](../../experiments/joint-temporal-v1-road-support-diagnosis.json) SHA`4bcbfa45...`.
- Eight outcome-blind starts: high-speed straight, left/right corner entry, braking turn; two per regime on six layout cells/five consumed roads. Exactly24 resets/768 decisions/4296raw including1224warmup,232champion acts, no retries. Prefix and baseline-repeat parity pass. Grouped TRAIN verification, not fresh holdout.
- Temporal state carries actual actions and corrects HUD/motion;7/8 anchors valid versus3/8 current-frame motion flags,143/174 unique baseline endpoints valid. No claim that a valid flag certifies truth. Inferred mapping yaw misses its heuristic bound25/44 intervals; forward bound misses1. Measured-image118 intervals stay within saved component bounds.
- H4 centre action-effect directions right/forward/yaw/speed match8/8 each; median relative errors `.09170/.02088/.11998/.00515`. Identical19-state stress sets preserve material correct direction on7/8,8/8,7/8,8/8 respectively. These are correlated coordinates of eight starts, not32 independent trials or scalar-cost proof.
- Full new-cost labels exist only forright1/straight2: baseline2,alternative0,unknown6; both supported robust orders correct. Natural-prefix comparison coverage isstraight2/25,left0/2,right2/5,braking2/13. Allfour actual alternative GT-progress wins haveunknown new-cost labels; old/new scalarforms overlap on onlyone pair. Cost-bias absence is NOT established.
- Diagnosis findsfive in-view reference holes from one ambiguous `.52<gray<.54` boundarypixel, rejected by the immediate-known-edge plus contiguous-centre rule. First7 ego regions are77/77repaired; exact causal mean-pose oracle mapping rescuesZEROlabels. No demonstrated implementation/order/schema bug. Brake2 separately has invalid lateral uncertainty and nohistory repair. Do not declare unknown pixels safe or modify the frozen thresholds after results.
- All16 actual baseline/alternative suffixes remain physically safe(4wheel road contact,zero obstaclecontact/damage), but8 fail the declared same-scenario `.5world/.05rad` pose tube. Five predicted-safe paths all miss containment: failed safety certificates, NOT five accidents. No blanket absolute-RMSE-versus-effect rejection.
- Pilot gate0/4 regimes; pilot resets0. Timing is not the blocker: measured nominal+observer/scorer segment-sum p99~169.7ms/max173.1ms, not integrated successor.act. Computational raster slicing preserved costs/masks with disclosed existing float32 distance-transform ULPs; no model/weight/gate changes.
- 200 tests pass; independent404-file/856-pin audit checks108499 assertions and134525 numeric elements with max discrepancy1.707e-15. Collection259.60s includes capture, not inference. Exact champion ZIP/all11sources remain unchanged; no Git write, unrelated RLPD work untouched.

Next priorities are bounded/interval road-reference support without invented free
space, and calibrated joint-state/inferred-motion uncertainty. Use the retained
data for diagnosis/development, not as untouched validation after revising these
rules. Do not expand MPC/search or claim full controller readiness from component
sign accuracy. The authorized paired budget is complete and the conditional pilot
was not triggered.

## Relative Action Probe Completed (2026-10-05)

**Small physics model has local response signal; observation quality/coverage is
the next bottleneck. No controller is adopted.** The user's conditional TRAIN
collection was exercised after a separate, predeclared relative-effect protocol.
The earlier archive absolute gate remains failed0/3, not waived or rewritten.
Absolute error alone did not settle the user's relative-comparison question.

- [Probe result](../../experiments/joint-prediction-probe-v1-result.json), [frozen protocol](../../experiments/joint-prediction-probe-v1-protocol.json), [primary analysis](../../runs/joint-prediction-probe-v1/analysis.json) SHA`78dc71fc...`, [independent audit](../../experiments/joint-prediction-probe-v1-audit.json) SHA`866c1ed4...`.
- Three consumed Track1 cells, one start each, baseline/repeat/one joint alternative: exactly 9 resets, 126 decisions, 963 raw ticks including459 warmup; 33 champion acts, none after the anchor. Exact prefix raw/image/accessible-state parity and full baseline-repeat equality hold. No retries or fresh-cell claim.
- New synchronized truth allows direct HUD validation on57 unique frame-label records: speed/joint0/yaw-rate absolute p95 `1.0993 units/s / .0029509rad / .0290693rad/s`. Wheel-omega p95 `5.07-10.58rad/s`, with abstentions. These are three similar early acceleration starts, not broad coverage.
- At.32s, uncorrected source model with actual instantaneous state has median position/yaw/speed-change errors `.008409/.002454/.002735` across six endpoints. Same model with runtime estimates has `.647272/.037612/1.362050`. This isolates substantial state-initialization error in this narrow regime; it is not a global dynamics/safety proof.
- Source-default runtime H4 effect signs match3/3 for each component; median relative errors right/forward/yaw/speed are `.0814/.0403/.0328/.00420`. Frozen local tracking-proxy order matches3/3, but only **1/3 starts is runtime-valid**; two motion-invalid fallback cases are diagnostic only. This proxy is not lap utility and does not validate collision safety.
- The archived FIT4x inertia correction transfers poorly: runtime.32s position/speed-change median errors `1.7209/8.4191`, versus source default `.6473/1.3620`. Both parameter sets were frozen before probe data; no post-probe fitting or promotion. Do not interpret the correction as physical mass.
- 80 focused synthetic/integrity tests pass. Independent audit checks499 hashes/1161 arrays/6885 numeric comparisons with zero discrepancy. Collection464.50s includes heavy capture/fsync, NOT inference latency; maximum sampled RSS324,120,576B. Champion ZIP/all11 sources and142 environment pins remain intact; no official action or Git write.

Retain the public-physics prior and investigate wheel-state/HUD decoding plus
causal signed-motion/slip fusion with calibrated uncertainty and abstention.
Do not move directly to MPC, full replacement or shield-backed safety claims.
The nine-reset probe is complete; no additional collection is authorized by its
result. Missing data, observation failure and model-transfer failure remain
distinct categories. Champion micro-optimization stays closed.

## Joint Prediction Validation (2026-10-05)

**Observer signal confirmed; the current small predictor fails its collection
gate.** The user's 2026-10-05 instruction authorized implementation, geometry-split
archive validation and, only if promising, bounded TRAIN comparison collection.
Missing prior joint branches is a **data gap**, not a technical-failure rule.
Champion micro-optimization remains closed; no full controller was implemented.

- [Plan](../../experiments/joint-prediction-v1-plan.json), [study result](../../experiments/joint-prediction-v1-result.json), and [primary frozen run](../../runs/joint-prediction-v1/result.json), SHA`1f192e92...`.
- Source-informed HUD speed/joint/yaw/wheel decoding, causal body-motion registration, and a small public-physics NumPy model are isolated in `haic/algorithms/joint_control/`. No privileged runtime inputs.
- 42 consumed TRAIN archives/9 roads split 4 FIT, 2 CAL, 3 TEST; repeats/windows stay with their geometry. Speed p95 on TEST is 1.170-1.194 units/s. Valid motion displacement p95 is .343-.348 units; validity 3070/5168 intervals. HUD yaw integral consistency p95 .0189-.0236rad is NOT direct instantaneous-yaw ground truth. Actual joint/omega labels are absent.
- Exactly9 FIT-only effective inertia corrections selected scales4/4/1 at the grid boundary; no expansion or held retuning. The omega-omission ablation was negative. Source-default and all correction evidence remain preserved.
- TEST.32s geometry-equal mean medians: estimated position/yaw/speed-change error`.7289/.1835/1.6164`, versus same-state persistence`.8718/.2663/.7007`. Available-label partial oracle`.6922/.1717/1.5921` does not fully isolate dynamics because instantaneous wheel/terrain states remain missing.
- All3 TEST roads have sufficient support and relative position/yaw signal, but yaw p95`.439-.489rad` exceeds the unchanged`.25rad` gate; median exceeds`.08rad`. That archive protocol ended at gate0/3 with **zero TRAIN resets/steps**. The later distinct relative-effect probe above is not a pass of that gate or a rewrite of its evidence.
- Runtime-valid.32s windows2689/5145; windows overlap. Existing RLPD/Oracle data and FIT speed support up to about43.72 do not validate the faster champion distribution or unseen actions. No lap/safety/shield-inheritance claim.
- [Independent passive audit](../../experiments/joint-prediction-v1-audit.json), SHA`2faf77e5...`, finds no discrepancy:1512 forecast-error arrays/580608 values reconstruct exactly,1296 prediction-summary groups and gate agree. Motion-vector numeric errors remain primary-run/hash-backed, not independently re-executed because per-frame vectors were not saved.
- 45 focused tests pass. Frozen-source full evaluation 30.37s/150,081,536B peak RSS; original champion ZIP/all 11 sources unchanged. No Agent, official evaluation, submission, neural training or Git write.

Keep the observation/prediction components and negative evidence. Do not expand
the inertia grid or loosen the archive gate after TEST. The subsequent separately
scoped probe above supplies instantaneous-state diagnostics without relabeling
the archive failure or missing branch data, and without implementing a controller.

## Champion Micro-Optimization Stopped By User (2026-10-04)

**Stop further local heuristic optimization of the current KOI champion.** The
user's2026-10-04T15:23:01Z decision closes the campaign beyond sprint72-relief:
do not propose or execute another small threshold, margin, recovery, steering,
corner or pedal patch, diagnostic or A/B without a new explicit instruction.
Keep `crossing_projection + collision-shield v1` as the unchanged submission
baseline and retain all negative source/protocol/result evidence. This is a
research-allocation decision, not proof that every heuristic improvement is
impossible; independent model lanes are not automatically changed.

The observed lesson is closed-loop sensitivity: a source-correct local pedal
change can improve a local speed profile yet change later perception, restriction
timing and steering enough to lose a finish. Do not overstate the particular
mechanism. All observed impact triggers remained vetoed, with zero genuine
impact-clear in both arms, so an impact-veto release is not the demonstrated DNF
cause. Also, undershoot6.825->2.218 belongs to the first2/0006 window, whose
brake-to-gas count is1->0. Repeated cycles3->0 is a DIFFERENT, own-window aggregate;
only4/13 baseline station windows support complete matched comparison.

At that decision, keeping the champion and researching a separate joint controller
were alternatives, not implementation authorization. Subsequent2026-10-04
feasibility and2026-10-05 observer/prediction validation were separately authorized
and are recorded above; neither selected or implemented a successor controller.
Privileged evaluation heading/progress is not a runtime input, and shield reuse
alone is not a safety proof. See the
[durable decision](../decisions/INDEX.md#stop-local-optimization-of-the-koi-champion).

## Current Submission Baseline (2026-10-02)

**User-designated baseline: crossing_projection + collision-shield v1.** At
2026-10-02T01:36:47Z the user reported that the current official submission
finished Track4 in **18.4s** and ranked **6th overall** at report time, and
explicitly selected it as the new submission baseline. The user binds this report
to ZIP SHA-256
`c9e376a049805a8669af9c3959e09f3f96565530a5a4feb631977237ed64f801`.
This is a **user-reported official result and user-bound package identity**, not
independent verification: no site submission ID, server receipt or model-confirmation
receipt is available. No new upload or server confirmation is performed here.

On2026-10-04 the user additionally summarized this champion as finishing all four
official tracks and placing sixth overall. This remains a dated user report, not
an independently verified current leaderboard or per-track same-model audit.

Frozen reference artifact:
[`20261002-crossing-projection-collision-shield-v1-baseline/`](../../submissions/20261002-crossing-projection-collision-shield-v1-baseline/),
with the exact ZIP, manifest, all11 source members, standalone stdlib restoration
helper, dependency versions, provenance and checksums. ZIP/member hashes match the
original receipt; isolated exact-copy restoration and source-only ZIP rebuilding
both reproduce the same ZIP SHA-256. These were file operations only: no Agent
execution, simulator reset, performance test or new official action.
The original experimental ZIP/receipt, shield source`ad772bde...`, standalone
crossing ZIP`a4b35c56...`, root Agent and previous candidates are retained unchanged.
The original research **NOT_ADOPTED / failed+20ms efficiency gate remains unchanged**;
this submission designation is separate from research adoption. No retuning or
evaluation is authorized by this documentation update. Historical packaging and
research designations below remain dated evidence, not the current selection.

## Fixed Sprint72 Handback Relief Rejected (2026-10-04)

**REJECTED / NOT ADOPTED; this exact specification is CLOSED without retuning.**
On `research/koi-sprint72-relief`, candidate ZIP `e7062c66...` inserts only the
diagnosed pre-arrival brake law `clip(.02*(v-72),0,.15)` with gas0, exact T60 and
the unchanged spatial/obstacle/intervention guards. Actual issued brakes enter
the original history. Champion `c9e376a0...`, all11 frozen sources and final
shield `ad772bde...` remain unchanged; no root Agent or existing policy edit.

The [frozen protocol](../../runs/koi-sprint72-relief-v1/protocol.json),
SHA`7fca6b24...`, completed16 contemporary natural episodes/eight matched consumed
TRAIN layouts/five geometries, with no omitted mate or retry. The
[result](../../experiments/koi-sprint72-relief-v1-result.json), SHA`b20da6b3...`,
preserves exact pre-divergence state/pixel/raw parity and all48 objects per arm.
Finishes remain7/8 but **one old finish is lost** (1/3184000013) and one gained
(2/3184000006). Damage, collision decisions, contacts and hit objects stay0.
All-wheel-offroad ticks64->366 and longest spell1.14->6.88s regress. Common6
laps average+26.667ms;1/0015+180ms and2/0015+80ms violate the20ms ceiling.

Local undershoot can improve:2/0006's first cap-window minimum65.175->69.782.
However only4/13 baseline station windows remain fully comparable; six meet an
earlier candidate restriction, two reverse passages are unreached, and one end
is unreached. No full-cohort matched undershoot/cycle mean is available. Own-window
repeated brake-gas cycles3->0 and whole-episode transitions215->198 are not valid
complete matched gains; on the common6 finishes transitions135->136 instead.

The four baseline reverse decisions194/195/447/448 remain counted and runtime-
eligible, not hidden by a heading veto. Candidate changes earlier driving and
finishes that cell in21.8s without the old reverse phase; it does not execute
those identical reverse states. This positive rescue cannot offset the lost finish.
Candidate changes39 nominal decisions; later steering and arrival timing/entry
speeds differ despite unchanged code. All232/204 impact triggers are vetoed, with
zero genuine impact-clear in either arm; do not blame an unobserved veto release.

Integrated159 tests+132 subtests and both exact CPU21 zero-reset import preflights
pass. The [independent audit](../../experiments/koi-sprint72-relief-v1-audit.json),
SHA`cfe02a62...`, reproduces the exact frozen result and verifies116 artifacts,
all4094 issued-brake/history decisions, all8 prefixes and all96 object records
without new driving; no discrepancy changes the rejection. Preserve source/run/
result as negative evidence. No coefficient sweep,
deadband/hysteresis/FP fix, repeat, large generalization, protected/private/official
evaluation or submission follows. See [section32](../architecture/koi-baseline-analysis-2026-09-30.md#32-fixed-sprint72-handback-relief-2026-10-04).

## Corner Entry Pre-Positioning Closed At Diagnosis (2026-10-04)

**No repeated feasibility improvement; no candidate or A/B.** This separate
hypothesis moved outside preparation10/20/30 decisions earlier, rather than
reopening direct apex pulling from the recorded entry. The
[natural analysis](../../experiments/koi-corner-entry-natural-v1.json) retains all
62 sharp-corner observations from eight consumed champion episodes/five geometries:
55 complete, one censored, six unreached. Complete geometric entries are6 outside,
15 center and34 inside. Earlier-outside comparisons with both shorter path and
lower maximum front-wheel angle occur on only one geometry, with approach/layout
confounding; they are not causal A/B or replicated evidence.

The [counterfactual diagnosis](../../experiments/koi-corner-preposition-diagnosis-v1.json)
tests288 bounded paths for the prior four rejected opportunities/two geometries,
including gradual setup, genuine outside entry, apex and complete return costs.
Nineteen paths have positive net length savings, but none reaches the2-unit
meaningful-gain screen; the maximum is1.459. Removing demand/overlap vetoes AND
accepting any positive saving leaves only two paths on the same case/geometry,
saving.187/.228. These are offline geometric predictions, not dynamic feasibility
proofs or measured lap gains. Brief offroad remains allowed, not a hard boundary.

Champion ZIP/all11 sources, root Agent, pedals, avoidance and shield remain
unchanged. No new resets, blind/official action, runtime candidate or A/B.
Exit-throttle, target-speed increases, overavoidance and direct-apex cutting stay
closed. See [section31](../architecture/koi-baseline-analysis-2026-09-30.md#31-outside-entry-pre-positioning-diagnosis-2026-10-04).

## Direct-Apex Corner Cutting Closed At The Passive Gate (2026-10-04)

**No candidate or A/B; no repeated feasible-looking gain in the bounded family.**
The [diagnosis](../../experiments/koi-corner-cutting-diagnosis-v1.json) reads only
the same eight frozen-champion consumed TRAIN episodes/five geometries. It retains
62 sharp-corner observations,55 complete;44/55 actual paths are already shorter
than their corresponding centerline arcs. Fifteen complete windows have no recorded
obstacle/recovery interference. Four of those, on two geometries, have meaningful
GEOMETRIC savings, but none passes the combined steering/lateral-demand/reentry
screen at unchanged logged speed. Do not misstate this as zero geometric opportunity.

Road boundaries were SOFT: predicted short offroad was explicitly allowed. The
bounded .75/1.5/2.25/3-unit inward shifts were screened separately from the
unconstrained chord lower bound. Their steering, grass exposure, reentry heading
and time estimates are offline proxies, not replayed trajectories or physical
impossibility proofs. Close this minimal fixed-speed direction without expanding
to a planner or changing pedals. Frozen champion/shield, avoidance and straight
control remain unchanged; exit-throttle, target-speed and overavoidance stay closed.
No new reset, blind/official action or measured lap improvement. See
[section30](../architecture/koi-baseline-analysis-2026-09-30.md#30-soft-boundary-corner-cutting-diagnosis-2026-10-04).

## Corner Target-Speed Headroom Not Established (2026-10-04)

**Direction CLOSED at the passive diagnosis gate; no candidate or A/B.** On
`research/koi-corner-target-speed`, the [diagnosis](../../experiments/koi-corner-target-diagnosis-v1.json)
uses only the existing eight frozen-champion TRAIN episodes/five roads/2,130
decisions. The active target is a road-center pixel-spread mapping, not calibrated
physical curvature. Its issued pedal formula matches832 non-overridden decisions.

Geometry-defined corner/followup windows retain68 complete, one censored/nonforward
and nine unreached observations. Of23 unconfounded complete passages,12 enter above
their HUD-scale target and remain contact-loss/damage/collision-free, but only one
passage on one road supplies sustained same-spread-band headroom under the stated
diagnostic screen. No mapping band has repeated support across distinct roads.
One other unconfounded corner already has17 partial-wheel-offroad raw ticks,
including two all-wheel-offroad ticks. Local projected margin is a proxy, not
exact curved-road clearance or proof of a physical speed limit.

The result does not establish excessive preview braking or a safe narrow mapping
increase. No target, lookahead, steering, avoidance or shield change; frozen ZIP,
all11 sources and worktree shield verify unchanged. Exit-throttle and overavoidance
remain closed. No new resets, blind/official evaluation, or measured A/B lap delta.
See [section29](../architecture/koi-baseline-analysis-2026-09-30.md#29-corner-target-speed-headroom-diagnosis-2026-10-04).

## Corner-Exit Throttle Hypothesis Not Confirmed (2026-10-04)

**Direction CLOSED at the diagnosis gate; no candidate or A/B run.** Passive
[diagnosis](../../experiments/koi-corner-exit-diagnosis-v1.json) covers eight existing
frozen-shield consumed TRAIN episodes/five roads/2,130 decisions, not fresh data.
Of36 putative post-turn alignment onsets,30 already issue gas>=.5; four of the
remaining six exceed their current target. All six fail the existing projected
free-space gate and have19.62-70.38deg of upcoming road heading change within28.8m.
These are not established unnecessary post-exit holds. Active targets match the
current-frame road-spread formula exactly on1,601 eligible decisions;832 eligible
non-sprint pedal pairs match the current-speed formula within2.8e-8. The corridor
EMA's pedals are overwritten, and no active corner-exit holding timer was found.

No acceleration/entry/mid-corner/avoidance/shield logic changed. Exact frozen ZIP,
all11 members and worktree v1 shield hashes match; overavoidance remains CLOSED.
Zero new simulator resets, generalization or official actions. A/B safety, exit
time and lap deltas are unmeasured, not zero. See
[section28](../architecture/koi-baseline-analysis-2026-09-30.md#28-corner-exit-throttle-diagnosis-2026-10-04).

## Bounded Magnitude Rejected; Overavoidance Closed (2026-10-02)

The user authorized one isolated reaction-based replacement of nominal `.34/.55`
avoidance with current-observation clearance/risk-dependent bounded magnitude,
followed by the exact unchanged submitted v1 shield. The adapter owns only the
original nominal and last diagnostics: no plan, action feedback, recovery/release
timer, margin reduction, steering-release or persistent trajectory controller.

The [four-pair frozen study](../../runs/koi-avoidance-magnitude-v1/protocol.json),
SHA`1456126e...`, completed all8 contemporary episodes on four consumed ordinary
TRAIN layouts/two roads. The [primary result](../../experiments/koi-avoidance-magnitude-v1-result.json),
SHA`55d489f9...`, is **REJECTED / NOT ADOPTED**: finishes4/4->3/4, lost1/gained0,
damage0->1.4, collision-positive decisions0->7, physical contact events0->4 and
hit objects0->3, retaining all24 objects/arm. New hits are1/0013 object4,
1/0015 object1 and3/0015 object1;1/0015 loses its finish. All four pairs match
geometry/initial state/pixels and exact raw/decision prefixes before divergence.

Only19/24 baseline windows and19/24 return followups remain comparable. Apparent
three-retained-cell means (lateral-48.188963%, path-0.526375%, issued steering
integral-12.230624%) exclude the failed cell and are NOT valid four-cell efficiency
evidence. On road0013, path+2.813968% and steering integral+14.947015% worsen
despite lower lateral deviation; lap+80ms also violates the20ms ceiling. Kept3
lap mean-120ms omits the lost finish. Common19 return-censor count5->1 and
descriptive bound13.131535->9.258961s do not repair five missing followups.

Nominal98/820 candidate decisions changed; unchanged shield interventions9->25
do not establish safety preservation. Integrated zero-reset validation277 tests
+133 subtests PASS; both exact CPU21 arms pass zero-reset import preflight.
Source/environment/model pins and root/champion/shield/prior candidates rehash
unchanged. No fresh/protected/Track4 geometry, official action or baseline edit.

**Discard this candidate and close overavoidance optimization under the user's
stop criterion.** Preserve frozen source/run/result as negative evidence; no
retuning, repeat, new controller or automatic next experiment. This is rejection
of this consumed-TRAIN candidate, not a proof that every possible avoidance
controller is ineffective. Details and fixed gates are in
[section27](../architecture/koi-baseline-analysis-2026-09-30.md#27-stateless-bounded-avoidance-magnitude-2026-10-02).

The [independent primary audit](../../experiments/koi-avoidance-magnitude-v1-audit.json),
SHA`7bc1297f...`, verifies all63 study-root files and frozen inventories, exactly
reproduces the result and17 gates (7 pass/10 fail), and independently confirms
safety,19/24 coverage and invalid four-cell efficiency denominators. No artifact
discrepancy, Agent/policy execution, environment reset or driving replay.

## Nominal Trajectory Candidate Rejected (2026-10-02)

The user authorized isolated nominal avoidance trajectory/side selection and
matched A/B on existing consumed TRAIN ordinary-obstacle cases. The new bilateral
approach/pass/rejoin generator, complete lagged pursuit projection and per-frame
remaining-plan validation precede the same byte-identical v1 shield in both arms.
No margin reduction, steering-release or recovery-state tuning was reopened.

The [frozen eight-pair study](../../runs/koi-nominal-trajectory-v1/protocol.json),
SHA`77bda836...`, completed all16 episodes on five consumed roads, including
ordinary four layouts/two roads. The
[result](../../experiments/koi-nominal-trajectory-v1-result.json), SHA`b020c3cb...`,
is **REJECTED / NOT ADOPTED**: finishes7/8->7/8, damage/collision decisions/hit
objects0->0, all38 baseline windows and43 common return followups preserved, but
ordinary max lateral+0.068713%, path only-0.014782%, issued steering integral
+4.193204% and steering variation+4.125609%. Common-return bound worsens; kept7
lap mean+20ms and3/3184000015+140ms fail efficiency. Road0013 remains exact no-op,
so no improvement on both ordinary geometries is established.

Discard this efficiency-regressing candidate from adoption and further repeats;
preserve its source/run/result as negative evidence. The exact submission baseline
and collision-shield v1 remain immutable and rehash unchanged. No root Agent,
fresh/protected/Track4 geometry, upload, model confirmation or official action.
The existing KOI analysis
[section26](../architecture/koi-baseline-analysis-2026-09-30.md#26-separate-nominal-trajectory-selection-2026-10-02)
records the source diagnosis, fixed gates and outcome. Fixed-demand steering and
centroid-only flank selection are observed source deficiencies; the new model
planner's cost is not evidence of shorter real closed-loop driving.

The [independent primary audit](../../experiments/koi-nominal-trajectory-v1-audit.json)
rehashes all64 primary files and frozen inventories, reproduces the result and all15
gates, and confirms only4/2132 nominal actions changed. Selected complete rejoin
plans actually abort after one/two holds; fallback transitions and later inherited
steering costs were not included in the original trajectory-bank cost. No artifact
discrepancy or reason for a driving repeat was found.

## Minimum-Intervention Collision Shield (2026-10-01)

**Historical user packaging designation (15:45 UTC):** keep the v1 research result
**NOT_ADOPTED** and its failed+20ms efficiency gate unchanged, but designate the
exact frozen v1 separately as an **experimental submission candidate** for imminent
obstacle-collision DNF prevention. Ordinary-overavoidance reduction is a separate
follow-up, not part of this candidate. No tuning, added recovery, or extra driving
evaluation. Local package
[`koi-collision-shield-v1-experimental-submission.zip`](../../submissions/koi-collision-shield-v1-experimental-submission.zip),
SHA`c9e376a0...`, preserves shield source`ad772bde...` and all crossing dependencies
byte-for-byte. The [receipt](../../submissions/koi-collision-shield-v1-experimental-submission.receipt.json)
records static/ZIP checks and64 exact synthetic CPU21 action/diagnostic comparisons,
zero simulator resets and unchanged prior ZIPs/root/v2/result. This is a packaging
designation, not a baseline replacement, research-gate pass, upload or confirmation.

**Current user direction:** close long-lived collision_recovery development as
FAILED / NOT ADOPTED; preserve its evidence and keep crossing_projection fixed.
Implement a separate pixel-only `CollisionShieldAgent`, not recovery v2, and test
only the same six already-consumed TRAIN cells (five road seeds), with ordinary
controls and corner/obstacle challenges. No Track4 geometry, protected/fresh cells,
root Agent change, steering-release change or official action.

The independent runtime uses a .32s lagged footprint projection: candidate steering
for only the actual .08s hold, then assumed baseline continuation. Only a predicted
baseline collision triggers the collision-free finite steering grid; minimize
action deviation plus a soft observed-road penalty, with unchanged pedals.
Immediate handback on clear/unknown/infeasible prediction; at most six changed
actions per encounter. Three observed threat-associated clear decisions rearm;
unresolved threat dropout disables rearming until reset, never retains control.
No heading recovery or reward-budget inference. Pixel projection is not a safety
guarantee, and already collision-free baseline overavoidance remains untouched.

The [frozen six-pair A/B](../../runs/koi-collision-shield-v1/protocol.json),
SHA`afeeb356...`, completed all12 episodes with exact logged-state/pixel/raw prefixes.
The [result](../../experiments/koi-collision-shield-v1-result.json) shows an important
consumed-TRAIN safety signal: finishes3/6->5/6, kept3/lost0/gained2, damage1.4->0,
collision-positive decisions7->0 and hit objects2->0, with no new hit. Only17/1639
candidate decisions changed in16 bursts; longest burst2 actions/.16s, largest encounter5
changed actions. Physical minimum clearance-.012577->.990591m; longest full-offroad
duration stays1.14s. Aggregate full-offroad occupancy1.20->1.28s.

**NOT ADOPTED under the frozen gate, not a rejection of the safety signal.**
The preserved3/3184000002 lap slows17.68->17.86s (+180ms), exceeding the predeclared
per-kept-lap+20ms limit; all other gates pass. The three kept laps average+40ms and
path+.714544m. Two ordinary controls retain2/2 finishes and no hits, but lateral
integral64.843274->65.955569m*s and path+0.582198m across the two do NOT demonstrate
less overavoidance. One control is an exact no-op; the other changes4 decisions,
finishes60ms faster but has higher max lateral6.691664->6.901651m. Remaining
2/3184000006 noncollision departure begins within the identical115-decision prefix,
before its first shield intervention. Do not infer a new heading-recovery remedy.

No gate relaxation, policy tuning, repeat, new cell or promotion follows this result.
Runtime/evaluator/analysis and preserved-regression checks:51 tests+31 subtests PASS;
both frozen arms pass zero-reset CPU21 preflight. The
[independent primary audit](../../experiments/koi-collision-shield-v1-audit.json)
rehashes all48 episode/raw/decision/process files and205 frozen inventory pins,
independently reproduces the metrics and all9 gate outcomes, and finds no artifact
discrepancy. Encounter bookkeeping may persist across baseline actions; it is not
continuous intervention. All six episodes end with conservative rearm blocked.
See
[implementation and outcome](../architecture/koi-baseline-analysis-2026-09-30.md#25-minimum-intervention-collision-shield).

## Crossing-Based Collision Recovery (2026-10-01)

**Current user direction:** crossing_projection is the KOI comparison baseline.
Preserve steering-release v2 and its submission derivative unchanged; do not
extend near_release or infer adoption from its reported public-track finishes.
The user authorized focused consumed TRAIN development, not official evaluation.

**Closed by the newer user direction above; no further state-machine extension.**
The separate pixel-only `CollisionRecoveryAgent` directly wraps frozen crossing
ZIP `a4b35c56...`, with emergency collision priority, soft road boundaries,
explicit heading alignment, nearest-edge re-entry and stable handback. It has no
steering-release dependency. The [six-pair result](../../experiments/koi-collision-recovery-v1-result.json)
is **NOT ADOPTED**: baseline3/6 versus candidate2/6 finishes, one gained and two
lost, one new hit on a baseline-clean object. Damage1.4->1.0 and collision-positive
decisions7->5 do not offset this. Longest all-wheels-offroad duration1.14->7.96s;
three candidate departures never regain road contact. The only mutually finished
lap slows19.96->23.84s. No repeat is warranted as a claimed improvement.

This follows a separate rejected v2-based priority-only [initial test](../../experiments/koi-collision-priority-v1-result.json)
(4/6->3/6 finishes), preserved with its frozen source. Both are outcome-selected,
reused TRAIN comparisons, not fresh generalization. Root Agent, crossing and v2
sources/ZIPs remain unchanged. The [recovery diagnosis](../../experiments/koi-collision-recovery-diagnosis-v1.json)
distinguishes collision stall, backward on-road travel and unrecovered departure;
physical road reacquisition alone is not heading-safe recovery.

Official Participants `dfb7a2d...` confirms retirement on101 consecutive decisions
whose summed raw reward is negative (`counter > 100`), NOT101 physics ticks or a
wheel-contact predicate. Agent receives no reward/counter; runtime recovery timers
are explicitly proxies. The passive evaluator records the exact decision counter
and physical contact loss separately. See KOI architecture section24 for details.

## Frozen v2 Generalization Rejection (2026-10-01)

**Current user direction: stop all further work; documentation only.** User visual
observation: ordinary obstacles still provoke excessive lateral avoidance in v2,
occasionally enough to leave the road. It is reduced versus before but remains
visibly excessive. Margin reduction and steering-release have not resolved this
problem and are currently on hold. Future reconsideration belongs at the
**avoidance trajectory / side-selection** stage, not further release/margin tuning.

**User-directed freeze:** steering-release v2 ZIP
`b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce`,
its runtime/parameters/manifest, crossing_projection source-reconstructed baseline
and root Agent remain unchanged. The old24 TRAIN conditions (tracks1/2/3 x
38300-38303 and50300-50303) are closed to further tuning or evaluation.

The completed attempt was authorized as one separately frozen **nonprotected
TRAIN/dev generalization A/B**, not protected confirmation, blind or official evaluation.
The [zero-reset exposure audit](../../experiments/koi-steering-generalization-v1-exposure.json)
cleared and claimed24 new road seeds3184000001-3184000024 across tracks1/2/3;
these are72 obstacle-layout pairs, not72 independent roads. Its freshness scope is
candidate-lineage-unseen, not project-global never-used:15 unrelated legacy PPO
sampling uncertainties remain disclosed. No protected observations were read.
Separate audit/evaluate/analyze adapters preserve every old source/result artifact.
Predeclare coverage, finish/safety, lateral/avoidance/path/return/lap and geometry
consistency gates. The completed matched prefix now supplies irreversible
predeclared safety counterexamples: **NOT A GENERALIZED ADOPTION CANDIDATE**.
A future correction must be a separate candidate, not an in-placev2 update or
automatic next run. No correction or further environment interaction is made.

**Original execution incomplete, negative evidence decisive:** the [frozen protocol](../../experiments/koi-steering-generalization-v1.json)
SHA`ccc6720f...`, [independent review](../../experiments/koi-steering-generalization-v1-preflight-review.json)
SHA`67157ff6...` and [mandatory forensic guard](../../experiments/koi-steering-generalization-v1-evidence-guard.json)
SHA`74597759...` passed zero-reset validation. New-study regression checks passed
135 tests plus20 subtests. Serial CPU21 A/B ended naturally with `child_timeout`;
the [guarded result](../../experiments/koi-steering-generalization-v1-result.json)
SHA`2572e09dfc8dd4a1399a1168ea15692b6be946cbb20a72b18d0532ea00d5843d`
preserves95 valid completed episodes, one queued/bootstrap timeout and48 unrun
slots from144 planned. Outputs/reset ledger remain under
`runs/koi-steering-generalization-v1/`. Do not edit pinned sources, restart or
relabel partial exposure. Only guarded finalization is authoritative; the raw loader's
failure-time partial-pin omission remains explicitly documented and tested.

Freshness auditing adds wall cost beyond the historical resource reference. The
fixed4584.848s budget was not changed. The final slot3/3184000016/v2 timed out in
2.123655s without episode/raw/decision files; its reset intent/process/logs remain
conservatively recorded, not fresh. The reviewed boundary manager refused before
any lock/signal because the operator had already finalized. Prepared continuation
and composite code was never frozen or run; no successful boundary receipt exists.
The144-slot/24-road plan is **not complete** and cannot be relabeled passed.

The [matched diagnosis](../../experiments/koi-steering-generalization-v1-diagnosis-result.json)
excludes the unmatched baseline episode: **47 matched pairs on16 new roads**, each
with identical geometry/initial state+pixels/first10 actual actions. B41/47 versus
C40/47 finishes; kept36/lost5/gained4/neither2. All five losses are valid natural
episodes (four off-track, one crash), distinct from the operational tail. They span
four geometries:3/3184000004,1/3184000005,1+3/3184000013,2/3184000014. Two new
baseline-clean hits are objects2 and4 in the first two cells. Appending unexecuted
cells cannot erase these strict lost-finish/no-new-hit counterexamples, so no
additionalv2 evaluation/partial retry is needed to reject generalized promotion.

Matched damage4.4->3.4, collision-positive decisions22->17 and hit objects5->4
(282 objects per arm) improve in aggregate but hide two per-cell regressions and
two new hits. Whole-episode centerline lateral proxy max29.9527->260.0025m.
On173 preserved windows of190 baseline-eligible windows, conditional equal-cell
lateral-10.4390%, avoidance integral-15.9312%, duration-3.5129%, path+0.0378% and
actual steer integral-2.7570%; the missing17 windows invalidate survivor-based
adoption. All12 eligible measured geometries have negative lateral/integral means,
but this does not establish the safety/completion chain. Common return censors
increase37->43 on235 followups;20 of255 baseline followups are lost. Kept36 laps
19.091111->19.037222s (-53.889ms) omit five baseline finishes and cannot offset them.

[Independent postrun audit](../../experiments/koi-steering-generalization-v1-postrun-audit.json)
SHA`c4bd5b11...` exactly reproduces the guarded result and validates every source/
raw/stream/process/ledger/model binding. [Five-failure audit](../../experiments/koi-steering-generalization-v1-failure-audit-result.json)
SHA`55897acf...` confirms exact action/physical/raw prefix and divergent PRE-state
equality: first changes are near_release steps43/41/70/93/33, with no generation
suppression in any loss. All17 release-held decisions and selected-object passages
remain sampled collision-free; contacts are DIFFERENT later unreleased objects,
and three failures instead lose road/heading recovery after passage. Do not infer
immediate released-object contact or blame the dormant generation correction.

The local selected-object lag proxy has no road/heading recovery or next-object
safety condition. Trajectory/perception-feedback mediation is a plausible but
untested mechanism, not a proven unique cause. The former release road-recovery
H1 is deferred with the steering-release approach. Future reconsideration is at
avoidance trajectory/side-selection; no current implementation, tuning, checks or
evaluation. Detailed observation/
hypothesis and geometry tables are in KOI architecture section23. `off_track` is
a reward-streak/simulator-termination label, not a wheel-contact predicate.

Postrun audit publication conservatively makes the frozen broad root-metadata
scanner report `HOLD` for its consumed outcome identities. Preserve it unchanged;
no exclusion/waiver/freshness relabel is added and no further reset is attempted.
This does not invalidate the already verified negative counterexamples. The
planned24-road matrix remains incomplete; no population/official generalization
claim is made, but strict generalized adoption is conclusively rejected.

The consumed-r2 section below is historical supporting evidence, not permission
to reuse its24 conditions. Its former future-confirmation wording is superseded
only by this user-authorized TRAIN generalization scope.

## KOI Steering Generation/Hold/Release (2026-10-01)

**User-directed closure:** all minimum-clearance variants are FAILED / NOT
ADOPTED. Both speed-target adjustment and safety-margin reduction directions are
closed; do not reopen them as steering-release tuning. Preserve prior source,
ZIP, protocols, negative results and audits. Crossing_projection remains the
fixed source-reconstructed KOI baseline; no root Agent or official-model change.

**Final r2: INTERNAL ADOPTION CANDIDATE, not a baseline replacement or official
model.** The [completed result](../../experiments/koi-steering-release-ab-r2-result.json)
and [frozen protocol](../../runs/koi-steering-release-ab-20261001-r2/protocol.json)
cover all48 slots on the same24 consumed TRAIN layouts (tracks1/2/3 x
38300-38303 and50300-50303; eight geometry seeds). All17 gates pass: B21/C24
finishes, kept21/lost0/gained3, damage1.4->0, collision-positive decisions7->0,
whole-episode hit objects2->0 and no new hit. All119 baseline fixed windows across
21 cells and all136 comparable return followups are retained.

Primary equal-cell relative changes are max lateral-12.544446%, avoidance
duration-3.312396%, avoidance integral-17.805737%, path-0.169631% and actual steer
integral-5.367767%. Duration does NOT meet5%; the avoidance gate passes through
integral, with lateral/integral each qualifying on seven geometry seeds. The38300
reset-seam cells are excluded from these window means, not from finish/safety or
the matched21 laps:19.054285714->19.016190476s (-38.095238ms). Candidate's own24-lap
mean is unmatched. Full144-object prospective return statuses are B113 returned/
23 next-entry censored/6 unpassed/2 invalid versus C125 returned/17 next-entry
censored/1 fixed-time censored/1 invalid. On the common136 followups, censors
fall23->19; onset-delay/censored-bound mixture0.726180453->0.666090505s is
descriptive, not an unbiased144-object return mean, KM or RMST estimate.

The evidenced mechanism is avoidance steering generation/hold/release, not
clearance magnitude. The
[source-exact diagnosis](../../experiments/koi-steering-release-baseline-diagnosis-v1.json)
reconstructs all5703 decisions/22779 ticks with zero issued/sum residual; it retains
all144 objects and finds36 same-object projection-off/laterally-separated/outward
avoidance cases. Old transverse fixture extents are unarchived: conservative yaw
bounds plus observed exact fixture clearance are not counterfactual safety proof;
new observer transverse intervals are measured separately. Missing projection or
detector dropout is not safe clearance/rear-clear. The
[first48-slot A/B](../../runs/koi-steering-release-ab-20261001-v1/protocol.json)
completed: its [result](../../experiments/koi-steering-release-ab-v1-result.json)
and [independent audit](../../experiments/koi-steering-release-ab-v1-audit.json)
find23 candidate finishes vs21, but kept20/lost1/gained3 and two new hit objects,
so NOT ADOPTED. Conditional117-window lateral/integral gains cannot offset two
missing baseline windows or lost return coverage. Both fail-cell diagnoses find
later inherited subpixel-motion flank flips, not contacts during released holds.
Explicitv2 ZIPb1911d7d.../source52f54099.../term helpercb1149b0... suppresses ONLY
a same-component motion-only flip when the current offset sign still favors the
selected flank; all unambiguous/sign-crossing choices
and coefficients/targets/margins remain unchanged. DefaultFalse preserves oldv1;
all9 frozen baseline dependencies remain byte-identical, including original6px
band and all margins/targets. Recorded final preflight:525 tests+28subtests and
actual CPU21 source smoke pass. The small correction was actually reevaluated
over the full r2, not inferred from the two diagnosed failures alone.
The zero-reset publisher-schema failure and v1a identical-policy fix are preserved.
The [final independent raw audit](../../experiments/koi-steering-release-ab-r2-audit.json)
rehashes all48 episode/raw/process sets,158 frozen copies,1046 evidence pins,
both model inventories and96 ledger rows, and independently reproduces all17 PASS
gates with zero formula/return mismatches. Its source/post-generation distinction
and all144-object safety/136-return denominators are preserved. Both-returned109
conditional onset means are0.724220183->0.646972477s, not a136/144-object mean.
CPU21 deterministic rebuild reproduces the same final ZIPb1911d7d....
See the [final analysis](../architecture/koi-baseline-analysis-2026-09-30.md#22-final-r2-internal-adoption-candidate)
and [evidence index](../experiments/INDEX.md#koi-steering-lifecycle-2026-10-01).
The successful steering hypothesis is not closed for lack of improvement, but
consumed TRAIN evidence does not establish fresh/generalized performance. The
only next gate is separately user-authorized future confirmation with provenance
and exposure checks; no evaluation is queued automatically and no fresh cells are
promised. Crossing_projection stays FIXED and root Agent remains unchanged.

## KOI Minimum-Clearance Result (Closed, 2026-09-30)

**User-directed closure:** adaptive-v1 is FAILED / NOT ADOPTED and its
speed-target hypothesis is closed. Preserve its frozen implementation, ZIP and
negative A/B evidence; do not pursue another adaptive speed-target variant.
`ContactContinuityAgent('crossing_projection')` is the fixed KOI improvement
baseline, reconstructed from immutable `c4e224d` sources, not restoration of the
missing original crossing ZIP.

**Minimum-clearance candidates NOT ADOPTED:** all three separate48-slot A/B
comparisons completed on only the same24 consumed TRAIN layouts, tracks1/2/3 x
38300-38303 and50300-50303 (eight geometry seeds). The isolated steering candidate
uses detected obstacle bounds, skin-inclusive full hull/wheel footprint, supported
asphalt and minimum lateral target, not a steering multiplier. Baseline speed
targets/pedal calculation and all frozen dependencies are unchanged.

2026-10-01 user direction closes all variants as failed and ends safety-margin
reduction; the following is preserved historical evidence, not a next tuning gate.

[v1 result](../../experiments/koi-minimum-clearance-ab-v1-result.json) preserves21
finishes and damage1.4/collision7 but changes only5 decisions, with+0.038408%
cell-weighted path and+8.571429ms kept-lap mean change. A diagnosed full-command
versus actual.08s action-hold mismatch was directly corrected in separatev2;
[r2 result](../../experiments/koi-minimum-clearance-ab-r2-result.json) loses one
baseline finish and adds a new obstacle hit, so fails safety regardless of its
surviving-window path decrease. Original results/ZIPs/source copies are preserved.

The geometry-derived current-projection clearance correctionv3, ZIP
`594fca15005ec38cc67801824e9725fc84da440c2309083f02a24955b0ab8fca`,
requires a supported target lane, safe applied.08s command and projected full
footprint separation on the intended flank. The [final r3 result](../../experiments/koi-minimum-clearance-ab-r3-result.json)
restores21/24 finishes, kept21/lost0/gained0, per-cell damage/collision unchanged,
no new hit and all119 baseline windows. Cell-weighted path reduction is only
0.138079%, below the predeclared2%/two-geometry gate; matched21-lap mean is
2.857143ms slower. Max lateral grows9.334206->9.608379 units and steering metrics
do not improve. Final335 tests+28 subtests and independent source reviews pass.
The [final primary audit](../../experiments/koi-minimum-clearance-ab-r3-audit.json)
rehashes all48 episode/raw pairs,150 source copies and both model inventories,
independently verifies every gate, and confirms3/38301's restored221 decisions/
881 raw records exactly match the recorded baseline. Completed138-passage minimum
clearance falls1.371783->0.662154 units; all144-object safety still preserves the
two preexisting hit objects. No real return_centerline calls occurred. Of144
return rows,118 per arm are window-exit censored; only one both-returned pair
has0.24->0.28s, not a cohort improvement.
No meaningful shorter/safe route or early-return improvement is established;
crossing_projection stays fixed, not replaced by any candidate. All three frozen
results, negative evidence and audits are routed from the experiment index.
No fresh/protected/official action, new-road claim, root Agent change or Git write.
See the [active KOI analysis](../architecture/koi-baseline-analysis-2026-09-30.md#15-minimum-clearance-가설과-adaptive-v1-종료).

## KOI Adaptive Avoidance A/B (Closed, 2026-09-30)

The user-directed first unchanged comparison completed all 48 episodes on only
the already-consumed TRAIN cohorts: tracks1/2/3 x38300-38303 and50300-50303.
There are 24 matched layout cells but only eight road geometry seeds. Both the
frozen crossing source reconstruction and adaptive-v1 ZIP finished21/24, with
kept21/lost0/gained0, total damage1.4 and seven collision-positive decisions each.
No operational error, censor or newly hit baseline-clean obstacle was found.
This does not reproduce the missing original crossing ZIP or historical report.

**Adaptive-v1 is not adopted:** mutually finished mean lap time increased
19.05429->19.07905s. The 120 valid common-entry obstacle pairs across21 cells
averaged0.649009->0.649890s; the cell-weighted mean change was+0.000839s, not a
reduction and below20ms raw timing resolution. Only7/5710 candidate decisions
changed pedals. One clean full physical passage had actual minimum44.162,
but the predeclared valid-common-entry/two-geometry speed gate was not met;
the seam-ambiguous38300 cohort remains excluded from common-entry statistics.
Do not confuse a target/peak above44 with sustained replicated passage speed.
Models/ZIPs remain unchanged; no tuning, second run, fresh/protected evaluation
or official action occurred. See the [frozen result](../../experiments/koi-adaptive-ab-v1-result.json)
and [measurement plan and interpretation](../architecture/koi-baseline-analysis-2026-09-30.md#13-사용자-지정-consumed-train-ab-검증).
The [independent primary audit](../../experiments/koi-adaptive-ab-v1-audit.json)
verifies all48 episode/raw hashes and distinguishes138 completed-passage speeds
(mean of segment means42.94373->43.00255) from the140-encounter summary containing
two incomplete encounters. Its one real above44 case has unchanged passage time.
The designated KOI improvement baseline remains crossing_projection; no other
research line's status or official-model designation is changed by this result.

## RLPD Coupled Recovery In Progress (2026-09-29)

The user authorized and resumed TRAIN-only curve-entry overspeed / steering-speed
coupling recovery validation after a Kilo-only restart. This authorization
supersedes the historical migration pause below for this RLPD iteration only.
Current collection uses only consumed G0 track-1 geometries 4272000001-4272000012;
confirmation/blind/private cells and official actions remain out of scope.

The [closed-loop validation result](../../experiments/rlpd-recovery-validation-v1-result.json)
now covers 42 completed branches, 18,411 decisions and 43 reset intents including
one externally killed duplicate attempt with unknown extra cost. The
[separate continuation](../../experiments/rlpd-coupled-recovery-r2.json) reused
28 immutable traces and collected only the missing 14. Full steering/pedal
Oracle feedback for 12/25 decisions preceded original-actor continuation to real
episode end, with lateral error, speed, damage and progress followed for at least
five seconds after handoff. Historical archive divergence was not repaired.

At 12 decisions, 2/10 failures became finishes but 2/4 finished-parent controls
were harmed; at 25 decisions, 3/10 became finishes and 2/4 controls were harmed.
Only one and three rescues respectively also passed local qualification. This
supports state-specific recovery data, not unconditional Oracle repair.

Partial primary receipts already show that an unchanged actor can pass the
five-second local condition and fail later. Learning therefore requires actual
paired full-finish rescue plus local qualification, with preserved-finish harm
controls. The data window includes the executed intervention and first 63 actor
decisions after handoff, not unexecuted Oracle proposals. Strict preparation
produced 665 accepted rows (653 unique), including 339 unique failure-support
rows across three geometries. Matched 8,192-decision raw-SAC fine-tuning of two
V5-seed50 learner-state copies completed under the
[frozen learning protocol](../../experiments/rlpd-recovery-learning-v1.json):
32online/32prior control versus 32online/16prior/16recovery. Contemporary
pixel-only finish/curve-entry evaluation completed 36 uncensored episodes in a
corrected r2 after a summary-field error stopped the retained first attempt.
Finishes are V5 5/12, matched control 4/12 and recovery replay 1/12. Recovery lost
all five original finishes and gained one different road; its prospective
curve-associated terminal failures are 1/12 versus source 1/12 and control 0/12.
The Q-only replay intervention is rejected. See the
[verified full evaluation](../../experiments/rlpd-recovery-evaluation-v1-result.json).
All three current
critics still prefer their own actor on the 87 Oracle failure-support images;
this is a ranking diagnostic, not evidence that the teacher is optimal or that
driving performance failed. See the
[action comparison](../../experiments/rlpd-recovery-action-comparison-v1-result.json).
The next isolated iteration adds joint native deterministic-mean guidance only
on proven failure Oracle rows, plus frozen-original-mean retention on prior and
handoff/preserved-control rows. It keeps raw SAC/environment reward unchanged
and reports extra actor optimizer work separately. It completed 8,192 decisions,
8,191 SAC updates and 8,191 extra actor updates under the
[guided child protocol](../../experiments/rlpd-recovery-guided-learning-v1.json)
after independent review and actual-data zero-reset preflight; 164 tests and
33 subtests passed. The [actual-visit regression review](../../experiments/rlpd-recovery-regression-review-v1-result.json)
shows changed heading/steering visits and lost handoff retention, not a uniform
saturation shift or a blanket speed/brake-only mechanism. The
[guided result checkpoint](../../experiments/rlpd-recovery-guided-v1-result.json)
records successful export/hash checks and markedly lower joint error on the
87 selected failure Oracle images. The full comparison nevertheless finished
original 5/12 versus guided 2/12, losing four original finishes and preserving
only one. The sparse curve-terminal proxy fell 1 to 0 while mean progress fell
0.6944 to 0.4804, so the guided intervention is rejected too. Action matching is
not closed-loop improvement. See the [verified guided evaluation](../../experiments/rlpd-recovery-guided-evaluation-v1-result.json).
The third isolated iteration freezes the original encoder/critics/temperature
and fits only actor correction with explicit initial/successful-path protection.
It completed 2,048 offline actor updates with all declared frozen components
bitwise unchanged, then finished 24 uncensored evaluation episodes. Completion
is tied at 5/12, but four original finishes were lost and four different roads
gained; damage and curve-terminal counts increased. It fails the preservation/
improvement gate. See the [actor-only outcome](../../experiments/rlpd-recovery-actor-only-evaluation-v1-result.json).
The fourth hypothesis uses the original actor as exact default when inactive
and a pixel-feature local-support gate to invoke the learned joint correction
for a held 12-decision sequence. Protected calibration prevents fresh triggers
on its sampled reference points, not harm guarantees during active holds or
whole episodes. The gate is now built with 87/87 covered prototypes and no fresh
trigger on 16,023 reference queries; inactive actions were source-exact. Its
[frozen protocol](../../experiments/rlpd-local-recovery-gate-v1.json) and independent
review passed, and the full suite passed 225 tests plus 37 subtests. Original-
versus-gated evaluation completed 24 uncensored episodes: original5/12 versus
gated6/12, all five old finishes retained and geometry4272000010 gained. Mean
damage0.2->0.05 and progress0.6944->0.7566 improve; sparse curve-terminal counts
remain1/1. This is the first retained consumed-TRAIN improvement, not fresh
generalization. The [unchanged-policy repeat](../../experiments/rlpd-local-recovery-gate-evaluation-repeat-v1-result.json)
reproduced the same counts and aggregates. The [primary trigger-path review](../../experiments/rlpd-local-recovery-gate-trajectory-review-v1-result.json)
verified only60/4896 decisions used correction: two12-decision holds rescue
geometry10, then383 source-only decisions finish. At5.04s after its first
trigger, heading is -0.049rad, speed21.99m/s, lateral2.28m, damage0, progress0.2465
versus source heading2.937rad/progress0.1549. All five preserved finishes had no
gate activity and exact trajectories. This closes the internal recovery-data
validation gate, not a fresh-road/independent-training or official-model gate.
See the [verified local-gate outcome](../../experiments/rlpd-local-recovery-gate-evaluation-v1-result.json).
No geometry/pose/Oracle inputs
are allowed at runtime and no fresh generalization is claimed.
There is
no fresh-road improvement or candidate promotion. See the
[active RLPD iteration](../plans/rlpd-completion-first-research-2026-09-26.md).

The [zero-reset precursor review](../../experiments/rlpd-recovery-precursor-review-v1-result.json)
confirms heading/lateral thresholds occur in successful controls too. Seed52's
archived traces lack curvature and same-state Oracle actions, so overspeed and
steering opposition are unassessed rather than inferred from old actors.
The [foundation checks](../../experiments/rlpd-recovery-foundation-check-v1-result.json)
establish synthetic code/checkpoint feasibility only, not driving performance.

Last refreshed: 2026-10-02 for rejected stateless bounded magnitude and user-directed
overavoidance closure, preserving the exact submission baseline. This is the
current-state source of truth, not an experiment changelog. Evidence and
historical decisions are linked below.

**Instance-migration freeze (2026-09-27 05:10 UTC):** The user stopped new
experiments and requested safe Vast.ai shutdown. The last live DrQ learner was
interrupted after its step-16,384 checkpoint; its partial ledger ends at step
21,037 and is not an exact resumable result. TD-MPC2 v1 remains partial and v2
has zero resets. No training, evaluation, confirmation/blind or official action
should be launched during migration. See the detailed
[instance handoff](instance-migration-handoff-2026-09-27.md), including the
Git-external transfer inventory and deletion gate. A Git push alone is not
proof this instance can be removed.

**New-instance environment check (2026-09-28):** The previous training receipts
record RTX 5070 Ti (compute capability 12.0) and Torch 2.11.0+cu128; the new
host has RTX 4060 Ti (8.9) and Torch 2.1.0+cu121. Python 3.11.14 and driver
580.173.02 match. Core dependencies, native Box2D, isolated regression tests,
CUDA/cuDNN and source/resource preflight passed. More importantly, the actual
RLPD batch-64, TD-MPC2 batch-256 and DrQ 32:32 batch-64 update paths all ran
on generated data with finite GPU metrics, without constructing or resetting
the HAIC environment; synthetic checkpoint and actor-load checks also passed.
**Development and a new, separately identified TRAIN runtime appear technically
feasible**, but this does not reproduce the old Torch/GPU environment, establish
long-run throughput, validate the whole artifact transfer, or make either
partial learner exactly resumable. The isolated `/tmp/kilo/haic-cpu21`
Torch 2.1.0+cpu interpreter is restored and import-checked with CUDA disabled.
It has since run TD-MPC2's source-bound CPU diagnostic for 16 capped episodes
on four already-consumed TRAIN cells; this is internal evaluation only, not
official Agent packaging or acceptance. The interpreter remains ephemeral and
shared.
Keep the migration stop:
no learner run or evaluation cell until a new user instruction, restored-evidence
verification and the relevant frozen protocol/exposure/resource gates. See the
[historical inventory](../../experiments/instance-migration-training-packages-2026-09-27.txt),
[initial validation](../../talk/messages/20260928T072107Z-k8v2-new-instance-gpu-validation.md)
and [synthetic training checks](../../talk/messages/20260928T074037Z-k8v2-synthetic-training-feasibility.md).

## Current Position

**DrQ-v2 is CLOSED by user direction (2026-09-29).** Do not initiate further
DrQ-v2 research, training, evaluation, speed/completion variants or official
actions. DrQ rows below and later historical snapshots record past work, not
an active queue or a candidate promotion. Preserve the original actor, local
ZIP and all negative-result evidence; see the
[closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line).

| Area | Current status |
|---|---|
| TD-MPC2 current research status | **Temporarily PAUSED by user direction on 2026-09-30 to focus on other ideas.** Scheduled resource follow-up cancelled; no active TD operator. The next raw-MSE reward-head-only control is implemented but unexecuted: final-source v3 benchmark, experiment protocol and real run are absent. No automatic restart when memory recovers; explicit user reopening is required. Earlier negative gates and artifacts remain preserved. See the [detailed paused plan](../plans/active/tdmpc2-pixel-online-baseline.md#user-requested-focus-pause-2026-09-30). |
| DrQ-v2 speed-only follow-up | **CLOSED with the entire DrQ-v2 line by user direction.** A frozen pad-4 seed1 brake-only inference intervention failed to preserve completed roads on 32 repeatedly reused development cells: track1 5/16 control versus 3/16 treatment (three lost finishes), track2 1/8 versus 0/8 (one lost), track3 0/8 in both arms. Only two mutually completed track1 laps shortened by 1.74/1.28 seconds; no cross-track speed result exists. The 128-episode CPU21 run passed two-reload parity and action-trace hashes. Do not deploy or pursue another DrQ speed hypothesis. The original actor/local ZIP and evidence remain unchanged. See the [`frozen result`](../../experiments/drqv2-speed-reused-development-v1-result.json) and [closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line). |
| Validated internal baseline | Native DrQ-v2 control with augmentation pad 4. Two unique control actors, one per training seed, were evaluated on two fresh internal confirmation cohorts; the recorded control outcomes range from 4 to 7 finishes in 32 cells. |
| Active research direction | DreamerV3 is CLOSED. Other algorithm work is listed separately below; there is no active Dreamer experiment or implementation task. |
| Current blocker | None for DreamerV3: the research line is closed under the current design and budget. This is not a theoretical impossibility judgment; further progress would require design-level rework. |
| DrQ-v2 narrow tuning | Closed. The predeclared steering-logit L2 and pad=1 follow-ups both regressed; do not launch a third narrow tuning axis. |
| DrQ-v2 teacher-replay plan | The isolated r3 collection completed its fixed 16,384 decisions/source cap, but one source had only 3/4 required finished geometries. A3 is inconclusive; no paired learner training, screen, confirmation, or blind evaluation occurred. Preserve both datasets as consumed and do not extend the cap or substitute a source. See `docs/experiments/INDEX.md`. |
| DrQ-v2 training-only geometry study | Completed: consumed-road failure analysis, blind-safe seed audit, 120 distinct TRAIN roads + 16 separate TRAIN-DIAGNOSTIC roads in six measured families, static/finish-logic sanity, and 272 sealed frozen actor diagnostics. The same 136 training-only roads gave 10 both-actor finishes, 74 provisional boundary, 50 difficult-with-progress, 2 unresolved and zero selected malformed geometries. No new learner training or held-out/blind action. See `docs/experiments/drqv2-geometry-augmentation-v1.md`. |
| DrQ-v2 geometry-mix fine-tuning | All six frozen r6 online-only runs completed 32,768 additional decisions and 22,768 updates each, sampling TRAIN only. On the same 16 previously designated TRAIN-DIAGNOSTIC roads, canonical repeat-0 finishes were uniform 1/32, failure-weighted 4/32, and easy-retention 3/32 across the two learner seeds; the unchanged source actors finished 11/32. This is descriptive, unranked development evidence, not fresh generalization, confirmation, blind, or official HAIC performance; no weights were selected or promoted. r4/r5 partial attempts remain preserved and are not resumed or counted. See the r6 [`active plan`](../plans/active/drqv2-geometry-mix-plan.md), [`protocol`](../../experiments/drqv2-geometry-mix-v1-r6.json), six run `result.json` files, and [`diagnostic manifest`](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json). |
| DrQ-v2 r7 source retention | Twelve frozen 32:32 source/online TRAIN runs completed the r6 budget with and without one pre-fixed actor preservation loss. On the same 16 reused TRAIN-DIAGNOSTIC roads, r7a kept 1/3/1 of the eleven source-success actor/road cells and gained 2/7/4; r7b kept 4/1/5 and gained 5/7/5 (uniform/failure-weighted/easy-retention). All six arms lost at least six old successes; the predeclared >=9 kept and >=2 gained retention contract failed despite higher total finishes. No model promotion or fresh held-out/confirmation/blind/official action. See the [`r7 evidence report`](../experiments/drqv2-retention-r7.md), [`result`](../../experiments/drqv2-retention-r7-result.json), and [`384-episode trace manifest`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/manifest.json). |
| DrQ-v2 final-source retention and reconstruction r1 rev2 | Offline parity passed 5,998 archived source actions/666 prior state hashes; both frozen source TRAIN pools contain 100,000 decisions. Five of six original learners completed; the 21,037-decision partial arm and stale step-16,384 checkpoint remain immutable and cannot be resumed exactly. The user explicitly reopened this line. Revision 2 binds the terminal TD-MPC2 v2 TRAIN and 16-episode CPU diagnostic on four already-consumed track-1 cells. A first seed0/uniform launch failed before environment creation; its pinned failure/runtime receipts show 0 resets, decisions, or updates. After the duplicate Torch interop setup was removed from the child launcher, the revision-2 preflights passed with zero interaction. Seed0/uniform then completed 32,768 decisions/22,768 updates on RTX 4060 Ti/Torch 2.11.0+cu128; its 67 TRAIN episodes/cells and eight checkpoint/actor/sample-trace hashes verify. The 120 parent catalog seeds are reused, not fresh. Five arms remain; no six-arm sample audit or final-source diagnostic exists yet. The >=9/11 kept AND >=2/21 gained retention gate is **unevaluated**. See the [child protocol](../../experiments/drqv2-final-source-replay-reconstruction-r1.json), [exposure receipt](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/preflight-exposure-audit.json), [zero-interaction setup failure](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/preflight-failure-20260928T104204Z-seed0-uniform.json), [arm0 result](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/learner-0-uniform-final_source/result.json), [DrQ report](../experiments/drqv2-final-source-replay-v1.md), [active plan](../plans/active/drqv2-geometry-mix-plan.md), [sample adapter](../../scripts/audit_drq_final_source_reconstruction_samples.py), and [diagnostic adapter](../../scripts/diagnose_drq_final_source_reconstruction.py). |
| Independent TD-MPC2 pixel baseline | Paper-faithful 5M-size-class model, episodic termination and MPPI use the TD-MPC2-only HAIC pixel/action adapter. V1 stopped at 10,061 decisions after a planner-reset inference-tensor failure; its 29 episodes had no finishes and the checkpoint is provenance-only. User-directed v2 from scratch passed restore/exposure/source/resource gates and ended at a valid episode boundary: 12,058 decisions, 37 episodes, 12,057 updates (10,000 pretraining; 2,057 post-seed), and 1,629.3 seconds. It stopped because 1,942 decisions remained, fewer than the frozen 2,000-step max episode. All 37 TRAIN episodes had no finish; the reset regression did not recur. The run used the new RTX 4060 Ti/Torch 2.1.0+cu121, not the previous runtime. Source-bound Torch 2.1 CPU export passed with zero resets. The 16-episode capped diagnostic on the same four consumed TRAIN cells had prior 0/8 finishes, 0 censored, mean progress 0.05252/reward -46.904; MPPI 0/8 finishes, 1/8 censored at 500 steps, mean progress 0.03487/reward -50.155. The evaluator marks `capped_finish_comparison_valid: false`; these reused-TRAIN proxies establish no strategy benefit/failure or generalization. No fresh, confirmation, blind, official or promotion action. See the [`active plan`](../plans/active/tdmpc2-pixel-online-baseline.md), [`v1 failure`](../../experiments/tdmpc2-reused-train-pilot-v1-failure.json), [`v2 exposure`](../../experiments/tdmpc2-reused-train-pilot-v2-exposure.json), [`v2 result`](../../runs/tdmpc2-reused-train-20260927-v2/result.json), and [`paired diagnostic`](../../runs/tdmpc2-reused-train-20260927-v2/evaluation-result.json). |
| User-directed pixel RLPD pilot | V2 is closed stop/hold. The separate long-horizon v1 passed screen, strict confirmation, and blind; RLPD seed 11 at 131,072 steps remains an internal candidate. Entropy V1–V3 aborted before interaction. V4 consumed fresh teacher/student runs but stopped before screen on source-hash drift; its held-outs are retired. V5 used fresh prior data and four 131,072-step target-arm runs; its screen had 29/192 canonical finishes and passed all target/seed gates. Strict confirmations were author-target 17/32 and 7/32 versus +1.5 target 6/32 and 6/32; all were eligible and operationally clean. Author-target passed the paired dominance gate, and its selected seed-50 actor finished 7/24 on the internal blind (mean progress 0.657). This remains two-seed internal evidence, not an official result. See `docs/experiments/INDEX.md`. |
| Pixel RLPD completion-first G0 | Complete TRAIN-only observational diagnosis: two frozen actors on 12 shared geometries, 24 episodes and 10,049 decisions; each finished 3/12 and failed 9/12. Contact and centerline-distance events also occurred on successful controls; the 20-decision low-directed-motion event appeared only on nonfinishes in this cohort. No causal remedy or learner experiment is selected. All cells are consumed; see `experiments/rlpd-g0-completion-v1-result.json`. |
| Current submission baseline | User-designated crossing_projection + collision-shield v1, ZIP`c9e376a0...`, as of2026-10-02. Pixel RLPD seed11 retains its historical internal-study candidate status, not the submission baseline. See `docs/results/MODEL_STATUS.md`. |
| Official external state | User reports Track4 finished in18.4s and6th overall at2026-10-02T01:36:47Z, bound by the user to the selected ZIP. No site submission ID, server receipt, independent package-to-result verification or model-confirmation receipt is recorded. This update is not an upload or server confirmation. |

**Latest DrQ final-source status (2026-09-28 14:03 UTC; supersedes the earlier DrQ table row):** revision-2 child preflight passed. Four of six arms completed full TRAIN budgets: all three source-seed-0 mixtures and seed1/uniform. Their result/step/episode/checkpoint/actor/sample-trace artifacts independently verify; all cells are from the reused parent TRAIN catalog. Two source-seed-1 arms remain, next failure-weighted. Per the user's pause request no DrQ process is active. The prior partial arm/setup failure remain preserved; no child sample audit, 192-episode diagnostic or retention-gate decision exists. See the [revision-2 protocol](../../experiments/drqv2-final-source-replay-reconstruction-r1.json), [arm results](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/), and [DrQ report](../experiments/drqv2-final-source-replay-v1.md).

**DrQ resumption update (2026-09-29; supersedes the 4/6 status above):**
The user reopened development and both remaining source-seed-1 TRAIN arms
completed 32,768 decisions and 22,768 updates each under the same revision-2
protocol. All six child results now exist; the last two arms' checkpoint,
actor and sample-trace hashes were independently rechecked, with no OOM or
protected-cell action. The source-bound six-arm CPU21 sample audit then stopped
on the first `seed0/uniform` online warmup comparison: its first 10,000 replay
`frames` were not byte-identical to the historical r7b comparator. This is a
**provenance gate failure**, not a learner or retention outcome. No passing
sample receipt or 192-episode TRAIN-DIAGNOSTIC output exists; the fixed
>=9/11 kept AND >=2/21 gained gate remains **unevaluated**. The original
migration-partial arm remains immutable. See the [audit HOLD](../../talk/messages/20260929T074909Z-t8p3-drq-sixarm-audit-hold.md)
and [DrQ report](../experiments/drqv2-final-source-replay-v1.md). Do not run
the diagnostic or infer cross-runtime bitwise comparability until the audit
discrepancy is characterized under a separately source-bound decision.

**Latest DrQ final-source outcome (2026-09-29 09:43 UTC; supersedes the
earlier DrQ HOLD as the operational status, not as an original-audit pass):**
All six revision-2 TRAIN learners remain complete. The original six-arm
sample audit **failed** on cross-runtime historical warmup byte parity and
its receipt remains absent. A distinct [source-pinned postrun r2 protocol](../../experiments/drqv2-final-source-replay-reconstruction-postrun-r2.json)
passed its separate zero-reset six-arm replay provenance audit
([receipt](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/pre-evaluation-postrun-r2-sample-audit.json))
and CPU21 zero-reset preflight, then completed exactly 192 episodes (96
deterministic canonical pairs) on 16 **previously consumed** track-1
TRAIN-DIAGNOSTIC roads. Its [manifest](../../runs/20260928-drqv2-final-source-replay-reconstruction-r1/train-diagnostic-postrun-r2/manifest.json)
SHA-256 is `e159681ede6f07f1432d5f1b5484bf7c584581704585d8c27f10688bf016dd9f`;
all 195 listed output hashes independently verify. Per mixture, from 11
known source-success and 21 source-failure actor/road cells, uniform kept
2/11 and gained 4/21, failure-weighted kept 5/11 and gained 7/21, and
easy-retention kept 6/11 and gained 6/21. **None meets kept >=9/11 AND
gained >=2/21** even as a descriptive development signal. The original
matched replay-only gate is not repaired or causally evaluated; the new
cross-GPU trajectories and unproven child replay pixel parity limit this
to reused-TRAIN descriptive evidence. No DrQ candidate selection, fresh,
confirmation/blind or official evaluation is opened. See the
[result note](../../talk/messages/20260929T094736Z-t8p3-drq-r2-diagnostic-no-signal.md)
and [full report](../experiments/drqv2-final-source-replay-v1.md).

**TD-MPC2 current status (2026-09-30): PAUSED to focus on other ideas.**
The user explicitly stopped this research line and cancelled its scheduled
follow-up. The 00:25 UTC resource wakeup was cancelled; no active TD
operator or pending continuation is left. Do not restart automatically
when memory recovers or on delayed delivery of an old alert. A new explicit
user direction is required to reopen this line. The next unexecuted idea
was a reward-head-only objective control (categorical CE -> decoded raw
reward MSE), starting from the original overshoot100k parent, not the
failed adapted model. Actual MSE checkpoint loads, updates and resets
remain zero; no final-source v3 benchmark, frozen MSE protocol or MSE
run exists. See the [detailed paused experiment and restart boundary](../plans/active/tdmpc2-pixel-online-baseline.md#user-requested-focus-pause-2026-09-30)
for exact hypothesis, data split, source hashes, budgets, unresolved
memory estimate, and fit/excluded/action-choice gates. Completed
negative results below remain evidence; this is a focus pause, not
a performance verdict on the unexecuted MSE treatment.

**Historical TD-MPC2 direction (2026-09-28; superseded by the focus pause above):**
The user classifies v2 as operational but undertrained (only 2,057 online
post-seed updates), not a failed algorithm. A separately frozen comparison
of 3D independent and 2D exclusive-pedal iid seed actions completed on the
same four already-consumed obstacle-enabled track-1 TRAIN roads, twice each.
Mean progress was 0.06897 for 3D versus 0.03682 for 2D (7/8 paired 3D wins),
with zero finish/damage in either arm. 2D had less negative whole-episode raw
return but shorter episodes; the predeclared progress-first rule selects 3D
for a new, separate ~100k from-scratch run. This is reused-TRAIN action-source
diagnosis, not trained-policy or generalization evidence. The independent
[100k TRAIN protocol](../../experiments/tdmpc2-long-reused-train-v1.json)
(SHA-256 `bc1a2746845cbef89275c9b51163c273955ef1664fe833da35ed17e534e8c885`)
passed the zero-reset preflight and started from scratch at 2026-09-28 15:44 UTC
in `runs/tdmpc2-long-20260928-v1/`. It **stopped partially** at 10,020
decisions/updates (28 complete TRAIN episodes, 279 decisions in the open
episode) with an action-bounds `ValueError` after pretraining. No 20k/40k/70k/100k
checkpoint, exact resume, or longer-budget learning conclusion exists. Its
[partial failure receipt](../../experiments/tdmpc2-long-reused-train-v1-failure.json)
preserves ledger hashes. All 10,020 applied actions were valid; an unbounded
float32 MPPI weighted-elite **diagnostic mean** has a reproduced one-ULP
overflow mechanism. The failing next-state value itself was not logged, so
this is a strong inference rather than a direct measurement. A new run must
use the separately source-pinned correction and path. That corrected
[retry v2 protocol](../../experiments/tdmpc2-long-reused-train-v2.json) at
SHA-256 `d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`
passed zero-reset preflight and started a separate persistent TRAIN run in
`runs/tdmpc2-long-20260928-v2/` at 16:27 UTC. Its first 20k whole-episode
checkpoint at **20,099 decisions/updates**, second at **40,024**, third at
**70,361** and final at **100,354** are sealed. Training completed normally
in 17,459 wall seconds; the separate predeclared CPU full-episode reused-
TRAIN evaluation is running.
The [frozen 20k result](../../experiments/tdmpc2-long-reused-train-v2-20k-result.json)
reports 0/67 finishes on reused training roads and identical-anchor H3
reward-ranking **24/40** informative pairs versus the v2 pilot's 25/40.
The fixed 256-window TRAIN seed probe has 0/768 positive terminal labels,
so its low termination loss does not verify event detection. The
[frozen 40k result](../../experiments/tdmpc2-long-reused-train-v2-40k-result.json)
reports 0/124 cumulative TRAIN finishes (0/57 in the 20k-to-40k interval).
Those later episodes progressed farther on average (0.182) and earned more
raw return (+36.36), but were longer and had higher damage (0.158), so this
is not a matched policy gain. The SAME replay probe's reward MAE worsened
0.059 to 0.088, and same-anchor H3 return ranking remained **24/40** versus
the pilot's 25/40. Neither 20k nor 40k proves world-model survival,
learning failure or >=50% completion; the final source-bounded result follows.
A [separately balanced terminal replay probe](../../experiments/tdmpc2-long-v2-balanced-terminal-20k40k.json)
used the SAME 67 positive raw-termination and 335 ordinary-negative
transitions at both checkpoints. The true-next encoded image path recognized
0/67 positives at each checkpoint, whereas the model-predicted rollout latent
path recognized 42/67 then 43/67 (both 335/335 specificity). The probe
contains no finishes or plain time limits; this is an in-TRAIN latent-path
diagnostic, neither proof of an algorithm bug nor finish-event detection.
The [frozen 70k result](../../experiments/tdmpc2-long-reused-train-v2-70k-result.json)
has **0/220** cumulative TRAIN finishes. Its 96 episodes since 40k averaged
progress 0.3463, raw return +213.62, damage 0.3979; 19/96 reached half
progress, five reached three-quarters, none finished. On the SAME twelve
reconstructed TRAIN anchors the H3 reward-return rank improved to **33/40**
informative pairs versus 24/40 at 20k/40k and 25/40 in the pilot. This is
a source-verified directional **within-TRAIN model survival signal**, not
independent-road replication, MPPI/Q validation or frozen-policy completion.
The balanced predicted-latent terminal probe recalls 53/67 raw endings at
70k while true-next-image recall remains 0/67; no finish/timeout cases
occur. Q spread scale rose to 133.12. Do not retune H, MPPI or reward before
the predeclared full-episode evaluation.
The next active TRAIN episode ledger records the [first single finish](../../talk/messages/20260928T200236Z-k3p7-tdmpc-first-training-finish.md)
at decision 74,216 on one reused road, 646 decisions, raw return 692.61,
damage 0.2 and `finished=true`; the evolving-policy denominator at that
boundary was 1/232 completed TRAIN episodes. This happened *after* the
70k checkpoint and is not evidence that the frozen 70k model finishes,
replicates, or approaches >=50%. Continue unchanged to 100k.
By decision/update 88,684 the still-updating TRAIN learner had
[four completed-road events](../../talk/messages/20260928T204539Z-k3p7-tdmpc-four-training-finishes.md)
in 275 completed episodes, all four in the 55 episodes since 70k and
spread over three of its four repeatedly trained geometries. These
on-policy events are not a frozen-actor completion rate; no 100k model
was available at that intermediate decision boundary.
The [completed 100k model result](../../experiments/tdmpc2-long-reused-train-v2-100k-result.json)
at 100,354 decisions/updates and 307 episodes binds all four checkpoint,
training/step-ledger and read-only diagnostic SHAs. All 14/307 evolving-
policy TRAIN finishes occurred among the 87 episodes after70k, spread over
all four reused roads (**14/87**, *not* frozen-policy finish rate). The
identical-anchor H3 raw-return ranking is **31/40** at100k versus pilot
25/40,20k/40k24/40,70k33/40: directional within-TRAIN reward-model signal,
not independent generalization. Fixed seed probe reward MAE 0.05184 is
below constant 0.37949 but has no positive terminal labels; balanced
predicted-latent raw termination recall is 59/67, true-next 0/67,
without finish/time-limit examples. The
[full-episode evaluation protocol](../../experiments/tdmpc2-full-consumed-train-v1.json)
SHA `874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd`
passed zero-reset CPU preflight and [completed 16/16 full episodes](../../experiments/tdmpc2-full-consumed-train-v1-result.json)
on **only** those same four consumed roads, x2 repeats each for prior and
MPPI. The frozen final actor had **prior 0/8 finishes and MPPI 0/8**, with
no censored episodes. MPPI had mean progress 0.507/raw return +385.93/damage
0.45 versus prior 0.327/+237.73/0.35; CPU action latency means 0.739s
versus 0.00193s. Same road/reset conditions do not match trajectories.
The evolving training collector's 14/87 late finishes cannot replace either
frozen 0/8 denominator. The 100k baseline is operational and its in-TRAIN
reward-model rank rose to 31/40, but local full-episode policy completion
remains weak; this does not prove TD-MPC2 intrinsically failed or measure
fresh-road >=50% performance. The separate
[single-axis damage-target protocol](../../experiments/tdmpc2-damage-shaping-v1.json)
SHA `4f037f39ac7ab4f56961723478048a72d604972c86be17b9b0b9dc5786cbd2c8`
passed zero-reset preflight, runtime/resource checks and 88 synthetic TD
tests, then started from scratch in `runs/tdmpc2-damage-20260928-v1/`
on the SAME four consumed TRAIN roads. Only replay's training reward is
changed; official raw environment return/progress/damage/finish, H=3,
5M/batch256/default MPPI/10k seed and 100k budget remain unchanged. The
[first treatment 20k checkpoint](../../experiments/tdmpc2-damage-shaping-v1-20k-result.json)
is sealed at 20,149 decisions/updates and 68 completed TRAIN episodes
(0 finishes). Across all 20,149 steps damage delta is zero: raw and
replay-target reward sums both equal -3,976.78, so the penalty DID NOT FIRE
before this model was saved. The first actual +0.2 damage/one-unit target
cost was logged later at decision 22,123. No frozen treatment-policy
evaluation or evidence of shaping benefit exists yet. The separate shaped-
target [40k checkpoint](../../experiments/tdmpc2-damage-shaping-v1-40k-result.json)
now seals 40,154 decisions/updates, 131 completed TRAIN episodes and **0/131
finishes**. Since20k, 63 episodes averaged raw progress0.1805, raw return
+47.20, damage0.206; 0/63 finished. Exactly 65 step damage increments
summed to13.0, so raw-minus-training reward sum is 65.0; primary raw reward
is unmodified. The seed-fixed probe has zero damage and zero terminal
positives, so its training-target MAE0.04874 neither measures penalized
states nor proves policy benefit. Baseline and treatment first planned
actions diverged *before* any penalty; one-seed differences are not causal
or transition matched. The [sealed 70k result](../../experiments/tdmpc2-damage-shaping-v1-70k-result.json)
at 70,481 decisions/updates records 2/203 evolving TRAIN episode finishes,
both on ONE reused road (2/72 since40k). The latter 72 averaged raw
progress0.402, raw return+226.98 and damage0.394; cumulative replay-only
penalty was 207 units from 207 positive damage steps. This is neither a
frozen-policy finish rate nor causal proof of the shaping hypothesis.
The [completed damage-only 100k source](../../experiments/tdmpc2-damage-shaping-v1-100k-result.json)
sealed at 100,186 decisions/updates, 286 evolving-policy TRAIN episodes
(10 finishes, eight of 83 episodes since70k across three reused roads) and
exactly 391 replay-only penalty units from 391 positive damage increments.
Those in-training outcomes are not the final model's rate. Its separately
[source-bound frozen CPU full-episode result](../../experiments/tdmpc2-damage-full-consumed-train-v1-result.json)
finished **prior 0/8 and MPPI 2/8**, with both MPPI successes being two
repeats of ONE heavily trained road; all 16 episodes ended uncensored.
This failed the predeclared local MPPI >=4/8 threshold and cannot establish
fresh-road, protected or official >=50% generalization. RAW-target frozen
baseline MPPI/prior each finished0/8 under matched road/reset conditions,
but trajectories differ and the two training runs' FIRST planned actions
forked BEFORE any damage cost, so an attributable shaping improvement is
unproven. MPPI mean damage remained0.45 under both learning objectives.
The separate shaped-
checkpoint CPU evaluator's earlier source-level
[HOLD was resolved](../../talk/messages/20260928T232547Z-k3p7-tdmpc-damage-evaluator-code-ready.md):
it now binds shaped replay/probe/producer raw telemetry to complete ledgers,
exact reset-intent phases and baseline source hashes, and 123 synthetic
tests passed. A distinct SHA-bound completed-treatment evaluation protocol
then passed zero-reset preflight and produced the valid 16-episode RAW
outcome above; it did not modify the frozen raw-baseline evaluator or
protected roads.
The [seed-boundary comparison](../../talk/messages/20260928T232052Z-k3p7-tdmpc-damage-first-plan-fork.md)
matched all 28 complete random-action seed episodes and decision10,000's
action/raw reward, but the very first planned action at decision10,001
differs BEFORE any observed damage penalty. No 10k model/RNG snapshot exists,
so the cause is unknown: matching source settings is not a transition-
matched or causal single-seed treatment comparison. Preserve the completed
source; stronger controls are needed before attributing any difference
solely to shaping.
The [no-reset seed replay byte audit](../../experiments/tdmpc2-raw-vs-damage-seed-parity-v1.json)
used strict SHA-bound RAW and DAMAGE 20k checkpoint replays: **0/10,000**
differences in seed actions, raw rewards, pre/next pixel stacks or
termination flags, and the same 256 frozen seed-probe windows/IDs/bytes.
This strengthens input-data parity, not equivalence of missing 10k model/
optimizer/RNG states or a causal explanation for the first planned fork.
The [no-reset frozen 100k CPU/GPU probe](../../experiments/tdmpc2-final-100k-freeze-parity-v1.json)
strict-loaded identical weights and aligned four archived TRAIN pixel-shift
fixtures: all nine latent/prior/reward/Q/termination output types met fixed
numeric tolerances (maximum latent difference 5.60e-6, decoded Q 1.45e-4).
With identical CPU pre-plan RNG and warm-start, MPPI selected the same elite;
training-only final Gaussian changed the resulting action by L2 0.0823 on
one archived state. This does not prove why frozen evaluation finished 0/8:
historical stochastic augmentation, full-episode CPU/GPU trajectories and
evaluation failure actions are unavailable from these fixed probes.
The [frozen result](../../experiments/tdmpc2-exploration-v2-result.json)
and [updated plan](../plans/active/tdmpc2-pixel-online-baseline.md) contain
denominators, tradeoffs, ordered diagnostics and future mismatch hypothesis;
v2's original protocol/checkpoint remain untouched. No confirmation, blind,
official or promotion action is opened.

The separate [v2 read-only H=3 logged-sequence score](../../experiments/tdmpc2-v2-logged-ranking-diagnostic.json)
found 37 replay episode-start returns indistinguishable to 1e-6, making all
666 across-start rankings ties. It performed zero resets and is not a verified
same-state prefix branch or a world-model survival/failure signal.

The separate [parity-checked v2 branch result](../../experiments/tdmpc2-v2-prefix-branches-v1-result.json)
did yield distinguishable H=3 returns at later within-TRAIN anchors: 12
predeclared anchors, five candidate suffixes, 72 reused-road resets, 80/120
real-return ties and 25 concordant versus 15 discordant informative pairs
(62.5%) across seven informative anchors. Accessible reconstructed prefix
states, raw commands and pixels matched before every branch; historical
hidden Box2D solver-state identity is not proven. Pairwise observations are
dependent; 20k/40k ranked 24/40, one pair below the pilot, whereas 70k
ranked 33/40 and final100k 31/40 on the SAME fixed pairs. The improvement
over the short pilot is within reused TRAIN roads, not a fresh-road result.

**TD-MPC2 H=5 action-choice gate (2026-09-29):** The separate
[logged H5 screen](../../experiments/tdmpc2-h5-logged-v1-result.json)
passed a weak in-replay reward/termination gate without resets; it did not
test action choice. The subsequently [frozen five-action real branch result](../../experiments/tdmpc2-h5-branches-v1-result.json)
completed 72 consumed-TRAIN resets at 12 reconstructed anchors on the same
four roads, all 60 five-step suffixes, with accessible prefix parity and
no early endings. Of 120 within-anchor actual H5 return pairs, 29 tied.
The predeclared reward-only gate **passed narrowly** (56/91 concordant,
four informative roads, >60%). But on the SAME five candidate actions and
actual H5 outcomes, H3 reward scored 63/91 while H5 reward scored 56/91;
the terminal/Q-including sampled-policy planner-score proxy scored H3
**65/91** and H5 **51/91**. At the fixed twelve anchors H3/H5 planner
top-choice tied for best real H5 return on 8/12 and 7/12, respectively.
The old three-action H3 receipt is not a matched comparator for these new
five-action outcomes. The one-draw planner proxy used different randomly
selected Q-head pairs at 10/12 anchors, so the 65/91 versus 51/91 difference
does not isolate horizon or establish full-MPPI policy inferiority; mean
selected-action H5 regret differed by only 0.00395. Dependent pairs, four
repeatedly trained roads and unknown historical hidden Box2D state further
restrict inference. The separate [read-only all-ten-fixed-critic-pairs control](../../experiments/tdmpc2-h5-fixed-q-v1-result.json)
held each Q pair and bootstrap-policy RNG stream constant across horizons:
H3 ordered 63-72/91 versus H5 42-53/91 across all ten pairs, 0 new resets.
This strengthens the **five fixed candidates** decision hold, not a claim
about full MPPI policy or a horizon-only causal mechanism.
**Do not launch H5 full-episode MPPI** based solely on the necessary reward
gate: actual planner choice quality is not sufficiently positive. The
subsequent one-axis [frozen H3 training-mode action-noise result](../../experiments/tdmpc2-h3-noise-full-consumed-train-v1-result.json)
changed only final-action `eval_mode=True` to `False` on the unchanged RAW100k
checkpoint/MPPI. It completed all **8/8** full episodes on the same four
consumed TRAIN roads x2, with **2/8 finishes on two distinct roads**, zero
censor and peak action latency 3.994s. This missed the fixed >=4/8 across
>=2 roads local gate. The historical frozen H3 no-noise result was 0/8,
but RNG, actions and trajectories diverge; no matched causal improvement,
independent-road generalization or explanation of evolving-TRAIN finishes
is established. The next selected hypothesis is limited TRAIN **road
coverage** from repeated four-road training; a new candidate-specific
cross-lane-consumed TRAIN reuse inventory passed synthetic safety checks
but is **BLOCKED** on unresolved typed legacy/result-only metadata and an
unconditional fail-closed entry. A [proposed 24-road manifest](../../experiments/tdmpc2-consumed-train-24-proposal-v1.json)
selected from catalog order is metadata only, not a cell claim: its
inventory confirmed 24 TRAIN IDs/four per family and 921 typed exposure
rows but classifies numerous known TD/DrQ TRAIN records as ambiguous.
This cannot certify reuse OR prove all 24 cells collide. No diverse
training reset was made; coverage remains untested. Do not turn catalog
membership into clearance.
The subsequent [all-five-head Q-average H3 diagnostic](../../experiments/tdmpc2-h3-all-q-v1-result.json)
on the SAME 12 consumed branch anchors was 65/91 concordant real-H5
pairs (versus original stochastic H3 65/91), with 9/12 tied-best
choices and regret 0.79416 (versus 8/12 and 0.79535). It failed the
predeclared >=66/91 concordance requirement, so no all-Q full-episode
evaluation was run. The next model-level hypothesis is counterfactual
multi-step reward/latent target quality using ONLY the original four
consumed TRAIN roads, subject to isolated learner/preflight validation;
the H5 per-discounted-step branch reward error rose from H3 0.330 to
H5 0.397. This is a mechanism hypothesis, not causal evidence. The fresh
multi-track diagnostic audit remains BLOCKED and no protected cells opened.
The isolated [no-reset reward-overshoot replay audit](../../experiments/tdmpc2-reward-overshoot-target-audit-v1-result.json)
passed its predeclared data gate: 99,126 of 99,740 eligible H3 TRAIN
windows (99.38%) contain both logged steps 4/5, and all four old roads
have nonconstant suffix rewards. A separate reward-overshoot replay/
learner module passed 38 focused synthetic and original regression tests
without changing the frozen base modules. This is **data/code feasibility
only**: the separate `scripts/train_tdmpc2_reward_overshoot.py` runner
passed 63 synthetic/base tests and independent read-only review found
no blocking invariant violation. A host/container NVIDIA PID-namespace
bug and a self-GPU-utilization false alarm were fixed BEFORE the
[synthetic CUDA benchmark](../../experiments/tdmpc2-reward-overshoot-throughput-v1-result.json):
full `learner.update(replay)` timing extrapolated 18,693 seconds, under
the predeclared <20,600-second resource screen and unchanged 21,600-second
TRAIN wall cap. This is synthetic speed, **not learned performance**.
The separate [variant protocol](../../experiments/tdmpc2-reward-overshoot-train-v1.json)
passed a zero-reset SHA/runtime/resource preflight; four exact consumed
TRAIN cells were independently rechecked for active claims and declared
as **non-exclusive, already-consumed TRAIN reuse**. An exclusive new
from-scratch [H3/RAW run](../../experiments/tdmpc2-reward-overshoot-train-v1-result.json)
finished at the first >=100k whole-episode checkpoint (100,159
decisions/updates; 309 episodes; 19,902 seconds <21,600-second cap).
Its separately [SHA-bound no-reset real-branch score](../../experiments/tdmpc2-overshoot-old-branch-score-v1-result.json)
matched the SAME 12 anchors/60 candidate action bytes and actual H5
outcomes: H5 reward rank rose **56/91 -> 74/91**, H3-prefix rank
**31/40 -> 34/40**, and 4/4 roads' H5 rank did not regress. But
H5 absolute error per discounted step worsened **0.39668 -> 0.40691**,
so the *predeclared four-way development gate FAILS*. Mean H5 return
underprediction also increased (-1.084 -> -1.368). The positive rank
is internal TRAIN action-order evidence, NOT a matched learner-policy
generalization or official score; its unchanged H3 default-MPPI
eight-full-episode evaluator remains DORMANT under this failed gate.
Collector finishes are not frozen-policy finishes. Diagnose the observed
rank-versus-magnitude split before choosing a distinct one-axis
intervention; reward calibration/dynamics/Q are hypotheses, not causes.

**TD-MPC2 post-100k mechanism split (2026-09-29):** The separate
[no-reset terminal/Q decomposition](../../experiments/tdmpc2-overshoot-planner-terms-v1-result.json)
reproduced old/new reward scores on those SAME 60 actual five-action
branches. No predicted termination crossed the planner's threshold;
terminal gating changed no reward ordering. New H5 reward ranked
**74/91**, but each of ten fixed Q-head pairs reduced the complete
planner score to **31-36/91** with top-choice regret **2.076** versus
reward-only **0.502**. This is five fixed choices, NOT full MPPI's 512
proposals or a fresh-policy result. The independent [reward-blind,
same-episode positive-event probe](../../experiments/tdmpc2-overshoot-positive-events-v1-result.json)
passed its separate head-mechanism screen on 4/4 old TRAIN roads:
new true-observation-latent positive-event bias -1.15 to -1.37 raw
reward/step and MAE 1.40-1.53 versus original head MAE 0.47-0.59.
Its initial probe had a REAL old307/new309 episode-ID mismatch and
failed zero-load preflight; the corrected source/tests then passed and
the final receipt required 0 resets/updates. Old model was scored on
its own replay and new model on old replay; this is not a causal
reward-head attribution. Prioritize action choice: a **new single-axis
H5 planner Q-weight 1->0** was evaluated in a separate
[source-bound real five-action branch result](../../experiments/tdmpc2-overshoot-q0-branches-v1-result.json)
on disjoint ORIGINAL RAW episode4..7 anchors over the same four
consumed TRAIN roads. All **72 reset intents**/60 full H5 suffixes
completed; 93 informative actual-H5 pairs and 27 real ties. New
H5 Q0 score ranked **68/93 (73.1%)**, better than every fixed-Q1
pair's **39-41/93**, but below H3 Q0's **70/93** on the SAME outcomes.
H5 Q0 tied the best actual action on **7/12**, below the predeclared
8/12, and mean H5 regret 1.279 exceeded H3 Q0's 1.028. Thus the
entire Q0 gate **FAILED** despite partial pairwise improvement: no H5
Q0 or original H5 full-policy episode was released. An independent
4/4 same-cell audit verified consumed TRAIN reuse; original RAW
complete ledger/checkpoint and source were rehashed before EACH reset.
Accessible prefix parity is not a hidden Box2D solver-state proof,
and five fixed candidates are not the complete MPPI policy. The
head-only positive-event target remains a candidate *hypothesis* but
requires its own isolated TRAIN-only design, evaluation and failure
rules. Original overshoot four-way FAIL and blocked fresh-grid isolation
remain unchanged; no official/protected model confirmation exists.

**Head-only adaptation data gate:** Its first predeclared adaptation-
excluded original RAW ep12..19 split FAILED the >=100 positive-event
labels per road screen at 38/87/38/39, with **zero optimizer steps**.
A separately fixed v2 TRAIN-only split ep12..43 has
144/199/137/151 positive labels across the same roads, passing
*only* data sufficiency; ep44..306 is proposed fitting data and
ep8..11 remains excluded for a possible third real branch cohort.
The [source-bound v2 head-only pilot](../../experiments/tdmpc2-head-only-adaptation-v2-result.json)
now completed **512 updates**, zero environment resets, in 367.66s
under its 1800s cap; non-head tensors were bitwise unchanged.
It **FAILED** the adaptation-excluded error gate on 0/4 qualifying
roads: positive-event MAE worsened on every road (1.06-1.68 before,
2.27-2.53 after), and nonpositive MAE worsened beyond0.05 on3 roads.
Its training CE decreased but is not calibration or driving evidence.
No old-branch preservation score, new ep8..11 branch reset or policy
evaluation is authorized for that adapted checkpoint. A synthetic CUDA
forecast passed separately; real no-load integration caught and fixed
forecast boolean metadata being mistaken for a numeric resource field
before any checkpoint load. Previously exposed source TRAIN episodes
are not independent holdouts. Next diagnose fit/excluded state, action
phase and reward-distribution shift, not assume a head-only remedy.
Original RAW ledger recount: fittingep44..306 has28,518 positive
events/86,581 decisions (32.94%); excludedep12..43 has631/9,929
(6.36%), including388/6,156 random and243/3,773 early-planned
decisions. The split differs in collection maturity as well as action
mode. A new zero-update frozen-head fit-versus-transfer diagnostic is
being prepared to test whether raw calibration improved even IN fitting
data; do not explain the excluded FAIL by distribution shift alone or
spend the reserved episode8..11 branch resets.
That [frozen-head no-update audit](../../experiments/tdmpc2-head-fit-transfer-v1-result.json)
now rejects a transfer-only explanation: on86,581 fitting targets,
naturalCE improved1.5066->1.2208 but positive rawMAE and absolute
bias worsened onALL4 roads (fitgate0/4). Next objective-only control
starts from ORIGINAL overshoot parent, not failed adapted head:
CE->standard decoded rawrewardMSE with split/sampler/optimizer/
512-update budget unchanged. Isolated MSE code passed91 combinedtests
and independent finalization review; no actual MSE adaptation yet.
Its memory readiness gate is BLOCKED: observed raw cgroup headroom
35.39GB versus fixed sourcepeak+24GiB requirement42.67GB. Older
generated benchmark receipts were tied to superseded source and cannot
release finalsource; a new v3 source-matched benchmark/protocol is
would be required if the user explicitly reopened the line. Automatic
resource checks are now cancelled. The 42.67 GB bound used the full
100k peak plus an agent-selected 24 GiB reserve; it is not a measured
head-only requirement or official rule. Its conservatism was acknowledged
in the user's follow-up, with no code or budget revision authorized.
The current stop is to focus on other ideas, not a memory-wait queue.

**TD-MPC2 continuation objective (user direction, 2026-09-28):** Continue
isolated learning and evaluation iterations toward >=50% **full-episode**
finishes and the strongest project model. The first longer-budget run was a
10,020-decision *operational partial failure*, not an algorithm learning
verdict; its diagnostic overflow was corrected in a separately source-pinned
retry completed at 100,354 decisions. Cycling its four already-consumed
track-1 TRAIN roads cannot
demonstrate unseen-road completion even if those roads eventually finish.
After a corrected 100k run, predeclare a separate, cross-lane-audited
training-excluded TRAIN-only diagnostic with multiple track IDs and complete
2,000-decision episodes, explicit finish denominators and matched frozen-model
reference cells; **no such fresh cells are certified or allocated yet**.
Existing catalog reservations and TD partial reset-intent records require a
TD-aware cross-lane audit; the RLPD G1 claim CLI does not scan TD
`training.jsonl` and cannot clear TD allocations. See the
[allocation audit](../../talk/messages/20260928T164442Z-k3p7-tdmpc-train-grid-audit.md).
An isolated TD read-only candidate auditor now checks known claims,
catalog reservations, partial reset intents and protected IDs, but
intentionally **always reports `BLOCKED`** until legacy exposure/result-only
coverage is independently closed. It cannot reserve roads or authorize
evaluation; its [scope/result note](../../talk/messages/20260928T170516Z-k3p7-tdmpc-cell-auditor-blocked.md)
records the missing gate.
The separate [reused multi-track feasibility assessment](../../talk/messages/20260928T192157Z-k3p7-tdmpc-reused-multitrack-path.md)
suggests a lower-tier TD-training-excluded, **cross-lane-consumed** TRAIN
catalog cohort could be predeclared after the 100k source result and a
different, lineage-bound reuse audit. No cells have been selected or
cleared; the fresh-grid BLOCKED status is unchanged, and even 12/24
finishes there would not prove unseen-road or official performance.
The currently best
independently evaluated RLPD seed-11 actor finished 12/24 screen, 12/32 on a
different confirmation cohort and 9/24 blind: those cohorts cannot be pooled
or reused for TD tuning, and 50% generalization is not established. See the
[TD-MPC2 active plan](../plans/active/tdmpc2-pixel-online-baseline.md) for the
bounded stepwise gate. Official model confirmation and submission remain
separate user-authorized actions.

**Local TRAIN resource gate provenance (2026-09-29):** The
[current official Participants README](https://github.com/2026-HAIC/Participants)
limits the *submitted* CPU Agent (1,024 MB participant process, 10s import/
construction, 5s reset/act); it does not prescribe a minimum free RAM,
disk, idle GPU/CPU, or wall-time limit for local training. Full resource
rules on the dynamic official website could not be retrieved. The pilot
16 GiB raw-cgroup/5 GiB disk/7,200s and later long/DAMAGE 16 GiB raw-cgroup
and disk/21,600s were locally source-pinned safety/budget choices, not user-
supplied or universal thresholds. Prior TD v2 was held at ~14.2 GiB raw
cgroup FREE despite ~25 GiB potentially reclaimable inactive file cache;
its prior learner failure was a planner-reset bug, not a proven OOM.
Reapplying an initial FREE-RAM/disk floor after the same learner grows replay
and writes snapshots can cause an avoidable non-resumable stop; conversely
the completed raw100k run reached ~16,124 MiB peak RSS and wrote ~6.74 GB
of snapshots, so no blanket fail-open is safe. Future studies follow the
[measured incremental-capacity workflow](../workflows/run-experiment.md),
with this [origin audit](../../talk/messages/20260929T022000Z-k3p7-resource-floor-origin-result.md)
as context. Current DAMAGE source/protocol and other frozen/teammate-owned
studies retain their own executable gates unchanged; this documentation
correction is NOT a retroactive change to the active run or an official
model action.

## DreamerV3 Research Line: CLOSED

Decision as of 2026-09-27: **additional DreamerV3 work is not worth pursuing under
the current design and budget.** The local implementation, available data, and
experiment contract have not solved long-horizon prior dynamics; further progress
would require design-level rework. This is not a claim that DreamerV3 is
theoretically impossible.

- Posterior reconstruction and short, logged-action predictions can look
  reasonable, but multi-step free prior rollout is unstable; 32-decision prediction
  error worsened beyond a simple repeat baseline. Posterior reconstruction is not
  open-loop prediction.
- Static random B1 v1-v9, random/teacher data, two DrQ-source replay, and H8
  prior-image auxiliary work did not establish stable improvement across two
  learner seeds and both scored action-source strata. The strict H8
  paired-improvement-and-repeat gate failed.
- The evidence does not support explaining the failure solely by unused actions,
  an obvious off-by-one alignment bug, or one missing prior-image auxiliary loss:
  action-input sensitivity was observed, the tested real transition trace found no
  target offset, and prior-image/H8 treatments were actually tried but failed
  their declared gates.
- Residual causes remain hypotheses, not proven mechanisms: limited state/action
  coverage; possible redundancy between the four-frame stack and RSSM memory or a
  reconstruction shortcut; the offline-world-model-first structure; and long-rollout
  dynamics drift.
- Stop local searches over loss weights, horizons, update counts, and similar
  parameters. No further experiment, training, collection, evaluation, promotion,
  official submission, or protected evaluation-cell use is part of this line.

The latest available multi-source+H8 checkpoints are world-model diagnostics at
[`seed 0`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-0/world-model-checkpoint.pt)
and
[`seed 1`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-1/world-model-checkpoint.pt).
Both receipts report `actor_trained=false` and `promotion_eligible=false`; neither
is a performance actor, and no Dreamer checkpoint is selected as a best model.
The final evidence/gate index is [`docs/experiments/INDEX.md`](../experiments/INDEX.md).
The former active plan is now a [closed status pointer](../plans/active/dreamerv3-recovery-strategy.md)
to its [archive](../plans/archived/dreamerv3-recovery-strategy-2026-09-27.md).
Any future contingency is only the
[`DEFERRED / LAST-RESORT ONLY` revival plan](../plans/dreamerv3-revival-plan.md).

## Active Work

The user-authorized DrQ-v2 geometry-mix study is a separate matched experiment;
it neither reopens teacher-replay r1-r3 nor changes DreamerV3's closed status.
Its three fixed family mixtures and diagnostic-only protocol are frozen at
[`drqv2-geometry-mix-v1-r6`](../../experiments/drqv2-geometry-mix-v1-r6.json).
Mix r1-r3 were superseded before any interaction. R4 and r5 each had one partial
learner-0 uniform TRAIN run (16,384 decisions, 6,384 updates) and stopped before
full checkpoint: r4 had an undefined catalog-index helper, r5 detected mutable
documentation in the checkpoint source-hash list. Both were mid-episode and
cannot be resumed. R6 restarts from the frozen source checkpoint with an
explicit TRAIN pool, new RNG schedule and executable-only runtime source hashes.
Partial TRAIN exposure and failure receipts are preserved but are not candidates
or evaluations.

R6 completed with all six run results and both step-16,384 and step-32,768
checkpoint/replay-sample trace hashes preserved under
[`runs/20260925-drqv2-geometry-mix-v1-r6`](../../runs/20260925-drqv2-geometry-mix-v1-r6/);
the active plan indexes the final checkpoint and replay-trace hashes. The frozen
CPU21 development diagnostic used only the 16 catalog TRAIN-DIAGNOSTIC roads:
256 episodes across eight actor roles and two repeats, with all 128 repeat pairs
deterministically identical. Its manifest binds every trace hash and the family
summary is linked from the experiment index. This is not a fresh holdout,
confirmation, blind, or official evaluation, and model updates are not themselves
evidence of generalization or a promotion decision.

The subsequent [r6 regression diagnosis](../experiments/drqv2-geometry-mix-r6-regression.md)
paired all 32 source-actor/road cells with each variant on those **reused**
TRAIN-DIAGNOSTIC roads: uniform lost 11 source wins and gained 1 new win;
failure-weighted lost 10/gained 3/kept 1; easy-retention lost 10/gained 2/kept
1. A bounded action-replay of the 11 archived source-success episodes (5,998
decisions) matched the old traces exactly and measured deterministic actor,
twin-critic and encoder outputs on identical source observations without a new
policy rollout or update. Early actor/encoder drift is directly observed; Q1
reordering on some cells and incomplete source-state replay retention remain
contributors to test, not proven causes. That diagnosis alone did not authorize
training, promotion or a new evaluation partition.

The subsequent separately authorized [r7 retention study](../experiments/drqv2-retention-r7.md)
kept r6's geometry mixtures and exact 32,768-decision/22,768-update budget.
An offline hybrid probe found early action differences from both encoder and
actor-head changes; a TRAIN-only gradient probe fixed one source-action
preservation weight. The actual source/online sample traces verified 32:32 for
every minibatch, and all twelve runs and 384 reused-development episodes passed
independent artifact, repeat and paired-cell checks. r7a's replay-only relative
to r7b's added preservation term did **not** preserve most of the eleven old
successful actor/road cells under any mixture. Failure-weighted r7a's 10/32
total finishes in particular comprise just **three kept plus seven gained**;
easy-retention r7b's 10/32 comprise **five kept plus five gained**. Both old
success retention and adaptation remain separate unsolved questions. Reused
TRAIN-DIAGNOSTIC cells are consumed development feedback, not a fresh holdout;
the [frozen result](../../experiments/drqv2-retention-r7-result.json) does not
promote a model or authorize protected/official action.

### Historical DreamerV3 Evidence (CLOSED; No Follow-up)

The former active plan is preserved in the
[`archive`](../plans/archived/dreamerv3-recovery-strategy-2026-09-27.md).
The details below are a dated record of completed diagnostics and failed gates,
not open prerequisites or authorization for another study. The initial fidelity
work and nine B1 static-random-data trials
are complete; all failed their frozen gates. The
[`B1 failure diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)
separates implementation errors, sparse training signal, and gate design defects.
No completion or official score claim follows from its offline diagnostics.

Dreamer P0/D0 has synthetic-tested actor/return contracts, reset-origin and
completed short-episode replay, and a retrospective scored-window BCE reference.
The regression suite covering the P1 teacher collector, sealed-dataset replay
bridge, fail-closed cross-lane ID inspector and adjacent contracts passed 184
synthetic/mock tests, not driving evidence.
Fractional short-episode sampling now skips unavailable windows; the collector
recomputes pinned audit evidence instead of trusting a self-declared pass receipt.
The inspector parses r6's geometry-sampler RNG separately from road IDs but
cannot certify exhaustive historical allocations and never issues a pass.
B1 remains no-go; no repaired Dreamer driving policy or P1/P1b dataset exists.
A complete independently verifiable seed inventory, fresh immutable protocol,
random-arm collector and offline learner runner remain necessary for P1/P1b.

A separate, non-promoting [reused-TRAIN engineering diagnostic](../../experiments/dreamerv3-reused-train-diagnostic-v1.json)
has since completed on four previously allocated r6 TRAIN roads. Its
[random](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/random/collection-result.json)
and [DrQ-source](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/teacher/collection-result.json)
collections used 1,306 and 2,548 decisions respectively, with zero versus one
local finish in four complete episodes each. Under a separate
[offline protocol](../../experiments/dreamerv3-reused-train-offline-v1.json),
two learner seeds per source completed 64 model-only updates each, without new
environment steps or actor/critic training; the four run receipts are linked in
the [experiment index](../experiments/INDEX.md). Differently sized reused data
and in-distribution training losses do not establish a matched prediction gain
or driving improvement. Fresh P1/P1b remains blocked, with no new Dreamer actor,
held-out result, or official result.

The separately frozen [reused-TRAIN development and update-budget result](../../experiments/dreamerv3-reused-train-development-v1-result.json)
scored two world-model seeds per source on the same four *training-excluded* but
previously allocated r6 TRAIN roads in each action-source stratum. Both new
collections finished 0/4 roads. At 64 model-only updates, all models missed the
shifted-repeat image baseline on teacher-action development; increasing only the
model update count to 256 worsened both teacher-trained seeds' image error on
both reused development strata. Each stratum had four independent terminal
events among 256 scored decisions and no finished development geometry;
average terminal BCE is not evidence of finish-event discrimination. The
second score reused development already inspected after 64 updates, so it is
consumed tuning feedback, not a fresh holdout or P1b result. The B1/P1 gates,
no-trained-student-actor status and prohibition on official claims remain.

Read-only [action-input](../../experiments/dreamerv3-reused-train-action-input-v1-result.json),
[eight-prior sampling](../../experiments/dreamerv3-reused-train-sampling-v1-result.json),
and [posterior/prior](../../experiments/dreamerv3-reused-posterior-gap-v1-result.json)
diagnostics on those SAME consumed road episodes show that teacher-data model
predictions respond to changed future action inputs, but not consistently in
the correct direction against logged targets. Averaging eight prior paths does
not eliminate their 9-32-step image error above frozen-anchor repeat.
Target-conditioned posterior reconstruction is not prediction; one-step prior
is near the true-previous-frame repeat, while 32-step free prior worsens
despite lower raw same-transition KL. That motivated the subsequent bounded
multi-step prior-image treatment reported below; its 256 base plus 64
additional optimizer steps cannot by itself isolate a loss-shape benefit.
No Dreamer actor/policy result, fresh evaluation or official score exists.

The separate [prior-image auxiliary training and score](../../experiments/dreamerv3-reused-train-prior-image-score-v1-result.json)
completed 256 ordinary plus 64 additional world-model optimizer steps per
arm/seed without actor or environment updates. Both teacher-data seeds improved
paired image MSE on both *consumed* development action-source strata, but
teacher-action values **0.015952/0.016135** remained above the predeclared
frozen-anchor repeat **0.014884**. The local teacher-data gate therefore failed;
the statically proposed third-per-family four roads were not opened, and the
extra optimizer steps prevent a pure loss-shape conclusion. A distinct
[12-road source0 diversity collection](../../experiments/dreamerv3-reused-train-diversity-v1-result.json)
used other, previously allocated r6 TRAIN cells: random finished 0/12 and
source0 finished 2/12 roads in two shape families, below its separately fixed
>=3-road teacher support gate. Both sealed archives were preserved but **no
Dreamer learner was released by the source0-only protocol**. A prior DrQ
catalog summary already has one source1 attempt on each same TRAIN road. The separately frozen
[source1 replication](../../experiments/dreamerv3-reused-train-source1-v1-result.json)
completed 5,192 decisions on all 12 roads, finishing 2/12: one road overlaps
source0's two finishes, so the *source-union* is three distinct roads in three
shape families. Its descriptive complement gate permitted **design only** of a
new multi-source study; source0's failed single-source gate stays failed.
Historical source1 also measured these same roads, so this is replication/variation,
not unseen-road support. The original Dreamer offline bridge accepts only one
actor ID/hash and its replay loses tags; the separate
[mixed-source learner](../../experiments/dreamerv3-reused-train-multisource-v1-result.json)
validated each archive independently, bound 24 episodes/10,612 decisions on
only 12 road IDs to a per-seed identical hashed replay lineage and completed
256 model-only updates for each of two seeds. Its predeclared
[four-road image score](../../experiments/dreamerv3-reused-train-multisource-score-v1-result.json)
failed: both seeds missed frozen-anchor repeat on random-action development,
and source0-action development had only seed0's tiny below-repeat value.
Development source/outcome receipts were visible before the scorer source was
finalized, so this training-excluded r6 TRAIN set is consumed iterative tuning,
not a blind/fresh validation target. No actor/critic was trained, no student
driving or official score exists. A separate multi-source data plus H8
prior-image auxiliary interaction was subsequently trained and scored as a
separate model-only study; its predeclared two-seed/two-stratum image gate
failed. The real fresh P1 cross-lane seed audit remains non-passing.

The isolated [mixed-source+H8 model-only training result](../../experiments/dreamerv3-reused-train-multisource-prior-v1-result.json)
used the original source0/source1 sealed 10,612-decision replay and identical
per-seed SHA-bound lineage. Both fixed learner seeds completed 256 ordinary
plus 64 additional prior-image world-model optimizer steps, with zero new
environment steps and unchanged actor/critic/target and their optimizers.
After the owner freed memory, the independently rechecked raw cgroup headroom
met the unchanged 12 GiB floor plus projected peak; no OOM kill was added.
Auxiliary TRAIN frame loss changed in opposite directions across seeds, so this
is pipeline/training evidence, **not** a scored prediction gain. The separately
pinned [read-only comparison](../../experiments/dreamerv3-reused-train-multisource-prior-score-v1-result.json)
against the earlier pure-256 models on the SAME already-consumed four-road
source0/random development cells failed: seed0 worsened on both source strata;
seed1 improved versus pure256 on both but random-action image MSE `0.001509`
remained above frozen-anchor repeat `0.001494`. The rule required BOTH seeds
to improve and beat repeat on BOTH strata. The first metadata-only score
preflight stopped on a 1-MiB catalog cap before ZIP decoding; a catalog-only
bounded correction, 33 scorer tests and new hashes preceded the first actual
score. Additional 64 optimizer steps preclude a compute-matched loss-shape
claim, and the development material was previously consumed iterative tuning.
Previously allocated seventh-per-family r6 TRAIN roads remain unopened,
the fresh P1 audit fails closed, and no Dreamer student actor or official result
exists. The earlier [memory hold](../../talk/messages/20260926T181924Z-k9r4-dreamer-resource-hold-final-check.md)
was resolved for this study after the owner's resource change; do not relax
limits for any future experiment.

For the blocked fresh P1 path, the Dreamer collector now accepts the **exact**
historical G0 `*-seed-audit.json` only as a SHA-pinned prior audit, not as a
protocol or a generic foreign seed audit. Synthetic manifest/hash-drift tests
and the full 367-test Dreamer regression suite passed. The auditor now guards
candidate-bearing known typed road/cell aliases, ledger road IDs and untyped
`start/end` and `start/count` ranges, without failing disjoint synthetic cells.
These fixes do **not** certify candidate freshness: the actual Dreamer P1 audit
still returns `passed=False`; other unknown field/range encodings, TRAIN
start/partial/claim records, own-protocol bootstrap and re-audit before each
reset remain unproven. No fresh P1 road has been selected or driven.

## Completed Teacher-Replay Gate

The user-directed DrQ-v2 teacher-replay protocol is frozen at
[`r3`](../../experiments/drqv2-teacher-replay-v1-r3.json). The r3 datasets passed
source and integrity checks, but one actor missed the preregistered coverage
hurdle; therefore there is no learner/evaluation result or candidate promotion.
The original r1 wrapper-seam attempt and r2 trainer-preflight attempt are retained
as zero-learning aborts, not model failures. Do not reuse r2/r3 teacher datasets,
open the reserved evaluation pools, or infer performance from collection metrics.
The stop receipt and per-source evidence are indexed in
[`docs/experiments/INDEX.md`](../experiments/INDEX.md).

## Completed Pixel RLPD Pilot Gate

The user-directed pixel-RLPD v2 protocol and result are recorded at
[`protocol`](../../experiments/pixel-rlpd-offpolicy-pilot-v2.json) and
[`result`](../../experiments/pixel-rlpd-offpolicy-pilot-v2-result.json). The
teacher dataset met its fixed 8,192-decision/two-finish coverage gate, and each of
the four SAC/RLPD student runs completed 16,384 decisions. The custom 12-cell per-
actor screen (two repeats per cell) produced one canonical finish in 96 canonical
episodes: the selected
RLPD seed-1 8,192-step actor finished once; selected seed 0 finished zero times.
The pilot therefore fails its preregistered two-seed promotion gate. All eight
screened actors passed CPU reload, determinism, and operational eligibility, but
this does not offset the completion gate. Do not open v2's reserved conditional
full, confirmation, or blind cells. The user requested a new feedback/retraining
iteration; any continuation must be a separate protocol with fresh code/data/screen
and held-out allocations, not an extension of v2.

## Fresh RLPD Follow-up (Completed)

The separate hypothesis protocol is
[`pixel-rlpd-long-horizon-followup-v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json)
(SHA-256 `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`). Its
seed and both teacher-ledger audits were completed before interaction, with no
known exact recorded overlap; historical schedules remain incomplete. The fresh
16,384-decision teacher collection had 6 distinct-geometry finishes; the four
matched 131,072-decision student runs completed. Both screen gates passed on 24
canonical cells per actor: seed 10 selected RLPD/SAC finished 7/24 and 3/24; seed
11 finished 12/24 and 3/24. Both strict confirmations passed (3/32 vs 2/32, and
12/32 vs 7/32). The first confirmation command failed before validation/workers and
consumed zero cells; actor-specific, hash-linked projections of the immutable
screen rows then passed `previous_evaluation_metadata()` without replaying screen
cells. The preselected RLPD seed-11, 131,072-step finalist passed the 24-cell blind
with 9 canonical finishes (mean progress 0.679). All CPU reload/determinism/resource
checks passed. This remains internal CarRacing evidence, not an official score.
Full detail is at
[`pixel-rlpd-long-horizon-followup-v1-result`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json).

## Next RLPD Question

The frozen RLPD temperature convention remains `target_entropy=-1.5`; it is not
proven optimal. Entropy V1–V3 aborted before environment interaction and their
allocations are retired. V4 passed initial protocol/runtime/geometry checks, collected
fresh prior data and completed four matched runs, but the pre-screen source recheck
detected a changed `generalization-policy.md`; V4 evaluation never opened. V5 screen
gate passed all four target/seed minima with 29 finishes in 192 canonical episodes.
Fresh strict confirmations were 17/32 and 7/32 for the author target versus 6/32
and 6/32 for the +1.5 target; all four were eligible, deterministic, CPU-reload
identical, and had zero operational failures. Author-target passed the paired
dominance gate, and the frozen tie-break selected its seed-50 actor for the one
24-cell internal blind, which finished 7 episodes (mean progress 0.657). This
two-seed CarRacing ablation is internal proxy evidence, not official performance or
model confirmation. See the
[`V5 result`](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json).

The separately authorized [RLPD completion-first G0](../../experiments/rlpd-g0-completion-v1-result.json)
closed its fixed TRAIN-only diagnostic budget without learner updates or held-out
evaluation. Its 12 geometry clusters produced 24 paired observational episodes,
not a matched training-treatment comparison. Read-only trace review found no
unique initiating cause: two qualified nonfinishes were road-near and oppositely
headed long before 95% visited-tile progress, while an early contact/centerline
warning can also precede a real finish. The [fixed-window offline extraction](../../experiments/rlpd-g0-fixed-windows-v1-result.json)
sealed 144 slots across those same 24 consumed TRAIN traces, with 101 anchored
windows and 43 explicit missing anchors; all six successful-parent controls were
retained, but no manual or causal labels exist. A separate
[pixel-only motion score](../../experiments/rlpd-g0-pixel-motion-v1-result.json)
covered all 10,049 decisions; 74 retrospectively satisfy the contact-stall
telemetry definition and all six finishes remain in its control distribution.
Those labels occupy four failed episodes on three road geometries; their score
range overlaps finished-control decisions. The [gate decision](../../experiments/rlpd-g0-pixel-motion-v1-decision.json)
selects no threshold or policy. The [frozen-encoder representation probe](../../experiments/rlpd-visual-representation-probe-v2-result.json)
completed 256 CPU head-only updates with the actor unchanged, but all 188
diagnostic true positives came from two correlated episodes on ONE geometry and
none of its diagnostic roads supplied a finished parent. It can at most decode
the existing visual speed HUD on reused TRAIN roads, not justify an intervention.
A new source-hashed geometry-level positive and successful-parent coverage gate,
followed by separate exact-prefix parity/harm evidence, must precede G1. The
RLPD-specific parity comparator and original-G0 SHA binder now pass synthetic
state/action tests and file-only two-decision prefix checks on 24/24 consumed
TRAIN episodes, but **no real reset/replay or G1 branch** was executed; hidden
Box2D equality remains unproven. None of these observations proves recovery
data, value shaping, or memory is the causal fix. The r5 seed-audit erratum is
narrow and does not retroactively attest the malformed receipt.

The proposed next RLPD G1 coverage design fixes 24 new TRAIN geometries x two
unchanged actors, with V5 seed-50 source-primary only for its G0-informed
post-contact-stall feasibility question and seed-11 fully reported as comparator.
Pure synthetic coverage checks require all 48 actor-road slots and distinct
pixel-positive failure/finished-parent roads; they do not certify annotations
or road freshness. A separate G1 v2 seed auditor now checks candidate-specific
TRAIN collisions, consumption, reservations and exclusions; unrelated DrQ/Dreamer
JSON or ledger changes are provenance warnings, not collisions. The read-only
consumed-G0 control `--seed-start 4272000001` remains `BLOCKED` with 12 actual
consumed-road intersections, not a proposed G1 batch. Current r5 TRAIN ledger,
abort, supersession, metrics and trace bytes are independently checked without
validating the malformed historical receipt SHA or changing the G0 v1 audit.
The shared TRAIN claim registry re-audits under a lock if a future batch is
separately selected; legacy/other-lane allocation races remain a limitation.
No G1 seed batch, claim, frozen protocol or environment reset has been allocated.

Additional RLPD-only G1 preparation is synthetic: a blocked-by-default
collector skeleton checks future audit/claim/source identity and preserves all
48 attempted, censored or unrun slots; a separate image-review module seals
opaque pixel-only packets and restricted mappings before outcome joining. The
candidate-scoped G1 auditor now fails closed on candidate-relevant `{start,end}`
seed ranges (conservative inclusive overlap because endpoint semantics are
unknown), self-protocol exclusion overlap, and unknown exclusion fields. Its
28-test regression suite and the unchanged collector/coverage/image-review
suite (30 tests) pass synthetically on the new host; see the
[auditor correction result](../../talk/messages/20260928T081534Z-u3k9-rlpd-auditor-fix-result.md).
Destination hashes for the seed-11 actor and V5 seed-50 actor/checkpoint match
their recorded SHA receipts, but this is only a targeted subset: the full RLPD
`runs/` and protected `evaluations/` closure has no destination tree-checksum
manifest, and the `transferred/` RLPD/evaluation mirror has no readable payload.
The original proposed G1 design treated full restoration verification as an
execution gate; the separately directed new-host study below uses targeted
SHA-bound inputs without claiming whole-tree restoration.
These checks do not establish road freshness, genuine blinded annotation, or a
pixel trigger. The historical capped collector was disabled because a
mid-episode four-core-hour stop could not be enforced by the source-pinned
G0 `run_cell`;
positive/finished-parent G1 coverage also remained unestablished. At that
pre-2026-09-29 snapshot no G1 seed batch, claim, frozen protocol or reset
existed; the separately authorized live study is recorded below.

**2026-09-29 RLPD immediate TRAIN restart:** The user removed the *local*
four-core-hour research cap for a separate experiment, not for the already
disabled G1 collector. A new [source-bound protocol](../../experiments/rlpd-reused-train-immediate-v1.json)
ran exactly one already-consumed G0 track-1/4272000001 TRAIN episode using
the unchanged V5 seed-50 actor. [Old G0](../../runs/20260926-rlpd-g0-completion-v1/cells.jsonl)
recorded off_track/456; the [new run](../../runs/20260929-rlpd-reused-train-immediate-v1/result.json)
recorded finished/531. Initial pixels matched, but next frames first diverged
at zero-based decision 33; the tiny first native-action difference is observed,
not a causal explanation. The [bounded evidence result](../../experiments/rlpd-reused-train-immediate-v1-result.json)
binds both trace SHAs and limitations. No learner/policy update, new road,
G1 coverage, protected cell or official action occurred. Whole-tree RLPD
restoration remains unverified; the historical capped G1 collector was still
disabled at the time of this separate repeat.
Independent review found no current source/actor/trace hash mismatch, but
future v1 freezes do not enforce historical G0 source equality and an ENOSPC
failure can strand its attempt receipt. Preserve this completed source/protocol
unchanged; use a corrected new operator for any later repeat.

**2026-09-29 RLPD real G1 collection and learner complete, validation next:**
The [uncapped source-bound G1 TRAIN protocol](../../experiments/rlpd-g1-newhost-20260929-v1.json)
consumed 24 newly claimed track-1 roads, 48/48 source-actor episodes, zero
censored/unrun. The [result](../../experiments/rlpd-g1-newhost-20260929-v1-result.json)
and [manifest](../../runs/20260929-rlpd-g1-newhost-v1/manifest.json) bind 48/48
trace/receipt hashes: frozen V5 seed-50 finished 12/24, seed-11 11/24;
paired road outcomes were 7 both finish, 5 primary only, 4 comparator only,
8 neither. This is observational TRAIN collection, not a treatment comparison
or official score. No image review was sealed. At least five <209-decision
slots cannot provide the frozen anchor-200-plus-eight ordinary pixel window;
required `unknown` labels make this cohort's zero-unknown G1 coverage `pass`
unreachable. Do not retroactively change the rubric or top up any of its roads.

Separately the [new-host learner result](../../experiments/rlpd-newhost-reused-train-20260929-v1-result.json)
binds authentic V5 prior bytes, 33 current source hashes and 131,072 online
decisions/130,072 actor-critic gradient steps on the 12 already consumed G0
TRAIN roads. Both saved actor exports (65,536 and 131,072 decisions) and full
checkpoints match their SHA receipts and strict-load on CPU. These training
interactions, including 19 finished episodes during an evolving policy, are
NOT either frozen export's finish rate. G1 roads were not learner inputs.

**Post-training TRAIN sanity is negative:** The final new-host and V5 actors
passed bit-exact double CPU reload/native adapter preflight with no reset. The
[frozen same-host paired screen](../../experiments/rlpd-newhost-train-screen-20260929-v1-result.json)
completed 24/24 episodes on the 12 ALREADY CONSUMED G0 TRAIN roads, one V5
seed-50 and one new-host seed-52 final actor per matching road. The new actor
finished 1/12, V5 5/12 (1 both, 0 new-only, 4 V5-only, 7 neither), with
zero censored slots and independently verified cell/trace hashes. The negative
in-sample result does NOT identify a training cause or imply fresh-road
generalization: seed-52 learned online on these roads, V5 learned on a
different 4-track/64-seed pool and drove G0 later without updates. Historical
G0 V5 finished 3/12 vs 5/12 in this contemporaneous run with four road
outcome flips, so old G0 cannot replace the current comparator. Immediate
same-host first-40-decision traces show higher new-host |steer|/brake and
lower speed, but the resulting paths diverge. Exploratory no-reset CPU
inference on 10,049 identically reconstructed archived G0 TRAIN preaction
pixel stacks gives mean brake .2401 final131072 versus .0968 step65536;
the final action brakes more on 9,057/10,049 common inputs, without a
constant-action collapse. Step65536 has NO measured finish rate. The cause
of poorer final driving remains an untested early-control/retention hypothesis,
not a proven critic or adapter defect: V5 and newhost use the same actor-update
code and V5 also shows finite Q/gradient spikes. Do not promote the final
actor or open protected cells. A future *separately frozen* TRAIN treatment
must test whether it preserves V5 finishes as well as rescuing failures.
Historical whole-tree restore remains unverified; confirmation/blind/official
cells stay closed.

## Competition Schedule And Access

The first mock in the current site-verified schedule has passed; the next event
listed is the second mock. The old hostname failed DNS resolution on 2026-09-25,
but the [replacement competition site](https://scholarships-hardwood-headers-influenced.trycloudflare.com/)
was reachable on 2026-09-26 and its bundle still published the recorded dates.
The owner reports a higher daily submission quota than the bundle's fixed upload
text; verify the authenticated effective limit before any approved upload. Dates,
timezones, the site-access check, quota provenance, and the unverified
confirmation-cutoff report are recorded only in
[`docs/competition/info.md`](../competition/info.md). Recheck the official
schedule before any external action.

## Read Next

- Current candidate provenance and confirmation status:
  [`docs/results/MODEL_STATUS.md`](../results/MODEL_STATUS.md)
- Evaluation and partition discipline:
  [`docs/evaluation/protocol.md`](../evaluation/protocol.md) and
  [`docs/evaluation/generalization-policy.md`](../evaluation/generalization-policy.md)
- Durable evidence and prior decisions:
  [`docs/experiments/INDEX.md`](../experiments/INDEX.md) and
  [`docs/decisions/INDEX.md`](../decisions/INDEX.md)
- External actions and official rules:
  [`docs/competition/`](../competition/)
- Multi-agent discussion protocol and recent working messages:
  [`talk/README.md`](../../talk/README.md)

## Evidence Boundary

The DrQ-v2 and DreamerV3 statements above summarize frozen experiment records, not
new measurements. Consult the linked JSON protocol/result artifacts before making a
new comparison or changing a candidate status.
