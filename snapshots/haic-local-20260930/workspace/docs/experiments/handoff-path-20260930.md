# Handoff/path mechanism comparison

Design and standing authorization: docs/plans/active/handoff-path-20260930-design.md and design-approval.json. Frozen plan c196aeaf10b00dc00a9ac8f30887aab2c4b2d143d332dfb6a63e692e079b636a, formal predecessor connected-guarded-extra6-20260930, separate CLI design/execution approvals. Four new mechanisms plus stablecontrol and frozen dual reference; tracks1-3 x consumedTRAIN38300..38303,72episodes. 2CPU2GiB1050seconds. Source changes: new haic_agent/handoff_path_runtime.py and actor route in training/fixed60_completion.py. Historical stable ZIP SHA verified7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835. No new ZIP/external action.

Reviewer review_fixed_speed, read-only, editedpaths[]: reset/API and control/reference appear consistent; found final impact-recovery frame and obstacle hold off-by-one, missing locked side and single-row fallback. Pre-freeze corrected pre-call impact gating, parent authorization acceleration, encounter side lock until>3missingframes, zero/one-row steering memory. Remembered steering includes temporal/obstacle components; it is not pure geometric heading. Activation and actual action changes logged separately.

Prior dual12 telemetry audit:74 transitions from preview-authorized to off; residual median0.0048647,max0.0706636; only1 exceeds.04. Near/middle row missing frames0. This weakens handoff discontinuity and missing-road hypotheses for that specific failure. Current registered comparison retained to measure activation; no retuning while running.

First observed early-gap failure1:38300: step254 speed52.7 detects object(y27.5,x37.2,roadcenter40.5), base side+1 and early gap right. Step255 base side flips-1, parentsteer-.367 but new gap's retainedright residual makes-.286. Step256 parent-.7 becomes-.58. Step257side+1;step258contact speed9.7. The additive new locked side and mutable parent side issue opposing commands. This is direct evidence of control conflict; not proof that every failure shares this cause. Next design should choose one geometric passing side/target and replace inherited avoidance, with explicit ablation, not sum independent opposing avoiders. No hidden simulator state used by controller; physical trace is offline diagnostic only.

Run currently executing; completion and gates pending.


## Completed comparison
72episodes completed671.59seconds,2CPU2GiB,peakRSS315969536bytes;maxact42.014ms,arm-p95about19.4-19.8ms. Import not separately measured; create/reset captured in raw/batch_audit. All72rawhashes valid,invalid/error0,first10equal. All24stable/dual full action/trajectory/progress replays match prior records. Heading arm exactly matches dual on12/12(heading_replay_audit).

|Arm|Completion|Median finished ms|New action changes|
|---|---:|---:|---:|
|stable control|12/12|20520|0|
|dual reference|12/12|20650|0|
|handoff decay|11/12|20380|2|
|early gap|8/12|19940|163|
|heading continuation|12/12|20650|0|
|straight acceleration|9/12|19300|908|

Incomplete-time medians are not improvements. Straight acceleration peak69.00 and nine finished laps do not offset three failures. Handoff two changes occur2:38302steps146/249 with obstacle y60.375/60.167; residual+.03066/-.01207 followed byofftrackprogress.9777 withoutcontact. Tiny intervention is not inherently safe. Heading branch inactive:12/12 is reference behavior, not recovery success.

Decision REVISE, gates ruleUNKNOWN/mechanismUNKNOWN(headinginactive)/outcomeFAIL. No release, no ZIP, no external actions. This is second full comparable non-improving guarded cycle after guarded full12, partial4 and full6+extra6 are not additional independent cycles. Preserve stable checkpoint and dual source. Next mechanism work should replace competing obstacle terms with one permitted geometric path, logging side selection and clearance; do not add another independent side command. Remaining independent directions to design: image-based obstacle motion prediction, lateral-velocity-aware path tracking, and braking-distance/curvature speed planning from the same selected path. These are design suggestions, not registered/approved executable plans. Registration must preserve consumedTRAIN identities and formal predecessor linkage; no reserved validation reuse.
