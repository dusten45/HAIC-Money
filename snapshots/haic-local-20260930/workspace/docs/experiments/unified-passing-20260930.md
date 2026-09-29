# Unified passing comparison, 2026-09-30

Plan e6ecf8ecd654ea27f9be752054d66e3f14c50c65ae8c76f7fdf3b2f4e0a0ab65. Formal predecessor handoff-path-20260930. Preimplementation design approval and separate source-bound CLI design/execution approvals cite standing local user authorization. New source haic_agent/unified_passing_runtime.py and runner actor route. Fivearms x12 consumedTRAIN tracks1-3/seeds38300..38303=60episodes; no heldout/confirmation/blind use. Original stableZIP preserved.

|Arm|Finishes|Median finished ms|Changed actions|Contact records|
|---|---:|---:|---:|---:|
|Stable|12/12|20520|0|1|
|Unified passing target|10/12|20600|557|13|
|Image-relative offset forecast|8/12|20470|448|22|
|Lateral observation|12/12|20590|2460|2|
|Distance/curvature speed|12/12|23280|3329|0|

Peak speed of distance-speed70.145 does not compensate for slower lap median. Lower medians among partial completions cannot win completion-first comparison. All four mechanisms change actions; no weight sweep performed.

60rawhashes match;invalid/error0;first10equal;all12control entire action/trajectory/progress replay previousstable. 367.84seconds,2CPU2GiB,peakRSS339279872bytes,maxact9.902ms,p95about4.6-4.8ms. Full latency/create/reset/progress/P90 in batch_audit.json. Import time not separately measured. Read-only reviewer found tiny gas/brake overlap and requested telemetry; beforefreeze anypositivebrake excludesgas, replacement road/avoid and nominal inheritedavoid logged. Forecast is image-relative, not physical objectmotion; matching heuristics don't prove identity. Lateral action is bounded residual after parent clipping.

Prior1:38300 additive-side conflict no longer occurs: unified candidate finishes21.98s withoutcontact. But2:38301unified failsprogress.697 after firstcontactstep86(speed19.3). Atsteps83-85object liesright of observedroadcenter by3.99-5.16pixels while earlier passing_side remains+1; nominal inheritedside is-1 but its term is replaced, not summed. Replacementavoid+.234..+.34 competes with road curvature correction within the chosen path and steers-.042,-.024,-.10. Predicted counterpart also fails2:38301progress.690 and2:38302/.465,2:38303/.386. Removing duplicate avoiders resolves arithmetic opposition but does not establish feasibility of retainedside. Width/vehicle clearance and reachable turning path are still unmodeled. This is a feasibility hypothesis from traces, not proof of a unique physical cause.

Third valid comparable nonimprovement after guarded full12 and handoff full12 => PIVOT. Preserve historical checkpoint7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835 and all source/runtime artifacts. GatesUNKNOWN/PASS/FAIL; no release or package. Switch away from serial residual additions/initial-side locking: next design should evaluate multiple image-space paths against road/obstacle masks and observed turning response before choosing one. Independent directions to preregister: road/obstacle clearance path search; image-based motion/turn-response estimation; finite-horizon action rollout/scoring; speed planning on the selected feasible path. These are prospective directions, not implemented or registered. Whole-map coordinates remain unavailable and must not enter inference.
