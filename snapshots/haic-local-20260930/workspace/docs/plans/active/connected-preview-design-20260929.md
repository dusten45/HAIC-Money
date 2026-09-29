# Connected image-path preview — heartbeat 14:25 UTC

Goal: preserve original stable ZIP completion while shortening laps toward half-time. Latest user identifies near-sighted steering and asks to exploit visible road/history. Prior pedal R2/R3 two non-improvements remain in history. This batch changes perception/steering architecture, not another pedal constant sweep. No active HAIC containers found at start. Deadline for this heartbeat14:45UTC; at most2CPU/2GiB and20minutes total. No external actions.

Design: add pixel-only road connected component tracing using NumPy/OpenCV (already provided). Threshold existing asphalt range, close3pixel holes, anchor component at road row54 nearcar, distance transform for center preference. A bounded beam follows the component in2D with smooth heading and no self loops. Trace can turn sideways and is not limited to x=f(y). HUD excluded; false connections/occlusion remain risks, no claimed true map access.

Four independent directions with one candidate each:
1 connected_pursuit: use a speed-dependent farther point on traced path for road steering, retain inherited obstacle/damping terms and pedals.
2 dual_preview: retain near-field road steering and add far-minus-near path bearing feedforward, retaining pedals.
3 motion_memory: connected pursuit plus optical-flow estimated rigid transform of past path, used during short detection loss if current pixel support exists. Feature transform is fitted in scaled camera coordinates; validate inliers/residual/displacement/rotation,3-frame maximum age. Only pixels, no rawyaw/coordinates.
4 shared_path: connected pursuit and a common path-derived straight acceleration window up to75, preserving stable pedals outside that window. Not unconditional fullgas or claimed85target.

Control is unchanged preview_row_repair behavior. All first10actions unchanged. Stable checkpoint SHA7bc38a3bdeab7cfdf437cfbf5c06f15a81cfa18fad5317748ab13a259b8ea835, never move reference. New files haic_agent/connected_preview_runtime.py; evaluator actor route only. No broad commits/worktree moves because required research dependencies are uncommitted; preserve existing work.

Register5arms×6consumedTRAIN cells tracks1–3/seeds38300,38302=30episodes,1200decisions/episode. Run timeout atmostremainingheartbeatbudget; profile outer1800s is not permission to exceed20min. Record path pixel points, selected target, reach distance, motionmemory use/quality, steering and pedal activation. Hash rawoutputs, compare control exacttrace and first10prefix, aggregate finish/time/incompleteprogress/collisions/damage/latency/reset/RSS. Accept mechanism only if farther path actually affects action, not from code presence. Endpoint6/6and faster thanstable21.59s, broader12cell/exactZIP required before any improvement/promotion claim. Failure means lostfinishes or no useful lookahead; third valid comparable non-improvement triggers PIVOT preserving strategy/checkpoint.

Implementation plan (executing-plans): create runtime and actor routing; scope read-only review; freeze exact source/plan hash; record design+execution standing approvals; execute registered CLI; diagnose representative failures; integrate exactly3gates; save next-action status. No unrelated unit tests. New package validation remainsUNKNOWN until actually done.

Standing local authorization2026-09-29 and heartbeat authorize design/implementation/run without questions. Per-design SHA recorded beforeimplementation, exact executableplan approvals beforeexecution.
