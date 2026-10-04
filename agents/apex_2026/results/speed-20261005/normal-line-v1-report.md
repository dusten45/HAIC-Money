# Supported normal-line V1: completed, target unmet

Frozen standalone source `fast_normal_line_agent.py` SHA256
`0c937c6201c12df110bf43940c030b6adfae40c29274acfd6e285c3db08f7f4f`
embeds exact RearClear parent `093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`.
Only `_ridge` changes; all act/steering feedback/force bounds/hazard gates/recovery
and rear-pass clearance transition remain inherited.

## Causal investigation

One exact mandatory T2 source-bound replay saved235 cameras in ignored
`.haic-artifacts/apex-speed-20261005/rear-geometry-track2/`.
Both driving replay and offline instrumented controller asserted235 identical
float32 actions against RearClear's benchmark trace. These checks are diagnostic
replay, not another benchmark. The privileged road and full-track point-model
route enter labels only; neither enters candidate inference.

The controller used ridge on176 frames and confidence on59. Clear low-target
frames54/161 show large path-fit curvature errors, but their actual raw path
target is100: final speed caps come from pursuit/reference curvature and prior
target+6 ramp. Both corridor solves are infeasible there. Changing the QP's
second hard heading anchor would therefore not address these cases.

Most low-speed hairpins have genuine local road curvature around0.08–0.09/m;
center-following ridge caps around45–50m/s have physical geometric support.
There is still room to select a line within road width. The full-track optimized
point route is an optimistic local optimum, not an actual lap, lower bound,
camera candidate, or evidence that the original time goal is impossible.

Saved frame150 supplies a supported path intervention: bounded normal offsets
reduce10m local fitted curvature from0.08966 to0.06197/m. An important ablation
separates this from arclength timing: the original ridge begins3m ahead but uses
that point as arc0; raw target48.45m/s becomes68.86 just by prepending ego origin.
Optimized path plus origin gives68.55. Thus the cap improvement is largely arc
origin/window alignment, not solely curvature optimization. Normal optimization
also gives full modeled body-corner camera depth4.13m; raw/origin-only path has
unknown image-end depth0 and is rejected by the proposed final guard.

## Single candidate

The base ridge retains its exact5.8m support guard. Resample that path at2m,
optimize normal offsets bounded±3m using NumPy projected accelerated quadratic
descent (800 iterations), with camera ego position/heading in the bending
objective and a weak offset penalty. Prepend ego origin so path arc starts at
the actual ego. Sample oriented body corners at lateral±1.6m and longitudinal
−2.4/+2.6m along every chord at spacing≤0.5m. Require camera depth≥1.9 throughout
the entire path; unknown image ends reject to the exact parent ridge. No partial
end checking, global circle relaxations, force changes, or blind parameter grid.

The235-camera audit admits13/176 ridge references and rejects163 exactly to
parent. This conservative support explains limited opportunity. Corner checks
use modeled path tangent and remain a camera heuristic: tire slip, steering
lag and actual body tracking are not certified.

The saved-frame target/clearance test failed against the unchanged parent
(48.45 versus required>56.45) before implementation. Four new focused tests
passed;20 focused pytest tests passed including parent arc/rear/support tests.
Regression checks cover real frame150, clipped frame83 exact fallback,
analytic straight preservation, reset and nonfinite observation recovery.

## Fresh mandatory benchmark

Exactly one fresh four-track benchmark, workers2/maxsteps700, source frozen:

| Track / seed | RearClear parent seconds | Normal V1 seconds | Contacts |
|---|---:|---:|---:|
|1 /516237|15.54|15.22|0|
|2 /644062|18.72|19.96|0|
|3 /1007|16.56|16.48|0|
|4 /18800|18.14|19.16|0|

All four actually finished. Original10–13s target remains unmet. The candidate
regresses T2 by1.24s and T4 by1.02s, so it is not adopted. Small T1/T3 gains do
not outweigh those regressions. Existing parent results remain their original
fresh runs; none is described as new validation.

Trace-only localization finds the T2 candidate initially leads0.16s at progress
0.6, then trails1.12s at0.7. Steps158–170 contain repeated inherited recovery
actions, braking nearly to zero twice. The receipt reports0 full-offtrack but13
partial-offtrack samples. Zero obstacle contacts therefore does not imply clean
road tracking. An exact new173-action camera prefix investigates this recovery
onset separately; no second candidate or benchmark is being run.

Receipt `normal-line-v1.json`, compact camera audit, lineage and zero-context
source patch accompany this report. No holdout opened, official/root sources
edited, or Git operations performed by this lane.
