# TRAIN-only geometry speed teacher feasibility

Earlier V1–V5 full-state teachers did not produce a fast reliable lap on consumed TRAIN cells 1:43 and 2:102. This probe retained the stable-risk pixel actor's steering and obstacle response, while the TRAIN-only teacher used track curvature and simulator speed to plan gas and braking. Those privileged values are not submission inference inputs.

Both arms finished both TRAIN cells with zero collisions, damage or invalid actions. The control laps were 23.78 and 22.56 s; teacher laps were 23.38 and 21.86 s. The paired finished-lap median improved from 23.17 to 22.62 s, **2.37%**. The teacher changed gas 299 times and added braking 41 times. It missed the preregistered 3% feasibility threshold, so the v2 outcome is **REVISE → GATE_REVIEW_REVISE → STOPPED**, with no release, demonstration collection, student training, or candidate score claim.

The full trace suggests an experiment to isolate teacher braking: on track 1 at progress 0.7, the teacher speed was 30.2 versus the pixel control's 39.3 despite earlier gains. This association does not prove cause. The next registered TRAIN revision will preserve the geometry target and gas action but coast instead of adding brake when above target.

Evidence: [run report](../../../artifacts/haic-research-v2/haic2-geometry-speed-teacher-train-feasibility-20260929/report.json) and [gate report](../../../runs/haic-research-v2/haic2-geometry-speed-teacher-train-feasibility-20260929/integration_report.json). The frozen source is `tmp/haic2-geometry-speed-teacher-frozen-20260929/`.
