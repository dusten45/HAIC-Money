# TRAIN-only geometry coast teacher feasibility passed

The revised TRAIN-only teacher preserved the frozen pixel actor's steering and braking. It used privileged track geometry to decide when to add gas and when to coast, without adding any brake action. On already-consumed TRAIN cells 1:43 and 2:102, the pixel control and teacher each finished 2/2 with zero invalid actions. Paired laps were 23.78→22.34 s and 22.56→21.38 s. Median finished lap improved **23.17→21.86 s (5.65%)**, passing the preregistered 3% feasibility gate. The teacher raised gas on 263 decisions and coasted on 42 decisions.

The v2 record is `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED` without release. This is a teacher feasibility result on TRAIN, not a pixel-only student or submission-candidate score. The geometry teacher itself cannot enter a submission. The next gate is a frozen-teacher check across twelve newly assigned TRAIN cells before label collection.

Evidence: [full run report](../../../artifacts/haic-research-v2/haic2-geometry-speed-coast-train-feasibility-20260929/report.json) and [gate report](../../../runs/haic-research-v2/haic2-geometry-speed-coast-train-feasibility-20260929/integration_report.json). Frozen source: `tmp/haic2-geometry-speed-coast-frozen-20260929/`.
