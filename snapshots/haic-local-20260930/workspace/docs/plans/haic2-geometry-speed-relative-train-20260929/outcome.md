# Fixed teacher on new TRAIN block: completion gate failed

On tracks 1–3 × seeds 5004–5007, the frozen geometry-coast teacher and stable-risk pixel control each validly finished 11/12. The teacher alone failed cell 1:5005, retiring off track at 73.0% progress after three obstacle collisions. Control finished that cell in 23.34 seconds. Control alone failed 2:5005; this does not cancel the teacher-only failure under the preregistered gate.

On the ten jointly completed cells, teacher median finished lap was 21.28 seconds versus control 22.40 seconds, a 5.0% improvement. The teacher added gas on 1,541 decisions, coasted on 351, added no brakes and produced zero invalid actions. This is a real speed mechanism but not a safe teacher for label collection under the registered completion-first rule. The v2 decision is `REVISE → GATE_REVIEW_REVISE → STOPPED` without release. These TRAIN cells are consumed; teacher results are not submission-candidate completion rates or official scores.

The failed teacher reached progress 0.73 at step 200, approached an obstacle to 8.89 units, then collided at steps 202, 204 and 205. The trace alone does not prove which earlier gas intervention caused the collision. A revised teacher needs an independent obstacle clearance mechanism or a new speed policy that preserves the pixel driver's obstacle trajectory, tested on fresh TRAIN cells.

Evidence: `artifacts/haic-research-v2/haic2-geometry-speed-teacher-relative-train-20260929/report.json` and `runs/haic-research-v2/haic2-geometry-speed-teacher-relative-train-20260929/integration_report.json`. Frozen source: `tmp/haic2-geometry-speed-teacher-relative-train-frozen-20260929/`.
