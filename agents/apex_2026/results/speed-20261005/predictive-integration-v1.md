# Standalone predictive-control integration V1

The standalone `fast_predictive_agent.py` preserves the exact093aaa77 RearClear
source prefix and selects its new predictive child. It imports only NumPy,
math, copy and time. An offline AST builder assembles pure functions from the
frozen four-tire model84a91143, observer80d079c5, HUD calibrationdf8ee78e,
camera geometry565d398e and joint planner341c6157. No simulator, project or
runtime-file import is included in the generated candidate.

Each action calls the parent once to update existing camera/pass memory,
observes the current frame, evaluates72 fixed joint/pedal combinations over
.32s with an emergency stop after the executed .08s block, then advances the
observer only with the final emitted action. Current detections and a valid
parent-transported active circle are both constraints. The planner contains
no inherited scalar speed target or pursuit steering proposal. Three fixed
representative states perturb speed+/−1.6m/s, side+/−.35m/s, yaw+/−.15rad/s
and joint+/−.006rad with correlated rolling speed; they are not a complete
independent uncertainty box or a safety certificate.

The conservative planner road field is separate from the parent's hole-filled
field. It fills only actual dark pixels inside the known car footprint and
classified detected circle cores within1.2m, preserving bright grass islands
and unknown boundaries. The full25-point body stencil and circle rectangle
checks cover every predicted raw tick. A hull-chord velocity adapter corrects
the frozen geometry's COM/hull projection-speed mismatch without changing its
source. Existing dense ridge-chord support remains required. Unsupported
images, suspect innovations/clipped rear bars, yaw outside calibration and
speed below20m/s retain the parent action. A per-model-call wall-clock check
bounds planning to3s, then falls back; persistent observer advancement is
outside that planner timeout and still occurs once.

Ten meaningful integration tests were written and observed RED before code,
then GREEN in .68s. They cover pure-export equivalence, exact-prefix/rebuild,
causal observer equivalence, one parent/one final-action advance, retained
input pixels/grass island, rotating-COM hull progress, detached bounded
scenarios, low-speed/reset behavior, timeout fallback and finite invalid-input
behavior. The wider apex suite had412 passes and17 failures: all17 came from
the parallel replay helper's active RED tests while that helper was still
absent. These failures are not described as a green full suite. Root separately
reported32 focused predictive/component tests passed.

An offline four-frame action profile uses only the existing consumed track-3
camera prefix and its logged legal controls for observer history. It is not a
candidate rollout, lap, fresh coverage or privileged-state initialization:

| Saved step | Decision | Action | Time |
| --- | --- | --- | ---: |
|20 | predictive | steer−.15155, gas1 |2.143s |
|40 | predictive | steer−.00141, brake.25 |1.128s |
|60 | parent, unsupported reference | steer+.06401, gas.3 |.056s |
|100 | parent, speed below20 | steer−.01871, gas1 |.025s |

Three-scenario step40 requires more braking than root's earlier nominal-only
helper probe; that difference remains visible. No10–13s or faster lap claim
is supported by this profile. Runtime is a local loaded-host observation,
not an official5s certification; timeout selection can depend on host load.
The separately source-bound mandatory benchmark is run by root with this
candidate frozen. This integration receipt does not substitute for it.
