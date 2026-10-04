# Fixed batch control planner, V1

The independent NumPy helper batches the physics calculations while preserving
the frozen scalar policy. It changes neither the 72 constant controls, the
three supplied camera scenarios, the score, nor the first-0.08-second emergency
braking requirement. No standalone agent was edited and no episode was run.

The new helper is `research/speed_20261005/predictive_batch_control.py`, SHA256
`263909e1163fdce76e6704246e7a612c7b940f40dab89a34338bd1c57bfbaeca`.
Its complete scalar prefix equals frozen planner
`341c61575651774cd5b582c644e47da021fc27737f43cb606b1a57fa561caf99`
apart from renaming its class to `_ScalarPlannerReference`. Its batch model is
frozen `8ae7239b81d9b41e649e2ef704c3a6bbf3a8b2a933ed499b7bf13f566cb82d53`;
that kernel already passed separate scalar recurrence and independent audits.

Eight focused tests passed in 8.17 seconds. They cover exact float32 candidate
order, launch and 100 m/s scenarios, unavoidable-hazard fallback, first-stop
masking and unpadded backup geometry, exception/input fallback, and two actual
saved-camera cases. Every scalar result field on those finite cases matched;
commands and integer counts matched exactly, floating metrics within 1e-10.
An independent read-only audit found no formula discrepancy and confirmed the
class-prefix AST identity. Its two additional three-control/three-state
differentials matched every scalar result in 0.244 seconds.

Each action/scenario pair advances the constant command for 16 raw ticks. The
backup starts from tick four, holds the same actual front-joint target after
the same float32 legal-command round trip, and uses gas zero/brake exactly one.
Stopped rows freeze immediately; geometry receives only the actual prefix and
tail up to the first stop, without padded poses. Center-of-mass integration and
conversion back to the hull origin match the scalar predictor. Logical model
steps retain scalar visited-scenario counts; actual simultaneous work has
separate `batch_calls` and `batched_model_rows` diagnostics.

The fixed profile includes field construction and all planner geometry checks:

| Supplied case | Scalar total | Batch total | Speedup | Feasible controls |
| --- | ---: | ---: | ---: | ---: |
| Saved camera, action 20, three scenarios | 2030.52 ms | 857.80 ms | 2.37x | 56 |
| Saved camera, action 40, three scenarios | 1081.60 ms | 657.92 ms | 1.64x | 16 |
| Straight 100 m/s, three scenarios | 1083.39 ms | 630.03 ms | 1.72x | 18 |

The actual-camera fixture is an export from the existing clear-supported T3
120-action camera prefix. It is not a replay of the new predictive agent.
The observer reconstructs state from cameras and past logged legal actions;
no diagnostic truth initializes or corrects it. Field construction is rerun
from each saved camera and exactly matches the exported strict field. These
timers exclude camera decoding, observer update, ridge extraction and scenario
construction. One measurement per fixed case on this shared host does not
prove a worst-case latency bound. Per-trajectory geometry now dominates the
remaining planner cost.

The successful differential contract assumes pure, finite model and geometry
callbacks. Batched geometry evaluates additional rows eagerly: a callback
exception on a scalar-unvisited row can produce a different `planning_error`
and failure-step count. Both return `None` for the caller's legal baseline.
No grass/contact certificate, adoption, or simulated lap-time improvement is
claimed. The original early-10-second goal remains unmet.

## Standalone extraction contract

The planner constructor is `FixedBatchControlPlanner(predict_batch, geometry)`;
`plan(supplied_states, previous_float32_action)` has the existing scalar result
fields plus the two batch telemetry fields. State/scenario shapes and legal
action sign remain unchanged. The supplied callback receives arrays with a
leading batch axis and physical target `-legal_steer`, gas and brake; it
returns batch state and diagnostic arrays. Its state and commands are copies.

For the batch physics callback, copy `BATCH_DT`, `BATCH_ARRAY_SHAPES`,
`BATCH_SCALARS`, `_copy_batch`, `_rotate_batch`, and `predict_batch` from the
frozen batch model. `pack_states` and `unpack_state` are research conveniences
and are not needed by this planner. Copy `_ScalarPlannerReference` and
`FixedBatchControlPlanner` from this helper; they require only NumPy and the
supplied callbacks. Keep the existing hull-velocity geometry adapter and the
observer's final-emitted-action-only advance. A deadline wrapper should check
each batch callback; `model_steps` is logical scalar-equivalent work, not the
number of callback invocations.

The verification receipt pins source, tests, fixture and profile artifacts.
The ignored source freeze is
`.haic-artifacts/apex-speed-20261005/predictive-batch-control-v1/`.
