# Four-tire short-horizon predictor validation

Four fixed uniform-asphalt conditions compare an analytical forward model with
the unchanged official Car over 0.08 / 0.16 / 0.24 / 0.32 seconds. They are
privileged diagnostics, with **no maps, candidate changes, timed laps or
parameter grid**. Initial body and wheel state is exact simulator state.
This result does not validate a camera observer or a legal pedal controller.

The model includes the compound body/wheel center of mass and yaw inertia,
each wheel's local velocity and angle, the steering motor's 3 rad/s limit and
50/s proportional response, delayed joint movement after tire-force evaluation,
rear gas rise of 0.1 per raw step, engine power and wheel inertia, per-step
brake omega reduction, four individual 400 N friction circles, and the 100 m/s
translation limits. The approximation omits iterative Box2D joint/translation
constraint solving and wheel slip from off-asphalt or damaged conditions.

The four cases are moderate acceleration, tight-turn oversupply, settled
tight-turn braking, and reversal of a settled tight turn. Both initial
warmup and subsequent command are fixed in the source. No cases were selected
after seeing their results. The acceptance limits were also fixed before
execution: at 0.08 s, speed error at most 2 m/s, yaw error 0.1 rad/s, maximum
rear longitudinal-slip error 0.5 m/s and per-tire force-vector error 50 N;
at 0.32 s the corresponding limits are 3 m/s, 0.2 rad/s, 1 m/s and 100 N.

All four cases pass both fixed acceptance checks. Observed worst errors:

| Horizon | Speed | Yaw | Rear longitudinal slip | Per-tire force vector |
| --- | ---: | ---: | ---: | ---: |
| 0.08 s | 0.0342 m/s | 0.00762 rad/s | 0.0635 m/s | 8.14 N |
| 0.32 s | 0.0313 m/s | 0.0241 rad/s | 0.00754 m/s | 16.73 N |

The force comparison reconstructs instantaneous forces from the endpoint
wheel slips. It is not a comparison of the force integrated over the preceding
step. The report also records each final model step's tire demand including
the engine/brake omega update, which differs from that endpoint diagnostic.

Behavioral checks preserve an unforced straight 70 m/s rolling state for
0.32 seconds, verify brake-lock ordering before the wheel-force update, and
bound all modeled applied tire vectors by 400 N. Official/helper/source hashes
are checked before and after execution. The source and report remain frozen;
the root independently repeated all four accepted cases into an ignored
receipt. No earlier rejected scalar torque formula is rehabilitated here.

The result supports investigating stateful, short-horizon force prediction.
The next separate requirement is camera-only state quality: body speed/yaw,
rear rotation, joint angle and especially lateral body velocity. Exact
privileged initialization cannot be used in a submission. Observation error,
road/grip uncertainty and recovery must be measured before changing pedals.
