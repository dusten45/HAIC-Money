# Rear-wheel torque feedback: negative physical evidence

Eight new three-second uniform-asphalt diagnostics use the unmodified official
car with decisions every .08 seconds. These are not timed laps or legal camera
benchmarks. The source and official hashes are bound in `torque-v1.json`.

An instantaneous speed/curvature torque budget and a proposed rear-wheel
rotation feedback both hold the measured .03-radian turns at about 100 m/s.
Both also hold a .06-radian turn when initialized at 100 m/s. When initialized
at 70 m/s and accelerating into that turn, both spin: maximum body slip is
110.7 degrees for the instantaneous budget and 72.5 degrees for wheel feedback.
The feedback case ends with only 61.0 m/s average over its last .6 seconds.

Raw rear rotation was supplied as an oracle, so camera quantization is not the
cause of this failure. Before the spin, rear rolling speed is close to body
speed; the rear feedback is too late to diagnose lateral saturation. Its
estimated tire-force allocation and acceleration history do not establish a
safe combined-force budget. Do not implement this formula as a camera agent.

Earlier source-bound fixed-gas diagnostics separately measured gas .3 at
70 m/s and .06 radians: 94.06 m/s final-window speed and 0.71 degrees maximum
slip. Gas .4 already produces 30.74 degrees, and gas .5 produces 120.36 degrees.
These are earlier measurements, not new validations. They motivate a new,
isolated camera test of propulsion based on the emitted joint request and
its projected demand at cruise speed. Current speed alone can underestimate
the load reached while continuing to accelerate into the same turn.
