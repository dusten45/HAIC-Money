# Supplied camera-state joint controller

`research/speed_20261005/predictive_control.py` contains one NumPy-only class,
`FixedControlPlanner(predict_step, geometry)`. It accepts supplied camera-only
four-tire state scenarios and a pure prediction callback; it imports no project,
simulator, map or observation labels. Root integrates the separately verified
camera observer, geometry and public mechanical model into a standalone agent.
The helper cannot establish that its supplied state scenarios cover every
observation error.

`plan(states, previous_action)` returns a float32 three-vector if feasible, or
`action=None` with a reason for the existing legal agent's fallback. Invalid
state, a wheel-velocity override, invalid predictor output, or prediction
exceptions return `None`. The caller must advance its observer with the final
emitted action, including fallback. One to seven scenarios are supported.

The first prototype compares at most 72 constant actions: nine steering
offsets `{0, ±.03, ±.06, ±.12, ±.24}` from the previous legal command, clipped
to ±.4 and the .24 change bound, paired with eight separate pedal modes:
coast, gas .16/.3/.6/1, and brake .25/.55/1. Full brake is exactly float32 1.0.
Using float32 .9 would fall below the predictor's Python `>=.9` lock threshold;
that representation error was independently caught, reproduced RED and fixed.
The superseded source and profile are retained with a reversible patch.

Every action uses the supplied four-tire engine/omega/gas/joint prediction,
including the 3 rad/s steering motor and delayed joint movement. Its physical
joint target is the negative legal steer. Camera coordinates are x-right and
y-forward, while model angle and yaw are conventional CCW. Compound COM is
integrated with post-step velocity, then hull position is recovered as
`COM - R(angle) × local_center`. Camera body tests use that hull position.

There are 16 raw .02 second steps (a .32 second prediction). All supplied
scenarios must retain nonnegative raw-tick road body, rotated-rectangle circle,
and supported-reference-front slacks. The geometry checks the complete sampled
body, not only wheel or center points, and circle clearance includes rectangle
to circle distance. Progress follows a reachable ordered ridge arc rather than
camera y order. There is no old pursuit command, reference-speed ceiling, or
road-center rejoin objective.

The score primarily maximizes the minimum scenario's progress. Small soft
costs penalize excess unsaturated tire demand, rear rolling slip, steering
change and braking. Infeasible candidates are never promoted by their score.
If none remain, the helper returns `None`; it does not invent a recovery action.

Emergency stop feasibility starts after the **first executed .08 second block**.
It does not start after the unexecuted .32 second prediction, which would
artificially slow a receding-horizon controller on a short camera view. Backup
uses gas zero and exact locking brake 1.0, holding the actual predicted front
joint (with the command change bound), for at most .64 seconds. Each backup
pose must remain within the same supported geometry and known ridge endpoint.
Low-motion completion means speed ≤.5 m/s and |yaw| ≤.1 rad/s.

This backup assumes uniform asphalt and the supplied model/observer scenarios.
It is not a grip, damage, contact, steering-lag uncertainty, or collision
certificate. Physical-model brake-tail validation is a separate exact-state
diagnostic. Camera innovations and insufficient road confidence still require
fallback in the integrated agent.

The meaningful unit cases cover sign and motor mapping, input preservation,
real launch propulsion, feasible 100 m/s motion with a first-block stop inside
a 35 m straight reference, an unavoidable near circle returning `None`, every
supplied adverse scenario, float32 brake locking, and invalid/model-error
fallback. Synthetic local profiles include building the camera field and ridge;
they measure this host only and cannot establish official worst-case latency.
No world steps, timed laps, parameter grid or holdout is part of this helper
study. No pace-goal or agent-adoption claim follows from these unit checks.

Final helper source is
`341c61575651774cd5b582c644e47da021fc27737f43cb606b1a57fa561caf99`, with
corrected geometry
`565d398e4e55c1e8bb5dad5d38e6ffa2c1a42eb5a8675832c2103f1197741920`.
The geometry bounds arc projection by the maximum prior/post speed so it
matches semi-implicit pose integration without an added projection reserve.
Eighteen focused tests passed. Fixed profiles including camera field/ridge
construction measured 558 ms (launch, one scenario), 1574 ms (70 m/s, three),
1155 ms (100 m/s, three), and 3249 ms (70 m/s, seven). The 100 m/s case retains
100 m/s at .32 seconds, while the first-block backup stops at progress
29.832 m in .46 seconds, within the observed 35 m reference.

Those profiles do not prove worst-case compliance. The finite model-call
ceiling is `72 × scenarios × (16 + 32)`; seven scenarios could require 24,192
raw predictions in unusually permissive geometry. The integrated agent should
use a bounded small scenario set and an external execution deadline/call
budget, preserving `None` fallback before the official five-second allowance.
The prototype's scenario perturbations do not cover body-speed or rear-raster
uncertainty and are not a certified uncertainty set.
