# Camera observer and four-tire integration contract

This is a source-extraction contract for a new research controller. The
research modules themselves are not submission files: they import paths,
Box2D and privileged diagnostic helpers. A legal standalone source must copy
only the pure camera/math components and fixed public mechanical constants.

Copy these pure functions from model SHA256
`84a91143f8d42db7cf1007586c5c13038c6e0e6d371624f262abcaba26e46064`:
`rotate`, `cross`, `tire_inputs`, `tire_forces`, `predict_step`. `summarize`
is optional for numerical diagnostics. Their required modules are NumPy,
math and copy, with `DT=.02`. Do not copy `body_state`, `make_car`, `tick`,
`run_case`, the receipt loader, simulator imports or command-line code.

Copy `joint_measurement`, `body_measurement`, `decode` from calibration source
`df8ee78e05d702a39395c4076129c551074d96fdd842cc08db7d40e2aa41edd1`,
and `CameraObserver` plus `PUBLIC_GEOMETRY`/`SENSOR_BOUNDS` from observer source
`80d079c5f46e064e1d1c5af2f3844911d8d3c7dfab125dd40db84bcc214dd008`.
Initialize the calibration with the following fixed stationary-render fit,
without runtime file reads or simulator imports:

```python
calibration = {
    'body_joint': {
        'body_slope': .08573317526988666,
        'body_intercept': .3065159749445597,
        'joint_positive': [53.32976668292053, .3237707931261848],
        'joint_negative': [53.33307758282946, .11603323404605852],
    },
    'rear_intercept': np.array([.020195027723669583, .00796898243261536]),
    'rear_inverse': np.array([[282.49406468720383, -6.042857175259085e-14],
                              [2.3915441163734697e-13, 309.2412452796282]]),
    'yaw_slope': 2.1639217019081123,
    'yaw_intercept': .151470661163329,
}
```

`observer.observe(frame84)` returns a detached state dictionary and sensor
diagnostics. `observer.advance(emitted_legal_action)` applies four raw model
steps and stores their result. Call `observe` before planning; evaluate every
hypothetical plan on copies. Only the final emitted action advances persistent
observer history. A fallback action must also advance that history. `reset`
clears all history to a stationary state; no label, seed, track ID, pose map
or simulator state is permitted as input.

At each `observe`, the local frame has x positive right and y positive forward,
with the current hull origin at[0,0]. State velocity is the compound-center
velocity: positive side velocity/sideslip means movement to the body's right.
`angle=0`; standard mathematical positive yaw rotates the car left, so
`state['yaw']=-HUD_camera_right_yaw`. Official joint target is
`-legal_action[0]`. State keys are `angle`, `velocity[2]`, `yaw`, `mass`,
`inertia`, `local_center[2]`, `wheel_offsets[4,2]`, `joint[4]`, `omega[4]`,
`gas[4]`, and `wheel_velocity_override=None`. Public geometry is mass
7.301919967kg, inertia19.170645978kg*m^2, local center[0,-.080459174]m,
and wheel anchors[+/-1.1,+1.6],[+/-1.1,-1.64]m minus that center.

To obtain a hull trajectory, initialize center displacement to zero. After
each raw `predict_step`, add `.02*state['velocity']`; reconstruct hull-origin
displacement as `delta_CM + c - rotate(c, state['angle'])`, where c is the
initial local center and initial angle is zero. For a camera-fixed obstacle
point p, its future body coordinates are
`rotate(p-delta_hull, -state['angle'])`. Retain world/camerainitial coordinates
for all path/hull comparisons; repeatedly mixing transformed camera points
with initial pose coordinates is incorrect.

The initial-pose joint target must remain a command, not a kinematic yaw
constraint: motor slew is3rad/s with proportional gain50; four raw steps allow
at most .24rad movement. Rear throttle ramps by .1 per raw step when rising
and falls immediately. Axle spin evolves before tire force. Brake at least.9
sets all axle spins to zero each raw step. Forces use the coupled400N tire
circle,82N/(m/s) slip stiffness and .54m radius. Body/wheel translation caps
are100m/s. Independent gas/brake and pure bicycle curvature formulas do not
reproduce these coupled dynamics.

The observer supplies a point estimate and raster intervals, not a formal
state-confidence certificate. Statistical healthy-prefix errors and its
three falsely accepted grass frames are reported in
`physics-camera-observer.md`. Rear/body pixel uncertainty can imply large
force errors. Low-speed sideslip, clipped meters, recovery, unknown surface,
contact/damage and road-support failures require bounded fallback behavior.
No controller should treat `dynamics_innovation_suspect=False` as proof of
asphalt or full hull clearance.

The .72s locked-brake tail has exact-state uniform-asphalt validation in
`physics-brake-tail-compact-v1.json`; it is sampled after the executed first
.08s action block. The straight100m/s stop travels21.833m and ends by .46s.
Its measured pose accuracy does not cover camera-state errors, grass or
contacts. The plan must preserve complete sampled hull/circle clearance,
including stopping-tail turns, camera margins and outside-viewport unknowns.
