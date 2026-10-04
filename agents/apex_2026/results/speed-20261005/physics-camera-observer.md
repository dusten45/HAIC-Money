# Camera dynamics observer diagnostic

This is an observe-only diagnostic of a causal camera algorithm. It is not a
submission, a timed lap, or fresh track coverage. There are zero new world
steps. Only the existing consumed track-3/seed-1007 prefix supplies120 camera
frames and recorded actions. Simulator labels are evaluated after prediction;
they never initialize or update the observer.

The observer resets body/wheel/control history to zero and uses fixed public
mechanical geometry. Each new frame supplies body-speed magnitude, body yaw,
front-joint angle and two isolated rear-wheel bars. Lateral velocity and front
wheel spin come from the four-tire model and past emitted actions. Camera
joint/rear measurements project that model state onto raster-error intervals,
rather than replacing an accurate motor history with a quantized point.

The body/joint calibration used1091 stationary official renders and zero world
steps. Joint error on the .002rad grid is at most .004816rad. Body magnitude
error on the quarter-m/s grid is at most1.501780m/s, so the explicit sensor
bound is1.6m/s. The initial1.5m/s assertion failed; its error is not rounded
down. The body and joint gauge regions passed sign, zero-frame and mutual
independence checks. Rear calibration is the earlier isolated two-column fit
with1.933074m/s individual rolling-speed error over its sampled operating
range. Yaw uses the earlier nine-render affine mass fit; that fit has no full
continuous error certificate.

For the94 frames labeled all-asphalt before and after an action with initial
speed at least20m/s, all six predeclared statistical screening bounds pass:

| Quantity | Observed absolute error | Screening bound |
| --- | ---: | ---: |
| Current sideslip90th percentile | .155357 degrees |1 degree |
| Current sideslip99th percentile | .301115 degrees |3 degrees |
| Next .08s speed90th percentile | .645930m/s |2.5m/s |
| Next .08s body yaw90th percentile | .025392rad/s |.2rad/s |
| Next .08s lateral velocity90th percentile | .054594m/s |1m/s |
| Next .08s max rear longitudinal-slip90th percentile | .435547m/s |2.5m/s |

Maximum healthy .08s errors are .815958m/s speed, .110293rad/s yaw,
.286494m/s lateral velocity and .734716m/s rear longitudinal slip. Current
front-joint history is especially accurate: maximum error .0000114rad over
all120 frames, well inside the joint gauge's resolution.

All120 rows remain in the receipt, including17 frames with at least one wheel
on grass. Full-prefix forecast maxima are .996258m/s speed, .274312rad/s yaw,
.751174m/s lateral velocity and2.383293m/s rear longitudinal slip. Current
sideslip's raw maximum52.66 degrees is the initial numerical heading of a
body moving at1e-10m/s. Two low-speed recovery frames produce6.33/4.39 degree
errors, which remain visible rather than being counted as healthy coverage.

The camera innovation gate does not certify asphalt: three grass-contact
frames at steps82/83/84, all above20m/s, pass it. State quality is sufficient
to motivate a bounded research planner test, but that gate cannot establish a
safe-control envelope. A legal planner needs camera road/contact confidence,
state-error perturbations and sampled hull clearance throughout prediction.
The body and rear gauges' combined pixel uncertainty can still imply hundreds
of newtons of longitudinal tire-force uncertainty. No control gain, general
track reliability or10–13s lap result is established here.

The source, all calibration receipts, camera NPZ and original inspection JSON
are bound by SHA256 in `physics-camera-observer-provenance.json`. Three
behavioral tests passed: cold gas/motor-step bounds, complete reset, and two
byte-identical120-frame runs with only camera/actions passed to the observer.
