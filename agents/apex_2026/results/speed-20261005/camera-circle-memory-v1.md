The camera-only four-tire observer forecasts measured circle motion more
accurately than the inherited missing-circle transport in this fixed offline
study. This supports testing a model-consistent memory interface. It does not
establish an uncertainty bound, safe MPC policy, or faster lap.

The study reads 492 saved RearClear camera inputs: T1 first 82, T2 all 235,
T4 first 175. Every parent action exactly matches its canonical saved trace.
The observer receives only camera pixels and actual emitted legal actions.
There are no new world steps, laps, map inputs, holdout inputs or truth labels.
Frozen EarlyFar perception is used only to measure full component centroids;
the rejected EarlyFar control policy is not executed.

Each starting circle is transported for up to three actions. The first later
camera containing exactly one candidate within 6 m of any fixed model defines
one common comparison target for all models. This yields 91 associations,
86 at one action and five at two/three. They support temporal continuity, not
true object identity or semantic ground truth. Samples are correlated. The
six-meter gate could admit a wrong unique object, and unmatched cases are
reported separately. None of these 492 frames contains two detected circles;
multi-slot behavior has synthetic coverage only.

| Transport / forecast | Mean error | p90 | Maximum |
| --- | --- | --- | --- |
| Previous HUD, SE(2) | 0.628 m | 1.465 m | 2.559 m |
| Current HUD, exact inherited Euler | 0.445 m | 0.899 m | 1.638 m |
| Current HUD, SE(2) | 0.400 m | 0.854 m | 1.817 m |
| Previous/current endpoint mean, SE(2) | 0.268 m | 0.507 m | 1.196 m |
| Past-slope upcoming forecast, SE(2) | 0.660 m | 1.807 m | 2.542 m |
| Calibrated camera observer + four tires | 0.174 m | 0.285 m | 0.568 m |

Memory update at frame i transports frame i−1 to i. Both endpoint HUD samples
are then legally available; their mean is causal. The past-slope formula
`current + 0.5*(current−previous)` forecasts the upcoming interval i→i+1,
using no future pixels. These time alignments are separate in code and tests.
The endpoint mean improves average retrospective memory error but is not
uniformly better: T4 circle step 22→24 has current SE(2) error 0.476 m and
endpoint-mean error 1.196 m against the full current centroid. The upcoming
past-slope formula is worse than the constant previous HUD on this sample.

The physical forecast first observes the camera, then integrates four .02 s
steps using the actual emitted steer, gas and brake. Compound-center translation
is the sum of new velocity times .02; hull-origin translation also adds
`c−R(theta)*c`. World/model body yaw is negative HUD rightward yaw. A stationary
old point transports as `R(−theta)*(point−hull_translation)`. No world position
or angle is initialized from a simulator label. This pipeline also changes
HUD calibration and estimates lateral velocity, so its gain cannot be assigned
to one isolated component. All 91 cases have no camera innovation alarm; this
provides no failure-regime coverage or certified lateral-state bound.

The proposed memory keeps current and remembered circles as separate slots.
Each stores a camera-local center, uncertainty radius, missed count and internal
tracking ID. Before association, every old slot receives the one transform
consistent with the action actually sent. Unique gated matches may correct the
observed center; unmatched old slots remain, and ambiguous matches keep the
union of hypotheses and new observations. A different current object does not
erase a missing near object. An age of four actions alone cannot clear a hazard.
The research prototype exposes overflow beyond 12 slots and preserves the union;
it is not a bounded production implementation. A future bounded controller
must reject the optimistic plan on overflow or retain an aggregate occupied
region, rather than drop a possible obstacle silently. Unique geometric matching
is an identity hypothesis, not proof; its reliability needs further assessment.

For an old center p and nominal translation T, a conservative transport radius
would satisfy
`sigma_new >= sigma_old + epsilon_translation + 2*|p−T|*sin(epsilon_angle/2)`.
An observed circle starts with an assumed half-pixel raster radius, 0.471 m;
this is not measured ground-truth error. The observer's measured 0.568 m maximum
forecast residual cannot serve as a guaranteed process bound. A matched circle
can anchor translation for other slots using `new_anchor + R(turn)*(p−old_anchor)`,
but uncertainty must include both anchor errors and angular error times their
separation. No sensor or model uncertainty is reset merely because an innovation
gate was quiet. With no defensible lateral-motion bound, MPC cannot claim a
robust occupancy certificate.

Collision checks use the official 1.2 m radius and the rotated vehicle envelope
x±1.6 m, y[−2.4,2.6] m. In predicted body coordinates, closest solid-rectangle
distance is `hypot(max(abs(x)−1.6,0), max(−2.4−y,0,y−2.6))`. Inflate the circle
by requested margin, slot uncertainty, predicted translation uncertainty and
rotation uncertainty of the body envelope. Checking only corner distances
misses circles intersecting a body edge or its interior. The helper's tests
include these cases, a 90-degree body rotation and missing/ambiguous objects.
Raw-step endpoint checks alone are not a continuous swept-body proof; a future
planner needs swept interpolation bounds or a conservative segment envelope.

Obstacle relevance and braking distance must use the ordered path arc and every
candidate body pose, preserving competing projections around folds. Camera
forward y is insufficient: T4 step 132 shows a circle at y=27.631 m with ridge
projection s=56.309 m, while T2 step 65 has y=18.225 m and s=38.139 m. The ridge
may contain competing nearby arc projections. It begins 3.0–7.387 m from the
ego; adding that gap as a straight chord does not verify a reachable join. MPC
must start at the actual estimated ego pose/velocity/steering state, check its
body through that gap, and stop its usable horizon at the first unsupported
body footprint. These ordered camera references do not establish the true map
route or a safe return to an unobserved branch.

Time to contact should be the first time a predicted swept body intersects a
possible circle, not center-forward-distance divided by HUD magnitude. Inflate
both object and pose uncertainty when computing the earliest plausible contact;
an optimistic nominal contact time cannot define the braking deadline. On a
straight aligned approach with closing speed bounded away from zero, a spatial
uncertainty sigma corresponds approximately to sigma/closing-speed timing
uncertainty. That approximation fails when the object is laterally outside the
sweep or closing speed changes sign. T4 step 132 has HUD speed 82.83 m/s: y/v
is 0.334 s while ridge s/v is 0.680 s, even before body reach, braking, curved
tracking and the unverified ego join are considered. Neither ratio is an actual
contact prediction. Multiple projection branches require multiple contact-time
hypotheses rather than the earliest image row alone.

At 100 m/s a held action covers 8 m. Fully supported straight-view circle range
is about 32.604 m with current road row 5, so an illustrative one-action reserve,
front reach 2.6 m, circle radius 1.2 m and assumed raster uncertainty leave about
20.33 m before straight aligned contact. This is a budgeting illustration, not
a required no-brake latency or physical impossibility proof. Controls begin
immediately; the four-tire forecast must check the whole interval with shared
longitudinal/lateral force and motor lag. Persistent travel near 100 requires
earlier useful path commitment or smaller lateral displacement, rather than
merely replacing forward distance by a larger number or erasing a remembered
constraint. Earlier detection must affect only actual predicted hazard occupancy,
without automatically replacing a valid clear-road grip controller.

Two timing regressions first failed against current-only sample transport;
all eight focused research tests now pass. They cover timing, sign, rectangle
edge/interior, missing-object retention, association ambiguity, uncertainty
growth/reset and arc/fold projections. Source/input/calibration bindings are in
`camera-circle-memory-v1-lineage.json`. No standalone inference candidate was
created. The original ≤13 s target remains false for the tested driving agents.

Independent audit: every lineage and input hash matches. Two/three-gap results use intermediate camera/HUD/action updates as they become available; they are sequential causal updates, not a single initial-frame multistep forecast. The helper nearest-branch lists omit sufficiently farther branches and must not be treated as exhaustive collision support for predictive control.
