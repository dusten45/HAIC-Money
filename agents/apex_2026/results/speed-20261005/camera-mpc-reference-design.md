# Camera reference for predictive control research

Helper `research/speed_20261005/camera_mpc_reference.py` is NumPy-only and contains
no agent, map/seed access, simulator import or privileged lookup. It is supplied
geometry for the physics rollout planner, not a tested driving candidate.

## Coordinate and reference contract

Coordinates are camera x-right/y-forward in meters. Predicted angle is
conventional counterclockwise ψ: forward rotates to[−sinψ,cosψ]. Body offsets
therefore rotate to(lat cosψ−long sinψ,lat sinψ+long cosψ). Rollout velocities are
camera-world vectors, not body coordinates. Detected circles must convert from
detector(y,x) to helper(x,y). Transported circle memory can be supplied with an
explicit positional uncertainty; absent/offscreen circles are not asserted safe.

Perception supplies the ordered, supported parametric ridge and camera distance
field. Base ridge support remains a caller requirement, such as the established
5.8m prefix guard. The reference begins around3m ahead; a backward extension of
only its first tangent defines ego's progress origin. That extension invents no
asphalt and does not expand body support. Projection never globally associates
the closest hairpin branch: each tick searches only the monotone arc interval
reachable from previous projection under predicted speed×dt plus explicitly
supplied uncertainty. This avoids nearest-return-branch jumps.

Use along-reference progress as the objective. Lateral offset is returned for
diagnostics, without a strong centerline error cost; any supported lateral line
can progress. This avoids the existing artificial rapid road-center rejoin.
The ordered reference is frozen throughout all beam knots in one planning call.
Evaluate full prefixes, including their initial state, to preserve association
across knots. New-camera temporal branch association remains perception work,
not solved by this local projection helper.

## Costs and endpoint

`CameraGeometry.evaluate(positions[N,2], angles[N], velocities[N,2], dt=.02)`
returns progress_m, road_slack_m, obstacle_slack_m, reference_front_slack_m and
violation arrays, plus path tangents/lateral offsets and remaining support.

Road support samples a25-point oriented rectangle grid covering edges, corners
and interior at every predicted raw tick. Lateral bounds±1.6m; longitudinal
bounds−2.4/+2.6m. Camera depth requires1.9m. Sampling outside the image gives0;
an optional camera grass mask can override filled/occluded distance-field cells.
Mask generation/confidence remains the caller's perception responsibility.

Obstacles require BOTH center separation≥3.7m+uncertainty and actual rotated
rectangle-to-circle separation≥0.75m+uncertainty, using radius1.2m. A3.7m center
disk alone can miss a front-corner threat, hence the second check. Rectangle
clearance assumes the supplied circle center/radius and is not perception proof.

Supported endpoint slack equals supported extent−progress−actual yaw-dependent
body front support along the local reference tangent. It cannot be replaced by
merely finding asphalt beyond the supported prefix. The planner must reject any
tick with negative road, obstacle or endpoint slack. Initial pose can legitimately
be unsupported, producing no feasible rollout instead of inventing safety.

For a terminal stop, brakeagent proposes a model-based gas0/brake0.9 tail with
steering held at actual front joint. Every raw tail tick must pass the SAME body,
obstacle and endpoint checks until speed≤2m/s; inference beyond the model's
validated0.32s horizon remains explicitly unvalidated. The helper also exposes
a scalar remaining-distance speed bound with supplied braking acceleration,
delay and uncertainty; that bound is only an assumption check, not a stopping
certificate or replacement for the model tail.

## Verification and limits

Nine focused tests pass: off-center straight progress, synthetic hairpin branch
ambiguity, CCW body yaw, interior grass and unknown image, rectangle obstacle
threat, terminal front/delay reserve, actual saved149 unsafe arc, frozen wrapper/
nonfinite states, and supported prefix end even when more asphalt is visible.
`camera-mpc-reference-proof.json` binds helper/test/camera fixture hashes and
records saved149/154 constant-arc geometry. No driving episode, fresh benchmark,
candidate source modification or holdout was performed for this design.

This improves the definition of a camera cost, not dynamic accuracy or lap time.
Sparse raw-tick sampling, uncertain segmentation, filled small image holes,
reference topology mistakes, tire slip and longer stopping-tail model drift
remain limitations. The standalone four-tire model and legal observer require
their own validation; none is imported into this geometry helper.

## V2 integration correction

The frozen V1 helper82c8d5a5 used prior speed×dt for the projection window.
The four-tire model integrates position with post-step velocity; during
acceleration, V1 could therefore pin projection behind the actual predicted pose
and overstate supported endpoint margin. V2 uses
max(prior speed,post speed)×dt plus supplied uncertainty. This follows integration
and introduces no driving parameter tuning. Helper SHA256
`565d398e4e55c1e8bb5dad5d38e6ffa2c1a42eb5a8675832c2103f1197741920`.

A new high-acceleration synthetic test fails V1: predicted2m displacement is
credited0m, leaving optimistic endpoint slack+1.4m. V2 credits2m and exposes
slack−0.6m. Ten tests now pass. `camera-mpc-reference-v2-proof.json` binds source,
test and zero-context V1→V2 patch; the original proof remains a V1 receipt.
Exact V1 source/report/proof are recoverable in ignored
`.haic-artifacts/apex-speed-20261005/camera-mpc-reference-v1/` and the reverse
patch reconstructs the original source. No candidate edits or driving episodes
were performed for this correction.
