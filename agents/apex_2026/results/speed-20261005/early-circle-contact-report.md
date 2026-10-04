# Remaining early-circle contact: detection flicker and solver memory erasure

Exact111-action saved-camera replay identifies the same official obstacle as
the original clear-supported V2 contact. Old contact97 is replaced by contacts
98/99; the first new hit preserves72m/s instead of the old21m/s. Detection
improved, but collision avoidance is still incomplete. Global trajectories
first diverge ataction46, so comparing equal numbered later frames is not a
controlled single-action comparison.

New first-obstacle encounter detects frame94 at30.57m, misses95, detects96 at
17.83m, misses97, then contacts98 at5m. At both misses the real antialiased core
haswidth2,height4 and metricaspect0.625, just below unchanged0.65shapegate.
Area and surrounding-asphalt gates pass. This establishes raster shape flicker;
no global shape threshold relaxation is introduced in this investigation.

The remembered-circle SE2 sign is consistent: positive camera yaw (right turn)
moves a stationary forward circle toward camera-left. `_PathReference._route`
transports it once using HUD speed/yaw, retains pass_side and increments the
missing count. Labels show residual centroid/transport errors; correct sign
is not exact pose estimation or a clearance guarantee.

The decisive route-memory bug lies in `_CorridorReference._route`. The base
routine computes a valid remembered shifted pass, but FISTA applies circle
constraints only to current detections. On a miss it optimizes that route
back toward the center without the remembered hazard constraint.

|saved miss|remembered base clearance|old optimizer clearance|old emitted steer|base remembered steer|
|---|---|---|---|---|
|95|3.70m|2.73m|+0.0038|+0.0181|
|97|3.72m|2.40m|−0.0424|+0.0460|

Clearance here is reference-center distance to the camera-transported circle;
it is not actual tyre clearance. At97 optimizer reference curvature is
−.00534/m, yielding feedforward−.0173 and feedback−.0250. The original base
pass has+.01487/m and emitspositive steering. Thus the path is erased before
feedback; yaw feedback alone does not explain the countersteer.

The new memory-constraint experiment preserves a transported valid pass as
an optimizer constraint. Camera regressions restore reference clearances
3.78/3.79m and positive steering, with exact visible-circle/expired-memory/
reset behavior. Its separate fresh four-cell receipt determines actual driving
outcomes. No new detector hysteresis, holdout or parameter grid is claimed.
