# Committed camera path experiment

## Causal evidence

Frozen V2 `aab01ed34d45b2741e307675580d0588026e7e109aec83d174b1b462b9d3f0a8`
on consumed track 4, seed 18800, replans the same circle from actual ego
position each frame. At steps 110/111/112 the required peak curvature rises
from 0.02451 to 0.07147 to 0.15667 /m. The last exceeds the 0.13049 /m joint
limit and falls back. Only 0.16 simulated seconds separate the first and last
cameras. Later steps 128/132 also reject a single-valued cubic road with
residuals 5.17/4.81 m. These are diagnostic replays, not fresh validation.

## Hypothesis and bounded change

Create standalone `fast_preview_commit_agent.py` from exact frozen V2, leaving
V1/V2 and their tests unchanged. Retain its initial pass construction, full
vehicle half-width 1.6 m, road margin 0.30 m, official circle radius 1.2 m,
circle-hull margin 0.75 m, centerline separation 3.5 m, joint/force/steering
slew limits and V1 rear propulsion budget. Keep all clear-road defaults.

Once a path is accepted, transport that path rather than solve a new
ego-anchored Hermite pass. Decode signed rightward camera yaw from the red HUD
bar, using its complete 4.2-pixel height and 1.68 pixels per rad/s. Integrate
the average of consecutive yaw rates over the official 0.08-second action.
In metric coordinates (x right, y forward), apply standard positive rotation
by the rightward ego heading change: an old point ahead moves to the left.
Associate the
same observed circle only when its displacement differs by at most 2 m from
the HUD speed/yaw prediction. Use the associated observed circle displacement
to correct translation; this admits measured lateral slip without accessing
simulator state. Rigid transforms preserve the reference curve's curvature.

Trim the already passed prefix at the current ego projection; do not demand
that road now outside the visible rear support remains observed. Retain
parametric signed curvature from the original curve under the rigid transform.
Recheck the transported reference against the current parsed road and current
circle with every original threshold. Check the actual ego footprint too.
Reject non-monotone forward support, ambiguous/sparse road, inconsistent circle
association, tracking error over 1.5 m, heading error over 0.45 rad, or missing
current circle. Rejected commitments return to V2 planning and then inherited
behavior. No stale invisible circle is asserted safe.

Track the fixed curve with current-heading and lateral-error correction plus
local signed-curvature feedforward. Keep pursuit demand as a speed bound,
the actual emitted steering force bound and rear grip budget. This is a
camera geometric/feedback experiment, not a dynamic tire or lag certificate.
It does not fix the later single-valued road branch model.

## Verification sequence

1. Capture steps 110–112 through an exact frozen V2 action replay and save a
   compact camera-only regression fixture; do not label it new validation.
2. Write and observe RED tests for repeated-camera path commitment, HUD yaw
   area/sign, rigid-transform curvature, tracking and unchanged safety rejects.
3. Implement only the standalone candidate and its focused tests; rebind V2
   camera/geometry regressions. Obtain independent read-only review.
4. Freeze source/defaults/package and run exactly one fresh mandatory four-cell
   benchmark, workers 1, max steps 700, role benchmark. No holdout or full
   development run in this lane. Report every finish, contact and original
   13-second verdict separately. Parent owns commit/push and later selection.
