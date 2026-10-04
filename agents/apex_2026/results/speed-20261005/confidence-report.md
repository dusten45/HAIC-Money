# Deliberate camera road-support confidence

Why did graph V1 finish four tracks while correcting the gray car seed in
V2 lost three? Exact camera replays answer one causal part of this question.
The two track4 trajectories and actions are identical through action69.
At action70 the nearest asphalt-like pixel at camera row63,column43 belongs
to an isolated two-pixel, two-row gray car component. Initial graph V1 has
no usable road from that component and invokes bounded recovery. V2 selects
the larger road component, so it continues ordinary driving.

V1 applies left steer −0.294 and brake0.35 for three actions, reducing speed
from50.6 to24.3m/s. V2 steers−0.170 initially and uses gas0.30, increasing
speed to53.8m/s. Before divergence, interval-average sideslip is approximately
−3degrees, not a large slide. The image shows the car at the outer road edge:
a remote road component does not prove road support at the ego origin. The
isolated gray fragment was an accidental confidence warning, and correcting
perception removed that warning. Subsequent trajectories then differ, so
this does not prove every later failure has the same cause.

`graph_confidence_probe.py` reproduces all250 V1 and199 V2 actions exactly
from saved camera frames, with component support statistics. Its accuracy
check requires no new driving. Obtaining those two saved-frame replays used
the already consumed required track4 geometry, with exact original actions.
No holdout is opened or simulator state supplied to inference.

`fast_confidence_agent.py` embeds frozen graph V2 and deliberately checks
whether the nearest ego asphalt-like component contains at least24 pixels
spanning five sampled road rows (eight pixel row difference). Otherwise it
invokes the existing bounded recovery action. Major road parsing remains
V2's corrected component logic. This explicitly restores road-support
confidence instead of silently treating an isolated car fragment as a road.
The gate remains a camera heuristic, not a formal support certificate.

Saved action70 regression fails on V2 (gas0.30) and passes after the confidence
gate (gas0, brake0.35, left recovery steer). Synthetic connected road with a
car occlusion remains valid and accelerates. Ten focused tests pass.

Fresh source-frozen mandatory benchmark:

| track/seed | actual official lap | contacts |
| --- | --- | --- |
| 1 / 516237 | 15.04s | 0 |
| 2 / 644062 | 24.22s | 1 |
| 3 / 1007 | 19.86s | 0 |
| 4 / 18800 | 19.96s | 2 |

All four official finishes reproduce initial graph V1's exact action hashes.
This is a fresh verification of a deliberately expressed mechanism, not a
speed improvement over V1. The strict10–13s goal remains unmet; candidate is
not adopted. Further work should prevent support loss using accurate curve
geometry and feasible trajectories, rather than relying on late recovery.
