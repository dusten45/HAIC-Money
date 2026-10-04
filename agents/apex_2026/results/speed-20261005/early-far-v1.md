EarlyFar V1 inspects fully visible circles above the inherited row 10 boundary.
It copies RearClear V1 source `093aaa77…` exactly, renames its public class to
`_RearClearReference`, and appends one `_circles` override. Steering, throttle,
braking, defaults, route construction, ridge selection and pass memory are
unchanged. Candidate source SHA is
`1835002767a09f6f3839d6e9695c0a0d3b9b7ee47722ee06e7b7bad7cb4580fb`.

The causal evidence is saved legal camera input from exact RearClear replays.
All parent actions reproduce the canonical trace: track 1 first 82 inputs,
track 2 all 235 inputs, track 4 first 175 inputs. No new episode or holdout was used
for this perception analysis. Hashes and frame changes are in
`early-far-v1-perception.json`; the compact camera-only fixture is
`tests/fixtures/early-far-track2.npz`.

| Camera | Existing detection | New detection | Evidence |
| --- | --- | --- | --- |
| T2 step 198 | none | 32.334 m ahead | Fully visible 2×3 orange component at rows 7–9; next frame detects 25.279 m, a 7.05 m lead. |
| T2 step 166 | none | 32.502 m ahead | Fully visible 2×4 quantized core; legacy aspect 0.625 fails 0.65 threshold. Next-frame camera match error 0.243 m. |
| T2 step 180 | 30.864 m | 31.158 m | Same component crosses row 10; full centroid replaces the cropped centroid without duplication. |

The raw top camera row reaches 37.037 m, but that does not establish usable road
or whole-circle visibility. With current highest road support row 5, a 1.2 m
circle plus half a pixel on each axis limits fully supported center range to
32.604 m. At 100 m/s one action covers 8 m; crossing a hard row boundary can delay a
whole action despite the small increase in maximum measurable range.

The override scans only components starting above row 10, following their full
connected shape through the observed road region. It requires 4–20 pixels,
width 2–5, height 2–6, at least half-filled bounding area, orange peak ≥0.655,
radius-compatible pixel distances, no adjacent white curb, and at least 3 of 4
asphalt context probes. The known 1.2 m radius plus assumed half-pixel center
uncertainty must fit entirely inside current observed road support and image,
with at least 1 pixel extra margin at left/right road boundaries. Vertical
support requires that the radius envelope fits the observed row span, with no
additional 1-pixel margin; step 198 leaves about 0.459 pixel at the upper span.
No partial top or boundary-clipped circle
is accepted. Component size and radius support replace the old aspect gate for
these additional far components; existing near/mid detections retain the
original gates.

Existing hazards are preserved. Only an inherited centroid contained in the
same full component is replaced. Failed early checks return the exact parent
list, and all inherited missing-circle transport/reset behavior remains active.

The saved-camera audit changes 7 frames in 492 inputs: five earlier detections
and two cropped-centroid corrections. All have later inherited detections.
The T4 step 22 association error is 2.712 m using constant initial HUD yaw; it
falls to 1.402 m using trapezoidal consecutive HUD yaw/speed. The latter uses
future pixels solely for this offline diagnostic. It is not candidate input.
Camera continuity supports the labels but is not independent semantic ground
truth. T1/T4 inputs cover prefixes, not complete laps.

Before implementation, four behavioral tests failed against exact RearClear:
missed whole circle, quantized aspect rejection, cropped centroid and preserving
a simultaneous near hazard. After the override, all 8 new cases pass, together
with 33 relevant existing cases (41 total). Negative cases cover partial top
visibility, missing road support, radius crossing a road edge, grass, white
curb and elongated texture. Clear/reset/default/invalid actions match the parent.
All 6 inherited RearClear behavioral tests also pass when rebound to this new
candidate, including active-pass transport and post-pass ridge release.

Earlier awareness does not prove higher pace. Any detected circle also inhibits
the inherited ridge mode and starts its cooldown. The road-relative pass and
physical control budgets are unchanged; this is a perception experiment and a
geometric heuristic, not a dynamic safety certificate. Half-pixel centroid
uncertainty is an assumption. The bounded fresh screen rejects this integration.

| Mandatory track/seed | Fresh EarlyFar V1 | Fresh RearClear V1 | Contacts |
| --- | --- | --- | --- |
| 1 / 516237 | 15.60 s | 15.54 s | 0 |
| 2 / 644062 | 18.80 s | 18.72 s | 0 |
| 3 / 1007 | 16.74 s | 16.56 s | 0 |
| 4 / 18800 | DNF, progress 0.534050 | 18.14 s | 0 |

The benchmark used frozen source, defaults {}, workers 1, max 700 actions, and
exactly these 4 cells once. Track 4 retired `off_track` after 203 actions, with 98
fully off-road and 102 partial samples; it did not reach the action cap. All
worker errors are null. First 3 total 51.14 s versus parent 50.82 s (+0.32 s/0.63%).
Receipt `early-far-v1.json` and evaluator freeze bind the exact source. The
first action differences occur at steps 28/166/41/22; trace hashes and actions
are in `early-far-v1-lineage.json`. The saved traces alone do not establish the
later road-loss cause.

The detector observes useful whole-circle pixels earlier, but this isolated
integration provides no pace or completion gain. `screen_passed=false`, the
original ≤13 s goal is false, and the candidate is not adopted. No full
development or holdout suite was opened. The ZIP is an ignored research
artifact (10,963B, exact source member), with no official Linux certification.
