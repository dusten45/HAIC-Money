# V2 camera geometry and fallback localization

This is a read-only study of the completed mandatory V2/b70af action caches
for T2 and T4. The action-only cache audit already verified exact float32
actions, every post-action trace field and the original receipt's semantic
result. This study does not rerun predictive Agent.act, a control search or
a world. It calls the inherited camera parent once per saved frame, rebuilds
the observer causally from zero and advances memory with the actual logged
action. The separate privileged truth caches are not opened.

Reconstruction matches every recorded last-speed, last-target and emitted
steer field on all 217 T2 and 137 T4 frames. As an independent acceptance
check, the same reconstruction also matches every stored passive parent
field, camera sensor and post-action observer state on both full exact V1
replays (242 T2 plus 226 T4 frames). V2's unlogged full parent state is not
directly compared; the frozen prefix and recorded field checks support the
reconstruction. The candidate and builders stay unchanged.

| Camera branch | T2 | T4 |
| --- | ---: | ---: |
| Geometry supported | 96 | 24 |
| Strict-supported prefix too short | 8 | 3 |
| Initial full-body road unsupported | 5 | 0 |
| Ineligible because speed is below 20m/s | 108 | 110 |

Every supported decision's recorded action is in the frozen control grid
and differs from the exact reconstructed parent action. This is consistent
with the predictive branch under the source's two possible action choices;
the original planner's diagnostics are not rerun or directly recovered.
Every deterministic unsupported or low-speed decision emits the exact
reconstructed parent action. Grid membership alone does not establish
predictive acceptance.

T2 first detects the eventual contacted circle at step 109, about 28.05m
ahead, while camera speed is 81.46m/s. The strict-supported center-reference
end moves from 25.00m to 18.46m to 13.23m on steps 109-111, stopping before
the circle's paint/antialias pixels. Recorded controls brake 0.25, 0.55 and
1.0. At step 112 the circle is 12.35m ahead, but its paint makes the first
unsupported reference sample appear at 9.615m, leaving fewer than five
original nodes. Geometry is rejected before a body check or control search.
The inherited action now steers 0.2206 and accelerates at gas 0.6. Steps
113-115 remain unsupported and accelerate at gas 1.0, 1.0 and 0.16. Step 116
contacts the circle, with recorded brake 0.3436. A detector miss at 114
correctly retains the transported circle as a constraint.

T4 exhibits the same mechanism. The circle is detected at step 25, about
29.98m ahead at camera speed 53.97m/s. Reference support shrinks to 28.25,
23.24, 19.03, 14.76 and 11.53m on steps 25-29. Recorded braking on 27-29 is
0.25, 0.55 and 0.25. At 29 the predictive action steers -0.2393, while the
camera parent's active passing side is right. At 30 circle paint reduces
the prefix below five original nodes; inherited fallback emits steer
0.000665 and gas 1.0. The parent's steering slew uses the preceding actual
predictive steer, so the switch also changes the intended turn while
limiting how quickly it can do so. Step 31 emits steer 0.1314 and gas 1.0;
step 32 emits steer 0.3704 and brake 0.5472 and contacts the circle. Misses
at 29 and 31 retain transported memory. Contacts at 33, 34 and 39 occur
during subsequent low-speed fallback. No pre-first-contact innovation or
road-loss flag explains either track's switch.

These observations support a control-handoff investigation before another
lap: the checked emergency tail is discarded when reference support fails,
and inherited fallback resumes a different pedal/steering plan. A causal
backup would need retained predicted states/actions, coordinate transport,
current obstacle checks and explicit invalidation; the earlier feasibility
check is not a guarantee in a later observation. Continuing an unsupported
candidate or simply filling grass is not justified by these receipts.

A second independent fixed synthetic check identifies a geometry conflict.
The public straight road has half-width 40/6=6.6667m. The frozen full-body
road guard requires 1.9m beyond its lateral half-width of 1.6m, limiting an
aligned hull center to 3.1667m from the road center. At a centered circle's
longitudinal crossing, the independent center-distance guard requires at
least 3.7m. The compatible interval is empty by 0.5333m. Removing only that
center guard would still require 3.55m for the rectangle plus circle radius
and 0.75m margin, conflicting by 0.3833m. The exact exported frozen functions
confirm these slacks on an ideal observed straight-road field. This is a
guard incompatibility, not a physical impossibility proof or a mandatory
track-specific model. Margin accounting should distinguish known vehicle
footprint from separately measured camera/model uncertainty; any changed
guard needs explicit raster, swept-body and contact-bound validation.

Circle paint also blocks a reference that is being used only for ordered
progress, before the actual offset trajectory is searched. A future bounded
perception fix would need a calibrated, measured circle occlusion envelope
and preserved actual body/circle collision tests. These data do not authorize
an arbitrary disk fill or inference beyond the viewport. The original four
10-13s target remains unmet; these cached studies add no new driving result.
