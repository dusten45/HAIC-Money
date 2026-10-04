# Mandatory lap-time feasibility audit

The official four map resets were sampled through the normal 50 step warmup,
without sending an Agent action or running a timed episode. The clock starts at
physics time 1.02 s, after warmup, with hull speed effectively zero and two
already visited road tiles on each map. The finish tracker requires 95% visited
tiles and a forward start-line crossing after departure. Six official obstacles
appear on each map. Exact source hashes and geometry are in
`feasibility-audit.json`; the reproducible helper is
`research/speed_20261005/audit_feasibility.py`.

| Track (seed) | Tiles / required | Centerline length, m | Proven minimum, s | Conditional wheel reach, s |
| --- | ---: | ---: | ---: | ---: |
| 1 (516237) | 280 / 266 | 983.500 | 0.04 | 3.16 |
| 2 (644062) | 336 / 320 | 1179.500 | 0.04 | 3.46 |
| 3 (1007) | 309 / 294 | 1085.000 | 0.04 | 3.16 |
| 4 (18800) | 279 / 266 | 979.824 | 0.04 | 2.80 |

**Proof and its limit.** At the timed start, all four finish trackers have not
departed the start area. Their first subsequent update returns before it can
recognize a crossing, so the earliest possible recorded finish timestamp is
the second 0.02 s update. This is a rigorous but unhelpfully weak 0.04 s lower
bound. Qualification requires many more tile contacts in practice; this audit
does not establish a stronger unconditional physical lower bound.

Box2D exposes `b2_maxTranslation = 2 m` per body per 0.02 s step. Assuming
this also caps each wheel center's **total** position change, the tile polygon
reach calculation gives the conditional numbers above: enough unvisited tiles
must be within reach of one of the four wheels, allowing every wheel to travel
independently through obstacles and grass. The wheel fixture's 0.608 m
circumscribed radius is included. This is **not a certified physical bound**:
Box2D caps velocity-integrated translation, while contact and joint position
corrections can change positions later in the step. This audit did not prove a
bound on those corrections.

At a hypothetical constant 100 m/s along each centerline, travel alone takes
9.835 / 11.795 / 10.850 / 9.798 s. Adding the separately measured ideal straight
launch penalty of 1.108 s gives 10.943 / 12.903 / 11.958 / 10.906 s. These
figures are **route estimates, not minimum lap-time proofs**: the 95% tile rule
and broad road permit corner cutting, while turns, obstacles and tire dynamics
can add time. Neither the proven nor the conditional calculation excludes an
early 10 second lap or establishes that it can be achieved. A faster route and
stable legal-camera control must be demonstrated empirically.
