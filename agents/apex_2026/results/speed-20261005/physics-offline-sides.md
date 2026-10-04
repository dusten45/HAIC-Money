# Alternative passing sides: offline diagnostic

No driving episodes or world steps were run. This study uses privileged full
road and obstacle geometry. It is not legal inference, a global time optimum,
a lap certificate, or a physical lower bound. Frozen agents and prior research
sources/results remain unchanged. SciPy 1.14.1 is optional research only.

The model remains 100 m/s maximum speed, lateral acceleration 219 m/s²,
forward acceleration 40 m/s², braking 150 m/s², initial speed zero, road-normal
point offsets ±4.8 m and obstacle-center segment clearance 3.7 m. The independent
acceleration limits omit tire coupling, hull sweep and tracking error.

We considered 2 / 4 / 2 / 4 side combinations. Eligibility is determined by
the point-car road-normal cross-section at each obstacle; it does not enumerate
all world-space trajectories. Each family first minimizes a convex curvature
approximation, then runs a local time search with at most 300 iterations.
Circle supporting halfplanes restrict each search to its chosen passage side.

| Track | Prior fixed-side time model | Best new side family | Local time solver | Segment-circle clearance |
| --- | ---: | ---: | --- | ---: |
| 1, seed 516237 | 12.067 s | 12.021 s | Converged | 3.700 m |
| 2, seed 644062 | 15.123 s | 15.064 s | Iteration limit | 3.700 m |
| 3, seed 1007 | 13.623 s | 13.579 s | Iteration limit | 3.700 m |
| 4, seed 18800 | 13.236 s | 13.143 s | Converged | 3.844 m |

Exact segment clearance and offset bounds were checked for all accepted
families with 1e-9 m numerical tolerance. The smaller changes on tracks 1 and 3
also reflect different initial routes and supporting planes, not solely a
change in obstacle side. The required all-track model time below 13 s was not
reached, so no additional teacher laps were scheduled.

The compact `physics-racingline-sides-v1.json` retains each best route and the
scalar results for all 12 families. Full trial coordinates and original
per-track receipts were moved byte-exact into ignored
`.haic-artifacts/apex-speed-20261005/`. The tracked receipts bind original raw
hashes and canonical hashes for every trial; all hashes were verified after
compaction. `physics-followup-provenance.json` binds the new sources/receipts.
