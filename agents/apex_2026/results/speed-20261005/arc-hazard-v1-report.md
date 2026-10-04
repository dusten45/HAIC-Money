# Camera parametric hazard prefix experiment

The frozen parent is RearClear
`093aaa77a0123138e52f51f576d54e6ae95b36b38929056f8a6a6a22215740dc`.
New standalone `fast_arc_hazard_agent.py` preserves its complete source prefix
with only the final parent class renamed. The new candidate source is
`8ec8b3014943a997eec15d9327515f99d648f98736e90897402a81c049a9734f`.
A zero-context patch reverses to the exact parent. Existing candidates and
official environment files remain unchanged. This is a research candidate,
not an adoption or original pace-goal claim.

The current camera row distance orders a hairpin's later circle before the
near bend. The new candidate measures the circle's relationship to the
supported ridge arc. It only defers circles whose avoidance footprints have a
known interior relationship to that path. Every circle must qualify; unknown
or off-reference association retains the exact parent action. It trims before
the earliest existing avoidance disk (3.7 m with default parameters), with
the existing 2.6 m front reserve. Retained arc must exceed
`v²/(2×braking_accel) + .08v + 2.6`. The ridge target additionally assumes zero
speed at its trimmed endpoint. No unseen extension or filled circle pixels
provide obstacle clearance.

Original ridge point support (5.8 m), chord support (1.9 m), circle detection,
force limits and existing active-pass behavior remain. Deferral additionally
requires 1.9 m support for sampled vehicle corners on the prospective emitted
constant turn, including the original .24 slew. These near 0–8 m samples do
not certify the full prefix, tyre behavior, steering lag, or dynamic safety.

| Saved input | Trimmed arc m | Stop budget m | Emitted body support m | Decision |
| --- | ---: | ---: | ---: | --- |
| T4 132 | 50.179 | 32.098 | 3.458 | defer, use trimmed ridge |
| T4 133 | 39.831 | 29.085 | 3.031 | defer, use trimmed ridge |
| T4 134 | 34.119 | 23.634 | 1.553 | exact parent fallback |
| T4 135 | 28.702 | 19.863 | 0.000 | exact parent fallback |
| T4 139 | 9.084 | 10.818 | 1.346 | exact parent fallback |

Original circles are cached for the action. All parent route, steering, and
ridge-phase calls see the same filtered labels only after qualification;
the ridge receives exactly the trimmed prefix. Outside the action, the
original detector reports the future hazard again. Reset clears all cache
and deferral state. A prior active pass, invalid/lost road, inadequate prefix,
or inadequate body support never clears cooldown or pass memory.

Existing exact camera captures were exported without another episode.
Before implementation, the focused suite had three expected failures (132/133
confidence rather than ridge, and the missing unknown-endpoint helper) and
eight passing preservation cases. After implementation, 32 focused tests
passed across hazard, rear-clear, arc-clear, arc-guard and supported-ridge
modules.

One fresh mandatory research screen used 700 actions maximum, default
parameters, two workers, and unchanged official finish semantics. The frozen
source remained unchanged throughout. All four finished without contacts.

| Track, seed | Fresh time s | Frozen parent time s | New minimum speed after input 40 m/s |
| --- | ---: | ---: | ---: |
| 1, 516237 | 15.54 | 15.54 | 16.854 |
| 2, 644062 | 19.14 | 18.72 | 29.067 |
| 3, 1007 | 16.60 | 16.56 | 44.385 |
| 4, 18800 | 21.94 | 18.14 | 1.349 |

The broad deferral rule does not improve pace and is rejected as a performance
candidate. The 10–13 second goal and the requested early-10-second goal remain
unmet; no formal adoption or extra-track claim is made. No holdout was opened.

Trace comparison shows the first changed action at T2 181, T3 63, and T4 24;
T1 actions are exactly unchanged. Thus the fresh T4 failure is not an exact
replay of the saved late 132/133 scene. The rule also changes an earlier
circle encounter. New T4 actions 108–180 remain below 20 m/s for 73 actions
(5.84 sampled seconds), compared with the parent's 28 actions (2.24 seconds).
The floor is 1.349 m/s at 112. Before recovery begins at 106, inputs 99–102
alternate steering +0.052/-0.188 by the complete .24 slew. Local stopping
prefix and body samples do not establish stable subsequent road/circle phase
transitions. No additional episode or threshold tuning was performed after
this rejected result.
