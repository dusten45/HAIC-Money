# Early-ten-second feasibility: consumed mandatory geometry only

The present geometric bound does **not** establish that early-ten-second laps
are impossible. It also supplies no executable early-ten-second route. Under
the explicitly conditional 100 m/s, nominal rigid-car model, the lower bounds
are 8.537 / 8.839 / 8.665 / 7.649 seconds. These are relaxed bounds rather than
predictions. The best measured RearClear laps remain
15.54 / 18.72 / 16.56 / 18.14 seconds; the requested pace is unachieved.

This study reconstructs only the four already-consumed mandatory layouts,
calling unchanged official `_create_track` through a geometry sink. It creates
polygon shapes but no Box2D world, vehicle, environment reset or physics step.
It creates no Agent, opens no holdout and runs no new lap. Official sources and
all frozen agents are unchanged. Exact input/source hashes are in the JSON and
lineage file; the saved actions reproduce the RearClear receipt's float32 action
hashes, and every saved progress value agrees with the reconstructed tile count.

## What the actual qualification rule requires

`FrictionDetector._contact` increments unique tile visits when the contacting
body has a `tiles` attribute. The four wheels have this attribute; the hull does
not. Qualification is `tile_visited_count / tile_count >= 0.95`, followed by the
official forward finish crossing. There is no required tile order, minimum
centerline arc length, or rule requiring the vehicle center to touch each tile.
Consequently, 95% of centerline length is not a universal distance lower bound.
An inside corner cut or passing close enough for a wheel to touch a tile can
retain credit. Up to the integer omission counts below can receive no contact.

| Track/seed | Tiles / required / omitted | Centerline polygon length | Sampled RearClear path | Conditional path lower bound | Conditional time lower bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 516237 | 280 / 266 / 14 | 983.50 m | 968.60 m | 853.66 m | 8.537 s |
| 2 / 644062 | 336 / 320 / 16 | 1179.50 m | 1159.33 m | 883.88 m | 8.839 s |
| 3 / 1007 | 309 / 294 / 15 | 1085.00 m | 1070.45 m | 866.48 m | 8.665 s |
| 4 / 18800 | 279 / 266 / 13 | 979.82 m | 953.13 m | 764.93 m | 7.649 s |

Centerline length is the closed polygon through official track points, not an
optimized route. Sampled path length joins successive 0.08 s output positions,
omits the first output interval, and can include the final confirmation tick
after the backdated finish candidate. Neither is a minimum or a rigorous
continuous-trajectory length. Sampled positions supply a useful pace diagnostic
only; they do not prove a lower bound.

## Exact relaxed coverage bound and integration error

Let each road tile polygon be `P_i`. Assume any wheel contact lies within 3 m of
the nominal compound car COM; expand each tile by this radius. Project the
expanded tiles along direction `u(theta)`, obtaining closed intervals
`[a_i, b_i]`. Set `k = ceil(0.95 N)`. Any qualifying COM path's projected range
must intersect at least `k` intervals and contain its nominal starting COM.

For an interval subset `S`, its required span containing projected start `s` is
`max(max_S a_i, s) - min(min_S b_i, s)`. The helper computes the exact minimum
over all size-`k` subsets. Sort `a` ascending; for every prefix `j >= k`, use its
`k`th-largest `b` as the greatest attainable minimum right endpoint, and minimize
the resulting span. This is exhaustive over the interval problem; it is not a
point-route optimizer or a global solution to tile coverage.

Cauchy's convex-hull perimeter formula gives, after closing the COM path by a
segment of length at most 10 m,

`COM path length >= integral_0^pi W_k,start(theta) dtheta - 10 m`.

Every feasible path must satisfy this inequality. The relaxation may select
different omitted tiles in different directions and ignores local road bends,
obstacles, wheel placement, steering, tire forces and path reachability. These
omissions explain why the bound is much shorter than current driving paths.

The helper uses 1024 midpoint bins. If all polygon vertices and the nominal
start lie within radius `D` of the chosen origin, the relaxed width is
`2D`-Lipschitz. Each bin of width `h` has integration error at most
`2D * h^2 / 4`. Subtracting these errors gives the reported lower bound, rather
than treating quadrature output as exact. Total error allowances are
0.910 / 0.974 / 0.982 / 0.811 m. Polygon vertices use the installed Box2D shape's
float32 representation. An additional 1e-6 m roundoff allowance is applied,
but floating-point/libm rounding is **not** machine-verified interval arithmetic.
The mathematical inequality is exact under its model; its numerical evaluation
has the stated practical rounding limitation.

Final independent review found two further premises in the existing numerical
model. First, it anchors the interval span at the **exact nominal** starting
COM. The official constructor initially places wheel centers without rotating
their offsets, so even spawn COM differs slightly from this nominal point;
post-warmup equality is unverified. Second, `k` includes untimed warmup tile
credits (the existing `feasibility-audit.json` records two on each layout).
The timed COM path must also intersect the expanded regions of those
precredited tiles for this numerical bound to apply. Neither premise has been
verified here. The helper and numerical JSON remain preserved research bytes;
the lineage explicitly records both additional model assumptions. A future
bound can instead subtract credited warmup tiles from `k` and deduct
`pi * start_uncertainty` from the anchored integral. That corrected computation
is outside this checkpoint. The reported 8.537 / 8.839 / 8.665 / 7.649 s values
therefore describe the specified hypothetical model and must not be presented
as unconditional bounds for official timed laps.

## Why the 100 m/s conversion is conditional

Installed Box2D has `b2_maxTranslation = 2 m`. Ordinary integration clamps each
body's integrated COM increment per 0.02 s tick; the convex average of those
increments bounds the compound COM increment by 2 m. Internal joint position
corrections conserve compound COM in the ideal collision-free model.

This is not an unconditional speed or displacement theorem for a full official
lap. Static-obstacle collision corrections can add displacement, and TOI solving
uses a shorter remaining timestep with its own translation clamp. Nominal
rigid-joint geometry also has not been given a universal finite-solver strain
bound. The 3 m contact radius is generous relative to the nominal wheel envelope
(approximately 2.64 m including polygon skins) but remains a model assumption.
The competition permits contacts short of retirement, so a collision-free
bound alone cannot prove the original task impossible for every legal strategy.

The finish time is backdated to the first candidate tick with valid forward
velocity, then confirmed later. A candidate need not be the first tick with
positive longitudinal position. Under the same rigid, collision-free model,
the compound COM offset from the hull origin is 0.080459 m, the hull-origin
increment is at most `2 + 2*offset`, and candidate longitudinal position is at
most `0.875 + 2 + 2*offset`. With finish half-width 6.666667 m and nominal start
near the line center, the closure is at most approximately 7.487 m. The 10 m
allowance is conservative under these assumptions.

A wheel-COM network relaxation avoids a car-to-wheel radius during the lap:
expand tiles by the wheel corner radius plus both polygon skins (0.62828 m),
and connect the four nominal starting wheel centers by a 7.64 m spanning tree.
It yields a sum-of-wheel-path lower bound from 386.11 to 445.70 m. Converting
that sum with four separate assumed 100 m/s wheel caps gives only roughly
0.97–1.11 s. Its geometry survives later joint strain, but the starting-tree
assumption, pre-timing contacts and actual wheel displacement bounds still need
care; this weak result does not resolve the target either.

## How much shortening the current paths would need

The following cuts compare current sampled path length with `100 * target_s`.
They assume constant 100 m/s from the start and no turns or braking, so actual
driving would need further time savings or shorter paths. Zero means the path
length alone fits that optimistic budget; it does not mean the lap is feasible.

| Track | For 10.0 s | For 10.5 s | For 11.0 s | For 13.0 s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0% | 0% | 0% | 0% |
| 2 | 13.74% | 9.43% | 5.12% | 0% |
| 3 | 6.58% | 1.91% | 0% | 0% |
| 4 | 0% | 0% | 0% | 0% |

Cutting 2–14% is not forbidden by the qualification wording and is not
excluded by any lower bound established here. Whether these particular layouts
admit the required shorter executable paths remains unresolved. There is no verified 2–14%
shortening witness with actual wheel contacts and obstacle clearance in this
study. A shorter curve touching expanded tile polygons would be a witness only
for this relaxation, not a minimum or an executable lap. A stronger witness
must continuously place the actual four wheel polygons and hull, preserve
95% sensor contacts, clear obstacles, return through the finish, and then satisfy
motor slew, shared tire forces and official retirement behavior. A generic
camera controller must additionally select it without a seed/map lookup.

Verification: eight focused tests pass, including exact exhaustive subset
comparisons with interval ties and a rectangle perimeter integration check.
An independent read-only math review checked the projection, closure and Box2D
caveats. The study establishes no adoption, new lap result or pace achievement.
