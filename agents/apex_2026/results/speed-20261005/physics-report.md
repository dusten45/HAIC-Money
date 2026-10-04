# Physics and racing-line diagnostics

These are **privileged diagnostics**, not legal camera agents, prospective
candidate benchmarks, or holdout results. Official physics, contact, damage,
watchdog and finish code were preserved. No candidate is adopted by this report.
The original four-track 10–13 second goal remains **unachieved**.

## Reproducible evidence

`physics-summary.json` binds all source hashes, 32 new teacher episodes,
action hashes, receipt hashes and ignored raw-trace hashes. The fresh archived
reference reproduced all four historical action hashes exactly. Historical
results are identified separately from these new executions. Source versions
are preserved by reversible zero-context patches; reversing both patches in
fresh `/tmp` files reproduced the exact historical source hashes.

SciPy **1.14.1** was installed in `.venv` for optional offline research only.
Official setup scripts and dependency requirements were not changed. Large
response traces and all per-action teacher traces are under the ignored
`.haic-artifacts/apex-speed-20261005/`; tracked receipts contain summaries and
hashes. No holdout was opened and no usage-limit error was encountered.

## Measured car response

- Wheelbase 3.24 m; wheel radius 0.54 m. Action steering `s` targets a front
  joint angle of `-s` radians, with a physical joint limit of ±0.4 radians.
- Measured stable curvature is approximately `1.033 * tan(delta) / 3.24`.
  At 100 m/s, front angles 0.03 and 0.06 radians demand approximately 96 and
  191 m/s² lateral acceleration. The latter is close to the grip limit.
- Four tires at 400 N and body mass 7.302 kg give a total ideal force limit of
  219.12 m/s². Acceleration and steering share each tire's force budget.
- Box2D's maximum translation is 2 m per 0.02 s physics step; fresh probes
  plateau at about **100 m/s**. Larger requested cruise speeds add no top speed.
- Ideal straight full-gas launch reaches 70 m/s at 1.56 s and 99 m/s at 2.42 s.
  Its measured time cost relative to instant 100 m/s over three seconds is
  1.108 s. Wheel rotational inertia limits launch acceleration to about
  43.77 m/s² despite larger force/body-mass bounds.
- Straight brake commands 0.1, 0.2 and 0.5 deliver approximately 31, 62 and
  154 m/s². Brake ≥0.9 locks wheels and removes the measured steering response.
  Combined steering/braking requires a smaller shared force budget.
- Full gas at high speed can spin the rear wheels even on moderate curves.
  Fixed gas 0.3 sustains the 100 m/s, 0.06-radian case; 0.5, 0.7 and 0.8 spin.
  The instantaneous formula `0.008*v*sqrt(1-(alat/210)^2)` remains unsafe while
  accelerating from 70 m/s into that turn because wheel state has memory.

The rear-wheel HUD radius calibration is one rendered state, not a complete
sensor fit. A wheel/HUD speed excess gate applied during straight launch can
cut useful throttle: at physical speed 25.4 m/s, measured rear rolling speed is
58.5 m/s. Curvature and wheel history matter.

## New mandatory teacher episodes

All rows below are new executions using the unmodified official environment.
Each setting was fixed before its four mandatory cells. Times are official
simulation finish times, in seconds; DNF means no official finish.

| Diagnostic setting | Track 1 | Track 2 | Track 3 | Track 4 | Contacts per track |
| --- | ---: | ---: | ---: | ---: | --- |
| Archived reference | 13.78 | 17.74 | 15.98 | DNF | 0 / 0 / 0 / 1 |
| Wider obstacle return | 13.72 | 17.64 | 15.74 | DNF | 0 / 0 / 0 / 1 |
| Fixed small lateral shift | DNF | DNF | DNF | DNF | 1 / 1 / 1 / 5 |
| Feedforward, small shift | DNF | DNF | 20.48 | DNF | 5 / 1 / 1 / 5 |
| Discrete pursuit, brake feedforward | 15.42 | 18.84 | 17.38 | DNF | 0 / 0 / 0 / 5 |
| Continuous pursuit, brake feedforward | 15.34 | 18.82 | 17.46 | 16.38 | 0 / 0 / 0 / 1 |
| Optimized spline route, lateral 190 | 14.66 | 18.50 | 17.24 | 15.80 | 0 / 0 / 0 / 0 |
| Optimized spline route, lateral 210 | 14.22 | 17.98 | 16.66 | 15.60 | 0 / 0 / 0 / 0 |

Budget used: **8 settings × 4 cells = 32 episodes**. No further teacher
episodes were run. The last setting has no fully off-road samples, but has
2 / 5 / 4 / 5 samples with some wheels off asphalt. It does not certify that
the complete rotated hull stayed inside the road.

The small fixed shift failed because official obstacle lateral offsets are
uniform over [-4, 4] m, including positions near road center. Obstacle-relative
clearance is required. Restoring a short lane shift after every obstacle also
adds curvature: the old 22 m cosine return has about twice the approach's peak
curvature. A wider return improved three reference times slightly, but did not
solve track 4. Brake feedforward plus conservative rear throttle improved
completion coverage while costing speed; it is not a validated speed fix.

## Offline route estimates

All models use the entire map, 100 m/s maximum speed, lateral limit 219 m/s²,
forward acceleration 40 m/s², braking 150 m/s², initial speed zero, lateral
point bounds ±4.8 m and obstacle-center separation at least 3.7 m. The speed
profile is propagated forward and backward around one complete route. The
terminal speed is free, not another standing stop.

| Offline model | Track 1 | Track 2 | Track 3 | Track 4 |
| --- | ---: | ---: | ---: | ---: |
| 24–60 periodic spline knots, local time objective | 12.719 | 15.938 | 14.512 | 13.626 |
| Every point free, converged convex curvature approximation | 12.305 | 15.554 | 14.085 | 13.542 |
| Every point free, 150-iteration local time search | 12.067 | 15.130 | 13.701 | 13.236 |

The spline search reached its 90-iteration limit and has numerical separation
violations up to 15 micrometers, retained explicitly in its receipts. The
convex curvature refinement converged on all tracks and passes exact segment
circle-distance checks. It minimizes an approximate curvature objective, not
time. The full-point time search converged only on track 4 at this checkpoint;
the remaining searches reached their iteration limit. Its exact obstacle
segment clearance is 3.704 / 3.719 / 3.726 / 3.742 m, with offsets within ±4.8 m.

These are **local route time estimates**, not global optimum certificates or
physical lap-time lower bounds. Independent acceleration limits omit tire
force coupling; road point bounds omit hull sweep; the models omit steering
latency, wheel spin, damage, tracking errors and the official finish gate.
Their estimated times cannot establish that a faster physical route is
impossible. Further offline optimization may improve them.

Measured road-center lengths are 983.500 / 1179.500 / 1085.000 / 979.824 m.
Instant 100 m/s plus the measured straight-launch penalty would cost
10.943 / 12.903 / 11.958 / 10.906 s on those center paths. Corner cutting can
shorten the route; these figures are illustrative and are not shortest-lap
lower bounds. Track 2 has very little slowdown margin on its center path.

## Camera-controller implications

The observed stable 100 m/s, 191 m/s² turn shows that curve slowing is not
always required. Reaching that regime needs accurate curvature, joint-angle
feedforward, bounded pursuit correction, yaw damping and rear torque control.
The measured response settles after the joint angle moves; steering-rate and
trajectory response should be treated separately.

A legal camera candidate can use measured road curvature and a lane offset
that stays on one passing side longer. The requested lane shift must follow
actual obstacle position and verified corridor clearance. Use the rear HUD
as a coarse wheel-spin signal with calibrated crop/raster limits, while
preserving useful straight-launch throttle. The failed feedforward and small
shift experiments remain negative evidence; this report does not recommend
adopting their control equations.
