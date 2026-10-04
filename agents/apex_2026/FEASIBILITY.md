# REQUIRED-cell feasibility and HUD diagnostic

Evidence collected 2026-10-04. This is evaluator-only analysis, not a runtime
policy or an official performance claim. No holdout/development cells were read
or reset. There is no proof that 10–13 seconds is globally impossible.

The persisted [primary summary](results/feasibility.json) contains all calibration
rows, geometry lengths and receipt/coordinate hashes, release aggregates,
synthetic speeds/distances at integer seconds, relaxed estimates, package/source
provenance, and exact source text plus hashes of the two original executed probe
scripts. Temporary paths below identify the original artifacts. The persisted
summary keeps their hashes without copying large traces or coordinate arrays.

The replay CLIs [feasibility_probe.py](diagnostics/feasibility_probe.py) and
[physical_limits.py](diagnostics/physical_limits.py) adapt those original scripts
to explicit input/output paths and refuse existing output artifacts. Their hashes
are recorded separately; they did not produce the recorded measurements. During
persistence only their `--help` and compilation were checked, with no new resets
or synthetic measurements. To repeat explicitly on REQUIRED cells in fresh paths:

```bash
.venv/bin/python agents/apex_2026/diagnostics/feasibility_probe.py \
  --release-dir /tmp/apex-release-r0 --output-dir /tmp/apex-feasibility-repeat
.venv/bin/python agents/apex_2026/diagnostics/physical_limits.py \
  --geometry-dir /tmp/apex-feasibility-repeat --output-dir /tmp/apex-limits-repeat
```

The first command consumes four new REQUIRED resets; the second uses synthetic
physics and existing geometry only. The first also requires the existing release
trace inputs; it does not rerun release driving episodes.

## What was inspected and consumed

Existing release evidence: `/tmp/apex-release-r0/t{1,2,3,4}.json` and `.jsonl`.
Exactly four additional official resets, each with the unchanged default warmup
and **zero agent actions**, obtained geometry and car mass:

| Track | Required seed | Receipt |
|---|---:|---|
| 1 | 516237 | `/tmp/apex-feasibility/geometry-t1.json` |
| 2 | 644062 | `/tmp/apex-feasibility/geometry-t2.json` |
| 3 | 1007 | `/tmp/apex-feasibility/geometry-t3.json` |
| 4 | 18800 | `/tmp/apex-feasibility/geometry-t4.json` |

Each receipt was written as started before reset and completed afterward.
`/tmp/apex-feasibility/probe.py` reproduces these probes but intentionally refuses
to overwrite existing reset receipts. Two additional synthetic diagnostics used
no CarRacing resets: indicator-only rendering and 1,000 steps of isolated car
physics on a synthetic uniform road. Neither is an official episode.

## Required speeds

Exact closed polygon centerline length comes from generated track coordinates.
Release path length sums hull positions at the action sampling interval and is
an approximation; the terminal action can extend beyond the scored crossing.

| Track | Tiles | Centerline m | Release lap s | Release path m | Release mean / peak m/s | Centerline average for 13s / 10s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 280 | 983.500 | 18.520 | 978.819 | 53.045 / 72.582 | 75.654 / 98.350 |
| 2 | 336 | 1179.500 | 21.700 | 1168.027 | 53.922 / 75.155 | 90.731 / 117.950 |
| 3 | 309 | 1085.000 | 20.300 | 1075.803 | 53.079 / 74.610 | 83.462 / 108.500 |
| 4 | 279 | 979.824 | DNF | 692.872 | 31.875 / 73.836 | 75.371 / 97.982 |

Release tracks 1–3 have zero damage and full tile progress. Track 4 ended at
0.741935 progress and 0.4 damage; its mean includes stopping and is not a lap
pace estimate. The target requires much higher sustained speed than the current
release, especially track 2. Centerline length is **not** a minimum legal racing
line distance, so dividing it by a speed limit cannot prove a global lap bound.

## Physical constraints and explicitly limited approximations

The unmodified source runs Box2D at 50 Hz. Installed Box2D reports
`b2_maxTranslation=2.0`, implying approximately 100 m/s per-body translation cap
at that step size. This is a numerical physics limit, not an engine-speed limit.
The car's summed body mass is 7.30191997. Each wheel force magnitude is capped at
400 on undamaged road (`road_friction=1`), giving a very optimistic combined
center-of-mass acceleration bound of 1600/mass = 219.12045 m/s² absent external
contacts. Actual acceleration cannot generally devote all four contact forces to
one direction. Only the rear wheels receive engine power; tire slip, wheel spin
and steering consume force budget. Grass force cap uses multiplier 0.6; damage
reduces grip/engine/steering further.

Front steering joints have angle limits ±0.4 rad and motor rate cap 3 rad/s.
The wheelbase is 3.24 m. A no-slip bicycle approximation gives minimum radius
3.24/tan(0.4) ≈ 7.66 m. This is an approximation, not a hard drift-path bound.

The synthetic straight-road probe uses the exact official `Car` dynamics, full
gas, zero steering/brake, no damage and all wheels marked on uniform road. It
measured speed 45.53 m/s at 1 s, 87.56 at 2 s and approximately 100 at 3 s;
distance was 889.21 m at 10 s and 1189.21 m at 13 s. Peak sampled acceleration
was 60.35 m/s². This shows a substantial launch cost (~111 m behind hypothetical
instantaneous 100 m/s motion) even without turns. It is a controlled diagnostic,
not a universal optimal acceleration proof. Source and samples:
`/tmp/apex-feasibility/limits.py`, `synthetic-straight.json`.

An optimistic **fixed-centerline point-mass approximation**, allowing the full
219.12 m/s² as lateral acceleration, instantaneous speed changes, no launch cost,
no obstacles and cap 100 m/s, integrates `ds/min(100,sqrt(a/curvature))` to:
track 1 **10.757 s**, track 2 **13.533 s**, track 3 **12.158 s**, track 4
**11.452 s**. Polygon curvature is estimated from neighboring segment headings.
These are useful difficulty indicators, **not proven lower bounds for legal
racing lines**. A wider smoother line or corner cutting changes both curvature
and distance. Track 2's 13-second target is particularly demanding even under
this generous centerline approximation. Artifact: `centerline-relaxation.json`.

Finish rules also prevent replacing the lap with a simple early turn-back:
`FinishLineTracker` requires at least 95% tile progress, departure from the start
area, entry through the back of the finish zone, a forward center crossing with
lateral/forward velocity ratio at most 4, and subsequent exit through the front.
The recorded finish time is the qualifying center-crossing timestamp, established
only when the front exit confirms it. The 95% threshold alone does not finish.

## Verified pixel-speed calibration issue

The official renderer draws the white HUD bar with height
`(H/40) * 0.02 * true_speed`. Calibration rendered this exact indicator to the
1000×800 surface, used the official 96×96 smoothscale, then the unchanged
`image_preprocessing` to 84×84 grayscale. It did not alter an environment or
feed privileged values into a candidate. Full data: `hud-calibration.json`.

| True speed m/s | Line estimator `(mass-.27)/.085` | Current rollout estimator |
|---:|---:|---:|
| 10 | 9.51 | 0.00 |
| 30 | 29.72 | 9.99 |
| 50 | 49.88 | 28.01 |
| 70 | 70.09 | 47.43 |
| 90 | 90.16 | 66.85 |
| 100 | 100.00 | 76.84 |

The line estimator's `[77:83,10:13]` mass is accurate throughout the physically
relevant range; its clip at 100 is consistent with the numerical speed cap.
Its two-frame average can introduce transient lag, which this static calibration
does not measure. The rollout estimator's `[74:81,11:12]` crop omits the bottom
of the bar and substantially underestimates speed. The geometric slope alone
is insufficient after this crop and downsampling. This is a concrete candidate
fix opportunity; no candidate source was edited during this analysis.

A speed-estimation correction does not itself establish faster completion.
Actual matched REQUIRED episodes would be needed to measure its effect.

## Additional trajectory-model mismatch clue

The earlier rollout acceleration expression
`gas*31/(1+0.055*v) - brake*38 - 0.004*v**2` predicts **−1.733 m/s²**
at 50 m/s with full gas and no brake. Existing REQUIRED track-1 release trace
step 15 instead measures **43.773 m/s²** over 80 ms, starting at 50.778 m/s,
with action `[0,1,0]`. The original before/after telemetry is embedded in the
[primary summary](results/feasibility.json), together with the exact prediction
at that observed speed. The isolated official dynamics' rapid 45.53→87.56 m/s
increase from 1 to 2 seconds is consistent with this being a material dynamics
model mismatch. This is a measured discrepancy, not proof of a universal
acceleration law or proof that fixing it improves laps. Candidate code may have
changed since this earlier expression was inspected.
