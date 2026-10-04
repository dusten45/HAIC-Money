# Faster present-pursuit camera study

This standalone research candidate targets all four required laps at at most
13 seconds, with useful speed maintained on curves. It preserves the robust
definitions and guided hazard protection from the frozen guarded-preview
reference. No root agent or official physics file changes.

Frozen source SHA256:
`a0cbdf690ead624597968ed8237e4712178dbc70af6b380f9fb6ed9cc3abb8ac`
(132,343 bytes). `preview-v1-lineage.json` binds its exact robust prefix,
embedded parameters, tests and source snapshot; `preview-v1-source.patch`
records the complete source change. Runtime parameters are `{}`.

## Calibration and concrete planning defect

Official input steering maps to the negative front-wheel target in radians,
bounded at 0.4 radians. Wheelbase is 3.24 metres. Camera scales are 1.3608
horizontal and 1.701 vertical pixels per metre; the white HUD measures physical
metres per second. Existing guarded-preview traces show median HUD error only
0.60–1.06 m/s below post-action actual speed. Their median actual speeds are
44.68/41.68/40.44/40.25 m/s; 82–86% of targets are below 60 m/s. These historical
traces diagnose the old policy and are not fresh validation of this candidate.

An aligned rasterized road with radius 60 metres at physical-gauge nominal 70 m/s
exposes a false corner ceiling. The old default target is 55.65 m/s, with
braking. At explicit lateral acceleration 200 and preview 0.12, pursuit steering is 0.05790
radians, close to the geometric 0.05395; the old quadratic still cuts the target
to 65.35. The present-pursuit force ceiling would allow over 100 in that frame.
This is evidence of quantization sensitivity, not proof that every upcoming
curve permits 100 m/s.

Independent official-Car characterization (`physics-response-v1.json` and
`physics-throttle-v1.json`) confirms near-kinematic steering within about 3% on
stable asphalt turns. It also shows full gas at 100 m/s can spin even at 0.03
radians. The rear drive and lateral tire demand share one friction circle.
The theoretical total limit is 219.12 m/s²; it is an upper bound. Box2D's maximum
translation produces an approximately 100 m/s steady ceiling.

An initial quasistatic gas budget still spun at 0.06 radians while accelerating
from 70. Fixed gas 0.3 remained stable from 70 and 100, whereas 0.5 and above spun.
Those are privileged uniform-asphalt diagnostics, with no road edges or
obstacles; they are not legal camera lap benchmarks or completion evidence.

## Bounded controller changes

The separate quadratic-window speed minimum is removed. Present pursuit
curvature still limits requested steering and target speed, and the inherited
visible-horizon and obstacle budgets remain. The lookahead coefficient is
0.12 seconds, with lateral acceleration 190 m/s², cruise 100, gas demand gates 0.90/0.98 and
clear-curve routing sweep 40. Robust definitions are copied exactly; their
constructor receives cruise 100, propulsion 1 and corner boost 0.5.

The public action then reserves rear grip using
`gas <= .008*speed*sqrt(max(0,1-(lateral_acceleration/210)^2))` while turning at
at least 25 m/s. It caps gas at 0.3 when the emitted steering would require at
least 130 m/s² at 100 m/s, matching the measured problematic acceleration regime.
Straight acceleration remains available. Existing robust hazard brake, gas and
effective target restrictions remain; any brake request clears gas.

Thirteen focused tests pass after four observed failures: false bend braking,
slow offset correction, excessive high-speed gas, and the transient turn budget.
The combined Apex controller/evaluator/package regressions pass 239/239.
The isolated ZIP is 26,345 bytes and contains byte-identical `agent.py`;
ZIP SHA256 `c2462faf582ae13ab815ae600e65e8bb06149230c3bf8bdff73575e29df77efa`.
This package check does not establish official container certification.

## Fresh mandatory screen

The single cold four-cell mandatory benchmark completed with frozen source
and parameters `{}`. `preview-v1-mandatory.json` preserves the exact source,
environment, parameter and action hashes. All operational errors are null.
No additional geometry or holdout was opened.

| Track, seed | Result | Contacts |
|---|---:|---:|
| 1, 516237 | 21.12 s | 0 |
| 2, 644062 | 27.40 s | 0 |
| 3, 1007 | 24.82 s | 0 |
| 4, 18800 | DNF, progress 0.52330 | 2 |

The first three times improve on frozen guarded-preview's 21.92/27.64/26.14,
but remain slower than fresh hybrid's 19.62/26.20/22.90. Track 4 loses a previous
finish, with damage 0.4 and official retirement `off_track`. Sampled full
off-road counts are zero; partial samples are 6/1/0/0, which does not exclude
events between decisions. Completion and the original pace goal both fail.

Measured local maxima are initialization 20.77 ms, reset 0.065 ms, action 53.41 ms
and RSS 98.42 MiB, within the stated runtime limits. These are local measures,
without official container certification.

`preview-v1-trace-summary.json` records diagnostic aggregates. The completed
tracks' median speeds rise to 46.56/43.90/43.91 m/s, still far from the 100 m/s
steady ceiling. Their target-below 60 fractions fall to 45–52%, versus 82–86%
in the prior frozen guarded-preview traces. Brake fractions fall to 24–25%,
versus 30–35%. The planning change therefore affects real actions, but the
remaining pursuit, uncertain-road and obstacle restrictions prevent the
requested lap times. These comparisons do not rescue the failed track 4.

## Remaining limits

Present pursuit provides no corner-entry certificate beyond its observed
target. The rear-force approximation and empirical throttle threshold do not
observe sideslip, tire angular velocity or contact damage. Sparse/occluded road
geometry and the existing robust obstacle limitations remain. Legacy curve
window/step controls are accepted but do not affect this planner; a custom
braking override does not change the inherited 100 m/s² horizon assumption.

**Original goal achieved: false; promoted: false.** Three modest lap gains and
a useful camera behavior test do not establish four 13-second completed laps.
