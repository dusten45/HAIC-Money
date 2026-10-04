# High-speed camera continuation

The user resumed the strict 10–13 second target on 2026-10-05 (Asia/Seoul),
starting from `a30adbb`. Root inference and official physics remain protected.
Previous holdout seeds are consumed development data; no new holdout has opened.
No candidate in this series has achieved the goal or been adopted.

The user cancelled the subsequently proposed 04:00 KST time cutoff. There is
no active clock deadline. Continue toward the speed goal and stop immediately
on an actual Codex weekly quota exhaustion error. Fast mode is an app setting
and no available tool can change it from this session.

## Initial hypotheses and fresh screens

- Existing guarded preview averages about 40–45 m/s and brakes on 30–35% of
  decisions. Its HUD estimate agrees with true simulation speed within about
  1 m/s in those traces. Camera units are not the main source of slowdown.
- Raising permitted lateral demand and propulsion alone gives guarded preview
  **19.08 / 31.40 / 21.42 / DNF** seconds. Track 4 has three contacts; the other
  three finish without contacts. The earlier invalid gas-gate parameter attempt
  is preserved as an operational error and is not a performance rejection.
- `fast_envelope_agent.py` removes the fixed steering gas cutoff, uses pursuit
  demand for speed, and shifts only the estimated necessary obstacle distance.
  Five camera behavior tests pass after two relevant failures against the old
  controller. A full-gas test expectation was revised after direct physics
  measurements showed rear-wheel spinning; acceleration still exceeds the old
  fixed 0.3 cap on the tested mild bend.
- Its first four-track screen finishes **0/4**, with progress
  **0.286 / 0.473 / 0.524 / 0.416** and contacts **4 / 3 / 2 / 4**. It remains
  experimental. Camera replay detects the first obstacle early enough, but the
  minimum lateral clearance fails during high-speed passage. Motion, margins
  and wheel-spin history need further work.

These are fresh benchmark screens, not complete prospective development trials.
The unchanged evaluator records actual official finish time, source/parameters,
environment and action hashes. Traces remain in ignored
`.haic-artifacts/apex-speed-20261005/`. Physics calibration is diagnostic only;
its privileged wheel/pose access is not part of any submitted agent.

Parallel lanes continue on observed yaw feedback, improved curve planning,
rear-wheel grip budgeting and obstacle routes. There is no callable tool for
Codex weekly remaining usage; a quota error will trigger immediate stopping.

## Subsequent fresh evidence

The independent yaw-feedback path lane is substantially faster than the prior
guarded snapshot, but is not yet a qualifying candidate:

| Source / parameters | Track 1 | Track 2 | Track 3 | Track 4 |
|---|---:|---:|---:|---:|
| path V3, defaults | 15.52 s | 19.92 s | DNF | 18.32 s |
| path V3, wider-preview study | 15.30 s | 19.50 s | DNF | DNF |
| path V4, defaults | 15.40 s | 23.20 s | 18.00 s | DNF |
| faster preview V1, defaults | 21.12 s | 27.40 s | 24.82 s | DNF |
| envelope V3, defaults | 18.54 s | DNF | DNF | DNF |
| envelope V4, defaults | DNF | DNF | DNF | DNF |
| path V4 without local quadratic caps | 15.56 s | DNF | DNF | DNF |
| path V5, defaults | 16.40 s | DNF | DNF | DNF |
| path V5, yaw_gain=.15 | 15.70 s | DNF | DNF | 26.10 s |
| curved preview pass V2, defaults | 20.20 s | 26.08 s | 22.82 s | DNF |
| braking feedforward V1, defaults | 16.60 s | DNF | DNF | 17.04 s |
| braking feedforward V2, defaults | 16.56 s | DNF | DNF | 16.76 s |
| camera corridor smoothing V1, defaults | 15.30 s | DNF | DNF | DNF |
| grip reserve V1, defaults | 15.72 s | DNF | DNF | DNF |
| graph V1, defaults | 15.04 s | 24.22 s | 19.86 s | 19.96 s |
| graph V2, defaults | 14.84 s | DNF | DNF | DNF |
| explicit ego confidence V1, defaults | 15.04 s | 24.22 s | 19.86 s | 19.96 s |
| forecast propulsion V1, defaults | 15.72 s | 26.20 s | 23.96 s | DNF |
| full ridge V1, defaults | 15.04 s | DNF | DNF | DNF |
| clear-road ridge V1, defaults | 15.18 s | 29.62 s | 19.60 s | DNF |
| clear ridge support V2, defaults | 15.18 s | 30.24 s | 18.50 s | 18.38 s |

All four times must belong to one source/parameter setting; the fastest
individual laps above must not be combined into a synthetic result. None of
these studies meets even the four required 13 second laps. Old holdout remains
consumed, no new holdout has opened, and no source is promoted.

Path V4 corrects road-normal pass offsets, remembered-circle rotation, HUD
boundary contamination and straight-launch feedback. Seventeen focused tests
pass, and reversible patches reconstruct V1–V3 source bytes and their receipts.
Dropping its local curvature speed caps entirely loses three finishes; the
three passing camera behavior tests do not qualify that pursuit-only variant.

Envelope V2 prematurely brakes ordinary launch wheel rotation and completes
0/4 with almost no progress; its camera regressions were red. V3 gates this
feedback to high speed and passes eight tests. An independent camera test then
shows row73 grass/road can inflate measured rear speed by over 50 m/s, creating
false spin braking. V4 excludes that row and passes nine tests, but its four
driving runs still fail: correcting the measurement exposes further control
and road-tracking limitations. The crop correction is not a lap improvement.

Physical measurements show full throttle can spin a sustained high-speed bend,
while gas0.3 remains stable in the measured .06 rad steering cases. The installed
Box2D maximum translation gives an approximately 100 m/s ceiling; full straight
launch reaches 99 m/s at 2.42 seconds. Neither these measurements nor a privileged
route model is an official lap or a proof of the minimum possible lap time.
Official Participants main was rechecked and remains `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`.

Current verified source-level regressions include 83 tests covering the existing
Apex evaluator/package/controllers and the preview/envelope candidates. The
previous full baseline's 15 dependency/provenance failures remain disclosed in
the preceding cloud report. Independent curved-pass, current-pose obstacle and
HUD-yaw investigations continue toward the original goal.

Nine stationary official renders calibrated the yaw HUD; earlier cropping
underread moderate and fast rotation. Path V5 fixes the crop and an invented
late obstacle pass, but both of its fresh settings fail mandatory coverage.
The curved preview V2 pass has valid sampled camera geometry but undertracks
the planned displacement: required curvature grows .0245→.0715→.1567/m in
.16 seconds, eventually exceeding the physical joint limit .1305/m. A later
hairpin cannot be represented by its single-valued road cubic. These are
causal failure findings, not reasons to relax the clearance checks.

Braking V1's feedforward still reserves tire force at saturated lateral load;
V2 corrects that concrete algebraic bug with a RED-to-GREEN camera regression.
It improves the two finished laps slightly but leaves the other two DNF.
Corridor V1 minimizes sampled local curvature within visible road and circle
bounds and passes four focused regressions. Only track 1 finishes; track 3's
100% tile progress is still an official DNF because no finish was recorded.

Privileged physics teachers and full-map offline route optimizers remain
diagnostic and cannot be adopted as camera agents. Thirty-two new teacher
episodes culminate in 14.22/17.98/16.66/15.60 s with zero contacts, still above
the goal. Converged local point-model time estimates are about
12.07/15.12/13.62/13.24 s, not physical laps, global optima or lower bounds.
SciPy 1.14.1 is optional offline research only; inference stays NumPy-only.

## Road support and current verification

Initial graph V1's four mandatory finishes were discovered after interrupted
parent orchestration; all workers had completed and their valid receipts are
preserved. Its saved-camera branch regression still fails. Graph V2 fixes the
car-fragment seed but removes a bounded recovery warning, losing three laps.
Explicit confidence checks the nearest ego component deliberately and freshly
reproduces all four V1 action hashes. The exact V1 source alias received a new
20-cell development benchmark: four exact mandatory repeats, 12/16 extra
finishes (4/4,4/4,1/4,3/4 by track), 21 total contacts. No new holdout was used.

Metric ridge reconstruction improves selected nearly horizontal bend errors
from about 2.5m to below 1m but full ridge hazard control fails three laps.
Clear-road-only ridge V1 preserves confidence hazard/recovery actions and
finishes three. Offline curvature measurements expose false startup and
horizon bends; smoothing reduces average error without eliminating systematic
bias. A 5.8m distance-support prefix restores track 4 in a fresh V2 benchmark;
track 2 still has three contacts and takes 30.24s. No source is adopted.

The completed full-suite checkpoint has 1396 passes, 10 skips, 15 known failures
in 450.19s. Its collection predates later controller/test changes; focused
tests verify those separately. The failures match existing four Box2D-version
expectations and eleven historical receipt/provenance cases. Protected root
and official simulator sources remain unchanged. The feasibility audit proves
no useful lower bound excluding 10s; offline route times are not impossibility
proofs or physical lap results.
