# High-speed camera continuation

The user resumed the strict 10–13 second target on 2026-10-05 (Asia/Seoul),
starting from `a30adbb`. Root inference and official physics remain protected.
Previous holdout seeds are consumed development data; no new holdout has opened.
No candidate in this series has achieved the goal or been adopted.

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
