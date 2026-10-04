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
