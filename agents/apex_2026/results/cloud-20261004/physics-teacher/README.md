# Privileged physics teacher diagnostic, 2026-10-04

This is an **offline diagnostic**, not a legal camera agent, a prospective
development rejection, a profile result, or proof that 10–13 second laps are
possible or impossible. The controller read the simulator's complete road
waypoints, six obstacle positions, and vehicle pose and velocity. Official
`Agent.act()` receives only four camera frames; it cannot use those raw values.
No holdout geometry was opened. No root agent or official simulator file was
changed.

The exact [receipt](teacher-v1.json) is 21,267 bytes (SHA256
`c7892e11eb15f4024b8b49654bee1061b0a5af69a0431a84d7787b031c2993f4`).
The exact [throwaway source](apex_privileged_teacher_v1.py.txt) is 11,684 bytes
(SHA256 `58397b25c22e19cdb703bd3bcfe22e5fd443a1c4c333d207a28de0bbf45e3b80`).
Its `.txt` extension prevents confusing it with a submission Python module;
the source bytes and receipt's source hash are unchanged. The receipt also
contains seven official environment file hashes and the action trace hash for
each run. The official Participants `main` checked for this study was
`dfb7a2de2178825ca5c5ce20bab01ba67052ba31` (`variables-6`).

The source hardcodes `ROOT = Path("/workspace/HAIC-Money")` for provenance hashes.
It runs unchanged only with the checkout at that path and the project on
`PYTHONPATH`. Moving the checkout would require changing the source bytes and
would produce a different source hash; the archived receipt would remain the
record of these exact runs. The run used the existing project Python 3.11.16
environment, one worker, frame skip four, 50 raw warmup frames, and a bounded
600 action cap. All four DNFs retired before that cap. No rerun is part of
this archive.

## Measured mandatory cells

Lap time is the official finish physics tick minus the post-warmup start tick.
Progress alone never counts as a finish. The two rows per pair change cruise
and lateral acceleration targets from 110/180 to 140/210; both use the same
privileged obstacle route and 85 m/s² *assumed* braking envelope.

| Track / seed | 110/180 diagnostic | 140/210 diagnostic |
|---|---|---|
| 1 / 516237 | Finish 13.78 s; 0 contacts | Finish 13.26 s; 0 contacts |
| 2 / 644062 | Finish 17.74 s; 0 contacts | DNF 70.2% progress; 0 contacts |
| 3 / 1007 | Finish 15.98 s; 0 contacts | DNF 37.2% progress; 0 contacts |
| 4 / 18800 | DNF 74.2% progress; 1 contact | DNF 60.2% progress; 0 contacts |

All DNFs were `off_track` retirements, which the official wrapper defines by
101 consecutive actions with negative summed raw reward. The 110/180 track 4
run had zero sampled full or partial road exits despite its one contact; the
receipt does not establish that the contact caused the retirement. The faster
140/210 runs on tracks 2, 3, and 4 recorded 90, 33, and 104 sampled full road
exits respectively. This setting improved track 1 by 0.52 s but lost the road
on the other three tracks. Vehicle speed reached about 100 m/s in every run,
matching Box2D's 2 m maximum translation at a 0.02 s physics tick. The route
lengths in the receipt are those of this teacher's chosen path, not shortest
paths or lap time lower bounds.

## Camera control guidance from these measurements

- The teacher started its lateral obstacle shift roughly 32 m before each
  obstacle and used a 12–34 m steering preview. At 90–100 m/s, 32 m is only
  0.32–0.36 s, about four actions. A camera controller can measure orange
  obstacle pixels and visible road edges, then lock a pass side at first
  reliable detection and brake if the visible corridor cannot support it.
- The existing camera calibration uses `(63 - row) / 1.701` metres forward.
  The image top is about 37 m ahead; the current road parser's farthest row 5
  is about 34 m. The teacher's speed planner examined curvature as far as
  100 m ahead. A camera policy can measure its **actual farthest contiguous
  road row** and HUD speed each action, then lower its speed target when the
  visible braking distance is insufficient. For an illustrative 50 m/s
  upcoming turn, this teacher's unverified 85 m/s² braking assumption requires
  44 m to slow from 100 m/s. With only 34 m visible and a 5 m margin, the same
  formula caps approach speed near 86 m/s. Actual braking and the resulting
  safety margin need measurement in camera-only development runs.
- Log farthest valid road row, detected obstacle distance and side, HUD speed,
  commanded braking distance, and later road/contact outcomes together. This
  tests whether early detection or the braking envelope is the missing piece.
  Extrapolating curvature beyond the image is an uncertain prediction, so
  missing far road evidence should lower speed rather than certify a clear
  path. These are testable camera-only control ideas, not measured gains.
