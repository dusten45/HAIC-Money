# Small navigation lane: rejected benchmark candidates

The new `navigation_agent.py` is a 353-line, standalone NumPy camera controller.
It reconstructs the observed road centerline and adds smooth offsets relative to
that curve for circle obstacles. It contains no simulator, root-agent, seed or
model imports. The active root agent and official simulator were not edited.

## Sources and tests

- V1 SHA256: `d7a9f3727e0274871a5f60116a6fb3b94c784b1215d32096407d9729c53d4e22`.
- Preserved V2 SHA256: `d59a354d2e4189b4db7fdd3ab2e2b7028429263bdacdd5efb5033f48ae467683`.
- Evaluator throughout: `39a280612b3f5e7402b7a475c447a7cb9c465ddc0bc9e5c5f98300cdcad0325c`.
- Reverse `probes/navigation-v1-to-v2.patch` against V2 to recover V1; the
  recovered bytes were checked against the V1 SHA256 above. Its original source
  and large traces also remain in ignored `.haic-artifacts/` directories.
- Nine new behavioral tests first failed and then passed. The last run of all
  six existing Apex test files plus the navigation file passed 70 tests in
  1.43 seconds. The root agent did not depend on this new file.

Review identified a planned pose that moved sideways instantly and insufficient
forward visibility. V2 starts the path at the actual ego pose with zero initial
heading, checks the rotated hull against road edges and obstacle circles,
rejects curvature above `tan(0.4)/3.24`, checks steering lag, front clearance and
braking reachability, and uses bounded recovery below six metres of visible
forward road. These checks pass synthetic tests; they do not establish driving
success.

## Fresh mandatory screens

Every screen used four cold workers serially, frame skip 4, max 700 decisions,
and the official finish crossing after warmup. The `benchmark` role means these
screens do not advance prospective development rejection counts.

| Screen | Track 1 / 516237 | Track 2 / 644062 | Track 3 / 1007 | Track 4 / 18800 |
| --- | --- | --- | --- | --- |
| V1 default | 21.78 s, 4 contacts | DNF 30.65%, 5 contacts | 21.92 s, 0 contacts | DNF 52.33%, 4 contacts |
| V2 default | DNF 13.57%, 3 contacts | DNF 20.54%, 0 contacts | DNF 19.09%, 2 contacts | DNF 11.83%, 2 contacts |
| V2 conservative | DNF 13.57%, 1 contact | DNF 20.54%, 0 contacts | DNF 19.09%, 2 contacts | DNF 11.83%, 1 contact |

Defaults were cruise 95, lateral acceleration 125, preview 0.16, braking
acceleration 95, obstacle speed 48 and clearance 0.35. The conservative screen
used cruise 65, lateral acceleration 110, braking acceleration 120 and obstacle
speed 30 with the other defaults unchanged.

The requested diagnostic `(1,1764402399)` also failed: V1 crashed at 33.916%
with five contacts; V2 conservative stopped at 19.231% with two contacts and
retired as `off_track`. All receipts have source, parameters, environment and
action hashes. No worker reported a runtime error. The largest measured action
latency on these screens/diagnostics was below 40 ms; memory was below 100 MiB.

## What the experiment showed

V1's geometrically clear future path did not ensure that short-preview steering
reached it before contact on a bend. V2's necessary reachability checks exposed
blocked approaches but then stopped indefinitely, causing the official
no-progress retirement. Lower approach speed reduced contacts without restoring
completion. Thus the fixed lane-offset family does not yet provide a feasible
closed-loop route through these observed geometries.

All three mandatory screens are rejected. Neither the original 13-second goal
nor any relaxed profile was achieved. This source is not adopted, no additional
development suite was opened for it, and holdout remained unopened. A future
attempt needs a path search that finds reachable alternatives from the current
pose and handles blocked approaches without indefinite stopping.
