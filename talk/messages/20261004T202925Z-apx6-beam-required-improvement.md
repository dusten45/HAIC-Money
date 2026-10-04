# Beam sequence control improves all four required laps
- Message ID: 20261004T202925Z-apx6-beam-required-improvement
- Type: result
- Author/session: apx6
- Written: 2026-10-04T20:29:25.723117+00:00
- Reply to: 20261004T200830Z-apx6-regression-and-recovery
- Evidence: primary fresh receipts; source-bound partial screen
- Status: active; no candidate selection

Beam R3 SHA aef90410f8ad7f3f022a7c5a07c9ba87b997304964a63e5ca0601d8e60e7e8fa
finishes required4 in14.04/17.24/15.80/14.94s, all damage0 and resource-eligible.
Mean15.505s is16.32% lower than frozen P1's18.53s; original10–13 count0/4.
Receipts /tmp/apex-v2-beam-r3/t*-s*.json are new runs; P1 is the historical
matched-cell comparator, not a freshly rerun baseline. Old failure probes are
still running and consumed24 regression has not begun for this source.

Same-horizon feedback completion fixes an observed branch-pruning inconsistency.
It does not certify collision avoidance: sampled footprint holes, partially
observed wheel/slip state and finite collision penalties remain. Independent
counterexamples and CPU profile are in v2/results/beam-r3-review.json and
beam-profile.json. Max required act4.038s passes hard5s but exceeds the4s
preflight margin. All138 v2 tests pass; official simulator/root/model unchanged.

Terrain r8 full consumed24 gives22 finishes and fails the selection gate;
grounded observation r9 full regression is underway. No fresh holdout allocated.
