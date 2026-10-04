# V2 experiment scoreboard

Receipt snapshot: 2026-10-04T21:06:06.298114+00:00. No new resets.

Explicit planned scopes retain missing and started cells. DNF means completed without a finish. Required4 coverage is reported even for smaller diagnostic allocations. Each source + exact config is separate; source/dependency verification failures invalidate aggregate finish claims.

| Candidate | Required4 finishes | Required lap times (s; T1–4) | Declared scope finishes | Consumed subset | Pending / missing | Code / resource failures | Warnings |
|---|---:|---|---|---|---:|---|---|
| actuator-r0 50b0206 | 4/4 | 16.86 / 21.16 / 18.56 / 19.26 | screen6: 5/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| brake-r0 c99895c | 3/4 | 16.82 / 21.36 / 18.92 / DNF | screen6: 4/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| exact-pedal-r0 ece5533 | 1/4 | DNF / 21.50 / DNF / DNF | screen6: 1/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| force-r0 01f42dc | 2/4 | DNF / 20.32 / 18.58 / DNF | screen6: 3/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| friction-envelope-r0 73177ff | 1/4 | 14.90 / DNF / DNF / DNF | screen6: 1/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| predictive-r1 bb5f395 | 0/4 | DNF / DNF / DNF / DNF | screen6: 0/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| obstacle-shield-r0 070ccc2 | 3/4 | 16.78 / 20.76 / 18.56 / DNF | screen6: 4/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| obstacle-steering-r0 1121447 | 4/4 | 16.78 / 20.72 / 18.56 / 17.88 | screen6: 5/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| obstacle-road-r0 7b21335 | 4/4 | 16.78 / 20.68 / 18.56 / 17.86 | screen6: 5/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| kinematic-r0 7e2087e | 1/4 | 15.42 / DNF / DNF / DNF | screen6: 1/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| beam-r1 acbb747 | 0/4 | DNF / DNF / DNF / DNF | screen6: 0/6 | 0/2 | 0 / 0 | 6 / 0 | 0 |
| beam-r2 aa2198a | 0/4 | DNF / DNF / DNF / DNF | screen6: 0/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| beam-r3 952c7c0 | 4/4 | 14.04 / 17.24 / 15.80 / 14.94 | screen6: 5/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| recovery-r0 08c6ff9 | 4/4 | 16.78 / 20.68 / 18.56 / 18.10 | screen6: 4/6 | 0/2 | 0 / 0 | 0 / 0 | 1 |
| recovery-r1 e83b363 | 4/4 | 16.78 / 20.68 / 18.56 / 18.10 | screen6: 5/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| shield-r0 51775a9 | 3/4 | 21.28 / 28.12 / 25.66 / DNF | screen6: 5/6 | 2/2 | 0 / 0 | 0 / 0 | 1 |
| shield-r1 070d788 | 4/4 | 19.42 / 25.50 / 22.02 / 20.66 | screen6: 6/6 | 2/2 | 0 / 0 | 0 / 0 | 0 |
| speed-r0 aa02e09 | 3/4 | 16.48 / 20.04 / 18.38 / DNF | screen6: 4/6 | 1/2 | 0 / 0 | 0 / 0 | 0 |
| yaw-r0 8db09cc | 4/4 | 16.54 / 20.32 / 18.44 / 18.52 | screen6: 4/6 | 0/2 | 0 / 0 | 0 / 0 | 0 |
| geodesic-r0 d621f9f | 3/4 | 17.80 / 21.52 / DNF / 18.06 | required4: 3/4 | — | 0 / 0 | 0 / 0 | 0 |
| geodesic-frenet-r5 cf44d28 | 3/4 | 18.32 / 21.96 / 20.10 / DNF | required4: 3/4 | — | 0 / 0 | 0 / 0 | 0 |
| predictive-r0 63b698d | 3/4 | 24.20 / 24.86 / 25.06 / DNF | required4: 3/4 | — | 0 / 0 | 0 / 0 | 0 |
| fallback-r1 82b02f5 | 2/4 | 18.06 / 22.40 / DNF / DNF | screen8: 4/8 | 2/4 | 0 / 0 | 0 / 0 | 0 |
| geodesic-r2 f91a327 | 4/4 | 17.58 / 21.56 / 19.44 / 17.72 | screen8: 6/8 | 2/4 | 0 / 0 | 0 / 0 | 0 |
| geodesic-frenet-r6 6b96642 | 4/4 | 17.14 / 21.00 / 18.94 / 17.32 | required4+full_consumed_regression24: 24/28 | 20/24 | 0 / 0 | 0 / 0 | 0 |
| geodesic-footprint-r7 d39a5d8 | 4/4 | 17.14 / 21.10 / 18.84 / 17.22 | required4+full_consumed_regression24: 23/28 | 19/24 | 0 / 0 | 0 / 0 | 0 |
| racing-r3 e3c6fc5 | 4/4 | 19.26 / 23.80 / 21.58 / 19.48 | screen8: 7/8 | 3/4 | 0 / 0 | 0 / 0 | 0 |
| racing-r4 4f9300a | 4/4 | 21.92 / 27.00 / 23.64 / 22.00 | screen8: 6/8 | 2/4 | 0 / 0 | 0 / 0 | 0 |
| geodesic-r1 dde6bf2 | 4/4 | 18.74 / 22.18 / 20.64 / 19.92 | required4+full_consumed_regression24: 26/28 | 22/24 | 0 / 0 | 0 / 0 | 0 |
| terrain-r8 f8039f6 | 4/4 | 16.62 / 20.66 / 18.50 / 17.04 | required4+full_consumed_regression24: 26/28 | 22/24 | 0 / 0 | 0 / 0 | 0 |
| curvature-memory-r0 bfcc592 | 4/4 | 18.30 / 21.66 / 19.98 / 18.06 | screen9: 7/9 | 3/5 | 0 / 0 | 0 / 0 | 0 |
| terrain-grounded-r9 2361296 | 4/4 | 16.74 / 20.72 / 18.64 / 17.02 | required4+full_consumed_regression24: 25/28 | 21/24 | 0 / 0 | 0 / 0 | 0 |
| obstacle-steering-audit 1121447 | 0/4 | — / — / — / — | diagnostic1_not_benchmark: 0/1 | 0/1 | 0 / 0 | 0 / 0 | 0 |
| beam-r5 589c8de | 1/4 | 14.28 / — / — / — | screen6_initial2_authorized: 1/6 | 0/2 | 0 / 4 | 0 / 0 | 0 |
| motion-registration-baseline-diagnostic f54347e | 1/4 | 16.78 / — / — / — | diagnostic1: 1/1 | — | 0 / 0 | 0 / 0 | 0 |

Nonbenchmark diagnostics and sources:
- **curvature-memory-bugfix-untested**: Bug-fixed source used for counterfactual analysis only; no driving receipt, not evaluated.
- **beam-r4-offline**: Offline only:14 tests and120 independent geometry checks passed, but all19 recorded pre-impact first actions remain unchanged. State/mask2x2 counts0/4/2/2 demonstrate fragile zero-clearance plans. No closed-loop success or failure claim;0 resets.
- **beam-r5-frozen-action-capture**: One separately authorized diagnostic reset;140 frozen actions replayed with exact before/after parity and81 captured observation stacks. No policy act calls; not a screen retry or independent candidate evaluation.

**beam-r5 authorization:** Six cells declared; only oldT3 and required1 initially authorized. Remaining4 require separate review/authorization; no automatic full24. R4 remains offline-only.
Initial authorized phase: 1/2 finishes, 0 incomplete, 0 missing. Conditional phase: 0/4 completed, 0 incomplete, 4 missing. The screen denominator remains6.

V1 historical baseline (not new v2 validation):
- historical_required4: 4/4 finishes; completed 4; missing 0.
- historical_consumed24: 22/24 finishes; completed 24; missing 0.

Unregistered primary receipts: 0. Add an explicit registry allocation before comparing them.

The geodesic full24 regression includes its two earlier probes exactly once. Overlapping required/screen summaries must not be added together. Retry policy: deduplicate the same episode, then select the earliest started attempt per cell; preserve all attempts, never choose the fastest finish.

Rebuild from the repository root:

```sh
.venv/bin/python agents/apex_2026/v2/diagnostics/experiment_index.py
```

Use `--registry path.json` to replace the manual registry with `{label: {folders: [...], expected_cells: [[track,seed],...], scope: "..."}}`. Expected cells must be declared explicitly; new folders are not auto-promoted.

[Primary receipt index](results/experiment-index.json) · [Generator](diagnostics/experiment_index.py)

Diagnostic partitions are excluded from benchmark totals and do not constitute independent validation. Counterfactual-only sources have no driving evidence.

Finished-only medians in JSON are descriptive and never replace the full declared denominator. No candidate is promoted by this table.
