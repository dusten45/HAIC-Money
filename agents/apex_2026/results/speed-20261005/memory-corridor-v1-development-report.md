# Memory corridor V1: consumed development benchmark

## Frozen run

The candidate was committed as `42c10637f28ecaf9aaaa38ad782485240cb705c9`
before this fresh run. Source SHA256:
`377b925b98880c5ba55bf5179f79e37c4400f5883ae0ed800e601910d7df2474`.
Parameters are `{}`; evaluator SHA256 is
`39a280612b3f5e7402b7a475c447a7cb9c465ddc0bc9e5c5f98300cdcad0325c`.

Command: `.venv/bin/python -m agents.apex_2026.evaluate --source
agents/apex_2026/fast_memory_corridor_agent.py --suite development --max-steps
700 --workers 2 --role benchmark --output
agents/apex_2026/results/speed-20261005/memory-corridor-v1-development.json
--trace-dir .haic-artifacts/apex-speed-20261005/memory-corridor-v1-development-traces`.
The process completed with exit status 0. No source or parameter changes were
made, and no new holdout episodes were opened. The four extra geometry seeds
are previously consumed development data. Benchmark role does not advance the
prospective rejection chain or authorize a relaxed profile.

## Results

| Track | Mandatory seed | Actual lap, s | Contacts | Extra finishes |
| --- | --- | ---: | ---: | ---: |
| 1 | 516237 | 15.20 | 0 | 2/4 |
| 2 | 644062 | 19.72 | 0 | 4/4 |
| 3 | 1007 | 18.78 | 0 | 3/4 |
| 4 | 18800 | 21.14 | 0 | 4/4 |

Extra completion is **13/16**, with 15 obstacle contacts in total. Completed
extra laps range from **14.76 to 25.52 s**. No mandatory or extra lap meets
13 s. At 18 s, only 1/4 mandatory and 4/16 extra cells qualify. The selected
13 s profile fails and the requested four early-10-second laps remain unmet.
This is a research checkpoint, not a formally adopted candidate.

| Extra seed | Track 1 | Track 2 | Track 3 | Track 4 |
| --- | --- | --- | --- | --- |
| 2867319041 | DNF, progress .4399, 4 contacts | 20.32 s, 0 contacts | 20.40 s, 0 contacts | 24.20 s, 0 contacts |
| 359018627 | DNF, progress .4146, 3 contacts | 21.22 s, 0 contacts | DNF, progress .1603, 0 contacts | 16.78 s, 0 contacts |
| 1764402399 | 21.04 s, 3 contacts | 16.04 s, 0 contacts | 18.90 s, 0 contacts | 19.18 s, 2 contacts |
| 4029571806 | 17.52 s, 0 contacts | 18.02 s, 0 contacts | 25.52 s, 2 contacts | 14.76 s, 1 contact |

All three DNFs retired as `off_track`; they are not operational errors. The
track-3 failure without obstacle contacts shows that obstacle avoidance alone
does not explain the remaining failures.

Compared with the earlier component V1 benchmark on these same 20 cells,
extra completion changes from 12/16 to 13/16 and extra contacts from 18 to 15.
Three finishes are gained: `(3,2867319041)`, `(4,1764402399)` and
`(3,4029571806)`. Two track-1 finishes are lost: `(1,2867319041)` and
`(1,359018627)`. Across the 14 cells completed by both sources, total lap time
changes from 285.32 to 263.64 s. These paired observations do not establish
unseen reliability or erase the lost finishes. Comparison input
`component-v1-development.json` has SHA256
`eadd8c54fec902ba4fc6d63c28a5df2d5087c6c2e7f003b774fa52a04b016e7a`.

## Independent verification

All four required cold repeats match their previous `memory-corridor-v1.json`
rows on all 15 nonruntime fields, including exact action hashes. Their full
raw trace JSON also matches exactly. A second agent independently checked the
four row/hash repeats and environment binding.

All 20 trace action hashes were independently recomputed from row-major
float32 action bytes. Action shapes, finite values, official bounds, step
counts, contact counts, final progress and finite simulator/debug values pass.
Every row matches its per-cell receipt, source and parameters. There are exactly
20 declared unique cells and no worker errors. All 11 environment/evaluator
hashes remain frozen; the root agent and seven official files match starting
commit `14bb967` byte for byte.

Local worker maxima are initialization **0.764 ms**, reset **0.015 ms**,
action **39.843 ms**, and peak RSS **95.211 MiB**, all finite and within their
10 s / 5 s / 5 s / 1024 MiB limits. NumPy is already imported by the evaluation
worker before the recorded initialization interval. These are local harness
measurements, not official submission-container certification.

The complete input hashes, per-trace hashes, assertions, runtime values and
failure details are in `memory-corridor-v1-development-lineage.json` (SHA256
`c3acc7a27e4c7aad9f54ce3907b0851d2182c5f847a6033db3de2a5598788e8f`).
The development receipt SHA256 is
`adbbe5ed1f9f7e9bb8cc7a5afbd8dc443c73cc2fb793910c7ccc7773b4db3379`;
its freeze SHA256 is
`6954610d3a84a81f1adb0f69bd92eba6109420c93e384f311787d0ae2edbf824`.
