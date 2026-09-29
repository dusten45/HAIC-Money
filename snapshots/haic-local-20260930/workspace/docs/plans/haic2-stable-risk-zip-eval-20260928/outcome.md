# Frozen ZIP confirmation outcome

The exact candidate ZIP SHA-256 `7f9ef05e1e3366969f2d751191e926ad4cd9a517b376c8218bc1fbe61cb111f4` was compared with the fixed control ZIP SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40` on registered tracks 1–3, seeds 336–339. The Linux CPU diagnostic extracted each archive into its own clean directory and ran every arm and cell in a separate process. All 24 episodes completed technically; there were no invalid actions or contract limit failures.

| Metric | Candidate ZIP | Control ZIP |
|---|---:|---:|
| Valid finishes | 11/12 | **12/12** |
| Median finished lap | **23.54 s** | 25.23 s |
| Collisions / damage | 0 / 0 | 3 / 0.6 |
| Maximum act time | 32.51 ms | 23.47 ms |
| Maximum peak RSS | 301.5 MB | 301.5 MB |

The candidate was faster on all eleven jointly completed cells. On track 2 seed 336 it retired `off_track` after 280 decisions at progress 0.5862, while the control finished in 25.28 s. This is a candidate-only failure. The preregistered completion-first decision is **REJECT** despite the faster finished laps. Seeds 336–339 are consumed confirmation cells and cannot become fresh tuning evidence.

The run was registered after another exact-package comparison had already started using the same 336–339 cells. The first execution later failed its source-inventory check because shared files changed, but it produced the same 11/12 versus 12/12 result. Therefore this run is a **same-cell diagnostic reproduction, not independent fresh confirmation**. The original run files remain unchanged; a `CORRECTION` event referencing its integration report records the identity overlap. See [the first package investigation](../../experiments/stable-risk-exact-package-20260928.md).

The local package static checks and runtime limits passed. Official acceptance and score remain unknown. The next mechanism should investigate bend entry at speed using only pixel observations; the exact cause of the off-track event has not yet been measured. A separate registered TRAIN diagnostic should first show that a new mechanism changes actions, followed by fresh comparison cells. Do not change the frozen ZIP or reinterpret this run.

Evidence: `artifacts/haic-research-v2/haic2-stable-risk-zip-confirmation-20260928/report.json`, `runs/haic-research-v2/haic2-stable-risk-zip-confirmation-20260928/run_manifest.json`, and `runs/haic-research-v2/haic2-stable-risk-zip-confirmation-20260928/integration_report.json`.
