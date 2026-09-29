# Stable road-clearance ZIP: local package check (2026-09-28)

## Established controller evidence

The unchanged pixel-only controller `StableRiskEnvelopeAgent` outperformed selected full-road-guard on three registered controller comparisons: tune 324–327 **12/12 vs 12/12** with medians **23.22 vs 24.70 s**; held-out 328–331 **11/12 vs 9/12** with medians **22.36 vs 24.98 s**; one-time confirmation 332–335 **12/12 vs 11/12** with medians **22.71 vs 24.26 s**. All completion counts are actual finish-line crossings. These are controller-level local diagnostics, not package or official scores. The source is frozen at SHA-256 `31eed1bc0e566d89b7b1d5fc33cb1f28f7662f383e1ae21f7ae814142997c86b`.

## Exact local ZIP and package result

The registered `stable-risk-frozen-package-20260928` operation built `artifacts/haic-research-v2/stable-risk-frozen-package-20260928/submission.zip`, SHA-256 `fd837b6cfeb41bcf7011991b929a25c77ba22bfd6b631d3e46803921121e4169`, from the confirmed runtime and selected control ZIP SHA-256 `5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40`. The only new member is the confirmed runtime; the root entrypoint wraps the same full-road-guard controller. Static inventory and forbidden-source checks passed. Package run gates remained `UNKNOWN` pending driving verification and had no release.

On tracks 1–3 × seeds 336–339, the first exact-package comparison artifact `artifacts/haic-research-v2/stable-risk-exact-package-confirmation-20260928/report.json` shows the candidate **11/12** vs selected ZIP **12/12** actual finishes. Candidate finished-lap median **23.54 s** vs control **25.23 s**. The candidate's sole failure was track 2 / seed 336 at 58.62% progress, `off_track`; the control finished that cell. There were zero invalid actions. Import/create maxima were 1.08 s candidate and 1.06 s control; candidate max reset 0.00003 s, RSS 302 MB, max act 34.5 ms. The faster finishes cannot override lower completion.

The first comparison's CLI execution began at 14:48:52 UTC and generated the complete result, but its post-run source inventory check failed when `haic_agent/anticipatory_fast_cruise_brake_runtime.py` was added concurrently to the shared root. The run has no `EXECUTION_FINISHED` or integration report and is **infrastructure-invalid for promotion**. Preserve that partial history and output without overwriting them.

A separate frozen-snapshot operation `haic2-stable-risk-zip-confirmation-20260928` registered the **same 336–339 cells at 14:51:13 UTC**, after the first run had started. It used ZIP SHA-256 `7f9ef05e1e3366969f2d751191e926ad4cd9a517b376c8218bc1fbe61cb111f4`; its bundled members match the first candidate byte-for-byte except the root `agent.py` docstring, and both entrypoints build `StableRiskEnvelopeAgent(ObstacleFullRoadGuardAgent(base))`. The frozen run also measured **11/12 vs 12/12**, medians **23.54 vs 25.23 s**, and recorded `REJECT → GATE_REVIEW_REJECT → STOPPED` with no release. Since its identities overlap the earlier execution, it is corroborating same-cell diagnostic evidence, not a second independent confirmation. No candidate ZIP is promoted.

## Decision and next mechanism

Reject the current speed envelope as the selected package because completion fell below the control. Keep the selected full-road-guard ZIP. Use fresh TRAIN-only cells to investigate whether stronger road reacquisition or an earlier bend/obstacle risk signal can preserve the speed gain; register a new four-direction batch and fresh tune identities before another comparison. Do not alter or tune on seeds 336–339. No competition-site upload, official submission, or model confirmation occurred.
