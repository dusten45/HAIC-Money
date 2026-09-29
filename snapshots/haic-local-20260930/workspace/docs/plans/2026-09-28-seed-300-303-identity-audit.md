# Seed 300–303 split identity audit

The v2 manifest for `apex-line-three-track-heldout-20260928` registered tracks 1–3 × seeds 300–303 as `held_out`. Its `EXECUTION_STARTED` event is 2026-09-28 13:16:40 UTC and `EXECUTION_FINISHED` is 13:24:38 UTC. Its result was apex 11/12 versus selected control 12/12, so apex was rejected.

The separate `anticipatory-bend-fresh-tune-20260928` manifest was created at 13:27:02 UTC and lists exactly the same 12 track/seed identities as fresh tune. That was after the first run completed and consumed the cells as held-out. Its report says control 12/12, candidate 12/12, candidate median 24.87 versus 25.06 s, but those cells were **already opened**. Its code may have executed correctly; the `fresh tune` identity and independent evidence claim do not hold for the combined project record. Treat that later result as a consumed-cell diagnostic only. Preserve both immutable run records and use a genuinely new, registered split for any anticipatory-bend advancement.

Evidence: `runs/haic-research-v2/apex-line-three-track-heldout-20260928/{run_manifest.json,events.jsonl,integration_report.json}` and `runs/haic-research-v2/anticipatory-bend-fresh-tune-20260928/{run_manifest.json,integration_report.json}`. This audit changes interpretation, not prior run files.

## A second collision: seeds 308–311

The isolated v2 plan `haic2-four-direction-fresh-tune-20260928` registered tracks 1–3 × seeds 308–311 and emitted `EXECUTION_STARTED` at **13:34:19 UTC**. The separate `anticipatory-bend-package-confirmation-20260928` manifest was created at **13:44:24 UTC** with exactly the same twelve identities. Thus the latter confirmation cells had already been opened by the earlier four-direction tune execution. The confirmation report's candidate 11/12 versus control 10/12 and all three `PASS` gates are a recorded local diagnostic, but **not an independent confirmation set** in the project-wide split ledger. The later package cannot claim the required fresh confirmation evidence from these cells. Preserve both run histories and use new reserved cells if that candidate is considered again.

Evidence: `tmp/haic2-speed-envelope-frozen-20260928/runs/haic-research-v2/haic2-four-direction-fresh-tune-20260928/{execution_plan.json,events.jsonl}` and `runs/haic-research-v2/anticipatory-bend-package-confirmation-20260928/{run_manifest.json,integration_report.json}`. The four-direction process was still live when this audit was written; this note does not infer its final result.
