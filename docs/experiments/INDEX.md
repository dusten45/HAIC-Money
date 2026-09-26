# New HAIC experiments

List only experiments created under the v2 research workflow. Historical `RESULTS.md` rows and legacy run directories remain at their original paths and are not imported here.

| Run ID | Hypothesis | Registered split and control | Decision | Manifest | Integration report |
|---|---|---|---|---|---|

For each new record, include the mechanism, observable signal, endpoint, falsifier, resource gate, all evaluated cells, and source paths. Preserve failures and unmatched observations.

## Recording new runs

Use `python -m haic_research.cli --root <repo> report <run-id> --outcome <outcome> --report <gate-report.json> --result <candidate.json> [--control <control.json>] [--promote-sota]`. Inputs are explicit repository JSON files outside historical paths. Existing gate-only reporting remains available without result inputs.

The writer creates immutable `docs/experiments/<run-id>.md` detail and appends a run summary table below. Result and comparison JSON are confined to that named v2 run and must precede the immutable gate report. Duplicate run IDs are refused; a correction uses a new run record and references the original evidence rather than overwriting it. Failed, diagnostic and unmatched observations can be recorded and cannot establish SOTA. No actual v2 experiment was created by implementation tests.
