# Archive identifies a newer baseline than the downloadable release
- Message ID: `20260930T073800Z-b8k4-baseline-identity-correction`
- Type: response
- Author/session: `b8k4`
- Written: 2026-09-30T07:38:00Z
- Reply to: `20260930T072443Z-b8k4-koi-baseline-analysis-scope`
- Evidence: observed frozen archive metadata and runtime code
- Status: analysis only

Correction to the earlier inference: the user selected the strongest model in
the KOI commits, not `arrival_speed` by name. The downloadable release is arrival,
but `snapshots/haic-local-20260930/workspace/docs/experiments/current-best-validated.json`
explicitly names the later `contact-package-20260930/submission-crossing-projection.zip`
at SHA-256 `4447c8dfad1392ddbbb83f0b3d23b8d5237af2b9be785cdfd08edaf6c4623f64`
and nests arrival under `previous_current`. Analyze both the latest archived
candidate and its exactly available release parent. Do not transfer either
cohort's scores to the other candidate or describe archival summary counts as
independently reverified raw episodes. No improvement work or external action.
