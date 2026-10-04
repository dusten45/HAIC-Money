# Frozen KOI arrival-speed baseline analysis only
- Message ID: `20260930T072443Z-b8k4-koi-baseline-analysis-scope`
- Type: coordination
- Author/session: `b8k4`
- Written: 2026-09-30T07:24:43Z
- Reply to: none
- Evidence: direct user scope and observed Git/release metadata
- Status: analysis only

User designates the model delivered by KOI commits `1e3dd6d`/`c4e224d`
(merged by `7dfb327`/`283cc59`) as the future improvement baseline and requests
precise analysis and documentation, explicitly no improvement work. The release
entry point is `FarHazardAgent('arrival_speed')`, not this worktree's research
agent or the later local RLPD gate.

`git pull --ff-only --no-autostash origin main` fetched successfully but refused
integration because existing README/current-state/index/RLPD-plan edits overlap.
User approved immutable extraction of `c4e224d` into
`/tmp/kilo/koi-baseline-c4e224d`; all existing tracked, staged and untracked work
is preserved. Planned workspace edits are a new baseline analysis document under
`docs/architecture/` and these separate coordination/result notes only. No shared
runtime/status/plan edits, environment resets, training, tuning or official action.
