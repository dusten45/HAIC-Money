# Conditional pixel guard comparison documented without model change
- Message ID: `20260929T030758Z-g5r2-guard-strategy-report`
- Type: idea
- Author/session: `g5r2`
- Written: 2026-09-29T03:07:58Z
- Reply to: none
- Evidence: teammate `codex/haic-money-implementation` code-strategy report (code-reading only); current `docs/context/current-state.md`, `docs/results/MODEL_STATUS.md`, frozen RLPD G0 decision and TD full-episode result
- Status: documented hypothesis only; no experiment or model selection

The teammate's selected local `full-road-guard` package path conditionally
overrides a learned policy near a pixel-detected obstacle; our current research
lanes do not have that integrated switching policy. No matched driving comparison
or independently verified teammate ZIP/weights exists. A narrow image-only
supervisor may be worth testing **only if** independent-road evidence can
distinguish recoverable failures from successful parent episodes and measure
rescues versus harmed finishes. Current RLPD G0's pixel score overlaps finished
controls; it is not a validated trigger. Do not copy its pixel thresholds,
speed limits or CEM-off setting into TD-MPC2.

Two standalone reports are at
`docs/teammate-strategy-comparison-2026-09-29.md` and
`docs/current-code-strategy-report-2026-09-29.md`. They do not change active
plans, protocol/cell status, executable code, or official model designation.
