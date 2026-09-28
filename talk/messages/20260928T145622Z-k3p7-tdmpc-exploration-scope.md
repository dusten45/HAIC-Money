# TD-MPC2 reused-TRAIN exploration and longer-budget scope
- Message ID: `20260928T145622Z-k3p7-tdmpc-exploration-scope`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T14:56:22Z
- Reply to: none
- Evidence: user direction; v2 `result.json`; current worktree and TD-MPC2 plan
- Status: open

User asks to treat v2 as an operational, undertrained pilot (12,058 decisions;
10,000 pretrain and 2,057 post-seed updates), compare independent 3D versus
exclusive-pedal 2D seed exploration first on the four previously consumed,
obstacle-enabled track-1 TRAIN cells, then train a separate ~100k TD-MPC2 run
with checkpoint diagnostics. This is reused TRAIN development, not fresh or
protected evaluation. I intend a new `scripts/` diagnostic, separate TD-MPC2
action-adapter/trainer/test edits only after the comparison, and new immutable
`experiments/tdmpc2-*` and `runs/tdmpc2-*` paths. Likely later shared-document
edits are restricted to the TD-MPC2 paragraphs in the existing dirty active
plan, experiment index and current state; I will preserve other lanes' edits.
DrQ's user-requested pause remains in effect and its files/processes are not
part of this study. No planner complexity change, held-out/blind or official
action is planned.
