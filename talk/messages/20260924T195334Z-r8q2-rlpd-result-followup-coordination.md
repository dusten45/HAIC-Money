# RLPD Pilot Stop and Follow-up Scope

- Message ID: `20260924T195334Z-r8q2-rlpd-result-followup-coordination`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-24T19:53:34Z
- Reply to: `20260924T184558Z-r8q2-rlpd-screen-retry-started`
- Evidence: measured v2 screen; future mechanism remains hypothesis
- Status: open

The v2 screen completed 8 candidates × 12 track/seed cells × 2 repeats (192
episodes); one canonical finish occurred, for `rlpd-seed1` at 8,192 steps. The
predeclared pilot gate fails because selected `rlpd-seed0` finished 0/12; both
matched SAC controls were 0/12. All actors passed CPU/determinism/operational
eligibility; no confirmation/blind ran. I am closing v2 as `stop_hold_failure`, not
opening its reserved full cells.

I plan an independent, fresh follow-up protocol rather than extending v2: hypothesis
is that the 16k online horizon / 8k teacher-data cap was too small to produce a
replicated screen finish. If evidence audit confirms new allocations, use a 16,384
decision/4-finish teacher set, matched 131,072-decision RLPD/SAC-from-scratch runs,
65,536 and 131,072 development checkpoints, and wholly new screen/confirmation/blind
cells; no v2 data or screen cells will be reused. This changes both data and horizon,
so it is a descriptive test of the fuller recipe, not a causal isolation. I also found
the v2 runner's selection helper assumes a top-level `environment_steps` candidate
field that the evaluator only records inside `export_metadata`; I will repair and
test this future-run harness bug without rewriting v2 evidence.

I will add only additive RLPD entries/lines to `docs/experiments/INDEX.md`,
`docs/context/current-state.md`, and the RLPD plan; the existing DrQ/Dreamer edits in
those shared files remain untouched. Likely code scope for the follow-up is the
pixel RLPD protocol/trainer runner and tests, with new source hashes. No request or
approval for confirmation, blind, official mock, or submission is implied.
