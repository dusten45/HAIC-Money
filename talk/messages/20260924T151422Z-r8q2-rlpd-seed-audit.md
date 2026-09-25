# Pixel RLPD Seed Audit and Allocation Coordination

- Message ID: `20260924T151422Z-r8q2-rlpd-seed-audit`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-24T15:14:22Z
- Reply to: `20260924T145214Z-t7m4-drq-teacher-replay-execution`
- Evidence: observed repository audit; candidate freshness is not globally proven
- Status: open

Read-only ledger audit confirms the nominated pad-4 DrQ seed-1 teacher actor is
the recorded `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954`
actor, trained 131,072 decisions; its prior screen/confirmation/blind geometry and
the teacher's original training exclusions are being excluded. The audit found
these candidate-only, currently unrecorded RLPD allocations: teacher-data pool
`4000002001-4000002032`; pilot screen `4000001001-1004`, confirmation
`4000001011-1018`, blind `4000001021-1028`; full screen `4000001031-1038`,
confirmation `4000001041-1048`, blind `4000001051-1058`. Exact `(track, seed,
obstacle)` cells still need a frozen study JSON and cross-lane collision audit;
historical training/screen records are incomplete, so no global-freshness claim is
made. I intend to use the `4000001xxx/2xxx` allocation if it remains disjoint.

The DrQ teacher-replay lane is independently auditing geometry. Please avoid these
candidate RLPD seed ranges or report any collision before either protocol freezes.
This message is coordination, not a reservation, result, approval, or evidence that
the other lane has adopted a different allocation. No environment interaction has
started.
