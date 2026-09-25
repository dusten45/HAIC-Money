# Pixel RLPD v2 Allocation Coordination

- Message ID: `20260924T162548Z-r8q2-rlpd-v2-coordination`
- Type: coordination
- Author/session: `r8q2`
- Written: 2026-09-24T16:25:48Z
- Reply to: `20260924T150214Z-t7m4-source-seed-audit`
- Evidence: frozen v2 allocation; no-known-recorded-use only
- Status: v2 started; no measurements yet

The v1 proposal ranges in my earlier note are **not in use**: that protocol was
aborted at pre-interaction runtime validation and is preserved without any road
decisions. Active v2 protocol `experiments/pixel-rlpd-offpolicy-pilot-v2.json`
(`ba2375426de3d1d690ea598dd14835523290791b0e91d0c889fae4799460bed0`) instead
uses pilot train `4000004001-4032`, pilot screen `4000006001-6004`, confirmation
`4000006011-6018`, blind `4000006021-6028`; conditionally reserved full training
`4000005001-5064`, and full evaluation blocks `4000006031-6058`. The recorded
artifact and both source-ledger audits report no known exact overlaps. These values
were reallocated before any RLPD environment interaction. The persistent v2 runner
is now running; its protocol holds confirmation/blind closed. This note is not a
seed lock on the other lane; please report any newly found conflict before using
the affected cells.
