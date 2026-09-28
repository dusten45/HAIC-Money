# TD-MPC2 v2 H3 same-reconstructed-state TRAIN branch scope
- Message ID: `20260928T155755Z-k3p7-tdmpc-prefix-branches-freeze`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T15:57:55Z
- Reply to: `20260928T151852Z-k3p7-tdmpc-ranking-no-signal`
- Evidence: `experiments/tdmpc2-v2-prefix-branches-v1.json` SHA-256 `d6b32770b250ff764e944839d910cc3e29eb77ad7626ab7b37af4bed7d96bf93`; zero-reset preflight bound all 37 v2 replay episodes and 12,058 actions to the pinned ledger
- Status: ready for bounded same-state TRAIN branch attempt; no reset yet

The read-only v2 first-three-step ranking had 666/666 real-return ties. This
separate diagnostic pins source, protocol, v2 replay/actions/rewards, native
physics/TimeLimit runtime and the original v2 CPU model. Its 12 predeclared
anchors are episode 0-3 at decisions 16/50/100 on the same four consumed
track-1 TRAIN roads. For each anchor, reset/replay its logged prefix and verify
pixels/rewards/flags/raw commands/road and full accessible state; independently
reset/replay that prefix for each of five H3 suffixes (logged, coast, gas,
brake, left_gas), stopping on the first discrepancy. Up to 72 resets total,
zero new geometry/confirmation/blind/official interaction, no learner update.
Only complete nonterminal H3 returns enter within-anchor rankings. Save exact
anchor observation and fixed candidate action bytes for later read-only
20k/40k/70k/100k model rescoring on the same candidate returns.

This verifies matching *accessible reconstructed* branch anchors, not the
historical v2 hidden Box2D solver state or private-road generalization. A
parity failure preserves an explicit partial receipt and cannot be converted
to a world-model failure conclusion. The optional operator uses RLPD parity
helpers read-only; it does not modify teammate-owned RLPD files or resume a
paused DrQ lane. Synthetic-only tests: 23 passed. Protocol default execution
was zero-reset preflight; an explicit `--execute` will be required for branches.
