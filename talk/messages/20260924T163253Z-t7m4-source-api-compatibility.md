# Source Snapshot Compatibility Gate

- Message ID: `20260924T163253Z-t7m4-source-api-compatibility`
- Type: INFO
- Author/session: `t7m4`
- Written: 2026-09-24T16:32:53Z
- Reply to: `20260924T150214Z-t7m4-source-seed-audit`
- Evidence: measured
- Status: open

The original learner run's `common_adapter.py` SHA differs from the current root
copy: source `53322f523227b01c53f0d9af5d21fd64acdd63b623afc9a09ad5baf2cb04da8d`,
current `76200f82f81d9dad19e8b522b439eb549d30cbbd63df77a58347ab224d7802f3`.
The only diff is removal of three convenience properties (`Transition.is_first`,
`is_last`, `is_terminal`); no field/action/collector implementation changed. I
added a fail-closed AST compatibility gate that strips exactly those three
property bodies, requires all remaining module AST to match, and snapshots both
the original source run's full code hash map and current study source map. This
compatibility has been verified for the original run source copy before A0 freeze;
all other original training-source hashes must equal current source hashes.
