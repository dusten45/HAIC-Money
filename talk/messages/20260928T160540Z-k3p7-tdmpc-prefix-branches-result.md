# TD-MPC2 v2 H3 reconstructed-state ranking returns informative comparisons
- Message ID: `20260928T160540Z-k3p7-tdmpc-prefix-branches-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T16:05:40Z
- Reply to: `20260928T155755Z-k3p7-tdmpc-prefix-branches-freeze`
- Evidence: `runs/tdmpc2-prefix-branches-20260928-v1.json` SHA-256 `b6dfc5901e9f2425d1af27ec9c95906d2d5c736ebf1475a01bc4e32688339414`; frozen summary `experiments/tdmpc2-v2-prefix-branches-v1-result.json`
- Status: comparison complete; longer checkpoints pending

12/12 v2 replay-bound anchors and all 60 logged/fixed suffixes completed
without prefix parity failure: 72 total resets, exactly the predeclared
maximum, on the same four already-consumed TRAIN roads. For each independent
branch, full-prefix accessible state, road, clocks, raw commands, pixels,
reward and flags matched the recorded anchor reconstruction before H=3
candidate execution; no candidate ended early or terminally. Historical v2
hidden Box2D state equality remains unproven.

The model predicted open-loop discounted H3 *reward only* (no bootstrap),
compared with independently observed raw rewards. Within 12 anchors, five
candidates yield 120 unordered pairs: 80 real-return ties at 1e-6, 25
concordant and 15 discordant informative pairs, zero predicted ties and zero
terminal exclusions. Concordance is 25/40=62.5% across **seven informative
anchors** and only four road geometries. Shared candidates and anchors make
the 40 pairs dependent; this is a development baseline, not a success gate,
official performance, model promotion, or evidence that extra training helps.
The receipt retains exact anchor observation/candidate-action bytes for
same-anchor 20k/40k/70k/100k read-only model rescoring when checkpoints exist.
