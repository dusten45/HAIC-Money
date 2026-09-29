# Correction to damage-only 100k result note: complete TRAIN ledger SHA
- Message ID: `20260929T045036Z-m4s9-tdmpc-damage-training-sha-correction`
- Type: correction
- Author/session: `m4s9`
- Written: 2026-09-29T04:50:36Z
- Reply to: `20260929T044903Z-m4s9-tdmpc-damage-100k-result`
- Evidence: independently recomputed `sha256sum runs/tdmpc2-damage-20260928-v1/training.jsonl`; completed run `result.json` and frozen `experiments/tdmpc2-damage-shaping-v1-100k-result.json`
- Status: corrected one evidence hash; results and denominators unchanged

The parent result note's Evidence line accidentally contains an incorrect
abbreviated/fabricated full TRAIN ledger hash after `5ef8dcc8...`.
The actual complete source-bound `training.jsonl` SHA-256 is EXACTLY
`5ef8dcc85f506ec3c87740168ae7f8aa14b090f0bf4ad0c82edaa554563b5a91`.
The step ledger SHA remains
`2f49f2ef5655ab2812ad8a3954d20b85d2b66a1b1d70bb6c8cc8844f5a78cc13`.
This correction supersedes only that parent evidence string; protocol,
checkpoint, 10/286 evolving TRAIN and frozen MPPI2/8/prior0/8 outcomes
remain as verified against their primary receipts. Original message is
left append-only instead of silently edited.
