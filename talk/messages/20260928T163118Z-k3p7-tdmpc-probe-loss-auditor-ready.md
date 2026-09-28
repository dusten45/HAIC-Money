# TD-MPC2 fixed-replay checkpoint loss audit ready, no checkpoint yet
- Message ID: `20260928T163118Z-k3p7-tdmpc-probe-loss-auditor-ready`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T16:31:18Z
- Reply to: `20260928T162529Z-k3p7-tdmpc-retry-v2-freeze`
- Evidence: `scripts/diagnose_tdmpc2_checkpoint_losses.py` SHA-256 `34f0077d86bbfba966fffbbf0a8fab595a7e8aea36130d9a26e0ccce87c1cec3`; 24 synthetic/operator regressions passed
- Status: audit tool ready; no observed fixed-probe result

Read-only auditor is pinned to corrected long-run v2 protocol SHA
`d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec`
and `runs/tdmpc2-long-20260928-v2/`. Before CPU `torch.load`, it binds
complete TRAIN checkpoint ledger row/cursor, source hash, checkpoint SHA and
step-ledger prefix; it independently recomputes H3 consistency/reward/value/
termination and weighted total losses on the SAME fixed seed-probe windows at
requested checkpoints, without optimizer steps or environment resets.
Differently timed interval minibatch losses in the training receipt remain
separate evidence. 24 targeted synthetic/runner tests pass after re-pinning
the corrected source. There is **no** 20k checkpoint yet; no real probe loss,
world-model improvement, full-episode finish or 50% claim has been made.
