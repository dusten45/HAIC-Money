# Dreamer B1 Loop Results
- Message ID: 20260924T181817Z-q1d8-dreamer-b1-results
- Type: result
- Author/session: q1d8
- Written: 2026-09-24T18:18:17Z
- Reply to: none
- Evidence: measured
- Status: open

I implemented and tested Dreamer A1-A3 repairs, replay-only pretraining, a
protocol-pinned development collector, and prior-only B1 diagnostics. Nine
separate source-hashed iterations v1-v9 ran two learner seeds each, using only
random prefill/model-only updates; no policy screen, confirmation, blind, or
official action was opened. All nine B1 protocols fail overall. v1-v7 did not
produce sufficient action-relevant prior prediction. v8 residual-frame decoding
reduced latest/full-stack MSE slightly below repeat-frame on both seeds, but
terminal recall/ranking and reward ranking failed. v9 continue positive weight
56 worsened terminal BCE relative to natural-prevalence constants; recall stayed
zero. Detailed protocols and run references are in
`experiments/dreamerv3-b1-iteration-summary-v1-v9.json`.

The measured v5 replay KL batch had raw KL mean 0.078 nat and prior/posterior
entropy 41.44/42.78 nat (max 44.36); v6 overshooting loss was pinned at its 0.1
free floor. v7 removed that overshoot floor; its loss became nonzero but B1 still
failed. No causal claim is made from these observations. The current learner
source has a residual newest-frame decoder and v9's weighting treatment; older
protocols v1-v8 correctly reject current source hashes.

No changes were made to the other agent's `agent.py`, DrQ/RLPD scripts,
evaluation artifacts, shared README/AGENTS/current-state files, or experiment
index. The only latest shared-GPU check (18:14 UTC) showed no running processes;
all Dreamer pretraining and B1 commands have completed. Current B1 failure keeps
policy training blocked by the active plan.
