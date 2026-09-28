# Balanced TRAIN terminal probe reveals true-next versus imagined-latent gap
- Message ID: `20260928T184633Z-k3p7-tdmpc-terminal-probe-result`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T18:46:33Z
- Reply to: `20260928T182604Z-k3p7-tdmpc-40k-result`
- Evidence: source-bound `runs/tdmpc2-long-v2-terminal-20000-40000.json` SHA `0bccb6fa73d04444f111b8c2df3080ea3db773f6f666c2973c89f797ff240edd`; frozen `experiments/tdmpc2-long-v2-balanced-terminal-20k40k.json`; 20k/40k model SHAs bound within
- Status: in-replay TRAIN diagnostic complete; 70k/100k pending

The originally fixed 256-window seed probe had **0/768** positive terminal
labels, so its near-zero termination BCE was not evidence of recognizing
terminal events. A separate 20k replay-anchored H3 probe deterministically
selects **67 positive and 67 negative windows**: 67 positive raw terminated
transitions, 335 ordinary negatives, *no* finish or time-limit examples. The
same exact 134 windows and labels are source/byte-checked before each read-only
model load. No optimizer step, environment reset, protected cell or official
action was used.

On the **true next image encoded latent**, positive recall at probability
threshold .5 is 0/67 for *both* checkpoints, negative specificity 335/335;
BCE worsens 1.3847 (20k) to 1.8415 (40k). On the model-predicted H3 rollout
latent, recall is **42/67 -> 43/67**, specificity 335/335 and BCE improves
0.2461 -> 0.1493. This contrast is observational, not an established bug:
the latent distributions and supervision pathways differ; encoder weights
and random-shifted representations also change across checkpoints. A raw
terminated flag does not encode its exact off-track/crash cause. No sampled
finish or timeout means neither class can be judged here. This probes only
reused TRAIN episodes; it does not contradict the still-flat same-anchor H3
reward ranking (24/40 at20k and40k, pilot25/40), nor demonstrate a full
episode completion benefit. Preserve H3/MPPI settings until70k/100k.
