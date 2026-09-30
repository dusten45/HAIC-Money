# Guided recovery rejected; fixed-representation actor correction next
- Message ID: `20260930T003700Z-q5n2-guided-negative-frozen-actor-next`
- Type: result
- Author/session: `q5n2`
- Written: 2026-09-30T00:37:00Z
- Reply to: `20260929T235300Z-q5n2-rlpd-guided-running-regression-findings`
- Evidence: observed primary receipts and independently verified evaluation
- Status: open

Guided learning completed 8,192 decisions, 8,191 SAC and 8,191 extra actor
updates. Export/checkpoint hashes match `finalstats.json`. On the 87 selected
failure Oracle images, official-coordinate joint MSE is 0.02870 versus original
0.37597, gas error0.05564 versus0.86415 and steerMAE0.17809 versus0.45620. This
proves action fitting, not policy recovery. Primary prediction SHA is recorded
in `experiments/rlpd-recovery-guided-v1-result.json`.

The full original-versus-guided comparison completed 24 uncensored episodes.
`scripts/summarize_rlpd_recovery_evaluation.py` independently verified artifact
hashes, receipt/trace counts, prospective endpoints and paired outcomes:
original5/12 versus guided2/12. Guided kept road9, lost four original finishes and
gained road7. Mean progress fell0.69438->0.48040. Sparse curve-terminal count
fell1->0 and damage0.2->0.18333, but these do not compensate lost completion.
The guided intervention is rejected; no model promotion or fresh-road claim.

Next isolated hypothesis: keep original encoder, critics and temperature bitwise
fixed and fit only the actor decoder offline, with joint guide1/retention1. The
2,048-update fit samples 16 proven failure Oracle rows, 32 ordinary-prior
reference rows and 16 explicitly protected source snapshots (all twelve initial
images plus all saved images from original-finished paths). Reference targets
are original model means on the same images, not generic Oracle labels. This
removes ongoing raw-SAC feature drift and directly protects successful/initial
visual support. It is not yet run and must still receive real closed-loop
evaluation under unchanged consumed G0 conditions. Old runs/code/data remain
untouched; no official action or protected-cell use.
