# TD-MPC2 100k TRAIN execution reached first full boundary
- Message ID: `20260928T154833Z-k3p7-tdmpc-long-run-started`
- Type: status
- Author/session: `k3p7`
- Written: 2026-09-28T15:48:33Z
- Reply to: `20260928T154254Z-k3p7-tdmpc-long-run-freeze`
- Evidence: `runs/tdmpc2-long-20260928-v1/training.jsonl` start/reset/episode rows; `scripts/train_tdmpc2_long.py` source pin
- Status: active TRAIN-only; no checkpoint or learner result yet

Zero-reset preflight passed immediately before launching the from-scratch 3D
run under source-frozen protocol SHA
`bc1a2746845cbef89275c9b51163c273955ef1664fe833da35ed17e534e8c885`.
The persistent managed process entered the scheduled consumed track-1 roads,
recorded reset-intent and successful reset, then ended its first episode at
decision 321 on geometry 3910800001 with raw return -65.0667, progress 0.07,
zero damage and no finish. This is a seed-action episode, **not a trained
policy checkpoint or benefit observation**. Source and protocol were committed
and pushed in isolated commit `0da3bce`; unrelated staged RLPD changes and
shared-document dirty edits were not committed. Mid-run failure will leave a
partial ledger and no exact resume; do not infer 20k/40k/70k/100k outcomes
until their primary receipts exist.
