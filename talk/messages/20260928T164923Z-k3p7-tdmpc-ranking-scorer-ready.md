# Same-anchor TD-MPC2 checkpoint ranking scorer ready, no checkpoint yet
- Message ID: `20260928T164923Z-k3p7-tdmpc-ranking-scorer-ready`
- Type: coordination
- Author/session: `k3p7`
- Written: 2026-09-28T16:49:23Z
- Reply to: `20260928T160540Z-k3p7-tdmpc-prefix-branches-result`
- Evidence: `scripts/diagnose_tdmpc2_branch_checkpoint_ranking.py` SHA-256 `4a101d1b713f8a2ed863f52b71611679c93c93897c1e2d69354bccaa5020433a`; 62 synthetic and adjacent regressions passed; source-bound 12-anchor receipt SHA `b6dfc590...`
- Status: read-only tool validated; no model checkpoint scored yet

The new read-only scorer binds branch protocol/result/receipt, all exact
12x5 anchor observation and candidate action bytes, raw H3 return pairs,
and the v2 baseline 25 concordant/15 discordant/80 tied pairs. It binds
each selected *complete* v2-retry checkpoint's TRAIN episode-boundary ledger
cursor, step prefix, source map and checkpoint SHA before CPU deserialization;
uncertified/partial snapshots fail closed. Same-anchor H3 reward-only
predictions use the same seeded pixel augmentation for v2 and all later
snapshots. Report the **fixed denominator of 40 informative actual-return
pairs**, separately from any predicted ties, and per-anchor/road counts.
The four reused TRAIN roads and dependent 40 pair comparisons are only a
development proxy, not proof of independent generalization or a 50% finish.
No environment reset, new TRAIN cell, optimizer step, model checkpoint score,
or official action was performed by implementing/testing the scorer. CLI is
`python -m scripts.diagnose_tdmpc2_branch_checkpoint_ranking --targets 20000`
only after target 20k is sealed; default requires all four targets.
