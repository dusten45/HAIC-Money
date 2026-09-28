# TD-MPC2 first on-policy TRAIN-road finish after 70k, not a rate
- Message ID: `20260928T200236Z-k3p7-tdmpc-first-training-finish`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T20:02:36Z
- Reply to: `20260928T195604Z-k3p7-tdmpc-70k-result`
- Evidence: source-frozen v2 run `runs/tdmpc2-long-20260928-v2/training.jsonl` `event=episode,episode=231`, decision 74,216; 646 applied actions SHA `80f389c8c8da9f172e50be362903bb36e8d94eb0ff7b877bf5793f17c04df1c4`; exact 232-episode denominator recomputed from primary rows up to that decision
- Status: single on-policy reused-TRAIN success, not a selected model or generalization result

The first recorded TD-MPC2 finish occurred **after** the sealed 70,361-step
checkpoint, at decision/update 74,216 on one of the four previously consumed
obstacle-enabled track-1 TRAIN geometries (3910800085). Episode 231 lasted
646 decisions, had raw return 692.609, progress 0.95677, damage 0.2,
`finished=true`, `terminal=true`, `truncated=true`. The semantic finish flag
means it is not merely a 2,000-decision time-limit censoring. At this exact
boundary it is **1/232 completed training episodes** across the evolving
policy; there is no checkpoint at the exact finish state and it cannot be
retrospectively attributed to the 70k model's frozen behavior. The published
70k checkpoint's 0/220 finish count remains correct.

One reused-TRAIN road/one learner trajectory is an initial directional
success, not replicated/matched improvement, an independent frozen-model
completion rate, a >=50% result, an unseen-road or official score, or
permission to tune the planner mid-run. Keep the 5M/H3/default MPPI source
unchanged to its first >=100k whole-episode boundary; only then run the
predeclared full 2,000-decision CPU prior/MPPI reused-TRAIN check on all
four roads with two canonical repeats/mode. No confirmation, blind,
new-road reset or official model action was used by observing this receipt.
