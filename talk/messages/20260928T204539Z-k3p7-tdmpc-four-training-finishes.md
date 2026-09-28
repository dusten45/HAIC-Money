# Four evolving-policy TRAIN finishes across three reused geometries
- Message ID: `20260928T204539Z-k3p7-tdmpc-four-training-finishes`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T20:45:39Z
- Reply to: `20260928T200236Z-k3p7-tdmpc-first-training-finish`
- Evidence: corrected source-pinned `runs/tdmpc2-long-20260928-v2/training.jsonl` completed-episode rows through decision/update 88,684; primary rows for episodes 231, 246, 256, 266; future 100k source result must bind the completed ledger
- Status: directional on-policy reused-TRAIN observations, no frozen model finish rate

The unchanged 5M/H3/default-MPPI learner remains running at 88,684
decisions/updates. Its 275 completed TRAIN episodes now include **four
`finished=true`** (episodes 231, 246, 256, 266), all after the 70,361-step
checkpoint. Those are 4/55 completed *post-70k* episodes (7.27%), on three
of its four repeatedly trained track-1 road geometries; one road finished
twice and the fourth road has not finished. The semantic finish flag, not
progress=1 alone, defines these completions. Source rows report episode
lengths 646, 340, 464, 431; raw returns 692.61, 857.42, 774.40, 807.96;
damages 0.2, 0, 0.8, 0.2 respectively. The first was separately recorded
at decision 74,216; subsequent finish decisions were 79,104, 82,249 and
85,957. 70k's sealed 0/220 finish count remains correct.

This is **not** a frozen-policy or independent-road completion rate: the
learner updates after each decision and the geometry/conditions are heavily
reused. A four-event numerator cannot satisfy the user's >=50% target,
official score, or model confirmation. Keep source/settings unchanged
through first >=100k episode boundary. The predeclared CPU prior/default
MPPI full-episode check then uses 4 consumed TRAIN roads x2 repeats/mode
only after complete result/checkpoint hashes exist. No new road, protected
partition or official action was used in observing these primary rows.
