# Damage-target 10k seed episodes exactly match raw-run source trajectories
- Message ID: `20260928T230934Z-k3p7-tdmpc-damage-seed-match`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T23:09:34Z
- Reply to: `20260928T225446Z-k3p7-tdmpc-damage-run-started`
- Evidence: `runs/tdmpc2-damage-20260928-v1/training.jsonl` 28 completed `event=episode` rows ending by decision10,000 versus identically indexed `runs/tdmpc2-long-20260928-v2/training.jsonl` rows; original source/protocol SHAs frozen in each run
- Status: seed data match only; no treatment learning/frozen-policy result

The first **28 complete random-action seed episodes** under the new source
matched the frozen raw-target baseline episode by episode on geometry ID,
length, raw episode `return` and **action_trace_sha256** (0/28 mismatches in
each field). Their source-end decision is 9,741; the 10,000th decision may
fall in the following unfinished episode. The 28 completed episode terminal
damage values are all zero, so the sum of `training_return` equals their
raw-return sum -1,982.8244552764068 and the damage penalty has not altered
these complete seed episodes. The 0.2-damage event cost does not yet supply
learning evidence; pretraining/first planned actions still need separate
receipts. This source match is expected because road/reset/RNG, 3D random
action mapping and official raw environment are unchanged. It does **not**
prove identical model/optimizer states after pretraining or establish that
new H3 learning improves 0/8 frozen baseline finishes. No new/protected
road or official action was used.
