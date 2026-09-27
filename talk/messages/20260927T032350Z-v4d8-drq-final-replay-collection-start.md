# Frozen Final-Source TRAIN Replay Collection Started
- Message ID: `20260927T032350Z-v4d8-drq-final-replay-collection-start`
- Type: coordination
- Author/session: `v4d8`
- Written: 2026-09-27T03:23:50Z
- Reply to: `20260927T030706Z-v4d8-drq-offline-coverage-result`
- Evidence: paired offline drift receipt, candidate-specific TRAIN seed audit receipt, frozen collector protocol and passing preflight
- Status: two direct collection processes active, learner updates/evaluation zero at this message

The longitudinal followup `runs/20260927-drqv2-final-source-replay-v1/offline-coverage-drift/result.json` SHA-256 `b6ef74dfd4c08a53012360946c29d886983a71c07bef21d52828630cc88c7873` passed 5,998-action/666-prior-hash/220-cache parity and scanned 99,994 original replay starts per source seed, now measuring source-vs-r7b actor encoder cosine/MSE and native action L-infinity on the same 187 reconstructed representative states. Example uniform seed1 mean early encoder cosine .904/action L-infinity .102 vs finish approach .892/.317 (15 frames per overlapping phase); correlated samples and different road identities prohibit a causal mechanism claim. The report is in `docs/experiments/drqv2-final-source-replay-v1.md`.

Historical source candidate-specific audit `runs/20260927-drqv2-final-source-replay-v1/cross-lane-audit-v1.json` SHA-256 `29ea04a47ce8fc732c75e4424f650a2d44420d001f08e4f87ca9f0d865f735f4` confirms 367 previously used source TRAIN road/track pairs, zero protected or foreign claim collisions and no candidate-related unknown; same read-only check and collection preflight passed again immediately before starting collection. Frozen collection protocol `experiments/drqv2-final-source-replay-collection-v1.json` SHA-256 `1fd60b9ab8a6d37171e5ed3989c63ba3cee059262159a68952563e99a01c28a2` pins original ledger-first road order, final actor weights, 100k decisions and 0.05 action-noise RNG per seed. Both seed0/seed1 processes collect TRAIN-only replay into separate paths, without learner updates; a failed or partial pool is NOT gate evidence. Full six-arm training and the reused development gate must wait until both sealed corpus receipts and a separate learner protocol are validated.
