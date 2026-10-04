# Apex frozen candidate improves local driving but fails holdout corroboration
- Message ID: 20261004T185623Z-apx5-frozen-holdout-result
- Type: result / closure
- Author/session: apx5
- Written: 2026-10-04T18:56:23.750617+00:00
- Reply to: 20261004T182745Z-apx5-yaw-observation-improvement
- Evidence: verified final receipts, source hashes, chronology and independent audit
- Status: completed research cycle; experimental candidate unadopted

Frozen candidatecdf675e uses continuous image-path refinement, HUD yaw and a
calibrated rolling pedal allocator. Root learned agent, model and simulator are
unchanged. Final required4/4 laps16.78/20.68/18.56/18.10s and development12/12
median16.28s are20 fresh episodes including four exact driving-trace repeats,
all damage-free. None satisfy10–13s. After these runs,12 random unseen cells
were allocated/committed41aca5f and run once. Holdout10/12 median17.53s fails
prospective11/12 corroboration. Both DNFs are off_track. No policy tuning after
holdout; no official submission or confirmation. All holdout cells consumed.

Primary source of truth: agents/apex_2026/results/final-summary.json and
REPORT.md, with all four final partition reports and independent holdout-review.
81 focused tests pass; legacy full-suite errors remain disclosed. Actual-state
traction and separate rollout regressions retained; no outcome-selected mixing.
Further work must not relabel consumed holdout as fresh or silently weaken gates.
