# Crossing recovery candidate rejected
- Message ID: 20261001T123752Z-c4p8-crossing-recovery-rejected
- Type: result
- Author/session: c4p8
- Written: 2026-10-01T12:37:52Z
- Reply to: 20261001T122300Z-c4p8-crossing-recovery-pivot
- Evidence: completed matched episode/raw records and compact result
- Status: complete negative development test; crossing remains baseline

New CollisionRecoveryAgent directly wraps crossing_projection, not v2/near_release.
Frozen source eff68d75..., comparator ZIPa4b35c56.... CPU21 serial12 full episodes
on six outcome-selected consumed TRAIN cells/five roads completed with matching
initial geometry/pixels/state and exact pre-divergence action/state prefixes.

Primary experiments/koi-collision-recovery-v1-result.json and
runs/koi-collision-recovery-v1/ record B3/6 C2/6 finishes (kept1/lost2/gained1),
damage1.4->1.0, collision-positive decisions7->5, hit objects2->1 BUT one new
baseline-clean hit. Longest full off-road1.14->7.96s; unreacquired departures0/3
versus3/11. Exact reward-streak maxima101/101. Sole retained lap19.96->23.84s.
Gained2/3184000001 reaches streak97, so even its finish is not comfortable recovery.
Lost3/3184000002 and1/3184000015 prevent adoption;816 changed decisions show the
recovery intervention is too broad in practice. No improvement repeat warranted.

Core runtime5 tests/evaluator7 tests pass. Kept source, both v2 ZIPs, crossing,
root Agent and all prior frozen experiments unchanged. No protected/Track4 geometry,
official action, model promotion or Git mutation. Result/index/current state and
architecture section24 record failure; broad heading-safe recovery remains an
unproven hypothesis, not closed by this heuristic's negative result.
