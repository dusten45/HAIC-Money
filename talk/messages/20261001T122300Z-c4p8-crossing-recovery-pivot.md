# Crossing baseline and heading-first recovery
- Message ID: 20261001T122300Z-c4p8-crossing-recovery-pivot
- Type: result
- Author/session: c4p8
- Written: 2026-10-01T12:23:00Z
- Reply to: 20261001T120420Z-c4p8-collision-priority-scope
- Evidence: measured initial A/B plus new user direction

User explicitly restores crossing_projection as baseline and rejects extending
near_release. New collision_recovery.py will wrap the unchanged crossing runtime
directly, with explicit heading alignment and safe edge re-entry before handback.
No Track4 geometry, protected cells or official action.

The completed initial v2-based priority test is NOT ADOPTED: six matched reused
TRAIN cells, v2 4/6 finishes versus candidate3/6, one lost finish (3/3184000002),
damage1.4->1.2 but longest all-wheels-offroad4->402 physics ticks. This reinforces
the recovery concern, not a reason to promote collision-only priority. Original
v2/source/ZIP and this run's frozen source remain unchanged. Primary result:
experiments/koi-collision-priority-v1-result.json; runs/koi-collision-priority-v1/.

Exact public rule verified: retirement on101 consecutive decisions with negative
summed raw reward, not physical wheel departure; act(observation) cannot see that
counter. New runtime timers are explicitly proxies; observer records actual count.
