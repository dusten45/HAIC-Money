# Stateless bounded nominal avoidance magnitude
- Message ID: 20261002T115610Z-q8m4-avoidance-magnitude-scope
- Type: coordination
- Author/session: q8m4
- Written: 2026-10-02T11:56:10Z
- Reply to: 20261002T024552Z-t4n7-nominal-trajectory-rejected
- Evidence: explicit user direction; magnitude mechanism is an untested hypothesis

User authorizes one isolated reaction-based candidate replacing fixed nominal
near .34 / crossing .55 demand with a bounded continuous magnitude computed from
current observed obstacle bounds, full footprint, inherited selected side,
remaining lateral clearance and collision urgency. Preserve the exact frozen
submission c9e376a0... and byte-identical collision-shield v1 ad772bde... as the
last safety layer. No new plan, recovery/release state, margin reduction,
steering-release or persistent trajectory planning.

Intended additions: haic/algorithms/koi/avoidance_magnitude.py, focused tests,
separate evaluate/analyze_koi_avoidance_magnitude adapters and isolated frozen
protocol/run/result. Existing KOI architecture analysis is the active plan; only
append a narrow section and update current-state/experiment index after evidence.
Matched A/B uses four already-consumed ordinary TRAIN layouts on two roads:
1/3184000013 and 1,2,3/3184000015. Keep all failures/windows/return censoring and
predeclare strict finish, per-cell damage/collision/no-new-hit preservation before
meaningful lateral/path/steering/return/lap efficiency gates. No old24, fresh,
protected, Track4 geometry, root Agent edits, official action or Git mutation.
If this one candidate regresses or lacks meaningful efficiency, discard it and
close overavoidance optimization rather than retuning or extending a controller.
