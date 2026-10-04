# Separate collision-first candidate
- Message ID: 20261001T120420Z-c4p8-collision-priority-scope
- Type: coordination
- Author/session: c4p8
- Written: 2026-10-01T12:04:20Z
- Reply to: none
- Evidence: user direction; mechanism not yet verified

Preserve steering-release v2, both ZIPs, original runtime/dependencies and root
Agent. User reports official Track 1/2/3 finishes and Track 4 failure; no external
evaluation or Track 4 geometry is requested here. Investigate archived TRAIN/dev
corner/obstacle steering conflict, then implement a separate collision-priority
wrapper with temporary soft road boundaries and immediate recovery. Intended new
files: haic/algorithms/koi/collision_priority.py, focused tests and separate
diagnosis/evaluation scripts and run artifacts. Reuse completed nonprotected TRAIN
cells only, explicitly development, not fresh confirmation. No old24 reset planned.

User explicitly requests fast focused work and prohibits excessive checks. Limit
validation to exact off-track rule, unchanged originals, targeted tests and matched
requested metrics; no repeated Git inspection or new broad audit framework.
