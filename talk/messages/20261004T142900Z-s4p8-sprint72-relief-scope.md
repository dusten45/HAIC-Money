# One fixed sprint72 handback relief candidate
- Message ID: 20261004T142900Z-s4p8-sprint72-relief-scope
- Type: coordination
- Author/session: s4p8
- Written: 2026-10-04T14:29:00Z
- Reply to: none
- Evidence: user-authorized implementation and small consumed TRAIN A/B

Create research/koi-sprint72-relief without disturbing existing staged/unstaged
work. Preserve submitted crossing_projection + collision-shield v1 ZIP and all
source members, root Agent, prior candidates and frozen evidence.

Implement only the diagnosed pre-arrival branch: v>=72, preview T==60, unchanged
sprint space condition S, no near/far/unresolved track, no impact-clear/recovery/
geometry-repair intervention, parent pedals exactly gas0/brake.15 and its steering.
Replace pedals with gas0 and clip(.02*(v-72),0,.15). Record actual issued brake in
brake_history; no fabricated .15 history. No cap/target/arrival/steering/shield/
floating-point boundary change, timer, deadband or hysteresis.

Intended isolated files: haic/algorithms/koi/sprint72_relief.py, dedicated operator
and analysis scripts/tests, runs/koi-sprint72-relief-v1 and its experiment result.
Existing KOI analysis/current-state/index receive narrow result updates. Freeze a
small matched consumed TRAIN cohort including the diagnosed reverse-travel case;
do not exclude the four reverse decisions or use hidden state in the policy.
Measure full-episode safety/finish, cap undershoot/brake-gas oscillation, trajectories
through the next real restriction, arrival timing and downstream steering/impact.
Failure closes this exact specification, with no retune or expanded experiment.
No fresh/protected/official/generalization evaluation, submission or confirmation.
