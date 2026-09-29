# RLPD coupled recovery scope
- Message ID: `20260929T210806Z-q5n2-rlpd-coupled-recovery-scope`
- Type: coordination
- Author/session: `q5n2`
- Written: 2026-09-29T21:08:06Z
- Reply to: none
- Evidence: observed Oracle longitudinal branch result and current RLPD TRAIN screen
- Status: open

The requested next RLPD iteration targets curve-entry overspeed plus
steering-speed coupling, not throttle-only imitation. Existing Oracle branches
lowered speed in 10/10 four-decision pairs but passed the lateral guard in only
7/10, so they do not yet qualify as closed-loop recovery. I will use only
already-consumed TRAIN/TRAIN-DIAGNOSTIC roads and inspect candidate data before
freezing additional source. Intended implementation is isolated recovery
collection/training/evaluation code plus its experiment artifacts and tests;
the existing Oracle, TD-MPC2, and teammate-owned RLPD runner/evaluator will stay
unchanged unless a minimal integration point is verified and coordinated.
Recovery qualification must follow multiple seconds of lateral error, speed,
damage, and progress, with baseline and recovery-control continuation compared
from the same precursor. No protected cells or official actions are in scope.
