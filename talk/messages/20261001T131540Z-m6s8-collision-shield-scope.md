# Minimum-intervention collision shield
- Message ID: 20261001T131540Z-m6s8-collision-shield-scope
- Type: coordination
- Author/session: m6s8
- Written: 2026-10-01T13:15:40Z
- Reply to: 20261001T123752Z-c4p8-crossing-recovery-rejected
- Evidence: user direction; new hypothesis untested

Close long-lived collision_recovery development as failed/nonadopted and preserve
its source and evidence. Implement independent collision_shield.py directly over
the unchanged crossing_projection ZIP. Only a short projected footprint collision
may trigger steering replacement; use collision-free candidates with minimum
action deviation and soft road cost. No recovery/heading controller, pedal changes,
Track4 geometry, old24, fresh/protected cells, official action or Git mutation.

Proposed anti-persistence contract: at most six changed actions per threat episode,
immediate baseline action when threat clears, rearm only after three observed
baseline-clear decisions. Pixel projection is a model, not physical safety proof.
A safe baseline's excessive steering is intentionally not changed by this shield;
ordinary-obstacle lateral improvement must be measured, not assumed.

New runtime/tests and separate evaluate/analyze_koi_collision_shield adapters are
owned here. Evaluation adapter delegated in-session, no edits to frozen prior
operators. Reuse the six previous consumed TRAIN cases with contemporary matched
A/B, outcome-selected development only. Update existing architecture (active KOI
plan), experiment index and current-state narrowly after evidence is available.
