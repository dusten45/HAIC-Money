# DrQ Speed Pilot Cohort And Source Boundary
- Message ID: `20260929T104612Z-s8d2-drq-speed-cohort`
- Type: coordination
- Author/session: `s8d2`
- Written: 2026-09-29T10:46:12Z
- Reply to: `20260929T102901Z-s8d2-drq-speed-scope`
- Evidence: hypothesis; immutable seed1 actor SHA `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954`; prior r6 TRAIN-DIAGNOSTIC ledger and residual-options pilot iteration-1 result; source-specific audit pending
- Status: source under synthetic test; NO road reset yet

The proposed new, distinct speed hypothesis only attenuates light braking (factor 0.25) with steering magnitude <=0.20, gas >=0.80 and brake <=0.30 after the unchanged pad-4 seed1 actor has mapped to official actions. Steering, gas, high brake, checkpoint and existing submission ZIP are unchanged. Root `agent.py` and the frozen native evaluator cannot inject this treatment, so I am preparing isolated `haic/algorithms/drq_v2/speed_control.py` and `scripts/evaluate_drq_speed_train.py`, plus tests and a NEW frozen protocol; no shared evaluator or environment edits.

Proposed matched cohort: the pre-existing 16 *track-1 TRAIN-DIAGNOSTIC* seeds from r6 plus tracks 2 and 3 x residual-options development seeds 4000000001..4000000008. These are MIXED, repeatedly consumed **development**, not new TRAIN, fresh confirmation, blind, or official evidence. Per-track finish retention is mandatory, and speed is measured only on cells that BOTH fixed policies finish. Earlier DRQ retention work forbade continuing to tune its closed retention line on track1 diagnostics; this is a user-directed *different* speed question, and the prior outcomes are explicitly disclosed. A three-seed x three-track all-TRAIN alternative has exact prior reset evidence but too few known potential jointly completed laps to establish speed on all three tracks. The new 32-cell cohort is intentionally outcome-exposed and cannot justify a generalization or promotion claim. A separate candidate-specific cross-lane/protected-cell audit and zero-reset preflight must pass before the first reset; please flag any overlap/conflict now. No official upload/confirmation action is planned.
