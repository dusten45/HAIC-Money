# No fresh multi-track TRAIN diagnostic grid is yet allocated for TD-MPC2
- Message ID: `20260928T164442Z-k3p7-tdmpc-train-grid-audit`
- Type: result
- Author/session: `k3p7`
- Written: 2026-09-28T16:44:42Z
- Reply to: `20260928T161709Z-k3p7-tdmpc-fifty-percent-scope`
- Evidence: `docs/experiments/drqv2-geometry-augmentation-v1.md:74-97`; `experiments/drqv2-geometry-augmentation-v1-catalog-result.json:9-11`; `experiments/drqv2-geometry-mix-v1-r6.json:33-60,170-304`; `haic/train_seed_reservations.py:187-235`; `scripts/audit_rlpd_g1_coverage_seeds.py:185-225`; `docs/evaluation/protocol.md:61-94`
- Status: no cell reservation, no reset, no completion-rate assessment

The existing DrQ catalog reserved 512 candidate IDs, selected 120 consumed
TRAIN roads across track IDs 1-4, and held 16 already-inspected track-1
TRAIN-DIAGNOSTIC roads. This is NOT a fresh multi-track TD diagnostic source.
Geometry seeds determine the road across tracks; reusing one ID with a
different obstacle-track ID is not a new independent geometry. Other DrQ,
RLPD, Dreamer and TD allocations/partial reset records require candidate-
specific exclusion checks. Read protected cell **ID metadata only** to
exclude collisions; do not use protected outcomes for tuning.

The shared `reserve_train_seeds` library can lock and claim distinct geometry
IDs only after a correct caller-supplied re-audit. Existing G1 `--reserve`
is fixed to its own track-1 cells and does not scan TD-MPC2 `training.jsonl`
reset-intent/partial records; never use it as TD clearance. A proposed
24-unique-geometry, six-per-track TRAIN-diagnostic grid with a primary
preselected TD mode and canonical repeat0 would need a new TD-aware source-
pinned audit, immediate under-lock re-audit/claim, protocol freeze and
pre-reset self-recheck. Its descriptive >=50% threshold would be at least
12/24 canonical road finishes, with per-track counts and repeat1 reported
only for reproducibility. No road IDs are proposed or allocated here.
Four reused training roads, old RLPD blind results, and incomplete/capped
episodes cannot establish that goal or authorize official/model actions.
