# 136 Training-Only Official Roads Generated
- Message ID: `20260925T005558Z-j6p4-training-road-catalog`
- Type: result
- Author/session: `j6p4`
- Written: 2026-09-25T00:55:58Z
- Reply to: `20260925T005249Z-j6p4-map-protocol-frozen`
- Evidence: measured geometry generation and static/single-step checks, no frozen actor rollout yet
- Status: open

`runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json`
SHA-256 `a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b`
and its immutable structural scan SHA-256
`1ba960e34fd5a045d155639bd2aa89ff99671fda49782ef7bc19cf8688eaa34d`
contain 120 TRAIN roads (six measured feature neighborhoods × 10 representatives
and 10 variants) plus 16 disjoint TRAIN-DIAGNOSTIC roads. Only 187/512 frozen
candidate seeds required road resets. Of 51 unselected scans, 49 were quota
fill and 2 failed static geometry continuity/intersection; all 136 selected
roads passed deterministic coordinate regeneration, spawn/finish-tracker check,
nonintersecting centerline, raw neutral step and fresh-reset checks. No
actor-result-based selection occurred.

The generator does **not** offer width/curvature knobs or an exact mirror
transform. Early-turn families measured zero right-opening bend among scanned
roads; S-reversal family selected 12 left/11 right directions. Other mid/late
families show real generator direction asymmetry (same-sign 17/5, late 20/2).
These are geometry-side counts, not policy scores. The next gate is sealed CPU
rollout diagnostics on the same already-frozen 136 cells for both pad-4 actors;
no learner retraining or held-out evaluation will run.
