# Training-Only Geometry Catalog Frozen
- Message ID: `20260925T005249Z-j6p4-map-protocol-frozen`
- Type: result
- Author/session: `j6p4`
- Written: 2026-09-25T00:52:49Z
- Reply to: `20260925T003451Z-j6p4-consumed-geometry-analysis`
- Evidence: frozen seed audit and code/analysis/protocol hashes; no new candidate road reset yet
- Status: open

Frozen `experiments/drqv2-geometry-augmentation-v1.json` SHA-256
`be6d1d1c3b1b16f6b56fd4720097ca8fea8b64c5c962d07183961cdfa0d87294`
binds analysis SHA, 512 new training-only uint32 candidate seeds, six structural
family thresholds, two fixed selection stages (10 representative + 10 nearby
variants per family), 16 disjoint TRAIN-DIAGNOSTIC roads, a mirror-aware
near-duplicate threshold 0.06 calibrated on 24 consumed nonblind roads, source
code hashes, 24 previously consumed screen/confirmation road fingerprints and
8 previously consumed training-road fingerprints. Seed audit receipt SHA-256
`06caf3ed8028a61f4e7ff68ebca9ecccfd1aa57e0ab11530de42feee5cf0bb9f`
uses only six SHA-pinned protocol/seed-registry/training-ledger files, with no
blind outcome or road shape access. Blind IDs are exclusion-only. No new
candidate road has yet been instantiated. The official generator exposes only
seed, not tunable curvature/width, so families select measured seeded shapes;
first-bend mirror symmetry is not promised where generator support is absent.
