# DrQ Training-Only Geometry Study Handoff
- Message ID: `20260925T012753Z-j6p4-geometry-final-handoff`
- Type: result
- Author/session: `j6p4`
- Written: 2026-09-25T01:27:53Z
- Reply to: `20260925T011940Z-j6p4-road-diagnostic-result`
- Evidence: frozen catalogs/protocols/shards and final verification
- Status: closed

The requested training-only study is complete without learner retraining:
120 TRAIN seeds plus 16 disjoint TRAIN-DIAGNOSTIC seeds, all official-generator
roads (seed-only control, fixed width), six evidence-backed measured families,
two frozen source-actor diagnostic attempts per road, four sealed CPU21 shards,
and no blind road/outcome reset or held-out actor reevaluation. The per-road
joined artifact `experiments/drqv2-geometry-augmentation-v1-final-set.json`
SHA-256 `338a845a47235927a4e7e0704d022e2cf46d5c5c805e4fe64e82c780aa2c0f30`
lists every road's generator seed, family, hypothesis, inclusion reason, nearest
consumed nonblind signature distance, difficulty and both actor traces. The
logical finish-tracker proof for all 136 is pinned by SHA-256
`c5f152c14309634238e98c064c15d34641801fd45ad1e952423937de9dbba4cf`;
for 81 roads no frozen actor finish was observed, so physical finishability
remains unresolved, not malformed.

Zero selected roads failed static/physics/finish-logic checks. The two preselection
rejected seeds had broken loop seams, not model-quality failures. Diagnostic
summaries: 10 both-actor single-attempt finishes, 74 useful/boundary, 50
progressing but no finish, 2 unresolved, 0 pathological. The source0 opening
cluster recurred on eight new collision-free roads near the first measured bend.
The fixed family-weight training sampler is implemented and one-step TRAIN-only
smoke/restart-next-seed parity passed, but the legacy DrQ trainer has **not** been
replaced or resumed; any long learner run needs its own frozen protocol and
episode-boundary sampler checkpoint integration. Full A-J evidence, limits and
future teacher-replay freshness proposal:
`docs/experiments/drqv2-geometry-augmentation-v1.md`.

Validation: 80 new geometry tests passed (+28 subtests); unchanged DrQ/trainer
49 tests and inference/evaluator 44 tests passed. Frozen catalog/diagnostic
source hashes match current files. No result or provenance was overwritten.
