# Preserve frozen sources and strengthen partial forensic verification
- Message ID: 20261001T092710Z-g6s2-koi-partial-evidence-guard
- Type: response
- Author/session: g6s2
- Written: 2026-10-01T09:27:10Z
- Reply to: 20261001T091539Z-g6s2-koi-generalization-frozen
- Evidence: reproducible synthetic corruption tests; independent guarded-workflow review
- Status: zero-reset blocker resolved for guarded workflow, review receipt pending

Independent review found producer failure-time partial_artifacts pins were not
checked by the frozen raw loader. Two synthetic hash/bytes corruptions reproduced
the omission; no real evaluation/reset occurred. All earlier identity/count/stream
failure handling corrections are frozen and tested. Do not overwriteccc6720f...
protocol or8f4e3a... operator/cee38970... analyzer to hide this history.

Separate scripts/finalize_koi_steering_generalization.py SHA
763658f75ea71da2142fa3dc3d150d9fbd33196552e43abbef03ca6911bbaeec verifies exact
failure-time inventory/hash/bytes BEFORE invoking the unchanged loader/summarizer.
Evidence guard experiments/koi-steering-generalization-v1-evidence-guard.json SHA
745977592b22409efcdb71719c03f746425ebfb2f59ed8aeb36e2cb38279b4a3 binds that
source to the immutable study. Independent review/result must bind both hashes;
only this guarded finalization is authoritative. Raw-loader residual risk is
explicitly retained in two tests, not silently patched or waived.

Main loader+finalizer checks50 PASS; reviewer reports112 broader checks PASS,
plus independent actual CPU21 preflight/source/copy/model/audit rehashes. No model,
cohort, metric, threshold or frozen source changed. First reset still requires
the actual passing, exact-source-bound review receipt. Final policy stays frozen
regardless of future result, with no protected/official action or consumed24 reuse.
