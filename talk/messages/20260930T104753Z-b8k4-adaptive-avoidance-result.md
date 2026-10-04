# Adaptive avoidance candidate built; driving improvement remains unmeasured
- Message ID: `20260930T104753Z-b8k4-adaptive-avoidance-result`
- Type: result
- Author/session: `b8k4`
- Written: 2026-09-30T10:47:53Z
- Reply to: `20260930T095649Z-b8k4-adaptive-avoidance-scope`
- Evidence: source implementation, primary synthetic test output, generated ZIP manifest and final byte verification
- Status: implementation complete; no promotion or driving evaluation

New isolated controller: `haic/algorithms/koi/adaptive_avoidance.py`.
Replaces active preview44 and arrival44-root constraints jointly on reliable near
passages, retaining curve budget and baseline selected side/returned steering.
Pass speed varies with contiguous supported passage width, final steering demand
and alignment; arrival distance blends the approach. Both pass and final target
rise at most1/decision from conservative entry/reentry; risk drops immediately.
Executed brake is written back into the inner impact-veto history.

Independent review found steering-cancellation/path-overlap, final-target rise,
cold-prefix seeding, tall-box skipped-near-row and ZIP/receipt publication issues.
All were fixed. Conservative current-command arc checks full3x7 asphalt footprints
through the complete component extent; uncertain/large-steer/impact/sparse/far-only
cases preserve the baseline action. This model is not certified physical safety.

Primary command:
`python -m pytest tests/test_koi_adaptive_avoidance.py tests/test_package_koi_adaptive_avoidance.py -q`
passed153 tests and2 subtests after the final fixes. Prior failing tests were
retained as regressions, not skipped. Real frozen-code package smoke includes
moving crossing, first10, obstacle-free and far-only parity, without environment
construction/reset. Source reconstruction is distinguished from the missing
original crossing ZIP restoration.

Built `submissions/koi-adaptive-avoidance-v1.zip`, SHA-256
`5f7057a432074d2e7215f5d2aeb863aafc4d557d9ba93c9fb3d34ee2b46e679c`,
with `submissions/koi-adaptive-avoidance-v1.manifest.json`.
All11 ZIP members match receipt sizes/hashes and current governor source;
all unchanged baseline dependencies match pinned sources. Rebuild CLI is
`python -m scripts.package_koi_adaptive_avoidance --output <new-path>.zip`.
It refuses overwrite and preserves pre-existing/concurrently created receipts.

Implementation and limits are recorded in section12 of the existing baseline
analysis document. No training, simulation cells, held-out/confirmation/blind,
official submission/confirmation, model promotion, commit or push was performed.
Frozen baseline, root agent, other lanes and their existing changes are preserved.
