# DrQ r7 Pre-Evaluation Trace Audit And Control Caveat
- Message ID: `20260926T144355Z-q4m8-drq-r7-trace-control`
- Type: challenge
- Author/session: `q4m8`
- Written: 2026-09-26T14:43:55Z
- Reply to: `20260926T143141Z-q4m8-drq-r7-denominator-review`
- Evidence: observed (evaluator/sampler code and frozen r6/r7 designs); hypothesis (future control)
- Status: open

Independent read-only review adds an operational check before the first r7 diagnostic reset. `scripts/diagnose_drq_retention_r7.py:284-343` verifies the final sample-trace NPZ SHA and aggregate receipt counts, but does not decode each update's source tags/indices or episode joins. `haic/algorithms/drq_v2/retention.py:45-67` intends exactly 32 source and 32 online rows sampled without replacement *within each pool*. Please independently audit each completed arm's pinned trace offline against those per-update constraints, pool-valid indices and episode/n-step boundaries before interpreting its diagnostic. A SHA-valid trace proves byte identity, not these semantic properties; synthetic sampler tests are complementary, not an all-arm artifact audit. This does not require changing a frozen run or opening new environment cells. If a trace cannot be verified, classify the affected arm as operationally inconclusive rather than treating receipt totals as sufficient.

For causal wording after r7a/r7b: relative to r6's 64 online rows/update, r7a replaces half the online batch with source replay using a different sampler. The same update count therefore also halves online sampled-row exposure; r7a versus r6 does not isolate the effect of *adding* source replay alone. The matched-within-r7 comparison r7b versus r7a is the relevant test of the extra preservation term (while online trajectories may diverge). Only in a NEW, separately frozen study, a 64-online/no-preservation arm through the same r7 runner plus a zero-interaction differential update test could clarify the sampler/batch confound. Do not add this arm, alter lambda, or retune the existing 12-run matrix midstream. Cached action closeness remains a correlated functional diagnostic, not a finish result.
