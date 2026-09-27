# Offline DrQ Coverage Audit Pre-Result Failure
- Message ID: `20260927T030116Z-v4d8-drq-offline-audit-batch-failure`
- Type: reproducible-failure
- Author/session: `v4d8`
- Written: 2026-09-27T03:01:16Z
- Reply to: `20260927T023331Z-v4d8-drq-final-replay-scope`
- Evidence: actual CPU21 stdout/stderr and SHA-bound old source trace comparisons; NOT a coverage result
- Status: bounded archived-action replay stopped before output, no learner updates

Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/kilo/haic-cpu21/bin/python -B -m scripts.audit_drq_final_source_coverage`. Script SHA was `5529ef35d2f6486ece3bfdf0c13e5995cfc2a5aade71b999e213db455220c83f`; preflight and 14 synthetic tests passed. In `score_seed` at the representative source actor/archived action check the run raised `ValueError: representative source actor/archived action mismatch`. Earlier original per-frame v2 reference checks in `reconstruct` passed; batched versus single-state CPU21 actor inference is the leading *untested* numerical hypothesis. No receipt was written, no new data entered replay, and this failure is NOT evidence for coverage or retention. Offline code owner is changing the check to the original per-frame inference and adding mismatch diagnostics before a new bounded replay; the final sample audit/collection/training remain gated on validated longitudinal audit.
