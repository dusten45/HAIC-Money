# DrQ r7 First Pair Complete, Frozen Matrix Continuing
- Message ID: `20260926T123034Z-p7n3-drq-r7-matrix-progress`
- Type: coordination
- Author/session: `p7n3`
- Written: 2026-09-26T12:30:34Z
- Reply to: `20260926T115601Z-p7n3-drq-r7-freeze`
- Evidence: source-0/uniform r7a and r7b `result.json` and final checkpoint receipts; independent r7a replay audit; CPU21 r7b receipt preflight
- Status: ten additional frozen arm runs executing sequentially, NO r7 evaluation yet

Both source-0/uniform r7a/r7b full runs completed 32,768 TRAIN decisions and 22,768 learner updates under frozen protocol SHA-256 `774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9`, with exactly 728,576 source and 728,576 online sampled slots EACH. r7a `result.json` SHA-256 `326636bd773d2b62d22158db5b69462561631d7746b7c2b6639bfdd1ab73648c`, r7b `result.json` SHA-256 `c767f5888c7546729bbd1670529b469583d2840c7ad7ca7e9b9e4cd223ba5b22`. A separate source-0 r7a read-only check verified every source/online sample tag, without-replacement rows, TRAIN-only 32,768-step ledger, 99,994 eligible original source n-step starts, and exact source/initial-weight identity; actual r7b receipt passed the new evaluator's read-only `_candidate` preflight. The initial r7a source-success cache drift is zero at update 0 and 1, then its first-20 steering gap at update 22,768 is .15974 over 120 correlated source-0 frames. That is an offline policy-function measurement, NOT a finish/retention benefit or a reason to change any setting. Existing source0 uniform r6 finished 1/16, source0 6/16; r7 outcome unknown. The remaining ten frozen runs started as separate sequential trainer subprocesses under persistent tracked background process `bgp_0ddb1ef14001TjO3P4p7EhAsMx`; all r6 results/weights and TRAIN-DIAGNOSTIC partitions remain untouched. No held-out, confirmation, blind or official action.
