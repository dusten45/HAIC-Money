# DrQ r7 Longitudinal Offline Coverage Result
- Message ID: `20260927T030706Z-v4d8-drq-offline-coverage-result`
- Type: result
- Author/session: `v4d8`
- Written: 2026-09-27T03:07:06Z
- Reply to: `20260927T030116Z-v4d8-drq-offline-audit-batch-failure`
- Evidence: `runs/20260927-drqv2-final-source-replay-v1/offline-coverage/result.json`, SHA-256 `e1746c05a1e11af31444a99505067e400e53734c535bfda8a8b778963e34b042`; executable `scripts/audit_drq_final_source_coverage.py`, SHA-256 `f66655521c5226f718ea292cebc5878744c0e53385521ebfa9c493eca6ac335f`
- Status: offline audit passed; no training, no candidate evaluation or official score

Archived official-action replay exactly matched all 11 source-success trajectories (5,998 decisions), all 666 previously hashed float32 pre-action observations, and all 220 cached first-20 frames. Every seed's 99,994 valid original evolving-source replay starts was scanned, and r7b sample trace confirms 99,915/99,994 unique starts actually drawn per mixture for source seed 0 and 99,917/99,994 for seed 1 (each arm 728,576 source slots). The source-vs-stored action native L-infinity mean across 99,994 replay starts was 0.843 for seed 0 and 0.691 for seed 1; only 26,132 and 18,117 respective starts had <=0.10 difference. These values are from the PRIMARY receipt above, not an earlier talk summary.

Representative phase counts are 18 frames per phase for seed 0's six successful actor-road cells and 15 for seed 1's five, apart from overlaps among phases. Source-encoder feature-neighbor count with cosine >=0.95 across ALL original replay starts is 4,660 for the 33 early phase representatives, 10 for middle, 4 for major corners, 36 for late, and 56 for finish-approach (sum of receipt `per_seed_phase_denominators.*.*.feature_neighbor_count_ge_0_95_sum`, each phase denominator 33*99,994 comparisons). Recovery is a hazard/heading proxy, not proven recovery. No source-success diagnostic road is identical to a TRAIN replay road. The original 5,998 float32 pixels were not all archived; 187 reconstructed representative stacks were checked against quantized replay stacks without claiming full original byte identity. Feature proximity/action agreement and action drift are descriptive evidence, not a causal explanation for r7 loss or official performance.

The earlier batch-vs-singleton actor check failure was corrected BEFORE this replay; all archived telemetry and prior hashes now pass. Next is only the owner-requested SHA-bound final-frozen-source TRAIN replay matched comparison, fixed 32:32/lambda=.5 and >=9/11 retained plus >=2/21 gained gate. No r7c or parameter sweep follows.
