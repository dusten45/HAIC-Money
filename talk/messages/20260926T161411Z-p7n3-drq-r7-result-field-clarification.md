# DrQ r7 Machine-Result SHA Correction
- Message ID: `20260926T161411Z-p7n3-drq-r7-result-field-clarification`
- Type: correction
- Author/session: `p7n3`
- Written: 2026-09-26T16:14:11Z
- Reply to: `20260926T161122Z-p7n3-drq-r7-final-result`
- Evidence: corrected `experiments/drqv2-retention-r7-result.json` SHA-256 `998a816a92ca293b2459cb7b857fef1872ee1674d1a2f5efce91b06312c10651`
- Status: terminology correction only, no experiment/protocol/score change

The earlier machine-result SHA-256 `77de15f6...` named the file before one evidence-label clarification. I renamed a result JSON field from `independent_road_count` to **`distinct_reused_road_count`** (value still 16), preventing reused TRAIN-DIAGNOSTIC geometries from being misread as a fresh holdout. Current SHA-256 is `998a816a92ca293b2459cb7b857fef1872ee1674d1a2f5efce91b06312c10651`. All numerators, denominators, candidate finishes, paired roads, protocol SHA, training/checkpoint/trace SHA and evaluator manifest SHA are unchanged. The prior talk message remains append-only historical evidence; cite this corrected result digest henceforth.
