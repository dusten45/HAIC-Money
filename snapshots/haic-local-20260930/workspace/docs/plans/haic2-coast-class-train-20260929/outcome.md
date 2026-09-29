# TRAIN-only three-class shadow distillation: ADVANCE to TRAIN driving check

The unchanged stable pixel driver completed all 12 TRAIN cells, tracks 1–3 × seeds 5016–5019. Only successful control trajectories entered the model: 1,669 training and 560 within-TRAIN validation examples. Hold/boost/coast labels were 44/1,518/107 for training and 17/512/31 for validation. At the fixed 0.65 acceptance threshold, the pixel model predicted 18 coasts in validation, 11 correct: precision 0.611 and recall 0.355. Training/validation losses were finite at 0.303/0.642, and a tensor-only checkpoint round trip succeeded. Checkpoint SHA-256: `f1afc4016f00ac699e8212f0007dde27f829831294f84b9fd3236a6cd0e3ab86`. Teacher suggestions were never executed and geometry is absent from runtime inputs.

All preregistered label and model activation gates passed. Formal outcome `ADVANCE`, gates rule compliance `UNKNOWN`, mechanism activation `PASS`, competitive outcome `NOT_APPLICABLE`, no release. These 5016–5019 identities are consumed TRAIN cells. The student has not driven yet; a separately registered TRAIN closed-loop check is required before TUNE.

Frozen source and checkpoint: `tmp/haic2-coast-class-train-frozen-20260929/`. The source SHA-256 entries in the run manifest refer to that snapshot.
