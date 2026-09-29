# Pixel speed student TRAIN activation

The stable pixel control validly finished 11/12 new TRAIN cells (tracks 1–3 × seeds 5012–5015). Shadow geometry suggestions were queried but never executed. Successful control trajectories supplied 1,567 eligible training examples with 1,394 positive gas labels; within-TRAIN validation supplied 531 examples and the student exceeded its fixed 0.8 gas threshold on 434. Training and validation losses were finite (0.136 and 0.149). The checkpoint SHA-256 is `1c139842763d9b3feb2a94380fa6b08b2e8612988bf7ef02ea31ee7a5aab51f6`.

The registered TRAIN activation gate passed. The v2 state is `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED` without release. This is no closed-loop student evidence: 89% of training labels were positive, so the model may act like broad acceleration. Freeze this checkpoint and compare it with the same stable control on newly registered TUNE cells, completion first. All 5012–5015 cells remain consumed TRAIN identities.

Evidence: `artifacts/haic-research-v2/haic2-safe-speed-distill-train-20260929/report.json` and `runs/haic-research-v2/haic2-safe-speed-distill-train-20260929/integration_report.json`. Frozen source: `tmp/haic2-safe-speed-distill-frozen-20260929/`.
