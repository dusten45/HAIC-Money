# Pixel speed student passed held-out controller gate

On untouched held-out tracks 1–3 × seeds 5208–5211, the frozen pixel-only speed student and stable-risk control each validly finished **12/12**. Student median finished lap was **22.20 seconds** versus **22.92 seconds**, **3.14% faster**, with 1,620 gas interventions and zero invalid actions. The trusted TRAIN checkpoint was converted to tensor-only storage before the unchanged weights-only runtime loaded it; the runtime-safe checkpoint SHA-256 was `3c3cafc0d549c3777a6489ede397a4bd4f2390f8f2b1597ae6e0960dd4dc2194`.

The result meets the preregistered completion and 2% paired-median gates. V2 recorded `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED`, no release because exact package rule compliance is still unknown. Material risk remains: student was slower on track 2 seeds 5209–5211, and student alone had two collisions with 0.4 damage on 3:5211 while still finishing. These held-out cells are consumed and cannot be retuned or reused as independent confirmation. Next step is exact ZIP packaging and fresh confirmation, without changing the checkpoint or speed rule.

Evidence: `artifacts/haic-research-v2/haic2-safe-speed-student-heldout-20260929/report.json` and `runs/haic-research-v2/haic2-safe-speed-student-heldout-20260929/integration_report.json`. Frozen source: `tmp/haic2-safe-speed-student-heldout-frozen-20260929/`.
