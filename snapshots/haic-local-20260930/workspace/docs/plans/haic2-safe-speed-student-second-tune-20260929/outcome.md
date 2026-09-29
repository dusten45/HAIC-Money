# Pixel speed student passed the second TUNE gate

On new TUNE tracks 1–3 × seeds 5204–5207, the fixed pixel student validly finished 10/12 versus stable-risk control 8/12. Both failed 2:5204; student alone failed 2:5205; control alone failed 1:5205, 1:5207 and 3:5206. On seven jointly completed cells, median finished lap was 21.70 versus 23.30 seconds, **6.87% faster**. Student gas intervened 1,902 times and invalid actions were zero. The comparison used the exact TRAIN checkpoint, converted to weights-only tensor storage before runtime loading, with no training or policy edit.

This meets the preregistered aggregate-completion and paired-lap gates. V2 recorded `ADVANCE → GATE_REVIEW_ADVANCE → STOPPED` without release because exact submission compliance remains unknown. The 5204–5207 cells are consumed TUNE. This is a local controller diagnostic, not an official score or SOTA promotion. Freeze source and checkpoint for a separately registered held-out comparison; treat the student-only failure as a real risk to inspect.

Evidence: `artifacts/haic-research-v2/haic2-safe-speed-student-second-tune-20260929/report.json` and `runs/haic-research-v2/haic2-safe-speed-student-second-tune-20260929/integration_report.json`. Frozen source: `tmp/haic2-safe-speed-student-second-tune-frozen-20260929/`.
