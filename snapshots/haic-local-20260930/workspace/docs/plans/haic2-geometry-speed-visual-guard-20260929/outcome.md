# Visual obstacle guard teacher: pivot

The new TRAIN 5008–5011 block finished 11/12 for each arm. The visual obstacle lockout activated on 1,881 teacher decisions, but candidate-only cell 1:5008 retired off track at 95.65% progress with no collisions; the control finished in 20.70 seconds. The control alone crashed on 3:5010. On the ten jointly finished cells, candidate median lap was 23.18 versus control 23.74 seconds, a 2.36% gain below the registered 3% gate. Invalid actions were zero. The v2 result is `PIVOT → GATE_REVIEW_PIVOT → STOPPED`, no release.

This follows two earlier twelve-cell teacher comparisons that also failed their registered completion gates: fixed geometry coast on 5000–5003 and 5004–5007. The three blocks used different gates or teacher revisions, so they are not pooled into a replicated candidate score. The pattern supports abandoning this geometry-speed teacher route for now. TRAIN 5000–5011 are consumed and remain diagnostic only. Next work should target valid finish and obstacle/finish recovery with a pixel-only mechanism, then compare lap time when completion ties.

Evidence: `artifacts/haic-research-v2/haic2-geometry-speed-visual-guard-train-20260929/report.json` and `runs/haic-research-v2/haic2-geometry-speed-visual-guard-train-20260929/integration_report.json`. Frozen source: `tmp/haic2-geometry-speed-visual-guard-frozen-20260929/`.
