# Local score improvement: stable road-clearance speed gate

## Fact

The new `stable_risk_envelope` keeps the selected full-road-guard actor as its default. It raises gas to 0.20 only when pixels show contiguous asphalt clearance around the car, all seven road rows are visible, no bright obstacle is detected, the inherited steering/brake are small, the bend is not rapidly growing, and rendered speed is below a bend-conditioned target. Submitted inference would still use only the four grayscale frames; this study did not build a submission ZIP.

| Split, tracks 1–3 | Frozen candidate finished | Selected control finished | Median finished lap, candidate | Median finished lap, control |
|---|---:|---:|---:|---:|
| Fresh TUNE seeds 324–327 | **12/12** | 12/12 | **23.22 s** | 24.70 s |
| Independent HELD_OUT seeds 328–331 | **11/12** | 9/12 | **22.36 s** | 24.98 s |
| One-time CONFIRMATION seeds 332–335 | **12/12** | 11/12 | **22.71 s** | 24.26 s |

The candidate was faster in all 12 paired tune cells, all 8 jointly completed held-out cells, and all 11 jointly completed confirmation cells. It produced no invalid actions. Its action latency p95 was 21.57 ms on tune, 20.44 ms on held-out, and 16.55 ms on confirmation; the respective maximums were 56.90, 53.68, and 36.74 ms. Candidate collision/damage totals were 0/0, 4/0.8, and 0/0 across these splits. The only held-out candidate nonfinish was track 3 seed 330 at reported progress 1.0 without a valid finish crossing. The full traces and per-episode metrics are in the matching reports.

The four-direction screen followed the research contract: road-clearance speed, bend-exit burst, centered-obstacle preview brake, and mild-bend outer entry each activated on consumed TRAIN cells. On fresh tune, the exit-burst arm finished 10/12 and was rejected; the obstacle-preview and outer-entry arms finished 12/12 but had slower median laps than the selected control. The road-clearance gate alone advanced to held-out and one-time confirmation with its source frozen.

All three evaluated stages have v2 `run_manifest.json`, append-only `events.jsonl`, and `integration_report.json` in `runs/haic-research-v2/`, plus full `report.json` in `artifacts/haic-research-v2/`. The frozen executable copy is `tmp/haic2-stable-risk-frozen-20260928/`; the reusable runtime is `haic_agent/stable_risk_envelope_runtime.py` and the registered diagnostic profile is `stable_risk_envelope`.

## Interpretation and limits

This is a measured **local controller-level** improvement in valid completion and lap time, supported by independent held-out and one-time confirmation. It is not an official score, ZIP-exact validation, or permission to upload. The official rule-compliance gate remains `UNKNOWN`, so no v2 release or SOTA promotion was recorded. The confirmed cells are consumed and cannot be reused for tuning. The next local phase is packaging the frozen runtime, then checking that exact ZIP on a separate registered set; do not modify this confirmed candidate from the confirmation failures.

## Later ZIP-exact check

The unchanged package was evaluated on fresh confirmation seeds 336–339. It lost one valid finish to the control (11/12 versus 12/12), so the later frozen-ZIP decision is **REJECT**. See [ZIP-exact outcome](../haic2-stable-risk-zip-eval-20260928/outcome.md). The earlier controller-level measurements remain historical evidence for their registered sets.
