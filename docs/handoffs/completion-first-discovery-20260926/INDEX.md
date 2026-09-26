# Completion first discovery batch — 2026-09-26

Phase completed: DISCOVER / HYPOTHESIZE. This handoff records hypotheses only; it is not an approved implementation or run plan. No `run_manifest.json`, plan hash, split, maps, seeds, denominator or execution approvals have been registered yet.

## Central assignment and validation

The coordinator grouped four candidate hypotheses into four distinct action/state-change directions. The configured batch rule is satisfied: **4 directions, 4 candidates, at most 1 per direction** (limits: ≥4 directions, ≤8 candidates, ≤2 per direction). The in-memory `haic_research.coordinator.assign_work` validation returned 4 assignments. No training or simulation ran.

| Assignment | Direction | Owner | Handoff |
|---|---|---|---|
| 1 | Pixel obstacle-to-car lateral feature | `discover_perception` | [HZ-PERCEPT-01](1-0afbd31b55ea.md) |
| 2 | Temporal pixel obstacle approach-rate cue | `discover_perception` | [HZ-PERCEPT-02](2-ee0cf2563e29.md) |
| 3 | PPO throttle-range calibration | `discover_control` | [PPO-PEDAL-RANGE-COMPLETION-01](3-4f55118bea37.md) |
| 4 | CEM model-uncertainty fallback | central coordinator | [CEM-UNCERTAINTY-FALLBACK-01](4-b4e137054bc3.md) |

Source inspection established that the first three can target the learned PPO candidate. The CEM path is potentially candidate-eligible when packaged with the same trained policy, but that eligibility and runtime budget require confirmation in design. The two corridor-control ideas returned by `discover_control` remain **diagnostic-only / eligibility unknown**, so they are excluded from this candidate batch.

## Integration

- Completion rate is the primary endpoint. Lap time and other metrics apply only as registered tie-breakers when completion rates tie.
- Every hypothesis has an identical-matched-control requirement and an activation/falsifier check. An inactive mechanism cannot justify threshold tuning.
- Map/seed/split/episode-count identities and the control package/checkpoint must be registered before comparison. Reserved confirmation/blind data are not available for tuning.
- Source facts are separated from inferences. None of these hypotheses has measured effect evidence; no gain is claimed.
- Next coordinator step: use the [Battlecode-style tournament design draft](battlecode-style-tournament-draft.md) to reconcile overlap and resource costs, select a concrete four-direction design, define a valid registered control and evaluation set, then request the separate design approval. No implementation or execution approval is implied by this report.

## Agent reports

The two independent agent reports are preserved in their handoffs. The central coordinator authored the fourth source-grounded candidate hypothesis. All assignments were read-only and no shared source code or experiment records were edited by agents.
