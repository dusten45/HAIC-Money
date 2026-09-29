# Anticipatory bend steering: first fresh tune comparison

## Question and causal path

`hypothesis_id`: `anticipatory-bend-neutral-steer-20260928`

The pixel observation can show a consistent far-road bend before the actor starts turning. When all seven road rows are visible, the near road is within three pixels of image center, the far road shifts at least five pixels consistently, no obstacle is active, and the actor's steering magnitude is below 0.05, a bounded steering cue toward the far road should start the turn earlier without replacing the learned policy. The candidate leaves gas and brake untouched. The endpoint is a valid finish-line crossing with at least the control's completion rate; only at equal completion do lap times matter.

| Required field | Registered value |
|---|---|
| `source_ref` | Submission #6 track 3 replay recorded in `docs/sources/INDEX.md`; four consumed tune diagnostics under `runs/haic-research-v2/anticipatory-bend-center-guard-20260928/` |
| `rule_or_requirement` | Complete the registered tracks reliably, including a qualified finish-line crossing. |
| `observable_information` | Pixel-derived near, middle, and far road centers; the base actor action; pixel obstacle diagnostic. No map coordinates or simulator state enter inference. |
| `allowed_action_or_state_change` | Change only steering, capped at magnitude 0.12, when the preview conditions above hold. |
| `expected_success_endpoint` | At least the control's finished-cell count over the same 12 cells; lower median finished lap time if completion ties. |
| `eligible_state` | Fresh tune diagnostic; not a held-out, confirmation, release, or official submission result. |
| `control` | `artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip`, SHA-256 `5C55670AB5AEF4BF64115909792650A5E76BC3576A64F3AAEB3780EFABC18E40`. |
| `falsifier` | Any candidate-only nonfinish, invalid action, or lower completion rate; absent preview activation also fails the mechanism gate. |
| `smallest_decisive_experiment` | Matched control/candidate on tracks 1, 2, and 3 × seeds 300, 301, 302, and 303, alternating arm order, 12 cells per arm. These cells were not found in the current v2 run manifests or plan documents at registration. |
| `resource_and_risk_gate` | Linux Python 3.11 CPU, 2,000 decisions per episode, 4.5 s action budget, no official site action. Cap the diagnostic at 1,800 wall-clock seconds. Freeze source hashes before execution and record each cell in immutable v2 manifest/events/report artifacts. |

The candidate source is `haic_agent/anticipatory_bend_runtime.py`, SHA-256 `27459733EDF6D845B61296F68FA82EB17AF9672E2A6C88337FF8AB6D9A2F91D2` at design drafting. Recheck this hash immediately before running. The four consumed cells had control 4/4 and candidate 4/4, with median finished lap 25.61 versus 25.46 s; they established behavior and cannot serve as fresh performance evidence.

Record completion, lap times, incomplete progress, collisions, damage, invalid actions, preview activations, action latency, and runtime/source hashes. Do not promote or package based on this tune alone. A later held-out and unchanged-package confirmation require new registered cells and evidence.
