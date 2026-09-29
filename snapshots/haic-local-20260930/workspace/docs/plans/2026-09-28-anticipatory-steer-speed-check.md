# Anticipatory steering with steering-aware speed: local check

## Hypothesis and boundary

`hypothesis_id`: `anticipatory-steer-speed-envelope-20260928`. The fixed pixel-only anticipatory steering ZIP can begin a visible turn before the base actor reacts. A small gas increase only when the full road is visible, the car is near the visible road center, there is no active obstacle or brake, steering demand is small, and pixel-estimated speed is below a steering-and-curvature-dependent target may improve lap time without a material road departure. The actor, obstacle response and early steering are unchanged. The speed addition uses only the same image observation and action available at inference.

| Field | Registered value |
| --- | --- |
| `source_ref` | Submission #6 track 3 replay; `anticipatory-bend-heldout-20260928`; consumed-cell package diagnostic `anticipatory-bend-package-confirmation-20260928`; current Participants README |
| `rule_or_requirement` | Return a finite `[steer, gas, brake]` action; finish reliably without off-track retirement. |
| `observable_information` | Image-derived road centers and speed bar, current steering/gas/brake, pixel obstacle diagnostic. |
| `allowed_action_or_state_change` | Raise gas to at most 0.16 only under the registered visual and action gates; steering and brake untouched. |
| `expected_success_endpoint` | Completion at least tied with the fixed anticipatory ZIP on matched cells, no candidate-only off-track retirement, and lower median finished lap on a completion tie. |
| `eligible_state` | New local tune, followed only if successful by separate held-out and unchanged-package confirmation. No official inference. |
| `control` | Fixed anticipatory ZIP SHA-256 `CE28AB744956BC4898D102C41FF6184EE4AC5A819A8559A656A6AEDDA4008F66`. |
| `falsifier` | Any candidate-only nonfinish or invalid action; greater completion loss; or material road departure increase as defined below. |
| `smallest_decisive_experiment` | Tracks 1–3 × seeds 1700–1703, 12 paired tune cells per arm. |
| `resource_and_risk_gate` | Linux Python 3.11 CPU, 2 CPUs, 2 GiB, 2,000 decisions per episode, 1,800 s wall cap, no site action. |

Register tune cells `1..3 × 1700..1703`, held-out `1..3 × 1704..1707`, and confirmation `1..3 × 1708..1711`. A v2 manifest and plan search found none of these identities before registration. Do not open held-out or confirmation for tuning. Another concurrent project run using any of these identities before execution invalidates their split status; audit timestamps before and after each stage. The existing 300–311 cells are consumed and may be used only to diagnose mechanism behavior, never as fresh evidence.

The speed candidate will use all seven road centers, near center within 2.5 pixels of image center, absolute far-near shift at most 7 pixels, absolute steering at most 0.10, no active obstacle, no brake above 0.01, and image-estimated speed below `46 - 1.1*abs(far-near) - 30*abs(steer) - 2` speed units. Then gas becomes `max(base_gas, 0.16)`, without capping an already larger actor command. This is one mechanism candidate, no threshold sweep. Count actual gas increases and their overlap with early steering.

Evaluation-only instrumentation may read simulator wheel-road contacts and negative-reward off-track counter; these values never enter the submitted agent. Report per cell: completion and lap time, progress, off-track retirement, decisions with all wheels off road, longest all-wheel-off-road streak, any-wheel-off-road fraction, maximum negative-reward streak, collisions, damage, mean/max speed, invalid actions, and p50/p95/max action latency. Treat all-wheel-off-road streak longer than control by more than 10 decisions or any-wheel-off-road fraction more than 2 percentage points higher as a material road-departure increase. Compare completion first, then the repository's fixed tie-breakers; a speed benefit cannot override a completion loss or material road-departure increase. Preserve source, package and evaluator hashes and every cell in an immutable v2 run. A successful tune is a basis to open held-out, not an official score.
