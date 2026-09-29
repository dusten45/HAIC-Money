# Early steering and speed coupling: local safety check

The fixed early steering candidate remained the comparison ZIP throughout. The user asked whether anticipating a turn prevents large road departures and whether speed can be raised based on steering. These runs use only pixels and current actions at inference; evaluator-only wheel-road contacts and negative-reward streaks measure road exposure and never enter the controller. This is local Linux Python 3.11 CPU evidence, not an official score or proof about submission #6's video.

The first speed candidate set gas to at least 0.16 when visible road geometry, steering and estimated speed appeared safe. On fresh tune tracks 1–3 × seeds 1700–1703 it raised gas 533 times, finished **11/12** versus fixed preview **12/12**, and alone retired off-track at track 2 seed 1700. Its any-wheel-off fraction was **14.10% versus 11.91%**, with all-wheels-off decisions **263 versus 177**. It had zero gas increases on the same decisions as preview activations. Reject this broad gate. Full records: `runs/haic-research-v2/anticipatory-steer-speed-tune-20260928/`.

An evaluation-only observation on consumed tune cells 2/1700 and 3/1701 found 19 preview steering events, with median image-estimated speed 42.3 and 6 events at or below 34. That diagnostic did not tune on held-out or confirmation cells. The first stable-road ZIP had a wrong module import and stopped after one control episode; `anticipatory-steer-speed-safe-tune-20260928` is infrastructure-invalid. The package was corrected in a new immutable ZIP and a new plan/tune split.

The corrected stable-road candidate raised gas to at least 0.13 after five aligned low-steer decisions. On tracks 1–3 × tune seeds 1724–1727 it finished **12/12** versus preview **10/12**, with median completed lap **23.28 versus 26.53 s**, and reduced aggregate all-wheels-off decisions **91 versus 198**. Yet track 2 seed 1724 had an any-wheel-off fraction **23.71% versus 21.57%**, exceeding the preregistered +2 percentage point per-cell safety tolerance by 0.15 percentage points. The original report's `REJECT` stands; this result does not justify opening its reserved held-out cells. Full records: `runs/haic-research-v2/anticipatory-steer-speed-safe-r2-tune-20260928/`.

The final gentler candidate raised gas to at least 0.12 only below image-estimated speed 41 after seven consecutive fully visible, aligned, nearly straight low-steer decisions. Its package SHA-256 is `07A50E0321F699496C926C6ABEC2FD0FCC0A91F06FBCE153A79842F653371AEA`. On fresh tune tracks 1–3 × seeds 1800–1803, both arms finished **11/12**, the candidate median was **26.94 versus 27.16 s**, and no cell exceeded the preregistered exposure limit. The unchanged ZIP then ran on held-out tracks 1–3 × seeds 1804–1807:

| Held-out metric | Early steering | Early steering + gentle speed |
| --- | ---: | ---: |
| Completed | 12/12 | 12/12 |
| Median finished lap | 25.96 s | 26.04 s |
| Mean simulator speed | 40.70 | 40.99 |
| Any-wheel-off fraction | 10.01% | 10.41% |
| All-wheels-off decisions | 90 | 94 |
| Longest all-wheels-off streak | 47 | 47 |
| Collisions / damage / invalid actions | 0 / 0 / 0 | 0 / 0 / 0 |
| Gas increases | 0 | 265 |

No candidate-only nonfinish or preregistered material road-exposure increase occurred on held-out. The small mean speed gain did not lower the finished-lap median, so the held-out outcome is `REVISE`; the reserved confirmation 1808–1811 must remain unopened for this candidate. Full records: `runs/haic-research-v2/anticipatory-steer-speed-gentle-tune-20260928/` and `runs/haic-research-v2/anticipatory-steer-speed-gentle-heldout-20260928/`.

Practical conclusion: early steering did activate on all studied splits. Raising gas broadly near low steering increased speed but produced a candidate-only off-track failure. A small increase after sustained stable road visibility did not cause a large measured road-departure increase in the held-out set, but did not deliver a lap-time benefit there. Keep the fixed early steering ZIP as the current local comparison for this mechanism. Further work should diagnose why actual preview events occur at relatively high speed and whether timing, braking or corner exit control can improve without using consumed held-out or confirmation cells for tuning.
