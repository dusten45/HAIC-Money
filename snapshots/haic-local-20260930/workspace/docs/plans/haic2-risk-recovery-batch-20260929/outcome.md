# Four-direction pixel-risk batch: no candidate advanced

The exact stable-risk control and four new pixel-only safeguards ran on already-consumed TRAIN 1:43 and 2:102. All five arms finished 2/2. Candidate action changes were 45 curve approach, 17 edge closure, 20 lateral drift, and 24 obstacle closing across the two cells. This established activation only; it did not establish a performance gain.

The same frozen sources then ran on newly registered TUNE tracks 1–3 × seeds 344–347, twelve matched cells per arm and sixty total episodes. Those cells are now consumed.

| Arm | Valid finishes | Median finished lap | Action changes |
|---|---:|---:|---:|
| Stable-risk control | **12/12** | **25.40 s** | — |
| Curve approach brake | 10/12 | 25.52 s | 275 |
| Road edge closure brake | 10/12 | 25.55 s | 121 |
| Lateral drift brake | 10/12 | 25.53 s | 65 |
| Obstacle closing brake | 11/12 | 25.76 s | 149 |

All four candidates lost valid finishes and had slower median finished laps. Curve approach retired off-track on 1:347 and 2:346; edge closure and lateral drift on 2:346 and 3:347; obstacle closing on 2:346. All sixty episodes had zero invalid actions. The registered completion-first decision is **REJECT** for all four; no held-out or confirmation cells were opened, no ZIP was built, and no SOTA or official model changed.

These interventions reacted to single-frame or adjacent-frame risk signals by imposing a brake. The measured failures argue against treating small reactive braking as a sufficient repair for the stable-risk ZIP's off-track behavior. The next search should examine a different control mechanism, such as a trajectory-aware teacher trained only on TRAIN geometry and then distilled into a pixel-input actor. This is a hypothesis, not measured performance.

Evidence: [TRAIN report](../../../artifacts/haic-research-v2/haic2-risk-recovery-four-direction-train-20260929/report.json), [TUNE report](../../../artifacts/haic-research-v2/haic2-risk-recovery-four-direction-tune-20260929/report.json), and [TUNE gate report](../../../runs/haic-research-v2/haic2-risk-recovery-four-direction-tune-20260929/integration_report.json). Frozen executable source remains at `tmp/haic2-risk-recovery-batch-frozen-20260929/`.
