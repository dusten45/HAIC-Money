# High-speed road-loss and collision recovery — 2026-09-29

## Outcome

Implemented six candidates across four recovery directions and ran 36 consumed TRAIN episodes, followed by 12 exact-ZIP episodes on fresh TRAIN. Target speed remains 60 in the existing estimator's units; this is not a km/h claim. Pedals remain unchanged. Brief road departures are diagnostics, not the former 1% rejection gate.

| Comparison | Original | Recovery | Meaning |
|---|---:|---:|---|
| Consumed TRAIN tracks 1–3, seeds 38200–38201 | 3/6 | impact_clear 4/6 | Preserves all three old finishes; rescues 1:38201 in 17.86 s |
| Fresh TRAIN tracks 1–3, seeds 38210–38211 | 3/6 | impact_clear 3/6 | Same three finishes, same median 18.84 s; no new rescue |

These sets are reported separately: consumed selection data and fresh TRAIN are not an independent held-out certification. Fresh candidate finishes are 1:38211 (18.76 s), 2:38210 (18.84 s), 3:38211 (18.84 s). Both fail 1:38210 at42.81%, 2:38211 at37.54% with one collision, and3:38210 at53.92%.

## Why departure becomes persistent

1. In all three original failed cells, the first center-road departure follows obstacle avoidance opposing road-following. The additive avoidance term has magnitude0.34. Example2:38200 decision53: road term−0.387 becomes−0.047 before temporal correction. This is evidence of competing controls near obstacles, not proof that globally removing avoidance is safe. An earlier global restriction lost valid finishes.
2. Missing road rows default to image center42. Example2:38200 decision112: near center30.5, missing far row42, road command becomes+0.225 despite the preceding left turn. When all rows vanish, default steering becomes0. The car can continue fast in the wrong direction.
3. Collision does not immediately terminate the episode. In1:38201, physical speed falls58.7→29.02 at decision100; gas is already0.6 afterward. Continuing acceleration alone did not recover forward progress. The environment's `off_track` label follows101 consecutive negative-reward decisions, not direct proof that every one of those decisions was geometrically outside asphalt. Re-entering previously visited road need not reset this condition.

## Implemented and measured

- Partial-row extrapolation:3/6,93 effective action changes.
- Short remembered turn:3/6,131 changes.
- Wider pixel road search:3/6,102 changes.
- Post-impact steering slew limit:3/6,11 changes.
- Stateful remembered turn until reacquisition:3/6,230 changes.
- Post-impact avoidance clearing (`impact_clear`):4/6 consumed TRAIN. On a pixel-estimated speed drop, for up to12 decisions, road steering and temporal damping replace the obstacle-added command when both road rows are visible. Speed target and pedals are unchanged; no privileged collision state enters inference.

At1:38201 the old and new first100 actions and poses, including the collision, are identical. There are two effective post-impact overrides relative to damping on the candidate's observations. Later trajectories naturally diverge. This controlled local observation supports the recovery mechanism. On fresh2:38211 it activates12 times but does not improve progress or completion, so it is not a general collision solution.

## Package and checks

Experimental ZIP: [submission-high-speed-recovery.zip](../../artifacts/haic-research-v2/fixed-high-speed-impact-fresh-20260929/submission-high-speed-recovery.zip),8267 bytes,six files, SHA-256 `f1eb42a4492e87f5d61641322cae3c0d5e335e895bef7ee4206ad1a570eda018`.

Both exact ZIPs ran in isolated child processes. All six first10 action/pose prefixes matched; all control finishes survived. Zero invalid actions. Candidate cold import/create maximum0.321s, reset<0.001s, action maximum17.42ms, peakRSS<284MB. Four focused recovery tests and v2 synthetic memory/impact checks passed. Read-only package review found no actionable defect. No official score, SOTA promotion, upload or model confirmation. Original delivered ZIP remains preserved.

Formal fresh outcome: ADVANCE for experimental local availability only, with rule_compliance UNKNOWN, mechanism_activation PASS, competitive_or_product_outcome UNKNOWN; STOPPED without release. Fresh results satisfy the registered no-regression floor but do not establish a performance improvement.

## Evidence

- [Four-direction comparison](../../artifacts/haic-research-v2/fixed-high-speed-recovery-train-20260929/recovery_comparison.json)
- [Stateful continuation comparison](../../artifacts/haic-research-v2/fixed-high-speed-recovery-v2-train-20260929/recovery_comparison.json)
- [Frozen exact-ZIP fresh comparison](../../artifacts/haic-research-v2/fixed-high-speed-impact-fresh-20260929/report.json)
- [Recovery implementation](../../haic_agent/fixed_high_speed_recovery_v2.py)

Remaining work is forward road reacquisition after the image loses the road, and resolving obstacle/road steering conflict before departure. Simple remembered turns and extrapolation were insufficient in these cells. Neither higher gas nor disabling retirement is an evidenced solution.
