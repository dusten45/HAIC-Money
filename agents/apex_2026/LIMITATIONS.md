# Measured limits and tested improvements

This is the 2026-10-05 KST Apex investigation, not a claim about official ranking.
The unchanged simulator supplies only image observations to the runtime agent.
Telemetry used below belongs to research diagnostics and is never an agent input.

## Perception and motion state

Optical flow failed on roughly half of the fast line-planner decisions. The old
controller omitted motion preview when flow failed, including critical turns.
A calibrated decoder reads yaw and actual front-wheel angle from the public HUD.
Matched required-cell times improved from18.32/22.80/20.18/18.42s to
17.14/21.42/19.48/17.92s, with all eight laps damage-free. This changes both yaw
accuracy and availability of prediction through flow dropouts; it does not
isolate those two effects. The [independent review](results/hud-review.json)
checks source/configuration, timestamps, pixel-only boundaries and real-trace
measurement error. Lateral slip still depends on optical flow.

## Path geometry and anticipation

Integer-grid path steps generated artificial curvature even on straight roads.
Continuous constrained refinement removed that source of unnecessary slowdown.
The sampled horizontal free intervals do not guarantee Euclidean vehicle-footprint
clearance or collision-free connecting segments. No such guarantee is claimed.

The remaining HUD100 development failure enters a bend behind an obstacle at
88.35m/s. Its speed target remains100 until the obstacle appears; it then falls
abruptly and braking immediately produces about184m/s² deceleration. The car
overshoots and becomes unstable during recovery. Thus this failure is not
explained by delayed pedal execution. Limited visible horizon, obstacle-bypass
geometry and recovery control remain relevant. See the
[reconstructed cap/trajectory evidence](results/line-development-failure-t1-seed3601050001.json).

## Pedal allocation and traction

The current controller's positive throttle bias and brake switch leave no coast
phase near target speed. Short brake/reacceleration sequences are measured;
some follow genuine target changes, so not all are waste. Isolated unchanged
Car dynamics support a rolling-state pedal model across20–90m/s, with much
larger errors under tire saturation or slip. The separate P1 pedal-allocation experiment subsequently finished required4/4
and development12/12. Frozen holdout10/12 failed corroboration, so the observed
local improvement is not a universal robustness claim. Failed optical flow
represents unknown slip as zero; the rolling-model domain is not established
on those frames.
See [pedal evidence](results/pedal-allocation.json).

Traction reservation originally depends on commanded steering. Current wheel
angle and measured yaw may indicate larger lateral demand during reversals;
the separate actual-state traction experiment rescued one development cell
but failed required track4 and was rejected. Calibrated
friction approximations are not formal stability guarantees.

## Independent rollout planner

Joint steering/pedal/pulse planning achieved four required finishes but only
6/12 development finishes. More accurate instantaneous state did not repair
this. Even when an80ms step is predicted well, measured0.96s open-loop position
error averages2.50m (90th percentile5.74m), large relative to road width. Neither
HUD-state revision nor yaw-only ablation qualified. See the
[52-episode ledger](results/rollout-experiments.json) and
[prediction diagnostic](results/rollout-prediction-check.json).

## Goal and generalization limits

No investigated candidate has completed all four required tracks in10–13s.
The target demands a substantially different speed/line tradeoff; current
centerline relaxations are diagnostics, not a proof that the target is impossible.
Candidate selection uses repeatedly consumed required/development cells and
therefore does not measure unseen-road generalization. The final source/configuration
freeze preceded the one-time holdout, whose10/12 finishes failed the11/12 gate. Historical or partially reused receipts
are labeled explicitly; final validation must be newly executed.
