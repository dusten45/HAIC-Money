# 2026-09-25 visual-control diagnosis

## Scope and conclusion

Read-only comparison of the two requested TUNE artifacts and their registered route/map files. No rollout, replay, evaluation, training, test, or code edit was performed. The only new file is this diagnosis.

The post-warmup-named checkpoint's steering is plainly **not constant**, unlike the fresh actor. That is evidence of changing actions, but not proof that steering changes in response to visual input: every `controller` entry is `null`, and the traces contain neither image/frame data nor the visual feature vector given to the actor. In addition, the warmup-named artifact's own `experiment` field says `fresh PPO Lagrangian safety-cost paired screen`; the two TUNE JSONs alone therefore do not establish that teacher warm-up was the only difference.

For the fresh `seed8104-adaptive` off-track run, the earliest route departure is most consistent with a **persistent steering bias**: the car leaves the initial bend as the mapped centerline becomes straight, while emitting the same steering and pedal values at every decision. Speed amplifies the excursion, but is not the best explanation for its onset. In the separate warmup-named `seed8104-adaptive` run, the final loss is more consistent with **insufficient steering/speed management through a tightening bend**; it is collision-free, but the trace is not literally obstacle-free at the first departure point.

## Evidence from the two TUNE artifacts

Both files report `split=tune`, two evaluations per arm, one TUNE geometry, no held-out/official maps opened, and no checkpoint updates during evaluation. The registered route is `custom-track-haic-tune-20260922` with seed labels `20260922` and `20260926`; those labels reuse the same map geometry, so they are repeat runs, not two independent track designs.

| Artifact / arm | Episode result | Steering evidence |
|---|---|---|
| `ppo-teacher-warmup-route-rescue-20260925/tune-evaluation.json`, `seed8104-adaptive` | Both labels: progress `0.588477`, 253 decisions, `off_track`, 0 collisions, damage `0`; max speed `65.884 m/s`, mean `49.635 m/s` | Across 506 trace rows, steer range `-0.592…+0.490` (SD `0.132`). Example: step 1 `-0.279`, step 30 `+0.002`, step 60 `-0.452`, step 80 `+0.010`. Actions vary; the trace cannot tie that variation to a visual cue. |
| `lagrangian-fresh-runtime-architecture-20260925/tune-evaluation.json`, `seed8104-adaptive` | Both labels: progress `0.090535`, 174 decisions, `off_track`, 0 collisions, damage `0`; max speed `58.579 m/s`, mean `40.223 m/s` | Across 348 trace rows, steer is exactly `-0.121`, gas `0.066`, brake `0`. This is a fixed-output failure, not evidence of a responsive steering policy. |

In the warmup-named trace, the final departure sequence is:

| Step | Progress | Speed (m/s) | Steer | Gas / brake | Approx. distance to nearest centerline vertex | Nearest obstacle distance |
|---:|---:|---:|---:|---:|---:|---:|
| 208 | `0.588477` | 44.9 | -0.046 | 0.153 / 0 | 6.5 | 18.2 |
| 209 | `0.588477` | 45.2 | -0.133 | 0.085 / 0 | 9.3 | 19.5 |
| 210 | `0.588477` | 46.4 | -0.029 | 0.289 / 0 | 12.6 | 21.7 |

The route file has 243 centerline points and `geometry.width=8`. Near the closest route indices 160→158, the local point-to-point turning radius estimated from adjacent centerline heading changes tightens from roughly `18.1 m` to `12.9 m`. The vehicle's progress stops at `0.588477` as its centerline-vertex distance grows; by step 253 it is at `61.9 m/s`, steer `-0.094`, and over `214 m` from the nearest centerline vertex. The nearest obstacle distance has grown to `222.9 m`; there are no collision frames. This supports a high-speed/understeering interpretation of the final excursion, but the vertex-distance and radius calculations are geometric approximations, not the simulator's per-step off-track flag.

The fresh actor departs earlier and without nearby obstacle contact:

| Step | Progress | Speed (m/s) | Steer / gas / brake | Nearest centerline-vertex distance | Nearest obstacle distance |
|---:|---:|---:|---:|---:|---:|
| 39 | `0.074074` | 30.2 | `-0.121 / 0.066 / 0` | 7.1 m | 148.8 m |
| 40 | `0.074074` | 30.6 | `-0.121 / 0.066 / 0` | 9.0 m | 146.4 m |
| 42 | `0.074074` | 31.4 | `-0.121 / 0.066 / 0` | 13.3 m | 141.4 m |

At route indices 16–22, adjacent centerline samples are nearly straight (about `3.95 m` apart with approximately `0°` local heading change). Progress stalls while the same `-0.121` steering persists; speed later rises to about `58.6 m/s`. Thus the trace points first to steering not relaxing as the initial turn exits into a straight, with speed worsening the resulting loop. It does not point to obstacle handling or a sharp local curve as the trigger.

## Limits and data-quality note

- In both TUNE files, all decision rows for `seed8104-adaptive` have `controller=null` (warmup-named: 506/506; fresh: 348/348). The row fields contain action, position/yaw, speed, progress, collision, and nearest-obstacle distance, but no image/frame, actor visual features, or per-step off-track flag.
- The warmup-named file is `ppo-teacher-warmup-route-rescue-20260925/tune-evaluation.json`, but its `experiment` value is `fresh PPO Lagrangian safety-cost paired screen`. Its checkpoint paths point inside the named artifact directory, yet this JSON does not record the warm-up source or epochs. Treat “post-warmup” as the run label, not a verified causal treatment, until provenance is reconciled.
- The two episode seeds in each arm map to one geometry. Repeated identical outcomes increase confidence in reproducibility on this map, not generalization across maps.
- `off_track` is recorded only as the terminal reason; the logged trace cannot identify the exact decision at which the evaluator first considered the car off track.

## Minimum telemetry for a decisive follow-up

At each decision, align and save: (1) the exact actor input frame/stack reference or the actual visual feature vector used by the actor; (2) pre-squash policy means and sampled/executed steer, throttle, and brake; and (3) route-relative lateral error, heading error, local curvature/lookahead radius, speed, and an explicit per-step off-track flag. A few pre-action frames around the first divergence are enough for visual inspection; the compact feature/action series should cover the whole episode. This would distinguish “the curve was not represented” from “the actor saw it but produced too little steering/braking.”

## One-variable PPO hypothesis

Hypothesis: the warmup-named actor's final failure is driven primarily by excessive speed for the tightening bend, rather than by obstacle contact. Starting from the same verified warm-start checkpoint, change only PPO's `safe_speed_reward_target` by `-8 m/s` (for example, `50 → 42 m/s` if 50 is confirmed as the actual baseline); hold optimizer, training budget, reward weights, seeds, and all other settings fixed. Prediction: speed should fall before route indices 160–158 and centerline distance should remain within the route width long enough to pass progress `0.588477`. If speed falls but the same departure and weak steering remain, this hypothesis is not supported and the next diagnosis should focus on the steering response. This is a one-geometry TUNE discriminator only, not a generalization claim.

## Source artifacts

- `artifacts/haic/ppo-teacher-warmup-route-rescue-20260925/tune-evaluation.json`
- `artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/tune-evaluation.json`
- `training/maps/site/site_map_split.json`
- `training/maps/site/custom-track-haic-tune-20260922.json`
