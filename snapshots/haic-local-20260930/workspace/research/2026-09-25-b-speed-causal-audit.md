# Strategy B: episode-level speed and failure audit

Date: 2026-09-25  
Scope: read-only aggregation of the existing Strategy B TUNE traces. No training, rollout, replay, evaluation, test, or code change was performed for this audit. This note is not a promotion decision.

## Source records

- Raw episode and action traces: `artifacts/haic/ppo-teacher-warmup-route-rescue-20260925/tune-evaluation.json`
- Run parameters: `artifacts/haic/ppo-teacher-warmup-route-rescue-20260925/preflight.json`
- Existing throttle expansion screen: `artifacts/haic/throttle-envelope-ppo-2seed-u8-20260925T135214KST/paired-comparison.json`
- TRAIN hazard exposure counts and registered next-step rationale: `RESULTS.md` (2026-09-25 hazard exposure and throttle screen entries)
- Reward implementation: `training/train_policy.py`, especially `shape_transition_reward()` and the `recovery_clearance_reward` gate
- Existing shaping helper with no training call site: `training/hazard_potential.py`

## Episode-level result

The tune file has eight episode rows, but only four distinct policy traces. Within each arm, environment seeds `20260922` and `20260926` have byte-equivalent decision traces. All eight rows use the same TUNE geometry (`custom-track-haic-tune-20260922`); the map count is one. Thus the table below reports each distinct arm once, not eight independent samples.

| Arm | Decisions | Progress | End reason | Collisions / damage | Mean / P50 / P90 / max speed (m/s) |
|---|---:|---:|---|---:|---:|
| seed8104-off | 289 | .741 | crash | 5 / 1.0 | 47.29 / 49.79 / 55.80 / 63.20 |
| seed8104-adaptive | 253 | .588 | off-track | 0 / 0.0 | 49.64 / 50.53 / 61.33 / 65.88 |
| seed8105-off | 163 | .251 | off-track | 1 / 0.2 | 16.48 / 0.01 / 55.61 / 56.86 |
| seed8105-adaptive | 301 | .551 | crash | 5 / 1.0 | 52.01 / 53.25 / 66.44 / 72.91 |

Completion was 0/8. Three distinct arms ended through crash/contact and one went off-track without contact. The non-stall arms maintained roughly 47–52 m/s mean speed before retiring; their failure is not explained by an intentionally slow target. The training target in this run was 70 m/s.

## Is the low mean speed caused by a slow-speed target?

No for the conspicuously slow `seed8105-off` episode. Its ten decisions immediately before the first contact averaged 50.66 m/s (`gas=.007`, `brake=.036`). On the last pre-contact decision (step 60), the obstacle was 7.30 distance units away, speed was 45.82 m/s, gas was `.003`, brake was `0`, and steer was `-.361`. Contact followed on step 61.

After the contact, 101 of the remaining 102 decisions were below 1 m/s and also had gas above `.25`; progress stayed at `.251` through retirement. The episode P50 speed was `.01 m/s` while its P90 was 55.61 m/s. This distribution identifies post-contact immobilization as the cause of its low mean, rather than a policy that deliberately drove slowly.

The other three arms show a different failure mode. Their pre-contact ten-decision mean speeds were 50.91, 54.17, or (with no contact) sustained high pace; they ended at progress `.551–.741` through repeated collisions or off-track retirement. The first-impact approach snapshots are:

| Arm | Last pre-contact step: distance, speed, gas, brake, steer | Result |
|---|---|---|
| seed8104-off | 284: 6.97, 48.86, `.018`, `0`, `-.579` | First contact at 285; five contact decisions and damage 1.0 |
| seed8105-off | 60: 7.30, 45.82, `.003`, `0`, `-.361` | Contact at 61, followed by the 102-decision near-stall |
| seed8105-adaptive | 296: 6.03, 51.96, `.122`, `0`, `-.361` | Contact at 297; five contact decisions and damage 1.0 |
| seed8104-adaptive | no contact | Off-track at progress `.588`; last recorded speed 61.91 |

The speed trace therefore separates into two causes: an isolated low mean dominated by collision-induced stall, and otherwise high-speed but incomplete runs caused by collision or course departure. Raising the global speed target or pedal range does not address either failure mechanism.

## Check against global throttle expansion

The already-recorded throttle expansion 1.0→3.5 screen raised mean episode speed from 33.35 to 40.89 m/s and completed-lap median from 20.64 to 19.54 s. Completion remained 2/4, valid sub-13-second finishes remained 0/4, collision onsets rose 10→12, and mean final damage rose `.50→.60`. This is evidence that pedal expansion can increase pace, but not that it solves completion. Do not repeat global throttle expansion as the next treatment.

## Causal interpretation

The current direct evidence supports two linked policy failures:

1. The policy does not consistently establish and hold a safe lateral path before contact. Three distinct arms contacted obstacles at high approach speed, with no brake on the final pre-contact decision. The independent TRAIN hazard audit also found urgent-brake activation of only 9.5% and 7.6% on its two maps, and urgent non-neutral steering split close to chance between moving away from and toward the obstacle.
2. Once one arm became immobilized, the policy kept applying throttle without changing progress. The latest run had `recovery_clearance_reward=0.0` and disabled recovery-action rewards. Existing `recovery_clearance_reward` is gated to damaged, low-speed, high-risk, non-collision, on-track states; it cannot teach the pre-impact maneuver that prevents the dominant contact sequence.

This establishes a policy/action-credit problem more strongly than a throttle-envelope problem. It does not yet distinguish a visual localization miss from a poor action chosen despite adequate visual localization: raw pixels were not retained in this TUNE trace. The single-geometry, duplicated-seed evaluation also cannot support a generalization claim.

## One untested single-variable PPO hypothesis

**Hypothesis:** on urgent, risk-positive TRAIN observations, PPO receives speed/brake shaping but no positive pre-impact signal for choosing the side that increases obstacle clearance. A small signed lateral-clearance reward will make the steering response directional and reduce contact without increasing the throttle envelope.

**Single variable:** add `preimpact_signed_clearance_coef`, with paired arms `0.0` and `1.0`; keep initialization, PPO budget, seeds, all other rewards, throttle/brake transforms, and actor inputs fixed. Apply it only in TRAIN when visible hazard risk is at least `.8` and `abs(relative_px) > 1`, using the registered geometry proxy `relative_px = 12 * obstacle_lateral + 42 * road_center_offset` and signed action term `-sign(relative_px) * steer`, scaled by visible risk and the existing `REWARD_SCALE`. The coefficient `1.0` gives at most `.1` reward per full-scale action at unit risk because `REWARD_SCALE=.1`. This is distinct from the global throttle screen and from the existing post-damage recovery-clearance term.

**Falsification / stop rule:** in a paired two-PPO-seed TRAIN screen, reject the hypothesis if both seeds do not gain at least 15 percentage points in the fraction of urgent non-neutral steering actions directed away from the obstacle versus their matched zero-coefficient controls. Do not advance to TUNE if this mechanism gate fails. If it passes, keep the candidate only when locked TUNE completion does not regress and collision/damage do not increase; one TUNE geometry is still insufficient for promotion, so held-out and official checks remain required.

No new experiment was started by this audit, and SOTA/RESULTS pointers were not changed.
