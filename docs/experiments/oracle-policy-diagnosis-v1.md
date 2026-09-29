# Oracle-v1 Policy Failure Diagnosis

## Decision

The clearest repeated RLPD risk signature is **high speed and continued throttle
at curve entry**, followed by a growing lateral offset. On seed `4272000005`, the
seed-50 actor was at 29.37 m/s, 3.19 m from centerline, with curvature 0.0886
and no damage at decision 31. Its action was `[-0.799, 0.929, 0.050]`; the
immutable Oracle proposed `[-0.064, 0.000, 0.500]` on that exact policy state.
The separate Oracle rollout was at 11.30 m/s, about 0.001 m from centerline,
and completed the same road without damage.

This is evidence for a state-conditioned speed/braking mismatch, not a causal
claim that throttle alone explains completion failures. Two short interventions
reduced speed in all ten selected RLPD cases, but both worsened the lateral-error
guard in three. **No behavior-cloning update or learner training was run.** The
bounded next direction is RLPD recovery/control data that preserves the joint
steering-speed context; a closed-loop multi-decision gate should precede any
training. Simple full-vector and throttle/brake-only Oracle imitation are held.

## Frozen Cohort And Method

`oracle-v1` is read-only. Its controller and provenance hashes were identical
before and after the work. No Fresh repository, private track, confirmation,
blind, submission, or official evaluation was accessed.

The cohort contains 45 current-runtime deterministic actor rollouts on 26 unique
track-1 cells. Every geometry was already consumed in TRAIN or
TRAIN-DIAGNOSTIC; there are no fresh-cell claims.

| Frozen policy cohort | Previously failed episodes selected | Current replay failures | Current outcome changes from archive |
|---|---:|---:|---:|
| DrQ-v2 r6 unchanged-source seeds 0/1, canonical repeat 0, failed rows only | 21/21 | 17/21 | 4 historical failures replayed as finishes |
| RLPD G0 long-horizon seed 11 and V5 author seed 50, all 12 roads each | 18/24; six archived finishes retained as controls | 16/24 | 8 total: five old failures replayed as finishes, three archived finishes replayed as failures |
| Total | 39 historical failures in 45 selected episodes | 33/45 | 12/45 |

The 21 DrQ failures span 14 geometry seeds; the 24 RLPD episodes span 12 other
geometry seeds. They are reported as separate studies, not a matched cross-model
comparison. The geometry families below are the frozen DrQ catalog labels; RLPD
roads are reported by their exact IDs.

The collector reloaded each frozen `actor.pt` with the root deterministic
`Agent`, then ran the actor and Oracle under the same track ID, geometry seed,
frame-skip 4, 50 raw no-op warm-up, obstacle setting, and cohort decision cap
(1,200 DrQ; 2,000 RLPD). Before comparison it required equal reset observation
hashes and exact road-coordinate hashes. RLPD's new-runtime reset stack matched
the archived initial stack on all 24 cases; its first action differed from the
archived action by at most `2.98e-7`. The code separately reports archived and
current outcomes because earlier same-cell repeats can diverge despite these
checks: a documented repeat changed outcome after the first differing next
frame at decision 33.

Each current rollout records policy/Oracle actions, same-state Oracle proposals,
position and velocity, speed, heading and heading error, curvature, path and
centerline errors, arc length, progress, reward, damage, collision/off-track
signals, per-decision observation hashes, and one observation-stack snapshot per
25 decisions. Raw traces and their SHA-256 values are under
`runs/oracle-policy-diagnosis-v1/`; the compact, per-episode comparisons and
aggregate result are in `experiments/oracle-policy-diagnosis-v1-result.json`.

“First material divergence” is the first single physical signal sustained for
three decisions: position gap >=2 m, heading gap >=0.30 rad, or speed gap >=2
m/s. “First failure precursor” is the earliest threshold event listed in the
protocol, including curvature/overspeed/braking mismatch, opposed steering,
sustained heading error, lateral excursion, damage, and low speed. Both labels
are descriptive, not causal. The earliest speed divergence occurred at median
decision 4 for DrQ and decision 2 for RLPD; it is not itself called failure
onset. For seed 50 / geometry `4272000005`, the first rule-selected curve-entry
precursor was decision 31.

## Findings

Across the 33 **current replay failures**, the frozen first-precursor rules
classified:

| First measured precursor | DrQ r6 current failures | RLPD G0 current failures | Total |
|---|---:|---:|---:|
| Curve-entry overspeed/action mismatch | 6 | 10 | 16 |
| Sustained heading error | 6 | 1 | 7 |
| Steering opposed to Oracle on the same policy state | 3 | 4 | 7 |
| Lateral excursion | 2 | 0 | 2 |
| Contact/damage increase | 0 | 1 | 1 |

The RLPD overspeed events occurred at decisions 31–68 on seeds 1, 3, 5, 8, 9,
10, and 11 (with seeds 3, 5, and 10 represented by both actors). At the
seed-50/`4272000005` event, the preceding policy window shows speed about 29.3
to 29.7 m/s, centerline offset 1.54 to 3.19 m, and heading error moving from
-0.264 rad to -0.013 rad. Across those decisions the policy kept requesting
roughly 0.92–0.94 gas and at most 0.10 brake; the Oracle requested zero gas and
0.50 brake when queried at the policy's exact state. Damage stayed at zero in
this window, so damage was not the initiating event. The Oracle's own separate
time-aligned path is not the same pose: it remained around 11–13 m/s, near-zero
centerline/heading error, and reached lower tile progress at that elapsed step
before eventually finishing. These metrics are tied to the trace window, not an
official score.

DrQ failures were distributed rather than confined to one catalog family:

| DrQ geometry family | Current failure precursors |
|---|---|
| `easy-curvature-anchor` | 2 overspeed, 1 steering opposition |
| `finish-approach-turn` | 2 heading error, 1 lateral excursion |
| `mid-road-left-right-reversal` | 1 overspeed, 1 heading error |
| `mid-road-sustained-or-same-turn` | 2 overspeed, 1 heading error |
| `opening-delayed-high-turn` | 1 heading error, 1 lateral excursion, 1 steering opposition |
| `opening-short-entry-left-turn` | 1 overspeed, 1 heading error, 1 steering opposition |

This sample contains only `track_id=1`, so it cannot support claims about
cross-track frequency. All 12 RLPD oracle rollouts finished without damage.
Across the 14 unique DrQ roads the Oracle finished 13; on seed `3910800172` it
reached tile progress 1.0 and damage 0.2 but did not finish within the frozen
1,200-decision cap. Oracle-v1 is therefore a strong fixed reference on this
cohort, not an infallible guarantee on every geometry.

## Bounded Falsification

The selected research line was RLPD recovery/offline action data. Its first
hypothesis was that a single Oracle action at the measured curve-entry event
would correct the speed error without a short-horizon lateral or damage
regression. Ten pairs used exact-state-verified policy prefixes from the ten
RLPD overspeed failures. The control applied the saved policy action; treatment
replaced one action with the full Oracle vector; both then replayed the same next
three recorded policy actions.

| Full-vector one-action branch, after four decisions | Result |
|---|---:|
| Speed at least 0.5 m/s lower | 10/10; median treatment-control `-4.407 m/s` |
| Absolute center-error increase <=0.5 m | 7/10 |
| Damage increase <=0 | 10/10 |
| Predeclared all-gates hypothesis | **Not supported** |

To isolate steering from speed, the same ten pairs then kept policy steering and
substituted only Oracle gas/brake, again followed by three recorded policy
actions:

| Longitudinal-only one-action branch, after four decisions | Result |
|---|---:|
| Speed at least 0.5 m/s lower | 10/10; median treatment-control `-4.412 m/s` |
| Absolute center-error increase <=0.5 m | 7/10 |
| Damage increase <=0 | 10/10 |
| Predeclared longitudinal-only hypothesis | **Not supported** |

The lateral guard failed on RLPD seed 11 / geometries `4272000008` and
`4272000009`, and seed 50 / geometry `4272000010` (increases of about 0.95,
1.35, and 1.44 m in the longitudinal-only branch). Thus the Oracle's lower
speed action has a repeatable local kinematic effect, but applying it without
a compatible steering/recovery sequence is not safe under this test. These are
four-decision open-loop suffix branches, not full-episode policy improvements.
The result rejects plain single-action teacher substitution as a sufficient
fix; it does not show that a trained recovery policy cannot work.

## Reproduction And Artifacts

Frozen protocols and results:

- `experiments/oracle-policy-diagnosis-v1.json`
- `experiments/oracle-policy-diagnosis-v1-continuation.json`
- `experiments/oracle-policy-diagnosis-v1-result.json`
- `experiments/oracle-policy-action-branch-v1.json`
- `experiments/oracle-policy-action-branch-v1-result.json`
- `experiments/oracle-policy-longitudinal-branch-v1.json`
- `experiments/oracle-policy-longitudinal-branch-v1-result.json`

The initial low-priority collector invocation hit its 1,200-second shell limit
after persisting all 45 policy traces and seven Oracle traces. The hash-checked
continuation did not repeat policy episodes; it collected only the 19 missing
Oracle cells. The complete run has 45 current policy traces, 26 Oracle traces,
and 71 trace receipts. Both one-action branch protocols and their short runs
are recorded under `runs/oracle-policy-action-branch-v1/` and
`runs/oracle-policy-longitudinal-branch-v1/`.

Commands (all bounded to one CPU thread at low scheduling priority):

```bash
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.diagnose_oracle_policy_failures \
  --output runs/oracle-policy-diagnosis-v1
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.resume_oracle_policy_diagnosis \
  --output runs/oracle-policy-diagnosis-v1
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.branch_oracle_policy_action
nice -n 19 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.branch_oracle_longitudinal_action
```

The non-reset preflights are `--preflight-only` on the diagnosis/resume command
and the respective branch commands. `python -m scripts.summarize_oracle_policy_diagnosis
--policy-id rlpd-seed50 --geometry-seed 4272000005` emits the worked per-step
example window. `tests/test_oracle_policy_failure_diagnosis.py` covers sustained
divergence and precursor ordering.

The collector, continuation and branch protocols bind actor, source trace,
geometry catalog, environment, and Oracle hashes. Oracle controller SHA-256
remained `a447aa7559a9f2fdddfc65202c739dd1228dd607b8cc40aa9b8f9bb82a28d839`;
its local provenance SHA-256 remained
`f56a53a769a9b104fee35cd64d686e4fd7dd428d84f81ea8d0095383d376a61c`.
