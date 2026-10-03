# Project Context

## Current Objective

Develop the strongest rule-compliant agent while preserving frozen studies.
The active bare route remains `_CompoundClearingBrakeCarryController`. The new
`_TemporalReachabilityController` tracks obstacle motion, checks visible road
passage and steering reachability, and retains a safe side through uncertain
camera rows. Its source-bound consumed ten-cell diagnostic now finishes10/10
versus baseline3/10, with contacts4 versus23. It recovers the formerly stalled
2/4089604952 and crashed3/3857792434 cells, but shared completed time is
7.81% slower across three pairs versus a registered5% limit. Far-side speed
relief at targets24/30 caused a new3/385 DNF and was rejected. The candidate
is **not activated**; fresh v6 screen/confirmation/blind remain the required
generalization gates. See `experiments/temporal-reachability-dev-v1-result.json`
and `experiments/temporal-reachability-generalization-v1.json`.

Prior feasible-corridor v1-v5 comparisons were rejected; their frozen results
are `experiments/feasible-corridor-dev-v1-result.json` through v5. Exact v5
replay found a late false right-side opening on2/408, while the v6 consumed
trace showed that missing nearby road rows could wrongly block an escape on
3/385. Source-bound v6 steering corrections fix both diagnostic cells. The
18 speed cap when no full corridor is visible remains the principal pace cost
on1/17 and2/644. Reused-cell fixes cannot establish unseen-track performance.

The frozen `_ObservedEgoSideSwitchController` is **REJECTED** on its
new-geometry screen:
11->17 finishes and mean progress0.675->0.777 on eight geometry seeds crossed
with IDs1--4, but contacts65->73 and damage13.0->14.6, with paired safety vetoes
on seven cells. All68 cold episodes, source hashes and two deterministic repeat
pairs passed independent audit. Its confirmation/blind partitions stay sealed.
Full record: `experiments/observed-ego-side-switch-generalization-v1-result.json`.

The preceding23-cell reused development comparison improved finishes13->17 and
contacts32->31, illustrating why it was insufficient for promotion. Exact
replays of two fresh regressions found that the ego-position rule blocked a
beneficial baseline obstacle-side switch at screen steps116 and195. The
diagnostic feasible-corridor controller uses visible road edges and obstacle
width; `Agent` still routes to the baseline. Its first consumed-cell version
failed the safety gate and cannot advance. The earlier observed-margin
candidate also remains **REJECTED**;
neither rejected study's confirmation/blind seeds may be recycled.

Among the prior fresh screen's six margin-candidate DNFs, four had the wrapper's
`off_track` label. That label means101 consecutive negative-reward decisions;
three of the four had no fully off-road samples at decision boundaries. Exact
source-bound baseline telemetry on consumed1/3857792434 and3/3857792434
reproduced the original action hashes: both cars stayed on road with road vision
and an obstacle continuously present, but were almost stationary with no new
tile for the last101 decisions. Road-loss fallback was not the cause. Future
obstacle-stall recovery needs a separately registered candidate and clearance
evidence; the ego-switch screen is rejected.
Track2 still has five-contact obstacle failures. Windows ZIP static/smoke passed
for the old frozen screen sources; official-like Linux CPU certification is
unavailable on this host because WSL lacks Python3.11/dependencies and Docker
is stopped.

No validated SOTA champion exists. Fresh confirmation/blind, website submission
details, and Linux certification remain unverified. No further DrQ outcome was
verified in this checkout. Competition documents and source-cited `report.pdf` were reconstructed
on2026-09-30 from pinned official sources and existing artifacts.

## Frozen Contract

- Do not modify `core/`, `env_wrapper.py`, or `damage.py` for experiments.
- Four `84x84` grayscale frames, CHW float32 `[0,1]`; official continuous action
  `[steer,gas,brake]` in `[-1,0,0]..[1,1,1]`. No fifth plane or smoothing.
- Raw official reward, no normalization, shaping, finish bonus or collision penalty.
- One decision per environment step; frame skip 4 only inside the environment.
  Pure time limits bootstrap; finish/crash/off-track endpoints do not.
- Finish requires 95% progress AND a valid forward finish crossing. Geometry is
  seed-driven; `track_id` changes obstacles. Reserve geometry seeds across ALL IDs.
- Exported CPU actor results are authoritative. Rank finish rate, then progress,
  then completed lap time. Freeze actor hashes before confirmation; never select
  on confirmation/blind. Diagnostic receipts cannot promote or unlock blind.
- CPU gate: Python 3.11, Torch 2.1 CPU, NumPy 1.26; init10s, reset/action5s,
  process1,024MB, ZIP500MB. Latest official participant commit rechecked on
  2026-10-03 is `dfb7a2de2178825ca5c5ce20bab01ba67052ba31`; environment files remain
  identical to the prior frozen reference. No trainer/SB3/prohibited imports in
  submission inference. Official submission-container execution remains separate.

## Completed L2 Study

Protocol: `experiments/drqv2-steering-logit-v1.json`. All four runs completed in
`runs/20260922-drq-steering-l2-v1-fast/`, frozen source `8e5fa46`:
control/L2 coefficients0/0.001, training seeds0/1, **131,072 decisions and121,073
updates each**, replay100,000, batch64, warmup10,000, sampler917, tracks1--4.
Every run selected its131,072 checkpoint over65,536 using the same CPU screen.

| Arm / Seed | Screen Finishes | Fresh Confirmation Finishes | Confirmation Progress |
| --- | --- | --- | --- |
| control / 0 | 7/24 | 6/32 | 0.735757 |
| control / 1 | 5/24 | 7/32 | 0.638823 |
| L2 / 0 | 3/24 | 3/32 | 0.519913 |
| L2 / 1 | 0/24 | 0/32, diagnostic-only | 0.422361 |

**Reject L2 at0.001:** paired finish deltas are **-3 and-7**. No promotion,
scale-up, submission or blind evaluation. Do not retrospectively nominate a
control for blind after seeing confirmation. Controls themselves reproduce
nonzero completion across both training seeds; this is not DrQ family rejection.

- Screen: IDs101--103/seeds31001--31008. Confirmation: IDs211--214/seeds32101--32108,
  now consumed. Blind IDs221--223/seeds32201--32208 remains untouched/reserved.
- Independent final audit verified all8 checkpoints, matching episode-index
  streams/exclusions, frozen actor/config/source lineage, and all640 CPU executions.
  Exact paired traces, zero operational failures; max action4.064ms,
  initialization1.218s, whole-worker RSS344.5625MiB.
- Those640 executions represent320 canonical actor/checkpoint-cell observations,
  56 distinct cells and16 geometry seeds, NOT640 independent trials.
- Full result: `experiments/drqv2-steering-logit-v1-result.json`.
  Mechanics: `experiments/drqv2-steering-logit-v1-diagnostics.json`.
  Operational history: `experiments/drqv2-l2-execution.json`.

## Final Controlled Follow-Up

**STATUS UNVERIFIED:** augmentation padding **4 versus1**,
`steering_logit_l2=0` in BOTH arms. No combined repair. This is attempt2 of2.
Protocol: `experiments/drqv2-augmentation-pad-v1.json`; explicit root
`runs/20260922-drq-augmentation-pad-v1/`, frozen source `a28ef02`. The protocol
plans all four arm/seed combinations from scratch with the same131,072 budget
and all other settings unchanged.
Operator state is in `experiments/drqv2-pad-execution.json`. Its last local
update was2026-09-22; the declared run root and pad actor are absent from this
Windows checkout. The original Linux/GPU host was not checked, so do not infer
completion or global failure from the stale local `training` label. All222 related tests,
CPU21 preflight, four-job GPU/CPU smoke and idempotent recovery passed before launch.

- Reuse only the consumed development screen101--103/31001--31008.
- Fresh confirmation311--314/33101--33108; fresh blind321--323/33201--33208,
  two CPU reloads each. Reserve180 seeds including every previous reservation.
- Keep strict positive confirmation finish gain in BOTH seeds, nonzero treatment
  screen/confirmation and all CPU gates. Finalist fixed by screen before confirmation.
- If this final trial does not reproduce improvement, **stop further DrQ tuning**.
  No third axis, coefficient search, reward/frame-skip/architecture bundle, or new
  algorithm. Positive evidence permits only a separately declared next stage.

Why this one axis: two2,048-state control replay pools, each shared across all four
actors, show current controls' pad4 view-pair steering sign disagreement at
9.03--9.33%, versus3.32--3.71% for pad1. L2 removes exactly-zero squash derivatives
in both seeds, but seed1 becomes MORE saturated and both seeds lose completion.
The penalty also changes longitudinal policy on identical inputs. History is used;
no new step/terminal bug was found. These are mechanisms/associations, not causes.
Smaller test shifts naturally change actions less; **only matched fresh completion
can accept weaker training augmentation**. L2 seed0 already reduced shift sensitivity
while losing finishes, so lower sensitivity or saturation alone is not success.

Additional observations: critic gradients are heavily clipped yet finite, with
Adam confounding naive scale interpretations; negative-reward tails dominate replay,
but L2 seed0 had MORE finish support than control0. Finishing policies also oscillate.
Do not attribute failure simply to insufficient finish data, gas/brake overlap,
noise decay, or steering sign changes. Pre-study diagnostics are preserved in
`experiments/drqv2-pre-l2-diagnostics.json`.

## Demonstration Replay Implementation

Implemented, not trained or performance-evaluated. A training-only corridor
teacher artifact stores raw-reward, terminal-safe DrQ transitions with official
three-dimensional actions converted once to native symmetric coordinates. DrQ
keeps this artifact in a replay separate from online experience and draws an
exact fixed number of demonstration rows per learner batch. The demonstration
replay, sampling RNG and source lineage are checkpointed; exported actors remain
unchanged and contain no teacher or replay. Correctness coverage uses only fake
environments and synthetic transitions.

## Local Visual Diagnostics

The local simulator can run an `agent.py` against a reproducible map with camera
frames enabled. `local_simulator.diagnostic` writes the run log, a standalone
HTML replay with synchronized camera/action/trajectory telemetry, and optionally
an annotated MP4. The report's saturation, gas/brake overlap, and collision
signals are observational flags only; no performance evaluation is performed.
The Track Lab web server now exposes a local Agent catalog under
`<artifact-root>/agents`, lets the browser choose a ready model, and passes that
selection to the automatic run endpoint. Map files, official seeds, and generated
custom tracks remain selectable in the same UI.
Track Lab derives collision locations from the existing `steps[].collision` and
`steps[].position` telemetry without changing the run schema. Live manual maps
retain numbered collision markers, and loaded/automatic replays reveal each event
from its occurrence onward with its step and vehicle-center world coordinates. These coordinates
are sampled at the end of the collision-bearing control step, not at the Box2D
contact manifold.

The three `evaluation_videos` maps were recovered from independent minimap shape,
track-length and obstacle fingerprints as official `(track_id, seed)` cells:
`(1,516237)`, `(2,644062)`, and `(3,1007)`. Runnable map specs are under
`training/maps/evaluation/`; the evidence and video hashes are in
`experiments/evaluation-video-track-seed-recovery-v1-result.json`. These are
diagnostic/replay-derived cells and must not be represented as unseen evaluation.

## Bare Baseline Runtime Repair

The neural weights in bare `model.pt` are bypassed after road detection, so the
controller selected by `Agent` is the effective policy. The accumulated
`_RacingLineController` was not backed by a matched protocol: on the deterministic
consumed development cell track1/seed42 it twice stopped at progress `0.816254`
after 782 decisions. Its visual corridor hazard fired on 680/782 recorded frames,
making the safety state too broad for targeted longitudinal repair.

Experiment `racing-line-hazard-recovery-v1` tried to preserve overspeed braking
under that gate. Although 92 tests passed, the preregistered smoke crashed at
progress `0.356890`, with damage 1.0 and five collisions; the change was rolled
back and recorded as `REJECT`.

The replacement `_StableCompletionController` is a standalone behavioral copy
of the smaller controller at source commit `52976fe`. Only bare state-dict routing
changed. It matches the frozen implementation on all 782 recorded frames and on
the full canonical 356-step trajectory. The single preregistered consumed-cell
smoke finished in 28.4s with progress `0.964664`, zero damage and zero collisions,
matching the independently reproduced reference exactly. This justifies the local
runtime repair but is diagnostic, reused-cell evidence: it does not establish
unseen-track generalization or SOTA. See
`experiments/stable-completion-controller-v1-result.json` for hashes and raw-log
pointers.

The submission-4 video protocols compose speed cap30 for obstacle+floor-speed
curves, half-strength curve-command retention when avoidance would cancel a bend,
and a clear-straight-only gas envelope0.10. Their final guarded source is commit
`5192c2b`; all results remain `INCOMPARABLE` without environment execution.

Submission-5 videos show Track1 finished27.50s (versus28.36), Track2 finished32.96s
(versus prior DNF20.5%), and Track3 failed at19.4% from repeated contact with the
orange obstacle while still inside the road. Track3's replay obstacle is visible
well before contact, but the replay is top-down and does not establish policy-camera
detection timing. Track1/2 remain center/outside-biased through slow U/hairpin
sections. This supports only the hypothesis of earlier reactions to geometry already
detected in the current policy camera, not track-ID, minimap, timing,
memorized-coordinate, or global inside-edge behavior.

At commit `8a3428d`, the final bare route is `_ObstaclePriorityController`. On an
already detected straight obstacle it floors inherited urgency at0.50; on a
centered/aligned current corridor with agreeing rows30/34 and no obstacle it applies
only a +/-0.05 distant-road preview. Independent review found that preview history
could weaken an immediately opposing obstacle command, so a separately registered
gate restores that first command to the no-preview obstacle control; same-direction
and non-obstacle transitions remain exact. Always-inside bias was rejected. The
frozen stable class, `model.pt`, official environment, DrQ, explicit-export and HAIC
paths remain unchanged. Final affected tests are159 passed; a pre-final broad run
was386 passed,10 skipped. Track execution was intentionally not performed, so
completion, lap-time and unseen-track improvement are not established. See the
three `guarded-distant-*-v1-result.json` / `anticipatory-*-v1-result.json` records.

The new unbound 4_7 videos visibly finish Track1 in27.18s, Track2 in33.22s and
Track3 in30.58s. Versus submission5, Track1 improves0.32s, Track2 regresses0.26s,
and Track3 changes from a19.4% collision-limit DNF to a finish. Matching T3 frames
show the same approach progress but a safe pass instead of contact, rotation and
stall. The replay does not expose policy-camera frames, actions or speed; therefore
it does not establish low top speed or an optimal apex line. A global inside bias
would also lack clearance calibration and can oppose obstacle avoidance.

At commit `a920139`, the routed bare controller was `_LaunchThrottleController`.
Only when the inherited policy already classifies a valid, obstacle-free, settled
straight and requests its full-gas branch (estimated speed strictly below transient
target minus8) does emitted gas change from0.10 to0.11. Target speed, stored EMA,
braking boundary, curves, previews, obstacles, steering, model and other runtime
paths are unchanged. Exact-boundary and transition tests plus the affected local
suite passed135 tests; an isolated temporary package passed route/action/reset
smoke. No environment run measured performance, so the result is `INCOMPARABLE`.
See `experiments/video-4-7-evidence-ledger-v1.json` and
`experiments/clear-straight-launch-throttle-v1-result.json`.

The `_CompoundSpeedMarginController` diagnostic at commit `ae77e3a` lowered the
compound transient target from30 to24. Newer user-attributed logs subsequently
showed seed21 retiring at the identical progress despite lower hairpin speed, while
a same-map seed42 finish was1.34s slower and contained five speed-below-28 segments.
Because those logs lack source/model/package hashes, this is adaptive development
evidence rather than a controlled comparison. The target24 route was retired at
`f7b4743`; the historical class and tests remain for auditability.

Consumed official-generator runs for local seeds11,17 and21 all ended off-track
after an outside obstacle pass into a continuing hairpin: speed had fallen to
about29--30, but
absolute steering unwound to0.012--0.027 just before road loss. Only seed17 had an
earlier collision, so collision was not the common terminal mode. Run logs lack
source/model/package hashes and therefore are motivation, not bound confirmation.
The prior bare route was `_TranslationInvariantExitController` at commit
`5f9af82`. It retained the clear-curve speed envelope from `_FastCornerCarryController`
but fixes its exit classification: the historical sweep substitutes image center
for unsampled rows, so a perfectly parallel corridor displaced6px was misread as
sweep6, target53.76 and brake0.091 at speed60. The new gate requires every sampled
row30--54 to align relative to actual row54, offset<=6, mild steering and no
obstacle/latch; it then uses target68 and straight gas while inherited recentering
steering remains bit-exact. In the same synthetic offset6 case the action changes
from brake0.091 to gas0.135, while real curves, a rows46/50 S-kink, obstacles and
all latch frames remain control-exact. The change passed20 focused and196 affected/
local/submission tests, isolated exact-file package smoke and final independent
review. `_CoherentCurveAttackController` at commit `42e8226` subsequently applied
a20% steering gain on every coherent obstacle-free curve. Consumed same-map run
chronology then exposed a seed42 regression: FastCornerCarry immediately preceded
a25.90s zero-collision finish, while TranslationInvariantExit and CoherentCurveAttack
immediately preceded five- and four-collision DNFs at the identical37.81% obstacle.
The failures began avoidance one progress tile later and arrived2.8--3.5 farther
from centerline despite similar speed. Separately, seeds11/17/21 lost the road at
speed29--30 only after post-obstacle steering unwound almost to zero.

`_PostObstacleCurveRetentionController` at commit `61df0b6` restored
`_FastCornerCarryController` as the recovery baseline and added the same20%/+0.06
bounded gain only on obstacle miss frames1--4 when stored
avoidance side, inherited steering and every sampled road row agree on turn
direction. Current obstacles, latch-free curves, opposite/S geometry and the fifth
clearing frame receive no new gain; the0.48 cap,0.07 slew and reversal protections
remain. Seven focused and212 affected tests, isolated package smoke and independent
audit passed; see
`experiments/post-obstacle-curve-retention-v1-result.json`.

The prior bare route was `_AggressiveCompoundPaceController` at commit `42f5609`,
above the retained adaptive-target, latch-release, compound brake-carry and
doubled-straight layers.
Submission12 videos finish in23.60/28.18/25.44s, but13/26 obstacle encounters still
lose at least10% on a rendered-video speed proxy. The controller removes the hard
target30 discontinuity only for a currently recognized far/moderate compound hazard:
`30+8*clip((44-y)/12)*clip((12-sweep)/6)`, with target at most38 for y<44 and
6<=sweep<12. The latch remains30, so the first detector miss restores target30;
near obstacles, extreme curvature and final requested steering above0.28 also keep
target30. The existing softened compound brake remains active throughout31--38.
At the maximum cell/speed40 this changes brake0.082 to0.058 without changing any
gas/brake constants or lateral behavior. Six focused and244 affected tests,
exact-file package smoke and two independent
audits passed. Closed-loop clearance/completion/time remain unmeasured, so the
candidate is `INCOMPARABLE`; see
`experiments/aggressive-compound-pace-v1-result.json`.

The current bare route is `_CompoundClearingBrakeCarryController`, introduced at commit
`c4e5287`. It addresses a narrower transition exposed by code review: after the
four target30 detector-miss frames, the fifth latch-clearing frame could restore
the stronger generic curve supplement for one decision before the faster clear-
curve path became eligible. Only after a far/moderate adaptive obstacle with a
nonzero steering request at most0.28, sweep in `[6,12)`, and no later steering
reversal, the clearing-frame brake becomes
`max(base, compound, control - 0.04)`. Target, gas and steering stay exact;
current-obstacle and miss1--4 behavior, ordinary curves, unsafe geometry, road
loss and the following frame remain exact. Four focused and248 affected static
tests plus two independent audits passed. Intentional off-road and global speed/
steering changes were rejected. Its original result was static-only; see
`experiments/compound-clearing-brake-carry-v1-result.json`.

New paired diagnostics restored actual driving evidence:
`compound-brake-onset-v1` ran14 episodes on seven reused cells. Both control and
candidate finished4/7 with identical trajectories; the candidate never changed a
brake command and was not activated (INCONCLUSIVE). Control results: seed42 crash
at69.26%, seed11 off-track18.92%, seed17 clean25.32s, seed21 off-track33.73%; the
three video-recovered maps finish23.66/28.12/25.26s, with two collisions on track2.
The separate `visible-compound-base-brake-v1` did alter braking, but lost seed17's
clean finish: candidate off-track at54.82%, with zero collision damage. At equal
progress53.49%, speed increased30.49 to36.99 while the later steering unwound to
zero. Seed42/21 also increased off-road samples. REJECT; stopped after9 complete
episodes, with the next candidate interrupted and remaining cells unrun. Neither
candidate is active; both remain for audit. Raw logs/hashes are referenced by the
two result JSONs. Next priority: source-bound policy-camera and internal
road/obstacle/steering telemetry at failure transitions to separate perception
loss from steering arbitration before further speed changes. No fresh holdout
was opened and no SOTA claim follows from these reused cells.

The user-requested untracked root `submission.zip` SHA is `3ac70e3a...`; it embeds
the prior `_CompoundObstacleBrakeCarryController` agent SHA `413afda3...` and
unchanged model SHA `c101c696...`. It has no official receipt binding and is stale
relative to the current source; preserve it but do not treat
it as the new candidate package.

## Runtime And Infrastructure

- Current Windows `.venv`: Python3.11.15, Torch2.1.0+cpu, NumPy1.26.0. It was used
  for the video-diagnostic unit, package and inference checks. Frozen DrQ run
  receipts declare the separate Linux `/venv/main` Python3.11.14,
  Torch2.11.0+cu128, RTX5070Ti training runtime; preserve it and re-verify its host
  before resumed learning rather than inferring it from this desktop environment.
- CPU gate interpreter: `/tmp/kilo/haic-cpu21/bin/python`, Torch2.1.0+cpu,
  NumPy1.26.0, Gymnasium0.29.1, OpenCV4.8.1.78. Preserve its symlink path when
  invoking it: resolving to the base executable loses venv isolation.
- `train_drqv2.py`: explicit run/checkpoint paths, CPU selection, replay/optimizer/
  CPU+CUDA RNG, sampler/warmup state, exact partial-episode reconstruction, incumbent
  preservation, source/runtime/protocol validation. No `_latest` dependency.
- `run_drqv2_matched.py`: executes immutable source/command snapshots, verifies
  budgets and episode prefixes, freezes actors, validates sealed CPU receipts,
  and supports non-mutating evaluation recovery. No retuning during a study.
- `diagnose_drqv2.py`: CPU21-only shared-replay offline actor comparisons, no env
  cells or optimizer updates. Raw large artifacts remain local; check existence
  after moving servers. Compact protocols/results and code are committed.
- Initial slow L2 run was stopped at last-logged22k per arm, before ANY checkpoint
  or evaluation. A CUDA-only gather preserves all scalar RNG draws/output and
  tested learner state, removing128 per-view host synchronizations. All arms
  restarted equally; live learning metrics matched through17k, throughput about3x.
  Sealed restart receipt and prefix proof are in the run roots/execution JSON.
- CPU/GPU restore/export tests, batch64 smokes, full four-job harness and idempotent
  recovery passed. RNG parity covers288 augmentation cases and successive updates.

## Earlier Evidence And Caveats

- Historical actor `runs/20260921-043514_drqv2-pilot-131072/actor.pt` recorded4/24,
  progress0.807, damage0.45. Exact historical cells/source provenance are incomplete.
  Its subsequent frozen CPU21 screen was0/24 and diagnostic confirmation4/32 on
  both reloads; no promotion. Details: `experiments/drqv2-promotion-v1-result.json`.
  Old confirmation31101--31108 is consumed; blind31201--31208 remains untouched.
- This is a native DrQ-v2 variant, not an exact author implementation: encoder
  sharing/strides and actor/augmentation details differ. These stayed fixed in L2.
- Sampled PPO at1,048,576 decisions failed0/24 development,0/32 confirmation,0/24
  blind; curriculum, collision, EMA/Markov and action-codebook variants did not
  establish completion. They remain historical, not active alternatives.
- Historical continuous PPO uses5 channels, EMA, shaping/normalization and a
  different effective sampler. It is budget evidence, not a matched algorithm-only
  control. The archived official submission is still an old PPO baseline.
- Concurrent upstream `b1ad528` introduced separate policy/planner/site work,
  preserved via merge9649c62. It is not used/trained here; official environment
  sources and the functioning GPU stack remain unchanged.
- Known unrelated merged-suite failure: `tests/test_submission_layout.py` reads
  inherited `ru_maxrss` in `training/package_submission.py` after CUDA-heavy tests;
  it passes alone. DrQ uses process-local `/proc/self/status` and passes its gates.
