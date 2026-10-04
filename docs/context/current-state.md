# Current Research State

## Apex v2 active — renewed user authorization (2026-10-05 KST)

The user requested creative, decisive further improvement after the v1 report.
New structural work is isolated under [agents/apex_2026/v2](../../agents/apex_2026/v2/README.md).
V1 frozen source/results remain unchanged and unadopted. All exposed old24 cells
are consumed regression data; v2 holdout24 is unallocated until final freeze.
See the [prospective design](../../agents/apex_2026/v2/DESIGN.md).
The [v2 progress record](../../agents/apex_2026/v2/PROGRESS.md) retains rejected
force/brake/actuator probes and geodesic r1's22/24 consumed regression result.
No matched improvement yet; geometry and observable-state physics work continues.
The [complete v2 scoreboard](../../agents/apex_2026/v2/EXPERIMENTS.md) separates
all small screens, execution failures and full regression results. R6's8/8 screen
fell to20/24 in broader testing; recovery r1 rescues one old failure while
preserving required laps, but its other old failure remains unresolved.


## Apex opposite-strategy cycle closed (2026-10-05 KST)

The user-authorized `agents/apex_2026/` lane completed on branch
`research/apex-2026-opposite-20261005`, preserving main/root agent and official
simulator. The user removed the clock deadline. See the
[final report](../../agents/apex_2026/REPORT.md), [frozen candidate](../../agents/apex_2026/candidate/agent.py),
and [source-bound summary](../../agents/apex_2026/results/final-summary.json).

The pixel-only continuous-path/HUD/rolling-pedal candidate was frozen and pushed
in `cdf675e` before all final validation and unseen allocation. Fresh required
laps are16.78/20.68/18.56/18.10s, all damage-free; declared development finishes
12/12 with median16.28s; four required repeats reproduce exact driving traces.
The original10–13s objective is unmet. The one-time fresh holdout finishes10/12
(median completed17.53s), below the prospective11/12 corroboration gate.
**Candidate remains frozen, experimental and unadopted; no official action.**
No policy was tuned after holdout. All holdout seeds are consumed, never fresh
again. Negative trials and exact sources remain preserved.81 focused tests pass;
pre-existing complete-suite errors are recorded separately. Further research
must use a distinct validation cycle and must not recycle this holdout.

## RLPD Coupled Recovery In Progress (2026-09-29)

The user authorized and resumed TRAIN-only curve-entry overspeed / steering-speed
coupling recovery validation after a Kilo-only restart. This authorization
supersedes the historical migration pause below for this RLPD iteration only.
Current collection uses only consumed G0 track-1 geometries 4272000001-4272000012;
confirmation/blind/private cells and official actions remain out of scope.

The [closed-loop validation result](../../experiments/rlpd-recovery-validation-v1-result.json)
now covers 42 completed branches, 18,411 decisions and 43 reset intents including
one externally killed duplicate attempt with unknown extra cost. The
[separate continuation](../../experiments/rlpd-coupled-recovery-r2.json) reused
28 immutable traces and collected only the missing 14. Full steering/pedal
Oracle feedback for 12/25 decisions preceded original-actor continuation to real
episode end, with lateral error, speed, damage and progress followed for at least
five seconds after handoff. Historical archive divergence was not repaired.

At 12 decisions, 2/10 failures became finishes but 2/4 finished-parent controls
were harmed; at 25 decisions, 3/10 became finishes and 2/4 controls were harmed.
Only one and three rescues respectively also passed local qualification. This
supports state-specific recovery data, not unconditional Oracle repair.

Partial primary receipts already show that an unchanged actor can pass the
five-second local condition and fail later. Learning therefore requires actual
paired full-finish rescue plus local qualification, with preserved-finish harm
controls. The data window includes the executed intervention and first 63 actor
decisions after handoff, not unexecuted Oracle proposals. Strict preparation
produced 665 accepted rows (653 unique), including 339 unique failure-support
rows across three geometries. Matched 8,192-decision raw-SAC fine-tuning of two
V5-seed50 learner-state copies completed under the
[frozen learning protocol](../../experiments/rlpd-recovery-learning-v1.json):
32online/32prior control versus 32online/16prior/16recovery. Contemporary
pixel-only finish/curve-entry evaluation completed 36 uncensored episodes in a
corrected r2 after a summary-field error stopped the retained first attempt.
Finishes are V5 5/12, matched control 4/12 and recovery replay 1/12. Recovery lost
all five original finishes and gained one different road; its prospective
curve-associated terminal failures are 1/12 versus source 1/12 and control 0/12.
The Q-only replay intervention is rejected. See the
[verified full evaluation](../../experiments/rlpd-recovery-evaluation-v1-result.json).
All three current
critics still prefer their own actor on the 87 Oracle failure-support images;
this is a ranking diagnostic, not evidence that the teacher is optimal or that
driving performance failed. See the
[action comparison](../../experiments/rlpd-recovery-action-comparison-v1-result.json).
The next isolated iteration adds joint native deterministic-mean guidance only
on proven failure Oracle rows, plus frozen-original-mean retention on prior and
handoff/preserved-control rows. It keeps raw SAC/environment reward unchanged
and reports extra actor optimizer work separately. It completed 8,192 decisions,
8,191 SAC updates and 8,191 extra actor updates under the
[guided child protocol](../../experiments/rlpd-recovery-guided-learning-v1.json)
after independent review and actual-data zero-reset preflight; 164 tests and
33 subtests passed. The [actual-visit regression review](../../experiments/rlpd-recovery-regression-review-v1-result.json)
shows changed heading/steering visits and lost handoff retention, not a uniform
saturation shift or a blanket speed/brake-only mechanism. The
[guided result checkpoint](../../experiments/rlpd-recovery-guided-v1-result.json)
records successful export/hash checks and markedly lower joint error on the
87 selected failure Oracle images. The full comparison nevertheless finished
original 5/12 versus guided 2/12, losing four original finishes and preserving
only one. The sparse curve-terminal proxy fell 1 to 0 while mean progress fell
0.6944 to 0.4804, so the guided intervention is rejected too. Action matching is
not closed-loop improvement. See the [verified guided evaluation](../../experiments/rlpd-recovery-guided-evaluation-v1-result.json).
The third isolated iteration freezes the original encoder/critics/temperature
and fits only actor correction with explicit initial/successful-path protection.
It completed 2,048 offline actor updates with all declared frozen components
bitwise unchanged, then finished 24 uncensored evaluation episodes. Completion
is tied at 5/12, but four original finishes were lost and four different roads
gained; damage and curve-terminal counts increased. It fails the preservation/
improvement gate. See the [actor-only outcome](../../experiments/rlpd-recovery-actor-only-evaluation-v1-result.json).
The fourth hypothesis uses the original actor as exact default when inactive
and a pixel-feature local-support gate to invoke the learned joint correction
for a held 12-decision sequence. Protected calibration prevents fresh triggers
on its sampled reference points, not harm guarantees during active holds or
whole episodes. The gate is now built with 87/87 covered prototypes and no fresh
trigger on 16,023 reference queries; inactive actions were source-exact. Its
[frozen protocol](../../experiments/rlpd-local-recovery-gate-v1.json) and independent
review passed, and the full suite passed 225 tests plus 37 subtests. Original-
versus-gated evaluation completed 24 uncensored episodes: original5/12 versus
gated6/12, all five old finishes retained and geometry4272000010 gained. Mean
damage0.2->0.05 and progress0.6944->0.7566 improve; sparse curve-terminal counts
remain1/1. This is the first retained consumed-TRAIN improvement, not fresh
generalization. The [unchanged-policy repeat](../../experiments/rlpd-local-recovery-gate-evaluation-repeat-v1-result.json)
reproduced the same counts and aggregates. The [primary trigger-path review](../../experiments/rlpd-local-recovery-gate-trajectory-review-v1-result.json)
verified only60/4896 decisions used correction: two12-decision holds rescue
geometry10, then383 source-only decisions finish. At5.04s after its first
trigger, heading is -0.049rad, speed21.99m/s, lateral2.28m, damage0, progress0.2465
versus source heading2.937rad/progress0.1549. All five preserved finishes had no
gate activity and exact trajectories. This closes the internal recovery-data
validation gate, not a fresh-road/independent-training or official-model gate.
See the [verified local-gate outcome](../../experiments/rlpd-local-recovery-gate-evaluation-v1-result.json).
No geometry/pose/Oracle inputs
are allowed at runtime and no fresh generalization is claimed.
There is
no fresh-road improvement or candidate promotion. See the
[active RLPD iteration](../plans/rlpd-completion-first-research-2026-09-26.md).

The [zero-reset precursor review](../../experiments/rlpd-recovery-precursor-review-v1-result.json)
confirms heading/lateral thresholds occur in successful controls too. Seed52's
archived traces lack curvature and same-state Oracle actions, so overspeed and
steering opposition are unassessed rather than inferred from old actors.
The [foundation checks](../../experiments/rlpd-recovery-foundation-check-v1-result.json)
establish synthetic code/checkpoint feasibility only, not driving performance.

Last refreshed: 2026-09-29 for the user-directed DrQ-v2 closure. This is the
current-state source of truth, not an experiment changelog. Evidence and
historical decisions are linked below.

**Instance-migration freeze (2026-09-27 05:10 UTC):** The user stopped new
experiments and requested safe Vast.ai shutdown. The last live DrQ learner was
interrupted after its step-16,384 checkpoint; its partial ledger ends at step
21,037 and is not an exact resumable result. TD-MPC2 v1 remains partial and v2
has zero resets. No training, evaluation, confirmation/blind or official action
should be launched during migration. See the detailed
[instance handoff](instance-migration-handoff-2026-09-27.md), including the
Git-external transfer inventory and deletion gate. A Git push alone is not
proof this instance can be removed.

## Current Position

**DrQ-v2 is CLOSED by user direction (2026-09-29).** Do not initiate further
DrQ-v2 research, training, evaluation, speed/completion variants or official
actions. DrQ rows below and later historical snapshots record past work, not
an active queue or a candidate promotion. Preserve the original actor, local
ZIP and all negative-result evidence; see the
[closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line).

| Area | Current status |
|---|---|
| DrQ-v2 speed-only follow-up | **CLOSED with the entire DrQ-v2 line by user direction.** A frozen pad-4 seed1 brake-only inference intervention failed to preserve completed roads on 32 repeatedly reused development cells: track1 5/16 control versus 3/16 treatment (three lost finishes), track2 1/8 versus 0/8 (one lost), track3 0/8 in both arms. Only two mutually completed track1 laps shortened by 1.74/1.28 seconds; no cross-track speed result exists. The 128-episode CPU21 run passed two-reload parity and action-trace hashes. Do not deploy or pursue another DrQ speed hypothesis. The original actor/local ZIP and evidence remain unchanged. See the [`frozen result`](../../experiments/drqv2-speed-reused-development-v1-result.json) and [closure decision](../decisions/INDEX.md#close-the-drq-v2-research-line). |
| Validated internal baseline | Native DrQ-v2 control with augmentation pad 4. Two unique control actors, one per training seed, were evaluated on two fresh internal confirmation cohorts; the recorded control outcomes range from 4 to 7 finishes in 32 cells. |
| Active research direction | DreamerV3 is CLOSED. Other algorithm work is listed separately below; there is no active Dreamer experiment or implementation task. |
| Current blocker | None for DreamerV3: the research line is closed under the current design and budget. This is not a theoretical impossibility judgment; further progress would require design-level rework. |
| DrQ-v2 narrow tuning | Closed. The predeclared steering-logit L2 and pad=1 follow-ups both regressed; do not launch a third narrow tuning axis. |
| DrQ-v2 teacher-replay plan | The isolated r3 collection completed its fixed 16,384 decisions/source cap, but one source had only 3/4 required finished geometries. A3 is inconclusive; no paired learner training, screen, confirmation, or blind evaluation occurred. Preserve both datasets as consumed and do not extend the cap or substitute a source. See `docs/experiments/INDEX.md`. |
| DrQ-v2 training-only geometry study | Completed: consumed-road failure analysis, blind-safe seed audit, 120 distinct TRAIN roads + 16 separate TRAIN-DIAGNOSTIC roads in six measured families, static/finish-logic sanity, and 272 sealed frozen actor diagnostics. The same 136 training-only roads gave 10 both-actor finishes, 74 provisional boundary, 50 difficult-with-progress, 2 unresolved and zero selected malformed geometries. No new learner training or held-out/blind action. See `docs/experiments/drqv2-geometry-augmentation-v1.md`. |
| DrQ-v2 geometry-mix fine-tuning | All six frozen r6 online-only runs completed 32,768 additional decisions and 22,768 updates each, sampling TRAIN only. On the same 16 previously designated TRAIN-DIAGNOSTIC roads, canonical repeat-0 finishes were uniform 1/32, failure-weighted 4/32, and easy-retention 3/32 across the two learner seeds; the unchanged source actors finished 11/32. This is descriptive, unranked development evidence, not fresh generalization, confirmation, blind, or official HAIC performance; no weights were selected or promoted. r4/r5 partial attempts remain preserved and are not resumed or counted. See the r6 [`active plan`](../plans/active/drqv2-geometry-mix-plan.md), [`protocol`](../../experiments/drqv2-geometry-mix-v1-r6.json), six run `result.json` files, and [`diagnostic manifest`](../../runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/manifest.json). |
| DrQ-v2 r7 source retention | Twelve frozen 32:32 source/online TRAIN runs completed the r6 budget with and without one pre-fixed actor preservation loss. On the same 16 reused TRAIN-DIAGNOSTIC roads, r7a kept 1/3/1 of the eleven source-success actor/road cells and gained 2/7/4; r7b kept 4/1/5 and gained 5/7/5 (uniform/failure-weighted/easy-retention). All six arms lost at least six old successes; the predeclared >=9 kept and >=2 gained retention contract failed despite higher total finishes. No model promotion or fresh held-out/confirmation/blind/official action. See the [`r7 evidence report`](../experiments/drqv2-retention-r7.md), [`result`](../../experiments/drqv2-retention-r7-result.json), and [`384-episode trace manifest`](../../runs/20260926-drqv2-retention-r7/train-diagnostic/manifest.json). |
| DrQ-v2 final-source retention and migration stop | Offline longitudinal parity passed 5,998 archived source actions/666 prior state hashes; full old replay scan found far fewer feature-near middle/corner/late/finish-approach than early states, without identifying a causal mechanism. Both final-source-policy TRAIN pools sealed 100,000 decisions each. **Five of six** matched 32:32, lambda=0.5 learners completed; seed1/easy-retention stopped for instance migration after a valid step-16,384 checkpoint, although the retained TRAIN ledger extends to step 21,037/gradient 11,037. There is no exact restart implementation, completed sixth result, all-six sample audit or final-source TRAIN-DIAGNOSTIC episode. The fixed >=9/11 retained AND >=2/21 gained gate remains **unevaluated**, not failed or passed. All DrQ work is paused for migration; no automatic new experiment or restart. Transfer Git-external source/control/final pools, five complete arms and the entire partial sixth directory. See the [detailed DrQ handoff](../experiments/drqv2-final-source-replay-v1.md) and [migration stop receipt](../../experiments/drqv2-final-source-replay-v1-migration-stop.json). |
| Independent TD-MPC2 pixel baseline | Paper/current-official-source faithful 5M-size-class world model, episodic termination and MPPI are isolated from DrQ/PPO; HAIC alone requires four-channel grayscale/64px and action conversion. The first frozen reused-TRAIN pilot recorded 10,061 decisions and 10,000 pretraining plus 60 later updates but failed at a next-episode planner reset because an inference tensor was zeroed outside inference mode. Its 29 completed episodes had no finishes and no complete planned episode; within-replay model fit cannot establish driving benefit. The regression is repaired and tested. Separate v2 full source/resource preflight passed once with zero resets, but shared disk had only 5.5GiB free against its unchanged >=5GiB floor and continued peer writes, so v2 has NOT run. No TD-MPC2 candidate, held-out or official action. See the [`active plan`](../plans/active/tdmpc2-pixel-online-baseline.md), [`v1 failure`](../../experiments/tdmpc2-reused-train-pilot-v1-failure.json), and [`v2 exposure`](../../experiments/tdmpc2-reused-train-pilot-v2-exposure.json). |
| User-directed pixel RLPD pilot | V2 is closed stop/hold. The separate long-horizon v1 passed screen, strict confirmation, and blind; RLPD seed 11 at 131,072 steps remains an internal candidate. Entropy V1–V3 aborted before interaction. V4 consumed fresh teacher/student runs but stopped before screen on source-hash drift; its held-outs are retired. V5 used fresh prior data and four 131,072-step target-arm runs; its screen had 29/192 canonical finishes and passed all target/seed gates. Strict confirmations were author-target 17/32 and 7/32 versus +1.5 target 6/32 and 6/32; all were eligible and operationally clean. Author-target passed the paired dominance gate, and its selected seed-50 actor finished 7/24 on the internal blind (mean progress 0.657). This remains two-seed internal evidence, not an official result. See `docs/experiments/INDEX.md`. |
| Pixel RLPD completion-first G0 | Complete TRAIN-only observational diagnosis: two frozen actors on 12 shared geometries, 24 episodes and 10,049 decisions; each finished 3/12 and failed 9/12. Contact and centerline-distance events also occurred on successful controls; the 20-decision low-directed-motion event appeared only on nonfinishes in this cohort. No causal remedy or learner experiment is selected. All cells are consumed; see `experiments/rlpd-g0-completion-v1-result.json`. |
| Best single-model designation | No official or competition-confirmed model is designated. Pixel RLPD seed 11 is an internal, limited-geometry blind-tested candidate only. See `docs/results/MODEL_STATUS.md`. |
| Official external state | This repository has no committed official server submission identifier, public-track result, or model-confirmation receipt. A local package archive is not proof of an official submission. |

## DreamerV3 Research Line: CLOSED

Decision as of 2026-09-27: **additional DreamerV3 work is not worth pursuing under
the current design and budget.** The local implementation, available data, and
experiment contract have not solved long-horizon prior dynamics; further progress
would require design-level rework. This is not a claim that DreamerV3 is
theoretically impossible.

- Posterior reconstruction and short, logged-action predictions can look
  reasonable, but multi-step free prior rollout is unstable; 32-decision prediction
  error worsened beyond a simple repeat baseline. Posterior reconstruction is not
  open-loop prediction.
- Static random B1 v1-v9, random/teacher data, two DrQ-source replay, and H8
  prior-image auxiliary work did not establish stable improvement across two
  learner seeds and both scored action-source strata. The strict H8
  paired-improvement-and-repeat gate failed.
- The evidence does not support explaining the failure solely by unused actions,
  an obvious off-by-one alignment bug, or one missing prior-image auxiliary loss:
  action-input sensitivity was observed, the tested real transition trace found no
  target offset, and prior-image/H8 treatments were actually tried but failed
  their declared gates.
- Residual causes remain hypotheses, not proven mechanisms: limited state/action
  coverage; possible redundancy between the four-frame stack and RSSM memory or a
  reconstruction shortcut; the offline-world-model-first structure; and long-rollout
  dynamics drift.
- Stop local searches over loss weights, horizons, update counts, and similar
  parameters. No further experiment, training, collection, evaluation, promotion,
  official submission, or protected evaluation-cell use is part of this line.

The latest available multi-source+H8 checkpoints are world-model diagnostics at
[`seed 0`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-0/world-model-checkpoint.pt)
and
[`seed 1`](../../runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1/learner-1/world-model-checkpoint.pt).
Both receipts report `actor_trained=false` and `promotion_eligible=false`; neither
is a performance actor, and no Dreamer checkpoint is selected as a best model.
The final evidence/gate index is [`docs/experiments/INDEX.md`](../experiments/INDEX.md).
The former active plan is now a [closed status pointer](../plans/active/dreamerv3-recovery-strategy.md)
to its [archive](../plans/archived/dreamerv3-recovery-strategy-2026-09-27.md).
Any future contingency is only the
[`DEFERRED / LAST-RESORT ONLY` revival plan](../plans/dreamerv3-revival-plan.md).

## Active Work

The user-authorized DrQ-v2 geometry-mix study is a separate matched experiment;
it neither reopens teacher-replay r1-r3 nor changes DreamerV3's closed status.
Its three fixed family mixtures and diagnostic-only protocol are frozen at
[`drqv2-geometry-mix-v1-r6`](../../experiments/drqv2-geometry-mix-v1-r6.json).
Mix r1-r3 were superseded before any interaction. R4 and r5 each had one partial
learner-0 uniform TRAIN run (16,384 decisions, 6,384 updates) and stopped before
full checkpoint: r4 had an undefined catalog-index helper, r5 detected mutable
documentation in the checkpoint source-hash list. Both were mid-episode and
cannot be resumed. R6 restarts from the frozen source checkpoint with an
explicit TRAIN pool, new RNG schedule and executable-only runtime source hashes.
Partial TRAIN exposure and failure receipts are preserved but are not candidates
or evaluations.

R6 completed with all six run results and both step-16,384 and step-32,768
checkpoint/replay-sample trace hashes preserved under
[`runs/20260925-drqv2-geometry-mix-v1-r6`](../../runs/20260925-drqv2-geometry-mix-v1-r6/);
the active plan indexes the final checkpoint and replay-trace hashes. The frozen
CPU21 development diagnostic used only the 16 catalog TRAIN-DIAGNOSTIC roads:
256 episodes across eight actor roles and two repeats, with all 128 repeat pairs
deterministically identical. Its manifest binds every trace hash and the family
summary is linked from the experiment index. This is not a fresh holdout,
confirmation, blind, or official evaluation, and model updates are not themselves
evidence of generalization or a promotion decision.

The subsequent [r6 regression diagnosis](../experiments/drqv2-geometry-mix-r6-regression.md)
paired all 32 source-actor/road cells with each variant on those **reused**
TRAIN-DIAGNOSTIC roads: uniform lost 11 source wins and gained 1 new win;
failure-weighted lost 10/gained 3/kept 1; easy-retention lost 10/gained 2/kept
1. A bounded action-replay of the 11 archived source-success episodes (5,998
decisions) matched the old traces exactly and measured deterministic actor,
twin-critic and encoder outputs on identical source observations without a new
policy rollout or update. Early actor/encoder drift is directly observed; Q1
reordering on some cells and incomplete source-state replay retention remain
contributors to test, not proven causes. That diagnosis alone did not authorize
training, promotion or a new evaluation partition.

The subsequent separately authorized [r7 retention study](../experiments/drqv2-retention-r7.md)
kept r6's geometry mixtures and exact 32,768-decision/22,768-update budget.
An offline hybrid probe found early action differences from both encoder and
actor-head changes; a TRAIN-only gradient probe fixed one source-action
preservation weight. The actual source/online sample traces verified 32:32 for
every minibatch, and all twelve runs and 384 reused-development episodes passed
independent artifact, repeat and paired-cell checks. r7a's replay-only relative
to r7b's added preservation term did **not** preserve most of the eleven old
successful actor/road cells under any mixture. Failure-weighted r7a's 10/32
total finishes in particular comprise just **three kept plus seven gained**;
easy-retention r7b's 10/32 comprise **five kept plus five gained**. Both old
success retention and adaptation remain separate unsolved questions. Reused
TRAIN-DIAGNOSTIC cells are consumed development feedback, not a fresh holdout;
the [frozen result](../../experiments/drqv2-retention-r7-result.json) does not
promote a model or authorize protected/official action.

### Historical DreamerV3 Evidence (CLOSED; No Follow-up)

The former active plan is preserved in the
[`archive`](../plans/archived/dreamerv3-recovery-strategy-2026-09-27.md).
The details below are a dated record of completed diagnostics and failed gates,
not open prerequisites or authorization for another study. The initial fidelity
work and nine B1 static-random-data trials
are complete; all failed their frozen gates. The
[`B1 failure diagnosis`](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json)
separates implementation errors, sparse training signal, and gate design defects.
No completion or official score claim follows from its offline diagnostics.

Dreamer P0/D0 has synthetic-tested actor/return contracts, reset-origin and
completed short-episode replay, and a retrospective scored-window BCE reference.
The regression suite covering the P1 teacher collector, sealed-dataset replay
bridge, fail-closed cross-lane ID inspector and adjacent contracts passed 184
synthetic/mock tests, not driving evidence.
Fractional short-episode sampling now skips unavailable windows; the collector
recomputes pinned audit evidence instead of trusting a self-declared pass receipt.
The inspector parses r6's geometry-sampler RNG separately from road IDs but
cannot certify exhaustive historical allocations and never issues a pass.
B1 remains no-go; no repaired Dreamer driving policy or P1/P1b dataset exists.
A complete independently verifiable seed inventory, fresh immutable protocol,
random-arm collector and offline learner runner remain necessary for P1/P1b.

A separate, non-promoting [reused-TRAIN engineering diagnostic](../../experiments/dreamerv3-reused-train-diagnostic-v1.json)
has since completed on four previously allocated r6 TRAIN roads. Its
[random](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/random/collection-result.json)
and [DrQ-source](../../runs/20260926-dreamerv3-reused-train-diagnostic-v1/collection/teacher/collection-result.json)
collections used 1,306 and 2,548 decisions respectively, with zero versus one
local finish in four complete episodes each. Under a separate
[offline protocol](../../experiments/dreamerv3-reused-train-offline-v1.json),
two learner seeds per source completed 64 model-only updates each, without new
environment steps or actor/critic training; the four run receipts are linked in
the [experiment index](../experiments/INDEX.md). Differently sized reused data
and in-distribution training losses do not establish a matched prediction gain
or driving improvement. Fresh P1/P1b remains blocked, with no new Dreamer actor,
held-out result, or official result.

The separately frozen [reused-TRAIN development and update-budget result](../../experiments/dreamerv3-reused-train-development-v1-result.json)
scored two world-model seeds per source on the same four *training-excluded* but
previously allocated r6 TRAIN roads in each action-source stratum. Both new
collections finished 0/4 roads. At 64 model-only updates, all models missed the
shifted-repeat image baseline on teacher-action development; increasing only the
model update count to 256 worsened both teacher-trained seeds' image error on
both reused development strata. Each stratum had four independent terminal
events among 256 scored decisions and no finished development geometry;
average terminal BCE is not evidence of finish-event discrimination. The
second score reused development already inspected after 64 updates, so it is
consumed tuning feedback, not a fresh holdout or P1b result. The B1/P1 gates,
no-trained-student-actor status and prohibition on official claims remain.

Read-only [action-input](../../experiments/dreamerv3-reused-train-action-input-v1-result.json),
[eight-prior sampling](../../experiments/dreamerv3-reused-train-sampling-v1-result.json),
and [posterior/prior](../../experiments/dreamerv3-reused-posterior-gap-v1-result.json)
diagnostics on those SAME consumed road episodes show that teacher-data model
predictions respond to changed future action inputs, but not consistently in
the correct direction against logged targets. Averaging eight prior paths does
not eliminate their 9-32-step image error above frozen-anchor repeat.
Target-conditioned posterior reconstruction is not prediction; one-step prior
is near the true-previous-frame repeat, while 32-step free prior worsens
despite lower raw same-transition KL. That motivated the subsequent bounded
multi-step prior-image treatment reported below; its 256 base plus 64
additional optimizer steps cannot by itself isolate a loss-shape benefit.
No Dreamer actor/policy result, fresh evaluation or official score exists.

The separate [prior-image auxiliary training and score](../../experiments/dreamerv3-reused-train-prior-image-score-v1-result.json)
completed 256 ordinary plus 64 additional world-model optimizer steps per
arm/seed without actor or environment updates. Both teacher-data seeds improved
paired image MSE on both *consumed* development action-source strata, but
teacher-action values **0.015952/0.016135** remained above the predeclared
frozen-anchor repeat **0.014884**. The local teacher-data gate therefore failed;
the statically proposed third-per-family four roads were not opened, and the
extra optimizer steps prevent a pure loss-shape conclusion. A distinct
[12-road source0 diversity collection](../../experiments/dreamerv3-reused-train-diversity-v1-result.json)
used other, previously allocated r6 TRAIN cells: random finished 0/12 and
source0 finished 2/12 roads in two shape families, below its separately fixed
>=3-road teacher support gate. Both sealed archives were preserved but **no
Dreamer learner was released by the source0-only protocol**. A prior DrQ
catalog summary already has one source1 attempt on each same TRAIN road. The separately frozen
[source1 replication](../../experiments/dreamerv3-reused-train-source1-v1-result.json)
completed 5,192 decisions on all 12 roads, finishing 2/12: one road overlaps
source0's two finishes, so the *source-union* is three distinct roads in three
shape families. Its descriptive complement gate permitted **design only** of a
new multi-source study; source0's failed single-source gate stays failed.
Historical source1 also measured these same roads, so this is replication/variation,
not unseen-road support. The original Dreamer offline bridge accepts only one
actor ID/hash and its replay loses tags; the separate
[mixed-source learner](../../experiments/dreamerv3-reused-train-multisource-v1-result.json)
validated each archive independently, bound 24 episodes/10,612 decisions on
only 12 road IDs to a per-seed identical hashed replay lineage and completed
256 model-only updates for each of two seeds. Its predeclared
[four-road image score](../../experiments/dreamerv3-reused-train-multisource-score-v1-result.json)
failed: both seeds missed frozen-anchor repeat on random-action development,
and source0-action development had only seed0's tiny below-repeat value.
Development source/outcome receipts were visible before the scorer source was
finalized, so this training-excluded r6 TRAIN set is consumed iterative tuning,
not a blind/fresh validation target. No actor/critic was trained, no student
driving or official score exists. A separate multi-source data plus H8
prior-image auxiliary interaction was subsequently trained and scored as a
separate model-only study; its predeclared two-seed/two-stratum image gate
failed. The real fresh P1 cross-lane seed audit remains non-passing.

The isolated [mixed-source+H8 model-only training result](../../experiments/dreamerv3-reused-train-multisource-prior-v1-result.json)
used the original source0/source1 sealed 10,612-decision replay and identical
per-seed SHA-bound lineage. Both fixed learner seeds completed 256 ordinary
plus 64 additional prior-image world-model optimizer steps, with zero new
environment steps and unchanged actor/critic/target and their optimizers.
After the owner freed memory, the independently rechecked raw cgroup headroom
met the unchanged 12 GiB floor plus projected peak; no OOM kill was added.
Auxiliary TRAIN frame loss changed in opposite directions across seeds, so this
is pipeline/training evidence, **not** a scored prediction gain. The separately
pinned [read-only comparison](../../experiments/dreamerv3-reused-train-multisource-prior-score-v1-result.json)
against the earlier pure-256 models on the SAME already-consumed four-road
source0/random development cells failed: seed0 worsened on both source strata;
seed1 improved versus pure256 on both but random-action image MSE `0.001509`
remained above frozen-anchor repeat `0.001494`. The rule required BOTH seeds
to improve and beat repeat on BOTH strata. The first metadata-only score
preflight stopped on a 1-MiB catalog cap before ZIP decoding; a catalog-only
bounded correction, 33 scorer tests and new hashes preceded the first actual
score. Additional 64 optimizer steps preclude a compute-matched loss-shape
claim, and the development material was previously consumed iterative tuning.
Previously allocated seventh-per-family r6 TRAIN roads remain unopened,
the fresh P1 audit fails closed, and no Dreamer student actor or official result
exists. The earlier [memory hold](../../talk/messages/20260926T181924Z-k9r4-dreamer-resource-hold-final-check.md)
was resolved for this study after the owner's resource change; do not relax
limits for any future experiment.

For the blocked fresh P1 path, the Dreamer collector now accepts the **exact**
historical G0 `*-seed-audit.json` only as a SHA-pinned prior audit, not as a
protocol or a generic foreign seed audit. Synthetic manifest/hash-drift tests
and the full 367-test Dreamer regression suite passed. The auditor now guards
candidate-bearing known typed road/cell aliases, ledger road IDs and untyped
`start/end` and `start/count` ranges, without failing disjoint synthetic cells.
These fixes do **not** certify candidate freshness: the actual Dreamer P1 audit
still returns `passed=False`; other unknown field/range encodings, TRAIN
start/partial/claim records, own-protocol bootstrap and re-audit before each
reset remain unproven. No fresh P1 road has been selected or driven.

## Completed Teacher-Replay Gate

The user-directed DrQ-v2 teacher-replay protocol is frozen at
[`r3`](../../experiments/drqv2-teacher-replay-v1-r3.json). The r3 datasets passed
source and integrity checks, but one actor missed the preregistered coverage
hurdle; therefore there is no learner/evaluation result or candidate promotion.
The original r1 wrapper-seam attempt and r2 trainer-preflight attempt are retained
as zero-learning aborts, not model failures. Do not reuse r2/r3 teacher datasets,
open the reserved evaluation pools, or infer performance from collection metrics.
The stop receipt and per-source evidence are indexed in
[`docs/experiments/INDEX.md`](../experiments/INDEX.md).

## Completed Pixel RLPD Pilot Gate

The user-directed pixel-RLPD v2 protocol and result are recorded at
[`protocol`](../../experiments/pixel-rlpd-offpolicy-pilot-v2.json) and
[`result`](../../experiments/pixel-rlpd-offpolicy-pilot-v2-result.json). The
teacher dataset met its fixed 8,192-decision/two-finish coverage gate, and each of
the four SAC/RLPD student runs completed 16,384 decisions. The custom 12-cell per-
actor screen (two repeats per cell) produced one canonical finish in 96 canonical
episodes: the selected
RLPD seed-1 8,192-step actor finished once; selected seed 0 finished zero times.
The pilot therefore fails its preregistered two-seed promotion gate. All eight
screened actors passed CPU reload, determinism, and operational eligibility, but
this does not offset the completion gate. Do not open v2's reserved conditional
full, confirmation, or blind cells. The user requested a new feedback/retraining
iteration; any continuation must be a separate protocol with fresh code/data/screen
and held-out allocations, not an extension of v2.

## Fresh RLPD Follow-up (Completed)

The separate hypothesis protocol is
[`pixel-rlpd-long-horizon-followup-v1`](../../experiments/pixel-rlpd-long-horizon-followup-v1.json)
(SHA-256 `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329`). Its
seed and both teacher-ledger audits were completed before interaction, with no
known exact recorded overlap; historical schedules remain incomplete. The fresh
16,384-decision teacher collection had 6 distinct-geometry finishes; the four
matched 131,072-decision student runs completed. Both screen gates passed on 24
canonical cells per actor: seed 10 selected RLPD/SAC finished 7/24 and 3/24; seed
11 finished 12/24 and 3/24. Both strict confirmations passed (3/32 vs 2/32, and
12/32 vs 7/32). The first confirmation command failed before validation/workers and
consumed zero cells; actor-specific, hash-linked projections of the immutable
screen rows then passed `previous_evaluation_metadata()` without replaying screen
cells. The preselected RLPD seed-11, 131,072-step finalist passed the 24-cell blind
with 9 canonical finishes (mean progress 0.679). All CPU reload/determinism/resource
checks passed. This remains internal CarRacing evidence, not an official score.
Full detail is at
[`pixel-rlpd-long-horizon-followup-v1-result`](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json).

## Next RLPD Question

The frozen RLPD temperature convention remains `target_entropy=-1.5`; it is not
proven optimal. Entropy V1–V3 aborted before environment interaction and their
allocations are retired. V4 passed initial protocol/runtime/geometry checks, collected
fresh prior data and completed four matched runs, but the pre-screen source recheck
detected a changed `generalization-policy.md`; V4 evaluation never opened. V5 screen
gate passed all four target/seed minima with 29 finishes in 192 canonical episodes.
Fresh strict confirmations were 17/32 and 7/32 for the author target versus 6/32
and 6/32 for the +1.5 target; all four were eligible, deterministic, CPU-reload
identical, and had zero operational failures. Author-target passed the paired
dominance gate, and the frozen tie-break selected its seed-50 actor for the one
24-cell internal blind, which finished 7 episodes (mean progress 0.657). This
two-seed CarRacing ablation is internal proxy evidence, not official performance or
model confirmation. See the
[`V5 result`](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json).

The separately authorized [RLPD completion-first G0](../../experiments/rlpd-g0-completion-v1-result.json)
closed its fixed TRAIN-only diagnostic budget without learner updates or held-out
evaluation. Its 12 geometry clusters produced 24 paired observational episodes,
not a matched training-treatment comparison. Read-only trace review found no
unique initiating cause: two qualified nonfinishes were road-near and oppositely
headed long before 95% visited-tile progress, while an early contact/centerline
warning can also precede a real finish. The [fixed-window offline extraction](../../experiments/rlpd-g0-fixed-windows-v1-result.json)
sealed 144 slots across those same 24 consumed TRAIN traces, with 101 anchored
windows and 43 explicit missing anchors; all six successful-parent controls were
retained, but no manual or causal labels exist. A separate
[pixel-only motion score](../../experiments/rlpd-g0-pixel-motion-v1-result.json)
covered all 10,049 decisions; 74 retrospectively satisfy the contact-stall
telemetry definition and all six finishes remain in its control distribution.
Those labels occupy four failed episodes on three road geometries; their score
range overlaps finished-control decisions. The [gate decision](../../experiments/rlpd-g0-pixel-motion-v1-decision.json)
selects no threshold or policy. The [frozen-encoder representation probe](../../experiments/rlpd-visual-representation-probe-v2-result.json)
completed 256 CPU head-only updates with the actor unchanged, but all 188
diagnostic true positives came from two correlated episodes on ONE geometry and
none of its diagnostic roads supplied a finished parent. It can at most decode
the existing visual speed HUD on reused TRAIN roads, not justify an intervention.
A new source-hashed geometry-level positive and successful-parent coverage gate,
followed by separate exact-prefix parity/harm evidence, must precede G1. The
RLPD-specific parity comparator and original-G0 SHA binder now pass synthetic
state/action tests and file-only two-decision prefix checks on 24/24 consumed
TRAIN episodes, but **no real reset/replay or G1 branch** was executed; hidden
Box2D equality remains unproven. None of these observations proves recovery
data, value shaping, or memory is the causal fix. The r5 seed-audit erratum is
narrow and does not retroactively attest the malformed receipt.

The proposed next RLPD G1 coverage design fixes 24 new TRAIN geometries x two
unchanged actors, with V5 seed-50 source-primary only for its G0-informed
post-contact-stall feasibility question and seed-11 fully reported as comparator.
Pure synthetic coverage checks require all 48 actor-road slots and distinct
pixel-positive failure/finished-parent roads; they do not certify annotations
or road freshness. A separate G1 v2 seed auditor now checks candidate-specific
TRAIN collisions, consumption, reservations and exclusions; unrelated DrQ/Dreamer
JSON or ledger changes are provenance warnings, not collisions. The read-only
consumed-G0 control `--seed-start 4272000001` remains `BLOCKED` with 12 actual
consumed-road intersections, not a proposed G1 batch. Current r5 TRAIN ledger,
abort, supersession, metrics and trace bytes are independently checked without
validating the malformed historical receipt SHA or changing the G0 v1 audit.
The shared TRAIN claim registry re-audits under a lock if a future batch is
separately selected; legacy/other-lane allocation races remain a limitation.
No G1 seed batch, claim, frozen protocol or environment reset has been allocated.

Additional RLPD-only G1 preparation is synthetic: a blocked-by-default
collector skeleton checks future audit/claim/source identity and preserves all
48 attempted, censored or unrun slots; a separate image-review module seals
opaque pixel-only packets and restricted mappings before outcome joining.
Thirty focused synthetic tests pass. The real collector remains disabled
because a mid-episode four-core-hour stop cannot yet be enforced by the
source-pinned G0 `run_cell`; no pixel trigger, genuine blinded annotation or
positive/finished-parent G1 coverage has been established. Two conditional
candidate-relevant G1 auditor false-pass shapes (`{start,end}` seed ranges and
self-protocol exclusions) are awaiting peer-owner correction. No new G1
interaction or policy promotion is justified by these preparatory files.

## Competition Schedule And Access

The first mock in the current site-verified schedule has passed; the next event
listed is the second mock. The old hostname failed DNS resolution on 2026-09-25,
but the [replacement competition site](https://scholarships-hardwood-headers-influenced.trycloudflare.com/)
was reachable on 2026-09-26 and its bundle still published the recorded dates.
The owner reports a higher daily submission quota than the bundle's fixed upload
text; verify the authenticated effective limit before any approved upload. Dates,
timezones, the site-access check, quota provenance, and the unverified
confirmation-cutoff report are recorded only in
[`docs/competition/info.md`](../competition/info.md). Recheck the official
schedule before any external action.

## Read Next

- Current candidate provenance and confirmation status:
  [`docs/results/MODEL_STATUS.md`](../results/MODEL_STATUS.md)
- Evaluation and partition discipline:
  [`docs/evaluation/protocol.md`](../evaluation/protocol.md) and
  [`docs/evaluation/generalization-policy.md`](../evaluation/generalization-policy.md)
- Durable evidence and prior decisions:
  [`docs/experiments/INDEX.md`](../experiments/INDEX.md) and
  [`docs/decisions/INDEX.md`](../decisions/INDEX.md)
- External actions and official rules:
  [`docs/competition/`](../competition/)
- Multi-agent discussion protocol and recent working messages:
  [`talk/README.md`](../../talk/README.md)

## Evidence Boundary

The DrQ-v2 and DreamerV3 statements above summarize frozen experiment records, not
new measurements. Consult the linked JSON protocol/result artifacts before making a
new comparison or changing a candidate status.
