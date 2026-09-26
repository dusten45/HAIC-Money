# DreamerV3 Completion-First Research Proposal

Date: 2026-09-26. **Originally research and design only.** Subsequent,
separately authorized P0 contract repairs and synthetic tests are recorded in
the [active recovery plan](active/dreamerv3-recovery-strategy.md); they do not
establish driving performance or authorize a new P1/P1b study. This report
remains a research proposal, not a frozen execution protocol; it does not
replace the failed B1 gates, reopen partitions, or designate a model.

## Recommendation

**Revised after peer-direction review on 2026-09-26:** the immediate priority is
to distinguish contract defects, missing task support, and an existing policy's
localized driving failures **before choosing a treatment**. The earlier opening
recommended a completion-grounded combination too strongly relative to its evidence.

1. Revalidate source/actor identity, actor-objective and data contracts. Before
   interpreting a future packaged deployment, separately verify its identity and
   local-versus-package action parity; a generic smoke test is not that comparison.
2. If an eligible trained policy exists, characterize its failures **and successes**
   before selecting an intervention. If it does not, diagnose support/prediction
   limitations instead. Repaired Dreamer has no trained driving policy in the B1
   record, unlike the partly successful RLPD candidate.
3. For Dreamer's verified support gap, retain the conditional **data-only** learner
   comparison after contract checks and separate authorization: random versus
   useful teacher whole episodes, retaining failures as well as successes. Do not
   require recovery branches or a finish objective for this experiment.
4. When a trained student and localized failure evidence exist, test one bounded
   intervention, then the entire pixel-realizable policy that selects, executes
   and ends it. Count rescued failures and destroyed baseline finishes separately.
5. Add task-semantic supervision, finish-value optimization, or backward curriculum
   only when a discriminating diagnosis supports that particular hypothesis.
   Preserve raw reward/environment, isolate each treatment, and require actual
   standalone/composite full-reset completion before promotion.

The distinctive proposal is the combination of **where useful data enter the
pipeline, actual finish semantics, and action-conditioned recovery evidence**.
Teacher replay, auxiliary heads, reverse curricula, and recovery policies are
existing ideas; their integration here is a project-specific hypothesis, not a
claim of a new published algorithm.

No method can honestly guarantee completion on arbitrary unseen tracks from
these records. This report validates mechanisms and identifies falsifiable
experiments; it does **not** demonstrate improved Dreamer completion.

## Peer Direction Review

The previous report did **not** include the two later `n8w3` notes. This revision
reads them, their parent thread, and the separate RLPD report, then checks their
claims rather than treating peer advice as an instruction or approval:

- [Failure-first completion gate, 06:34:21 UTC](../../talk/messages/20260926T063421Z-n8w3-completion-failure-gate.md).
- [Recovery selection boundary, 06:37:49 UTC](../../talk/messages/20260926T063749Z-n8w3-recovery-selection-boundary.md).
- [RLPD research result, 06:38:44 UTC](../../talk/messages/20260926T063844Z-r4f7-rlpd-completion-research-result.md)
  and its [report](rlpd-completion-first-research-2026-09-26.md).

Their empirical subject is primarily RLPD/DrQ. They mention Dreamer to preserve
its independent lane and execution stop; they do not diagnose the repaired
Dreamer actor or authorize a new experiment.

| Peer point or inference to check | Comparison with the original report | Decision |
|---|---|---|
| Diagnose where useful progress is lost before choosing a treatment | Our opening elevated a completion-value/representation package before establishing the dominant policy failure. | **Accept and change priority.** Add D0 below and make optional treatments conditional. |
| Oracle best-of-branches is not a policy | We already required past-observable anchors and named continuations, but did not require the complete selector, actions, termination and handback as one explicit gate. | **Accept and strengthen.** Add the complete-controller gate, including Dreamer recurrent carry. |
| Report rescues and formerly successful laps lost | Our aggregate non-regression requirement could hide harmful switches. | **Accept and strengthen.** Require the paired outcome table and a preregistered lost-success limit. |
| Verify the actual candidate package, not only generic smoke/reload checks | Our small in-memory export test was explicitly limited, but the future package handoff was not a separate named comparison. | **Accept at deployment handoff.** Do not make a nonexistent released ZIP a prerequisite for model-only diagnosis. |
| Blind results, clock resets, hidden labels and global braking are unsafe shortcuts | These cautions were already explicit in the report. | **Retain as agreement**, not new discoveries from this revision. |
| Possible extrapolation, not a peer claim: recovery-first is the right next learner experiment for Dreamer | That conclusion does not follow from RLPD's failure distribution; B1 did not train a repaired Dreamer policy. | **Do not transfer blindly.** Keep the conditional Dreamer data-only comparison and defer recovery control until a relevant student exists. |
| Reassessment of our proposal: a finish objective or extra memory is the established remedy | Neither our structural argument nor the peer aggregates establish this treatment effect. | **Defer.** Keep both as hypotheses with separate discriminating gates. |

### Evidence Check

Read-only recounts of consumed **nonblind** receipts verified V5 author-target
confirmation: 64 canonical episodes, 24 finishes, 40 failures, of which 35 were
`off_track` and five crash. The all-actor V5 screen had 192 canonical episodes,
29 finishes and 163 failures; only four failed at progress >=0.95. That last
`4/163` pools eight actors across two target arms and two checkpoints. It is not
the selected actor's rate: selected author seed 50 had 3/16 such screen failures,
seed 51 had 0/17, and author confirmations together had 1/40. These aggregates
argue against treating missed late crossing as a blanket explanation, not against
every possible long-horizon benefit of a finish critic.

The already-published long-horizon result confirms 9/24 blind finishes and
14 `off_track` plus one crash among the 15 nonfinishes. No blind episode/trajectory
file was inspected in this follow-up. Its 24 cells are eight geometries with
three track/obstacle variants, not 24 independent roads. None of these fractions
is a Dreamer treatment effect or a reason to tune on those cells.

Recount inputs were the consumed V5 screen and author seed-50/51 confirmation
episode logs in the corresponding local `evaluations/` run directories. These
large raw logs are retained locally, not versioned with this report; the
[V5 result](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json)
and [screen report](../../evaluations/20260925T162014319049Z_pixel-rlpd-entropy-target-ablation-v5-screen/report.md)
contain the published aggregates.
Their SHA-256 values were respectively
`ceb22013b235207184ed05ea84d3b6032e15178f960b4b77093175bd6d0fe383`,
`839d607abb3dc81a6f5c819da6166ea9b1c02375af06d12be91e28bbd4831c0c`,
and `cdfd7a3dfe0c7d9211c79418cd90a025956a2d3e7974971c93feb8beae91091a`.
Only canonical repeat 0 contributed to counts; the repeat pairs agreed.
The blind statement uses only the [already-published aggregate](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json)
and [frozen partition metadata](../../experiments/pixel-rlpd-long-horizon-followup-v1.json).

Local source independently confirms the event distinctions in `env_wrapper.py:70-99`
and `core/finish_line.py:41-95`. The package distinction also checks out:
`package_submission.py:390-450` tests shape, finiteness and reset consistency on
zero images but has no local-candidate action comparator;
`evaluate_policy.py:802-891` compares repeat signatures, not independently the
frozen local candidate against its extracted ZIP on shared observations.
Before a future deployment comparison, freeze actor/source/ZIP identity and use
the same permitted observation sequence, reset/action contract and declared
numeric tolerance in the intended CPU runtime. No package was built or tested
in this follow-up, and no actual deployment mismatch is claimed.

Failure typing is observational: opening-turn loss of progress, road-bound
stall/loop, obstacle-contact recovery failure, missed qualified crossing, or
unknown/mixed. It is not a causal diagnosis of excess speed, insufficient memory,
or the wrong objective. A matched branch can demonstrate a local intervention
effect without uniquely identifying its mechanism.

### Revised Decision Order

```text
Identity and contract checks
  -> D0: eligible trained policy exists?
     No: audit task support/prediction; consider P1/P1b data-only study.
     Yes: classify its TRAIN-side failures AND successes.
       -> choose one targeted diagnostic, including successful-parent cases.
       -> useful branches? Test a complete causal pixel/history-based policy.
       -> compare full-reset rescues AND lost successes on separate geometry.
  -> objective, semantic memory or curriculum only for a supported next question.
```

This is a research-priority correction, not a new runtime gate already passed.
Existing B1 stops and authorization/partition boundaries remain unchanged.
Follow-up verification used two independent read-only checks, source inspection
and the illustrative rescue/harm arithmetic below. The original 14 unit tests
were not rerun for this documentation revision; no implementation, model update,
environment interaction, package build or official action occurred.

## Audited Starting Point

The codebase is a local, compact PyTorch DreamerV3-derived implementation, not an
unmodified upstream release. The old failed policy and current repaired learner
must not be treated as the same model.
The current path uses four 84x84 image channels, a 512-dimensional embedding,
256 deterministic hidden units and 16x16 categorical latents; default learning
windows are 32 decisions with 8 burn-in decisions and 15 imagined decisions.
No post-repair driving policy was trained or selected by the nine B1 studies.

| Evidence | Observation | What it does not establish |
|---|---|---|
| Historical policy pilot | Steering-saturated, zero-finish failure before the subsequent repairs. The frequently quoted progress `0.019383949` and steering saturation `0.953982120` correspond to the 43 post-warmup-end-labeled episodes through decision 5,672, not the final 10k log. | Failure of every Dreamer implementation; a matched random-policy comparison. |
| B1 v1-v9 | Nine studies, two learner seeds each; all 18 frozen world-model gates failed. These were random collection plus model-only updates, not 18 trained driving policies. | The current actor's completion rate or a causal test of actor repairs. |
| Recomputed B1 exposure | 198,656 training-decision exposures and 8,200 model-only updates in aggregate. v2-v9 repeat the same training episode summaries for each learner seed; 571 completed-episode exposures reduce to 71 distinct completed summaries over 37 unique geometry seed IDs. | 18 independent training-data populations or 198,656 distinct, diverse transitions. |
| Recomputed development support | Nine distinct development sets: 288 episodes, 98,113 decisions, 288 `off_track` endings, zero finishes, zero truncations, zero episodes with final progress at least 0.2. Each set was shared by two learner seeds. | Any validation of late-lap recovery, finish-gate recognition, or complete-lap action ranking. |
| v8/v9 training | 35/12,288 and 36/12,288 terminal transitions for the two seeds; completed random episodes never reached 20% progress. | A success/failure classifier with positive **finish** examples. |
| B1 terminal prediction | All 180 context/horizon rows across the 18 reports have terminal recall zero. v8's small image-baseline gain did not fix the terminal/reward gates. | That image reconstruction is entirely useless, or that changing only a classifier threshold would solve driving. |
| v9 weighting | `pos_weight=56` upweighted `continue=1`, the majority class, not terminal events. | A failed test of correctly weighted terminal-positive learning. |
| Longer context audit | The frozen matched-anchor v9 short/8/10/full-context analysis retained zero terminal recall for both seeds. | That more memory is sufficient to fix the existing representation, or that long memory can never help lap closure. |
| Other algorithms | RLPD long-horizon internal blind was 9/24; entropy V5 was 7/24. These are different candidates and cohorts. | A matched ranking against Dreamer, a 95%-reliable driver, or official HAIC results. |

Primary records: [feasibility](../../experiments/dreamerv3-feasibility-gate.json),
[B1 summary](../../experiments/dreamerv3-b1-iteration-summary-v1-v9.json),
[B1 diagnosis](../../experiments/dreamerv3-b1-failure-diagnosis-v1.json),
[seed-0 context audit](../../experiments/dreamerv3-b1-context-audit-v9-seed0.json),
[seed-1 context audit](../../experiments/dreamerv3-b1-context-audit-v9-seed1.json),
[RLPD long-horizon result](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json),
and [RLPD V5 result](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json).
New aggregates above were recomputed from their existing run reports, episode
logs, and development NPZ metadata/reward/flag arrays; no new episode was run.

The missing-support hypothesis is stronger than an episode-count explanation:

| Existing training source | Complete episodes | Finished geometries | Episodes ending at progress >= 0.2 | Median final progress |
|---|---|---|---|---|
| Dreamer v9 random, seeds 0 / 1 | 35 / 36 | 0 / 0 | 0 / 0 | 0.07358 / 0.07279 |
| Separate RLPD long-horizon prior | 36 | 6 | 34 | 0.86058 |
| Separate RLPD V5 prior | 35 | 8 | 33 | 0.87324 |

These are **unmatched data-support comparisons**, not evidence that swapping the
data alone improves Dreamer. They show that a similar number of episodes can
cover radically different parts of the task. All proposed Dreamer data still
require their own approved, fresh allocation and provenance.

B1's comparator and reward statistic also limit interpretation. At context 5,
horizon 1, v8 model terminal BCE was `0.072729235/0.072817956`; the constant fitted
to the **same scored-window prevalence** gives `0.072654808`. The apparent
12-13% gain against the old training-prevalence comparator becomes 0.10%/0.22%
worse against the matched constant. This retrospective constant is a diagnostic
reference, not a deployable predictor fitted on future data. B1 reward Spearman
ranks episode-average returns under recorded actions, not alternative actions at
one state. All frozen failures stand, but this static-random-data gate has not
been demonstrated necessary or sufficient for successful online Dreamer control.

Two interpretation corrections matter. Logged `kl_dyn_mean`/`kl_rep_mean` are
**post-free-nat-clamp**, not raw KL. Also, the summary's v6 claim that overshooting
stayed exactly at its floor across updates is too strong: last-100 means are
`0.1000324035/0.1000389402`, and earlier values exceeded the floor substantially.
v7 last-100 means are `0.0029169120/0.0030619941`, not the summary's approximate
`1e-4..4e-4`. Do not infer zero model gradients or latent collapse from these
floored summaries. Frozen records remain unchanged.

## The Actual Completion Problem

### Progress Is Not Finish

`core/finish_line.py:41-95` requires a qualified, forward passage through a finish
gate: depart the start, enter appropriately from behind, cross within the allowed
width and direction, and exit through the front. `progress >= 0.95`, and even
`progress == 1`, do not replace `finished`.

Consumed diagnostic evidence contains `progress=1`, `finish_qualified=true`,
`finished=false`, `off_track`, and zero damage:
[geometry failure analysis](../../experiments/drqv2-geometry-augmentation-v1-analysis.json),
the rows around lines 5394-5410 and 5580-5596. These establish the distinction,
not why those particular vehicles missed the crossing. Their consumed evaluation
trajectories are **not** proposed training data.

### Survival Is Not Finish Either

`env_wrapper.py:70-74` increments `off_track_count` when the **summed decision
reward is negative**, resets it otherwise, and retires when it exceeds 100.
This is not a direct geometric grass detector. With four raw frames per decision
at 50 Hz, 101 decisions correspond nominally to 8.08 simulated seconds.

The local base environment awards each tile once, subtracts `0.1` per raw tick,
and does not award an additional raw finish bonus
(`core/vendor/car_racing.py:108-111,579-584`). After tile coverage, merely circling
safely or braking does not replenish the progress-stall budget. Optimizing raw
discounted tile reward is therefore **not equivalent** to maximizing completion
probability. This is an objective distinction, not proof that it caused every
observed failure.

A recovery must resume **directed useful travel before retirement**, or complete
the qualified finish crossing. "Stay on road", "minimize speed", and "stay alive"
are inadequate goals by themselves.

### Do Not Repackage Failed Braking Heuristics

The [DrQ prefix-branch result](../../experiments/drqv2-residual-options-prefix-branch-v1-result.json)
had locally reproducible tested prefixes, but COAST both rescued and derailed
runs. Two BRAKE branches reduced final progress from about `0.6360` and `0.5037`
to `0.1544`, without producing a finish. The
[full residual-options pilot](../../experiments/drqv2-residual-options-pilot-result.json)
tied the baseline at 5/32 in one iteration and regressed to 3/32 versus 5/32 in
the other. These small, reused-development results reject a universal
brake/coast prescription, not every conditional recovery policy or Dreamer.

## Newly Checked Implementation Boundaries

These findings concern the inspected working-tree source. No fix was made.

| Finding | Verification | Implication |
|---|---|---|
| Attached imagined features in actor score loss | `dreamer_v3.py:1121-1137,1198-1204` retains a path from earlier reparameterized actions through later states. `Actor.log_prob()` detaches the sampled action, not the feature input. A two-step in-memory probe gave identical forward loss `3.7071619`, but an unwanted earlier-action gradient norm `0.004543067`; actor-gradient difference versus stopped features was `0.006581852` L2. | Revalidate stopped-feature score-function semantics before any new actor learning. This cannot explain B1 `model_only=True` failures, which return before this code. |
| Slow target used for return construction | Current calls at `dreamer_v3.py:1145,1164` use `critic_target`. Pinned upstream configs explicitly set both `imag_loss.slowtar=False` and `repl_loss.slowtar=False`, and upstream stops imagined features. | A fidelity mismatch with the plan/upstream configuration, not a demonstrated driving failure. Name and test the intended contract instead of calling all A3 work complete. |
| Startup supervision gap | With sequence length 32 and burn-in 8, two 39-decision episodes yielded no usable sequence. A 40-decision episode supplied learning targets only for transitions 8-39. The first eight transitions are always prefix-only. | Early driving and short failures need an explicitly tested reset-start/variable-prefix sampling design; not blindly larger replay. |
| Event balancing is not class balancing | A 32-transition learning window has at most one episode-end terminal in this sampler. Even all terminal-anchored windows give at most 1/32 terminal targets. | `terminal_window_fraction=0.5` does not mean 50% positive terminal labels. Count actual labels and independent events. |
| Missing task/source fields | Replay contains pixels, actions, rewards and boundary data, but no complete source/finish/progress/damage contract (`dreamer_v3.py:444-490`). | Teacher, recovery, and semantic targets require deliberate data-schema work later; they are not existing flags to switch on. |
| CPU recurrent export | A tiny in-memory learner, training exporter and deployment exporter agreed exactly across three observations and two resets. | No current deterministic export bug was demonstrated. Stochastic-training versus argmax-deployment mismatch remains a separate, untested performance hypothesis. |

Fourteen selected existing pure contract tests passed. They include detached
sample-score checks but miss the attached-feature path above. Passing these
tests is not a policy-learning result. The upstream comparison used commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`, both `agent.py` **and** `configs.yaml`;
the function-level `slowtar=True` default alone is not its configured behavior.
The probes used local Torch 2.11 CPU, not the pinned Torch 2.1 competition-like
runtime; the small export check is not submission eligibility evidence.

## Proposed Design

### 1. Completion And Recovery Data Before Scale

First obtain fresh TRAIN-only evidence that the relevant states are reachable
and controllable. Use a source chosen and frozen from training-side evidence,
not whichever candidate looked best on a blind result. Do not reuse another
study's retired teacher data or consume its reserved roads.

The data must contain complete laps, normal driving, recoverable deviations,
failed deviations, and the approach/crossing after high tile coverage. A teacher
that finishes a few easy roads but never handles the intended corner families
is not a universal expert. If a pixel teacher cannot provide coverage, a
training-only state-informed demonstrator is a **separate conditional option**:
its legality, physical fidelity, success, and extra resources need verification
before use. It is not assumed to exist or to solve the hard roads.

Preserve the complete transition contract: result frame, proposed versus executed
native/official actions, controller/source, episode/geometry, raw reward,
`finished`, retirement reason, termination/truncation, elapsed decisions, progress
and available privileged training labels. Never feed those privileged labels
directly to the deployed actor.

An initial sampling proposal is 50% natural whole-episode sequences, 25% genuine
successful-lap suffixes, and 25% matched recovery/failure neighborhoods, with
geometry/source caps. This is an ablatable proposal, not an optimized ratio.
Keep failed trajectories; success-only imitation erases the error states the
student will visit. Evaluate calibration on the natural distribution and account
for selection/oversampling when fitting event probabilities.

The old requirement that random-data B1 pass before teacher treatments remains
in force for the old plan. **Adopting a new data-first study requires a separately
approved and frozen protocol**, not relabeling v1-v9 as successes.

### 2. Task-Grounded Belief, Not Just Better Video

Keep the RSSM, raw reward head and image reconstruction as useful regularizers.
Add only targets that can discriminate driving outcomes:

| Component | Proposed training target | Deployment boundary |
|---|---|---|
| Fast local geometry | Visible directed road corridor, upcoming curvature, obstacle-relative clearance | Predict from allowed pixels; no full unseen centerline or map input. |
| Fast vehicle motion | Speed, heading error, yaw/slip and approach-to-edge dynamics | HUD and image history provide evidence, not perfect state observability. |
| Action-conditioned progress | New-tile/directed-progress events under the executed action | Keep raw reward unchanged; do not reward repeated local loops. |
| Competing events | Finish, damage/retirement, progress-stall, and timeout labels | Correct output supervision; no future label leakage into recurrent inputs. |
| Slow lap belief | Coverage/loop closure, approach to the start area, uncertainty in stall budget and finish phase | Learn from pixel/action history; add only an exact internal decision clock. |

The local renderer includes motion-related HUD signals, but no explicit finish
line marker. Exact damage, visited tiles, finish-tracker phase and the
negative-reward streak are not inputs to `Agent.act(obs)`. Some evidence may be
recoverable from history, HUD digits and subtle visited-tile shading; **exact
identifiability is unproven**. Predict distributions with uncertainty, not a
fictional fully observed state. `training/labels.py` provides useful training-only
labels, but its `off_track` field is a retirement flag, not an occupancy mask.

Start with small diagnostic/auxiliary heads on the existing belief. Introduce a
separate slow memory or start-area image memory only if grouped TRAIN-development
tests show that current recurrence lacks the needed information. Do not bundle
a new encoder, larger RSSM, RGB input, and hierarchy into the first comparison.
Retain the general RSSM rather than imposing an unverified sufficient semantic
bottleneck. "Slow memory" does not mean its targets change slowly: stall-counter
resets and finish-gate transitions can be abrupt.

The decisive metric is **correct same-state action ordering and calibrated
false-safe error**, not low whole-image MSE. A time-only predictor, HUD-only
predictor, constant-prevalence model, and frozen-teacher action are important
baselines. Predicting "this random episode usually ends now" is not learning how
to prevent that ending.
Check prior-only predictions as well as posterior probes, including actions the
student optimizer favors. Good calibration on teacher actions alone does not
protect against model exploitation on different actions.

### 3. A Deadline-Conditioned Finish Critic

Let `b_t` be the student's history-derived belief and `tau` the remaining
**declared task** decision budget. Define a separate value:

```text
F_pi(b, tau) = P_pi(real finish before failure and before the deadline | b)

target = 1                                      if the result is a real finish
target = 0                                      if fatal/stall failure or deadline
target = F_pi(b_next, tau - 1)                   if still live
unfinished F_pi(b, 0) = 0
```

Use the same event precedence as the environment if events coincide. Arbitrary
collection interruption is censored, not a task failure. The ordinary Dreamer
value may legitimately bootstrap a time-limit truncation; the **finish-before-
deadline** value has a different boundary. Keep both contracts explicit.

This value has no reward discount: `gamma=1` is appropriate for this finite-horizon
probability, not a recommendation to change every Dreamer loss to undiscounted
learning. Current ordinary `gamma=0.99` discounts a finish bonus 500 decisions
away by `0.00657048`, and 1,000 decisions away by `0.00004317`. Merely appending a
small terminal bonus is not the same proposal. A 15-decision imagination window
is also not the total value horizon; critic bootstrapping can propagate beyond it.

Train the actor to improve real, policy-consistent finish reachability, with
speed/raw reward only secondary to a predeclared completion constraint. Keep a
standard raw-reward actor/objective as the matched control. This deliberately
changes the optimization objective and must be named accordingly.
Validate the finish head passively before allowing it to influence actions.
Give both objective arms identical actor-visible clock information and auxiliary
losses; otherwise the objective contrast also changes the information available.
A predicted finish must never stop the real episode, suppress action calls, or
count as success. Only the environment's actual canonical finish can do that.

Three safeguards are essential:

- A teacher's successful suffix labels `F_teacher`, not automatically `F_student`.
  Use it for supported initialization/imitation, then collect student-continuation
  outcomes or a justified policy-evaluation target. A supervised behavior-success
  predictor must not silently become a current-policy critic.
- Do not apply an unconstrained max over unsupported offline actions. Model and
  critic overestimation can invent finishes. Limit early policy change to supported
  actions, use real branch evidence, and test uncertainty/error association before
  treating an ensemble score as a safety signal.
- Preserve the ordinary continue head's proper termination semantics. Finishes and
  failures both end ordinary value propagation; a separate finish/event head
  distinguishes their desirability. High continuation alone is not the objective.

### 4. Counterfactual Recovery Curriculum

Archive real TRAIN prefixes near a corner, deviation, or impending stall. Reset
the unchanged simulator and replay the **executed** prefix. First verify parity
of observations, rewards, physical diagnostics, counters, visited tiles and finish
phase to the extent observable. Recompute the student's recurrent belief from the
same full prefix under its current weights; never transplant a teacher's hidden
state or zero the student's memory at takeover.

At one matched anchor, compare a small frozen action/short-option set, then use
the same declared **continuation policy** in each branch. A common open-loop
action suffix answers a different question and may punish a good intervention
because its later timing changed. Distinguish first-action effects, short-option
effects and student-closed-loop effects; merge none of them into one label.
Discard actions that become identical after clipping and report tied outcomes.
Choose evaluation anchors by predeclared past-observable rules, not by discovering
which candidate later wins. Keep attempted, unrecoverable and censored branches
in the denominator. Outcome-selected rescue pairs are training examples, not an
unbiased estimate of intervention benefit or natural failure prevalence.

Reward a branch label for restoring directed progress or finishing **before the
remaining stall/task budget**, not merely reducing lateral error. If no tested
action can rescue a late anchor, move the next predefined diagnostic anchor
earlier to test anticipation. Failure of a finite candidate set does not prove
the state uncontrollable under every action.

Existing DrQ prefix parity covered 2,048 decisions across 16 cells but only four
geometry seeds. It does not establish exact long-lap replay or Dreamer recurrent
parity. If parity fails, the branch experiment is inconclusive; do not create
counterfactual labels from mismatched states.

**Complete-controller gate, added after peer review:** a useful branch is only
feasibility for the tested intervention and continuation at that anchor. Before
promotion, freeze and test the entire policy using permitted pixels/action history:
its intervention trigger, recovery-action selector, duration/termination rule,
handback rule, repeated-intervention handling, and memory updates. Include
nonintervention states and successful-parent episodes. An oracle best branch,
simulator-only trigger, hidden-state recovery action, or hindsight handback cannot
stand in for this policy's result.

During recovery, advance the base Dreamer belief with every actual observation
and **executed native action**, preserve its clock, and do not reset/freeze its
carry. Current export code stores its own proposed action as `prev_a`
(`agent.py:297-304`, `dreamer_v3.py:778-787`). That is appropriate for the current
unmodified actor, but blindly overriding its returned action with a recovery
controller would leave the wrong action in its next recurrent update. This is a
future composite-controller integration requirement, not a newly claimed bug in
the base actor or evidence that such a controller already exists.

If recovery is distilled into a monolithic student, evaluate that actual student;
explicit options need not remain in deployment. In either case, require normal-
start whole-lap comparison on separate TRAIN-DIAGNOSTIC geometries. Student-only
handback after a short recovery is necessary evidence, not proof of eventual finish.

After full-lap examples exist, an optional backward curriculum gives the student
the final 25, then 50, 100, and longer decisions of a real successful trajectory.
These are proposed stage lengths, not measurements. Expand only under frozen
TRAIN-development mastery rules, eventually removing the teacher prefix entirely.
Preserve elapsed time, damage, visited tiles and stall/finish state. Every repeated
prefix is charged to the interaction budget. This is reverse **control takeover**,
not teleportation, a reset of the deadline, or replaying pixels without physics.

## What To Test First

This is a proposed decision tree, not a runnable or frozen protocol. No seed IDs
are allocated by this report. Fix source/config, exclusions, budgets and gates
before any future collection. Existing B1 stops are not overridden.

| Stage | Smallest discriminating question | Control and acceptance | Stop or redirect |
|---|---|---|---|
| P0: contracts | Is the intended objective/data path actually implemented? | Pin actor/source identity and test stopped-feature gradients, configured return target, reset-start coverage, task-deadline labels, executed-action carry and CPU parity. Separately check actual local/ZIP parity before interpreting deployment. | No actor experiment while a required contract fails. Generic smoke is not candidate-specific package parity. |
| D0: diagnosis, before treatment selection | Is there an eligible trained policy to diagnose, or only model/support evidence? | For an existing frozen policy, describe TRAIN-side failures and successes using pre-failure pixels, executed actions, speed, new tile visits, negative-reward count, damage and finish phase where available. Internal state is diagnostic supervision only. Without a trained policy, audit support/prediction and route conditionally to P1/P1b. | Record missing fields or mixed/unknown modes; do not infer physical road departure from `off_track`, invent missing Dreamer policy evidence, or use RLPD's distribution as Dreamer's. New diagnostic collection requires its own authorization/protocol. |
| P1: ordinary data support | Can a frozen TRAIN source supply useful whole-episode driving support? | Freeze one source and a draft 32,768-decision whole-episode collection cap, retaining success and failure. Seek at least 12 distinct finished geometries spanning intended families and report uncovered families. This is an operational proposal, not a power calculation. **No matched-recovery-anchor requirement for P1b.** | Missing support at the cap leaves the model hypothesis inconclusive and the coverage gate unmet. Do not double the cap, replace the source, or open evaluation roads after seeing the result. |
| P1b: data only | Does supported driving data improve the ordinary model before any new objective or architecture? | Same corrected learner, two seeds, updates and preregistered TRAIN geometry pool: capped random collection versus capped teacher whole episodes, retaining successes and failures. No added BC loss, semantic loss, special success sampler, finish objective, or curriculum. Compare grouped development prediction and decision relevance. | Old B1 data are context, not the matched random control. No gain rejects the simple data-only explanation, not all later combinations. |
| R1: recovery, conditional | Does a localized failure admit a useful intervention AND a deployable selector? | Only after D0 motivates recovery and a relevant student exists: freeze a separate total branch/prefix budget; the earlier 20 outcome-differing-anchor proposal belongs here, not in P1. Log all attempts plus successful parents, verify parity, then pass the complete-controller and full-reset rescue/harm gates. | A best-of-K oracle or short rescue is not a policy. Failure to meet the anchor count is inconclusive; no adaptive cap extension. Do not force recovery if another hypothesis is better supported. |
| P2: representation | Is useful action-relevant information learnable from deployment inputs? | On fixed data, compare ordinary WM versus the smallest semantic-head addition; same architecture budget where practical, updates, two learner seeds, and geometry-grouped development. Require event calibration better than same-distribution constants and action-ranking confidence above chance, not merely better video. | Time-only/HUD-only baseline wins, all-zero finish predictor, no recoverable pairs, or insufficient positive independent roads: diagnose or mark inconclusive. |
| P3: objective | Does finish reachability improve driving rather than just predictions? | Fixed data/representation, compare ordinary raw-return objective against finish-first objective; hold BC strength, replay, budgets and checkpoint opportunities fixed. Include a same-data BC-only diagnostic to expose a broken world-model policy-learning path. Require student-only full-reset completion improvement for both learner seeds on the same development cells. | Lower model loss without finishes, survival by stalling, or one-seed-only gains do not justify scale-up. |
| P4: curriculum, conditional | Does backward scheduling improve control takeover? | Compare backward scheduling against uniform takeover sampling over the same frozen prefix bank, source and **total** interaction budget including replayed prefixes. A separate whole-episode control can assess whether prefix access is worth its cost. | Suffix mastery without full-reset transfer rejects promotion. Do not combine this change with a new objective/encoder. |
| P5: generalization | Does one frozen standalone student reliably finish diverse fresh roads? | Predeclared geometry-family split, CPU/reset/resource gates, screen selection then fresh confirmation; reserved blind only once its new protocol permits. Report each learner seed separately. | No tuning after confirmation/blind; no assisted or per-track-best aggregation. |

Do not run all rows as a single large package or a mandatory serial ladder.
After P0 and D0, P1b remains the conditional first learner comparison for the
current Dreamer support gap; it is not the first action before diagnosis.
R1, P2, P3 and P4 are separate branches selected by evidence, not an automatic
sequence of added complexity.
A passive P2 probe can guide whether additional representation learning is needed;
do not make every proposed head mandatory before testing the objective.
If P1 cannot produce support, the next useful research is
demonstrator/observability/controllability diagnosis, not an expensive new RSSM.
If BC can finish but imagined-policy learning destroys that ability, freeze the
good initialization and investigate the policy objective/model exploitation.

Before attributing an improvement specifically to world-model action selection,
compare real-continuation action labels with short-imagination labels at matched
anchors, candidate actions, continuation policy and distillation procedure. Report
their different acquisition/compute costs. Simulator-assisted imitation with a
Dreamer encoder is not by itself evidence that imagined policy learning helped.
Recovery-branch collection versus ordinary continuation is another **separate**
data treatment, not silently bundled into P1b.

Freeze TRAIN prefix-bank selection and split by ancestor episode and geometry,
not by overlapping windows. Its outcome-selected suffix success rates cannot
replace the actual student's full-reset arrival-distribution success rates.

A proposed practical scale-up hurdle is at least 12/24 full-reset finishes on
24 distinct TRAIN-development geometries **for each** learner seed, without
regressing the same-data control or CPU eligibility. This is stricter than
nonzero feasibility but still only a design threshold, not a reliability claim
or a replacement for the old frozen Phase D. A reliability target such as 95%
belongs to a later adequately sized, fresh, fixed-model test.

**Rescue and harm accounting, added after peer review:** on matched normal-start
episodes for the frozen baseline and actual complete candidate, report all four
outcomes. Let `n01` be baseline fail / candidate finish, `n10` baseline finish /
candidate fail, `n11` both finish and `n00` both fail:

```text
net finish-rate change = (n01 - n10) / N
rescue rate            = n01 / (n00 + n01)
lost-success rate      = n10 / (n10 + n11)
```

Report counts and denominators, learner seeds and geometry-family strata. Freeze
a permissible lost-success rate and seed/family regression limits alongside the
net-improvement rule before evaluation. Use geometry-clustered uncertainty and
do not pool different selected checkpoints. If the baseline has no finishes,
lost-success rate is undefined, not zero; insufficient independent roads makes
the harm claim inconclusive. Outcome-selected rescue pairs cannot estimate these
full-reset rates, and zero observed harm does not prove safety.

A new four-case algebra check illustrates why this matters: baseline finishes
`[1,1,0,0]`, a recovery alternative gives `[0,1,1,0]`, and a causal selector
intervenes on cases one and three. Oracle best-of-branches yields 3/4 finishes,
but that selector yields 2/4: one rescue and one lost baseline finish, net zero.
The arithmetic/assertions passed without any policy or environment execution;
these are illustrative arrays, not measured driving outcomes.

For a deadline-dependent actor, make the student's own elapsed/remaining decision
clock available to its policy as well as the finish critic. Otherwise identical
pixel beliefs with different deadlines cannot induce different actions. This is
an internal counter derived from reset/action history, not an oracle stall counter.

Count source training, teacher collection, random prefill, student actions,
repeated prefixes, branch continuation and rejected attempts separately and in
total. Count updates, labels/queries, GPU time and CPU latency too. Equal student
online steps do not imply equal total data or compute. The current custom
evaluator needs a compatible frozen Dreamer/geometry contract; do not assume
existing DrQ geometry tooling already supports the proposed actor.

## Validation Performed In This Session

### Evidence And Code Checks

- Three independent read-only audits covered Dreamer implementation, frozen data,
  and task/environment mechanics. A separate conceptual review challenged the
  proposed design. No implementation, new training, driving rollout, evaluation
  partition, package upload or model confirmation was performed.
- Reaggregated the existing B1 run reports and development arrays without loading
  new roads or mining reserved/blind trajectories. Read already published blind
  aggregate results only for status, not road-specific tuning labels.
- Ran 14 existing pure unit contracts, synthetic actor-gradient and replay-window
  probes, and in-memory deterministic export parity. These are correctness
  checks, not a rerun of the historical 139-test integration claim.
- Checked pinned official Dreamer source/config and the primary literature below.
  Literature supports mechanisms and cautions, not HAIC completion guarantees.

### Mathematical Checks

For ordered road sections, the chain rule gives:

```text
P(finish) = product_i P(pass section i | passed all previous sections)
```

No independence assumption is needed for that conditional product. The following
**illustration**, not measured road statistics, uses 20 equal conditional rates:

| Per-section success | Whole-lap success |
|---|---|
| 0.95 | 0.358486 |
| 0.98 | 0.667608 |
| 0.99 | 0.817907 |
| 0.995 | 0.904610 |
| 0.999 | 0.980189 |

For 95% lap success the equal conditional rate would need to be
`0.95**(1/20) = 0.997438621`. This explains why improving average progress is not
enough: small repeated tail risks can dominate completion. Actual sections have
different, policy-dependent hazards; this arithmetic does not estimate them.

For a constant continue predictor with natural terminal prevalence `36/12288`
and continue-positive weight 56, minimizing weighted BCE gives:

```text
p_continue = 1 - 36/12288                         = 0.9970703125
q_weighted = 56*p_continue/(56*p_continue+1-p_continue)
                                                  = 0.999947533
natural BCE at p_continue                          = 0.020013847
natural BCE at q_weighted                          = 0.028925351
```

This analytically confirms the weighting direction; it does not predict the
trained network's exact outputs. Correctly upweighting rare failures still does
not create finish examples, and weighted outputs are not naturally calibrated
probabilities without appropriate correction/validation.

A five-state finite-horizon Bellman toy checked success, failure, near-goal,
two-steps-away and a waiting loop. Near-goal had a 0.9 finish / 0.1 failure
transition. The recurrence yielded `F(near,1)=0.9`, `F(far,1)=0`,
`F(far,2)=0.9`, and `F(waiting,tau)=0`; success was 1 and unfinished deadline
was 0. This verifies boundary algebra only, not a learned critic's calibration.
An initial inline command had a syntax error before executing; the corrected
recurrence and all assertions passed.

Finally, even all-success tests do not prove certainty. Under IID Bernoulli
sampling from a fixed target distribution, the one-sided exact 95% lower bound
after `n/n` successes is `0.05**(1/n)`: 24/24 gives 0.882654, 32/32 gives
0.910632, and 59/59 gives 0.950492. Repeated CPU reloads or obstacle variants on
the same geometry are not additional independent roads. Clustered/family-shifted
track samples need corresponding analysis; this simple bound is not a guarantee
for official private tracks.

## Reproduction And Provenance

Inspected HEAD: `d94f77110080ae5755252d46300a5b587b4e020f`. The tree already
contained other work, so HEAD alone does not identify the inspected implementation.
Source fingerprints recorded during this audit:

| File | SHA-256 |
|---|---|
| `dreamer_v3.py` | `015189e90c50c233cffe77f9c0c52e2f02c9b2e2a946c1da7b94d7254045aa42` |
| `train_dreamerv3.py` | `d180fbdf36d3324a48cf9b2783f2dddf203d7b6cd4ae4cf73cd79c75d5038328` |
| `common_adapter.py` | `76200f82f81d9dad19e8b522b439eb549d30cbbd63df77a58347ab224d7802f3` |
| `env_wrapper.py` | `8687215412da0a1f34534623d5050030d85c629a70cb922e9d2199c513a3941e` |
| `core/finish_line.py` | `7c16dda1378dcd65aed46d9da96608aa055124c2f27d8fb00c309078361d4204` |
| B1 diagnosis JSON | `ee15cb9a1269356beab4c0950035bac53c8a7506fa8acbd73b0fc9f2caf8f29f` |
| B1 iteration summary JSON | `62dcf6b915716d3b4442619916514b40941994469a26f7903b28b154d3b3e45c` |

Development decision totals for v1 through v9 were respectively
`11862, 10885, 10421, 10931, 11104, 10203, 10426, 11049, 11232`.
Training run roots and exact frozen inputs are routed by the B1 summary/diagnosis;
the aggregate counts are exposures, not new or independent datasets.

The RLPD V5 prior-data manifest and episode rows give **eight** finished geometries
in 35 complete episodes, 15,915 stored transitions plus 469 discarded decisions,
within the 16,384-decision cap. The early talk message saying six is inconsistent
with those primary rows. The separate long-horizon dataset has six finishes in
36 complete episodes. Neither is a dataset-use authorization for Dreamer.

### Exact Contract And Gradient Checks

The selected unit-test invocation was:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  tests.test_dreamerv3.TestDreamerV3.test_rssm_sequence_applies_each_action_to_the_successor_observation \
  tests.test_dreamerv3.TestDreamerV3.test_rssm_burnin_context_does_not_receive_learning_gradients \
  tests.test_dreamerv3.TestDreamerV3.test_actor_bounded_normal_score_gradient_and_entropy \
  tests.test_dreamerv3.TestDreamerV3.test_twohot_reward_and_value_heads_encode_decode_raw_returns \
  tests.test_dreamerv3.TestDreamerV3.test_kl_free_nats_are_applied_to_both_gradient_paths \
  tests.test_dreamerv3.TestDreamerV3.test_terminal_imagination_weight_stays_zero_after_terminal \
  tests.test_dreamerv3.TestDreamerV3.test_terminal_positive_weight_increases_positive_continue_gradient \
  tests.test_dreamerv3.TestDreamerV3.test_lambda_return_stops_bootstrapping_after_terminal \
  tests.test_dreamerv3.TestDreamerV3.test_sequence_replay_keeps_terminal_successor_separate_from_reset \
  tests.test_dreamerv3.TestDreamerV3.test_sequence_replay_uses_truncation_successor_not_next_episode \
  tests.test_dreamerv3.TestDreamerV3.test_sequence_replay_returns_same_episode_burnin_prefix \
  tests.test_dreamerv3.TestDreamerV3.test_terminal_window_fraction_anchors_sequences_on_terminal_actions \
  tests.test_dreamerv3.TestDreamerV3.test_warmup_to_policy_handoff_matches_continuous_recurrent_prefix \
  tests.test_dreamerv3.TestDreamerV3.test_observe_replaces_proposed_online_action_with_executed_native_action
```

Result: 14 tests passed in 0.496 seconds. The following is the actor probe's
executable body, formatted from its inline `python -c` invocation. It creates no
environment, calls no optimizer, and writes no research code file:

```python
import torch
from dreamer_v3 import Actor, CategoricalRSSM

torch.set_num_threads(1)
torch.manual_seed(260926)
rssm = CategoricalRSSM(
    action_dim=3, embed_dim=8, hidden_dim=8,
    num_categoricals=2, num_classes=3,
)
actor = Actor(in_features=rssm.state_dim, action_dim=3)
h, z = torch.zeros(4, 8), torch.zeros(4, 6)
a0, _ = actor(torch.cat([h, z], -1))
h1, z1, _, _ = rssm.step_prior(h, z, a0.clamp(-1, 1))
s1 = torch.cat([h1, z1], -1)
a1, _ = actor(s1)
loss = -actor.log_prob(s1, a1).mean()
reference = -actor.log_prob(s1.detach(), a1).mean()
through = torch.autograd.grad(loss, a0, retain_graph=True)[0]
stopped = torch.autograd.grad(
    reference, a0, allow_unused=True, retain_graph=True,
)[0]
params = tuple(actor.parameters())
actual = torch.autograd.grad(loss, params, retain_graph=True)
expected = torch.autograd.grad(reference, params)
delta = torch.cat([(a - b).reshape(-1) for a, b in zip(actual, expected)])
print(loss.item(), reference.item(), through.norm().item(), stopped)
print(delta.norm().item(), delta.abs().max().item())
```

Outputs, rounded: `3.7071619, 3.7071619, 0.004543067, None`, then
`0.006581852, 0.005271666`. This is the state-dependence counterexample, not a
complete replacement test suite or a measurement of driving performance.

Replay probe parameters were capacity 80 with two 39-transition complete episodes,
then capacity 40 with one 40-transition complete episode; zero uint8 frames of
shape `(4,84,84)`, zero actions, reward equal to transition index, terminal at the
last transition, `sample_sequence(1,32,burnin=8)` and, for the second case,
`terminal_fraction=1.0`. The export probe used learner seed 17, `embed_dim=32`,
`hidden_dim=16`, two categoricals/four classes, capacity 1, and constant float32
observations `0.0,1.0,0.3`, twice with resets. Its learner/export/deployment state
dictionaries were copied in memory, without saving or loading a checkpoint file.

## Primary References

| Source checked on 2026-09-26 | Relevant support | Transfer limitation |
|---|---|---|
| [DreamerV3 paper, v2](https://arxiv.org/html/2301.04104v2) | RSSM, score-function actor, return normalization, joint world-model/policy learning. | Generic benchmark success does not certify this local implementation or random-data B1 design. |
| [Pinned upstream agent](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py) and [config](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/configs.yaml) | Stopped imagined features and configured return-target semantics. | Not a reason to copy its largest model or assume deterministic export is identical. |
| [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300) | Expanding competence outward from goal-near states. | Arbitrary reset states are not available here; use real prefix replay and preserve budgets/state. |
| [Go-Explore](https://arxiv.org/abs/2004.12919) | Remember and return to useful states before exploration. | No claim that simulator restoration or Atari results transfer automatically. |
| [DAgger](https://arxiv.org/abs/1011.0686) | Student-distribution errors undermine pure offline imitation. | Requires a competent labeler on those states; teacher failure cannot be cured by querying it more. |
| [Recovery RL](https://arxiv.org/abs/2010.15920) | Separate task optimization from learned recovery behavior. | Road occupancy is not this task's full safety/finish condition; no formal safety certificate follows. |
| [LOMPO](https://arxiv.org/abs/2012.11547) | Latent uncertainty can constrain offline world-model exploitation. | Disagreement needs validation against actual error and under distribution shift. |
| [Director](https://arxiv.org/abs/2206.04114) | Long-horizon hierarchical behavior from pixel world models. | Deferred option if credit assignment remains the bottleneck, not the first implementation. |
| [RLPD](https://arxiv.org/abs/2302.02948) | Prior trajectories can help online learning, including imperfect data. | It does not prove that any teacher dataset or identical mixing recipe works for Dreamer. |

The research remains internal CarRacing proxy work. Before any future privileged
training or external competition action, verify the relevant current official
rules; pixel-only deployment compatibility is not blanket permission for every
data-collection method. Follow the
[evaluation protocol](../evaluation/protocol.md) and
[partition policy](../evaluation/generalization-policy.md). No official track
result, submission, confirmation or guaranteed completion is claimed.
