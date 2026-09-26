# RLPD Completion-First Research

Date: 2026-09-26. Session: `r4f7`. Revised after the peer-direction comparison
requested on 2026-09-26; see Section 1.1.

**Original report status: research and validation only; no implementation or execution approval at drafting.**
This report answers the request to audit RLPD and propose a substantially stronger
route to finishing. No training, driving-data collection, new evaluation cell,
submission, or model confirmation was performed. Existing blind episode details
were not inspected. Published blind aggregates below describe historical status,
not a tuning dataset. Concurrent work and frozen experiment artifacts are unchanged.

**Implementation update, 2026-09-26:** The user subsequently authorized internal
implementation, synthetic tests, and then ongoing internal learning iterations.
The [separate frozen G0 protocol](../../experiments/rlpd-g0-completion-v1.json)
and [observational result](../../experiments/rlpd-g0-completion-v1-result.json)
now cover 12 consumed TRAIN-only geometries and two frozen actors with no learner
updates. This later work does not turn the original research report into a
retroactive execution protocol. A causal G1 treatment remains unselected pending
successful-control and failure-window analysis; official submission and model
confirmation remain separate actions.

**G0 follow-up gate:** Independent read-only checks of all 24 hashed traces found
18 failures and six finishes on five distinct road geometries. The first measured
event among failures was contact in 9, centerline-distance proxy in 6, and low
directed motion in 3; one finished control contacted an obstacle and two briefly
exceeded the centerline threshold. Of three qualified nonfinishes, two traveled
road-near in the wrong direction long before qualification; the third later
became centerline-far. This is chronology, not causal attribution or a matched
actor-treatment result. The [fixed-window freeze](../../experiments/rlpd-g0-fixed-windows-v1.json)
and [unreviewed extraction result](../../experiments/rlpd-g0-fixed-windows-v1-result.json)
now cover ALL 24 already consumed TRAIN traces, including six successful
controls; 101 of 144 anchor slots have bounded windows and 43 are explicitly
missing. Before choosing G1, review those windows with an unknown/mixed category.
Do not train a rescuer,
completion critic or memory variant merely because one proxy is frequent.
In a small retrospective subset, two contacted failures visibly stalled while a
different actor's contacted finish continued; pre-flip wrong-way images remained
ambiguous against successful bends. The [frozen pixel-only motion rule](../../experiments/rlpd-g0-pixel-motion-v1.json)
has therefore scored all 24 reused G0 TRAIN episodes without selecting a
threshold, changing an actor or claiming rescue. Its
[result](../../experiments/rlpd-g0-pixel-motion-v1-result.json) was independently
verified on all 10,049 correlated decisions. Labels occupy four failed episodes
on only three geometries, and continuous scores overlap finished controls. The
[recorded gate decision](../../experiments/rlpd-g0-pixel-motion-v1-decision.json)
therefore rejects a single scalar intervention threshold. A source-hashed,
geometry-grouped [frozen-encoder diagnostic probe](../../experiments/rlpd-visual-representation-probe-v2-result.json)
has now completed one CPU head-only training iteration with j<20 burn-in and
unchanged actor parameters. It reported 188/190 retrospective diagnostic
low-speed decisions at the fixed cutoff, but ALL 188 true positives came from
the two actor episodes on one geometry; no diagnostic geometry has a finished
parent. The visual HUD displays true speed, so this probe does not establish
optical-flow decoding, contact causation, useful trigger safety or rescue.
First require independently supported positive and successful-parent diagnostic
road coverage under a separate frozen protocol, then an RLPD-specific complete
state/prefix parity and harmed-finish branch. The contacted comparison is not
matched by actor or road; the G1 intervention remains unselected.

## 1. Recommendation

**Keep RLPD as the data-efficient learning backbone, but first establish WHERE
the frozen RLPD policy loses directed progress and WHAT precedes that loss. Then
test the smallest targeted intervention. Recovery data is a leading conditional
hypothesis, not the predetermined first treatment. High raw return is still not
the same task as reliable completion.**

Working name: **Finish-First Recovery RLPD (F2R-RLPD)**. This is a proposed local
synthesis for the recovery branch, not a claim of a new published algorithm,
demonstrated improvement, or a mandatory architecture for every failure type.

The intended sequence is:

1. Freeze actor/source, action mode, adapters, and runtime identity. Require a
   separate exact-package handoff check before attributing a deployment difference
   to driving behavior; package creation is not required for local diagnosis.
2. On newly allocated TRAIN-only roads, classify the first observed loss of directed
   progress, with successful controls and mixed/unknown categories. Choose a bounded,
   falsifiable hypothesis rather than asserting a cause from a retirement label.
3. Test ONE targeted intervention, which may prevent a bad turn entry, recover after
   a contact, or correct stagnation/finish approach. Require actual finish benefit
   AND preservation of existing finishes. If evidence is unresolved, retain a small
   competing-hypothesis diagnostic rather than starting broad retraining.
4. Establish that the entire resulting behavior is realizable from permitted pixels/
   history. A privileged teacher, hindsight selector, or short successful suffix is
   not a deployed policy. Representation feasibility can precede distillation when
   current observations cannot distinguish the useful actions.
5. If a targeted data gap is supported, add verified successes AND failed alternatives
   and test the prior-data intervention in isolation. If objective/credit or history
   is implicated instead, test completion value or memory separately, not automatically
   after a recovery-data run. Their calibration/realizability gates still apply.
6. Replicate full-lap gains on fresh, predeclared geometry partitions. Improve lap time
   after reliability, not instead of it.

The important change is **diagnose before prescribing, then change only the
supported bottleneck**, not another entropy sign, steering clamp, or larger network.
Prevention may be more valuable than recovery after control has already been lost.
If a proposed teacher cannot rescue relevant student states, or the student cannot
recognize and execute the correction, do not spend a large learner budget on it.

No current evidence permits an honest promise of finishing every unseen private
track. The achievable engineering target is a much higher, explicitly measured
finish rate on diverse, unseen geometries, with a predeclared lower confidence
bound and retained failure cases. Section 8 defines that standard.

### 1.1 Comparison with other-session advice

The initial audit read existing RLPD result discussions, but missed the two n8w3
posts arriving while this report was being drafted. The user's follow-up prompted
an explicit comparison of [failure-first advice](../../talk/messages/20260926T063421Z-n8w3-completion-failure-gate.md),
[recovery/package boundaries](../../talk/messages/20260926T063749Z-n8w3-recovery-selection-boundary.md),
and the relevant [Dreamer research result](../../talk/messages/20260926T063301Z-f3c8-dreamer-completion-research-result.md).
Peer messages are suggestions, not execution authorization or primary evidence.

| Peer point | Check and comparison | Decision |
|---|---|---|
| Classify failure before selecting a remedy | Our Section 3 already admits that terminal summaries cannot distinguish causes, but the original headline and G0 exit still privileged recovery | ACCEPT as a better-supported order; G0 must now output a failure taxonomy and a targeted falsification choice |
| Exact released-ZIP identity/action parity differs from generic smoke and CPU actor reload | Source inspection confirms generic smoke uses zero observations and no local-reference action comparison; exact seed-11 CPU evaluation does exist, but a release-ZIP parity receipt was not established | ADD a distinct handoff gate, without invalidating local CPU evidence or requiring a new ZIP just to diagnose local failures |
| Recovery branch success is not a deployable fix | Prefix parity, original time/damage/history, actual student handback, pixel-only trigger AND actions, harmed-finish controls, and normal-start testing were already required in Sections 4 and 8 | RETAIN; do not present these safeguards as newly demonstrated performance |
| DrQ opening-turn failures suggest an RLPD target | Frozen DrQ evidence includes collision-free early low-progress failures on two consumed roads, but no matched RLPD causal test | KEEP as a competing hypothesis only; do not import its road-specific diagnosis or unconditionally reduce speed |
| A small deployed controller is only a conditional possibility | Neither peer note establishes useful, visually selectable RLPD actions with net full-lap benefit; n8w3 explicitly rejects treating the order as a promised shortest route | AGREE with that limitation; no automatic switch to a deployed controller |
| Dreamer completion/history cautions transfer | The task boundary and teacher/student distinction apply to both lanes, while Dreamer B1/RSSM defects concern a different learner | RETAIN shared constraints; do not use Dreamer failures to explain RLPD or adopt its architecture |

The cited seed-11 result JSON also confirms the peer's published aggregate of
9 finishes, 14 `off_track`, and one crash in 24 canonical blind cells. This check
read the already-published RESULT aggregate only, not blind episodes or road
details, and is not a new tuning signal. The revised research order is justified
by the missing RLPD-specific failure timing and available nonblind evidence, not
by reopening blind for diagnosis.

The stronger peer point is methodological, not a rival proven algorithm. Do not
turn "failure first" into an endless requirement to prove a unique cause from
observation alone: chronology selects hypotheses; matched interventions test them.

## 2. Current Position

The frozen protocols/results, not older prose or talk messages, govern these facts.
Counts use canonical `repeat == 0`; the second CPU reload is not another road.

| Study | Matched comparison | Result | Interpretation |
|---|---|---|---|
| Pixel pilot v2, 16,384 decisions/student | RLPD vs otherwise identical online-only pixel SAC | Selected screen actors: seed 0, 0/12 vs 0/12; seed 1, 1/12 vs 0/12 | Failed the two-seed pilot gate; later partitions remain closed |
| Long-horizon v1, 131,072 decisions/student | RLPD vs matching SAC, seeds 10/11 | Screen: 7/24 vs 3/24 and 12/24 vs 3/24; confirmation: 3/32 vs 2/32 and 12/32 vs 7/32 | Prior-data recipe helped in both paired seeds, but seed sensitivity is substantial |
| Long-horizon v1 finalist | One screen-preselected seed-11 actor | Published blind: 9/24 finishes, mean progress 0.679 | Internal candidate, not dependable completion or official performance |
| Entropy v5, 131,072 decisions/student | Author target -1.5 vs +1.5, seeds 50/51 | Confirmation: 17/32 vs 6/32 and 7/32 vs 6/32 | Increasing the entropy target is not supported as the next fix |
| Entropy v5 finalist | Author-target seed 50 | Published blind: 7/24 finishes, mean progress 0.657 | Separate cells/candidate from long-horizon v1; NOT a matched regression comparison |

Sources: [pilot result](../../experiments/pixel-rlpd-offpolicy-pilot-v2-result.json),
[long-horizon result](../../experiments/pixel-rlpd-long-horizon-followup-v1-result.json),
[entropy v5 result](../../experiments/pixel-rlpd-entropy-target-ablation-v5-result.json),
and each corresponding protocol JSON.

The designated internal candidate remains long-horizon seed 11 at step 131,072:
`runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt`,
SHA-256 `f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1`.
This report neither selects a replacement nor changes official model status.

Important qualifications:

- "Long-horizon" increased interaction/training budget, not the Bellman discount
  horizon: the learner still uses `gamma=0.99` and one-step backups.
- Increasing the prior cap and learner budget together between pilot and follow-up
  did not isolate their individual effects. The within-study SAC/RLPD comparison
  is the relevant matched evidence.
- Long-horizon v1 froze its single finalist on screen. Entropy v5's frozen
  `evaluation.blind_finalist` rule selects the winning-target seed using
  confirmation metrics. That is not independent confirmation after final seed
  selection and differs from the generic no-selection-on-confirmation policy.
  Preserve its historical protocol provenance; do not call it screen-preselected.
- Each 24-cell blind grid comprises eight geometry seeds with three track-ID
  obstacle variants, not 24 independent road geometries. Confirmation similarly
  has eight geometries with four variants. Do not manufacture narrow confidence
  intervals by counting variants or reloads as independent samples.
- Entropy v1-v3 preflight failures and v4's pre-screen source-hash stop are not
  measurements of a driving policy's failure. V4 trained models but opened no
  evaluation; its retired allocations are not reusable fresh cells.
- Historical status prose is partly stale: the old RLPD proposal and model ledger
  still contain pre-v5 next-step wording. This report uses frozen v5 evidence
  without silently rewriting shared status or historical plans.

## 3. Verified Mechanisms

### 3.1 Raw return is not the finish objective

The official-compatible environment rewards first visits to tiles and subtracts
`0.1` per raw frame. It does not add an explicit finish bonus. A finish requires
at least 95% tile coverage AND a valid forward finish-line traversal and exit.
`progress >= 0.95`, remaining alive, and crossing backward are not finishes.

Sources: `core/vendor/car_racing.py:110-111,575-588`,
`core/finish_line.py:41-95`, `env_wrapper.py:79-100`.

Illustrative arithmetic, NOT measured trajectories: ignoring common initial
offsets, a slow finish at 95% coverage after 1,000 four-frame decisions earns
approximately `950 - 400 = 550`, while an 80%-coverage failure after 400 decisions
can earn `800 - 160 = 640`. Thus the raw-return objective can rank a failure above
a finish even without discounting. A collision-ending failure need not receive
the separate out-of-bounds `-100` penalty.

This proves a possible objective conflict. It does **not** prove that reward
misalignment caused a particular recorded failure, nor that dense tile reward
cannot learn a finishing policy.

### 3.2 The credit and observation horizons are short

At 50 FPS and frame skip 4, one decision normally advances 0.08 simulation seconds.
Four stacked observations span three intervals, approximately 0.24 seconds.

| Quantity | Value |
|---|---:|
| `1 / (1 - 0.99)` effective discount scale | 100 decisions, approximately 8 seconds |
| Discount half-life | approximately 5.52 seconds |
| `0.99 ** 250` | 0.08106 |
| `0.99 ** 500` | 0.006570 |
| `0.99 ** 1000` | 0.00004317 |

These are algebraic scales, not hard memory cutoffs. Dense local reward may still
propagate useful behavior. Nevertheless, a distant finish signal appended to the
unchanged objective will be heavily attenuated.

The current policy is a feed-forward pixel encoder, 50-dimensional latent, and
two 256-unit hidden layers. It has no persistent memory, previous-action input,
elapsed-step input, damage input, or progress input. Its CNN learns through raw
TD loss and is copied to the actor with actor-CNN gradients stopped.
Sources: `haic/algorithms/rlpd/model.py:18-35,92-114`,
`haic/algorithms/rlpd/agent.py:227-286`, root `agent.py:841-898`.

**Hypothesis:** hidden damage, recent control response, no-progress duration, and
route phase create some action ambiguity that a 0.24-second stack cannot resolve.
Speed is partly visible in the image/HUD; it is incorrect to claim the model has
no speed information. A recurrent model cannot recover information that no
permitted observation history contains.

### 3.3 The retirement label does not identify its physical cause

`off_track` means more than 100 consecutive decisions whose aggregate raw reward
is negative. It is not a geometric test of whether the car is on grass. Stopping,
circling on visited road, or driving backward can trigger it. At the default skip,
101 such decisions are about 8.08 seconds. Time since the FIRST departure or
missed turn is more informative than the final retirement frame.

Five collision events exhaust damage. Earlier collisions already reduce grip,
engine output, and steering response (`damage.py:17-21,33-36`). Therefore, a low
count of terminal `crash` labels would not establish that collisions are irrelevant
to later `off_track` retirements. Damage is not a dedicated HUD field in
`core/vendor/car_racing.py:717-780`.

Consequences: do not prescribe unconditional braking, do not train a "grass"
classifier from `off_track` labels, and do not reset damage or the negative-reward
counter when constructing a recovery example.

### 3.4 RLPD is not directly cloning successful behavior

The current offline/online split is 32:32 in a batch of 64. Sampling inside each
source is uniform over transitions, with replacement. It is not uniform over
geometries, successful episodes, late-lap states, or recovery opportunities.
Teacher actions train Q regression, not a BC loss. Ten Q heads share an encoder,
training data, and targets; they are not ten independent safety certificates.
UTD is 1, not 10. The target subset is one random head; the actor uses the ensemble
mean. Entropy is absent from the Bellman backup, not from actor optimization.

Sources: `haic/algorithms/rlpd/agent.py:62-110,227-286`,
`haic/algorithms/rlpd/replay.py:244-268`,
`scripts/train_rlpd_entropy_ablation.py:218-240`.

The existing teacher is a fallible frozen DrQ actor. Finishing examples establish
that some behavior works, not that this teacher can recover the student's errors.
Purely collecting more easy-lane frames can repeatedly expose the same coverage
gap. Conversely, failures remain useful off-policy data: deleting every failed
episode or copying the teacher everywhere is not the recommendation.

### 3.5 What the existing episodes actually show

The following is a new read-only recount of recorded TRAIN, screen, and consumed
confirmation artifacts, not new driving. Because confirmation episode details
were mined for this diagnosis, those cohorts are now development-contaminated for
future hypotheses. They retain their original historical result, but cannot serve
again as fresh confirmation. Blind episode files were not opened.

| Nonblind cohort | Finishes | `off_track` | `crash` | Denominator |
|---|---:|---:|---:|---:|
| Pilot v2 screen, all eight exported actors | 1 | 82 | 13 | 96 canonical episodes |
| Long-horizon v1 screen, all eight actors | 35 | 147 | 10 | 192 |
| Entropy v5 screen, all eight actors | 29 | 152 | 11 | 192 |
| Long-horizon RLPD seed 10 confirmation | 3 | 27 | 2 | 32 |
| Long-horizon RLPD seed 11 confirmation | 12 | 17 | 3 | 32 |
| V5 author-target seed 50 confirmation | 17 | 12 | 3 | 32 |
| V5 author-target seed 51 confirmation | 7 | 23 | 2 | 32 |

No timeout or operational-failure category occurred in these cohorts. Of the 40
failed author-target v5 confirmation episodes, 35 (87.5%) ended with `off_track`.
The high-priority diagnostic is thus what caused sustained lack of new-tile reward:
missed turns, obstacle contacts, a spin, slow/stationary behavior, a wrong-way loop,
or failure to cross the line. Episode summaries alone cannot choose among them.

Three particularly useful counterexamples prevent a misleading diagnosis:

1. The selected v5 author seed-50 screen has three failed episodes already at
   >=95% progress and qualified to finish. Two reach progress 1.0 with zero final
   damage but retire `off_track`, with raw rewards 724.982 and 745.309. Its consumed
   confirmation contains another progress-1.0, zero-damage `off_track` failure,
   raw reward 727.843. Complete coverage is demonstrably not a finish. These facts
   do not establish the car's path or which crossing condition it missed.
2. That is NOT the main explanation for all failures: only 4 of all 163 v5 screen
   failures reach >=95% progress. Failure counts in bins `<.25`, `[.25,.5)`,
   `[.5,.75)`, `[.75,.95)`, and `>=.95` are 61, 48, 37, 13, and 4. A finish-line
   specialist alone leaves most failures untouched. These bins pool checkpoints
   and entropy arms and describe diagnostic coverage, not a finalist's failure rate.
3. More training is not monotonically better: long-horizon RLPD seed 10 goes from
   7/24 deterministic screen finishes at 65,536 decisions to 0/24 at 131,072;
   seed 11 goes from 9/24 to 12/24. Seed 10 nevertheless has 15/67 finishes among
   TRAIN episodes ending in the last 32,768 decisions. Training finishes cannot
   replace deterministic deployment selection. TRAIN policy stochasticity, changing
   weights, and different roads mean this is NOT a controlled test of action noise.

Actual raw-return overlap also exists. The frozen v5 author seed-51 65,536-step
screen actor has a failure with reward 651.965 and a finish with reward 528.405,
on different cells. This independently illustrates why reward ranking is not
finish ranking; it is not a same-state proof that changing reward will help.

Prior-data composition was recomputed from episode manifests:

| Dataset | Complete episodes | Distinct finished geometries | Transitions | Transitions belonging to successful episodes |
|---|---:|---:|---:|---:|
| Pilot v2 | 19 | 3 | 8,170 | 1,541 (18.862%) |
| Long-horizon v1 | 36 | 6 | 16,305 | 3,048 (18.694%) |
| Entropy v5 | 35 | 8 | 15,915 | 4,094 (25.724%) |

V5's 16,384-decision collection cap also spent 469 decisions on a discarded
partial episode. Its 35 complete episodes consist of 8 finishes, 24 `off_track`,
and 3 crashes. The old collection talk message says six finishes; the immutable
manifest, result, and recount agree on EIGHT. These percentages are fractions of
transitions whose parent episodes succeeded, not fractions of useful recovery
labels or evidence that all other transitions are harmful.

The v5 target intervention did affect optimization: logged last-32,768-decision
mean entropies were -1.48449/-1.50760 for author seeds 50/51 versus
1.48934/1.49487 for positive-target seeds. Mean temperatures were .10586/.11169
versus .79609/.78332. Recorded scalar metrics were finite, execution mismatch
zero, and source mix exactly 32:32. No numerical collapse was visible at the log
cadence; this is not an exhaustive check of every update.

Finally, the paired gains are less uniform than a pass/fail gate suggests. Long-
horizon confirmation has treatment-only/control-only/both finishes of 3/2/0 for
seed 10 and 9/4/3 for seed 11. V5 author versus positive has 14/3/3 for seed 50 but
6/5/1 for seed 51. The latter is only a one-cell net gain, with mixed per-geometry
effects. Passing both aggregate seed gates is not dominance on every road.

## 4. Failure-First Pipeline

### 4.0 Diagnose before choosing the intervention

G0 is no longer just bookkeeping before a predetermined rescue experiment. Its
required output is an RLPD-specific, actor/action-mode-bound failure taxonomy,
geometry-clustered frequencies, uncertainty/missingness, and a proposed minimal
intervention with a falsification rule. Use ordinary starts and successful-parent
controls, not only a retrospectively selected set of dramatic failures.

In a separately authorized TRAIN-only diagnostic, record permitted pixels and
executed steering/gas/brake alongside diagnostic-only speed/heading/road position,
new tile visits, summed raw decision reward, consecutive-negative-reward counter,
contacts/damage, elapsed/remaining time, and finish-crossing phase. Preserve the
distinction between cumulative visited-tile progress and directed spatial motion.
Inspect the first observable loss and preceding window, not merely the final
retirement frame after 101 consecutive negative summed decision rewards. Freeze
event definitions and tie/missingness rules; log observations separately from inferred mechanisms.
Privileged telemetry must never become undeclared inference input.

| RLPD-specific precursor pattern to investigate | First targeted falsification | Do not infer |
|---|---|---|
| Repeated opening-turn control loss without preceding contact/damage | A narrowly specified pre-departure turn-entry or heading correction, followed by unchanged-student continuation | DrQ's cause automatically transfers; all-road braking or steering damping helps |
| Contact/damage followed by degraded control | Separate contact avoidance from post-damage recovery; preserve real damage/history and test actual finish rescue versus harm | Low terminal-crash counts make obstacles irrelevant; a nominal-lap teacher is a recovery oracle |
| Road-bound stall/loop or qualified missed crossing | Directed-progress/finish-phase action comparison; valid suffix support or completion-value work only if the evidence supports it | `off_track` proves grass, or progress 1.0 identifies which crossing condition failed |
| Useful privileged-state actions are poorly distinguishable from current pixels | Compare same-stack, permitted-history, and privileged diagnostic probes on held TRAIN geometries before selecting memory/supervision/distillation | Any probe error proves missing information rather than data/optimization error; more memory guarantees a solution |
| Mixed, sparse, or unresolved evidence | A bounded comparison of the smallest competing explanations; preserve unknown cases | One dominant causal mechanism must be forced or a broad recovery search is justified |

Diagnostic prevalence is not expected treatment benefit. Choose the first test
using observed coverage, feasible correction, possible harm, and cost; report the
choice as a hypothesis, not an identified optimum. Neither a narrow failure cluster
nor a temporal association establishes causality without an intervention contrast.

An initial proposed diagnostic cap is 12 fresh TRAIN geometries x two frozen actors
x one predeclared obstacle variant each x at most 2,000 decisions: at most 48,000
agent decisions, no learner updates and no interventions. This is a separate
diagnostic proposal, not permission to start, and not an additional free allowance
inside an existing frozen protocol. Count reset/warmup raw frames separately; keep
budget-censored results unknown. Do not extend the pool until a preferred pattern
appears. No seeds or actual runs are allocated here.

### 4.1 Prove recoverability before training

This is the recovery/prevention branch selected AFTER G0 supports a relevant
hypothesis. It is not the only possible G0 exit and need not precede a clearly
indicated representation or finish-phase feasibility test.

Use only TRAIN/TRAIN-DIAGNOSTIC geometries allocated by a new audited protocol.
Freeze the source actor, rescuers, intervention durations, anchor selection rule,
candidate count, and compute cap. Do not inspect old blind roads to choose these.
Policy identity must include the actor hash AND action-selection mode, observation/
action adapters, and reset/history contract. Use the deterministic exported policy
as the reference for deployment claims; stochastic training control with the same
weights is a different continuation policy.

At each anchor, restore the exact preceding state by reset plus the same executed
action prefix. Verify observation/reward/termination parity and relevant local
state, including damage effects, the per-tile visited bitmap (not just its count),
the full finish-tracker phase/history, and the outer TimeLimit elapsed decisions
(not a restarted anchor deadline). Rebuild
any proposed recurrent policy state by the same visual/action history. Do not
teleport the car or reset counters. Existing DrQ prefix parity is a precedent,
not a proof that a new RLPD pipeline has parity.

Compare branches from the SAME state:

- Continue the frozen RLPD actor unchanged.
- Execute a short, predefined recovery behavior, then hand control back to that
  same actor and continue until real finish, failure, or the original deadline.
- Retain the full successful and failed branch records, not just the best branch.

Anchor near the first observable departure, collision, or progress stall, plus
predeclared earlier offsets. Only probing the last few frames of the 101-step
retirement countdown can miss the recoverable moment. Include ordinary states
and states from trajectories that would finish, so intervention harm is measured.

Candidate rescue behaviors should be coherent maneuvers: stabilize heading and
rejoin a traversable corridor, brake BEFORE a tight curve, or pass an obstacle on
a chosen side. A geometry-aware controller or trajectory search may generate labels
during training, but must itself pass this gate. A full-state teacher is not an
assumed oracle; approximate curvature/friction speed formulas are design priors,
not exact Box2D guarantees. First try at most one or two fixed rescuers instead of
searching a large menu until retrospective successes appear.

Report the paired finish transition table:

| Baseline finish | Recovery finish | Meaning |
|---|---|---|
| no | yes | Actual rescue |
| yes | no | Harmed finish |
| yes | yes | Preserved finish; only then consider time/damage |
| no | no | Not a completion rescue, even if progress or survival improves |

The main diagnostic is `rescued_failures - harmed_finishes`, with geometry-cluster
uncertainty and all failures retained. Best-of-K branching is an **oracle upper
bound**: choosing the successful branch AFTER seeing outcomes is not a deployable
policy. A failure to find a rescue is evidence against this candidate set/anchor
budget, not proof that the state is physically unrecoverable.
Only fully observed paired outcomes enter the rescue/harm table. If the collection
cap censors either branch, retain the pair as unknown and report missingness;
do not count the observed member as an unpaired win or the censored one as failure.

### 4.2 Prove visual realizability and deployment selection

Learn or specify the ENTIRE deployed control path using permitted observations.
A visual trigger that calls a privileged full-state rescuer does not pass: both
the trigger and the resulting recovery actions must be realizable without that
state, or a pixel-only student must successfully reproduce them. Split data by
entire geometry and episode, never by neighboring frames. Test on separate
TRAIN-DIAGNOSTIC geometries under normal start-to-finish rollouts, where the policy
does not know future failure.

For a claim about the first unchanged RLPD actor, this means the same four-frame
input, not an undeclared longer history or clock. A selector using permitted action
history/self-maintained time can be a valuable separate feasibility probe, but its
success cannot establish realizability by the original feed-forward actor. Freeze
that richer representation as a separate treatment if it is needed.

Separate two gaps: **no feasible rescue exists in the tested menu** versus **a
rescue exists but the image-only policy cannot choose it reliably**. The latter
motivates memory/perception work, not more labels from an inaccessible oracle.
Opposite-side obstacle passes must remain distinct coherent targets; averaging
left and right successful controls may point directly at the obstacle.

For the first deployable route, distill validated behavior into the actor rather
than shipping a simulator, teacher, branch search, or privileged risk critic.
An inference-time safety supervisor is a different experiment requiring its own
CPU/export contract and evidence of harmless switching.

### 4.3 Test the data intervention in isolation

First compare normal prior collection with recovery-targeted prior collection
under the same RLPD architecture, raw objective, student decision/update budget,
checkpoint opportunities, and source split. Match or explicitly report ALL prior
collection costs, including replayed prefixes and rejected branches. Otherwise a
benefit could simply reflect much more simulator use.

Then, separately, test geometry/episode and recovery stratification WITHIN the
offline half. A candidate batch is 32 online + 16 ordinary prior + 16 recovery;
these are proposed numbers, not an established optimum. Preserve ordinary driving
coverage and failure negatives. Disclose sampling probabilities; oversampled risk
data cannot be treated as natural-prevalence calibrated probabilities without
correction and testing on the natural rollout distribution.

A later, separate auxiliary imitation term may anchor only demonstrated useful
recovery actions. It must not blindly imitate all teacher actions or use an
unvalidated Q filter as evidence that the teacher is better. Such a term is a
RLPD-derived variant, not faithful original RLPD.

**Action accounting is non-negotiable:** Bellman transitions contain the action
actually executed. Store policy proposals and supervised rescue labels separately.
Do not credit a proposed unsafe student action with the outcome of a substituted
teacher action when the final policy will have no teacher.

## 5. Completion Objective

This remains a conditional research branch, not an obligatory next treatment after
any low completion count. Missing finish reward proves possible objective mismatch,
not that this is the dominant cause of the candidate's losses of directed progress.

Keep the existing raw-return critic as a diagnostic/comparator. Add a distinct
finite-horizon success value in a separately declared treatment:

```text
F_pi(H_t, h_t, a) = P(real finish before failure/deadline | H_t, h_t, a, then pi)
h_t = remaining decisions under the declared evaluation budget

y_F = 1                                      if this transition truly finishes
      0                                      if it fails or exhausts that budget
      E[a' ~ pi(.|H_next,h_t-1)] F_target(...) otherwise
```

For deterministic deployment, `pi` means the exported `tanh(mean)` control law
with its exact adapters/reset behavior, so the expectation uses that single action.
Bind this mode in every continuation-policy identity and calibration rollout.
Do not estimate the stochastic learner's `F_pi` and label it the deterministic
export's finish probability just because their weight hashes match. If stochastic
policy improvement is studied, its deployment-mode gap needs a separate test.

`H_t` denotes permitted history or its learned representation, not privileged
environment state. Finish takes precedence when a successful transition also has
`truncated=True`. The finite remaining horizon makes undiscounted reachability
well-defined: this is NOT setting the existing continuing raw-return SAC discount
to 1. No entropy bonus belongs in a probability target.
The training-only critic can receive the recorded remaining horizon while the
actor retains its existing input contract. If time/history is also added to the
actor, compare against a raw-objective control with the SAME added inputs before
attributing a gain to the completion objective.

The replay currently lacks explicit per-transition `finished` labels. Add a
future data contract distinguishing true success, physical retirement, actual
evaluation timeout, and collection-budget censoring. A collection cut short before
the task deadline is **unknown**, not a failed lap. The raw-return critic's
nonterminal-time-limit bootstrap mask must not be copied into this objective.

Monte Carlo outcomes of teacher or rescue-composite rollouts estimate THAT
behavior's success, not the current student's success probability. A rescued
prefix followed by frozen actor `pi_0` yields evidence about that intervention
under `pi_0`. It is not a value label for an arbitrary later actor. Preserve the
continuation-policy hash, and validate off-policy learning using fresh TRAIN
rollouts of each frozen policy version. Naive multi-step teacher returns introduce
off-policy bias; do not call them exact credit propagation.

Before using `F` to change the actor, require calibration against a constant
base-rate/horizon baseline and correct ranking of held-geometry paired branches.
Report Brier/log loss, reliability by horizon/geometry family, and action-ranking
errors; a low image loss or tiny ensemble variance is insufficient.

The policy-improvement target should then be **finish probability first**, with a
bounded policy-change constraint to reduce extrapolation. A precise research
formulation is maximize estimated `J_F(pi)` subject to a predeclared KL step bound
to the previous actor; retain baseline behavior on unsupported states. Only after
the reliability gate passes should a separate stage improve speed subject to a
non-regression constraint on `J_F`. Approximate neural values and KL bounds do not
make these true-probability constraints certified.

This changes the author SAC objective and is explicitly a new recipe. Do not
silently reuse an entropy coefficient tuned to large raw Q values with a [0,1]
completion Q: the relative scale can overwhelm the success signal. Nor does a
large arbitrary `finish_bonus - collision_penalty` establish lexicographic
completion preference. Compare objective variants on identical frozen data and
budgets, and retain raw reward only as a separate reported metric.

## 6. Conditional Extensions

| Extension | Evidence needed first | Main risk |
|---|---|---|
| Small action-conditioned recurrent actor | Observable-history probes outperform four-frame probes on geometry-held TRAIN data, followed by a closed-loop gain | Hidden-state/burn-in mistakes and added export complexity; memory is not automatically useful |
| Auxiliary road/obstacle, heading, speed, slip-response targets | Pixel features fail held-geometry control-relevant prediction despite adequate rescue coverage | Privileged labels may encode information absent from images; pixel accuracy need not improve action ranking |
| Asymmetric full-state completion/safety critic with pixel actor | It improves actual student decisions over a pixel critic in a separate ablation | A pure-state critic replacement would remove the present critic-trained visual encoder path; preserve a visual learning objective |
| Finish-suffix/backward curriculum | Real successful suffixes exist and prefix replay reproduces their full history | Teleporting near the finish loses visited tiles, damage, tracker history, and time; suffix success is not full-lap success |
| Larger UTD or longer off-policy returns | Stable value/action ranking, sufficient coverage, and explicit compute controls | Replaying a narrow wrong target faster can worsen overfitting; uncorrected teacher multi-step targets are biased |

For a valid suffix curriculum, obtain the anchor by executing the real prefix,
retain its spent time/damage/visited tiles, and progressively move the learning
anchor earlier. Always retain normal-start episodes and evaluate only unassisted
full laps. Prefix simulator cost counts in the budget. This is an adaptation of
reverse curricula, not their original arbitrary-reset assumption.

## 7. Counter-Evidence And Rejected Shortcuts

- DrQ residual-options pilot v1 produced 5/32 finishes versus 5/32 baseline, with
  four paired wins and four losses; progress fell from about 0.685 to 0.570.
  Margin variant v2 reduced intervention fraction from about 0.742 to 0.138 but
  finished 3/32 versus 5/32. Less intervention was not sufficient.
- Its prefix study already did exact-prefix options followed by base-policy
  continuation. The same COAST option both rescued and derailed finishes. Two
  BRAKE branches reduced progress from 0.6360 and 0.5037 to 0.1544. Therefore,
  branching itself is not new; the proposed differentiation is student-specific
  failure-state coverage, finish-based harm testing, visual realizability, and
  learning the resulting behavior.
- DrQ teacher-replay r3 stopped at data coverage (four versus three distinct finish
  geometries at its fixed cap), with no learner/evaluation. It is not evidence
  that training on better prior data fails.
- Entropy v5 favors the author target in both matched seeds. Do not relabel +1.5
  as a bug fix or make further entropy search the main completion strategy.
- "Never crash" is not "finish": a stationary agent can avoid immediate collisions
  but retire from no progress. A short-horizon survival score is not a finish score.
- Shared-encoder Q variance, sigmoid outputs, or lower ensemble quantiles are not
  certified safety bounds. Learned risk needs external outcome calibration.
- More training, larger images, longer frame stacks, action damping, and a new
  algorithm name are not explanations of the existing failures by themselves.

Sources: [residual pilot](../../experiments/drqv2-residual-options-pilot-result.json),
[prefix branches](../../experiments/drqv2-residual-options-prefix-branch-v1-result.json),
[teacher replay r3](../../experiments/drqv2-teacher-replay-v1-r3-result.json).

## 8. Proposed Gates

These are proposed decision criteria to freeze BEFORE future execution, not an
already-frozen protocol, seed allocation, or authorization. Never enlarge a cap or
weaken a gate after observing failure. A gate can be inconclusive from insufficient
coverage without proving the entire direction wrong.

| Gate | Isolated question | Advance / reject |
|---|---|---|
| G0: Identity and failure diagnosis | Is the exact local candidate/runtime identified, and what first precedes its loss of directed progress? | Actor/mode/contract parity, a geometry-clustered failure taxonomy with successful controls and mixed/unknown cases, then a bounded falsification choice; resume checks apply when resumption is used, prefix parity before branching |
| G1: Targeted-intervention feasibility | Does the G0-selected prevention/recovery/phase intervention improve actual finishes without excess harmed finishes? | Positive paired net finish benefit across geometry families and original state/time preservation; recovery uses Section 4.1, not an automatic default for every G0 outcome |
| G2: Visual deployability | Can the entire deployed control path realize the benefit from its declared visual input/history without hindsight or privileged recovery actions? | Benefit remains on separate TRAIN-DIAGNOSTIC geometries and normal-start rollouts; reject a visual trigger with a privileged rescuer or oracle best-of-K alone |
| G3: Prior-data treatment | Does targeted recovery data help otherwise unchanged RLPD? | At least two independently trained seeds improve against their matched controls; report matched geometry-cluster intervals and collection costs |
| G4: Completion value/objective | Does a calibrated finish value improve decisions beyond the data treatment? | First pass policy-specific ranking/calibration, then paired full-lap improvement; reject on calibration failure or raw-Q/entropy scale confounds |
| G5: Robust replication | Is the resulting recipe reliable, not merely a nonzero-finisher? | Proposed development target: >=90% pooled finish, each of at least three learner seeds >=80%, no predeclared geometry family below 80%; retain uncertainty rather than treating these thresholds as a statistical proof |
| G6: Frozen confirmation/blind | Does ONE preselected actor generalize? | Freeze actor on screen; fresh confirmation and one untouched blind; no finalist reselection or tuning after either |

For the recovery branch only, an initial G1 study should be bounded rather than
another unbounded search; the G0 diagnosis must first justify selecting it:
propose 12 new TRAIN geometry seeds, two frozen RLPD source actors, at most two
predeclared anchors per episode and two rescuers, with a hard total of 131,072
simulator decisions INCLUDING repeated prefixes/continuations. Require evidence
on at least eight distinct geometries; otherwise report inconclusive, not a pass.
Also require, per source actor, fully paired controls on at least three distinct
successful-parent geometries and three distinct failed-parent geometries. These
are proposed minimum coverage numbers, not a precision guarantee. If the fixed
pool/cap does not provide them, G1 is inconclusive; do not pass on rescue counts
without actual opportunities to measure harmed finishes.
Freeze the sampling/order/cap accounting before collecting; scarce successful-parent
controls cannot be replaced by hindsight-selected easy roads. No concrete seed
numbers are allocated by this report.

G3/G4 must not change data, replay proportions, actor memory, loss, and UTD together.
Use fresh independently trained baseline runs for causal comparisons; comparisons
against an old candidate with different prior/training history are performance
benchmarks, not algorithm-only effects. When extra teacher/search compute cannot
be matched, report the confound explicitly rather than claiming equal-budget gains.

### Separate package handoff gate

Before attributing a future local-versus-deployed discrepancy to driving behavior,
bind the exact handed-off ZIP to its actor and source. This is distinct from G0's
local identity check and does NOT require producing a release package before
ordinary local failure diagnosis.

Verified existing evidence is stronger than a generic smoke: the seed-11 candidate
and inspected screen/confirmation actor snapshots have SHA-256
`f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1`; the inspected
confirmation records Python 3.11.14 / Torch 2.1.0+cpu with deterministic reload and
zero operational failures. Its `archive_sha256` identifies copied actor bytes, not
a released ZIP (`evaluate_policy.py:192-214,802-839,1007-1043`). The archived root
`Agent` path is meaningful CPU inference evidence and must not be dismissed.

However, `tests/test_submission_package.py:117-166` uses a newly initialized
seed-23 actor and dummy provenance, not this candidate. `package_submission.py:390-451,499-526`
smokes repeated zero observations for shape/finite/reset/tag/timing behavior;
it does not compare the extracted package's actions to this local candidate on
the same input corpus. The inspected submission ledger/manifest identifies a
baseline1 artifact, not a seed-11 ZIP parity receipt. That is an evidence gap,
NOT proof the owner never packaged or submitted a model elsewhere.

The future handoff check should identify the actual ZIP, record its hash/member
hashes/dependency inventory, bind `model.pt` and declared `agent.py` revision, and
compare frozen evaluator Agent versus EXTRACTED package Agent in clean, pinned
Python 3.11 / CPU Torch 2.1 processes. Use identical prehashed synthetic and, if
authorized, existing TRAIN-only observation sequences and reset schedules; no
new driving or blind corpus is needed. Record two fresh reload traces, official
float32 actions, finite/bounds checks, maximum absolute error and a predeclared
equality criterion, runtime/import provenance, latency and whole-worker memory.
Isolate import paths/user-site settings to prevent accidental workspace fallback.
Passing finite fixtures is package attribution evidence, not a full-lap guarantee
or official server validation. No such ZIP test or new package was run in this task.

### Reliability standard

For a strong single-model reliability statement, make an independent geometry the
statistical unit. One defensible cluster outcome is "this geometry finishes under
ALL of its predeclared obstacle variants." For `n` IID clusters and zero failures,
the exact one-sided 95% binomial lower bound is `0.05 ** (1/n)`:

- Even eight perfect geometry clusters give a lower bound of only 0.6877.
- 59/59 perfect, independently sampled geometry clusters give a bound of 0.9505.
- 299/299 give a bound of 0.9900, still not 100% certainty.

These are sample-size illustrations, not achieved results. With failures, compute
the appropriate exact/cluster interval; do not keep collecting until a bound passes.
The IID argument applies only to a declared sampling distribution, not arbitrary
private tracks or repeated variants. For stratified families, report family and
mixture estimates with their correct sampling assumptions. A five-seed result is
also not five independent teacher datasets when the prior is shared.

Compounding matters: if each of 20 sequential segments has conditional survival
0.95, lap survival is `0.95 ** 20 = 0.3585`; 0.995 per segment gives 0.9046. These
are hypothetical conditional probabilities, not a measured 20-segment HAIC model.
The useful lesson is to target the worst recurring failure states, not average
frame-level action accuracy.

## 9. Validation Performed

### Official contract and synthetic tests

The current competition site was reachable, but its fetched page was only an app
shell. The official Participants README and commit API were read on 2026-09-26;
main was `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`. The README confirms `variables-6`,
pixel-only observations, completion semantics, and Linux/Python 3.11 CPU inference.
No schedule, quota, upload, or official evaluation action is inferred here.

The following local files exactly matched the corresponding raw official files
at that revision by SHA-256, ruling out a divergence in these inspected mechanisms:

| File | SHA-256 |
|---|---|
| `env_wrapper.py` | `8687215412da0a1f34534623d5050030d85c629a70cb922e9d2199c513a3941e` |
| `core/finish_line.py` | `7c16dda1378dcd65aed46d9da96608aa055124c2f27d8fb00c309078361d4204` |
| `core/vendor/car_racing.py` | `fcd3c36087d7b2721efcd27538b2c9576fa48a77b276ca17c2185b410b0a8c69` |

Executed existing tests, with no driving environment:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_local_contract.TestFinishLineTracker tests.test_local_contract.TestEnvironmentContract -v
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B -m unittest tests.test_rlpd_model tests.test_rlpd_replay tests.test_rlpd_augmentation -v
```

Results: 11 finish/wrapper/damage synthetic tests and 26 RLPD model/replay/augmentation
tests passed. These validate selected contracts, not the proposed policy's efficacy.
The inspected RLPD core learner/replay files match frozen long-horizon/v5 snapshots;
HEAD `d94f77110080ae5755252d46300a5b587b4e020f` alone is not full provenance because
the shared worktree contains substantial pre-existing changes.

### Two exact-resume defects reproduced, not fixed

An independent static audit found two gaps despite those passing tests. A bounded
in-memory synthetic probe then reproduced both:

1. `FrameStackReplay.state_dict()` omits the valid-row permutation. A two-step
   terminated episode has live order `[1,0]`, restored as `[0,1]`. The same stored
   RNG state therefore yields different sampled transitions.
2. `_has_valid_stack_prefix()` checks `len(STACK_SHAPE)-1 == 2` predecessors rather
   than the three needed for a four-frame stack. After ring overwrite, restore
   admits a row whose oldest frame is gone, and stack reconstruction raises.

Locations: `haic/algorithms/rlpd/replay.py:155-164,200-227,271-288,330-343`.
Existing restore tests cover a single transition, not the multi-row ordering or
restored overwritten prefix (`tests/test_rlpd_replay.py:106-122,157-167`).

Reproducer body, executed with `python -B -c` and the same no-bytecode/CPU-only
environment settings above; it writes no file and uses only synthetic arrays:

```python
import copy
from tests.test_rlpd_replay import transition
from haic.algorithms.rlpd.replay import FrameStackReplay

r = FrameStackReplay(6, seed=8, source="online")
r.add(transition(step_index=0))
r.add(transition(step_index=1, terminated=True))
q = FrameStackReplay(6, seed=999, source="online")
q.load_state_dict(copy.deepcopy(r.state_dict()))
print(r._valid_rows[:r.valid_count], q._valid_rows[:q.valid_count])
print(r.sample(8)["indices"], q.sample(8)["indices"])

r = FrameStackReplay(5, seed=3, source="online")
for i in range(5):
    r.add(transition(episode=3, step_index=i, base=30, terminated=i == 4))
r.add(transition(episode=4, step_index=0, base=60, geometry_seed=8001))
q = FrameStackReplay(5, seed=999, source="online")
q.load_state_dict(copy.deepcopy(r.state_dict()))
print(r._valid_rows[:r.valid_count], q._valid_rows[:q.valid_count])
print(q._has_valid_stack_prefix(3))
q._observation_stack(3)
```

Observed: live/restored order `[1,0]` / `[0,1]`; samples
`[0,1,1,0,1,1,0,0]` / `[1,0,0,1,0,0,1,1]`; overwritten valid rows `[4]` / `[3,4]`;
prefix check `True`, then `RuntimeError: replay stack crossed an episode or ring boundary`.
The diagnostic intentionally exited nonzero. No affected prior training resume was
established: these are future-resume correctness blockers, NOT an explanation for
the existing finish rates or a refutation of actor-only CPU reload evidence.

### Evidence boundary

The nonblind recount used `episodes.jsonl` under these existing evaluation roots:

```text
evaluations/20260924T184536689584Z_pixel-rlpd-offpolicy-pilot-v2-screen/
evaluations/20260925T002348999661Z_pixel-rlpd-long-horizon-followup-v1-screen/
evaluations/20260925T025158881289Z_pixel-rlpd-long-horizon-followup-v1-confirmation/
evaluations/20260925T025455593182Z_pixel-rlpd-long-horizon-followup-v1-confirmation/
evaluations/20260925T025850459044Z_pixel-rlpd-long-horizon-followup-v1-confirmation/
evaluations/20260925T030142060474Z_pixel-rlpd-long-horizon-followup-v1-confirmation/
evaluations/20260925T162014319049Z_pixel-rlpd-entropy-target-ablation-v5-screen/
evaluations/20260925T191038725317Z_pixel-rlpd-entropy-target-ablation-v5-confirmation/
evaluations/20260925T200824835493Z_pixel-rlpd-entropy-target-ablation-v5-confirmation/
evaluations/20260925T201206242723Z_pixel-rlpd-entropy-target-ablation-v5-confirmation/
evaluations/20260925T201631697790Z_pixel-rlpd-entropy-target-ablation-v5-confirmation/
```

Recount rules: filter `repeat == 0`, count explicit `finished` and the recorded
`termination_class` (`finished`, `off_track`, or `crash` here); never turn
`progress >= .95` into success. Prior manifests use `retire_reason` for failures.
Pair actors by
`(track_id, seed)` within the SAME study, cluster geometry by seed, and use the
frozen actor/checkpoint identities. Check repeat 1 only for deterministic agreement,
not in the denominator. Sum complete prior episodes from each study's
`runs/<study>/prior-data/manifest.json`; student episode/metric logs remain under
their individual run directories. Last-quarter TRAIN counts select episodes by
ending global step and are not a fixed-policy matched evaluation.

Provenance hashes independently checked during this audit (manifest SHA is the
FILE hash, not the separately recorded dataset-content digest):

| Study | Protocol JSON SHA-256 | Prior manifest file SHA-256 | Screen episodes file SHA-256 |
|---|---|---|---|
| Pilot v2 | `ba2375426de3d1d690ea598dd14835523290791b0e91d0c889fae4799460bed0` | `45fda2393348c95cee4b5069838063a0f9011e980307922676059012b56a94c7` | `b4306aac2b72724d7f67e80305da6ad01249024b191560c21eb6ee26b42896ef` |
| Long-horizon v1 | `2960b561f24f6dc23a42dd0d49fd1ef7a5a0864e0e3f7543cb6920bc465a4329` | `4e18cd373ffe79e515f343bf8b5f6385916e817d0cd06657e438a2a10f3dabcf` | `976b573f2d33463ebb49da4dcab91f1c9c8914dd2f166d88b5caecad7e97a5f2` |
| Entropy v5 | `2e7df4152e2282bcd34a6d2a160737cc13c6f1a03d769bd69832e1fc51326546` | `2dad0686e8c7c84954f096375f31495673bbad25c988034613b4145cb97fd770` | `ceb22013b235207184ed05ea84d3b6032e15178f960b4b77093175bd6d0fe383` |

Compact reproduction of the central v5 recount, from the repository root using
standard-library Python only. This reads existing files and does not run an agent,
environment, or evaluation:

```python
import hashlib
import json
from collections import Counter
from pathlib import Path

roots = [
    "20260925T162014319049Z_pixel-rlpd-entropy-target-ablation-v5-screen",
    "20260925T191038725317Z_pixel-rlpd-entropy-target-ablation-v5-confirmation",
    "20260925T200824835493Z_pixel-rlpd-entropy-target-ablation-v5-confirmation",
]
for root in roots:
    raw = (Path("evaluations") / root / "episodes.jsonl").read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    rows = [r for r in rows if r["repeat"] == 0]
    print(root, hashlib.sha256(raw).hexdigest(), len(rows),
          sum(r["finished"] for r in rows),
          Counter(r["termination_class"] for r in rows),
          len({r["seed"] for r in rows}))

p = Path("runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/manifest.json")
raw = p.read_bytes()
m = json.loads(raw)
episodes = m["episodes"]
print(hashlib.sha256(raw).hexdigest(), len(episodes),
      sum(e["steps"] for e in episodes),
      len({e["geometry_seed"] for e in episodes if e["finished"]}),
      sum(e["steps"] for e in episodes if e["finished"]),
      m["discarded_decisions"])
```

Expected canonical counts: screen 29 finishes / 152 `off_track` / 11 crashes;
author seed-50 confirmation 17/12/3; author seed-51 confirmation 7/23/2.
Each root has eight geometry seeds. Prior output is 35 episodes, 15,915 stored
transitions, eight finished geometries, 4,094 transitions from finished episodes,
and 469 discarded partial decisions. The main audit independently reproduced all
three screen aggregates and all three prior-manifest totals after the child audit.

An independent report review also tightened three design gates: deterministic
versus stochastic continuation identity, pixel-only access for ALL recovery actions
rather than the trigger alone, and minimum successful-parent harm-control coverage
with explicit exclusion/reporting of censored pairs. These are design corrections,
not implementation changes or new policy evidence.

The validation establishes implementation facts, objective mismatch possibilities,
arithmetic, prior negative evidence, and a falsifiable research design. It does
not establish that F2R-RLPD raises completion: that requires G1-G6 after separate
implementation/execution approval. None of the proposed methods has been trained
or driven in this task.

## 10. Primary References

| Reference | What supports this proposal | What it does NOT establish |
|---|---|---|
| [Ball et al., RLPD, 2023](https://arxiv.org/html/2302.02948v4), [author code](https://github.com/ikostrikov/rlpd/tree/c90fd4baf28c9c9ef40a81460a2e395092844f88) | Symmetric prior/online sampling, normalized ensembles, environment-sensitive design choices | HAIC completion; a ten-head safety bound; imitation/pretraining as original RLPD |
| [Ross et al., DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html) | Label learner-visited states to address sequential distribution shift | Perfect expert labels, low realizable visual error, recoverability, or unseen-track guarantees; its kart example is not this damage-limited task |
| [Thananjeyan et al., Recovery RL, 2021](https://arxiv.org/html/2010.15920v2) | Separate task behavior and learned recovery regions; use offline constraint data | Finish probability or a teacher-free guarantee. Its task-action relabeling is for a shielded MDP and must not be copied into unshielded student replay |
| [Pinto et al., Asymmetric Actor Critic, 2017](https://arxiv.org/html/1710.06542v1) | Training-only full-state critics with visual actors; evidence that full-state expert DAgger can underperform asymmetric learning | Automatic transfer to RLPD/HAIC; resolving genuinely unobservable action ambiguity |
| [Florensa et al., Reverse Curriculum Generation, 2017](https://proceedings.mlr.press/v78/florensa17a.html) | Expand learnable starts backward from successful goals | Safe arbitrary resets in a task with irreversible damage, visited tiles, and finish-tracker history |
| [Hsu et al., Safety and Liveness Guarantees through Reach-Avoid RL, 2021](https://arxiv.org/html/2112.12288v1), [author code](https://github.com/SafeRoboticsLab/safety_rl) | Goal-reaching and avoidance as distinct from ordinary reward; policy validation/supervision | Neural-pixel certification. Contractive discounted reach-avoid margins are not calibrated finite-horizon finish probabilities |

Official sources checked: [competition site](https://scholarships-hardwood-headers-influenced.trycloudflare.com/),
[Participants at the checked revision](https://github.com/2026-HAIC/Participants/tree/1c11db8afc2fbfcfb610672b7ee0ecd122c97741).

**Revised decision:** first bind the local candidate/runtime and classify its loss
of directed progress on bounded TRAIN-only diagnostics. Then choose one targeted
falsification, not recovery by default. Keep harm-aware recovery, pixel-only
realizability, completion value, and memory as conditional branches, testing each
supported learning change in isolation. Verify the exact package separately before
deployment attribution. Preserve unknowns and negative results rather than claiming
that either the original recipe or the peer suggestion guarantees completion.
