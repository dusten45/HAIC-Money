# DrQ-v2 Rival Research Proposal: Frozen Driver Options and Road Coverage

## Status and Decision Boundary

**Status: historical design proposal, not the active plan.** The active work remains
[`active/dreamerv3-recovery-strategy.md`](active/dreamerv3-recovery-strategy.md).
This file describes two **independent** alternatives, not a combined treatment:

1. **A (priority):** keep an existing DrQ-v2 driver frozen and learn a small,
   model-free controller that selectively changes its actions for a short time.
2. **B (separate fallback):** retain ordinary DrQ-v2 but change the distribution
   of *training roads at reset*, subject to evidence of undercovered road motifs.

### Outcome as of 2026-09-25

An A-family, separately frozen **single-driver** [residual-options pilot](../../experiments/drqv2-residual-options-pilot-result.json)
completed two development iterations: option/base finishes were 5/5 then 3/5
on the same consumed 32-cell development grid. It stopped without the proposed
two-driver promotion gate, fresh confirmation, blind evaluation, or model promotion.
The subsequent [training-only prefix-branch diagnostic](../../experiments/drqv2-residual-options-prefix-branch-v1-result.json)
completed without promoting a full-episode policy. B's training-road **reset
reweighting** remains a separate, untested proposal; the completed
[training-only road-shape diagnostic](../experiments/drqv2-geometry-augmentation-v1.md)
does not test B. The design below predates these outcomes and does not authorize
training, opening blind cells, official submission, or model confirmation. A new
study would require separate authorization, a frozen protocol, and fresh audited
partitions; see [`run-experiment`](../workflows/run-experiment.md).

### 한눈에 보는 제안

- **A (우선):** 완주 사례가 있는 DrQ 운전자를 고정한다. 매 2번의 환경 결정마다
  작은 Q 제어기가 기존 행동 유지, 조향 보정, 가속 해제, 약한 제동 중 하나를
  선택한다. 원래 운전자만 쓰는 같은 도로의 평가와 비교한다.
- **B (별도 실험):** DrQ의 모델·보상·replay는 유지하고 훈련 episode의 도로
  seed만 사전에 정한 곡률 패턴 빈도로 다시 뽑는다. 먼저 훈련 도로의 특정
  패턴 노출이 실제 부족한지 조사하고, 부족하지 않다면 실험하지 않는다.
- **둘 다 아직 가설이다.** 정합성 검사, 훈련에 쓸 셀과 평가 셀의 분리,
  새로 동결한 비교 프로토콜, CPU/패키지 제한을 통과하기 전에는 결과나
  공식 성능을 주장하지 않는다. A와 B를 한 번에 적용하지 않는다.

### Why these are different research questions

The [active plan](active/dreamerv3-recovery-strategy.md) repairs DreamerV3
transitions/objectives, tests its world model, then conditionally considers a DrQ
teacher and episode-balanced *replay*. A preserves the existing DrQ policy at
inference and learns only a constrained, model-free intervention decision; it does
not train an RSSM, run CEM, copy DrQ into a student, or revise the DrQ steering
loss. B changes the *environment-reset road distribution*, not replay sampling,
custom-map evaluation, or Dreamer training. A and B must not be run together in a
first comparison: that would confound the intervention and data hypotheses.

| Fact from existing evidence | Interpretation and limit |
|---|---|
| Two distinct native pad-4 DrQ control actors had 4-7 finishes per 32-cell internal confirmation cohort. | There is a useful but limited frozen driver. These historical cohorts are **not** future matched controls or official scores. See [`MODEL_STATUS.md`](../results/MODEL_STATUS.md) and the [L2](../../experiments/drqv2-steering-logit-v1-result.json) and [padding](../../experiments/drqv2-augmentation-pad-v1-result.json) result records. |
| Steering-logit L2 and pad 1 each lost against their matched pad-4 controls. | Do not repeat either axis or infer that steering saturation caused every failure. See [`../decisions/INDEX.md`](../decisions/INDEX.md). |
| `retire_reason="off_track"` is assigned after more than 100 consecutive **negative-reward decisions**. | It is not a road-boundary measurement; neither braking nor steering rescue is established by the label. See [`env_wrapper.py`](../../env_wrapper.py). |
| A prior multidiscrete PPO run reported zero finishes in its recorded holdout; corridor-teacher imitation also had a small unsuccessful tune check. | Neither is a matched test of a frozen DrQ driver plus a small option controller. Do not relabel their results as evidence for or against A. See the [PPO run](../../runs/20260920-214647_ppo-cnn-multidiscrete-control-v1/best_model_metrics.json) and [pedal-policy history](archived/2026-09-21-haic-pedal-policy-plan.md). |
| A historical 32-cell grid crossed eight geometry seeds with four track IDs. | Thirty-two cells are not 32 independent road shapes; B's geometry-coverage mechanism remains untested. See the frozen [pad study protocol](../../experiments/drqv2-augmentation-pad-v1.json) and [`../evaluation/generalization-policy.md`](../evaluation/generalization-policy.md). |

## Shared Contracts and Non-Negotiable Limits

- Preserve `core/`, `env_wrapper.py`, and `damage.py` as the official local mirror.
  No performance change to environment physics, reward, obstacles, termination, or
  frame cadence. The present input is four `(84,84)` grayscale frames as
  `float32` in `[0,1]`; the action returned by `Agent.act()` is official
  `[steer, gas, brake]`, with bounds `[-1,1] x [0,1] x [0,1]`.
- One environment decision is one `CarEnvironment.step`, ordinarily covering four
  raw simulation frames. Discounting and budgets below count **decisions**, not
  raw frames. Keep the official raw environment reward in both studies; do not
  call raw reward, progress, damage, or an invented risk score an official rank.
- A submitted agent may use only its observation and its own stored episode state.
  Track IDs, geometry, simulator `info`, raw reward, future observations, and
  oracle labels are not available to it. Training-only metadata never leaks into
  `agent.py` inference. Both proposed paths require a single immutable package
  tested on CPU; an ensemble of per-track packages is not one model.
- The locally mirrored limits are Python 3.11, CPU-only, construction <=10 s,
  each reset/act <=5 s, and process peak RSS <=1,024 MB. See
  [`../competition/restrictions.md`](../competition/restrictions.md); recheck the
  official site and Participants repository before **any** approved external
  action. A passing local package test is not an official evaluation.
- Exclude previously consumed cells and every study's reserved blind geometry
  seeds **across all training track IDs**. The known consumed confirmation seeds
  `31101-31108`, `32101-32108`, and `33101-33108` are not fresh; reserved
  `31201-31208`, `32201-32208`, and `33201-33208` are not tuning cells. The
  historical pilot schedule is incomplete, so check run/protocol receipts before
  claiming any proposed new seed is unused. See
  [`../evaluation/generalization-policy.md`](../evaluation/generalization-policy.md).
- In a new study, choose exact unused `(track_id, geometry_seed, obstacles)` cells,
  disjoint training exclusions, checkpoint opportunity, source/runtime hashes,
  numerical thresholds and stop conditions **before the first result**. All
  comparisons use the same new cells and `repeat == 0` once per cell; multiple CPU
  reloads audit determinism rather than multiplying independent observations.

**Paired evaluation is not a built-in confirmation mode.** The current
`evaluate_policy.py` accepts **one** actor for confirmation/blind and requires
that exact actor's eligible, immutable screen receipt under the identical
protocol hash. A new study-specific orchestrator must evaluate each treatment
and control candidate separately on the same frozen screen and confirmation
matrices, bind every candidate's screen-to-confirmation lineage, then compare
canonical outcomes cell by cell in a *separate paired decision receipt*.
Checkpoint screens also need separate actor receipts. If a control has zero
screen finishes, the existing `--diagnostic-confirmation` path can collect a
matched operational comparator, but that confirmation is **non-promoting**
and cannot open blind for the control. A treatment with zero screen finishes
does **not** advance to promotion confirmation under this proposal. Do not
weaken the evaluator's single-actor, lineage, or blind gates to make a paired
table appear to work. The new validator must verify protocol/cell/actor/source
hashes, CPU eligibility and receipt status for every row before comparing.

## A. Frozen Driver Plus Short-Horizon Option Controller

### Hypothesis and scope

Some failures of an otherwise partially capable DrQ-v2 policy may be locally
avoidable by selecting a short, bounded correction or by choosing not to
intervene. This original action-choice hypothesis is **not established** by the
later limited pilot or diagnostic, and is not a claim that
`off_track` means the wheels left the road. The control is the **same frozen
driver with no option controller** on the **same cells**. Two existing baseline
actors, one per independent DrQ training seed, are described in
[`MODEL_STATUS.md`](../results/MODEL_STATUS.md); first verify their full export
provenance and hash. Their old confirmation numbers are historical context only.

This is an *add-on* comparison, not a claim of equal total training compute: the
old 131,072-decision DrQ training cost for each driver is inherited by both arms;
all additional option exploration and updates belong to the treatment and must
be reported separately. The candidate package contains **both** the frozen
driver and the option selector; it is one immutable actor, not a switch between
separately submitted models or track-specific models.

### Decision and action contract

An **option boundary** occurs at the first decision after `reset` and again after
the current option ends. At a boundary, the option selector observes the current
four-frame stack and the current deterministic frozen-DrQ action. For the first
protocol, choose one option for a fixed **two-decision** maximum duration. At
each of those decisions recompute the *base DrQ action on the current frame*;
only the choice of correction remains fixed. If the episode ends after the first
decision, the realized option duration is one, not two. Reset must clear the
option ID and remaining duration before the next episode. A one-decision
version is a **different** predeclared study, not a post-confirmation tweak.
Evaluation chooses the highest predicted option value; an exact value tie
deterministically prefers `KEEP`. Exploration randomness is training-only.

Define the default candidate actions in **official** coordinates, all applied
to that decision's freshly computed base action `b=(steer,gas,brake)`:

| Option | Proposed applied action | Semantics |
|---|---|---|
| `KEEP` | Exactly the frozen driver's official action | Control action; bypass a needless conversion round trip so the trace can be identical. |
| `STEER_MINUS` | `(clip(b.steer-0.15,-1,1), b.gas, b.brake)` | Bounded negative steering correction; no unverified physical left/right label. |
| `STEER_PLUS` | `(clip(b.steer+0.15,-1,1), b.gas, b.brake)` | Bounded positive steering correction. |
| `COAST` | `(b.steer, 0, 0)` | Stop accelerating without commanding a brake. |
| `BRAKE` | `(b.steer, 0, max(b.brake,0.25))` | Bounded gentle brake; still clip to `[0,1]`. |

**Before implementation freeze the signs, magnitudes, and two-decision duration
in the A protocol.** The numbers above are starting **design choices**, not
empirically optimal coefficients. Testing several values on confirmation would
invalidate that comparison. In all branches apply the common action bounds
once; do not mix native symmetric gas/brake with official `[0,1]` pedals. For
training, convert a non-`KEEP` official action to the collector's native action
with `ActionAdapter.to_native`, feed that native action into
`EpisodeCollector.step`, and record the **actually executed** official action
(`transition.applied_action`) and inverse-mapped native action
(`transition.action`). For `KEEP`, pass the original native frozen-DrQ action
through the existing adapter without rounding it through official and back.
In the exported `Agent.act()` path return the corresponding official action.
Verify training/inference applied-action parity at clipping boundaries. In
particular, native `[0,0,0]` maps to official `[0,0.5,0.5]`: it is **not**
`KEEP` and is not coasting. `KEEP` means an identity residual on the *real*
base action, including its native pedals.

The selector's first candidate implementation should reuse the **frozen DrQ
visual encoder** with a small trainable option-value head conditioned on its
features and the base action. Nothing in DrQ (encoder, actor, augmentation,
critics) is fine-tuned. If the current CPU export omits the required encoder
features or parity fails, resolve the export contract and measure resource cost
**before** evaluating performance; do not silently substitute a second,
untested CNN. The existing `agent.py` and `evaluate_policy.py` handle tagged
DrQ and Dreamer exports, not a composed option model by default. A new export
tag and compatible evaluator/package path are an implementation requirement,
not existing capability. In particular, `candidate_metadata()` presently
classifies non-Dreamer `.pt` exports as DrQ, while numeric summary eligibility
only checks the DrQ label. Fix **both** the model classification and the
new label's limits before counting any result as eligible.

### Data, objective, and budget

At option boundaries during **training-only** episodes, explore every option
with a protocol-fixed exploration schedule. A replay of the frozen driver's
`KEEP` trajectories cannot identify the effects of untried corrections.
Record the offered and executed option, base and applied action at every
decision, option-selection probability/exploration mode, track/geometry seed,
episode, actor hash, actual duration, and end reason. Log coverage by option
and by *distinct episodes/geometries*, not just a large count of correlated
frames. Do not reuse screen, confirmation, blind, or official-public trajectories
as training data.

A compact **semi-Markov** replay row at each option boundary is
`(phi_t, b_t, option, [r_t,...,r_{t+k-1}], k, phi_{t+k}, b_{t+k}, terminated,
truncated, terminal, final_observation_present, actor_hash, cell_id)`, with `k` equal to
the actual number of *environment decisions*, all rewards raw, and `phi` the
deterministic **frozen** DrQ encoder features computed on the real four-frame
observation. Generate the final features from the true final frame before any
reset, not from the next episode's first frame. Binding the replay to a frozen
actor hash is essential: features cached from another encoder are invalid.
The `phi_{t+k}` and `b_{t+k}` fields may be absent when a genuine terminal
does not need bootstrapping; they must exist at a bootstrappable truncation.
Storing both full `(4,84,84)` uint8 images for 100,000 option rows would
allocate roughly 5.26 GiB; two 256-dimensional `float32` feature vectors per
row would take about 195 MiB instead, if the frozen encoder indeed exposes
those features and the policy uses **exactly those inference-available**
features. Measure the actual feature size, replay capacity and memory before
training; do not claim the suggested size is the measured system footprint.
The baseline DrQ per-decision n-step replay cannot be relabelled into option
transitions by changing only an index. If an episode terminates, do not
bootstrap; if it is merely truncated, bootstrap only from a valid final
observation under the predeclared time-limit contract, never from a reset
observation. `finished=True` is terminal even when the raw environment also
sets `truncated=True`. Treat insufficient final-observation data as a broken
contract, not as a silent terminal conversion.

For a Double-Q-style selector, a target at option boundary is

```text
R_k = sum_{i=0}^{k-1} gamma^i * raw_reward[t+i]
o_star = argmax_o Q_online(o_next, base_action_next, o)
target = R_k + (gamma^k if not terminal else 0)
                 * Q_target(o_next, base_action_next, o_star)
```

Compute the second term only if the final observation is valid and a bootstrap
is permitted. The option is chosen **once** at the boundary; the frozen driver
still recomputes its base action inside the option. No learned dynamics,
imagined rewards, future track geometry, privileged `info` at inference, or
reward shaping enters this objective. A training-only diagnostic may compare
Q calibration and realized intervention returns but cannot certify completion.

**Suggested preregistration budget, not permission to run:** use the two
identified frozen DrQ actors; for each, train two independent option-learner RNG
seeds with a maximum of 32,768 additional environment decisions each. Reserve
the first 2,048 of each budget for **uniformly random five-option selection at
option boundaries** as a proposed coverage default, then use a proposed fixed
`epsilon=0.2` at option boundaries for the remainder. Freeze the exact seeded
schedule, update count, replay size, `gamma` (initial design `0.99` per
*decision*), target-network schedule and final checkpoint in the protocol.
Report total **additional** decisions for all four learners,
baseline driver decisions, option-selection frequencies and elapsed compute
separately. Do not call option update count an environment-decision count. These
values are bounded feasibility suggestions, not historical results or permission
to scale after a failed gate. For the first pilot, write the final learner
checkpoint at an **episode boundary at or before the cap** and report the
actual decisions. A claimed mid-episode resume would additionally require
restoring/replaying the environment state and proving the next observation,
held option, frozen-driver action, Q and target weights, optimizer, replay
cursor/content and learner/exploration RNG all match; saving the
option ID and RNG alone does not restore a simulator episode.

### Implementation map (only after separate authorization)

| Boundary | Proposed work | Required observable proof |
|---|---|---|
| New reusable algorithm code under `haic/algorithms/drq_v2/` | Frozen driver adapter, five-option transforms, option boundary/remaining-duration state, option-value head, semi-Markov replay and update. | `KEEP` reproduces the original deterministic action trace exactly; clipping, two-decision holds and one-decision early termination are deterministic. |
| `common_adapter.py` integration only if necessary | Reuse `ActionAdapter` and `EpisodeCollector`; preserve executed native versus applied official action and actual final observation. | Native/official round trip, all three axes at bounds, terminal versus time-limit truncation, no cross-episode replay. |
| Training CLI in `scripts/` | Freeze base actor, collect only training cells, train the option head, export manifest with both hashes and the option specification. | Every intervention arm has actual executed-option coverage; actor weights/hash are unchanged; exact consumed decision/update accounting. |
| `agent.py`, `evaluate_policy.py`, `package_submission.py` compatibility boundary | Recognize **one new tagged composite export**, load on CPU without extra runtime dependencies, implement reset/act, accurate algorithm metadata and numeric eligibility for this actor type; put the tiny inference head in `agent.py` or include its module in **both** packaging and evaluator source snapshots. | Two CPU reloads yield the same trace; startup/reset/act/RSS and package static checks pass. Existing package inclusion and DrQ-labelled numeric summary checks must not be assumed to cover the new tag. |
| New study-specific orchestrator/receipt validator in `scripts/` | Run one candidate per frozen confirmation receipt, ensure each has its own preceding screen; bind treatment and same-driver control outcomes on identical canonical cells after both are sealed. | Reject absent/changed protocol hash, unmatched actor lineage, consumed cells, CPU failure or duplicate/missing canonical results; never mistake a diagnostic control for a promoted candidate. |
| `tests/` | Synthetic option-duration/discount tests, reward/end-observation markers, reset/`KEEP` parity, native/applied action tests, CPU export/package tests. | No agent interaction is needed to falsify a broken transition or inference contract. |

The training learner, replay and optimizer must **not** be packaged for inference;
prove training-head versus exported-head behavior parity with fixed feature/action
inputs. Do not edit protected environment files to expose geometry, reward history, or
an action oracle at inference. New one-off diagnosis CLIs belong in `scripts/`;
generated trajectories and model files belong in `runs/` and evaluations in
`evaluations/`, not beside this Markdown.

An implementer can proceed in this order **without starting a performance run**:

1. Seal a proposed study protocol and the two exact frozen driver/export hashes;
   audit old and newly reserved geometry seeds and source/runtime versions.
2. Write pure option-transform/hold/reset tests and option-transition marker
   tests; implement only enough reusable code to pass them.
3. Verify frozen encoder-feature parity and `KEEP` trace equality between the
   training adapter, original CPU DrQ and the new CPU tag. Reject if packaging
   metadata, numeric limits, or first-call/reset behavior differs.
4. Add the training-only explorer, feature replay and learner; prove frozen
   driver hashes, boundary-checkpoint RNG/option-state continuity and actual
   option execution coverage before any A2 screen. Do not enable mid-episode
   resume without the separate environment-state/prefix parity gate.
5. Run the frozen screen/confirmation process **only after its separate
   authorization**. A0/A1 tests are not A2/A3 driving evidence.

Relevant existing regression suites after code work are
`python -m unittest tests.test_drqv2 tests.test_agent_inference
tests.test_local_contract tests.test_evaluate_policy
tests.test_submission_package`, plus the new option-specific tests. Record
exact commands, source/export/protocol hashes and CPU receipts; do not
describe a test plan in this document as a passing test result.

### A: gates and decision rule

1. **A0, contract:** Freeze both base actor hashes, option semantics, data split,
   CPU/runtime contract and protocol before any outcome. `KEEP` action/termination
    traces match each base actor across reset and episode boundaries; the other
    four options pass exact executed-action and duration tests, including
    clipping-induced options identical to `KEEP` and `finished=True` with
    `truncated=True`. Test the first `act()` even when no explicit `reset()`
    preceded it and do not carry a held option into the next episode. A
    synthetic two-decision marker with `gamma=0.9`, rewards `1,2` must yield
    `R_k=2.8` and exponent `gamma^2`; early finish uses `k=1` and no bootstrap.
    Fail fast on any misalignment, non-finite action, inability to bootstrap
    truncations correctly,
    or CPU/package limit breach. Do not learn a selector to hide a contract bug.
2. **A1, training-only feasibility:** collect actual option executions with the
   stated decision cap. Check that every option has coverage across independent
    training episodes and geometry seeds, that the driver was frozen, and that
    losses/actions remain finite. As a suggested coverage gate, each option
    must actually execute in at least ten distinct training episodes spanning
    four training-only geometry seeds; prefreeze this and the handling of
    clipped identical actions, rather than merely counting proposals. A missing
    option or repeated early collapse is
   **inconclusive/failed data collection**, not evidence of a bad controller.
   Choose one predetermined final checkpoint per learner; there is no
   confirmation-driven checkpoint rescue.
3. **A2, fresh screen:** on a *new* declared development grid, screen each
   unchanged frozen driver **and each option learner separately**, then compare
   their recorded outcomes on **identical cells**. Seal individual actor
   receipts; a multi-candidate screen pointer cannot serve as one candidate's
   preceding confirmation receipt in the current evaluator.
   Select at most one of the two option-learner seeds *per frozen driver* using
   only screen completion and the protocol's predeclared tie-breakers. Require
   nonzero composite completion, no operational failures, and a paired finish
   count at least as high as its unchanged driver before confirmation. If no
   seed qualifies for either driver, stop. Freeze package/checkpoint/option
   hashes and screen receipt after selection.
4. **A3, one fresh confirmation:** run one individually selected composite
   and one unchanged-driver control *separately*, each with its own screen
   lineage, on the *same new 32 canonical cells per pair*, e.g. four track IDs
   by eight geometry seeds when the exclusion audit permits it. A control
   with zero screen finishes may use only the marked **diagnostic,
   non-promoting** confirmation route; report that caveat explicitly. The new
   paired validator combines the sealed receipts. The suggested **research
   gate** is at least **two additional
   finishes per 32** for **each** driver, no action/API/resource failure and no
   use of confirmation results to change the learner, option definitions or
   threshold. Report paired wins/losses and geometry-clustered uncertainty;
   a +2/32 margin is a predeclared practical hurdle, **not** a universal
   statistical guarantee or official-score improvement. Treat only one driver
   gaining, or progress improving without the required finishes, as rejection
   for this protocol. Open any study-specific reserved blind partition only
   if the *separately approved frozen protocol* explicitly calls for it.

For any accepted internal candidate, also report lap times **only among
finishers**, incomplete progress, negative-reward retirement labels, collisions,
intervention rate and CPU resources. These diagnose the mechanism, not replace
the primary paired finish outcome. A composite's result does not promote either
standalone DrQ actor or prove every private-track result.

### A: ways this can fail

- The limited four-frame observation may not distinguish states needing opposite
  corrections. A temporary improvement in raw reward may sacrifice finishes.
- Off-policy Q can overestimate rarely executed options. The remedy for an
  unsupported option is better *training-only coverage or a rejected pilot*,
  not inventing counterfactual outcomes from frozen-driver replay.
- Holding a correction for two decisions can be harmful through a sharp turn;
  braking can extend a negative-reward streak. Neither the `off_track` label nor
  steering-saturation diagnostics establish the right intervention.
- Reusing frozen visual features could conceal a hazard the DrQ encoder never
  represented. A second encoder or new timing is a **new treatment** requiring
  its own CPU gate and protocol; it is not a silent rescue after confirmation.

## B. Training-Road Coverage: Independent Fallback Study

### Hypothesis, never assumed fact

Uniform episode-level fresh-road generation may insufficiently cover *sequences*
of difficult bends within a fixed training budget. The existing sampler already
changes geometry across episodes, so rare-motif undercoverage must first be
**measured on training-only data**; old completion and retirement labels do not
prove it. The official local generator's road shape is seeded, while track ID
and the same seed jointly influence obstacle placement. Reweighting seeds
therefore also changes the *distribution of obstacle configurations* even with
unchanged obstacle-generation code; B can test a **training-road sampling
strategy**, not isolate a geometry-only causal effect. A single geometry seed
repeated across track IDs is not several independent road geometries.

### B0: preregistered training-only feasibility audit

**Freeze a separate B0 diagnostic JSON protocol before inspecting seed-exposure
or driving outcomes.** It specifies generator/descriptor hashes, the permitted
seed pools, both frozen diagnostic DrQ actor hashes, cell/episode budget,
baseline/reference calculation, positive thresholds, `inconclusive` rules
and rejection conditions. B0 is not B1's training protocol, a new screen, or a
fresh confirmation result. Only after B0 passes should a **separate** B1
control-versus-treatment JSON protocol be frozen. A historical diagnostic read
after its outcomes are known can motivate B0 but cannot pass this prospective
gate by itself.

Audit the baseline's **training** road catalogue and realized training episodes
at the geometry-seed level. As a concrete initial catalog, preselect **2,048
unique permitted uint32 seeds**, record generator revision and each generated
road's centerline and fingerprint, and apply the union of **all** historical
consumed/reserved and newly allocated study exclusions before generating any
training map. Both arms must use this identical catalog. Before B0 outcomes,
also seal a **separate** training-range diagnostic pool, excluded from both B1
training and later screen/confirmation; do not promote B0 cells to fresh B1
evidence. No reserved blind cell enters either pool.

For the **single initial reweighting axis**, resample each unmodified closed
centerline at uniform arc-length spacing; calculate signed successive segment
turns using `atan2(cross,dot)` and wrap each angular difference. For each
cyclic window covering approximately eight road widths, accumulate the
**absolute** heading changes. Use that road's 90th percentile window value
as its severity descriptor; compute the 80th percentile descriptor threshold
from the *fixed training catalog only*. Assign the top 20% to `high_turn`,
the rest to `other`, and freeze equality/tie handling before training. Keep
window spacing, width definition and road generation version in the descriptor
hash. This scalar measures local concentrated turning, **not** every kind of
corner difficulty; record separate length, total absolute turn, ordered
left/right reversals and rigid-motion-normalized centerline fingerprints for
diagnosis, but do not silently reweight those additional features. Cyclic
starting position, global translation and rotation must not alter the scalar;
do not count rotations or alternative obstacle layouts as new geometries.
The threshold and bin fraction are **proposed experimental settings**, not
validated notions of a difficult road. Record obstacle-setting strata
separately from road-shape bins.

The suggested **bounded B0 diagnostic** uses 16 distinct `high_turn` and 16
distinct `other` geometries drawn uniformly *within each bin* from that
sealed diagnostic pool by a frozen RNG. Assign track IDs 1-4 in balanced
counts (four seeds per ID and bin), fixing the ID and seed for each cell
before any driving outcome; run one `max_steps=2000`, `frame_skip=4`
episode per geometry for **each of the two frozen DrQ actors**: 64 episodes,
at most 128,000 decisions. Track ID and obstacle setting are preassigned
independently of the bin; compare identical cells across actors. All 16
geometries per bin must return complete episode-level outcomes for each actor
or B0 is `inconclusive`. This is a **local diagnostic proxy**, not official
evaluation or an independent confirmation sample.

For a hypothetical uniform-catalog reset sampler, estimate the
**duration-weighted reference** from bin-wise observed mean episode lengths:

```text
p_high_decisions = (0.20 * mean_length_high)
                   / (0.20 * mean_length_high + 0.80 * mean_length_other)
```

Compute this per actor with the same local horizon, and compare episode-level
negative-reward-retirement rates by bin (the `off_track` label is **not** a
physical road-crossing label). The proposed go/no-go gate is
`p_high_decisions <= 0.15` **and** `retire_rate_high - retire_rate_other >= 0.10`
for **both** actors. These are preregistration **heuristics**, not proven
statistical effect sizes; report geometry-clustered uncertainty and obstacle
strata. The 20% factor applies to uniform *reset draws*, **not** observed
decisions. If the planned events/cells are insufficient or a fingerprint/bin
is unstable, classify B0 `inconclusive` and do not scale. If the complete
diagnostic misses either numeric gate, reject B for this protocol rather
than lowering it afterward. Do not use known confirmation outcomes to
select new thresholds. If a historical DrQ run did not record geometry
seeds, do not infer exposure from its final reward: label historical
exposure unknown.

### B1: single-variable training design

Subject to a positive B0 audit and a **separate** frozen protocol, choose one
finite catalogue of fresh training-only geometry seeds in advance. Both arms
use the same catalogue, same four training track IDs, same official obstacle-
generation code, raw reward, pad-4 native DrQ-v2 architecture, replay, optimizer,
augmentation, training decision/update budget and checkpoint opportunities.
Only the *episode-reset seed-selection probabilities* differ:

- **Control:** uniform over the catalogue for each new training episode
  (expected 20% `high_turn` seeds by reset, modulo ties).
- **Treatment:** first choose `high_turn` or `other` with fixed 50/50
  probability, then draw a seed uniformly within that bin. This has the same
  finite seed support but reweights high-turn resets; `50/50` is a **proposal**,
  not a sweep or proven optimum.

Both arms are conditional on the **same finite training catalogue**, unlike
the historical unconstrained uint32 seed stream. The matched estimand is the
effect of the seed reweighting *within this catalogue*, not an unqualified
improvement over every historical native DrQ run. Keep the track-ID sampler
and its RNG stream identical by design; actual visited track/seed pairs will
still differ when episode lengths or seed selections diverge.

Keep the total at the historical comparison's 131,072 environment decisions
per learner seed only if the fresh B protocol accepts that budget; use at least
two independent learner seeds with matched source revision. As an initial
checkpoint proposal, freeze **65,536 and 131,072 decisions** as the only
selection opportunities, select on the new screen cells, and seal the selected
actor before confirmation. Record actual
per-decision and per-episode geometry/motif frequencies, repeated seeds,
reset attempts, training track/obstacle IDs, wall-clock time and update counts.
Episode lengths differ, so equal numbers of *resets* do not imply equal numbers
of training decisions in each bin. Candidate seed catalog construction and
reset overhead are not zero-cost. Because a seed also affects obstacles, report
realized obstacle-layout strata by arm instead of saying that the obstacle
distribution was controlled. Do not quietly switch to episode-balanced
replay, teacher data, custom-map physics, auxiliary targets, new action heads or
the A option controller during B.

The implementation seam is the DrQ training reset/sampling path:
`train_drqv2.py` builds a sampled environment via `build_sampled_env`, whose
sampler in `train.py` independently chooses a track ID and a geometry seed at
episode reset. `HaicTrack.reset` ignores a caller-provided reset `seed`, so
calling `collector.reset(seed=chosen_seed)` does **not** implement the catalog;
change the training-only sampler behind `build_sampled_env` or inject a
study-specific sampler at that boundary instead. Existing
`run_drqv2_matched.py` supports the L2/pad studies, not a road-distribution
arm; use a separately scoped runner and protocol rather than relabeling old
results. The new runner must include the sampler and descriptor in its source
snapshot; previous L2/pad studies' equal-geometry-stream-prefix check cannot
hold when training seed frequencies intentionally differ. Keep the track-ID
draw RNG independent from the treatment seed-selection RNG, save and restore
both full RNG states and the catalog hash in checkpoints, and log the chosen
track ID, geometry seed and bin at **every reset**. Put the reusable
descriptor/sampler under `haic/algorithms/drq_v2/` and an audit CLI under
`scripts/`. Preserve the official environment and existing DrQ CPU actor
export untouched. Add tests for catalog determinism, exclusion across all
track IDs, bin accounting, unchanged transition/action/reward contracts, and
the equal-budget schedule. RNG persistence by itself does not prove a
mid-episode environment resume: either checkpoint/restart at a declared
episode boundary or separately prove full environment-prefix parity. Gate
order: validate catalog/generator fingerprints
and exclusion validator; implement catalog-backed sampler plus RNG checkpoint
parity; validate uniform and 50/50 draw frequencies without environment
interaction; freeze the B0 diagnostic protocol **before** observing its
outcomes; after B0 passes, freeze B1's protocol **before** any B1 training
or evaluation. The current actor inference/export tests should be
unchanged; add new runner/sampler tests to the regression suite.

### B2: matched evaluation and rejection

Use fresh **study-specific** screen cells for fixed checkpoint selection,
with **separate immutable receipts for each checkpoint and control/treatment
actor**, then a newly frozen actor/protocol and untouched fresh confirmation
cells. Run each sealed candidate individually on confirmation with its own
preceding eligible screen receipt; if a control screened at zero finishes,
its confirmation is diagnostic and non-promoting. The new paired validator
compares each treatment seed with its matched uniform-control seed on the
**same canonical cells** after validating both receipts; two CPU reloads are
a determinism check. Suggested primary
gate: at least **two additional finishes per 32** in **both** treatment learner
seeds, no operational failure; report distinct geometry-level gains/losses and
uncertainty with geometry seed as the grouping unit. The +2 criterion is a
proposed preregistration hurdle, not demonstrated power or proof of a private
track gain. Progress and lap time cannot rescue a zero-finish treatment.
Failure to demonstrate B0 undercoverage, a treatment that does not change
realized motif exposure, or an improvement confined to one learner seed is a
stop/rejection for this protocol. Do not reuse consumed confirmation or a
different study's reserved blind cells.

## Decision Summary

| Question | Study A | Study B |
|---|---|---|
| Primary changed variable | Presence of a trained short option selector over one **unchanged frozen driver** | Training-road reset distribution for **freshly trained ordinary DrQ** |
| Matched control | Same frozen driver and same fresh evaluation cells | Fresh uniform-sampled pad-4 DrQ runs on same catalogue and fresh evaluation cells |
| Training cost statement | Base DrQ cost **plus** additional intervention decisions | Same per-arm environment decisions and learner update budget; sampler overhead separately reported |
| First reason to stop | `KEEP` parity/transition or CPU package failure; no actual option coverage | No training-only undercoverage or incoherent road bins |
| Performance gate | Matched finish gain in both sealed base-driver pairs | Matched finish gain in both independent learner-seed pairs |

Priority A preserves demonstrated driving ability and tests a mechanism other
than Dreamer fidelity. B is a separate data-distribution bet only after a
training-only diagnostic establishes a reason to take it. Neither is an
extension of the rejected DrQ L2/pad sweep, an officially submitted model, or
evidence that the new hypothesis works. Update experiment/result/current-state
documents **only after** an authorized, frozen study actually changes their
source-of-truth status.
