# Oracle-Teacher Visual PPO Revival

Status: **PROPOSED / PLANNING ONLY**, 2026-09-30. The user requested a plan,
not implementation, collection, training or evaluation. No cells are allocated
by this document. DrQ-v2 and Dreamer remain closed; TD-MPC2 remains paused.
Existing KOI and RLPD work is independent and unchanged.

## Decision And Hypothesis

Select **visual PPO**, developed as Oracle BC -> student-state DAgger ->
completion-preserving PPO fine-tuning. Do not resume the historical checkpoints
or repeat from-scratch PPO tuning.

Hypothesis: a competent joint-action teacher can establish a usable visual driving
policy before on-policy reward optimization, while DAgger covers states reached
by the imperfect student. PPO can then improve that policy without destroying
its completed roads. Each clause has a separate gate; successful imitation alone
does not establish a PPO improvement.

Independent reviews found TD-MPC2 technically suitable for supervised actor/prior
learning, but its explicit focus pause and unresolved planner-score risks make
it less suitable for this scope-preserving first proposal. Dreamer's free-prior
dynamics problem is not automatically repaired by better actor labels. PPO has
primary unsuccessful baseline evidence and reusable imitation/on-policy pieces.

## Evidence And Limitations

| Evidence | What it establishes |
|---|---|
| `runs/20260920-214647_ppo-cnn-multidiscrete-control-v1/best_model_metrics.json` | At 1,048,576 steps, seen 0/24 and historical holdout 0/24 finishes; holdout progress 0.61065 with 13 crash and 11 off-track endings. |
| `runs/20260921-000225_ppo-cnn-continuous-box-control-v1/best_model_metrics.json` | Selected step 786,432, not final: seen 0/24 and historical holdout 0/24 finishes; holdout progress 0.38201. |
| `haic/oracle_v1/controller.py` and `PROVENANCE.json` | A privileged pure-pursuit/speed teacher; provenance records 100/100 finishes on 50 exposed local configurations, not a universal guarantee or official result. |
| `experiments/rlpd-recovery-guided-evaluation-v1-result.json` | Joint teacher fitting can regress closed-loop driving: original 5/12 versus guided 2/12, with four original finishes lost. |
| `experiments/rlpd-recovery-validation-v1-result.json` | Executed Oracle intervention can harm finished-parent controls; a short local recovery condition is insufficient to establish full-episode rescue. |

Historical PPO results used different action/control and reward contracts,
including five input planes and smoothing. Their reported holdouts are consumed,
not new evaluation cells, and are not matched comparators for this proposal.
The new student follows the custom visual PPO implementation family, whereas
the two large historical runs used the root SB3 family. This is a new PPO-based
design, not a single-axis causal explanation of those old failures.

The archived pedal-policy plan also reports low BC training error with no driving
success, but its original outputs are unavailable locally. Treat that report as
historical context, not reverified numerical evidence.

## Frozen Student And Teacher Contract

- Student input: only the official-shaped four grayscale frames `(4,84,84)` in
  `[0,1]`. Use a compact full-frame CNN with image-derived HUD features, following
  `haic_agent/networks.py`. First version is feedforward; no RSSM, GRU or planner.
- Student output: three continuous coordinates `[steer, gas, brake]`, with bounds
  `[-1,1]`, `[0,1]`, `[0,1]`. Use a three-coordinate squashed Normal and affine
  pedal mapping; deterministic evaluation transforms the actor mean.
- BC matches the three **official-coordinate** deterministic actions. Exact zero
  teacher pedals remain valid supervised targets; do not require inverse-tanh
  of those boundary labels. PPO stores exact pretransform samples and transformed
  log probabilities, including pedal Jacobians, rather than inverting clipped
  environment actions.
- Teacher: unchanged `OracleController`, `target_speed=12.0`,
  `avoid_obstacles=True`. Pin controller, environment and runtime hashes. Query
  before executing the action associated with that observation.
- Environment: existing unmodified local official mirror, obstacles enabled,
  frame skip 4, 50 raw warmup noops. Use explicit actual-end versus evaluator-cap
  semantics with a proposed 2,000-decision ceiling. Never count a capped live
  episode as a finish or an uncensored comparison.
- PPO learns unchanged raw environment reward. No old route/damage reward shaping,
  reward normalization, privileged auxiliary targets, EMA steering, CEM or MPPI.
  Initially omit auxiliary-head optimization entirely.
- Oracle pose, velocity, centerline, future obstacle positions and road IDs are
  training/diagnostic data only. Student forward/export must not read them.

The old custom actor caps gas at 0.02 and brake at 0.03.
`training/imitation.py:40-63` additionally rescales the pixel corridor teacher and
multiplies pedals by 0.75; its collector executes the rescaled action. These
paths **cannot be reused unchanged** or hidden behind an Oracle teacher factory.
The new collector executes original Oracle actions during expert rollouts.
Neither reducing Oracle pedals nor raising the teacher speed is a v1 treatment.

Oracle sees information unavailable to a short pixel history, so some action
targets may be visually ambiguous. Held-road error, per-channel error, near-zero
pedal imbalance and left/right/obstacle state responsiveness are diagnostics.
Persistent good fit with poor student-only completion rejects the proposed
supervision/representation contract; it is not permission for a loss-weight sweep.

## Data And Accounting

Propose TRAIN 60 roads, 12 per track ID 1-5, with distinct geometry seeds across
tracks; development 20 roads, four per track; confirmation 40 roads, eight per
track; terminal blind 40 roads, eight per track. These are proposed counts only.
Exact IDs require cross-lane allocation/exposure/exclusion audits and claims;
current RLPD unseen-cohort claims and all retired/protected pools remain excluded.

Split entire geometry seeds, never random frames from the same road. BC fitting
uses 50 TRAIN roads, ten per track; ten TRAIN-only roads, two per track, are an
offline supervision diagnostic excluded from all fitting and online training.
DAgger/PPO use only those 50 fitting roads. These ten diagnostic roads are
consumed TRAIN evidence, not confirmation. Do not collect Oracle trajectories
or labels on development, confirmation or blind roads.

Persist reset intents and every attempted episode, including teacher failures,
student failures, partials and unused scheduled slots. Each transition has
pre/next observation hashes, episode/cell identity, teacher target, student
proposal, actual executed action and action role, raw reward, terminal/truncation
flags, actual finish, cap status, source/runtime/protocol hashes and timestamp.
Store privileged diagnostics separately from model input tensors.

Queried but unexecuted Oracle actions are supervised labels only. They do not
inherit the student's successor or reward as demonstrated transitions. Expert,
mixed DAgger and pure-student episode outcomes have separate denominators.
No teacher intervention or rescue is allowed in any candidate evaluation.

All following budgets/gates are **proposed research choices**, not measured
resource feasibility, competition requirements or a frozen execution protocol.

## Staged Development

| Stage | Proposed work and cap | Gate and interpretation |
|---|---|---|
| 0. Contract | Implement isolated source and synthetic tests; zero environment resets. | Action/label alignment, saturation likelihoods, terminal/truncation-safe GAE, TRAIN-only input boundaries, CPU export/reload and source binding pass. Resource forecast measured before any real reset. |
| 1. Teacher pilot | First 20 TRAIN roads, four per track; one complete Oracle episode each, at most 40,000 decisions. Pilot is part of, not additional to, the 60-road dataset. | At least 16/20 finishes and at least 3/4 on each track, no invalid actions; caps/partials block the full-finish gate. Retain every failure; do not replace hard roads. Otherwise stop student work and diagnose teacher suitability. |
| 2. Expert data + BC | Complete the fixed 60-road schedule, at most 120,000 total teacher decisions. Only after the teacher gate passes, fit two independent initialization/minibatch seeds for 8,192 BC updates each, batch 64; exclude the ten diagnostic roads. | Teacher finishes at least 50/60 and 9/12 per track. After fixed final BC export, each student alone finishes at least 5/20 development roads spanning at least three tracks. Below this, do not launch DAgger/PPO just because BC MSE is small. |
| 3. DAgger | Two fixed rounds per student seed, each at most 16,384 interaction decisions and 4,096 BC updates, batch 64. First round chooses whole teacher/student actions with teacher probability 0.5; second round is student-only with Oracle query labels. | Fit aggregated expert plus learner-state labels using fixed 32/32 minibatches. Evaluate fixed round-2 exports alone. Each seed must finish at least 10/20, at least one on each track, retain at least 90% of its BC finishes and not increase mean damage by more than 0.05. This is a PPO-entry feasibility gate, not proof DAgger helped. |
| 4. PPO isolation | From each same seed-specific round-2 student, create equal-weight/model/optimizer/RNG starting copies: plain PPO control and PPO + fixed teacher-BC retention treatment. Each arm gets at most 65,536 pure-student decisions. | Same raw reward, cells, schedule, PPO update count and checkpoints. Only retention differs. Freeze the teacher-label pool before forking; no new DAgger or Oracle takeover in either PPO arm. Keep the unchanged round-2 student as a third evaluation baseline. |
| 5. Independent evaluation | Development selection, fresh confirmation, then one terminal blind test for a predeclared finalist if gates pass. | Both training seeds must support the same treatment direction. No checkpoint/weight/seed replacement or retraining after confirmation; blind is not a tuning loop. |

BC sampling first balances roads/episodes, then frames, rather than allowing long
straight episodes to dominate. Predeclare fixed longitudinal/turn/obstacle error
reports and label-support counts; do not outcome-select easier samples. DAgger
keeps real failures and off-nominal teacher queries, including disagreement,
instead of assuming an Oracle good from reset is always a good recovery teacher.
Report pure-student round-2 and development outcomes separately from the assisted
first round. If student-state guidance degrades viable trajectories, reject it
under the retention gate rather than describing teacher assistance as success.

Initial PPO choices: rollout 1,024, minibatch 64, four epochs, gamma 0.99,
GAE lambda 0.95, clip ratio 0.2, learning rate 1e-4, entropy coefficient 0.001,
value coefficient 0.5, gradient norm cap 0.5. Before each fork, fit only the value
head for 1,024 updates, batch 64, on discounted actual raw returns from complete
expert fitting episodes, with encoder/actor/std bitwise frozen. Both arms inherit
that same checkpoint and set initial pretransform exploration std to 0.1 in all
three coordinates; source-bind the setting before interaction. Treatment
adds `1.0 * BC_MSE` on 64 frozen teacher-labelled examples per PPO optimizer
step, with no extra optimizer steps. Its separate, fixed minibatch RNG must not
advance PPO action-sampling or permutation RNG streams. Report auxiliary forward/backward work
and wall cost separately; equal decisions/updates do not mean equal computation.
No parameter search is part of v1.

Forks have matched starting state and reset conditions, not identical trajectories
after action divergence. If plain PPO wins, report that result honestly; it does
not validate teacher retention. If neither PPO arm improves the frozen DAgger
policy, conclude imitation is useful but PPO revival remains unsupported.

## Selection And Stop Rules

For PPO screen, compare only predeclared 32,768 and 65,536 exports; choose by
complete finish count, then mean progress, then lower mean damage, then earlier
checkpoint. Evaluate the frozen BC/DAgger baselines on the same development
schedule and publish kept/lost/gained/neither cells for every comparison.

The PPO development gate requires, **for each training seed**, at least two net
additional finishes over its frozen DAgger policy, at least 90% of old finishes
retained, and mean damage no more than 0.05 worse. A higher total achieved by
replacing most completed roads fails. Treatment must also not have fewer total
finishes than plain PPO. If DAgger is already near the cohort ceiling, absence
of improvement is inconclusive for PPO potential, not grounds to change the gate.

Confirmation tests frozen selected control, treatment and DAgger actors from both
seeds on the same 40 fresh internal cells. Proposed treatment gate: at least
20/40 finishes per seed, at least 90% of each seed's DAgger finishes retained,
positive net finish gain over DAgger, no mean damage increase above 0.05, and
nonnegative finish difference versus plain PPO in both seeds. Publish per-track
counts, uncertainty and paired discordances; dependent track variants and reload
repeats do not multiply the road/training-seed denominator. These thresholds are
screening decisions, not a statistical claim of superiority.

Pick the treatment blind finalist by the frozen development rule before
confirmation. Blind opens only if the confirmation gate passes; evaluate it and
its seed-matched frozen DAgger baseline on 40 untouched cells, once, with full
uncertainty and paired outcomes. A 20/40 observation is not proof the population
completion rate is at least 50%. It is not official HAIC performance.

Report lap time only on jointly completed roads, plus all-road completion,
progress, damage, timeout/crash/off-track/invalid-action counts. First target is
reliable completion; exceeding the conservative 12 m/s teacher is a later,
separate hypothesis. Historical KOI/RLPD results on different cells cannot rank
this student against them. A project-baseline comparison would require separately
authorized, frozen common cells and exact available baseline provenance.

Hard stop on privileged input leakage, invalid actions/nonfinite losses,
source/protocol mismatch, unresolved candidate-related seed identity, missing
critical resource telemetry or projected inability to finish the declared phase.
Keep partial evidence and issue a failure receipt; do not reuse a stopped run
path, extend caps, swap failed roads or relabel consumed cells as fresh.

## Implementation Boundary And Deliverables

Existing pieces to reuse after compatibility tests: the pixel/HUD encoder design
in `haic_agent/networks.py`, GAE/storage in `training/rollout.py`, PPO loss pattern
in `training/ppo.py`, and aggregation/BC patterns in `training/imitation.py`.
Do not import the old shaped trainer as a turnkey experiment. DAgger currently
is not integrated into its main training flow.

Proposed isolated reusable code belongs under `haic/algorithms/ppo_teacher/`;
collection/training/evaluation/export CLIs under `scripts/`; focused regression
tests under `tests/`. Future study protocols/results belong in `experiments/`,
large data/checkpoints/ledgers in distinct `runs/` paths. Do not patch shared
Oracle/environment/old PPO modules or move teammate-owned files. A separate
CPU actor adapter/packaging mode is required: the old custom dynamics/CEM package
must not enter this PPO-only inference path accidentally.

Before runtime execution, review the implementation and freeze a study JSON with exact cells/claims,
source/runtime/commands, initialization and sampling seeds, full episode slots,
all update/decision/reset/wall caps, retention loss, checkpoint rule and gates.
Forecast remaining RAM across cgroup ancestors, GPU reserved memory, checkpoint
serialization and disk/temp space from synthetic measurements rather than copying
an old arbitrary free-resource floor. At two DAgger rounds and four PPO arms,
proposed expert/DAgger/PPO interaction ceilings total 447,680 decisions before
evaluation; warmups, any pilot/recovery diagnostics and all evaluation episodes
must be accounted for separately. Reserve no roads until the candidate audit
passes, and recheck own claims immediately before resets.

CPU acceptance follows the then-current official contract. For planning, the
local mirror lists CPU Torch 2.1, 1,024 MB process memory, 10-second construction
and 5-second reset/act bounds. Prove actor-only reload/action parity and absence
of teacher/environment dependencies before package eligibility. Recheck official
sources before any separately authorized submission or confirmation. This plan
authorizes neither external action nor candidate promotion.

Only this proposed plan and coordination notes change now. Current-state,
experiment index and model status remain unchanged because no experiment or
research-line activation occurred. Update their relevant sections only when a
subsequent authorized execution changes their source-of-truth status.
