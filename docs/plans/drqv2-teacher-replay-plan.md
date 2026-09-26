# DrQ-v2 Teacher Replay: Implementation and Study Plan

## Status and Decision Boundary

| Field | Plan |
|---|---|
| Status | Historical research design; the separately frozen r3 study stopped at A3 as inconclusive. This document alone authorizes no new training, evaluation, package release, or official action. |
| Relationship to current work | Independent of the active DreamerV3 recovery plan and the separate [pixel RLPD study](pixel-rlpd-offpolicy-plan.md); any new study needs its own frozen protocol and fresh partitions. |
| Claim under test | With a fixed additional online-decision and learner-update budget, replaying new training-only trajectories from an already competent frozen DrQ-v2 driver may improve repeated unseen-track completion when fine-tuning a copy of that driver. |
| Primary variable | Use of a fixed, source-matched teacher replay buffer in 25% of each learner minibatch. The paired control uses only newly collected online replay. |
| Not claimed | Equal *total* training cost, RLPD reproduction, causal benefit from steering desaturation, official score improvement, or an automatic replacement of either existing DrQ-v2 control. |
| First stopping point | A two-source, bounded 32,768-additional-online-decision pilot with frozen CPU screen and conditional fresh confirmation. No automatic 131,072-decision extension. |

### Outcome as of 2026-09-25

The separately frozen [r3 protocol](../../experiments/drqv2-teacher-replay-v1-r3.json)
spent the fixed 16,384 teacher decisions per source. Learner 0 reached four distinct
finished geometries, but learner 1 reached only three against the required four;
[A3 is inconclusive](../../experiments/drqv2-teacher-replay-v1-r3-result.json).
No student/control learner was trained and no screen, confirmation, or blind cells
were opened. Preserve the r2/r3 teacher datasets as consumed evidence; do not reuse
them for a fresh attempt, extend r3's cap, or substitute a source. The design below
is historical context, not permission to restart this study or open its reserved
cells.

This is a design specification, **not** a frozen experiment protocol or an active-plan
status update. At the original design gate, before the first teacher episode, a
separate JSON protocol had to pin
the finished implementation's source/configuration, the actual source-checkpoint
hashes, exact permitted seeds, the budgets and all screen/confirmation/reserved
blind cells. No number below is a measured optimum. Do not fill in evaluation
seeds after seeing driving outcomes. See [run-experiment](../workflows/run-experiment.md),
[evaluation protocol](../evaluation/protocol.md), and
[generalization policy](../evaluation/generalization-policy.md).

## Why This Study Is Distinct

- The two native pad-4 DrQ controls are the internal validated baseline, **not**
  official submissions: one unique actor per original learner seed recorded 4-7
  finishes per 32-cell confirmation cohort. Their full source lineage is in
  [model status](../results/MODEL_STATUS.md) and the
  [padding result](../../experiments/drqv2-augmentation-pad-v1-result.json).
- Steering-logit L2 at 0.001 and pad 1 each regressed in paired DrQ studies. This
  plan changes neither axis and does not launch another narrow tuning sweep.
- The stopped residual-options pilot selects a second controller over a frozen
  driver. This plan has no runtime option selector, second CNN, braking rule,
  planner, or intervention margin. It fine-tunes **one** actor with real executed
  transitions. The stopped pilot is not a control for this study; see
  [its result](../../experiments/drqv2-residual-options-pilot-result.json).
- [RLPD](https://arxiv.org/abs/2302.02948) motivates the independent prior/online
  replay streams, but its SAC actor, entropy objective, ensemble critics, pixel
  settings, and 50:50 sampling are **not** implemented here. See the separate
  [RLPD plan](pixel-rlpd-offpolicy-plan.md) for a real pixel-RLPD study.

## Environment, Model, and Inference Contract

One environment decision means one `CarEnvironment.step()` call. Its default
`frame_skip=4` happens inside `env_wrapper.py`; neither the collector nor the
learner repeats it. The observation is `float32 (4,84,84)` in `[0,1]`: one
grayscale frame is appended per environment decision after frame skip, with four
copies of the initial frame at reset. The deployed actor sees only this stack;
track ID, geometry, reward, `info`, collision history, and future frames are
training/evaluation metadata, not inference inputs.

The training coordinate is symmetric native `[steer, gas, brake]` in `[-1,1]^3`.
`ActionAdapter.to_official()` maps the pedals to `[0,1]` exactly once before the
environment step. Always use `EpisodeCollector.step(native_action)` and put
`transition.action`, the inverse of the **executed** official action, into both
replays. Retain `transition.applied_action` for audit. Native `[0,0,0]` means
official `[0,0.5,0.5]`, **not** coasting. There is no new rule suppressing
simultaneous pedals or smoothing steering. The inference actor returns a finite
official shape-`(3,)` action using the existing DrQ path in `agent.py`.

The model stays `DrQActor` (`drq_v2.py:365-410`): a four-channel CNN, 256-feature
encoder, 256-unit policy trunk, and deterministic three-axis `tanh` output.
There is no runtime teacher, critic, replay, privileged feature, ensemble, or
recurrent state. Export a **single** `haic-drq-v2-actor-v1` actor through the
existing `model.pt` loader; its training provenance must separately say
`DrQ-v2 + teacher replay`. Do not relabel an actor as a different learning
algorithm to bypass `evaluate_policy.py`'s DrQ eligibility. The active root
`package_submission.py`, not the historical visual PPO packager, is the package
path. No trainer dependency belongs in the ZIP.

## Source Checkpoints and Fork Semantics

Use the two full native pad-4 control checkpoints, not just their actor-only
exports:

| Original learner | Actor SHA-256 in model ledger | Full source |
|---|---|---|
| 0 | `433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37` | `runs/20260922-drq-augmentation-pad-v1-restart/control-seed0/checkpoints/step-000131072/checkpoint.pt` |
| 1 | `c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954` | `runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/checkpoints/step-000131072/checkpoint.pt` |

At the freeze gate, recompute each actor/full-checkpoint hash and verify source
revision, export parity, config, four-frame spec, action fingerprint, raw reward,
frame skip, and source manifest. Missing or inconsistent full checkpoints stop
this study; an actor-only file cannot warm-start a paired critic comparison.

For **each** original learner, instantiate one online-only control and one
teacher-replay treatment from byte-identical initial actor, Q1, Q2, and target-Q
weights. Fresh optimizers with identical hyperparameters are initialized for
the two arms; do not silently reuse different Adam moments or old replay. Store
the source hashes and initial weight hashes in both arm receipts. Implement an
explicit weight-only fork: `DrQv2Agent.load_checkpoint()` currently restores
replay, counters, optimizers and RNG and strictly checks the configuration, so
calling `--resume` and then resetting some fields would not establish this
experiment. Preserve the original checkpoints unchanged.

Use distinct study counters: `source_environment_steps=131072`,
`additional_online_steps=0` at fork, `study_gradient_steps=0`. Both arms use
the original exploration schedule evaluated at
`source_environment_steps + additional_online_steps`; it has already reached
the declared final exploration standard deviation of `0.05`. Do **not** reset
only the treatment's exploration schedule to `0.2`. Log actual native and
official per-axis actions, boundary mass, and effective noise for both arms.
Each arm has its own RNG states, initialized from the study's declared seeds;
track-selection, policy-noise, replay-sampling, augmentation, and learner RNG
streams must be individually checkpointed. The two arms can use identical
indexed reset RNG schedules but will not see identical *realized* road prefixes
when their episode lengths diverge.

## Partitions and Teacher Collection

Before any collection, enumerate the union of all documented consumed
screen/confirmation/blind geometry seeds, reserved blind seeds from **all**
studies, each new study partition, and development-only cells. In addition,
audit the two source runs' `episodes.jsonl` and original sampler provenance:
the new screen/confirmation/blind geometry seeds must be absent from **both
source actors' original 131,072-decision training episodes**, not only their
new teacher/student training. Exclude all evaluation/reserved geometry seeds
across **every** training track ID, not merely matching `(track_id, seed)`
pairs. The earlier pilot's exact grid is incomplete, so a read-only
run/protocol audit must qualify any freshness claim. Reserve different
teacher-training and online-training geometry pools within track IDs 1-4;
these pools may differ in episode counts but neither may touch evaluation
partitions.
Do not reuse an RLPD study's screen/confirmation or a Dreamer development
episode as this study's fresh evidence.

Collect an immutable dataset independently for each frozen source actor:

1. Use the official local wrapper unmodified, the real sampled reset path in
   `build_sampled_env`, `frame_skip=4`, official obstacles and raw reward.
   `collector.reset(seed=chosen_seed)` alone does **not** control the sampled
   road; validate or inject the training-only sampler at the reset seam.
2. Execute the source actor deterministically on each observation. Save each
   transition's four-channel input and actual final observation, proposed
   native action, executed native/official action, raw reward, end flags and
   `finished`, `retire_reason`, progress and damage **as training-only labels**.
3. Set a maximum of **16,384 actual environment decisions per source**. Stage
   episode rows until completion; if the decision cap interrupts an episode,
   record its spent decisions but do not mark it terminal or insert an
   incomplete final episode into teacher replay. Keep every complete episode,
   successful or unsuccessful. Never post-select finishes into a success-only
   replay or tune selection on its later evaluation outcome.
4. Serialize or index images sufficient to reconstruct every trainable
   `(o_t,a_t,r_{t+1},o_{t+1})`; hashes without pixels cannot train the student.
   Seal dataset bytes/hash, episode ledger, source actor hash, sampled road IDs,
   collector source hash and actual decision count before either arm learns.

An initial data-feasibility gate proposes **at least four finished episodes on
four distinct geometry seeds from each source actor** within that cap. It is a
preregistered coverage heuristic, not statistical proof that the demonstration
distribution generalizes. If it fails, report `inconclusive: insufficient
teacher completion coverage`; do not enlarge the cap or choose a different
source actor after seeing the data. The dataset includes terminal failures so
that the critic still observes their consequences. Every sample source is
tagged with its original actor and episode, even though those tags are not
inference inputs.

## Replay and Update Algorithm

Keep the immutable teacher buffer separate from a fresh online buffer for each
arm. Use `Uint8Replay`'s latest-frame `uint8 (84,84)` rows, episode/step/sequence
indices, sparse terminal/truncation result stacks, and n-step builder; do not
naively store both full `uint8 (4,84,84)` stacks at every row. Teacher capacity
is at most 16,384 rows per source and online capacity is 100,000. Existing
100,000-frame storage alone uses 705,600,000 bytes (about 673 MiB); this is
**training** memory and excludes boundary stacks, batches, optimizers and
other process allocations. An extra 16,384 latest-frame rows use about 110
MiB before metadata. Measure real host/GPU peak use rather than equating
`replay.memory_bytes` with whole-process RSS. Neither buffer enters inference.

Replay episode IDs must remain unambiguous across collection sources, ring
overwrites and restarts. Preserve the existing n-step contract: for a sample
starting at decision `t`, sum up to three **decision-level** raw rewards,
stop at `terminated` or `truncated`, set discount `0` on actual terminal, and
use `gamma^k` on a pure time-limit truncation only with the *real final
observation*. `finished=True` is terminal even if `truncated=True`. Reject a
missing boundary frame or an attempted cross-episode next observation. The
teacher replay stores the executed native action and audits the applied
official action; neither is an imagined counterfactual. Uniform valid-row
sampling within each buffer and no
success-only weighting are the first treatment.

The **fixed first-pilot** learning settings, inherited from the pad-4 study
where applicable, are:

| Setting | Proposed value and accounting |
|---|---|
| Additional online budget | 32,768 environment decisions **per arm per source**. The shared teacher collection is additional; never hide the 131,072-decision historical training of each source. |
| Online-only startup | Use the pretrained actor with its declared exploration noise for the first 10,000 online decisions. Do not use uniform random actions or learner updates in either arm during this period. |
| Learning phase | Starting after 10,000 online decisions, attempt exactly one critic update per additional online decision if both required buffers contain valid rows. Abort on an unexpected invalid batch rather than silently unequalizing update counts. |
| Minibatch | 64 rows: control `64 online`; treatment `48 online + 16 fixed teacher`. Run actor updates on the **same sampled mixture** as critic updates. An action decision is not a minibatch sample or gradient step. |
| Unchanged learner | Pad `4`; feature/hidden `256/256`; `n_step=3`; `gamma=0.99`; actor/critic LR `1e-4`; Polyak `tau=0.01`; actor and target update every second critic update; target noise std `0.2`, clip `0.5`; steering-logit L2 `0`; no reward shaping or normalization. |
| Checkpoint opportunities | Exactly 16,384 and 32,768 **additional online decisions**, after that decision's prescribed update. Export both for CPU screen; select only from these two. |

The nominal phase has `32,768 - 10,000 = 22,768` critic updates and 11,384
actor/target updates per arm if every declared update succeeds. Source optimizer
steps do not count as study updates. Because actor update uses a separately
augmented observation, preserve the existing critic-current, critic-next and
actor-view random shifts, applied consistently to the two replay sources.
Do not add an auxiliary reward, BC, `Q`-filter, Plan2Explore, action smoothing,
or intervention head to this **single-variable** experiment.

For teacher and online row `i`, with actual n-step horizon `k_i`:

```text
y_i = R_i + d_i * min(Q1_target(o_next_i, a_next_i),
                     Q2_target(o_next_i, a_next_i))
d_i = 0 for a terminal, otherwise gamma ** k_i
a_next_i = clip(actor(o_next_i) + clipped_target_noise, -1, 1)
L_critic = mean((Q1(o_i, executed_native_i) - y_i)^2
              + (Q2(o_i, executed_native_i) - y_i)^2)
L_actor = -mean(Q1(augmented_o_i, actor(augmented_o_i)))
```

Gradients must not flow through targets, noise or target networks; freeze the
critic during the actor update. These are the current DrQ losses, **not** the
RLPD SAC loss and not a direct behavior-cloning objective. Critic action
coverage and actor behavior on teacher states are diagnostics: extra old-driver
rows may anchor road-state values or may cause policy drift/Q overestimation.
Neither outcome is guaranteed. Log source-specific TD errors, actor actions
relative to the teacher on teacher states, `Q1-Q2`, return scale, action
saturation, raw reward, actual sample fractions and distinct episode/geometry
counts. Never use an offline critic value as proof of finish improvement.

## Code and Artifact Boundaries

| Proposed boundary | Implementation and proof |
|---|---|
| `haic/algorithms/drq_v2/` | New reusable teacher-row metadata, immutable dataset index, two-source sampler, and weight-only fork/update helper. Reuse `common_adapter` and `drq_v2` primitives; leave old checkpoint format, `DrQv2Agent.update()` behavior, and the rejected L2/pad runner unchanged. Prove source and teacher buffer hashes. |
| `scripts/` | Study-specific teacher collector, two-arm trainer, provenance/partition auditor and paired-receipt validator. Run project imports from repo root with `python -m`. Current `run_drqv2_matched.py` hardcodes the L2/pad axes; it is not an arbitrary prior-replay runner. |
| `runs/`, `evaluations/`, `experiments/` | Actual pixel datasets/checkpoints and logs under `runs/`, single-actor screen/confirmation receipts under `evaluations/`, immutable **future** JSON protocol/result in `experiments/`. Do not version large replay/weights simply to make a prose statement. |
| `agent.py`, `package_submission.py` | Use the existing tagged DrQ actor-only CPU implementation and active root packager; no teacher/critic inference import. Confirm the strict same-architecture source/fork/export action trace and check root-only ZIP and prohibited Python imports. |
| `evaluate_policy.py` | Evaluate each exported actor separately using its own fixed screen receipt and the **same** study SHA before confirmation. A new paired validator joins already sealed per-actor canonical cells; do not weaken the existing single-actor lineage requirement or treat a diagnostic zero-screen control as promoted. |

Do not place new research Python files at repository root, alter
`core/`, `env_wrapper.py`, or `damage.py`, or rewrite pre-existing source actor
weights. The actual runnable CLI arguments, source snapshots and output paths
are specified and hashed in the frozen protocol after these components exist,
not invented in this prose plan.

### Proposed Implementation Interfaces

These were proposed names and contracts at the time of writing. Several were
later implemented for separately frozen studies; consult current code and the
r3 protocol for the actual CLI and contract. This design did not itself write
code or allocate output directories.

| New component | Proposed minimal interface and invariant |
|---|---|
| `haic/algorithms/drq_v2/teacher_replay.py` | `fork_from_source(full_checkpoint, study_config, seed)` validates the source and copies actor, two critics and both targets; it initializes new matched optimizers, online replay and RNG **without** calling legacy `load_checkpoint()` as a study resume. `TeacherDataset` loads immutable complete episodes with a source/collection digest. `TwoSourceReplay.sample()` returns a batch and source indices with quotas `0:64` or `16:48`, failing if either pool lacks valid n-step rows. |
| `scripts/collect_drq_teacher.py` | Proposed CLI accepts `--protocol-file`, `--source-learner-seed {0,1}`, and `--run-dir`. It creates a new directory exclusively, verifies source hashes and allowed reset cells **before** environment creation, records raw decision count including dropped partial episodes, seals immutable teacher pixels and a source/episode manifest. |
| `scripts/train_drq_teacher_replay.py` | Proposed CLI accepts `--protocol-file`, `--source-learner-seed {0,1}`, `--arm {online-only,teacher-replay}`, `--teacher-dataset`, and `--run-dir`. It rejects teacher mismatches, revalidates exclusions at every reset, separates source/global exploration steps from study update counts and exports actor-only checkpoints at precisely the two declared online steps. |
| `scripts/compare_drq_teacher_replay.py` | Proposed CLI accepts a frozen protocol plus the immutable per-actor screen/confirmation receipts. It validates identical canonical cells, source/arm/actor/protocol hashes, eligibility, two reload trace identity and prior-screen lineage, then emits a **separate** per-cell paired decision receipt; it never mutates an evaluator receipt. |
| `tests/test_drq_teacher_replay.py`, `tests/test_drq_teacher_study.py` | New synthetic batch/fork/budget/terminal and protocol/receipt tests alongside existing DrQ, evaluator and package regression suites. No environment evaluation is required to fail a broken contract. |

The future frozen protocol should contain `hypothesis`, `source_actor_sha256`,
`source_checkpoint_sha256`, `source_revision`, environment/action/observation
fingerprints, teacher source/training seed pools and collector cap, both arm
configs and online/exploration/update/checkpoint budgets, every proposed
development/screen/confirmation/blind cell, exclusion union, CPU constraints,
selection/acceptance/stop rules and source snapshots for **all** new scripts.
Teacher dataset bytes/hash and student actor hashes do not exist before
collection; bind them in immutable collection and selection receipts before
the next dependent step. Run-local filenames, contents and commands should
match that frozen contract. A locally generated package is never proof of
an official upload.

## Executable Test and Gate Order

1. **A0, source and static protocol:** inspect/re-hash both full checkpoints
   and actors, reserve fresh disjoint partitions, exclude their **geometry
   seed union** from every new collector and reject any intersection with the
   original source-training seed ledgers; record dependency versions. Tests
   reject wrong actor/source hash, duplicate cells, reused confirmation/blind
   seeds, a new evaluation
   geometry seen during source training, unsupported archive fields and absent
   source snapshots.
2. **A1, data and replay correctness:** synthetic marked two-episode streams
   prove `o_t`/`o_{t+1}` and reward/action alignment, native/official clipping,
   finish-as-terminal even with truncation, pure time-limit bootstrap, early
   1/2/3-step endings, reset padding, ring overwrite and missing-final-frame
   fail-closed behavior. Check that the frozen dataset includes only
   preregistered training seeds and all complete teacher episodes, with no
   fabricated terminal on budget interruption.
3. **A2, learner and fork correctness:** at the fork both arms and the original
   actor have identical deterministic CPU actions on a fixed observation set.
   Critic loss uses executed native actions and detached targets; actor loss
   updates only intended parameters. Verify exact `16:48` vs `0:64` rows,
   source coverage, update totals, RNG isolation and loss finiteness. A full
   checkpoint restores both buffer contents/cursors, optimizer, learner and
   sampler RNG and source identities. Mid-episode restart requires actual
   environment-prefix replay and action/observation parity; otherwise resume
   only from a complete episode boundary.
4. **A3, bounded teacher collection:** audit real reset IDs and dataset byte
   hashes; enforce the four-independent-geometry-finish-per-source proposed
   coverage gate. Insufficient events are inconclusive, not a license to view
   confirmation or rerun another teacher seed.
5. **A4, 32k training pilot and CPU screen:** run both original sources and both
   arms under the frozen budget. Export at the two fixed opportunities and run
   the existing isolated CPU evaluator on every candidate checkpoint. Initialization
   `<=10s`, each reset/act `<=5s`, whole worker RSS `<=1GiB`, valid action, two
   independent CPU reloads with identical traces/outcomes and package static
   checks are hard gates. The active packager's smoke does not itself measure
   peak worker RSS, so keep the evaluator's measurement.
6. **A5, fresh confirmation and optional blind:** complete the screen-only
   selection and seal actor/source/protocol/checkpoint hashes *before* the
   first confirmation. Treat confirmation strictly as pass/fail; a study blind
   is a predeclared terminal check on one screen-fixed treatment finalist,
   opened only after the full confirmation gate. Never recycle the prior
   DrQ/Dreamer/option screen or a different study's reserved blind.

Relevant existing regression suites include `tests.test_drqv2`,
`tests.test_train_drqv2`, `tests.test_agent_inference`,
`tests.test_evaluate_policy`, and `tests.test_submission_package`; add new
teacher-buffer/fork/receipt tests rather than treating old unit tests as
evidence for this new treatment. No test or experiment in this list has been
executed by the creation of this document.

## Frozen Evaluation and Selection Rule

For a first study, propose a **new** 24-cell screen (three track IDs by eight
distinct geometry seeds), a separate **new** 32-cell confirmation (four track
IDs by eight distinct geometry seeds) and a reserved **new** 24-cell blind
(three IDs by eight seeds). Those are shape examples, **not allocated IDs**;
audit and freeze the concrete disjoint cells before teacher collection. A
shared geometry with different obstacle/track IDs remains one correlated road
shape. Run each actor twice in isolated CPU workers, count only canonical
`repeat == 0` for outcome summaries, and compare treatment, online-only
control and unchanged source actor on identical canonical cells. The source
actor also needs its own study-compatible screen receipt; old confirmation
numbers are context only.

Within each source/arm choose at most one eligible checkpoint by the existing
ordered screen tuple: most completions, then greater canonical mean progress
as defined by the study's metric contract, then shorter **completed** lap time;
exact ties retain 16,384. Freeze **both** treatment source actors for the
replication comparison and nominate at most one blind finalist using screen
results only, with original learner 0 breaking an exact tie. A treatment
source with zero screen finishes cannot pass. Require at least one treatment
screen finish and no fewer finishes than its same-source online-only control
for **each** original learner before any promotion confirmation. If a control
has zero screen finishes, the evaluator's explicit diagnostic-confirmation
path can yield a paired scientific comparator, but it is non-promoting and
cannot authorize control blind.

The proposed **practical confirmation hurdle**, frozen before collection, is
`F_teacher(source) >= F_online_only(source) + 2` **and**
`F_teacher(source) >= F_unchanged_source(source)` for *each* of the two
original learner sources, with treatment gains on at least two distinct
geometry seeds, nonzero treatment completion and no CPU/receipt failure.
Compare per-cell paired wins, losses and ties; report geometry-clustered
uncertainty. A `+2/32` difference is a proposed decision hurdle, **not** a
statistical significance claim or an estimate of private-track performance.
Progress or raw reward cannot rescue a finish tie/regression. Completed lap
time only compares finishers; damage, `retire_reason`, steering and Q metrics
are failure diagnostics. The one screen-fixed blind finalist must finish at
least one canonical blind cell with matching reload traces and all runtime
gates, if blind was predeclared and confirmation passes; otherwise do not open
it.

Failure at any prior gate stops promotion. No checkpoint, replay fraction,
teacher length, study geometry or source actor is replaced after confirmation.
If the pilot fails, record a bounded rejected/inconclusive attempt rather
than trying a third DrQ coefficient or retroactively reopening old blind
cells. A positive result permits consideration of a **new**, separately
frozen larger-budget study; it does not itself authorize scale-up, an official
submission or model confirmation.

## Accounting, Risks, and Follow-Up Conditions

- Per source actor the historical 131,072 decisions are inherited, not free.
  Report teacher collector decisions (including discarded partial episodes),
  per-arm additional online decisions, environment episodes, critic/actor
  update counts, wall-clock GPU time, CPU evaluation decisions and teacher
  dataset creation separately. Both arms match *new online decisions and
  update opportunities*; treatment has extra teacher data, so this is not an
  equal-total-data or algorithm-only comparison with a fresh DrQ run.
- Teacher rows may preserve road knowledge, but may also oversample the old
  policy's errors; actor Q gradients on teacher states can exploit extrapolated
  values. Dataset completion coverage, source-specific TD error, applied-action
  support, online action spread and closed-loop finishes test these mechanisms.
  Steering saturation exists even among finishers, so do not impose an ad hoc
  steering-logit penalty or infer that desaturation is a completion surrogate.
- The official target is one model that handles unseen tracks. Avoid
  track-specific actor selection, using published tracks as iterative tuning,
  treating a local custom map as a private holdout, or joining several
  submissions' per-track bests. Official ranking places completion ahead of
  incomplete progress; local raw reward, damage and robustness remain proxies.
- If fixed demonstration replay helps but actions drift from the source on
  **training-only** diagnostics, a separately preregistered advantage-filtered
  imitation treatment may be proposed. It is **not** part of v1. If action
  support/exploration is the bottleneck, test the independent pixel-RLPD plan
  rather than silently changing this actor to SAC midway through the study.

The current locally mirrored official contract is Python 3.11/Linux CPU,
Torch 2.1.0, `Agent` import/construction `<=10s`, each `reset`/`act` `<=5s`,
participant process memory `<=1,024 MB` and a compliant root ZIP; the local
evaluator separately enforces its whole-worker 1 GiB cap. The official ZIP
limits include compressed `<=500 MB`, uncompressed `<=2 GB`, at most 1,000
files and inference weights entirely in the archive; see
[restrictions](../competition/restrictions.md). Recheck the actual
[Participants source](https://github.com/2026-HAIC/Participants) and
[competition website](https://scholarships-hardwood-headers-influenced.trycloudflare.com/)
immediately before any separately authorized official action. A local
package or candidate designation is not official confirmation.
