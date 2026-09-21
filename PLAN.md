# HAIC Algorithm Migration Plan

## Goal

Replace the PPO+CnnPolicy learning algorithm without changing the official
environment. The primary objective is repeated unseen-track completion. Mean
progress, damage, steering smoothness, training throughput, checkpoint size,
and CPU inference latency are secondary evidence. Lap-time optimization and
submission updates remain blocked until a candidate has repeated unseen
completion.

The requested order is preserved:

1. DrQ-v2
2. DreamerV3
3. TD-MPC2
4. Dreamer 4

An algorithm is not promoted because of a higher progress score alone. Every
candidate must pass the same real-environment CPU screen, confirmation, and
blind protocol.

## Frozen Contract

- Leave `core/`, `env_wrapper.py`, and `damage.py` unchanged.
- Use the current Gymnasium environment and one high-level transition per
  `CarEnvironment.step`; `frame_skip=4` remains inside the environment and is
  never repeated by an algorithm adapter.
- Observation spec: four `84x84` grayscale frames, CHW, `float32`, `[0, 1]`.
  A fifth Markov/action-control plane is a separate representation axis and
  must not be mixed into the first algorithm comparison.
- Primary algorithm track: continuous action `[steer, gas, brake]` with
  official bounds `[-1, 0, 0]..[1, 1, 1]`. Map native symmetric action heads
  through one explicit `ActionAdapter`; do not let each library invent its own
  gas/brake convention.
- First algorithm comparison uses no stateful smoothing and no collision
  penalty. Existing smoothing and Markov experiments remain historical
  evidence, not hidden confounders.
- Keep the sampled-track training stream, official obstacles, independent
  `(track_id, seed)` variation, and CPU-pinned evaluator. Record the exact
  schedule because independent episode lengths prevent bit-for-bit paired
  rollouts unless a precomputed schedule is used.
- Preserve termination semantics explicitly: finish/crash is terminal for
  bootstrapping; pure time-limit truncation is not. Record `finished`,
  `progress`, `damage`, `retire_reason`, `terminated`, and `truncated`.

## Common Adapter

Add an algorithm-neutral layer instead of extending PPO-specific functions:

- `ObservationSpec`: shape, dtype, channel order, normalization, and optional
  control-plane fingerprint.
- `ActionSpec` and `ActionAdapter`: native algorithm action space, official
  HAIC action space, clipping/order, frame-skip unit, and fingerprint.
- `Transition` and `EpisodeCollector`: applied action, raw reward, terminal
  flags, discount, track/seed metadata, and no cross-episode leakage.
- `ReplayAdapter`: transition replay for DrQ-v2 and episode/sequence replay for
  world-model algorithms. Sequence samplers must never cross episode bounds.
- `PolicyAdapter`: `reset_episode()`, stateful `act(observation, deterministic)`,
  CPU load, and deterministic trace output.
- `TrainerAdapter`: collect/update, checkpoint, restore, RNG state, replay
  metadata, and source hashes.
- Generic manifest: algorithm, source commit, environment/action/observation
  fingerprints, reward contract, frame skip, max steps, seeds, dependency
  lockfile, model/optimizer/replay/RNG paths, and hardware.
- Generic evaluator bridge: reuse the existing isolated worker, fixed
  screen/confirmation/blind matrices, action trace hash, CPU reload agreement,
  latency, RSS, and archive smoke test.

No optional JAX, Hydra, TorchRL, or algorithm-specific dependency may be
imported by the submission runtime. Keep each training stack isolated and make
the exported agent dependency-minimal.

## Stage 0: Harness Gate

Implement and test the common adapter before implementing an algorithm.

- Build a random-policy collector and replay round-trip test.
- Verify CHW/HWC and uint8/float conversions with exact frame ordering.
- Verify action mapping, bounds, one update per high-level step, and no extra
  frame repeat.
- Verify finish, crash, truncation, reset, and sequence-boundary behavior.
- Verify CPU adapter load/act/reset parity on two episodes and a 5-second action
  budget.
- Verify checkpoint manifests and source hashes are sufficient to reconstruct
  semantics without `runs/_latest` or mutable adjacent files.
- Run a 2,000--10,000-step random/pipeline smoke only; it is not a score.

Failure at this stage blocks all four algorithms.

## Stage 1: DrQ-v2

DrQ-v2 is the first implementation because it is a PyTorch, model-free,
continuous pixel-control algorithm and can reuse the current CPU/Torch stack.
The primary source is the archived MIT-licensed
`facebookresearch/drqv2` implementation and paper
`https://arxiv.org/abs/2107.09645`.

Implement a small native adapter, not an SB3 wrapper:

- uint8 episode replay with three-step returns, terminal-aware sampling, and a
  fixed warm-up period. Start at 100,000 transitions rather than blindly
  allocating the reference 1M buffer: a full four-frame HAIC uint8 stack is
  about 2.63 GiB per 100,000 transitions before metadata;
- random-shift pixel augmentation applied consistently to critic/actor views;
- DDPG-style actor, twin critics/target critics, target updates, exploration
  schedule, and deterministic evaluation actor;
- keep internal actions in symmetric `[-1, 1]^3`, map gas/brake with
  `(u + 1) / 2`, and store that normalized action in replay. Convert the
  already-normalized HAIC pixels to uint8 only once, then use `obs / 255 - .5`
  inside the encoder;
- configurable replay/update ratio and batch size, with all RNGs in the
  manifest;
- standalone Torch actor/encoder export for the existing CPU submission
  contract.

DrQ-v2 gates:

1. Smoke: finite losses, replay occupancy, action bounds, checkpoint restore,
   deterministic CPU trace, and no memory growth over 2,000--10,000 steps.
2. Pilot: one fixed seed and one fresh screen at 131,072 environment steps.
   Stop for NaNs, replay/terminal bugs, action saturation, or inference over
   the submission budget.
3. Matched run: four training seeds, the same sampled-track schedule and
   budget as the PPO control, with an explicit CPU checkpoint selection.
4. Confirmation and blind: require nonzero unseen completion before promotion;
   otherwise retain the best progress result only as a rejected experiment.

Expected decision: DrQ-v2 is the most likely first viable replacement because
its training and export path can stay Torch-only. A progress gain without a
finish is not sufficient.

## Stage 2: DreamerV3

Use the official `danijar/dreamerv3` source and its Nature/paper reference
`https://arxiv.org/abs/2301.04104` in an isolated training environment. Do not
add JAX/Flax/Embodied dependencies to the submission environment. The official
JAX route is the fidelity reference; a pinned SheepRL/TorchRL prototype may be
used only as an explicitly labeled compatibility experiment because their
release cadence and APIs differ from the current Torch 2.1 environment.

Required adapter work:

- wrap the Gymnasium environment as an `embodied.Env`-style stream with
  `image`, reward, `is_first`, `is_last`, and `is_terminal`;
- sequence replay with explicit discounts and episode boundaries;
- map the continuous HAIC action to the library's symmetric action head and
  back through the canonical adapter;
- reset and carry RSSM state in `PolicyAdapter`; PPO-style stateless
  `predict()` cannot be reused;
- record world-model, actor, value, optimizer, replay, and RNG checkpoints;
- first prove CPU Torch/JAX inference and package limits with a compact model;
  only then consider a larger model or GPU training.
- encode `is_last = terminated or truncated`, but set `is_terminal` only for
  crash/off-track or a successful finish; a pure maximum-step truncation must
  remain bootstrap-able. Keep the aggregated four-frame reward and actual
  applied action in the sequence.
- do not apply PPO `VecNormalize` reward normalization. DreamerV3's symlog and
  two-hot reward/value targets are part of its own scale contract. Reset RSSM
  latent state, previous action, and any smoother state together at episode
  boundaries.
- begin with replay ratio `16` or `32`, 8--16 environments, 64-step sequences,
  and `0.1--0.25M` policy decisions. Do not copy the reference high replay
  ratio or 1M-step visual-control budget before the adapter is proven.

DreamerV3 gates:

1. Synthetic/short real-episode world-model reconstruction and reward/terminal
   calibration; pixel loss alone is not a pass.
2. 10,000-step online smoke with imagination loss finite and action sensitivity
   verified by intervention tests.
3. `0.1--0.25M`-decision pilot and CPU screen against the DrQ-v2/PPO control.
4. Full matched run only if the pilot produces real-environment progress and
   meets recurrent reset, latency, memory, and package gates.

Prefer TorchRL's DreamerV3 components as the long-term PyTorch path, with
SheepRL in an isolated Python 3.11 environment as a faster prototype. The
official JAX runtime has the highest fidelity but is a poor final submission
dependency, and archived/out-of-date Dreamer ports are not acceptable. If no
stack can be isolated or exported under the CPU contract, stop at a
training-only feasibility report rather than silently shipping a different
algorithm. Scale to `0.5--1M` decisions across three seeds only after the
100k--250k pilot shows real-environment utility.

## Stage 3: TD-MPC2

Use the official MIT-licensed `nicklashansen/tdmpc2` source and
`https://arxiv.org/abs/2310.16828`. The official implementation is GPU-first,
expects TorchRL/TensorDict/Hydra-era dependencies, and uses latent MPPI
planning. It expects RGB/64x64-like inputs while HAIC supplies 4x84x84
grayscale, so the adapter must make preprocessing explicit.

Implement in an isolated environment:

- compact 5M-class latent dynamics/Q model first;
- episode trajectory replay, terminal-aware targets, and deterministic planner
  seed/config;
- resize 84px observations to the upstream 64px encoder contract, normalize
  with the upstream pixel convention, use four input channels, and apply the
  symmetric continuous action mapping;
- planner horizon/sample/iteration configuration recorded in the checkpoint;
- standalone CPU inference path that does not import TorchRL or the trainer.

The official reference is GPU-first (Torch 2.7-era TorchRL/TensorDict/Hydra)
and its published 5M planner uses roughly 512 samples and six MPPI iterations.
Treat that configuration as a GPU reference only. Also test a small model and
reduced population/iteration planner, because one HAIC action must satisfy the
5-second CPU limit. Keep replay uint8/64px or a bounded capacity; a 1M replay
of full 84px float stacks is not a safe starting point. Track model-reward and
termination calibration explicitly because visited-tile rewards, damage, and
finish crossing are not fully observable in four camera frames.

Freeze the comparator before porting: use the explicit current continuous PPO
run `runs/20260921-000225_ppo-cnn-continuous-box-control-v1`, not `_latest`.
Evaluate three TD-MPC2 inference variants independently: policy-prior-only,
the published 512-sample/6-iteration MPPI planner, and a CPU-feasible reduced
planner such as 64x3 or 128x4. Train and evaluate each with its actual planner
configuration; never train with the published planner and silently reduce it
only during export. Fix a per-episode planner RNG or deterministic elite mode
so repeated CPU traces remain meaningful.

TD-MPC2 gates:

1. Dependency and synthetic planner smoke without modifying the project
   environment.
2. GPU 10,000-step pilot if CUDA is available; otherwise stop and record the
   hardware blocker rather than pretending CPU parity.
3. CPU planner benchmark at submission action limits, including RSS and model
   archive size. The default 512-sample/6-iteration planner is not assumed to
   be deployable.
4. 131,072-step screen, then matched training only if both learning and CPU
   planner gates pass.

TD-MPC2 is a potentially strong sample-efficiency candidate, but its planner
   and dependency boundary make it less likely than DrQ-v2 to be a practical
   HAIC submission without a deliberately small model.

## Stage 4: Dreamer 4

Treat Dreamer 4 as a feasibility/reproducibility stage, not as a routine
version bump. The primary source is
`https://arxiv.org/abs/2509.24527` and
`https://danijar.com/project/dreamer4/`. As of this plan there is no
author-provided official repository or checkpoint for the paper-faithful
agent. The paper describes a large offline Minecraft system with a causal video
tokenizer, action-conditioned transformer, flow/shortcut objectives, and a
roughly 2B-parameter system.

Before any code is copied:

- lock a specific source commit and license;
- reject unofficial/incomplete repositories as production dependencies;
- define a compact HAIC variant, offline dataset, continuous action encoding,
  reward/finish/damage targets, and leakage-free held-out tracks;
- test action sensitivity and reward/termination calibration, not only pixel
  prediction;
- enforce 500 MB archive, 1 GB process RSS, CPU Torch 2.1, initialization,
  and action latency limits.

The default decision is **do not implement paper-faithful Dreamer 4** for the
submission path. If research value justifies a pilot, make it a separately
named Dreamer-4-inspired world-model experiment with a strict compute budget
and a final distillation target into a small CPU actor. It cannot be promoted
without real-environment completion and a valid package.

## Matched Evaluation Matrix

For each implemented algorithm, record one explicit run directory and never
infer a candidate from `_latest`.

- Smoke: 1 seed, 2,000--10,000 steps, pipeline and checkpoint tests.
- Pilot: 2 seeds, 131,072 steps, fixed screen cells only.
- Matched: 4 seeds, same track/seed schedule, same observation/action/reward
  contract, and the same bounded environment-step budget.
- CPU selection: reload every checkpoint on CPU, compare deterministic traces,
  select by unseen finish rate then progress then completed lap time.
- Confirmation: two repeats on fresh cells for every candidate that passes the
  screen; include damage, retire reason, action trace, latency, and RSS.
- Blind: only the selected candidate, only after confirmation; no submission
  update unless repeated unseen completion is present.

Promotion thresholds:

- hard requirement: nonzero repeated unseen completion;
- secondary: improvement over the matched PPO control in completion/progress;
- operational: deterministic reload, CPU action budget, package size, and no
  prohibited runtime imports/network access;
- tie-breakers: damage and completed lap time, never raw progress alone.

## Decision Rules

- DrQ-v2 passes: use it as the first algorithmic replacement and still run the
  later stages only as research comparisons if resources permit.
- DrQ-v2 fails at the pipeline or CPU gate: fix the adapter once, then stop the
  branch; do not tune endless hyperparameter sweeps.
- DreamerV3 passes where DrQ-v2 fails: prefer it only if recurrent CPU state,
  export, and completion gates all pass.
- TD-MPC2 passes learning but fails planner/resource gates: retain its learning
  evidence, reject it as a submission algorithm, and do not package trainer
  dependencies.
- Dreamer 4 lacks a locked official source or violates resource limits: mark
  it blocked with the primary-source evidence and move to the most viable
  earlier algorithm.

Every stage ends with a short artifact summary in `CONTEXT.md`, but temporary
logs, replay buffers, model files, and evaluation matrices remain in their
explicit `runs/` and `evaluations/` directories. No commit or push is part of
this planning step.
