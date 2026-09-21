# Project Context

This file records the current durable state of the HAIC project. Replace stale
findings instead of accumulating experiment history here; detailed artifacts
belong in `runs/` and `submissions/`.

## Objective

Build a reliable `variables-6` CarRacing policy that completes unseen official
tracks consistently, then improve lap time while retaining a small, CPU-safe
submission implementation.

## Environment And Evaluation Contract

- The local `core/`, `env_wrapper.py`, and `damage.py` files are treated as
  official environment code and must not be changed for policy experiments.
- Observations are four `84x84` grayscale frames in `[0, 1]`; actions are
  continuous `[steer, gas, brake]`; the default action frame skip is 4.
- A finish requires at least 95% progress and a valid forward finish-line
  crossing. Progress alone is not a finish.
- Track geometry is primarily determined by `seed`; `track_id` changes the
  deterministic obstacle-placement stream. Training and evaluation must vary
  both independently.

## Current Baseline

- PPO `CnnPolicy` with reward normalization and a +100 finish bonus during
  training. Default stabilization settings are LR `1e-4`, five PPO epochs,
  clip range `0.2`, and target KL `0.03`.
- Training workers now sample a fresh `(track_id, seed)` pair every episode
  from reproducible independent streams. Fixed worker tracks are retained only
  as an explicit comparison mode.
- Periodic checkpoints include matching `VecNormalize` state. Selection ranks
  holdout finish rate, progress, then lap time from a saved checkpoint freshly
  loaded on CPU; per-episode outcomes and CPU reload agreement are recorded.

## Best Completed Evidence

- The sampled-track run
  `runs/20260920-022920_ppo-cnn-episode-sampler-v1/` completed 1,048,576
  steps. Its CPU-selected final checkpoint had 0/24 finishes and 0.429 mean
  progress on its development holdout.
- Its immutable two-repeat confirmation on 32 new cells had 0 finishes, 0.355
  mean progress, and exact repeated action traces. Its separate 24-cell blind
  protocol had 0 finishes and 0.489 mean progress.
- The legacy final control completed 2/24 blind cells (8.3%, 21.78--23.22 s),
  while the sampled policy completed none. Neither is a submission candidate.
- The two-stage no-obstacle-to-official curriculum also completed 0/32
  confirmation cells and 0/24 blind cells (0.334 mean progress). It regressed
  against both the sampled policy and legacy control, so do not extend it.
- A matched continuation from the curriculum checkpoint confirmed that a native
  collision penalty of 5.0 is not worthwhile: control had 0/32 finishes,
  0.453 progress, and 0.675 damage; treatment had 0/32, 0.431, and 0.644.
  Do not spend the blind grid on either arm.
- An exploratory single-pass CPU screen of older candidates also found no
  reliable policy; do not use it for model selection.
- Historical GPU evaluation metrics are not comparable to current CPU results:
  the original 0.508 holdout progress replayed as 0.345 under the locked
  runtime, despite identical policy weights. Treat current pinned CPU results
  as authoritative.
- This is a progress baseline, not a viable final submission. The newer
  `runs/20260919-154015_ppo-cnn-baseline1-server/` run was interrupted before
  producing a model, metrics, or checkpoints; `runs/_latest` currently points
  to it and must not be used automatically.

## Deployment

- `export_policy.py` exports an SB3 policy to a Torch-only `model.pt` and
  validates deterministic actor parity plus the stateful smoothing trace.
- Submission inference in `agent.py` uses only Torch and NumPy. Package
  validation includes `action_smoothing.py`, records config/fingerprint
  provenance, and runs a reset-plus-sequence CPU smoke test.
- The archived submission was exported from an older baseline, not the current
  Baseline 1.1 candidate. Do not treat it as the preferred submission.

## Reproducibility

- The intended development target is Python 3.11, Gymnasium 0.29.1, SB3 2.2.1,
  NumPy 1.26.0, and Torch 2.1.0. The current local virtualenv imports these
  versions successfully.
- Historical baseline artifacts were trained with Torch 2.11.0+cu128 from a
  dirty worktree, while the intended CPU runtime is Torch 2.1.0. Archive and
  environment hashes, CPU/thread settings, and per-cell actions are now
  captured by `evaluate_policy.py` protocol artifacts under `evaluations/`.

## Active Direction

- Two independent EMA-0.35 pairs had zero completion. In the corrected-horizon
  pair, EMA raised fresh holdout progress from 0.387 to 0.493 and reduced
  applied steering delta from 0.514 to 0.180, but did not pass the nonzero
  completion promotion threshold.
- Do not run another alpha sweep or spend blind seeds. The next single
  intervention is a Markov 5-channel action-control representation: a
  constant-plane capacity control versus a plane encoding prior applied steering
  under the same EMA-0.35 transform.
- Action-control parity is implemented in training, evaluator, export payload,
  and root-only submission inference. The new architecture is incompatible with
  existing 4-channel checkpoints and must train from scratch.
- `ppo-cnn-markov-plane-control-v1` completed with 0/24 fresh holdout finishes
  and 0.448 progress. `ppo-cnn-markov-prior-steering-v1` is the matched fresh
  treatment; only its fifth-plane information differs.
- `ppo-cnn-multidiscrete-control-v1` completed with 0/24 fresh holdout finishes
  and 0.611 progress on its `13301`--`13308` grid. This is the best current
  progress signal but not a promotion result; run one same-grid continuous Box
  control before changing the discrete mapping or using confirmation/blind cells.
- The same-grid 5-channel continuous Box control completed with 0/24 finishes
  and 0.382 progress, confirming a substantial MultiDiscrete progress advantage
  without completion. Its deterministic MultiDiscrete policy almost never
  braked (81/9,788 decisions) and lacks a gentle `+-0.25` steering action;
  use this telemetry to justify one mapping change rather than a parameter sweep.
- `ppo-cnn-multidiscrete-steer-longitudinal-v2-gentleturn-v1` is the sole
  follow-up treatment. It retains the V1 mapping and adds symmetric `+-0.25`
  steering targets, changing the policy head from 8 to 10 logits while keeping
  every other V1 setting and partition fixed.

## Stateful Smoothing Evidence (2026-09-20)

- The matched steering-EMA implementation is complete across training,
  evaluation, export, and submission. Alpha `0.35` reduced applied steering
  delta but lost progress; alpha `0.6` was better than the no-op control.
- Alpha `0.6` run: `runs/20260920-165819_ppo-cnn-smoothing-steering-ema060-v1/`.
  Confirmation progress `0.3405` vs control `0.3040`, applied steering delta
  `0.2128` vs `0.3421`; blind progress `0.3161`, delta `0.2182`, finish `0/24`.
- Immutable alpha `0.6` artifacts:
  `evaluations/20260920T172901961075Z_checkpoint-v1-screen/`,
  `evaluations/20260920T173325219035Z_checkpoint-v1-confirmation/`, and
  `evaluations/20260920T174939980170Z_checkpoint-v1-blind/`.
- Completion remains absent, so no submission or lap-time update is allowed.
  Preserve this evidence while the Markov action-control hypothesis proceeds.

## Current Priorities

1. DrQ-v2 Stage 0 real-environment smoke passed at 2,000 steps with finite
   updates, checkpoint restore parity, and exported actor artifact. The 10,000
   step CPU smoke also passed with 9,001 finite updates, a 130 MB training
   checkpoint, and a 1.2 MB actor export. GPU 10,000-step restore/export smoke
   also passed. `drqv2-pilot-131072` is running with a fixed CPU exported-actor
   evaluation grid on completion. The completed pilot achieved 4/24 CPU-export
   finishes (16.7%), 0.807 progress, and 0.45 damage on unseen fixed cells; add
   periodic CPU checkpoint selection and fresh confirmation before scaling it.
2. The interrupted PPO V2 gentle-turn run has no selected checkpoint or result and is
   not evidence. Do not resume it while the algorithm migration gate runs.
3. Require repeated nonzero unseen completion before using confirmation or blind
   partitions, optimizing lap time, or updating submission artifacts.
4. Retain immutable checkpoint evaluation artifacts and CPU-selection evidence
   for every candidate.
5. Export and validate an immutable submission only after a policy finishes
   unseen tracks reliably.
