# Project Context

## Objective And Algorithm Order

Replace PPO with a submission-capable algorithm that repeatedly finishes unseen
official `variables-6` tracks. The fixed order is **DrQ-v2, DreamerV3, TD-MPC2,
Dreamer 4**. PPO experiments are historical evidence, not an active improvement
project. `PLAN.md` contains the frozen contract and promotion gates.

## Frozen Environment And Deployment Contract

- Do not change `core/`, `env_wrapper.py`, or `damage.py` for experiments.
- Four `84x84` grayscale frames, CHW float32 `[0,1]`; continuous official
  `[steer,gas,brake]` bounds `[-1,0,0]..[1,1,1]`. No fifth plane, smoothing,
  reward normalization, finish shaping, or collision penalty for DrQ-v2.
- One decision per `CarEnvironment.step`; frame skip 4 occurs inside the
  environment only. Pure time limits bootstrap; finishes/crashes/off-track do not.
- A finish requires 95% progress AND a valid forward finish crossing. Geometry
  is seed-driven; `track_id` changes obstacles. Reserve geometry seeds across
  all training track IDs, not just complete `(track_id,seed)` pairs.
- Exported CPU actor results are authoritative. Selection order: unseen finish
  rate, mean progress, completed lap time. Repeated deterministic runs are not
  independent evaluation cells or training seeds.
- Official Participant source checked at
  `1c11db8afc2fbfcfb610672b7ee0ecd122c97741`: Python 3.11, Torch 2.1 CPU,
  NumPy 1.26; initialization 10 s, reset/action 5 s, process 1,024 MB,
  ZIP 500 MB. Submission Python must not import prohibited modules such as
  `pathlib`, `os`, `subprocess`, or trainer dependencies.

## Current DrQ-v2 Evidence

- Stage 0 real-environment 2,000-step smoke, CPU 10,000-step smoke and GPU
  restore/export smoke passed historically. Native implementation is in
  `common_adapter.py`, `drq_v2.py`, and `train_drqv2.py`.
- Completed pilot: `runs/20260921-043514_drqv2-pilot-131072/`, training seed 0,
  tracks 1--4, sampler 917, 131,072 high-level steps, batch 64, warmup 10,000,
  replay 100,000, one update per step, raw rewards and no smoothing.
- Its exported CPU actor recorded **4/24 finishes (16.7%), progress 0.807180419,
  damage 0.45**. This is the strongest migration signal, not final promotion.
- The actor (1,179,316 bytes) and checkpoint (1,204,218,555 bytes) are present on
  this server. Actor SHA256:
  `e0483df7c7887e9fd9e670450e223ec1b6f2598224ee23f9efe329cce4a2681e`.
- Historical provenance is incomplete: the exact 24 screen cells, per-cell
  traces, lap times, full environment hashes and command line were not saved.
  Do not infer the old grid from CLI defaults or claim proven historical
  disjointness. The old `cpu_action_parity` flag checked same-device restore,
  not actual exported CPU parity.

## Current Runtime

- Keep the working training environment: `.venv` points to `/venv/main`, Python
  3.11.14, Torch 2.11.0+cu128, RTX 5070 Ti. Actual CUDA matrix multiplication,
  convolution backward and Adam updates passed on 2026-09-21. Do not run a lock
  sync that downgrades it to Torch 2.1 or redesign the training image.
- Separate gate interpreter: `/tmp/kilo/haic-cpu21/bin/python`, Torch 2.1.0+cpu,
  NumPy 1.26.0, Gymnasium 0.29.1, OpenCV 4.8.1.78. SB3 2.2.1 is installed only
  for harness compatibility, not needed by DrQ inference.
- Historical pilot actor passed this CPU runtime with SB3 imports blocked:
  two identical 401-action real episodes on smoke cell `(1,0)`, peak process
  RSS 286 MiB, maximum measured action latency below 1 ms. This is an adapter
  compatibility smoke, not a completion or submission-package gate.
- Temporary CPU environment can be recreated with Python 3.11 and the above
  pinned versions, using the official PyTorch CPU wheel index for Torch only.

## Historical PPO Evidence

- Sampled PPO `runs/20260920-022920_ppo-cnn-episode-sampler-v1/`: 1,048,576
  decisions, 0/24 development finishes (progress 0.429), 0/32 confirmation
  (0.355), 0/24 blind (0.489). Legacy control finished only 2/24 blind cells.
- Curriculum failed (0/32 confirmation, 0/24 blind, progress 0.334). Collision
  penalty 5.0 also failed: control/treatment 0/32, progress 0.453/0.431.
- EMA, Markov-plane and MultiDiscrete variants did not establish completion.
  MultiDiscrete V1 progress 0.611 versus continuous control 0.382 remains
  historical progress evidence only. The gentle-turn V2 run was interrupted
  without a checkpoint; do not resume it during the migration gate.
- Historical GPU metrics are not comparable with pinned CPU metrics. Old
  confirmation seeds 20101--20108 and blind seeds 20201--20208 were repeatedly
  reused; they are not fresh partitions.
- The named continuous PPO control uses 5 channels, EMA 0.35, finish shaping
  and reward normalization. It is NOT a fully matched raw-reward DrQ control.
  SB3's first seeded reset also overrides the current PPO sampler's declared
  master with policy-seed-plus-worker streams. Do not claim identical training
  schedules from equal master-seed numbers.
- The archived submission is an old PPO baseline, not a migration candidate.
  Never infer the current candidate from mutable `runs/_latest`.

## Current Priority

1. Finish isolated exported-CPU periodic checkpoint selection, deterministic
   reload/resource gates, CUDA RNG restore, circular-replay regression tests,
   explicit episode schedule and safe partial-episode resume.
2. Freeze and execute `experiments/drqv2-promotion-v1.json`: new screen IDs
   101--103/seeds 31001--31008, confirmation IDs 111--114/seeds 31101--31108,
   reserved blind IDs 121--123/seeds 31201--31208, each with two CPU reloads.
   These are disjoint from recorded prior cells; undocumented historical pilot
   cells and its realized training schedule cannot be proven disjoint.
3. Require repeated nonzero confirmation and CPU gates before scale-up. The
   predeclared next design uses four DrQ seeds, the same indexed sampled-track
   stream, 1,048,576 decisions each and explicit CPU checkpoint selection. This
   matches PPO's budget, not its full historical contract. Reserve new final
   confirmation/blind cells for scale-up; do not reuse pilot validation cells.
4. If confirmation fails, inspect saturation, exploration, representation,
   replay/update and terminal telemetry before one evidence-backed intervention.
   Do not sweep hyperparameters or begin DreamerV3/TD-MPC2 prematurely.
5. No submission release or lap-time optimization until completion and blind
   gates pass. Preserve immutable evaluation artifacts and explicit actor hashes.
