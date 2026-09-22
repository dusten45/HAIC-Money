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
  damage 0.45** historically. Preserve this evidence, not a promotion verdict.
- The actor (1,179,316 bytes) and checkpoint (1,204,218,555 bytes) are present on
  this server. Actor SHA256:
  `e0483df7c7887e9fd9e670450e223ec1b6f2598224ee23f9efe329cce4a2681e`.
- Historical provenance is incomplete: the exact 24 screen cells, per-cell
  traces, lap times, full environment hashes and command line were not saved.
  Do not infer the old grid from CLI defaults or claim proven historical
  disjointness. The old `cpu_action_parity` flag checked same-device restore,
  not actual exported CPU parity.
- Fresh CPU21 evaluation of that SAME actor is now complete. Screen IDs
  101--103/seeds 31001--31008: **0/24 finishes**, progress **0.721822419**, damage
  **0.458333333**, repeated twice with exact action traces. The screen gate failed.
- To complete the requested replication without erasing that failure, the
  already-frozen confirmation IDs 111--114/seeds 31101--31108 were run in explicit
  **non-promoting diagnostic mode**: **4/32 finishes (12.5%) on BOTH reloads**,
  progress **0.696825936**, damage **0.43125**, completed-lap mean **39,380 ms**.
  Thus the completion signal reproduced, but consistency/promotion did not.
- **No checkpoint promotion, blind evaluation, million-step scale-up or submission
  release.** DrQ-v2 remains first candidate; do not interpret this as rejection of
  the algorithm family. The implementation is a native variant, not an exact
  author reproduction: separate encoders/strides, deterministic Q1-only actor
  objective and independent actor augmentation differ from the author source.
- Decision, telemetry, limitations and immutable artifact paths:
  `experiments/drqv2-promotion-v1-result.json`. The diagnostic-only decision was
  committed before confirmation in `experiments/drqv2-confirmation-diagnostic-v1.json`.
  These 56 fresh cells have 16 geometry seeds; 112 reload executions are not 112
  independent trials. Confirmation seeds 31101--31108 are now consumed.

## Implemented Pipeline And Verification

- `train_drqv2.py` saves explicit immutable checkpoint directories, exports the
  actor, and invokes `evaluate_policy.py` in a separate CPU21 interpreter.
  Use `--run-dir`, `--protocol-file`, `--eval-python`, `--eval-freq`, and optionally
  `--eval-workers`. No `_latest` lookup. Best selection uses CPU finish/progress/lap.
- Full trainer checkpoints carry replay, optimizer/CPU/CUDA RNG, warmup/sampler
  state, current-episode action prefix and selected incumbent. `--resume` verifies
  identical runtime/source/protocol, reconstructs physics and checks observations.
  Legacy checkpoints remain loadable for evidence, not exact live-training resume.
- Circular replay reconstruction, cross-episode rejection, real CPU export parity,
  CUDA determinism, selection ties and actual actor-loss logging are regression-tested.
  Submission `Agent` loads DrQ exports without training/SB3 imports. Packaging accepts
  an explicit `actor.pt`, archives it as `model.pt`, and records both identities.
- Committed-code GPU smoke: `runs/20260921-drqv2-selection-gpu-gate/`, 2,000 steps,
  1,001 updates, CPU evaluation at 1,000/2,000 selected 1,000. Explicit continuation:
  `runs/20260921-drqv2-selection-gpu-resume/`, 3,000 total steps/2,001 updates, same
  incumbent retained, bounded replay. CPU and CUDA closed-loop split-run tests are
  bitwise exact with deterministic CUDA settings.
- Tests: **130 passed, 1 skipped**. The skip needs a separate server-reference
  checkout. CPU21 contract/deployment subset: **63 passed, 2 CUDA-only skips**.
  Official environment sources and the working training stack are unchanged.

## Current Runtime

- Keep the working training environment: `.venv` points to `/venv/main`, Python
  3.11.14, Torch 2.11.0+cu128, RTX 5070 Ti. Actual CUDA matrix multiplication,
  convolution backward and Adam updates passed on 2026-09-21. Do not run a lock
  sync that downgrades it to Torch 2.1 or redesign the training image.
- Separate gate interpreter: `/tmp/kilo/haic-cpu21/bin/python`, Torch 2.1.0+cpu,
  NumPy 1.26.0, Gymnasium 0.29.1, OpenCV 4.8.1.78. SB3 2.2.1 is installed only
  for harness compatibility, not needed by DrQ inference.
- All 112 fresh evaluation workers passed CPU gates and repeat traces: maximum
  whole-worker RSS about **344 MiB**, action **2.66 ms**, initialization **1.02 s**.
  Temporary root-only ZIP/actual packaging CLI smoke also passed in CPU21 (~1.1 MB).
  This is not an actual official submission-container or competition score receipt.
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

1. Keep the failed screen gate and successful diagnostic replication both visible.
   Do not promote using a diagnostic receipt or reuse consumed confirmation as fresh.
   Blind IDs 121--123/seeds 31201--31208 remain untouched.
2. The user authorized actual controlled follow-ups on 2026-09-22, not an algorithm
   switch. L2 is implemented and ALL four arms are RUNNING from scratch at
   `runs/20260922-drq-steering-l2-v1-fast/`, source revision `8e5fa46`.
   The initial `runs/20260922-drq-steering-l2-v1/` was stopped with every arm last
   logged at22,000 steps, BEFORE any checkpoint/evaluation. Its sealed
   `engineering_abort.json` preserves the reason and zero holdout consumption.
   Frozen protocol (unchanged):
   `experiments/drqv2-steering-logit-v1.json` (SHA256
   `348d1534a429a313a2927200f027f8e14b957f109aa7e2451fecb0ae1c76b235`).
   Control/L2 coefficients 0/0.001, seeds 0/1, 131,072 decisions each, batch64,
   warmup10,000, replay100,000, sampler917, tracks1--4. Only the actor steering-logit
   penalty changes. CPU checkpoint selection is at65,536/131,072.
3. `run_drqv2_matched.py` executes an immutable source snapshot, verifies exact
   environment/gradient/replay budgets and episode-stream prefixes, freezes selected
   actors before confirmation, and preserves sealed receipts. Fresh confirmation:
   IDs211--214/seeds32101--32108; blind221--223/seeds32201--32208. All164 reserved
   geometry seeds are excluded from new training. L2 must improve confirmation
   finish count in BOTH seeds, with nonzero treatment screen/confirmation and all
   CPU gates; only the pre-frozen finalist can enter blind. No progress-only success.
4. Prerequisites passed: coefficient-zero bitwise CPU/CUDA baseline parity, nonzero
   L2 gradient despite squash saturation, exact resume, two actual batch64 2k GPU
   smokes and CPU exports. Four-job64-step full runner smoke and idempotent recovery
   passed, correctly rejecting zero finishes without running blind. Artifacts:
   `runs/20260922-drq-l2-control-smoke/`, `runs/20260922-drq-l2-treatment-smoke/`,
   `runs/20260922-drq-l2-runner-smoke-v2/`. No real-study outcome exists yet.
   All150 DrQ experiment/deployment tests pass together. Profiling found128 scalar
   GPU-to-CPU RNG synchronizations per augmentation view. A tested CUDA-only gather
   preserves the exact scalar draw order/output/RNG and16 successive learner updates
   per arm; a single batched RNG draw was rejected because it changed CUDA RNG state.
   Live learning metrics also match all four original prefixes exactly through17,000
   steps; wall time per1,000 steps fell from about94.5s to31.5s. Operator state and
   restart evidence are in `experiments/drqv2-l2-execution.json`.
5. Offline evidence is in `experiments/drqv2-pre-l2-diagnostics.json`; reusable
   `diagnose_drqv2.py` compares all actors on one fixed replay-state sample under
   CPU21. History IS used; no new frame-skip/terminal bug was found. Logit saturation,
   heavy critic clipping, sparse finish-boundary replay and shift sensitivity are
   measured, not causal proofs. At nominal wheel limits, steering magnitudes above
   0.46 share saturated motor commands: restoring tanh gradients alone may not
   improve physical control. The small L2 smoke already demonstrates this distinction.
6. Wait for the matched outcome before selecting at most ONE additional justified
   single-axis trial. Padding4 versus1 is only a conditional candidate if L2 restores
   its mechanism but not completion and standardized shift sensitivity remains.
   Never combine a rejected L2 repair with that trial, sweep coefficients, weaken
   gates, or start another algorithm. Stop after1--2 controlled attempts without
   replicated improvement. Million-step scale-up remains conditional, not launched.

## Concurrent Upstream Work

- Remote `b1ad528` introduced separate visual-policy/planner/site tools while this
  session began. It was preserved with merge `9649c62`, not overwritten. Its lazy
  dispatch leaves explicit DrQ inference unchanged; those algorithms are not used
  or trained in this study. No dependency sync or official-environment edit occurred.
- Post-merge DrQ suites:91 host passes; CPU21 subset74 run/3 CUDA skips. The full
  merged suite had355 passes/14 skips and one unrelated
  `tests/test_submission_layout.py` failure: upstream `training/package_submission.py`
  uses inherited Linux `ru_maxrss` after CUDA-heavy tests. That test passes alone.
  This was not changed; the DrQ gate measures process-local `/proc/self/status` RSS.
