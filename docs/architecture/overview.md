# Architecture Overview

This document maps current code boundaries. HAIC-Money is an experimental research
fork of the official Participants template: the upstream repository defines the
competition contract, while this repository adds research, evaluation, packaging,
and local-simulation code. Executable code and tests remain the source of truth for
behavior; historical design documents are retained under `docs/architecture/history/`.

## Official-Compatibility Boundary

- `core/`, `env_wrapper.py`, and `damage.py` implement the local official-environment
  mirror. Do not modify them to improve a research result without first treating the
  change as an environment divergence, not an HAIC improvement.
- `agent.py` is the submission entry point. It exposes `Agent`, `reset`, and `act`;
  its valid input/output contract is maintained in `docs/competition/restrictions.md`.
- `local_runner.py` exercises the local mirror. It is a development tool, not an
  official server or a substitute for the official competition source.
- `common_adapter.py` owns shared observation, action, transition, replay, policy,
  trainer, and evaluator contracts for native algorithm stacks. Its constraints are
  a local research contract; official source still governs the submission boundary.

## Inference Implementations

- `agent.py` dispatches `model.pt` to a raw baseline state dict, a tagged
  `haic-drq-v2-actor-v1`, or a tagged `haic-dreamerv3-actor-v1` export. It keeps
  imports lazy where a package mode does not need a training stack.
- If `model.pt` is absent, `agent.py` can fall back to the historical HAIC
  `policy.pt` and optional `dynamics.pt` PPO/CEM route. The active root packager does
  not package that fallback, so it is not the current DrQ/Dreamer submission path.
- `drq_v2.py` and `train_drqv2.py` provide the native DrQ-v2 training/export path.
- `dreamer_v3.py` and `train_dreamerv3.py` provide the native recurrent DreamerV3
  path and CPU actor export.
- `haic_agent/` contains the visual PPO/dynamics/CEM-era inference modules, runtime
  configuration, observation helpers, and the training-only corridor teacher.

These paths are alternatives and historical research layers, not evidence that a
single package combines every algorithm. Candidate provenance must name its exact
entry point, checkpoint, configuration, and source revision.

## Training and Local Research

- `train.py` and `export_policy.py` are the earlier Stable-Baselines PPO path.
- `training/` contains the visual PPO, latent dynamics, Track Lab site-map, closed-
  loop evaluation, imitation, and package-building paths. These PPO/CEM paths are
  historical research/implementation evidence, not the current submission route.
- `local_simulator/` and `web_simulator/` support custom tracks, logs, and local
  replay. Custom-track performance is local generalization evidence only.

## Evaluation and Packaging

- `evaluate_policy.py` evaluates a fixed candidate on local protocol partitions and
  binds screen, confirmation, and blind receipts.
- `run_drqv2_matched.py` launches the frozen matched DrQ-v2 study machinery.
- `training/evaluate_closed_loop.py` evaluates the visual PPO/planner and Track Lab
  paths; it is a separate local evaluation route.
- `package_submission.py` is the active root packager for baseline, DrQ-v2, and
  DreamerV3 `model.pt` exports. It creates a local ZIP/manifest and never uploads.
  It packages only its recognized inference modules, so it cannot represent every
  officially permitted dependency/module layout.
- `training/package_submission.py` is the separate visual PPO/CEM pilot path. It has
  different runtime checks and output behavior; it is not the active DrQ/Dreamer
  provenance path.

Reuse the matching existing packager and tests rather than recreating a validator in
a workflow. Their limits remain local evidence, not official-server validation.

See [`docs/evaluation/protocol.md`](../evaluation/protocol.md) for what can be
compared and [`docs/workflows/prepare-submission.md`](../workflows/prepare-submission.md)
for the external-package gate.

## Change Rules

Architecture changes must record the affected compatibility boundary, update this
map when the boundary changes, and preserve a testable official submission path.
Training-only labels, teachers, and simulators must not silently enter submission
inference.
