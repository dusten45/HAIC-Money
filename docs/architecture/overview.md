# Architecture Overview

This document maps current code boundaries. Executable code and tests remain the
source of truth for behavior; historical design documents are retained under
`docs/architecture/history/`.

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

- `agent.py` selects supported inference payloads and keeps imports lazy where a
  package mode does not need a training stack.
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
  loop evaluation, imitation, and package-building paths.
- `local_simulator/` and `web_simulator/` support custom tracks, logs, and local
  replay. Custom-track performance is local generalization evidence only.

## Evaluation and Packaging

- `evaluate_policy.py` evaluates a fixed candidate on local protocol partitions and
  binds screen, confirmation, and blind receipts.
- `run_drqv2_matched.py` launches the frozen matched DrQ-v2 study machinery.
- `training/evaluate_closed_loop.py` evaluates the visual PPO/planner and Track Lab
  paths; it is a separate local evaluation route.
- `package_submission.py` and `training/package_submission.py` package different
  inference paths. Reuse the matching existing packager and tests rather than
  recreating a validator in a workflow.

See [`docs/evaluation/protocol.md`](../evaluation/protocol.md) for what can be
compared and [`docs/workflows/prepare-submission.md`](../workflows/prepare-submission.md)
for the external-package gate.

## Change Rules

Architecture changes must record the affected compatibility boundary, update this
map when the boundary changes, and preserve a testable official submission path.
Training-only labels, teachers, and simulators must not silently enter submission
inference.
