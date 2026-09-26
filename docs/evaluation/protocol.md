# Internal Evaluation Protocol

This is the reusable discipline for local/internal research. It is not an official
HAIC score protocol and does not authorize an official submission. For a concrete
study, the frozen JSON protocol and its result/run artifacts are authoritative.

## Sources of Truth

- Generic evaluator behavior: `evaluate_policy.py` and its tests.
- Matched DrQ-v2 study behavior: `run_drqv2_matched.py`, `train_drqv2.py`, and the
  linked custom protocol JSON in `experiments/`. Legacy built-in
  `checkpoint-v1-*` protocols cannot evaluate native DrQ actors.
- Visual PPO/Track Lab behavior: `training/evaluate_closed_loop.py` and its declared
  split manifest.
- One study's cells, exclusions, thresholds, hashes, and outcomes: its immutable
  protocol/result artifact, never a prose summary.

## Required Sequence

1. Define a candidate and a frozen protocol before training or evaluation.
2. Keep the comparison matched where possible: source/runtime, observation/action
   contract, reward/termination semantics, budget, checkpoint opportunities,
   training exclusion set, and every non-active treatment variable stay fixed.
3. Run development/screen cells only for checkpoint selection. CPU-exported actors
   and their local CPU results are authoritative for the existing DrQ-v2 machinery.
4. Freeze each screen-selected checkpoint and actor hash, source/protocol hashes,
   screen receipts, and any rule for choosing a blind finalist before confirmation.
5. Run confirmation on its predeclared fresh internal cells. Do not re-rank or
   replace a seed/checkpoint, alter a parameter, or extend training after seeing it,
   except for a one-time finalist choice among already screen-selected actors if
   the frozen protocol explicitly predeclares confirmation-based selection.
6. Open a reserved internal blind partition only when its protocol's confirmation
   gate passes. Blind is a terminal test for the predeclared decision, not an
   iterative tuning loop. The [RLPD entropy v5 protocol](../../experiments/pixel-rlpd-entropy-target-ablation-v5.json)
   predeclared a winning-target and within-target finalist choice after confirmation;
   it did not reselect checkpoints or retrain actors.

The evaluator records source/environment/protocol/checkpoint hashes, action traces,
reload agreement, latency, RSS, and termination details. A diagnostic confirmation
after a zero-finish screen requires explicit `--diagnostic-confirmation`; it consumes
the declared confirmation evidence but is non-promoting and cannot authorize blind.

## Existing Matched DrQ-v2 Template

The completed L2 and padding records use four DrQ-specific arms/runs: control and
treatment for training seeds 0 and 1. Each uses 131,072 environment decisions,
`frame_skip=4`, training track IDs 1-4, two candidate checkpoints (65,536 and
131,072 decisions), and two isolated CPU reloads per evaluation cell. This is a
historical template, not a universal evaluator or budget for DreamerV3 or another
family.

For those studies, a selectable result is eligible, determinism-audited, CPU-reload
matched, and free of operational failures. Checkpoint selection is lexicographic:
canonical finish rate, then average progress, then lower completed lap time; exact
ties retain the earlier checkpoint. All arm/seed runs complete before
`frozen_candidates.json` seals the candidates. The treatment blind finalist is frozen
from screen evidence before confirmation. Confirmation and blind are pass/fail gates
only, never a replacement selection stage. Exact grids, exclusions, thresholds, and
receipt hashes are in the corresponding protocol JSON files.

## Freshness and Repeats

Fresh means no known prior recorded use under the declared internal protocol and
exclusion from training where required. Historical pilot screen scheduling is
incomplete, so freshness is not proof of global non-use. Two CPU reloads test
deterministic reproducibility; only canonical `repeat == 0` cells enter aggregates,
and repeat 1 does not double the number of independent tracks, geometry seeds, or
training runs. Obstacle variants sharing geometry are correlated and must not be
reported as independent training seeds.

## CPU and Package Gates

Current matched DrQ studies use a pinned local Linux/Python 3.11, CPU-only Torch
2.1.0, NumPy 1.26.0, Gymnasium 0.29.1, OpenCV 4.8.1.78 runtime with CUDA unavailable
and one Torch intra/inter-op thread. Their gates are init <=10 s, agent reset <=5 s,
max action <=5 s, and whole-worker `VmHWM` <=1 GiB. This is strong internal
operational evidence, but it is not proof of the official server container. The
official package gate remains governed by `docs/competition/restrictions.md` and
`docs/workflows/prepare-submission.md`.

## New Algorithm Families

Do not force a new algorithm into an old result grid without a compatible frozen
contract. First prove collection, terminal semantics, reset behavior, CPU export,
and package feasibility; then write a study-specific protocol with explicit
acceptance, rejection, and stop conditions.
