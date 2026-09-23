# Internal Evaluation Protocol

This is the reusable discipline for local/internal research. It is not an official
HAIC score protocol and does not authorize an official submission. For a concrete
study, the frozen JSON protocol and its result/run artifacts are authoritative.

## Sources of Truth

- Generic evaluator behavior: `evaluate_policy.py` and its tests.
- Matched DrQ-v2 study behavior: `run_drqv2_matched.py`, `train_drqv2.py`, and the
  linked protocol JSON in `experiments/`.
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
4. Freeze the selected checkpoint, actor hash, source/protocol hashes, and screen
   receipt before confirmation.
5. Run confirmation on its predeclared fresh internal cells. Do not re-rank,
   replace a seed/checkpoint, alter a parameter, or extend training after seeing it.
6. Open a reserved internal blind partition only when its protocol's confirmation
   gate passes. Blind is a terminal test for that decision, not an iterative tuning
   loop.

The evaluator records source/environment/protocol/checkpoint hashes, action traces,
reload agreement, latency, RSS, and termination details. A diagnostic confirmation
after a zero-finish screen is explicitly non-promoting and cannot authorize blind.

## Existing Matched DrQ-v2 Template

The completed L2 and padding records use four arms/runs: control and treatment for
training seeds 0 and 1. Each uses 131,072 environment decisions, `frame_skip=4`,
training track IDs 1-4, two candidate checkpoints (65,536 and 131,072 decisions),
and two isolated CPU reloads per evaluation cell. This is a historical template, not
a universal budget for DreamerV3 or another family.

For those studies, checkpoint selection is restricted to complete operational and
determinism-audited screen receipts and is lexicographic: canonical finish rate,
then average progress, then lower completed lap time; exact ties retain the earlier
checkpoint. The frozen treatment finalist is also chosen from screen evidence before
confirmation. Confirmation and blind are pass/fail gates only, never a replacement
selection stage. Exact grids, exclusions, and thresholds are in the corresponding
protocol JSON files.

## Freshness and Repeats

Fresh means unused under the declared internal protocol and excluded from training
where required. Two CPU reloads test deterministic reproducibility; they do not
double the number of independent tracks, geometry seeds, or training runs. Obstacle
variants sharing geometry are correlated and must not be reported as independent
training seeds.

## CPU and Package Gates

Current matched studies use a pinned local CPU runtime and compare exported actors
there. This is strong internal operational evidence, but it is not proof of the
official server container. The official package gate remains governed by
`docs/competition/restrictions.md` and
`docs/workflows/prepare-submission.md`.

## New Algorithm Families

Do not force a new algorithm into an old result grid without a compatible frozen
contract. First prove collection, terminal semantics, reset behavior, CPU export,
and package feasibility; then write a study-specific protocol with explicit
acceptance, rejection, and stop conditions.
