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

A TRAIN cell (one TRAIN road identified by `track_id`, `geometry_seed`, and relevant
obstacle/reset conditions) is fresh for a *particular* study only if its declared
exclusions and prior-use requirement are met: inspect frozen allocations, active
reservations, actual environment interaction (including partial/aborted resets),
prior-data/episode ledgers, and any earlier exposure of its geometry or outcomes.
Check geometry-seed exclusions across track IDs where the study requires them;
multiple obstacle variants are not independent geometries. An independently
changing DrQ/Dreamer/RLPD protocol or unrelated file is not by itself a TRAIN
freshness violation. Hash the specific evidence inspected for provenance, but
report repository-wide inventory changes separately as warnings, not as a reason
to label an unrelated candidate `BLOCKED`. Unknown records relevant to the
candidate must still block until resolved. Where fresh TRAIN-DIAGNOSTIC is an
explicit study requirement, previously used diagnostic roads remain excluded.

A declared TRAIN reservation prevents another lane from claiming the same road
even before any driving. Retiring an unobserved TRAIN reservation does not itself
prove reuse is safe: verify zero interaction and zero outcome/geometry exposure,
record an explicit release decision, then re-audit before any reallocation. Old
retired allocations are not silently recycled. Audit again immediately before
reservation/protocol freeze and before the first reset; a read-only audit is not an
atomic claim. Neither this TRAIN-only rule nor a release decision makes a consumed
screen, confirmation, or blind cell fresh, allows tuning on confirmation/blind,
or relaxes their predeclared gates. Official submission and model confirmation
still require separate authorization.

For a prospective RLPD G1 study,
`python -B -m scripts.audit_rlpd_g1_coverage_seeds --seed-start N` emits the
**v2 read-only** candidate inventory (no protocol or
permission to reset). Its `collisions`/`blockers` determine `BLOCKED`; global
experiment/source inventory changes and unrelated incomplete histories appear
under `provenance_warnings`, with source-specific SHA and the limited claim
`no_known_recorded_overlap`. Only after separately predeclaring a candidate
batch, review the warnings and run the same CLI with `--reserve --study-id ID`
to re-audit within the shared TRAIN claim lock and write immutable per-seed
claims. Its output retains `locked_audit` (the exact per-source hash inventory
examined under the lock), `locked_claim_audit`, and `train_claims_sha256`.
Freeze a new source-pinned `haic-rlpd-g1-coverage-protocol-v1` protocol with
the exact 24 TRAIN `cells`, `study_id`, and `train_claims_sha256` from the claim
output. Before the first reset, re-audit with the same `--seed-start N` plus
`--self-study-id ID`, `--self-protocol-path experiments/FILE.json`, and
`--self-protocol-sha256 SHA`; only exact own claims and the frozen protocol
SHA/digest are waived. Foreign claims or new interaction still block. A normal
read-only re-audit *without* self mode intentionally blocks the claimed batch.
The registry is cooperative: an independent lane bypassing it
can still race; check talk, namespace allocation and live ledgers. The historic
G0 v1 auditor and per-reset whole-catalog checks remain byte-for-byte frozen
with that completed run; its v1 receipt is not a G1 certificate.

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
