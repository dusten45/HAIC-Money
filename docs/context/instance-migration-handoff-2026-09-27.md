# Vast.ai Instance Migration Handoff (2026-09-27)

## Shutdown Boundary

The user directed **no new experiments** while retiring this instance. This
document preserves the state as of 2026-09-27 05:35 UTC; recheck live processes,
Git and archive bytes immediately before transfer/deletion. The TD-MPC2 v2
resource-check wakeup was cancelled. The only TD-MPC2 Kilo background process
had already failed and exited; no TD-MPC2 v2 road was ever reset. Another Kilo
session started DrQ final-source seed-1/easy-retention training after the stop
request. It was interrupted with SIGINT **after** its step-16,384 checkpoint
completed; subsequent `ps` checks found no Python learner/evaluator. Preserve
the entire partial run. Do not infer every Kilo session is quiescent forever
from one process snapshot; other sessions must independently honor the stop.

This is a preservation handoff, not authorization to resume training,
evaluation, confirmation/blind use, model confirmation, or official submission.
The first action after cloning is to verify the Git tip and transferred
artifacts, not to run a learner.

## Research At Stop

| Lane | Verified state and judgment | Exact continuation boundary |
|---|---|---|
| DrQ-v2 controls and r6/r7 | Native DrQ-v2 controls remain the internal baseline; r6 six-arm geometry-mix and r7 twelve-arm retention studies finished on TRAIN/reused development cells. r7 failed its predeclared old-success retention contract; no official model promotion. See [current state](current-state.md) and [index](../experiments/INDEX.md). | Results/protocols are complete; preserve replay-bearing checkpoints and diagnostic traces. Do not re-open consumed diagnostic or protected cells. |
| DrQ final-source replay v1 | Two source collections each sealed 100,000 TRAIN decisions and 99,997 valid replay starts; **five of six** learner arms completed their 32,768-decision/22,768-update budgets. Seed-1/easy-retention was stopped at **21,037 logged additional TRAIN decisions / 11,037 updates**, without a final result or diagnostic. No six-arm result, comparison, or promotion follows. [Stop receipt](../../experiments/drqv2-final-source-replay-v1-migration-stop.json). | The last full saved checkpoint is `runs/20260927-drqv2-final-source-replay-v1/learner-1-easy_retention-final_source/checkpoints/step-000016384/checkpoint.pt`, SHA-256 `9081cadf89539cfbde96206638f504c70e8b9e6659b57e80e2683fbdf0ef2637`, 802,467,664 bytes. Its actor SHA-256 is `3751318d80d94c43fdf88d258b65d7da7e6a61865acbdddec0eab1330e7c1181`. **4,653** later decisions are consumed but not checkpointed; the sealed metrics ledger SHA-256 is `79a46419b5cd3c6badea4d278aa1f5ac6efb2d34bb20a863fae91043baaf209c`. No exact same-protocol resume or step-32,768 checkpoint. |
| Pixel RLPD | Long-horizon seed-11 is an **internal** limited-geometry blind-tested candidate, not official. Its 131,072-step actor is `runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt`, SHA-256 `f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1`. V5 author-target seed-50 is another consumed-development finalist, not a matched replacement. G0 is observational diagnosis; G1 seed freshness remains unresolved. | Preserve final actors **and** replay/source checkpoints, prior data, confirmation/blind receipts. No further evaluation during migration. |
| DreamerV3 | Research line **closed under the current design/budget** after failed long-horizon prior/H8 gates. Latest two H8 checkpoints are world-model diagnostics (`actor_trained=false`, `promotion_eligible=false`), not deployable actors. See [current state](current-state.md) and the archived recovery plan. | Preserve all B1 and reused-source collections, score receipts and both diagnostic checkpoints for interpretation; no queued experiment. |
| Independent TD-MPC2 | Paper/current-repository 5M-size-class pixel baseline with HAIC-only adapter implemented and synthetic/CPU checks passed. Frozen reused-TRAIN **v1** recorded 10,061 decisions, 29 complete episodes, 10,000 pretraining plus 60 postseed updates at the last boundary; planner reset then failed on an inference tensor. No completed planned episode, paired prior/MPPI test, or driving-improvement judgment. See [failure receipt](../../experiments/tdmpc2-reused-train-pilot-v1-failure.json) and [plan](../plans/active/tdmpc2-pixel-online-baseline.md). | Preserve `runs/tdmpc2-reused-train-20260927-v1/boundary.pt`, SHA-256 `d3502e430a0c4bb3be9a993ab16434d6b5400afdc201177080f8cc4d035d8b4c`, 373,692,725 bytes, **together with** `training.jsonl`, SHA-256 `ff909545fe87fac34e649e016586b8babc22ab296aa33ad935ca4a1e13871961`. The ledger ends `checkpoint -> reset_intent -> partial`: valid saved boundary but **exact v1 resume forbidden**. Repaired reset is committed; separate [v2 protocol](../../experiments/tdmpc2-reused-train-pilot-v2.json) SHA-256 `209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb` has **zero resets and no model**, and specifies a *new from-scratch* attempt on already-consumed TRAIN roads. Do not start it until a new user instruction and preflight after migration. |

There is no committed official submission identifier or competition-confirmed
model. Raw reward/progress/finish counts above are internal research proxies.
The incomplete DrQ and TD-MPC2 attempts must not be relabeled as finished or
as fresh validation.

## Git-External Transfer Inventory

The read-only audit at approximately 05:24 UTC counted **allocated disk bytes**
(`du -B1`), not compressed-archive sizes. Peers committed code after the audit,
so counts of *non-Git* text files are an upper-bound snapshot, not an atomic
post-commit inventory. Active writers must remain stopped while copying.

| Class | Repository-relative paths | Allocated bytes at audit | Why |
|---|---|---:|---|
| **MUST TRANSFER** | `runs/` whole tree | 108,308,742,144 | 104,216,604,672 bytes then outside Git: replay-bearing `.pt` checkpoints, collections, source snapshots, traces, ignored logs, prior data, failed/partial evidence and model actors. Transfer complete directories, not selected `result.json` files. |
| **MUST TRANSFER** | `evaluations/` whole tree | 909,111,296 | 280,645,632 bytes then outside Git: internal screen/confirmation/blind manifests, previous snapshots and ignored candidate actors. Keep this access-controlled; do not publish raw protected results. |
| **MUST TRANSFER until Git push verified** | `experiments/` non-Git protocol/result/stop receipts | 200,704 outside Git (7,921,664 whole tree) | Some TD-MPC2 evidence is already pushed, but peer DrQ/Dreamer JSON and the DrQ migration stop receipt must be verified at the final Git tip or separately copied. |
| **MUST TRANSFER until Git push verified** | Untracked text under `docs/`, `talk/`, `haic/`, `scripts/`, `tests/` | 2,465,792 outside Git at audit | Source and append-only coordination from other research lanes. Preserve/commit reviewed files; do not silently discard any remaining untracked text. |
| **Git-covered** | `submissions/20260919T135436Z_baseline1-final/`, root `model.pt`, root `submission.zip` | 6,254,592 for submission directory | Already version-controlled. Archive is not proof of an official submission. Confirm the pushed commit contains them; no extra archive transfer needed if so. |

The conservative **non-Git transfer-relevant total at the audit snapshot was
104,499,916,800 bytes** (~104.5 GB decimal / 97.3 GiB). Only 270,336 bytes of
nested Python bytecode were categorized as regenerable, leaving
**104,499,646,464 bytes** to preserve if no further commits remove text from
the non-Git set. Instead of selecting individual binary files, copying all
`runs/` plus all `evaluations/` covers the irreproducible bulk in
**109,217,853,440 allocated bytes** (~109.2 GB), including their Git-covered
portions. Transferring all four artifact roots including `experiments/` and
`submissions/` allocated 109,232,029,696 bytes at audit; do not double-count
these roots and their subgroups. Recompute totals at copy time as peer commits
and disk allocation change.

### Partition Of `runs/`

Every row below is **included** in the 108,308,742,144-byte `runs/` root,
not additional storage to sum on top. Preserve the whole tree to cover
unnamed historical partial attempts. Nested frozen `source/` snapshots occupy
24,723,456 bytes and ignored run logs 2,314,240 bytes, already within this
root total; keep both with their receipts.

| Path group | Allocated bytes | Purpose |
|---|---:|---|
| `runs/20260927-drqv2-final-source-replay-v1/` | 6,614,634,496 | Two sealed source pools, five complete learners, stopped sixth partial (820,660,753 **apparent** bytes / approximately 820,678,656 allocated in last separate measurement). |
| `runs/20260926-drqv2-retention-r7/` | 20,609,380,352 | Twelve runs, source/online replay lineage, checkpoints and development traces. |
| `runs/20260925-drqv2-geometry-mix-v1-r6/` | 10,387,148,800 | Six runs, checkpoint/replay traces and development diagnostics. |
| `runs/20260922-drq-steering-l2-v1-fast/` | 9,047,994,368 | Historical controls and treatment checkpoints. |
| `runs/20260922-drq-augmentation-pad-v1-restart/` | 9,075,847,168 | Historical pad controls and treatment checkpoints. |
| `runs/20260924-pixel-rlpd-offpolicy-pilot-v2/` | 3,721,342,976 | Consumed prior collection, four learner runs and source snapshots. |
| `runs/20260924-pixel-rlpd-long-horizon-followup-v1/` | 10,739,867,648 | Prior data, four learners and the seed-11 internal candidate. |
| `runs/20260925-pixel-rlpd-entropy-target-ablation-v4/` | 5,705,342,976 | Consumed collection and four runs stopped before screen. |
| `runs/20260925-pixel-rlpd-entropy-target-ablation-v5/` | 10,729,406,464 | Collection, four learners and seed-50 provenance. |
| `runs/20260924-dreamerv3-b1-*/` | 11,438,149,632 | Nine historical world-model diagnosis branches. |
| `runs/20260926-dreamerv3-reused-train-{diagnostic,diversity,multisource,source1}-v1/` | 3,667,566,592 | Frozen collections and world-model diagnostic checkpoints, not performance actors. |
| `runs/20260924-drqv2-teacher-replay-v1-r{2,3}/` | 73,076,736 | Consumed teacher collections including failed-gate r3 datasets. |
| `runs/tdmpc2-reused-train-20260927-v1/` | 375,242,752 | One partial TRAIN ledger and replay/optimizer-bearing boundary. V2 has no run directory. |
| `runs/20260926-rlpd-g0-{completion,fixed-windows}-v1/` | 36,020,224 | Observational traces, claims and frozen windows. |
| Remaining `runs/` entries | 6,087,720,960 | Earlier PPO/DrQ, r4/r5 partials, catalogs, residual options, G0, smoke and other diagnostics. |

The DrQ owner independently identified a **19,339,067,392-byte minimum** for
its five-complete-plus-partial final-source handoff. That subtotal is already
inside the global run rows, not extra disk. It includes original seed-0/seed-1
step-131,072 source checkpoint **and `config.json`**, six r7b control run
directories, old source/development traces, both 100k-decision collection
pools, five complete and one partial final-source learners, and cross-lane
receipts. Both old source `config.json` files are Git-external and read by
`scripts/audit_drq_final_source_seeds.py`; a checkpoint-only transfer would
break this audit. Their SHA-256 values are
`0aec3e0edef58f4e9130ebd179f03a41fe1dcc85e1089c442386d5040b926cd0`
and `fcefa4b96ff4f74548f43ca0dda6a14390c98896f2ef9cf4ca4d6e73a4dbb625`.
The conservative whole-`runs/` rule also protects historical arms outside
this minimal DrQ-specific subset.

Targeted `sha256sum --check` succeeded for both final-source collection
pools/ledgers, all 30 receipt-bound checkpoint/actor/sample-trace files in the
five completed DrQ learners, the stopped sixth checkpoint/actor/metrics, both
RLPD finalist actors/checkpoints, both latest Dreamer diagnostic checkpoints,
and the TD-MPC2 ledger/boundary pair. This was **not** a hash of every
historical file; verify the complete copy before deleting this instance.

### Rebuildable Or Not Needed

- Rebuild Python environments instead of migrating root `.venv`: it is an
  **absolute symlink** to `/venv/main` and copying/dereferencing it would
  import an instance-specific interpreter. `.pytest_cache/` (151,552 bytes)
  and nested run `__pycache__/` bytecode (270,336 bytes) are regenerable; the
  bytecode is already included in the conservative whole-tree transfer.
- `evaluations/.pending-20260920T081735715893Z-q566zurj/` uses 51,060,736
  bytes but has no scored episode rows/manifest. It is abandoned-preparation
  evidence, **not** a result; keep it in the conservative whole-tree copy
  rather than deleting an ambiguous record during shutdown.
- Root `tmp/` images and root `model.pt`/`submission.zip` are Git tracked.
  There are no root `models/` or `checkpoints/` directories. Do not copy an
  ignored `.venv` or generated bytecode as a required dependency.
- Instance-local `/tmp/kilo/haic-cpu21/` and synthetic
  `/tmp/kilo/tdmpc2-*-synthetic*.pt` are disposable CPU-compatibility tooling
  and seeded **untrained** model exports, not trained results; recreate them
  from the committed checks rather than treating `/tmp/kilo` as a research
  checkpoint archive.

Sparse files matter: `runs/` was 113,632,943,467 apparent bytes but only
108,308,742,144 allocated, so a non-sparse copy can need ~5.3 GB more at
the destination. Do not construct a second full local tarball on the nearly
full instance. Use an authenticated **private direct transfer** with the
same repository-relative paths, preserving sparse files. Exclude `.git`,
the `.venv` symlink, credentials and machine secrets from any bulk transfer.
An owner found `/root/.vast_api_key` (mode 0644) and
`/root/.config/gh/hosts.yml` (mode 0600) outside this repository. The Vast key's
world-readable mode is a local credential exposure risk; neither credential
belongs in Git, the evidence archive, or this handoff. Re-provision credentials
privately on the new host rather than copying their values.

## Clone, Restore And Check (No Experiment)

1. After the final reviewed commits are pushed, clone `main` from
   `https://github.com/dusten45/HAIC-Money.git` to the same absolute workspace
   root `/workspace/HAIC-Money` if possible (some frozen DrQ receipts name that
   root). Check `git rev-parse HEAD '@{upstream}'` and verify the committed
   TD-MPC2 code, protocols, peer source/results and migration handoff exist.
   Do not interpret a clone of the original `fe73dab` as restored: it lacked
   all TD-MPC2 implementation files.
2. Privately restore **complete** `runs/` and `evaluations/` trees and any
   still-uncommitted `experiments/`, `docs/`, `talk/`, `haic/`, `scripts/`, and
   `tests/` paths with their original relative names. The new instance should
   have more than 120 GB available for the artifact trees, plus ample working
   headroom; sparse-preserving transfer is important. Example transfer from
   the old repo root once a new host exists:

   ```bash
   rsync -aS --numeric-ids --info=progress2 runs evaluations experiments NEW_HOST:/workspace/HAIC-Money/
   rsync -aS --numeric-ids --checksum --dry-run --itemize-changes runs evaluations experiments NEW_HOST:/workspace/HAIC-Money/
   rsync -a --numeric-ids talk/messages/ NEW_HOST:/workspace/HAIC-Money/talk/messages/
   ```

   The second command should report no changed files. It is expensive but
   necessary before removing the only local copy. If reviewed source/talk
   remains outside the final pushed Git tree, copy those paths privately too
   and confirm their hashes. Do **not** upload protected evaluation data to a
   public Git repository or ship an archive with unknown credentials.
3. Rebuild Python 3.11 (`.python-version`), install project requirements
   using [README](../../README.md) as a starting point, and separately verify
   the **training** GPU environment against the observed
   [package inventory](../../experiments/instance-migration-training-packages-2026-09-27.txt):
   Python 3.11.14, Torch 2.11.0+cu128 with CUDA 12.8, Gymnasium 0.29.1,
   Box2D-py 2.3.5, NumPy 1.26.0, OpenCV 4.8.1.78, Pygame 2.6.1. The old
   RTX 5070 Ti driver was 580.173.02. **`requirements.txt` and `uv.lock`
   pin Torch 2.1.0, not the recorded training 2.11 CUDA wheel**; blindly
   `uv sync`-ing an established GPU environment would change it. The separate
   CPU-only inference check needs Torch 2.1.0+cpu, not the old nonportable
   `/tmp/kilo/haic-cpu21` interpreter path.
4. Perform **read-only** verification first:

   ```bash
   git status --short --branch
   git rev-parse HEAD '@{upstream}'
   sha256sum experiments/tdmpc2-reused-train-pilot-v2.json scripts/train_tdmpc2.py haic/algorithms/tdmpc2/planner.py
   sha256sum runs/tdmpc2-reused-train-20260927-v1/training.jsonl runs/tdmpc2-reused-train-20260927-v1/boundary.pt
   sha256sum runs/20260927-drqv2-final-source-replay-v1/learner-1-easy_retention-final_source/checkpoints/step-000016384/checkpoint.pt
   df -h .
   nvidia-smi
   ```

   Expected TD-MPC2 protocol/operator/planner SHA-256 values are respectively
   `209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb`,
   `580b867a826c72cebe4e5c116e782d442b96d681549792e0a224e52523a1fc0c`,
   and `e0c114570343fbd74f20611732c85766edbb1b33684527d2c7b37bda528eb009`.
   Other expected hashes are in the lane table above and frozen receipts.
   Recheck cgroup **raw** `memory.max - memory.current`, disk, GPU and OOM
   history for the new host; TD-MPC2 v2 freezes >=16 GiB raw memory and >=5 GiB
   available disk, plus a 7,200-second training wall cap.
5. After establishing the GPU interpreter, a TD-MPC2 **zero-reset,
   read-only** contract check is:

   ```bash
   python -B -m scripts.train_tdmpc2 --protocol experiments/tdmpc2-reused-train-pilot-v2.json --protocol-sha256 209f8a752a6ef61a3dcec6c770fe10fd71158779b83afbf72f9b5d72410c8bfb --run-dir runs/tdmpc2-reused-train-20260927-v2 --preflight
   ```

   Passing preflight is **not** authority to start `--run`. V1's `--resume`
   is forbidden. DrQ sixth-arm `step-000016384` is preserved research state,
   not an exact restart instruction. Dreamer is closed. Any new TRAIN action
   requires a separate user instruction, exposure audit, resource recheck and
   correctly frozen protocol after the migration.

## Deletion Gate

**Do not delete this Vast.ai instance yet.** Source/protocol commits and
pushes alone do not preserve ~104.5 GB of Git-external evidence. The new host
or private storage target has not been provided; no archive has been copied or
verified. Before termination, confirm all peer-owned commits reached upstream,
recheck `git status` including ignored files, prove every live learner/evaluator
is stopped, transfer the mandatory artifacts, verify the whole copy and
critical SHA-256 receipts **on the destination**, and validate that protected
data remained private. Until then this disk is the only known complete copy.
