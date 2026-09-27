"""Read-only, non-promoting open-loop score for the frozen mixed-source learner.

Freeze a separate JSON protocol at PROTOCOL_PATH and pass its SHA on the CLI.
Schema: format, purpose, study_id, training_protocol,
development_collection_protocol, training (seed-ordered result/checkpoint/lineage
path and SHA pairs), development (source0/random receipt, archive, digest and
source identity), source_sha256, scoring, resources. No environment is accessed
and no optimizer step occurs. An untrained reference agent is instantiated only
to audit seeded actor/critic state. Development ZIPs must not be opened before
this code, its synthetic tests and the separate score protocol have been frozen.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import common_adapter
import dreamer_v3
from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from haic.algorithms.drq_v2 import teacher_replay
from scripts.diagnose import dreamerv3_reused_train_open_loop as original


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-multisource-score-v1"
PURPOSE = "reused-TRAIN-training-excluded-open-loop-diagnostic"
STUDY_ID = "dreamerv3-reused-train-multisource-score-v1"
PROTOCOL_PATH = "experiments/dreamerv3-reused-train-multisource-score-v1.json"
TRAIN_PATH = "experiments/dreamerv3-reused-train-multisource-v1.json"
TRAIN_SHA256 = "1a7ddee161bf1958eb5c59152326b6f2fed8899eed41294e55e86edc5ccce42d"
DEV_PATH = "experiments/dreamerv3-reused-train-multisource-development-v1.json"
DEV_SHA256 = "0f2c622e008ab7e4bce98c376c2a6c3e4ef6e78d73d07d82d2716185ff9d2fb0"
R6_SHA256 = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
TRAIN_ROOT = "runs/20260926-dreamerv3-reused-train-multisource-v1"
DEV_ROOT = TRAIN_ROOT + "/development"
OUTPUT = TRAIN_ROOT + "/open-loop-v1"
TRAIN_DECISIONS = 10612
REPLAY_CAPACITY = 16384
RESULT_SHAS = ("ecb070a98f20a679f3743ed81a1423b4b899696871607a2990c7b7c5e19c6896",
               "516487dba3ca2b55c018bd2cbf874a820ebb5646ab58dbb06444d99db0aa7dc0")
CHECKPOINT_SHAS = ("cff37b13a7b38153f44a0035f548eb2925603a3357efe804e95255ef838546e6",
                   "35b61f080df5246adcd46bd5d259031909881575d4ebb44a382932e4ee0b12d0")
LINEAGE_SHA256 = "431990749c7c9069a5fb33269cfd726d38d1ff53a21f50f28f59380a8aade52a"
ROADS = (3910800008, 3910800011, 3910800019, 3910800007, 3910800074,
         3910800069, 3910800010, 3910800016, 3910800041, 3910800030,
         3910800081, 3910800070)
DEV_ROADS = (3910800012, 3910800024, 3910800005, 3910800052)
SOURCE_PATHS = frozenset({
    "scripts/diagnose/dreamerv3_reused_multisource_open_loop.py",
    "scripts/diagnose/dreamerv3_reused_train_open_loop.py",
    "dreamer_v3.py", "common_adapter.py", "haic/algorithms/drq_v2/teacher_replay.py",
})
RUNTIME_SOURCES = {
    "scripts/diagnose/dreamerv3_reused_multisource_open_loop.py": Path(__file__),
    "scripts/diagnose/dreamerv3_reused_train_open_loop.py": Path(original.__file__),
    "dreamer_v3.py": Path(dreamer_v3.__file__),
    "common_adapter.py": Path(common_adapter.__file__),
    "haic/algorithms/drq_v2/teacher_replay.py": Path(teacher_replay.__file__),
}


def _file(root: Path, name: str, *, max_bytes: int | None = None) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or Path(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"unsafe repository-relative input path: {name}")
    if name.split("/")[0] not in ("runs", "experiments", "scripts", "haic") and name not in (
            "dreamer_v3.py", "common_adapter.py"):
        raise ValueError(f"input outside allowed paths: {name}")
    path = root
    for part in name.split("/"):
        path /= part
        if path.is_symlink() or (name.startswith("runs/") and any(
                token in part.lower() for token in original._FORBIDDEN)):
            raise ValueError(f"symlink or protected input path: {name}")
    if not path.is_file() or max_bytes is not None and not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"missing or oversized input: {name}")
    return path


def _pin(root: Path, name: str, digest: str, *, max_bytes: int | None = None) -> Path:
    path = _file(root, name, max_bytes=max_bytes)
    if original._sha256(path) != original._hash(digest):
        raise ValueError(f"SHA-256 mismatch: {name}")
    return path


def _reference(root: Path, value: Any, expected: str, digest: str) -> dict[str, Any]:
    if value != {"path": expected, "sha256": digest}:
        raise ValueError(f"frozen reference mismatch: {expected}")
    return original._json(_pin(root, expected, digest, max_bytes=1024**2))


def _lineage(root: Path, train: dict[str, Any], rows: list[dict[str, Any]],
             receipts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if rows[0]["lineage_sha256"] != rows[1]["lineage_sha256"] or rows[0]["lineage_path"] == rows[1]["lineage_path"]:
        raise ValueError("separate seed lineage sidecars required")
    sequences = []
    for row in rows:
        ledger = original._json(_pin(root, row["lineage_path"], row["lineage_sha256"], max_bytes=1024**2))
        episodes = ledger.get("episodes")
        total = sum(train["datasets"][source]["stored_decisions"] for source in ("source0", "source1"))
        if (ledger.get("format") != "haic-dreamerv3-reused-train-multisource-v1-lineage"
                or ledger.get("range_convention") != "inclusive" or ledger.get("transition_count") != total
                or not isinstance(episodes, list) or len(episodes) != 2 * len(ROADS)):
            raise ValueError("lineage count/format mismatch")
        cursor = 0
        for index, episode in enumerate(episodes):
            source = ("source0", "source1")[index % 2]
            identity = train["datasets"][source]
            receipt_episode = receipts[source]["episode_rows"][index // 2]
            expected = {
                "source_id": identity["source_id"], "actor_sha256": identity["source_actor_sha256"],
                "archive_sha256": identity["archive_sha256"], "dataset_digest": identity["dataset_digest"],
                "road": {"track_id": 1, "geometry_seed": ROADS[index // 2]},
                "original_episode_id": str(receipt_episode["episode_id"]), "replay_episode_id": index,
                "sequence_id_start": cursor, "sequence_id_end": cursor + receipt_episode["decisions"] - 1,
            }
            if episode != expected or not 1 <= receipt_episode["decisions"] <= 2000:
                raise ValueError("lineage source/road/episode or contiguous sequence interval mismatch")
            cursor += receipt_episode["decisions"]
        if cursor != total or cursor != TRAIN_DECISIONS:
            raise ValueError("lineage not contiguous through 0..10611")
        sequences.append(episodes)
    if sequences[0] != sequences[1]:
        raise ValueError("seed lineages disagree")
    return sequences[0]


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _equal(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _checkpoint(path: Path, *, config: dict[str, Any], seed: int,
                metadata: dict[str, Any], lineage: list[dict[str, Any]]) -> tuple[Any, ...]:
    # Check SHA, size, cgroup and source pins before calling this pickle loader.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("format") != "haic-dreamerv3-checkpoint-v3"
            or payload.get("run_metadata") != metadata or payload.get("config") != config
            or type(payload.get("gradient_steps")) is not int or payload["gradient_steps"] != 256
            or type(payload.get("environment_steps")) is not int or payload["environment_steps"] != 0
            or payload.get("trainer_state") is not None):
        raise ValueError(f"learner-{seed}: model-only checkpoint metadata/config/counters mismatch")
    replay = payload.get("replay")
    count = sum(item["sequence_id_end"] - item["sequence_id_start"] + 1 for item in lineage)
    if (not isinstance(replay, dict) or any(replay.get(key) != value for key, value in (
            ("capacity", config["replay_capacity"]), ("size", count), ("cursor", count), ("total_steps", count)))
            or not np.array_equal(replay.get("sequence_ids"), np.arange(count))
            or not np.array_equal(np.flatnonzero(replay.get("is_first")),
                                  [item["sequence_id_start"] for item in lineage])
            or not np.array_equal(np.flatnonzero(replay.get("is_last")),
                                  [item["sequence_id_end"] for item in lineage])
            or set(replay.get("boundary_observations", {})) != {
                item["sequence_id_end"] for item in lineage}):
        raise ValueError(f"learner-{seed}: checkpoint replay lost or overwrote lineage")
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            reference = DreamerV3Agent(DreamerV3Config(**config), seed=seed)
            for name in ("actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer"):
                if not _equal(payload.get(name), getattr(reference, name).state_dict()):
                    raise ValueError(f"learner-{seed}: actor/critic seeded initialization or optimizer changed: {name}")
    finally:
        np.random.set_state(numpy_state)
    del reference
    modules = original.ConvEncoder(4, config["embed_dim"]), original.CategoricalRSSM(
        action_dim=3, embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"],
        num_categoricals=config["num_categoricals"], num_classes=config["num_classes"],
        unimix=config["unimix"])
    encoder, rssm = modules
    modules += (original.ConvDecoder(rssm.state_dim, 1),
                original.RewardHead(rssm.state_dim, config["twohot_bins"]),
                original.ContinueHead(rssm.state_dim))
    for name, module in zip(("encoder", "rssm", "decoder", "reward_head", "continue_head"), modules, strict=True):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
    return modules


def preflight(protocol_path: Path, protocol_sha256: str, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate checkpoint metadata/lineage before any ZIP read or output creation."""
    root = Path(repo_root).resolve()
    if original._relative(root, protocol_path) != PROTOCOL_PATH:
        raise ValueError("score protocol path is fixed")
    protocol = original._json(_pin(root, PROTOCOL_PATH, protocol_sha256, max_bytes=1024**2))
    if (set(protocol) != {"format", "purpose", "study_id", "training_protocol",
                          "development_collection_protocol", "training", "development",
                          "source_sha256", "scoring", "resources"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != STUDY_ID):
        raise ValueError("not the separate frozen non-promoting mixed-source score protocol")
    if original._relative(root, output_dir) != OUTPUT:
        raise ValueError("score output directory is fixed and separate")
    output = root / OUTPUT
    if (not output.parent.is_dir() or any(path.is_symlink() for path in
            (root / "runs", root / TRAIN_ROOT, output.parent))):
        raise ValueError("score output parent must exist and not be a symlink")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"score output collision: {output}")

    train = _reference(root, protocol["training_protocol"], TRAIN_PATH, TRAIN_SHA256)
    dev = _reference(root, protocol["development_collection_protocol"], DEV_PATH, DEV_SHA256)
    train_cells = [{"track_id": 1, "geometry_seed": road} for road in ROADS]
    dev_cells = [{"track_id": 1, "geometry_seed": road} for road in DEV_ROADS]
    if (train.get("format") != "haic-dreamerv3-reused-train-multisource-v1"
            or train.get("purpose") != "reused-TRAIN-engineering-diagnostic"
            or train.get("cells") != train_cells or len(set(ROADS)) != len(ROADS)
            or train.get("learner", {}).get("seeds") != [0, 1]
            or train["learner"].get("updates") != 256
            or set(train.get("datasets", {})) != {"source0", "source1"}
            or train["datasets"]["source0"]["source_id"] != "drq-source-0"
            or train["datasets"]["source1"]["source_id"] != "drq-source-1"
            or train["datasets"]["source0"]["source_actor_sha256"] == train["datasets"]["source1"]["source_actor_sha256"]):
        raise ValueError("frozen mixed-source training identity or 256-update budget mismatch")
    config = train["learner"]["config"]
    if (not isinstance(config, dict) or set(config) != {field.name for field in fields(DreamerV3Config)}
            or config.get("device") != "cpu" or config.get("replay_capacity") != REPLAY_CAPACITY
            or sum(item["stored_decisions"] for item in train["datasets"].values()) != TRAIN_DECISIONS):
        raise ValueError("complete frozen CPU model configuration/replay capacity required")
    r6_ref = train["r6_protocol"]
    if dev.get("r6_protocol") != r6_ref:
        raise ValueError("development and training must bind the same r6 TRAIN allocation")
    r6 = _reference(root, r6_ref, "experiments/drqv2-geometry-mix-v1-r6.json", R6_SHA256)
    pool, diagnostic = r6.get("training_pool", {}), r6.get("diagnostic_pool", {})
    if (dev.get("format") != "haic-dreamerv3-reused-train-diagnostic-v1"
            or dev.get("study_id") != "dreamerv3-reused-train-multisource-development-v1"
            or dev.get("purpose") != train["purpose"]
            or dev.get("freshness_claim") != "reused r6 TRAIN cells; not fresh P1/P1b, evaluation, or promotion"
            or dev.get("cells") != dev_cells or dev.get("episode_schedule") != [0, 1, 2, 3]
            or dev.get("frame_skip") != 4 or dev.get("max_steps") != 2000
            or dev.get("budgets") != {"random_decision_cap": 8000, "teacher_decision_cap": 8000}
            or pool.get("partition") != "TRAIN" or diagnostic.get("partition") != "TRAIN-DIAGNOSTIC"
            or 1 not in pool.get("track_ids", [])
            or not set(ROADS + DEV_ROADS) <= set(pool.get("geometry_seeds", []))
            or set(ROADS) & set(DEV_ROADS)
            or set(ROADS + DEV_ROADS) & set(diagnostic.get("geometry_seeds", []))
            or dev.get("source_actor", {}).get("source_id") != train["datasets"]["source0"]["source_id"]
            or dev["source_actor"].get("actor_sha256") != train["datasets"]["source0"]["source_actor_sha256"]
            or dev["source_actor"].get("checkpoint_sha256") != train["datasets"]["source0"]["source_checkpoint_sha256"]):
        raise ValueError("development must use exact training-excluded four reused r6 TRAIN roads/source0")
    random = dev.get("random", {})
    collector_sha = dev.get("source_sha256", {}).get("scripts/diagnose/collect_dreamerv3_reused_train.py")
    policy = {"policy": "uniform-native-random-v1", "collector_sha256": collector_sha,
              "rng_seed": 7395, "native_bounds": [-1.0, 1.0], "action_dim": 3}
    random_sha = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    if (random != {"rng_seed": 7395, "source_id": "uniform-native-random-seed-7395",
                   "source_actor_sha256": random_sha}
            or not isinstance(dev.get("source_sha256"), dict)):
        raise ValueError("independently frozen development random source identity required")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("exact imported scorer/model/archive executable SHA map required")
    for name, digest in sources.items():
        path = _pin(root, name, digest, max_bytes=4 * 1024**2)
        if original._sha256(RUNTIME_SOURCES[name]) != digest or name in train["source_sha256"] and (
                train["source_sha256"][name] != digest):
            raise ValueError(f"executing or trained source SHA-256 mismatch: {name}")
    for name, digest in train["source_sha256"].items():
        _pin(root, name, digest, max_bytes=4 * 1024**2)
    for name in ("common_adapter.py", "haic/algorithms/drq_v2/teacher_replay.py"):
        if dev["source_sha256"].get(name) != sources[name]:
            raise ValueError("collection decoder differs from scored training source")
    _pin(root, "scripts/diagnose/collect_dreamerv3_reused_train.py", collector_sha, max_bytes=4 * 1024**2)

    scoring = protocol["scoring"]
    if (scoring != {"context_decisions": 8, "window_decisions": 32, "latent_seed": 120927,
                    "context_mode": "reset-origin-full-prefix-terminal", "baselines": original.BASELINES,
                    "min_complete_roads_per_stratum": 4}):
        raise ValueError("fixed 8-context/32-target/reset-terminal windows and latent seed required")
    resources = protocol["resources"]
    limits = {"max_archive_bytes": 64 * 1024**2, "max_uncompressed_bytes": 512 * 1024**2,
              "max_decisions_per_archive": 32768, "max_checkpoint_bytes": 1024**3,
              "max_cgroup_memory_bytes": 128 * 1024**3, "min_cgroup_available_bytes": 128 * 1024**3,
              "min_disk_available_bytes": 1024**4}
    if not isinstance(resources, dict) or set(resources) != set(limits):
        raise ValueError("exact ZIP/checkpoint/cgroup/disk resource caps required")
    for key, cap in limits.items():
        original._positive(resources[key], key, cap)
    if (resources["min_cgroup_available_bytes"] < 8 * 1024**3
            or resources["min_disk_available_bytes"] < 2 * 1024**3):
        raise ValueError("score cgroup/disk floor below safe minimum")
    needed = max(resources["min_cgroup_available_bytes"], 512 * 1024**2
                 + 2 * resources["max_checkpoint_bytes"] + 2 * resources["max_archive_bytes"]
                 + 2 * resources["max_uncompressed_bytes"])
    cgroup = original._cgroup()
    if cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"] or cgroup["available_bytes"] < needed:
        raise ValueError("cgroup lacks frozen score-time headroom")
    if os.statvfs(output.parent).f_bavail * os.statvfs(output.parent).f_frsize < resources["min_disk_available_bytes"]:
        raise ValueError("disk lacks frozen score-time headroom")

    training = protocol["training"]
    if not isinstance(training, list) or len(training) != 2:
        raise ValueError("exactly two frozen training learner seeds required")
    train_receipts = {}
    train_archives = {}
    for label in ("source0", "source1"):
        row = train["datasets"][label]
        receipt = original._json(_pin(root, row["receipt_path"], row["receipt_sha256"], max_bytes=1024**2))
        if (receipt.get("status") != "completed" or receipt.get("arm") != "teacher"
                or receipt.get("source_id") != row["source_id"]
                or receipt.get("source_actor_sha256") != row["source_actor_sha256"]
                or receipt.get("archive_sha256") != row["archive_sha256"]
                or receipt.get("dataset_digest") != row["dataset_digest"]
                or receipt.get("stored_decisions") != row["stored_decisions"]
                or receipt.get("allowed_cells") != train_cells
                or receipt.get("complete_episode_count") != 12 or receipt.get("schedule_exhausted") is not True
                or receipt.get("partial_decisions") != 0 or receipt.get("unresolved_decision_calls") != 0
                or not isinstance(receipt.get("episode_rows"), list) or len(receipt["episode_rows"]) != 12):
            raise ValueError(f"{label}: training receipt/source/coverage mismatch")
        for i, episode in enumerate(receipt["episode_rows"]):
            if (episode.get("attempt") != i or episode.get("episode_id") != i
                    or episode.get("status") != "complete" or episode.get("complete") is not True
                    or type(episode.get("terminal")) is not bool or type(episode.get("finished")) is not bool
                    or any(episode.get(key) != value for key, value in train_cells[i].items())):
                raise ValueError(f"{label}: training receipt episode mismatch")
        archive = _file(root, row["archive_path"], max_bytes=resources["max_archive_bytes"])
        if archive.parent != (root / row["receipt_path"]).parent:
            raise ValueError("training archive/receipt directory mismatch")
        train_archives[label] = archive
        train_receipts[label] = receipt
    checked_training = []
    for seed, row in enumerate(training):
        prefix = f"{TRAIN_ROOT}/learner-{seed}"
        expected = {"seed": seed, "result_path": prefix + "/training-result.json",
                    "result_sha256": RESULT_SHAS[seed], "checkpoint_path": prefix + "/world-model-checkpoint.pt",
                    "checkpoint_sha256": CHECKPOINT_SHAS[seed], "lineage_path": prefix + "/lineage.json",
                    "lineage_sha256": LINEAGE_SHA256}
        if row != expected:
            raise ValueError("both fixed training result/checkpoint/lineage SHA pairs required")
        result = original._json(_pin(root, row["result_path"], row["result_sha256"], max_bytes=1024**2))
        checkpoint = _file(root, row["checkpoint_path"], max_bytes=resources["max_checkpoint_bytes"])
        _pin(root, row["checkpoint_path"], row["checkpoint_sha256"], max_bytes=resources["max_checkpoint_bytes"])
        for key, value in (("format", "haic-dreamerv3-reused-train-multisource-v1-result"),
                           ("status", "complete"), ("seed", seed), ("purpose", train["purpose"]),
                           ("study_id", train["study_id"]), ("protocol_sha256", TRAIN_SHA256),
                           ("sources", train["datasets"]), ("checkpoint_path", "world-model-checkpoint.pt"),
                           ("checkpoint_sha256", row["checkpoint_sha256"]),
                           ("lineage_path", row["lineage_path"]), ("lineage_sha256", LINEAGE_SHA256),
                           ("model_only_updates", 256), ("environment_steps", 0),
                           ("distinct_training_roads", 12), ("episodes", 24), ("decisions", TRAIN_DECISIONS),
                           ("actor_critic_target_optimizers_unchanged", True), ("actor_trained", False),
                           ("fresh_claim", False), ("p1b_claim", False), ("promotion_eligible", False)):
            if type(result.get(key)) is not type(value) or result[key] != value:
                raise ValueError(f"learner-{seed}: training result {key} mismatch")
        checked_training.append({"row": row, "result": result, "checkpoint": checkpoint,
                                 "metadata": {"purpose": train["purpose"], "study_id": train["study_id"],
                                              "seed": seed, "protocol_sha256": TRAIN_SHA256,
                                              "lineage_path": row["lineage_path"], "lineage_sha256": LINEAGE_SHA256,
                                              "sources": train["datasets"], "actor_trained": False,
                                              "fresh_claim": False, "p1b_claim": False,
                                              "promotion_eligible": False}})
    lineage = _lineage(root, train, training, train_receipts)

    development = protocol["development"]
    if not isinstance(development, dict) or set(development) != {"source0", "random"}:
        raise ValueError("both distinct development action strata required")
    checked_dev = {}
    for label, arm in (("source0", "teacher"), ("random", "random")):
        row = development[label]
        prefix = f"{DEV_ROOT}/{arm}"
        if (not isinstance(row, dict) or set(row) != {"receipt_path", "receipt_sha256", "archive_path",
                                                      "archive_sha256", "dataset_digest", "source_id",
                                                      "source_actor_sha256"}
                or row["receipt_path"] != prefix + "/collection-result.json"
                or row["archive_path"] != prefix + "/support-dataset.npz"):
            raise ValueError(f"{label}: development must use the separate fixed arm archive/receipt")
        source = dev["source_actor"] if label == "source0" else random
        actor_sha = source["actor_sha256"] if label == "source0" else source["source_actor_sha256"]
        if row["source_id"] != source["source_id"] or row["source_actor_sha256"] != actor_sha:
            raise ValueError(f"{label}: forged development source identity")
        receipt = original._json(_pin(root, row["receipt_path"], row["receipt_sha256"], max_bytes=1024**2))
        archive = _file(root, row["archive_path"], max_bytes=resources["max_archive_bytes"])
        for key, value in (("format", "haic-dreamerv3-reused-train-collection-result-v1"),
                           ("status", "completed"), ("study_id", dev["study_id"]),
                           ("purpose", dev["purpose"]), ("protocol_sha256", DEV_SHA256),
                           ("r6_protocol_sha256", r6_ref["sha256"]), ("arm", arm),
                           ("source_id", row["source_id"]), ("source_actor_sha256", actor_sha),
                           ("source_checkpoint_sha256", source["checkpoint_sha256"] if label == "source0" else None),
                           ("random_rng_seed", 7395 if label == "random" else None),
                           ("archive_sha256", row["archive_sha256"]), ("dataset_digest", row["dataset_digest"]),
                           ("dataset_path", "support-dataset.npz"), ("allowed_cells", dev_cells),
                           ("excluded_seeds", sorted(diagnostic["geometry_seeds"])),
                           ("decision_cap", 8000), ("complete_episode_count", 4),
                           ("schedule_attempts", 4), ("schedule_exhausted", True),
                           ("partial_decisions", 0), ("unresolved_decision_calls", 0)):
            if type(receipt.get(key)) is not type(value) or receipt[key] != value:
                raise ValueError(f"{label}: development receipt {key} mismatch")
        episodes = receipt.get("episode_rows")
        if not isinstance(episodes, list) or len(episodes) != 4:
            raise ValueError(f"{label}: four independent complete road episodes required")
        for index, episode in enumerate(episodes):
            if (episode.get("attempt") != index or episode.get("episode_id") != index
                    or episode.get("complete") is not True or episode.get("status") != "complete"
                    or any(episode.get(key) != value for key, value in dev_cells[index].items())
                    or type(episode.get("decisions")) is not int or not 1 <= episode["decisions"] <= 2000
                    or type(episode.get("terminal")) is not bool or type(episode.get("finished")) is not bool
                    or type(episode.get("terminated")) is not bool or type(episode.get("truncated")) is not bool
                    or not (episode["terminated"] or episode["truncated"])):
                raise ValueError(f"{label}: incomplete/wrong development episode")
        if (receipt.get("stored_decisions") != sum(ep["decisions"] for ep in episodes)
                or receipt.get("decisions_spent") != receipt["stored_decisions"]
                or receipt.get("distinct_finished_geometries") != sorted(
                    episode["geometry_seed"] for episode in episodes if episode["finished"])):
            raise ValueError(f"{label}: development decision/finish denominators mismatch")
        checked_dev[label] = {"row": row, "receipt": receipt, "archive": archive}
    if development["source0"]["source_actor_sha256"] == development["random"]["source_actor_sha256"]:
        raise ValueError("development action strata must have distinct source hashes")
    for seed, model in enumerate(checked_training):
        model["modules"] = _checkpoint(model["checkpoint"], config=config, seed=seed,
                                       metadata=model["metadata"], lineage=lineage)
    return {"root": root, "protocol": protocol, "train": train, "dev": dev, "output": output,
            "training": checked_training, "training_receipts": train_receipts,
            "training_archives": train_archives, "development": checked_dev, "lineage": lineage,
            "cgroup": cgroup, "required_available_bytes": needed}


def _dataset(path: Path, row: dict[str, Any], resources: dict[str, int],
             roads: tuple[int, ...], study_id: str, arm: str,
             receipt: dict[str, Any]) -> teacher_replay.TeacherDataset:
    dataset = original._load_dataset(path, row, resources, tuple((1, road) for road in roads), study_id, arm)
    episodes = receipt["episode_rows"]
    if dataset.transition_count != receipt["stored_decisions"] or len(dataset.episodes) != len(episodes):
        raise ValueError("sealed archive count differs from receipt")
    for index, (episode, expected) in enumerate(zip(dataset.episodes, episodes, strict=True)):
        if (not episode.complete or episode.episode_id != str(expected["episode_id"])
                or episode.metadata.get("attempt") != index or episode.track_id != 1
                or episode.geometry_id != str(roads[index]) or episode.steps != expected["decisions"]
                or bool(episode.terminal[-1]) is not expected["terminal"]
                or bool(episode.finished[-1]) is not expected["finished"]
                or bool(episode.terminated[-1]) is not expected["terminated"]
                or bool(episode.truncated[-1]) is not expected["truncated"]):
            raise ValueError("sealed episode source/road/terminal differs from receipt")
    return dataset


def score(protocol_path: Path, protocol_sha256: str, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, output_dir, repo_root=repo_root)
    resources = checked["protocol"]["resources"]
    if original._cgroup()["available_bytes"] < checked["required_available_bytes"]:
        raise ValueError("cgroup lacks archive-load headroom")
    torch.set_num_threads(1)
    baselines = {}
    for source in ("source0", "source1"):
        row = checked["train"]["datasets"][source]
        receipt = checked["training_receipts"][source]
        dataset = _dataset(checked["training_archives"][source], row, resources, ROADS,
                           receipt["study_id"], "teacher", receipt)
        n = dataset.transition_count
        baselines[source] = {"training_decisions": n,
                             "training_terminal_events": sum(int(ep.terminal.sum()) for ep in dataset.episodes),
                             "training_reward_sum": sum(float(ep.rewards.astype(np.float64).sum()) for ep in dataset.episodes)}
        del dataset
    total = sum(part["training_decisions"] for part in baselines.values())
    baseline = {"training_decisions": total,
                "training_terminal_events": sum(part["training_terminal_events"] for part in baselines.values()),
                "constant_reward": original._finite(sum(part["training_reward_sum"] for part in baselines.values()) / total)}
    baseline["terminal_prevalence"] = original._finite(baseline["training_terminal_events"] / total)
    datasets = {}
    for label, arm in (("source0", "teacher"), ("random", "random")):
        entry = checked["development"][label]
        datasets[label] = _dataset(entry["archive"], entry["row"], resources, DEV_ROADS,
                                   checked["dev"]["study_id"], arm, entry["receipt"])
    report = {"format": FORMAT, "purpose": PURPOSE, "status": "engineering_proxy_only",
              "study_id": STUDY_ID, "score_protocol_sha256": protocol_sha256,
              "training_protocol_sha256": TRAIN_SHA256, "development_collection_protocol_sha256": DEV_SHA256,
              "source_sha256": checked["protocol"]["source_sha256"], "sampling": checked["protocol"]["scoring"],
              "training_baselines": baseline, "training_source_denominators": baselines,
              "training_episode_attempts": 24, "distinct_training_roads": 12,
              "training": [{"learner_seed": seed, "result_sha256": model["row"]["result_sha256"],
                            "checkpoint_sha256": model["row"]["checkpoint_sha256"],
                            "lineage_sha256": model["row"]["lineage_sha256"]}
                           for seed, model in enumerate(checked["training"])],
              "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
              "student_actor_trained": False,
              "interpretation": "Internal reused-r6-TRAIN prediction proxy only; development was excluded from this "
                                "learner but previously consumed by r6. Two training sources share twelve road IDs. "
                                "Mixed teacher 10612 versus old random 4137 decisions is not matched. "
                                "Last-logged reward uses the preceding recorded reward, even within a window. "
                                "No actor, ranking, fresh P1/P1b, confirmation, blind or official inference.",
              "strata": {}, "gate": {"name": "both-seeds-both-action-strata-strict-image-mse-below-shifted-repeat",
                                    "passed": False, "meaning": "non-promoting internal engineering proxy only"}}
    for stratum_index, label in enumerate(("source0", "random")):
        dataset = datasets[label]
        stratum = {"source_id": checked["development"][label]["row"]["source_id"],
                   "development_archive_sha256": checked["development"][label]["row"]["archive_sha256"],
                   "collected_complete_episodes": len(dataset.episodes),
                   "collected_independent_roads": len({ep.geometry_id for ep in dataset.episodes}),
                   "collected_finished_episodes": sum(bool(ep.finished[-1]) for ep in dataset.episodes),
                   "collected_terminal_episodes": sum(bool(ep.terminal[-1]) for ep in dataset.episodes),
                   "models": []}
        for seed, model in enumerate(checked["training"]):
            if original._sha256(model["checkpoint"]) != model["row"]["checkpoint_sha256"]:
                raise ValueError("checkpoint changed since preflight")
            modules = model["modules"]
            episodes = [original._score_episode(ep, modules, baseline["constant_reward"],
                                                baseline["terminal_prevalence"],
                                                120927 + stratum_index * 1_000_003 + index * 1009 + seed)
                        for index, ep in enumerate(dataset.episodes)]
            roads = defaultdict(list)
            for episode in episodes:
                roads[(episode["track_id"], episode["geometry_seed"])].append(episode)
            aggregate = original._aggregate(episodes)
            complete = (len(episodes) == 4 and len(roads) == 4 and aggregate["episode_count"] == 4
                        and aggregate["window_count"] == 8 and aggregate["window_label_uses"] == 256)
            metrics = aggregate["metrics_episode_mean"]
            passed = bool(complete and metrics is not None and
                          metrics["image_mse"] < metrics["shifted_repeat_mse"])
            stratum["models"].append({"learner_seed": seed, "checkpoint_sha256": model["row"]["checkpoint_sha256"],
                                      "aggregate": aggregate,
                                      "roads": [{"track_id": track, "geometry_seed": road, **original._aggregate(items)}
                                                for (track, road), items in sorted(roads.items())],
                                      "episodes": episodes, "image_gate_passed": passed})
        report["strata"][label] = stratum
    report["gate"]["passed"] = all(model["image_gate_passed"] for stratum in report["strata"].values()
                                    for model in stratum["models"])
    if any(model["aggregate"]["episode_count"] != 4 for stratum in report["strata"].values()
           for model in stratum["models"]):
        report["status"] = "coverage_inconclusive"
    # Check mutable inputs again before making the only output directory.
    preflight(protocol_path, protocol_sha256, output_dir, repo_root=checked["root"])
    for entry in checked["development"].values():
        if original._sha256(entry["archive"]) != entry["row"]["archive_sha256"]:
            raise ValueError("development archive changed during scoring")
    for source, archive in checked["training_archives"].items():
        if original._sha256(archive) != checked["train"]["datasets"][source]["archive_sha256"]:
            raise ValueError("training archive changed during scoring")
    output = checked["output"]
    output.mkdir(exist_ok=False)
    with (output / "score-result.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = score(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root)
    print(json.dumps({"status": result["status"], "gate_passed": result["gate"]["passed"],
                      "output": str(args.output / "score-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
