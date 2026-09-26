"""Read-only open-loop diagnosis on sealed, training-excluded, REUSED r6 TRAIN roads.

Protocol format: haic-dreamerv3-reused-train-score-v1. This is neither the
online B1 gate nor a fresh P1/P1b, policy evaluation, or promotion procedure.
Run from the repository root with ``python -m scripts.diagnose.dreamerv3_reused_train_open_loop``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch

from dreamer_v3 import (CategoricalRSSM, ContinueHead, ConvDecoder, ConvEncoder,
                        DreamerV3Config, RewardHead)
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-score-v1"
PURPOSE = "reused-TRAIN-training-excluded-open-loop-diagnostic"
OFFLINE_PATH = "experiments/dreamerv3-reused-train-offline-v1.json"
OFFLINE_SHA256 = "2cb3883109ef43a001d741cdc5055c6448a8a312094f687d907454d7bf6296ba"
R6_SHA256 = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
TRAIN_ROOT = "runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline"
DEVELOPMENT_ID = "dreamerv3-reused-train-development-v1"
DEVELOPMENT_ROADS = (3910800002, 3910800009, 3910800057, 3910800044)
COLLECTOR = "scripts/diagnose/collect_dreamerv3_reused_train.py"
SOURCE_PATHS = frozenset({
    "scripts/diagnose/dreamerv3_reused_train_open_loop.py", "dreamer_v3.py",
    "common_adapter.py", "haic/algorithms/drq_v2/teacher_replay.py",
})
BASELINES = ["shifted-repeat", "training-mean-reward", "last-logged-reward",
             "training-terminal-prevalence"]
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN = ("blind", "confirm", "screen", "eval", "held-out", "holdout", "submission")
_NPZ_KEYS = {"manifest", "digest", "transition_offsets", "frame_offsets", "frames", "actions",
             "applied_actions", "rewards", "progress", "damage", "terminated", "truncated",
             "finished", "terminal"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("expected lowercase SHA-256")
    return value


def _relative(root: Path, supplied: Path | str) -> str:
    path = Path(supplied)
    try:
        return (path if path.is_absolute() else root / path).relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path must be repository-relative") from exc


def _path(root: Path, name: str, kind: str, *, exists: bool = True) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"{kind}: unsafe path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{kind}: unnormalized path")
    if kind == "protocol":
        allowed = len(parts) == 2 and parts[0] == "experiments" and parts[-1].startswith(
            "dreamerv3-reused-train-") and parts[-1].endswith(".json")
    elif kind == "r6":
        allowed = name == "experiments/drqv2-geometry-mix-v1-r6.json"
    elif kind == "source":
        allowed = name in SOURCE_PATHS
    elif kind == "training_result":
        allowed = (len(parts) == 5 and "/".join(parts[:3]) == TRAIN_ROOT
                   and parts[3] in ("random-seed-0", "random-seed-1", "teacher-seed-0", "teacher-seed-1")
                   and parts[-1] == "training-result.json")
    elif kind == "checkpoint":
        allowed = (len(parts) == 5 and "/".join(parts[:3]) == TRAIN_ROOT
                   and parts[3] in ("random-seed-0", "random-seed-1", "teacher-seed-0", "teacher-seed-1")
                   and parts[-1] == "world-model-checkpoint.pt")
    elif kind in ("receipt", "archive"):
        allowed = (len(parts) >= 5 and parts[0] == "runs" and parts[-2] in ("random", "teacher")
                   and parts[-1] == ("collection-result.json" if kind == "receipt" else "support-dataset.npz"))
    elif kind == "output":
        allowed = len(parts) >= 3 and parts[0] == "runs" and "diagnos" in name.lower()
    else:
        raise ValueError(f"unknown path kind: {kind}")
    if not allowed or (parts[0] == "runs" and any(
        token in part.lower() for part in parts[1:] for token in _FORBIDDEN
    )):
        raise ValueError(f"{kind}: not an allowed reused-TRAIN path: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{kind}: symlinks forbidden: {name}")
    if exists and not path.is_file():
        raise ValueError(f"{kind}: missing regular file: {name}")
    if not exists and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"output already exists: {name}")
    return path


def _pinned(root: Path, name: str, digest: str, kind: str) -> Path:
    path = _path(root, name, kind)
    if _sha256(path) != _hash(digest):
        raise ValueError(f"{kind}: SHA-256 mismatch: {name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("JSON exceeds 1 MiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                        parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    if not isinstance(result, dict):
        raise ValueError("JSON must be an object")
    return result


def _ref(root: Path, value: Any, kind: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{kind} requires path and SHA-256")
    path = _pinned(root, value["path"], value["sha256"], kind)
    return path, _json(path)


def _cells(value: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("explicit reused TRAIN cells required")
    result = []
    for cell in value:
        if (not isinstance(cell, dict) or set(cell) != {"track_id", "geometry_seed"}
                or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)
                or type(cell["geometry_seed"]) is not int or not 0 <= cell["geometry_seed"] < 2**32):
            raise ValueError("invalid reused TRAIN cell")
        result.append((cell["track_id"], cell["geometry_seed"]))
    if len(result) != len({road for _, road in result}):
        raise ValueError("duplicate geometry seed")
    return tuple(result)


def _positive(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _cgroup() -> dict[str, int]:
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError("finite cgroup-v2 memory limit/current required") from exc
    if limit <= 0 or used < 0 or used > limit:
        raise ValueError("invalid cgroup memory counters")
    return {"limit_bytes": limit, "used_bytes": used, "available_bytes": limit - used}


def _dataset_bytes(path: Path, digest: str, cap: int, uncompressed_cap: int,
                   max_decisions: int) -> bytes:
    size = path.stat().st_size
    if not 0 < size <= cap:
        raise ValueError("archive compressed size exceeds frozen cap")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode) or os.fstat(stream.fileno()).st_size != size:
            raise ValueError("archive size/type changed before read")
        data = stream.read(cap + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != _hash(digest):
        raise ValueError("archive size or SHA-256 mismatch")
    try:
        with ZipFile(BytesIO(data)) as archive:
            members = archive.infolist()
            if (len(members) != len(_NPZ_KEYS) or {member.filename for member in members}
                    != {f"{key}.npy" for key in _NPZ_KEYS}
                    or any(member.file_size > uncompressed_cap for member in members)
                    or sum(member.file_size for member in members) > uncompressed_cap):
                raise ValueError("dataset ZIP entries exceed frozen names/size bound")
            with archive.open("actions.npy") as stream:
                if np.lib.format.read_magic(stream) != (1, 0):
                    raise ValueError("unsupported archive action header")
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                if len(shape) != 2 or shape[1] != 3 or dtype != np.dtype("float32") or not 1 <= shape[0] <= max_decisions:
                    raise ValueError("archive action rows exceed frozen decision cap")
    except (BadZipFile, KeyError, EOFError, OSError) as exc:
        raise ValueError("invalid bounded teacher dataset ZIP") from exc
    return data


def _load_dataset(path: Path, row: dict[str, Any], resources: dict[str, int],
                  expected_cells: tuple[tuple[int, int], ...], study_id: str, arm: str) -> TeacherDataset:
    data = _dataset_bytes(path, row["archive_sha256"], resources["max_archive_bytes"],
                          resources["max_uncompressed_bytes"], resources["max_decisions_per_archive"])
    dataset = TeacherDataset.from_bytes(data, expected_digest=row["dataset_digest"])
    cells = []
    for episode in dataset.episodes:
        if (episode.source_id != row["source_id"] or episode.source_actor_sha256 != row["source_actor_sha256"]
                or not episode.geometry_id.isdecimal() or str(int(episode.geometry_id)) != episode.geometry_id
                or not (episode.terminated[-1] or episode.truncated[-1])
                or episode.metadata.get("study_id") != study_id or episode.metadata.get("arm") != arm):
            raise ValueError("sealed episode source, boundary or study identity mismatch")
        cells.append((episode.track_id, int(episode.geometry_id)))
        for step in range(episode.steps):
            terminal = (bool(episode.terminated[step]) or bool(episode.finished[step])
                        or episode.retire_reasons[step] in ("crash", "off_track", "out_of_bounds"))
            if bool(episode.terminal[step]) != terminal:
                raise ValueError("sealed terminal label contradicts episode outcome")
    if not set(cells) <= set(expected_cells) or len(cells) != len(set(cells)):
        raise ValueError("sealed episodes contain undeclared or duplicate road cells")
    return dataset


def preflight(protocol_path: Path, protocol_sha256: str, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Reject identity, resource and partition mismatches before loading archives/checkpoints."""
    root = Path(repo_root).resolve()
    protocol_name = _relative(root, protocol_path)
    if not protocol_name.startswith("experiments/dreamerv3-reused-train-score-"):
        raise ValueError("a separate frozen reused-TRAIN scoring protocol is required")
    protocol = _json(_pinned(root, protocol_name, protocol_sha256, "protocol"))
    if (set(protocol) != {"format", "purpose", "study_id", "offline_protocol", "training",
                          "development_collection_protocol", "development", "source_sha256", "scoring", "resources"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != DEVELOPMENT_ID):
        raise ValueError("invalid separate reused-TRAIN scoring contract")
    output_name = _relative(root, output_dir)
    output = _path(root, output_name, "output", exists=False)
    if not output.parent.is_dir():
        raise ValueError("output must be a NEW separate runs directory with an existing parent")

    offline_ref = protocol["offline_protocol"]
    if (not isinstance(offline_ref, dict) or offline_ref != {"path": OFFLINE_PATH, "sha256": OFFLINE_SHA256}):
        raise ValueError("exact first-loop offline protocol SHA-256 required")
    _, offline = _ref(root, offline_ref, "protocol")
    if (offline.get("format") != "haic-dreamerv3-reused-train-offline-v1"
            or offline.get("purpose") != "reused-TRAIN-engineering-diagnostic"
            or offline.get("study_id") != "dreamerv3-reused-train-diagnostic-v1"
            or offline.get("learner", {}).get("seeds") != [0, 1]
            or offline["learner"].get("updates") != 64):
        raise ValueError("first-loop offline protocol identity/budget mismatch")
    _, training_collection = _ref(root, offline["collection_protocol"], "protocol")
    r6_ref = training_collection.get("r6_protocol")
    if (not isinstance(r6_ref, dict) or r6_ref != {
        "path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": R6_SHA256
    }):
        raise ValueError("exact r6 TRAIN protocol required")
    _, r6 = _ref(root, r6_ref, "r6")
    pool = r6.get("training_pool", {})
    diagnostic = r6.get("diagnostic_pool", {})
    if (pool.get("partition") != "TRAIN" or diagnostic.get("partition") != "TRAIN-DIAGNOSTIC"
            or pool.get("track_ids") != [1, 2, 3, 4]):
        raise ValueError("r6 TRAIN-only partition contract mismatch")
    train_cells = _cells(training_collection.get("cells"))
    if (set(train_cells) != set(_cells(offline["datasets"]["random"]["allowed_cells"]))
            or set(train_cells) != set(_cells(offline["datasets"]["teacher"]["allowed_cells"]))):
        raise ValueError("first-loop training cell lists disagree")

    dev_ref = protocol["development_collection_protocol"]
    _, development = _ref(root, dev_ref, "protocol")
    dev_cells = _cells(development.get("cells"))
    allowed = {(track, road) for track in pool["track_ids"] for road in pool["geometry_seeds"]}
    if (development.get("format") != "haic-dreamerv3-reused-train-diagnostic-v1"
            or development.get("purpose") != "reused-TRAIN-engineering-diagnostic"
            or development.get("freshness_claim") != "reused r6 TRAIN cells; not fresh P1/P1b, evaluation, or promotion"
            or development.get("study_id") != DEVELOPMENT_ID
            or development.get("r6_protocol") != r6_ref
            or development.get("episode_schedule") != [0, 1, 2, 3]
            or development.get("frame_skip") != 4 or development.get("max_steps") != 2000
            or development.get("budgets") != {"random_decision_cap": 8000, "teacher_decision_cap": 8000}
            or dev_cells != tuple((1, road) for road in DEVELOPMENT_ROADS)
            or set(dev_cells) & set(train_cells) or not set(dev_cells) <= allowed
            or {road for _, road in dev_cells} & set(diagnostic["geometry_seeds"])):
        raise ValueError("development must use the four frozen disjoint reused r6 TRAIN roads")
    if (development.get("source_actor") != training_collection.get("source_actor")
            or development.get("random", {}).get("rng_seed") != 7392
            or training_collection.get("random", {}).get("rng_seed") != 7391
            or development.get("source_sha256") != training_collection.get("source_sha256")
            or not isinstance(development.get("source_sha256"), dict)
            or COLLECTOR not in development["source_sha256"]):
        raise ValueError("development source actor or independent random seed differs from freeze")
    collector_sha = _hash(development["source_sha256"][COLLECTOR])
    random_policy = {"policy": "uniform-native-random-v1", "collector_sha256": collector_sha,
                     "rng_seed": 7392, "native_bounds": [-1.0, 1.0], "action_dim": 3}
    random_hash = hashlib.sha256(json.dumps(random_policy, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    if (development["random"].get("source_id") != "uniform-native-random-seed-7392"
            or development["random"].get("source_actor_sha256") != random_hash
            or offline["datasets"]["random"]["source_id"] != "uniform-native-random-seed-7391"
            or random_hash == offline["datasets"]["random"]["source_actor_sha256"]):
        raise ValueError("development random policy is not independently source-pinned")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("exact scorer/model executable source SHA map required")
    for name, digest in sources.items():
        _pinned(root, name, digest, "source")
    if sources["scripts/diagnose/dreamerv3_reused_train_open_loop.py"] != _sha256(Path(__file__)):
        raise ValueError("executing scorer differs from frozen source")
    for name in ("dreamer_v3.py", "common_adapter.py", "haic/algorithms/drq_v2/teacher_replay.py"):
        if offline["source_sha256"].get(name) != sources[name]:
            raise ValueError("current model/archive decoder source differs from training executable")

    scoring = protocol["scoring"]
    if (not isinstance(scoring, dict) or set(scoring) != {
        "context_decisions", "window_decisions", "min_terminal_episodes_per_stratum",
        "min_geometries_per_stratum", "latent_seed", "baselines", "context_mode"
    } or scoring["context_decisions"] != 8 or type(scoring["context_decisions"]) is not int
            or scoring["window_decisions"] != 32 or type(scoring["window_decisions"]) is not int
            or _positive(scoring["min_terminal_episodes_per_stratum"], "min terminal episodes", 32768) < 4
            or _positive(scoring["min_geometries_per_stratum"], "min geometries", 32768) < 4
            or type(scoring["latent_seed"]) is not int or not 0 <= scoring["latent_seed"] < 2**32
            or scoring["baselines"] != BASELINES
            or scoring["context_mode"] != "reset-origin-full-prefix-terminal"):
        raise ValueError("frozen 8-context/32-decision two-window baseline contract required")
    resources = protocol["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "max_archive_bytes", "max_uncompressed_bytes", "max_decisions_per_archive",
        "max_checkpoint_bytes", "max_cgroup_memory_bytes", "min_cgroup_available_bytes"
    }:
        raise ValueError("exact ZIP, checkpoint and cgroup resource caps required")
    for name, maximum in (("max_archive_bytes", 512 * 1024**2),
                          ("max_uncompressed_bytes", 1024 * 1024**2),
                          ("max_decisions_per_archive", 32768),
                          ("max_checkpoint_bytes", 1024 * 1024**2),
                          ("max_cgroup_memory_bytes", 128 * 1024**3),
                          ("min_cgroup_available_bytes", 128 * 1024**3)):
        _positive(resources[name], name, maximum)
    if resources["min_cgroup_available_bytes"] >= resources["max_cgroup_memory_bytes"]:
        raise ValueError("impossible cgroup resource floor")
    needed = max(resources["min_cgroup_available_bytes"],
                 512 * 1024**2 + 2 * resources["max_checkpoint_bytes"]
                 + 2 * resources["max_archive_bytes"] + resources["max_uncompressed_bytes"])
    cgroup = _cgroup()
    if cgroup["limit_bytes"] > resources["max_cgroup_memory_bytes"] or cgroup["available_bytes"] < needed:
        raise ValueError("cgroup lacks frozen score-time headroom")

    training = protocol["training"]
    dev_rows = protocol["development"]
    if (not isinstance(training, dict) or set(training) != {"random", "teacher"}
            or not isinstance(dev_rows, dict) or set(dev_rows) != {"random", "teacher"}):
        raise ValueError("both training models and both development source strata are required")
    checked_training = {}
    checked_development = {}
    inputs: set[Path] = set()
    for arm in ("random", "teacher"):
        rows = training[arm]
        if (not isinstance(rows, list) or len(rows) != 2
                or [row.get("seed") for row in rows if isinstance(row, dict)] != [0, 1]):
            raise ValueError(f"{arm}: exact two fixed training learner seeds required")
        checked_training[arm] = []
        for seed, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != {
                "seed", "result_path", "result_sha256", "checkpoint_path", "checkpoint_sha256"
            }:
                raise ValueError("four exact training result/checkpoint pairs required")
            expected_dir = f"{TRAIN_ROOT}/{arm}-seed-{seed}"
            if (row["result_path"] != f"{expected_dir}/training-result.json"
                    or row["checkpoint_path"] != f"{expected_dir}/world-model-checkpoint.pt"):
                raise ValueError("checkpoint and result must be in the original four training runs")
            result_file = _pinned(root, row["result_path"], row["result_sha256"], "training_result")
            checkpoint_file = _path(root, row["checkpoint_path"], "checkpoint")
            if not 0 < checkpoint_file.stat().st_size <= resources["max_checkpoint_bytes"]:
                raise ValueError("checkpoint exceeds frozen size cap")
            if _sha256(checkpoint_file) != _hash(row["checkpoint_sha256"]):
                raise ValueError("checkpoint SHA-256 mismatch")
            result = _json(result_file)
            train_row = offline["datasets"][arm]
            for key, expected in (
                ("format", "haic-dreamerv3-reused-train-offline-result-v1"),
                ("status", "complete"), ("purpose", offline["purpose"]),
                ("study_id", offline["study_id"]), ("arm", arm), ("seed", seed),
                ("offline_protocol_sha256", OFFLINE_SHA256),
                ("collection_protocol_sha256", offline["collection_protocol"]["sha256"]),
                ("checkpoint_path", "world-model-checkpoint.pt"),
                ("checkpoint_sha256", row["checkpoint_sha256"]),
                ("collection_receipt_sha256", train_row["receipt_sha256"]),
                ("collection_receipt_path", train_row["receipt_path"]),
                ("model_only_updates", 64), ("environment_steps", 0),
                ("actor_trained", False), ("actor_critic_target_unchanged", True),
                ("fresh_claim", False), ("p1b_claim", False), ("promotion_eligible", False),
            ):
                if type(result.get(key)) is not type(expected) or result[key] != expected:
                    raise ValueError(f"{arm}/{seed}: training result {key} disagrees with freeze")
            for key in ("archive_path", "archive_sha256", "dataset_digest", "source_id", "source_actor_sha256"):
                if result.get(key) != train_row[key]:
                    raise ValueError(f"{arm}/{seed}: training source {key} disagrees with offline protocol")
            _check_checkpoint_metadata(checkpoint_file, arm=arm, seed=seed,
                                       dataset_digest=train_row["dataset_digest"],
                                       config=offline["learner"]["config"])
            checked_training[arm].append({"checkpoint": checkpoint_file, "result": result,
                                          "checkpoint_sha256": row["checkpoint_sha256"]})
            inputs.update((checkpoint_file, result_file))
        train_row = offline["datasets"][arm]
        _pinned(root, train_row["receipt_path"], train_row["receipt_sha256"], "receipt")
        train_archive = _path(root, train_row["archive_path"], "archive")
        checked_training[arm][0]["archive"] = train_archive
        inputs.add(train_archive)

        row = dev_rows[arm]
        if not isinstance(row, dict) or set(row) != {
            "receipt_path", "receipt_sha256", "archive_path", "archive_sha256",
            "dataset_digest", "source_id", "source_actor_sha256"
        }:
            raise ValueError(f"{arm}: development archive and receipt SHA pins required")
        receipt_path = _pinned(root, row["receipt_path"], row["receipt_sha256"], "receipt")
        archive_path = _path(root, row["archive_path"], "archive")
        if (receipt_path.parent != archive_path.parent or receipt_path.parent.name != arm
                or archive_path == train_archive or not 0 < archive_path.stat().st_size <= resources["max_archive_bytes"]
                or not isinstance(row["receipt_path"], str) or "/development/" not in row["receipt_path"]):
            raise ValueError(f"{arm}: development must use a separate arm archive/receipt")
        receipt = _json(receipt_path)
        source = development["random"] if arm == "random" else development["source_actor"]
        actor_hash = source["source_actor_sha256" if arm == "random" else "actor_sha256"]
        for key, expected in (
            ("format", "haic-dreamerv3-reused-train-collection-result-v1"),
            ("status", "completed"), ("purpose", development["purpose"]),
            ("study_id", DEVELOPMENT_ID), ("protocol_sha256", dev_ref["sha256"]),
            ("arm", arm), ("source_id", source["source_id"]),
            ("source_actor_sha256", actor_hash), ("dataset_path", "support-dataset.npz"),
            ("archive_sha256", row["archive_sha256"]), ("dataset_digest", row["dataset_digest"]),
            ("decision_cap", 8000), ("r6_protocol_sha256", R6_SHA256),
        ):
            if type(receipt.get(key)) is not type(expected) or receipt[key] != expected or row.get(key, expected) != expected:
                raise ValueError(f"{arm}: development receipt {key} disagrees with frozen sources")
        if (_cells(receipt.get("allowed_cells")) != dev_cells
                or receipt.get("excluded_seeds") != sorted(diagnostic["geometry_seeds"])
                or type(receipt.get("stored_decisions")) is not int
                or not 1 <= receipt["stored_decisions"] <= 8000
                or type(receipt.get("unresolved_decision_calls")) is not int
                or receipt["unresolved_decision_calls"] != 0):
            raise ValueError(f"{arm}: development receipt cell/decision evidence mismatch")
        checked_development[arm] = {"row": row, "receipt": receipt, "archive": archive_path}
        inputs.update((receipt_path, archive_path))
    if (dev_rows["random"]["source_id"] == dev_rows["teacher"]["source_id"]
            or dev_rows["random"]["source_actor_sha256"] == dev_rows["teacher"]["source_actor_sha256"]):
        raise ValueError("development arms must have distinct logged action source identities")
    if output in inputs or any(output in path.parents or path in output.parents for path in inputs):
        raise ValueError("output cannot contain or overwrite any input")
    return {"root": root, "protocol": protocol, "output": output, "offline": offline,
            "training_collection": training_collection, "training_cells": train_cells,
            "development_cells": dev_cells, "training": checked_training,
            "development": checked_development, "cgroup": cgroup, "required_available_bytes": needed}


def _finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite open-loop model prediction or metric")
    return result


def _mean(values: list[float]) -> float | None:
    return _finite(sum(values) / len(values)) if values else None


def _check_checkpoint_metadata(path: Path, *, arm: str, seed: int, dataset_digest: str,
                               config: dict[str, Any]) -> dict[str, Any]:
    # Only call after SHA, fixed original location, size and cgroup preflight.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("run_metadata")
    expected = {"purpose": "reused-TRAIN-engineering-diagnostic",
                "study_id": "dreamerv3-reused-train-diagnostic-v1", "arm": arm,
                "seed": seed, "offline_protocol_sha256": OFFLINE_SHA256,
                "dataset_digest": dataset_digest, "actor_trained": False,
                "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False}
    if (payload.get("format") != "haic-dreamerv3-checkpoint-v3" or metadata != expected
            or type(payload.get("environment_steps")) is not int or payload["environment_steps"] != 0
            or type(payload.get("gradient_steps")) is not int or payload["gradient_steps"] != 64
            or payload.get("config") != config):
        raise ValueError(f"{arm}/{seed}: exact model-only checkpoint metadata/config required")
    if set(config) != {field.name for field in fields(DreamerV3Config)} or config.get("device") != "cpu":
        raise ValueError("scoring requires complete frozen CPU model config")
    return payload


def _checkpoint_modules(path: Path, *, arm: str, seed: int, dataset_digest: str,
                        config: dict[str, Any]) -> tuple[Any, ...]:
    payload = _check_checkpoint_metadata(path, arm=arm, seed=seed,
                                         dataset_digest=dataset_digest, config=config)
    encoder = ConvEncoder(in_channels=4, embed_dim=config["embed_dim"])
    rssm = CategoricalRSSM(action_dim=3, embed_dim=config["embed_dim"],
                           hidden_dim=config["hidden_dim"], num_categoricals=config["num_categoricals"],
                           num_classes=config["num_classes"], unimix=config["unimix"])
    decoder = ConvDecoder(in_features=rssm.state_dim, out_channels=1)
    reward = RewardHead(in_features=rssm.state_dim, bins=config["twohot_bins"])
    continuation = ContinueHead(in_features=rssm.state_dim)
    for name, module in (("encoder", encoder), ("rssm", rssm), ("decoder", decoder),
                         ("reward_head", reward), ("continue_head", continuation)):
        module.load_state_dict(payload[name], strict=True)
        module.eval()
    del payload
    return encoder, rssm, decoder, reward, continuation


def _score_episode(episode: Any, modules: tuple[Any, ...], baseline_reward: float,
                   prevalence: float, rng_seed: int) -> dict[str, Any]:
    encoder, rssm, decoder, reward_head, continue_head = modules
    length = episode.steps
    cell = {"track_id": episode.track_id, "geometry_seed": int(episode.geometry_id)}
    result: dict[str, Any] = {"episode_id": episode.episode_id, **cell, "decisions": length,
                              "terminal_event": bool(episode.terminal[-1]),
                              "finished": bool(episode.finished[-1]), "windows": [], "metrics": None}
    if length < 41:
        result["coverage_note"] = "requires 8 reset-context decisions and two distinct 32-decision anchors"
        return result
    anchors = (("reset", 8), ("terminal", length - 32))
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(rng_seed)
        # The terminal anchor replays the actual reset-origin observation/action
        # prefix, not a fake reset eight steps before the terminal window.
        max_anchor = anchors[-1][1]
        embeds = []
        for start in range(0, max_anchor + 1, 32):
            stop = min(start + 32, max_anchor + 1)
            stacks = np.stack([episode.observation(i) for i in range(start, stop)])
            embeds.append(encoder(torch.from_numpy(stacks)).unsqueeze(0))
        embeddings = torch.cat(embeds, dim=1)
        context_actions = torch.from_numpy(episode.actions[:max_anchor].copy()).unsqueeze(0)
        first = torch.zeros((1, max_anchor), dtype=torch.bool)
        first[:, 0] = True
        states, _, _ = rssm.observe_sequence(embeddings, context_actions, first)
        for name, anchor in anchors:
            state = states[:, anchor].clone()
            stack = torch.from_numpy(episode.observation(anchor).copy()).unsqueeze(0)
            repeat = stack.clone()
            metrics: dict[str, list[float]] = defaultdict(list)
            positive = 0
            for index in range(anchor, anchor + 32):
                action = torch.from_numpy(episode.actions[index].copy()).unsqueeze(0)
                h, z = torch.split(state, [rssm.hidden_dim, rssm.stoch_dim], dim=-1)
                next_h, next_z, _, _ = rssm.step_prior(h, z, action)
                state = torch.cat((next_h, next_z), dim=-1)
                latest = (stack[:, -1:] + decoder(state)).clamp(0.0, 1.0)
                stack = torch.cat((stack[:, 1:], latest), dim=1)
                repeat = torch.cat((repeat[:, 1:], repeat[:, -1:]), dim=1)
                target = torch.from_numpy(episode.observation(index + 1)[-1].copy()).unsqueeze(0)
                actual = _finite(episode.rewards[index])
                predicted = _finite(reward_head.pred(state).item())
                terminal = bool(episode.terminal[index])
                positive += int(terminal)
                p = min(max(_finite(1.0 - torch.sigmoid(continue_head(state)).item()), 1e-7), 1 - 1e-7)
                prior = min(max(prevalence, 1e-7), 1 - 1e-7)
                metrics["image_mse"].append(_finite(torch.mean((latest[:, -1] - target) ** 2).item()))
                metrics["shifted_repeat_mse"].append(_finite(torch.mean((repeat[:, -1] - target) ** 2).item()))
                metrics["reward_mse"].append(_finite((predicted - actual) ** 2))
                metrics["training_constant_reward_mse"].append(_finite((baseline_reward - actual) ** 2))
                previous = float(episode.rewards[index - 1]) if index else baseline_reward
                metrics["last_logged_reward_mse"].append(_finite((previous - actual) ** 2))
                metrics["terminal_bce"].append(_finite(-math.log(p if terminal else 1 - p)))
                metrics["training_prevalence_bce"].append(_finite(-math.log(prior if terminal else 1 - prior)))
            result["windows"].append({"kind": name, "context_anchor_decision": anchor,
                                       "reset_origin_prefix_decisions": anchor, "first_target_decision": anchor,
                                       "last_target_decision": anchor + 31, "label_uses": 32,
                                       "terminal_positive_label_uses": positive,
                                       "metrics": {key: _mean(values) for key, values in metrics.items()}})
    start, terminal_start = (anchor for _, anchor in anchors)
    unique = set(range(start, start + 32)) | set(range(terminal_start, terminal_start + 32))
    result["unique_scored_decisions"] = len(unique)
    result["duplicate_label_uses"] = 64 - len(unique)
    result["unique_terminal_positive_labels"] = sum(bool(episode.terminal[index]) for index in unique)
    result["metrics"] = {key: _mean([window["metrics"][key] for window in result["windows"]])
                         for key in result["windows"][0]["metrics"]}
    return result


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in episodes if row["metrics"] is not None]
    return {"episode_count": len(scored), "window_count": sum(len(row["windows"]) for row in scored),
            "window_label_uses": sum(32 * len(row["windows"]) for row in scored),
            "unique_scored_decisions": sum(row["unique_scored_decisions"] for row in scored),
            "duplicate_label_uses": sum(row["duplicate_label_uses"] for row in scored),
            "terminal_positive_label_uses": sum(sum(window["terminal_positive_label_uses"]
                                                    for window in row["windows"]) for row in scored),
            "unique_terminal_positive_labels": sum(row["unique_terminal_positive_labels"] for row in scored),
            "metrics_episode_mean": {key: _mean([row["metrics"][key] for row in scored])
                                     for key in scored[0]["metrics"]} if scored else None}


def score(protocol_path: Path, protocol_sha256: str, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, output_dir, repo_root=repo_root)
    protocol = checked["protocol"]
    resources = protocol["resources"]
    current = _cgroup()
    if (current["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or current["available_bytes"] < checked["required_available_bytes"]):
        raise ValueError("cgroup lacks archive-load headroom after checkpoint preflight")
    torch.set_num_threads(1)
    training_baselines = {}
    dev_datasets = {}
    for arm in ("random", "teacher"):
        train_row = checked["offline"]["datasets"][arm]
        train_dataset = _load_dataset(checked["training"][arm][0]["archive"], train_row, resources,
                                      checked["training_cells"], checked["offline"]["study_id"], arm)
        if (train_dataset.transition_count != train_row["stored_decisions"]
                or set((ep.track_id, int(ep.geometry_id)) for ep in train_dataset.episodes)
                != set(checked["training_cells"])):
            raise ValueError("training archive cells/decisions disagree with offline freeze")
        count = train_dataset.transition_count
        training_baselines[arm] = {
            "training_decisions": count,
            "training_terminal_events": sum(int(ep.terminal.sum()) for ep in train_dataset.episodes),
            "constant_reward": _finite(sum(float(ep.rewards.astype(np.float64).sum())
                                           for ep in train_dataset.episodes) / count),
        }
        training_baselines[arm]["terminal_prevalence"] = _finite(
            training_baselines[arm]["training_terminal_events"] / count)
        del train_dataset
        dev = checked["development"][arm]
        dev_dataset = _load_dataset(dev["archive"], dev["row"], resources,
                                    checked["development_cells"], DEVELOPMENT_ID, arm)
        receipt = dev["receipt"]
        if (dev_dataset.transition_count != receipt["stored_decisions"]
                or len(dev_dataset.episodes) != receipt["complete_episode_count"]):
            raise ValueError("development archive episode/decision counts disagree with receipt")
        complete_rows = [row for row in receipt["episode_rows"] if row.get("status") == "complete"]
        if len(complete_rows) != len(dev_dataset.episodes):
            raise ValueError("development receipt complete episodes disagree with sealed data")
        for episode, row in zip(dev_dataset.episodes, complete_rows, strict=True):
            if (type(row.get("episode_id")) is not int or str(row["episode_id"]) != episode.episode_id
                    or episode.metadata.get("attempt") != row.get("attempt")
                    or (episode.track_id, int(episode.geometry_id)) != (row.get("track_id"), row.get("geometry_seed"))
                    or episode.steps != row.get("decisions") or bool(episode.terminal[-1]) != row.get("terminal")
                    or bool(episode.finished[-1]) != row.get("finished")):
                raise ValueError("development episode does not match its collection receipt")
        dev_datasets[arm] = dev_dataset

    report = {"format": FORMAT, "purpose": PURPOSE, "score_protocol_sha256": protocol_sha256,
              "offline_protocol_sha256": OFFLINE_SHA256,
              "development_collection_protocol_sha256": protocol["development_collection_protocol"]["sha256"],
              "source_sha256": protocol["source_sha256"], "sampling": protocol["scoring"],
              "status": "descriptive_only", "fresh_claim": False, "p1b_claim": False,
              "promotion_eligible": False, "student_actor_trained": False,
              "interpretation": "Training-excluded but explicitly reused r6 TRAIN roads. No fresh, held-out, "
                                "confirmation, blind, official, actor or promotion inference; window labels "
                                "on one road are not independent experiments. Last-logged-reward uses the "
                                "preceding recorded reward, including inside the scored window.",
              "strata": {}}
    with torch.no_grad():
        for source_arm in ("random", "teacher"):
            dataset = dev_datasets[source_arm]
            terminal_episodes = sum(bool(ep.terminal[-1]) for ep in dataset.episodes)
            usable = [ep for ep in dataset.episodes if ep.steps >= 41]
            scored_terminals = sum(bool(ep.terminal[-1]) for ep in usable)
            geometry_count = len({(ep.track_id, ep.geometry_id) for ep in usable})
            reasons = []
            if scored_terminals < protocol["scoring"]["min_terminal_episodes_per_stratum"]:
                reasons.append("fewer_than_minimum_independent_scored_terminal_episodes")
            if geometry_count < protocol["scoring"]["min_geometries_per_stratum"]:
                reasons.append("fewer_than_minimum_independent_scored_geometries")
            stratum = {"development_archive_sha256": checked["development"][source_arm]["row"]["archive_sha256"],
                       "source_id": checked["development"][source_arm]["row"]["source_id"],
                       "collected_complete_episodes": len(dataset.episodes), "collected_terminal_episodes": terminal_episodes,
                       "scored_episode_count": len(usable), "scored_independent_terminal_episodes": scored_terminals,
                       "scored_independent_geometries": geometry_count,
                       "coverage": "inconclusive" if reasons else "descriptive_sufficient_minimum",
                       "coverage_reasons": reasons, "models": []}
            for model_arm in ("random", "teacher"):
                baseline = training_baselines[model_arm]
                for seed, model_row in enumerate(checked["training"][model_arm]):
                    if _sha256(model_row["checkpoint"]) != model_row["checkpoint_sha256"]:
                        raise ValueError("checkpoint changed since scoring preflight")
                    modules = _checkpoint_modules(model_row["checkpoint"], arm=model_arm, seed=seed,
                                                   dataset_digest=checked["offline"]["datasets"][model_arm]["dataset_digest"],
                                                   config=checked["offline"]["learner"]["config"])
                    episodes = [_score_episode(ep, modules, baseline["constant_reward"],
                                               baseline["terminal_prevalence"],
                                               protocol["scoring"]["latent_seed"] + (source_arm == "teacher") * 1_000_003
                                               + index * 1009 + seed)
                                for index, ep in enumerate(dataset.episodes)]
                    del modules
                    roads = defaultdict(list)
                    for row in episodes:
                        roads[(row["track_id"], row["geometry_seed"])].append(row)
                    stratum["models"].append({
                        "model_arm": model_arm, "learner_seed": seed,
                        "checkpoint_sha256": model_row["checkpoint_sha256"],
                        "training_baselines": baseline,
                        "aggregate": _aggregate(episodes),
                        "roads": [{"track_id": track, "geometry_seed": road, **_aggregate(items)}
                                  for (track, road), items in sorted(roads.items())],
                        "episodes": episodes,
                    })
            report["strata"][source_arm] = stratum
    if any(item["coverage"] == "inconclusive" for item in report["strata"].values()):
        report["status"] = "coverage_inconclusive"
    # Never create/overwrite output until all sources and finite metrics have been checked.
    for name, digest in protocol["source_sha256"].items():
        _pinned(checked["root"], name, digest, "source")
    _pinned(checked["root"], _relative(checked["root"], protocol_path), protocol_sha256, "protocol")
    for arm in ("random", "teacher"):
        row = checked["development"][arm]["row"]
        _pinned(checked["root"], row["receipt_path"], row["receipt_sha256"], "receipt")
        if _sha256(checked["development"][arm]["archive"]) != row["archive_sha256"]:
            raise ValueError("development archive changed while scoring")
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
    print(json.dumps({"status": result["status"], "output": str(args.output / "score-result.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
