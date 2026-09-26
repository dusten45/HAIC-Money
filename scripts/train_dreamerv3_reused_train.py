"""Offline, non-promoting Dreamer world-model diagnostic on explicitly reused TRAIN.

Freeze this separate offline protocol only after both collection receipts exist. Run
one arm and one learner seed per invocation; no actor training or environment access.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
from typing import Any

import numpy as np
import torch

import common_adapter
import dreamer_v3
from dreamer_v3 import DreamerV3Agent, DreamerV3Config, NoValidSequenceError
from haic.algorithms.dreamer_v3 import offline
from haic.algorithms.dreamer_v3.offline import replay_from_dataset_bytes
from haic.algorithms.drq_v2 import teacher_replay


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-dreamerv3-reused-train-offline-v1"
PURPOSE = "reused-TRAIN-engineering-diagnostic"
COLLECTION_FORMAT = "haic-dreamerv3-reused-train-diagnostic-v1"
R6_PATH = "experiments/drqv2-geometry-mix-v1-r6.json"
R6_SHA256 = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
SOURCE_PATHS = frozenset({
    "scripts/train_dreamerv3_reused_train.py", "dreamer_v3.py",
    "haic/algorithms/dreamer_v3/offline.py",
    "haic/algorithms/drq_v2/teacher_replay.py", "common_adapter.py",
})
COLLECTOR_PATH = "scripts/diagnose/collect_dreamerv3_reused_train.py"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN = ("blind", "confirm", "screen", "eval", "held-out", "holdout", "submission")


class UnavailableSequenceError(RuntimeError):
    """The sealed replay cannot supply the frozen learner sequence."""


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parts(name: str) -> tuple[str, ...]:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or Path(name).is_absolute()):
        raise ValueError("paths must be normalized repository-relative paths")
    parts = tuple(name.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("paths must be normalized repository-relative paths")
    return parts


def _relative(root: Path, path: Path) -> str:
    path = Path(path)
    try:
        return (path if path.is_absolute() else root / path).relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path must be inside the repository") from exc


def _path(root: Path, name: str, kind: str, *, existing: bool = True) -> Path:
    parts = _parts(name)
    if kind in ("protocol", "r6"):
        permitted = len(parts) == 2 and parts[0] == "experiments" and name.endswith(".json")
    elif kind == "source":
        permitted = name.endswith(".py") and (
            parts[0] in ("scripts", "haic", "core") or name in {
                "dreamer_v3.py", "common_adapter.py", "train.py", "drq_v2.py",
                "env_wrapper.py", "damage.py", "tracking.py", "action_smoothing.py",
                "action_representation.py",
            }
        )
    elif kind in ("receipt", "archive"):
        permitted = (len(parts) >= 4 and parts[0] == "runs"
                     and parts[-2] in ("random", "teacher")
                     and parts[-1] == ("collection-result.json" if kind == "receipt" else "support-dataset.npz"))
    elif kind == "output":
        permitted = len(parts) >= 3 and parts[0] == "runs"
    else:
        raise ValueError(f"unsupported path kind {kind}")
    if not permitted or (parts[0] == "runs" and any(
        token in part.lower() for part in parts[1:] for token in _FORBIDDEN
    )):
        raise ValueError(f"{kind}: not an allowed TRAIN path: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{kind}: symlink path forbidden: {name}")
    if existing and not path.is_file():
        raise ValueError(f"{kind}: missing regular file: {name}")
    if not existing and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"no overwrite or resume: {name}")
    return path


def _pinned(root: Path, name: str, digest: str, kind: str) -> Path:
    path = _path(root, name, kind)
    if _sha256(path) != _digest(digest, f"{kind} SHA-256"):
        raise ValueError(f"{kind}: SHA-256 mismatch: {name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"JSON exceeds 1 MiB: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {token}")))
    if not isinstance(value, dict):
        raise ValueError("pinned JSON must be an object")
    return value


def _cells(value: Any) -> set[tuple[int, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("TRAIN cells must be a nonempty explicit list")
    cells = set()
    seeds = set()
    for cell in value:
        if (not isinstance(cell, dict) or set(cell) != {"track_id", "geometry_seed"}
                or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)
                or type(cell["geometry_seed"]) is not int or not 0 <= cell["geometry_seed"] < 2**32
                or cell["geometry_seed"] in seeds):
            raise ValueError("TRAIN cells require unique uint32 road IDs on tracks 1..4")
        seeds.add(cell["geometry_seed"])
        cells.add((cell["track_id"], cell["geometry_seed"]))
    return cells


def _seeds(value: Any) -> set[int]:
    if (not isinstance(value, list) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in value)
            or len(set(value)) != len(value)):
        raise ValueError("excluded_seeds must contain distinct uint32 road IDs")
    return set(value)


def _positive(value: Any, name: str, maximum: int) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return value


def _cgroup_memory() -> dict[str, int]:
    """Require a finite cgroup-v2 memory limit; never guess host free memory."""
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError("finite cgroup-v2 memory.max and memory.current are required") from exc
    if limit <= 0 or used < 0 or used > limit:
        raise ValueError("invalid cgroup memory limit/current usage")
    return {"limit_bytes": limit, "used_bytes": used, "available_bytes": limit - used}


def preflight(protocol_path: Path, protocol_sha256: str, arm: str, seed: int,
              output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate both arms and resource bounds before touching any archive bytes."""
    root = Path(repo_root).resolve()
    name = _relative(root, protocol_path)
    if not name.startswith("experiments/dreamerv3-reused-train-"):
        raise ValueError("offline protocol must be a separate Dreamer reused-TRAIN experiment")
    protocol = _json(_pinned(root, name, protocol_sha256, "protocol"))
    if (set(protocol) != {"format", "purpose", "study_id", "collection_protocol", "source_sha256",
                          "datasets", "learner", "resources"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or not isinstance(protocol["study_id"], str)
            or not protocol["study_id"].startswith("dreamerv3-reused-train-")):
        raise ValueError("not a separate frozen reused-TRAIN offline protocol")
    if arm not in ("random", "teacher") or type(seed) is not int:
        raise ValueError("one random/teacher arm and one frozen integer learner seed required")
    output_name = _relative(root, output_dir)
    output = _path(root, output_name, "output", existing=False)
    if not output.parent.is_dir():
        raise ValueError("output parent must be an existing TRAIN run directory")

    ref = protocol["collection_protocol"]
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("collection protocol path and SHA-256 are required")
    collection = _json(_pinned(root, ref["path"], ref["sha256"], "protocol"))
    if (collection.get("format") != COLLECTION_FORMAT or collection.get("purpose") != PURPOSE
            or collection.get("study_id") != protocol["study_id"]):
        raise ValueError("collection protocol format, purpose or study_id mismatch")
    r6_ref = collection.get("r6_protocol")
    if (not isinstance(r6_ref, dict) or set(r6_ref) != {"path", "sha256"}
            or r6_ref != {"path": R6_PATH, "sha256": R6_SHA256}):
        raise ValueError("collection must pin the known reused r6 TRAIN protocol")
    r6 = _json(_pinned(root, r6_ref["path"], r6_ref["sha256"], "r6"))
    pool = r6.get("training_pool")
    diagnostic = r6.get("diagnostic_pool")
    if (r6.get("study_id") != "drqv2-geometry-mix-v1-r6" or not isinstance(pool, dict)
            or pool.get("partition") != "TRAIN" or not isinstance(diagnostic, dict)
            or diagnostic.get("partition") != "TRAIN-DIAGNOSTIC"):
        raise ValueError("r6 TRAIN catalog allocation and TRAIN-DIAGNOSTIC split required")
    collection_cells = _cells(collection.get("cells"))
    if not collection_cells <= {
        (track, road) for track in pool["track_ids"] for road in pool["geometry_seeds"]
    } or {road for _, road in collection_cells} & set(diagnostic["geometry_seeds"]):
        raise ValueError("collection cells are outside known-reused r6 TRAIN roads")
    collection_budgets = collection.get("budgets")
    if (collection.get("frame_skip") != 4 or type(collection.get("frame_skip")) is not int
            or type(collection.get("max_steps")) is not int or not 0 < collection["max_steps"] <= 2000
            or not isinstance(collection_budgets, dict)
            or set(collection_budgets) != {"random_decision_cap", "teacher_decision_cap"}):
        raise ValueError("collection must freeze both decision caps and the unshaped environment cadence")
    for label in ("random", "teacher"):
        _positive(collection_budgets[f"{label}_decision_cap"], f"{label} decision cap", 32768)
    random_source = collection.get("random")
    teacher_source = collection.get("source_actor")
    if (not isinstance(random_source, dict) or not isinstance(teacher_source, dict)
            or any(key not in random_source for key in ("source_id", "source_actor_sha256"))
            or any(key not in teacher_source for key in ("source_id", "actor_sha256"))):
        raise ValueError("collection protocol must identify both frozen source policies")

    collection_sources = collection.get("source_sha256")
    if not isinstance(collection_sources, dict) or COLLECTOR_PATH not in collection_sources:
        raise ValueError("collection executable sources must include the exact collector")
    for source_name, digest in collection_sources.items():
        _pinned(root, source_name, digest, "source")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("offline protocol must pin exact executable source hashes")
    for source_name, digest in sources.items():
        _pinned(root, source_name, digest, "source")
    if (sources["scripts/train_dreamerv3_reused_train.py"] != _sha256(Path(__file__))
            or sources["dreamer_v3.py"] != _sha256(Path(dreamer_v3.__file__))
            or sources["haic/algorithms/dreamer_v3/offline.py"] != _sha256(Path(offline.__file__))
            or sources["haic/algorithms/drq_v2/teacher_replay.py"] != _sha256(Path(teacher_replay.__file__))
            or sources["common_adapter.py"] != _sha256(Path(common_adapter.__file__))):
        raise ValueError("executing offline learner differs from pinned executable source")

    learner = protocol["learner"]
    if not isinstance(learner, dict) or set(learner) != {"seeds", "config", "updates"}:
        raise ValueError("learner must freeze seeds, complete config and update budget")
    seeds = learner["seeds"]
    if (not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(value) is not int or not 0 <= value < 2**32 for value in seeds)
            or seed not in seeds):
        raise ValueError("learner seed is not one of two frozen matched seeds")
    updates = _positive(learner["updates"], "model-only updates", 10000)
    values = learner["config"]
    if not isinstance(values, dict) or set(values) != {field.name for field in fields(DreamerV3Config)}:
        raise ValueError("complete identical DreamerV3Config is required for both arms")
    default = asdict(DreamerV3Config(device="cpu"))
    if any(type(value) is not type(default[key]) or (type(value) is float and not math.isfinite(value))
           for key, value in values.items()) or values["device"] != "cpu":
        raise ValueError("offline learner requires finite, typed CPU config")
    config = DreamerV3Config(**values)
    if (not 1 <= config.batch_size <= 32 or not 1 <= config.seq_len <= 64
            or not 0 <= config.burnin_steps <= 64 or config.replay_capacity < 1
            or any(not 0 <= fraction <= 1 for fraction in (
                config.terminal_window_fraction, config.reset_start_fraction, config.short_episode_fraction
            )) or min(config.embed_dim, config.hidden_dim, config.num_categoricals, config.num_classes,
                       config.twohot_bins) <= 0 or config.twohot_bins % 2 != 1
            or config.embed_dim > 512 or config.hidden_dim > 512
            or config.num_categoricals > 32 or config.num_classes > 32 or config.twohot_bins < 3
            or config.model_lr <= 0 or config.grad_clip_norm <= 0
            or config.continue_positive_weight <= 0 or not 0 <= config.unimix <= 1):
        raise ValueError("unsafe or unavailable frozen Dreamer sequence/config")

    resources = protocol["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "max_archive_bytes", "max_cgroup_memory_bytes", "min_cgroup_available_bytes"
    }:
        raise ValueError("frozen archive and cgroup resource caps required")
    archive_cap = _positive(resources["max_archive_bytes"], "max_archive_bytes", 512 * 1024**2)
    memory_cap = _positive(resources["max_cgroup_memory_bytes"], "max_cgroup_memory_bytes", 128 * 1024**3)
    free_floor = _positive(resources["min_cgroup_available_bytes"], "min_cgroup_available_bytes", memory_cap)
    datasets = protocol["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != {"random", "teacher"}:
        raise ValueError("both random and teacher frozen datasets required")
    for label, source in (("random", random_source), ("teacher", teacher_source)):
        row = datasets[label]
        if (not isinstance(row, dict) or row.get("source_id") != source["source_id"]
                or row.get("source_actor_sha256") != source[
                    "source_actor_sha256" if label == "random" else "actor_sha256"
                ]):
            raise ValueError(f"{label}: source identity differs from collection protocol")
    checked = {}
    collection_root = None
    for label, row in datasets.items():
        if not isinstance(row, dict) or set(row) != {
            "receipt_path", "receipt_sha256", "archive_path", "archive_sha256", "dataset_digest",
            "source_id", "source_actor_sha256", "stored_decisions", "allowed_cells", "excluded_seeds"
        }:
            raise ValueError(f"{label}: incomplete frozen dataset identity")
        receipt_name, archive_name = row["receipt_path"], row["archive_path"]
        receipt_file = _pinned(root, receipt_name, row["receipt_sha256"], "receipt")
        archive_file = _path(root, archive_name, "archive")
        if (receipt_name.split("/")[:-1] != archive_name.split("/")[:-1]
                or receipt_name.split("/")[-2] != label
                or output == archive_file or output == receipt_file
                or output in archive_file.parents or archive_file in output.parents):
            raise ValueError(f"{label}: paths must be in the same collection arm and outside output")
        current_root = tuple(receipt_name.split("/")[:-2])
        if collection_root is not None and current_root != collection_root:
            raise ValueError("both collection arms must share a single pinned collection root")
        collection_root = current_root
        _digest(row["archive_sha256"], "archive SHA-256")
        _digest(row["dataset_digest"], "dataset digest")
        _digest(row["source_actor_sha256"], "source actor SHA-256")
        _positive(row["stored_decisions"], "stored_decisions", config.replay_capacity)
        if (not isinstance(row["source_id"], str) or not row["source_id"].strip()
                or _cells(row["allowed_cells"]) != collection_cells):
            raise ValueError(f"{label}: invalid source, stored decisions or paired TRAIN cells")
        excluded = _seeds(row["excluded_seeds"])
        if ({road for _, road in collection_cells} & excluded
                or set(diagnostic["geometry_seeds"]) - excluded):
            raise ValueError(f"{label}: TRAIN cells overlap exclusions or omit r6 TRAIN-DIAGNOSTIC")
        receipt = _json(receipt_file)
        for key, expected in (
            ("format", "haic-dreamerv3-reused-train-collection-result-v1"),
            ("purpose", PURPOSE), ("status", "completed"),
            ("study_id", protocol["study_id"]), ("protocol_sha256", ref["sha256"]),
            ("arm", label), ("source_id", row["source_id"]),
            ("source_actor_sha256", row["source_actor_sha256"]),
            ("archive_sha256", row["archive_sha256"]), ("dataset_digest", row["dataset_digest"]),
            ("stored_decisions", row["stored_decisions"]), ("dataset_path", "support-dataset.npz"),
            ("decision_cap", collection_budgets[f"{label}_decision_cap"]),
        ):
            if type(receipt.get(key)) is not type(expected) or receipt[key] != expected:
                raise ValueError(f"{label}: collection receipt {key} differs from frozen protocol")
        if (_cells(receipt.get("allowed_cells")) != collection_cells
                or _seeds(receipt.get("excluded_seeds")) != excluded
                or type(receipt.get("decisions_spent")) is not int
                or receipt.get("schedule_exhausted") is not True
                or type(receipt.get("schedule_attempts")) is not int
                or receipt["schedule_attempts"] != len(collection_cells)
                or type(receipt.get("complete_episode_count")) is not int
                or receipt["complete_episode_count"] < 1
                or type(receipt.get("partial_decisions")) is not int
                or receipt["partial_decisions"] != 0
                or type(receipt.get("unresolved_decision_calls")) is not int
                or receipt["unresolved_decision_calls"] != 0
                or receipt["decisions_spent"] != row["stored_decisions"]
                or not row["stored_decisions"] <= receipt["decisions_spent"] <= receipt["decision_cap"]):
            raise ValueError(f"{label}: incomplete collection schedule, cells or decision counts")
        size = archive_file.stat().st_size
        if not 0 < size <= archive_cap:
            raise ValueError(f"{label}: archive size exceeds frozen cap before archive read")
        checked[label] = {"archive": archive_file, "receipt": receipt, "row": row, "size": size}
    if (datasets["random"]["source_id"] == datasets["teacher"]["source_id"]
            or datasets["random"]["source_actor_sha256"] == datasets["teacher"]["source_actor_sha256"]
            or _seeds(datasets["random"]["excluded_seeds"]) != _seeds(datasets["teacher"]["excluded_seeds"])):
        raise ValueError("random and teacher require distinct sources and identical exclusions")
    if ("excluded_seeds" in collection
            and _seeds(collection["excluded_seeds"]) != _seeds(datasets[arm]["excluded_seeds"])):
        raise ValueError("collection exclusions differ from offline protocol")
    cgroup = _cgroup_memory()
    # Model/optimizer, autograd, replay and checkpoint copies need headroom beyond compressed bytes.
    needed = max(free_floor, 256 * 1024**2 + 3 * config.replay_capacity * (4 * 84 * 84 + 64)
                 + 4 * max(item["size"] for item in checked.values()))
    if (cgroup["limit_bytes"] > memory_cap or cgroup["available_bytes"] < needed):
        raise ValueError("cgroup memory exceeds frozen cap or lacks preflight headroom")
    return {"root": root, "protocol": protocol, "config": config, "datasets": checked,
            "output": output, "cgroup": cgroup, "required_available_bytes": needed}


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _archive_bytes(path: Path, *, expected_size: int, cap: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size or info.st_size > cap:
            raise ValueError("archive size or type changed after resource preflight")
        data = source.read(cap + 1)
    if len(data) != expected_size:
        raise ValueError("archive size changed while reading bounded bytes")
    return data


def _unchanged(agent: DreamerV3Agent, snapshots: dict[str, dict[str, torch.Tensor]]) -> bool:
    return all(torch.equal(value, snapshots[name][key]) for name in snapshots
               for key, value in getattr(agent, name).state_dict().items())


def train_arm(protocol_path: Path, protocol_sha256: str, arm: str, seed: int,
              output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, arm, seed, output_dir, repo_root=repo_root)
    protocol = checked["protocol"]
    row = checked["datasets"][arm]["row"]
    output = checked["output"]
    output.mkdir(exist_ok=False)
    completed = 0
    phase = "archive verification"
    try:
        archive = checked["datasets"][arm]["archive"]
        cgroup = _cgroup_memory()
        if (cgroup["limit_bytes"] > protocol["resources"]["max_cgroup_memory_bytes"]
                or cgroup["available_bytes"] < checked["required_available_bytes"]):
            raise ValueError("cgroup memory lacks frozen archive-load headroom")
        data = _archive_bytes(archive, expected_size=checked["datasets"][arm]["size"],
                              cap=protocol["resources"]["max_archive_bytes"])
        if hashlib.sha256(data).hexdigest() != row["archive_sha256"]:
            raise ValueError("dataset archive SHA-256 differs from frozen protocol")
        phase = "sealed replay materialization"
        replay, audit = replay_from_dataset_bytes(
            data, archive_sha256=row["archive_sha256"], dataset_digest=row["dataset_digest"],
            source_id=row["source_id"], source_actor_sha256=row["source_actor_sha256"],
            allowed_cells=[(cell["track_id"], cell["geometry_seed"]) for cell in row["allowed_cells"]],
            excluded_seeds=row["excluded_seeds"], capacity=checked["config"].replay_capacity,
        )
        # The bridge validates the entire sealed archive; read only its small
        # per-episode metadata/finish array again to retain actual finish IDs.
        with np.load(BytesIO(data), allow_pickle=False) as sealed:
            episodes = json.loads(sealed["manifest"].tobytes())["episodes"]
            offsets = sealed["transition_offsets"]
            finished = sealed["finished"]
            finished_cells = sorted({
                (episode["track_id"], int(episode["geometry_id"]))
                for episode, end in zip(episodes, offsets[1:]) if bool(finished[end - 1])
            })
        del data
        if (audit["decisions"] != row["stored_decisions"] or replay.size != row["stored_decisions"]
                or len(finished_cells) != audit["distinct_finished_cells"]):
            raise ValueError("actual stored TRAIN decisions differ from frozen collection")
        terminal_events = int(replay.is_terminal[:replay.size].sum())
        phase = "model-only updates"
        torch.set_num_threads(1)
        agent = DreamerV3Agent(checked["config"], seed=seed)
        agent.replay = replay
        snapshots = {name: {key: value.detach().clone() for key, value in getattr(agent, name).state_dict().items()}
                     for name in ("actor", "critic", "critic_target")}
        with (output / "update-metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(protocol["learner"]["updates"]):
                previous = agent.gradient_steps
                try:
                    metrics = agent.update(model_only=True)
                except NoValidSequenceError as exc:
                    raise UnavailableSequenceError(
                        f"unavailable replay sequence at model-only update {index + 1}"
                    ) from exc
                if not metrics:
                    raise UnavailableSequenceError(f"unavailable replay sequence at model-only update {index + 1}")
                if (not isinstance(metrics, dict) or "loss_wm" not in metrics
                        or any(not isinstance(key, str) or type(value) not in (int, float)
                               or not math.isfinite(value) for key, value in metrics.items())
                        or any(key.startswith("loss_actor") or key.startswith("loss_critic") for key in metrics)
                        or agent.gradient_steps != previous + 1 or agent.environment_steps != 0
                        or agent.actor_optimizer.state or agent.critic_optimizer.state
                        or not _unchanged(agent, snapshots)):
                    raise ValueError("model-only learner produced invalid metrics, counter or actor/critic update")
                current_cgroup = _cgroup_memory()
                if (current_cgroup["limit_bytes"] > protocol["resources"]["max_cgroup_memory_bytes"]
                        or current_cgroup["available_bytes"] < protocol["resources"]["min_cgroup_available_bytes"]):
                    raise ValueError("cgroup memory exceeded the frozen training resource floor")
                stream.write(json.dumps({"update": index + 1, "metrics": metrics}, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                completed += 1
        if (agent.gradient_steps != protocol["learner"]["updates"] or not _unchanged(agent, snapshots)
                or agent.environment_steps != 0):
            raise ValueError("model-only updates changed actor/critic/target or missed frozen budget")
        _pinned(checked["root"], _relative(checked["root"], protocol_path), protocol_sha256, "protocol")
        _pinned(checked["root"], protocol["collection_protocol"]["path"],
                protocol["collection_protocol"]["sha256"], "protocol")
        for source_name, source_digest in protocol["source_sha256"].items():
            _pinned(checked["root"], source_name, source_digest, "source")
        for label, frozen in checked["datasets"].items():
            _pinned(checked["root"], frozen["row"]["receipt_path"],
                    frozen["row"]["receipt_sha256"], "receipt")
            if _sha256(frozen["archive"]) != frozen["row"]["archive_sha256"]:
                raise ValueError(f"{label}: frozen archive changed during training")
        phase = "checkpoint and receipt"
        metrics_path = output / "update-metrics.jsonl"
        checkpoint_path = output / "world-model-checkpoint.pt"
        agent.save_checkpoint(checkpoint_path, run_metadata={
            "purpose": PURPOSE, "study_id": protocol["study_id"], "arm": arm, "seed": seed,
            "offline_protocol_sha256": protocol_sha256, "dataset_digest": row["dataset_digest"],
            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
            "promotion_eligible": False,
        })
        result = {
            "format": "haic-dreamerv3-reused-train-offline-result-v1",
            "purpose": PURPOSE, "study_id": protocol["study_id"], "status": "complete",
            "arm": arm, "seed": seed, "offline_protocol_sha256": protocol_sha256,
            "collection_protocol_sha256": protocol["collection_protocol"]["sha256"],
            "collection_receipt_path": row["receipt_path"], "collection_receipt_sha256": row["receipt_sha256"],
            "archive_path": row["archive_path"], "archive_sha256": row["archive_sha256"],
            "dataset_digest": row["dataset_digest"], "source_id": row["source_id"],
            "source_actor_sha256": row["source_actor_sha256"],
            "dataset_evidence": {
                **audit, "terminal_events": terminal_events,
                "finished_cells": [{"track_id": track, "geometry_seed": road}
                                   for track, road in finished_cells],
            },
            "collection_decisions_spent": checked["datasets"][arm]["receipt"]["decisions_spent"],
            "environment_steps": agent.environment_steps, "model_only_updates": completed,
            "actor_critic_target_unchanged": True, "actor_trained": False,
            "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
            "resource_snapshot": {"preflight_cgroup": checked["cgroup"],
                                  "required_available_bytes": checked["required_available_bytes"],
                                  "archive_size_bytes": checked["datasets"][arm]["size"],
                                  "replay_memory_bytes": replay.memory_bytes, "post_cgroup": _cgroup_memory(),
                                  "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
            "checkpoint_path": checkpoint_path.name, "checkpoint_sha256": _sha256(checkpoint_path),
            "metrics_path": metrics_path.name, "metrics_sha256": _sha256(metrics_path),
        }
        _write_json(output / "training-result.json", result)
        return result
    except BaseException as exc:
        _write_json(output / "abort.json", {
            "format": "haic-dreamerv3-reused-train-offline-abort-v1", "purpose": PURPOSE,
            "study_id": protocol["study_id"], "arm": arm, "seed": seed,
            "offline_protocol_sha256": protocol_sha256, "phase": phase,
            "completed_model_only_updates": completed, "error_type": type(exc).__name__,
            "error": str(exc), "actor_trained": False, "fresh_claim": False,
            "p1b_claim": False, "promotion_eligible": False,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=("random", "teacher"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = train_arm(args.protocol, args.protocol_sha256, args.arm, args.seed,
                       args.output, repo_root=args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
