"""Two separately pinned DrQ sources on the same consumed r6 TRAIN roads.

Schema: format, purpose, study_id, source0_protocol, source1_protocol,
source1_root_receipt, r6_protocol, catalog, historical_train_summary,
base_model_protocol, source_sha256, cells, datasets, learner, resources.
Datasets are keyed source0/source1; each pins receipt_path/sha256,
archive_path/sha256, dataset_digest, source_id, source_actor_sha256,
source_checkpoint_sha256, stored_decisions. Learner pins seeds, updates and
the full DreamerV3Config. Resources pin max_archive_bytes,
max_total_uncompressed_bytes, max_cgroup_memory_bytes,
min_cgroup_available_bytes and min_disk_available_bytes.

Freeze the new protocol and pass its SHA before invoking. One seed per invocation;
no environment, auxiliary objective, policy update, or evaluation is run here.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import fields
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import resource
from typing import Any
from zipfile import ZipFile

import torch

import common_adapter
from common_adapter import Transition
import dreamer_v3
from dreamer_v3 import DreamerV3Agent, DreamerV3Config, Uint8SequenceReplay
from haic.algorithms.dreamer_v3 import offline
from haic.algorithms.drq_v2 import teacher_replay
from scripts import train_dreamerv3_reused_train as single


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-dreamerv3-reused-train-multisource-v1"
PURPOSE = "reused-TRAIN-engineering-diagnostic"
STUDY_ID = "dreamerv3-reused-train-multisource-v1"
PROTOCOL_PATH = "experiments/dreamerv3-reused-train-multisource-v1.json"
OUTPUT_ROOT = "runs/20260926-dreamerv3-reused-train-multisource-v1"
REPLAY_CAPACITY = 16384
UPDATES = 256
ROADS = (3910800008, 3910800011, 3910800019, 3910800007,
         3910800074, 3910800069, 3910800010, 3910800016,
         3910800041, 3910800030, 3910800081, 3910800070)
FAMILIES = ("opening-short-entry-left-turn", "opening-delayed-high-turn",
            "easy-curvature-anchor", "mid-road-left-right-reversal",
            "mid-road-sustained-or-same-turn", "finish-approach-turn") * 2
REFS = {
    "source0_protocol": ("experiments/dreamerv3-reused-train-diversity-v1.json", "a2f3e92fd1b2e95f883b24e3ef28d58b3e70938a0dd867d2c9ec5da5bd46df83"),
    "source1_protocol": ("experiments/dreamerv3-reused-train-source1-v1.json", "cc6746d0a7c51b27112d1d0b7569d2483cdfc7f7c03d81e8dc7b4fcc42c47262"),
    "source1_root_receipt": ("runs/20260926-dreamerv3-reused-train-source1-v1/collection/collection-result.json", "502676b60faa51de06fc0f72f53edcb57e53710c063dbc965bc8e4689398984c"),
    "r6_protocol": ("experiments/drqv2-geometry-mix-v1-r6.json", "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"),
    "catalog": ("runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json", "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"),
    "historical_train_summary": ("experiments/drqv2-geometry-augmentation-v1-training-summary.json", "29fe9d2b8d4eff04879cc9c1390b4701e4f235af710856f39acf6592f65015ee"),
    "base_model_protocol": ("experiments/dreamerv3-reused-train-offline-v2.json", "ba837ebc3f64282bbd77b4444c6f337c23a3ebce12602d328e9626c1c406b290"),
}
SOURCES = {
    "source0": {
        "receipt_path": "runs/20260926-dreamerv3-reused-train-diversity-v1/collection/teacher/collection-result.json",
        "receipt_sha256": "00a388f5eabdd9707b096dfb735b2d717b309b86a9daa1423d2c16177fa2c353",
        "archive_path": "runs/20260926-dreamerv3-reused-train-diversity-v1/collection/teacher/support-dataset.npz",
        "archive_sha256": "c7e988303d485744c73d6bb987d40a44ae51e0f11855ba4d33e8255bf971c596",
        "dataset_digest": "4eb201475c013ef669dcfc12cab798747c15d2e82b6a0b7d21d6024d6d991d3d",
        "source_id": "drq-source-0", "source_actor_sha256": "433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37",
        "source_checkpoint_sha256": "4248750c7114afdf955ea85a60c842565498aba3a64e380fa81164c7eb340979",
        "stored_decisions": 5420,
    },
    "source1": {
        "receipt_path": "runs/20260926-dreamerv3-reused-train-source1-v1/collection/teacher/collection-result.json",
        "receipt_sha256": "21918ba3d83bd947e896d15a96b5aed4a34e4d723514cf892b77a0f0dc56fe48",
        "archive_path": "runs/20260926-dreamerv3-reused-train-source1-v1/collection/teacher/support-dataset.npz",
        "archive_sha256": "aa8af2512ef248c0d8927459b2acb175eb46ffb56afbcd9e3e0e668080bd0607",
        "dataset_digest": "f1176a0e7661cc5594a891e172b9918dc6e3a9a97477ed4281e52e1e3db5c126",
        "source_id": "drq-source-1", "source_actor_sha256": "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
        "source_checkpoint_sha256": "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4",
        "stored_decisions": 5192,
    },
}
SOURCE_PATHS = frozenset({
    "scripts/train_dreamerv3_reused_multisource.py", "scripts/train_dreamerv3_reused_train.py",
    "dreamer_v3.py", "haic/algorithms/dreamer_v3/offline.py",
    "haic/algorithms/drq_v2/teacher_replay.py", "common_adapter.py",
})
RUNTIME_SOURCES = {
    "scripts/train_dreamerv3_reused_multisource.py": Path(__file__),
    "scripts/train_dreamerv3_reused_train.py": Path(single.__file__),
    "dreamer_v3.py": Path(dreamer_v3.__file__),
    "haic/algorithms/dreamer_v3/offline.py": Path(offline.__file__),
    "haic/algorithms/drq_v2/teacher_replay.py": Path(teacher_replay.__file__),
    "common_adapter.py": Path(common_adapter.__file__),
}


class LineageReplay(Uint8SequenceReplay):
    """Reject even a sampled window crossing a differently sourced episode."""

    def bind(self, rows: list[dict[str, Any]]) -> None:
        self.lineage = rows
        self._starts = [row["sequence_id_start"] for row in rows]

    def sample_sequence(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        batch = super().sample_sequence(*args, **kwargs)
        for ids in batch["sequence_ids"].tolist():
            index = bisect_right(self._starts, ids[0]) - 1
            if (index < 0 or ids != list(range(ids[0], ids[0] + len(ids)))
                    or ids[-1] > self.lineage[index]["sequence_id_end"]):
                raise ValueError("sampled sequence crosses source/episode lineage")
        return batch


def _reference(root: Path, name: str, digest: str, *, max_bytes: int = 4 * 1024**2) -> dict[str, Any]:
    parts = single._parts(name)
    if parts[0] not in ("experiments", "runs") or not name.endswith(".json"):
        raise ValueError("reference is not a TRAIN JSON path")
    path = root
    for part in parts:
        path /= part
        if path.is_symlink() or any(token in part.lower() for token in single._FORBIDDEN):
            raise ValueError("reference symlink or protected partition")
    if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"reference missing or exceeds bound: {name}")
    if single._sha256(path) != single._digest(digest, "reference SHA-256"):
        raise ValueError(f"reference SHA-256 mismatch: {name}")
    return single._json(path) if path.stat().st_size <= 1024**2 else _large_json(path)


def _large_json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(value, dict):
        raise ValueError("reference must be an object")
    return value


def _disk_available(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _snapshot(agent: DreamerV3Agent) -> dict[str, Any]:
    # Clone optimizer tensors as well as parameters: an initialized optimizer
    # could otherwise mutate its existing state without changing its identity.
    import copy
    return {name: copy.deepcopy(getattr(agent, name).state_dict()) for name in (
        "actor", "critic", "critic_target", "actor_optimizer", "critic_optimizer")}


def _unchanged(agent: DreamerV3Agent, snapshot: dict[str, Any]) -> bool:
    return all(_same(state, getattr(agent, name).state_dict()) for name, state in snapshot.items())


def preflight(protocol_path: Path, protocol_sha256: str, seed: int, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if single._relative(root, protocol_path) != PROTOCOL_PATH:
        raise ValueError("new multi-source protocol path is fixed")
    protocol = _reference(root, PROTOCOL_PATH, protocol_sha256)
    if (set(protocol) != {"format", "purpose", "study_id", *REFS, "source_sha256", "cells",
                          "datasets", "learner", "resources"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != STUDY_ID):
        raise ValueError("not a separately frozen multi-source TRAIN protocol")
    output_name = single._relative(root, output_dir)
    if output_name != f"{OUTPUT_ROOT}/learner-{seed}":
        raise ValueError("output must be the fixed TRAIN-only learner-seed directory")
    output = single._path(root, output_name, "output", existing=False)
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    refs = {}
    for key, (name, digest) in REFS.items():
        if protocol[key] != {"path": name, "sha256": digest}:
            raise ValueError(f"{key}: reference identity differs from frozen source")
        refs[key] = _reference(root, name, digest)
    cells = [{"track_id": 1, "geometry_seed": road} for road in ROADS]
    if protocol["cells"] != cells or len(set(ROADS)) != len(ROADS):
        raise ValueError("must reuse exactly the twelve fixed TRAIN roads")
    r6, s0, s1 = (refs[key] for key in ("r6_protocol", "source0_protocol", "source1_protocol"))
    if (r6.get("study_id") != "drqv2-geometry-mix-v1-r6"
            or r6.get("training_pool", {}).get("partition") != "TRAIN"
            or 1 not in r6["training_pool"].get("track_ids", [])
            or not set(ROADS) <= set(r6["training_pool"].get("geometry_seeds", []))
            or r6.get("diagnostic_pool", {}).get("partition") != "TRAIN-DIAGNOSTIC"
            or set(ROADS) & set(r6["diagnostic_pool"].get("geometry_seeds", []))
            or r6.get("catalog_sha256") != REFS["catalog"][1]):
        raise ValueError("r6 TRAIN membership/catalog or diagnostic exclusion failed")
    for source, label, expected_format in ((s0, "source0", "haic-dreamerv3-reused-train-diagnostic-v1"),
                                            (s1, "source1", "haic-dreamerv3-reused-train-source1-v1")):
        if (source.get("format") != expected_format or source.get("purpose") != PURPOSE
                or source.get("r6_protocol") != protocol["r6_protocol"]
                or source.get("cells") != cells or source.get("episode_schedule") != list(range(len(ROADS)))
                or source.get("frame_skip") != 4 or source.get("max_steps") != 2000
                or source.get("source_actor", {}).get("source_id") != SOURCES[label]["source_id"]
                or source["source_actor"].get("actor_sha256") != SOURCES[label]["source_actor_sha256"]
                or source["source_actor"].get("checkpoint_sha256") != SOURCES[label]["source_checkpoint_sha256"]):
            raise ValueError(f"{label}: source protocol/cell/actor drift")
    if (s0.get("budgets", {}).get("teacher_decision_cap") != 24000
            or s1.get("budgets") != {"teacher_decision_cap": 24000}
            or s1.get("source0_protocol") != protocol["source0_protocol"]
            or s1.get("source0_teacher_receipt") != {
                "path": SOURCES["source0"]["receipt_path"],
                "sha256": SOURCES["source0"]["receipt_sha256"]}
            or any(s1.get(key) != protocol[key] for key in (
                "r6_protocol", "catalog", "historical_train_summary"))):
        raise ValueError("source1 provenance does not bind the failed source0 attempt")
    catalog = refs["catalog"]
    found = {}
    for item in catalog.get("train", []):
        if isinstance(item, dict) and item.get("geometry_seed") in ROADS:
            road = item["geometry_seed"]
            if road in found:
                raise ValueError("duplicate catalog TRAIN road")
            found[road] = item.get("family")
    if (catalog.get("format") != "haic-drq-training-geometry-catalog-v1"
            or found != dict(zip(ROADS, FAMILIES))):
        raise ValueError("pinned catalog family or TRAIN road mismatch")
    historical = refs["historical_train_summary"]
    if (historical.get("format") != "haic-drq-training-geometry-diagnostic-summary-v1"
            or historical.get("catalog_sha256") != REFS["catalog"][1]):
        raise ValueError("historical summary is not catalog-bound")
    prior = {}
    for item in historical.get("geometry", []):
        if isinstance(item, dict) and item.get("geometry_seed") in ROADS:
            road = item["geometry_seed"]
            if (road in prior or item.get("partition") != "train" or item.get("family") != found[road]
                    or type(item.get("by_source", {}).get("1", {}).get("finished")) is not bool):
                raise ValueError("historical source1 TRAIN metadata drift")
            prior[road] = item["by_source"]["1"]["finished"]
    if set(prior) != set(ROADS):
        raise ValueError("missing historical TRAIN road")
    source_hashes = protocol["source_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != SOURCE_PATHS:
        raise ValueError("incomplete executable source hashes")
    for name, path in RUNTIME_SOURCES.items():
        single._pinned(root, name, source_hashes[name], "source")
        if single._sha256(path) != source_hashes[name]:
            raise ValueError(f"executing source SHA-256 differs: {name}")
    learner = protocol["learner"]
    if (not isinstance(learner, dict) or set(learner) != {"seeds", "updates", "config"}
            or learner["seeds"] != [0, 1] or type(seed) is not int or seed not in (0, 1)
            or type(learner["updates"]) is not int or learner["updates"] != UPDATES):
        raise ValueError("exact two learner seeds and model-only update budget required")
    base = refs["base_model_protocol"].get("learner", {}).get("config")
    values = learner["config"]
    if (not isinstance(base, dict) or not isinstance(values, dict)
            or set(values) != {field.name for field in fields(DreamerV3Config)}
            or base.get("replay_capacity") != 8192
            or values != {**base, "replay_capacity": REPLAY_CAPACITY}
            or values.get("device") != "cpu" or any(
                type(value) is not type(base[key]) or (type(value) is float and not math.isfinite(value))
                for key, value in values.items())):
        raise ValueError("CPU model config must equal frozen v2 except replay capacity")
    config = DreamerV3Config(**values)
    if not isinstance(protocol["datasets"], dict) or protocol["datasets"] != SOURCES:
        raise ValueError("both exact distinct teacher archive/receipt/source pairs required")
    if sum(row["stored_decisions"] for row in SOURCES.values()) > config.replay_capacity:
        raise ValueError("combined replay capacity would silently overwrite a source")
    resources = protocol["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "max_archive_bytes", "max_total_uncompressed_bytes", "max_cgroup_memory_bytes",
        "min_cgroup_available_bytes", "min_disk_available_bytes"}:
        raise ValueError("archive/uncompressed/cgroup/disk caps required")
    archive_cap = single._positive(resources["max_archive_bytes"], "archive cap", 64 * 1024**2)
    unpack_cap = single._positive(resources["max_total_uncompressed_bytes"], "uncompressed cap", 512 * 1024**2)
    memory_cap = single._positive(resources["max_cgroup_memory_bytes"], "memory cap", 128 * 1024**3)
    free_floor = single._positive(resources["min_cgroup_available_bytes"], "memory floor", memory_cap)
    disk_floor = single._positive(resources["min_disk_available_bytes"], "disk floor", 1024**4)
    if free_floor < 8 * 1024**3 or disk_floor < 2 * 1024**3:
        raise ValueError("cgroup/disk floors below safe model-only minimum")
    checked = {}
    for label, row in SOURCES.items():
        receipt_path = single._pinned(root, row["receipt_path"], row["receipt_sha256"], "receipt")
        archive = single._path(root, row["archive_path"], "archive")
        receipt = single._json(receipt_path)
        if (row["receipt_path"].rsplit("/", 1)[0] != row["archive_path"].rsplit("/", 1)[0]
                or not 0 < archive.stat().st_size <= archive_cap):
            raise ValueError(f"{label}: archive path/size exceeds bound")
        source_protocol = s0 if label == "source0" else s1
        if (receipt.get("format") != "haic-dreamerv3-reused-train-collection-result-v1"
                or receipt.get("status") != "completed" or receipt.get("arm") != "teacher"
                or receipt.get("purpose") != PURPOSE or receipt.get("study_id") != source_protocol["study_id"]
                or receipt.get("protocol_sha256") != REFS[f"{label}_protocol"][1]
                or receipt.get("r6_protocol_sha256") != REFS["r6_protocol"][1]
                or any(receipt.get(key) != row[key] for key in (
                    "source_id", "source_actor_sha256", "source_checkpoint_sha256",
                    "archive_sha256", "dataset_digest", "stored_decisions"))
                or receipt.get("dataset_path") != "support-dataset.npz"
                or receipt.get("allowed_cells") != cells
                or receipt.get("excluded_seeds") != sorted(r6["diagnostic_pool"]["geometry_seeds"])
                or receipt.get("schedule_exhausted") is not True
                or receipt.get("schedule_attempts") != len(ROADS)
                or receipt.get("complete_episode_count") != len(ROADS)
                or receipt.get("partial_decisions") != 0 or receipt.get("unresolved_decision_calls") != 0
                or receipt.get("decisions_spent") != row["stored_decisions"]
                or receipt.get("decision_cap") != 24000):
            raise ValueError(f"{label}: incomplete or mismatched teacher collection receipt")
        episodes = receipt.get("episode_rows")
        if not isinstance(episodes, list) or len(episodes) != len(ROADS):
            raise ValueError(f"{label}: missing collection episode rows")
        for i, (record, cell) in enumerate(zip(episodes, cells)):
            if (not isinstance(record, dict) or record.get("attempt") != i
                    or record.get("episode_id") != i or record.get("status") != "complete"
                    or record.get("complete") is not True or type(record.get("terminal")) is not bool
                    or type(record.get("finished")) is not bool
                    or any(record.get(key) != value for key, value in cell.items())
                    or type(record.get("decisions")) is not int
                    or not 1 <= record["decisions"] <= 2000):
                raise ValueError(f"{label}: missing/partial/heldout episode in receipt")
        if sum(record["decisions"] for record in episodes) != row["stored_decisions"]:
            raise ValueError(f"{label}: receipt episode decision count mismatch")
        if receipt.get("distinct_finished_geometries") != sorted(
                record["geometry_seed"] for record in episodes if record["finished"]):
            raise ValueError(f"{label}: finished-road receipt drift")
        checked[label] = {"row": row, "receipt": receipt, "archive": archive,
                          "size": archive.stat().st_size}
    finished = {label: {road for road in ROADS if checked[label]["receipt"]["episode_rows"][ROADS.index(road)]["finished"]}
                for label in SOURCES}
    union = finished["source0"] | finished["source1"]
    if (len(finished["source0"]) != 2 or len(finished["source1"]) != 2
            or len(finished["source1"] - finished["source0"]) < 1 or len(union) < 3
            or len({found[road] for road in union}) < 2):
        raise ValueError("distinct-road source-union support design gate failed")
    root_receipt = refs["source1_root_receipt"]
    if (root_receipt.get("protocol_sha256") != REFS["source1_protocol"][1]
            or root_receipt.get("status") != "completed"
            or root_receipt.get("catalog") != protocol["catalog"]
            or root_receipt.get("historical_source1_reference", {}).get("path") != REFS["historical_train_summary"][0]
            or root_receipt["historical_source1_reference"].get("sha256") != REFS["historical_train_summary"][1]
            or root_receipt.get("teacher") != checked["source1"]["receipt"]
            or root_receipt.get("source0_reference", {}).get("original_learner_gate_passed") is not False
            or root_receipt["source0_reference"].get("protocol_path") != REFS["source0_protocol"][0]
            or root_receipt["source0_reference"].get("protocol_sha256") != REFS["source0_protocol"][1]
            or root_receipt["source0_reference"].get("teacher_receipt_path") != SOURCES["source0"]["receipt_path"]
            or root_receipt["source0_reference"].get("teacher_receipt_sha256") != SOURCES["source0"]["receipt_sha256"]
            or root_receipt["source0_reference"].get("finished_geometries") != sorted(finished["source0"])
            or root_receipt.get("source1_finished_geometries") != sorted(finished["source1"])
            or root_receipt.get("union_finished_geometries") != sorted(union)
            or root_receipt.get("support_gate", {}).get("passed") is not True
            or root_receipt["support_gate"].get("requires_complete_twelve_attempts") is not True
            or root_receipt["support_gate"].get("authorizes_training") is not False
            or root_receipt["support_gate"].get("merges_archives") is not False):
        raise ValueError("source1 design-only receipt cannot authorize historical training")
    cgroup = single._cgroup_memory()
    replay_bytes = REPLAY_CAPACITY * (4 * 84 * 84 + 32)
    required = max(free_floor, 256 * 1024**2 + 3 * replay_bytes + 4 * unpack_cap)
    disk_required = max(disk_floor, 2 * replay_bytes + 256 * 1024**2)
    if cgroup["limit_bytes"] > memory_cap or cgroup["available_bytes"] < required:
        raise ValueError("cgroup memory below pinned capacity/headroom floor")
    if _disk_available(output.parent) < disk_required:
        raise ValueError("disk headroom below pinned checkpoint floor")
    return {"root": root, "protocol": protocol, "refs": refs, "checked": checked,
            "config": config, "output": output, "cgroup": cgroup,
            "required_available_bytes": required, "required_disk_bytes": disk_required,
            "uncompressed_cap": unpack_cap, "finished": finished}


def materialize(checked: dict[str, Any]) -> tuple[LineageReplay, list[dict[str, Any]], dict[str, Any]]:
    """Verify each source separately, then interleave whole episodes by road."""
    if sum(item["row"]["stored_decisions"] for item in checked["checked"].values()) > checked["config"].replay_capacity:
        raise ValueError("combined replay capacity would silently overwrite a source")
    replay = LineageReplay(capacity=checked["config"].replay_capacity)
    lineage: list[dict[str, Any]] = []
    datasets = {}
    audit = {}
    total_uncompressed = 0
    for label, item in checked["checked"].items():
        row = item["row"]
        raw = single._archive_bytes(item["archive"], expected_size=item["size"],
                                    cap=checked["protocol"]["resources"]["max_archive_bytes"])
        if hashlib.sha256(raw).hexdigest() != row["archive_sha256"]:
            raise ValueError(f"{label}: archive SHA-256 changed")
        with ZipFile(BytesIO(raw)) as archive:
            total_uncompressed += sum(member.file_size for member in archive.infolist())
        if total_uncompressed > checked["uncompressed_cap"]:
            raise ValueError("combined uncompressed archive exceeds pinned cap")
        # The original bridge enforces this source's SHA, digest, actor, native
        # action/terminal contract and complete T+1 frames before any joining.
        source_replay, audit[label] = offline.replay_from_dataset_bytes(
            raw, archive_sha256=row["archive_sha256"], dataset_digest=row["dataset_digest"],
            source_id=row["source_id"], source_actor_sha256=row["source_actor_sha256"],
            allowed_cells=[(1, road) for road in ROADS],
            excluded_seeds=item["receipt"]["excluded_seeds"], capacity=row["stored_decisions"])
        if (source_replay.size != row["stored_decisions"]
                or source_replay.total_steps != row["stored_decisions"]
                or audit[label]["decisions"] != row["stored_decisions"]
                or audit[label]["episodes"] != len(ROADS)
                or {tuple(cell) for cell in audit[label]["covered_cells"]} != {(1, road) for road in ROADS}
                or audit[label]["distinct_finished_cells"] != len(checked["finished"][label])):
            raise ValueError(f"{label}: single-source replay evidence drift")
        del source_replay
        dataset = teacher_replay.TeacherDataset.from_bytes(raw, expected_digest=row["dataset_digest"])
        del raw
        if len(dataset.episodes) != len(ROADS) or dataset.transition_count != row["stored_decisions"]:
            raise ValueError(f"{label}: source archive has missing episodes")
        for i, episode in enumerate(dataset.episodes):
            record = item["receipt"]["episode_rows"][i]
            if (not episode.complete or episode.source_id != row["source_id"]
                    or episode.source_actor_sha256 != row["source_actor_sha256"]
                    or episode.track_id != 1 or episode.geometry_id != str(ROADS[i])
                    or record.get("track_id") != episode.track_id
                    or record.get("geometry_seed") != ROADS[i]
                    or episode.episode_id != str(record["episode_id"])
                    or episode.steps != record["decisions"]
                    or bool(episode.finished[-1]) is not record["finished"]
                    or bool(episode.terminal[-1]) is not record["terminal"]
                    or bool(episode.terminated[-1]) is not record["terminated"]
                    or bool(episode.truncated[-1]) is not record["truncated"]
                    or not (episode.terminated[-1] or episode.truncated[-1])):
                raise ValueError(f"{label}: archive source/road/terminal/receipt drift")
        datasets[label] = dataset
    provenance: set[tuple[str, str, int]] = set()
    for i, road in enumerate(ROADS):
        for label in ("source0", "source1"):
            episode = datasets[label].episodes[i]
            row = checked["checked"][label]["row"]
            key = (episode.source_id, episode.episode_id, road)
            if key in provenance:
                raise ValueError("duplicate teacher episode provenance")
            provenance.add(key)
            start = replay.total_steps
            episode_id = 2 * i + (label == "source1")
            for step in range(episode.steps):
                replay.add(Transition(
                    observation=episode.observation(step), action=episode.actions[step],
                    applied_action=episode.applied_actions[step], reward=float(episode.rewards[step]),
                    next_observation=episode.observation(step + 1),
                    terminated=bool(episode.terminated[step]), truncated=bool(episode.truncated[step]),
                    terminal=bool(episode.terminal[step]),
                    info={"finished": bool(episode.finished[step]), "progress": float(episode.progress[step]),
                          "damage": float(episode.damage[step]), "retire_reason": episode.retire_reasons[step]},
                    episode_id=episode_id, step=step))
            if (replay.size != replay.total_steps or replay.total_steps - start != episode.steps
                    or replay.sequence_ids[start] != start or not replay.is_first[start]
                    or not replay.is_last[replay.total_steps - 1]
                    or replay.total_steps - 1 not in replay._boundary_observations):
                raise ValueError("replay dropped transitions, reset or T+1 boundary frame")
            lineage.append({
                "source_id": episode.source_id, "actor_sha256": row["source_actor_sha256"],
                "archive_sha256": row["archive_sha256"], "dataset_digest": row["dataset_digest"],
                "road": {"track_id": 1, "geometry_seed": road},
                "original_episode_id": episode.episode_id, "replay_episode_id": episode_id,
                "sequence_id_start": start, "sequence_id_end": replay.total_steps - 1,
            })
    if (len(lineage) != 2 * len(ROADS) or replay.size != sum(
            item["row"]["stored_decisions"] for item in checked["checked"].values())
            or replay.size > replay.capacity or replay.total_steps != replay.size):
        raise ValueError("combined replay lost episodes or silently overwrote transitions")
    replay.bind(lineage)
    return replay, lineage, audit


def train(protocol_path: Path, protocol_sha256: str, seed: int, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, seed, output_dir, repo_root=repo_root)
    output = checked["output"]
    output.mkdir(exist_ok=False)
    completed = 0
    phase = "archive verification"
    try:
        if (single._cgroup_memory()["available_bytes"] < checked["required_available_bytes"]
                or _disk_available(output) < checked["required_disk_bytes"]):
            raise ValueError("resource headroom fell below archive-load floor")
        replay, lineage, audit = materialize(checked)
        phase = "lineage sidecar"
        lineage_path = output / "lineage.json"
        single._write_json(lineage_path, {"format": FORMAT + "-lineage", "range_convention": "inclusive",
                                          "episodes": lineage, "transition_count": replay.size})
        lineage_sha = single._sha256(lineage_path)
        lineage_name = single._relative(checked["root"], lineage_path)
        phase = "model-only updates"
        torch.set_num_threads(1)
        agent = DreamerV3Agent(checked["config"], seed=seed)
        agent.replay = replay
        snapshot = _snapshot(agent)
        with (output / "update-metrics.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(UPDATES):
                previous = agent.gradient_steps
                metrics = agent.update(model_only=True)
                if (not isinstance(metrics, dict) or "loss_wm" not in metrics
                        or any(not isinstance(key, str) or type(value) not in (int, float)
                               or not math.isfinite(value) for key, value in metrics.items())
                        or any(key.startswith(("loss_actor", "loss_critic", "loss_aux")) for key in metrics)
                        or agent.gradient_steps != previous + 1 or agent.environment_steps != 0
                        or not _unchanged(agent, snapshot)
                        or any(not torch.isfinite(param).all().item() for module in (
                            agent.encoder, agent.rssm, agent.decoder, agent.reward_head, agent.continue_head)
                            for param in module.parameters())):
                    raise ValueError("model-only metrics, counters, optimizer or parameters invalid")
                current = single._cgroup_memory()
                if (current["limit_bytes"] > checked["protocol"]["resources"]["max_cgroup_memory_bytes"]
                        or current["available_bytes"] < checked["protocol"]["resources"]["min_cgroup_available_bytes"]
                        or _disk_available(output) < checked["protocol"]["resources"]["min_disk_available_bytes"]):
                    raise ValueError("cgroup/disk fell below training floor")
                stream.write(json.dumps({"update": index + 1, "metrics": metrics}, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                completed += 1
        if agent.gradient_steps != UPDATES or agent.environment_steps != 0 or not _unchanged(agent, snapshot):
            raise ValueError("model-only budget or actor/critic/optimizer invariant failed")
        _reference(checked["root"], PROTOCOL_PATH, protocol_sha256)
        for key, (name, digest) in REFS.items():
            _reference(checked["root"], name, digest)
        for name, digest in checked["protocol"]["source_sha256"].items():
            single._pinned(checked["root"], name, digest, "source")
        for row in SOURCES.values():
            single._pinned(checked["root"], row["receipt_path"], row["receipt_sha256"], "receipt")
            if single._sha256(checked["root"] / row["archive_path"]) != row["archive_sha256"]:
                raise ValueError("source archive changed during model-only training")
        if single._sha256(lineage_path) != lineage_sha:
            raise ValueError("lineage sidecar changed before checkpoint")
        phase = "checkpoint and training receipt"
        metadata = {"purpose": PURPOSE, "study_id": STUDY_ID, "seed": seed,
                    "protocol_sha256": protocol_sha256, "lineage_path": lineage_name,
                    "lineage_sha256": lineage_sha, "sources": checked["protocol"]["datasets"],
                    "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
                    "promotion_eligible": False}
        checkpoint_path = output / "world-model-checkpoint.pt"
        agent.save_checkpoint(checkpoint_path, run_metadata=metadata)
        result = {"format": FORMAT + "-result", "status": "complete", "purpose": PURPOSE,
                  "study_id": STUDY_ID, "seed": seed, "protocol_sha256": protocol_sha256,
                  "sources": checked["protocol"]["datasets"], "source_audits": audit,
                  "distinct_training_roads": len(ROADS), "episodes": len(lineage), "decisions": replay.size,
                  "union_finished_roads": sorted(checked["finished"]["source0"] | checked["finished"]["source1"]),
                  "lineage_path": lineage_name, "lineage_sha256": lineage_sha,
                  "model_only_updates": completed, "environment_steps": agent.environment_steps,
                  "actor_critic_target_optimizers_unchanged": True, "actor_trained": False,
                  "fresh_claim": False, "p1b_claim": False, "promotion_eligible": False,
                  "resource_snapshot": {"preflight_cgroup": checked["cgroup"],
                                        "replay_memory_bytes": replay.memory_bytes,
                                        "post_cgroup": single._cgroup_memory(),
                                        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                  "checkpoint_path": checkpoint_path.name, "checkpoint_sha256": single._sha256(checkpoint_path),
                  "metrics_path": "update-metrics.jsonl",
                  "metrics_sha256": single._sha256(output / "update-metrics.jsonl")}
        single._write_json(output / "training-result.json", result)
        return result
    except BaseException as exc:
        single._write_json(output / "abort.json", {
            "format": FORMAT + "-abort", "study_id": STUDY_ID, "purpose": PURPOSE,
            "seed": seed, "protocol_sha256": protocol_sha256, "phase": phase,
            "completed_model_only_updates": completed, "error_type": type(exc).__name__,
            "error": str(exc), "actor_trained": False, "fresh_claim": False,
            "p1b_claim": False, "promotion_eligible": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.protocol, args.protocol_sha256, args.seed, args.output,
                           repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
