"""Shared protocol, provenance, and offline-data helpers for pixel RLPD CLIs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np

from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition


ROOT = Path(__file__).resolve().parents[1]
STUDY_FORMAT = "haic-pixel-rlpd-study-v1"
DATASET_FORMAT = "haic-rlpd-prior-dataset-v1"
SUPPORTED_TEXT_SUFFIXES = {
    ".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def runtime_metadata(*, device: str) -> dict[str, Any]:
    import cv2
    import gymnasium
    import torch

    distributions = sorted(
        f"{distribution.metadata.get('Name', '').lower()}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    metadata = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "opencv": cv2.__version__,
        "device": device,
        "installed_distributions": distributions,
        "installed_distributions_sha256": canonical_sha256(distributions),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        properties = torch.cuda.get_device_properties(index)
        metadata["gpu"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return metadata


def geometry_seeds_from_ledger(path: Path) -> set[int]:
    seeds = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid episode ledger JSON at {path}:{line_number}") from exc
            seed = record.get("geometry_seed", record.get("seed"))
            if record.get("event") in {"reset", "resume"} and type(seed) is int:
                seeds.add(seed)
    return seeds


def audit_candidate_seed_tokens(seeds: list[int]) -> dict[str, Any]:
    """Search recorded source artifacts for exact candidate seed tokens."""
    seed_to_int = {str(seed): seed for seed in seeds}
    alternatives = "|".join(sorted(seed_to_int, key=len, reverse=True))
    token_pattern = re.compile(rf"(?<![A-Za-z0-9.])({alternatives})(?![A-Za-z0-9.])")
    searched = []
    known_uses = []
    excluded_dirs = {".git", ".venv", "__pycache__", "node_modules", "talk"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        # The source audit messages intentionally discuss candidate values and are
        # coordination, not evidence that an environment cell was consumed.
        if relative.parts[0] == "talk":
            continue
        searched.append(str(relative))
        seen_here = set()
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    for match in token_pattern.finditer(line):
                        seed = seed_to_int[match.group(1)]
                        if seed not in seen_here:
                            known_uses.append({"seed": seed, "artifact": str(relative)})
                            seen_here.add(seed)
        except (OSError, UnicodeError):
            continue
    return {
        "method": "Exact standalone uint32 token search in recorded text artifacts; decimal/hash substrings and current talk coordination are excluded.",
        "searched_artifact_count": len(searched),
        "searched_artifacts_sha256": canonical_sha256(searched),
        "known_recorded_uses": known_uses,
        "scope": "No known recorded use only; incomplete historical pilot/training schedules preclude a global non-use proof.",
    }


def verify_teacher_ledgers(protocol: dict[str, Any]) -> dict[str, list[int]]:
    audit = protocol.get("geometry_audit", {})
    ledgers = audit.get("teacher_source_ledgers")
    if not isinstance(ledgers, list) or len(ledgers) < 1:
        raise ValueError("protocol must record all source-teacher training ledgers")
    used = {}
    evaluation_seeds = {
        seed
        for partition in protocol.get("partitions", {}).values()
        for seed in partition.get("seeds", [])
    }
    for partition in protocol.get("future_full_reservation", {}).values():
        if isinstance(partition, dict):
            evaluation_seeds.update(partition.get("seeds", []))
    all_training_seeds = set(protocol.get("training_geometry_seeds", []))
    future_training = protocol.get("future_full_reservation", {})
    all_training_seeds.update(future_training.get("training_geometry_seeds", []))
    for item in ledgers:
        relative = item.get("path")
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"teacher source episode ledger is missing: {relative}")
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"teacher source episode ledger changed: {relative}")
        seeds = geometry_seeds_from_ledger(path)
        overlap = evaluation_seeds.intersection(seeds)
        if overlap:
            raise ValueError(f"evaluation seed overlaps recorded teacher data: {sorted(overlap)}")
        training_overlap = all_training_seeds.intersection(seeds)
        if training_overlap:
            raise ValueError(f"candidate teacher data reuses source training geometry: {sorted(training_overlap)}")
        if canonical_sha256(sorted(seeds)) != item.get("geometry_seed_set_sha256"):
            raise ValueError(f"teacher source geometry-seed audit changed: {relative}")
        used[str(relative)] = sorted(seeds)
    return used


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_protocol(path: Path, *, verify_sources: bool = True) -> dict[str, Any]:
    path = Path(path).resolve()
    protocol = json.loads(path.read_text())
    if not isinstance(protocol, dict) or protocol.get("format") != STUDY_FORMAT:
        raise ValueError("unsupported RLPD study protocol format")
    if protocol.get("status") != "frozen":
        raise ValueError("RLPD study protocol is not frozen")
    if protocol.get("frame_skip") != 4 or protocol.get("max_steps") != 2000:
        raise ValueError("study must preserve the frozen frame_skip=4/max_steps=2000 contract")
    if protocol.get("training_track_ids") != [1, 2, 3, 4]:
        raise ValueError("training IDs must match the declared HAIC study contract")
    training_seeds = protocol.get("training_geometry_seeds")
    if (
        not isinstance(training_seeds, list)
        or not training_seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in training_seeds)
        or len(set(training_seeds)) != len(training_seeds)
    ):
        raise ValueError("training geometry seeds must be unique uint32 integers")
    data_cells = protocol.get("teacher_data_cells")
    if not isinstance(data_cells, list) or not data_cells:
        raise ValueError("protocol must freeze teacher_data_cells")
    if any(
        not isinstance(cell, dict)
        or cell.get("track_id") not in protocol["training_track_ids"]
        or cell.get("geometry_seed") not in training_seeds
        or cell.get("obstacles") is not True
        for cell in data_cells
    ):
        raise ValueError("teacher data cells must use only frozen training IDs/seeds with obstacles")
    data_seeds = [cell["geometry_seed"] for cell in data_cells]
    if len(set(data_seeds)) != len(data_seeds):
        raise ValueError("teacher collection geometry seeds must be unique")
    budget = protocol.get("teacher_data_budget", {})
    if (
        type(budget.get("decisions")) is not int
        or budget["decisions"] not in {8192, 16384}
        or type(budget.get("minimum_distinct_finishes")) is not int
        or budget["minimum_distinct_finishes"] <= 0
    ):
        raise ValueError("teacher-data decision/finish gate must use a fixed supported budget")
    student = protocol.get("student_training", {})
    learner = student.get("learner", {})
    candidate_steps = student.get(
        "candidate_steps",
        protocol.get("evaluation", {}).get("screen_candidate_steps"),
    )
    if (
        type(student.get("steps_per_run")) is not int
        or student["steps_per_run"] <= 1000
        or not isinstance(student.get("learner_seeds"), list)
        or not student["learner_seeds"]
        or any(type(seed) is not int or seed < 0 for seed in student["learner_seeds"])
        or len(set(student["learner_seeds"])) != len(student["learner_seeds"])
        or student.get("first_update_step") != 1000
        or student.get("random_decisions") != 1000
        or student.get("policy_takeover_step") != 2000
        or student.get("updates_per_decision") != 1
        or student.get("batch_size") != 64
        or student.get("offline_batch_size") != 32
        or student.get("online_batch_size") != 32
        or student.get("backup_entropy") is not False
        or student.get("expected_gradient_steps_per_run") != student["steps_per_run"] - 1000
        or not isinstance(candidate_steps, list)
        or not candidate_steps
        or candidate_steps[-1] != student["steps_per_run"]
        or candidate_steps != sorted(candidate_steps)
        or any(
            type(step) is not int or not 1000 < step <= student["steps_per_run"]
            for step in candidate_steps
        )
        or len(set(candidate_steps)) != len(candidate_steps)
    ):
        raise ValueError("student budget, candidate checkpoints, or warmup contract is invalid")
    expected_learner = {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "temperature_lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "num_qs": 10,
        "num_min_qs": 1,
        "target_entropy": -1.5,
        "initial_alpha": 0.1,
        "augmentation_pad": 4,
        "n_step": 1,
        "weight_decay": 0.0,
    }
    if learner != expected_learner:
        raise ValueError("RLPD learner hyperparameters differ from the frozen author-pixel contract")
    if (
        student.get("replay_capacity") != 100000
        or student.get("offline_capacity") != budget["decisions"]
    ):
        raise ValueError("replay capacities differ from the frozen memory/data budget")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError("frozen protocol requires screen, confirmation, and blind allocations")
    partition_seeds: set[int] = set()
    for name, matrix in partitions.items():
        tracks, seeds = matrix.get("track_ids"), matrix.get("seeds")
        if (
            not isinstance(tracks, list) or not tracks
            or any(type(track) is not int or track <= 0 for track in tracks)
            or len(set(tracks)) != len(tracks)
            or not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)
            or type(matrix.get("repeats")) is not int or matrix["repeats"] != 2
        ):
            raise ValueError(f"invalid frozen {name} cell matrix")
        if partition_seeds.intersection(seeds):
            raise ValueError("evaluation geometry seeds must be disjoint across partitions")
        partition_seeds.update(seeds)
        if set(seeds).intersection(training_seeds):
            raise ValueError("evaluation geometry may not be used for student/teacher training")
    future = protocol.get("future_full_reservation")
    future_seeds = set()
    future_training_seeds = []
    if future is not None:
        if not isinstance(future, dict) or future.get("status") != "reserved_not_opened; requires new full protocol and successful pilot gate":
            raise ValueError("conditional full-stage cells must be explicitly reserved and unopened")
        future_training_seeds = future.get("training_geometry_seeds")
        future_track_ids = future.get("training_track_ids")
        future_cells = future.get("teacher_data_cells")
        future_budget = future.get("teacher_data_budget")
        if (
            not isinstance(future_track_ids, list)
            or future_track_ids != [1, 2, 3, 4]
            or not isinstance(future_training_seeds, list)
            or len(future_training_seeds) != 64
            or len(set(future_training_seeds)) != 64
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in future_training_seeds)
            or set(future_training_seeds).intersection(training_seeds)
            or set(future_training_seeds).intersection(partition_seeds)
            or not isinstance(future_cells, list)
            or len(future_cells) != 64
            or future_budget != {"decisions": 16384, "minimum_distinct_finishes": 4}
        ):
            raise ValueError("future full-stage training pool/teacher budget is not uniquely reserved")
        if any(
            not isinstance(cell, dict)
            or cell.get("track_id") not in future_track_ids
            or cell.get("geometry_seed") not in future_training_seeds
            or cell.get("obstacles") is not True
            for cell in future_cells
        ) or len({cell["geometry_seed"] for cell in future_cells}) != len(future_cells):
            raise ValueError("future full teacher cells must uniquely use their training-only seed pool")
        for name in ("screen", "confirmation", "blind"):
            matrix = future.get(name)
            if not isinstance(matrix, dict):
                raise ValueError(f"full-stage {name} reservation is missing")
            tracks, seeds = matrix.get("track_ids"), matrix.get("seeds")
            if (
                not isinstance(tracks, list) or not tracks
                or len(set(tracks)) != len(tracks)
                or any(type(track) is not int or track <= 0 for track in tracks)
                or not isinstance(seeds, list) or not seeds
                or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
                or matrix.get("repeats") != 2
            ):
                raise ValueError(f"invalid future full {name} cell matrix")
            if future_seeds.intersection(seeds) or partition_seeds.intersection(seeds):
                raise ValueError("pilot and reserved full evaluation geometry must all be disjoint")
            if set(seeds).intersection(training_seeds) or set(seeds).intersection(future_training_seeds):
                raise ValueError("reserved full evaluation geometry may not be used for training")
            future_seeds.update(seeds)
    if protocol.get("reserved_training_seeds") is None:
        raise ValueError("protocol must freeze cross-track training exclusions")
    reserved = protocol["reserved_training_seeds"]
    if (
        not isinstance(reserved, list)
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in reserved)
        or len(set(reserved)) != len(reserved)
        or not partition_seeds.issubset(reserved)
        or not future_seeds.issubset(reserved)
        or set(training_seeds).intersection(reserved)
        or set(future_training_seeds).intersection(reserved)
    ):
        raise ValueError("reserved_training_seeds must include evaluation and exclude training geometry")
    teacher = protocol.get("teacher", {})
    if (
        teacher.get("actor_sha256") != "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954"
        or teacher.get("checkpoint_sha256") != "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4"
    ):
        raise ValueError("teacher identity differs from the predeclared pad-4 seed-1 actor")
    if verify_sources:
        source_hashes = protocol.get("source_hashes")
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError("protocol must freeze source hashes before collection/training")
        for relative, expected in source_hashes.items():
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)
            ):
                raise ValueError(f"source SHA-256 is malformed: {relative}")
            source = (ROOT / relative).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file():
                raise ValueError(f"invalid source provenance path: {relative}")
            if sha256_file(source) != expected:
                raise ValueError(f"study source changed after freeze: {relative}")
    source_geometry = {}
    if verify_sources:
        geometry_audit = protocol.get("geometry_audit", {})
        token_audit = geometry_audit.get("candidate_seed_token_audit", {})
        if token_audit.get("known_recorded_uses") != []:
            raise ValueError("candidate geometry allocation has a recorded artifact collision")
        candidate_seeds = sorted(set(
            training_seeds
            + future_training_seeds
            + list(partition_seeds)
            + list(future_seeds)
        ))
        if (
            token_audit.get("candidate_seed_count") != len(candidate_seeds)
            or token_audit.get("candidate_seeds_sha256") != canonical_sha256(candidate_seeds)
            or token_audit.get("searched_artifacts_sha256") is None
        ):
            raise ValueError("candidate geometry token audit does not match frozen cells")
        source_geometry = verify_teacher_ledgers(protocol)
    protocol["_teacher_source_geometry"] = source_geometry
    protocol["_path"] = str(path)
    return protocol


def snapshot_sources(protocol: dict[str, Any], destination: Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in protocol["source_hashes"].items():
        source = ROOT / relative
        if sha256_file(source) != expected:
            raise ValueError(f"source changed during run setup: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _dataset_manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "dataset_sha256"}


def verify_dataset(
    dataset_dir: Path,
    protocol: dict[str, Any],
    *,
    require_eligible: bool = True,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != DATASET_FORMAT:
        raise ValueError("unsupported teacher dataset format")
    if manifest.get("teacher_actor_sha256") != protocol["teacher"]["actor_sha256"]:
        raise ValueError("offline dataset teacher hash does not match the frozen protocol")
    if manifest.get("study_protocol_sha256") != sha256_file(protocol["_path"]):
        raise ValueError("offline dataset protocol hash does not match the study")
    if manifest.get("decisions_spent", 2**63) > protocol["teacher_data_budget"]["decisions"]:
        raise ValueError("offline collection exceeded its frozen decision budget")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("offline dataset manifest is missing its episode list")
    expected_seeds = set(protocol["training_geometry_seeds"])
    geometry = set()
    finish_geometry = set()
    used_decisions = 0
    episode_ids = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        relative = episode.get("path")
        if type(episode_id) is not int or episode_id in episode_ids:
            raise ValueError("offline dataset episode IDs are invalid or duplicated")
        episode_ids.add(episode_id)
        if not isinstance(relative, str):
            raise ValueError("offline episode path must be a string")
        path = (dataset_dir / relative).resolve()
        if not path.is_relative_to(dataset_dir) or not path.is_file():
            raise ValueError("offline episode path escapes its immutable dataset")
        if sha256_file(path) != episode.get("sha256"):
            raise ValueError("offline episode content hash mismatch")
        if episode.get("track_id") not in protocol["training_track_ids"]:
            raise ValueError("offline episode used a non-training track ID")
        seed = episode.get("geometry_seed")
        if seed not in expected_seeds:
            raise ValueError("offline episode used an undeclared training geometry")
        if seed in set(protocol["reserved_training_seeds"]):
            raise ValueError("offline episode used a reserved evaluation geometry")
        pair = (episode["track_id"], seed)
        if pair in geometry:
            raise ValueError("offline dataset repeats a teacher geometry")
        geometry.add(pair)
        used_decisions += int(episode.get("steps", 0))
        if episode.get("finished") is True:
            finish_geometry.add(pair)
    if used_decisions != manifest.get("stored_decisions"):
        raise ValueError("offline dataset stored-decision count is inconsistent")
    spent = manifest.get("decisions_spent")
    if (
        type(spent) is not int
        or not used_decisions <= spent <= protocol["teacher_data_budget"]["decisions"]
        or manifest.get("discarded_decisions") != spent - used_decisions
        or manifest.get("distinct_finish_geometries") != len(finish_geometry)
    ):
        raise ValueError("offline dataset decision/finish accounting is inconsistent")
    if manifest.get("dataset_sha256") != canonical_sha256(_dataset_manifest_payload(manifest)):
        raise ValueError("offline dataset manifest hash mismatch")
    minimum = protocol["teacher_data_budget"]["minimum_distinct_finishes"]
    if require_eligible and len(finish_geometry) < minimum:
        raise ValueError("teacher dataset is ineligible: too few distinct-geometry finishes")
    manifest["verified_finish_geometries"] = len(finish_geometry)
    manifest["verified_stored_decisions"] = used_decisions
    return manifest


def _reconstruct_observation(frames: np.ndarray, index: int) -> np.ndarray:
    start = max(0, index - 3)
    recent = [frames[position] for position in range(start, index + 1)]
    recent = [recent[0]] * (4 - len(recent)) + recent
    return np.stack(recent).astype(np.uint8, copy=False)


def load_offline_replay(
    dataset_dir: Path,
    protocol: dict[str, Any],
    *,
    seed: int,
    require_eligible: bool = True,
) -> tuple[FrameStackReplay, str, dict[str, Any]]:
    manifest = verify_dataset(dataset_dir, protocol, require_eligible=require_eligible)
    capacity = protocol["teacher_data_budget"]["decisions"]
    replay = FrameStackReplay(capacity, seed=seed, source="offline")
    for metadata in manifest["episodes"]:
        path = Path(dataset_dir) / metadata["path"]
        with np.load(path, allow_pickle=False) as episode:
            frames = np.asarray(episode["frames"])
            initial = np.asarray(episode["initial_observation"])
            final = np.asarray(episode["final_observation"])
            if (
                frames.ndim != 3 or frames.shape[1:] != (84, 84) or frames.dtype != np.uint8
                or initial.shape != (4, 84, 84) or initial.dtype != np.uint8
                or final.shape != (4, 84, 84) or final.dtype != np.uint8
            ):
                raise ValueError("offline episode pixel arrays violate the frozen observation contract")
            length = len(frames)
            fields = {
                name: np.asarray(episode[name])
                for name in (
                    "proposed_actions", "executed_actions", "applied_actions", "rewards",
                    "terminated", "truncated", "terminal",
                )
            }
            if any(len(value) != length for value in fields.values()) or length != metadata["steps"]:
                raise ValueError("offline episode transition arrays have inconsistent lengths")
            if not np.array_equal(initial, np.repeat(frames[0:1], 4, axis=0)):
                raise ValueError("reset observation did not follow first-frame stack padding")
            for step in range(length):
                observation = initial if step == 0 else _reconstruct_observation(frames, step)
                next_observation = (
                    _reconstruct_observation(frames, step + 1)
                    if step + 1 < length else final
                )
                replay.add(PixelTransition(
                    observation=observation,
                    proposed_action=fields["proposed_actions"][step],
                    executed_action=fields["executed_actions"][step],
                    applied_action=fields["applied_actions"][step],
                    reward=float(fields["rewards"][step]),
                    next_observation=next_observation,
                    terminated=bool(fields["terminated"][step]),
                    truncated=bool(fields["truncated"][step]),
                    terminal=bool(fields["terminal"][step]),
                    episode_id=metadata["episode_id"],
                    step=step,
                    track_id=metadata["track_id"],
                    geometry_seed=metadata["geometry_seed"],
                ))
    replay.finalize()
    if replay.valid_count != len(replay):
        raise ValueError("offline replay includes transitions without terminal-safe successors")
    return replay, manifest["dataset_sha256"], manifest
