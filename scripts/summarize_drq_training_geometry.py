"""Summarize only complete, sealed training-geometry diagnostic shards.

Run from the repository root with ``python -m scripts.summarize_drq_training_geometry``.
This reads no road, environment, evaluation or blind outcome artifact. Blind seed
IDs in the frozen protocol are used solely to exclude non-training cells.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

from scripts import diagnose_drq_training_geometry as diagnostic


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
CATALOG_SHA256 = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
PROTOCOL_PATH = "experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json"
PROTOCOL_SHA256 = "5935664e132603de62cd35601fda12b998ea2009548332736720bab905d2c3f3"
DIAGNOSTICS_ROOT = "runs/20260925-drqv2-geometry-augmentation-v1/diagnostics"
SHARD_COUNT = 4
FORMAT = "haic-drq-training-geometry-diagnostic-summary-v1"
HIGH_PROGRESS = 0.9
MEANINGFUL_PROGRESS = 0.2
OFFTRACK_SEMANTICS = (
    "consecutive wrapper decisions with aggregated raw reward < 0; resets on nonnegative reward; "
    "retires when >100, not a geometric road boundary"
)
COLLISION_SEMANTICS = "one Boolean per wrapper decision; any collision among up to 4 raw frames"
TRACE_DTYPES = {
    "reward": "float32", "progress": "float32", "damage": "float32",
    "native_action": "float32", "official_action": "float32",
    "collision_action": "bool", "off_track_counter": "int16",
    "sparse_step": "int32", "sparse_nearest_point_index": "int32",
    "sparse_nearest_point_fraction": "float32", "sparse_nearest_distance_m": "float32",
    "sparse_speed_m_s": "float32",
}
CELL_KEYS = frozenset({
    "format", "role", "ranked", "partition", "geometry_seed", "track_id", "geometry_id",
    "cell_id", "source_learner_seed", "source_actor_sha256", "source_actor_weights_sha256",
    "road_coordinate_sha256", "signature_sha256", "family", "stage", "steps", "max_steps",
    "raw_reward_sum", "finished", "terminal_progress", "max_progress", "terminal_damage",
    "max_damage", "collision_actions", "collision_semantics", "terminated", "truncated",
    "terminal", "retire_reason", "termination_class", "timeout_at_max_steps",
    "off_track_counter_final", "off_track_counter_max", "off_track_semantics",
    "official_action_axes", "native_action_axes", "steering_throttle_time_bins",
    "sparse_road_samples", "first_road_sample", "last_road_sample", "finish_line",
    "lap_time_ms", "trace_path", "trace_sha256", "catalog_sha256", "protocol_sha256",
})
MANIFEST_KEYS = frozenset({
    "format", "role", "ranked", "score_selection", "performance_scope", "created_at_utc",
    "catalog_path", "catalog_sha256", "catalog_protocol_sha256", "analysis_sha256",
    "scan_path", "scan_sha256", "protocol_path", "protocol_sha256", "source_actors",
    "shard_index", "shard_count", "geometry_ids", "cell_count", "frame_skip",
    "max_steps", "sparse_stride", "schema", "file_sha256",
})


def _same(actual: Any, expected: Any, label: str) -> None:
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{label} differs from the frozen training diagnostic")


def _number(value: Any, label: str, *, low: float = -math.inf, high: float = math.inf) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be finite and in [{low}, {high}]")
    return float(value)


def _close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(_number(actual, label), expected, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{label} differs from the sealed trace")


def _source_actors(protocol: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    sources = protocol.get("source_actors")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("exactly two source actors are required")
    by_seed = {}
    for source in sources:
        if not isinstance(source, dict) or set(source) != {
            "learner_seed", "actor_path", "actor_sha256", "actor_weights_sha256",
        }:
            raise ValueError("source actor provenance is incomplete")
        seed = diagnostic._integer(source["learner_seed"], "source seed", maximum=1)
        if seed in by_seed or (source["actor_sha256"], source["actor_weights_sha256"]) != diagnostic.SOURCE_HASHES[seed]:
            raise ValueError("source actor provenance differs from frozen pad-4 sources")
        path = diagnostic._repo_path(root, source["actor_path"], "runs")
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["actor_sha256"]:
            raise ValueError("source actor export changed")
        by_seed[seed] = source
    if set(by_seed) != {0, 1}:
        raise ValueError("source actor pair is incomplete")
    return [by_seed[seed] for seed in (0, 1)]


def _static_sanity(rows: list[dict[str, Any]]) -> None:
    # Nearby official tiles may overlap at valid tight turns; it is a warning, not a veto.
    for row in rows:
        structure, verification = row["structure"], row["verification"]
        if (structure.get("centerline_connected") is not True
                or type(structure.get("centerline_self_intersections")) is not int
                or structure["centerline_self_intersections"] != 0
                or structure.get("finish_static_only") is not True
                or any(verification.get(key) is not True for key in (
                    "regenerated_coordinate_hash_match", "raw_reset", "one_valid_raw_step",
                    "fresh_tracker_after_reset",
                ))
                or verification.get("finish_plausibility") != "static_only_not_a_successful_lap"):
            raise ValueError(f"catalog static/sanity invalid for training geometry {row['geometry_seed']}")


def _trace(shard_dir: Path, row: dict[str, Any]) -> None:
    path = shard_dir / row["trace_path"]
    steps = row["steps"]
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(TRACE_DTYPES) or len(archive.files) != len(TRACE_DTYPES):
                raise ValueError("trace has missing/extra/duplicate arrays")
            arrays = {name: archive[name] for name in TRACE_DTYPES}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid sealed trace: {path}") from error
    if arrays["sparse_step"].ndim != 1:
        raise ValueError("trace sparse_step must be a vector")
    for name, dtype in TRACE_DTYPES.items():
        array = arrays[name]
        shape = ((steps, 3) if name.endswith("_action") and name != "collision_action"
                 else (len(arrays["sparse_step"]),) if name.startswith("sparse_") else (steps,))
        if array.shape != shape or array.dtype != np.dtype(dtype) or not np.isfinite(array).all():
            raise ValueError(f"trace {name} has invalid dtype, shape or values")
    reward, progress, damage = (arrays[key] for key in ("reward", "progress", "damage"))
    if (np.any((progress < 0) | (progress > 1)) or np.any(damage < 0)
            or np.any(arrays["off_track_counter"] < 0)):
        raise ValueError("trace progress, damage or negative-reward counter is invalid")
    counter = 0
    for value, observed in zip(reward, arrays["off_track_counter"]):
        counter = counter + 1 if value < 0 else 0
        if counter != observed or counter > 101:
            raise ValueError("off_track_counter is not consecutive negative-reward decisions")
    _close(row["raw_reward_sum"], float(np.sum(reward, dtype=np.float64)), "raw reward sum")
    for key, expected in (
        ("terminal_progress", progress[-1]), ("max_progress", np.max(progress)),
        ("terminal_damage", damage[-1]), ("max_damage", np.max(damage)),
    ):
        _close(row[key], float(expected), key)
    _same(row["collision_actions"], int(arrays["collision_action"].sum()), "collision actions")
    _same(row["off_track_counter_final"], int(arrays["off_track_counter"][-1]), "off-track final")
    _same(row["off_track_counter_max"], int(arrays["off_track_counter"].max()), "off-track max")
    if row["termination_class"] == "off_track" and counter != 101:
        raise ValueError("off-track retirement without 101 consecutive negative-reward decisions")
    for name, low, high in (("official_action", -1, 1), ("native_action", -1, 1)):
        action = arrays[name]
        if np.any((action < low) | (action > high)) or (name == "official_action" and np.any(action[:, 1:] < 0)):
            raise ValueError("trace action axes are outside the execution contract")
        axes = row[f"{name}_axes"]
        if not isinstance(axes, dict) or set(axes) != {"steer", "gas", "brake"}:
            raise ValueError("action axis summary is incomplete")
        for index, axis in enumerate(("steer", "gas", "brake")):
            if not isinstance(axes[axis], dict) or set(axes[axis]) != {
                "mean", "std", "min", "max", "near_boundary_fraction",
            }:
                raise ValueError("action axis summary is incomplete")
            for stat, expected in (
                ("mean", action[:, index].mean()), ("std", action[:, index].std()),
                ("min", action[:, index].min()), ("max", action[:, index].max()),
            ):
                _close(axes[axis][stat], float(expected), f"{name}.{axis}.{stat}")
            _number(axes[axis]["near_boundary_fraction"], "action boundary fraction", low=0, high=1)
    samples = row["sparse_road_samples"]
    sparse_steps = arrays["sparse_step"]
    if (not isinstance(samples, list) or len(samples) != len(sparse_steps) or len(samples) < 2
            or sparse_steps[0] != 0 or sparse_steps[-1] != steps
            or np.any(np.diff(sparse_steps) <= 0)):
        raise ValueError("sparse road locations lack reset/terminal observations")
    _same(row["first_road_sample"], samples[0], "first sparse road location")
    _same(row["last_road_sample"], samples[-1], "last sparse road location")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {
            "step", "nearest_point_index", "nearest_point_fraction", "nearest_distance_m", "speed_m_s",
        }:
            raise ValueError("sparse road location is incomplete")
        _same(sample["step"], int(sparse_steps[index]), "sparse step")
        _same(sample["nearest_point_index"], int(arrays["sparse_nearest_point_index"][index]),
              "sparse nearest point index")
        for key, low, high in (
            ("nearest_point_fraction", 0, 1), ("nearest_distance_m", 0, math.inf),
            ("speed_m_s", 0, math.inf),
        ):
            _number(sample[key], f"sparse {key}", low=low, high=high)
            _close(sample[key], float(arrays[f"sparse_{key}"][index]), f"sparse {key}")


def _cell(row: Any, catalog_row: dict[str, Any], actor: dict[str, Any],
          catalog_sha: str, protocol_sha: str, max_steps: int, shard_dir: Path) -> None:
    if not isinstance(row, dict) or set(row) != CELL_KEYS:
        raise ValueError("cell has missing/extra fields or leaked partition data")
    seed, source = catalog_row["geometry_seed"], actor["learner_seed"]
    expected = {
        "format": diagnostic.RESULT_FORMAT, "role": "training_diagnostic", "ranked": False,
        "partition": catalog_row["partition"], "geometry_seed": seed,
        "track_id": catalog_row["track_id"], "geometry_id": str(seed),
        "cell_id": f"train-geometry:{catalog_row['partition']}:seed-{seed}:track-{catalog_row['track_id']}:source-{source}",
        "source_learner_seed": source, "source_actor_sha256": actor["actor_sha256"],
        "source_actor_weights_sha256": actor["actor_weights_sha256"],
        "road_coordinate_sha256": catalog_row["road_coordinate_sha256"],
        "signature_sha256": catalog_row["signature_sha256"], "family": catalog_row["family"],
        "stage": catalog_row["stage"], "max_steps": max_steps,
        "trace_path": f"traces/seed-{seed}-source-{source}.npz",
        "catalog_sha256": catalog_sha, "protocol_sha256": protocol_sha,
        "collision_semantics": COLLISION_SEMANTICS, "off_track_semantics": OFFTRACK_SEMANTICS,
    }
    for key, value in expected.items():
        _same(row[key], value, f"cell {key}")
    diagnostic._sha(row["trace_sha256"], "cell trace hash")
    diagnostic._integer(row["steps"], "cell steps", minimum=1, maximum=max_steps)
    if (type(row["finished"]) is not bool or type(row["timeout_at_max_steps"]) is not bool
            or any(type(row[key]) is not bool for key in ("terminated", "truncated", "terminal"))
            or not (row["terminated"] or row["truncated"])
            or row["termination_class"] not in {
                "finished", "crash", "off_track", "out_of_bounds", "timeout", "truncated",
            } or row["finished"] != (row["termination_class"] == "finished")
            or row["timeout_at_max_steps"] != (row["termination_class"] == "timeout")
            or (row["timeout_at_max_steps"] and row["steps"] != max_steps)
            or row["terminal"] != (row["terminated"] or row["finished"] or row["retire_reason"] in {
                "crash", "off_track", "out_of_bounds",
            })
            or row["retire_reason"] not in (None, "crash", "off_track", "out_of_bounds")
            or (row["termination_class"] in ("crash", "off_track")
                and row["retire_reason"] != row["termination_class"])):
        raise ValueError("cell finish, termination or timeout semantics are inconsistent")
    if (not isinstance(row["finish_line"], dict)
            or type(row["finish_line"].get("crossed")) is not bool
            or type(row["finish_line"].get("qualified")) is not bool
            or row["finish_line"]["crossed"] != row["finished"]
            or row["finished"] and not row["finish_line"]["qualified"]
            or (row["lap_time_ms"] is None) == row["finished"]):
        raise ValueError("cell finish-line evidence is inconsistent")
    for flag, time in (("qualified", "qualified_time_s"), ("crossed", "crossing_time_s")):
        value = row["finish_line"].get(time)
        if (value is None) == row["finish_line"][flag]:
            raise ValueError("cell finish-line time and crossing flag differ")
        if value is not None:
            _number(value, f"finish-line {time}", low=0)
    if row["finished"]:
        _number(row["lap_time_ms"], "lap time", low=0)
    _trace(shard_dir, row)


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(sum(values) / len(values)), "min": float(min(values)), "max": float(max(values))}


def _difficulty(first: dict[str, Any], second: dict[str, Any]) -> str:
    if first["finished"] and second["finished"]:
        return "too_easy"
    if first["finished"] or second["finished"] or max(first["max_progress"], second["max_progress"]) >= HIGH_PROGRESS:
        return "useful_boundary"
    if max(first["max_progress"], second["max_progress"]) >= MEANINGFUL_PROGRESS:
        return "difficult_but_learnable"
    return "unresolved_difficult"


def _actor_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": len(rows), "finishes": sum(row["finished"] for row in rows),
        "high_progress_nonfinishes": sum(not row["finished"] and row["max_progress"] >= HIGH_PROGRESS for row in rows),
        "termination_classes": dict(sorted(Counter(row["termination_class"] for row in rows).items())),
        **{name: _stats([row[key] for row in rows]) for name, key in (
            ("terminal_progress", "terminal_progress"), ("max_progress", "max_progress"),
            ("steps", "steps"), ("terminal_damage", "terminal_damage"),
            ("max_damage", "max_damage"), ("raw_reward_sum", "raw_reward_sum"),
            ("collision_actions", "collision_actions"),
            ("off_track_counter_final", "off_track_counter_final"),
            ("off_track_counter_max", "off_track_counter_max"),
        )},
        **{f"sparse_{when}_{key}": _stats([row[f"{when}_road_sample"][key] for row in rows])
           for when in ("first", "last") for key in ("nearest_point_fraction", "nearest_distance_m")},
        "mean_of_cell_action_means": {
            channel: {axis: float(sum(row[channel][axis]["mean"] for row in rows) / len(rows))
                      for axis in ("steer", "gas", "brake")}
            for channel in ("official_action_axes", "native_action_axes")
        },
    }


def _contrast(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    return {
        "paired_geometry_count": len(pairs),
        "source0_only_finishes": sum(first["finished"] and not second["finished"] for first, second in pairs),
        "source1_only_finishes": sum(second["finished"] and not first["finished"] for first, second in pairs),
        "source1_minus_source0": {key: _stats([second[key] - first[key] for first, second in pairs])
                                   for key in ("terminal_progress", "max_progress", "steps",
                                               "terminal_damage", "collision_actions", "off_track_counter_max")},
        "source1_minus_source0_action_mean": {
            channel: {axis: _stats([second[channel][axis]["mean"] - first[channel][axis]["mean"]
                                    for first, second in pairs])
                      for axis in ("steer", "gas", "brake")}
            for channel in ("official_action_axes", "native_action_axes")
        },
        "evidence_limit": "One cell per actor per geometry; paired descriptions, not repeated trials or significance.",
    }


def summarize(*, root: Path, catalog_path: Path, catalog_sha256: str,
              protocol_path: Path, protocol_sha256: str, diagnostics_root: Path,
              output: Path) -> dict[str, Any]:
    """Validate every frozen cell before atomically claiming the new result path."""
    root = root.resolve()
    catalog_path = diagnostic._repo_path(root, catalog_path, "runs")
    protocol_path = diagnostic._repo_path(root, protocol_path, "experiments")
    diagnostics_root = diagnostic._repo_path(root, diagnostics_root, "runs")
    output = diagnostic._repo_path(root, output, "experiments")
    _same(catalog_path.relative_to(root).as_posix(), CATALOG_PATH, "catalog path")
    _same(protocol_path.relative_to(root).as_posix(), PROTOCOL_PATH, "protocol path")
    _same(diagnostics_root.relative_to(root).as_posix(), DIAGNOSTICS_ROOT, "diagnostics root")
    _same(diagnostic._sha(catalog_sha256, "catalog hash"), CATALOG_SHA256, "catalog hash")
    _same(diagnostic._sha(protocol_sha256, "protocol hash"), PROTOCOL_SHA256, "protocol hash")
    if output.exists() or output.parent != root / "experiments" or output.suffix != ".json":
        raise FileExistsError("output must be a new, top-level experiments/ JSON artifact")
    catalog = diagnostic._read_pinned_json(catalog_path, catalog_sha256, "catalog")
    protocol = diagnostic._read_pinned_json(protocol_path, protocol_sha256, "protocol")
    rows = diagnostic._catalog_rows(catalog, protocol, catalog_sha256)
    _static_sanity(rows)
    for key, expected in (
        ("catalog_path", CATALOG_PATH), ("catalog_scan_path", catalog["scan_path"]),
        ("catalog_scan_sha256", catalog["scan_sha256"]),
    ):
        _same(protocol.get(key), expected, f"protocol {key}")
    actors = _source_actors(protocol, root)
    if (not diagnostics_root.is_dir() or set(item.name for item in diagnostics_root.iterdir()) != {
            f"shard-{index:04d}-of-{SHARD_COUNT:04d}" for index in range(SHARD_COUNT)}):
        raise ValueError("all four and only four COMPLETE diagnostic shards are required")
    all_cells: dict[tuple[int, int], dict[str, Any]] = {}
    manifest_hashes = []
    for index in range(SHARD_COUNT):
        shard_dir = diagnostic._repo_path(root, diagnostics_root / f"shard-{index:04d}-of-{SHARD_COUNT:04d}", "runs")
        expected_rows = [row for offset, row in enumerate(rows) if offset % SHARD_COUNT == index]
        traces = {f"traces/seed-{row['geometry_seed']}-source-{seed}.npz"
                  for row in expected_rows for seed in (0, 1)}
        manifest_path = diagnostic._repo_path(root, shard_dir / "manifest.json", "runs")
        seal_path = diagnostic._repo_path(root, shard_dir / "manifest.sha256", "runs")
        if not manifest_path.is_file() or not seal_path.is_file():
            raise ValueError(f"shard {index} is incomplete: manifest and manifest.sha256 required")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if seal_path.read_bytes() != f"{manifest_sha}  manifest.json\n".encode("ascii"):
            raise ValueError(f"shard {index} manifest.sha256 seal is missing or changed")
        manifest = diagnostic._read_pinned_json(manifest_path, manifest_sha, "shard manifest")
        if set(manifest) != MANIFEST_KEYS:
            raise ValueError("shard manifest fields/partition differ from writer")
        expected = {
            "format": diagnostic.MANIFEST_FORMAT, "role": "training_diagnostic", "ranked": False,
            "score_selection": False, "catalog_path": CATALOG_PATH, "catalog_sha256": catalog_sha256,
            "catalog_protocol_sha256": catalog["protocol_sha256"], "analysis_sha256": catalog["analysis_sha256"],
            "scan_path": catalog["scan_path"], "scan_sha256": catalog["scan_sha256"],
            "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha256,
            "source_actors": actors, "shard_index": index, "shard_count": SHARD_COUNT,
            "geometry_ids": [row["geometry_seed"] for row in expected_rows],
            "cell_count": 2 * len(expected_rows), "frame_skip": 4,
            "max_steps": protocol["max_steps"], "sparse_stride": diagnostic.SPARSE_STRIDE,
        }
        for key, value in expected.items():
            _same(manifest[key], value, f"shard {index} {key}")
        schema = manifest["schema"]
        if (not isinstance(schema, dict) or set(schema) != {"cell", "trace_arrays"}
                or schema["cell"] != diagnostic.RESULT_FORMAT
                or not isinstance(schema["trace_arrays"], dict)
                or set(schema["trace_arrays"]) != set(TRACE_DTYPES)):
            raise ValueError("unknown diagnostic shard trace schema")
        inventory = manifest["file_sha256"]
        if not isinstance(inventory, dict) or set(inventory) != traces | {"cells.jsonl"}:
            raise ValueError("shard file inventory has missing/extra/non-training files")
        trace_dir = diagnostic._repo_path(root, shard_dir / "traces", "runs")
        if (set(item.name for item in shard_dir.iterdir()) != {
                "manifest.json", "manifest.sha256", "cells.jsonl", "traces",
            } or not trace_dir.is_dir()
                or set(item.name for item in trace_dir.iterdir()) != {Path(name).name for name in traces}):
            raise ValueError("shard contains missing or extra/unsealed files")
        for relative, digest in inventory.items():
            diagnostic._sha(digest, "inventory SHA-256")
            path = diagnostic._repo_path(root, shard_dir / relative, "runs")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"shard {index} inventory file changed: {relative}")
        raw = (shard_dir / "cells.jsonl").read_text(encoding="utf-8")
        lines = raw.splitlines()
        if len(lines) != 2 * len(expected_rows) or not raw.endswith("\n"):
            raise ValueError(f"shard {index} has incomplete/extra diagnostic rows")
        expected_by_seed = {row["geometry_seed"]: row for row in expected_rows}
        observed: set[tuple[int, int]] = set()
        for line in lines:
            cell = json.loads(line, object_pairs_hook=diagnostic._object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON {value}")))
            if not isinstance(cell, dict):
                raise ValueError("cell JSON row must be an object")
            seed = cell.get("geometry_seed")
            source = cell.get("source_learner_seed")
            if type(seed) is not int or seed not in expected_by_seed or type(source) is not int or source not in (0, 1):
                raise ValueError("cell seed/actor is not in its assigned training shard")
            key = (seed, source)
            if key in observed or key in all_cells:
                raise ValueError("duplicate cell ID or seed/actor coverage")
            _cell(cell, expected_by_seed[seed], actors[source], catalog_sha256, protocol_sha256,
                  protocol["max_steps"], shard_dir)
            if cell["trace_sha256"] != inventory[cell["trace_path"]]:
                raise ValueError("cell trace hash differs from shard inventory")
            observed.add(key)
            all_cells[key] = cell
        if observed != {(row["geometry_seed"], seed) for row in expected_rows for seed in (0, 1)}:
            raise ValueError("shard lacks exact two-source coverage")
        manifest_hashes.append({"shard_index": index, "manifest_sha256": manifest_sha})
    if len(all_cells) != len(rows) * 2:
        raise ValueError("catalog did not receive exactly two diagnostic cells per geometry")
    geometry = []
    families: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for catalog_row in rows:
        seed = catalog_row["geometry_seed"]
        pair = (all_cells[(seed, 0)], all_cells[(seed, 1)])
        families[catalog_row["family"]].append(pair)
        category = _difficulty(*pair)
        geometry.append({
            "geometry_seed": seed, "track_id": catalog_row["track_id"],
            "partition": catalog_row["partition"], "family": catalog_row["family"],
            "stage": catalog_row["stage"], "road_coordinate_sha256": catalog_row["road_coordinate_sha256"],
            "static_sanity": "passed_frozen_catalog", "finishes": sum(cell["finished"] for cell in pair),
            "provisional_difficulty": category,
            "finish_reachability": ("observed_both_actors" if all(cell["finished"] for cell in pair)
                                    else "observed_one_actor" if any(cell["finished"] for cell in pair)
                                    else "unknown_no_finish_observed"),
            "by_source": {str(source): {
                "cell_id": cell["cell_id"], "trace_sha256": cell["trace_sha256"],
                "finished": cell["finished"], "termination_class": cell["termination_class"],
                "terminal_progress": cell["terminal_progress"], "max_progress": cell["max_progress"],
                "steps": cell["steps"], "raw_reward_sum": cell["raw_reward_sum"],
                "terminal_damage": cell["terminal_damage"],
                "max_damage": cell["max_damage"], "collision_actions": cell["collision_actions"],
                "off_track_counter_final": cell["off_track_counter_final"],
                "off_track_counter_max": cell["off_track_counter_max"],
                "official_action_mean": {axis: cell["official_action_axes"][axis]["mean"]
                                         for axis in ("steer", "gas", "brake")},
                "native_action_mean": {axis: cell["native_action_axes"][axis]["mean"]
                                       for axis in ("steer", "gas", "brake")},
                "first_road_sample": cell["first_road_sample"],
                "last_road_sample": cell["last_road_sample"],
            } for source, cell in enumerate(pair)},
            "source_contrast": _contrast([pair]),
        })
    family_summaries = {}
    for name, pairs in sorted(families.items()):
        cells = [cell for pair in pairs for cell in pair]
        family_summaries[name] = {
            "geometry_count": len(pairs), "cell_count": len(cells),
            "partition_geometry_counts": dict(sorted(Counter(pair[0]["partition"] for pair in pairs).items())),
            "provisional_difficulty_counts": dict(sorted(Counter(_difficulty(*pair) for pair in pairs).items())),
            "by_source": {str(source): _actor_summary([pair[source] for pair in pairs]) for source in (0, 1)},
            "source_contrast": _contrast(pairs),
        }
    result = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "training_diagnostic", "ranked": False, "score_selection": False,
        "performance_scope": "Internal training-only raw CarRacing proxies; not official performance or held-out evaluation.",
        "catalog_path": CATALOG_PATH, "catalog_sha256": catalog_sha256,
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha256,
        "diagnostics_root": diagnostics_root.relative_to(root).as_posix(),
        "shard_manifests": manifest_hashes, "geometry_count": len(rows), "cell_count": len(all_cells),
        "source_actors": actors, "partition_geometry_counts": dict(sorted(Counter(
            row["partition"] for row in rows).items())),
        "difficulty_thresholds": {"high_progress_at_least": HIGH_PROGRESS,
                                  "meaningful_progress_at_least": MEANINGFUL_PROGRESS,
                                  "progress_metric": "visited tile fraction; not directed lap distance"},
        "difficulty_interpretation": {
            "too_easy": "Both actors finished once on this geometry; not replicated ease.",
            "useful_boundary": "At least one finish or high-progress cell, but not both finished.",
            "difficult_but_learnable": "No finish; at least one meaningful-progress cell.",
            "unresolved_difficult": "Neither actor reached meaningful progress; reachability unknown.",
        },
        "difficulty_counts": dict(sorted(Counter(item["provisional_difficulty"] for item in geometry).items())),
        "pathological_count": 0,
        "evidence_limits": [
            "Every catalog road retained; pathological is reserved for failed frozen static/sanity checks, which abort this summary.",
            "Only one CPU rollout per source actor and geometry; no replicated finish or significance inference.",
            "No observed finish leaves finish reachability unknown, not impossible.",
            "Sparse first/last nearest-road positions are sampled lower-bound location evidence only; not directed travel or a finish.",
            OFFTRACK_SEMANTICS,
            COLLISION_SEMANTICS,
        ],
        "geometry": geometry, "families": family_summaries,
    }
    if (hashlib.sha256(catalog_path.read_bytes()).hexdigest() != catalog_sha256
            or hashlib.sha256(protocol_path.read_bytes()).hexdigest() != protocol_sha256
            or any(hashlib.sha256((diagnostics_root / f"shard-{item['shard_index']:04d}-of-{SHARD_COUNT:04d}"
                                   / "manifest.json").read_bytes()).hexdigest() != item["manifest_sha256"]
                   for item in manifest_hashes)):
        raise ValueError("frozen input changed while summarizing")
    with output.open("xb") as destination:
        destination.write((json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new immutable experiments/ JSON file")
    args = parser.parse_args(argv)
    try:
        result = summarize(root=ROOT, catalog_path=args.catalog, catalog_sha256=args.catalog_sha256,
                           protocol_path=args.protocol, protocol_sha256=args.protocol_sha256,
                           diagnostics_root=args.diagnostics_root, output=args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "geometry_count": result["geometry_count"],
                      "cell_count": result["cell_count"], "difficulty_counts": result["difficulty_counts"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
