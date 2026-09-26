"""Frozen, non-selecting DrQ source diagnostics on cataloged training roads only.

Run from the repository root with ``python -m scripts.diagnose_drq_training_geometry``.
The protocol and catalog must be independently frozen and SHA-pinned before use.
No development/confirmation/blind partition can be passed as an execution target.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from common_adapter import ActionSpec, EpisodeCollector, ObservationSpec, file_sha256
from drq_v2 import load_exported_actor
from train import build_env


ROOT = Path(__file__).resolve().parents[1]
CATALOG_FORMAT = "haic-drq-training-geometry-catalog-v1"
PROTOCOL_FORMAT = "haic-drq-training-geometry-diagnostics-v1"
RESULT_FORMAT = "haic-drq-training-geometry-diagnostic-cell-v1"
MANIFEST_FORMAT = "haic-drq-training-geometry-diagnostic-shard-v1"
MAX_DECISIONS = 1200
SPARSE_STRIDE = 25
TRAIN_TARGET = 120
DIAGNOSTIC_TARGET = 16
AUDITED_POOL_MIN = 256
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# These are the two independently trained, already frozen pad-4 source exports.
SOURCE_HASHES = {
    0: ("433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37",
        "81a8c0785e7f586481af40ae628714aec858c1c0a89db27c79e709311997ba93"),
    1: ("c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
        "e0984f709d1971754f2e9b64a69bf216922ca5bd0791ad247181b218f830b736"),
}
PARTITIONS = frozenset({"train", "train_diagnostic", "screen", "confirmation", "blind"})
EXECUTABLE_PARTITIONS = ("train", "train_diagnostic")
CATALOG_KEYS = frozenset({
    "format", "protocol_sha256", "analysis_sha256", "generated_at_utc",
    "train", "train_diagnostic", "scanned_count", "rejected_count", "symmetry",
    "seed_audit", "seed_audit_receipt", "source_sha256",
    "finish_evidence_limit", "scan_path", "scan_sha256", "unused_candidate_count",
})
CATALOG_ROW_KEYS = frozenset({
    "geometry_seed", "track_id", "family", "stage", "direction",
    "road_coordinate_sha256", "signature_sha256", "cyclic_signature_sha256", "signature", "features",
    "structure", "verification",
})


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label}: nonempty lowercase SHA-256 required")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0, maximum: int = 2**32 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label}: expected integer in [{minimum}, {maximum}]")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read_pinned_json(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    expected_sha = _sha(expected_sha, f"{label} SHA-256")
    raw = path.read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"{label} is empty or changed after freeze")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON {value}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _repo_path(root: Path, path: str | Path, directory: str) -> Path:
    """Do not follow symlinks into sealed evaluation or external directories."""
    root = root.resolve()
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError(f"{directory}/ path is required")
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        parts = candidate.relative_to(root).parts
    except ValueError as error:
        raise ValueError(f"path must be under the repository's {directory}/ directory") from error
    if len(parts) < 2 or parts[0] != directory or any(part in (".", "..") for part in parts):
        raise ValueError(f"path must be under the repository's {directory}/ directory")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink in pinned input/output path")
    return current


def _seeds(value: Any, label: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label}: explicit seed list required")
    seeds = [_integer(seed, label) for seed in value]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{label}: duplicate geometry seed")
    return seeds


def _catalog_rows(catalog: dict[str, Any], protocol: dict[str, Any],
                  catalog_sha256: str) -> list[dict[str, Any]]:
    if catalog.get("format") != CATALOG_FORMAT:
        raise ValueError("unrecognized catalog format")
    if set(catalog) - CATALOG_KEYS:
        raise ValueError("unknown catalog field/partition")
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("unrecognized diagnostic protocol format")
    if _sha(protocol.get("catalog_sha256"), "protocol catalog hash") != catalog_sha256:
        raise ValueError("diagnostic protocol does not bind this immutable catalog")
    if (_sha(protocol.get("catalog_protocol_sha256"), "catalog generation protocol hash")
            != _sha(catalog.get("protocol_sha256"), "catalog protocol hash")):
        raise ValueError("catalog generation protocol lineage differs")
    _sha(catalog.get("analysis_sha256"), "catalog analysis receipt hash")
    _sha(catalog.get("scan_sha256"), "catalog frozen scan hash")
    if not isinstance(catalog.get("scan_path"), str) or not catalog["scan_path"].startswith("runs/"):
        raise ValueError("catalog does not bind a training-only structural scan")
    if not isinstance(catalog.get("seed_audit"), dict) or catalog["seed_audit"].get("passed") is not True:
        raise ValueError("catalog lacks a passing pre-generation training seed audit")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) - PARTITIONS:
        raise ValueError("unknown protocol partition")
    if not all(name in partitions for name in (*EXECUTABLE_PARTITIONS, "blind")):
        raise ValueError("training partitions and blind exclusion IDs are mandatory")
    if protocol.get("role") != "training_diagnostic" or protocol.get("score_selection") is not False:
        raise ValueError("protocol must prohibit model and road score selection")
    if protocol.get("frame_skip") != 4 or protocol.get("raw_reward") is not True or protocol.get("obstacles") is not True:
        raise ValueError("diagnostic must use raw reward, official obstacles and frame_skip=4")
    _integer(protocol.get("max_steps"), "max_steps", minimum=1, maximum=MAX_DECISIONS)

    partition_seeds = {}
    for name, value in partitions.items():
        if not isinstance(value, dict) or set(value) != {"seeds"}:
            raise ValueError(f"partitions.{name} must contain only seed IDs")
        partition_seeds[name] = _seeds(value["seeds"], f"partitions.{name}")
    reserved = set(_seeds(protocol.get("reserved_training_seeds"), "reserved_training_seeds"))
    if not set(partition_seeds["blind"]).issubset(reserved):
        raise ValueError("blind seeds missing from global reserved exclusions")
    excluded = reserved | set().union(*(set(ids) for name, ids in partition_seeds.items()
                                       if name not in EXECUTABLE_PARTITIONS))

    rows: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    seen_roads: set[str] = set()
    for partition in EXECUTABLE_PARTITIONS:
        entries = catalog.get(partition)
        target = TRAIN_TARGET if partition == "train" else DIAGNOSTIC_TARGET
        if not isinstance(entries, list) or len(entries) != target:
            raise ValueError(f"catalog {partition} must contain {target} training-only rows")
        expected = set(partition_seeds[partition])
        actual: set[int] = set()
        for row in entries:
            if not isinstance(row, dict) or set(row) != CATALOG_ROW_KEYS:
                raise ValueError("catalog row has missing or unknown geometry fields/partition")
            seed = _integer(row.get("geometry_seed"), "catalog geometry_seed")
            track_id = _integer(row.get("track_id"), "catalog track_id", minimum=1)
            road_sha = _sha(row.get("road_coordinate_sha256"), "road coordinate hash")
            _sha(row.get("signature_sha256"), "road signature hash")
            _sha(row.get("cyclic_signature_sha256"), "cyclic road signature hash")
            if (seed in seen_seeds or seed in excluded or road_sha in seen_roads
                    or seed not in expected):
                raise ValueError("duplicate/unknown/reserved/blind geometry in catalog")
            if track_id != 1 or not isinstance(row.get("family"), str) or not row["family"]:
                raise ValueError("catalog track ID/family is not frozen official training geometry")
            if row.get("stage") not in (("representative", "variant") if partition == "train"
                                        else ("diagnostic",)):
                raise ValueError("unknown catalog stage/partition")
            for key in ("signature", "features", "structure", "verification"):
                if not isinstance(row.get(key), dict) or not row[key]:
                    raise ValueError(f"catalog row lacks {key}")
            signature_sha = hashlib.sha256(json.dumps(
                row["signature"], sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if (row["signature_sha256"] != signature_sha
                    or row["features"].get("signature_sha256") != signature_sha
                    or row["features"].get("signature") != row["signature"]
                    or row["verification"].get("regenerated_coordinate_hash_match") is not True):
                raise ValueError("catalog road signature or static regeneration changed")
            actual.add(seed)
            seen_seeds.add(seed)
            seen_roads.add(road_sha)
            rows.append({**row, "partition": partition})
        if actual != expected:
            raise ValueError(f"catalog {partition} differs from frozen protocol seed IDs")
    audit = catalog["seed_audit"]
    if (audit.get("format") != "haic-drq-training-seed-audit-v1"
            or audit.get("matched_collisions") != [] or audit.get("blind_data_access") !=
            "none; protocol blind seed IDs are exclusion-only"):
        raise ValueError("catalog seed audit is not blind-safe and collision-free")
    proposed = _seeds(audit.get("proposed_seeds"), "catalog seed audit")
    if len(proposed) < AUDITED_POOL_MIN or not seen_seeds.issubset(proposed) or set(proposed) & excluded:
        raise ValueError("selected road IDs are not exclusively audited training seeds")
    digest = hashlib.sha256(json.dumps(proposed, separators=(",", ":")).encode("ascii")).hexdigest()
    if _sha(audit.get("proposed_seeds_sha256"), "audit seed-list hash") != digest:
        raise ValueError("catalog seed audit has changed")
    return sorted(rows, key=lambda row: row["geometry_seed"])


def _weights_sha256(actor: Any) -> str:
    digest = hashlib.sha256()
    state = actor.state_dict()
    if not state:
        raise ValueError("source actor has no weights")
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_sources(protocol: dict[str, Any], root: Path) -> dict[int, Any]:
    sources = protocol.get("source_actors")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("exactly two frozen pad-4 source actors required")
    loaded = {}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("source actor entry is invalid")
        learner_seed = _integer(source.get("learner_seed"), "source learner seed", maximum=1)
        expected_file, expected_weights = SOURCE_HASHES[learner_seed]
        actor_sha = _sha(source.get("actor_sha256"), "source actor SHA-256")
        weights_sha = _sha(source.get("actor_weights_sha256"), "source actor weights SHA-256")
        if learner_seed in loaded or (actor_sha, weights_sha) != (expected_file, expected_weights):
            raise ValueError("source actor IDs/weights differ from frozen pad-4 sources")
        actor_path = _repo_path(root, source.get("actor_path"), "runs")
        if not actor_path.is_file() or file_sha256(actor_path) != actor_sha:
            raise ValueError("source actor export changed after freeze")
        # The exported config is part of the pinned bytes; it is not a trainer load.
        payload = torch.load(actor_path, map_location="cpu", weights_only=False)
        if (payload.get("format") != "haic-drq-v2-actor-v1"
                or payload.get("config", {}).get("augmentation_pad") != 4
                or payload["config"].get("steering_logit_l2", 0.0) != 0.0):
            raise ValueError("source export is not the original pad-4 control")
        first, action_adapter, observation_spec = load_exported_actor(actor_path, device="cpu")
        second, reload_adapter, reload_spec = load_exported_actor(actor_path, device="cpu")
        if (action_adapter.spec.fingerprint != ActionSpec().fingerprint
                or reload_adapter.spec.fingerprint != ActionSpec().fingerprint
                or observation_spec.fingerprint != ObservationSpec().fingerprint
                or reload_spec.fingerprint != ObservationSpec().fingerprint
                or _weights_sha256(first) != weights_sha or _weights_sha256(second) != weights_sha):
            raise ValueError("source actor reload/weights/action contract differs")
        for fill in (0.0, 0.5, 1.0):
            observation = np.full((4, 84, 84), fill, dtype=np.float32)
            with torch.inference_mode():
                a = first.act(observation, deterministic=True)
                b = second.act(observation, deterministic=True)
            if not np.array_equal(a, b) or np.asarray(a).shape != (3,) or not np.isfinite(a).all():
                raise ValueError("source actor CPU reload is nondeterministic")
        loaded[learner_seed] = (first, action_adapter, observation_spec, actor_path)
    if set(loaded) != {0, 1}:
        raise ValueError("source actor learner-seed pair is incomplete")
    return loaded


def _road_sample(env: Any, step: int) -> dict[str, Any]:
    raw = env.unwrapped
    track = np.asarray(raw.track, dtype=np.float64)
    if track.ndim != 2 or track.shape[1] != 4 or len(track) < 20 or not np.isfinite(track).all():
        raise ValueError("raw official road has no finite centerline")
    points = track[:, 2:4]
    position = np.asarray(raw.car.hull.position, dtype=np.float64)
    velocity = np.asarray(raw.car.hull.linearVelocity, dtype=np.float64)
    if position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("car pose/speed is not finite")
    distances = np.linalg.norm(points - position, axis=1)
    index = int(np.argmin(distances))
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    total = float(lengths.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("raw road perimeter is invalid")
    return {
        "step": step,
        "nearest_point_index": index,
        "nearest_point_fraction": float(lengths[:index].sum() / total),
        "nearest_distance_m": float(distances[index]),
        "speed_m_s": float(np.linalg.norm(velocity)),
    }


def _road_sha256(track: Any) -> str:
    rows = np.asarray(track, dtype="<f8")
    if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
        raise ValueError("raw road coordinates are invalid")
    return hashlib.sha256(np.ascontiguousarray(rows[:, 2:4]).tobytes()).hexdigest()


def _offtrack_wrapper(env: Any) -> Any:
    current = env
    for _ in range(12):
        if hasattr(current, "off_track_counter") and hasattr(current, "max_off_track_steps"):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise ValueError("original off-track wrapper counter is inaccessible")


def _axis_summary(actions: np.ndarray, *, official: bool) -> dict[str, Any]:
    return {
        axis: {
            "mean": float(actions[:, index].mean()),
            "std": float(actions[:, index].std()),
            "min": float(actions[:, index].min()),
            "max": float(actions[:, index].max()),
            "near_boundary_fraction": (
                float(np.mean((actions[:, index] <= 0.01) | (actions[:, index] >= 0.99)))
                if official and index > 0 else float(np.mean(np.abs(actions[:, index]) >= 0.99))
            ),
        }
        for index, axis in enumerate(("steer", "gas", "brake"))
    }


def _temporal_bins(actions: np.ndarray) -> list[dict[str, Any]]:
    bins = []
    for quarter in range(4):
        start = len(actions) * quarter // 4
        stop = len(actions) * (quarter + 1) // 4
        sample = actions[start:stop]
        bins.append({
            "start_step": start,
            "end_step_exclusive": stop,
            "mean_steer": float(sample[:, 0].mean()) if len(sample) else None,
            "mean_abs_steer": float(np.abs(sample[:, 0]).mean()) if len(sample) else None,
            "steering_near_boundary_fraction": float(np.mean(np.abs(sample[:, 0]) >= 0.99)) if len(sample) else None,
            "mean_gas": float(sample[:, 1].mean()) if len(sample) else None,
            "gas_under_fifth_fraction": float(np.mean(sample[:, 1] <= 0.2)) if len(sample) else None,
            "gas_over_half_fraction": float(np.mean(sample[:, 1] > 0.5)) if len(sample) else None,
        })
    return bins


def _finish_state(env: Any, info: dict[str, Any]) -> dict[str, Any]:
    tracker = getattr(env.unwrapped, "finish_line_tracker", None)
    finished = bool(info.get("finished", False))
    qualified = bool(info.get("finish_qualified", False))
    finish_time = info.get("finish_time_s")
    qualification_time = info.get("finish_qualified_time_s")
    if finished and (not qualified or finish_time is None):
        raise ValueError("finish crossing is inconsistent with qualification")
    if tracker is None:
        raise ValueError("official finish-line tracker is missing")
    return {
        "qualified": qualified,
        "qualified_time_s": float(qualification_time) if qualification_time is not None else None,
        "crossed": finished,
        "crossing_time_s": float(finish_time) if finish_time is not None else None,
        "departed_start_area": bool(tracker.departed_start_area),
        "crossing_from_back": bool(tracker.crossing_from_back),
        "candidate_crossing_time_s": (
            float(tracker.candidate_crossing_time_s)
            if tracker.candidate_crossing_time_s is not None else None
        ),
        "last_longitudinal_m": float(tracker.previous_longitudinal),
        "last_lateral_m": float(tracker.previous_lateral),
    }


def diagnose_cell(row: dict[str, Any], learner_seed: int, loaded: tuple[Any, ...],
                  max_steps: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    actor, adapter, spec, _ = loaded
    seed, track_id = row["geometry_seed"], row["track_id"]
    env = build_env(track_id=track_id, seed=seed, max_steps=max_steps, frame_skip=4,
                    reward_shaping=False, obstacles=True, collision_penalty=0.0)
    collector = EpisodeCollector(env, action_adapter=adapter, observation_spec=spec)
    try:
        observation, reset_info = collector.reset()
        if (reset_info.get("seed") != seed or reset_info.get("track_id") != track_id
                or getattr(env.unwrapped, "track_seed", seed) != seed):
            raise ValueError("reset selected a seed/track outside the frozen catalog")
        if _road_sha256(env.unwrapped.track) != row["road_coordinate_sha256"]:
            raise ValueError("reset road coordinates differ from frozen training catalog")
        wrapper = _offtrack_wrapper(env)
        if wrapper.max_off_track_steps != 100:
            raise ValueError("original 100-negative-decision off-track contract changed")
        sparse = [_road_sample(env, 0)]
        start_time_s = float(env.unwrapped.t)
        rewards, progress, damage = [], [], []
        native_actions, official_actions = [], []
        collisions, offtrack = [], []
        terminal = None
        with torch.inference_mode():
            for step in range(1, max_steps + 1):
                action = actor.act(observation, deterministic=True)
                transition = collector.step(action)
                if (transition.info.get("seed") != seed or transition.info.get("track_id") != track_id):
                    raise ValueError("transition returned a seed/track outside the catalog")
                native_actions.append(transition.action.copy())
                official_actions.append(transition.applied_action.copy())
                rewards.append(float(transition.reward))
                progress.append(float(transition.info["progress"]))
                damage.append(float(transition.info["damage"]))
                collisions.append(bool(transition.info["collision"]))
                offtrack.append(int(wrapper.off_track_counter))
                if step == 1 or step % SPARSE_STRIDE == 0 or transition.done or step == max_steps:
                    if sparse[-1]["step"] != step:
                        sparse.append(_road_sample(env, step))
                terminal = transition
                if transition.done:
                    break
                observation = transition.next_observation
        assert terminal is not None
        native = np.asarray(native_actions, dtype=np.float32)
        official = np.asarray(official_actions, dtype=np.float32)
        arrays = {
            "reward": np.asarray(rewards, dtype=np.float32),
            "progress": np.asarray(progress, dtype=np.float32),
            "damage": np.asarray(damage, dtype=np.float32),
            "native_action": native,
            "official_action": official,
            "collision_action": np.asarray(collisions, dtype=np.bool_),
            "off_track_counter": np.asarray(offtrack, dtype=np.int16),
            "sparse_step": np.asarray([item["step"] for item in sparse], dtype=np.int32),
            "sparse_nearest_point_index": np.asarray([item["nearest_point_index"] for item in sparse], dtype=np.int32),
            "sparse_nearest_point_fraction": np.asarray([item["nearest_point_fraction"] for item in sparse], dtype=np.float32),
            "sparse_nearest_distance_m": np.asarray([item["nearest_distance_m"] for item in sparse], dtype=np.float32),
            "sparse_speed_m_s": np.asarray([item["speed_m_s"] for item in sparse], dtype=np.float32),
        }
        if any(not np.isfinite(values).all() for values in arrays.values()):
            raise ValueError("non-finite cell trace")
        finished = bool(terminal.info.get("finished", False))
        retired = terminal.info.get("retire_reason")
        reason = ("finished" if finished else str(retired) if retired in ("crash", "off_track")
                  else "out_of_bounds" if terminal.terminated else "timeout" if len(rewards) == max_steps
                  else "truncated")
        if reason == "timeout" and (finished or retired in ("crash", "off_track")):
            raise ValueError("timeout cannot hide finish or retirement")
        finish = _finish_state(env, terminal.info)
        result = {
            "format": RESULT_FORMAT,
            "role": "training_diagnostic",
            "ranked": False,
            "partition": row["partition"],
            "geometry_seed": seed,
            "track_id": track_id,
            "geometry_id": str(seed),
            "cell_id": f"train-geometry:{row['partition']}:seed-{seed}:track-{track_id}:source-{learner_seed}",
            "source_learner_seed": learner_seed,
            "source_actor_sha256": SOURCE_HASHES[learner_seed][0],
            "source_actor_weights_sha256": SOURCE_HASHES[learner_seed][1],
            "road_coordinate_sha256": row["road_coordinate_sha256"],
            "signature_sha256": row["signature_sha256"],
            "family": row["family"],
            "stage": row["stage"],
            "steps": len(rewards),
            "max_steps": max_steps,
            "raw_reward_sum": float(sum(rewards)),
            "finished": finished,
            "terminal_progress": progress[-1],
            "max_progress": max(progress),
            "terminal_damage": damage[-1],
            "max_damage": max(damage),
            "collision_actions": sum(collisions),
            "collision_semantics": "one Boolean per wrapper decision; any collision among up to 4 raw frames",
            "terminated": terminal.terminated,
            "truncated": terminal.truncated,
            "terminal": terminal.terminal,
            "retire_reason": retired,
            "termination_class": reason,
            "timeout_at_max_steps": reason == "timeout",
            "off_track_counter_final": offtrack[-1],
            "off_track_counter_max": max(offtrack),
            "off_track_semantics": "consecutive wrapper decisions with aggregated raw reward < 0; resets on nonnegative reward; retires when >100, not a geometric road boundary",
            "official_action_axes": _axis_summary(official, official=True),
            "native_action_axes": _axis_summary(native, official=False),
            "steering_throttle_time_bins": _temporal_bins(official),
            "sparse_road_samples": sparse,
            "first_road_sample": sparse[0],
            "last_road_sample": sparse[-1],
            "finish_line": finish,
            "lap_time_ms": round((finish["crossing_time_s"] - start_time_s) * 1000)
            if finished else None,
        }
        return result, arrays
    finally:
        env.close()


def run_shard(*, root: Path, catalog_path: Path, catalog_sha256: str,
              protocol_path: Path, protocol_sha256: str, output_root: Path,
              shard_index: int, shard_count: int) -> dict[str, Any]:
    root = root.resolve()
    _integer(shard_count, "shard_count", minimum=1, maximum=1024)
    _integer(shard_index, "shard_index", minimum=0, maximum=shard_count - 1)
    catalog_path = _repo_path(root, catalog_path, "runs")
    protocol_path = _repo_path(root, protocol_path, "experiments")
    output_root = _repo_path(root, output_root, "runs")
    catalog_sha256 = _sha(catalog_sha256, "catalog hash")
    protocol_sha256 = _sha(protocol_sha256, "protocol hash")
    catalog = _read_pinned_json(catalog_path, catalog_sha256, "catalog")
    protocol = _read_pinned_json(protocol_path, protocol_sha256, "protocol")
    rows = _catalog_rows(catalog, protocol, catalog_sha256)
    _repo_path(root, catalog["scan_path"], "runs")  # Pointer only; never read prior structural references.
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if not selected:
        raise ValueError("empty shard: reduce shard_count")
    loaded = _load_sources(protocol, root)
    shard_dir = output_root / f"shard-{shard_index:04d}-of-{shard_count:04d}"
    if shard_dir.exists():
        raise FileExistsError(f"immutable shard already exists: {shard_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(exist_ok=False)
    traces = shard_dir / "traces"
    traces.mkdir()
    jsonl = shard_dir / "cells.jsonl"
    inventory: dict[str, str] = {}
    max_steps = protocol["max_steps"]
    with jsonl.open("x", encoding="utf-8") as stream:
        for row in selected:
            for learner_seed in (0, 1):
                if (file_sha256(catalog_path) != catalog_sha256
                        or file_sha256(protocol_path) != protocol_sha256
                        or any(file_sha256(item[3]) != SOURCE_HASHES[seed][0]
                               for seed, item in loaded.items())):
                    raise ValueError("frozen input changed during shard run; partial shard is invalid")
                result, arrays = diagnose_cell(row, learner_seed, loaded[learner_seed], max_steps)
                relative = f"traces/seed-{row['geometry_seed']}-source-{learner_seed}.npz"
                with (shard_dir / relative).open("xb") as destination:
                    np.savez_compressed(destination, **arrays)
                result["trace_path"] = relative
                result["trace_sha256"] = file_sha256(shard_dir / relative)
                inventory[relative] = result["trace_sha256"]
                result["catalog_sha256"] = catalog_sha256
                result["protocol_sha256"] = protocol_sha256
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    inventory["cells.jsonl"] = file_sha256(jsonl)
    if (file_sha256(catalog_path) != catalog_sha256 or file_sha256(protocol_path) != protocol_sha256
            or any(file_sha256(item[3]) != SOURCE_HASHES[seed][0]
                   for seed, item in loaded.items())):
        raise ValueError("frozen inputs changed before shard seal")
    manifest = {
        "format": MANIFEST_FORMAT,
        "role": "training_diagnostic",
        "ranked": False,
        "score_selection": False,
        "performance_scope": "internal raw-reward/geometry diagnostics only; not official HAIC performance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_path": str(catalog_path.relative_to(root)),
        "catalog_sha256": catalog_sha256,
        "catalog_protocol_sha256": catalog["protocol_sha256"],
        "analysis_sha256": catalog["analysis_sha256"],
        "scan_path": catalog["scan_path"],
        "scan_sha256": catalog["scan_sha256"],
        "protocol_path": str(protocol_path.relative_to(root)),
        "protocol_sha256": protocol_sha256,
        "source_actors": [
            {"learner_seed": seed, "actor_path": str(loaded[seed][3].relative_to(root)),
             "actor_sha256": SOURCE_HASHES[seed][0], "actor_weights_sha256": SOURCE_HASHES[seed][1]}
            for seed in (0, 1)
        ],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "geometry_ids": [row["geometry_seed"] for row in selected],
        "cell_count": len(selected) * 2,
        "frame_skip": 4,
        "max_steps": max_steps,
        "sparse_stride": SPARSE_STRIDE,
        "schema": {"cell": RESULT_FORMAT, "trace_arrays": {
            "reward": "float32 raw reward per high-level decision",
            "progress": "float32 visited-tile fraction per decision (not directed lap fraction)",
            "damage": "float32 damage per decision",
            "native_action": "float32 [step, steer/gas/brake] symmetric [-1,1] executed",
            "official_action": "float32 [step, steer/gas/brake] official [-1,1]/[0,1]/[0,1] executed",
            "collision_action": "bool any raw-frame collision in decision",
            "off_track_counter": "int16 consecutive negative-total-reward decisions",
            "sparse_step": "int32 decision number including reset step 0 and terminal step",
            "sparse_nearest_point_index": "int32 raw road centerline point index",
            "sparse_nearest_point_fraction": "float32 centerline arclength fraction from spawn",
            "sparse_nearest_distance_m": "float32 distance from car to nearest road point",
            "sparse_speed_m_s": "float32 car hull speed at sparse decision",
        }},
        "file_sha256": dict(sorted(inventory.items())),
    }
    with (shard_dir / "manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, sort_keys=True, indent=2, allow_nan=False)
        destination.write("\n")
    with (shard_dir / "manifest.sha256").open("x", encoding="ascii") as destination:
        destination.write(f"{file_sha256(shard_dir / 'manifest.json')}  manifest.json\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new runs/ directory for immutable, separately sealed shards")
    parser.add_argument("--shard-index", type=int, required=True, help="zero-based shard index")
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run_shard(root=ROOT, catalog_path=args.catalog, catalog_sha256=args.catalog_sha256,
                       protocol_path=args.protocol, protocol_sha256=args.protocol_sha256,
                       output_root=args.output_root, shard_index=args.shard_index,
                       shard_count=args.shard_count)
    print(json.dumps({"role": result["role"], "ranked": False,
                      "geometry_count": len(result["geometry_ids"]),
                      "cell_count": result["cell_count"],
                      "shard_index": args.shard_index, "shard_count": args.shard_count,
                      "manifest_sha256": file_sha256(args.output_root / f"shard-{args.shard_index:04d}-of-{args.shard_count:04d}" / "manifest.json")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
