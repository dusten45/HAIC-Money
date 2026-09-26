"""Frozen, TRAIN-only RLPD G0 diagnostic; never a ranking evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec
from haic.algorithms.rlpd.g0_diagnostic import (
    G0Cell, G0Decision, G0EventRules, G0Identity, G0Telemetry,
    classify_g0_episode, validate_g0_trace,
)


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"[0-9a-f]{64}")
SOURCE_FILES = frozenset({
    "agent.py", "common_adapter.py", "train.py", "env_wrapper.py",
    "action_smoothing.py", "tracking.py",
    "core/__init__.py", "core/vendor/__init__.py", "scripts/__init__.py",
    "haic/__init__.py", "haic/algorithms/__init__.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py", "core/obstacle_contacts.py",
    "core/finish_line.py", "core/track_variables.py", "action_representation.py",
    "damage.py", "haic/algorithms/rlpd/model.py",
    "haic/algorithms/rlpd/g0_diagnostic.py", "scripts/diagnose_rlpd_g0.py",
    "scripts/audit_rlpd_g0_seeds.py", "scripts/prepare_rlpd_g0_protocol.py",
    "scripts/summarize_rlpd_g0.py",
    "requirements.txt",
})
EXPECTED_ACTORS = {
    "long-horizon-seed11": {
        "path": "runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt",
        "sha256": "f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1",
        "training_seed": 11, "study_id": "pixel-rlpd-long-horizon-followup-v1", "arm": "rlpd",
    },
    "entropy-v5-author-seed50": {
        "path": "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/rlpd-author-target-seed50/checkpoints/step-000131072/actor.pt",
        "sha256": "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98",
        "training_seed": 50, "study_id": "pixel-rlpd-entropy-target-ablation-v5", "arm": "rlpd-author-target",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _file(root: Path, value: Any, prefix: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"path must be repository-relative under {prefix}/")
    relative = Path(value)
    if relative.parts[0] != prefix or any(part in (".", "..") for part in relative.parts):
        raise ValueError(f"path must remain under {prefix}/")
    path = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink in frozen path: {value}")
    if not path.is_file():
        raise ValueError(f"frozen file is missing: {value}")
    return path


def _pinned_json(path: Path, expected: str) -> dict[str, Any]:
    if sha256_file(path) != _sha(expected, str(path)):
        raise ValueError(f"frozen JSON hash changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen JSON must be an object")
    return value


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _check_audit_inventory(root: Path, audit: dict[str, Any]) -> None:
    inventory = audit.get("source_inventory")
    if (
        not isinstance(inventory, list) or not inventory
        or _canonical_sha(inventory) != _sha(audit.get("source_inventory_sha256"), "audit inventory")
    ):
        raise ValueError("G0 geometry audit source inventory is malformed")
    _sha(audit.get("excluded_inventory_sha256"), "excluded seed inventory")
    paths = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "kind", "sha256", "bytes"}:
            raise ValueError("G0 geometry audit has malformed source inventory entry")
        relative = item["path"]
        if not isinstance(relative, str) or not relative.startswith(("experiments/", "runs/")):
            raise ValueError("G0 geometry audit inventory escaped protocol/run trees")
        if relative in paths or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("G0 geometry audit source inventory has duplicate/unsafe paths")
        paths.add(relative)
        path = _file(root, relative, relative.split("/", 1)[0])
        if sha256_file(path) != _sha(item["sha256"], relative) or path.stat().st_size != item["bytes"]:
            raise ValueError(f"G0 geometry audit source changed: {relative}")


def _check_experiment_content_inventory(root: Path, audit: dict[str, Any]) -> None:
    inventory = audit.get("experiment_content_inventory")
    if (not isinstance(inventory, list) or not inventory
            or _canonical_sha(inventory) != _sha(
                audit.get("experiment_content_inventory_sha256"), "experiment content inventory"
            )):
        raise ValueError("G0 experiment content inventory is malformed")
    paths = set()
    for item in inventory:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ValueError("G0 experiment content inventory entry is malformed")
        relative = item["path"]
        if not isinstance(relative, str) or relative in paths:
            raise ValueError("G0 experiment content path is duplicated or malformed")
        paths.add(relative)
        path = _file(root, relative, "experiments")
        if sha256_file(path) != _sha(item["sha256"], relative) or path.stat().st_size != item["bytes"]:
            raise ValueError(f"G0 experiment content changed: {relative}")


def preflight(
    root: Path, protocol_path: str, protocol_sha256: str, *,
    actor_factory=None, payload_loader=None, audit_verifier=None,
    expected_actors=None,
) -> dict[str, Any]:
    """Validate all cells, sources and CPU actors before any environment creation."""
    root = root.resolve()
    protocol_file = _file(root, protocol_path, "experiments")
    protocol = _pinned_json(protocol_file, protocol_sha256)
    if protocol.get("format") != "haic-rlpd-g0-diagnostic-v1" or protocol.get("status") != "frozen":
        raise ValueError("G0 protocol must have the frozen diagnostic format")
    if protocol.get("partition") != "TRAIN" or protocol.get("frame_skip") != 4:
        raise ValueError("G0 must use TRAIN-only four-frame-skip control")
    if protocol.get("max_steps") != 2000 or protocol.get("decision_cap") != 48000:
        raise ValueError("G0 budget differs from its fixed 24x2000 cap")
    if protocol.get("reward_shaping") is not False or protocol.get("collision_penalty") != 0:
        raise ValueError("G0 cannot modify the raw reward environment")
    if protocol.get("interventions") is not False or protocol.get("learner_updates") != 0:
        raise ValueError("G0 may not intervene or train")
    try:
        rules = G0EventRules(**protocol["event_rules"])
    except (KeyError, TypeError) as exc:
        raise ValueError("G0 event rules must be fully frozen") from exc
    if rules.max_decisions != protocol["max_steps"] or rules.negative_reward_limit != 100:
        raise ValueError("G0 classifier disagrees with wrapper/budget semantics")
    distance = protocol.get("centerline_far_threshold_m")
    if type(distance) not in (int, float) or not np.isfinite(distance) or distance <= 0:
        raise ValueError("G0 needs a finite centerline-distance proxy threshold")

    rows = protocol.get("cells")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("G0 must freeze exactly 12 geometry cells")
    cells = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"partition", "track_id", "geometry_seed", "obstacles"}:
            raise ValueError("G0 cells need explicit TRAIN track/geometry/obstacle identity")
        track_id, seed = row["track_id"], row["geometry_seed"]
        if (row["partition"] != "TRAIN" or row["obstacles"] is not True
                or type(track_id) is not int or track_id not in (1, 2, 3, 4)
                or type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("G0 has a non-TRAIN or malformed geometry cell")
        cells.append((track_id, seed))
    if len({seed for _, seed in cells}) != 12:
        raise ValueError("G0 geometry seeds are not 12 independent allocated IDs")

    audit_path = _file(root, protocol.get("geometry_audit_path"), "experiments")
    audit = _pinned_json(audit_path, protocol.get("geometry_audit_sha256"))
    from scripts.audit_rlpd_g0_seeds import R5_ERRATUM_PATH
    erratum_ref = audit.get("r5_erratum")
    if (not isinstance(erratum_ref, dict)
            or erratum_ref.get("path") != R5_ERRATUM_PATH
            or erratum_ref.get("path") != protocol.get("r5_erratum_path")
            or erratum_ref.get("sha256") != protocol.get("r5_erratum_sha256")
            or erratum_ref.get("original_ledger_bytes_attested") is not False):
        raise ValueError("G0 protocol must explicitly pin the narrow r5 provenance erratum")
    erratum_file = _file(root, erratum_ref["path"], "experiments")
    if sha256_file(erratum_file) != _sha(erratum_ref["sha256"], "r5 erratum"):
        raise ValueError("G0 r5 provenance erratum changed")
    if (
        audit.get("format") != "haic-rlpd-g0-seed-audit-v1"
        or audit.get("status") != "no_known_recorded_overlap"
        or audit.get("passed") is not True
        or audit.get("cells") != rows
        or type(audit.get("collision_count")) is not int or audit["collision_count"] != 0
        or audit.get("ambiguities") != []
        or audit.get("protocol_frozen") is not False
        or audit.get("candidate_seeds") != [seed for _, seed in cells]
        or audit.get("candidate_seeds_sha256") != _canonical_sha([seed for _, seed in cells])
        or audit.get("candidate_rule") != "seed_start + offset for offsets 0..11, in that order; no replacement on collision"
        or audit.get("seed_start") != cells[0][1]
    ):
        raise ValueError("G0 cells lack a matching, clean frozen geometry audit")
    _check_audit_inventory(root, audit)
    _check_experiment_content_inventory(root, audit)
    if audit_verifier is None:
        from scripts.audit_rlpd_g0_seeds import audit_g0_seeds
        audit_verifier = audit_g0_seeds
    fresh = audit_verifier(cells[0][1], cells[0][0], repo_root=root, r5_erratum=erratum_ref["path"])
    if any(fresh.get(key) != audit.get(key) for key in (
        "format", "status", "passed", "cells", "candidate_seeds_sha256",
        "excluded_inventory_sha256", "source_inventory_sha256", "source_inventory", "r5_erratum",
        "experiment_content_inventory_sha256", "experiment_content_inventory",
    )):
        raise ValueError("G0 frozen geometry audit disagrees with a complete fresh audit")

    source_hashes = protocol.get("source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != SOURCE_FILES:
        raise ValueError("G0 executable source map is incomplete or includes mutable documents")
    for relative, expected in source_hashes.items():
        path = _file(root, relative, relative.split("/", 1)[0])
        if sha256_file(path) != _sha(expected, relative):
            raise ValueError(f"G0 executable source hash changed: {relative}")

    actors = protocol.get("actors")
    if not isinstance(actors, list) or len(actors) != 2:
        raise ValueError("G0 requires exactly two frozen source actors")
    if expected_actors is None:
        expected_actors = EXPECTED_ACTORS
    ids = set()
    actor_paths = []
    for entry in actors:
        if not isinstance(entry, dict) or set(entry) != {
            "id", "path", "sha256", "source_sha256", "export_protocol_sha256", "action_mode",
            "candidate_path", "candidate_sha256", "training_seed", "environment_steps",
            "checkpoint_path", "checkpoint_sha256", "source_study_protocol_path",
        }:
            raise ValueError("G0 actor identity is incomplete")
        if (not isinstance(entry["id"], str) or re.fullmatch(r"[a-z0-9-]{1,48}", entry["id"]) is None
                or entry["id"] in ids or entry["action_mode"] != "exported_tanh_mean"):
            raise ValueError("G0 actor IDs must be distinct deterministic exports")
        ids.add(entry["id"])
        expected = expected_actors.get(entry["id"])
        if (expected is None or entry["path"] != expected["path"]
                or entry["sha256"] != expected["sha256"]
                or type(entry["training_seed"]) is not int
                or entry["training_seed"] != expected["training_seed"]
                or entry["environment_steps"] != 131072):
            raise ValueError("G0 actor is not an independently designated frozen candidate")
        path = _file(root, entry["path"], "runs")
        if sha256_file(path) != _sha(entry["sha256"], "actor"):
            raise ValueError("G0 actor bytes differ from frozen protocol")
        candidate = _file(root, entry["candidate_path"], "runs")
        candidate_payload = _pinned_json(candidate, entry["candidate_sha256"])
        candidate_actor = candidate_payload.get("actor_path")
        candidate_checkpoint = candidate_payload.get("checkpoint_path")
        if (not isinstance(candidate_actor, str) or Path(candidate_actor).is_absolute()
                or ".." in Path(candidate_actor).parts
                or candidate.parent.parent.parent / candidate_actor != path
                or not isinstance(candidate_checkpoint, str) or Path(candidate_checkpoint).is_absolute()
                or ".." in Path(candidate_checkpoint).parts
                or (candidate.parent.parent.parent / candidate_checkpoint).relative_to(root).as_posix()
                != entry["checkpoint_path"]
                or candidate_payload.get("checkpoint_sha256") != entry["checkpoint_sha256"]
                or candidate_payload.get("format") != "haic-rlpd-candidate-v1"
                or candidate_payload.get("actor_sha256") != entry["sha256"]
                or candidate_payload.get("protocol_sha256") != entry["export_protocol_sha256"]
                or candidate_payload.get("study_id") != expected["study_id"]
                or candidate_payload.get("arm") != expected["arm"]
                or candidate_payload.get("training_seed") != entry["training_seed"]
                or candidate_payload.get("environment_steps") != entry["environment_steps"]):
            raise ValueError("G0 candidate receipt does not bind the designated actor")
        checkpoint_file = _file(root, entry["checkpoint_path"], "runs")
        study_protocol = _file(root, entry["source_study_protocol_path"], "experiments")
        if (sha256_file(checkpoint_file) != _sha(entry["checkpoint_sha256"], "source checkpoint")
                or entry["source_study_protocol_path"] != f"experiments/{expected['study_id']}.json"
                or sha256_file(study_protocol) != entry["export_protocol_sha256"]):
            raise ValueError("G0 source checkpoint/study protocol provenance changed")
        actor_paths.append(path)
    if set(ids) != set(expected_actors) or len(set(actor_paths)) != 2 or len({row["sha256"] for row in actors}) != 2:
        raise ValueError("G0 source actors must be two distinct designated exports")

    if payload_loader is None:
        payload_loader = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    if actor_factory is None:
        from agent import Agent
        actor_factory = lambda path: Agent(model_path=path)

    obs_spec, action_spec = ObservationSpec(), ActionSpec()
    adapter = ActionAdapter(action_spec)
    fixture_rng = np.random.default_rng(52731)
    fixtures = (
        np.zeros((4, 84, 84), dtype=np.float32),
        fixture_rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8).astype(np.float32) / 255.0,
    )
    loaded = {}
    for entry, path in zip(actors, actor_paths):
        payload = payload_loader(path)
        if (
            not isinstance(payload, dict) or payload.get("format") != "haic-rlpd-pixel-actor-v1"
            or payload.get("source_sha256") != _sha(entry["source_sha256"], "actor source")
            or payload.get("protocol_sha256") != _sha(entry["export_protocol_sha256"], "actor protocol")
            or payload.get("observation_spec") != asdict(obs_spec)
            or payload.get("action_spec") != asdict(action_spec)
        ):
            raise ValueError("G0 actor export metadata conflicts with the frozen identity")
        first, second = actor_factory(path), actor_factory(path)
        for observation in fixtures:
            first.reset(observation)
            second.reset(observation)
            a = np.asarray(first.act(observation))
            b = np.asarray(second.act(observation))
            if a.shape != (3,) or a.dtype != np.float32 or not np.array_equal(a, b):
                raise ValueError("G0 CPU actor reload/action trace mismatch")
            with torch.inference_mode():
                native = first.model(torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0))
            mapped = adapter.to_official(native.squeeze(0).detach().numpy(), clip=False)
            if not np.allclose(a, mapped, atol=1e-6, rtol=0):
                raise ValueError("G0 root Agent action differs from native actor/adapters")
        loaded[entry["id"]] = (first, path)
    return {
        "protocol": protocol, "protocol_path": protocol_file,
        "protocol_sha256": protocol_sha256, "audit_path": audit_path, "audit": audit,
        "cells": cells, "rules": rules, "actors": loaded,
        "audit_verifier": audit_verifier,
    }


def _uint8_pixels(observation: np.ndarray) -> np.ndarray:
    ObservationSpec().validate(observation)
    pixels = np.rint(observation * 255).astype(np.uint8)
    if not np.allclose(pixels.astype(np.float32) / 255.0, observation, rtol=0, atol=1e-6):
        raise ValueError("G0 observation cannot be reconstructed from recorded uint8 pixels")
    return pixels


def _road_sample(raw: Any, points: np.ndarray, arclength: np.ndarray) -> dict[str, float]:
    position = np.asarray(raw.car.hull.position, dtype=np.float64)
    velocity = np.asarray(raw.car.hull.linearVelocity, dtype=np.float64)
    if position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("G0 raw pose/velocity is invalid")
    distances = np.linalg.norm(points - position, axis=1)
    index = int(np.argmin(distances))
    tangent = points[(index + 1) % len(points)] - points[index]
    angle = float(raw.car.hull.angle) + np.pi / 2.0
    heading = float(np.arctan2(tangent[1], tangent[0]))
    return {
        "centerline_fraction": float(arclength[index] / arclength[-1]),
        "centerline_distance_m": float(distances[index]),
        "speed_m_s": float(np.linalg.norm(velocity)),
        "heading_error_rad": float(np.arctan2(np.sin(angle - heading), np.cos(angle - heading))),
        "x_m": float(position[0]), "y_m": float(position[1]),
    }


def _wrapper(env: Any) -> Any:
    for _ in range(12):
        if hasattr(env, "off_track_counter") and hasattr(env, "max_off_track_steps"):
            return env
        env = getattr(env, "env", None)
        if env is None:
            break
    raise ValueError("G0 original off-track wrapper is inaccessible")


def run_cell(
    *, row: dict[str, Any], actor: Any, actor_info: dict[str, Any], actor_payload: bytes,
    rules: G0EventRules, centerline_far_threshold_m: float, env_factory,
    attempt: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Drive one prevalidated TRAIN cell; raw state is diagnostic-only."""
    from common_adapter import EpisodeCollector

    track_id, seed = row["track_id"], row["geometry_seed"]
    from core.vendor.car_racing import PLAYFIELD
    if attempt is not None:
        attempt.update(phase="environment_creation", decision_calls=0, decisions_completed=0)
    env = env_factory(
        track_id=track_id, seed=seed, max_steps=rules.max_decisions,
        frame_skip=4, reward_shaping=False, obstacles=True, collision_penalty=0.0,
    )
    collector = EpisodeCollector(env)
    adapter = ActionAdapter()
    try:
        if attempt is not None:
            attempt["phase"] = "reset"
        raw = env.unwrapped
        counters = {"resets": 0, "reset_initial_raw_frames": 0,
                    "reset_noop_raw_frames": 0, "driven_raw_frames": 0,
                    "driven_raw_calls": 0}
        if attempt is not None:
            attempt["raw_counters"] = counters
        raw_commands = []
        raw_phase_bits = []
        reset_phase = [True]
        original_step, original_reset = raw.step, raw.reset

        def counted_step(action):
            command = None
            if action is None:
                counters["reset_initial_raw_frames"] += 1
            elif reset_phase[0]:
                if not np.array_equal(np.asarray(action), [0.0, 0.0, 0.0]):
                    raise ValueError("G0 reset warmup issued a non-noop raw action")
                counters["reset_noop_raw_frames"] += 1
            else:
                command = np.asarray(action, dtype=np.float64).copy()
                counters["driven_raw_calls"] += 1
            result = original_step(action)
            if command is not None:
                raw_commands.append(command)
                counters["driven_raw_frames"] += 1
            if command is not None:
                tracker = getattr(raw, "finish_line_tracker", None)
                if tracker is None:
                    raw_phase_bits.append(255)
                else:
                    # Flags 1/2/4/8/16: qualified/departed/back-crossing/candidate/finished; 255 is missing.
                    raw_phase_bits.append(
                        (1 if getattr(tracker, "qualified_time_s", None) is not None else 0)
                        | (2 if getattr(tracker, "departed_start_area", False) else 0)
                        | (4 if getattr(tracker, "crossing_from_back", False) else 0)
                        | (8 if getattr(tracker, "candidate_crossing_time_s", None) is not None else 0)
                        | (16 if getattr(tracker, "finish_time_s", None) is not None else 0)
                    )
            return result

        def counted_reset(*args, **kwargs):
            counters["resets"] += 1
            return original_reset(*args, **kwargs)

        raw.step, raw.reset = counted_step, counted_reset
        observation, info = collector.reset()
        reset_phase[0] = False
        if attempt is not None:
            attempt.update(phase="preflight_after_reset", reset_count=counters["resets"],
                           reset_initial_raw_frames=counters["reset_initial_raw_frames"],
                           reset_noop_raw_frames=counters["reset_noop_raw_frames"])
        if info.get("track_id") != track_id or info.get("seed") != seed:
            raise ValueError("G0 reset selected an unallocated cell")
        actor.reset(observation)
        if getattr(raw, "track_id", None) != track_id or getattr(raw, "track_seed", None) != seed:
            raise ValueError("G0 raw road differs from requested track/geometry identity")
        wrapper = _wrapper(env)
        if wrapper.max_off_track_steps != rules.negative_reward_limit:
            raise ValueError("G0 original off-track threshold drifted")
        if (counters["resets"] < 1
                or counters["reset_initial_raw_frames"] != counters["resets"]
                or counters["reset_noop_raw_frames"] != wrapper.warmup_steps):
            raise ValueError("G0 reset/raw-frame accounting differs from original wrapper")
        points = np.asarray(raw.track, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 4 or len(points) < 20 or not np.isfinite(points).all():
            raise ValueError("G0 raw road centerline is missing or malformed")
        points = points[:, 2:4]
        lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        arclength = np.concatenate(([0.0], np.cumsum(lengths)))
        if not np.isfinite(arclength[-1]) or arclength[-1] <= 0:
            raise ValueError("G0 raw road has invalid perimeter")
        road_sha256 = hashlib.sha256(np.ascontiguousarray(points.astype("<f8")).tobytes()).hexdigest()
        initial_stack = _uint8_pixels(observation)
        expected_stack = initial_stack.copy()
        previous = _road_sample(raw, points, arclength)
        previous_tiles = int(raw.tile_visited_count)
        reset_initial_frames = counters["reset_initial_raw_frames"]
        reset_noop_frames = counters["reset_noop_raw_frames"]
        last_t = float(raw.t)
        arrays: dict[str, list[Any]] = {
            key: [] for key in (
                "observation_frame", "next_frame", "proposed_native_action", "executed_native_action",
                "commanded_official_action", "raw_official_action", "summed_reward", "new_tiles", "directed_delta",
                "centerline_distance_m", "centerline_fraction", "speed_m_s", "heading_error_rad",
                "x_m", "y_m", "progress", "damage", "contact", "off_track_counter",
                "finish_qualified", "finished", "terminated", "truncated", "finish_phase",
                "raw_frames", "raw_finish_phase_bits",
            )
        }
        if attempt is not None:
            attempt.update(phase="driving", partial_arrays=arrays, initial_stack=initial_stack)
        telemetry = []
        for step in range(rules.max_decisions):
            official = np.asarray(actor.act(observation))
            if official.shape != (3,) or official.dtype != np.float32:
                raise ValueError("G0 actor produced a non-float32 official action")
            with torch.inference_mode():
                native_model = actor.model(torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0))
            native_model = native_model.squeeze(0).detach().numpy()
            if not np.allclose(adapter.to_official(native_model, clip=False), official, rtol=0, atol=1e-6):
                raise ValueError("G0 actor native/root Agent action parity failed")
            proposed = np.asarray(native_model, dtype=np.float32).copy()
            previous_commands = len(raw_commands)
            previous_phase = len(raw_phase_bits)
            if attempt is not None:
                attempt["decision_calls"] += 1
            transition = collector.step(proposed)
            if attempt is not None:
                attempt["decisions_completed"] += 1
                attempt["driven_raw_frames"] = counters["driven_raw_frames"]
            commands = raw_commands[previous_commands:]
            if (transition.info.get("track_id") != track_id or transition.info.get("seed") != seed
                    or not np.allclose(transition.applied_action, official, rtol=0, atol=1e-6)):
                raise ValueError("G0 environment executed a different cell or action")
            if (not 1 <= len(commands) <= 4 or any(
                command.shape != (3,) or not np.allclose(command, official, rtol=0, atol=1e-6)
                for command in commands
            )):
                raise ValueError("G0 raw CarRacing.step received a substituted action")
            phase_bits = raw_phase_bits[previous_phase:]
            if len(phase_bits) != len(commands):
                raise ValueError("G0 raw finish phase sampling missed a driven frame")
            current = _road_sample(raw, points, arclength)
            frac_delta = (current["centerline_fraction"] - previous["centerline_fraction"] + 0.5) % 1.0 - 0.5
            directed = frac_delta if abs(frac_delta) <= 0.05 and max(
                previous["centerline_distance_m"], current["centerline_distance_m"]
            ) <= centerline_far_threshold_m else None
            tiles = int(raw.tile_visited_count)
            if tiles < previous_tiles:
                raise ValueError("G0 visited-tile count decreased")
            tracker = getattr(raw, "finish_line_tracker", None)
            if tracker is None:
                phase = None
            elif transition.info.get("finished"):
                phase = "finished"
            elif getattr(tracker, "candidate_crossing_time_s", None) is not None:
                phase = "candidate_crossing"
            elif getattr(tracker, "crossing_from_back", False):
                phase = "crossing_from_back"
            elif transition.info.get("finish_qualified"):
                phase = "qualified"
            else:
                phase = "unqualified"
            time_s = float(raw.t)
            raw_frames = len(commands)
            if abs(time_s - last_t - raw_frames / 50) > 1e-5:
                raise ValueError("G0 raw-frame accounting is inconsistent")
            contact = bool(transition.info["collision"])
            record = G0Telemetry(
                step=step, summed_reward=float(transition.reward), new_tiles=tiles - previous_tiles,
                directed_delta=directed,
                centerline_far=current["centerline_distance_m"] > centerline_far_threshold_m,
                contact=contact, damage=float(transition.info["damage"]),
                progress=float(transition.info["progress"]),
                finish_qualified=bool(transition.info["finish_qualified"]), finish_phase=phase,
                finished=bool(transition.info["finished"]), terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
                out_of_bounds=max(abs(current["x_m"]), abs(current["y_m"])) > PLAYFIELD,
                retire_reason=transition.info["retire_reason"],
            )
            telemetry.append(record)
            pixels = _uint8_pixels(observation)
            next_pixels = _uint8_pixels(transition.next_observation)
            if not np.array_equal(pixels, expected_stack):
                raise ValueError("G0 current observation differs from preserved frame history")
            expected_stack = np.concatenate((expected_stack[1:], next_pixels[-1][None]), axis=0)
            if not np.array_equal(next_pixels, expected_stack):
                raise ValueError("G0 next observation differs from preserved frame history")
            values = {
                "observation_frame": pixels[-1], "next_frame": next_pixels[-1],
                "proposed_native_action": proposed, "executed_native_action": transition.action,
                "commanded_official_action": transition.applied_action,
                "raw_official_action": commands[0],
                "summed_reward": record.summed_reward, "new_tiles": record.new_tiles,
                "directed_delta": np.nan if directed is None else directed,
                "centerline_distance_m": current["centerline_distance_m"],
                "centerline_fraction": current["centerline_fraction"],
                "speed_m_s": current["speed_m_s"], "heading_error_rad": current["heading_error_rad"],
                "x_m": current["x_m"], "y_m": current["y_m"], "progress": record.progress,
                "damage": record.damage, "contact": contact,
                "off_track_counter": int(wrapper.off_track_counter),
                "finish_qualified": record.finish_qualified, "finished": record.finished,
                "terminated": record.terminated, "truncated": record.truncated,
                "finish_phase": phase or "unknown", "raw_frames": raw_frames,
                "raw_finish_phase_bits": np.array(phase_bits + [255] * (4 - len(phase_bits)), dtype=np.uint8),
            }
            for key, value in values.items():
                arrays[key].append(value)
            previous, previous_tiles, last_t = current, tiles, time_s
            if transition.done:
                break
            observation = transition.next_observation
        else:
            transition = None

        last = telemetry[-1]
        end_reason = (
            "finished" if last.finished else
            "retired" if last.retire_reason in ("off_track", "crash") else
            "out_of_bounds" if last.terminated and last.out_of_bounds else
            "task_timeout" if last.truncated and len(telemetry) == rules.max_decisions else
            "unknown" if last.terminated or last.truncated else "collection_censored"
        )
        summary = classify_g0_episode(telemetry, rules=rules, end_reason=end_reason)
        stack = initial_stack.copy()

        def decisions():
            nonlocal stack
            for index in range(len(telemetry)):
                yield G0Decision(
                    step=index, observation=stack.astype(np.float32) / 255.0,
                    proposed_native_action=np.asarray(arrays["proposed_native_action"][index]),
                    executed_native_action=np.asarray(arrays["executed_native_action"][index]),
                    applied_official_action=np.asarray(arrays["commanded_official_action"][index]),
                )
                stack = np.concatenate((stack[1:], np.asarray(arrays["next_frame"][index])[None]), axis=0)

        identity = G0Identity(
            actor_sha256=actor_info["sha256"], source_sha256=actor_info["source_sha256"],
            protocol_sha256=actor_info["export_protocol_sha256"],
            observation_fingerprint=ObservationSpec().fingerprint,
            action_fingerprint=ActionSpec().fingerprint,
            action_mode=actor_info["action_mode"], reset_contract="stateless-export-episode-reset",
        )
        validate_g0_trace(
            identity=identity, actor_payload=actor_payload,
            observation_spec=ObservationSpec(), action_spec=ActionSpec(),
            cell=G0Cell("TRAIN", seed, track_id), allowed_train_cells={(track_id, seed)},
            decisions=decisions(),
        )
        trace = {key: np.asarray(values) for key, values in arrays.items()}
        trace["initial_stack"] = initial_stack
        result = {
            "partition": "TRAIN", "track_id": track_id, "geometry_seed": seed,
            "actor_id": actor_info["id"], "actor_sha256": actor_info["sha256"],
            "road_centerline_sha256": road_sha256, "steps": len(telemetry),
            "action_semantics": "actor native proposal, collector executed native, and raw CarRacing.step official commands separately observed",
            "finish_phase_sampling": "raw-tick post-step flags captured; classifier still treats crossing mechanism as unassessed",
            "driven_raw_frames": counters["driven_raw_frames"],
            "reset_initial_raw_frames": reset_initial_frames,
            "reset_noop_raw_frames": reset_noop_frames,
            "summary": asdict(summary),
            "raw_reward_sum": float(sum(arrays["summed_reward"])),
            "max_progress": float(max(arrays["progress"])),
        }
        if attempt is not None:
            attempt["phase"] = "complete"
        return result, trace
    finally:
        env.close()


def _check_cell_inputs(root: Path, context: dict[str, Any]) -> None:
    protocol = context["protocol"]
    for relative, digest in protocol["source_hashes"].items():
        if sha256_file(root / relative) != digest:
            raise ValueError("G0 source drift during collection; preserve partial run")
    if (sha256_file(context["protocol_path"]) != context["protocol_sha256"]
            or sha256_file(context["audit_path"]) != protocol["geometry_audit_sha256"]):
        raise ValueError("G0 frozen protocol/audit changed during collection")
    erratum_ref = context["audit"]["r5_erratum"]
    if sha256_file(root / erratum_ref["path"]) != erratum_ref["sha256"]:
        raise ValueError("G0 r5 provenance erratum changed during collection")
    _check_audit_inventory(root, context["audit"])
    _check_experiment_content_inventory(root, context["audit"])
    fresh = context["audit_verifier"](
        context["cells"][0][1], context["cells"][0][0], repo_root=root,
        r5_erratum=erratum_ref["path"],
    )
    if any(fresh.get(key) != context["audit"].get(key) for key in (
        "status", "cells", "excluded_inventory_sha256", "source_inventory_sha256", "r5_erratum",
        "experiment_content_inventory_sha256", "experiment_content_inventory",
    )):
        raise ValueError("G0 geometry inventory changed before cell reset")
    for entry in protocol["actors"]:
        actor_path = context["actors"][entry["id"]][1]
        if sha256_file(actor_path) != entry["sha256"]:
            raise ValueError("G0 actor drift during collection")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, sort_keys=True, indent=2, allow_nan=False)
        destination.write("\n")


def _claim_geometry_pool(root: Path, context: dict[str, Any], output_root: str) -> None:
    claims = root / "runs/rlpd-g0-claims"
    if claims.is_symlink():
        raise ValueError("G0 claim directory must not be a symlink")
    claims.mkdir(exist_ok=True)
    seeds = sorted(seed for _, seed in context["cells"])
    lock_path = claims / ".lock"
    if lock_path.is_symlink():
        raise ValueError("G0 claim lock must not be a symlink")
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for existing in claims.glob("*.json"):
            if existing.is_symlink():
                raise ValueError("G0 claim must not be a symlink")
            receipt = json.loads(existing.read_text(encoding="utf-8"))
            if receipt.get("format") != "haic-rlpd-g0-geometry-claim-v1":
                raise ValueError("unknown G0 pool claim; fail closed")
            recorded = receipt.get("geometry_seeds")
            if not isinstance(recorded, list) or any(type(seed) is not int for seed in recorded):
                raise ValueError("malformed G0 pool claim; fail closed")
            if set(seeds) & set(recorded):
                raise ValueError("G0 geometry pool already reserved by an earlier attempt")
        _write_json_exclusive(claims / f"{_canonical_sha(seeds)}.json", {
            "format": "haic-rlpd-g0-geometry-claim-v1", "status": "reserved-once",
            "geometry_seeds": seeds, "protocol_sha256": context["protocol_sha256"],
            "geometry_audit_sha256": context["protocol"]["geometry_audit_sha256"],
            "output_root": output_root,
        })


def collect(
    root: Path, context: dict[str, Any], output_root: str, *, env_factory=None,
) -> dict[str, Any]:
    if env_factory is None:
        from train import build_env
        env_factory = build_env
    if not isinstance(output_root, str) or Path(output_root).is_absolute() or (
        not Path(output_root).parts or Path(output_root).parts[0] != "runs"
        or any(part in (".", "..") for part in Path(output_root).parts)
    ):
        raise ValueError("G0 output must be a new repository-relative runs/ directory")
    output = root / output_root
    current = root
    for part in Path(output_root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("G0 output path must not traverse a symlink")
    if output.exists():
        raise FileExistsError(output)
    _check_cell_inputs(root, context)
    _claim_geometry_pool(root, context, output_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    (output / "attempts").mkdir()
    (output / "aborts").mkdir()
    protocol = context["protocol"]
    spent = 0
    rows = []
    road_hashes: dict[int, str] = {}
    for cell in protocol["cells"]:
        for actor_info in protocol["actors"]:
            try:
                _check_cell_inputs(root, context)
                if spent + protocol["max_steps"] > protocol["decision_cap"]:
                    raise ValueError("G0 requested cell exceeds frozen decision cap")
            except Exception as exc:
                _write_json_exclusive(output / "halt.json", {
                    "status": "pre-reset-halt", "cell": cell, "actor_id": actor_info["id"],
                    "completed_cells": len(rows), "completed_decisions": spent,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                raise
            actor, path = context["actors"][actor_info["id"]]
            name = f"seed-{cell['geometry_seed']}-track-{cell['track_id']}-{actor_info['id']}"
            _write_json_exclusive(output / "attempts" / f"{name}.json", {
                "status": "reserved_attempt", "cell": cell, "actor_id": actor_info["id"],
                "decision_reservation": protocol["max_steps"], "completed_decisions_before": spent,
                "protocol_sha256": context["protocol_sha256"],
            })
            attempt: dict[str, Any] = {}
            spent_before = spent
            try:
                result, trace = run_cell(
                    row=cell, actor=actor, actor_info=actor_info, actor_payload=path.read_bytes(),
                    rules=context["rules"],
                    centerline_far_threshold_m=protocol["centerline_far_threshold_m"],
                    env_factory=env_factory, attempt=attempt,
                )
                known_road = road_hashes.get(cell["geometry_seed"])
                if known_road is not None and known_road != result["road_centerline_sha256"]:
                    raise ValueError("G0 paired actors received different road centerlines")
                _check_cell_inputs(root, context)
                spent += result["steps"]
                road_hashes[cell["geometry_seed"]] = result["road_centerline_sha256"]
                relative = f"traces/{name}.npz"
                attempt["phase"] = "writing_trace_and_ledger"
                with (output / relative).open("xb") as destination:
                    np.savez_compressed(destination, **trace)
                result["trace_path"] = relative
                result["trace_sha256"] = sha256_file(output / relative)
                with (output / "cells.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                rows.append(result)
            except Exception as exc:
                partial = attempt.get("partial_arrays")
                partial_path = None
                partial_error = None
                if partial and len(partial["summed_reward"]) > 0:
                    partial_path = f"aborts/{name}.npz"
                    try:
                        with (output / partial_path).open("xb") as destination:
                            np.savez_compressed(destination, initial_stack=attempt["initial_stack"],
                                                **{key: np.asarray(value) for key, value in partial.items()})
                    except Exception as archive_exc:
                        partial_error = f"{type(archive_exc).__name__}: {archive_exc}"
                        partial_path = None
                _write_json_exclusive(output / "aborts" / f"{name}.json", {
                    "status": "invalid_partial_attempt", "cell": cell, "actor_id": actor_info["id"],
                    "decision_reservation": protocol["max_steps"],
                    "decision_calls_at_least": attempt.get("decision_calls", 0),
                    "decisions_completed": attempt.get("decisions_completed", 0),
                    "driven_raw_frames_observed": attempt.get("raw_counters", {}).get("driven_raw_frames", 0),
                    "driven_raw_calls_at_least": attempt.get("raw_counters", {}).get("driven_raw_calls", 0),
                    "reset_count_observed": attempt.get("raw_counters", {}).get("resets", 0),
                    "phase": attempt.get("phase", "before_environment_creation"),
                    "completed_decisions_before": spent_before,
                    "partial_trace_path": partial_path,
                    "partial_trace_sha256": sha256_file(output / partial_path) if partial_path else None,
                    "partial_trace_error": partial_error,
                    "error": f"{type(exc).__name__}: {exc}",
                    "no_retry_or_relabel": True,
                })
                raise
    try:
        _check_cell_inputs(root, context)
    except Exception as exc:
        _write_json_exclusive(output / "halt.json", {
            "status": "post-collection-source-drift", "completed_cells": len(rows),
            "completed_decisions": spent, "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    manifest = {
        "format": "haic-rlpd-g0-diagnostic-result-v1", "role": "TRAIN-only-failure-diagnostic",
        "ranked": False, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(context["protocol_path"].relative_to(root)),
        "protocol_sha256": context["protocol_sha256"],
        "geometry_audit_sha256": protocol["geometry_audit_sha256"],
        "decisions_spent": spent, "decision_cap": protocol["decision_cap"],
        "cell_count": len(rows), "geometry_count": len(context["cells"]),
        "driven_raw_frames": sum(row["driven_raw_frames"] for row in rows),
        "reset_initial_raw_frames": sum(row["reset_initial_raw_frames"] for row in rows),
        "reset_noop_raw_frames": sum(row["reset_noop_raw_frames"] for row in rows),
        "cells_sha256": sha256_file(output / "cells.jsonl"),
    }
    _write_json_exclusive(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", help="collect into a new runs/ directory; omit for preflight only")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = preflight(ROOT, args.protocol, args.protocol_sha256)
    if args.output_root:
        manifest = collect(ROOT, result, args.output_root)
        print(json.dumps(manifest, sort_keys=True))
    else:
        print(json.dumps({"status": "preflight-only", "cells": len(result["cells"]),
                          "actors": list(result["actors"])}))


if __name__ == "__main__":
    main()
