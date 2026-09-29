"""Collect matched privileged-oracle/policy traces on consumed TRAIN cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("experiments/oracle-policy-diagnosis-v1.json")
DRQ_ROOT = Path("runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic")
DRQ_CATALOG = Path("runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json")
RLPD_ROOT = Path("runs/20260926-rlpd-g0-completion-v1")
DRQ_ACTORS = {
    "drq-seed0": {
        "actor_path": "runs/20260922-drq-augmentation-pad-v1-restart/control-seed0/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37",
    },
    "drq-seed1": {
        "actor_path": "runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
    },
}
RLPD_ACTORS = {
    "rlpd-seed11": {
        "actor_id": "long-horizon-seed11",
        "actor_path": "runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "f5c048d9cff711705c55887a433d433b3f7eed13ed104f3f370d750a2fba17e1",
    },
    "rlpd-seed50": {
        "actor_id": "entropy-v5-author-seed50",
        "actor_path": "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/rlpd-author-target-seed50/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98",
    },
}
MIN_POLICY_FAILURES = 20
MAX_POLICY_FAILURES = 50
FRAME_SKIP = 4
WARMUP = 50
TARGET_SPEED = 12.0
COHORT_CONTRACT = {
    "drq-r6": {
        "ledger": "runs/20260925-drqv2-geometry-mix-v1-r6/train-diagnostic/episodes.jsonl",
        "arm": "unchanged-source",
        "canonical_repeat": 0,
        "selection": "historically failed episodes only",
        "episodes": 21,
        "historical_failures": 21,
    },
    "rlpd-g0": {
        "ledger": "runs/20260926-rlpd-g0-completion-v1/cells.jsonl",
        "actors": ["long-horizon-seed11", "entropy-v5-author-seed50"],
        "selection": "all stored same-cell episodes, preserving six finishes as controls",
        "episodes": 24,
        "historical_failures": 18,
    },
    "total_policy_episodes": 45,
    "historical_failures": 39,
    "unique_track_seed_cells": 26,
    "allowed_partition": "previously consumed TRAIN or TRAIN-DIAGNOSTIC only",
}
ANALYSIS_CONTRACT = {
    "oracle": {"target_speed_m_s": 12.0, "avoid_obstacles": True},
    "environment": {"track_ids": [1], "frame_skip": 4, "warmup_raw_noops": 50,
                     "obstacles": True, "collision_penalty": 0.0,
                     "decision_caps": {"drq-r6": 1200, "rlpd-g0": 2000}},
    "material_divergence": {"position_gap_m": 2.0, "heading_gap_rad": 0.30,
                            "speed_gap_m_s": 2.0, "same_signal_for_decisions": 3},
    "first_precursor": {"curve_abs_curvature_min": 0.025,
                         "curve_speed_m_s_min": 14.0,
                         "heading_error_abs_rad_min": 0.55,
                         "centerline_offset_abs_m_min": 6.0,
                         "low_speed_m_s_max": 1.5,
                         "low_speed_window_decisions": 20,
                         "opposed_steering_min_abs_delta": 0.60},
    "taxonomy_is_causal": False,
    "control_conditioning": "same initial observation and exact road hash; no hidden-state equivalence after action divergence",
}


class DiagnosisError(RuntimeError):
    """Frozen input, reset, trace, or output contract failed."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DiagnosisError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise DiagnosisError(f"expected JSON objects in ledger: {path}")
    return rows


def _path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise DiagnosisError(f"path escapes repository: {relative}")
    return path


def _expected_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    values = protocol.get("source_sha256")
    if not isinstance(values, dict) or not values:
        raise DiagnosisError("protocol is missing source_sha256")
    return values


def verify_source_hashes(protocol: dict[str, Any]) -> None:
    if protocol.get("cohort") != COHORT_CONTRACT:
        raise DiagnosisError("policy cohort selection differs from the frozen contract")
    if protocol.get("analysis_contract") != ANALYSIS_CONTRACT:
        raise DiagnosisError("environment or divergence thresholds differ from the frozen contract")
    for relative, expected in _expected_hashes(protocol).items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise DiagnosisError(f"invalid source hash for {relative}")
        path = _path(relative)
        if _sha_file(path) != expected:
            raise DiagnosisError(f"source hash mismatch: {relative}")
    code_hash = protocol.get("code_sha256", {}).get("runner")
    if code_hash != _sha_file(Path(__file__).resolve()):
        raise DiagnosisError("diagnostic runner differs from the frozen protocol")


def _policy_summary(row: dict[str, Any], *, study: str, policy_id: str) -> dict[str, Any]:
    return {
        "study": study,
        "policy_id": policy_id,
        "track_id": int(row["track_id"]),
        "geometry_seed": int(row["geometry_seed"]),
        "original_finished": bool(row["finished"]),
        "original_steps": int(row["steps"]),
        "original_reason": row.get("retire_reason") or row.get("summary", {}).get("outcome"),
        "original_max_progress": float(row.get("max_progress", row.get("terminal_progress", 0.0))),
        "original_damage": float(row.get("max_damage", row.get("terminal_damage", 0.0))),
        "family": row.get("family"),
        "source_trace_path": row["trace_path"],
        "source_trace_sha256": row["trace_sha256"],
        "source_actor_sha256": row["actor_sha256"],
        "source_road_centerline_sha256": row.get("road_centerline_sha256"),
        "max_steps": 1200 if study == "drq-r6" else 2000,
    }


def _catalog_road_hash(value: Any, seed: int) -> str | None:
    if isinstance(value, dict):
        if value.get("geometry_seed") == seed and isinstance(value.get("road_coordinate_sha256"), str):
            return value["road_coordinate_sha256"]
        for child in value.values():
            match = _catalog_road_hash(child, seed)
            if match is not None:
                return match
    elif isinstance(value, list):
        for child in value:
            match = _catalog_road_hash(child, seed)
            if match is not None:
                return match
    return None


def load_cases() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select all 21 canonical DrQ failures and all 24 frozen RLPD G0 traces."""
    drq_ledger = DRQ_ROOT / "episodes.jsonl"
    drq_manifest = _read_json(DRQ_ROOT / "manifest.json")
    drq_catalog = _read_json(ROOT / DRQ_CATALOG)
    drq_rows = _read_jsonl(drq_ledger)
    if drq_manifest.get("files_sha256", {}).get("episodes.jsonl") != _sha_file(drq_ledger):
        raise DiagnosisError("DrQ episode ledger is not bound by its run manifest")
    drq_cases = []
    for row in drq_rows:
        if (row.get("arm") != "unchanged-source" or row.get("repeat") != 0
                or row.get("canonical_repeat") is not True or row.get("finished") is not False):
            continue
        source_seed = int(row["source_learner_seed"])
        policy_id = f"drq-seed{source_seed}"
        if policy_id not in DRQ_ACTORS:
            raise DiagnosisError(f"unexpected DrQ source actor: {policy_id}")
        row = dict(row)
        row["trace_path"] = row.get("trace_path") or (
            f"traces/seed-{row['geometry_seed']}/unchanged-source-seed{source_seed}/repeat-0.npz"
        )
        trace_relative = row["trace_path"]
        trace_path = (ROOT / DRQ_ROOT / trace_relative).resolve()
        expected = drq_manifest.get("files_sha256", {}).get(f"train-diagnostic/{trace_relative}")
        if expected is None:
            expected = drq_manifest.get("files_sha256", {}).get(trace_relative)
        if expected is None or _sha_file(trace_path) != expected:
            raise DiagnosisError(f"DrQ trace is not bound by its manifest: {trace_relative}")
        row["trace_sha256"] = expected
        row["trace_path"] = trace_path.relative_to(ROOT).as_posix()
        row["actor_sha256"] = row["actor_sha256"]
        row["steps"] = int(row["steps"])
        row["max_progress"] = row["max_progress"]
        row["max_damage"] = row["max_damage"]
        row["road_centerline_sha256"] = _catalog_road_hash(drq_catalog, int(row["geometry_seed"]))
        if row["road_centerline_sha256"] is None:
            raise DiagnosisError(f"DrQ road missing from frozen catalog: {row['geometry_seed']}")
        drq_cases.append(_policy_summary(row, study="drq-r6", policy_id=policy_id))

    rlpd_ledger = RLPD_ROOT / "cells.jsonl"
    rlpd_manifest = _read_json(RLPD_ROOT / "manifest.json")
    if _sha_file(rlpd_ledger) != rlpd_manifest.get("cells_sha256"):
        raise DiagnosisError("RLPD cell ledger is not bound by its run manifest")
    rlpd_rows = _read_jsonl(rlpd_ledger)
    rlpd_cases = []
    for row in rlpd_rows:
        policy_id = next((name for name, spec in RLPD_ACTORS.items()
                          if spec["actor_id"] == row.get("actor_id")), None)
        if policy_id is None:
            continue
        trace_path = (ROOT / RLPD_ROOT / row["trace_path"]).resolve()
        expected = row["trace_sha256"]
        if _sha_file(trace_path) != expected:
            raise DiagnosisError(f"RLPD trace hash mismatch: {trace_path}")
        normalized = dict(row)
        normalized["finished"] = row.get("summary", {}).get("outcome") == "finished"
        normalized["max_damage"] = row.get("max_damage", 0.0)
        normalized["trace_path"] = trace_path.relative_to(ROOT).as_posix()
        rlpd_cases.append(_policy_summary(normalized, study="rlpd-g0", policy_id=policy_id))

    cases = sorted(drq_cases + rlpd_cases,
                   key=lambda row: (row["study"], row["policy_id"], row["geometry_seed"]))
    failures = sum(not row["original_finished"] for row in cases)
    if not (20 <= failures <= 50) or len(drq_cases) != 21 or len(rlpd_cases) != 24:
        raise DiagnosisError(
            f"frozen cohort changed: DrQ={len(drq_cases)}, RLPD={len(rlpd_cases)}, failures={failures}"
        )
    if len({(row["track_id"], row["geometry_seed"]) for row in cases}) != 26 or failures != 39:
        raise DiagnosisError("the frozen cell/failure census changed")
    hashes = {
        "drq_ledger": _sha_file(drq_ledger),
        "drq_manifest": _sha_file(DRQ_ROOT / "manifest.json"),
        "drq_catalog": _sha_file(ROOT / DRQ_CATALOG),
        "rlpd_ledger": _sha_file(rlpd_ledger),
        "rlpd_manifest": _sha_file(RLPD_ROOT / "manifest.json"),
    }
    return cases, hashes


def _actor_path(policy_id: str) -> tuple[Path, str]:
    spec = (DRQ_ACTORS | RLPD_ACTORS)[policy_id]
    path = _path(spec["actor_path"])
    actual = _sha_file(path)
    if actual != spec["actor_sha256"]:
        raise DiagnosisError(f"frozen actor hash mismatch: {spec['actor_path']}")
    return path, actual


def _make_env(track_id: int, seed: int, max_steps: int):
    from gymnasium.wrappers.time_limit import TimeLimit

    from core.vendor.car_racing import CarRacing
    from env_wrapper import CarEnvironment

    base = CarRacing(continuous=True, render_mode="rgb_array")
    env = CarEnvironment(
        TimeLimit(base, max_episode_steps=max_steps * FRAME_SKIP + 200),
        skip_frames=FRAME_SKIP,
        no_operation=WARMUP,
    )
    observation, info = env.reset(seed=seed, options={"track_id": track_id})
    if not isinstance(info, dict):
        raise DiagnosisError("reset info is not a dictionary")
    if int(getattr(base, "track_id", -1)) != track_id:
        raise DiagnosisError("environment reset to an unexpected track")
    return env, base, observation


def _state(base: Any, controller: Any) -> dict[str, Any]:
    hull = base.car.hull
    position = np.asarray(hull.position, dtype=np.float64)
    velocity = np.asarray(hull.linearVelocity, dtype=np.float64)
    action, diag = controller.act()
    if (position.shape != (2,) or velocity.shape != (2,)
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise DiagnosisError("simulator pose or velocity is invalid")
    return {
        "position": position,
        "velocity": velocity,
        "heading": float(hull.angle),
        "speed": float(np.linalg.norm(velocity)),
        "oracle_action": np.asarray(action, dtype=np.float32),
        "arc_length": float(diag["arc_length"]),
        "center_error": float(diag["center_error"]),
        "path_error": float(diag["path_error"]),
        "curvature": float(diag["curvature"]),
        "heading_error": float(diag["heading_error"]),
        "waypoint_index": int(diag["waypoint_index"]),
    }


def _obs_hash(observation: np.ndarray) -> str:
    value = np.ascontiguousarray(observation)
    if value.shape != (4, 84, 84) or value.dtype != np.float32:
        raise DiagnosisError(f"unexpected policy observation: {value.dtype} {value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise DiagnosisError("observation is non-finite or outside [0, 1]")
    return _sha_bytes(value.tobytes())


def _track_hash(base: Any) -> str:
    points = np.asarray(base.track, dtype=np.float64)[:, 2:4].astype("<f8", copy=False)
    return _sha_bytes(np.ascontiguousarray(points).tobytes())


def _empty_trace() -> dict[str, list[Any]]:
    return defaultdict(list)


def _collect_episode(case: dict[str, Any], mode: str, actor: Any | None = None):
    from haic.oracle_v1 import OracleController

    max_steps = int(case["max_steps"])
    env, base, observation = _make_env(case["track_id"], case["geometry_seed"], max_steps)
    try:
        track_hash = _track_hash(base)
        if (case.get("source_road_centerline_sha256") is not None
                and case["source_road_centerline_sha256"] != track_hash):
            raise DiagnosisError("reset road differs from the source catalog coordinates")
        controller = OracleController(base, target_speed=TARGET_SPEED, avoid_obstacles=True)
        if actor is not None:
            actor.reset(observation)
        trace = _empty_trace()
        initial_observation_sha = _obs_hash(observation)
        snapshots = []
        terminated = truncated = False
        info: dict[str, Any] = {}
        total_reward = 0.0
        for step in range(max_steps):
            state = _state(base, controller)
            if step % 25 == 0:
                snapshots.append((step, np.rint(observation * 255.0).astype(np.uint8)))
            if mode == "oracle":
                action = state["oracle_action"]
            else:
                if actor is None:
                    raise DiagnosisError("policy rollout has no frozen actor")
                action = np.asarray(actor.act(observation), dtype=np.float32)
            if action.shape != (3,) or not np.isfinite(action).all() or not env.action_space.contains(action):
                raise DiagnosisError(f"invalid applied action at step {step}: {action}")
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = _state(base, controller)
            values = {
                "step": step,
                "observation_sha256": _obs_hash(observation),
                "policy_or_oracle_action": action.copy(),
                "oracle_action_at_state": state["oracle_action"].copy(),
                "action_delta": action - state["oracle_action"],
                "pre_position": state["position"],
                "post_position": next_state["position"],
                "pre_velocity": state["velocity"],
                "post_velocity": next_state["velocity"],
                "pre_heading": state["heading"],
                "post_heading": next_state["heading"],
                "pre_speed": state["speed"],
                "post_speed": next_state["speed"],
                "arc_length": state["arc_length"],
                "center_error": state["center_error"],
                "path_error": state["path_error"],
                "curvature": state["curvature"],
                "heading_error": state["heading_error"],
                "waypoint_index": state["waypoint_index"],
                "reward": float(reward),
                "progress": float(info.get("progress", env._calculate_progress())),
                "damage": float(info.get("damage", env.damage.damage)),
                "collision": bool(info.get("collision", False)),
                "off_track_counter": int(env.off_track_counter),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "finished": bool(info.get("finished", False)),
                "retire_reason": str(info.get("retire_reason") or ""),
                "next_observation_sha256": _obs_hash(next_observation),
            }
            for key, value in values.items():
                trace[key].append(value)
            total_reward += float(reward)
            observation = next_observation
            if terminated or truncated:
                break

        arrays = {}
        for key, values in trace.items():
            if key.endswith("sha256") or key == "retire_reason":
                arrays[key] = np.asarray(values, dtype="U64")
            else:
                arrays[key] = np.asarray(values)
        if snapshots:
            arrays["snapshot_steps"] = np.asarray([step for step, _ in snapshots], dtype="<i4")
            arrays["observation_snapshots"] = np.stack([image for _, image in snapshots])
        finish_time = base.finish_time_s
        summary = {
            "mode": mode,
            "track_id": case["track_id"],
            "geometry_seed": case["geometry_seed"],
            "max_steps": max_steps,
            "steps": len(trace["step"]),
            "finished": finish_time is not None,
            "finish_qualified": base.finish_qualified_time_s is not None,
            "finish_time_s": finish_time,
            "progress": float(env._calculate_progress()),
            "damage": float(env.damage.damage),
            "collision_positive_actions": int(sum(trace["collision"])),
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reason": ("finished" if finish_time is not None else info.get("retire_reason")
                       or ("terminated" if terminated else "truncated" if truncated else "runner_max_steps")),
            "initial_observation_sha256": initial_observation_sha,
            "road_centerline_sha256": track_hash,
            "source_road_hash_matches": (case.get("source_road_centerline_sha256") is None
                                         or case["source_road_centerline_sha256"] == track_hash),
        }
        return summary, arrays
    finally:
        env.close()


def _wrapped_angle_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(left - right), np.cos(left - right))


def first_material_divergence(policy: dict[str, np.ndarray], oracle: dict[str, np.ndarray], *,
                              position_m: float = 2.0, heading_rad: float = 0.30,
                              speed_m_s: float = 2.0, sustained: int = 3) -> dict[str, Any] | None:
    count = min(len(policy["post_position"]), len(oracle["post_position"]))
    if count == 0:
        return None
    position_gap = np.linalg.norm(
        policy["post_position"][:count] - oracle["post_position"][:count], axis=1,
    )
    heading_gap = np.abs(_wrapped_angle_delta(policy["post_heading"][:count],
                                              oracle["post_heading"][:count]))
    speed_gap = np.abs(policy["post_speed"][:count] - oracle["post_speed"][:count])
    candidates = []
    for name, values, threshold in (
        ("position", position_gap, position_m),
        ("heading", heading_gap, heading_rad),
        ("speed", speed_gap, speed_m_s),
    ):
        active = values >= threshold
        for start in range(0, count - sustained + 1):
            if bool(np.all(active[start:start + sustained])):
                candidates.append((start, name))
                break
    if not candidates:
        return None
    start, criterion = min(candidates)
    return {
        "step": start,
        "sustained_decisions": sustained,
        "criteria": [criterion],
        "position_gap_m": float(position_gap[start]),
        "heading_gap_rad": float(heading_gap[start]),
        "speed_gap_m_s": float(speed_gap[start]),
    }


def first_failure_precursor(trace: dict[str, np.ndarray], *, finished: bool) -> dict[str, Any]:
    """Find the earliest measured risk event; taxonomy labels are descriptive."""
    count = len(trace["step"])
    candidates: list[tuple[int, str, str]] = []
    for index in range(count):
        speed = float(trace["pre_speed"][index])
        curvature = abs(float(trace["curvature"][index]))
        policy = trace["policy_or_oracle_action"][index]
        teacher = trace["oracle_action_at_state"][index]
        if (curvature >= 0.025 and speed >= TARGET_SPEED + 2.0
                and policy[1] >= 0.35 and policy[2] < 0.05 and teacher[2] >= 0.10):
            candidates.append((index, "curve-entry-overspeed", "speed/action diverged from the oracle brake request in curvature"))
        if (speed >= 2.5 and abs(float(teacher[0])) >= 0.25
                and policy[0] * teacher[0] < 0 and abs(policy[0] - teacher[0]) >= 0.60):
            candidates.append((index, "steering-opposition", "policy steering opposed the oracle on the same policy state"))
        if index >= 2 and np.all(np.abs(trace["heading_error"][index - 2:index + 1]) >= 0.55):
            candidates.append((index - 2, "heading-error", "absolute heading error exceeded 0.55 rad for three decisions"))
        if index >= 1 and np.all(np.abs(trace["center_error"][index - 1:index + 1]) >= 6.0):
            candidates.append((index - 1, "lateral-excursion", "centerline offset exceeded the 6 m road half-width proxy for two decisions"))
        if trace["damage"][index] >= 0.2 and (index == 0 or trace["damage"][index - 1] < trace["damage"][index]):
            candidates.append((index, "contact-damage", "wrapper damage increased before episode termination"))
        if index >= 19:
            low_motion = (
                (trace["pre_speed"][index - 19:index + 1] < 1.5)
                & (np.abs(trace["center_error"][index - 19:index + 1]) < 6.0)
            )
            if np.all(low_motion):
                candidates.append((index - 19, "low-speed-stall", "speed stayed below 1.5 m/s for 20 decisions"))
    if candidates:
        index, label, evidence = min(candidates, key=lambda item: (item[0], item[1]))
        return {"step": int(index), "mode": label, "evidence": evidence,
                "causal": False}
    if not finished and float(np.max(trace["progress"])) >= 0.90:
        return {"step": count - 1, "mode": "late-progress-nonfinish",
                "evidence": "visited-tile progress reached at least 0.90 without a finish", "causal": False}
    return {"step": None, "mode": "unclassified", "evidence": "no predeclared precursor threshold crossed", "causal": False}


def analyze_pair(policy_summary: dict[str, Any], policy_trace: dict[str, np.ndarray],
                 oracle_summary: dict[str, Any], oracle_trace: dict[str, np.ndarray]) -> dict[str, Any]:
    if policy_summary["initial_observation_sha256"] != oracle_summary["initial_observation_sha256"]:
        raise DiagnosisError("policy and oracle did not receive identical reset observations")
    if policy_summary["road_centerline_sha256"] != oracle_summary["road_centerline_sha256"]:
        raise DiagnosisError("policy and oracle reset to different road geometry")
    if not np.allclose(policy_trace["pre_position"][0], oracle_trace["pre_position"][0], atol=1e-9, rtol=0):
        raise DiagnosisError("policy and oracle initial physical positions differ")
    action_deltas = np.abs(policy_trace["action_delta"])
    first_action_gap = int(np.flatnonzero(np.max(action_deltas, axis=1) > 1e-4)[0]) if np.any(action_deltas > 1e-4) else None
    divergence = first_material_divergence(policy_trace, oracle_trace)
    precursor = first_failure_precursor(policy_trace, finished=policy_summary["finished"])
    source_repeat_changed = policy_summary["original_finished"] != policy_summary["finished"]
    return {
        "study": policy_summary["study"],
        "policy_id": policy_summary["policy_id"],
        "actor_sha256": policy_summary["source_actor_sha256"],
        "track_id": policy_summary["track_id"],
        "geometry_seed": policy_summary["geometry_seed"],
        "source_trace_path": policy_summary["source_trace_path"],
        "source_trace_sha256": policy_summary["source_trace_sha256"],
        "geometry_family": policy_summary.get("family"),
        "original_finished": policy_summary["original_finished"],
        "replay_finished": policy_summary["finished"],
        "first_action_linf_delta_from_archived": policy_summary.get("first_action_linf_delta_from_archive"),
        "archived_initial_stack_match": policy_summary.get("archived_initial_stack_match"),
        "oracle_finished": oracle_summary["finished"],
        "replay_outcome_changed_from_archived_episode": source_repeat_changed,
        "policy_steps": policy_summary["steps"],
        "oracle_steps": oracle_summary["steps"],
        "first_material_divergence": divergence,
        "first_policy_oracle_action_gap_on_policy_state": first_action_gap,
        "first_failure_precursor": precursor,
        "policy_progress": policy_summary["progress"],
        "oracle_progress": oracle_summary["progress"],
        "policy_damage": policy_summary["damage"],
        "oracle_damage": oracle_summary["damage"],
        "policy_collision_actions": policy_summary["collision_positive_actions"],
        "oracle_collision_actions": oracle_summary["collision_positive_actions"],
        "common_step_action_delta_mean": np.mean(action_deltas, axis=0).astype(float).tolist(),
        "common_step_action_delta_p95": np.quantile(action_deltas, 0.95, axis=0).astype(float).tolist(),
        "reset_observation_sha256": policy_summary["initial_observation_sha256"],
        "road_centerline_sha256": policy_summary["road_centerline_sha256"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _write_trace(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha_file(path)


def run_diagnosis(output: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    protocol = _read_json(ROOT / PROTOCOL_PATH)
    verify_source_hashes(protocol)
    cases, source_hashes = load_cases()
    actor_paths = {}
    for policy_id in sorted({case["policy_id"] for case in cases}):
        actor_paths[policy_id] = _actor_path(policy_id)[0]
    failures = sum(not row["original_finished"] for row in cases)
    if preflight_only:
        return {
            "status": "preflight_passed_no_environment_resets",
            "selected_policy_episodes": len(cases),
            "historical_failures": failures,
            "unique_cells": len({(row["track_id"], row["geometry_seed"]) for row in cases}),
            "actors": {key: str(value.relative_to(ROOT)) for key, value in actor_paths.items()},
            "source_hashes": source_hashes,
        }
    if output.exists() or not output.parent.is_dir():
        raise DiagnosisError("output must be new and its parent directory must already exist")
    import torch

    from agent import Agent

    torch.set_num_threads(1)
    started = time.monotonic()
    output.mkdir()
    _write_json(output / "metadata.json", {
        "format": "haic-oracle-policy-diagnosis-run-v1",
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": _sha_file(ROOT / PROTOCOL_PATH),
        "runner_sha256": _sha_file(Path(__file__).resolve()),
        "oracle_controller_sha256": _sha_file(ROOT / "haic/oracle_v1/controller.py"),
        "oracle_provenance": _read_json(ROOT / "haic/oracle_v1/PROVENANCE.json"),
        "cohort_contract": COHORT_CONTRACT,
        "analysis_contract": ANALYSIS_CONTRACT,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "resource_contract": "serial rollouts; no learner updates; one torch CPU thread",
        "source_hashes": source_hashes,
        "historical_policy_episodes": len(cases),
        "historical_failures": failures,
    })

    actors = {key: Agent(model_path=str(path)) for key, path in actor_paths.items()}
    policies: dict[str, dict[str, Any]] = {}
    output_files: dict[str, str] = {}
    for index, case in enumerate(cases, 1):
        policy_id = case["policy_id"]
        source_trace_path = _path(case["source_trace_path"])
        with np.load(source_trace_path, allow_pickle=False) as archived:
            action_key = "official_action" if case["study"] == "drq-r6" else "commanded_official_action"
            archived_action = np.asarray(archived[action_key][0], dtype=np.float32)
            archived_stack = np.asarray(archived["initial_stack"]) if "initial_stack" in archived.files else None
        if archived_action.shape != (3,) or not np.isfinite(archived_action).all():
            raise DiagnosisError(f"invalid archived first action: {source_trace_path}")
        summary, trace = _collect_episode(case, "policy", actors[policy_id])
        summary["archived_first_action"] = archived_action.astype(float).tolist()
        summary["current_first_action"] = trace["policy_or_oracle_action"][0].astype(float).tolist()
        summary["first_action_linf_delta_from_archive"] = float(np.max(np.abs(
            archived_action - trace["policy_or_oracle_action"][0]
        )))
        if archived_stack is not None:
            summary["archived_initial_stack_match"] = bool(np.array_equal(
                archived_stack, trace["observation_snapshots"][0],
            ))
        summary.update({key: case[key] for key in (
            "study", "policy_id", "track_id", "geometry_seed", "original_finished",
            "original_steps", "original_reason", "original_max_progress", "original_damage",
            "source_trace_path", "source_trace_sha256", "source_actor_sha256", "family",
        )})
        name = f"{policy_id}-track{case['track_id']}-seed{case['geometry_seed']}"
        trace_path = output / "policies" / f"{name}.npz"
        summary["trace_path"] = trace_path.relative_to(output).as_posix()
        summary["trace_sha256"] = _write_trace(trace_path, trace)
        output_files[summary["trace_path"]] = summary["trace_sha256"]
        policies[name] = {"summary": summary, "trace": trace}
        with (output / "rollouts.jsonl").open("a") as ledger:
            ledger.write(json.dumps({"kind": "policy", "summary": summary}, sort_keys=True) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        print(f"policy {index:02d}/{len(cases)} {name}: "
              f"archived_finish={case['original_finished']} replay_finish={summary['finished']} "
              f"steps={summary['steps']}", flush=True)

    oracle_cells = sorted({(row["track_id"], row["geometry_seed"], row["max_steps"])
                           for row in cases})
    oracles: dict[tuple[int, int, int], dict[str, Any]] = {}
    for index, (track_id, seed, max_steps) in enumerate(oracle_cells, 1):
        case = {"track_id": track_id, "geometry_seed": seed, "max_steps": max_steps}
        summary, trace = _collect_episode(case, "oracle")
        name = f"oracle-track{track_id}-seed{seed}-cap{max_steps}"
        trace_path = output / "oracle" / f"{name}.npz"
        summary["trace_path"] = trace_path.relative_to(output).as_posix()
        summary["trace_sha256"] = _write_trace(trace_path, trace)
        output_files[summary["trace_path"]] = summary["trace_sha256"]
        oracles[(track_id, seed, max_steps)] = {"summary": summary, "trace": trace}
        with (output / "rollouts.jsonl").open("a") as ledger:
            ledger.write(json.dumps({"kind": "oracle", "summary": summary}, sort_keys=True) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
        print(f"oracle {index:02d}/{len(oracle_cells)} {name}: "
              f"finish={summary['finished']} steps={summary['steps']}", flush=True)

    comparisons = []
    for name, policy in policies.items():
        case = policy["summary"]
        oracle = oracles[(case["track_id"], case["geometry_seed"], case["max_steps"])]
        comparisons.append(analyze_pair(case, policy["trace"],
                                        oracle["summary"], oracle["trace"]))
    taxonomy = Counter(row["first_failure_precursor"]["mode"] for row in comparisons
                       if not row["replay_finished"])
    per_study = {}
    for study in ("drq-r6", "rlpd-g0"):
        rows = [row for row in comparisons if row["study"] == study]
        per_study[study] = {
            "episodes": len(rows),
            "historical_failures": sum(not row["original_finished"] for row in rows),
            "current_replay_failures": sum(not row["replay_finished"] for row in rows),
            "oracle_finishes_on_matched_cells": sum(row["oracle_finished"] for row in rows),
            "archived_outcome_changes": sum(row["replay_outcome_changed_from_archived_episode"] for row in rows),
            "taxonomy_current_failures": dict(sorted(Counter(
                row["first_failure_precursor"]["mode"] for row in rows
                if not row["replay_finished"]
            ).items())),
        }
    result = {
        "format": "haic-oracle-policy-diagnosis-result-v1",
        "status": "complete_descriptive_consumed_train_diagnostic",
        "diagnostic_only": True,
        "official_score": False,
        "fresh_generalization_claim": False,
        "policy_episode_count": len(comparisons),
        "historical_failure_count": sum(not row["original_finished"] for row in comparisons),
        "current_replay_failure_count": sum(not row["replay_finished"] for row in comparisons),
        "unique_oracle_cells": len(oracles),
        "cohort_contract": COHORT_CONTRACT,
        "analysis_contract": ANALYSIS_CONTRACT,
        "source_hashes": source_hashes,
        "per_study": per_study,
        "first_precursor_taxonomy": dict(sorted(taxonomy.items())),
        "thresholds": {
            "physical_divergence": {"position_gap_m": 2.0, "heading_gap_rad": 0.30,
                                    "speed_gap_m_s": 2.0, "sustained_decisions": 3},
            "precursor": {"curve_abs_curvature": 0.025, "overspeed_m_s": 14.0,
                           "heading_error_rad": 0.55, "centerline_offset_m": 6.0},
        },
        "limitations": [
            "All diagnostic geometries are previously consumed track-1 TRAIN or TRAIN-DIAGNOSTIC cells.",
            "DrQ r6 source archives contain per-decision actions/progress/damage but no pixels or dense pose; current-runtime frozen-actor reruns supply these signals.",
            "Some RLPD same-cell cross-runtime reruns previously changed after decision 33; this run therefore preserves archived and current replay outcomes separately.",
            "Separate policy and oracle rollouts share reset observation and road hash, not simulator hidden state after their first different actions.",
            "Threshold taxonomy labels first measured precursors, not proven causal mechanisms; cell repeats and road families are correlated.",
            "Oracle privileged state and outputs are diagnostic only and must not be used as a submitted Agent.",
        ],
        "comparisons": comparisons,
        "output_files_sha256": output_files,
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate frozen sources and cohort without constructing/resetting an environment")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        result = run_diagnosis(output, preflight_only=args.preflight_only)
    except Exception as error:
        print(f"diagnosis failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps({key: result[key] for key in (
        "status", "selected_policy_episodes", "policy_episode_count",
        "historical_failures", "current_replay_failure_count", "unique_cells",
        "unique_oracle_cells",
    ) if key in result}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
