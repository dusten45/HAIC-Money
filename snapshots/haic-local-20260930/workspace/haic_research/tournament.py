"""Validated candidate batches for completion-first local tournaments."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Mapping


class TournamentManifestError(ValueError):
    """A registered tournament candidate manifest is incomplete or unsafe."""


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLANNER_KEYS = {"horizon", "population", "iterations", "candidate_batch_size", "uncertainty_cost"}


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TournamentManifestError(f"{name} must be a safe nonempty identifier")
    return value


def _relative_file(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TournamentManifestError(f"{name} must be a project-relative POSIX file path")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts or ":" in value
            or path.as_posix() != value or any(ord(char) < 32 for char in value)):
        raise TournamentManifestError(f"{name} must stay inside the project")
    return path.as_posix()


def validate_candidate_manifest(payload: object, split_group: str) -> dict[str, Any]:
    """Validate a tune batch or a single held-out winner against the 4/8/2 rule."""
    if split_group not in {"tune", "held_out"}:
        raise TournamentManifestError("completion tournaments accept only tune or held_out cells")
    if not isinstance(payload, Mapping):
        raise TournamentManifestError("candidate manifest must be an object")
    expected_fields = {"schema_version", "comparison_id", "control_id", "stage", "candidates"}
    if split_group == "held_out":
        expected_fields.add("tune_source")
    if set(payload) != expected_fields:
        raise TournamentManifestError("candidate manifest has unknown or missing fields")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 2:
        raise TournamentManifestError("candidate manifest requires schema_version 2")
    stage = payload.get("stage")
    if stage != split_group:
        raise TournamentManifestError("candidate manifest stage must match the selected split group")
    comparison_id = _identifier(payload.get("comparison_id"), "comparison_id")
    control_id = _identifier(payload.get("control_id"), "control_id")
    tune_source = None
    if stage == "held_out":
        raw_source = payload.get("tune_source")
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            "run_id", "plan_hash", "summary_ref", "summary_sha256"
        }:
            raise TournamentManifestError("held_out requires an immutable tune_source reference")
        plan_hash = raw_source.get("plan_hash")
        summary_hash = raw_source.get("summary_sha256")
        if not isinstance(plan_hash, str) or not _SHA256.fullmatch(plan_hash):
            raise TournamentManifestError("tune_source.plan_hash must be a lowercase SHA-256")
        if not isinstance(summary_hash, str) or not _SHA256.fullmatch(summary_hash):
            raise TournamentManifestError("tune_source.summary_sha256 must be a lowercase SHA-256")
        tune_source = {
            "run_id": _identifier(raw_source.get("run_id"), "tune_source.run_id"),
            "plan_hash": plan_hash,
            "summary_ref": _relative_file(raw_source.get("summary_ref"), "tune_source.summary_ref"),
            "summary_sha256": summary_hash,
        }
        if not tune_source["summary_ref"].endswith("/summary.json"):
            raise TournamentManifestError("tune_source.summary_ref must point to a tournament summary.json")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TournamentManifestError("candidates must be a nonempty list")
    if stage == "tune" and not 4 <= len(candidates) <= 8:
        raise TournamentManifestError("tune tournament requires four to eight candidates")
    if stage == "held_out" and len(candidates) != 1:
        raise TournamentManifestError("held_out tournament must contain only the frozen tune winner")

    normalized = []
    ids: list[str] = []
    directions: list[str] = []
    for index, raw in enumerate(candidates):
        where = f"candidates[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != {
            "candidate_id", "direction", "hypothesis_id", "hypothesis_ref", "mode",
            "policy_checkpoint", "dynamics_checkpoint", "planner_settings",
        }:
            raise TournamentManifestError(f"{where} has unknown or missing fields")
        candidate_id = _identifier(raw.get("candidate_id"), f"{where}.candidate_id")
        hypothesis_id = _identifier(raw.get("hypothesis_id"), f"{where}.hypothesis_id")
        direction = raw.get("direction")
        if not isinstance(direction, str) or not direction.strip() or len(direction) > 160:
            raise TournamentManifestError(f"{where}.direction must be a concise nonempty label")
        hypothesis_ref = _relative_file(raw.get("hypothesis_ref"), f"{where}.hypothesis_ref")
        if not hypothesis_ref.startswith("docs/") or PurePosixPath(hypothesis_ref).suffix.lower() != ".md":
            raise TournamentManifestError(f"{where}.hypothesis_ref must point to current project documentation")
        mode = raw.get("mode")
        if not isinstance(mode, str) or mode not in {"ppo_only", "ppo_cem"}:
            raise TournamentManifestError(f"{where}.mode must be ppo_only or ppo_cem")
        policy_checkpoint = _relative_file(raw.get("policy_checkpoint"), f"{where}.policy_checkpoint")
        if PurePosixPath(policy_checkpoint).suffix.lower() not in {".pt", ".pth"}:
            raise TournamentManifestError(f"{where}.policy_checkpoint must be a .pt or .pth file")
        dynamics_value = raw.get("dynamics_checkpoint")
        dynamics_checkpoint = (
            _relative_file(dynamics_value, f"{where}.dynamics_checkpoint")
            if dynamics_value is not None else None
        )
        if dynamics_checkpoint is not None and PurePosixPath(dynamics_checkpoint).suffix.lower() not in {".pt", ".pth"}:
            raise TournamentManifestError(f"{where}.dynamics_checkpoint must be a .pt or .pth file")
        planner_settings = raw.get("planner_settings")
        if not isinstance(planner_settings, Mapping):
            raise TournamentManifestError(f"{where}.planner_settings must be an object")
        if mode == "ppo_only":
            if dynamics_checkpoint is not None or planner_settings:
                raise TournamentManifestError(f"{where} PPO-only arm cannot declare planner inputs")
            normalized_settings = {}
        else:
            if dynamics_checkpoint is None or set(planner_settings) != _PLANNER_KEYS:
                raise TournamentManifestError(f"{where} PPO+CEM arm requires dynamics and all fixed planner settings")
            normalized_settings = dict(planner_settings)
            for key, minimum, maximum in (
                ("horizon", 1, 32), ("population", 2, 1024),
                ("iterations", 1, 64), ("candidate_batch_size", 1, 1024),
            ):
                value = normalized_settings[key]
                if type(value) is not int or not minimum <= value <= maximum:
                    raise TournamentManifestError(f"{where}.planner_settings.{key} is outside its allowed range")
            cost = normalized_settings["uncertainty_cost"]
            if type(cost) not in {int, float} or not math.isfinite(cost) or not 0.0 <= cost <= 100.0:
                raise TournamentManifestError(f"{where}.planner_settings.uncertainty_cost is outside its allowed range")
            normalized_settings["uncertainty_cost"] = float(cost)
        ids.append(candidate_id)
        directions.append(direction.strip())
        normalized.append({
            "candidate_id": candidate_id,
            "direction": direction.strip(),
            "hypothesis_id": hypothesis_id,
            "hypothesis_ref": hypothesis_ref,
            "mode": mode,
            "policy_checkpoint": policy_checkpoint,
            "dynamics_checkpoint": dynamics_checkpoint,
            "planner_settings": normalized_settings,
        })
    if (control_id in ids or len(set(ids)) != len(ids)
            or len({item["hypothesis_id"] for item in normalized}) != len(normalized)):
        raise TournamentManifestError("candidate and hypothesis identifiers must be unique")
    if stage == "tune":
        counts = Counter(directions)
        if len(counts) < 4 or any(count > 2 for count in counts.values()):
            raise TournamentManifestError("tune batch must satisfy at least four directions and at most two per direction")
    return {
        "schema_version": 2,
        "comparison_id": comparison_id,
        "control_id": control_id,
        "stage": stage,
        "candidates": normalized,
        **({"tune_source": tune_source} if tune_source is not None else {}),
    }


def referenced_files(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every candidate, hypothesis, and frozen tune-evidence file to fingerprint."""
    paths: list[str] = []
    for candidate in manifest["candidates"]:
        paths.extend((candidate["hypothesis_ref"], candidate["policy_checkpoint"]))
        if candidate["dynamics_checkpoint"] is not None:
            paths.append(candidate["dynamics_checkpoint"])
    if manifest["stage"] == "held_out":
        paths.append(manifest["tune_source"]["summary_ref"])
    return tuple(paths)
