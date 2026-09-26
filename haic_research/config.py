"""Load and validate the HAIC research harness configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


METRICS = frozenset({
    "completion_rate", "median_finished_lap_ms", "mean_incomplete_progress",
    "p90_finished_lap_ms", "collisions", "damage", "act_latency_p95_ms",
})
PROFILE_MODULES = {
    "train_policy": "training.train_policy",
    "evaluate_closed_loop": "training.evaluate_closed_loop",
    "package_submission": "training.package_submission",
    "benchmark_corridor_diagnostic": "training.benchmark_corridor",
}
REQUIRED_SPLITS = frozenset({"train", "tune", "held_out", "confirmation", "blind", "official"})
SECRET_KEY = re.compile(r"(?:secret|password|passwd|token|credential|api[_-]?key|private[_-]?key)", re.I)


class ConfigError(ValueError):
    """The configuration cannot be loaded safely."""


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    message: str


@dataclass(frozen=True)
class HarnessConfig:
    repo_root: Path
    run_root: Path
    artifact_root: Path
    legacy_paths: tuple[Path, ...]
    primary_metric: str
    tie_breakers: tuple[str, ...]
    search_limits: tuple[int, int, int]
    pivot_after: int
    split_ids: tuple[str, ...]
    command_profiles: Mapping[str, Mapping[str, Any]]
    raw: Mapping[str, Any]


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key: {location}.{key}" if location else f"missing required key: {key}")
    return mapping[key]


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(mapping, key, "")
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be an object")
    return value


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ConfigError(f"{name} must be a project-relative path")
    return (root / value).resolve()


def load_config(root: Path) -> HarnessConfig:
    """Read harness.config.json from *root* and reject invalid settings."""
    root = Path(root).resolve()
    try:
        raw = json.loads((root / "harness.config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read harness.config.json: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration must be an object")
    for key in ("schema_version", "project", "paths", "official_sources", "workflow", "search", "metrics", "splits", "commands"):
        _required(raw, key, "")
    paths = _section(raw, "paths")
    workflow = _section(raw, "workflow")
    search = _section(raw, "search")
    metrics = _section(raw, "metrics")
    splits = _section(raw, "splits")
    commands = _section(raw, "commands")
    for key in ("project", "official_sources"):
        if not isinstance(raw[key], dict if key == "project" else list):
            raise ConfigError(f"{key} has an invalid type")
    legacy = _required(paths, "legacy_paths", "paths")
    ties = _required(metrics, "tie_breakers", "metrics")
    ids = _required(splits, "ids", "splits")
    profiles = _required(commands, "profiles", "commands")
    if not isinstance(legacy, list) or not isinstance(ties, list) or not isinstance(ids, list) or not isinstance(profiles, dict):
        raise ConfigError("legacy_paths, tie_breakers, ids, or profiles has an invalid type")
    limits = tuple(_required(search, key, "search") for key in (
        "min_independent_directions", "max_candidates_per_batch", "max_candidates_per_direction"
    ))
    config = HarnessConfig(
        repo_root=root,
        run_root=_resolve_path(root, _required(paths, "run_root", "paths"), "paths.run_root"),
        artifact_root=_resolve_path(root, _required(paths, "artifact_root", "paths"), "paths.artifact_root"),
        legacy_paths=tuple(_resolve_path(root, item, "paths.legacy_paths") for item in legacy),
        primary_metric=_required(metrics, "primary", "metrics"),
        tie_breakers=tuple(ties),
        search_limits=limits,
        pivot_after=_required(workflow, "pivot_after_valid_non_improving_cycles", "workflow"),
        split_ids=tuple(ids),
        command_profiles=profiles,
        raw=raw,
    )
    issues = validate_config(config)
    if issues:
        raise ConfigError("; ".join(issue.message for issue in issues))
    return config


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_config(config: HarnessConfig) -> list[ConfigIssue]:
    """Return every detected policy violation without reading legacy data."""
    issues: list[ConfigIssue] = []

    def add(path: str, message: str) -> None:
        issues.append(ConfigIssue(path, message))

    root = config.repo_root.resolve()
    if config.raw.get("schema_version") != 1 or type(config.raw.get("schema_version")) is not int:
        add("schema_version", "schema_version must be 1")
    if config.primary_metric != "completion_rate":
        add("metrics.primary", "metrics.primary must be completion_rate")
    for metric in config.tie_breakers:
        if not isinstance(metric, str) or metric not in METRICS or metric == "completion_rate":
            add("metrics.tie_breakers", f"unknown tie-breaker metric: {metric}")
    named_ties = [metric for metric in config.tie_breakers if isinstance(metric, str)]
    if len(set(named_ties)) != len(named_ties):
        add("metrics.tie_breakers", "metrics.tie_breakers contains duplicates")
    if config.search_limits != (4, 8, 2) or any(type(value) is not int for value in config.search_limits):
        add("search", "search limits must be 4/8/2")
    if config.pivot_after != 3 or type(config.pivot_after) is not int:
        add("workflow.pivot_after_valid_non_improving_cycles", "pivot count must be 3")
    if any(not isinstance(item, str) for item in config.split_ids) or set(item for item in config.split_ids if isinstance(item, str)) != REQUIRED_SPLITS or len(config.split_ids) != len(REQUIRED_SPLITS):
        add("splits.ids", "splits.ids must register train, tune, held_out, confirmation, blind, and official")

    roots = {"paths.run_root": config.run_root, "paths.artifact_root": config.artifact_root}
    forbidden = (root / "artifacts/haic", root / "submissions")
    for name, path in roots.items():
        resolved = path.resolve()
        if not _inside(resolved, root) or resolved == root:
            add(name, f"{name} must stay inside the project")
        if any(_inside(resolved, old) for old in forbidden):
            add(name, f"{name} must not use a legacy output root")
    if _inside(config.run_root.resolve(), config.artifact_root.resolve()) or _inside(config.artifact_root.resolve(), config.run_root.resolve()):
        add("paths", "run_root and artifact_root must be distinct and non-nested")
    for path in config.legacy_paths:
        if not _inside(path.resolve(), root):
            add("paths.legacy_paths", "legacy paths must stay inside the project")

    def check_relative(value: Any, name: str) -> None:
        if not isinstance(value, str) or not value or Path(value).is_absolute() or not _inside((root / value).resolve(), root):
            add(name, f"{name} must be a project-relative path inside the project")

    raw_paths = config.raw.get("paths", {})
    if isinstance(raw_paths, dict):
        check_relative(raw_paths.get("pdf_root"), "paths.pdf_root")
    sources = config.raw.get("official_sources", [])
    if isinstance(sources, list):
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                add(f"official_sources[{index}]", "official source must be an object")
                continue
            for key in ("path", "license_path"):
                if key in source:
                    check_relative(source[key], f"official_sources[{index}].{key}")
            if "related_paths" in source:
                related = source["related_paths"]
                if not isinstance(related, list):
                    add(f"official_sources[{index}].related_paths", "related_paths must be a list")
                else:
                    for item in related:
                        check_relative(item, f"official_sources[{index}].related_paths")

    for profile_id, profile in config.command_profiles.items():
        if profile_id not in PROFILE_MODULES or not isinstance(profile, dict) or profile.get("module") != PROFILE_MODULES.get(profile_id):
            add(f"commands.profiles.{profile_id}", f"unregistered command: {profile_id}")
            continue
        output = profile.get("output_root")
        if not isinstance(output, str) or not output or Path(output).is_absolute():
            add(f"commands.profiles.{profile_id}.output_root", "command output_root must be project-relative")
        else:
            output_path = (root / output).resolve()
            if output_path != config.artifact_root.resolve():
                add(f"commands.profiles.{profile_id}.output_root", "command output_root must equal artifact_root")
    if set(config.command_profiles) != set(PROFILE_MODULES):
        add("commands.profiles", "commands.profiles must register only the four HAIC local operations")
    commands = config.raw.get("commands", {})
    if isinstance(commands, dict) and (commands.get("default_mode") != "plan-only" or commands.get("allow_shell") is not False):
        add("commands", "commands must default to plan-only with shell disabled")

    def inspect_keys(value: Any, location: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                name = f"{location}.{key}" if location else str(key)
                if SECRET_KEY.search(str(key)):
                    add(name, f"secret-like config key: {name}")
                inspect_keys(child, name)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect_keys(child, f"{location}[{index}]")

    inspect_keys(config.raw)
    return issues
