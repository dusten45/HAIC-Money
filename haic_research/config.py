"""Load and validate the HAIC research harness configuration."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .models import WorkflowState


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
PROFILE_ARGUMENTS = {
    "train_policy": {"output", "train-only-site-map-split", "defer-tune", "total-steps", "updates", "seed", "max-decisions", "learning-rate", "resume"},
    "evaluate_closed_loop": {"output", "policy-checkpoint", "dynamics-checkpoint", "site-map-split", "plan-budget", "max-decisions"},
    "package_submission": {"output", "policy-checkpoint", "dynamics-checkpoint", "source-root", "evaluation-summary", "smoke-test"},
    "benchmark_corridor_diagnostic": {"output", "track", "site-map", "profile", "site-map-split", "site-group", "site-limit", "max-decisions"},
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


def load_config(root: Path, *, validate: bool = True) -> HarnessConfig:
    """Load config, rejecting invalid settings unless collecting validation issues."""
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
    issues = validate_config(config) if validate else []
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
    expected_ties = ("median_finished_lap_ms", "mean_incomplete_progress", "p90_finished_lap_ms",
                     "collisions", "damage", "act_latency_p95_ms")
    if config.tie_breakers != expected_ties:
        add("metrics.tie_breakers", "metrics.tie_breakers must use the exact completion-first tie-breaker order")
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
    raw_paths = config.raw.get("paths", {})
    for name, path in roots.items():
        resolved = path.resolve()
        raw_value = raw_paths.get(name.removeprefix("paths.")) if isinstance(raw_paths, dict) else None
        lexical = None
        if isinstance(raw_value, str) and not Path(raw_value).is_absolute():
            lexical = Path(os.path.abspath(root / raw_value))
        if not _inside(resolved, root) or resolved == root:
            add(name, f"{name} must stay inside the project")
        if any(
            _inside(resolved, old.resolve()) or (lexical is not None and _inside(lexical, old))
            for old in forbidden
        ):
            add(name, f"{name} must not use a legacy output root")
    if _inside(config.run_root.resolve(), config.artifact_root.resolve()) or _inside(config.artifact_root.resolve(), config.run_root.resolve()):
        add("paths", "run_root and artifact_root must be distinct and non-nested")
    for path in config.legacy_paths:
        if not _inside(path.resolve(), root):
            add("paths.legacy_paths", "legacy paths must stay inside the project")

    def check_relative(value: Any, name: str) -> None:
        if not isinstance(value, str) or not value or Path(value).is_absolute() or not _inside((root / value).resolve(), root):
            add(name, f"{name} must be a project-relative path inside the project")

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
        location = f"commands.profiles.{profile_id}"
        timeout = profile.get("timeout_seconds")
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            add(location, "command timeout_seconds must be finite and positive")
        schema = profile.get("arguments")
        if not isinstance(schema, dict) or not schema or "output" not in schema:
            add(location, "command arguments must contain a typed output schema")
            schema = {}
        for name, spec in schema.items():
            if name not in PROFILE_ARGUMENTS[profile_id] or not isinstance(spec, dict):
                add(location, f"unsupported command argument: {name}")
                continue
            kind = spec.get("type")
            if not isinstance(kind, str):
                add(location, f"invalid typed argument schema: {name}")
                continue
            if (spec.get("flag") != "--" + name or kind not in {"int", "float", "bool", "str", "path", "track"}
                    or type(spec.get("required")) is not bool or type(spec.get("repeated", False)) is not bool):
                add(location, f"invalid typed argument schema: {name}")
            if set(spec) - {"flag", "type", "required", "repeated", "choices", "minimum", "maximum", "path_role"}:
                add(location, f"unknown argument schema fields: {name}")
            role = spec.get("path_role")
            if role is not None and not isinstance(role, str):
                add(location, f"invalid path role: {name}")
                continue
            if (kind == "path" and role not in {"input_file", "input_directory", "output_file", "output_directory"}) or (kind != "path" and role is not None):
                add(location, f"invalid path role: {name}")
            if (name == "output" and role not in {"output_file", "output_directory"}) or (name != "output" and role in {"output_file", "output_directory"}):
                add(location, "only --output may declare an output path")
            path_inputs = {"train-only-site-map-split", "resume", "policy-checkpoint", "dynamics-checkpoint", "site-map-split", "site-map", "evaluation-summary", "source-root", "output"}
            expected_type = ("path" if name in path_inputs else "bool" if name in {"defer-tune", "smoke-test"}
                             else "float" if name in {"learning-rate", "plan-budget"}
                             else "str" if name in {"profile", "site-group"} else "track" if name == "track" else "int")
            expected_role = ("output_file" if profile_id in {"package_submission", "benchmark_corridor_diagnostic"} else "output_directory") if name == "output" else "input_directory" if name == "source-root" else "input_file" if name in path_inputs else None
            repeated = profile_id == "benchmark_corridor_diagnostic" and name in {"track", "site-map", "profile"}
            if kind != expected_type or role != expected_role or spec.get("repeated", False) != repeated:
                add(location, f"schema differs from the real argparse contract: {name}")
            if name == "defer-tune" and spec.get("choices") != [True]:
                add(location, "TRAIN-only profile requires --defer-tune=true")
            choices = spec.get("choices")
            if choices is not None and (not isinstance(choices, list) or not choices):
                add(location, f"choices must be a nonempty list: {name}")
            permitted = {"site-group": {"train", "tune", "held_out"},
                         "profile": {"safe", "fast", "fast_plus", "sprint_guarded", "race", "slow_obstacle", "strong_avoid"}}
            if name in permitted and (not isinstance(choices, list) or not choices
                                      or any(not isinstance(item, str) or item not in permitted[name] for item in choices)):
                add(location, f"choices differ from real argparse values: {name}")
            if kind in {"int", "float"}:
                floor = 0 if name == "seed" else 1 if kind == "int" else 1e-12
                minimum = spec.get("minimum")
                if type(minimum) not in {int, float} or not math.isfinite(minimum) or minimum < floor:
                    add(location, f"argument requires an explicit valid lower bound: {name}")
            for bound in ("minimum", "maximum"):
                value = spec.get(bound)
                if bound in spec and (kind not in {"int", "float"} or type(value) not in {int, float} or not math.isfinite(value)):
                    add(location, f"invalid numeric {bound}: {name}")
            if (type(spec.get("minimum")) in {int, float} and type(spec.get("maximum")) in {int, float}
                    and spec["minimum"] > spec["maximum"]):
                add(location, f"minimum exceeds maximum: {name}")
        required_inputs = {"policy-checkpoint", "dynamics-checkpoint"} if profile_id in {"evaluate_closed_loop", "package_submission"} else set()
        if profile_id == "train_policy":
            required_inputs = {"train-only-site-map-split", "defer-tune", "total-steps"}
        if any(not isinstance(schema.get(name), dict) or schema[name].get("required") is not True for name in required_inputs):
            add(location, "command is missing required explicit input arguments")
        output_name = profile.get("output_name")
        if (not isinstance(output_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", output_name)
                or output_name.endswith('.')):
            add(location, "output_name must be a safe single path component")
        if profile.get("requires_approval") != "execution":
            add(location, "command must require execution approval")
        if profile_id == "benchmark_corridor_diagnostic" and profile.get("sota_eligible") is not False:
            add(location, "benchmark diagnostics must be ineligible for SOTA")
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
    workflow = config.raw.get("workflow", {})
    if not isinstance(workflow, Mapping) or workflow.get("states") != [state.value for state in WorkflowState]:
        add("workflow.states", "workflow states must include every registered state in order")
    required_values = {
        "project": {"id": "haic-money", "scope": "haic-only"},
        "workflow": {
            "default_mode": "plan-only",
            "separate_approval_stages": ["design", "implementation", "execution"],
            "gates": ["rule_compliance", "mechanism_activation", "competitive_or_product_outcome"],
        },
        "search": {"threshold_sweep_requires_mechanism_activation": True},
        "metrics": {"official_score_is_separate": True},
        "splits": {"protected_ids": ["confirmation", "blind"], "blind_tuning_allowed": False,
                   "consumed_confirmation_reusable_as_fresh": False},
    }
    for section, fields in required_values.items():
        values = config.raw.get(section, {})
        for key, expected in fields.items():
            actual = values.get(key) if isinstance(values, Mapping) else None
            if actual != expected or (type(expected) is bool and actual is not expected):
                add(f"{section}.{key}", f"{section}.{key} must be {expected!r}")
    project = config.raw.get("project", {})
    name = project.get("name") if isinstance(project, Mapping) else None
    if not isinstance(name, str) or not name.strip() or name.strip().lower() in {"unknown", "default", "todo", "tbd"}:
        add("project.name", "project.name must identify the HAIC project")
    expected_sources = (
        {"precedence": 1, "id": "competition_site", "url": "https://ships-duo-ethical-saver.trycloudflare.com/"},
        {"precedence": 2, "id": "participants_repository", "url": "https://github.com/2026-HAIC/Participants"},
        {"precedence": 3, "id": "local_source_mirror", "path": "docs/sources/official-participants/README.md",
         "license_path": "docs/sources/official-participants/LICENSE",
         "source_commit": "1c11db8afc2fbfcfb610672b7ee0ecd122c97741"},
        {"precedence": 4, "id": "historical_evidence", "path": "COMPETITION_INFO.md", "related_paths": ["RESULTS.md"]},
    )
    if not isinstance(sources, list) or len(sources) != len(expected_sources):
        add("official_sources", "official_sources must register exactly the four source-precedence entries")
    else:
        for index, expected in enumerate(expected_sources):
            actual = sources[index]
            for key, value in expected.items():
                if (not isinstance(actual, Mapping) or actual.get(key) != value
                        or (key == "precedence" and type(actual.get(key)) is not int)):
                    add(f"official_sources[{index}].{key}", f"official_sources[{index}].{key} must be {value!r}")

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
