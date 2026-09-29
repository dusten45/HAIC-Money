"""Typed local HAIC commands bound to immutable plans and current-run approvals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .config import HarnessConfig, validate_config
from .models import Approval, GateStatus, Hypothesis, IntegrationReport, RunEvent, RunManifest, WorkflowState
from .records import (
    RecordError, RunTransaction, _canonical, _identifier, _json_value, _typed, _open, _read_json,
    create_run, run_transaction,
)
from .state import ApprovalError, TransitionError, transition
from .tournament import (
    TournamentManifestError,
    referenced_files,
    validate_candidate_manifest,
)


class CommandError(ValueError):
    """A command, plan, or run lifecycle is unsafe or inconsistent."""


class UnknownProfileError(CommandError):
    """Only the configured registered local operations may be used."""


@dataclass(frozen=True)
class CommandProfile:
    profile_id: str
    module: str
    allowed_args: Mapping[str, Mapping[str, object]]
    required_args: tuple[str, ...]
    timeout_seconds: float
    output_root: Path


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    succeeded: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


@dataclass(frozen=True)
class RunHistory:
    state: WorkflowState
    approvals: tuple[Approval, ...]
    execution_started: bool
    execution_finished: bool
    execution_succeeded: bool
    outcome: WorkflowState | None
    released: bool


_STAGES = {
    "design": (WorkflowState.DESIGN_PENDING_APPROVAL, WorkflowState.EXECUTE_PENDING_APPROVAL),
    # Retained only to replay or finish plans created under the former three-stage contract.
    "implementation": (WorkflowState.IMPLEMENT_PENDING_APPROVAL, WorkflowState.EXECUTE_PENDING_APPROVAL),
    "execution": (WorkflowState.EXECUTE_PENDING_APPROVAL, WorkflowState.EVALUATE),
}
_INITIAL = (WorkflowState.STOPPED, WorkflowState.DISCOVER, WorkflowState.HYPOTHESIZE,
            WorkflowState.DESIGN_PENDING_APPROVAL)
_OUTCOMES = frozenset({WorkflowState.ADVANCE, WorkflowState.REJECT, WorkflowState.REVISE, WorkflowState.PIVOT})
_EXECUTION_PLAN_SCHEMA_VERSION = 3
_SOURCE_POLICY = {
    "version": 2,
    "nonrecursive_python_directories": ["training", "haic_agent", "core", "core/vendor", "haic_research"],
    "root_files": ["agent.py", "env_wrapper.py", "damage.py"],
    "package_inference_files": ["agent.py", "haic_agent/__init__.py", "haic_agent/observation.py",
                                "haic_agent/pixel_features.py", "haic_agent/networks.py", "haic_agent/dynamics.py",
                                "haic_agent/planner.py", "haic_agent/runtime_config.py"],
}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CommandError(f"invalid plan JSON: {exc}") from exc


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _source_bytes(config: HarnessConfig, path: Path, *, role: str = "input_file") -> str:
    """Hash exact local file bytes through no-redirect/single-link checks."""
    checked = Path(_path(config, path, role))
    # records._open checks identity again after opening and does not follow links.
    with _open(checked, "r") as stream:
        digest = hashlib.sha256()
        while chunk := stream.buffer.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()


def _source_identity(config: HarnessConfig, profile_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Inventory concrete executable and harness code directories and named source files.

    It deliberately includes untracked local .py files and observes additions.
    It never walks subdirectories, maps, run roots, or installed packages.
    """
    hashes = {}
    directories = {}
    for relative in _SOURCE_POLICY["nonrecursive_python_directories"]:
        directory = config.repo_root / relative
        try:
            checked = _canonical(directory)
        except RecordError as exc:
            raise CommandError(str(exc)) from exc
        directories[relative] = checked.is_dir()
        if not checked.exists():
            continue
        if not checked.is_dir():
            raise CommandError("source inventory location must be a directory")
        for path in sorted(checked.iterdir()):
            if path.suffix.lower() == ".py":
                hashes[path.relative_to(config.repo_root).as_posix()] = _source_bytes(config, path)
    for relative in _SOURCE_POLICY["root_files"]:
        path = config.repo_root / relative
        try:
            checked = _canonical(path)
        except RecordError as exc:
            raise CommandError(str(exc)) from exc
        hashes[relative] = _source_bytes(config, checked) if checked.exists() else None
    module_path = config.command_profiles[profile_id]["module"].replace(".", "/") + ".py"
    if module_path not in hashes:
        raise CommandError("registered executable module source is missing")
    packaged = {}
    if profile_id in {"package_submission", "package_stable_risk", "package_speed_budget"}:
        source_root = Path(arguments.get("source-root", config.repo_root))
        if source_root != config.repo_root:
            for relative in _SOURCE_POLICY["package_inference_files"]:
                path = source_root / relative
                packaged[path.relative_to(config.repo_root).as_posix()] = _source_bytes(config, path)
    return dict(policy=_SOURCE_POLICY, directories=directories, files=hashes, package_sources=packaged,
                interpreter={"executable": str(Path(sys.executable).resolve()), "version": sys.version,
                             "implementation": sys.implementation.name, "cache_tag": sys.implementation.cache_tag})


def _input_identity(config: HarnessConfig, profile_id: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Hash explicitly named input files and map files referenced by named splits."""
    profile = config.command_profiles[profile_id]
    hashes = {}
    for name, spec in profile["arguments"].items():
        if name not in arguments or spec.get("path_role") not in {"input_file", "legacy_checkpoint_file"}:
            continue
        value = arguments[name]
        values = value if spec.get("repeated", False) else (value,)
        for index, raw_path in enumerate(values):
            checked = Path(_path(config, raw_path, spec["path_role"]))
            relative = checked.relative_to(config.repo_root).as_posix()
            key = f"{name}[{index}]" if spec.get("repeated", False) else name
            hashes[key] = {"path": relative, "sha256": _source_bytes(config, checked, role=spec["path_role"])}
            if name in {"train-only-site-map-split", "site-map-split"}:
                split_group = (
                    str(arguments.get("split-group"))
                    if profile_id == "evaluate_closed_loop" and "candidate-manifest" in arguments
                    else None
                )
                map_paths = _validate_split_references(
                    config, checked, train_only=name == "train-only-site-map-split", group=split_group)
                hashes[key]["maps"] = [
                    {"path": map_path.relative_to(config.repo_root).as_posix(),
                     "sha256": _source_bytes(config, map_path)}
                    for map_path in map_paths
                ]
            if name == "candidate-manifest":
                try:
                    candidate_payload = _read_json(checked)
                    normalized_manifest = validate_candidate_manifest(
                        candidate_payload, str(arguments.get("split-group", ""))
                    )
                except (RecordError, TournamentManifestError) as exc:
                    raise CommandError(f"invalid candidate tournament manifest: {exc}") from exc
                references = []
                if normalized_manifest["stage"] == "held_out":
                    source = normalized_manifest["tune_source"]
                    summary_path = Path(_path(config, source["summary_ref"], "input_file"))
                    expected_summary = (
                        config.artifact_root / source["run_id"]
                        / config.command_profiles["evaluate_closed_loop"]["output_name"]
                        / "summary.json"
                    )
                    if summary_path != expected_summary:
                        raise CommandError("tune source summary must be the registered run's evaluation summary")
                    summary_digest = _source_bytes(config, summary_path)
                    if summary_digest != source["summary_sha256"]:
                        raise CommandError("tune source summary hash differs from the manifest")
                for reference in referenced_files(normalized_manifest):
                    candidate_path = Path(_path(config, reference, "input_file"))
                    if reference == normalized_manifest.get("tune_source", {}).get("summary_ref"):
                        if config.artifact_root not in candidate_path.parents:
                            raise CommandError("tune source summary must be under the new artifact root")
                    references.append({
                        "path": candidate_path.relative_to(config.repo_root).as_posix(),
                        "sha256": _source_bytes(config, candidate_path),
                    })
                hashes[key]["references"] = references
    return {"policy": "explicit-input-bytes-v2", "files": hashes}


def _profile(config: HarnessConfig, profile_id: str) -> CommandProfile:
    if not isinstance(profile_id, str) or profile_id not in config.command_profiles:
        raise UnknownProfileError(f"unknown command profile: {profile_id}")
    issues = validate_config(config)
    if issues:
        raise CommandError("invalid command configuration: " + "; ".join(issue.message for issue in issues))
    profile = config.command_profiles[profile_id]
    schema = profile["arguments"]
    return CommandProfile(profile_id, profile["module"], schema,
                          tuple(name for name, spec in schema.items() if spec["required"]),
                          profile["timeout_seconds"], config.artifact_root)


def _safe_text(value: str) -> None:
    if (not value or "--" in value or any(character in value for character in ";|&<>`$\r\n\x00")
            or any(ord(character) < 32 for character in value)):
        raise CommandError("argument contains empty, control, option, or shell-operator text")


def _path(config: HarnessConfig, value: object, role: str, *, run_id: str | None = None) -> str:
    if not isinstance(value, (str, Path)):
        raise CommandError("path argument must be a string or Path")
    raw = str(value).replace("\\", "/")
    _safe_text(raw)
    parsed = Path(raw)
    if ".." in parsed.parts or any(part.endswith((".", " ")) for part in parsed.parts):
        raise CommandError("path traversal or ambiguous path component")
    candidate = parsed if parsed.is_absolute() else config.repo_root / parsed
    # Reject legacy paths lexically before any filesystem inspection.
    if candidate != config.repo_root and config.repo_root not in candidate.parents:
        raise CommandError("input/output path must remain inside the repository")
    for legacy in config.legacy_paths:
        if candidate == legacy or legacy in candidate.parents:
            if not (config.run_root == candidate or config.run_root in candidate.parents) and not (
                role == "legacy_checkpoint_file"
                and config.repo_root / "artifacts/haic" in candidate.parents
                and candidate.suffix.lower() == ".pt"
            ):
                raise CommandError("historical data paths are outside this command boundary")
    try:
        path = _canonical(candidate)
    except RecordError as exc:
        raise CommandError(str(exc)) from exc
    if role.startswith("output"):
        root = config.artifact_root / run_id if run_id else config.artifact_root
        if root not in path.parents:
            raise CommandError("output path must be under the run-specific artifact root")
        if run_id is None:
            relative = path.relative_to(root)
            if len(relative.parts) < 2:
                raise CommandError("output path must include a run identifier")
            try:
                _identifier(relative.parts[0])
            except RecordError as exc:
                raise CommandError(str(exc)) from exc
        if path.exists() and (path.is_dir() != (role == "output_directory")):
            raise CommandError("output path type does not match its schema")
    elif role in {"input_file", "legacy_checkpoint_file"}:
        if not path.is_file():
            raise CommandError("explicit input file does not exist")
        if role == "legacy_checkpoint_file" and (
            config.repo_root / "artifacts/haic" not in path.parents or path.suffix.lower() != ".pt"
        ):
            raise CommandError("legacy checkpoint must be a .pt file under artifacts/haic")
    elif role == "input_directory":
        if not path.is_dir():
            raise CommandError("explicit input directory does not exist")
    return str(path)


def _value(config: HarnessConfig, spec: Mapping[str, object], value: object, *, run_id: str | None = None) -> object:
    kind = spec["type"]
    if kind == "path":
        normalized = _path(config, value, spec["path_role"], run_id=run_id)
        if spec["path_role"] == "legacy_checkpoint_file":
            digest = hashlib.sha256(Path(normalized).read_bytes()).hexdigest()
            if digest != spec.get("sha256"):
                raise CommandError("legacy checkpoint hash does not match registered value")
    elif kind == "bool":
        if type(value) is not bool:
            raise CommandError("boolean argument requires a JSON boolean")
        normalized = value
    elif kind in {"int", "float"}:
        if type(value) not in ({int} if kind == "int" else {int, float}) or not math.isfinite(value):
            raise CommandError(f"{kind} argument requires a finite typed number")
        normalized = int(value) if kind == "int" else float(value)
        if "minimum" in spec and normalized < spec["minimum"] or "maximum" in spec and normalized > spec["maximum"]:
            raise CommandError("numeric argument is outside the permitted range")
    else:
        if not isinstance(value, str):
            raise CommandError("string argument requires a string")
        _safe_text(value)
        normalized = value
        if kind == "track":
            match = re.fullmatch(r"([1-9][0-9]*):([0-9]+)", value)
            if not match or int(match[2]) > 0xFFFFFFFF:
                raise CommandError("track must be TRACK_ID:UINT32_SEED")
            normalized = f"{int(match[1])}:{int(match[2])}"
    if "choices" in spec and normalized not in spec["choices"]:
        raise CommandError("argument is not one of the permitted values")
    return normalized


def normalize_arguments(config: HarnessConfig, profile_id: str, arguments: Mapping[str, object], *, run_id: str | None = None) -> dict[str, object]:
    """Normalize an explicit typed subset; output is fixed when planning a run."""
    profile = _profile(config, profile_id)
    if not isinstance(arguments, Mapping) or any(not isinstance(key, str) for key in arguments):
        raise CommandError("arguments must be an object with string keys")
    if set(arguments) - set(profile.allowed_args):
        raise CommandError("unknown command arguments: " + str(sorted(set(arguments) - set(profile.allowed_args))))
    supplied = dict(arguments)
    if run_id is not None:
        try:
            _identifier(run_id)
        except RecordError as exc:
            raise CommandError(str(exc)) from exc
        output = str(config.artifact_root / run_id / config.command_profiles[profile_id]["output_name"])
        if "output" in supplied and _path(config, supplied["output"], profile.allowed_args["output"]["path_role"], run_id=run_id) != output:
            raise CommandError("output must equal the registered run artifact destination")
        supplied["output"] = output
    if any(name not in supplied for name in profile.required_args):
        raise CommandError("missing required explicit command arguments")
    normalized = {}
    for name in sorted(supplied):
        spec, value = profile.allowed_args[name], supplied[name]
        if spec.get("repeated", False):
            if not isinstance(value, (list, tuple)) or not value:
                raise CommandError("repeated argument requires a nonempty list")
            normalized[name] = [_value(config, spec, item, run_id=run_id) for item in value]
        else:
            normalized[name] = _value(config, spec, value, run_id=run_id)
    if profile_id == "evaluate_closed_loop":
        has_manifest = "candidate-manifest" in normalized
        has_group = "split-group" in normalized
        if has_manifest != has_group:
            raise CommandError("candidate-manifest and split-group must be provided together")
        if has_manifest and "site-map-split" not in normalized:
            raise CommandError("candidate tournament requires site-map-split")
    if profile_id == "benchmark_corridor_diagnostic" and not any(name in normalized for name in ("track", "site-map", "site-map-split")):
        raise CommandError("diagnostic requires explicitly registered track or map input")
    for name in ("train-only-site-map-split", "site-map-split"):
        if name in normalized:
            split_group = (
                str(normalized["split-group"])
                if profile_id == "evaluate_closed_loop" and "candidate-manifest" in normalized
                else None
            )
            _validate_split_references(
                config, Path(normalized[name]),
                train_only=name == "train-only-site-map-split", group=split_group,
            )
    if profile_id == "evaluate_closed_loop":
        has_manifest = "candidate-manifest" in normalized
        if has_manifest:
            try:
                candidate_manifest = validate_candidate_manifest(
                    _read_json(Path(normalized["candidate-manifest"])),
                    str(normalized["split-group"]),
                )
            except (RecordError, TournamentManifestError) as exc:
                raise CommandError(f"invalid candidate tournament manifest: {exc}") from exc
            manifest_path = Path(normalized["candidate-manifest"])
            if config.repo_root / "docs" not in manifest_path.parents:
                raise CommandError("candidate tournament manifests must be stored under docs/")
            for name in ("policy-checkpoint", "dynamics-checkpoint"):
                checkpoint = Path(normalized[name])
                if config.artifact_root not in checkpoint.parents:
                    raise CommandError("tournament control checkpoints must come from the new artifact root")
            for reference in referenced_files(candidate_manifest):
                source = Path(_path(config, reference, "input_file"))
                if reference.lower().endswith((".pt", ".pth")):
                    if config.artifact_root not in source.parents:
                        raise CommandError("tournament candidate checkpoints must come from the new artifact root")
    return normalized


def _validate_split_references(
    config: HarnessConfig,
    split: Path,
    *,
    train_only: bool,
    group: str | None = None,
) -> tuple[Path, ...]:
    """Inspect only a named manifest; reject forbidden map targets before opening.

    Full split loaders open all three groups even if their later selection picks
    one group. TRAIN-only never touches the other groups' map paths.
    """
    try:
        payload = _read_json(split)
    except RecordError as exc:
        raise CommandError(f"invalid split manifest: {exc}") from exc
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise CommandError("split manifest requires schema_version 1")
    map_paths = []
    groups = ("train",) if train_only else ((group,) if group else ("train", "tune", "held_out"))
    if group is not None:
        if group not in {"tune", "held_out"} or train_only:
            raise CommandError("selected split group must be tune or held_out")
        declared_paths: dict[str, set[str]] = {}
        for group_name in ("train", "tune", "held_out"):
            declared = payload.get(group_name)
            if not isinstance(declared, list) or not declared:
                raise CommandError(f"split {group_name} requires nonempty explicit map entries")
            names = set()
            for entry in declared:
                raw = entry.get("map") if isinstance(entry, dict) else None
                if not isinstance(raw, str) or not raw or "\\" in raw:
                    raise CommandError(f"split {group_name} map requires a relative POSIX path")
                parsed = Path(raw)
                if parsed.is_absolute() or ".." in parsed.parts:
                    raise CommandError("split map must remain inside its manifest directory")
                canonical = Path(_path(config, split.parent / parsed, "input_file"))
                names.add(os.path.normcase(str(canonical)).casefold())
            declared_paths[group_name] = names
        for left, right in (("train", "tune"), ("train", "held_out"), ("tune", "held_out")):
            if declared_paths[left] & declared_paths[right]:
                raise CommandError(f"split {left} and {right} must reference different map files")
    for group_name in groups:
        entries = payload.get(group_name)
        if not isinstance(entries, list) or not entries:
            raise CommandError(f"split {group_name} requires nonempty explicit map entries")
        for entry in entries:
            raw = entry.get("map") if isinstance(entry, dict) else None
            if not isinstance(raw, str) or not raw:
                raise CommandError(f"split {group_name} map requires a relative file path")
            if "\\" in raw:
                raise CommandError("split map references must use forward slashes")
            parsed = Path(raw)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise CommandError("split map must remain inside its manifest directory")
            # _path rejects historical roots lexically before any lstat/open.
            # Only existence/type is needed; never open the referenced map here.
            map_paths.append(Path(_path(config, split.parent / parsed, "input_file")))
    return tuple(map_paths)


def _registered_split_cells(config: HarnessConfig, split: Path, group: str) -> tuple[tuple[str, int], ...]:
    """Resolve exact selected (map_id, seed) cells without opening other groups."""
    try:
        payload = _read_json(split)
        from training.site_maps import load_site_map

        map_paths = _validate_split_references(config, split, train_only=False, group=group)
        entries = payload[group]
        cells: list[tuple[str, int]] = []
        for entry, map_path in zip(entries, map_paths, strict=True):
            seeds = entry.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                raise CommandError(f"split {group} map requires nonempty explicit seeds")
            map_id = load_site_map(map_path).map_id
            for seed in seeds:
                if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF:
                    raise CommandError(f"split {group} seeds must be uint32 integers")
                cells.append((map_id, seed))
        if not cells or len(cells) != len(set(cells)):
            raise CommandError("selected split group must contain unique map/seed cells")
        return tuple(cells)
    except (RecordError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, CommandError):
            raise
        raise CommandError(f"cannot register exact split cells: {exc}") from exc


def build_argv(config: HarnessConfig, profile_id: str, arguments: Mapping[str, object]) -> list[str]:
    profile = _profile(config, profile_id)
    normalized = normalize_arguments(config, profile_id, arguments)
    if "output" not in normalized:
        raise CommandError("argv requires a planned output path")
    argv = [sys.executable, "-m", profile.module]
    for name, value in normalized.items():
        spec = profile.allowed_args[name]
        if spec["type"] == "bool":
            if value:
                argv.append(spec["flag"])
        else:
            values = value if spec.get("repeated", False) else [value]
            for item in values:
                argv.extend((spec["flag"], str(item)))
    return argv


def _manifest_metadata(manifest: RunManifest) -> dict[str, object]:
    return {name: value for name, value in _json_value(manifest).items() if name not in {"plan_hash", "approval_hash"}}


def _research(value: Mapping[str, object]) -> dict[str, object]:
    try:
        hypothesis = _typed(Hypothesis, dict(value))
    except (RecordError, TypeError, ValueError) as exc:
        raise CommandError(f"complete hypothesis research metadata required: {exc}") from exc
    if any(not isinstance(getattr(hypothesis, item.name), str) or not getattr(hypothesis, item.name).strip()
           for item in fields(Hypothesis) if item.name != "source_paths"):
        raise CommandError("hypothesis fields must be nonempty reviewable strings")
    if any(not isinstance(path, str) or not path.strip() for path in hypothesis.source_paths):
        raise CommandError("hypothesis source paths must be nonempty strings")
    return _json_value(hypothesis)


def _event(kind: str, state: WorkflowState, **kwargs) -> RunEvent:
    return RunEvent(datetime.now(timezone.utc), kind, state, str(uuid.uuid4()), **kwargs)


def register_plan(config: HarnessConfig, metadata: Mapping[str, object], profile_id: str,
                  arguments: Mapping[str, object], *, research: Mapping[str, object],
                  previous_run_dir: Path | None = None) -> Path:
    """Register one operation without invoking a runner or reading historical data."""
    if not isinstance(metadata, Mapping):
        raise CommandError("manifest metadata must be an object")
    data = dict(metadata)
    if data.pop("approval_hash", "") or data.pop("plan_hash", ""):
        raise CommandError("new plans cannot supply prior approval or plan hashes")
    try:
        run_id = data["run_id"]
        normalized = normalize_arguments(config, profile_id, arguments, run_id=run_id)
        destination = str(config.artifact_root / run_id)
        if "output_paths" in data and list(data["output_paths"]) != [normalized["output"]]:
            raise CommandError("manifest output paths must match the planned output")
        data["output_paths"] = [normalized["output"]]
        provisional = _typed(RunManifest, dict(data, plan_hash="pending", approval_hash=""))
    except (KeyError, RecordError, TypeError, ValueError) as exc:
        if isinstance(exc, CommandError):
            raise
        raise CommandError(f"invalid complete manifest metadata: {exc}") from exc
    for name in ("purpose", "hypothesis_hash", "candidate_revision", "control_revision", "candidate_package_hash", "control_package_hash"):
        if not isinstance(getattr(provisional, name), str) or not getattr(provisional, name).strip():
            raise CommandError(f"manifest {name} must be reviewable and nonempty")
    for name in ("data_ids", "map_ids", "split_ids"):
        if not getattr(provisional, name) or any(not isinstance(item, str) or not item.strip() for item in getattr(provisional, name)):
            raise CommandError(f"manifest requires registered {name}")
    if any(item not in config.split_ids for item in provisional.split_ids):
        raise CommandError("manifest uses unregistered split identities")
    if profile_id == "evaluate_closed_loop" and "candidate-manifest" in normalized:
        try:
            candidate_manifest = validate_candidate_manifest(
                _read_json(Path(normalized["candidate-manifest"])),
                str(normalized["split-group"]),
            )
        except (RecordError, TournamentManifestError) as exc:
            raise CommandError(f"invalid candidate tournament manifest: {exc}") from exc
        stage = str(normalized["split-group"])
        expected_revision = (
            candidate_manifest["candidates"][0]["candidate_id"]
            if stage == "held_out"
            else f"tournament-{candidate_manifest['comparison_id']}"
        )
        if (provisional.control_revision != candidate_manifest["control_id"]
                or provisional.candidate_revision != expected_revision
                or provisional.comparison_id != candidate_manifest["comparison_id"]
                or tuple(provisional.split_ids) != (stage,)):
            raise CommandError("run manifest control, candidate, comparison, and split identities must match the tournament manifest")
        cells = _registered_split_cells(config, Path(normalized["site-map-split"]), stage)
        if (set(provisional.map_ids) != {map_id for map_id, _ in cells}
                or set(provisional.seed_ids) != {str(seed) for _, seed in cells}
                or provisional.comparison_episode_count != len(cells)):
            raise CommandError("run manifest map/seed identities and denominator must match the selected exact cells")
    if not provisional.resource_limits or not provisional.permission_limits or not provisional.source_hashes:
        raise CommandError("manifest requires resource, permission and source hash metadata")
    payload = dict(schema_version=_EXECUTION_PLAN_SCHEMA_VERSION, manifest=_manifest_metadata(provisional), profile_id=profile_id,
                   arguments=normalized, profile=_json_value(config.command_profiles[profile_id]),
                   research=_research(research), artifact_destination=destination,
                   input_identity=_input_identity(config, profile_id, normalized),
                   executable_identity=_source_identity(config, profile_id, normalized))
    digest = _hash(payload)
    manifest = _typed(RunManifest, dict(data, plan_hash=digest, approval_hash=""))
    path = create_run(config, manifest, previous_run_dir=previous_run_dir)
    with run_transaction(path, config=config) as run:
        run.write_plan(dict(payload, plan_hash=digest))
        for state in _INITIAL:
            run.append(_event("STATE", state))
    return path


def _validated_plan(config: HarnessConfig, run: RunTransaction) -> dict[str, object]:
    plan = run.plan
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "manifest", "profile_id", "arguments", "profile", "research", "artifact_destination", "input_identity", "executable_identity", "plan_hash"}:
        raise CommandError("persisted execution plan is missing or malformed")
    digest = _hash({name: value for name, value in plan.items() if name != "plan_hash"})
    if digest != plan["plan_hash"] or digest != run.manifest.plan_hash:
        raise CommandError("persisted execution plan hash mismatch")
    if plan["manifest"] != _manifest_metadata(run.manifest):
        raise CommandError("manifest/source revisions differ from immutable plan")
    if plan["schema_version"] != _EXECUTION_PLAN_SCHEMA_VERSION or type(plan["schema_version"]) is not int:
        raise CommandError("unknown execution plan schema")
    profile_id = plan["profile_id"]
    _profile(config, profile_id)
    if plan["profile"] != _json_value(config.command_profiles[profile_id]):
        raise CommandError("current command profile fingerprint differs from approved plan")
    normalized = normalize_arguments(config, profile_id, plan["arguments"], run_id=run.manifest.run_id)
    if normalized != plan["arguments"] or plan["artifact_destination"] != str(config.artifact_root / run.manifest.run_id):
        raise CommandError("persisted arguments/output destination are not canonical")
    if plan["input_identity"] != _input_identity(config, profile_id, normalized):
        raise CommandError("planned input file bytes changed; a new run, plan and approvals are required")
    if plan["executable_identity"] != _source_identity(config, profile_id, normalized):
        raise CommandError("actual executable source/interpreter changed; a new run, plan and approvals are required")
    if _research(plan["research"]) != plan["research"]:
        raise CommandError("persisted hypothesis metadata is not canonical")
    return plan


def _reference(event: RunEvent, approvals: tuple[Approval, ...], stage: str) -> Approval:
    matched = [approval for approval in approvals if approval.stage == stage and approval.source_ref == event.approval_ref]
    if len(matched) != 1:
        raise ApprovalError("transition must reference that stage's current-run APPROVAL event")
    return matched[0]


def _replay(run: RunTransaction) -> RunHistory:
    events = run.events
    if len(events) < len(_INITIAL):
        raise CommandError("run has incomplete plan registration history")
    state = WorkflowState.STOPPED
    approvals = ()
    started = finished = succeeded = released = False
    outcome = None
    gates = run.report.gate_results if run.report else ()
    for index, event in enumerate(events):
        if index < len(_INITIAL):
            if event.kind != "STATE" or event.workflow_state != _INITIAL[index] or event.approval_ref is not None:
                raise CommandError("run must register STOPPED/DISCOVER/HYPOTHESIZE/DESIGN in order")
            if index:
                state = transition(state, event.workflow_state, plan_hash=run.manifest.plan_hash, approvals=(), gates=())
            continue
        requested = event.workflow_state
        if event.kind == "APPROVAL":
            stage = event.approval_stage
            if (stage not in _STAGES or state != _STAGES[stage][0] or requested != state
                    or event.approved_plan_hash != run.manifest.plan_hash
                    or any(approval.stage == stage for approval in approvals)):
                raise ApprovalError("approval stages cannot be bypassed, reordered, duplicated, or stale")
            source = event.resource_usage.get("authorization_source")
            if not isinstance(source, str) or not source.strip() or event.approval_ref is not None:
                raise ApprovalError("approval must record the user's authorization source")
            approvals += (Approval(stage, event.approved_plan_hash, event.timestamp, event.event_id),)
        elif event.kind == "EXECUTION_STARTED":
            if started or state != WorkflowState.EXECUTE_PENDING_APPROVAL or requested != WorkflowState.EVALUATE or event.execution_status != "running":
                raise CommandError("execution may start exactly once from its pending approval")
            _reference(event, approvals, "execution")
            state = transition(state, requested, plan_hash=run.manifest.plan_hash, approvals=approvals, gates=())
            started = True
        elif event.kind in {"EXECUTION_FINISHED", "EXECUTION_FAILED"}:
            if not started or finished or state != WorkflowState.EVALUATE or requested != state:
                raise CommandError("execution finish requires exactly one earlier start")
            _reference(event, approvals, "execution")
            succeeded = event.kind == "EXECUTION_FINISHED"
            if event.execution_status != ("succeeded" if succeeded else "failed"):
                raise CommandError("execution finish status does not match event kind")
            code = event.resource_usage.get("returncode")
            if succeeded and (type(code) is not int or code != 0):
                raise CommandError("successful execution requires exit code zero evidence")
            if not succeeded and (code is not None and (type(code) is not int or code == 0)):
                raise CommandError("failed execution cannot record a successful exit code")
            if not succeeded and not event.error:
                raise CommandError("failed execution requires error evidence")
            finished = True
        elif event.kind == "CYCLE_DECISION":
            if not finished or state != WorkflowState.EVALUATE or requested not in _OUTCOMES or outcome is not None:
                raise CommandError("cycle decision requires a finished execution and one evaluation outcome")
            if not succeeded and requested == WorkflowState.ADVANCE:
                raise CommandError("failed execution cannot advance")
            state = transition(state, requested, plan_hash=run.manifest.plan_hash, approvals=approvals, gates=())
            outcome = requested
        elif event.kind == "GATE_REVIEW":
            if outcome is None or requested != WorkflowState[f"GATE_REVIEW_{outcome.value}"] or event.correction_ref != "integration_report.json":
                raise CommandError("gate review must reference the matching outcome and persisted report")
            state = transition(state, requested, plan_hash=run.manifest.plan_hash, approvals=approvals, gates=gates)
        elif event.kind == "STATE":
            stage = next((name for name, edge in _STAGES.items() if (state, requested) == edge), None)
            if (stage is None and state == WorkflowState.DESIGN_PENDING_APPROVAL
                    and requested == WorkflowState.IMPLEMENT_PENDING_APPROVAL):
                stage = "design"  # Historical three-stage runs.
            if stage == "execution":
                raise CommandError("EVALUATE requires EXECUTION_STARTED, not a state label")
            if stage:
                _reference(event, approvals, stage)
            elif requested not in {WorkflowState.RELEASE_IF_GATE_PASS, WorkflowState.STOPPED} or event.approval_ref is not None:
                raise CommandError("unexpected lifecycle state event")
            state = transition(state, requested, plan_hash=run.manifest.plan_hash, approvals=approvals, gates=gates)
            if state == WorkflowState.RELEASE_IF_GATE_PASS:
                released = True
        else:
            raise CommandError(f"unsupported lifecycle event: {event.kind}")
    if started and not run.execution_reserved:
        raise CommandError("execution history has no immutable reservation")
    if started:
        start_event = next(event for event in events if event.kind == "EXECUTION_STARTED")
        if run.execution_claim != {"event_id": start_event.event_id, "plan_hash": run.manifest.plan_hash}:
            raise CommandError("execution claim must reference this exact run's start event and plan")
    return RunHistory(state, approvals, started, finished, succeeded, outcome, released)


def replay_run_history(config: HarnessConfig, run_dir: Path) -> RunHistory:
    """Validate the complete current-run plan/log and replay actual state edges.

    Task 7 must require history.released to prove the ADVANCE release path.
    """
    with run_transaction(run_dir, config=config) as run:
        _validated_plan(config, run)
        return _replay(run)


def load_run_plan(config: HarnessConfig, run_dir: Path) -> dict[str, object]:
    with run_transaction(run_dir, config=config) as run:
        plan = _validated_plan(config, run)
        _replay(run)
        return plan


def approve_run(config: HarnessConfig, run_dir: Path, stage: str, *, source_ref: str) -> Approval:
    if stage not in _STAGES or not isinstance(source_ref, str) or not source_ref.strip():
        raise ApprovalError("approval requires a registered stage and user authorization source reference")
    with run_transaction(run_dir, config=config) as run:
        _validated_plan(config, run)
        history = _replay(run)
        if history.state != _STAGES[stage][0] or any(approval.stage == stage for approval in history.approvals):
            raise ApprovalError("approval is not legal at the current stage")
        event = _event("APPROVAL", history.state, approval_stage=stage, approved_plan_hash=run.manifest.plan_hash,
                       resource_usage={"authorization_source": source_ref})
        approval = Approval(stage, run.manifest.plan_hash, event.timestamp, event.event_id)
        run.append(event)
        if stage != "execution":
            requested = transition(history.state, _STAGES[stage][1], plan_hash=run.manifest.plan_hash,
                                   approvals=history.approvals + (approval,), gates=())
            run.append(_event("STATE", requested, approval_ref=event.event_id))
        return approval


def execute_approved(config: HarnessConfig, run_dir: Path, plan_hash: str, profile_id: str,
                     arguments: Mapping[str, object], *, runner=subprocess.run) -> CommandResult:
    with run_transaction(run_dir, config=config) as run:
        plan = _validated_plan(config, run)
        history = _replay(run)
        if plan_hash != plan["plan_hash"] or profile_id != plan["profile_id"]:
            raise CommandError("caller command/hash differs from persisted approved plan")
        normalized = normalize_arguments(config, profile_id, arguments, run_id=run.manifest.run_id)
        if normalized != plan["arguments"]:
            raise CommandError("changed arguments require a new plan, run and approvals")
        if history.execution_started or run.execution_reserved:
            raise CommandError("execution has already been reserved; retry requires a new run")
        if history.state != WorkflowState.EXECUTE_PENDING_APPROVAL:
            raise ApprovalError("execution requires EXECUTE_PENDING_APPROVAL")
        execution = next((approval for approval in history.approvals if approval.stage == "execution"), None)
        if execution is None:
            raise ApprovalError("execution approval for the exact current plan hash is required")
        argv = build_argv(config, plan["profile_id"], plan["arguments"])
        profile = _profile(config, plan["profile_id"])
        state = transition(history.state, WorkflowState.EVALUATE, plan_hash=plan_hash, approvals=history.approvals, gates=())
        run.reserve_execution(_event("EXECUTION_STARTED", state, approval_ref=execution.source_ref, execution_status="running"))
    try:
        completed = runner(argv, cwd=config.repo_root, timeout=profile.timeout_seconds,
                           shell=False, capture_output=True, text=True, check=False)
        if type(completed.returncode) is not int:
            raise ValueError("runner did not return an integer exit code")
        result = CommandResult(completed.returncode, completed.returncode == 0,
                               completed.stdout or "", completed.stderr or "",
                               None if completed.returncode == 0 else f"process exited with code {completed.returncode}")
    except Exception as exc:
        # Preserve the permanent claim on timeouts, launch errors and runner failures.
        result = CommandResult(None, False, error=f"{type(exc).__name__}: {exc}")
    with run_transaction(run_dir, config=config) as run:
        _validated_plan(config, run)
        history = _replay(run)
        if not history.execution_started or history.execution_finished:
            raise CommandError("execution finish history is inconsistent")
        run.append(_event("EXECUTION_FINISHED" if result.succeeded else "EXECUTION_FAILED", WorkflowState.EVALUATE,
                          approval_ref=execution.source_ref,
                          execution_status="succeeded" if result.succeeded else "failed", error=result.error,
                          resource_usage={"returncode": result.returncode}))
    return result


def report_run(config: HarnessConfig, run_dir: Path, outcome: str | WorkflowState,
               report: IntegrationReport) -> RunHistory:
    try:
        outcome = WorkflowState(outcome)
    except ValueError as exc:
        raise CommandError("unknown evaluation outcome") from exc
    if outcome not in _OUTCOMES or not isinstance(report, IntegrationReport):
        raise CommandError("report requires an evaluation outcome and typed IntegrationReport")
    with run_transaction(run_dir, config=config) as run:
        _validated_plan(config, run)
        history = _replay(run)
        if history.state != WorkflowState.EVALUATE or not history.execution_finished:
            raise CommandError("report requires a finished execution in EVALUATE")
        if outcome == WorkflowState.ADVANCE and not history.execution_succeeded:
            raise CommandError("failed execution cannot advance")
        state = transition(history.state, outcome, plan_hash=run.manifest.plan_hash, approvals=history.approvals, gates=())
        review = transition(state, WorkflowState[f"GATE_REVIEW_{outcome.value}"], plan_hash=run.manifest.plan_hash,
                            approvals=history.approvals, gates=report.gate_results)
        # Validate every edge before any immutable report write.
        release = outcome == WorkflowState.ADVANCE and all(gate.status == GateStatus.PASS for gate in report.gate_results)
        target = WorkflowState.RELEASE_IF_GATE_PASS if release else WorkflowState.STOPPED
        transition(review, target, plan_hash=run.manifest.plan_hash, approvals=history.approvals, gates=report.gate_results)
        run.write_report(report)
        run.append(_event("CYCLE_DECISION", state))
        run.append(_event("GATE_REVIEW", review, correction_ref="integration_report.json"))
        run.append(_event("STATE", target))
        if release:
            run.append(_event("STATE", WorkflowState.STOPPED))
        return _replay(run)
