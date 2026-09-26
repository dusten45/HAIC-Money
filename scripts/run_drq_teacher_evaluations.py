"""Run the frozen DrQ teacher-replay screen/confirmation/blind evaluation gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import struct
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts import compare_drq_teacher_replay as compare


ARMS = ("unchanged-source", "online-only", "teacher-replay")
STUDENT_ARMS = ("online-only", "teacher-replay")
STAGES = ("screen", "confirmation", "blind", "all")
ORCHESTRATION_DIR = "drqv2-teacher-replay-v1"
STAGE_COMPLETE = "stage-complete.json"


class OrchestrationError(ValueError):
    """A frozen evaluation stage is invalid, incomplete or already consumed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise OrchestrationError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise OrchestrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_bytes(),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise OrchestrationError(f"cannot read valid JSON: {path}") from error


def _encoded_json(value: Any) -> bytes:
    try:
        return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise OrchestrationError("refusing to write a non-JSON orchestration artifact") from error


def _assert_beneath(path: Path, root: Path, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise OrchestrationError(f"{label} escapes its permitted root: {path}")
    if path.is_symlink():
        raise OrchestrationError(f"{label} must not be a symlink: {path}")
    return resolved


def _mkdir(path: Path, root: Path, *, exclusive: bool = False) -> None:
    _assert_beneath(path, root, "output directory")
    try:
        path.mkdir(parents=not exclusive, exist_ok=not exclusive)
    except FileExistsError as error:
        raise OrchestrationError(f"immutable output directory already exists: {path}") from error


def _write_exclusive(path: Path, value: Any, output_root: Path) -> str:
    _assert_beneath(path, output_root, "output file")
    if not path.parent.is_dir():
        raise OrchestrationError(f"output parent directory does not exist: {path.parent}")
    payload = _encoded_json(value)
    try:
        with path.open("xb") as target:
            target.write(payload)
            target.flush()
    except FileExistsError as error:
        raise OrchestrationError(f"refusing to overwrite immutable output: {path}") from error
    return hashlib.sha256(payload).hexdigest()


def _write_sealed(path: Path, value: dict[str, Any], output_root: Path) -> str:
    _assert_beneath(path, output_root, "sealed output file")
    if not path.parent.is_dir():
        raise OrchestrationError(f"sealed output parent does not exist: {path.parent}")
    try:
        return compare.write_sealed(path, value)
    except (OSError, ValueError) as error:
        raise OrchestrationError(str(error)) from error


def _repo_file(repo_root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise OrchestrationError(f"{label} path is absent")
    raw = Path(relative)
    if raw.is_absolute():
        raise OrchestrationError(f"{label} must be repository-relative: {relative}")
    resolved = (repo_root / raw).resolve()
    if not resolved.is_relative_to(repo_root.resolve()) or not resolved.is_file():
        raise OrchestrationError(f"{label} is missing or escapes the repository: {relative}")
    if raw.is_symlink() or resolved.is_symlink():
        raise OrchestrationError(f"{label} must not be a symlink: {relative}")
    return resolved


def _check_config(
    path: Path,
    protocol: dict[str, Any],
    identity: dict[str, Any],
    *,
    student: bool,
    protocol_sha256: str | None = None,
) -> None:
    value = read_json(path)
    config = value.get("config") if isinstance(value, dict) else None
    if not isinstance(config, dict):
        raise OrchestrationError(f"candidate run config lacks a config object: {path}")
    if config.get("frame_skip") != protocol["frame_skip"] or config.get("max_steps") != protocol["max_steps"]:
        raise OrchestrationError(f"candidate run config horizon differs from protocol: {path}")
    if student and (
        config.get("study_protocol_sha256") != protocol_sha256
        or config.get("source_learner_seed") != identity["source_learner_seed"]
        or config.get("arm") != identity["arm"]
        or config.get("algorithm") != "drq-v2"
        or config.get("total_steps")
        != 131072 + protocol["budgets"]["additional_online_decisions_per_arm_source"]
    ):
        raise OrchestrationError(f"student run config identity/budget differs from candidate catalog: {path}")


def _source_run_dir(source: dict[str, Any], actor_path: Path, repo_root: Path) -> Path:
    declared_config = _repo_file(repo_root, source["run_config_path"], "source run config")
    if sha256_file(declared_config) != source["run_config_sha256"]:
        raise OrchestrationError(f"source run config hash mismatch: {declared_config}")
    candidates = [parent for parent in actor_path.parents if (parent / "config.json").is_file()]
    if not candidates:
        raise OrchestrationError(f"source actor has no candidate_metadata-compatible config.json: {actor_path}")
    run_dir = candidates[0]
    config_path = run_dir / "config.json"
    if config_path.resolve() != declared_config.resolve():
        raise OrchestrationError(
            f"source config.json differs from its frozen run config: {config_path} != {declared_config}"
        )
    if not actor_path.is_relative_to(run_dir):
        raise OrchestrationError(f"source actor is outside its candidate run directory: {actor_path}")
    return run_dir


def build_actor_inventory(
    protocol: dict[str, Any], protocol_sha256: str, repo_root: Path, run_root: Path
) -> dict[tuple[int, str, int | None], dict[str, Any]]:
    """Bind the ten frozen screen actors from source records and four catalogs."""
    repo_root = repo_root.resolve()
    run_root = run_root.resolve()
    if not run_root.is_relative_to(repo_root) or not run_root.is_dir():
        raise OrchestrationError(f"study run_root is missing or outside the repository: {run_root}")

    sources = {source["learner_seed"]: source for source in protocol["source_actors"]}
    if (
        len(sources) != 2
        or set(sources) != {0, 1}
        or any(type(source["learner_seed"]) is not int for source in protocol["source_actors"])
    ):
        raise OrchestrationError("protocol source actors must uniquely identify learner seeds 0 and 1")
    inventory: dict[tuple[int, str, int | None], dict[str, Any]] = {}

    for learner in (0, 1):
        source = sources[learner]
        actor_path = _repo_file(repo_root, source["actor_path"], "source actor")
        checkpoint_path = _repo_file(repo_root, source["checkpoint_path"], "source checkpoint")
        if sha256_file(actor_path) != source["actor_sha256"]:
            raise OrchestrationError(f"source actor hash differs from protocol: {actor_path}")
        if sha256_file(checkpoint_path) != source["checkpoint_sha256"]:
            raise OrchestrationError(f"source checkpoint hash differs from protocol: {checkpoint_path}")
        run_dir = _source_run_dir(source, actor_path, repo_root)
        identity = {
            "source_learner_seed": learner,
            "arm": "unchanged-source",
            "actor_sha256": source["actor_sha256"],
            "checkpoint_sha256": None,
            "checkpoint_online_step": None,
        }
        _check_config(run_dir / "config.json", protocol, identity, student=False)
        inventory[(learner, "unchanged-source", None)] = {
            "identity": identity,
            "actor_path": actor_path,
            "run_dir": run_dir,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "checkpoint_manifest_path": _repo_file(
                repo_root, source["checkpoint_manifest_path"], "source checkpoint manifest"
            ),
            "checkpoint_manifest_sha256": source["checkpoint_manifest_sha256"],
            "config_path": run_dir / "config.json",
            "config_sha256": sha256_file(run_dir / "config.json"),
        }

    catalogs = sorted(run_root.rglob("checkpoint-catalog.json"))
    expected_catalog_roles = {(learner, arm) for learner in (0, 1) for arm in STUDENT_ARMS}
    observed_catalog_roles: set[tuple[int, str]] = set()
    declared_steps = protocol["budgets"]["checkpoint_online_steps"]
    if declared_steps != [16384, 32768]:
        raise OrchestrationError("checkpoint catalog inventory requires the frozen 16384/32768 steps")
    for catalog_path in catalogs:
        if catalog_path.is_symlink() or not catalog_path.resolve().is_relative_to(run_root):
            raise OrchestrationError(f"checkpoint catalog escapes the frozen run_root: {catalog_path}")
        catalog = read_json(catalog_path)
        learner = catalog.get("source_learner_seed") if isinstance(catalog, dict) else None
        arm = catalog.get("arm") if isinstance(catalog, dict) else None
        role = (learner, arm)
        if (
            type(learner) is not int
            or not isinstance(arm, str)
            or role not in expected_catalog_roles
            or role in observed_catalog_roles
        ):
            raise OrchestrationError(f"unexpected or duplicate checkpoint catalog role: {catalog_path}")
        source = sources[learner]
        if (
            type(catalog.get("schema_version")) is not int
            or catalog.get("schema_version") != 1
            or catalog.get("study_protocol_sha256") != protocol_sha256
            or catalog.get("source_actor_sha256") != source["actor_sha256"]
            or catalog.get("source_checkpoint_sha256") != source["checkpoint_sha256"]
        ):
            raise OrchestrationError(f"checkpoint catalog protocol/source identity mismatch: {catalog_path}")
        run_dir = catalog_path.parent.resolve()
        if not run_dir.is_relative_to(run_root):
            raise OrchestrationError(f"candidate run directory escapes protocol run_root: {run_dir}")
        config_path = run_dir / "config.json"
        base_identity = {
            "source_learner_seed": learner,
            "arm": arm,
            "actor_sha256": "",
            "checkpoint_sha256": "",
            "checkpoint_online_step": None,
        }
        if not config_path.is_file():
            raise OrchestrationError(f"checkpoint catalog run lacks config.json: {config_path}")
        _check_config(
            config_path, protocol, base_identity, student=True, protocol_sha256=protocol_sha256
        )
        result_path = run_dir / "result.json"
        result = read_json(result_path)
        if (
            result.get("completed") is not True
            or result.get("study_protocol_sha256") != protocol_sha256
            or result.get("source_learner_seed") != learner
            or result.get("arm") != arm
            or result.get("additional_online_steps")
            != protocol["budgets"]["additional_online_decisions_per_arm_source"]
            or result.get("checkpoint_catalog_path") != "checkpoint-catalog.json"
        ):
            raise OrchestrationError(f"checkpoint catalog has no matching completed trainer result: {result_path}")
        candidates = catalog.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(declared_steps):
            raise OrchestrationError(f"checkpoint catalog must contain exactly two candidates: {catalog_path}")
        seen_steps: set[int] = set()
        for record in candidates:
            if not isinstance(record, dict):
                raise OrchestrationError(f"invalid checkpoint catalog record: {catalog_path}")
            step = record.get("checkpoint_online_step")
            if type(step) is not int or step not in declared_steps or step in seen_steps:
                raise OrchestrationError(f"checkpoint catalog has an extra/duplicate step: {catalog_path}")
            seen_steps.add(step)
            identity = {
                "source_learner_seed": learner,
                "arm": arm,
                "actor_sha256": record.get("actor_sha256"),
                "checkpoint_sha256": record.get("checkpoint_sha256"),
                "checkpoint_online_step": step,
            }
            if any(
                record.get(field) != value
                for field, value in (
                    ("study_protocol_sha256", protocol_sha256),
                    ("source_learner_seed", learner),
                    ("arm", arm),
                    ("source_actor_sha256", source["actor_sha256"]),
                    ("source_checkpoint_sha256", source["checkpoint_sha256"]),
                )
            ):
                raise OrchestrationError(f"catalog candidate lineage mismatch: {catalog_path}, step {step}")
            if not all(isinstance(identity[key], str) and len(identity[key]) == 64
                       for key in ("actor_sha256", "checkpoint_sha256")):
                raise OrchestrationError(f"catalog candidate lacks valid actor/checkpoint hashes: {catalog_path}")
            actor_path = _repo_file(repo_root, record.get("actor_path"), "student actor")
            checkpoint_path = _repo_file(repo_root, record.get("checkpoint_path"), "student checkpoint")
            manifest_path = _repo_file(
                repo_root, record.get("checkpoint_manifest_path"), "student checkpoint manifest"
            )
            if not all(path.is_relative_to(run_root) for path in (actor_path, checkpoint_path, manifest_path)):
                raise OrchestrationError(f"catalog candidate artifact escapes study run_root: {catalog_path}")
            if (
                sha256_file(actor_path) != identity["actor_sha256"]
                or sha256_file(checkpoint_path) != identity["checkpoint_sha256"]
                or sha256_file(manifest_path) != record.get("checkpoint_manifest_sha256")
            ):
                raise OrchestrationError(f"catalog candidate artifact hash mismatch: {catalog_path}, step {step}")
            if not actor_path.is_relative_to(run_dir):
                raise OrchestrationError(f"catalog actor is outside its candidate run directory: {actor_path}")
            expected_checkpoint_dir = run_dir / "checkpoints" / f"step-{step:09d}"
            if (
                actor_path != expected_checkpoint_dir / "actor.pt"
                or checkpoint_path != expected_checkpoint_dir / "checkpoint.pt"
                or manifest_path != expected_checkpoint_dir / "checkpoint.manifest.json"
            ):
                raise OrchestrationError(f"catalog candidate paths differ from the frozen checkpoint layout: {catalog_path}")
            _check_config(
                config_path, protocol, identity, student=True, protocol_sha256=protocol_sha256
            )
            inventory[(learner, arm, step)] = {
                "identity": identity,
                "actor_path": actor_path,
                "run_dir": run_dir,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "checkpoint_manifest_path": manifest_path,
                "checkpoint_manifest_sha256": record["checkpoint_manifest_sha256"],
                "config_path": config_path,
                "config_sha256": sha256_file(config_path),
                "catalog_path": catalog_path.resolve(),
                "catalog_sha256": sha256_file(catalog_path),
            }
        if seen_steps != set(declared_steps):
            raise OrchestrationError(f"checkpoint catalog step set differs from protocol: {catalog_path}")
        observed_catalog_roles.add(role)

    if observed_catalog_roles != expected_catalog_roles:
        missing = sorted(expected_catalog_roles - observed_catalog_roles)
        raise OrchestrationError(f"missing completed student checkpoint catalogs: {missing}")
    expected_inventory = {
        (learner, "unchanged-source", None) for learner in (0, 1)
    } | {
        (learner, arm, step)
        for learner in (0, 1)
        for arm in STUDENT_ARMS
        for step in declared_steps
    }
    if set(inventory) != expected_inventory or len(inventory) != 10:
        raise OrchestrationError("candidate actor inventory is not the exact ten-actor screen grid")
    return inventory


def _role_key(identity: dict[str, Any]) -> tuple[int, str, int | None]:
    return (
        identity["source_learner_seed"], identity["arm"], identity["checkpoint_online_step"]
    )


def _identity_key(wrapper: dict[str, Any]) -> tuple[int, str, int | None]:
    return (wrapper["source_learner_seed"], wrapper["arm"], wrapper["checkpoint_online_step"])


def screen_score(metrics: dict[str, Any]) -> tuple[int, float, float]:
    """Return finishes desc, canonical progress desc, completed lap time asc."""
    finishes = metrics["finish_count"]
    progress = metrics["canonical_mean_progress"]
    lap_time = metrics["completed_mean_lap_time_ms"]
    if type(finishes) is not int or not math.isfinite(progress):
        raise OrchestrationError("screen score contains invalid canonical metrics")
    return finishes, progress, math.inf if lap_time is None else lap_time


def choose_screen_actor(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one actor by the preregistered tuple, retaining the earlier step on ties."""
    if not candidates:
        raise OrchestrationError("screen selection has no candidates")
    def key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
        identity = candidate["identity"]
        finishes, progress, lap_time = screen_score(candidate["metrics"])
        step = identity["checkpoint_online_step"]
        earlier_step = -1 if step is None else step
        return (-finishes, -progress, lap_time, earlier_step)
    return min(candidates, key=key)


class DrQTeacherEvaluationOrchestrator:
    def __init__(
        self,
        protocol_file: Path,
        repo_root: Path,
        *,
        evaluator_timeout_seconds: int = 300,
        package_backend=None,
    ):
        self.protocol_file = protocol_file.resolve()
        self.repo_root = repo_root.resolve()
        if not self.protocol_file.is_file() or not self.repo_root.is_dir():
            raise OrchestrationError("protocol file and repository root must exist")
        if type(evaluator_timeout_seconds) is not int or evaluator_timeout_seconds <= 0:
            raise OrchestrationError("evaluator timeout must be a positive integer")
        try:
            self.protocol, self.protocol_sha256 = compare.load_protocol(
                self.protocol_file, self.repo_root
            )
        except Exception as error:
            raise OrchestrationError(f"invalid frozen study protocol: {error}") from error
        self.run_root = (self.repo_root / self.protocol["run_root"]).resolve()
        if not self.run_root.is_relative_to(self.repo_root) or not self.run_root.is_dir():
            raise OrchestrationError("protocol run_root must exist safely beneath the repository")
        self.evaluations_root = self.run_root / "evaluations"
        _assert_beneath(self.evaluations_root, self.run_root, "evaluations root")
        self.base_root = self.evaluations_root / ORCHESTRATION_DIR
        _assert_beneath(self.base_root, self.evaluations_root, "orchestration root")
        self.evaluator_timeout_seconds = evaluator_timeout_seconds
        self.package_backend = package_backend
        self.inventory = build_actor_inventory(
            self.protocol, self.protocol_sha256, self.repo_root, self.run_root
        )

    def _stage_root(self, stage: str) -> Path:
        return self.base_root / stage

    def _cell_id(self, identity: dict[str, Any]) -> str:
        learner = identity["source_learner_seed"]
        arm = identity["arm"]
        step = identity["checkpoint_online_step"]
        return f"learner-{learner}-{arm}-source" if step is None else f"learner-{learner}-{arm}-step-{step}"

    def _cell_paths(self, stage_root: Path, actor: dict[str, Any]) -> dict[str, Path]:
        cell = stage_root / "actors" / self._cell_id(actor["identity"])
        return {
            "cell": cell,
            "started": cell / "started.json",
            "evaluator_output": cell / "evaluator-runs",
            "evaluator_pointer": cell / "evaluator-receipt.json",
            "wrapper": cell / "actor-receipt.json",
        }

    def _screen_actors(self) -> list[dict[str, Any]]:
        actors = list(self.inventory.values())
        return sorted(
            actors,
            key=lambda actor: (
                actor["identity"]["source_learner_seed"],
                ARMS.index(actor["identity"]["arm"]),
                actor["identity"]["checkpoint_online_step"] or -1,
            ),
        )

    def _package_screen_actors(
        self, stage_root: Path, actors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        package_root = stage_root / "packages"
        _mkdir(package_root, self.evaluations_root, exclusive=True)
        records = []
        for actor in actors:
            identity = actor["identity"]
            cell = self._cell_id(identity)
            archive_path = package_root / f"{cell}.zip"
            receipt_path = package_root / f"{cell}.json"
            if self.package_backend is None:
                from package_submission import (
                    build_submission,
                    model_payload_format,
                    smoke_submission,
                    submission_dependency_paths,
                    validate_agent_source,
                    validate_submission_archive,
                )

                agent_path = self.repo_root / "agent.py"
                expected_model = validate_agent_source(agent_path)
                model_format = model_payload_format(actor["actor_path"])
                if model_format != "haic-drq-v2-actor-v1" or expected_model != "model.pt":
                    raise OrchestrationError("A4 requires the root-model DrQ actor package format")
                dependencies = submission_dependency_paths(agent_path, model_format)
                if archive_path.exists() or archive_path.is_symlink():
                    raise OrchestrationError(f"refusing to overwrite immutable actor package: {archive_path}")
                build_submission(agent_path, actor["actor_path"], archive_path)
                validate_submission_archive(
                    archive_path,
                    expected_model,
                    expected_modules=[dependency.name for dependency in dependencies],
                )
                smoke = smoke_submission(
                    archive_path, self.protocol["runtime"]["evaluation_executable"]
                )
                package = {
                    "expected_model_filename": expected_model,
                    "model_format": model_format,
                    "modules": [dependency.name for dependency in dependencies],
                    "smoke": smoke,
                }
            else:
                package = self.package_backend(actor, archive_path)
            if not archive_path.is_file() or archive_path.is_symlink():
                raise OrchestrationError(f"A4 package backend did not produce a regular archive: {archive_path}")
            smoke = package.get("smoke")
            if (
                package.get("expected_model_filename") != "model.pt"
                or package.get("model_format") != "haic-drq-v2-actor-v1"
                or not isinstance(package.get("modules"), list)
                or not isinstance(smoke, dict)
                or type(smoke.get("finite")) is not bool
                or smoke.get("finite") is not True
                or type(smoke.get("reset_matches_first")) is not bool
                or smoke.get("reset_matches_first") is not True
                or type(smoke.get("unreset_matches_first")) is not bool
                or smoke.get("unreset_matches_first") is not True
                or smoke.get("shape") != [3]
                or type(smoke.get("init_seconds")) not in (int, float)
                or not 0 <= smoke["init_seconds"] <= 10
                or type(smoke.get("act_seconds")) not in (int, float)
                or not 0 <= smoke["act_seconds"] <= 5
            ):
                raise OrchestrationError(f"A4 root-package validation/smoke gate failed: {cell}")
            package_receipt = {
                "format": "haic-drq-teacher-replay-root-package-gate-v1",
                "study_protocol_sha256": self.protocol_sha256,
                "identity": identity,
                "actor_sha256": identity["actor_sha256"],
                "archive_path": str(archive_path.resolve()),
                "archive_sha256": sha256_file(archive_path),
                "expected_model_filename": package["expected_model_filename"],
                "model_format": package["model_format"],
                "modules": package["modules"],
                "smoke": smoke,
            }
            _write_exclusive(receipt_path, package_receipt, self.evaluations_root)
            records.append({
                "identity": identity,
                "archive_path": str(archive_path.resolve()),
                "archive_sha256": package_receipt["archive_sha256"],
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": sha256_file(receipt_path),
            })
        return records

    def _validate_screen_packages(
        self, stage_root: Path, actors: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        package_root = stage_root / "packages"
        expected_ids = {self._cell_id(actor["identity"]) for actor in actors}
        if not package_root.is_dir() or {path.stem for path in package_root.glob("*.json")} != expected_ids:
            raise OrchestrationError("A4 package receipt inventory differs from ten screen actors")
        records = []
        for actor in actors:
            cell = self._cell_id(actor["identity"])
            archive_path = package_root / f"{cell}.zip"
            receipt_path = package_root / f"{cell}.json"
            receipt = read_json(receipt_path)
            if (
                not archive_path.is_file()
                or archive_path.is_symlink()
                or set(receipt) != {
                    "format", "study_protocol_sha256", "identity", "actor_sha256", "archive_path",
                    "archive_sha256", "expected_model_filename", "model_format", "modules", "smoke",
                }
                or receipt.get("format") != "haic-drq-teacher-replay-root-package-gate-v1"
                or receipt.get("study_protocol_sha256") != self.protocol_sha256
                or receipt.get("identity") != actor["identity"]
                or receipt.get("actor_sha256") != actor["identity"]["actor_sha256"]
                or receipt.get("archive_path") != str(archive_path.resolve())
                or receipt.get("archive_sha256") != sha256_file(archive_path)
                or receipt.get("expected_model_filename") != "model.pt"
                or receipt.get("model_format") != "haic-drq-v2-actor-v1"
                or not isinstance(receipt.get("modules"), list)
            ):
                raise OrchestrationError(f"A4 package receipt/archive identity mismatch: {cell}")
            records.append({
                "identity": actor["identity"],
                "archive_path": str(archive_path.resolve()),
                "archive_sha256": sha256_file(archive_path),
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": sha256_file(receipt_path),
            })
        return records

    def _selected_screen_wrappers(self, screen: dict[str, Any]) -> dict[tuple[int, str], Path]:
        if not screen["gate_passed"]:
            raise OrchestrationError("screen gate failed; confirmation is forbidden")
        selection = read_json(screen["selection_path"])
        if not isinstance(selection, dict) or selection.get("study_protocol_sha256") != self.protocol_sha256:
            raise OrchestrationError("screen selection is malformed or belongs to another protocol")
        selected = selection.get("selected_identities")
        if not isinstance(selected, list) or len(selected) != 6:
            raise OrchestrationError("screen selection does not freeze six actor identities")
        result = {}
        for record in selected:
            if not isinstance(record, dict):
                raise OrchestrationError("invalid selected screen actor identity")
            role = (record.get("source_learner_seed"), record.get("arm"))
            if role in result:
                raise OrchestrationError("duplicate frozen source/arm selection")
            path = Path(record.get("screen_wrapper_path", "")).resolve()
            if sha256_file(path) != record.get("screen_wrapper_sha256"):
                raise OrchestrationError(f"selected screen wrapper hash mismatch: {path}")
            wrapper = read_json(path)
            if compare._identity(wrapper) != {
                key: record[key] for key in (
                    "source_learner_seed", "arm", "actor_sha256", "checkpoint_sha256",
                    "checkpoint_online_step",
                )
            }:
                raise OrchestrationError(f"selected screen wrapper identity mismatch: {path}")
            result[role] = path
        expected = {(learner, arm) for learner in (0, 1) for arm in ARMS}
        if set(result) != expected:
            raise OrchestrationError("screen selection does not include every source/arm role")
        return result

    def _stage_actor_list(self, stage: str, screen: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if stage == "screen":
            return self._screen_actors()
        if screen is None:
            raise OrchestrationError(f"{stage} requires a completed screen")
        if stage == "confirmation":
            selected_wrappers = self._selected_screen_wrappers(screen)
            roles = {(learner, arm) for learner in (0, 1) for arm in ARMS}
            actors = []
            for learner, arm in sorted(roles, key=lambda role: (role[0], ARMS.index(role[1]))):
                wrapper = read_json(selected_wrappers[(learner, arm)])
                identity = compare._identity(wrapper)
                actor = self.inventory.get(_role_key(identity))
                if actor is None or actor["identity"] != identity:
                    raise OrchestrationError("selected screen identity is absent from the checkpoint catalog")
                actors.append(actor)
            if len(actors) != 6:
                raise OrchestrationError("confirmation must contain exactly six selected actors")
            return actors
        if stage == "blind":
            finalist = read_json(screen["finalist_path"])
            identity = {
                "source_learner_seed": finalist["source_learner_seed"],
                "arm": finalist["arm"],
                "actor_sha256": finalist["actor_sha256"],
                "checkpoint_sha256": finalist["checkpoint_sha256"],
                "checkpoint_online_step": finalist["checkpoint_online_step"],
            }
            actor = self.inventory.get(_role_key(identity))
            if actor is None or actor["identity"] != identity:
                raise OrchestrationError("screen-fixed blind finalist differs from the actor catalog")
            return [actor]
        raise OrchestrationError(f"unsupported evaluation stage: {stage}")

    def _evaluator_prefix(self) -> list[str]:
        executable = self.protocol["runtime"]["evaluation_executable"]
        if not isinstance(executable, str) or not executable.strip():
            raise OrchestrationError("frozen runtime.evaluation_executable is absent")
        try:
            prefix = shlex.split(executable)
        except ValueError as error:
            raise OrchestrationError("runtime.evaluation_executable is malformed") from error
        if len(prefix) == 1:
            return [prefix[0], str((self.repo_root / "evaluate_policy.py").resolve())]
        if len(prefix) == 3 and prefix[1:] == ["-m", "evaluate_policy"]:
            return prefix
        raise OrchestrationError(
            "runtime.evaluation_executable must be a Python executable or '<python> -m evaluate_policy'"
        )

    def _verify_actor_unchanged(self, actor: dict[str, Any]) -> None:
        if sha256_file(self.protocol_file) != self.protocol_sha256:
            raise OrchestrationError("frozen study protocol changed during evaluation orchestration")
        identity = actor["identity"]
        if sha256_file(actor["actor_path"]) != identity["actor_sha256"]:
            raise OrchestrationError(f"candidate actor hash changed: {actor['actor_path']}")
        if sha256_file(actor["checkpoint_path"]) != actor["checkpoint_sha256"]:
            raise OrchestrationError(f"candidate checkpoint hash changed: {actor['checkpoint_path']}")
        if sha256_file(actor["checkpoint_manifest_path"]) != actor["checkpoint_manifest_sha256"]:
            raise OrchestrationError(
                f"candidate checkpoint manifest changed: {actor['checkpoint_manifest_path']}"
            )
        if sha256_file(actor["config_path"]) != actor["config_sha256"]:
            raise OrchestrationError(f"candidate config changed: {actor['config_path']}")
        catalog = actor.get("catalog_path")
        if catalog is not None and sha256_file(catalog) != actor["catalog_sha256"]:
            raise OrchestrationError(f"candidate checkpoint catalog changed: {catalog}")

    def _evaluate_actor(
        self,
        stage: str,
        actor: dict[str, Any],
        stage_root: Path,
        *,
        previous_wrapper_path: Path | None = None,
        diagnostic: bool = False,
    ) -> tuple[Path, Path]:
        self._verify_actor_unchanged(actor)
        evaluator_prefix = self._evaluator_prefix()
        previous_pointer = None
        if previous_wrapper_path is not None:
            previous_partition = "screen" if stage == "confirmation" else "confirmation"
            previous_wrapper, _, _ = compare._validate_wrapper(
                previous_wrapper_path, self.protocol, self.protocol_sha256, previous_partition
            )
            if compare._identity(previous_wrapper) != actor["identity"]:
                raise OrchestrationError(f"preceding evaluator actor differs from {stage} candidate")
            previous_pointer = Path(previous_wrapper["evaluator_receipt_path"]).resolve()
            if sha256_file(previous_pointer) != previous_wrapper["evaluator_receipt_sha256"]:
                raise OrchestrationError(f"previous evaluator pointer changed: {previous_pointer}")
        paths = self._cell_paths(stage_root, actor)
        _assert_beneath(paths["cell"], self.evaluations_root, "actor evaluation cell")
        _mkdir(paths["cell"], self.evaluations_root, exclusive=True)
        marker = {
            "schema_version": 1,
            "study_protocol_sha256": self.protocol_sha256,
            "partition": stage,
            "identity": actor["identity"],
            "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        _write_exclusive(paths["started"], marker, self.evaluations_root)

        command = [
            *evaluator_prefix,
            "--protocol-file", str(self.protocol_file),
            "--partition", stage,
            "--model", str(actor["actor_path"]),
            "--run-dir", str(actor["run_dir"]),
            "--evaluations-dir", str(paths["evaluator_output"]),
            "--output", str(paths["evaluator_pointer"]),
            "--workers", "1",
            "--max-steps", str(self.protocol["max_steps"]),
            "--frame-skip", str(self.protocol["frame_skip"]),
        ]
        if previous_pointer is not None:
            command.extend(("--previous-evaluation", str(previous_pointer)))
        if diagnostic:
            command.append("--diagnostic-confirmation")
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.evaluator_timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise OrchestrationError(f"{stage} evaluator did not complete for {self._cell_id(actor['identity'])}: {error}") from error
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "no evaluator output")[-4000:]
            raise OrchestrationError(
                f"{stage} evaluator failed for {self._cell_id(actor['identity'])}: {detail}"
            )
        self._verify_actor_unchanged(actor)
        pointer_path = paths["evaluator_pointer"]
        _assert_beneath(pointer_path, self.evaluations_root, "evaluator pointer")
        if pointer_path.is_symlink():
            raise OrchestrationError(f"evaluator pointer must not be a symlink: {pointer_path}")
        if not pointer_path.is_file():
            raise OrchestrationError(f"evaluator exited successfully without its immutable pointer: {pointer_path}")
        pointer_bytes = pointer_path.read_bytes()
        pointer = read_json(pointer_path)
        self._check_pointer_identity(pointer, stage, actor, diagnostic=diagnostic)
        wrapper = {
            "study_protocol_sha256": self.protocol_sha256,
            **actor["identity"],
            "partition": stage,
            "evaluator_receipt_path": str(pointer_path.resolve()),
            "evaluator_receipt_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        }
        if stage != "screen":
            if previous_wrapper_path is None:
                raise OrchestrationError(f"{stage} actor lacks a frozen preceding-stage wrapper")
            previous_wrapper = read_json(previous_wrapper_path)
            wrapper["screen_receipt_sha256"] = (
                hashlib.sha256(previous_wrapper_path.read_bytes()).hexdigest()
                if stage == "confirmation"
                else previous_wrapper["screen_receipt_sha256"]
            )
            wrapper["screen_candidate_identity"] = (
                compare._identity(previous_wrapper)
                if stage == "confirmation"
                else previous_wrapper["screen_candidate_identity"]
            )
        wrapper_path = paths["wrapper"]
        _write_exclusive(wrapper_path, wrapper, self.evaluations_root)
        return wrapper_path, pointer_path

    def _check_pointer_identity(
        self, pointer: dict[str, Any], stage: str, actor: dict[str, Any], *, diagnostic: bool
    ) -> None:
        if not isinstance(pointer, dict) or set(pointer) != compare.POINTER_KEYS:
            raise OrchestrationError("evaluator pointer has a missing or unsupported schema")
        if (
            pointer.get("partition") != stage
            or pointer.get("protocol_name") != f"{self.protocol['name']}-{stage}"
            or pointer.get("protocol_sha256") != self.protocol_sha256
            or pointer.get("diagnostic_only") is not diagnostic
            or not isinstance(pointer.get("evaluation_dir"), str)
            or not Path(pointer["evaluation_dir"]).is_absolute()
            or not isinstance(pointer.get("ranked"), list)
            or len(pointer["ranked"]) != 1
        ):
            raise OrchestrationError("evaluator pointer partition/protocol/diagnostic identity mismatch")
        evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
        cell = self._cell_paths(self._stage_root(stage), actor)["evaluator_output"]
        if not evaluation_dir.is_relative_to(cell.resolve()):
            raise OrchestrationError(f"evaluator output escaped its immutable actor location: {evaluation_dir}")
        result = pointer["ranked"][0]
        if not isinstance(result, dict) or any(
            result.get(field) != actor["identity"]["actor_sha256"]
            for field in ("archive_sha256", "policy_sha256", "evaluation_archive_sha256")
        ):
            raise OrchestrationError("evaluator result actor hash differs from the frozen catalog")
        if result.get("algorithm") != "drq-v2" or result.get("diagnostic_only") is not diagnostic:
            raise OrchestrationError("evaluator result algorithm/diagnostic identity mismatch")

    def _new_stage(self, stage: str) -> Path:
        _mkdir(self.evaluations_root, self.run_root)
        _mkdir(self.base_root, self.evaluations_root)
        root = self._stage_root(stage)
        _mkdir(root, self.evaluations_root, exclusive=True)
        _mkdir(root / "actors", self.evaluations_root, exclusive=True)
        return root

    def _completed_files(self, stage_root: Path) -> dict[str, str]:
        files = {}
        for path in sorted(stage_root.rglob("*")):
            if path.name == STAGE_COMPLETE:
                continue
            if path.is_symlink():
                raise OrchestrationError(f"completed stage contains a symlink: {path}")
            if path.is_file():
                relative = path.relative_to(stage_root).as_posix()
                files[relative] = sha256_file(path)
            elif path.is_dir() and path.name.startswith((".pending-", ".tmp-")):
                raise OrchestrationError(f"completed stage contains a partial evaluator directory: {path}")
        return files

    def _finish_stage(
        self, stage_root: Path, stage: str, *, diagnostic: bool, status: str, details: dict[str, Any]
    ) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "study_protocol_sha256": self.protocol_sha256,
            "partition": stage,
            "diagnostic_only": diagnostic,
            "status": status,
            "details": details,
            "files": self._completed_files(stage_root),
        }
        _write_exclusive(stage_root / STAGE_COMPLETE, record, self.evaluations_root)
        return record

    def _check_complete_file_set(self, stage_root: Path) -> dict[str, Any]:
        _assert_beneath(stage_root, self.evaluations_root, "completed stage directory")
        marker_path = stage_root / STAGE_COMPLETE
        if not marker_path.is_file():
            raise OrchestrationError(
                f"partial {stage_root.name} stage artifacts found; consumed actor cells will not be rerun"
            )
        _assert_beneath(marker_path, self.evaluations_root, "stage completion marker")
        marker = read_json(marker_path)
        required = {
            "schema_version", "study_protocol_sha256", "partition", "diagnostic_only",
            "status", "details", "files",
        }
        if not isinstance(marker, dict) or set(marker) != required:
            raise OrchestrationError(f"malformed completed-stage marker: {marker_path}")
        if (
            marker["schema_version"] != 1
            or marker["study_protocol_sha256"] != self.protocol_sha256
            or marker["partition"] != stage_root.name
            or type(marker["diagnostic_only"]) is not bool
            or not isinstance(marker["details"], dict)
            or not isinstance(marker["files"], dict)
        ):
            raise OrchestrationError(f"completed-stage identity mismatch: {marker_path}")
        actual = self._completed_files(stage_root)
        if marker["files"] != actual:
            raise OrchestrationError(f"completed-stage artifact inventory/hash mismatch: {stage_root}")
        return marker

    def _new_or_completed(self, stage: str) -> tuple[Path, dict[str, Any] | None]:
        stage_root = self._stage_root(stage)
        if stage_root.exists():
            _assert_beneath(stage_root, self.evaluations_root, "stage directory")
            marker = self._check_complete_file_set(stage_root)
            return stage_root, marker
        return self._new_stage(stage), None

    def _join(
        self,
        stage: str,
        stage_root: Path,
        wrappers: list[Path],
        lineage: list[Path],
    ) -> tuple[dict[str, Any], str, Path]:
        output = stage_root / "paired-decision.json"
        arguments = [
            "--protocol-file", str(self.protocol_file),
            "--partition", stage,
            "--output", str(output),
            "--repo-root", str(self.repo_root),
        ]
        for path in wrappers:
            arguments.extend(("--receipt", str(path.resolve())))
        for path in lineage:
            arguments.extend(("--lineage-pointer", str(path.resolve())))
        try:
            status = compare.main(arguments)
        except Exception as error:
            raise OrchestrationError(f"paired {stage} receipt join failed: {error}") from error
        if status != 0 or not output.is_file():
            raise OrchestrationError(f"paired {stage} receipt join failed")
        decision, digest = compare._sealed_json(output, "paired decision receipt")
        if output.with_suffix(output.suffix + ".sha256").read_text(encoding="ascii").strip() != digest:
            raise OrchestrationError(f"paired {stage} decision sidecar mismatch")
        return decision, digest, output

    def _metrics(self, wrapper_path: Path, stage: str = "screen") -> dict[str, Any]:
        wrapper, _, report = compare._validate_wrapper(
            wrapper_path, self.protocol, self.protocol_sha256, stage
        )
        canonical = list(report["canonical"].values())
        progress_values = [row["progress"] for row in canonical]
        if not canonical or any(type(value) not in (int, float) or not math.isfinite(value)
                                for value in progress_values):
            raise OrchestrationError("screen canonical progress values are malformed")
        lap_times = [row["lap_time_ms"] for row in canonical if row["finished"]]
        return {
            "identity": compare._identity(wrapper),
            "finish_count": report["finishes"],
            "canonical_cell_count": report["cell_count"],
            "canonical_mean_progress": sum(progress_values) / len(progress_values),
            "completed_mean_lap_time_ms": (
                sum(lap_times) / len(lap_times) if lap_times else None
            ),
        }

    def _expected_selected_wrapper_paths(self, screen: dict[str, Any]) -> list[Path]:
        selected = self._selected_screen_wrappers(screen)
        return [
            selected[(learner, arm)]
            for learner in (0, 1)
            for arm in ARMS
        ]

    def _validate_screen_completed(self, stage_root: Path, marker: dict[str, Any]) -> dict[str, Any]:
        if marker["diagnostic_only"] is not False or marker["status"] not in ("screen-passed", "screen-gate-failed"):
            raise OrchestrationError("completed screen stage has invalid status")
        actors = self._screen_actors()
        package_receipts = self._validate_screen_packages(stage_root, actors)
        wrappers = [self._cell_paths(stage_root, actor)["wrapper"] for actor in actors]
        self._validate_cell_records(stage_root, "screen", self._screen_actors(), wrappers)
        decision, decision_sha, decision_path = self._join_read_only("screen", wrappers, [])
        gate, _ = compare._sealed_json(stage_root / "screen-gate.json", "screen gate receipt")
        if (
            gate.get("paired_decision_sha256") != decision_sha
            or gate.get("passed") is not (marker["status"] == "screen-passed")
        ):
            raise OrchestrationError("screen gate decision/marker mismatch")
        expected_gate, passed, diagnostic_required, selection_path, finalist_path, _ = self._screen_selection_read_only(
            stage_root, wrappers, decision, sha256_file(decision_path), decision_path, package_receipts
        )
        if gate != expected_gate or passed is not (marker["status"] == "screen-passed"):
            raise OrchestrationError("completed screen selection/gate does not match its evaluator receipts")
        if passed:
            selection = read_json(selection_path)
            decision_record, decision_sha = compare._sealed_json(decision_path, "screen decision")
            decision_record["_path"] = str(decision_path.resolve())
            decision_record["_sha256"] = decision_sha
            finalist, _ = compare._sealed_json(finalist_path, "screen finalist")
            try:
                fixed_at = datetime.fromisoformat(finalist["fixed_at_utc"].replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as error:
                raise OrchestrationError("screen finalist timestamp is malformed") from error
            if fixed_at.tzinfo is None:
                raise OrchestrationError("screen finalist timestamp must include a timezone")
            compare._validate_selection_record(selection_path, self.protocol, self.protocol_sha256,
                                               decision_record, finalist)
            compare._validate_finalist_pointer(
                finalist_path, self.protocol, self.protocol_sha256, decision_record,
                (fixed_at + timedelta(seconds=1)).isoformat().replace("+00:00", "Z"),
            )
        elif selection_path is not None or finalist_path is not None:
            raise OrchestrationError("failed screen gate must not contain a blind finalist")
        details = marker["details"]
        if (
            details.get("gate_passed") is not passed
            or details.get("diagnostic_confirmation_required") is not diagnostic_required
            or details.get("paired_decision_path") != str(decision_path.resolve())
            or details.get("paired_decision_sha256") != decision_sha
            or details.get("screen_gate_path") != str((stage_root / "screen-gate.json").resolve())
            or details.get("selection_path") != (str(selection_path.resolve()) if selection_path else None)
            or details.get("finalist_path") != (str(finalist_path.resolve()) if finalist_path else None)
            or details.get("package_receipt_count") != len(package_receipts)
        ):
            raise OrchestrationError("completed screen stage details mismatch")
        return {
            "stage": "screen",
            "gate_passed": passed,
            "diagnostic_confirmation_required": diagnostic_required,
            "decision_path": decision_path,
            "gate_path": stage_root / "screen-gate.json",
            "selection_path": selection_path,
            "finalist_path": finalist_path,
            "wrapper_paths": wrappers,
            "package_receipts": package_receipts,
            "decision": decision,
        }

    def _screen_selection_read_only(
        self, stage_root: Path, wrappers: list[Path], decision: dict[str, Any],
        decision_sha: str, decision_path: Path, package_receipts: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], bool, bool, Path | None, Path | None, Path]:
        # Recompute the frozen gate without creating or changing any artifact.
        records = []
        metrics_by_key = {}
        wrappers_by_key = {}
        for path in wrappers:
            metrics = self._metrics(path)
            key = _role_key(metrics["identity"])
            metrics_by_key[key] = metrics
            wrappers_by_key[key] = path.resolve()
            score = screen_score(metrics)
            records.append({
                "candidate_identity": metrics["identity"],
                "wrapper_path": str(path.resolve()),
                "wrapper_sha256": sha256_file(path),
                "metrics": metrics,
                "score_tuple": [score[0], score[1], 0.0 if math.isinf(score[2]) else -score[2]],
            })
        chosen = {}
        for learner in (0, 1):
            for arm in STUDENT_ARMS:
                candidates = [
                    {"identity": metrics_by_key[(learner, arm, step)]["identity"],
                     "metrics": metrics_by_key[(learner, arm, step)],
                     "wrapper_path": wrappers_by_key[(learner, arm, step)]}
                    for step in self.protocol["budgets"]["checkpoint_online_steps"]
                ]
                chosen[(learner, arm)] = choose_screen_actor(candidates)
        selected = []
        for learner in (0, 1):
            for arm in ARMS:
                metrics = (metrics_by_key[(learner, arm, None)] if arm == "unchanged-source"
                           else chosen[(learner, arm)]["metrics"])
                identity = metrics["identity"]
                wrapper_path = (wrappers_by_key[(learner, arm, None)] if arm == "unchanged-source"
                                else chosen[(learner, arm)]["wrapper_path"])
                score = screen_score(metrics)
                selected.append({
                    **identity,
                    "screen_wrapper_path": str(wrapper_path),
                    "screen_wrapper_sha256": sha256_file(wrapper_path),
                    "score_tuple": [score[0], score[1], 0.0 if math.isinf(score[2]) else -score[2]],
                })
        joined_sources = {item["source_learner_seed"]: item
                          for item in decision["result"].get("sources", [])}
        failures = []
        for learner in (0, 1):
            teacher = chosen[(learner, "teacher-replay")]["metrics"]
            online = chosen[(learner, "online-only")]["metrics"]
            joined_source = joined_sources.get(learner)
            joined_candidate = next((item for item in (joined_source or {}).get("candidates", [])
                                     if item["arm"] == "teacher-replay"
                                     and item["checkpoint_online_step"]
                                     == chosen[(learner, "teacher-replay")]["identity"]["checkpoint_online_step"]), None)
            if teacher["finish_count"] <= 0:
                failures.append(f"learner {learner} selected teacher actor has zero canonical finishes")
            if teacher["finish_count"] < online["finish_count"]:
                failures.append(f"learner {learner} selected teacher actor trails selected online-only actor")
            if joined_candidate is None or joined_candidate.get("screen_only_eligible") is not True:
                failures.append(f"learner {learner} selected teacher actor fails the paired screen eligibility gate")
        diagnostic_required = any(
            (metrics_by_key[(learner, arm, None)] if arm == "unchanged-source"
             else chosen[(learner, arm)]["metrics"])["finish_count"] == 0
            for learner in (0, 1) for arm in ("unchanged-source", "online-only")
        )
        passed = not failures
        gate = {
            "schema_version": 1,
            "study_protocol_sha256": self.protocol_sha256,
            "partition": "screen",
            "paired_decision_path": str(decision_path.resolve()),
            "paired_decision_sha256": decision_sha,
            "passed": passed,
            "diagnostic_confirmation_required": diagnostic_required,
            "failures": failures,
            "candidate_receipts": [
                {key: item[key] for key in ("candidate_identity", "wrapper_path", "wrapper_sha256")}
                for item in records
            ],
            "package_receipts": package_receipts,
            "screen_scores": [
                {"candidate_identity": item["candidate_identity"],
                 "wrapper_sha256": item["wrapper_sha256"], "metrics": item["metrics"]}
                for item in records
            ],
            "selected_identities": selected,
        }
        selection_path = stage_root / "selected-candidates.json" if passed else None
        finalist_path = stage_root / "screen-finalist.json" if passed else None
        return gate, passed, diagnostic_required, selection_path, finalist_path, stage_root / "screen-gate.json"

    def _join_read_only(
        self, stage: str, wrappers: list[Path], lineage: list[Path]
    ) -> tuple[dict[str, Any], str, Path]:
        decision = compare.compare_receipts(self.protocol, self.protocol_sha256, stage, wrappers, lineage)
        path = self._stage_root(stage) / "paired-decision.json"
        if not path.is_file():
            raise OrchestrationError(f"completed {stage} stage lacks paired decision receipt")
        on_disk, actual_digest = compare._sealed_json(path, f"paired {stage} decision")
        if on_disk != decision:
            raise OrchestrationError(f"completed {stage} paired decision differs from its actor receipts")
        return decision, actual_digest, path

    def _validate_cell_records(
        self,
        stage_root: Path,
        stage: str,
        actors: list[dict[str, Any]],
        wrappers: list[Path],
        *,
        diagnostic: bool = False,
    ) -> None:
        _assert_beneath(stage_root, self.evaluations_root, "completed stage directory")
        expected_ids = {self._cell_id(actor["identity"]) for actor in actors}
        actor_root = stage_root / "actors"
        if not actor_root.is_dir() or {path.name for path in actor_root.iterdir()} != expected_ids:
            raise OrchestrationError(f"{stage} actor cell inventory differs from the frozen stage")
        for actor, wrapper_path in zip(actors, wrappers):
            paths = self._cell_paths(stage_root, actor)
            _assert_beneath(paths["cell"], self.evaluations_root, "completed actor cell")
            marker = read_json(paths["started"])
            if marker != {
                "schema_version": 1,
                "study_protocol_sha256": self.protocol_sha256,
                "partition": stage,
                "identity": actor["identity"],
                "started_at_utc": marker.get("started_at_utc"),
            }:
                raise OrchestrationError(f"{stage} actor start marker identity mismatch: {paths['started']}")
            if not isinstance(marker["started_at_utc"], str) or not wrapper_path.is_file():
                raise OrchestrationError(f"{stage} actor cell is incomplete: {paths['cell']}")
            wrapper = read_json(wrapper_path)
            expected_keys = compare.WRAPPER_KEYS if stage == "screen" else compare.WRAPPER_KEYS | compare.WRAPPER_LINEAGE_KEYS
            if set(wrapper) != expected_keys or wrapper.get("partition") != stage:
                raise OrchestrationError(f"malformed {stage} actor wrapper: {wrapper_path}")
            if compare._identity(wrapper) != actor["identity"] or wrapper.get("study_protocol_sha256") != self.protocol_sha256:
                raise OrchestrationError(f"{stage} actor wrapper identity mismatch: {wrapper_path}")
            pointer_path = paths["evaluator_pointer"].resolve()
            if (
                wrapper.get("evaluator_receipt_path") != str(pointer_path)
                or not pointer_path.is_file()
                or sha256_file(pointer_path) != wrapper.get("evaluator_receipt_sha256")
            ):
                raise OrchestrationError(f"{stage} evaluator pointer hash/path mismatch: {wrapper_path}")
            self._check_pointer_identity(read_json(pointer_path), stage, actor,
                                         diagnostic=diagnostic)

    def _verify_stage_marker(
        self, stage_root: Path, marker: dict[str, Any], expected_diagnostic: bool
    ) -> None:
        if marker["diagnostic_only"] is not expected_diagnostic:
            raise OrchestrationError("completed stage diagnostic mode differs from the requested stage")

    def _screen(self) -> dict[str, Any]:
        stage = "screen"
        stage_root, marker = self._new_or_completed(stage)
        if marker is not None:
            return self._validate_screen_completed(stage_root, marker)
        actors = self._screen_actors()
        package_receipts = self._package_screen_actors(stage_root, actors)
        wrappers = []
        for actor in actors:
            wrapper, _ = self._evaluate_actor(stage, actor, stage_root)
            wrappers.append(wrapper)
        decision, decision_sha, decision_path = self._join(stage, stage_root, wrappers, [])
        gate, passed, diagnostic_required, selection_path, finalist_path, gate_path = self._screen_selection(
            stage_root, wrappers, decision, decision_sha, decision_path, package_receipts
        )
        details = {
            "gate_passed": passed,
            "diagnostic_confirmation_required": diagnostic_required,
            "paired_decision_path": str(decision_path.resolve()),
            "paired_decision_sha256": decision_sha,
            "screen_gate_path": str(gate_path.resolve()),
            "selection_path": str(selection_path.resolve()) if selection_path else None,
            "finalist_path": str(finalist_path.resolve()) if finalist_path else None,
            "package_receipt_count": len(package_receipts),
        }
        self._finish_stage(
            stage_root, stage, diagnostic=False,
            status="screen-passed" if passed else "screen-gate-failed", details=details,
        )
        return {
            "stage": stage,
            "gate_passed": passed,
            "diagnostic_confirmation_required": diagnostic_required,
            "decision_path": decision_path,
            "gate_path": gate_path,
            "selection_path": selection_path,
            "finalist_path": finalist_path,
            "wrapper_paths": wrappers,
            "package_receipts": package_receipts,
            "decision": decision,
        }

    def _read_completed_screen(self) -> dict[str, Any]:
        stage_root = self._stage_root("screen")
        if not stage_root.exists():
            raise OrchestrationError("screen stage is not complete; run --stage screen first")
        marker = self._check_complete_file_set(stage_root)
        return self._validate_screen_completed(stage_root, marker)

    def _read_completed_confirmation(self, screen: dict[str, Any]) -> dict[str, Any]:
        stage_root = self._stage_root("confirmation")
        if not stage_root.exists():
            raise OrchestrationError("confirmation stage is not complete; run --stage confirmation first")
        marker = self._check_complete_file_set(stage_root)
        self._verify_stage_marker(stage_root, marker, screen["diagnostic_confirmation_required"])
        return self._validate_confirmation_completed(stage_root, marker, screen)

    def _screen_selection(
        self, stage_root: Path, wrapper_paths: list[Path], decision: dict[str, Any], decision_sha: str,
        decision_path: Path, package_receipts: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], bool, bool, Path | None, Path | None, Path]:
        gate_record, gate_passed, diagnostic_required, selection_path, finalist_path, gate_path = self._screen_selection_read_only(
            stage_root, wrapper_paths, decision, decision_sha, decision_path, package_receipts
        )
        _write_sealed(gate_path, gate_record, self.evaluations_root)
        if gate_passed:
            selected = gate_record["selected_identities"]
            by_role = {(item["source_learner_seed"], item["arm"]): item for item in selected}
            teacher_0 = by_role[(0, "teacher-replay")]
            teacher_1 = by_role[(1, "teacher-replay")]
            finalist = teacher_0 if teacher_0["score_tuple"] >= teacher_1["score_tuple"] else teacher_1
            if teacher_0["score_tuple"] == teacher_1["score_tuple"]:
                finalist = teacher_0
            selection_record = {
                "format": compare.SELECTION_FORMAT,
                "study_protocol_sha256": self.protocol_sha256,
                "screen_decision_path": str(decision_path.resolve()),
                "screen_decision_sha256": decision_sha,
                "candidate_receipts": gate_record["candidate_receipts"],
                "selected_identities": selected,
                "selection_rule": (
                    "canonical finish count desc, canonical mean progress desc, completed mean lap time asc; "
                    "earlier checkpoint on exact tuple tie; learner 0 on exact treatment finalist tie"
                ),
                "blind_finalist": finalist,
            }
            _write_exclusive(selection_path, selection_record, self.evaluations_root)
            finalist_record = {
                "schema_version": 1,
                "receipt_type": compare.FINALIST_TYPE,
                "study_protocol_sha256": self.protocol_sha256,
                "partition": "screen",
                "source_learner_seed": finalist["source_learner_seed"],
                "arm": finalist["arm"],
                "actor_sha256": finalist["actor_sha256"],
                "checkpoint_sha256": finalist["checkpoint_sha256"],
                "checkpoint_online_step": finalist["checkpoint_online_step"],
                "screen_receipt_sha256": finalist["screen_wrapper_sha256"],
                "screen_decision_path": str(decision_path.resolve()),
                "screen_decision_sha256": decision_sha,
                "selection_file_path": str(selection_path.resolve()),
                "selection_file_sha256": sha256_file(selection_path),
                "fixed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            _write_sealed(finalist_path, finalist_record, self.evaluations_root)
        return gate_record, gate_passed, diagnostic_required, selection_path, finalist_path, gate_path

    def _confirmation(self, screen: dict[str, Any], diagnostic_confirmation: bool) -> dict[str, Any]:
        if not screen["gate_passed"]:
            raise OrchestrationError("screen gate failed; no confirmation actors may be evaluated")
        diagnostic_required = screen["diagnostic_confirmation_required"]
        if diagnostic_confirmation is not diagnostic_required:
            if diagnostic_required:
                raise OrchestrationError(
                    "a selected unchanged-source/online-only actor had zero screen finishes; "
                    "confirmation requires --diagnostic-confirmation"
                )
            raise OrchestrationError("--diagnostic-confirmation is allowed only for a zero-finish screen control")
        stage = "confirmation"
        stage_root = self._stage_root(stage)
        if stage_root.exists():
            marker = self._check_complete_file_set(stage_root)
            self._verify_stage_marker(stage_root, marker, diagnostic_required)
            return self._validate_confirmation_completed(stage_root, marker, screen)
        stage_root = self._new_stage(stage)
        actors = self._stage_actor_list(stage, screen)
        screen_wrappers = self._selected_screen_wrappers(screen)
        wrapper_paths = []
        for actor in actors:
            identity = actor["identity"]
            screen_wrapper = screen_wrappers[(identity["source_learner_seed"], identity["arm"])]
            wrapper, _ = self._evaluate_actor(
                stage, actor, stage_root, previous_wrapper_path=screen_wrapper,
                diagnostic=diagnostic_required,
            )
            wrapper_paths.append(wrapper)
        if diagnostic_required:
            diagnostic_path = stage_root / "diagnostic-result.json"
            diagnostic = {
                "schema_version": 1,
                "receipt_type": "haic-drq-teacher-replay-diagnostic-confirmation-v1",
                "study_protocol_sha256": self.protocol_sha256,
                "partition": "confirmation",
                "diagnostic_only": True,
                "promoting": False,
                "blind_authorized": False,
                "screen_lineage": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in self._expected_selected_wrapper_paths(screen)
                ],
                "actor_receipts": [
                    {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for path in wrapper_paths
                ],
                "complete_actor_count": len(wrapper_paths),
            }
            _write_sealed(diagnostic_path, diagnostic, self.evaluations_root)
            details = {
                "diagnostic_result_path": str(diagnostic_path.resolve()),
                "diagnostic_result_sha256": sha256_file(diagnostic_path),
                "complete_actor_count": len(wrapper_paths),
                "passed": False,
            }
            self._finish_stage(stage_root, stage, diagnostic=True, status="diagnostic-complete", details=details)
            return {
                "stage": stage,
                "diagnostic_only": True,
                "passed": False,
                "wrapper_paths": wrapper_paths,
                "decision_path": None,
                "diagnostic_path": diagnostic_path,
            }

        lineage = self._expected_selected_wrapper_paths(screen)
        decision, decision_sha, decision_path = self._join(stage, stage_root, wrapper_paths, lineage)
        details = {
            "paired_decision_path": str(decision_path.resolve()),
            "paired_decision_sha256": decision_sha,
            "passed": decision["result"].get("passed") is True,
            "screen_lineage": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in lineage
            ],
        }
        self._finish_stage(stage_root, stage, diagnostic=False,
                           status="confirmation-passed" if details["passed"] else "confirmation-failed",
                           details=details)
        return {
            "stage": stage,
            "diagnostic_only": False,
            "passed": details["passed"],
            "wrapper_paths": wrapper_paths,
            "decision_path": decision_path,
            "decision": decision,
        }

    def _validate_confirmation_completed(
        self, stage_root: Path, marker: dict[str, Any], screen: dict[str, Any]
    ) -> dict[str, Any]:
        diagnostic = marker["diagnostic_only"]
        expected_status = "diagnostic-complete" if diagnostic else None
        if diagnostic and marker["status"] != expected_status:
            raise OrchestrationError("diagnostic confirmation stage has invalid status")
        actors = self._stage_actor_list("confirmation", screen)
        wrappers = [self._cell_paths(stage_root, actor)["wrapper"] for actor in actors]
        self._validate_cell_records(
            stage_root, "confirmation", actors, wrappers, diagnostic=diagnostic
        )
        lineage = self._expected_selected_wrapper_paths(screen)
        if diagnostic:
            for actor, wrapper in zip(actors, wrappers):
                previous = lineage[(actor["identity"]["source_learner_seed"] * 3)
                                   + ARMS.index(actor["identity"]["arm"])]
                self._validate_diagnostic_evaluator(wrapper, previous)
            path = stage_root / "diagnostic-result.json"
            diagnostic_record, diagnostic_sha = compare._sealed_json(path, "diagnostic confirmation receipt")
            expected_diagnostic = {
                "schema_version": 1,
                "receipt_type": "haic-drq-teacher-replay-diagnostic-confirmation-v1",
                "study_protocol_sha256": self.protocol_sha256,
                "partition": "confirmation",
                "diagnostic_only": True,
                "promoting": False,
                "blind_authorized": False,
                "screen_lineage": [
                    {"path": str(item.resolve()), "sha256": sha256_file(item)} for item in lineage
                ],
                "actor_receipts": [
                    {"path": str(item.resolve()), "sha256": sha256_file(item)} for item in wrappers
                ],
                "complete_actor_count": 6,
            }
            if diagnostic_record != expected_diagnostic or (
                diagnostic_sha != marker["details"].get("diagnostic_result_sha256")
                or marker["details"].get("diagnostic_result_path") != str(path.resolve())
                or marker["details"].get("complete_actor_count") != 6
                or marker["details"].get("passed") is not False
            ):
                raise OrchestrationError("diagnostic confirmation receipt is malformed")
            return {
                "stage": "confirmation", "diagnostic_only": True, "passed": False,
                "wrapper_paths": wrappers, "decision_path": None, "diagnostic_path": path,
            }
        decision, decision_sha, decision_path = self._join_read_only("confirmation", wrappers, lineage)
        passed = decision["result"].get("passed") is True
        if marker["status"] != ("confirmation-passed" if passed else "confirmation-failed"):
            raise OrchestrationError("confirmation stage status differs from the paired decision")
        if (
            marker["details"].get("passed") is not passed
            or marker["details"].get("paired_decision_sha256") != decision_sha
            or marker["details"].get("paired_decision_path") != str(decision_path.resolve())
            or marker["details"].get("screen_lineage") != [
                {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in lineage
            ]
        ):
            raise OrchestrationError("completed confirmation stage details mismatch")
        return {
            "stage": "confirmation", "diagnostic_only": False, "passed": passed,
            "wrapper_paths": wrappers, "decision_path": decision_path, "decision": decision,
        }

    def _validate_diagnostic_evaluator(self, wrapper_path: Path, previous_wrapper_path: Path) -> None:
        wrapper = read_json(wrapper_path)
        pointer_path = Path(wrapper["evaluator_receipt_path"]).resolve()
        pointer = read_json(pointer_path)
        if (
            wrapper.get("partition") != "confirmation"
            or wrapper.get("study_protocol_sha256") != self.protocol_sha256
            or sha256_file(pointer_path) != wrapper.get("evaluator_receipt_sha256")
            or pointer.get("diagnostic_only") is not True
            or pointer.get("partition") != "confirmation"
            or pointer.get("protocol_sha256") != self.protocol_sha256
            or pointer.get("protocol_name") != f"{self.protocol['name']}-confirmation"
            or not isinstance(pointer.get("ranked"), list) or len(pointer["ranked"]) != 1
        ):
            raise OrchestrationError("diagnostic evaluator pointer identity/hash mismatch")
        screen_wrapper = read_json(previous_wrapper_path)
        compare._validate_wrapper(previous_wrapper_path, self.protocol, self.protocol_sha256, "screen")
        if (
            wrapper.get("screen_receipt_sha256") != sha256_file(previous_wrapper_path)
            or wrapper.get("screen_candidate_identity") != compare._identity(screen_wrapper)
            or compare._identity(wrapper) != compare._identity(screen_wrapper)
        ):
            raise OrchestrationError("diagnostic confirmation wrapper has wrong screen-frozen lineage")
        result = pointer["ranked"][0]
        if (
            result.get("diagnostic_only") is not True
            or result.get("archive_sha256") != wrapper["actor_sha256"]
            or result.get("policy_sha256") != wrapper["actor_sha256"]
            or result.get("evaluation_archive_sha256") != wrapper["actor_sha256"]
            or result.get("eligible") is not True
            or result.get("determinism_audited") is not True
            or result.get("cpu_reload_matches") is not True
            or result.get("operational_failures") != 0
        ):
            raise OrchestrationError("diagnostic evaluator actor is incomplete or operationally invalid")
        directory = Path(pointer["evaluation_dir"]).resolve()
        stage_root = self._stage_root("confirmation")
        if not directory.is_relative_to(stage_root.resolve()):
            raise OrchestrationError("diagnostic evaluator artifact escapes confirmation output root")
        manifest = read_json(directory / "manifest.json")
        matrix = self.protocol["partitions"]["confirmation"]
        if (
            set(manifest) != compare.MANIFEST_KEYS
            or manifest.get("schema_version") != 2
            or manifest.get("partition") != "confirmation"
            or manifest.get("protocol") != f"{self.protocol['name']}-confirmation"
            or manifest.get("protocol_sha256") != self.protocol_sha256
            or manifest.get("diagnostic_only") is not True
            or manifest.get("cell_matrix") != matrix
            or manifest.get("frame_skip") != self.protocol["frame_skip"]
            or manifest.get("max_steps") != self.protocol["max_steps"]
        ):
            raise OrchestrationError("diagnostic evaluator manifest is malformed or mismatched")
        for name in compare.EVALUATOR_FILES:
            if not (directory / name).is_file():
                raise OrchestrationError(f"diagnostic evaluator output is incomplete: {directory / name}")
        if read_json(directory / "protocol.json") != matrix:
            raise OrchestrationError("diagnostic evaluator partition snapshot mismatch")
        protocol_snapshot = directory / "protocol_spec.json"
        if sha256_file(protocol_snapshot) != self.protocol_sha256 or read_json(protocol_snapshot) != self.protocol:
            raise OrchestrationError("diagnostic evaluator study protocol snapshot mismatch")
        candidate_archive = (directory / result["evaluation_archive_path"]).resolve()
        candidate_config = (directory / result["evaluation_run_config_path"]).resolve()
        source_config = Path(result["run_config_path"]).resolve()
        if (
            not candidate_archive.is_relative_to(directory)
            or not candidate_archive.is_file()
            or sha256_file(candidate_archive) != wrapper["actor_sha256"]
            or not candidate_config.is_relative_to(directory)
            or not candidate_config.is_file()
            or sha256_file(candidate_config) != result["run_config_sha256"]
            or not source_config.is_file()
            or sha256_file(source_config) != result["run_config_sha256"]
        ):
            raise OrchestrationError("diagnostic evaluator actor/config snapshot mismatch")
        worker_runtimes = manifest.get("worker_runtime")
        if not isinstance(worker_runtimes, list) or not worker_runtimes:
            raise OrchestrationError("diagnostic evaluator omitted worker runtime metadata")
        for runtime in worker_runtimes:
            compare._validate_runtime(runtime)
        previous_pointer = Path(screen_wrapper["evaluator_receipt_path"]).resolve()
        compare._validate_previous_lineage(
            directory, manifest.get("previous_evaluation"), wrapper, "confirmation", previous_pointer
        )
        summary = result
        if (
            set(summary) != compare.SUMMARY_KEYS
            or summary.get("algorithm") != "drq-v2"
            or summary.get("diagnostic_only") is not True
            or summary.get("eligible") is not True
            or summary.get("determinism_audited") is not True
            or summary.get("cpu_reload_matches") is not True
            or summary.get("operational_failures") != 0
            or summary.get("export_metadata", {}).get("format") != "haic-drq-v2-actor-v1"
            or summary.get("export_metadata", {}).get("action_spec", {}).get("frame_skip")
            != self.protocol["frame_skip"]
            or summary.get("expected_cells") != len(matrix["track_ids"]) * len(matrix["seeds"])
            or summary.get("canonical_episodes") != summary.get("expected_cells")
            or summary.get("expected_results") != 2 * summary.get("expected_cells")
            or summary.get("run_frame_skip") != self.protocol["frame_skip"]
            or summary.get("run_max_steps") != self.protocol["max_steps"]
        ):
            raise OrchestrationError("diagnostic evaluator summary has invalid cell/config counts")
        rows = []
        with (directory / "episodes.jsonl").open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
                except json.JSONDecodeError as error:
                    raise OrchestrationError(f"malformed diagnostic episode row {line_number}") from error
                if not isinstance(row, dict) or set(row) != compare.EPISODE_KEYS:
                    raise OrchestrationError(f"unsupported diagnostic episode schema at row {line_number}")
                if (
                    row.get("status") != "ok"
                    or row.get("loaded_archive_sha256") != wrapper["actor_sha256"]
                    or row.get("candidate_id") != summary.get("candidate_id")
                    or any(type(row.get(field)) is not int for field in ("track_id", "seed", "repeat", "steps"))
                    or any(type(row.get(field)) is not bool for field in ("finished", "terminated", "truncated"))
                    or not 0 < row.get("steps", 0) <= self.protocol["max_steps"]
                ):
                    raise OrchestrationError(f"invalid diagnostic episode identity at row {line_number}")
                for field in ("progress", "reward", "damage", "steering_delta_abs_mean"):
                    value = row.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise OrchestrationError(f"invalid diagnostic {field} at row {line_number}")
                for field, limit in compare.RESOURCE_LIMITS.items():
                    value = row.get(field)
                    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                        raise OrchestrationError(f"diagnostic resource limit failed for {field} at row {line_number}")
                if row["finished"]:
                    lap_time = row.get("lap_time_ms")
                    if type(lap_time) not in (int, float) or not math.isfinite(lap_time) or lap_time < 0:
                        raise OrchestrationError(f"invalid diagnostic lap time at row {line_number}")
                elif row.get("lap_time_ms") is not None:
                    raise OrchestrationError(f"unfinished diagnostic episode has lap time at row {line_number}")
                if not isinstance(row.get("actions"), list) or len(row["actions"]) != row["steps"]:
                    raise OrchestrationError(f"diagnostic action trace length mismatch at row {line_number}")
                action_digest = hashlib.sha256()
                for action in row["actions"]:
                    if not isinstance(action, list) or len(action) != 3 or any(
                        type(value) not in (int, float) or not math.isfinite(value) for value in action
                    ):
                        raise OrchestrationError(f"invalid diagnostic action at row {line_number}")
                    action_digest.update(struct.pack("<3f", *action))
                if action_digest.hexdigest() != row.get("action_trace_sha256"):
                    raise OrchestrationError(f"diagnostic action trace hash mismatch at row {line_number}")
                rows.append(row)
        expected = {
            (track, seed, repeat)
            for track in matrix["track_ids"]
            for seed in matrix["seeds"]
            for repeat in range(matrix["repeats"])
        }
        observed = {(row["track_id"], row["seed"], row["repeat"]) for row in rows}
        if len(rows) != len(expected) or observed != expected:
            raise OrchestrationError("diagnostic evaluator did not complete the exact confirmation grid")
        audits = read_json(directory / "determinism.json")
        if not isinstance(audits, list) or len(audits) != len(expected) // 2:
            raise OrchestrationError("diagnostic evaluator determinism audit is incomplete")
        grouped = {}
        for row in rows:
            grouped.setdefault((row["track_id"], row["seed"]), []).append(row)
        for cell, pair in grouped.items():
            pair.sort(key=lambda row: row["repeat"])
            if [row["repeat"] for row in pair] != [0, 1] or any(
                tuple(row.get(field) for field in compare.TRACE_FIELDS) !=
                tuple(pair[0].get(field) for field in compare.TRACE_FIELDS)
                for row in pair[1:]
            ):
                raise OrchestrationError(f"diagnostic reload traces differ at {cell}")
        audit_by_cell = {}
        for audit in audits:
            if not isinstance(audit, dict) or set(audit) != {
                "audited", "candidate_id", "matches_canonical", "repeats", "seed", "track_id",
            }:
                raise OrchestrationError("diagnostic evaluator has a malformed determinism record")
            cell = (audit["track_id"], audit["seed"])
            if cell in audit_by_cell or cell not in grouped or (
                audit["candidate_id"] != summary["candidate_id"]
                or audit["repeats"] != 2
                or audit["audited"] is not True
                or audit["matches_canonical"] is not True
            ):
                raise OrchestrationError("diagnostic evaluator determinism receipt mismatch")
            audit_by_cell[cell] = audit
        if set(audit_by_cell) != set(grouped):
            raise OrchestrationError("diagnostic evaluator determinism grid differs from episode grid")
        canonical = [row for row in rows if row["repeat"] == 0]
        finishes = sum(bool(row["finished"]) for row in canonical)
        mean_progress = sum(row["progress"] for row in canonical) / len(canonical)
        lap_times = [row["lap_time_ms"] for row in canonical if row["finished"]]
        reported = summary.get("summary", {})
        if (
            reported.get("n_episodes") != len(canonical)
            or not math.isclose(reported.get("finish_rate", -1), finishes / len(canonical), abs_tol=1e-12)
            or not math.isclose(reported.get("avg_progress", math.nan), mean_progress, abs_tol=1e-12)
            or (reported.get("avg_lap_time_ms") is not None if not lap_times else
                not math.isclose(reported.get("avg_lap_time_ms", math.nan), sum(lap_times) / len(lap_times), abs_tol=1e-12))
        ):
            raise OrchestrationError(
                "diagnostic evaluator summary disagrees with canonical episode rows: "
                f"reported={reported!r}, observed={{'n_episodes': {len(canonical)}, "
                f"'finish_rate': {finishes / len(canonical)}, 'avg_progress': {mean_progress}, "
                f"'avg_lap_time_ms': {sum(lap_times) / len(lap_times) if lap_times else None}}}"
            )
        observed_resources = {
            field: max((row[field] for row in rows), default=0.0)
            for field in compare.RESOURCE_LIMITS
        }
        reported_resources = summary.get("resources", {})
        if set(reported_resources) != set(observed_resources) or any(
            not math.isclose(reported_resources[field], value, abs_tol=1e-12)
            for field, value in observed_resources.items()
        ):
            raise OrchestrationError("diagnostic evaluator resource summary disagrees with episode rows")

    def _blind(self, screen: dict[str, Any], confirmation: dict[str, Any]) -> dict[str, Any]:
        if confirmation.get("diagnostic_only") or confirmation.get("passed") is not True:
            raise OrchestrationError("blind evaluation requires a passed standard paired confirmation")
        if not screen["gate_passed"] or screen["finalist_path"] is None:
            raise OrchestrationError("blind evaluation requires a predeclared screen finalist")
        stage = "blind"
        stage_root = self._stage_root(stage)
        actor = self._stage_actor_list(stage, screen)[0]
        if stage_root.exists():
            marker = self._check_complete_file_set(stage_root)
            if marker["diagnostic_only"] is not False:
                raise OrchestrationError("blind stage cannot be diagnostic")
            wrappers = [self._cell_paths(stage_root, actor)["wrapper"]]
            self._validate_cell_records(stage_root, stage, [actor], wrappers)
            lineages = [confirmation["decision_path"], screen["finalist_path"]]
            decision, decision_sha, decision_path = self._join_read_only(stage, wrappers, lineages)
            details = marker["details"]
            finalist_path = screen["finalist_path"]
            screen_wrapper = self._selected_screen_wrappers(screen)[
                (read_json(finalist_path)["source_learner_seed"], "teacher-replay")
            ]
            if (
                marker["status"] != "blind-complete"
                or details.get("paired_decision_sha256") != decision_sha
                or details.get("paired_decision_path") != str(decision_path.resolve())
                or details.get("screen_finalist_path") != str(finalist_path.resolve())
                or details.get("screen_finalist_sha256") != sha256_file(finalist_path)
                or details.get("screen_wrapper_path") != str(screen_wrapper.resolve())
                or details.get("screen_wrapper_sha256") != sha256_file(screen_wrapper)
                or details.get("confirmation_decision_path") != str(confirmation["decision_path"].resolve())
                or details.get("confirmation_decision_sha256") != sha256_file(confirmation["decision_path"])
            ):
                raise OrchestrationError("completed blind stage marker differs from its paired decision")
            return {"stage": stage, "wrapper_paths": wrappers, "decision_path": decision_path,
                    "decision": decision, "accepted": decision["result"].get("accepted") is True}

        stage_root = self._new_stage(stage)
        finalist_path = screen["finalist_path"]
        finalist = read_json(finalist_path)
        selected_screen_path = self._selected_screen_wrappers(screen)[
            (finalist["source_learner_seed"], "teacher-replay")
        ]
        confirmation_wrapper_path = next(
            path for path in confirmation["wrapper_paths"]
            if _identity_key(read_json(path)) == (
                finalist["source_learner_seed"], "teacher-replay", finalist["checkpoint_online_step"]
            )
        )
        wrapper, _ = self._evaluate_actor(
            stage, actor, stage_root, previous_wrapper_path=confirmation_wrapper_path
        )
        decision, decision_sha, decision_path = self._join(
            stage, stage_root, [wrapper], [confirmation["decision_path"], finalist_path]
        )
        details = {
            "paired_decision_path": str(decision_path.resolve()),
            "paired_decision_sha256": decision_sha,
            "screen_finalist_path": str(finalist_path.resolve()),
            "screen_finalist_sha256": sha256_file(finalist_path),
            "screen_wrapper_path": str(selected_screen_path.resolve()),
            "screen_wrapper_sha256": sha256_file(selected_screen_path),
            "confirmation_decision_path": str(confirmation["decision_path"].resolve()),
            "confirmation_decision_sha256": sha256_file(confirmation["decision_path"]),
        }
        self._finish_stage(stage_root, stage, diagnostic=False, status="blind-complete", details=details)
        return {
            "stage": stage, "wrapper_paths": [wrapper], "decision_path": decision_path,
            "decision": decision, "accepted": decision["result"].get("accepted") is True,
        }

    def run(self, stage: str, *, diagnostic_confirmation: bool = False) -> dict[str, Any]:
        if stage not in STAGES:
            raise OrchestrationError(f"stage must be one of {', '.join(STAGES)}")
        if stage != "confirmation" and stage != "all" and diagnostic_confirmation:
            raise OrchestrationError("--diagnostic-confirmation applies only to confirmation or all")
        if type(diagnostic_confirmation) is not bool:
            raise OrchestrationError("diagnostic_confirmation must be boolean")
        if stage != "all":
            if stage == "screen":
                return self._screen()
            screen = self._read_completed_screen()
            if stage == "confirmation":
                return self._confirmation(screen, diagnostic_confirmation)
            confirmation = self._read_completed_confirmation(screen)
            return self._blind(screen, confirmation)

        screen = self._screen()
        if not screen["gate_passed"]:
            raise OrchestrationError("screen gate failed; confirmation and blind were not opened")
        if screen["diagnostic_confirmation_required"] != diagnostic_confirmation:
            if screen["diagnostic_confirmation_required"]:
                raise OrchestrationError(
                    "zero-finish screen controls require --diagnostic-confirmation; blind will remain closed"
                )
            if diagnostic_confirmation:
                raise OrchestrationError("--diagnostic-confirmation was supplied without a zero-finish screen control")
        confirmation = self._confirmation(screen, diagnostic_confirmation)
        if confirmation.get("diagnostic_only") or not confirmation.get("passed"):
            return {
                "stage": "all",
                "screen": screen,
                "confirmation": confirmation,
                "blind": None,
                "blind_opened": False,
            }
        blind = self._blind(screen, confirmation)
        return {
            "stage": "all", "screen": screen, "confirmation": confirmation,
            "blind": blind, "blind_opened": True,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--diagnostic-confirmation", action="store_true")
    parser.add_argument("--evaluator-timeout-seconds", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        orchestrator = DrQTeacherEvaluationOrchestrator(
            args.protocol_file,
            args.repo_root,
            evaluator_timeout_seconds=args.evaluator_timeout_seconds,
        )
        result = orchestrator.run(args.stage, diagnostic_confirmation=args.diagnostic_confirmation)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
