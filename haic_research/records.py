"""Confined v2 run records with exclusive immutable and append-only writes.

There is no discovery/import API: callers must explicitly name one v2 run.
An existing lock (including one left after a crash) always requires operator
resolution; it is never silently removed or treated as expired.
"""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Mapping

from .config import HarnessConfig, validate_config
from .models import CheckpointRef, GateResult, IntegrationReport, RunEvent, RunManifest, WorkflowState


class RecordError(ValueError):
    """A record is invalid or cannot be persisted without replacing history."""


class PathSafetyError(RecordError):
    """A caller supplied an untrusted persistence path."""


class RecordLockError(RecordError):
    """Another writer or a stale lock prevents access."""


_FILES = ("run_manifest.json", "events.jsonl", "integration_report.json", ".records.lock",
          "execution_plan.json", ".execution.claim", "result_evidence.json", "comparison_evidence.json")
_GATES = frozenset({"rule_compliance", "mechanism_activation", "competitive_or_product_outcome"})
_RESERVED = {"con", "prn", "aux", "nul", "clock$", "conin$", "conout$"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)
} | {f"{prefix}{number}" for prefix in ("com", "lpt") for number in ("¹", "²", "³")}


def _identifier(value: str) -> None:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value)
            or value.endswith((".", " ")) or value.split(".")[0].lower() in _RESERVED):
        raise PathSafetyError("invalid or reserved run identifier")


def _no_redirect(path: Path) -> None:
    """Check existing components with lstat only, never follow redirected data."""
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PathSafetyError(f"redirected path component: {component}")
        if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
            raise PathSafetyError(f"hardlinked record or artifact: {component}")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise PathSafetyError(f"unsupported filesystem object: {component}")


def _canonical(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise PathSafetyError("persistence paths must be absolute without traversal")
    _no_redirect(path)
    if str(path) != str(path.resolve()):
        raise PathSafetyError("path aliases are not accepted")
    return path


def _config_roots(config: HarnessConfig) -> tuple[Path, Path, Path]:
    if not isinstance(config, HarnessConfig):
        raise PathSafetyError("a HarnessConfig is required")
    # Check lexical paths first, before validate_config resolves any components.
    root, run_root, artifact_root = map(_canonical, (config.repo_root, config.run_root, config.artifact_root))
    issues = validate_config(config)
    if issues:
        raise PathSafetyError("invalid config: " + "; ".join(item.message for item in issues))
    raw_paths = config.raw.get("paths", {})
    for name, path in (("run_root", run_root), ("artifact_root", artifact_root)):
        raw = raw_paths.get(name)
        if not isinstance(raw, str) or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise PathSafetyError("configured roots must use unaliased project-relative paths")
        if str(root / raw) != str(path):
            raise PathSafetyError("configured root does not match validated config")
        if path in config.legacy_paths:
            raise PathSafetyError("a legacy root cannot be a v2 record root")
    return root, run_root, artifact_root


def _run_path(run_dir: Path, config: HarnessConfig) -> Path:
    _, run_root, _ = _config_roots(config)
    path = _canonical(run_dir)
    _identifier(path.name)
    if str(path.parent) != str(run_root) or not path.is_dir():
        raise PathSafetyError("run_dir must be an existing direct child of configured run_root")
    for name in _FILES:
        child = path / name
        _no_redirect(child)
        if child.exists() and not child.is_file():
            raise PathSafetyError("record paths must be regular files")
    return path


def _checkpoint(ref: CheckpointRef | None, config: HarnessConfig) -> None:
    if ref is None:
        return
    root, _, artifact_root = _config_roots(config)
    if not isinstance(ref, CheckpointRef):
        raise RecordError("checkpoint_ref must be typed")
    # Treat both slash styles as separators on every platform.
    raw = ref.path.replace("\\", "/")
    parts = raw.split("/")
    if Path(raw).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise PathSafetyError("checkpoint must be a project-relative unaliased artifact path")
    for part in parts:
        _identifier(part)
    path = _canonical(root / Path(raw))
    if artifact_root not in path.parents:
        raise PathSafetyError("checkpoint reference must remain under configured artifact_root")
    if path.exists() and not path.is_file():
        raise PathSafetyError("checkpoint reference must identify a file")


@contextmanager
def _open(path: Path, mode: str):
    _no_redirect(path)
    before = path.stat() if path.exists() else None
    flags = {"r": os.O_RDONLY, "x": os.O_WRONLY | os.O_CREAT | os.O_EXCL,
             "a": os.O_WRONLY | os.O_APPEND}[mode]
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RecordError(f"immutable record already exists: {path.name}") from exc
    except OSError as exc:
        raise RecordError(f"cannot open record {path.name}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        _no_redirect(path)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or (before is not None and (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino))):
            raise PathSafetyError("record changed identity or has multiple links")
        current = path.stat()
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise PathSafetyError("record path changed while opening")
        with os.fdopen(descriptor, mode="r" if mode == "r" else "w", encoding="utf-8", newline="\n") as stream:
            descriptor = None
            yield stream
            if mode != "r":
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _lock(path: Path):
    lock = path / ".records.lock"
    try:
        with _open(lock, "x") as stream:
            stream.write(str(os.getpid()) + "\n")
        identity = lock.stat()
    except RecordError as exc:
        raise RecordLockError("run lock exists or cannot be safely acquired") from exc
    try:
        yield
    finally:
        _no_redirect(lock)
        current = lock.stat()
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            raise RecordLockError("run lock changed identity; refusing to remove it")
        lock.unlink()


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _json_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise RecordError("JSON mappings require string keys")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RecordError(f"unsupported record value: {type(value).__name__}")


def _encode(record: object) -> str:
    try:
        return json.dumps(_json_value(record), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise RecordError(f"invalid JSON record: {exc}") from exc


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecordError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode(text: str) -> object:
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(RecordError("invalid JSON number")))
    except (ValueError, TypeError) as exc:
        raise RecordError(f"invalid JSON record: {exc}") from exc


def _read_json(path: Path) -> object:
    return _decode(_read_text(path))


def _read_text(path: Path) -> str:
    try:
        with _open(path, "r") as stream:
            return stream.read()
    except (UnicodeError, OSError) as exc:
        raise RecordError(f"cannot read UTF-8 record {path.name}: {exc}") from exc


def _typed(cls, data):
    if not isinstance(data, dict) or set(data) - {item.name for item in fields(cls)}:
        raise RecordError(f"invalid {cls.__name__} fields")
    data = dict(data)
    try:
        if cls in {RunManifest, RunEvent}:
            if data.get("checkpoint_ref") is not None:
                data["checkpoint_ref"] = _typed(CheckpointRef, data["checkpoint_ref"])
        if cls is RunEvent:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
            if not isinstance(data.get("resource_usage", {}), dict):
                raise RecordError("resource_usage must be an object")
        if cls is RunManifest:
            for name in ("tool_versions", "runtime_versions", "resource_limits", "permission_limits", "source_hashes"):
                if not isinstance(data.get(name), dict):
                    raise RecordError(f"{name} must be an object")
        if cls is IntegrationReport:
            data["gate_results"] = tuple(_typed(GateResult, gate) for gate in data["gate_results"])
        return cls(**data)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise RecordError(f"invalid {cls.__name__}: {exc}") from exc


def _manifest(path: Path, config: HarnessConfig) -> RunManifest:
    manifest = _typed(RunManifest, _read_json(path / "run_manifest.json"))
    if manifest.run_id != path.name:
        raise PathSafetyError("manifest run_id does not match run directory")
    _checkpoint(manifest.checkpoint_ref, config)
    return manifest


def _validate_report(report: IntegrationReport) -> None:
    if not isinstance(report, IntegrationReport) or not all(isinstance(gate, GateResult) for gate in report.gate_results):
        raise RecordError("a typed IntegrationReport is required")
    names = [gate.name for gate in report.gate_results]
    if len(names) != len(_GATES) or set(names) != _GATES:
        raise RecordError("report must contain exactly one result for each registered gate")


def _report(path: Path) -> IntegrationReport:
    report = _typed(IntegrationReport, _read_json(path / "integration_report.json"))
    _validate_report(report)
    return report


def _validate_event(event: RunEvent, previous: list[RunEvent], manifest: RunManifest,
                    path: Path, config: HarnessConfig) -> None:
    if not isinstance(event, RunEvent):
        raise RecordError("a typed RunEvent is required")
    if event.event_id in {item.event_id for item in previous}:
        raise RecordError("event_id must be unique within the run")
    if event.event_id == "integration_report.json":
        raise RecordError("report reference is reserved and cannot be an event ID")
    if previous and event.timestamp < previous[-1].timestamp:
        raise RecordError("event timestamps must be nondecreasing")
    if event.kind == "APPROVAL" and event.approved_plan_hash != manifest.plan_hash:
        raise RecordError("approval must identify the current run's exact plan hash")
    if event.correction_ref is not None:
        if not isinstance(event.correction_ref, str) or not event.correction_ref:
            raise RecordError("correction_ref must be a nonempty reference string")
        if event.correction_ref == "integration_report.json":
            _report(path)
        elif event.correction_ref not in {item.event_id for item in previous}:
            raise RecordError("correction must reference an existing event or report")
    _checkpoint(event.checkpoint_ref, config)


def _events(path: Path, manifest: RunManifest, config: HarnessConfig) -> tuple[RunEvent, ...]:
    text = _read_text(path / "events.jsonl")
    if text and not text.endswith("\n"):
        raise RecordError("incomplete event history line")
    result = []
    # Unicode line separators are valid JSON string data, not JSONL delimiters.
    for row in text.split("\n")[:-1]:
        item = _typed(RunEvent, _decode(row))
        _validate_event(item, result, manifest, path, config)
        result.append(item)
    return tuple(result)


def read_manifest(run_dir: Path, *, config: HarnessConfig) -> RunManifest:
    path = _run_path(run_dir, config)
    with _lock(path):
        return _manifest(path, config)


def read_events(run_dir: Path, *, config: HarnessConfig) -> tuple[RunEvent, ...]:
    path = _run_path(run_dir, config)
    with _lock(path):
        return _events(path, _manifest(path, config), config)


def append_event(run_dir: Path, event: RunEvent, *, config: HarnessConfig) -> None:
    path = _run_path(run_dir, config)
    with _lock(path):
        manifest = _manifest(path, config)
        previous = _events(path, manifest, config)
        _validate_event(event, list(previous), manifest, path, config)
        encoded = _encode(event)
        with _open(path / "events.jsonl", "a") as stream:
            stream.write(encoded)


def write_integration_report(run_dir: Path, report: IntegrationReport, *, config: HarnessConfig) -> None:
    path = _run_path(run_dir, config)
    with _lock(path):
        _manifest(path, config)
        _validate_report(report)
        encoded = _encode(report)
        with _open(path / "integration_report.json", "x") as stream:
            stream.write(encoded)


def read_integration_report(run_dir: Path, *, config: HarnessConfig) -> IntegrationReport:
    path = _run_path(run_dir, config)
    with _lock(path):
        _manifest(path, config)
        return _report(path)


def write_execution_plan(run_dir: Path, plan: Mapping[str, object], *, config: HarnessConfig) -> None:
    """Write one immutable plan through the existing confinement and lock boundary."""
    with run_transaction(run_dir, config=config) as run:
        run.write_plan(plan)


def read_execution_plan(run_dir: Path, *, config: HarnessConfig) -> dict[str, object]:
    with run_transaction(run_dir, config=config) as run:
        if run.plan is None:
            raise RecordError("execution plan is missing")
        return run.plan


_TRANSACTION_KEY = object()


class RunTransaction:
    """Current-run snapshot and confined mutations while its records lock is held.

    Obtain only through run_transaction; callers must finish the context before
    invoking a runner. A claim remains immutable even after failure or a crash.
    """

    def __init__(self, path: Path, config: HarnessConfig, *, _key=None):
        if _key is not _TRANSACTION_KEY:
            raise RecordLockError("use run_transaction to acquire the records lock")
        self._active = True
        self._path = _run_path(path, config)
        self._config = config
        self.manifest = _manifest(self._path, config)
        self.events = _events(self._path, self.manifest, config)
        self.plan = _read_json(path / "execution_plan.json") if (path / "execution_plan.json").exists() else None
        if self.plan is not None and not isinstance(self.plan, dict):
            raise RecordError("execution plan must be an object")
        self.report = _report(path) if (path / "integration_report.json").exists() else None
        self.execution_reserved = (path / ".execution.claim").exists()
        self.execution_claim = _read_json(path / ".execution.claim") if self.execution_reserved else None

    def _active_path(self) -> Path:
        if not self._active:
            raise RecordLockError("run transaction has ended")
        return _run_path(self._path, self._config)

    def append(self, event: RunEvent) -> None:
        path = self._active_path()
        _validate_event(event, list(self.events), self.manifest, path, self._config)
        with _open(path / "events.jsonl", "a") as stream:
            stream.write(_encode(event))
        self.events += (event,)

    def write_plan(self, plan: Mapping[str, object]) -> None:
        path = self._active_path()
        if not isinstance(plan, Mapping):
            raise RecordError("execution plan must be an object")
        with _open(path / "execution_plan.json", "x") as stream:
            stream.write(_encode(plan))
        self.plan = _json_value(plan)

    def write_report(self, report: IntegrationReport) -> None:
        path = self._active_path()
        _validate_report(report)
        with _open(path / "integration_report.json", "x") as stream:
            stream.write(_encode(report))
        self.report = report

    def read_result_evidence(self, *, comparison: bool = False) -> dict[str, object]:
        path = self._active_path()
        name = "comparison_evidence.json" if comparison else "result_evidence.json"
        payload = _read_json(path / name)
        if not isinstance(payload, dict):
            raise RecordError("result evidence must be an object")
        return payload

    def write_result_evidence(self, payload: Mapping[str, object], *, comparison: bool = False) -> None:
        """Immutable named evidence, only before the gate report, under this lock."""
        path = self._active_path()
        if self.report is not None:
            raise RecordError("result evidence must precede the immutable gate report")
        if not isinstance(payload, Mapping):
            raise RecordError("result evidence must be an object")
        name = "comparison_evidence.json" if comparison else "result_evidence.json"
        with _open(path / name, "x") as stream:
            stream.write(_encode(payload))

    def reserve_execution(self, event: RunEvent) -> None:
        path = self._active_path()
        if self.execution_reserved or any(item.kind == "EXECUTION_STARTED" for item in self.events):
            raise RecordError("execution was already reserved; retry requires a new run")
        if event.kind != "EXECUTION_STARTED":
            raise RecordError("execution reservation requires EXECUTION_STARTED")
        _validate_event(event, list(self.events), self.manifest, path, self._config)
        # The same lock covers the claim and start append. A partial write leaves
        # a permanent claim and therefore cannot authorize another invocation.
        claim = {"event_id": event.event_id, "plan_hash": self.manifest.plan_hash}
        with _open(path / ".execution.claim", "x") as stream:
            stream.write(_encode(claim))
        self.execution_reserved = True
        self.execution_claim = claim
        self.append(event)
        _, _, artifact_root = _config_roots(self._config)
        destination = artifact_root / self.manifest.run_id
        _canonical(destination)
        destination.mkdir(parents=True, exist_ok=True)
        _canonical(destination)


@contextmanager
def run_transaction(run_dir: Path, *, config: HarnessConfig):
    """Atomically inspect and update only the explicitly named confined v2 run."""
    path = _run_path(run_dir, config)
    with _lock(path):
        run = RunTransaction(path, config, _key=_TRANSACTION_KEY)
        try:
            yield run
        finally:
            run._active = False


def _continuation(manifest: RunManifest, previous_run_dir: Path, config: HarnessConfig) -> None:
    path = _run_path(previous_run_dir, config)
    with _lock(path):
        parent = _manifest(path, config)
        events = _events(path, parent, config)
        if manifest.predecessor_run_id != parent.run_id:
            raise RecordError("predecessor run ID does not match explicitly named predecessor")
        if any(getattr(manifest, key) == getattr(parent, key) for key in ("run_id", "cycle_id", "plan_hash")):
            raise RecordError("continuation requires distinct run ID, cycle ID and plan hash")
        decisions = [(index, item) for index, item in enumerate(events) if item.kind == "CYCLE_DECISION"]
        if not decisions:
            raise RecordError("predecessor has no cycle decision")
        index, decision = decisions[-1]
        if (decision.event_id != manifest.predecessor_decision_ref
                or decision.workflow_state not in {WorkflowState.REVISE, WorkflowState.PIVOT}):
            raise RecordError("continuation must reference latest REVISE or PIVOT decision")
        review_state = WorkflowState[f"GATE_REVIEW_{decision.workflow_state.value}"]
        reviews = [i for i, item in enumerate(events) if i > index and item.kind == "GATE_REVIEW"
                   and item.workflow_state == review_state]
        if not reviews or not any(item.workflow_state == WorkflowState.STOPPED for item in events[reviews[-1] + 1:]):
            raise RecordError("decision requires later matching gate review and STOPPED event")
        if events[-1].workflow_state != WorkflowState.STOPPED:
            raise RecordError("predecessor must remain STOPPED")
        _report(path)
        if decision.workflow_state == WorkflowState.PIVOT:
            checkpoint = decision.checkpoint_ref or parent.checkpoint_ref
            if checkpoint is None or manifest.checkpoint_ref != checkpoint:
                raise RecordError("pivot must preserve selected predecessor checkpoint path and hash")
            _checkpoint(checkpoint, config)


def create_run(config: HarnessConfig, manifest: RunManifest, *, previous_run_dir: Path | None = None) -> Path:
    _, run_root, _ = _config_roots(config)
    if not isinstance(manifest, RunManifest):
        raise RecordError("a typed RunManifest is required")
    _identifier(manifest.run_id)
    _checkpoint(manifest.checkpoint_ref, config)
    linked = manifest.predecessor_run_id is not None or manifest.predecessor_decision_ref is not None
    if linked != (previous_run_dir is not None) or (linked and not all((manifest.predecessor_run_id, manifest.predecessor_decision_ref))):
        raise RecordError("continuation requires explicit predecessor path and both predecessor references")
    if previous_run_dir is not None:
        _continuation(manifest, previous_run_dir, config)
    encoded = _encode(manifest)
    path = run_root / manifest.run_id
    _no_redirect(path)
    if path.exists():
        raise RecordError("run identifier has already been used")
    run_root.mkdir(parents=True, exist_ok=True)
    _canonical(run_root)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RecordError("run identifier has already been used") from exc
    with _lock(_run_path(path, config)):
        with _open(path / "run_manifest.json", "x") as stream:
            stream.write(encoded)
        with _open(path / "events.jsonl", "x"):
            pass
    return path
