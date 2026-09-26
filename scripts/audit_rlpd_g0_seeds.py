"""Read-only-by-default geometry-seed preflight for a 12-road RLPD G0 TRAIN pool.

Run with ``python -m scripts.audit_rlpd_g0_seeds --seed-start N --track-id T``.
The caller chooses N/T BEFORE observing either actor. Candidate i is N+i, i=0..11;
collisions are NOT replaced. This is not a protocol freeze, a geometry check, or a
global non-use certificate: historical pilot schedules and unrecorded use are incomplete.
Only named seed metadata and TRAIN ledgers are opened; an opt-in r5
erratum additionally hashes two named TRAIN metrics/trace files. No evaluator/blind
episodes, environment, model or trainer is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


UINT32_MAX = (1 << 32) - 1
FRESHNESS_LIMITATION = "no known recorded overlap; historical pilot schedules are incomplete"
CATALOG_PROTOCOL = "experiments/drqv2-geometry-augmentation-v1.json"
CATALOG_PATH = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
# Filename-only snapshot: an unreviewed new experiment protocol must not silently
# reserve a proposed G0 seed outside the typed, hash-pinned source inventory.
EXPERIMENT_JSON_CATALOG_SHA256 = "29878b34f423ac17e934246e77e1253d0357fedb47b021172fcd456788a9474d"
EXPERIMENT_CONTENT_CATALOG_SHA256 = "b9629c40e703b19fae2ac4b0560e5d74e25000517053a3b7c357fdf09be7aebf"
REQUIRED_PROTOCOLS = (
    *(f"experiments/pixel-rlpd-offpolicy-pilot-v{v}.json" for v in (1, 2)),
    "experiments/pixel-rlpd-long-horizon-followup-v1.json",
    *(f"experiments/pixel-rlpd-entropy-target-ablation-v{v}.json" for v in range(1, 6)),
    CATALOG_PROTOCOL,
    *(f"experiments/drqv2-teacher-replay-v1{s}.json" for s in ("", "-r2", "-r3")),
    *(f"experiments/drqv2-geometry-mix-v1{s}.json" for s in ("", "-r2", "-r3", "-r4", "-r5", "-r6")),
    *(f"experiments/dreamerv3-b1-{name}.json" for name in (
        "world-model-local-v1", "world-model-local-v2", "terminal-balanced-local-v3",
        "observation-scale-local-v4", "prior-kl-local-v5", "overshooting-local-v6",
        "overshooting-floor-local-v7", "residual-frame-local-v8",
        "terminal-positive-weight-local-v9",
    )),
)
TEACHER_LEDGERS = tuple(
    f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{i}/episodes.jsonl"
    for i in (0, 1)
)
REQUIRED_LEDGERS = TEACHER_LEDGERS + tuple(
    f"runs/20260925-drqv2-geometry-mix-v1-{v}/learner-0-uniform/episodes.jsonl"
    for v in ("r4", "r5", "r6")
) + tuple(
    f"runs/20260924-pixel-rlpd-{study}/{arm}-seed{seed}/episodes.jsonl"
    for study in ("offpolicy-pilot-v2", "long-horizon-followup-v1")
    for arm in ("rlpd", "sac") for seed in ((0, 1) if study == "offpolicy-pilot-v2" else (10, 11))
) + tuple(
    f"runs/20260925-pixel-rlpd-entropy-target-ablation-v{version}/"
    f"rlpd-{arm}-target-seed{seed}/episodes.jsonl"
    for version, seeds in ((4, (40, 41)), (5, (50, 51)))
    for arm in ("author", "positive") for seed in seeds
) + tuple(
    f"runs/20260924-dreamerv3-b1-{study}/pretrain-seed-{seed}/episodes.jsonl"
    for study in (
        "world-model-local-v1", "world-model-local-v2", "terminal-balanced-local-v3",
        "observation-scale-local-v4", "prior-kl-local-v5", "overshooting-local-v6",
        "overshooting-floor-local-v7", "residual-frame-local-v8",
        "terminal-positive-weight-local-v9",
    ) for seed in (0, 1)
)
REQUIRED_COLLECTIONS = {
    f"runs/{run}/prior-data/collection.jsonl": f"experiments/{study}.json"
    for run, study in (
        ("20260924-pixel-rlpd-offpolicy-pilot-v2", "pixel-rlpd-offpolicy-pilot-v2"),
        ("20260924-pixel-rlpd-long-horizon-followup-v1", "pixel-rlpd-long-horizon-followup-v1"),
        ("20260925-pixel-rlpd-entropy-target-ablation-v4", "pixel-rlpd-entropy-target-ablation-v4"),
        ("20260925-pixel-rlpd-entropy-target-ablation-v5", "pixel-rlpd-entropy-target-ablation-v5"),
    )
}
REQUIRED_COLLECTION_SHA256 = {
    "runs/20260924-pixel-rlpd-offpolicy-pilot-v2/prior-data/collection.jsonl":
        "a26998d7139fc2e40c2bb1ac23f4ef5653fe94bcda714286bc2cc777eafb5275",
    "runs/20260924-pixel-rlpd-long-horizon-followup-v1/prior-data/collection.jsonl":
        "100ca6220e9ec8500038a56001af86e20ae6b98f474c4dc1a7edf885ab87cd32",
    "runs/20260925-pixel-rlpd-entropy-target-ablation-v4/prior-data/collection.jsonl":
        "a5298dab82388a3a6481bb2a81e30c9f2eee0f9726daa339cba908a4c2a4848b",
    "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/collection.jsonl":
        "92b86bc8b38e020d26b66bd2742f64ab929147dcb0a2f965bf415482d34d599f",
}
PARTIAL_RECEIPTS = tuple(
    f"runs/20260925-drqv2-geometry-mix-v1-{v}/protocol-superseded-after-partial.json"
    for v in ("r4", "r5")
)
ATTEMPT_RECEIPTS = tuple(
    f"runs/20260925-drqv2-geometry-mix-v1-{v}/learner-0-uniform/precheckpoint-abort.json"
    for v in ("r4", "r5")
)
TRAIN_ROOTS = (
    *(f"runs/20260924-{study}" for study in (
        "pixel-rlpd-offpolicy-pilot-v2", "pixel-rlpd-long-horizon-followup-v1")),
    *(f"runs/20260925-pixel-rlpd-entropy-target-ablation-v{v}" for v in (4, 5)),
    *(f"runs/20260924-dreamerv3-b1-{study}" for study in (
        "world-model-local-v1", "world-model-local-v2", "terminal-balanced-local-v3",
        "observation-scale-local-v4", "prior-kl-local-v5", "overshooting-local-v6",
        "overshooting-floor-local-v7", "residual-frame-local-v8",
        "terminal-positive-weight-local-v9")),
    "runs/20260925-drqv2-geometry-mix-v1-r6",
)
R5_ERRATUM_PATH = "experiments/drqv2-geometry-mix-v1-r5-seed-audit-erratum-v1.json"
R5_EPISTEMIC_LIMIT = (
    "The malformed r5 abort-receipt episodes_sha256 does not attest the actual "
    "episodes.jsonl bytes. This erratum independently checks the listed current "
    "ledger and recorded seed IDs; it does not repair the frozen receipt."
)
R5_PROFILE: dict[str, str | int] = {
    "attempt_receipt_path": "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/precheckpoint-abort.json",
    "attempt_receipt_sha256": "da17593a7853e30c19f37ee53c87de2300198309ceb1b046742bc5f56fe994d3",
    "supersession_path": "runs/20260925-drqv2-geometry-mix-v1-r5/protocol-superseded-after-partial.json",
    "supersession_sha256": "dfc5bcb050090278e3213af101d44e413ac0f70fce5727204bcb8424e0f3ca60",
    "ledger_path": "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/episodes.jsonl",
    "ledger_sha256": "8d192a9cc87ee3777e3647b2e49eeec7eb66b84eb12099a0ea7c569230889014",
    "protocol_path": "experiments/drqv2-geometry-mix-v1-r5.json",
    "protocol_sha256": "531ca7838c0e911eec70cf729073753367df97ddcb6aedea1fe92091c8327000",
    "step_metrics_path": "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/step-metrics.jsonl",
    "step_metrics_sha256": "47d381deae01fe77a9272f74869a3061941d11716d44c4052db665f4ee1a55f4",
    "trace_path": "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/replay-sample-trace-step-000016384.npz",
    "trace_sha256": "6d01e0e45eb6d0365c448f2d1b302961a8715dfd74a46b38f7569b25705f3e96",
    "invalid_receipt_episodes_sha256": "8d192a9cc87ee3777e364f2d49e49ecc7eb66b84eb12099a0ea7c569230889014",
    "ordered_reset_count": 36,
    "end_count": 35,
    "online_decisions": 16384,
    "learner_updates": 6384,
    "last_episode_id": 35,
    "last_episode_step": 412,
    "last_geometry_seed": 3910800035,
}
_FORBIDDEN = re.compile(r"blind|confirm|screen|eval|held.?out|holdout|submission", re.I)


class SeedAuditError(ValueError):
    """An incomplete, unsafe, or uninterpretable audit cannot certify a pool."""


class SeedAuditBlocked(SeedAuditError):
    """The complete report contains a collision or an unexplained exact token."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__("candidate seed collision or ambiguous evidence")
        self.report = report


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _uint32(value: Any, where: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise SeedAuditError(f"{where}: expected uint32 geometry seed")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, value in pairs:
        if key in record:
            raise SeedAuditError(f"duplicate JSON key: {key}")
        record[key] = value
    return record


def _json(raw: bytes, where: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                           parse_constant=lambda v: (_ for _ in ()).throw(SeedAuditError(f"{where}: {v}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SeedAuditError(f"{where}: malformed UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SeedAuditError(f"{where}: expected JSON object")
    return value


def _experiment_catalog(root: Path, *, expected: str | None = None,
                        expected_content: str | None = None,
                        candidates: list[int] | None = None,
                        track_id: int | None = None) -> tuple[str, list[dict[str, Any]], dict[str, bytes]]:
    directory = root / "experiments"
    if directory.is_symlink() or not directory.is_dir():
        raise SeedAuditError("missing or symlinked experiments directory")
    names: list[str] = []
    receipts: dict[str, tuple[dict[str, Any], str]] = {}
    files: dict[str, bytes] = {}
    for entry in directory.iterdir():
        if entry.suffix != ".json":
            continue
        if entry.is_symlink() or not entry.is_file():
            raise SeedAuditError(f"unsafe experiments JSON path: {entry.name}")
        raw = entry.read_bytes()
        relative = f"experiments/{entry.name}"
        if entry.name.endswith("-seed-audit.json"):
            receipt = _json(raw, relative)
            proposed = receipt.get("candidate_seeds")
            if (receipt.get("format") != "haic-rlpd-g0-seed-audit-v1"
                    or receipt.get("status") != "no_known_recorded_overlap"
                    or receipt.get("passed") is not True
                    or receipt.get("protocol_frozen") is not False
                    or not isinstance(proposed, list) or len(proposed) != 12
                    or any(type(seed) is not int for seed in proposed)
                    or receipt.get("candidate_seeds_sha256") != _digest(proposed)
                    or receipt.get("candidate_rule") !=
                    "seed_start + offset for offsets 0..11, in that order; no replacement on collision"
                    or receipt.get("seed_start") != proposed[0]
                    or proposed != list(range(proposed[0], proposed[0] + 12))
                    or not isinstance(receipt.get("source_inventory"), list)
                    or not receipt["source_inventory"]
                    or receipt.get("source_inventory_sha256") != _digest(receipt["source_inventory"])
                    or not isinstance(receipt.get("experiment_content_inventory"), list)
                    or not receipt["experiment_content_inventory"]
                    or receipt.get("experiment_content_inventory_sha256") !=
                    _digest(receipt["experiment_content_inventory"])):
                raise SeedAuditError(f"unrecognized experiment seed-audit artifact: {entry.name}")
            if expected is not None and receipt.get("experiments_catalog_sha256") != expected:
                raise SeedAuditError(f"seed-audit artifact has an unreviewed experiment catalog: {entry.name}")
            if (expected_content is not None
                    and receipt.get("experiment_content_inventory_sha256") != expected_content):
                raise SeedAuditError(f"seed-audit artifact has unreviewed experiment bytes: {entry.name}")
            if candidates is not None and proposed != candidates:
                raise SeedAuditError(f"seed-audit artifact reserves a different candidate batch: {entry.name}")
            receipts[relative] = (receipt, hashlib.sha256(raw).hexdigest())
            continue
        _json(raw, relative)
        names.append(relative)
        files[relative] = raw
    digest = _digest(sorted(names))
    if expected is not None and digest != expected and candidates is not None and track_id is not None:
        self_protocols = [name for name in names if "g0" in PurePosixPath(name).stem.lower()]
        if len(self_protocols) == 1:
            name = self_protocols[0]
            protocol = _json(files[name], name)
            audit_path = protocol.get("geometry_audit_path")
            referenced = receipts.get(audit_path) if isinstance(audit_path, str) else None
            if referenced is not None:
                receipt, receipt_sha = referenced
                cells = [{"track_id": track_id, "geometry_seed": seed,
                          "partition": "TRAIN", "obstacles": True} for seed in candidates]
                erratum = receipt.get("r5_erratum")
                if (protocol.get("format") == "haic-rlpd-g0-diagnostic-v1"
                        and protocol.get("status") == "frozen"
                        and protocol.get("partition") == "TRAIN"
                        and protocol.get("cells") == receipt.get("cells") == cells
                        and receipt.get("candidate_seeds_sha256") == _digest(candidates)
                        and protocol.get("geometry_audit_sha256") == receipt_sha
                        and isinstance(erratum, dict)
                        and erratum.get("path") == protocol.get("r5_erratum_path") == R5_ERRATUM_PATH
                        and erratum.get("sha256") == protocol.get("r5_erratum_sha256")
                        and erratum.get("original_ledger_bytes_attested") is False):
                    digest = _digest(sorted(set(names) - {name}))
                    del files[name]
    if expected is not None and digest != expected:
        raise SeedAuditError("experiment JSON catalog changed; review new cross-lane protocols before seed audit")
    evidence = [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                for name, raw in sorted(files.items())]
    return digest, evidence, files


def _map(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SeedAuditError(f"{where}: missing or unrecognized object")
    return value


def _seeds(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SeedAuditError(f"{where}: missing or empty seed list")
    seeds = [_uint32(seed, where) for seed in value]
    if len(seeds) != len(set(seeds)):
        raise SeedAuditError(f"{where}: duplicate geometry seed")
    return seeds


def _safe(root: Path, name: str, kind: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise SeedAuditError(f"{kind}: unsafe path")
    rel = PurePosixPath(name)
    parts = name.split("/")
    if rel.is_absolute() or rel.as_posix() != name or any(p in ("", ".", "..") for p in parts):
        raise SeedAuditError(f"{kind}: path must be normalized and relative: {name}")
    if kind == "protocol":
        allowed = len(parts) == 2 and parts[0] == "experiments" and name.endswith(".json")
    elif kind == "erratum":
        allowed = name == R5_ERRATUM_PATH
    elif kind == "auxiliary":
        allowed = name in (R5_PROFILE["step_metrics_path"], R5_PROFILE["trace_path"])
    elif kind == "catalog":
        allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "catalog.json"
    elif kind == "receipt":
        allowed = (len(parts) >= 3 and parts[0] == "runs" and parts[-1] in
                   ("protocol-superseded-after-partial.json", "precheckpoint-abort.json")
                   and not any(_FORBIDDEN.search(p) for p in parts[1:-1]))
    elif kind == "collection":
        allowed = name in REQUIRED_COLLECTIONS
    elif kind == "root":
        allowed = len(parts) == 2 and parts[0] == "runs"
    else:
        allowed = (len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "episodes.jsonl"
                   and not any(_FORBIDDEN.search(p) for p in parts[1:-1]))
    if not allowed:
        raise SeedAuditError(f"{kind}: outside permitted seed metadata: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise SeedAuditError(f"{kind}: symlink forbidden: {name}")
    return path


def _read(root: Path, name: str, kind: str, inventory: dict[str, dict[str, Any]]) -> bytes:
    path = _safe(root, name, kind)
    if not path.is_file() or name in inventory:
        raise SeedAuditError(f"{kind}: missing or duplicate required source: {name}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SeedAuditError(f"{kind}: cannot read {name}") from exc
    if not raw.strip():
        raise SeedAuditError(f"{kind}: empty source: {name}")
    inventory[name] = {"path": name, "kind": kind, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return raw


def _append(ids: dict[int, list[dict[str, str]]], seeds: list[int], path: str, field: str) -> None:
    for seed in seeds:
        source = {"path": path, "field": field}
        if source not in ids.setdefault(seed, []):
            ids[seed].append(source)


def _field(ids: dict[int, list[dict[str, str]]], obj: dict[str, Any], key: str,
           path: str, prefix: str = "") -> list[int]:
    where = f"{prefix}.{key}" if prefix else key
    seeds = _seeds(obj.get(key), f"{path}.{where}")
    _append(ids, seeds, path, where)
    return seeds


def _reject_seed_ranges(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if "seed" in key and ("range" in key or key.endswith("_start")):
                raise SeedAuditError(f"{path}: unsupported seed range/schema: {key}")
            _reject_seed_ranges(item, path)
    elif isinstance(value, list):
        for item in value:
            _reject_seed_ranges(item, path)


def _partitions(ids: dict[int, list[dict[str, str]]], obj: Any, path: str, prefix: str) -> None:
    partitions = _map(obj, f"{path}.{prefix}")
    if set(partitions) != {"screen", "confirmation", "blind"}:
        raise SeedAuditError(f"{path}.{prefix}: missing or unrecognized partition")
    for name, partition in partitions.items():
        _field(ids, _map(partition, f"{path}.{prefix}.{name}"), "seeds", path, f"{prefix}.{name}")


def _protocol(record: dict[str, Any], path: str, ids: dict[int, list[dict[str, str]]],
              references: list[tuple[str, str, str, set[int] | None]]) -> None:
    fmt = record.get("format")
    name = PurePosixPath(path).name
    expected = (
        "haic-pixel-rlpd-study-v1" if name.startswith("pixel-rlpd-") else
        "haic-drq-training-geometry-protocol-v1" if name.startswith("drqv2-geometry-augmentation-") else
        "haic-drq-geometry-mix-study-v1" if name.startswith("drqv2-geometry-mix-") else
        "haic-dreamerv3-study-protocol-v1" if name.startswith("dreamerv3-b1-") else
        None if name.startswith("drqv2-teacher-replay-") else "unsupported"
    )
    if fmt != expected or expected == "unsupported":
        raise SeedAuditError(f"{path}: protocol path/schema mismatch")
    if fmt == "haic-pixel-rlpd-study-v1":
        _field(ids, record, "reserved_training_seeds", path)
        train = _field(ids, record, "training_geometry_seeds", path)
        cells = record.get("teacher_data_cells")
        if not isinstance(cells, list) or not cells:
            raise SeedAuditError(f"{path}: missing teacher_data_cells")
        for n, cell in enumerate(cells):
            cell = _map(cell, f"{path}.teacher_data_cells[{n}]")
            seed = _uint32(cell.get("geometry_seed"), f"{path}.teacher_data_cells[{n}]")
            _append(ids, [seed], path, f"teacher_data_cells[{n}].geometry_seed")
        if {c["geometry_seed"] for c in cells} - set(train):
            raise SeedAuditError(f"{path}: teacher cells outside training seed allocation")
        _partitions(ids, record.get("partitions"), path, "partitions")
        if "future_full_reservation" in record:
            future = _map(record["future_full_reservation"], f"{path}.future_full_reservation")
            for key in ("training_geometry_seeds", "teacher_data_cells", "screen", "confirmation", "blind"):
                if key not in future:
                    raise SeedAuditError(f"{path}: incomplete future_full_reservation")
            _field(ids, future, "training_geometry_seeds", path, "future_full_reservation")
            if not isinstance(future["teacher_data_cells"], list) or not future["teacher_data_cells"]:
                raise SeedAuditError(f"{path}: empty future teacher data cells")
            for n, cell in enumerate(future["teacher_data_cells"]):
                seed = _uint32(_map(cell, path).get("geometry_seed"), path)
                _append(ids, [seed], path, f"future_full_reservation.teacher_data_cells[{n}].geometry_seed")
            _partitions(ids, {k: future[k] for k in ("screen", "confirmation", "blind")},
                        path, "future_full_reservation")
        audit = _map(record.get("geometry_audit"), f"{path}.geometry_audit")
        if "teacher_source_ledgers" in audit:
            ledgers = audit["teacher_source_ledgers"]
            if not isinstance(ledgers, list) or len(ledgers) != 2:
                raise SeedAuditError(f"{path}: missing teacher source ledger inventory")
            for n, ledger in enumerate(ledgers):
                ledger = _map(ledger, f"{path}.geometry_audit.teacher_source_ledgers[{n}]")
                ledger_seeds = _seeds(ledger.get("geometry_seeds"), path)
                _append(ids, ledger_seeds, path,
                        f"geometry_audit.teacher_source_ledgers[{n}].geometry_seeds")
                references.append((ledger.get("path"), ledger.get("sha256"), path, set(ledger_seeds)))
    elif fmt == "haic-drq-training-geometry-protocol-v1":
        seeds = _field(ids, record, "candidate_seeds", path)
        if len(seeds) != 512 or seeds != list(range(3910800001, 3910800513)):
            raise SeedAuditError(f"{path}: full DrQ 512-ID catalog reservation missing")
        for key in ("reserved", "heldout", "blind"):
            _field(ids, _map(record.get("exclusion_seed_ids"), path), key, path, "exclusion_seed_ids")
    elif fmt == "haic-drq-geometry-mix-study-v1":
        env = _map(record.get("environment"), f"{path}.environment")
        diag = _map(record.get("diagnostic_pool"), f"{path}.diagnostic_pool")
        if env.get("partition") != "TRAIN" or diag.get("partition") != "TRAIN-DIAGNOSTIC":
            raise SeedAuditError(f"{path}: wrong mix pool partitions")
        _field(ids, env, "geometry_seeds", path, "environment")
        _field(ids, diag, "geometry_seeds", path, "diagnostic_pool")
        if "training_pool" in record:
            pool = _map(record["training_pool"], path)
            if pool.get("partition") != "TRAIN":
                raise SeedAuditError(f"{path}: wrong training pool")
            _field(ids, pool, "geometry_seeds", path, "training_pool")
        catalog = _map(record.get("catalog"), f"{path}.catalog")
        references.append((catalog.get("path"), catalog.get("sha256"), path, None))
    elif fmt == "haic-dreamerv3-study-protocol-v1":
        _field(ids, record, "reserved_training_seeds", path)
        dev = _map(record.get("training_development"), path)
        _field(ids, dev, "seeds", path, "training_development")
        cells = dev.get("cells")
        if not isinstance(cells, list) or not cells:
            raise SeedAuditError(f"{path}: missing B1 development cells")
        for n, cell in enumerate(cells):
            seed = _uint32(_map(cell, path).get("seed"), f"{path}.training_development.cells[{n}]")
            _append(ids, [seed], path, f"training_development.cells[{n}].seed")
        _partitions(ids, record.get("partitions"), path, "partitions")
    elif isinstance(record.get("study_id"), str) and record["study_id"].startswith("drqv2-teacher-replay-v1"):
        for key in ("reserved_training_seeds", "known_excluded_geometry_seeds"):
            _field(ids, record, key, path)
        _field(ids, _map(record.get("geometry_audit"), path), "candidate_seeds", path, "geometry_audit")
        _partitions(ids, record.get("partitions"), path, "partitions")
        pools = _map(record.get("training_pools"), path)
        for key in ("online_training", "teacher_training"):
            _field(ids, _map(pools.get(key), path), "seeds", path, f"training_pools.{key}")
        actors = record.get("source_actors")
        if not isinstance(actors, list) or len(actors) != 2:
            raise SeedAuditError(f"{path}: missing teacher source actors")
        for n, actor in enumerate(actors):
            actor = _map(actor, path)
            actor_seeds = _field(ids, actor, "training_geometry_seeds", path, f"source_actors[{n}]")
            references.append((actor.get("episodes_path"), actor.get("episodes_sha256"), path,
                               set(actor_seeds)))
    else:
        raise SeedAuditError(f"{path}: unrecognized protocol schema {fmt!r}")


def _ledger(raw: bytes, path: str, ids: dict[int, list[dict[str, str]]],
            ambiguities: list[dict[str, Any]]) -> set[int]:
    observed: set[int] = set()
    for n, line in enumerate(raw.splitlines(), 1):
        row = _json(line, f"{path}:{n}")
        if type(row.get("track_id")) is not int or row["track_id"] < 0:
            raise SeedAuditError(f"{path}:{n}: missing track_id")
        if "event" in row and row["event"] not in ("reset", "end", "budget-stop"):
            raise SeedAuditError(f"{path}:{n}: unrecognized event")
        fields = [key for key in ("seed", "geometry_seed") if key in row]
        if not fields:
            raise SeedAuditError(f"{path}:{n}: missing ledger seed")
        values = [_uint32(row[key], f"{path}:{n}.{key}") for key in fields]
        for key, value in zip(fields, values):
            _append(ids, [value], path, f"line:{n}.{key}")
            observed.add(value)
        if len(set(values)) != 1:
            ambiguities.append({"path": path, "line": n, "fields": fields,
                                "reason": "seed and geometry_seed disagree"})
    return observed


def _collection(raw: bytes, path: str, protocol: dict[str, Any],
                ids: dict[int, list[dict[str, str]]]) -> None:
    lines = raw.splitlines()
    if not lines or len(lines) % 2:
        raise SeedAuditError(f"{path}: collection must contain complete reset/outcome pairs")
    training = set(_seeds(protocol.get("training_geometry_seeds"), path))
    allocated: set[tuple[int, int]] = set()
    for cell in protocol["teacher_data_cells"]:
        cell = _map(cell, f"{path}.teacher_data_cells")
        seed = _uint32(cell.get("geometry_seed"), f"{path}.teacher_data_cells.geometry_seed")
        track = cell.get("track_id")
        if (type(track) is not int or track not in (1, 2, 3, 4)
                or cell.get("obstacles") is not True or seed not in training
                or (seed, track) in allocated):
            raise SeedAuditError(f"{path}: malformed or duplicate TRAIN teacher allocation")
        allocated.add((seed, track))
    seen: set[int] = set()
    reset_keys = {"event", "cell_index", "episode_id", "geometry_seed", "track_id", "obstacles"}
    stored_keys = {"event", "episode_id", "geometry_seed", "track_id", "path", "sha256",
                   "teacher_actor_sha256", "finished", "progress", "retire_reason", "reward",
                   "steps", "terminal", "terminated", "truncated"}
    discarded_keys = {"event", "episode_id", "geometry_seed", "track_id", "decisions_spent", "reason"}
    for index in range(len(lines) // 2):
        reset = _json(lines[index * 2], f"{path}:{index * 2 + 1}")
        result = _json(lines[index * 2 + 1], f"{path}:{index * 2 + 2}")
        seed = _uint32(reset.get("geometry_seed"), f"{path}:{index * 2 + 1}.geometry_seed")
        track = reset.get("track_id")
        if (set(reset) != reset_keys or reset.get("event") != "reset"
                or type(reset.get("cell_index")) is not int or reset["cell_index"] != index
                or type(reset.get("episode_id")) is not int or reset["episode_id"] != index
                or type(track) is not int or track not in (1, 2, 3, 4)
                or reset.get("obstacles") is not True or seed in seen
                or seed not in training or (seed, track) not in allocated):
            raise SeedAuditError(f"{path}:{index * 2 + 1}: duplicate or non-TRAIN collection reset")
        seen.add(seed)
        _append(ids, [seed], path, f"line:{index * 2 + 1}.geometry_seed")
        if (result.get("event") not in ("stored_episode", "discarded_incomplete_episode")
                or type(result.get("episode_id")) is not int or result["episode_id"] != index
                or type(result.get("track_id")) is not int or result["track_id"] != track
                or _uint32(result.get("geometry_seed"), f"{path}:{index * 2 + 2}.geometry_seed") != seed):
            raise SeedAuditError(f"{path}:{index * 2 + 2}: mismatched collection outcome")
        if result["event"] == "stored_episode":
            if (set(result) != stored_keys
                    or result.get("path") != f"episodes/episode-{index:04d}.npz"
                    or not isinstance(result.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", result["sha256"])
                    or not isinstance(result.get("teacher_actor_sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", result["teacher_actor_sha256"])
                    or any(type(result.get(key)) is not bool for key in
                           ("finished", "terminal", "terminated", "truncated"))
                    or result["terminal"] is not True
                    or result["terminated"] == result["truncated"]
                    or type(result.get("steps")) is not int or result["steps"] <= 0
                    or type(result.get("progress")) not in (int, float)
                    or not math.isfinite(result["progress"]) or not 0 <= result["progress"] <= 1
                    or type(result.get("reward")) not in (int, float)
                    or not math.isfinite(result["reward"])
                    or (result.get("retire_reason") is not None
                        and (not isinstance(result["retire_reason"], str) or not result["retire_reason"]))):
                raise SeedAuditError(f"{path}:{index * 2 + 2}: malformed stored TRAIN episode metadata")
        elif (set(result) != discarded_keys or index != len(lines) // 2 - 1
              or type(result.get("decisions_spent")) is not int or result["decisions_spent"] <= 0
              or result.get("reason") != "teacher decision cap reached; no synthetic terminal inserted"):
            raise SeedAuditError(f"{path}:{index * 2 + 2}: malformed terminal discarded TRAIN episode")
        _append(ids, [seed], path, f"line:{index * 2 + 2}.geometry_seed")


def _training_ledgers(root: Path, training_roots: Sequence[str]) -> set[str]:
    paths: set[str] = set()
    for name in training_roots:
        directory = _safe(root, name, "root")
        if not directory.is_dir():
            raise SeedAuditError(f"missing TRAIN ledger root: {name}")
        for child in directory.iterdir():
            if child.is_symlink():
                raise SeedAuditError(f"symlink under TRAIN ledger root: {child.name}")
            if not child.is_dir() or _FORBIDDEN.search(child.name):
                continue  # Never traverse evaluator or blind directories.
            ledger = child / "episodes.jsonl"
            if ledger.is_symlink():
                raise SeedAuditError(f"symlink TRAIN ledger: {ledger.name}")
            if ledger.is_file():
                paths.add(f"{name}/{child.name}/episodes.jsonl")
    return paths


def _verify_r5_erratum(
    root: Path, erratum_path: str, candidates: list[int], track_id: int,
    catalog_path: str, inventory: dict[str, dict[str, Any]], raw_sources: dict[str, bytes],
) -> dict[str, Any]:
    """Accept only the known malformed r5 ledger-hash edge, not other hash drift."""
    raw = _read(root, erratum_path, "erratum", inventory)
    erratum = _json(raw, erratum_path)
    if (set(erratum) != {"format", "candidate_seeds", "candidate_seeds_sha256", "track_id",
                         "r5", "original_ledger_bytes_attested", "epistemic_limit"}
            or erratum["format"] != "haic-rlpd-g0-r5-receipt-erratum-v1"
            or erratum["candidate_seeds"] != candidates
            or erratum["candidate_seeds_sha256"] != _digest(candidates)
            or type(erratum["track_id"]) is not int or erratum["track_id"] != track_id
            or erratum["r5"] != R5_PROFILE
            or erratum["original_ledger_bytes_attested"] is not False
            or erratum["epistemic_limit"] != R5_EPISTEMIC_LIMIT):
        raise SeedAuditError(f"{erratum_path}: erratum schema, pins, candidate or limitation mismatch")
    for key in ("attempt_receipt", "supersession", "ledger", "protocol"):
        name, digest = R5_PROFILE[f"{key}_path"], R5_PROFILE[f"{key}_sha256"]
        if name not in inventory or inventory[name]["sha256"] != digest:
            raise SeedAuditError(f"{erratum_path}: {key} bytes disagree with pinned historical evidence")
    receipt = _json(raw_sources[R5_PROFILE["attempt_receipt_path"]], "r5 abort receipt")
    supersession = _json(raw_sources[R5_PROFILE["supersession_path"]], "r5 supersession")
    bad_hash = receipt.get("episodes_sha256")
    if (not isinstance(bad_hash, str) or not re.fullmatch(r"[0-9a-f]{65}", bad_hash)
            or bad_hash != R5_PROFILE["invalid_receipt_episodes_sha256"]
            or receipt.get("format") != "haic-drq-geometry-mix-partial-arm-abort-v1"
            or receipt.get("study_id") != "drqv2-geometry-mix-v1-r5"
            or receipt.get("protocol_path") != R5_PROFILE["protocol_path"]
            or receipt.get("protocol_sha256") != R5_PROFILE["protocol_sha256"]
            or receipt.get("run_dir") != R5_PROFILE["ledger_path"].rsplit("/", 1)[0]
            or receipt.get("source_seed") != 0 or receipt.get("variant") != "uniform"
            or receipt.get("step_metrics_sha256") != R5_PROFILE["step_metrics_sha256"]
            or receipt.get("online_replay_sample_trace_sha256") != R5_PROFILE["trace_sha256"]
            or receipt.get("unique_training_geometry_seeds_observed") != R5_PROFILE["ordered_reset_count"]
            or any(receipt.get(key) != R5_PROFILE[key] for key in
                   ("online_decisions", "learner_updates", "last_episode_id", "last_episode_step",
                    "last_geometry_seed"))
            or receipt.get("full_checkpoint_written") is not False
            or receipt.get("actor_candidate_written") is not False
            or receipt.get("resume_allowed") is not False
            or receipt.get("evaluation_receipts") != []
            or supersession.get("study_id") != "drqv2-geometry-mix-v1-r5"
            or supersession.get("attempt_receipt_path") != R5_PROFILE["attempt_receipt_path"]
            or supersession.get("attempt_receipt_sha256") != R5_PROFILE["attempt_receipt_sha256"]
            or supersession.get("superseded_protocol_path") != R5_PROFILE["protocol_path"]
            or supersession.get("superseded_protocol_sha256") != R5_PROFILE["protocol_sha256"]
            or supersession.get("online_decisions") != R5_PROFILE["online_decisions"]
            or supersession.get("learner_updates") != R5_PROFILE["learner_updates"]
            or supersession.get("full_checkpoint_written") is not False
            or supersession.get("resume_allowed") is not False
            or supersession.get("evaluation_receipts") != []
            or supersession.get("diagnostic_or_heldout_access") is not False):
        raise SeedAuditError(f"{erratum_path}: malformed hash is not the sole documented r5 receipt defect")
    for key in ("step_metrics", "trace"):
        name = R5_PROFILE[f"{key}_path"]
        auxiliary = _read(root, name, "auxiliary", inventory)
        if inventory[name]["sha256"] != R5_PROFILE[f"{key}_sha256"]:
            raise SeedAuditError(f"{erratum_path}: changed r5 {key} bytes")
        if key == "step_metrics":
            raw_sources[name] = auxiliary
    rows = [_json(line, f"r5 episodes:{n}") for n, line in
            enumerate(raw_sources[R5_PROFILE["ledger_path"]].splitlines(), 1)]
    resets = [row for row in rows if row.get("event") == "reset"]
    ends = [row for row in rows if row.get("event") == "end"]
    n = R5_PROFILE["ordered_reset_count"]
    if (len(rows) != n + R5_PROFILE["end_count"]
            or len(resets) != n or len(ends) != n - 1
            or [row.get("episode_id") for row in resets] != list(range(n))
            or [row.get("episode_id") for row in ends] != list(range(n - 1))
            or [row.get("event") for row in rows] != [
                event for i in range(n) for event in (("reset", "end") if i < n - 1 else ("reset",))
            ]):
        raise SeedAuditError(f"{erratum_path}: r5 ordered reset/end history differs")
    for reset, end in zip(resets, ends):
        if (reset.get("seed") != end.get("seed") or reset.get("track_id") != end.get("track_id")
                or reset.get("geometry_seed") != reset.get("seed")):
            raise SeedAuditError(f"{erratum_path}: r5 reset/end geometry identity differs")
    if (type(resets[-1].get("additional_online_step")) is not int
            or resets[-1]["additional_online_step"] + R5_PROFILE["last_episode_step"] + 1
            != R5_PROFILE["online_decisions"]
            or resets[-1].get("seed") != R5_PROFILE["last_geometry_seed"]
            or len({row["seed"] for row in resets}) != n):
        raise SeedAuditError(f"{erratum_path}: r5 partial-episode accounting differs")
    catalog = _json(raw_sources[catalog_path], "DrQ catalog")
    protocol = _json(raw_sources[R5_PROFILE["protocol_path"]], "r5 protocol")
    catalog_ids = set(_seeds(_map(catalog.get("seed_audit"), "catalog").get("proposed_seeds"), "catalog"))
    pool_ids = set(_seeds(_map(protocol.get("training_pool"), "r5 training_pool").get("geometry_seeds"),
                          "r5 training_pool"))
    if not {row["seed"] for row in resets} <= catalog_ids & pool_ids:
        raise SeedAuditError(f"{erratum_path}: r5 ledger seed outside reserved TRAIN catalog pool")
    return {"path": erratum_path, "sha256": inventory[erratum_path]["sha256"],
            "scope": "one malformed r5 abort-receipt episodes_sha256 reference only",
            "original_ledger_bytes_attested": False, "epistemic_limit": R5_EPISTEMIC_LIMIT}


def audit_g0_seeds(
    seed_start: int, track_id: int, *, repo_root: Path = Path("."),
    experiment_catalog_sha256: str = EXPERIMENT_JSON_CATALOG_SHA256,
    experiment_content_sha256: str = EXPERIMENT_CONTENT_CATALOG_SHA256,
    required_protocols: Sequence[str] = REQUIRED_PROTOCOLS,
    required_ledgers: Sequence[str] = REQUIRED_LEDGERS,
    required_collections: dict[str, str] = REQUIRED_COLLECTIONS,
    collection_sha256: dict[str, str] = REQUIRED_COLLECTION_SHA256,
    training_roots: Sequence[str] = TRAIN_ROOTS,
    catalog_path: str = CATALOG_PATH,
    catalog_protocol: str = CATALOG_PROTOCOL,
    partial_receipts: Sequence[str] = PARTIAL_RECEIPTS,
    attempt_receipts: Sequence[str] = ATTEMPT_RECEIPTS,
    r5_erratum: str | None = None,
) -> dict[str, Any]:
    """Return a read-only pass receipt, or raise SeedAuditBlocked/SeedAuditError.

    Source-list overrides exist for synthetic fixtures only; the CLI always uses
    the complete named production inventory. New live ledgers require re-auditing.
    """
    start = _uint32(seed_start, "seed_start")
    if start > UINT32_MAX - 11 or type(track_id) is not int or not 1 <= track_id <= 4:
        raise SeedAuditError("candidate range must fit uint32; track_id must be 1..4")
    candidates = list(range(start, start + 12))
    if r5_erratum is not None and r5_erratum != R5_ERRATUM_PATH:
        raise SeedAuditError(f"only the exact r5 erratum path is permitted: {R5_ERRATUM_PATH}")
    root = Path(repo_root).resolve()
    catalog_digest, experiment_evidence, experiment_files = _experiment_catalog(
        root, expected=experiment_catalog_sha256, expected_content=experiment_content_sha256,
        candidates=candidates, track_id=track_id)
    if _digest(experiment_evidence) != experiment_content_sha256:
        raise SeedAuditError("experiment JSON bytes changed; review cross-lane content before seed audit")
    if catalog_protocol not in required_protocols:
        raise SeedAuditError("missing required catalog protocol")
    inventory: dict[str, dict[str, Any]] = {}
    raw_sources: dict[str, bytes] = {}
    excluded: dict[int, list[dict[str, str]]] = {}
    ambiguities: list[dict[str, Any]] = []
    references: list[tuple[str, str, str, set[int] | None]] = []
    for path in sorted(required_protocols):
        raw = _read(root, path, "protocol", inventory)
        raw_sources[path] = raw
        record = _json(raw, path)
        _reject_seed_ranges(record, path)
        _protocol(record, path, excluded, references)
    raw = _read(root, catalog_path, "catalog", inventory)
    raw_sources[catalog_path] = raw
    catalog = _json(raw, catalog_path)
    if catalog.get("format") != "haic-drq-training-geometry-catalog-v1":
        raise SeedAuditError("unrecognized DrQ catalog schema")
    catalog_seeds = _field(excluded, _map(catalog.get("seed_audit"), catalog_path),
                           "proposed_seeds", catalog_path, "seed_audit")
    if catalog_seeds != list(range(3910800001, 3910800513)):
        raise SeedAuditError("DrQ catalog must reserve all 512 candidate IDs")
    if catalog.get("protocol_sha256") != inventory[catalog_protocol]["sha256"]:
        raise SeedAuditError("catalog protocol SHA-256 mismatch")
    for path in sorted(attempt_receipts):
        raw = _read(root, path, "receipt", inventory)
        raw_sources[path] = raw
        receipt = _json(raw, path)
        protocol = receipt.get("protocol_path")
        ledger_path = f"{path.rsplit('/', 1)[0]}/episodes.jsonl"
        if (receipt.get("format") != "haic-drq-geometry-mix-partial-arm-abort-v1"
                or protocol not in inventory
                or receipt.get("protocol_sha256") != inventory[protocol]["sha256"]
                or receipt.get("full_checkpoint_written") is not False
                or receipt.get("actor_candidate_written") is not False):
            raise SeedAuditError(f"{path}: invalid partial attempt receipt")
        last_seed = _uint32(receipt.get("last_geometry_seed"), f"{path}.last_geometry_seed")
        _append(excluded, [last_seed], path, "last_geometry_seed")
        references.append((ledger_path, receipt.get("episodes_sha256"), path, None))
    for path in sorted(partial_receipts):
        raw = _read(root, path, "receipt", inventory)
        raw_sources[path] = raw
        receipt = _json(raw, path)
        protocol = receipt.get("superseded_protocol_path")
        attempt = receipt.get("attempt_receipt_path")
        if (receipt.get("format") != "haic-drq-geometry-mix-precheckpoint-supersession-v1"
                or receipt.get("status") != "partial-train-attempt-incomplete"
                or protocol not in inventory
                or receipt.get("superseded_protocol_sha256") != inventory[protocol]["sha256"]
                or attempt not in inventory or inventory[attempt]["kind"] != "receipt"
                or ("attempt_receipt_sha256" in receipt and
                    receipt["attempt_receipt_sha256"] != inventory[attempt]["sha256"])):
            raise SeedAuditError(f"{path}: incomplete or mismatched partial receipt")
    expected_ledgers = set(required_ledgers)
    expected_ledgers.update(_training_ledgers(root, training_roots))
    ledger_sets: dict[str, set[int]] = {}
    for path in sorted(expected_ledgers):
        raw = _read(root, path, "ledger", inventory)
        raw_sources[path] = raw
        ledger_sets[path] = _ledger(raw, path, excluded, ambiguities)
    for path, protocol_path in sorted(required_collections.items()):
        if protocol_path not in inventory:
            raise SeedAuditError(f"{path}: required collection protocol missing: {protocol_path}")
        raw = _read(root, path, "collection", inventory)
        if inventory[path]["sha256"] != collection_sha256.get(path):
            raise SeedAuditError(f"{path}: frozen TRAIN collection ledger hash drift")
        raw_sources[path] = raw
        _collection(raw, path, _json(raw_sources[protocol_path], protocol_path), excluded)
    applied_erratum: dict[str, Any] | None = None
    for path, digest, source, expected_seeds in references:
        if (not isinstance(path, str) or path not in inventory
                or inventory[path]["kind"] not in ("catalog", "ledger")
                or not isinstance(digest, str)):
            raise SeedAuditError(f"{source}: required referenced catalog/teacher ledger absent or hash mismatch: {path}")
        if inventory[path]["sha256"] != digest:
            if (r5_erratum is None or applied_erratum is not None
                    or source != R5_PROFILE["attempt_receipt_path"]
                    or path != R5_PROFILE["ledger_path"]
                    or digest != R5_PROFILE["invalid_receipt_episodes_sha256"]):
                raise SeedAuditError(f"{source}: required referenced catalog/teacher ledger absent or hash mismatch: {path}")
            applied_erratum = _verify_r5_erratum(root, r5_erratum, candidates, track_id, catalog_path,
                                                 inventory, raw_sources)
        if expected_seeds is not None and ledger_sets[path] != expected_seeds:
            raise SeedAuditError(f"{source}: teacher source seed inventory disagrees with ledger: {path}")
    if r5_erratum is not None and applied_erratum is None:
        raise SeedAuditError("r5 erratum requested but its exact defective receipt edge was not found")
    # Search exact decimal tokens, never substrings of longer IDs, floats in words,
    # filenames or hexadecimal hashes. Unexplained matches block rather than assert use.
    token = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(map(str, candidates)) + r")(?![A-Za-z0-9_.])")
    typed = set(excluded)
    for path, raw in sorted(raw_sources.items()):
        for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            for match in token.finditer(line):
                seed = int(match.group())
                if seed not in typed:
                    ambiguities.append({"path": path, "line": n, "seed": seed,
                                        "reason": "exact token outside recognized geometry seed fields"})
    for path, raw in sorted(experiment_files.items()):
        if path in inventory:
            if inventory[path]["sha256"] != hashlib.sha256(raw).hexdigest():
                raise SeedAuditError(f"{path}: typed experiment source changed while auditing")
            continue
        for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            for match in token.finditer(line):
                ambiguities.append({"path": path, "line": n, "seed": int(match.group()),
                                    "reason": "exact token in otherwise-untyped experiment JSON"})
    collisions = [{"seed": seed, "sources": excluded[seed]} for seed in candidates if seed in excluded]
    evidence = [inventory[name] for name in sorted(inventory)]
    report = {
        "format": "haic-rlpd-g0-seed-audit-v1",
        "status": "blocked" if collisions or ambiguities else "no_known_recorded_overlap",
        "passed": not collisions and not ambiguities,
        "candidate_rule": "seed_start + offset for offsets 0..11, in that order; no replacement on collision",
        "seed_start": start,
        "candidate_seeds": candidates,
        "candidate_seeds_sha256": _digest(candidates),
        "cells": [{"track_id": track_id, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}
                  for seed in candidates],
        "excluded_seed_count": len(excluded),
        "excluded_inventory_sha256": _digest(sorted(excluded)),
        "collisions": collisions,
        "collision_count": len(collisions),
        "ambiguities": ambiguities,
        "source_inventory": evidence,
        "source_inventory_sha256": _digest(evidence),
        "experiments_catalog_sha256": catalog_digest,
        "experiment_content_inventory": experiment_evidence,
        "experiment_content_inventory_sha256": _digest(experiment_evidence),
        "experiment_content_scope": "all experiments/*.json except verified own G0 receipt and single self protocol",
        "freshness_claim": FRESHNESS_LIMITATION,
        "blind_episode_access": "none; protocol blind seed IDs used only as exclusions",
        "read_scope": "all experiment JSON metadata, named TRAIN ledgers/catalog/partial receipts, four prior-data collection.jsonl",
        "protocol_frozen": False,
        "r5_erratum": applied_erratum,
    }
    if not report["passed"]:
        raise SeedAuditBlocked(report)
    return report


def write_clean_receipt(report: dict[str, Any], *, repo_root: Path,
                        output: str, training_roots: Sequence[str] = TRAIN_ROOTS) -> None:
    """Opt-in exclusive receipt write after a fresh inventory check, not a protocol freeze."""
    if report.get("status") != "no_known_recorded_overlap" or report.get("passed") is not True:
        raise SeedAuditError("refusing to write a blocked or incomplete receipt")
    root = Path(repo_root).resolve()
    destination = _safe(root, output, "protocol")
    if (not output.startswith("experiments/") or output.count("/") != 1
            or not output.endswith("-seed-audit.json") or destination.is_file()):
        raise SeedAuditError("receipt output must be a new experiments/*-seed-audit.json")
    if not destination.parent.is_dir():
        raise SeedAuditError("receipt output directory does not exist")
    inventory = report.get("source_inventory")
    if (not isinstance(inventory, list) or not inventory
            or _digest(inventory) != report.get("source_inventory_sha256")):
        raise SeedAuditError("invalid source inventory in receipt")
    _, experiment_evidence, _ = _experiment_catalog(
        root, expected=report.get("experiments_catalog_sha256"),
        expected_content=report.get("experiment_content_inventory_sha256"),
        candidates=report.get("candidate_seeds"), track_id=report["cells"][0]["track_id"])
    if (not isinstance(report.get("experiment_content_inventory"), list)
            or experiment_evidence != report["experiment_content_inventory"]
            or _digest(experiment_evidence) != report.get("experiment_content_inventory_sha256")):
        raise SeedAuditError("experiment JSON content changed before receipt write")
    known_ledgers = {row["path"] for row in inventory if row.get("kind") == "ledger"}
    if not _training_ledgers(root, training_roots) <= known_ledgers:
        raise SeedAuditError("new TRAIN ledger discovered; rerun the full audit")
    for row in inventory:
        name, kind, expected = row["path"], row["kind"], row["sha256"]
        path = _safe(root, name, kind)
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise SeedAuditError(f"missing source during receipt recheck: {name}") from exc
        if actual != expected:
            raise SeedAuditError(f"source changed before receipt write: {name}")
    try:
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise SeedAuditError(f"receipt already exists: {output}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--track-id", type=int, required=True)
    parser.add_argument("--output", help="opt-in exclusive experiments/*-seed-audit.json receipt; default is read-only")
    parser.add_argument("--r5-erratum", metavar=R5_ERRATUM_PATH,
                        help="read-only exact r5 abort-receipt erratum; never waive other source hashes")
    args = parser.parse_args(argv)
    try:
        report = audit_g0_seeds(args.seed_start, args.track_id, repo_root=args.repo_root,
                                r5_erratum=args.r5_erratum)
        if args.output is not None:
            write_clean_receipt(report, repo_root=args.repo_root, output=args.output)
    except SeedAuditBlocked as exc:
        print(json.dumps(exc.report, sort_keys=True, indent=2))
        return 1
    except SeedAuditError as exc:
        print(f"G0 seed audit rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
