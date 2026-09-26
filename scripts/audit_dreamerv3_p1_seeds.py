"""Inspect current recorded geometry IDs without opening held-out episodes.

Catalog and TRAIN metadata are checked against the live tree, not a historical
G0 snapshot. This is not a global historical non-use certificate. In particular,
the P1 protocol/receipt SHA cycle has no verified bootstrap yet, so this auditor
cannot issue a passing collector receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FORMAT = "haic-dreamerv3-p1-cross-lane-seed-audit-v1"
READ_SCOPE = "SHA-pinned ID-only protocols, prior audits, TRAIN episode ledgers and TRAIN prior collection ledgers"
FRESHNESS_LIMITATION = "no known recorded overlap; historical pilot schedules are incomplete"
INVENTORY_BLOCKER = (
    "A typed, independently frozen complete current recorded source inventory and "
    "a P1 protocol/audit receipt SHA-256 bootstrap remain unresolved; historical "
    "pilot schedules are incomplete and global non-use cannot be inferred."
)
R5_BLOCKER = (
    "The r5 partial-run abort receipt has a malformed 65-character episodes_sha256; "
    "its original receipt cannot attest original ledger bytes, even when the "
    "current ledger/metrics/trace chain is independently rechecked."
)
KINDS = ("protocol", "prior_seed_audit", "training_ledger", "prior_collection_ledger")
# These are necessary collector inputs, never a sufficient inventory certificate.
KNOWN_INPUTS = {
    "protocol": frozenset({
        "experiments/drqv2-geometry-mix-v1-r6.json",
        "experiments/drqv2-teacher-replay-v1-r3.json",
        "experiments/pixel-rlpd-entropy-target-ablation-v5.json",
        "experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json",
    }),
    "prior_collection_ledger": frozenset({
        "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/collection.jsonl",
    }),
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_HELD_OUT = re.compile(r"blind|confirm|screen|eval|held.?out|holdout|submission", re.I)
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_AUDIT = "experiments/rlpd-g0-completion-v1-seed-audit.json"
G0_CLAIMS = "runs/rlpd-g0-claims"
G0_RUN = "runs/20260926-rlpd-g0-completion-v1"
R5_RUN = "runs/20260925-drqv2-geometry-mix-v1-r5"
R5_PROTOCOL = "experiments/drqv2-geometry-mix-v1-r5.json"
R5_ERRATUM = "experiments/drqv2-geometry-mix-v1-r5-seed-audit-erratum-v1.json"
# These pins are historical primary-artifact byte identities, not the G0 verifier's
# candidate-bound pass receipt. No other malformed hash receives an exception.
R5_PINS = {
    f"{R5_RUN}/learner-0-uniform/precheckpoint-abort.json": "da17593a7853e30c19f37ee53c87de2300198309ceb1b046742bc5f56fe994d3",
    f"{R5_RUN}/protocol-superseded-after-partial.json": "dfc5bcb050090278e3213af101d44e413ac0f70fce5727204bcb8424e0f3ca60",
    f"{R5_RUN}/learner-0-uniform/episodes.jsonl": "8d192a9cc87ee3777e3647b2e49eeec7eb66b84eb12099a0ea7c569230889014",
    f"{R5_RUN}/learner-0-uniform/step-metrics.jsonl": "47d381deae01fe77a9272f74869a3061941d11716d44c4052db665f4ee1a55f4",
    f"{R5_RUN}/learner-0-uniform/replay-sample-trace-step-000016384.npz": "6d01e0e45eb6d0365c448f2d1b302961a8715dfd74a46b38f7569b25705f3e96",
    R5_PROTOCOL: "531ca7838c0e911eec70cf729073753367df97ddcb6aedea1fe92091c8327000",
}
_TOKEN_BOUNDARY = r"(?<![A-Za-z0-9_.])(?:{})(?![A-Za-z0-9_.])"


class SeedAuditError(ValueError):
    """An unsafe, unpinned, missing, or uninterpretable declared input."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SeedAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw: bytes, where: str) -> dict[str, Any]:
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                            parse_constant=lambda x: (_ for _ in ()).throw(SeedAuditError(f"{where}: invalid {x}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SeedAuditError(f"{where}: invalid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise SeedAuditError(f"{where}: expected an object")
    return result


def _map(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SeedAuditError(f"{where}: missing or opaque object")
    return value


def _uint32(value: Any, where: str) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise SeedAuditError(f"{where}: expected uint32 geometry seed ID")
    return value


def _seeds(value: Any, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SeedAuditError(f"{where}: missing or empty geometry seed list")
    result = [_uint32(seed, where) for seed in value]
    if len(result) != len(set(result)):
        raise SeedAuditError(f"{where}: duplicate geometry seed")
    return result


def _source(root: Path, name: str, digest: str, kind: str) -> bytes:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise SeedAuditError(f"{kind}: path must be normalized and repository-relative")
    parts = name.split("/")
    if kind in ("protocol", "prior_seed_audit"):
        allowed = (len(parts) == 2 and parts[0] == "experiments" and name.endswith(".json")
                   and (bool(re.search(r"-geometry-audit(?:-final)?\.json\Z", name)) or name == G0_AUDIT)
                   == (kind == "prior_seed_audit")
                   and not re.search(r"result|diagnostic|outcome|road", parts[-1], re.I))
    else:
        allowed = (len(parts) >= 3 and parts[0] == "runs"
                   and parts[-1] == ("episodes.jsonl" if kind == "training_ledger" else "collection.jsonl"))
    if (not allowed or any(_HELD_OUT.search(part) for part in parts[1:])
            or not isinstance(digest, str) or not _HASH.fullmatch(digest)):
        raise SeedAuditError(f"{kind}: path or SHA-256 outside ID-only allowlist: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise SeedAuditError(f"{kind}: symlink forbidden: {name}")
    if not path.is_file():
        raise SeedAuditError(f"{kind}: missing pinned source: {name}")
    try:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise SeedAuditError(f"{kind}: ID-only source exceeds 32 MiB: {name}")
        raw = path.read_bytes()
    except OSError as exc:
        raise SeedAuditError(f"{kind}: cannot read pinned source: {name}") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SeedAuditError(f"{kind}: SHA-256 mismatch: {name}")
    return raw


def _metadata(root: Path, name: str, digest: str | None = None) -> bytes:
    """Read only a fixed run-metadata/experiment path, never episode payloads."""
    parts = name.split("/")
    if (not name or "\\" in name or any(part in ("", ".", "..") for part in parts)
            or (parts[0] == "runs" and _HELD_OUT.search(name))
            or parts[0] not in ("runs", "experiments")
            or (parts[0] == "experiments" and (len(parts) != 2 or not name.endswith(".json")))
            or (parts[0] == "runs" and not (
                name.startswith(G0_CLAIMS + "/") and len(parts) == 3 and name.endswith(".json")
                or name in (f"{G0_RUN}/manifest.json", f"{G0_RUN}/cells.jsonl")
                or name in R5_PINS))):
        raise SeedAuditError(f"unsafe ID metadata path: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise SeedAuditError(f"symlinked ID metadata: {name}")
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise SeedAuditError(f"missing or oversized ID metadata: {name}")
    raw = path.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise SeedAuditError(f"SHA-256 mismatch in ID metadata: {name}")
    return raw


def _experiment_catalog(root: Path) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    directory = root / "experiments"
    if directory.is_symlink() or not directory.is_dir():
        raise SeedAuditError("missing or symlinked experiments JSON catalog")
    contents: dict[str, bytes] = {}
    for entry in directory.iterdir():
        if entry.suffix != ".json":
            continue
        name = f"experiments/{entry.name}"
        raw = _metadata(root, name)
        _json(raw, name)
        contents[name] = raw
    evidence = [{"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                for name, raw in sorted(contents.items())]
    return evidence, contents


def _training_inventory(root: Path) -> tuple[set[str], set[str]]:
    directory = root / "runs"
    if directory.is_symlink() or not directory.is_dir():
        raise SeedAuditError("missing or symlinked TRAIN runs directory")
    ledgers: set[str] = set()
    collections: set[str] = set()
    def cannot_walk(exc: OSError) -> None:
        raise SeedAuditError(f"TRAIN inventory traversal failed: {exc}") from exc

    for folder, dirs, files in os.walk(directory, followlinks=False, onerror=cannot_walk):
        base = Path(folder)
        dirs[:] = [name for name in dirs if not _HELD_OUT.search(name)]
        for name in dirs:
            if (base / name).is_symlink():
                raise SeedAuditError(f"symlink in TRAIN inventory: {base / name}")
        for name, target in (("episodes.jsonl", ledgers), ("collection.jsonl", collections)):
            if name in files:
                relative = (base / name).relative_to(root).as_posix()
                if (base / name).is_symlink():
                    raise SeedAuditError(f"symlinked TRAIN ledger: {relative}")
                target.add(relative)
    return ledgers, collections


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def _add(ids: dict[int, list[dict[str, str]]], seeds: list[int], path: str, field: str) -> None:
    for seed in seeds:
        source = {"path": path, "field": field}
        if source not in ids.setdefault(seed, []):
            ids[seed].append(source)


def _protocol(record: dict[str, Any], path: str, ids: dict[int, list[dict[str, str]]],
              sources: Mapping[str, Mapping[str, str]]) -> None:
    """Only explicit, known geometry-ID layouts; never follow metadata pointers."""
    known: set[str] = set()

    def field(obj: Any, key: str, prefix: str = "") -> list[int]:
        value = _map(obj, f"{path}.{prefix}").get(key)
        name = f"{prefix}.{key}" if prefix else key
        seeds = _seeds(value, f"{path}.{name}")
        known.add(name)
        _add(ids, seeds, path, name)
        return seeds

    def cells(items: Any, prefix: str, key: str) -> set[int]:
        if not isinstance(items, list) or not items:
            raise SeedAuditError(f"{path}.{prefix}: missing geometry cells")
        values: set[int] = set()
        for index, item in enumerate(items):
            name = f"{prefix}[{index}].{key}"
            seed = _uint32(_map(item, f"{path}.{name}").get(key), f"{path}.{name}")
            known.add(name)
            _add(ids, [seed], path, name)
            values.add(seed)
        return values

    def partitions(obj: Any, prefix: str) -> None:
        parts = _map(obj, f"{path}.{prefix}")
        if set(parts) != {"screen", "confirmation", "blind"}:
            raise SeedAuditError(f"{path}.{prefix}: incomplete exclusion partitions")
        for part in ("screen", "confirmation", "blind"):
            field(parts[part], "seeds", f"{prefix}.{part}")

    fmt = record.get("format")
    name = path.removeprefix("experiments/")
    if name.startswith("drqv2-teacher-replay-") and fmt is None:
        if not isinstance(record.get("study_id"), str) or not record["study_id"].startswith("drqv2-teacher-replay-"):
            raise SeedAuditError(f"{path}: wrong teacher protocol identity")
        field(record, "reserved_training_seeds")
        field(record, "known_excluded_geometry_seeds")
        audit = _map(record.get("geometry_audit"), path)
        field(audit, "candidate_seeds", "geometry_audit")
        audit_path, audit_hash = audit.get("report_path"), audit.get("report_sha256")
        if sources["prior_seed_audit"].get(audit_path) != audit_hash or not audit_hash:
            raise SeedAuditError(f"{path}: referenced prior audit must be declared and SHA-pinned")
        partitions(record.get("partitions"), "partitions")
        pools = _map(record.get("training_pools"), path)
        for key in ("online_training", "teacher_training"):
            field(pools.get(key), "seeds", f"training_pools.{key}")
        actors = record.get("source_actors")
        if not isinstance(actors, list) or not actors:
            raise SeedAuditError(f"{path}: missing source actors")
        for index, actor in enumerate(actors):
            item = _map(actor, path)
            field(item, "training_geometry_seeds", f"source_actors[{index}]")
            if (not item.get("episodes_sha256")
                    or sources["training_ledger"].get(item.get("episodes_path")) != item["episodes_sha256"]):
                raise SeedAuditError(f"{path}: source actor training ledger must be declared and SHA-pinned")
    elif name.startswith("pixel-rlpd-") and fmt == "haic-pixel-rlpd-study-v1":
        field(record, "reserved_training_seeds")
        training = set(field(record, "training_geometry_seeds"))
        if cells(record.get("teacher_data_cells"), "teacher_data_cells", "geometry_seed") - training:
            raise SeedAuditError(f"{path}: teacher cells outside TRAIN allocation")
        partitions(record.get("partitions"), "partitions")
        if "future_full_reservation" in record:
            future = _map(record["future_full_reservation"], path)
            reserved = set(field(future, "training_geometry_seeds", "future_full_reservation"))
            if cells(future.get("teacher_data_cells"), "future_full_reservation.teacher_data_cells", "geometry_seed") - reserved:
                raise SeedAuditError(f"{path}: future teacher cells outside reserved allocation")
            partitions({key: future.get(key) for key in ("screen", "confirmation", "blind")},
                       "future_full_reservation")
        audit = _map(record.get("geometry_audit"), path)
        if "teacher_source_ledgers" in audit:
            ledgers = audit["teacher_source_ledgers"]
            if not isinstance(ledgers, list) or not ledgers:
                raise SeedAuditError(f"{path}: opaque teacher source ledger list")
            for index, ledger in enumerate(ledgers):
                item = _map(ledger, path)
                field(item, "geometry_seeds", f"geometry_audit.teacher_source_ledgers[{index}]")
                if (not item.get("sha256")
                        or sources["training_ledger"].get(item.get("path")) != item["sha256"]):
                    raise SeedAuditError(f"{path}: teacher source ledger must be declared and SHA-pinned")
    elif name.startswith("drqv2-geometry-augmentation-") and fmt == "haic-drq-training-geometry-protocol-v1":
        field(record, "candidate_seeds")
        exclusions = _map(record.get("exclusion_seed_ids"), path)
        for key in ("reserved", "heldout", "blind"):
            field(exclusions, key, "exclusion_seed_ids")
    elif name.startswith("drqv2-geometry-mix-") and fmt == "haic-drq-geometry-mix-study-v1":
        env = _map(record.get("environment"), path)
        diagnostic = _map(record.get("diagnostic_pool"), path)
        if env.get("partition") != "TRAIN" or diagnostic.get("partition") != "TRAIN-DIAGNOSTIC":
            raise SeedAuditError(f"{path}: wrong TRAIN/diagnostic partition")
        field(env, "geometry_seeds", "environment")
        field(diagnostic, "geometry_seeds", "diagnostic_pool")
        if "training_pool" in record:
            pool = _map(record["training_pool"], path)
            if pool.get("partition") != "TRAIN":
                raise SeedAuditError(f"{path}: wrong training_pool partition")
            field(pool, "geometry_seeds", "training_pool")
        if "diagnostics" in record and "geometry_seeds" in _map(record["diagnostics"], path):
            field(record["diagnostics"], "geometry_seeds", "diagnostics")
        if "runs" in record or "seeds_by_source" in record:
            sources_by_seed = record.get("seeds_by_source")
            runs = record.get("runs")
            if (not isinstance(sources_by_seed, list) or not sources_by_seed
                    or not isinstance(runs, list) or not runs):
                raise SeedAuditError(f"{path}: incomplete geometry-sampler RNG schedule")
            geometry_rng: dict[int, int] = {}
            for index, raw in enumerate(sources_by_seed):
                source = _map(raw, f"{path}.seeds_by_source[{index}]")
                learner = source.get("learner_seed")
                if type(learner) is not int or learner in geometry_rng:
                    raise SeedAuditError(f"{path}: invalid source learner seed")
                name = f"seeds_by_source[{index}].geometry_seed"
                geometry_rng[learner] = _uint32(source.get("geometry_seed"), f"{path}.{name}")
                known.add(name)
            seen_learners: set[int] = set()
            for index, raw in enumerate(runs):
                run = _map(raw, f"{path}.runs[{index}]")
                learner = run.get("learner_seed")
                name = f"runs[{index}].geometry_seed"
                if (type(learner) is not int or learner not in geometry_rng
                        or _uint32(run.get("geometry_seed"), f"{path}.{name}") != geometry_rng[learner]):
                    raise SeedAuditError(f"{path}: run geometry-sampler RNG differs from its source")
                seen_learners.add(learner)
                known.add(name)
            if seen_learners != set(geometry_rng):
                raise SeedAuditError(f"{path}: incomplete source RNG schedule")
    elif name.startswith("dreamerv3-b1-") and fmt == "haic-dreamerv3-study-protocol-v1":
        field(record, "reserved_training_seeds")
        development = _map(record.get("training_development"), path)
        dev_seeds = set(field(development, "seeds", "training_development"))
        if cells(development.get("cells"), "training_development.cells", "seed") != dev_seeds:
            raise SeedAuditError(f"{path}: development cells and IDs differ")
        partitions(record.get("partitions"), "partitions")
    elif path == G0_PROTOCOL and fmt == "haic-rlpd-g0-diagnostic-v1":
        if (record.get("status") != "frozen" or record.get("partition") != "TRAIN"
                or record.get("geometry_audit_path") != G0_AUDIT):
            raise SeedAuditError(f"{path}: G0 protocol must bind frozen TRAIN audit")
        cells(record.get("cells"), "cells", "geometry_seed")
        for index, cell in enumerate(record["cells"]):
            if (set(cell) != {"track_id", "geometry_seed", "partition", "obstacles"}
                    or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)
                    or cell["partition"] != "TRAIN" or cell["obstacles"] is not True):
                raise SeedAuditError(f"{path}: invalid G0 TRAIN cell {index}")
    else:
        raise SeedAuditError(f"{path}: unsupported protocol path/format; cannot skip unknown allocations")

    def check_unknown(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                field_name = f"{prefix}.{key}" if prefix else key
                if ((key in ("seed", "seeds", "geometry_seed", "geometry_seeds")
                     or key.endswith(("_seeds", "_seed_ids", "_seed_start"))
                     or "seed_range" in key or "geometry_id" in key)
                        and field_name not in known and key != "learner_seeds"):
                    raise SeedAuditError(f"{path}: unknown geometry seed field {field_name}")
                check_unknown(item, field_name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check_unknown(item, f"{prefix}[{index}]")

    check_unknown(record)


def _prior_audit(record: dict[str, Any], path: str, ids: dict[int, list[dict[str, str]]]) -> None:
    if path == G0_AUDIT:
        if (record.get("format") != "haic-rlpd-g0-seed-audit-v1"
                or record.get("status") != "no_known_recorded_overlap"
                or record.get("passed") is not True or record.get("collisions") != []
                or record.get("ambiguities") != []
                or record.get("freshness_claim") != FRESHNESS_LIMITATION
                or record.get("protocol_frozen") is not False):
            raise SeedAuditError(f"{path}: unsupported G0 historical audit")
        candidates = _seeds(record.get("candidate_seeds"), path)
        cells = record.get("cells")
        if (record.get("candidate_seeds_sha256") != _digest_json(candidates)
                or not isinstance(cells, list) or len(cells) != len(candidates)
                or [cell.get("geometry_seed") for cell in cells if isinstance(cell, dict)] != candidates
                or any(not isinstance(cell, dict) or set(cell) != {"track_id", "geometry_seed", "partition", "obstacles"}
                       or cell.get("partition") != "TRAIN" or cell.get("obstacles") is not True
                       or type(cell.get("track_id")) is not int or cell["track_id"] not in (1, 2, 3, 4)
                       for cell in cells)):
            raise SeedAuditError(f"{path}: G0 candidate/cell schedule differs")
        for name in ("source_inventory", "experiment_content_inventory"):
            items = record.get(name)
            if (not isinstance(items, list) or not items
                    or record.get(f"{name}_sha256") != _digest_json(items)):
                raise SeedAuditError(f"{path}: historical {name} metadata is inconsistent")
        # Its historical source inventory is NOT the present experiments catalog.
        _add(ids, candidates, path, "candidate_seeds")
        return
    if (record.get("passed") is not True or record.get("parse_errors") != []
            or record.get("global_freshness_claim") != FRESHNESS_LIMITATION):
        raise SeedAuditError(f"{path}: unsuccessful/unsupported prior seed audit")
    excluded = _seeds(record.get("known_excluded_geometry_seeds"), path)
    candidates = _seeds(record.get("candidate_seeds"), path)
    if (type(record.get("known_excluded_geometry_seed_count")) is not int
            or record["known_excluded_geometry_seed_count"] != len(excluded)):
        raise SeedAuditError(f"{path}: prior exclusion count differs")
    snapshots = record.get("source_snapshots")
    if (type(record.get("source_snapshot_count")) is not int
            or record["source_snapshot_count"] < 1 or not isinstance(snapshots, list)
            or len(snapshots) != record["source_snapshot_count"]
            or any(not isinstance(item, dict) or set(item) != {"path", "sha256"}
                   or not isinstance(item["path"], str) or not isinstance(item["sha256"], str)
                   or not _HASH.fullmatch(item["sha256"]) for item in snapshots)):
        raise SeedAuditError(f"{path}: incomplete prior audit snapshot metadata")
    # Snapshot paths are metadata, never file access or an authoritative inventory.
    _add(ids, excluded, path, "known_excluded_geometry_seeds")
    _add(ids, candidates, path, "candidate_seeds")


def _ledger(raw: bytes, path: str, ids: dict[int, list[dict[str, str]]], kind: str) -> None:
    if not raw.strip():
        raise SeedAuditError(f"{path}: empty TRAIN ledger")
    for number, line in enumerate(raw.splitlines(), 1):
        row = _json(line, f"{path}:{number}")
        if any(("seed_range" in key or key.endswith(("_seed_start", "_seed_ids", "_seeds")))
               for key in row):
            raise SeedAuditError(f"{path}:{number}: unknown TRAIN ledger seed schema")
        if type(row.get("track_id")) is not int or row["track_id"] not in (1, 2, 3, 4):
            raise SeedAuditError(f"{path}:{number}: invalid TRAIN track_id")
        if kind == "prior_collection_ledger":
            if row.get("event") not in ("reset", "stored_episode", "discarded_incomplete_episode") or "seed" in row:
                raise SeedAuditError(f"{path}:{number}: opaque prior collection row")
            key = "geometry_seed"
        else:
            if row.get("event", "reset") not in ("reset", "end", "budget-stop"):
                raise SeedAuditError(f"{path}:{number}: opaque training episode event")
            if "seed" not in row and "geometry_seed" not in row:
                raise SeedAuditError(f"{path}:{number}: missing training episode seed")
            if "seed" in row and "geometry_seed" in row and (
                _uint32(row["seed"], f"{path}:{number}.seed")
                != _uint32(row["geometry_seed"], f"{path}:{number}.geometry_seed")
            ):
                raise SeedAuditError(f"{path}:{number}: conflicting training episode IDs")
            key = "geometry_seed" if "geometry_seed" in row else "seed"
        seed = _uint32(row.get(key), f"{path}:{number}.{key}")
        _add(ids, [seed], path, f"line:{number}.{key}")


def _g0_metadata(root: Path, contents: Mapping[str, bytes],
                 ids: dict[int, list[dict[str, str]]]) -> list[dict[str, Any]]:
    if G0_PROTOCOL not in contents and G0_AUDIT not in contents:
        if (root / G0_CLAIMS).exists() or (root / G0_RUN).exists():
            raise SeedAuditError("G0 TRAIN metadata exists without its protocol and audit")
        return []
    if G0_PROTOCOL not in contents or G0_AUDIT not in contents:
        raise SeedAuditError("G0 protocol and historical audit must both exist")
    protocol = _json(contents[G0_PROTOCOL], G0_PROTOCOL)
    receipt = _json(contents[G0_AUDIT], G0_AUDIT)
    audit_hash = hashlib.sha256(contents[G0_AUDIT]).hexdigest()
    protocol_hash = hashlib.sha256(contents[G0_PROTOCOL]).hexdigest()
    if (protocol.get("geometry_audit_path") != G0_AUDIT
            or protocol.get("geometry_audit_sha256") != audit_hash
            or protocol.get("cells") != receipt.get("cells")):
        raise SeedAuditError("G0 protocol/cells/seed-audit chain differs")
    _prior_audit(receipt, G0_AUDIT, ids)
    claim_dir = root / G0_CLAIMS
    if claim_dir.is_symlink() or not claim_dir.is_dir():
        raise SeedAuditError("G0 consumed claim directory missing or symlinked")
    claim_paths = sorted(f"{G0_CLAIMS}/{entry.name}" for entry in claim_dir.iterdir()
                         if entry.suffix == ".json")
    expected_claim = f"{G0_CLAIMS}/{receipt['candidate_seeds_sha256']}.json"
    if claim_paths != [expected_claim]:
        raise SeedAuditError("G0 consumed claim inventory differs from audited candidate batch")
    claim_raw = _metadata(root, expected_claim)
    claim = _json(claim_raw, expected_claim)
    if (set(claim) != {"format", "geometry_audit_sha256", "geometry_seeds", "output_root",
                       "protocol_sha256", "status"}
            or claim.get("format") != "haic-rlpd-g0-geometry-claim-v1"
            or claim.get("status") != "reserved-once" or claim.get("output_root") != G0_RUN
            or claim.get("geometry_audit_sha256") != audit_hash
            or claim.get("protocol_sha256") != protocol_hash
            or claim.get("geometry_seeds") != receipt["candidate_seeds"]):
        raise SeedAuditError("G0 claim must bind the audited, frozen TRAIN cells")
    cells_path = f"{G0_RUN}/cells.jsonl"
    manifest_path = f"{G0_RUN}/manifest.json"
    cells_raw = _metadata(root, cells_path)
    manifest_raw = _metadata(root, manifest_path)
    manifest = _json(manifest_raw, manifest_path)
    if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("protocol_path") != G0_PROTOCOL
            or manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("geometry_audit_sha256") != audit_hash
            or manifest.get("cells_sha256") != hashlib.sha256(cells_raw).hexdigest()
            or type(manifest.get("geometry_count")) is not int
            or manifest["geometry_count"] != len(receipt["candidate_seeds"])
            or type(manifest.get("cell_count")) is not int
            or manifest["cell_count"] != len(cells_raw.splitlines())):
        raise SeedAuditError("G0 consumed cells manifest does not bind TRAIN metadata")
    if not cells_raw.splitlines():
        raise SeedAuditError("G0 consumed cells ledger is empty")
    actual: set[tuple[int, int]] = set()
    seen: set[tuple[int, int, str]] = set()
    allocated = {(cell["track_id"], cell["geometry_seed"]) for cell in protocol["cells"]}
    actors = protocol.get("actors")
    if (not isinstance(actors, list) or not actors or
            any(not isinstance(actor, dict) or not isinstance(actor.get("id"), str) for actor in actors)):
        raise SeedAuditError("G0 frozen actor inventory is opaque")
    actor_ids = {actor["id"] for actor in actors}
    if len(actor_ids) != len(actors):
        raise SeedAuditError("G0 frozen actor IDs are duplicated")
    for number, line in enumerate(cells_raw.splitlines(), 1):
        row = _json(line, f"{cells_path}:{number}")
        key = (row.get("track_id"), row.get("geometry_seed"))
        if (type(key[0]) is not int or type(key[1]) is not int or key not in allocated
                or row.get("partition") != "TRAIN"
                or row.get("actor_id") not in actor_ids
                or (*key, row["actor_id"]) in seen):
            raise SeedAuditError(f"{cells_path}:{number}: unallocated or opaque consumed TRAIN cell")
        actual.add(key)
        seen.add((*key, row["actor_id"]))
        _add(ids, [key[1]], cells_path, f"line:{number}.geometry_seed")
    if actual != allocated or len(seen) != len(allocated) * len(actor_ids):
        raise SeedAuditError("G0 consumed TRAIN cell coverage differs from protocol")
    return [{"path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for path, raw in ((expected_claim, claim_raw), (manifest_path, manifest_raw), (cells_path, cells_raw))]


def _r5_chain(root: Path, contents: Mapping[str, bytes],
              training_ledgers: Mapping[str, str], ids: dict[int, list[dict[str, str]]]) -> list[dict[str, Any]]:
    if R5_PROTOCOL not in contents:
        if (root / R5_RUN).exists() or R5_ERRATUM in contents:
            raise SeedAuditError("r5 partial TRAIN run lacks its frozen protocol")
        return []
    if R5_ERRATUM not in contents or hashlib.sha256(contents[R5_PROTOCOL]).hexdigest() != R5_PINS[R5_PROTOCOL]:
        raise SeedAuditError("r5 protocol or narrow historical erratum is missing/drifted")
    receipt_path = f"{R5_RUN}/learner-0-uniform/precheckpoint-abort.json"
    supersession_path = f"{R5_RUN}/protocol-superseded-after-partial.json"
    ledger_path = f"{R5_RUN}/learner-0-uniform/episodes.jsonl"
    metrics_path = f"{R5_RUN}/learner-0-uniform/step-metrics.jsonl"
    trace_path = f"{R5_RUN}/learner-0-uniform/replay-sample-trace-step-000016384.npz"
    if training_ledgers.get(ledger_path) != R5_PINS[ledger_path]:
        raise SeedAuditError("r5 TRAIN ledger must be declared with its independent current SHA-256")
    receipt_raw = _metadata(root, receipt_path, R5_PINS[receipt_path])
    supersession_raw = _metadata(root, supersession_path, R5_PINS[supersession_path])
    ledger_raw = _source(root, ledger_path, R5_PINS[ledger_path], "training_ledger")
    metrics_raw = _metadata(root, metrics_path, R5_PINS[metrics_path])
    trace_raw = _metadata(root, trace_path, R5_PINS[trace_path])
    receipt = _json(receipt_raw, receipt_path)
    supersession = _json(supersession_raw, supersession_path)
    erratum = _json(contents[R5_ERRATUM], R5_ERRATUM)
    invalid = receipt.get("episodes_sha256")
    if (not isinstance(invalid, str) or not re.fullmatch(r"[0-9a-f]{65}", invalid)
            or invalid != "8d192a9cc87ee3777e364f2d49e49ecc7eb66b84eb12099a0ea7c569230889014"
            or receipt.get("format") != "haic-drq-geometry-mix-partial-arm-abort-v1"
            or receipt.get("study_id") != supersession.get("study_id") != "drqv2-geometry-mix-v1-r5"
            or receipt.get("protocol_path") != R5_PROTOCOL
            or receipt.get("protocol_sha256") != R5_PINS[R5_PROTOCOL]
            or receipt.get("run_dir") != ledger_path.rsplit("/", 1)[0]
            or receipt.get("source_seed") != 0 or receipt.get("variant") != "uniform"
            or receipt.get("step_metrics_sha256") != R5_PINS[metrics_path]
            or receipt.get("online_replay_sample_trace_sha256") != R5_PINS[trace_path]
            or receipt.get("online_decisions") != supersession.get("online_decisions") != 16384
            or receipt.get("learner_updates") != supersession.get("learner_updates") != 6384
            or receipt.get("last_episode_id") != 35 or receipt.get("last_episode_step") != 412
            or receipt.get("unique_training_geometry_seeds_observed") != 36
            or any(receipt.get(key) is not False for key in
                   ("full_checkpoint_written", "actor_candidate_written", "resume_allowed", "episode_boundary_at_stop"))
            or receipt.get("evaluation_receipts") != []
            or supersession.get("format") != "haic-drq-geometry-mix-precheckpoint-supersession-v1"
            or supersession.get("status") != "partial-train-attempt-incomplete"
            or supersession.get("superseded_protocol_path") != R5_PROTOCOL
            or supersession.get("superseded_protocol_sha256") != R5_PINS[R5_PROTOCOL]
            or supersession.get("attempt_receipt_path") != receipt_path
            or supersession.get("attempt_receipt_sha256") != R5_PINS[receipt_path]
            or supersession.get("full_checkpoint_written") is not False
            or supersession.get("resume_allowed") is not False
            or supersession.get("diagnostic_or_heldout_access") is not False
            or supersession.get("evaluation_receipts") != []):
        raise SeedAuditError("r5 abort/supersession bytes do not establish the one malformed edge")
    pins = _map(erratum.get("r5"), R5_ERRATUM)
    for field, path in (("attempt_receipt", receipt_path), ("supersession", supersession_path),
                        ("ledger", ledger_path), ("protocol", R5_PROTOCOL),
                        ("step_metrics", metrics_path), ("trace", trace_path)):
        if pins.get(f"{field}_path") != path or pins.get(f"{field}_sha256") != R5_PINS[path]:
            raise SeedAuditError(f"r5 erratum {field} historical pin differs")
    if (set(erratum) != {"format", "candidate_seeds", "candidate_seeds_sha256", "track_id",
                        "r5", "original_ledger_bytes_attested", "epistemic_limit"}
            or set(pins) != {"attempt_receipt_path", "attempt_receipt_sha256",
                             "supersession_path", "supersession_sha256", "ledger_path", "ledger_sha256",
                             "protocol_path", "protocol_sha256", "step_metrics_path", "step_metrics_sha256",
                             "trace_path", "trace_sha256", "invalid_receipt_episodes_sha256",
                             "ordered_reset_count", "end_count", "online_decisions", "learner_updates",
                             "last_episode_id", "last_episode_step", "last_geometry_seed"}
            or erratum.get("format") != "haic-rlpd-g0-r5-receipt-erratum-v1"
            or pins.get("invalid_receipt_episodes_sha256") != invalid
            or erratum.get("original_ledger_bytes_attested") is not False
            or pins.get("online_decisions") != 16384 or pins.get("learner_updates") != 6384
            or pins.get("ordered_reset_count") != 36 or pins.get("end_count") != 35
            or pins.get("last_episode_id") != 35 or pins.get("last_episode_step") != 412
            or pins.get("last_geometry_seed") != receipt.get("last_geometry_seed")):
        raise SeedAuditError("r5 erratum must document only the malformed receipt edge")
    if G0_AUDIT in contents:
        g0 = _json(contents[G0_AUDIT], G0_AUDIT)
        if (erratum.get("candidate_seeds") != g0.get("candidate_seeds")
                or erratum.get("candidate_seeds_sha256") != g0.get("candidate_seeds_sha256")
                or erratum.get("track_id") != g0["cells"][0]["track_id"]
                or not isinstance(erratum.get("epistemic_limit"), str)
                or "does not attest" not in erratum["epistemic_limit"]):
            raise SeedAuditError("r5 erratum is not the G0-bounded historical correction")
    rows = [_json(line, f"{ledger_path}:{n}") for n, line in enumerate(ledger_raw.splitlines(), 1)]
    resets = [row for row in rows if row.get("event") == "reset"]
    ends = [row for row in rows if row.get("event") == "end"]
    if (len(rows) != 71 or len(resets) != 36 or len(ends) != 35
            or [row.get("event") for row in rows] != [
                event for i in range(36) for event in (("reset", "end") if i < 35 else ("reset",))]
            or [row.get("episode_id") for row in resets] != list(range(36))
            or [row.get("episode_id") for row in ends] != list(range(35))
            or len({row.get("seed") for row in resets}) != 36
            or any(reset.get("seed") != reset.get("geometry_seed")
                   or reset.get("seed") != end.get("seed")
                   or reset.get("track_id") != end.get("track_id")
                   for reset, end in zip(resets, ends))
            or resets[-1].get("seed") != receipt.get("last_geometry_seed")
            or type(resets[-1].get("additional_online_step")) is not int
            or resets[-1]["additional_online_step"] + 413 != 16384):
        raise SeedAuditError("r5 partial TRAIN reset/end ledger differs from bounded receipt")
    metrics = metrics_raw.splitlines()
    if len(metrics) != 16384:
        raise SeedAuditError("r5 step metrics decision count differs")
    for index, line in enumerate(metrics, 1):
        row = _json(line, f"{metrics_path}:{index}")
        episode_id = row.get("episode_id")
        if (row.get("additional_online_step") != index
                or type(episode_id) is not int or not 0 <= episode_id < 36
                or row.get("geometry_seed") != resets[episode_id]["seed"]
                or row.get("track_id") != resets[episode_id]["track_id"]
                or row.get("source_seed") != 0 or row.get("variant") != "uniform"):
            raise SeedAuditError(f"r5 metrics/ledger disagree at decision {index}")
    last = _json(metrics[-1], f"{metrics_path}:16384")
    if last.get("episode_id") != 35 or last.get("episode_step") != 412 or last.get("gradient_steps") != 6384:
        raise SeedAuditError("r5 final metrics/abort accounting differs")
    _add(ids, [receipt["last_geometry_seed"]], receipt_path, "last_geometry_seed")
    return [{"path": path, "sha256": R5_PINS[path], "bytes": len(raw)} for path, raw in
            ((receipt_path, receipt_raw), (supersession_path, supersession_raw),
             (ledger_path, ledger_raw), (metrics_path, metrics_raw), (trace_path, trace_raw))]


def _required_protocol(name: str) -> bool:
    stem = name.removeprefix("experiments/").removesuffix(".json")
    return bool(re.fullmatch(
        r"drqv2-teacher-replay-v1(?:-r[23])?|drqv2-geometry-augmentation-v1|"
        r"drqv2-geometry-mix-v1(?:-r[2-6])?|pixel-rlpd-(?:offpolicy-pilot|"
        r"long-horizon-followup|entropy-target-ablation)-v[1-9][0-9]*|"
        r"dreamerv3-b1-(?:world-model-local-v[12]|terminal-balanced-local-v3|"
        r"observation-scale-local-v4|prior-kl-local-v5|overshooting-local-v6|"
        r"overshooting-floor-local-v7|residual-frame-local-v8|"
        r"terminal-positive-weight-local-v9)|rlpd-g0-completion-v1", stem))


def audit_dreamerv3_p1_seeds(
    study_id: str, proposed_cells: Sequence[Mapping[str, int]], *, repo_root: Path,
    protocol_sources: Mapping[str, str], prior_audit_sources: Mapping[str, str],
    training_ledgers: Mapping[str, str], prior_collection_ledgers: Mapping[str, str],
) -> dict[str, Any]:
    """Return bounded current-recorded evidence, not a collector pass certificate."""
    if not isinstance(study_id, str) or not study_id.startswith("dreamerv3-p1-") or not study_id.strip():
        raise SeedAuditError("study_id must name a new dreamerv3-p1- study")
    if (isinstance(proposed_cells, (str, bytes)) or not isinstance(proposed_cells, Sequence)
            or not proposed_cells):
        raise SeedAuditError("proposed TRAIN cells must be explicit and nonempty")
    proposed: list[int] = []
    for index, cell in enumerate(proposed_cells):
        if (not isinstance(cell, Mapping) or set(cell) != {"track_id", "geometry_seed"}
                or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)):
            raise SeedAuditError(f"proposed cell {index}: expected TRAIN track_id 1..4 and geometry_seed")
        proposed.append(_uint32(cell["geometry_seed"], f"proposed cell {index}"))
    if len(proposed) != len(set(proposed)):
        raise SeedAuditError("proposed TRAIN geometry seeds must be unique across all track IDs")
    proposed.sort()

    root = Path(repo_root)
    if root.is_symlink() or any(_HELD_OUT.search(part) for part in root.parts) or not root.is_dir():
        raise SeedAuditError("repo root must be an existing, non-held-out, non-symlink directory")
    declared = dict(zip(KINDS, (protocol_sources, prior_audit_sources,
                                training_ledgers, prior_collection_ledgers)))
    for kind, manifest in declared.items():
        if not isinstance(manifest, Mapping) or not manifest:
            raise SeedAuditError(f"{kind}: explicit PATH=SHA256 manifest required")
        if any(not isinstance(path, str) or not isinstance(digest, str) or not _HASH.fullmatch(digest)
               for path, digest in manifest.items()):
            raise SeedAuditError(f"{kind}: invalid PATH=SHA256 manifest")
        if not KNOWN_INPUTS.get(kind, frozenset()) <= set(manifest):
            raise SeedAuditError(f"{kind}: missing known cross-lane source; this minimum is not exhaustive")
    if len(set().union(*(set(manifest) for manifest in declared.values()))) != sum(map(len, declared.values())):
        raise SeedAuditError("same input declared in multiple source categories")

    catalog, contents = _experiment_catalog(root)
    catalog_paths = set(contents)
    missing = {name for name in catalog_paths if _required_protocol(name)} - set(protocol_sources)
    if missing:
        raise SeedAuditError(f"undeclared current experiment protocol: {sorted(missing)[0]}")
    if G0_AUDIT in contents and G0_AUDIT not in prior_audit_sources:
        raise SeedAuditError(f"undeclared current G0 consumed audit: {G0_AUDIT}")
    if set(protocol_sources) - catalog_paths or set(prior_audit_sources) - catalog_paths:
        raise SeedAuditError("declared experiment metadata missing from current JSON catalog")
    ledgers, collections = _training_inventory(root)
    for kind, discovered, supplied in (("training_ledger", ledgers, training_ledgers),
                                        ("prior_collection_ledger", collections, prior_collection_ledgers)):
        if discovered != set(supplied):
            raise SeedAuditError(f"{kind}: omitted/new TRAIN source inventory: "
                                 f"{sorted(discovered ^ set(supplied))[0]}")

    ids: dict[int, list[dict[str, str]]] = {}
    evidence: list[dict[str, Any]] = []
    for kind in KINDS:
        for path, digest in sorted(declared[kind].items()):
            raw = _source(root, path, digest, kind)
            if kind == "protocol":
                _protocol(_json(raw, path), path, ids, declared)
            elif kind == "prior_seed_audit":
                _prior_audit(_json(raw, path), path, ids)
            else:
                _ledger(raw, path, ids, kind)
            evidence.append({"kind": kind, "path": path, "sha256": digest, "schema_checked": True})

    auxiliary = _g0_metadata(root, contents, ids)
    auxiliary.extend(_r5_chain(root, contents, training_ledgers, ids))
    # The old G0 experiment-content inventory was a pre-G0 snapshot, not an
    # authority for today's catalog. An unknown exact candidate token in any
    # untyped experiment JSON is a stop rather than evidence of non-use.
    token = re.compile(_TOKEN_BOUNDARY.format("|".join(map(str, proposed))))
    typed = set(protocol_sources) | set(prior_audit_sources)
    for path, raw in sorted(contents.items()):
        if path in typed:
            if hashlib.sha256(raw).hexdigest() != declared["protocol"].get(path, declared["prior_seed_audit"].get(path)):
                raise SeedAuditError(f"{path}: experiment changed during catalog audit")
            continue
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if token.search(line):
                raise SeedAuditError(f"{path}:{number}: candidate ID in untyped experiment JSON")

    matches = [{"seed": seed, "sources": ids[seed]} for seed in proposed if seed in ids]
    retired = [row for row in matches if any(source["path"].startswith("experiments/")
                                               for source in row["sources"])]
    return {
        "format": FORMAT, "study_id": study_id, "passed": False,
        "purpose": "Dreamer P1 TRAIN-only geometry seed IDs",
        "inventory_complete": False, "inventory_blockers": [INVENTORY_BLOCKER, R5_BLOCKER],
        "schema_validation": "pass", "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "proposed_seeds": proposed, "proposed_seed_count": len(proposed),
        "proposed_seeds_sha256": hashlib.sha256(json.dumps(proposed, separators=(",", ":")).encode("ascii")).hexdigest(),
        "matched_collisions": matches, "retired_pool_collisions": retired, "parse_errors": [],
        "source_evidence": evidence, "read_paths": [row["path"] for row in evidence],
        "experiment_catalog_sha256": _digest_json(sorted(contents)),
        "experiment_content_inventory": catalog,
        "experiment_content_inventory_sha256": _digest_json(catalog),
        "auxiliary_train_metadata": auxiliary,
        "inventory_scope": "live experiments/*.json names/content; non-held-out runs/**/episodes.jsonl "
                           "and collection.jsonl names/pinned bytes; G0 claims/cells and bounded r5 metadata",
        "read_scope": READ_SCOPE, "freshness_claim": FRESHNESS_LIMITATION,
        "blind_data_access": "none; partition seed IDs are exclusion-only",
        "structural_blind_geometry_comparison": "not performed",
    }


def _manifest(entries: list[str], kind: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise SeedAuditError(f"{kind}: expected PATH=SHA256")
        path, digest = entry.rsplit("=", 1)
        if path in result:
            raise SeedAuditError(f"{kind}: duplicate path: {path}")
        result[path] = digest
    return result


def _cell(value: str) -> dict[str, int]:
    parts = value.split(":")
    if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise SeedAuditError("--cell must be TRACK_ID:GEOMETRY_SEED (decimal integers)")
    return {"track_id": int(parts[0]), "geometry_seed": int(parts[1])}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--cell", action="append", required=True, metavar="TRACK_ID:SEED")
    for flag in ("protocol", "prior-audit", "training-ledger", "prior-collection-ledger"):
        parser.add_argument(f"--{flag}", action="append", required=True, metavar="PATH=SHA256")
    args = parser.parse_args(argv)
    try:
        receipt = audit_dreamerv3_p1_seeds(
            args.study_id, [_cell(value) for value in args.cell], repo_root=args.repo_root,
            protocol_sources=_manifest(args.protocol, "protocol"),
            prior_audit_sources=_manifest(args.prior_audit, "prior_seed_audit"),
            training_ledgers=_manifest(args.training_ledger, "training_ledger"),
            prior_collection_ledgers=_manifest(args.prior_collection_ledger, "prior_collection_ledger"),
        )
    except SeedAuditError as exc:
        print(f"Dreamer P1 seed audit rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 1  # Self-protocol bootstrap and complete typed recorded inventory remain unresolved.


if __name__ == "__main__":
    sys.exit(main())
