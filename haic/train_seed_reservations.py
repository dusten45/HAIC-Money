"""Cross-lane, TRAIN-only geometry-seed claims; not a freshness audit.

Use one shared, pre-existing registry directory for all lanes. Each seed has one
immutable ``seed-<uint32>.json`` claim, regardless of track variant or obstacles.
Callers supply a candidate-specific re-audit which checks historical interaction,
allocations, exposure, exclusions and partial ledgers. The audit runs under the
registry flock and must return the exact candidate cells and explicit empty
collision/blocker/consumed/reserved lists. This module validates that contract but
cannot establish the completeness or truth of an external auditor's sources.

Claims stay reserved: consumption is evidenced by run ledgers, not inferred from
claim status or automatically changed here. Retiring an unobserved claim requires
separate verification of zero use/exposure and an explicit release procedure;
this module provides neither release nor automatic reuse. Existing claim files,
including consumed or retired-unobserved records, always block a new claim.
Batch preflight avoids predictable partial writes. Filesystem failure after the
first O_EXCL creation cannot be rolled back without deleting immutable evidence:
PartialBatchError reports the affected seeds, and even an incomplete file remains
on disk to block subsequent claims until explicitly investigated.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


CLAIM_FORMAT = "haic-train-seed-claim-v1"
AUDIT_FORMAT = "haic-train-seed-audit-v1"
_NAME = re.compile(r"seed-(0|[1-9][0-9]*)\.json\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CELL_KEYS = {"partition", "track_id", "geometry_seed", "obstacles"}
_AUDIT_KEYS = {"format", "partition", "status", "cells", "collisions", "blockers",
               "consumed_seeds", "reserved_seeds", "source"}
_CLAIM_KEYS = {"format", "study_id", "protocol_id", "protocol_path",
               "protocol_sha256", "partition", "track_id", "geometry_seed",
               "obstacles", "status", "claimed_at_utc", "audit_source",
               "audit_digest_sha256"}


class ReservationError(ValueError):
    """Invalid audit, unsafe registry or malformed claim; nothing may be reused."""


class SeedUnavailableError(ReservationError):
    """At least one requested geometry seed already has an immutable claim."""


class PartialBatchError(ReservationError):
    """I/O failed after creating claims; created_seeds may include an incomplete file."""

    def __init__(self, created_seeds: Sequence[int]) -> None:
        self.created_seeds = tuple(created_seeds)
        super().__init__(f"partial TRAIN claim batch; inspect seeds {self.created_seeds}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ReservationError(f"{field}: expected nonempty text")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReservationError(f"{field}: expected lowercase SHA-256")
    return value


def _cell(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CELL_KEYS:
        raise ReservationError("cell: expected partition, track_id, geometry_seed, obstacles")
    if value["partition"] != "TRAIN":
        raise ReservationError("cell: only TRAIN may be claimed")
    for key in ("track_id", "geometry_seed"):
        if type(value[key]) is not int or not 0 <= value[key] < 2**32:
            raise ReservationError(f"cell.{key}: expected uint32")
    if type(value["obstacles"]) is not bool:
        raise ReservationError("cell.obstacles: expected bool")
    return {key: value[key] for key in ("partition", "track_id", "geometry_seed", "obstacles")}


def _claim(value: Any, seed: int) -> None:
    if not isinstance(value, dict) or set(value) != _CLAIM_KEYS or value["format"] != CLAIM_FORMAT:
        raise ReservationError(f"seed-{seed}: unknown claim format/fields")
    if _cell({key: value[key] for key in _CELL_KEYS})["geometry_seed"] != seed:
        raise ReservationError(f"seed-{seed}: filename and record disagree")
    _text(value["study_id"], "study_id")
    if value["protocol_id"] is not None:
        _text(value["protocol_id"], "protocol_id")
    if value["protocol_path"] is not None:
        _text(value["protocol_path"], "protocol_path")
    if value["protocol_sha256"] is not None:
        if value["protocol_path"] is None:
            raise ReservationError("protocol_sha256 requires protocol_path")
        _sha(value["protocol_sha256"], "protocol_sha256")
    if value["status"] not in ("reserved", "consumed", "retired-unobserved"):
        raise ReservationError(f"seed-{seed}: unknown claim status")
    timestamp = _text(value["claimed_at_utc"], "claimed_at_utc")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReservationError(f"seed-{seed}: invalid UTC timestamp") from exc
    if not timestamp.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise ReservationError(f"seed-{seed}: timestamp must be UTC Z")
    _text(value["audit_source"], "audit_source")
    _sha(value["audit_digest_sha256"], "audit_digest_sha256")


def validate_train_claim(value: Any, seed: int) -> None:
    """Validate a claim discovered by a read-only candidate inventory."""
    _claim(value, seed)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReservationError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _existing_claims(directory_fd: int) -> set[int]:
    seeds: set[int] = set()
    for name in sorted(os.listdir(directory_fd)):
        if name == ".gitkeep":
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size != 0:
                raise ReservationError("unsafe registry directory marker")
            continue
        match = _NAME.fullmatch(name)
        if match is None or int(match[1]) >= 2**32:
            raise ReservationError(f"unsafe or unknown registry entry: {name}")
        seed = int(match[1])
        try:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 dir_fd=directory_fd)
        except OSError as exc:
            raise ReservationError(f"{name}: unsafe or unreadable claim") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise ReservationError(f"{name}: not a small regular claim file")
            raw = os.read(descriptor, 65537)
            if len(raw) > 65536:
                raise ReservationError(f"{name}: oversized claim file")
        finally:
            os.close(descriptor)
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                                parse_constant=lambda x: (_ for _ in ()).throw(
                                    ReservationError(f"{name}: invalid JSON constant {x}")))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ReservationError(f"{name}: malformed JSON") from exc
        _claim(record, seed)
        seeds.add(seed)
    return seeds


def _audit_digest(report: Any, cells: list[dict[str, Any]]) -> tuple[str, str]:
    if type(report) is not dict or set(report) != _AUDIT_KEYS or report["format"] != AUDIT_FORMAT:
        raise ReservationError("re-audit: unknown report format/fields")
    if report["partition"] != "TRAIN" or report["status"] != "clear":
        raise ReservationError("re-audit: TRAIN clearance required")
    if not isinstance(report["cells"], list) or [_cell(cell) for cell in report["cells"]] != cells:
        raise ReservationError("re-audit: candidate cells differ")
    for field in ("collisions", "blockers", "consumed_seeds", "reserved_seeds"):
        if type(report[field]) is not list or report[field] != []:
            raise ReservationError(f"re-audit: {field} must be an explicit empty list")
    source = _text(report["source"], "re-audit.source")
    try:
        raw = json.dumps(report, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ReservationError("re-audit: unhashable evidence") from exc
    return hashlib.sha256(raw).hexdigest(), source


def reserve_train_seeds(
    registry_root: Path,
    cells: Sequence[Mapping[str, Any]],
    re_audit: Callable[[tuple[Mapping[str, Any], ...]], Mapping[str, Any]],
    *,
    study_id: str,
    protocol_id: str | None = None,
    protocol_path: str | None = None,
    protocol_sha256: str | None = None,
) -> list[dict[str, Any]]:
    """Claim a complete TRAIN batch after a candidate-specific re-audit under flock.

    The shared registry directory must already exist and not pass through symlinks.
    ``re_audit`` receives immutable candidate cells and must return a dict with
    AUDIT_FORMAT, ``partition='TRAIN'``, ``status='clear'``, the identical ``cells``
    in order, empty ``collisions``, ``blockers``, ``consumed_seeds`` and
    ``reserved_seeds`` lists, and a nonempty ``source`` identifying its evidence.
    The saved digest hashes the entire validated report, not a status string.
    No call to this function by itself certifies external audit completeness.
    """
    prepared = [_cell(cell) for cell in cells]
    if not prepared or len({cell["geometry_seed"] for cell in prepared}) != len(prepared):
        raise ReservationError("cells: require nonempty, distinct geometry seeds across tracks")
    _text(study_id, "study_id")
    if protocol_id is not None:
        _text(protocol_id, "protocol_id")
    if protocol_path is not None:
        _text(protocol_path, "protocol_path")
    if protocol_sha256 is not None:
        if protocol_path is None:
            raise ReservationError("protocol_sha256 requires protocol_path")
        _sha(protocol_sha256, "protocol_sha256")
    if not callable(re_audit):
        raise ReservationError("re_audit: callable required")

    root = Path(registry_root).absolute()
    if any(part.is_symlink() for part in (root, *root.parents)):
        raise ReservationError("registry path: symlink is unsafe")
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        requested = {cell["geometry_seed"] for cell in prepared}
        if requested & _existing_claims(directory_fd):
            raise SeedUnavailableError("geometry seed already claimed across tracks")

        report = re_audit(tuple(MappingProxyType(cell.copy()) for cell in prepared))
        audit_digest, audit_source = _audit_digest(report, prepared)
        if requested & _existing_claims(directory_fd):
            raise SeedUnavailableError("geometry seed claimed during re-audit")
        directory_stat = os.stat(root, follow_symlinks=False)
        locked_stat = os.fstat(directory_fd)
        if (directory_stat.st_dev, directory_stat.st_ino) != (locked_stat.st_dev, locked_stat.st_ino):
            raise ReservationError("registry directory changed during re-audit")

        timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        claims = [{"format": CLAIM_FORMAT, "study_id": study_id,
                   "protocol_id": protocol_id, "protocol_path": protocol_path,
                   "protocol_sha256": protocol_sha256, **cell, "status": "reserved",
                   "claimed_at_utc": timestamp, "audit_source": audit_source,
                   "audit_digest_sha256": audit_digest} for cell in prepared]
        payloads = [json.dumps(claim, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
                    for claim in claims]
        created: list[int] = []
        try:
            for claim, raw in zip(claims, payloads):
                seed = claim["geometry_seed"]
                fd = os.open(f"seed-{seed}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory_fd)
                created.append(seed)
                try:
                    view = memoryview(raw)
                    while view:
                        written = os.write(fd, view)
                        if not written:
                            raise OSError("short claim write")
                        view = view[written:]
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(directory_fd)
        except OSError as exc:
            if created:
                raise PartialBatchError(created) from exc
            raise ReservationError("claim creation failed without writing a claim") from exc
        return claims
    finally:
        os.close(directory_fd)
