"""Audit training-only official-generator seed IDs without opening held-out data.

Every source is named and SHA-pinned by the caller. This module never discovers
files, follows protocol pointers, reads geometry, or opens evaluator artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


UINT32_MAX = (1 << 32) - 1
FRESHNESS_LIMITATION = "no known recorded overlap; historical pilot schedules are incomplete"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_PART = re.compile(r"blind|confirm|screen|eval|held.?out|holdout|submission", re.I)
_SEED_LIST_KEYS = frozenset({
    "reserved_training_seeds", "known_excluded_geometry_seeds",
    "training_geometry_seeds", "reused_development_geometry_seeds",
})


class SeedAuditError(ValueError):
    """Unpinned, inaccessible, unsafe, or uninterpretable seed evidence."""


class SeedCollisionError(SeedAuditError):
    """The audited seed batch intersects frozen exclusions or a source ledger."""

    def __init__(self, report: dict[str, Any]) -> None:
        super().__init__(f"{len(report['matched_collisions'])} proposed geometry seed(s) collide")
        self.report = report


def _uint32(value: Any, context: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise SeedAuditError(f"{context}: expected a uint32 seed ID")
    return value


def _seed_list(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise SeedAuditError(f"{context}: expected a nonempty list of uint32 seed IDs")
    seeds = [_uint32(seed, context) for seed in value]
    if len(seeds) != len(set(seeds)):
        raise SeedAuditError(f"{context}: duplicate seed ID")
    return seeds


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SeedAuditError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_json(raw: bytes, context: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SeedAuditError(f"{context}: invalid JSON constant {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SeedAuditError(f"{context}: malformed UTF-8 JSON") from exc


def _sources(
    root: Path, declarations: Mapping[str, str], kind: str,
) -> list[tuple[str, Path, str]]:
    if not isinstance(declarations, Mapping) or not declarations:
        raise SeedAuditError(f"{kind}: at least one SHA-pinned source is required")
    result: list[tuple[str, Path, str]] = []
    for name, expected in declarations.items():
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            raise SeedAuditError(f"{kind}: invalid source path")
        relative = PurePosixPath(name)
        parts = relative.parts
        if (relative.is_absolute() or name != relative.as_posix()
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise SeedAuditError(f"{kind}: paths must be relative and normalized: {name}")
        if kind in ("protocol", "prior_seed_audit"):
            allowed = len(parts) == 2 and parts[0] == "experiments" and parts[1].endswith(".json")
            is_prior = parts[-1].endswith("-geometry-audit.json")
            allowed &= is_prior if kind == "prior_seed_audit" else not is_prior
            allowed &= not re.search(r"result|diagnostic|outcome|road", parts[-1], re.I)
        else:
            allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "episodes.jsonl"
            allowed &= not any(_FORBIDDEN_PART.search(part) for part in parts[1:-1])
        if not allowed or not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise SeedAuditError(f"{kind}: path or SHA outside allowlist: {name}")
        path = root.joinpath(*parts)
        # Refuse symlinks rather than resolving into a sealed directory.
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise SeedAuditError(f"{kind}: symlink source is forbidden: {name}")
        if not path.is_file():
            raise SeedAuditError(f"{kind}: missing source: {name}")
        result.append((name, path, expected))
    return sorted(result)


def _add(ids: dict[int, list[dict[str, str]]], seeds: list[int], path: str, field: str) -> None:
    for seed in seeds:
        ids.setdefault(seed, []).append({"path": path, "field": field})


def _protocol_ids(record: Any, path: str, ids: dict[int, list[dict[str, str]]]) -> int:
    if not isinstance(record, dict):
        raise SeedAuditError(f"{path}: opaque protocol schema")
    for key in record:
        if (key.endswith("_seeds") and key not in _SEED_LIST_KEYS | {"learner_seeds"}
                or re.search(r"blind.*(?:outcome|road|geometry|path)|road|outcome", key, re.I)):
            raise SeedAuditError(f"{path}: opaque or unsafe protocol field {key}")
    if "learner_seeds" in record:
        learners = record["learner_seeds"]
        if (not isinstance(learners, list) or not learners
                or any(type(seed) is not int or seed < 0 for seed in learners)):
            raise SeedAuditError(f"{path}: invalid non-geometry learner RNG seeds")
    recognized = 0
    for key in sorted(_SEED_LIST_KEYS):
        if key in record:
            _add(ids, _seed_list(record[key], f"{path}.{key}"), path, key)
            recognized += 1
    partitions = record.get("partitions")
    if partitions is not None:
        if not isinstance(partitions, dict) or not partitions:
            raise SeedAuditError(f"{path}.partitions: opaque schema")
        for name, item in partitions.items():
            if not isinstance(item, dict) or set(item) - {"seeds", "track_ids", "repeats"}:
                raise SeedAuditError(f"{path}.partitions.{name}: opaque schema")
            _add(ids, _seed_list(item.get("seeds"), f"{path}.partitions.{name}.seeds"),
                 path, f"partitions.{name}.seeds")
            recognized += 1
    pools = record.get("training_pools")
    if pools is not None:
        if not isinstance(pools, dict) or not pools:
            raise SeedAuditError(f"{path}.training_pools: opaque schema")
        for name, item in pools.items():
            if not isinstance(item, dict) or "seed_range" in item:
                raise SeedAuditError(f"{path}.training_pools.{name}: opaque schema")
            _add(ids, _seed_list(item.get("seeds"), f"{path}.training_pools.{name}.seeds"),
                 path, f"training_pools.{name}.seeds")
            recognized += 1
    audit = record.get("geometry_audit")
    if audit is not None:
        if not isinstance(audit, dict):
            raise SeedAuditError(f"{path}.geometry_audit: opaque schema")
        if "candidate_seeds" not in audit and "report_path" in audit:
            raise SeedAuditError(f"{path}.geometry_audit: missing candidate seed IDs")
        if "candidate_seeds" in audit:
            _add(ids, _seed_list(audit["candidate_seeds"], f"{path}.geometry_audit.candidate_seeds"),
                 path, "geometry_audit.candidate_seeds")
            recognized += 1
    actors = record.get("source_actors")
    if actors is not None:
        if not isinstance(actors, list):
            raise SeedAuditError(f"{path}.source_actors: opaque schema")
        for index, actor in enumerate(actors):
            if not isinstance(actor, dict) or "training_geometry_seeds" not in actor:
                raise SeedAuditError(f"{path}.source_actors[{index}]: missing training seeds")
            if any(key.endswith("_seeds") and key != "training_geometry_seeds" for key in actor):
                raise SeedAuditError(f"{path}.source_actors[{index}]: opaque seed schema")
            _add(ids, _seed_list(actor["training_geometry_seeds"],
                                 f"{path}.source_actors[{index}].training_geometry_seeds"),
                 path, f"source_actors[{index}].training_geometry_seeds")
            recognized += 1
    if not recognized:
        raise SeedAuditError(f"{path}: opaque protocol schema; no partition/exclusion seeds")
    return recognized


def _prior_audit_ids(record: Any, path: str, ids: dict[int, list[dict[str, str]]]) -> int:
    if not isinstance(record, dict) or record.get("passed") is not True or record.get("parse_errors") != []:
        raise SeedAuditError(f"{path}: prior seed audit is not frozen and successful")
    if record.get("global_freshness_claim") != FRESHNESS_LIMITATION:
        raise SeedAuditError(f"{path}: missing historical-coverage limitation")
    seeds = _seed_list(record.get("known_excluded_geometry_seeds"),
                       f"{path}.known_excluded_geometry_seeds")
    if record.get("known_excluded_geometry_seed_count") != len(seeds):
        raise SeedAuditError(f"{path}: prior exclusion count disagrees with seed IDs")
    if (type(record.get("source_snapshot_count")) is not int
            or record["source_snapshot_count"] < 1
            or not isinstance(record.get("source_snapshots"), list)
            or len(record["source_snapshots"]) != record["source_snapshot_count"]):
        raise SeedAuditError(f"{path}: prior audit source snapshot inventory is incomplete")
    candidates = _seed_list(record.get("candidate_seeds"), f"{path}.candidate_seeds")
    _add(ids, seeds, path, "known_excluded_geometry_seeds")
    _add(ids, candidates, path, "candidate_seeds")
    return len(set(seeds) | set(candidates))


def _ledger_ids(raw: bytes, path: str, ids: dict[int, list[dict[str, str]]]) -> int:
    found: set[int] = set()
    if not raw.strip():
        raise SeedAuditError(f"{path}: empty training episode ledger")
    for number, line in enumerate(raw.splitlines(), 1):
        row = _parse_json(line, f"{path}:{number}")
        if not isinstance(row, dict) or ("seed" in row) == ("geometry_seed" in row):
            raise SeedAuditError(f"{path}:{number}: opaque training episode seed schema")
        if type(row.get("track_id")) is not int or row["track_id"] < 0:
            raise SeedAuditError(f"{path}:{number}: missing/invalid track ID")
        if "event" in row and row["event"] not in ("reset", "end"):
            raise SeedAuditError(f"{path}:{number}: opaque training episode event")
        seed = _uint32(row["seed"] if "seed" in row else row["geometry_seed"], f"{path}:{number}")
        if seed not in found:
            _add(ids, [seed], path, "episodes.jsonl.seed")
            found.add(seed)
    return len(found)


def audit_training_seeds(
    proposed_seeds: Sequence[int], *, repo_root: Path,
    protocol_sources: Mapping[str, str], training_ledgers: Mapping[str, str],
    prior_audit_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Return an auditable pass receipt; raise on invalid inputs or any exact match.

    Sources are relative path -> expected SHA-256 mappings, not directory roots.
    A collision raises SeedCollisionError with the complete failed ``.report``.
    Historical unrecorded use and structural road similarity are not certified.
    """
    if isinstance(proposed_seeds, (str, bytes)) or not isinstance(proposed_seeds, Sequence):
        raise SeedAuditError("proposed seeds must be a sequence of uint32 IDs")
    proposed = [_uint32(seed, "proposed seeds") for seed in proposed_seeds]
    if len(proposed) < 100 or len(proposed) != len(set(proposed)):
        raise SeedAuditError("at least 100 unique proposed training seeds are required")
    root = Path(repo_root).resolve()
    if any(_FORBIDDEN_PART.search(part) for part in root.parts):
        raise SeedAuditError("repo root cannot be inside a held-out or submission area")
    protocol = _sources(root, protocol_sources, "protocol")
    prior = _sources(root, prior_audit_sources, "prior_seed_audit")
    ledgers = _sources(root, training_ledgers, "training_ledger")
    paths = [name for name, _, _ in protocol + prior + ledgers]
    if len(paths) != len(set(paths)):
        raise SeedAuditError("the same source was declared more than once")

    excluded: dict[int, list[dict[str, str]]] = {}
    source_evidence = []
    for kind, sources in (("protocol", protocol), ("prior_seed_audit", prior),
                          ("training_ledger", ledgers)):
        for name, path, expected in sources:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise SeedAuditError(f"{name}: unable to read declared source") from exc
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                raise SeedAuditError(f"{name}: source SHA-256 mismatch")
            if kind == "training_ledger":
                count = _ledger_ids(raw, name, excluded)
            else:
                record = _parse_json(raw, name)
                if kind == "protocol":
                    audit = record.get("geometry_audit") if isinstance(record, dict) else None
                    if isinstance(audit, dict) and ("report_path" in audit or "report_sha256" in audit):
                        if (audit.get("report_path") not in prior_audit_sources
                                or prior_audit_sources[audit["report_path"]] != audit.get("report_sha256")):
                            raise SeedAuditError(f"{name}: referenced prior seed audit must be declared and SHA-pinned")
                    actors = record.get("source_actors") if isinstance(record, dict) else None
                    if isinstance(actors, list):
                        for actor in actors:
                            if isinstance(actor, dict) and ("episodes_path" in actor or "episodes_sha256" in actor):
                                if (actor.get("episodes_path") not in training_ledgers
                                        or training_ledgers[actor["episodes_path"]] != actor.get("episodes_sha256")):
                                    raise SeedAuditError(f"{name}: source actor ledger must be declared and SHA-pinned")
                count = (_protocol_ids(record, name, excluded) if kind == "protocol"
                         else _prior_audit_ids(record, name, excluded))
            source_evidence.append({"kind": kind, "path": name, "sha256": actual,
                                    "seed_count_or_fields": count})

    collisions = [{"seed": seed, "sources": excluded[seed]} for seed in proposed if seed in excluded]
    report = {
        "format": "haic-drq-training-seed-audit-v1",
        "passed": not collisions,
        "purpose": "training-only official-generator geometry seed IDs",
        "proposed_seed_count": len(proposed),
        "proposed_seeds": proposed,
        "proposed_seeds_sha256": hashlib.sha256(
            json.dumps(proposed, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "known_excluded_seed_count": len(excluded),
        "matched_collisions": collisions,
        "source_evidence": source_evidence,
        "read_paths": [row["path"] for row in source_evidence],
        "read_scope": "only explicitly SHA-pinned experiments protocol/audit JSON and runs training episodes.jsonl",
        "freshness_claim": FRESHNESS_LIMITATION,
        "blind_data_access": "none; protocol blind seed IDs are exclusion-only",
        "structural_blind_geometry_comparison": "not performed",
    }
    if collisions:
        raise SeedCollisionError(report)
    return report


def _declarations(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for entry in values:
        if "=" not in entry:
            raise SeedAuditError(f"{label}: expected PATH=SHA256")
        path, digest = entry.rsplit("=", 1)
        if path in result:
            raise SeedAuditError(f"{label}: duplicate path {path}")
        result[path] = digest
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seeds", nargs="+", type=int)
    seeds.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int, help="number of consecutive seeds from --seed-start")
    parser.add_argument("--protocol", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("--prior-audit", action="append", default=[], metavar="PATH=SHA256")
    parser.add_argument("--training-ledger", action="append", default=[], metavar="PATH=SHA256")
    args = parser.parse_args(argv)
    try:
        if (args.seed_start is None) != (args.seed_count is None):
            raise SeedAuditError("--seed-start and --seed-count must be supplied together")
        if args.seed_count is not None:
            _uint32(args.seed_start, "--seed-start")
            if not 100 <= args.seed_count <= 100_000 or args.seed_start + args.seed_count > UINT32_MAX + 1:
                raise SeedAuditError("--seed-count must be 100..100000 and stay within uint32")
        proposed = (args.seeds if args.seeds is not None
                    else list(range(args.seed_start, args.seed_start + args.seed_count)))
        report = audit_training_seeds(
            proposed, repo_root=args.repo_root,
            protocol_sources=_declarations(args.protocol, "--protocol"),
            prior_audit_sources=_declarations(args.prior_audit, "--prior-audit"),
            training_ledgers=_declarations(args.training_ledger, "--training-ledger"),
        )
    except SeedCollisionError as exc:
        print(json.dumps(exc.report, sort_keys=True, indent=2))
        return 1
    except SeedAuditError as exc:
        print(f"seed audit rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
