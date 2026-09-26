"""Audit recorded geometry use before freezing teacher-replay partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOTS = ("docs", "experiments", "evaluations", "runs", "submissions", "talk")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".py", ".txt", ".yaml", ".yml"}
EXCLUDED_PATH_PARTS = {".git", "__pycache__", ".pytest_cache"}
GEOMETRY_LIST_KEYS = {
    "geometry_seeds",
    "reserved_training_seeds",
    "excluded_seeds",
    "reset_seeds",
    "development_seeds",
    "evaluation_seeds",
    "candidate_seeds",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer_values(value: Any) -> set[int]:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**32:
        return {value}
    if isinstance(value, list):
        result: set[int] = set()
        for item in value:
            result |= _integer_values(item)
        return result
    return set()


def _collect_geometry_seeds(value: Any, *, track_context: bool = False) -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, list):
        for item in value:
            seeds |= _collect_geometry_seeds(item, track_context=track_context)
        return seeds
    if not isinstance(value, dict):
        return seeds

    local_track_context = track_context or "track_id" in value or "track_ids" in value
    event = value.get("event")
    for key, item in value.items():
        if key in GEOMETRY_LIST_KEYS or key == "seeds":
            seeds |= _integer_values(item)
        elif key in {"geometry_seed", "reset_seed"}:
            seeds |= _integer_values(item)
        elif key == "seed" and (local_track_context or event in {"reset", "end"}):
            seeds |= _integer_values(item)
        elif isinstance(item, (dict, list)):
            seeds |= _collect_geometry_seeds(item, track_context=local_track_context)
    return seeds


def _source_files(repo_root: Path, exclude: set[str]) -> list[Path]:
    files: set[Path] = set()
    for root_name in ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part in EXCLUDED_PATH_PARTS for part in path.parts):
                continue
            relative = path.relative_to(repo_root).as_posix()
            if relative in exclude:
                continue
            files.add(path)
    return sorted(files, key=lambda path: path.as_posix())


def _read_json_records(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".jsonl":
        return [json.loads(text)]
    records = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def audit_geometry(
    repo_root: Path,
    candidate_seeds: list[int],
    *,
    exclude_paths: set[str] | None = None,
) -> dict[str, Any]:
    if not candidate_seeds or len(candidate_seeds) != len(set(candidate_seeds)):
        raise ValueError("candidate seeds must be a non-empty unique list")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in candidate_seeds):
        raise ValueError("candidate seeds must be exact uint32 integers")
    repo_root = repo_root.resolve()
    sources = _source_files(repo_root, exclude_paths or set())
    known_geometry_seeds: set[int] = set()
    snapshots = []
    parse_errors = []
    candidate_hits: dict[int, list[str]] = {seed: [] for seed in candidate_seeds}

    patterns = {
        seed: re.compile(rf"(?<![A-Za-z0-9.]){seed}(?![A-Za-z0-9.])")
        for seed in candidate_seeds
    }
    for path in sources:
        relative = path.relative_to(repo_root).as_posix()
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        snapshots.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
        if path.suffix.lower() in {".json", ".jsonl"}:
            try:
                for record in _read_json_records(path):
                    known_geometry_seeds |= _collect_geometry_seeds(record)
            except (OSError, json.JSONDecodeError, ValueError) as error:
                parse_errors.append(str(error))
        for seed, pattern in patterns.items():
            if pattern.search(text):
                candidate_hits[seed].append(relative)

    return {
        "schema_version": 1,
        "scope_roots": list(ROOTS),
        "method": (
            "Recursively hash UTF-8 JSON/JSONL/text evidence in the frozen scope; "
            "extract geometry, partition, exclusion and recorded episode reset seeds; "
            "search exact candidate tokens with alphanumeric/decimal boundaries."
        ),
        "global_freshness_claim": "no known recorded overlap; historical pilot schedules are incomplete",
        "candidate_seeds": candidate_seeds,
        "candidate_hits": {str(seed): paths for seed, paths in candidate_hits.items()},
        "known_excluded_geometry_seeds": sorted(known_geometry_seeds),
        "known_excluded_geometry_seed_count": len(known_geometry_seeds),
        "source_snapshots": snapshots,
        "source_snapshot_count": len(snapshots),
        "parse_errors": parse_errors,
        "passed": not parse_errors and not any(candidate_hits.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-seeds", required=True,
                        help="comma-separated proposed geometry seeds, audited across all track IDs")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--exclude-path", action="append", default=[],
                        help="repository-relative generated output to omit from the prior-use scan")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        seeds = [int(value) for value in args.candidate_seeds.split(",") if value]
        report = audit_geometry(args.repo_root, seeds, exclude_paths=set(args.exclude_path))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "source_snapshots"},
                     indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
