"""Read-only, candidate-scoped TD-MPC2 TRAIN-DIAGNOSTIC allocation inventory.

Input is an existing experiments JSON with format
``haic-tdmpc2-train-diagnostic-candidates-v1``, purpose
``TRAIN-DIAGNOSTIC`` and exactly 24 explicit TRAIN cells (six on each track 1-4,
distinct geometry_seed values, obstacles=true). No IDs are generated. A report
is NOT an allocation, an under-lock re-audit, or permission to reset. In
particular, incomplete historical exposure coverage deliberately blocks even
when no recorded intersection is found. Protected outcomes are never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from haic.train_seed_reservations import ReservationError, validate_train_claim
from scripts.audit_rlpd_g1_coverage_seeds import (
    CATALOG, CATALOG_PROTOCOL, G0_AUDIT, G0_CELLS, G0_CLAIMS, G0_PROTOCOL,
    TRAIN_CLAIMS, InventoryError, _Inventory, _directory_scope, _digest,
    _json, _mentions_candidate_road, _sha, _uint,
)

FORMAT = "haic-tdmpc2-train-diagnostic-candidates-v1"
TD_JOURNALS = (
    "runs/tdmpc2-reused-train-20260927-v1/training.jsonl",
    "runs/tdmpc2-reused-train-20260927-v2/training.jsonl",
    "runs/tdmpc2-long-20260928-v1/training.jsonl",
    "runs/tdmpc2-long-20260928-v2/training.jsonl",
)
_SKIP_EXPERIMENT = re.compile(
    r"(?:^|[-_])(?:result|results|score|evaluation|confirmation|blind|screen|"
    r"analysis|summary|failure|preflight|decision|receipt|exposure)(?:[-_.]|$)", re.I,
)
_RUN_FILES = {"episodes.jsonl", "collection.jsonl", "training.jsonl", "cells.jsonl",
              "run-config.json", "run_config.json", "precheckpoint-abort.json",
              "collection-result.json"}


def _cells(inv: _Inventory, candidate_path: str) -> tuple[list[dict[str, Any]], str]:
    parts = PurePosixPath(candidate_path).parts
    if (len(parts) != 2 or parts[0] != "experiments" or not parts[1].endswith(".json")
            or "\\" in candidate_path or _directory_scope(parts[1]) != "train"):
        raise InventoryError("candidate-path: require a non-protected experiments/*.json file")
    raw = inv.read(candidate_path)
    obj = _json(raw, candidate_path)
    if set(obj) != {"format", "purpose", "cells"} or obj["format"] != FORMAT or obj["purpose"] != "TRAIN-DIAGNOSTIC":
        raise InventoryError("candidate input: unknown format, purpose or extra fields")
    rows = obj["cells"]
    if not isinstance(rows, list) or len(rows) != 24:
        raise InventoryError("candidate input: require exactly 24 explicitly declared cells")
    tracks: Counter[int] = Counter()
    seeds: set[int] = set()
    for index, cell in enumerate(rows):
        if not isinstance(cell, dict) or set(cell) != {"partition", "track_id", "geometry_seed", "obstacles"}:
            raise InventoryError(f"candidate cells[{index}]: unknown cell fields")
        track = cell["track_id"]
        seed = _uint(cell["geometry_seed"], f"candidate cells[{index}].geometry_seed")
        if (type(track) is not int or track not in (1, 2, 3, 4)
                or cell["partition"] != "TRAIN" or cell["obstacles"] is not True):
            raise InventoryError(f"candidate cells[{index}]: require obstacle-enabled TRAIN track 1-4")
        if seed in seeds:
            raise InventoryError(f"candidate cells[{index}]: cross-track geometry alias {seed}")
        tracks[track] += 1
        seeds.add(seed)
    if any(tracks[track] != 6 for track in (1, 2, 3, 4)):
        raise InventoryError("candidate input: require six independent geometry IDs per track")
    return rows, _sha(raw)


def _relevant(value: Any, candidates: set[int], context: str = "") -> bool:
    """Conservatively identify unknown road aliases and overlapping seed ranges."""
    if isinstance(value, dict):
        keys = {str(key).casefold(): key for key in value}
        road = any(word in context for word in ("seed", "road", "cell", "geometry"))
        if road and "start" in keys and ("end" in keys or "count" in keys):
            start = value[keys["start"]]
            end = value[keys["end"]] if "end" in keys else None
            count = value[keys["count"]] if "count" in keys else None
            if type(start) is not int or (end is not None and type(end) is not int) or (count is not None and type(count) is not int):
                return True  # Unknown bounds cannot prove disjointness.
            if end is not None and any(min(start, end) <= seed <= max(start, end) for seed in candidates):
                return True  # End semantics may be inclusive.
            if count is not None and (count <= 0 or any(start <= seed < start + count for seed in candidates)):
                return True
        for key, item in value.items():
            name = str(key).casefold()
            if name.endswith(("seed_start", "geometry_seed_start", "road_start")):
                prefix = name.removesuffix("start")
                count = value.get(prefix + "count")
                end = value.get(prefix + "end")
                if count is not None or end is not None:
                    if _relevant({"start": item, "count": count} if count is not None else
                                 {"start": item, "end": end}, candidates, name):
                        return True
            if _relevant(item, candidates, f"{context}.{name}"):
                return True
        return False
    if isinstance(value, list):
        if "range" in context and any(word in context for word in ("seed", "road", "cell", "geometry")):
            if len(value) != 2 or any(type(item) is not int for item in value):
                return True  # An unreviewed range encoding has unknown bounds.
            if any(min(value) <= seed <= max(value) for seed in candidates):
                return True
        return any(_relevant(item, candidates, context) for item in value)
    return (any(word in context for word in ("seed", "road", "cell", "geometry"))
            and ((type(value) is int and value in candidates)
                 or (isinstance(value, str) and value.isdecimal() and int(value) in candidates)))


def _source_relevant(raw: bytes, path: str, candidates: set[int], token: re.Pattern[str]) -> bool:
    if _mentions_candidate_road(raw, token, candidates):
        return True
    try:
        return _relevant(_json(raw, path), candidates)
    except InventoryError:
        # JSONL and damaged files are covered by the line-aware token check.
        return False


def _walk_train(root: Path) -> list[str]:
    run_root = root / "runs"
    if run_root.is_symlink() or not run_root.is_dir():
        raise InventoryError("runs: missing or symlinked directory")
    paths: list[str] = []

    def onerror(exc: OSError) -> None:
        raise InventoryError(f"runs: discovery failed: {exc}") from exc

    for current, dirs, files in os.walk(run_root, followlinks=False, onerror=onerror):
        location = Path(current)
        for name in list(dirs):
            child = location / name
            if child.is_symlink():
                raise InventoryError(f"{child.relative_to(root)}: unsafe symlink")
            scope = _directory_scope(name)
            if scope == "ambiguous":
                raise InventoryError(f"{child.relative_to(root)}: ambiguous protected/TRAIN directory")
            if scope == "protected":
                dirs.remove(name)  # Never enumerate or open protected episode/outcome files.
        for name in files:
            if name not in _RUN_FILES:
                continue
            path = location / name
            if path.is_symlink():
                raise InventoryError(f"{path.relative_to(root)}: unsafe symlink")
            paths.append(path.relative_to(root).as_posix())
    return sorted(paths)


def _td_ledger(inv: _Inventory, path: str) -> None:
    """Retain reset intent and open partial identity; never infer release at EOF."""
    rows = inv.read(path).splitlines()
    if not rows or not inv.raw[path].endswith(b"\n"):
        raise InventoryError(f"{path}: empty or torn TD journal")
    phase = "start"
    active: tuple[int, int] | None = None
    for index, line in enumerate(rows, 1):
        row = _json(line, f"{path}:{index}")
        event = row.get("event")
        if any(("seed" in key or "road_id" in key or "geometry" in key)
               and key not in ("geometry_seed", "seed_schedule") for key in row):
            raise InventoryError(f"{path}:{index}: unknown TD road identity field")
        if index == 1:
            if event != "start" or not isinstance(row.get("protocol_sha256"), str):
                raise InventoryError(f"{path}: TD journal has no source-bound start")
            continue
        if event in ("reset_intent", "reset", "episode"):
            track = row.get("track_id")
            if type(track) is not int or track not in (1, 2, 3, 4):
                raise InventoryError(f"{path}:{index}: missing TRAIN track identity")
            seed = inv.record(row.get("geometry_seed"), path, f"line:{index}.{event}.geometry_seed")
            cell = (track, seed)
            if (event == "reset_intent" and phase not in ("start", "ended")
                    or event == "reset" and (phase != "intent" or active != cell)
                    or event == "episode" and (phase != "reset" or active != cell)):
                raise InventoryError(f"{path}:{index}: TD reset/episode identity mismatch")
            phase = {"reset_intent": "intent", "reset": "reset", "episode": "ended"}[event]
            active = cell
        elif event == "partial":
            if phase not in ("intent", "reset", "ended"):
                raise InventoryError(f"{path}:{index}: orphan partial record")
            phase = "partial"
        elif event not in ("step", "updates", "metric", "diagnostic", "checkpoint"):
            raise InventoryError(f"{path}:{index}: unknown TD event {event!r}")
        if "geometry_seed" in row and event not in ("reset_intent", "reset", "episode"):
            inv.record(row["geometry_seed"], path, f"line:{index}.{event}.geometry_seed")
    inv.typed_paths.add(path)
    # A live journal, failed reset-intent, and mid-episode partial are all exposure.
    # Their recorded road remains in inv.ids, even if no successful reset followed.


def audit_tdmpc2_train_diag_seeds(
    candidate_path: str, *, repo_root: Path = Path("."),
    expected_source_inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Inventory explicitly declared cells; always block until coverage is reviewed."""
    root = Path(repo_root).resolve()
    inv = _Inventory(root)
    cells, candidate_sha = _cells(inv, candidate_path)
    candidates = {cell["geometry_seed"] for cell in cells}
    if (expected_source_inventory_sha256 is not None
            and not re.fullmatch(r"[0-9a-f]{64}", expected_source_inventory_sha256)):
        raise InventoryError("expected-source-inventory-sha256: expected lowercase SHA-256")
    token = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(map(str, sorted(candidates))) + r")(?![A-Za-z0-9_.])")
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def issue(path: str, field: str, reason: str, *, mandatory: bool = False) -> None:
        entry = {"path": path, "field": field, "reason": reason}
        raw = inv.raw.get(path, b"")
        if (mandatory or any(s["path"] == path for seed in candidates for s in inv.ids.get(seed, []))
                or _source_relevant(raw, path, candidates, token)):
            issues.append(entry)
        else:
            warnings.append(entry)

    # Protocols and seed-audit files supply protected ID exclusions, NOT results.
    experiments = root / "experiments"
    try:
        if experiments.is_symlink() or not experiments.is_dir():
            raise InventoryError("experiments: missing or unsafe directory")
        for entry in sorted(experiments.iterdir()):
            if entry.suffix != ".json" or entry.name == PurePosixPath(candidate_path).name:
                continue
            if entry.is_symlink():
                raise InventoryError(f"{entry.relative_to(root)}: unsafe experiment symlink")
            if _SKIP_EXPERIMENT.search(entry.name):
                continue  # Never open protected or outcome-bearing result artifacts.
            path = entry.relative_to(root).as_posix()
            try:
                data = _json(inv.read(path), path)
                inv.fields(data, path)
                inv.typed_paths.add(path)
            except InventoryError as exc:
                issue(path, "protocol/ID metadata", str(exc))
    except (InventoryError, OSError) as exc:
        issues.append({"path": "experiments", "field": "discovery", "reason": str(exc)})

    # Verify the full DrQ reservation, not only its 120 selected TRAIN roads.
    try:
        protocol = _json(inv.read(CATALOG_PROTOCOL), CATALOG_PROTOCOL)
        catalog = _json(inv.read(CATALOG), CATALOG)
        reserved = list(range(3910800001, 3910800513))
        if (protocol.get("candidate_seeds") != reserved
                or catalog.get("protocol_sha256") != _sha(inv.raw[CATALOG_PROTOCOL])
                or catalog.get("seed_audit", {}).get("proposed_seeds") != reserved):
            raise InventoryError("DrQ 512-ID reservation/protocol/catalog mismatch")
        for group, count in (("train", 120), ("train_diagnostic", 16)):
            rows = catalog.get(group)
            if not isinstance(rows, list) or len(rows) != count:
                raise InventoryError(f"DrQ catalog {group}: missing selected roads")
            selected = [row.get("geometry_seed") if isinstance(row, dict) else None for row in rows]
            if len(set(selected)) != count or not set(selected) <= set(reserved):
                raise InventoryError(f"DrQ catalog {group}: IDs outside reservation")
        for seed in reserved:
            inv.record(seed, CATALOG_PROTOCOL, "candidate_seeds (512-ID reservation)")
    except (InventoryError, TypeError, AttributeError) as exc:
        issue(CATALOG_PROTOCOL, "catalog reservation", str(exc), mandatory=True)

    for path in (G0_PROTOCOL, G0_AUDIT):
        try:
            obj = _json(inv.read(path), path)
            inv.fields(obj, path)
        except InventoryError as exc:
            issue(path, "G0 allocation", str(exc), mandatory=True)
    try:
        for index, line in enumerate(inv.read(G0_CELLS).splitlines(), 1):
            obj = _json(line, f"{G0_CELLS}:{index}")
            inv.record(obj.get("geometry_seed"), G0_CELLS, f"line:{index}.geometry_seed")
        inv.typed_paths.add(G0_CELLS)
    except InventoryError as exc:
        issue(G0_CELLS, "G0 partial/complete cells", str(exc), mandatory=True)

    for directory in (G0_CLAIMS, TRAIN_CLAIMS):
        try:
            folder = root / directory
            if folder.is_symlink() or not folder.is_dir():
                raise InventoryError(f"{directory}: missing or unsafe claims")
            for entry in sorted(folder.iterdir()):
                if entry.name in ((".gitkeep",) if directory == TRAIN_CLAIMS else (".lock",)):
                    if entry.is_symlink() or entry.stat().st_size:
                        raise InventoryError(f"{entry}: unsafe registry marker")
                    continue
                if entry.suffix != ".json" or entry.is_symlink():
                    raise InventoryError(f"{entry}: unknown or unsafe claim entry")
                path = entry.relative_to(root).as_posix()
                claim = _json(inv.read(path), path)
                if directory == TRAIN_CLAIMS:
                    match = re.fullmatch(r"seed-(0|[1-9][0-9]*)\.json", entry.name)
                    if match is None:
                        raise InventoryError(f"{path}: invalid registry filename")
                    seed = _uint(int(match[1]), path)
                    try:
                        validate_train_claim(claim, seed)
                    except ReservationError as exc:
                        raise InventoryError(f"{path}: {exc}") from exc
                    inv.record(seed, path, "geometry_seed")
                else:
                    inv.seeds(claim.get("geometry_seeds"), path, "geometry_seeds")
        except (InventoryError, OSError) as exc:
            issue(directory, "claims", str(exc), mandatory=True)

    try:
        paths = _walk_train(root)
    except InventoryError as exc:
        issue("runs", "discovery", str(exc), mandatory=True)
        paths = []
    for path in TD_JOURNALS:
        if path not in paths:
            issue(path, "TD journal", "missing historical TD training.jsonl", mandatory=True)
    for path in paths:
        try:
            if path.endswith("/training.jsonl"):
                _td_ledger(inv, path)
            elif path.endswith("/cells.jsonl"):
                for index, line in enumerate(inv.read(path).splitlines(), 1):
                    row = _json(line, f"{path}:{index}")
                    inv.record(row.get("geometry_seed"), path, f"line:{index}.geometry_seed")
                inv.typed_paths.add(path)
            elif path.endswith("/episodes.jsonl") or path.endswith("/collection.jsonl"):
                inv.ledger(path, collection=path.endswith("/collection.jsonl"))
            else:
                obj = _json(inv.read(path), path)
                inv.fields(obj, path)
                inv.typed_paths.add(path)
        except InventoryError as exc:
            # A torn record after a known reset must not erase that intersection.
            issue(path, "TRAIN ledger/marker", str(exc))

    for path, raw in sorted(inv.raw.items()):
        if path == candidate_path or path in inv.typed_paths or path == CATALOG:
            continue
        if _mentions_candidate_road(raw, token, candidates):
            entry = {"path": path, "field": "untyped candidate ID",
                     "reason": "candidate occurs outside certified road allocation fields"}
            if entry not in issues:
                issues.append(entry)

    inventory = [{"path": path, "sha256": _sha(raw), "bytes": len(raw)}
                 for path, raw in sorted(inv.raw.items())]
    digest = _digest(inventory)
    if expected_source_inventory_sha256 is not None and digest != expected_source_inventory_sha256:
        warnings.append({"path": "source_inventory", "field": "names+bytes",
                         "reason": "snapshot drift; re-audit relevant cells, not a global hash collision"})
    collisions = [{"geometry_seed": cell["geometry_seed"], "track_id": cell["track_id"],
                   "sources": inv.ids[cell["geometry_seed"]]}
                  for cell in cells if cell["geometry_seed"] in inv.ids]
    # Historical schedules, result-only exposure receipts and protected ID metadata
    # beyond the inspected protocols are not independently complete. No clearance.
    issues.append({"path": "historical_inventory", "field": "coverage",
                   "reason": "historical TRAIN/exposure and protected-ID inventory not independently certified; review omitted result-only receipts and legacy schedules before any claim"})
    return {"format": "haic-tdmpc2-train-diagnostic-inventory-v1", "status": "BLOCKED",
            "candidate_path": candidate_path, "candidate_sha256": candidate_sha,
            "cells": cells, "collisions": collisions, "blockers": issues,
            "provenance_warnings": warnings, "source_inventory": inventory,
            "source_inventory_sha256": digest, "blind_episode_reads": 0,
            "reservation_or_claim": False,
            "limitation": "Read-only candidate intersection inventory; not a fresh-grid certificate or permission to reset"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, help="existing experiments JSON with 24 explicit cells")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-source-inventory-sha256")
    args = parser.parse_args()
    try:
        report = audit_tdmpc2_train_diag_seeds(
            args.candidate_path, repo_root=args.repo_root,
            expected_source_inventory_sha256=args.expected_source_inventory_sha256)
    except InventoryError as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2  # A read-only audit never clears an allocation.


if __name__ == "__main__":
    raise SystemExit(main())
