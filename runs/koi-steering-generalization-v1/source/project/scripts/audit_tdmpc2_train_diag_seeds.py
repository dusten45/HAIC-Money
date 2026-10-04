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
from copy import deepcopy
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
               "collection-result.json", "config.json", "protocol.json", "study_protocol.json"}
_RUN_LOCAL_IDS = {"config.json", "protocol.json", "study_protocol.json",
                  "run-config.json", "run_config.json"}
_RESIDUAL = "runs/20260924-drqv2-residual-options-pilot"
_FINAL_SOURCE = "runs/20260927-drqv2-final-source-replay-v1/collection"
_SAFE_TRAIN_RECEIPTS = {
    f"{_RESIDUAL}/iteration-1/failure.json",
    f"{_RESIDUAL}/iteration-1-retry/result.json",
    f"{_RESIDUAL}/iteration-1-retry/counter-audit.json",
    f"{_RESIDUAL}/iteration-2/result.json",
    "runs/tdmpc2-exploration-20260928-v1/failure.json",
    *(f"{_FINAL_SOURCE}/seed{seed}/receipt.json" for seed in (0, 1)),
}
_FINAL_SOURCE_PROTOCOL = "experiments/drqv2-final-source-replay-v1.json"
_FINAL_SOURCE_COLLECTION = "experiments/drqv2-final-source-replay-collection-v1.json"
_FINAL_SOURCE_R7 = "experiments/drqv2-retention-r7.json"
_FINAL_SOURCE_RNG = {"track_seed", "geometry_seed", "actor_rng_seed", "replay_rng_seed",
                     "target_noise_seed", "update_rng_seed"}


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


def _range_key(key: Any) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).casefold()


def _relevant(value: Any, candidates: set[int], context: str = "", path: str = "") -> bool:
    """Conservatively identify unknown road aliases and overlapping seed ranges."""
    if isinstance(value, dict):
        if (path.endswith("drqv2-final-source-replay-v1.json")
                or path.endswith("drqv2-final-source-replay-reconstruction-r1.json")):
            if (context.endswith(".rng_seeds") and set(value) == _FINAL_SOURCE_RNG
                    and all(type(item) is int and 0 <= item <= 2**32 - 1 for item in value.values())):
                return False  # Validated separately: sampler RNG, never a road ID.
        if (path == _FINAL_SOURCE_COLLECTION and context.endswith(".collection_rng_seeds")
                and set(value) == {"action_noise_seed"}
                and type(value["action_noise_seed"]) is int
                and 0 <= value["action_noise_seed"] <= 2**32 - 1):
            return False
        keys = {_range_key(key): key for key in value}
        road = any(word in context for word in ("seed", "road", "cell", "geometry"))
        if len(keys) != len(value) and (road or any(
                any(word in key for word in ("seed", "road", "cell", "geometry"))
                for key in keys)):
            return True  # Conflicting spellings cannot establish disjointness.
        interval_keys: set[str] = set()
        interval = any(word in context for word in ("range", "interval", "span"))
        if road and (len({"start", "min", "lower", "from"} & keys.keys()) > 1
                     or len({"end", "max", "upper", "to"} & keys.keys()) > 1):
            return True  # Multiple bound conventions cannot prove disjointness.
        if road and ("start" in keys or "min" in keys or "lower" in keys or "from" in keys):
            start = value.get(keys.get("start", keys.get("min", keys.get("lower", keys.get("from", "")))))
            end = value.get(keys.get("end", keys.get("max", keys.get("upper", keys.get("to", "")))))
            count = value.get(keys.get("count", ""))
            if type(start) is not int or (end is not None and type(end) is not int) or (count is not None and type(count) is not int):
                return True
            if end is not None and any(min(start, end) <= seed <= max(start, end) for seed in candidates):
                return True  # End semantics may be inclusive.
            if count is not None and (count <= 0 or any(start <= seed < start + count for seed in candidates)):
                return True
            if end is None and count is None:
                return True  # No upper bound: cannot prove the interior is disjoint.
            interval_keys = {"start", "end", "count", "min", "max", "lower", "upper", "from", "to"}
        elif road and interval:
            return True  # Unrecognized interval bounds could include any candidate.
        for key, item in value.items():
            name = _range_key(key)
            if name in interval_keys:
                continue  # Bounds are not standalone road IDs.
            if name.endswith(("seed_start", "geometry_seed_start", "road_start")):
                prefix = name.removesuffix("start")
                count = value.get(keys.get(prefix + "count", ""))
                end = value.get(keys.get(prefix + "end", ""))
                if _relevant({"start": item, "count": count} if count is not None else
                             {"start": item, "end": end} if end is not None else
                             {"start": item}, candidates, name, path):
                    return True
            if name.endswith(("seed_min", "geometry_seed_min", "road_min")):
                prefix = name.removesuffix("min")
                bound = value.get(keys.get(prefix + "max", ""))
                if _relevant({"min": item, "max": bound} if bound is not None else
                             {"min": item}, candidates, name, path):
                    return True
            if name.endswith(("seed_end", "road_end")):
                if (not any(field in keys for field in (name.removesuffix("end") + "start",
                                                       name.removesuffix("end") + "min"))
                        and (type(item) is not int or any(seed <= item for seed in candidates))):
                    return True
            if name.endswith(("seed_count", "road_count")) and not any(
                    field in keys for field in (name.removesuffix("count") + "start",
                                                name.removesuffix("count") + "min")):
                return True  # Count without an origin could contain any road.
            if name.endswith(("seed_max", "road_max")) and not any(
                    field in keys for field in (name.removesuffix("max") + "min",
                                                name.removesuffix("max") + "start")):
                if type(item) is not int or any(seed <= item for seed in candidates):
                    return True
            if name.endswith(("seed_end", "road_end", "seed_count", "road_count", "seed_max", "road_max")):
                continue  # A paired bound was checked as an interval above.
            if any(word in name for word in ("range", "interval", "span")) and any(
                    word in name for word in ("seed", "road", "cell", "geometry")):
                if not isinstance(item, (dict, list)):
                    return True  # An unknown interval encoding cannot establish disjointness.
                if isinstance(item, dict) and not any(_range_key(k) in
                                                       ("start", "min", "lower", "from") for k in item):
                    return True
            if _relevant(item, candidates, f"{context}.{name}", path):
                return True
        return False
    if isinstance(value, list):
        if any(word in context for word in ("range", "interval", "span")) and any(
                word in context for word in ("seed", "road", "cell", "geometry")):
            if len(value) != 2 or any(type(item) is not int for item in value):
                return True  # An unreviewed range encoding has unknown bounds.
            if any(min(value) <= seed <= max(value) for seed in candidates):
                return True
        return any(_relevant(item, candidates, context, path) for item in value)
    if isinstance(value, str) and re.search(
            r"(?i)\b(?:geometry[_ -]?seed|road[_ -]?id|seed)\s*[:=]\s*(?:"
            + "|".join(map(str, sorted(candidates))) + r")(?!\d)", value):
        return True
    return (any(word in context for word in ("seed", "road", "cell", "geometry"))
            and ((type(value) is int and value in candidates)
                 or (isinstance(value, str) and value.isdecimal() and int(value) in candidates)))


def _source_relevant(raw: bytes, path: str, candidates: set[int], token: re.Pattern[str]) -> bool:
    try:
        return _relevant(_json(raw, path), candidates, path=path)
    except InventoryError:
        if path.endswith(".jsonl"):
            return any(_source_relevant(line, path + f":{i}", candidates, token)
                       for i, line in enumerate(raw.splitlines(), 1))
        return bool(re.search(rb"(?i)(?:seed|road|geometry)[_ -]*(?:range|interval|span|start|end|min|max)", raw) or
                    _mentions_candidate_road(raw, token, candidates))


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
            path = location / name
            relative = path.relative_to(root).as_posix()
            if name not in _RUN_FILES and relative not in _SAFE_TRAIN_RECEIPTS:
                continue
            if path.is_symlink():
                raise InventoryError(f"{path.relative_to(root)}: unsafe symlink")
            paths.append(relative)
    return sorted(paths)


def _td_ledger(inv: _Inventory, path: str) -> None:
    """Retain reset intent and open partial identity; never infer release at EOF."""
    rows = inv.read(path).splitlines()
    if not rows:
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
    if not inv.raw[path].endswith(b"\n"):
        raise InventoryError(f"{path}: torn TD journal after intact rows")
    inv.typed_paths.add(path)
    # A live journal, failed reset-intent, and mid-episode partial are all exposure.
    # Their recorded road remains in inv.ids, even if no successful reset followed.


def _run_local_id_fields(inv: _Inventory, obj: dict[str, Any], path: str) -> dict[str, Any]:
    """Only declarations with explicit road roles are collisions; RNG is not."""
    def sampler_role(value: Any) -> bool:
        return isinstance(value, str) and any(
            word in value.casefold() for word in ("sampler", "learner", "rng"))

    safe = deepcopy(obj)
    roots = [(safe, "")]
    if "config" in safe:
        if not isinstance(safe["config"], dict):
            raise InventoryError(f"{path}: candidate-possible unreviewed run-local config")
        roots.append((safe["config"], "config."))
    for source, prefix in roots:
        if "partition" in source and source["partition"] not in ("TRAIN", "TRAIN-DIAGNOSTIC"):
            if any(field in source for field in ("excluded_training_seeds", "training_geometry_seeds")):
                raise InventoryError(f"{path}.{prefix}partition: unreviewed run-local road role")
        if (any(sampler_role(source.get(key)) for key in ("role", "seed_role", "source_role"))
                and any(field in source for field in ("excluded_training_seeds", "training_geometry_seeds"))):
            raise InventoryError(f"{path}.{prefix}role: contradictory run-local road declaration")
        for field, role in (("excluded_training_seeds", "TRAIN exclusion"),
                            ("training_geometry_seeds", "declared TRAIN road")):
            if field in source:
                inv.seeds(source.pop(field), path, f"{prefix}{field} ({role}; unbound source)")
    for source, prefix in list(roots):
        if "protocol" in source:
            if not isinstance(source["protocol"], dict):
                raise InventoryError(f"{path}.{prefix}protocol: unreviewed run-local source")
            roots.append((source["protocol"], f"{prefix}protocol."))
    roles = {"train", "training", "train_diagnostic", "screen", "development",
             "confirmation", "blind", "heldout", "reserved"}
    for source, prefix in roots:
        if "partitions" not in source:
            continue
        partitions = source["partitions"]
        if not isinstance(partitions, dict):
            raise InventoryError(f"{path}.{prefix}partitions: unreviewed run-local source")
        for role, row in partitions.items():
            if not isinstance(row, dict) or "seeds" not in row:
                continue
            if role not in roles:
                raise InventoryError(f"{path}.{prefix}partitions.{role}: unknown road role")
            tracks = row.get("track_ids")
            if tracks is not None and (not isinstance(tracks, list) or not tracks
                                       or any(type(track) is not int or track not in (1, 2, 3, 4)
                                              for track in tracks)):
                raise InventoryError(f"{path}.{prefix}partitions.{role}: invalid track scope")
            if "partition" in row and row["partition"] not in (
                    role, role.upper(), role.replace("_", "-").upper()):
                raise InventoryError(f"{path}.{prefix}partitions.{role}: contradictory partition role")
            if any(sampler_role(row.get(key)) for key in ("role", "seed_role", "source_role")):
                raise InventoryError(f"{path}.{prefix}partitions.{role}: contradictory seed role")
            inv.seeds(row.pop("seeds"), path,
                      f"{prefix}partitions.{role}.seeds (declared {role}; unbound source)")

    def unreviewed(value: Any) -> bool:
        if isinstance(value, dict):
            return any((any(word in str(key).casefold() for word in
                            ("seed", "road", "geometry", "rng", "sampler")) or unreviewed(item))
                       for key, item in value.items())
        if isinstance(value, list):
            return any(unreviewed(item) for item in value)
        return isinstance(value, str) and re.search(
            r"(?i)\b(?:seed|geometry[_ -]?seed|road[_ -]?id)\s*[:=]\s*\d+", value) is not None

    if unreviewed(safe):
        raise InventoryError(f"{path}: candidate-possible unreviewed run-local seed/road role")
    return safe


def _id_fields(inv: _Inventory, obj: dict[str, Any], path: str) -> None:
    """Inspect road IDs, excluding only reviewed sampler schedules."""
    if path == _FINAL_SOURCE_COLLECTION:
        sources = obj.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise InventoryError(f"{path}: missing collection source schedules")
        safe = dict(obj)
        safe["sources"] = {}
        for seed, row in sources.items():
            if (seed not in ("0", "1") or not isinstance(row, dict)
                    or not isinstance(row.get("collection_rng_seeds"), dict)
                    or set(row["collection_rng_seeds"]) != {"action_noise_seed"}):
                raise InventoryError(f"{path}.sources.{seed}: unknown collection sampler")
            _uint(row["collection_rng_seeds"]["action_noise_seed"], f"{path}.sources.{seed}.action_noise_seed")
            safe["sources"][seed] = {key: value for key, value in row.items()
                                     if key != "collection_rng_seeds"}
        inv.fields(safe, path)
    elif path in (_FINAL_SOURCE_PROTOCOL,
                "experiments/drqv2-final-source-replay-reconstruction-r1.json"):
        if (obj.get("r6_protocol_path") != "experiments/drqv2-geometry-mix-v1-r6.json"
                or obj.get("r6_protocol_sha256") != _sha(inv.read(obj["r6_protocol_path"]))):
            raise InventoryError(f"{path}: unbound r6 TRAIN pool")
        runs = obj.get("runs")
        if not isinstance(runs, list) or not runs:
            raise InventoryError(f"{path}: missing final-source runs")
        safe = dict(obj)
        safe["runs"] = []
        for index, row in enumerate(runs):
            if not isinstance(row, dict) or not isinstance(row.get("rng_seeds"), dict):
                raise InventoryError(f"{path}.runs[{index}]: unknown sampler schema")
            rng = row["rng_seeds"]
            if set(rng) != _FINAL_SOURCE_RNG or any(type(v) is not int or not 0 <= v <= 2**32 - 1
                                                    for v in rng.values()):
                raise InventoryError(f"{path}.runs[{index}]: unknown sampler RNG fields")
            safe["runs"].append({key: value for key, value in row.items() if key != "rng_seeds"})
        inv.fields(safe, path)
    elif path.startswith("runs/") and PurePosixPath(path).name in _RUN_LOCAL_IDS:
        obj = _run_local_id_fields(inv, obj, path)
    else:
        inv.fields(obj, path)
    # The shared typed inspector does not reject every legacy interval spelling.
    # A disjoint unknown interval is only a warning; an overlapping one blocks.
    def has_range(value: Any) -> bool:
        if isinstance(value, dict):
            return any((re.search(r"(?:seed|road|geometry|cell).*?(?:range|interval|span|_start|_end|_count|_min|_max)$", _range_key(key), re.I)
                        is not None or has_range(item)) for key, item in value.items())
        return isinstance(value, list) and any(has_range(item) for item in value)

    if has_range(obj):
        raise InventoryError(f"{path}: unreviewed road interval metadata")
    inv.typed_paths.add(path)


def _receipt_road_keys(value: Any, path: str, allowed: set[str], prefix: str = "") -> None:
    """A typed TRAIN receipt must not hide a second, unreviewed road alias."""
    if isinstance(value, list):
        for item in value:
            _receipt_road_keys(item, path, allowed, prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else key
            if (any(word in key.casefold() for word in ("seed", "road", "geometry"))
                    and field not in allowed):
                raise InventoryError(f"{path}.{field}: unreviewed road identity alias")
            _receipt_road_keys(item, path, allowed, field)


def _residual_episodes(inv: _Inventory, path: str) -> tuple[int, int]:
    raw = inv.read(path)
    decisions = 0
    count = 0
    for index, line in enumerate(raw.splitlines(), 1):
        row = _json(line, f"{path}:{index}")
        _receipt_road_keys(row, f"{path}:{index}", {"geometry_seed"})
        if (row.get("event") is not None or row.get("episode") != index
                or type(row.get("decisions")) is not int or row["decisions"] <= 0
                or type(row.get("track_id")) is not int or row["track_id"] not in (1, 2, 3, 4)):
            raise InventoryError(f"{path}:{index}: unknown completed TRAIN episode")
        inv.record(row.get("geometry_seed"), path, f"line:{index}.geometry_seed (actual)")
        counts = row.get("option_counts")
        if counts is not None and (not isinstance(counts, dict)
                or any(type(n) is not int or n < 0 for n in counts.values())
                or sum(counts.values()) != row["decisions"]):
            raise InventoryError(f"{path}:{index}: completed episode option counts disagree")
        decisions += row["decisions"]
        count += 1
    if not raw.endswith(b"\n"):
        raise InventoryError(f"{path}: torn episode tail")
    inv.typed_paths.add(path)
    return count, decisions


def _residual_result(inv: _Inventory, path: str) -> None:
    obj = _json(inv.read(path), path)
    tail = obj.get("incomplete_final_episode")
    if not isinstance(tail, dict):
        raise InventoryError(f"{path}: missing incomplete TRAIN tail")
    seed = inv.record(tail.get("geometry_seed"), path, "incomplete_final_episode.geometry_seed (actual partial)")
    _receipt_road_keys(obj, path, {"incomplete_final_episode.geometry_seed"})
    if (type(tail.get("track_id")) is not int or tail["track_id"] not in (1, 2, 3, 4)
            or type(tail.get("decisions")) is not int or tail["decisions"] <= 0
            or tail.get("status") != "budget_interrupted; not an episode outcome"
            or obj.get("status") != "completed_bounded_training_pilot"):
        raise InventoryError(f"{path}: unknown partial TRAIN status for {seed}")
    ledger = f"{PurePosixPath(path).parent}/episodes.jsonl"
    count, decisions = _residual_episodes(inv, ledger)
    if type(obj.get("completed_episodes")) is not int or obj["completed_episodes"] != count:
        raise InventoryError(f"{path}: completed episode count disagrees with {ledger}")
    actual = decisions + tail["decisions"]
    counts = obj.get("option_action_counts")
    if (not isinstance(counts, dict)
            or set(counts) != {"KEEP", "STEER_MINUS", "STEER_PLUS", "COAST", "BRAKE"}
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != actual):
        raise InventoryError(f"{path}: option action counts disagree with actual decisions")
    if path.endswith("/iteration-1-retry/result.json"):
        correction_path = f"{PurePosixPath(path).parent}/counter-audit.json"
        correction = _json(inv.read(correction_path), correction_path)
        if (correction.get("status") != "post-run-accounting-correction; original artifacts preserved"
                or correction.get("completed_episode_decisions") != decisions
                or correction.get("incomplete_final_episode_decisions") != tail["decisions"]
                or correction.get("actual_collector_step_calls") != actual
                or correction.get("sum_of_per_option_action_counts") != actual
                or correction.get("result_json_environment_decisions_field") != obj.get("environment_decisions")
                or obj.get("environment_decisions") != actual + 1):
            raise InventoryError(f"{path}: corrected decision/episode/tail counts disagree")
        inv.typed_paths.add(correction_path)
    elif obj.get("environment_decisions") != actual:
        raise InventoryError(f"{path}: decision/episode/tail counts disagree")
    inv.typed_paths.add(path)


def _train_failure(inv: _Inventory, path: str) -> None:
    obj = _json(inv.read(path), path)
    if path.startswith(_RESIDUAL):
        _receipt_road_keys(obj, path, set())
        if (not str(obj.get("status", "")).startswith("implementation_smoke_failed")
                or type(obj.get("environment_decisions_completed")) is not int):
            raise InventoryError(f"{path}: unknown early failure schema")
        if obj["environment_decisions_completed"] > 0:
            raise InventoryError(f"{path}: positive TRAIN decisions with no recorded road ID")
        return  # Zero decisions alone do not release any prior allocation.
    cell = obj.get("reset_cell")
    if not isinstance(cell, dict):
        raise InventoryError(f"{path}: missing reset identity")
    _receipt_road_keys(obj, path, {"reset_cell.geometry_seed"})
    inv.record(cell.get("geometry_seed"), path, "reset_cell.geometry_seed (actual reset; 0 decisions)")
    if (set(cell) != {"geometry_seed", "track_id"} or type(cell["track_id"]) is not int
            or cell["track_id"] not in (1, 2, 3, 4)
            or obj.get("status") != "aborted_after_first_reset_before_first_action"
            or obj.get("attempted_episodes") != 1 or obj.get("decisions") != 0):
        raise InventoryError(f"{path}: unknown zero-action reset schema")
    protocol = "experiments/tdmpc2-exploration-v1.json"
    if obj.get("protocol_sha256") != _sha(inv.read(protocol)):
        raise InventoryError(f"{path}: reset protocol binding mismatch")
    inv.typed_paths.add(path)


def _final_source_receipt(inv: _Inventory, path: str) -> None:
    """Retain known reset rows; never certify a receipt without full schedule parity."""
    seed = PurePosixPath(path).parent.name.removeprefix("seed")
    source = _json(inv.read(_FINAL_SOURCE_PROTOCOL), _FINAL_SOURCE_PROTOCOL)
    collection = _json(inv.read(_FINAL_SOURCE_COLLECTION), _FINAL_SOURCE_COLLECTION)
    original = _json(inv.read(_FINAL_SOURCE_R7), _FINAL_SOURCE_R7)
    receipt = _json(inv.read(path), path)
    _receipt_road_keys(receipt, path, {
        "source_seed", "geometry_seeds", "excluded_diagnostic_roads",
        "collection_rng_seeds", "collection_rng_seeds.action_noise_seed",
    })
    links = source.get("source_replay")
    sources = collection.get("sources")
    if not isinstance(links, dict) or not isinstance(sources, dict):
        raise InventoryError(f"{path}: missing source protocol bindings")
    linked = links.get(seed)
    source_row = sources.get(seed)
    original_sources = original.get("source_replay")
    original_row = original_sources.get(seed) if isinstance(original_sources, dict) else None
    if (not isinstance(linked, dict) or not isinstance(source_row, dict)
            or not isinstance(original_row, dict)):
        raise InventoryError(f"{path}: unknown collection source")
    ledger_path = f"{PurePosixPath(path).parent}/episodes.jsonl"
    if (seed not in ("0", "1") or linked.get("receipt_path") != path
            or linked.get("receipt_sha256") != _sha(inv.raw[path])
            or source.get("r6_protocol_path") != "experiments/drqv2-geometry-mix-v1-r6.json"
            or collection.get("r6_protocol_path") != source["r6_protocol_path"]
            or source.get("r6_protocol_sha256") != _sha(inv.read(source["r6_protocol_path"]))
            or collection.get("r6_protocol_sha256") != source["r6_protocol_sha256"]
            or source.get("r7_protocol_path") != _FINAL_SOURCE_R7
            or collection.get("r7_protocol_path") != _FINAL_SOURCE_R7
            or source.get("r7_protocol_sha256") != _sha(inv.raw[_FINAL_SOURCE_R7])
            or collection.get("r7_protocol_sha256") != _sha(inv.raw[_FINAL_SOURCE_R7])
            or source.get("collection_protocol_path") != _FINAL_SOURCE_COLLECTION
            or source.get("collection_protocol_sha256") != _sha(inv.raw[_FINAL_SOURCE_COLLECTION])
            or collection.get("catalog_sha256") != source.get("catalog_sha256")
            or collection.get("catalog_sha256") != _sha(inv.read(CATALOG))
            or receipt.get("catalog_sha256") != collection["catalog_sha256"]
            or receipt.get("collection_protocol_sha256") != _sha(inv.raw[_FINAL_SOURCE_COLLECTION])
            or receipt.get("format") != "haic-drq-final-source-pool-v1"
            or receipt.get("partition") != "TRAIN" or receipt.get("source_seed") != int(seed)
            or receipt.get("episode_ledger_path") != ledger_path
            or receipt.get("completed") is not True):
        raise InventoryError(f"{path}: unverifiable source/receipt binding")
    if any(receipt.get(field) != source_row.get(field) for field in
           ("original_ledger_sha256", "schedule_sha256", "source_actor_sha256",
            "source_checkpoint_sha256", "collection_rng_seeds")):
        raise InventoryError(f"{path}: original source/schedule binding mismatch")
    original_path = (f"runs/20260922-drq-augmentation-pad-v1-restart/"
                     f"control-seed{seed}/episodes.jsonl")
    if (original_row.get("episode_ledger_path") != original_path
            or original_row.get("episode_ledger_sha256") != receipt.get("original_ledger_sha256")
            or _sha(inv.read(original_path)) != receipt["original_ledger_sha256"]):
        raise InventoryError(f"{path}: original TRAIN episode ledger unbound")
    raw = inv.read(ledger_path)
    if _sha(raw) != receipt.get("episode_ledger_sha256") or not raw.endswith(b"\n"):
        raise InventoryError(f"{path}: unbound or torn collection ledger")
    resets: set[int] = set()
    pending: tuple[int, int, int] | None = None
    steps = 0
    episodes = 0
    lines = raw.splitlines()
    for index, line in enumerate(lines, 1):
        row = _json(line, f"{ledger_path}:{index}")
        _receipt_road_keys(row, f"{ledger_path}:{index}", {"seed", "geometry_seed", "source_seed"})
        event = row.get("event")
        track = row.get("track_id")
        road = row.get("geometry_seed")
        if (event not in ("reset", "end", "capped_partial") or type(track) is not int
                or track not in (1, 2, 3, 4) or type(road) is not int or road != row.get("seed")
                or row.get("source_seed") != int(seed)):
            raise InventoryError(f"{ledger_path}:{index}: unknown collection identity")
        road = inv.record(road, ledger_path, f"line:{index}.geometry_seed (actual)")
        episode_id = row.get("episode_id")
        if type(episode_id) is not int:
            raise InventoryError(f"{ledger_path}:{index}: missing episode ID")
        identity = (road, track, episode_id)
        if event == "reset":
            if (pending is not None or row.get("partition") != "TRAIN"
                    or identity[2] != episodes or row.get("collection_step") != steps):
                raise InventoryError(f"{ledger_path}:{index}: invalid reset order")
            pending = identity
            resets.add(road)
            episodes += 1
        else:
            if (pending != identity or type(row.get("steps")) is not int or row["steps"] < 0
                    or row.get("collection_step") != steps + row["steps"]):
                raise InventoryError(f"{ledger_path}:{index}: invalid completed/partial episode")
            steps += row["steps"]
            pending = None
            if event == "capped_partial" and index != len(lines):
                raise InventoryError(f"{ledger_path}:{index}: partial not last")
    step_path = f"{PurePosixPath(path).parent}/steps.jsonl"
    if (receipt.get("step_ledger_path") != step_path
            or not isinstance(receipt.get("step_ledger_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["step_ledger_sha256"])):
        raise InventoryError(f"{path}: missing pinned TRAIN step ledger")
    step_raw = inv.read(step_path)
    if (_sha(step_raw) != receipt["step_ledger_sha256"] or not step_raw.endswith(b"\n")
            or len(step_raw.splitlines()) != steps):
        raise InventoryError(f"{path}: TRAIN step ledger SHA/count mismatch")
    visited = receipt.get("geometry_seeds")
    if (pending is not None or type(receipt.get("decisions")) is not int
            or receipt["decisions"] != steps
            or receipt.get("scheduled_episodes_consumed") != episodes
            or type(visited) is not list
            or any(type(road) is not int or not 0 <= road <= 2**32 - 1 for road in visited)
            or set(visited) != resets or len(visited) != len(resets)):
        raise InventoryError(f"{path}: visited roads or step/episode counts disagree with ledger")
    if (collection.get("decisions") != 100000 or collection.get("capacity") != 100000
            or receipt.get("decisions") != 100000 or receipt.get("capacity") != 100000):
        raise InventoryError(f"{path}: not the fixed complete 100k-decision TRAIN collection")
    # SHA/count agreement does not establish step road identity or that the full
    # source schedule was followed. The episode resets remain recorded, not released.
    raise InventoryError(f"{path}: step-road and original schedule parity not independently verified")


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
                _id_fields(inv, data, path)
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
            if path in _SAFE_TRAIN_RECEIPTS:
                if path.endswith("/counter-audit.json"):
                    continue  # Checked together with its specific corrected result.
                if path.startswith(_RESIDUAL) and path.endswith("/result.json"):
                    _residual_result(inv, path)
                    issue(path, "source binding", "residual-options run protocol/manifest not independently bound")
                elif path.endswith("/failure.json"):
                    _train_failure(inv, path)
                else:
                    _final_source_receipt(inv, path)
            elif path.startswith(_RESIDUAL) and path.endswith("/episodes.jsonl") and "iteration-1/" not in path:
                _residual_episodes(inv, path)
            elif path.endswith("/training.jsonl"):
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
                _id_fields(inv, obj, path)
        except InventoryError as exc:
            # A torn record after a known reset must not erase that intersection.
            issue(path, "TRAIN ledger/marker", str(exc),
                  mandatory=path == f"{_RESIDUAL}/iteration-1/failure.json"
                  or path.startswith("runs/") and PurePosixPath(path).name in _RUN_LOCAL_IDS
                  or path.startswith(_FINAL_SOURCE) and path.endswith("/receipt.json"))

    for path, raw in sorted(inv.raw.items()):
        if path == candidate_path or path in inv.typed_paths or path == CATALOG:
            continue
        if _source_relevant(raw, path, candidates, token):
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
    # Narrow receipts reduce invisible collisions, not historical closure or clearance.
    issues.append({"path": "historical_inventory", "field": "coverage",
                   "reason": "historical TRAIN/exposure and protected-ID inventory not independently certified; unbound positive-interaction roads, unscheduled historical runs and unreviewed receipts remain; no claim or reset permitted"})
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
