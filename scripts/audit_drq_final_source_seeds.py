"""Candidate-specific, read-only audit of reused original DrQ source TRAIN roads.

Run before freezing the final-source collector protocol, and re-run before its
first reset. A passing receipt is a snapshot, not a fresh-road claim or a global
inventory attestation. Protected outcome directories are never opened.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R7_PROTOCOL = "experiments/drqv2-retention-r7.json"
CATALOG_PROTOCOL = "experiments/drqv2-geometry-augmentation-v1.json"
CLAIMS = "experiments/train-seed-claims"
FIRST_SEQUENCE = 31_072
LAST_SEQUENCE = 131_071
CAPACITY = 100_000
TOTAL_STEPS = 131_072
SOURCE_COUNTS = (367, 321)
FORMAT = "haic-drq-final-source-cross-lane-audit-v1"
_ROAD_KEY = re.compile(r"(?:seed|road|cell|geometry|reserv|blind|confirm|heldout|holdout|diagnostic|retir)", re.I)
_PROTECTED = re.compile(r"blind|confirm|heldout|holdout|diagnostic|screen|private|retir", re.I)
_CLAIM_NAME = re.compile(r"seed-(0|[1-9][0-9]*)\.json\Z")
_SKIP_DIR = re.compile(r"blind|confirm|heldout|holdout|diagnostic|screen|private|eval|submission", re.I)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def schedule_sha256(schedule: list[dict]) -> str:
    # Match the collector's schedule serialization, not a pair-array research hash.
    raw = json.dumps(schedule, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return digest(raw.encode("utf-8"))


def _unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json(raw: bytes, name: str) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda token: require(False, f"{name}: {token}"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: malformed JSON") from exc
    require(type(value) is dict, f"{name}: expected JSON object")
    return value


def _read(root: Path, relative: str, expected: str | None = None) -> tuple[bytes, str]:
    path = Path(relative)
    require(not path.is_absolute() and all(part not in (".", "..") for part in path.parts)
            and path.parts and "\\" not in relative, f"unsafe source path: {relative}")
    target = root
    for part in path.parts:
        target = target / part
        require(not target.is_symlink(), f"symlinked audit input: {relative}")
    require(target.is_file(), f"missing audit input: {relative}")
    raw = target.read_bytes()
    actual = digest(raw)
    if expected is not None:
        require(actual == expected, f"pinned source SHA mismatch: {relative}")
    return raw, actual


def _ledger(raw: bytes, name: str) -> list[dict]:
    episodes: list[dict] = []
    previous_end = 0
    for number, line in enumerate(raw.splitlines(), 1):
        row = _json(line, f"{name}:{number}")
        if row.get("event") == "reset":
            require(row.get("episode_id") == len(episodes) and type(row.get("episode_id")) is int
                    and type(row.get("track_id")) is int and row["track_id"] in (1, 2, 3, 4)
                    and type(row.get("seed")) is int and 0 <= row["seed"] < 2**32
                    and (not episodes or "end_step" in episodes[-1]),
                    f"{name}:{number}: invalid original TRAIN reset")
            episodes.append({"original_episode_id": row["episode_id"],
                             "track_id": row["track_id"], "geometry_seed": row["seed"],
                             "start_sequence": previous_end})
        elif row.get("event") == "end":
            require(bool(episodes) and "end_step" not in episodes[-1]
                    and row.get("episode_id") == episodes[-1]["original_episode_id"]
                    and row.get("seed") == episodes[-1]["geometry_seed"]
                    and row.get("track_id") == episodes[-1]["track_id"]
                    and type(row.get("steps")) is int and 0 < row["steps"] <= 2000
                    and type(row.get("global_step")) is int
                    and row["global_step"] == previous_end + row["steps"]
                    and row["global_step"] <= TOTAL_STEPS,
                    f"{name}:{number}: original TRAIN episode end does not join reset")
            previous_end = row["global_step"]
            episodes[-1]["end_step"] = previous_end
        else:
            raise ValueError(f"{name}:{number}: unrecognized original ledger event")
    require(bool(episodes) and "end_step" not in episodes[-1]
            and episodes[-1]["start_sequence"] < TOTAL_STEPS,
            f"{name}: final original budget-cap reset missing or ambiguous")
    return episodes


def _checkpoint_state(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    require(type(payload) is dict and payload.get("format") == "haic-drq-v2-checkpoint-v1"
            and payload.get("environment_steps") == TOTAL_STEPS,
            f"{path}: not the original 131072-step source checkpoint")
    return payload["replay"]


def _pinned_checkpoint(root: Path, relative: str, expected: str) -> Path:
    require(type(relative) is str and not Path(relative).is_absolute()
            and "\\" not in relative and type(expected) is str and len(expected) == 64,
            f"unsafe checkpoint identity: {relative}")
    path = root
    for part in Path(relative).parts:
        require(part not in (".", "..") and not path.is_symlink(),
                f"unsafe checkpoint path: {relative}")
        path = path / part
    require(path.is_file() and not path.is_symlink(), f"missing source checkpoint: {relative}")
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    require(sha.hexdigest() == expected, f"pinned source SHA mismatch: {relative}")
    return path


def derive_schedule(episodes: list[dict], state: dict) -> tuple[list[dict], int]:
    require(state.get("capacity") == CAPACITY and state.get("size") == CAPACITY
            and state.get("next_sequence") == LAST_SEQUENCE + 1
            and state.get("action_dim") == 3 and state.get("n_step") == 3
            and state.get("gamma") == .99, "source checkpoint replay contract differs")
    positions = np.arange(FIRST_SEQUENCE, LAST_SEQUENCE + 1, dtype=np.int64)
    slots = positions % CAPACITY
    require(np.array_equal(np.asarray(state["sequence_ids"])[slots], positions),
            "original checkpoint retained sequence window incomplete")
    ids = np.asarray(state["episode_ids"])[slots]
    steps = np.asarray(state["episode_steps"])[slots]
    boundary = int(ids[0])
    require(0 <= boundary < len(episodes) and episodes[boundary]["start_sequence"] < FIRST_SEQUENCE
            and episodes[boundary].get("end_step", TOTAL_STEPS) > FIRST_SEQUENCE,
            "first retained sequence has no containing original episode")
    expected_ids = np.empty_like(positions)
    expected_steps = np.empty_like(positions)
    for row in episodes[boundary:]:
        start = row["start_sequence"]
        end = row.get("end_step", TOTAL_STEPS)
        selected = (positions >= start) & (positions < end)
        expected_ids[selected] = row["original_episode_id"]
        expected_steps[selected] = positions[selected] - start
    require(np.array_equal(ids, expected_ids) and np.array_equal(steps, expected_steps),
            "checkpoint episode identity/steps do not join original source ledger")
    ordered = episodes[boundary:] + episodes[:boundary]
    schedule = [{key: row[key] for key in ("original_episode_id", "track_id", "geometry_seed")}
                for row in ordered]
    require(len({row["geometry_seed"] for row in schedule}) == len(schedule),
            "original source schedule contains a duplicate geometry seed")
    return schedule, boundary


def _road_mentions(value: object, candidates: set[int], field: tuple[str, ...] = ()):
    if type(value) is dict:
        name = ".".join(field).lower()
        start = value.get("seed_start", value.get("geometry_seed_start"))
        count = value.get("seed_count", value.get("geometry_seed_count"))
        if "seed_range" in name or "road_range" in name:
            start = value.get("start", start)
            count = value.get("count", count)
        if type(start) is int and type(count) is int and count > 0:
            for seed in candidates:
                if start <= seed < start + count:
                    yield field + ("range",), seed
        elif "seed_start" in value and type(value.get("seed_start")) is int:
            explicit = value.get("candidate_seeds")
            rule = value.get("candidate_rule")
            span = re.fullmatch(r"seed_start \+ offset for offsets 0\.\.(\d+), in that order; no replacement on collision",
                                rule) if type(rule) is str else None
            if (type(explicit) is list and span is not None and
                    len(explicit) == int(span[1]) + 1 and
                    explicit == list(range(value["seed_start"], value["seed_start"] + len(explicit)))):
                for seed in candidates.intersection(explicit):
                    yield field + ("range",), seed
            else:
                yield field + ("unparsed_range",), None
        elif any("seed_range" in key or "seed_start" in key for key in value):
            yield field + ("unparsed_range",), None
        for key, item in value.items():
            yield from _road_mentions(item, candidates, field + (key,))
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _road_mentions(item, candidates, field + (str(index),))
    elif field and _ROAD_KEY.search(".".join(field)):
        if type(value) is int and value in candidates:
            yield field, value
        elif type(value) is str:
            if value.isdecimal() and int(value) in candidates:
                yield field, int(value)
            else:
                for token in re.finditer(r"\b(?:geometry[_ -]?seed|road[_ -]?id|seed)\s*[:=]\s*(\d+)\b", value, re.I):
                    if int(token[1]) in candidates:
                        yield field, int(token[1])


def _metadata(root: Path, candidates: set[int]) -> tuple[list[dict], list[str]]:
    directory = root / "experiments"
    require(directory.is_dir() and not directory.is_symlink(), "missing/unsafe experiments directory")
    expected_sources: list[dict] = []
    inventory: list[str] = []
    tokens = re.compile(r"\b(?:" + "|".join(map(str, sorted(candidates))) + r")\b")
    for path in sorted(directory.glob("*.json")):
        relative = path.relative_to(root).as_posix()
        raw, source_sha = _read(root, relative)
        # An unrelated malformed file is not a global freshness failure.
        try:
            obj = _json(raw, relative)
        except ValueError:
            require(not tokens.search(raw.decode("utf-8", errors="replace"))
                    and b"seed_range" not in raw and b"seed_start" not in raw,
                    f"candidate-ambiguous experiment metadata: {relative}")
            continue
        if obj.get("format") == FORMAT:
            roads = obj.get("evidence", {}).get("expected_source_exclusions", {}).get("historical_cells")
            require(obj.get("passed") is True and obj.get("partition") == "TRAIN"
                    and obj.get("protected_or_reserved_overlap") == []
                    and obj.get("ambiguous_records") == []
                    and type(roads) is list
                    and {row.get("geometry_seed") for row in roads if type(row) is dict} == candidates,
                    f"candidate-ambiguous prior source audit: {relative}")
            hits = list(_road_mentions(obj, candidates))
            require(all(".".join(part for part in field if not part.isdecimal()) ==
                        "evidence.expected_source_exclusions.historical_cells.geometry_seed"
                        for field, _ in hits),
                    f"candidate-ambiguous prior source audit: {relative}")
            continue  # An earlier audit receipt is not an allocation or claim.
        hits = list(_road_mentions(obj, candidates))
        if not hits:
            continue
        for field, seed in hits:
            keys = ".".join(part for part in field if not part.isdecimal()).lower()
            own_catalog = (relative == CATALOG_PROTOCOL
                           and keys == "exclusion_seed_ids.reserved")
            teacher = re.fullmatch(
                r"experiments/drqv2-teacher-replay-v1(?:-r[23])?"
                r"(?:-geometry-audit(?:-final)?)?\.json", relative) is not None
            rlpd = re.fullmatch(
                r"experiments/pixel-rlpd-(?:offpolicy-pilot-v[12]|long-horizon-followup-v1|"
                r"entropy-target-ablation-v[1-5])\.json", relative) is not None
            old_source_exclusion = (
                relative == "experiments/drqv2-geometry-augmentation-v1-diagnostic-protocol.json"
                and keys == "reserved_training_seeds"
            ) or teacher and keys in (
                "reserved_training_seeds", "known_excluded_geometry_seeds",
                "source_actors.training_geometry_seeds"
            ) or rlpd and keys in (
                "reserved_training_seeds", "geometry_audit.teacher_source_ledgers.geometry_seeds"
            )
            if own_catalog or old_source_exclusion:
                if seed is not None:
                    expected_sources.append({"path": relative, "field": keys, "geometry_seed": seed})
                continue
            if seed is None:
                raise ValueError(f"candidate-ambiguous seed range: {relative}:{keys}")
            if _PROTECTED.search(keys) or any(key in keys for key in (
                    "training_geometry_seeds", "candidate_seeds", "reserved_training_seeds",
                    "geometry_seed", "road_ids", "seeds")):
                raise ValueError(f"protected or foreign allocated road {seed}: {relative}:{keys}")
            raise ValueError(f"candidate-ambiguous metadata road {seed}: {relative}:{keys}")
        inventory.append(f"{relative}:{source_sha}")
    catalog_raw, catalog_sha = _read(root, CATALOG_PROTOCOL)
    catalog = _json(catalog_raw, CATALOG_PROTOCOL)
    require(catalog.get("format") == "haic-drq-training-geometry-protocol-v1"
            and type(catalog.get("exclusion_seed_ids")) is dict
            and type(catalog["exclusion_seed_ids"].get("reserved")) is list,
            "source-history exclusion catalog incomplete")
    reserved = catalog["exclusion_seed_ids"]["reserved"]
    require(candidates <= set(reserved), "original source seeds missing from known-used TRAIN exclusion catalog")
    return expected_sources, sorted(inventory + [f"{CATALOG_PROTOCOL}:{catalog_sha}"])


def _registry(root: Path, candidates: set[int]) -> dict:
    path = root / CLAIMS
    require(path.is_dir() and not path.is_symlink(), "shared TRAIN claims registry missing/unsafe")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        observed = []
        for name in sorted(os.listdir(descriptor)):
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if name == ".gitkeep":
                require(stat.S_ISREG(info.st_mode) and info.st_size == 0,
                        "unsafe shared TRAIN claims marker")
                continue
            match = _CLAIM_NAME.fullmatch(name)
            if match is None:
                require(not any(re.search(rf"(?<!\d){seed}(?!\d)", name) for seed in candidates),
                        f"candidate-ambiguous claim filename: {name}")
                continue
            seed = int(match[1])
            if seed in candidates:
                raise ValueError(f"foreign immutable TRAIN claim for geometry seed {seed}")
            require(stat.S_ISREG(info.st_mode), f"unsafe TRAIN claim: {name}")
            observed.append(name)
        return {"directory": CLAIMS, "candidate_claim_count": 0,
                "other_claim_names": observed}
    finally:
        os.close(descriptor)


def _own_collection(root: Path, relative: str, seed: int, schedule: list[dict],
                    original_sha: str, checkpoint_sha: str, actor_sha: str) -> str:
    base = f"runs/20260927-drqv2-final-source-replay-v1/collection/seed{seed}"
    require(relative == f"{base}/episodes.jsonl", "unrecognized final source collection ledger")
    receipt_path = f"{base}/receipt.json"
    receipt_raw, receipt_sha = _read(root, receipt_path)
    receipt = _json(receipt_raw, receipt_path)
    protocol_path = "experiments/drqv2-final-source-replay-collection-v1.json"
    _, protocol_sha = _read(root, protocol_path)
    require(receipt.get("format") == "haic-drq-final-source-pool-v1"
            and receipt.get("completed") is True and receipt.get("source_seed") == seed
            and receipt.get("partition") == "TRAIN" and receipt.get("excluded_diagnostic_roads") is True
            and receipt.get("collection_protocol_sha256") == protocol_sha
            and receipt.get("schedule_sha256") == schedule_sha256(schedule)
            and receipt.get("original_ledger_sha256") == original_sha
            and receipt.get("source_checkpoint_sha256") == checkpoint_sha
            and receipt.get("source_actor_sha256") == actor_sha
            and receipt.get("episode_ledger_path") == relative
            and receipt.get("step_ledger_path") == f"{base}/steps.jsonl"
            and receipt.get("pool_path") == f"{base}/pool.pt"
            and receipt.get("decisions") == CAPACITY
            and receipt.get("capacity") == CAPACITY,
            f"own source collection receipt incomplete or different: {receipt_path}")
    raw, _ = _read(root, relative, receipt.get("episode_ledger_sha256"))
    step_path = _pinned_checkpoint(root, f"{base}/steps.jsonl", receipt.get("step_ledger_sha256"))
    _pinned_checkpoint(root, f"{base}/pool.pt", receipt.get("pool_sha256"))
    resets = []
    pending = None
    for number, line in enumerate(raw.splitlines(), 1):
        row = _json(line, f"{relative}:{number}")
        if row.get("event") == "reset":
            index = len(resets)
            require(pending is None and index < len(schedule)
                    and row.get("episode_id") == index and row.get("schedule_index") == index
                    and row.get("source_seed") == seed and row.get("partition") == "TRAIN"
                    and row.get("original_episode_id") == schedule[index]["original_episode_id"]
                    and row.get("track_id") == schedule[index]["track_id"]
                    and row.get("seed") == row.get("geometry_seed") == schedule[index]["geometry_seed"],
                    f"own source collection reset diverges from schedule: {relative}:{number}")
            pending = index
            resets.append(row["geometry_seed"])
        else:
            require(row.get("event") in ("end", "capped_partial") and pending is not None
                    and row.get("episode_id") == pending and row.get("schedule_index") == pending
                    and row.get("source_seed") == seed
                    and row.get("original_episode_id") == schedule[pending]["original_episode_id"]
                    and row.get("track_id") == schedule[pending]["track_id"]
                    and row.get("seed") == row.get("geometry_seed") == schedule[pending]["geometry_seed"],
                    f"own source collection outcome missing or ambiguous: {relative}:{number}")
            pending = None
    require(pending is None and resets and receipt.get("scheduled_episodes_consumed") == len(resets)
            and receipt.get("geometry_seeds") == sorted(resets),
            "own final-source collection road inventory does not join receipt")
    boundary = schedule[0]["original_episode_id"]
    prefix_start = sum(row["original_episode_id"] >= boundary for row in schedule)
    require(receipt.get("retained_window_schedule_episodes") == prefix_start
            and receipt.get("historical_prefix_episodes_consumed") == max(0, len(resets) - prefix_start),
            "own collection prefix contingency differs from frozen schedule")
    prefix_steps = 0
    decision = 0
    with step_path.open("rb") as stream:
        for decision, line in enumerate(stream, 1):
            row = _json(line, f"{step_path}:{decision}")
            index = row.get("schedule_index")
            require(type(index) is int and 0 <= index < len(resets)
                    and row.get("decision") == decision and row.get("sequence_id") == decision - 1
                    and row.get("source_seed") == seed and row.get("partition") == "TRAIN"
                    and row.get("episode_id") == index
                    and row.get("original_episode_id") == schedule[index]["original_episode_id"]
                    and row.get("track_id") == schedule[index]["track_id"]
                    and row.get("geometry_seed") == schedule[index]["geometry_seed"]
                    and row.get("schedule_phase") == (
                        "historical_prefix_once" if index >= prefix_start else "retained_window"),
                    f"own source pool step has foreign or ambiguous road: {step_path}:{decision}")
            prefix_steps += int(index >= prefix_start)
    require(decision == CAPACITY and prefix_steps == receipt.get("historical_prefix_decisions"),
            "own source pool step count or fallback share differs from receipt")
    return receipt_sha


def _run_evidence(root: Path, candidates: dict[int, int], schedules: dict[int, list[dict]],
                  ledgers: dict[str, str], checkpoints: dict[str, str], actors: dict[int, str]) -> dict:
    directory = root / "runs"
    require(directory.is_dir() and not directory.is_symlink(), "runs directory missing/unsafe")
    token = re.compile(rb"\b(?:" + b"|".join(str(seed).encode() for seed in sorted(candidates)) + rb")\b")
    seen = []
    own_receipts = {}

    def onerror(error: OSError):
        raise ValueError(f"runs discovery failed: {error}") from error

    for current, dirs, files in os.walk(directory, followlinks=False, onerror=onerror):
        parent = Path(current)
        for name in dirs[:]:
            child = parent / name
            require(not child.is_symlink(), f"symlinked runs directory: {child}")
            if _SKIP_DIR.search(re.sub(r"[^a-z0-9]", "", name.casefold())):
                dirs.remove(name)  # Never open protected evaluation/diagnostic episodes.
        for name in files:
            if name not in ("episodes.jsonl", "collection.jsonl", "cells.jsonl"):
                continue
            relative = (parent / name).relative_to(root).as_posix()
            target = parent / name
            require(not target.is_symlink(), f"unsafe TRAIN ledger path: {relative}")
            with target.open("rb") as stream:
                relevant = any(token.search(line) for line in stream)
            if not relevant:
                continue
            raw, source_sha = _read(root, relative)
            if relative.startswith("runs/20260927-drqv2-final-source-replay-v1/collection/"):
                match = re.fullmatch(
                    r"runs/20260927-drqv2-final-source-replay-v1/collection/seed([01])/episodes\.jsonl",
                    relative)
                require(match is not None, f"candidate-ambiguous final collection: {relative}")
                seed = int(match[1])
                own_receipts[str(seed)] = _own_collection(
                    root, relative, seed, schedules[seed], ledgers[str(seed)],
                    checkpoints[str(seed)], actors[seed])
                seen.append({"path": relative, "sha256": source_sha,
                             "candidate_road_rows": len(raw.splitlines())})
                continue
            matched = 0
            for number, line in enumerate(raw.splitlines(), 1):
                if not token.search(line):
                    continue
                row = _json(line, f"{relative}:{number}")
                seeds = [row[key] for key in ("geometry_seed", "seed") if key in row]
                if not seeds or any(type(seed) is not int for seed in seeds):
                    raise ValueError(f"candidate-ambiguous TRAIN ledger: {relative}:{number}")
                if not any(seed in candidates for seed in seeds):
                    require(not list(_road_mentions(row, set(candidates))),
                            f"candidate-ambiguous additional road field: {relative}:{number}")
                    continue
                require(len(set(seeds)) == 1 and type(row.get("track_id")) is int,
                        f"candidate-ambiguous TRAIN road identity: {relative}:{number}")
                seed = seeds[0]
                require(row["track_id"] == candidates[seed],
                        f"cross-track road identity collision: {relative}:{number}")
                require(row.get("partition", "TRAIN") == "TRAIN",
                        f"protected TRAIN ledger road: {relative}:{number}")
                require(row.get("event") in ("reset", "end", "budget-stop", "capped_partial",
                                             "stored_episode", "discarded_incomplete_episode")
                        or name == "cells.jsonl" and row.get("partition") == "TRAIN",
                        f"candidate-ambiguous TRAIN ledger event: {relative}:{number}")
                require(relative.startswith("runs/20260922-drq-")
                        or relative in ("runs/20260922-drq-augmentation-pad-v1-restart/"
                                        "control-seed0/episodes.jsonl",
                                        "runs/20260922-drq-augmentation-pad-v1-restart/"
                                        "control-seed1/episodes.jsonl"),
                        f"foreign TRAIN interaction on historical source seed: {relative}:{number}")
                matched += 1
            if matched:
                seen.append({"path": relative, "sha256": source_sha,
                             "candidate_road_rows": matched})
    # G0 immutable claim is not housed in the shared registry.
    g0 = root / "runs/rlpd-g0-claims"
    if g0.is_dir():
        require(not g0.is_symlink(), "unsafe G0 claim directory")
        for path in sorted(g0.glob("*.json")):
            relative = path.relative_to(root).as_posix()
            raw, _ = _read(root, relative)
            if token.search(raw):
                raise ValueError(f"candidate-ambiguous or foreign G0 TRAIN reservation: {relative}")
    return {"previous_same_lane_interactions": seen, "verified_own_pool_receipt_sha256": own_receipts,
            "protected_outcome_directories_opened": 0}


def audit(root: Path) -> dict:
    root = root.resolve(strict=True)
    r7_raw, r7_sha = _read(root, R7_PROTOCOL)
    r7 = _json(r7_raw, R7_PROTOCOL)
    require(r7.get("format") == "haic-drq-retention-study-v1"
            and set(r7.get("source_replay", {})) == {"0", "1"},
            "source replay protocol missing two original source seeds")
    r6_path = r7.get("r6_protocol_path")
    require(type(r6_path) is str and type(r7.get("r6_protocol_sha256")) is str,
            "r6 allocation lineage unspecified")
    r6_raw, r6_sha = _read(root, r6_path, r7["r6_protocol_sha256"])
    r6 = _json(r6_raw, r6_path)
    require(r6.get("format") == "haic-drq-geometry-mix-study-v1"
            and r6.get("environment", {}).get("partition") == "TRAIN"
            and r6.get("environment", {}).get("obstacles") is True,
            "source allocation does not have pinned TRAIN obstacles")
    catalog_ref = r6.get("catalog")
    require(type(catalog_ref) is dict and type(catalog_ref.get("path")) is str
            and type(catalog_ref.get("sha256")) is str,
            "r6 diagnostic catalog provenance unavailable")
    catalog_raw, catalog_sha = _read(root, catalog_ref["path"], catalog_ref["sha256"])
    catalog = _json(catalog_raw, catalog_ref["path"])
    diagnostic = catalog.get("train_diagnostic")
    require(type(diagnostic) is list and len(diagnostic) == 16
            and all(type(row) is dict and type(row.get("track_id")) is int
                    and type(row.get("geometry_seed")) is int for row in diagnostic)
            and r6.get("diagnostic_pool", {}).get("partition") == "TRAIN-DIAGNOSTIC"
            and set(r6["diagnostic_pool"].get("geometry_seeds", [])) == {
                row["geometry_seed"] for row in diagnostic},
            "r6 TRAIN-DIAGNOSTIC catalog does not join frozen allocation")
    schedules = {}
    boundaries = {}
    ledgers = {}
    checkpoints = {}
    configs = {}
    source_actors = {}
    require(type(r6.get("source_actors")) is list
            and {row.get("learner_seed") for row in r6["source_actors"]} == {0, 1},
            "original source actor lineage missing")
    for seed in (0, 1):
        source = r7["source_replay"][str(seed)]
        ledger_path = source["episode_ledger_path"]
        expected_path = ("runs/20260922-drq-augmentation-pad-v1-restart/"
                         f"control-seed{seed}/episodes.jsonl")
        require(ledger_path == expected_path,
                f"seed{seed}: original source ledger path differs")
        config_path = f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}/config.json"
        config_raw, configs[str(seed)] = _read(root, config_path)
        config = _json(config_raw, config_path)
        settings = config.get("config")
        source_protocol = settings.get("protocol") if isinstance(settings, dict) else None
        reward = settings.get("reward_contract") if isinstance(settings, dict) else None
        require(isinstance(settings, dict) and isinstance(source_protocol, dict)
                and settings.get("training_obstacles") == "official"
                and settings.get("track_ids") == source_protocol.get("training_track_ids") == [1, 2, 3, 4]
                and settings.get("frame_skip") == source_protocol.get("frame_skip") == 4
                and settings.get("max_steps") == source_protocol.get("max_steps") == 2000
                and isinstance(reward, dict) and reward.get("reward_shaping") is False
                and reward.get("norm_reward") is False,
                f"seed{seed}: source interactions were not official-obstacle TRAIN")
        ledger_raw, ledgers[str(seed)] = _read(root, ledger_path, source["episode_ledger_sha256"])
        episodes = _ledger(ledger_raw, ledger_path)
        require(len(episodes) == SOURCE_COUNTS[seed],
                f"seed{seed}: original source episode count differs")
        checkpoint_path = source["checkpoint_path"]
        require(checkpoint_path == ("runs/20260922-drq-augmentation-pad-v1-restart/"
                                    f"control-seed{seed}/checkpoints/step-000131072/checkpoint.pt"),
                f"seed{seed}: original checkpoint path differs")
        checkpoint = _pinned_checkpoint(root, checkpoint_path, source["checkpoint_sha256"])
        checkpoints[str(seed)] = source["checkpoint_sha256"]
        actor = next(row for row in r6["source_actors"] if row["learner_seed"] == seed)
        require(actor.get("checkpoint_sha256") == checkpoints[str(seed)]
                and type(actor.get("actor_sha256")) is str,
                f"seed{seed}: original source actor checkpoint lineage differs")
        source_actors[seed] = actor["actor_sha256"]
        state = _checkpoint_state(checkpoint)
        schedules[seed], boundaries[str(seed)] = derive_schedule(episodes, state)
        del state
    original_zero = schedules[0][-boundaries["0"]:] + schedules[0][:-boundaries["0"]]
    original_one = schedules[1][-boundaries["1"]:] + schedules[1][:-boundaries["1"]]
    require(original_zero[:len(original_one)] == original_one,
            "two source ledgers disagree on indexed original TRAIN road stream")
    cells = {row["geometry_seed"]: row["track_id"] for row in schedules[0]}
    require(len(cells) == SOURCE_COUNTS[0]
            and all(cells.get(row["geometry_seed"]) == row["track_id"] for row in schedules[1]),
            "original source TRAIN streams disagree on cross-track road identity")
    require(not ({row["geometry_seed"] for row in diagnostic} & set(cells)),
            "source schedule overlaps TRAIN-DIAGNOSTIC catalog")
    own_refs, inventory = _metadata(root, set(cells))
    catalog_protocol_sha = digest(_read(root, CATALOG_PROTOCOL)[0])
    require(catalog.get("protocol_sha256") == catalog_protocol_sha,
            "historical TRAIN exclusion catalog protocol SHA drift")
    registry = _registry(root, set(cells))
    train_ledgers = _run_evidence(root, cells, schedules, ledgers, checkpoints, source_actors)
    ref_counts = {}
    for ref in own_refs:
        key = (ref["path"], ref["field"])
        ref_counts[key] = ref_counts.get(key, 0) + 1
    return {"format": FORMAT, "passed": True, "partition": "TRAIN",
            "schedule_sha256": {str(seed): schedule_sha256(schedules[seed]) for seed in (0, 1)},
            "original_ledger_sha256": ledgers, "excluded_diagnostic_roads": True,
            "protected_or_reserved_overlap": [], "ambiguous_records": [],
            "evidence": {"r7_protocol_sha256": r7_sha, "r6_protocol_sha256": r6_sha,
                         "diagnostic_catalog_sha256": catalog_sha,
                         "source_checkpoint_sha256": checkpoints,
                         "source_training_config_sha256": configs,
                         "original_ledger_sha256": ledgers,
                         "source_schedule": {str(seed): {"containing_episode": boundaries[str(seed)],
                                            "primary_count": SOURCE_COUNTS[seed] - boundaries[str(seed)],
                                            "earlier_once_count": boundaries[str(seed)],
                                            "total_count": SOURCE_COUNTS[seed]}
                                             for seed in (0, 1)},
                         "distinct_historical_source_train_cells": len(cells),
                         "expected_source_exclusions": {"count": len(own_refs),
                             "distinct_geometry_seeds": len({ref["geometry_seed"] for ref in own_refs}),
                             "historical_cells": [{"track_id": cells[seed], "geometry_seed": seed}
                                                  for seed in sorted(cells)],
                             "references_by_source": [{"path": path, "field": field, "count": count}
                                                      for (path, field), count in sorted(ref_counts.items())]},
                         "matched_experiment_sources": inventory,
                         "shared_train_claims": registry, "actual_train_ledgers": train_ledgers,
                         "limitations": ["Historical source TRAIN reuse, not new/fresh TRAIN geometry.",
                             "Read-only candidate-scoped snapshot, not an atomic claim or global freshness audit.",
                             "Recheck active claims and SHA-bound schedule before freeze and before first reset.",
                             "Unrecorded exposure and future cross-lane allocations remain unknown."]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--preflight-only", action="store_true", help="print audit; never write")
    parser.add_argument("--output", type=Path, help="new receipt path under experiments/ or runs/")
    args = parser.parse_args()
    if not args.preflight_only:
        require(args.output is not None, "--output required unless --preflight-only")
    report = audit(args.repo_root)
    if args.preflight_only:
        print(json.dumps(report, sort_keys=True, indent=2))
        return
    root = args.repo_root.resolve(strict=True)
    output = args.output
    if output.is_absolute():
        output = output.relative_to(root)
    require(output.suffix == ".json" and all(part not in (".", "..") for part in output.parts)
            and (len(output.parts) == 2 and output.parts[0] == "experiments"
                 or len(output.parts) >= 3 and output.parts[:2] == (
                     "runs", "20260927-drqv2-final-source-replay-v1"))
            and not any(_SKIP_DIR.search(part) for part in output.parts[1:]),
            "receipt must be a new unprotected experiments/ or final-source runs/ JSON path")
    parent = root
    for part in output.parts[:-1]:
        parent = parent / part
        require(parent.is_dir() and not parent.is_symlink(), "receipt parent missing or unsafe")
    destination = root / output
    require(not destination.exists() and not destination.is_symlink(), "audit receipt already exists")
    # No directory creation and no overwrite; a failed audit never reaches this point.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "receipt": output.as_posix(),
                      "sha256": digest(destination.read_bytes())}, sort_keys=True))


if __name__ == "__main__":
    main()
