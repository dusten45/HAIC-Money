"""Read-only inventory for an EXPLICIT 24-road TD TRAIN reuse proposal.

This is deliberately not a clearance/claim tool. Legacy result-only interaction
records cannot yet be exhaustively joined to road IDs, so even a clean snapshot
returns BLOCKED. Protected outcome files and the environment are never opened.
Run from the repository root with ``python -B -m scripts.audit_tdmpc2_consumed_train_reuse``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any


FORMAT = "haic-tdmpc2-consumed-train-reuse-candidates-v1"
CATALOG = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
CATALOG_RESULT = "experiments/drqv2-geometry-augmentation-v1-catalog-result.json"
CATALOG_PROTOCOL = "experiments/drqv2-geometry-augmentation-v1.json"
R6_PROTOCOL = "experiments/drqv2-geometry-mix-v1-r6.json"
ACTOR_RECEIPT = "experiments/drqv2-geometry-augmentation-v1-final-set.json"
CATALOG_SHA256 = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
CATALOG_PROTOCOL_SHA256 = "be6d1d1c3b1b16f6b56fd4720097ca8fea8b64c5c962d07183961cdfa0d87294"
TD4 = {3910800001, 3910800004, 3910800034, 3910800085}
TD_ORDER = (3910800001, 3910800004, 3910800034, 3910800085)
COMPLETE_DECISIONS = 100000
TD_JOURNALS = (
    "runs/tdmpc2-reused-train-20260927-v1/training.jsonl",
    "runs/tdmpc2-reused-train-20260927-v2/training.jsonl",
    "runs/tdmpc2-long-20260928-v1/training.jsonl",
    "runs/tdmpc2-long-20260928-v2/training.jsonl",
    "runs/tdmpc2-damage-20260928-v1/training.jsonl",
)
COMPLETE_TD = (
    ("experiments/tdmpc2-long-reused-train-v2-100k-result.json",
     "experiments/tdmpc2-long-reused-train-v2.json", "runs/tdmpc2-long-20260928-v2"),
    ("experiments/tdmpc2-damage-shaping-v1-100k-result.json",
     "experiments/tdmpc2-damage-shaping-v1.json", "runs/tdmpc2-damage-20260928-v1"),
)
_GENERATION_SOURCES = {
    "core/finish_line.py", "core/vendor/car_racing.py",
    "haic/algorithms/drq_v2/geometry_features.py",
    "scripts/generate_drq_training_geometry.py",
}
_PROTECTED = re.compile(r"blind|confirm|screen|held.?out|holdout|private|submission|evaluation", re.I)
_OUTCOME = re.compile(r"(?:^|[-_])(?:result|results|score|summary|analysis|failure|receipt|preflight)(?:[-_.]|$)", re.I)
_ROAD_FILE = {"episodes.jsonl", "training.jsonl", "collection.jsonl", "cells.jsonl",
              "run-config.json", "run_config.json", "config.json", "protocol.json",
              "study_protocol.json", "result.json", "receipt.json", "failure.json"}
_KNOWN_TRAIN_PREFIXES = (
    "runs/20260925-drqv2-geometry-mix-v1-r4/",
    "runs/20260925-drqv2-geometry-mix-v1-r5/",
    "runs/20260925-drqv2-geometry-mix-v1-r6/",
    "runs/20260926-drqv2-retention-r7/",
    "runs/20260927-drqv2-final-source-replay-v1/collection/",
    "runs/20260927-drqv2-final-source-replay-v1/learner-",
    "runs/20260928-drqv2-final-source-replay-reconstruction-r1/learner-",
    "runs/20260926-dreamerv3-reused-train-",
)
_KNOWN_PROTOCOLS = {
    CATALOG_PROTOCOL, R6_PROTOCOL, ACTOR_RECEIPT,
    *(row[1] for row in COMPLETE_TD),
}


class AuditError(ValueError):
    """Evidence is missing, ambiguous or unsafe; no reuse conclusion follows."""


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuditError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _parse(raw: bytes, path: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda item: (_ for _ in ()).throw(AuditError(item)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"{path}: malformed JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path}: expected JSON object")
    return value


def _path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if (not relative or path.is_absolute() or path.as_posix() != relative
            or any(part in (".", "..") for part in relative.split("/")) or "\\" in relative):
        raise AuditError(f"unsafe evidence path: {relative}")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise AuditError(f"symlinked evidence path: {relative}")
    return current


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(root: Path, relative: str, *, cap: int = 64 * 1024 * 1024) -> tuple[dict[str, Any], str]:
    path = _path(root, relative)
    if not path.is_file() or path.stat().st_size > cap:
        raise AuditError(f"{relative}: missing or oversized evidence")
    raw = path.read_bytes()
    return _parse(raw, relative), hashlib.sha256(raw).hexdigest()


def _uint(seed: Any, context: str) -> int:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise AuditError(f"{context}: expected uint32 geometry seed")
    return seed


def _manifest(root: Path, relative: str) -> tuple[list[dict[str, Any]], str]:
    parts = PurePosixPath(relative).parts
    if (len(parts) != 2 or parts[0] != "experiments" or not parts[1].endswith(".json")
            or _PROTECTED.search(parts[1]) or _OUTCOME.search(parts[1])):
        raise AuditError("candidate must be an existing, non-protected experiments/*.json manifest")
    obj, sha = _read(root, relative)
    if (set(obj) != {"format", "purpose", "cells"} or obj["format"] != FORMAT
            or obj["purpose"] != "consumed-TRAIN-development"
            or not isinstance(obj["cells"], list) or len(obj["cells"]) != 24):
        raise AuditError("candidate must contain exactly 24 explicit TRAIN cells; no generated IDs")
    seen: set[int] = set()
    for index, cell in enumerate(obj["cells"]):
        if (not isinstance(cell, dict) or set(cell) != {"partition", "track_id", "geometry_seed", "obstacles"}
                or cell["partition"] != "TRAIN" or type(cell["track_id"]) is not int
                or cell["track_id"] != 1 or cell["obstacles"] is not True):
            raise AuditError(f"cells[{index}]: only track-1 obstacle-enabled TRAIN is allowed")
        seed = _uint(cell["geometry_seed"], f"cells[{index}]")
        if seed in seen:
            raise AuditError(f"cells[{index}]: duplicate geometry seed")
        seen.add(seed)
    if not TD4 <= seen:
        raise AuditError("candidate must retain all four original TD TRAIN geometries")
    return obj["cells"], sha


def _catalog(root: Path, seeds: set[int]) -> tuple[dict[int, str], set[int], list[dict[str, str]]]:
    result, result_sha = _read(root, CATALOG_RESULT)
    protocol, protocol_sha = _read(root, CATALOG_PROTOCOL)
    catalog, catalog_sha = _read(root, CATALOG)
    if (result.get("status") != "success" or result.get("catalog_path") != CATALOG
            or result.get("catalog_sha256") != catalog_sha == CATALOG_SHA256
            or result.get("protocol_sha256") != protocol_sha == CATALOG_PROTOCOL_SHA256
            or catalog.get("protocol_sha256") != protocol_sha or result.get("train_count") != 120
            or result.get("train_diagnostic_count") != 16
            or catalog.get("format") != "haic-drq-training-geometry-catalog-v1"
            or protocol.get("format") != "haic-drq-training-geometry-protocol-v1"):
        raise AuditError("catalog/result/generation protocol identity or SHA mismatch")
    reserved = list(range(3910800001, 3910800513))
    if (protocol.get("candidate_seeds") != reserved
            or catalog.get("seed_audit", {}).get("proposed_seeds") != reserved
            or catalog.get("seed_audit", {}).get("passed") is not True):
        raise AuditError("512-ID catalog reservation or seed audit mismatch")
    exclusions = protocol.get("exclusion_seed_ids")
    if (not isinstance(exclusions, dict) or set(exclusions) != {"blind", "heldout", "reserved"}
            or any(not isinstance(values, list) or any(type(x) is not int for x in values)
                   for values in exclusions.values())):
        raise AuditError("protected/reserved geometry ID metadata unavailable")
    forbidden = set().union(*(set(values) for values in exclusions.values()))
    if seeds & forbidden:
        raise AuditError("candidate aliases a reserved or protected ID across track variants")
    source = catalog.get("source_sha256")
    pinned = protocol.get("source_sha256")
    if not isinstance(source, dict) or set(source) != _GENERATION_SOURCES or not isinstance(pinned, dict):
        raise AuditError("generation source pins missing")
    evidence = [{"path": CATALOG_RESULT, "sha256": result_sha},
                {"path": CATALOG, "sha256": catalog_sha},
                {"path": CATALOG_PROTOCOL, "sha256": protocol_sha}]
    for path, expected in sorted(source.items()):
        actual = _sha(_path(root, path))
        if pinned.get(path) != expected or actual != expected:
            raise AuditError(f"catalog generation source pin mismatch: {path}")
        evidence.append({"path": path, "sha256": actual})
    rules = protocol.get("family_rules")
    if not isinstance(rules, list) or len(rules) != 6:
        raise AuditError("six frozen catalog family rules missing")
    names = [row["name"] for row in rules if isinstance(row, dict) and isinstance(row.get("name"), str)]
    if len(names) != 6 or len(set(names)) != 6:
        raise AuditError("six frozen catalog family rules missing")
    train, diagnostic = catalog.get("train"), catalog.get("train_diagnostic")
    if not isinstance(train, list) or not isinstance(diagnostic, list) or (len(train), len(diagnostic)) != (120, 16):
        raise AuditError("catalog TRAIN and TRAIN-DIAGNOSTIC partitions missing")
    families: dict[int, str] = {}
    diagnostics: set[int] = set()
    signatures: set[str] = set()
    stages: Counter[tuple[str, str]] = Counter()
    for partition, rows in (("TRAIN", train), ("TRAIN-DIAGNOSTIC", diagnostic)):
        for row in rows:
            if not isinstance(row, dict):
                raise AuditError("catalog road row is not typed")
            seed = _uint(row.get("geometry_seed"), "catalog road")
            signature = row.get("road_coordinate_sha256")
            verification = row.get("verification")
            if (seed not in reserved or type(row.get("track_id")) is not int or row["track_id"] != 1
                    or row.get("family") not in names or not isinstance(signature, str)
                    or len(signature) != 64 or signature in signatures
                    or not isinstance(verification, dict)
                    or not all(verification.get(key) is True for key in
                               ("raw_reset", "one_valid_raw_step", "regenerated_coordinate_hash_match"))
                    or seed in families or seed in diagnostics):
                raise AuditError("catalog road has ambiguous partition, family, track or coordinate hash")
            signatures.add(signature)
            if partition == "TRAIN":
                families[seed] = row["family"]
                stage = row.get("stage")
                if stage not in ("representative", "variant"):
                    raise AuditError("catalog TRAIN stage mismatch")
                stages[(row["family"], stage)] += 1
            else:
                diagnostics.add(seed)
                if row.get("stage") != "diagnostic":
                    raise AuditError("TRAIN-DIAGNOSTIC stage mismatch")
    if any(stages[(name, stage)] != 10 for name in names for stage in ("representative", "variant")):
        raise AuditError("catalog family/stage quota mismatch")
    if seeds & diagnostics or not seeds <= families.keys():
        raise AuditError("candidate is not exclusively within the selected catalog TRAIN pool")
    if Counter(families[seed] for seed in seeds) != Counter({name: 4 for name in names}):
        raise AuditError("candidate must have four unique roads in each of six families")
    return families, diagnostics, evidence


def _r6(root: Path, families: dict[int, str], diagnostics: set[int]) -> dict[str, str]:
    r6, sha = _read(root, R6_PROTOCOL)
    source = r6.get("catalog")
    train = r6.get("training_pool")
    diag = r6.get("diagnostic_pool")
    environment = r6.get("environment")
    if (not isinstance(source, dict) or source.get("path") != CATALOG
            or source.get("sha256") != CATALOG_SHA256
            or source.get("protocol_path") != CATALOG_PROTOCOL
            or source.get("protocol_sha256") != CATALOG_PROTOCOL_SHA256
            or not isinstance(train, dict) or not isinstance(train.get("geometry_seeds"), list)
            or len(train["geometry_seeds"]) != 120 or set(train["geometry_seeds"]) != families.keys()
            or train.get("track_ids") != [1, 2, 3, 4]
            or not isinstance(diag, dict) or not isinstance(diag.get("geometry_seeds"), list)
            or len(diag["geometry_seeds"]) != 16 or set(diag["geometry_seeds"]) != diagnostics
            or not isinstance(environment, dict) or environment.get("obstacles") is not True
            or environment.get("track_ids") != [1, 2, 3, 4]):
        raise AuditError("DrQ r6 TRAIN/diagnostic pool or catalog lineage mismatch")
    return {"path": R6_PROTOCOL, "sha256": sha}


def _actor_receipt(root: Path, seeds: set[int], families: dict[int, str]) -> dict[str, str]:
    receipt, sha = _read(root, ACTOR_RECEIPT)
    roads = receipt.get("roads")
    if (receipt.get("format") != "haic-drq-training-geometry-final-set-v1"
            or receipt.get("catalog_path") != CATALOG
            or receipt.get("catalog_sha256") != CATALOG_SHA256
            or receipt.get("catalog_protocol_sha256") != CATALOG_PROTOCOL_SHA256
            or not isinstance(roads, list) or len(roads) != 136):
        raise AuditError("DrQ source actor TRAIN receipt/catalog lineage mismatch")
    visited: set[int] = set()
    for road in roads:
        if not isinstance(road, dict) or type(road.get("geometry_seed")) is not int:
            raise AuditError("DrQ source actor receipt has untyped road")
        seed = road["geometry_seed"]
        if seed not in seeds:
            continue  # Neither noncandidate development outcomes nor diagnostic outcomes are used.
        actors = road.get("frozen_source_actor_results")
        if (seed in visited or road.get("partition") != "TRAIN" or road.get("track_id") != 1
                or road.get("family") != families[seed] or not isinstance(actors, dict)):
            raise AuditError(f"DrQ source actor candidate is ambiguous: {seed}")
        for actor in ("0", "1"):
            row = actors.get(actor)
            if (not isinstance(row, dict) or row.get("cell_id") !=
                    f"train-geometry:train:seed-{seed}:track-1:source-{actor}"
                    or type(row.get("steps")) is not int or row["steps"] <= 0):
                raise AuditError(f"DrQ source actor candidate lacks positive typed interaction: {seed}")
        visited.add(seed)
    if visited != seeds:
        raise AuditError("DrQ source actor receipt omits a proposed TRAIN candidate")
    return {"path": ACTOR_RECEIPT, "sha256": sha}


def _td_rows(root: Path, path: str, *, expected_sha: str | None = None,
             expected_protocol: str | None = None) -> tuple[int, int, str]:
    file = _path(root, path)
    if not file.is_file():
        raise AuditError(f"missing TD training journal: {path}")
    sha = _sha(file)
    if expected_sha is not None and sha != expected_sha:
        raise AuditError(f"TD training journal SHA mismatch: {path}")
    phase = "start"
    cell: tuple[int, int] | None = None
    count = decisions = 0
    with file.open("rb") as stream:
        first = stream.readline()
        start = _parse(first, f"{path}:1")
        if (start.get("event") != "start" or not isinstance(start.get("protocol_sha256"), str)
                or expected_protocol is not None and start["protocol_sha256"] != expected_protocol):
            raise AuditError(f"TD journal has no bound start: {path}")
        for line_number, raw in enumerate(stream, 2):
            row = _parse(raw, f"{path}:{line_number}")
            event = row.get("event")
            if event in ("reset_intent", "reset", "episode"):
                seed = _uint(row.get("geometry_seed"), f"{path}:{line_number}")
                track = row.get("track_id")
                if (type(track) is not int or track != 1 or seed != TD_ORDER[count % 4]
                        or type(row.get("episode")) is not int or row["episode"] != count):
                    raise AuditError(f"TD reset identity or schedule mismatch: {path}:{line_number}")
                identity = (track, seed)
                if (event == "reset_intent" and phase not in ("start", "ended")
                        or event == "reset" and (phase != "intent" or cell != identity)
                        or event == "episode" and (phase != "reset" or cell != identity)):
                    raise AuditError(f"torn TD reset_intent/reset/episode chain: {path}:{line_number}")
                if event == "episode":
                    if type(row.get("length")) is not int or row["length"] <= 0:
                        raise AuditError(f"TD episode has no positive length: {path}:{line_number}")
                    if row.get("decisions") != decisions + row["length"]:
                        raise AuditError(f"TD decision cursor does not join episode length: {path}:{line_number}")
                    count += 1
                    decisions += row["length"]
                phase = {"reset_intent": "intent", "reset": "reset", "episode": "ended"}[event]
                cell = identity
            elif event == "partial":
                if phase not in ("intent", "reset", "ended"):
                    raise AuditError(f"orphan TD partial: {path}:{line_number}")
                phase = "partial"
            elif event not in ("step", "updates", "metric", "diagnostic", "checkpoint"):
                raise AuditError(f"unknown TD journal event: {path}:{line_number}")
            if not raw.endswith(b"\n"):
                raise AuditError(f"torn TD journal EOF: {path}:{line_number}")
    if phase in ("intent", "reset"):
        raise AuditError(f"unclosed TD reset_intent or live episode: {path}")
    return count, decisions, sha


def _complete_td(root: Path, final_path: str, protocol_path: str, run: str) -> list[dict[str, str]]:
    final, final_sha = _read(root, final_path)
    protocol, protocol_sha = _read(root, protocol_path)
    result_path = f"{run}/result.json"
    result, result_sha = _read(root, result_path)
    cells = [{"track_id": 1, "geometry_seed": seed} for seed in (3910800001, 3910800004,
                                                              3910800034, 3910800085)]
    if (final.get("training_protocol") != protocol_path or final.get("training_protocol_sha256") != protocol_sha
            or final.get("run_result") != result_path or final.get("run_result_sha256") != result_sha
            or final.get("run_status") != result.get("status") != "completed_boundary_at_least_100k"
            or final.get("decisions") != result.get("decisions")
            or type(result.get("decisions")) is not int or result["decisions"] < COMPLETE_DECISIONS
            or result.get("updates") != result["decisions"]
            or protocol.get("cells") != cells or protocol.get("episode_schedule") != [0, 1, 2, 3]
            or protocol.get("environment") != {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}
            or protocol.get("run_dir") != run or result.get("protocol_sha256") != protocol_sha
            or result.get("source_sha256") != protocol.get("source_sha256")):
        raise AuditError(f"RAW/DAMAGE complete source lineage mismatch: {run}")
    ledger = f"{run}/training.jsonl"
    steps = f"{run}/steps.jsonl"
    required_ledger_sha = final.get("training_ledger_sha256", final.get("complete_training_ledger_sha256"))
    episodes, decisions, ledger_sha = _td_rows(root, ledger, expected_sha=required_ledger_sha,
                                                expected_protocol=protocol_sha)
    if (episodes != result.get("episodes") or decisions != result.get("decisions")
            or ledger_sha != result.get("training_ledger_sha256")
            or final.get("step_ledger_sha256", final.get("complete_step_ledger_sha256")) != result.get("step_ledger_sha256")
            or _sha(_path(root, steps)) != result["step_ledger_sha256"]):
        raise AuditError(f"RAW/DAMAGE full training receipt/ledger mismatch: {run}")
    source = protocol["source_sha256"]
    producer = "scripts/train_tdmpc2_damage.py" if "damage" in run else "scripts/train_tdmpc2_long.py"
    if not isinstance(source, dict) or source.get(producer) != _sha(_path(root, producer)):
        raise AuditError(f"TD source producer pin mismatch: {producer}")
    return [{"path": name, "sha256": digest} for name, digest in
            ((final_path, final_sha), (protocol_path, protocol_sha), (result_path, result_sha), (ledger, ledger_sha),
             (steps, result["step_ledger_sha256"]))]


def _candidate_token(seeds: set[int]) -> re.Pattern[bytes]:
    return re.compile(rb"(?<![0-9])(?:" + b"|".join(str(s).encode() for s in sorted(seeds)) + rb")(?![0-9])")


def _interval_hit(value: Any, seeds: set[int]) -> bool:
    if isinstance(value, list):
        return any(_interval_hit(item, seeds) for item in value)
    if not isinstance(value, dict):
        return False
    names = {re.sub(r"(?<=[a-z])(?=[A-Z])", "_", key).lower(): item
             for key, item in value.items()}
    for prefix in ("seed", "geometry_seed"):
        for left, right in (("from", "to"), ("start", "end")):
            low, high = names.get(f"{prefix}_{left}"), names.get(f"{prefix}_{right}")
            if type(low) is int and type(high) is int and low <= high:
                if any(low <= seed <= high for seed in seeds):
                    return True
    return any(_interval_hit(child, seeds) for child in value.values())


def _mentions(raw: bytes, token: re.Pattern[bytes], seeds: set[int]) -> bool:
    if token.search(raw):
        return True
    if not any(marker in raw for marker in (b"seedFrom", b"seed_from", b"seedStart", b"seed_start",
                                            b"geometrySeedFrom", b"geometry_seed_from")):
        return False
    try:
        return _interval_hit(_parse(raw, "interval metadata"), seeds)
    except AuditError:
        return False  # Legacy malformed ranges remain under the mandatory coverage blocker.


def _protected_ids(value: Any, seeds: set[int], protected: bool = False) -> bool:
    if isinstance(value, dict):
        return any(_protected_ids(child, seeds,
                                  protected or bool(re.search(r"blind|confirm|screen|held.?out|holdout|private",
                                                              key, re.I)))
                   for key, child in value.items())
    if isinstance(value, list):
        return any(_protected_ids(child, seeds, protected) for child in value)
    return protected and (type(value) is int and value in seeds
                          or isinstance(value, str) and bool(_candidate_token(seeds).search(value.encode())))


def _protocol_metadata(root: Path, seeds: set[int], candidate: str, blockers: list[dict[str, str]],
                       warnings: list[dict[str, str]]) -> None:
    folder = _path(root, "experiments")
    if not folder.is_dir():
        raise AuditError("missing experiments metadata directory")
    token = _candidate_token(seeds)
    for entry in sorted(folder.iterdir()):
        if not entry.name.endswith(".json") or entry.name == PurePosixPath(candidate).name:
            continue
        relative = f"experiments/{entry.name}"
        if entry.is_symlink():
            warnings.append({"path": relative, "reason": "unreviewed metadata symlink"})
            continue
        # These are outcome files, not the ID metadata allowed by this operator.
        if (relative == ACTOR_RECEIPT or _OUTCOME.search(entry.name)
                or _PROTECTED.search(entry.name) and not entry.name.endswith("-protocol.json")):
            continue
        raw = entry.read_bytes()
        if not _mentions(raw, token, seeds):
            continue
        try:
            obj = _parse(raw, relative)
        except AuditError as exc:
            blockers.append({"path": relative, "reason": str(exc)})
            continue
        if _protected_ids(obj, seeds):
            blockers.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                             "reason": "candidate aliases protected ID metadata"})
            continue
        if relative in _KNOWN_PROTOCOLS:
            continue  # Already checked from pinned lineage or separately inventoried below.
        if relative.startswith("experiments/dreamerv3-reused-train-"):
            cells = obj.get("cells")
            r6 = obj.get("r6_protocol")
            if (obj.get("purpose") == "reused-TRAIN-engineering-diagnostic"
                    and isinstance(cells, list) and isinstance(r6, dict)
                    and r6.get("path") == R6_PROTOCOL
                    and r6.get("sha256") == _sha(_path(root, R6_PROTOCOL))
                    and all(isinstance(cell, dict) and type(cell.get("track_id")) is int
                            and cell["track_id"] == 1 and type(cell.get("geometry_seed")) is int
                            for cell in cells)):
                warnings.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "reason": "typed cross-lane consumed Dreamer TRAIN source"})
                continue
        if (re.fullmatch(r"experiments/pixel-rlpd-entropy-target-ablation-v[1-5]\.json", relative)
                and isinstance(obj.get("reserved_training_seeds"), list)
                and seeds <= set(x for x in obj["reserved_training_seeds"] if type(x) is int)):
            warnings.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                             "reason": "historical RLPD prior exclusion, not a TRAIN claim"})
            continue
        blockers.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                         "reason": "candidate appears in unclassified protocol allocation/protected ID metadata"})


def _claims(root: Path, seeds: set[int], blockers: list[dict[str, str]],
            warnings: list[dict[str, str]]) -> None:
    for directory in ("experiments/train-seed-claims", "runs/rlpd-g0-claims"):
        folder = _path(root, directory)
        if not folder.is_dir():
            raise AuditError(f"missing TRAIN claims directory: {directory}")
        token = _candidate_token(seeds)
        for item in sorted(folder.iterdir()):
            relative = f"{directory}/{item.name}"
            if item.name in (".gitkeep", ".lock"):
                continue
            if item.is_symlink() or not item.is_file():
                target = blockers if token.search(relative.encode()) else warnings
                target.append({"path": relative, "reason": "unreviewed claim entry; identity unavailable"})
                continue
            if item.name.startswith("seed-") and item.name.endswith(".json"):
                try:
                    claimed_seed = int(item.name[5:-5])
                except ValueError:
                    claimed_seed = None
                if claimed_seed in seeds:
                    blockers.append({"path": relative, "sha256": _sha(item),
                                     "reason": "active or retired foreign TRAIN claim"})
                    continue
            raw = item.read_bytes()
            if _mentions(raw, token, seeds):
                blockers.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "reason": "candidate in foreign/ambiguous TRAIN claim"})
            elif item.name.endswith(".json"):
                warnings.append({"path": relative, "reason": "unrelated foreign TRAIN claim"})


def _run_inventory(root: Path, seeds: set[int], families: dict[int, str],
                   blockers: list[dict[str, str]], warnings: list[dict[str, str]]
                   ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    folder = _path(root, "runs")
    if not folder.is_dir():
        raise AuditError("missing TRAIN run directory")
    token = _candidate_token(seeds)
    sources: list[dict[str, str]] = []
    exposures: list[dict[str, Any]] = []

    def onerror(exc: OSError) -> None:
        raise AuditError(f"TRAIN run discovery failed: {exc}") from exc

    for current, dirs, files in os.walk(folder, followlinks=False, onerror=onerror):
        location = Path(current)
        for name in dirs[:]:
            child = location / name
            if child.is_symlink():
                relative = child.relative_to(root).as_posix()
                target = blockers if token.search(relative.encode()) else warnings
                target.append({"path": relative, "reason": "unreviewed TRAIN run symlink"})
                dirs.remove(name)
                continue
            if _PROTECTED.search(name) or "diagnostic" in name.lower():
                dirs.remove(name)  # Do not enumerate or read protected outcomes.
        for name in files:
            if name not in _ROAD_FILE:
                continue
            path = location / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                target = blockers if token.search(relative.encode()) else warnings
                target.append({"path": relative, "reason": "unreviewed TRAIN run symlink"})
                continue
            if relative in TD_JOURNALS:
                continue
            with path.open("rb") as stream:
                relevant = any(_mentions(line, token, seeds) for line in stream)
            if not relevant:
                continue
            sources.append({"path": relative, "sha256": _sha(path)})
            if name not in ("episodes.jsonl", "collection.jsonl", "cells.jsonl"):
                blockers.append({"path": relative, "reason": "candidate in unbound run-local or result-only metadata"})
                continue
            if not relative.startswith(_KNOWN_TRAIN_PREFIXES):
                blockers.append({"path": relative, "reason": "candidate in unknown cross-lane TRAIN ledger"})
                continue
            with path.open("rb") as stream:
                lines = stream.readlines()
            if not lines or not lines[-1].endswith(b"\n"):
                blockers.append({"path": relative, "reason": "torn candidate-relevant TRAIN ledger"})
                continue
            for index, line in enumerate(lines, 1):
                if not _mentions(line, token, seeds):
                    continue
                try:
                    row = _parse(line, f"{relative}:{index}")
                    seed = row.get("geometry_seed", row.get("seed"))
                    if (seed not in seeds or type(row.get("track_id")) is not int
                            or row["track_id"] not in (1, 2, 3, 4)
                            or "geometry_seed" in row and "seed" in row and row["seed"] != seed
                            or row.get("partition", "TRAIN") != "TRAIN"
                            or row.get("catalog_sha256", CATALOG_SHA256) != CATALOG_SHA256
                            or row.get("geometry_family", families[seed]) != families[seed]):
                        raise AuditError("untyped or contradictory road identity")
                    exposures.append({"path": relative, "line": index, "geometry_seed": seed,
                                      "track_id": row["track_id"], "exact_track1_cell": row["track_id"] == 1})
                except (AuditError, TypeError) as exc:
                    blockers.append({"path": relative, "reason": f"line {index}: {exc}"})
                    break
    return sources, exposures


def audit(candidate_path: str, *, repo_root: Path = Path(".")) -> dict[str, Any]:
    """Collect source-specific evidence; never clear, reserve, select or reset a road."""
    root = repo_root.resolve(strict=True)
    report: dict[str, Any] = {
        "format": "haic-tdmpc2-consumed-train-reuse-inventory-v1", "status": "BLOCKED",
        "classification": "cross-lane-consumed TRAIN development; never fresh or protected evaluation",
        "candidate_path": candidate_path, "cells": [], "source_inventory": [], "typed_exposures": [],
        "blockers": [], "provenance_warnings": [], "protected_outcome_reads": 0,
        "reservation_or_claim": False, "coverage_certified": False,
    }
    blockers, warnings = report["blockers"], report["provenance_warnings"]
    try:
        cells, report["candidate_sha256"] = _manifest(root, candidate_path)
        report["cells"] = cells
        seeds = {cell["geometry_seed"] for cell in cells}
        families, excluded, source = _catalog(root, seeds)
        report["families"] = dict(Counter(families[s] for s in seeds))
        report["excluded_diagnostic_geometry_seed_count"] = len(excluded)
        report["source_inventory"].extend(source)
        report["source_inventory"].append(_r6(root, families, excluded))
        report["source_inventory"].append(_actor_receipt(root, seeds, families))
    except (AuditError, OSError, TypeError, KeyError) as exc:
        blockers.append({"path": candidate_path, "reason": str(exc)})
        return report
    for final, protocol, run in COMPLETE_TD:
        try:
            report["source_inventory"].extend(_complete_td(root, final, protocol, run))
        except (AuditError, OSError, TypeError, KeyError) as exc:
            blockers.append({"path": final, "reason": str(exc)})
    for path in TD_JOURNALS:
        try:
            count, decisions, sha = _td_rows(root, path)
            report["source_inventory"].append({"path": path, "sha256": sha})
            if path in (row[2] + "/training.jsonl" for row in COMPLETE_TD) and (count == 0 or decisions == 0):
                raise AuditError("completed TD source has no episodes")
        except (AuditError, OSError, TypeError, KeyError) as exc:
            blockers.append({"path": path, "reason": str(exc)})
    try:
        _protocol_metadata(root, seeds, candidate_path, blockers, warnings)
    except (AuditError, OSError, TypeError, KeyError) as exc:
        blockers.append({"path": "experiments", "reason": str(exc)})
    try:
        _claims(root, seeds, blockers, warnings)
    except (AuditError, OSError, TypeError, KeyError) as exc:
        blockers.append({"path": "claims", "reason": str(exc)})
    try:
        sources, exposures = _run_inventory(root, seeds, families, blockers, warnings)
        report["source_inventory"].extend(sources)
        report["typed_exposures"] = exposures
    except (AuditError, OSError, TypeError, KeyError) as exc:
        blockers.append({"path": "runs", "reason": str(exc)})
    blockers.append({"path": "historical_exposure_coverage", "reason":
                     "FAIL-CLOSED: current protected-ID metadata, result-only and legacy positive-interaction "
                     "sources cannot be proven exhaustive by this candidate inventory; independent "
                     "schema-by-schema closure required before any reuse decision"})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, help="existing explicit experiments/*.json manifest")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    report = audit(args.candidate_path, repo_root=args.repo_root)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 2  # No mode of this operator clears a road or exits successfully.


if __name__ == "__main__":
    raise SystemExit(main())
