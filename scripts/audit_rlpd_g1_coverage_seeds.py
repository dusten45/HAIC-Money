"""Read-only, candidate-scoped inventory for a HYPOTHETICAL G1 TRAIN cohort.

``python -B -m scripts.audit_rlpd_g1_coverage_seeds --seed-start N`` prints a
snapshot, not a reservation, receipt, protocol, or permission to reset an env.
The 24 proposed IDs are N+i (i=0..23); collisions are never replaced. No
evaluator episode file, simulator, model, or trainer is imported or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from haic.train_seed_reservations import ReservationError, reserve_train_seeds, validate_train_claim
from scripts.audit_rlpd_g0_seeds import REQUIRED_COLLECTIONS, REQUIRED_LEDGERS, R5_PROFILE


UINT32_MAX = 2**32 - 1
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_AUDIT = "experiments/rlpd-g0-completion-v1-seed-audit.json"
G0_MANIFEST = "runs/20260926-rlpd-g0-completion-v1/manifest.json"
G0_CELLS = "runs/20260926-rlpd-g0-completion-v1/cells.jsonl"
G0_CLAIMS = "runs/rlpd-g0-claims"
CATALOG_PROTOCOL = "experiments/drqv2-geometry-augmentation-v1.json"
CATALOG = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
R7_PROTOCOL = "experiments/drqv2-retention-r7.json"
R6_PROTOCOL = "experiments/drqv2-geometry-mix-v1-r6.json"
R5_ABORT = "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/precheckpoint-abort.json"
R5_LEDGER = "runs/20260925-drqv2-geometry-mix-v1-r5/learner-0-uniform/episodes.jsonl"
R5_G1_ATTESTATION = "experiments/rlpd-g1-r5-source-attestation-v1.json"
R5_ABORT_SHA = "da17593a7853e30c19f37ee53c87de2300198309ceb1b046742bc5f56fe994d3"
R5_LEDGER_SHA = "8d192a9cc87ee3777e3647b2e49eeec7eb66b84eb12099a0ea7c569230889014"
R5_BAD_HASH = "8d192a9cc87ee3777e364f2d49e49ecc7eb66b84eb12099a0ea7c569230889014"
TRAIN_CLAIMS = "experiments/train-seed-claims"
_EXCLUDED = re.compile(r"(^|[-_])(eval(?:uation|uator)?s?|blind|private|confirm(?:ation)?|screen|held[-_]?out|holdout|submission)([-_]|$)", re.I)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_GEOMETRY_KEYS = {"geometry_seed", "geometry_seeds", "training_geometry_seeds",
                  "reserved_training_seeds", "candidate_seeds", "known_excluded_geometry_seeds"}
_LIST_KEYS = _GEOMETRY_KEYS - {"geometry_seed"}
_PASSTHROUGH_FORMATS = {
    "haic-rlpd-g0-r5-receipt-erratum-v1",  # Candidate-bound G0, NEVER a G1 waiver.
}
_REUSED_DREAMER_PROTOCOLS = {
    "dreamerv3-reused-train-diagnostic-v1.json",
    "dreamerv3-reused-train-development-v1.json",
    "dreamerv3-reused-train-diversity-v1.json",
    "dreamerv3-reused-train-source1-v1.json",
    "dreamerv3-reused-train-multisource-v1.json",
    "dreamerv3-reused-train-multisource-development-v1.json",
}


def _directory_scope(name: str) -> str:
    """Classify protected directories; ambiguous names block rather than vanish."""
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    if "unblind" in normalized:
        return "ambiguous"
    if _EXCLUDED.search(name):
        return "protected"
    markers = ("blind", "private", "heldout", "holdout", "evaluation", "evaluator", "eval",
               "confirmation", "confirm", "screen", "submission")
    for marker in markers:
        if normalized.startswith(marker):
            if normalized[len(marker):] in (
                "", "s", "episodes", "episode", "results", "result", "traces",
                "trace", "data", "dataset", "files", "file",
            ):
                return "protected"
            return "ambiguous"
    if any(marker in normalized for marker in markers):
        return "ambiguous"
    return "train"


class InventoryError(ValueError):
    """A source or field cannot be safely interpreted."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii"))


def _uint(value: Any, ref: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise InventoryError(f"{ref}: expected uint32 road ID")
    return value


def _obj(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise InventoryError(f"duplicate JSON key {key}")
        obj[key] = value
    return obj


def _json(raw: bytes, ref: str) -> dict[str, Any]:
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_obj,
                         parse_constant=lambda v: (_ for _ in ()).throw(InventoryError(f"{ref}: {v}")))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise InventoryError(f"{ref}: invalid UTF-8 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise InventoryError(f"{ref}: expected JSON object")
    return obj


def _mapping(value: Any, ref: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InventoryError(f"{ref}: expected object")
    return value


def _sequence(value: Any, ref: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise InventoryError(f"{ref}: expected nonempty list")
    return value


def _mentions_candidate_road(raw: bytes, token: re.Pattern[str], candidates: set[int]) -> bool:
    """Search unknown metadata for road identities, not steps or learner RNGs."""
    text = raw.decode("utf-8", errors="replace")

    def inspect(value: Any, context: str = "") -> bool:
        if isinstance(value, dict):
            if "seed" in context and "range" in context:
                first = value.get("start")
                count = value.get("count")
                if type(first) is int and type(count) is int and count > 0:
                    if any(first <= seed < first + count for seed in candidates):
                        return True
            first = value.get("seed_start", value.get("geometry_seed_start"))
            count = value.get("seed_count", value.get("geometry_seed_count"))
            if type(first) is int and type(count) is int and count > 0:
                if any(first <= seed < first + count for seed in candidates):
                    return True
            return any(inspect(item, f"{context}.{key.casefold()}") for key, item in value.items())
        if isinstance(value, list):
            return any(inspect(item, context) for item in value)
        if isinstance(value, str) and re.search(
                r"(?i)\b(?:geometry[_ -]?seed|road[_ -]?id|seed)\s*[:=]\s*(?:"
                + token.pattern + r")", value):
            return True
        return (any(word in context for word in ("seed", "road", "cell", "geometry"))
                and ((type(value) is int and value in candidates)
                     or (isinstance(value, str) and value.isdecimal() and int(value) in candidates)))

    try:
        if not token.search(text) and "seed_range" not in text and "seed_start" not in text:
            return False
        return inspect(json.loads(text, object_pairs_hook=_obj))
    except (ValueError, InventoryError):
        for line in text.splitlines():
            if not token.search(line) and "seed_range" not in line and "seed_start" not in line:
                continue
            try:
                if inspect(json.loads(line, object_pairs_hook=_obj)):
                    return True
            except (ValueError, InventoryError):
                return True  # Candidate token in an unparseable evidence record.
    return False


def _walk_sources(root: Path) -> tuple[list[str], list[str], list[str], list[str]]:
    """Discover TRAIN ledgers/receipts without descending into held-outs."""
    run_root = root / "runs"
    if run_root.is_symlink() or not run_root.is_dir():
        raise InventoryError("runs: missing or symlinked directory")
    episodes: list[str] = []
    collections: list[str] = []
    start_markers: list[str] = []
    receipts: list[str] = []

    def onerror(exc: OSError) -> None:
        raise InventoryError(f"runs: discovery failed: {exc}") from exc

    for current, dirs, files in os.walk(run_root, followlinks=False, onerror=onerror):
        location = Path(current)
        for dirname in list(dirs):
            child = location / dirname
            if child.is_symlink():
                raise InventoryError(f"{child.relative_to(root)}: unsafe symlink")
            scope = _directory_scope(dirname)
            if scope == "ambiguous":
                raise InventoryError(f"{child.relative_to(root)}: ambiguous protected/TRAIN directory")
            if scope == "protected":
                dirs.remove(dirname)  # Never open or even enumerate blind episode files.
        for filename in files:
            if filename not in ("episodes.jsonl", "collection.jsonl", "run-config.json",
                                "run_config.json", "precheckpoint-abort.json", "collection-result.json"):
                continue
            path = location / filename
            if path.is_symlink():
                raise InventoryError(f"{path.relative_to(root)}: unsafe symlink")
            relative = path.relative_to(root).as_posix()
            if filename == "episodes.jsonl":
                episodes.append(relative)
            elif filename == "collection.jsonl":
                collections.append(relative)
            elif filename == "collection-result.json":
                receipts.append(relative)
            else:
                start_markers.append(relative)
    return sorted(episodes), sorted(collections), sorted(start_markers), sorted(receipts)


class _Inventory:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.raw: dict[str, bytes] = {}
        self.ids: dict[int, list[dict[str, str]]] = {}
        self.blockers: list[dict[str, str]] = []
        self.typed_paths: set[str] = set()

    def block(self, path: str, field: str, reason: str) -> None:
        item = {"path": path, "field": field, "reason": reason}
        if item not in self.blockers:
            self.blockers.append(item)

    def read(self, relative: str) -> bytes:
        parts = relative.split("/")
        if (not relative or PurePosixPath(relative).is_absolute() or "\\" in relative
                or any(part in ("", ".", "..") for part in parts)):
            raise InventoryError(f"{relative}: unsafe source path")
        if relative in self.raw:
            return self.raw[relative]
        path = self.root
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise InventoryError(f"{relative}: unsafe symlink")
        if not path.is_file():
            raise InventoryError(f"{relative}: missing source")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise InventoryError(f"{relative}: unreadable source: {exc}") from exc
        if not raw:
            raise InventoryError(f"{relative}: empty source")
        self.raw[relative] = raw
        return raw

    def record(self, value: Any, path: str, field: str) -> int:
        seed = _uint(value, f"{path}.{field}")
        source = {"path": path, "field": field}
        if source not in self.ids.setdefault(seed, []):
            self.ids[seed].append(source)
        return seed

    def seeds(self, value: Any, path: str, field: str) -> list[int]:
        seeds = [self.record(v, path, f"{field}[{i}]") for i, v in enumerate(_sequence(value, f"{path}.{field}"))]
        if len(seeds) != len(set(seeds)):
            raise InventoryError(f"{path}.{field}: duplicate road ID")
        return seeds

    def fields(self, value: Any, path: str, prefix: str = "", *, rng: bool = False) -> None:
        """Typed seed fields only. Other seed-looking fields are blockers, not guesses."""
        if isinstance(value, list):
            for index, item in enumerate(value):
                self.fields(item, path, f"{prefix}[{index}]", rng=rng)
            return
        if not isinstance(value, dict):
            return
        if (PurePosixPath(path).name.startswith("drqv2-geometry-mix-")
                and re.fullmatch(r"(?:runs|seeds_by_source)\[\d+\]", prefix)):
            schedule = {"actor_rng_seed", "geometry_seed", "learner_seed", "replay_rng_seed",
                        "study_seed", "target_noise_seed", "track_seed", "update_rng_seed"}
            allowed = schedule | ({"run_dir", "source_seed", "variant"} if prefix.startswith("runs[") else set())
            if set(value) != allowed:
                raise InventoryError(f"{path}.{prefix}: unknown mix RNG/road identity schema")
            for key in schedule:
                _uint(value[key], f"{path}.{prefix}.{key}")
            return  # geometry_seed here is a sampler RNG, not a road ID.
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else key
            if key == "rng_seeds":
                if (path != R7_PROTOCOL and "drqv2-geometry-mix-" not in path
                        and not path.startswith("runs/20260926-drqv2-retention-r7/")):
                    raise InventoryError(f"{path}.{field}: unreviewed RNG versus road identity")
                rng_map = _mapping(item, f"{path}.{field}")
                expected = {"track_seed", "geometry_seed", "actor_rng_seed",
                            "replay_rng_seed", "target_noise_seed", "update_rng_seed"}
                if "drqv2-geometry-mix-" in path:
                    expected |= {"learner_seed", "study_seed"}
                    if path.endswith("/run-config.json"):
                        expected |= {"run_dir", "source_seed", "variant"}
                if set(rng_map) != expected:
                    raise InventoryError(f"{path}.{field}: unknown RNG schedule schema")
                if path.endswith("/run-config.json") and "drqv2-geometry-mix-" in path:
                    if (rng_map["run_dir"] != str(PurePosixPath(path).parent)
                            or rng_map["variant"] not in ("uniform", "failure_weighted", "easy_retention")):
                        raise InventoryError(f"{path}.{field}: run RNG identity differs")
                for rng_key, rng_value in rng_map.items():
                    if rng_key in {"run_dir", "variant"}:
                        continue
                    _uint(rng_value, f"{path}.{field}.{rng_key}")
                continue  # Sampler geometry_seed is NOT a road ID.
            if key == "learner_seeds" and prefix == "" and PurePosixPath(path).name.startswith("dreamerv3-b1-"):
                if not isinstance(item, list) or not item or any(type(seed) is not int for seed in item):
                    raise InventoryError(f"{path}.{field}: invalid learner RNG seeds")
                continue
            if key == "seeds" and prefix == "learner" and PurePosixPath(path).name.startswith("dreamerv3-reused-train-"):
                if not isinstance(item, list) or not item or any(type(seed) is not int for seed in item):
                    raise InventoryError(f"{path}.{field}: invalid learner RNG seeds")
                continue
            if key == "exclusion_seed_ids":
                excluded = _mapping(item, f"{path}.{field}")
                if set(excluded) != {"blind", "heldout", "reserved"}:
                    raise InventoryError(f"{path}.{field}: unknown protected geometry exclusions")
                self.fields(excluded, path, field)
                continue
            if key == "structural_reference_seeds":
                for index, row in enumerate(_sequence(item, f"{path}.{field}")):
                    ref = _mapping(row, f"{path}.{field}[{index}]")
                    if type(ref.get("consumed")) is not bool or "geometry_seed" not in ref:
                        raise InventoryError(f"{path}.{field}[{index}]: ambiguous reference road")
                    self.fields(ref, path, f"{field}[{index}]")
                continue
            if key in _LIST_KEYS:
                self.seeds(item, path, field)
            elif key == "geometry_seed":
                self.record(item, path, field)
            elif key == "last_geometry_seed":
                self.record(item, path, field)
            elif key == "seeds" and (prefix.startswith("partitions.") or prefix.startswith("training_pools.")
                                     or prefix == "training_development" or prefix.startswith("exclusion_seed_ids")
                                     or prefix.startswith("future_full_reservation.")):
                self.seeds(item, path, field)
            elif key in ("seed", "reserved", "heldout", "blind") and (
                    (key == "seed" and (prefix.startswith("training_development.cells[")
                                        or prefix.startswith("cells[")))
                    or (key != "seed" and prefix == "exclusion_seed_ids")):
                if key == "seed":
                    self.record(item, path, field)
                else:
                    self.seeds(item, path, field)
            elif (key in ("seeds", "seed_ids", "road_ids") or key.endswith(
                ("_seeds", "_seed_ids", "_road_ids", "_seed_range")
            )):
                raise InventoryError(f"{path}.{field}: unknown seed-bearing field")
            elif "geometry_seed" in key or key in (
                    "seed_range", "seed_start", "road_seeds", "road_id", "road_seed"):
                raise InventoryError(f"{path}.{field}: unknown seed-bearing field")
            else:
                self.fields(item, path, field, rng=rng)

    def protocol(self, path: str, obj: dict[str, Any]) -> None:
        fmt = obj.get("format")
        name = PurePosixPath(path).name
        if name == "drqv2-retention-r7.json":
            if fmt != "haic-drq-retention-study-v1" or len(_sequence(obj.get("runs"), f"{path}.runs")) != 12:
                raise InventoryError(f"{path}: unknown r7 protocol schema/runs")
            if (obj.get("r6_protocol_path") != R6_PROTOCOL
                    or obj.get("r6_protocol_sha256") != _sha(self.read(R6_PROTOCOL))):
                raise InventoryError(f"{path}.r6_protocol_sha256: r6 TRAIN allocation source drift")
            run_dirs: set[str] = set()
            matrix: set[tuple[int, str, str]] = set()
            for i, run in enumerate(obj["runs"]):
                run = _mapping(run, f"{path}.runs[{i}]")
                run_dir = run.get("run_dir")
                cell = (run.get("source_seed"), run.get("variant"), run.get("condition"))
                if (not isinstance(run_dir, str) or not run_dir.startswith("runs/20260926-drqv2-retention-r7/learner-")
                        or run_dir in run_dirs or cell in matrix or type(run.get("source_seed")) is not int
                        or "rng_seeds" not in run):
                    raise InventoryError(f"{path}.runs[{i}].run_dir: unknown allocation")
                run_dirs.add(run_dir)
                matrix.add(cell)
            if matrix != {(s, v, c) for s in (0, 1) for v in
                          ("uniform", "failure_weighted", "easy_retention") for c in ("r7a", "r7b")}:
                raise InventoryError(f"{path}.runs: missing/duplicate r7 TRAIN allocation")
        elif name == "rlpd-g0-completion-v1.json":
            if fmt != "haic-rlpd-g0-diagnostic-v1" or obj.get("partition") != "TRAIN":
                raise InventoryError(f"{path}: unknown G0 protocol schema")
        elif name == "drqv2-geometry-augmentation-v1.json":
            if fmt != "haic-drq-training-geometry-protocol-v1":
                raise InventoryError(f"{path}: unknown DrQ catalog protocol")
        elif name in _REUSED_DREAMER_PROTOCOLS:
            expected = ("haic-dreamerv3-reused-train-diagnostic-v1" if name in {
                "dreamerv3-reused-train-diagnostic-v1.json",
                "dreamerv3-reused-train-development-v1.json",
                "dreamerv3-reused-train-diversity-v1.json",
                "dreamerv3-reused-train-multisource-development-v1.json",
            } else PurePosixPath(name).stem.replace("dreamerv3-", "haic-dreamerv3-"))
            if fmt != expected:
                raise InventoryError(f"{path}: unknown reused Dreamer TRAIN protocol schema")
        elif name.startswith("pixel-rlpd-"):
            if fmt != "haic-pixel-rlpd-study-v1":
                raise InventoryError(f"{path}: unknown RLPD protocol schema")
        elif name.startswith("drqv2-geometry-mix-"):
            if fmt != "haic-drq-geometry-mix-study-v1":
                raise InventoryError(f"{path}: unknown DrQ mix protocol schema")
        elif name.startswith("dreamerv3-b1-"):
            if fmt != "haic-dreamerv3-study-protocol-v1":
                raise InventoryError(f"{path}: unknown Dreamer B1 protocol schema")
        elif name.startswith("drqv2-teacher-replay-"):
            if not str(obj.get("study_id", "")).startswith("drqv2-teacher-replay-v1"):
                raise InventoryError(f"{path}: unknown teacher replay protocol schema")
        elif fmt in _PASSTHROUGH_FORMATS:
            return
        else:
            raise InventoryError(f"{path}: unreviewed protocol schema {fmt!r}")
        self.fields(obj, path)
        if name == "drqv2-geometry-augmentation-v1.json" and obj.get("candidate_seeds") != list(range(3910800001, 3910800513)):
            raise InventoryError(f"{path}.candidate_seeds: incomplete 512-road reservation")
        self.typed_paths.add(path)

    def ledger(self, path: str, *, collection: bool = False) -> None:
        lines = self.read(path).splitlines()
        if not lines:
            raise InventoryError(f"{path}: empty TRAIN ledger")
        pending: tuple[int, int, int | None] | None = None
        for index, line in enumerate(lines, 1):
            obj = _json(line, f"{path}:{index}")
            event = obj.get("event")
            if event not in (("reset", "stored_episode", "discarded_incomplete_episode") if collection
                             else ("reset", "end", "budget-stop")):
                raise InventoryError(f"{path}:{index}.event: unknown TRAIN ledger schema {event!r}")
            for key in obj:
                if (("seed" in key or key == "road_ids") and key not in
                        ("seed", "geometry_seed", "source_seed", "rng_seed", "actor_rng_seed")
                        or key in ("road_id", "road_ids")):
                    raise InventoryError(f"{path}:{index}.{key}: unknown seed-bearing ledger field")
            if collection:
                reset_keys = {"event", "cell_index", "episode_id", "geometry_seed",
                              "track_id", "obstacles"}
                outcome_keys = {"event", "episode_id", "geometry_seed", "track_id"}
                if ((event == "reset" and set(obj) != reset_keys)
                        or (event != "reset" and not outcome_keys <= set(obj))):
                    raise InventoryError(f"{path}:{index}: unknown prior-data collection identity schema")
                if type(obj.get("episode_id")) is not int or obj["episode_id"] < 0:
                    raise InventoryError(f"{path}:{index}.episode_id: invalid collection identity")
                if event == "reset" and (type(obj.get("cell_index")) is not int
                                         or obj["cell_index"] != obj["episode_id"]):
                    raise InventoryError(f"{path}:{index}.cell_index: invalid collection reset")
            track = obj.get("track_id")
            if type(track) is not int or not 1 <= track <= 4:
                raise InventoryError(f"{path}:{index}.track_id: invalid track")
            seeds = [self.record(obj[key], path, f"line:{index}.{key}") for key in
                     ("geometry_seed", "seed") if key in obj]
            if not seeds or len(set(seeds)) != 1:
                raise InventoryError(f"{path}:{index}: missing or disagreeing road seed fields")
            if collection and event == "reset" and obj.get("obstacles") is not True:
                raise InventoryError(f"{path}:{index}.obstacles: expected TRAIN obstacles true")
            identity = (seeds[0], track, obj.get("episode_id") if collection else None)
            if event == "reset":
                if pending is not None:
                    raise InventoryError(f"{path}:{index}: previous reset has no outcome")
                pending = identity
            elif event == "budget-stop":
                if pending is None or pending != identity:
                    raise InventoryError(f"{path}:{index}: orphan or mismatched budget-stop")
                pending = None
            else:
                if pending != identity:
                    raise InventoryError(f"{path}:{index}: orphan or mismatched outcome")
                pending = None
                if event == "discarded_incomplete_episode" and index != len(lines):
                    raise InventoryError(f"{path}:{index}: discard must terminate collection")
        self.typed_paths.add(path)  # Seed identity is known even when the run stopped mid-episode.
        if pending is not None:
            rlpd_study = next((study for study in (
                "20260925-pixel-rlpd-entropy-target-ablation-v5",
                "20260924-pixel-rlpd-long-horizon-followup-v1",
            ) if path.startswith(f"runs/{study}/")), None)
            if rlpd_study is not None:
                run_dir = str(PurePosixPath(path).parent)
                receipt_path = f"{run_dir}/result.json"
                receipt = _json(self.read(receipt_path), receipt_path)
                study_id = rlpd_study.split("-", 1)[1]
                protocol = f"experiments/{study_id}.json"
                candidates = receipt.get("candidates")
                last = candidates[-1] if isinstance(candidates, list) and candidates else {}
                previous = _json(lines[-2], f"{path}:previous") if len(lines) > 1 else {}
                if (receipt.get("format") != ("haic-rlpd-entropy-run-result-v1" if "entropy" in study_id
                                               else "haic-rlpd-run-result-v1")
                        or receipt.get("study_id") != study_id
                        or not isinstance(last, dict)
                        or type(receipt.get("environment_steps")) is not int
                        or receipt["environment_steps"] <= 0
                        or last.get("protocol_sha256") != _sha(self.read(protocol))
                        or last.get("environment_steps") != receipt.get("environment_steps")
                        or type(last.get("resume_episode_steps")) is not int
                        or last["resume_episode_steps"] <= 0
                        or type(previous.get("global_step")) is not int
                        or previous.get("event") != "end"
                        or previous["global_step"] + last["resume_episode_steps"] != receipt["environment_steps"]
                        or pending[0] not in self.ids):
                    raise InventoryError(f"{path}: final reset lacks matching SHA-bound budget-cap result")
                return  # The final reset and all partial interaction remain consumed.
            if path.startswith("runs/20260926-drqv2-retention-r7/"):
                run_dir = str(PurePosixPath(path).parent)
                receipt_path = f"{run_dir}/result.json"
                receipt = _json(self.read(receipt_path), receipt_path)
                config_path = f"{run_dir}/run-config.json"
                config = _json(self.read(config_path), config_path)
                final = receipt.get("candidates", [])
                final = final[-1] if isinstance(final, list) and final else {}
                reset = _json(lines[-1], f"{path}:last reset")
                previous = _json(lines[-2], f"{path}:previous") if len(lines) > 1 else {}
                if (receipt.get("format") != "haic-drq-retention-run-result-v1"
                        or receipt.get("completed") is not True
                        or receipt.get("study_id") != "drqv2-retention-r7"
                        or receipt.get("study_protocol_sha256") != _sha(self.read(R7_PROTOCOL))
                        or type(receipt.get("additional_online_steps")) is not int
                        or receipt["additional_online_steps"] <= 0
                        or config.get("format") != "haic-drq-retention-run-config-v1"
                        or config.get("protocol_sha256") != receipt["study_protocol_sha256"]
                        or not isinstance(final, dict)
                        or final.get("checkpoint_online_step") != receipt.get("additional_online_steps")
                        or type(reset.get("additional_online_step")) is not int
                        or previous.get("event") != "end"
                        or previous.get("additional_online_step") != reset["additional_online_step"]
                        or not 0 <= reset["additional_online_step"] < receipt["additional_online_steps"]):
                    raise InventoryError(f"{path}: final reset lacks matching SHA-bound r7 budget-cap result")
                return  # Final checkpoint was written mid-episode; reset is consumed.
            if path not in (
                "runs/20260925-drqv2-geometry-mix-v1-r4/learner-0-uniform/episodes.jsonl",
                R5_LEDGER,
            ):
                raise InventoryError(
                    f"{path}: unclosed final reset; no typed cap/partial receipt verified "
                    "(live/incomplete use cannot be cleared)")
            abort_path = str(PurePosixPath(path).with_name("precheckpoint-abort.json"))
            abort = _json(self.read(abort_path), abort_path)
            protocol_path = ("experiments/drqv2-geometry-mix-v1-r4.json"
                             if path != R5_LEDGER else "experiments/drqv2-geometry-mix-v1-r5.json")
            if (
                abort.get("format") != "haic-drq-geometry-mix-partial-arm-abort-v1"
                or abort.get("run_dir") != str(PurePosixPath(path).parent)
                or abort.get("protocol_path") != protocol_path
                or abort.get("protocol_sha256") != _sha(self.read(protocol_path))
                or abort.get("episode_boundary_at_stop") is not False
                or abort.get("last_geometry_seed") != pending[0]
                or abort.get("resume_allowed") is not False
                or abort.get("evaluation_receipts") != []
                or abort.get("episodes_sha256") != (
                    R5_BAD_HASH if path == R5_LEDGER else _sha(self.raw[path])
                )
                or path == R5_LEDGER and _sha(self.raw[path]) != R5_LEDGER_SHA
                or path == R5_LEDGER and _sha(self.raw[abort_path]) != R5_ABORT_SHA
            ):
                raise InventoryError(f"{path}: unclosed reset lacks matching typed partial-abort receipt")


def audit_g1_coverage_seeds(
    seed_start: int, *, repo_root: Path = Path("."),
    expected_experiments_sha256: str | None = None,
    expected_sources_sha256: str | None = None,
    r5_attestation_sha256: str | None = None,
    self_study_id: str | None = None,
    self_protocol_path: str | None = None,
    self_protocol_sha256: str | None = None,
    required_ledgers: Sequence[str] = REQUIRED_LEDGERS,
    required_collections: Sequence[str] = tuple(REQUIRED_COLLECTIONS),
) -> dict[str, Any]:
    """Return BLOCKED or no_known_recorded_overlap for a fixed hypothetical cohort.

    Global snapshot pins are advisory; specific frozen sources still retain SHA.
    Self-claim mode requires a separately frozen exact protocol and claim digest.
    Source-list overrides are for synthetic fixtures only. This function writes
    nothing and does not make any seed allocation or freshness guarantee.
    """
    start = _uint(seed_start, "seed_start")
    if start > UINT32_MAX - 23:
        raise InventoryError("seed_start: 24-road batch exceeds uint32")
    candidates = list(range(start, start + 24))
    proposed_cells = [{"track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}
                      for seed in candidates]
    if self_study_id is not None:
        if (not self_study_id.strip() or self_protocol_path is None
                or self_protocol_sha256 is None or not _SHA.fullmatch(self_protocol_sha256)
                or PurePosixPath(self_protocol_path).parts != ("experiments", PurePosixPath(self_protocol_path).name)
                or not self_protocol_path.endswith(".json")):
            raise InventoryError("self-claim check requires study ID and frozen experiments protocol path/SHA")
    elif self_protocol_path is not None or self_protocol_sha256 is not None:
        raise InventoryError("self-protocol path/SHA require an explicit study ID")
    root = Path(repo_root).resolve()
    inv = _Inventory(root)
    if root == Path(__file__).resolve().parents[1] and (
            tuple(required_ledgers) != tuple(REQUIRED_LEDGERS)
            or tuple(required_collections) != tuple(REQUIRED_COLLECTIONS)):
        inv.block("runs", "source-list overrides", "synthetic-only inventory overrides cannot pass repository preflight")
    experiments: list[str] = []
    try:
        directory = root / "experiments"
        if directory.is_symlink() or not directory.is_dir():
            raise InventoryError("experiments: missing or symlinked directory")
        experiments = sorted(f"experiments/{entry.name}" for entry in directory.iterdir()
                             if entry.suffix == ".json")
        for path in experiments:
            try:
                raw = inv.read(path)
                obj = _json(raw, path)
            except InventoryError as exc:
                inv.block(path, "JSON", str(exc))
                continue
            name = PurePosixPath(path).name
            if path == self_protocol_path:
                continue  # SHA, exact cells and claim chain checked below.
            if (name == G0_PROTOCOL.split("/")[-1] or name == CATALOG_PROTOCOL.split("/")[-1]
                    or name == R7_PROTOCOL.split("/")[-1]
                    or name in _REUSED_DREAMER_PROTOCOLS
                    or (name.startswith(("pixel-rlpd-", "drqv2-geometry-mix-", "drqv2-teacher-replay-",
                                          "dreamerv3-b1-"))
                        and not any(marker in name for marker in
                                    ("-result", "-analysis", "-diagnostic", "-summary", "-preflight",
                                     "-erratum", "-score", "-sampling", "-posterior", "-budget"))
                        and not name.endswith("-seed-audit.json"))):
                try:
                    inv.protocol(path, obj)
                except InventoryError as exc:
                    inv.block(path, "schema", str(exc))
            elif name == G0_AUDIT.split("/")[-1]:
                if obj.get("format") != "haic-rlpd-g0-seed-audit-v1":
                    inv.block(path, "format", "unknown G0 audit schema")
                else:
                    try:
                        inv.seeds(obj.get("candidate_seeds"), path, "candidate_seeds")
                        inv.typed_paths.add(path)
                    except InventoryError as exc:
                        inv.block(path, "candidate_seeds", str(exc))
            elif name == R5_G1_ATTESTATION.split("/")[-1]:
                pass  # Verified separately below, never as a source of road IDs.
            elif name.endswith("-result.json") or name.endswith("-preflight.json"):
                pass  # Outcome metadata never substitutes for a typed allocation.
            elif obj.get("format") in _PASSTHROUGH_FORMATS:
                pass  # Historical G0 erratum is intentionally NOT a G1 attestation.
            else:
                inv.block(path, "format", f"unreviewed experiment JSON schema {obj.get('format')!r}")
    except InventoryError as exc:
        inv.block("experiments", "discovery", str(exc))

    for path in (G0_PROTOCOL, G0_AUDIT, CATALOG_PROTOCOL, R7_PROTOCOL):
        if path not in inv.raw:
            inv.block(path, "source", "missing mandatory typed protocol")
    try:
        g0 = _json(inv.read(G0_PROTOCOL), G0_PROTOCOL)
        audit = _json(inv.read(G0_AUDIT), G0_AUDIT)
        seeds = list(range(4272000001, 4272000013))
        cells = [{"track_id": 1, "geometry_seed": seed, "partition": "TRAIN", "obstacles": True}
                 for seed in seeds]
        if g0.get("cells") != cells or audit.get("cells") != cells or audit.get("candidate_seeds") != seeds:
            raise InventoryError("G0 protocol/audit: 12 consumed TRAIN cells differ")
        if g0.get("geometry_audit_sha256") != _sha(inv.read(G0_AUDIT)):
            raise InventoryError("G0 protocol.geometry_audit_sha256: source drift")
        for seed in seeds:
            inv.record(seed, G0_PROTOCOL, "cells.geometry_seed")
    except InventoryError as exc:
        inv.block(G0_PROTOCOL, "cells", str(exc))
    try:
        protocol = _json(inv.read(CATALOG_PROTOCOL), CATALOG_PROTOCOL)
        catalog = _json(inv.read(CATALOG), CATALOG)
        proposed = _mapping(catalog.get("seed_audit"), f"{CATALOG}.seed_audit").get("proposed_seeds")
        if (protocol.get("candidate_seeds") != list(range(3910800001, 3910800513))
                or proposed != protocol["candidate_seeds"] or catalog.get("protocol_sha256") != _sha(inv.read(CATALOG_PROTOCOL))):
            raise InventoryError(f"{CATALOG}.seed_audit.proposed_seeds: missing/drifted full 512-road reservation")
        inv.seeds(proposed, CATALOG, "seed_audit.proposed_seeds")
    except InventoryError as exc:
        inv.block(CATALOG, "seed_audit", str(exc))
    # Preserve candidate IDs from G0 primary rows and the separate claim even if
    # their manifest cross-hash is damaged; chain failure is still reported below.
    try:
        claim_dir = root / G0_CLAIMS
        if claim_dir.is_dir() and not claim_dir.is_symlink():
            for entry in sorted(claim_dir.iterdir()):
                if entry.suffix == ".json":
                    relative = f"{G0_CLAIMS}/{entry.name}"
                    claim = _json(inv.read(relative), relative)
                    inv.seeds(claim.get("geometry_seeds"), relative, "geometry_seeds")
        for index, row in enumerate(inv.read(G0_CELLS).splitlines(), 1):
            obj = _json(row, f"{G0_CELLS}:{index}")
            if "geometry_seed" in obj:
                inv.record(obj["geometry_seed"], G0_CELLS, f"line:{index}.geometry_seed")
            else:
                raise InventoryError(f"{G0_CELLS}:{index}: missing road identity")
    except InventoryError as exc:
        inv.block(G0_CELLS, "G0 primary cells/claims", str(exc))
    try:
        manifest = _json(inv.read(G0_MANIFEST), G0_MANIFEST)
        claim_dir = root / G0_CLAIMS
        if claim_dir.is_symlink() or not claim_dir.is_dir():
            raise InventoryError(f"{G0_CLAIMS}: missing/unsafe G0 pool claim")
        claim_paths = sorted(f"{G0_CLAIMS}/{p.name}" for p in claim_dir.iterdir() if p.suffix == ".json")
        if len(claim_paths) != 1:
            raise InventoryError(f"{G0_CLAIMS}: expected exactly one G0 pool claim")
        claim = _json(inv.read(claim_paths[0]), claim_paths[0])
        rows = inv.read(G0_CELLS)
        if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
                or manifest.get("cell_count") != 24 or manifest.get("geometry_count") != 12
                or manifest.get("protocol_sha256") != _sha(inv.read(G0_PROTOCOL))
                or manifest.get("geometry_audit_sha256") != _sha(inv.read(G0_AUDIT))
                or manifest.get("cells_sha256") != _sha(rows)
                or claim.get("format") != "haic-rlpd-g0-geometry-claim-v1"
                or claim.get("status") != "reserved-once"
                or claim.get("geometry_seeds") != list(range(4272000001, 4272000013))
                or claim.get("protocol_sha256") != manifest["protocol_sha256"]
                or claim.get("geometry_audit_sha256") != manifest["geometry_audit_sha256"]):
            raise InventoryError("G0 manifest/claim/protocol/cells hash or pool mismatch")
        if len(rows.splitlines()) != 24:
            raise InventoryError(f"{G0_CELLS}: incomplete 24 TRAIN rows")
        actors = [actor.get("id") for actor in _sequence(g0.get("actors"), f"{G0_PROTOCOL}.actors")]
        if actors != ["long-horizon-seed11", "entropy-v5-author-seed50"]:
            raise InventoryError(f"{G0_PROTOCOL}.actors: two original G0 actors missing")
        for index, line in enumerate(rows.splitlines(), 1):
            cell = _json(line, f"{G0_CELLS}:{index}")
            if (cell.get("partition") != "TRAIN" or cell.get("track_id") != 1
                    or cell.get("geometry_seed") != 4272000001 + (index - 1) // 2
                    or cell.get("actor_id") != actors[(index - 1) % 2]):
                raise InventoryError(f"{G0_CELLS}:{index}: unexpected TRAIN cell")
            inv.record(cell["geometry_seed"], G0_CELLS, f"line:{index}.geometry_seed")
    except InventoryError as exc:
        inv.block(G0_MANIFEST, "cells/claim", str(exc))

    self_claim_sources: list[dict[str, str]] = []
    self_claims_verified = False
    try:
        claims = root / TRAIN_CLAIMS
        if claims.is_symlink() or not claims.is_dir():
            raise InventoryError(f"{TRAIN_CLAIMS}: missing or symlinked shared TRAIN registry")
        for entry in sorted(claims.iterdir()):
            if entry.name == ".gitkeep":
                if entry.is_symlink() or not entry.is_file() or entry.stat().st_size:
                    raise InventoryError(f"{TRAIN_CLAIMS}: unsafe registry marker")
                continue
            if not re.fullmatch(r"seed-(0|[1-9][0-9]*)\.json", entry.name):
                raise InventoryError(f"{TRAIN_CLAIMS}/{entry.name}: unknown claim entry")
            path = f"{TRAIN_CLAIMS}/{entry.name}"
            claim = _json(inv.read(path), path)
            seed = _uint(int(entry.name[5:-5]), path)
            try:
                validate_train_claim(claim, seed)
            except ReservationError as exc:
                raise InventoryError(f"{path}: {exc}") from exc
            inv.typed_paths.add(path)
            if self_study_id is not None and seed in candidates:
                cell = proposed_cells[seed - start]
                if (claim["study_id"] != self_study_id or claim["protocol_id"] != self_study_id
                        or any(claim[key] != cell[key] for key in cell)
                        or claim["protocol_path"] not in (None, self_protocol_path)
                        or claim["protocol_sha256"] not in (None, self_protocol_sha256)
                        or not claim["audit_source"].startswith("haic-rlpd-g1-coverage-seed-inventory-v2:")):
                    raise InventoryError(f"{path}: cannot waive another or mismatched TRAIN claim")
                self_claim_sources.append({"path": path, "sha256": _sha(inv.raw[path])})
            else:
                inv.record(seed, path, "geometry_seed")
        if self_study_id is not None:
            protocol = _json(inv.read(self_protocol_path), self_protocol_path)
            if (len(self_claim_sources) != len(candidates)
                    or _sha(inv.raw[self_protocol_path]) != self_protocol_sha256
                    or set(protocol) - {"format", "status", "partition", "study_id", "cells",
                                        "train_claims_sha256", "source_hashes", "source_actors",
                                        "budget", "runtime", "exclusions", "track_id", "obstacles"}
                    or protocol.get("format") != "haic-rlpd-g1-coverage-protocol-v1"
                    or protocol.get("status") != "frozen"
                    or protocol.get("partition") != "TRAIN"
                    or protocol.get("study_id") != self_study_id
                    or protocol.get("cells") != proposed_cells
                    or protocol.get("train_claims_sha256") != _digest(sorted(
                        self_claim_sources, key=lambda source: source["path"]))):
                raise InventoryError("frozen self-protocol/24 claims SHA or exact cells disagree")
            inv.typed_paths.add(self_protocol_path)
            self_claims_verified = True
    except (InventoryError, OSError) as exc:
        inv.block(TRAIN_CLAIMS, "claims", str(exc))

    try:
        episode_paths, collection_paths, marker_paths, receipt_paths = _walk_sources(root)
    except InventoryError as exc:
        inv.block("runs", "discovery", str(exc))
        episode_paths, collection_paths, marker_paths, receipt_paths = [], [], [], []
    ledger_dirs = {str(PurePosixPath(path).parent) for path in episode_paths + collection_paths}
    for marker in marker_paths:
        try:
            record = _json(inv.read(marker), marker)
            inv.fields(record, marker)
            inv.typed_paths.add(marker)
        except InventoryError as exc:
            inv.block(marker, "start-marker", str(exc))
        if str(PurePosixPath(marker).parent) not in ledger_dirs:
            inv.block(marker, "started TRAIN run", "start/abort marker has no TRAIN episode or collection ledger")
    for path in set(required_ledgers) - set(episode_paths):
        inv.block(path, "episodes.jsonl", "missing required TRAIN ledger")
    for path in set(required_collections) - set(collection_paths):
        inv.block(path, "collection.jsonl", "missing required prior-data collection ledger")
    for path in episode_paths:
        try:
            inv.ledger(path)
        except InventoryError as exc:
            inv.block(path, "ledger", str(exc))
    for path in collection_paths:
        try:
            inv.ledger(path, collection=True)
        except InventoryError as exc:
            inv.block(path, "collection", str(exc))
    for path in receipt_paths:
        try:
            obj = _json(inv.read(path), path)
            inv.fields(obj, path)
            inv.typed_paths.add(path)
        except InventoryError as exc:
            inv.block(path, "collection receipt", str(exc))
    try:
        r7 = _json(inv.read(R7_PROTOCOL), R7_PROTOCOL)
        for i, run in enumerate(_sequence(r7.get("runs"), f"{R7_PROTOCOL}.runs")):
            directory = _mapping(run, f"{R7_PROTOCOL}.runs[{i}]").get("run_dir")
            if not isinstance(directory, str) or not directory.startswith("runs/20260926-drqv2-retention-r7/learner-"):
                raise InventoryError(f"{R7_PROTOCOL}.runs[{i}].run_dir: unknown r7 run")
            if (root / directory).exists() and f"{directory}/episodes.jsonl" not in episode_paths:
                inv.block(directory, "episodes.jsonl", "started r7 run missing TRAIN ledger")
    except InventoryError as exc:
        inv.block(R7_PROTOCOL, "runs", str(exc))

    r5_verified = False
    try:
        abort = _json(inv.read(R5_ABORT), R5_ABORT)
        actual = _sha(inv.read(R5_LEDGER))
        if (_sha(inv.read(R5_ABORT)) != R5_ABORT_SHA or actual != R5_LEDGER_SHA
                or abort.get("episodes_sha256") != R5_BAD_HASH or len(R5_BAD_HASH) != 65):
            raise InventoryError("r5 historical abort/ledger bytes or malformed 65-hex reference changed")
        # This is independent verification of CURRENT preserved TRAIN evidence,
        # not a repair of the frozen abort receipt's invalid episodes_sha256.
        if R5_ABORT_SHA == R5_PROFILE["attempt_receipt_sha256"]:
            for name, sha in (("protocol", "protocol_sha256"),
                              ("supersession", "supersession_sha256"),
                              ("step_metrics", "step_metrics_sha256"),
                              ("trace", "trace_sha256")):
                if _sha(inv.read(str(R5_PROFILE[f"{name}_path"]))) != R5_PROFILE[sha]:
                    raise InventoryError(f"r5 {name} preserved evidence SHA changed")
            supersession = _json(inv.raw[str(R5_PROFILE["supersession_path"])], "r5 supersession")
            metrics = inv.raw[str(R5_PROFILE["step_metrics_path"])].splitlines()
            last_metric = _json(metrics[-1], "r5 last metric") if metrics else {}
            resets = [_json(row, "r5 reset") for row in inv.raw[R5_LEDGER].splitlines()
                      if b'"event": "reset"' in row]
            if (abort.get("protocol_path") != R5_PROFILE["protocol_path"]
                    or abort.get("protocol_sha256") != R5_PROFILE["protocol_sha256"]
                    or abort.get("step_metrics_sha256") != R5_PROFILE["step_metrics_sha256"]
                    or abort.get("online_replay_sample_trace_sha256") != R5_PROFILE["trace_sha256"]
                    or abort.get("last_geometry_seed") != R5_PROFILE["last_geometry_seed"]
                    or abort.get("last_episode_step") != R5_PROFILE["last_episode_step"]
                    or abort.get("online_decisions") != R5_PROFILE["online_decisions"]
                    or abort.get("learner_updates") != R5_PROFILE["learner_updates"]
                    or abort.get("resume_allowed") is not False
                    or abort.get("evaluation_receipts") != []
                    or len(resets) != R5_PROFILE["ordered_reset_count"]
                    or resets[-1].get("seed") != abort["last_geometry_seed"]
                    or len(metrics) != abort["online_decisions"]
                    or last_metric.get("additional_online_step") != abort["online_decisions"]
                    or last_metric.get("geometry_seed") != abort["last_geometry_seed"]
                    or last_metric.get("episode_step") != abort["last_episode_step"]
                    or supersession.get("attempt_receipt_sha256") != R5_ABORT_SHA
                    or supersession.get("status") != "partial-train-attempt-incomplete"
                    or supersession.get("resume_allowed") is not False):
                raise InventoryError("r5 abort/supersession/metrics/ledger identity differs")
        if r5_attestation_sha256 is not None:
            if not _SHA.fullmatch(r5_attestation_sha256):
                raise InventoryError("G1-scoped attestation SHA-256 must be 64 lowercase hex")
            att = _json(inv.read(R5_G1_ATTESTATION), R5_G1_ATTESTATION)
            if (_sha(inv.raw[R5_G1_ATTESTATION]) != r5_attestation_sha256
                    or att != {"format": "haic-rlpd-g1-r5-source-attestation-v1",
                               "scope": "r5 malformed receipt hash; independently hashed TRAIN ledger road IDs only",
                               "receipt_path": R5_ABORT, "receipt_sha256": R5_ABORT_SHA,
                               "ledger_path": R5_LEDGER, "ledger_sha256": R5_LEDGER_SHA,
                               "invalid_receipt_episodes_sha256": R5_BAD_HASH,
                               "original_ledger_bytes_attested": False}):
                raise InventoryError("G1-specific r5 attestation source/SHA/schema disagrees")
        r5_verified = True
    except InventoryError as exc:
        inv.block(R5_ABORT, "episodes_sha256", str(exc))

    sources = [{"path": path, "sha256": _sha(raw), "bytes": len(raw)}
               for path, raw in sorted(inv.raw.items())]
    experiment_sources = [row for row in sources if row["path"].startswith("experiments/")]
    experiment_sha = _digest(experiment_sources)
    source_sha = _digest(sources)
    repository_changed = ((expected_experiments_sha256 is not None
                           and experiment_sha != expected_experiments_sha256)
                          or (expected_sources_sha256 is not None
                              and source_sha != expected_sources_sha256))
    warnings: list[dict[str, str]] = []
    if repository_changed:
        warnings.append({"path": "repository_inventory", "field": "names+bytes",
                         "reason": "repository_changed_since_audit_snapshot; inspect candidate evidence, not global SHA"})
    changed_sources: dict[str, bytes] = {}
    try:
        current_experiments = sorted(f"experiments/{p.name}" for p in (root / "experiments").iterdir()
                                      if p.suffix == ".json")
        now_episodes, now_collections, now_markers, now_receipts = _walk_sources(root)
        changed_names = ((set(current_experiments) - set(experiments))
                         | (set(now_episodes) - set(episode_paths))
                         | (set(now_collections) - set(collection_paths))
                         | (set(now_markers) - set(marker_paths))
                         | (set(now_receipts) - set(receipt_paths)))
        for path in sorted(changed_names):
            changed_sources[path] = (root / path).read_bytes()
        if (current_experiments != experiments or now_episodes != episode_paths
                or now_collections != collection_paths or now_markers != marker_paths
                or now_receipts != receipt_paths):
            warnings.append({"path": "source_inventory", "field": "names",
                             "reason": "sources changed during read-only audit; re-audit before reservation"})
        for path, raw in inv.raw.items():
            real = root / path
            if any((root / PurePosixPath(*PurePosixPath(path).parts[:i])).is_symlink()
                   for i in range(1, len(PurePosixPath(path).parts) + 1)):
                raise InventoryError(f"{path}: source became symlink")
            current = real.read_bytes()
            if current != raw:
                changed_sources[path] = current
                warnings.append({"path": path, "field": "bytes",
                                 "reason": "source bytes changed during read-only audit; re-audit before reservation"})
    except (InventoryError, OSError) as exc:
        inv.block("source_inventory", "stability", str(exc))
    # An opaque source with a candidate token cannot be dismissed as unrelated.
    # Inspect the latest bytes too, so an in-flight edit adding a candidate blocks.
    token = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(map(str, candidates)) + r")(?![A-Za-z0-9_.])")
    for path, raw in sorted(changed_sources.items()):
        if _mentions_candidate_road(raw, token, set(candidates)):
            inv.block(path, "concurrent candidate evidence", "candidate ID added or changed during audit")
    candidate_blockers: list[dict[str, str]] = []
    for issue in inv.blockers:
        path = issue["path"]
        raw = inv.raw.get(path, b"")
        if (any(source["path"] == path for seed in candidates for source in inv.ids.get(seed, []))
                or (path not in inv.typed_paths and _mentions_candidate_road(raw, token, set(candidates)))
                or "source-list overrides" in issue["field"]
                or (path in ("runs", "experiments") and issue["field"] == "discovery")
                or (path in (G0_PROTOCOL, G0_AUDIT, CATALOG_PROTOCOL, R7_PROTOCOL)
                    and issue["field"] in ("JSON", "schema", "source"))
                or (r5_attestation_sha256 is not None and path == R5_ABORT
                    and issue["field"] == "episodes_sha256")
                or issue["field"] == "claims"
                or "symlink" in issue["reason"]
                or "ambiguous protected/TRAIN" in issue["reason"]
                or "source became" in issue["reason"]):
            candidate_blockers.append(issue)
        else:
            warnings.append(issue)
    # Audit JSON, receipts and new protocol formats that have not yet received
    # typed road extraction; exact candidate tokens remain unresolved evidence.
    for path, raw in sorted(inv.raw.items()):
        if (path.startswith("experiments/") and path not in inv.typed_paths
                and _mentions_candidate_road(raw, token, set(candidates))):
            issue = {"path": path, "field": "untyped candidate token",
                     "reason": "candidate occurs outside recognized road allocation fields"}
            if issue not in candidate_blockers:
                candidate_blockers.append(issue)
    collisions = [{"geometry_seed": seed, "sources": inv.ids[seed]}
                  for seed in candidates if seed in inv.ids]
    return {
        "format": "haic-rlpd-g1-coverage-seed-inventory-v2",
        "status": "BLOCKED" if candidate_blockers or collisions else "no_known_recorded_overlap",
        "protocol_frozen": self_claims_verified,
        "self_claims_verified": self_claims_verified,
        "train_claims_sha256": (_digest(sorted(self_claim_sources, key=lambda source: source["path"]))
                                 if self_claims_verified else None),
        "candidate_rule": "fixed seed_start+i, i=0..23; no replacement or top-up",
        "seed_start": start,
        "candidate_seeds": candidates,
        "cells": proposed_cells,
        "actors": ["entropy-v5-author-seed50 SOURCE-primary (predeclare elsewhere)",
                   "long-horizon-seed11 comparator (no promotion)"],
        "collisions": collisions,
        "blockers": candidate_blockers,
        "provenance_warnings": warnings,
        "repository_changed_since_audit_snapshot": repository_changed,
        "inventory_complete": not warnings,
        "experiment_inventory": experiment_sources,
        "experiment_inventory_sha256": experiment_sha,
        "source_inventory": sources,
        "source_inventory_sha256": source_sha,
        "blind_episode_reads": 0,
        "r5_evidence": {"status": "current TRAIN evidence verified" if r5_verified else "unverified",
                        "receipt_sha256": R5_ABORT_SHA, "current_ledger_sha256": R5_LEDGER_SHA,
                        "original_ledger_bytes_attested": False,
                        "limitation": "malformed historical receipt hash; current TRAIN reset IDs independently hashed"},
        "limitation": "Candidate-scoped recorded overlap only; warnings require review before reserve/freeze; never a global freshness certificate or permission to allocate/run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", required=True, type=int, help="hypothetical N, no allocation")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-experiments-sha256")
    parser.add_argument("--expected-sources-sha256")
    parser.add_argument("--r5-attestation-sha256")
    parser.add_argument("--reserve", action="store_true", help="claim a cleared TRAIN batch under shared lock")
    parser.add_argument("--study-id", help="required with --reserve; must describe a separately planned study")
    parser.add_argument("--protocol-path", help="optional future frozen protocol path for a claim")
    parser.add_argument("--protocol-sha256", help="SHA of --protocol-path if already frozen")
    parser.add_argument("--self-study-id", help="pre-reset recheck of this study's already claimed batch")
    parser.add_argument("--self-protocol-path", help="frozen G1 protocol containing the claim digest")
    parser.add_argument("--self-protocol-sha256", help="exact frozen G1 protocol SHA for pre-reset recheck")
    args = parser.parse_args()
    try:
        if args.reserve and args.self_study_id:
            parser.error("--reserve cannot waive its own as-yet-unfrozen claim")
        report = audit_g1_coverage_seeds(
            args.seed_start, repo_root=args.repo_root,
            expected_experiments_sha256=args.expected_experiments_sha256,
            expected_sources_sha256=args.expected_sources_sha256,
            r5_attestation_sha256=args.r5_attestation_sha256,
            self_study_id=args.self_study_id,
            self_protocol_path=args.self_protocol_path,
            self_protocol_sha256=args.self_protocol_sha256)
        if args.reserve:
            if not args.study_id:
                parser.error("--reserve requires --study-id")
            if report["status"] != "no_known_recorded_overlap":
                parser.error("candidate collision/ambiguity: refusing TRAIN claim")
            root = args.repo_root.resolve()
            locked_report: dict[str, Any] | None = None
            locked_claim_audit: dict[str, Any] | None = None
            def re_audit(cells: tuple) -> dict[str, Any]:
                nonlocal locked_report, locked_claim_audit
                fresh = audit_g1_coverage_seeds(args.seed_start, repo_root=root)
                if fresh["cells"] != [dict(cell) for cell in cells] or fresh["status"] != "no_known_recorded_overlap":
                    raise InventoryError("G1 candidate evidence changed before claim")
                locked_report = fresh
                locked_claim_audit = {"format": "haic-train-seed-audit-v1", "partition": "TRAIN",
                                      "status": "clear", "cells": fresh["cells"],
                                      "collisions": fresh["collisions"], "blockers": fresh["blockers"],
                                      "consumed_seeds": [], "reserved_seeds": [],
                                      "source": f"haic-rlpd-g1-coverage-seed-inventory-v2:{fresh['source_inventory_sha256']}"}
                return locked_claim_audit
            report["claims"] = reserve_train_seeds(
                root / TRAIN_CLAIMS, report["cells"], re_audit,
                study_id=args.study_id, protocol_id=args.study_id,
                protocol_path=args.protocol_path, protocol_sha256=args.protocol_sha256)
            report["locked_audit"] = locked_report
            report["locked_claim_audit"] = locked_claim_audit
            report["train_claims_sha256"] = _digest(sorted([
                {"path": f"{TRAIN_CLAIMS}/seed-{claim['geometry_seed']}.json",
                 "sha256": _sha((root / TRAIN_CLAIMS / f"seed-{claim['geometry_seed']}.json").read_bytes())}
                for claim in report["claims"]], key=lambda source: source["path"]))
    except (InventoryError, ReservationError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "no_known_recorded_overlap" else 2


if __name__ == "__main__":
    raise SystemExit(main())
