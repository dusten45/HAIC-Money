"""Zero-interaction candidate-unseen KOI TRAIN inventory and immutable claims.

Fixed 24 uint32 road IDs, three track variants, two frozen policies. No simulator,
policy imports, outcomes, replacement or top-up. Callers hold the shared registry
directory flock through audit_cell, reset-intent publication and actual reset.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

from haic.train_seed_reservations import (
    AUDIT_FORMAT, ReservationError, reserve_train_seeds, validate_train_claim,
)
from scripts.audit_rlpd_g0_seeds import REQUIRED_COLLECTIONS, REQUIRED_LEDGERS, REQUIRED_PROTOCOLS
from scripts.audit_rlpd_g1_coverage_seeds import InventoryError, _Inventory, _digest, _json, _sha
from scripts.audit_rlpd_gate_unseen_train import (
    _catalog_counts, _discover, _ppo_resolution, _relevant,
)

STUDY_ID = "koi-steering-generalization-v1"
FORMAT = "haic-koi-steering-generalization-audit-v1"
AUDIT_PATH = f"experiments/{STUDY_ID}-exposure.json"
PROTOCOL_PATH = f"experiments/{STUDY_ID}.json"
RUN_PATH = f"runs/{STUDY_ID}"
REGISTRY = "experiments/train-seed-claims"
SEED_START, COHORT_SIZE = 3184000001, 24
ARMS = ("crossing_projection", "steering_release_v1")
MODEL_HASHES = {
    ARMS[0]: "a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8",
    ARMS[1]: "b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce",
}
MANIFEST_SHA = "b1a97164434d66f3e83cfda938d606a1bf6aab910adafbac5fd7dbb59be981a4"
R2_PATH = "runs/koi-steering-release-ab-20261001-r2/protocol.json"
R2_SHA = "9c0a9e4f11c3a1884bfc1bdd1be8841d76c81ca1dc3fa851096f23617396ee96"
FRESHNESS = "candidate-lineage-unseen TRAIN/dev; all known explicit cross-lane interactions, allocations, protected/retired/reserved/claimed road IDs excluded; not project-global never-used"
IMPLEMENTATION = (
    "scripts/audit_koi_steering_generalization.py", "haic/train_seed_reservations.py",
    "scripts/audit_rlpd_gate_unseen_train.py", "scripts/audit_rlpd_g0_seeds.py",
    "scripts/audit_rlpd_g1_coverage_seeds.py", "scripts/audit_tdmpc2_train_diag_seeds.py",
    "docs/evaluation/generalization-policy.md",
)


class StableInventory(_Inventory):
    """Reject symlink/device sources and changing reads, not unrelated hash drift."""

    def read(self, relative: str) -> bytes:
        if relative in self.raw:
            return self.raw[relative]
        parts = relative.split("/")
        if (Path(relative).is_absolute() or "\\" in relative
                or any(part in ("", ".", "..") for part in parts)):
            raise InventoryError(f"{relative}: unsafe path")
        path = self.root
        for part in parts:
            path /= part
            if path.is_symlink():
                raise InventoryError(f"{relative}: symlinked source")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise InventoryError(f"{relative}: not regular metadata")
                raw = stream.read()
                after = os.fstat(stream.fileno())
            stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if stamp(before) != stamp(after) or stamp(before) != stamp(path.stat()):
                raise InventoryError(f"{relative}: source changed while reading")
        except OSError as exc:
            raise InventoryError(f"{relative}: missing/unreadable source") from exc
        self.raw[relative] = raw
        return raw


def proposed_cells() -> list[dict[str, Any]]:
    return [{"partition": "TRAIN", "track_id": track, "geometry_seed": seed, "obstacles": True}
            for seed in range(SEED_START, SEED_START + COHORT_SIZE) for track in (1, 2, 3)]


def claim_cells() -> list[dict[str, Any]]:
    # The common registry excludes each geometry globally, regardless of track.
    return [cell for cell in proposed_cells() if cell["track_id"] == 1]


def scheduled_slots() -> list[dict[str, Any]]:
    rows = []
    for cell in proposed_cells():
        track, seed = cell["track_id"], cell["geometry_seed"]
        arms = ARMS if (track + seed) % 2 else tuple(reversed(ARMS))
        rows.extend({"mode": arm, "track_id": track, "seed": seed, "status": "unrun"} for arm in arms)
    return rows


def _lineage(inv: StableInventory, seeds: set[int]) -> dict[str, Any]:
    """Pin the source-only policy closure and its explicit development ancestry."""
    def pin(path: str, expected: str | None = None) -> bytes:
        raw = inv.read(path)
        if expected is not None and _sha(raw) != expected:
            raise InventoryError(f"{path}: frozen lineage hash differs")
        return raw

    r2 = _json(pin(R2_PATH, R2_SHA), R2_PATH)
    manifest_path = "submissions/koi-steering-release-v2.manifest.json"
    manifest = _json(pin(manifest_path, MANIFEST_SHA), manifest_path)
    if (r2["model_hashes"] != MODEL_HASHES or manifest["candidate_zip_sha256"] != MODEL_HASHES[ARMS[1]]
            or manifest["candidate"] != "koi-steering-release-v2"
            or manifest["stabilize_ambiguous_flank"] is not True
            or manifest["speed_target_changed"] is not False or manifest["margin_reduction"] is not False):
        raise InventoryError("KOI frozen model identity differs")
    paths = ("runs/koi-steering-release-ab-20261001-r2/crossing-projection-source-reconstruction.zip",
             "submissions/koi-steering-release-v2.zip")
    source_text = []
    for arm, path in zip(ARMS, paths):
        raw = pin(path, MODEL_HASHES[arm])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            expected = r2["model_source_sha256"][arm]
            if (len(names) != len(set(names)) or set(names) != set(expected)
                    or any(not name.endswith(".py") or Path(name).is_absolute()
                           or ".." in Path(name).parts or "\\" in name for name in names)):
                raise InventoryError("KOI policy closure has unknown/non-source members")
            for name in names:
                data = archive.read(name)
                if _sha(data) != expected[name]:
                    raise InventoryError("KOI source member differs")
                source_text.append(data.decode("utf-8"))
        if arm == ARMS[1] and {row["path"]: row["sha256"] for row in manifest["files"]} != expected:
            raise InventoryError("KOI manifest source closure differs")
    # Exact immutable bytes were reviewed as deterministic pixels/history source,
    # not a learned actor or a PPO/prior-data loader. Reject external artifact refs.
    joined = "\n".join(source_text)
    if re.search(r"torch\.load|pickle\.load|joblib\.load|\.pt\b|\.npz\b|runs/|evaluations/", joined):
        raise InventoryError("source-only policy has an unclassified external data dependency")
    lineage_paths = [R2_PATH, manifest_path, *paths]
    for run in ("koi-adaptive-ab-20260930-v1", "koi-minimum-clearance-ab-20260930-v1",
                "koi-minimum-clearance-ab-20260930-r2", "koi-minimum-clearance-ab-20260930-r3",
                "koi-steering-release-ab-20261001-v1"):
        path = f"runs/{run}/protocol.json"
        if run == "koi-steering-release-ab-20261001-v1":
            expected_sha = "26d43bf9e0eceff84c667e244683376344a4bc1356ea818084ca66be8963e04d"
        else:
            pins = [sha for name, sha in r2["consumed_reuse_evidence"]["source_sha256"].items()
                    if name.endswith("/" + path)]
            if len(pins) != 1:
                raise InventoryError("KOI predecessor protocol lacks unique frozen ancestry pin")
            expected_sha = pins[0]
        obj = _json(pin(path, expected_sha), path)
        cells = obj.get("cells")
        consumed = {(track, seed) for track in (1, 2, 3)
                    for seed in (*range(38300, 38304), *range(50300, 50304))}
        if (not isinstance(cells, list) or not cells
                or any(not isinstance(c, dict) or type(c.get("geometry_seed")) is not int
                       or type(c.get("track_id")) is not int or c["geometry_seed"] in seeds for c in cells)
                or len(cells) != 24 or {(c["track_id"], c["geometry_seed"]) for c in cells} != consumed):
            raise InventoryError(f"{path}: candidate selection/development allocation unknown or overlapping")
        lineage_paths.append(path)
    diagnosis_path = "scripts/diagnose_koi_steering_release.py"
    diagnosis_pin = next(row["sha256"] for row in r2["source_copies"]
                         if row["file"] == "source/project/" + diagnosis_path)
    diagnosis = pin(diagnosis_path, diagnosis_pin)
    if (b"range(38300, 38304)" not in diagnosis or b"range(50300, 50304)" not in diagnosis
            or token_for(seeds).search(diagnosis.decode("utf-8"))):
        raise InventoryError("KOI diagnosis producer has unknown selection cells")
    lineage_paths.append(diagnosis_path)
    evidence = [{"path": p, "sha256": _sha(inv.raw[p]), "bytes": len(inv.raw[p])} for p in sorted(lineage_paths)]
    return {"status": "verified", "sources": evidence, "source_pins_sha256": _digest(evidence),
            "model_hashes": MODEL_HASHES, "model_source_sha256": r2["model_source_sha256"],
            "candidate_manifest_sha256": MANIFEST_SHA,
            "why_unrelated": "Exact source-only KOI ZIP closure has no learned weights/prior-data/PPO artifact loading. KOI development/selection protocols use explicit consumed 38300-38303/50300-50303 cells, all excluded. Legacy PPO sampling is not this candidate's data/weight ancestry.",
            "limitation": "Formal source/data lineage, not a certificate of all human influence or unrecorded project-global interaction."}


def token_for(seeds: set[int]) -> re.Pattern[str]:
    return re.compile(r"(?<!\d)(?:" + "|".join(map(str, sorted(seeds))) + r")(?!\d)")


def _reviewed_counts(obj: dict[str, Any], path: str, raw: bytes) -> dict[str, Any]:
    if path != "experiments/koi-steering-release-consumed-audit-v1.json":
        return obj
    if _sha(raw) != "a9a6422b0f63ffc1ab9f45d94785cac8c3770b678f9b522bdf6a8f7177bacfe9":
        raise InventoryError("reviewed consumed audit count schema hash differs")
    normalized = dict(obj)
    allocation = dict(obj["allocation_recheck"])
    for key in ("explicit_seed_range_lists_checked", "seed_start_base_count_context_records_checked"):
        if type(allocation.get(key)) is not int or allocation[key] < 0:
            raise InventoryError("reviewed metadata cardinality malformed")
        allocation.pop(key)
    normalized["allocation_recheck"] = allocation
    return normalized


def _unknown_road(value: Any) -> bool:
    if isinstance(value, list):
        return any(_unknown_road(item) for item in value)
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key in ("geometry_seed", "road_id") and (type(item) is not int or not 0 <= item < 2**32):
            return True
        if _unknown_road(item):
            return True
    return False


def _scoped_source(receipt: dict[str, Any]) -> str:
    keys = ("format", "study_id", "cells", "freshness_definition", "freshness_scope",
            "project_global_unseen_certified", "sources", "warnings", "legacy_uncertainties",
            "legacy_ppo_dispositions", "candidate_lineage_evidence", "geometry_seed_count",
            "matched_cell_count", "episode_slot_count", "zero_interaction",
            "environment_constructions", "environment_resets", "blind_observation_reads")
    try:
        return "koi-candidate-unseen-TRAIN/evidence-sha256:" + _digest({key: receipt[key] for key in keys})
    except (KeyError, TypeError, ValueError) as exc:
        raise InventoryError("malformed scoped claim evidence") from exc


def verify_claims(receipt: dict[str, Any], *, root: Path = Path(".")) -> list[dict[str, Any]]:
    if (receipt.get("format") != FORMAT or receipt.get("study_id") != STUDY_ID
            or receipt.get("cells") != proposed_cells() or receipt.get("status") != "clear"
            or receipt.get("collisions") != [] or receipt.get("blockers") != []
            or receipt.get("zero_interaction") is not True
            or receipt.get("freshness_definition") != FRESHNESS
            or receipt.get("freshness_scope") != "candidate-lineage-unseen"
            or receipt.get("project_global_unseen_certified") is not False
            or any(type(receipt.get(key)) is not int or receipt[key] != value for key, value in
                   (("geometry_seed_count", 24), ("matched_cell_count", 72), ("episode_slot_count", 144),
                    ("environment_constructions", 0), ("environment_resets", 0), ("blind_observation_reads", 0)))
            or any(row not in receipt.get("warnings", []) for row in receipt.get("legacy_uncertainties", []))):
        raise InventoryError("invalid/blocked KOI receipt")
    contract = receipt.get("claim_audit")
    expected = {"format": AUDIT_FORMAT, "partition": "TRAIN", "status": "clear", "cells": claim_cells(),
                "collisions": [], "blockers": [], "consumed_seeds": [], "reserved_seeds": [],
                "source": _scoped_source(receipt)}
    if contract != expected:
        raise InventoryError("claim audit evidence chain differs")
    assert isinstance(contract, dict)
    pins = receipt.get("claims")
    if not isinstance(pins, list) or len(pins) != COHORT_SIZE:
        raise InventoryError("exactly24 global geometry claims required")
    inv, records = StableInventory(Path(root).absolute()), []
    for cell, pin in zip(claim_cells(), pins):
        seed = cell["geometry_seed"]
        path = f"{REGISTRY}/seed-{seed}.json"
        raw = inv.read(path)
        row = _json(raw, path)
        validate_train_claim(row, seed)
        if (pin != {"path": path, "sha256": _sha(raw)} or row["study_id"] != STUDY_ID
                or row["audit_digest_sha256"] != _digest(contract) or row["audit_source"] != contract["source"]
                or any(row[key] != value for key, value in cell.items()) or row["status"] != "reserved"
                or any(row[key] is not None for key in ("protocol_id", "protocol_path", "protocol_sha256"))):
            raise InventoryError(f"{path}: self-claim chain differs")
        records.append(row)
    return records


def load_audit(path: Path | str, expected_sha256: str, *, root: Path = Path(".")) -> dict[str, Any]:
    inv = StableInventory(Path(root).absolute())
    relative = Path(path).relative_to(inv.root).as_posix() if Path(path).is_absolute() else str(path)
    if relative != AUDIT_PATH:
        raise InventoryError("unexpected audit receipt path")
    raw = inv.read(relative)
    if _sha(raw) != expected_sha256:
        raise InventoryError("audit receipt hash differs")
    receipt = _json(raw, relative)
    verify_claims(receipt, root=root)
    return receipt


def _self_metadata(inv: StableInventory, path: str, raw: bytes, receipt: dict[str, Any],
                   protocol_sha: str | None) -> None:
    protocol_raw = inv.read(PROTOCOL_PATH)
    protocol = _json(protocol_raw, PROTOCOL_PATH)
    audit_raw = inv.read(AUDIT_PATH)
    lineage = receipt["candidate_lineage_evidence"]
    if (protocol_sha is None or _sha(protocol_raw) != protocol_sha or _json(audit_raw, AUDIT_PATH) != receipt
            or protocol.get("study", protocol.get("study_id")) != STUDY_ID
            or _digest(protocol.get("cells")) != _digest(proposed_cells()) or protocol.get("episodes") != 144
            or protocol.get("model_hashes") != MODEL_HASHES
            or protocol.get("model_source_sha256") != lineage.get("model_source_sha256")
            or protocol.get("candidate_manifest_sha256") != MANIFEST_SHA
            or _digest(protocol.get("schedule")) != _digest(scheduled_slots())
            or protocol.get("freshness", {}).get("audit_receipt") != {"path": AUDIT_PATH, "sha256": _sha(audit_raw)}
            or any(protocol.get(key) != value for key, value in
                   (("max_decisions", 1200), ("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("official_action", False)))):
        raise InventoryError("self protocol/receipt/model/cohort identity differs")
    if path in (PROTOCOL_PATH, RUN_PATH + "/protocol.json"):
        if _sha(raw) != protocol_sha:
            raise InventoryError("copied self protocol differs")
        return
    if path == RUN_PATH + "/exposure-audit.json":
        if raw != audit_raw:
            raise InventoryError("copied self exposure receipt differs")
        return
    if path != RUN_PATH + "/reset-ledger.jsonl":
        raise InventoryError("unclassified self-study metadata")
    if raw and not raw.endswith(b"\n"):
        raise InventoryError("torn self reset ledger")
    rows = [_json(line, path) for line in raw.splitlines()]
    if len(rows) > 288:
        raise InventoryError("self ledger exceeds144 slots")
    for index, row in enumerate(rows):
        slot = scheduled_slots()[index // 2]
        track, seed, arm = slot["track_id"], slot["seed"], slot["mode"]
        identity = f"{track}:{seed}:{arm}"
        if index % 2 == 0:
            if (set(row) != {"status", "arm", "track", "seed", "slot_id", "time"}
                    or row.get("status") != "reset_intent" or row.get("arm") != arm
                    or row.get("track") != track or row.get("seed") != seed
                    or type(row.get("track")) is not int or type(row.get("seed")) is not int
                    or row.get("slot_id") != identity or type(row.get("time")) not in (int, float)):
                raise InventoryError("self reset-intent identity/order differs")
        else:
            keys = {"mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage", "collisions",
                    "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes",
                    "status", "file", "sha256", "slot_id", "process_file", "process_sha256"}
            if (set(row) != keys or row.get("status") != "completed" or row.get("mode") != arm
                    or row.get("track_id") != track or row.get("seed") != seed or row.get("slot_id") != identity
                    or type(row.get("track_id")) is not int or type(row.get("seed")) is not int
                    or row.get("file") != f"{track}-{seed}-{arm}.json"
                    or row.get("process_file") != f"{track}-{seed}-{arm}.bound-process.json"
                    or any(not isinstance(row.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
                           for key in ("sha256", "process_sha256"))):
                raise InventoryError("self completed identity/order differs")
    # A final intent is conservative partial exposure, authenticated, never released.


def audit(root: Path = Path("."), *, authenticated_receipt: dict[str, Any] | None = None,
          self_protocol_sha256: str | None = None,
          required_sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    root, seeds = Path(root).absolute(), set(range(SEED_START, SEED_START + COHORT_SIZE))
    if required_sources is not None and root == Path(__file__).resolve().parents[1]:
        raise InventoryError("source overrides are synthetic-only")
    production = required_sources is None
    required = tuple(REQUIRED_PROTOCOLS) + tuple(REQUIRED_LEDGERS) + tuple(REQUIRED_COLLECTIONS) if production else required_sources
    inv = StableInventory(root)
    collisions, blockers, warnings, dispositions, uncertainties = [], [], [], [], []
    lineage = {"status": "synthetic-empty-lineage", "sources": []}
    own = set()
    if authenticated_receipt is not None:
        verify_claims(authenticated_receipt, root=root)
        own = {pin["path"] for pin in authenticated_receipt["claims"]}
        if self_protocol_sha256 is not None:
            try:
                _self_metadata(inv, PROTOCOL_PATH, inv.read(PROTOCOL_PATH),
                               authenticated_receipt, self_protocol_sha256)
            except InventoryError as exc:
                blockers.append({"path": PROTOCOL_PATH, "reason": str(exc)})
    token = token_for(seeds)
    try:
        paths = set(_discover(root)) | set(required or ())
        # Earlier registry used batch geometry_seeds files. Preserve all of it.
        old_registry = root / "runs/rlpd-g0-claims"
        if old_registry.exists():
            if old_registry.is_symlink() or not old_registry.is_dir():
                raise InventoryError("unsafe historical G0 claims")
            paths.update(p.relative_to(root).as_posix() for p in old_registry.iterdir() if p.name != ".lock")
        # The older discovery omitted KOI reset-ledger; all run intents matter.
        for path in root.glob("runs/**/reset-ledger.jsonl"):
            relative = path.relative_to(root).as_posix()
            if not re.search(r"blind|confirm|private|held[-_]?out|holdout|screen", relative, re.I):
                paths.add(relative)
    except (InventoryError, OSError) as exc:
        paths = set(required or ())
        blockers.append({"path": ".", "reason": str(exc)})
    if production:
        try:
            lineage = _lineage(inv, seeds)
        except (InventoryError, OSError, KeyError, TypeError, zipfile.BadZipFile, UnicodeError) as exc:
            blockers.append({"path": "candidate-lineage", "reason": str(exc)})
    for path in sorted(paths):
        try:
            raw = inv.read(path)
            if path in own:
                continue
            if path == AUDIT_PATH:
                if authenticated_receipt is None or _json(raw, path) != authenticated_receipt:
                    raise InventoryError("existing self audit is not authenticated")
                continue
            if (path == PROTOCOL_PATH or path.startswith(RUN_PATH + "/")) and authenticated_receipt is not None:
                _self_metadata(inv, path, raw, authenticated_receipt, self_protocol_sha256)
                continue
            if path.startswith(REGISTRY + "/"):
                match = re.fullmatch(r"seed-(\d+)\.json", Path(path).name)
                if match is None:
                    raise InventoryError("unknown shared registry entry")
                validate_train_claim(_json(raw, path), int(match[1]))
            normalized = raw
            if not path.endswith(".jsonl"):
                obj = _reviewed_counts(_catalog_counts(_json(raw, path), inv), path, raw)
                if _unknown_road(obj):
                    raise InventoryError("unresolved explicit road identity")
                cfg = obj.get("config")
                if (path.startswith("runs/") and Path(path).name == "config.json"
                        and isinstance(cfg, dict) and cfg.get("algorithm") == "PPO" and "sampled_seed_range" in cfg):
                    obj, disposition = _ppo_resolution(inv, path, obj)
                    dispositions.append(disposition)
                normalized = json.dumps(obj, sort_keys=True).encode("ascii")
            else:
                if _relevant(raw, path, seeds, token):
                    entry = {"path": path, "sha256": _sha(raw)}
                    if token.search(raw.decode("utf-8", errors="replace")):
                        collisions.append({**entry, "reason": "explicit candidate road in complete/partial ledger"})
                    else:
                        blockers.append({**entry, "reason": "candidate-possible ledger interval/identity"})
                for line in raw.splitlines():
                    try:
                        obj = _json(line, path)
                    except InventoryError:
                        if _relevant(line, path, seeds, token):
                            raise InventoryError("candidate-relevant malformed partial ledger")
                        if re.search(rb'"(?:geometry_seed|road_id)"\s*:\s*(?:null|"(?:unknown)?"|$)', line):
                            raise InventoryError("partial ledger has unresolved road identity")
                        continue
                    if _unknown_road(obj):
                        raise InventoryError("unresolved explicit road identity in ledger")
            if not path.endswith(".jsonl") and _relevant(normalized, path, seeds, token):
                entry = {"path": path, "sha256": _sha(raw)}
                if token.search(normalized.decode("utf-8", errors="replace")):
                    collisions.append({**entry, "reason": "explicit candidate road identity across lanes/tracks"})
                else:
                    blockers.append({**entry, "reason": "candidate-possible interval/unknown road identity"})
            if path.endswith(".jsonl") and raw and not raw.endswith(b"\n"):
                # Never drop a complete matching prefix because the last row is torn.
                tail = raw.splitlines()[-1]
                if _relevant(tail, path, seeds, token):
                    blockers.append({"path": path, "reason": "candidate-relevant torn partial ledger"})
                else:
                    warnings.append({"path": path, "reason": "disjoint torn ledger retained; not proof of zero interaction"})
        except (InventoryError, ReservationError, OSError) as exc:
            blockers.append({"path": path, "reason": str(exc)})
    for row in dispositions:
        if row["status"] == "HOLD":
            if lineage["status"] == "verified":
                warning = {**row, "status": "WARN-unrelated-legacy-uncertainty",
                           "why_unrelated": lineage["why_unrelated"],
                           "lineage_source_pins_sha256": lineage["source_pins_sha256"],
                           "project_global_non_use_proven": False}
                warnings.append(warning)
                uncertainties.append(warning)
            else:
                blockers.append({"path": row["path"], "reason": "unclassified candidate-relevant legacy sampling uncertainty"})
    if production:
        for path in IMPLEMENTATION:
            try:
                inv.read(path)
            except InventoryError as exc:
                blockers.append({"path": path, "reason": str(exc)})
    sources = [{"path": p, "sha256": _sha(raw), "bytes": len(raw)} for p, raw in sorted(inv.raw.items())
               if p not in own and p not in (AUDIT_PATH, PROTOCOL_PATH) and not p.startswith(RUN_PATH + "/")]
    return {"format": FORMAT, "study_id": STUDY_ID, "partition": "TRAIN", "cells": proposed_cells(),
            "geometry_seed_count": 24, "matched_cell_count": 72, "episode_slot_count": 144,
            "status": "clear" if not collisions and not blockers else "HOLD", "collisions": collisions,
            "blockers": blockers, "warnings": warnings, "legacy_uncertainties": uncertainties,
            "legacy_ppo_dispositions": dispositions, "candidate_lineage_evidence": lineage,
            "sources": sources, "source_count": len(sources), "source_pins_sha256": _digest(sources),
            "freshness_definition": FRESHNESS, "freshness_scope": "candidate-lineage-unseen",
            "project_global_unseen_certified": False, "zero_interaction": True,
            "environment_constructions": 0, "environment_resets": 0, "blind_observation_reads": 0,
            "claims": [], "claim_audit": None}


def audit_cell(receipt: dict[str, Any], cell: dict[str, Any], *, root: Path = Path("."),
               self_protocol_sha256: str | None = None) -> dict[str, Any]:
    if cell not in proposed_cells() or set(cell) != set(proposed_cells()[0]):
        raise InventoryError("cell outside fixed TRAIN cohort/conditions")
    report = audit(root, authenticated_receipt=receipt, self_protocol_sha256=self_protocol_sha256)
    if report["status"] != "clear":
        raise InventoryError("pre-reset audit HOLD: " + json.dumps(report["collisions"] + report["blockers"]))
    return report


def claim(root: Path = Path("."), *, required_sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    def recheck(cells: Any) -> dict[str, Any]:
        nonlocal report
        report = audit(root, required_sources=required_sources)
        if list(map(dict, cells)) != claim_cells() or report["status"] != "clear":
            raise InventoryError("under-lock audit HOLD: " + json.dumps(report["collisions"] + report["blockers"]))
        contract = {"format": AUDIT_FORMAT, "partition": "TRAIN", "status": "clear", "cells": claim_cells(),
                    "collisions": [], "blockers": [], "consumed_seeds": [], "reserved_seeds": [],
                    "source": _scoped_source(report)}
        report["claim_audit"] = contract
        return contract
    reserve_train_seeds(Path(root) / REGISTRY, claim_cells(), recheck, study_id=STUDY_ID)
    inv = StableInventory(Path(root).absolute())
    report["claims"] = [{"path": path, "sha256": _sha(inv.read(path))} for path in
                        (f"{REGISTRY}/seed-{cell['geometry_seed']}.json" for cell in claim_cells())]
    verify_claims(report, root=root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.claim and args.output is None:
        args.output = Path(AUDIT_PATH)
    if args.output is not None:
        target = args.root / args.output
        if (args.output.as_posix() != AUDIT_PATH or target.exists() or target.is_symlink()
                or any(p.is_symlink() for p in (target.parent, *target.parent.parents))):
            parser.error("only an absent, safe owned exposure receipt may be written")
    report = claim(args.root) if args.claim else audit(args.root)
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    if args.output is not None:
        with (args.root / args.output).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "cells": len(report["cells"]),
                      "source_count": report["source_count"], "claims": len(report["claims"]),
                      "collisions": report["collisions"], "blockers": report["blockers"],
                      "receipt_sha256": _sha(raw)}, sort_keys=True))
    return 0 if report["status"] == "clear" else 2


if __name__ == "__main__":
    raise SystemExit(main())
