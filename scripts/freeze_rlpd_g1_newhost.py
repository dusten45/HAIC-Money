"""Freeze and self-audit a separately claimed, uncapped G1 TRAIN collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from haic.train_seed_reservations import validate_train_claim
from scripts import diagnose_rlpd_g1_coverage as g1
from scripts.audit_rlpd_g1_coverage_seeds import audit_g1_coverage_seeds


ROOT = Path(__file__).resolve().parents[1]
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_SHA = "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b"
RUBRIC = "experiments/rlpd-g1-newhost-image-rubric-v1.json"
STUDY = "rlpd-g1-new-host-20260929-v1"
START = 2800000001


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def within(relative: str, *, first: str | None = None) -> Path:
    path = Path(relative)
    if (path.is_absolute() or not path.parts or path.as_posix() != relative
            or any(part in (".", "..") for part in path.parts)
            or (first is not None and path.parts[0] != first)):
        raise ValueError(f"unsafe study path: {relative}")
    target = ROOT
    for part in path.parts:
        target /= part
        if target.is_symlink():
            raise ValueError(f"symlinked study path: {relative}")
    return target


def write_exclusive(relative: str, value: dict) -> str:
    target = within(relative, first="experiments")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return digest(target)


def cells() -> list[dict]:
    return [{"partition": "TRAIN", "track_id": 1, "geometry_seed": START + offset,
             "obstacles": True} for offset in range(24)]


def claims_sha(rows: list[dict]) -> str:
    sources = []
    for row in rows:
        seed = row["geometry_seed"]
        relative = f"experiments/train-seed-claims/seed-{seed}.json"
        path = within(relative, first="experiments")
        claim = json.loads(path.read_text(encoding="utf-8"))
        validate_train_claim(claim, seed)
        if (claim["study_id"] != STUDY or claim["protocol_id"] != STUDY
                or claim["status"] != "reserved"
                or any(claim[key] != row[key] for key in row)
                or claim["protocol_path"] not in (None, "experiments/rlpd-g1-newhost-20260929-v1.json")):
            raise ValueError(f"claim does not bind the exact G1 study/cell: {relative}")
        sources.append({"path": relative, "sha256": digest(path)})
    return g1._digest(sorted(sources, key=lambda entry: entry["path"]))


def freeze(relative: str) -> dict:
    target = within(relative, first="experiments")
    if target.exists():
        raise FileExistsError(target)
    original = within(G0_PROTOCOL, first="experiments")
    if digest(original) != G0_SHA:
        raise ValueError("historical G0 actor/cell protocol changed")
    g0 = json.loads(original.read_text(encoding="utf-8"))
    rows = cells()
    claim_digest = claims_sha(rows)
    rubric = within(RUBRIC, first="experiments")
    if json.loads(rubric.read_text(encoding="utf-8")).get("status") != "frozen-before-train-collection":
        raise ValueError("pixel-only review rubric not frozen before reset")
    actors = []
    for actor_id, path in zip(g1.ACTORS, g1.ACTOR_PATHS):
        entry = next(actor for actor in g0["actors"] if actor["id"] == actor_id)
        if entry["path"] != path or entry["sha256"] != g1.ACTOR_HASHES[actor_id]:
            raise ValueError("historical G0 actor identity changed")
        if digest(within(path, first="runs")) != entry["sha256"]:
            raise ValueError("historical actor export changed")
        actors.append({key: entry[key] for key in (
            "id", "path", "sha256", "source_sha256", "export_protocol_sha256",
            "action_mode")})
    protocol = {
        "format": "haic-rlpd-g1-coverage-protocol-v1", "status": "frozen",
        "partition": "TRAIN", "study_id": STUDY, "cells": rows,
        "source_hashes": {path: digest(within(path)) for path in sorted(g1.SOURCE_FILES)},
        "source_actors": actors,
        "budget": {"max_decisions_per_episode": 2000, "max_total_decisions": 96000,
                   "max_total_raw_frames": 386448, "max_core_hours": None},
        "runtime": {"frame_skip": 4, "reward_shaping": False, "collision_penalty": 0,
                    "interventions": False, "learner_updates": 0,
                    "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50},
        "event_rules": {"max_decisions": 2000, "negative_reward_limit": 100,
                        "stall_window": 20, "tile_window": 20,
                        "directed_delta_epsilon": 0.01,
                        "centerline_far_threshold_m": 10.0},
        "train_claims_sha256": claim_digest,
        "image_rubric_sha256": digest(rubric),
        "exclusions": {"training_geometry_seeds": [
            row["geometry_seed"] for row in g0["cells"]]},
    }
    if relative != "experiments/rlpd-g1-newhost-20260929-v1.json":
        raise ValueError("G1 claim protocol path must be fixed")
    protocol_sha = write_exclusive(relative, protocol)
    return {"protocol_path": relative, "protocol_sha256": protocol_sha,
            "train_claims_sha256": claim_digest}


def self_audit(protocol_path: str, expected_sha: str, receipt_path: str) -> dict:
    if protocol_path != "experiments/rlpd-g1-newhost-20260929-v1.json":
        raise ValueError("not the claimed G1 protocol")
    source = within(protocol_path, first="experiments")
    if digest(source) != expected_sha:
        raise ValueError("G1 protocol changed after claims")
    report = audit_g1_coverage_seeds(
        START, repo_root=ROOT, self_study_id=STUDY,
        self_protocol_path=protocol_path, self_protocol_sha256=expected_sha)
    if (report["status"] != "no_known_recorded_overlap"
            or report["protocol_frozen"] is not True
            or report["self_claims_verified"] is not True
            or report["cells"] != cells()
            or report["train_claims_sha256"] != claims_sha(cells())):
        raise ValueError("candidate G1 self-audit is not clear")
    receipt_sha = write_exclusive(receipt_path, report)
    return {"audit_path": receipt_path, "audit_sha256": receipt_sha,
            "status": report["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--protocol", required=True)
    audit = sub.add_parser("self-audit")
    audit.add_argument("--protocol", required=True)
    audit.add_argument("--protocol-sha256", required=True)
    audit.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = (freeze(args.protocol) if args.operation == "freeze" else
              self_audit(args.protocol, args.protocol_sha256, args.receipt))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
