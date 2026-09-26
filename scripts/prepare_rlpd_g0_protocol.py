"""Freeze a bounded, TRAIN-only RLPD G0 protocol from a clean seed-audit receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.audit_rlpd_g0_seeds import R5_ERRATUM_PATH, audit_g0_seeds
from scripts.diagnose_rlpd_g0 import (
    EXPECTED_ACTORS, ROOT, SOURCE_FILES, _check_audit_inventory, _check_experiment_content_inventory, _file,
    _pinned_json, _write_json_exclusive, sha256_file,
)


def freeze_g0_protocol(
    root: Path, *, audit_path: str, audit_sha256: str, output: str,
    audit_verifier=None, payload_loader=None, expected_actors=None,
) -> tuple[dict, str]:
    """Freeze bytes and allocation, not a diagnosis; do not create an environment."""
    root = root.resolve()
    if expected_actors is None:
        expected_actors = EXPECTED_ACTORS
    if audit_verifier is None:
        audit_verifier = audit_g0_seeds
    if payload_loader is None:
        payload_loader = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    audit = _pinned_json(_file(root, audit_path, "experiments"), audit_sha256)
    if (
        audit.get("format") != "haic-rlpd-g0-seed-audit-v1"
        or audit.get("status") != "no_known_recorded_overlap"
        or audit.get("passed") is not True
        or audit.get("collision_count") != 0
        or audit.get("ambiguities") != []
        or len(audit.get("cells", [])) != 12
        or not isinstance(audit.get("r5_erratum"), dict)
        or audit["r5_erratum"].get("path") != R5_ERRATUM_PATH
        or audit["r5_erratum"].get("original_ledger_bytes_attested") is not False
    ):
        raise ValueError("G0 protocol cannot freeze an incomplete seed audit")
    _check_audit_inventory(root, audit)
    _check_experiment_content_inventory(root, audit)
    cells = audit["cells"]
    fresh = audit_verifier(
        audit["seed_start"], cells[0]["track_id"], repo_root=root, r5_erratum=R5_ERRATUM_PATH,
    )
    if any(fresh.get(key) != audit.get(key) for key in (
        "cells", "candidate_seeds_sha256", "excluded_inventory_sha256",
        "source_inventory_sha256", "source_inventory", "r5_erratum",
        "experiment_content_inventory_sha256", "experiment_content_inventory",
    )):
        raise ValueError("G0 audit inventory changed before protocol freeze")
    sources = {}
    for relative in sorted(SOURCE_FILES):
        path = _file(root, relative, relative.split("/", 1)[0])
        sources[relative] = sha256_file(path)
    actors = []
    for actor_id, expected in expected_actors.items():
        path = _file(root, expected["path"], "runs")
        if sha256_file(path) != expected["sha256"]:
            raise ValueError(f"G0 designated actor bytes changed: {actor_id}")
        candidate = path.with_name("candidate.json")
        candidate_path = candidate.relative_to(root).as_posix()
        candidate_sha = sha256_file(candidate)
        candidate_receipt = _pinned_json(_file(root, candidate_path, "runs"), candidate_sha)
        payload = payload_loader(path)
        checkpoint_rel = candidate_receipt.get("checkpoint_path")
        if (not isinstance(checkpoint_rel, str) or Path(checkpoint_rel).is_absolute()
                or ".." in Path(checkpoint_rel).parts):
            raise ValueError("G0 source candidate checkpoint path is unsafe")
        checkpoint_path = (candidate.parent.parent.parent / checkpoint_rel).relative_to(root).as_posix()
        checkpoint = _file(root, checkpoint_path, "runs")
        source_protocol_path = f"experiments/{expected['study_id']}.json"
        source_protocol = _file(root, source_protocol_path, "experiments")
        if (
            candidate_receipt.get("format") != "haic-rlpd-candidate-v1"
            or candidate_receipt.get("actor_sha256") != expected["sha256"]
            or candidate_receipt.get("training_seed") != expected["training_seed"]
            or candidate_receipt.get("study_id") != expected["study_id"]
            or candidate_receipt.get("arm") != expected["arm"]
            or candidate_receipt.get("environment_steps") != 131072
            or sha256_file(checkpoint) != candidate_receipt.get("checkpoint_sha256")
            or sha256_file(source_protocol) != candidate_receipt.get("protocol_sha256")
            or payload.get("format") != "haic-rlpd-pixel-actor-v1"
            or payload.get("training_seed") != expected["training_seed"]
            or payload.get("environment_steps") != 131072
            or payload.get("protocol_sha256") != candidate_receipt.get("protocol_sha256")
        ):
            raise ValueError(f"G0 actor/export/receipt identity disagrees: {actor_id}")
        actors.append({
            "id": actor_id, "path": expected["path"], "sha256": expected["sha256"],
            "candidate_path": candidate_path, "candidate_sha256": candidate_sha,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": candidate_receipt["checkpoint_sha256"],
            "source_study_protocol_path": source_protocol_path,
            "source_sha256": payload["source_sha256"],
            "export_protocol_sha256": payload["protocol_sha256"],
            "training_seed": expected["training_seed"], "environment_steps": 131072,
            "action_mode": "exported_tanh_mean",
        })
    protocol = {
        "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
        "purpose": "first-loss-and-completion-diagnostic-not-ranking",
        "geometry_audit_path": audit_path, "geometry_audit_sha256": audit_sha256,
        "r5_erratum_path": R5_ERRATUM_PATH,
        "r5_erratum_sha256": audit["r5_erratum"]["sha256"],
        "cells": cells, "actors": actors, "source_hashes": sources,
        "frame_skip": 4, "max_steps": 2000, "decision_cap": 48000,
        "reward_shaping": False, "collision_penalty": 0.0,
        "interventions": False, "learner_updates": 0,
        "centerline_far_threshold_m": 9.0,
        "centerline_far_definition": "nearest track center point distance > 9.0 m is a proxy, not physical grass occupancy",
        "event_rules": {
            "max_decisions": 2000, "negative_reward_limit": 100,
            "stall_window": 20, "tile_window": 30, "directed_delta_epsilon": 0.0001,
        },
        "censoring": "2,000-decision task timeout is separate from a prematurely stopped collection",
        "execution_order": "cells in audited order, each with both actors in listed order",
        "actor_attribution": "two frozen local exports diagnosed on the same roads; no matched training intervention claim",
    }
    destination = root / output
    if (not isinstance(output, str) or Path(output).is_absolute()
            or Path(output).parent != Path("experiments") or ".." in Path(output).parts
            or "g0" not in Path(output).stem.lower()
            or Path(output).suffix != ".json"
            or destination.exists()):
        raise ValueError("G0 protocol output must be a new experiments/*g0*.json file")
    _write_json_exclusive(destination, protocol)
    return protocol, sha256_file(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol, digest = freeze_g0_protocol(
        ROOT, audit_path=args.audit, audit_sha256=args.audit_sha256, output=args.output,
    )
    print(json.dumps({"protocol_sha256": digest, "status": "frozen",
                      "cells": len(protocol["cells"]), "actors": len(protocol["actors"])}))


if __name__ == "__main__":
    main()
