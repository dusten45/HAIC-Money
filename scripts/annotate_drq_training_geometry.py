"""Join immutable training-road catalog and frozen actor diagnostics per road."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from haic.algorithms.drq_v2.geometry_features import signature_distance


CATALOG = Path("runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json")
CATALOG_SHA = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
SCAN = Path("runs/20260925-drqv2-geometry-augmentation-v1/catalog/scan.json")
SCAN_SHA = "1ba960e34fd5a045d155639bd2aa89ff99671fda49782ef7bc19cf8688eaa34d"
PROTOCOL = Path("experiments/drqv2-geometry-augmentation-v1.json")
PROTOCOL_SHA = "be6d1d1c3b1b16f6b56fd4720097ca8fea8b64c5c962d07183961cdfa0d87294"
SUMMARY = Path("experiments/drqv2-geometry-augmentation-v1-training-summary.json")
SUMMARY_SHA = "29fe9d2b8d4eff04879cc9c1390b4701e4f235af710856f39acf6592f65015ee"
OUTPUT = Path("experiments/drqv2-geometry-augmentation-v1-final-set.json")


def _pinned(root: Path, relative: Path, expected: str) -> dict[str, Any]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"frozen training-only evidence path is missing: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"frozen training-only evidence hash changed: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"frozen evidence must be a JSON object: {relative}")
    return value


def annotate(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / OUTPUT
    if output.exists():
        raise FileExistsError("per-road final set already exists; do not overwrite provenance")
    catalog = _pinned(root, CATALOG, CATALOG_SHA)
    scan = _pinned(root, SCAN, SCAN_SHA)
    protocol = _pinned(root, PROTOCOL, PROTOCOL_SHA)
    summary = _pinned(root, SUMMARY, SUMMARY_SHA)
    if (catalog.get("protocol_sha256") != PROTOCOL_SHA
            or scan.get("protocol_sha256") != PROTOCOL_SHA
            or summary.get("catalog_sha256") != CATALOG_SHA
            or summary.get("protocol_sha256") != "5935664e132603de62cd35601fda12b998ea2009548332736720bab905d2c3f3"):
        raise ValueError("training-road catalog and CPU diagnostic lineage disagree")
    selected = catalog["train"] + catalog["train_diagnostic"]
    if len(catalog["train"]) != 120 or len(catalog["train_diagnostic"]) != 16:
        raise ValueError("training-only partitions must remain the original 120+16")
    results = {row["geometry_seed"]: row for row in summary["geometry"]}
    selected_scan = {row["geometry_seed"]: row for row in scan["rows"] if row["status"] == "selected"}
    seeds = [row["geometry_seed"] for row in selected]
    if (len(set(seeds)) != 136 or set(results) != set(seeds)
            or set(selected_scan) != set(seeds)):
        raise ValueError("catalog, scan and diagnostic road inventories differ")
    rules = {item["name"]: item for item in protocol["family_rules"]}
    reference = protocol["structural_reference_seeds"]
    previous_train = protocol["training_corpus_signatures"]
    threshold = protocol["near_duplicate_threshold"]
    forbidden = set().union(*(set(protocol["exclusion_seed_ids"][key])
                              for key in ("reserved", "heldout", "blind")))
    if set(seeds) & forbidden:
        raise ValueError("training-road catalog overlaps exclusion-only held-out/blind IDs")
    annotated = []
    for row in selected:
        seed = row["geometry_seed"]
        result, source_scan = results[seed], selected_scan[seed]
        partition = "TRAIN" if row in catalog["train"] else "TRAIN-DIAGNOSTIC"
        if (result["road_coordinate_sha256"] != row["road_coordinate_sha256"]
                or result["family"] != row["family"]
                or result["partition"] != partition.lower().replace("-", "_")
                or source_scan["family"] != row["family"]
                or source_scan["stage"] != row["stage"]
                or source_scan["road_coordinate_sha256"] != row["road_coordinate_sha256"]):
            raise ValueError(f"road {seed} changed between selection and frozen actor diagnosis")
        known_distance = min(
            signature_distance(row["signature"], past["signature"], allow_mirror=True)
            for past in reference
        )
        prior_train_distance = min(
            signature_distance(row["signature"], past["signature"], allow_mirror=True)
            for past in previous_train
        )
        if known_distance < threshold or prior_train_distance < threshold:
            raise ValueError(f"road {seed} violates preregistered structural near-duplicate threshold")
        annotated.append({
            "geometry_id": str(seed),
            "geometry_seed": seed,
            "track_id": row["track_id"],
            "partition": partition,
            "family": row["family"],
            "stage": row["stage"],
            "geometric_turn_direction": row["direction"],
            "targeted_failure_hypothesis": rules[row["family"]]["targeted_failure_mode"],
            "generator_parameters": {
                "generator_version": protocol["generator_version"],
                "geometry_seed": seed,
                "track_id_for_obstacles": row["track_id"],
                "fixed_half_width_m": row["features"]["fixed_half_width_m"],
                "checkpoint_count": protocol["generator_control"]["checkpoint_count"],
                "curvature_width_explicit_controls": False,
            },
            "measured_features": {
                key: row["features"][key] for key in (
                    "perimeter_m", "abs_curvature_p90_per_m", "early_strongest_bend",
                    "mid_strongest_bend", "late_strongest_bend", "mid_reversals_65m",
                    "mid_consecutive_same_sign_65m",
                )
            },
            "road_coordinate_sha256": row["road_coordinate_sha256"],
            "known_consumed_heldout_nearest_signature_distance": round(known_distance, 6),
            "known_training_nearest_signature_distance": round(prior_train_distance, 6),
            "blind_structural_proximity": "unverified; blind seed IDs excluded without reading shapes",
            "selected_before_any_actor_result": True,
            "inclusion_reason": (
                "precommitted measured-feature family quota, independent of actor score"
                if partition == "TRAIN"
                else "precommitted disjoint TRAIN-DIAGNOSTIC quota, not learner replay"
            ),
            "physical_sanity": row["verification"],
            "static_structure": row["structure"],
            "difficulty": result["provisional_difficulty"],
            "finish_reachability": result["finish_reachability"],
            "frozen_source_actor_results": result["by_source"],
        })
    artifact = {
        "format": "haic-drq-training-geometry-final-set-v1",
        "catalog_path": CATALOG.as_posix(), "catalog_sha256": CATALOG_SHA,
        "catalog_protocol_path": PROTOCOL.as_posix(), "catalog_protocol_sha256": PROTOCOL_SHA,
        "diagnostic_summary_path": SUMMARY.as_posix(), "diagnostic_summary_sha256": SUMMARY_SHA,
        "scan_path": SCAN.as_posix(), "scan_sha256": SCAN_SHA,
        "train_geometry_count": len(catalog["train"]),
        "train_diagnostic_geometry_count": len(catalog["train_diagnostic"]),
        "difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in annotated).items())),
        "removed_for_policy_failure": 0,
        "preselection_malformed_seed_ids": [
            row["geometry_seed"] for row in scan["rows"]
            if row.get("reason") == "invalid_static_structure"
        ],
        "roads": annotated,
    }
    with output.open("xb") as destination:
        destination.write((json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return {"output": OUTPUT.as_posix(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "train": len(catalog["train"]), "train_diagnostic": len(catalog["train_diagnostic"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        print(json.dumps(annotate(args.repo_root), sort_keys=True))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
