"""One source1-only complement on twelve already consumed r6 TRAIN roads.

The caller must separately freeze experiments/dreamerv3-reused-train-source1-v1.json
and pass its SHA-256. Required keys: format, study_id, purpose, freshness_claim,
source0_protocol, source0_teacher_receipt, r6_protocol, catalog,
historical_train_summary, source_sha256, cells, episode_schedule, frame_skip,
max_steps, budgets, source_actor, spec_fingerprints. No protocol, simulator,
learner, or output is created on import.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from common_adapter import ActionSpec, ObservationSpec
from scripts.diagnose import collect_dreamerv3_reused_train as reused


ROOT = reused.ROOT
FORMAT = "haic-dreamerv3-reused-train-source1-v1"
STUDY_ID = "dreamerv3-reused-train-source1-v1"
PROTOCOL_PATH = "experiments/dreamerv3-reused-train-source1-v1.json"
OUTPUT_PATH = "runs/20260926-dreamerv3-reused-train-source1-v1/collection"
WRAPPER_PATH = "scripts/diagnose/collect_dreamerv3_reused_source1.py"
ORIGINAL_COLLECTOR_SHA256 = "3217f00a626694372b4c935d6d4503bdc8d407867d62df53cee42d6d9c5dea17"
SOURCE0_PROTOCOL = "experiments/dreamerv3-reused-train-diversity-v1.json"
SOURCE0_PROTOCOL_SHA256 = "a2f3e92fd1b2e95f883b24e3ef28d58b3e70938a0dd867d2c9ec5da5bd46df83"
SOURCE0_RECEIPT = "runs/20260926-dreamerv3-reused-train-diversity-v1/collection/teacher/collection-result.json"
SOURCE0_RECEIPT_SHA256 = "00a388f5eabdd9707b096dfb735b2d717b309b86a9daa1423d2c16177fa2c353"
CATALOG_PATH = "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json"
CATALOG_SHA256 = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
HISTORICAL_PATH = "experiments/drqv2-geometry-augmentation-v1-training-summary.json"
HISTORICAL_SHA256 = "29fe9d2b8d4eff04879cc9c1390b4701e4f235af710856f39acf6592f65015ee"
ACTOR_SHA256 = "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954"
CHECKPOINT_SHA256 = "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4"
WEIGHTS_SHA256 = "e0984f709d1971754f2e9b64a69bf216922ca5bd0791ad247181b218f830b736"
SOURCE_REVISION = "3ea51d6"
FAMILIES = {
    3910800008: "opening-short-entry-left-turn",
    3910800011: "opening-delayed-high-turn",
    3910800019: "easy-curvature-anchor",
    3910800007: "mid-road-left-right-reversal",
    3910800074: "mid-road-sustained-or-same-turn",
    3910800069: "finish-approach-turn",
    3910800010: "opening-short-entry-left-turn",
    3910800016: "opening-delayed-high-turn",
    3910800041: "easy-curvature-anchor",
    3910800030: "mid-road-left-right-reversal",
    3910800081: "mid-road-sustained-or-same-turn",
    3910800070: "finish-approach-turn",
}
SOURCE0_FINISHES = (3910800011, 3910800041)
HISTORICAL_SOURCE1_FINISHES = (3910800008, 3910800011, 3910800041, 3910800074)


def _reference(root: Path, name: str, digest: str) -> Path:
    """Pin fixed receipt/catalog paths without admitting symlinked parents."""
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden: {name}")
    if not path.is_file() or reused.sha256(path) != digest:
        raise ValueError(f"pinned reference SHA-256 mismatch: {name}")
    return path


def _check_extra_pins(root: Path, wrapper_sha256: str) -> None:
    _reference(root, WRAPPER_PATH, wrapper_sha256)
    if reused.sha256(Path(__file__)) != wrapper_sha256:
        raise ValueError("executing source1 wrapper differs from pinned source")
    if reused.sha256(Path(reused.__file__)) != ORIGINAL_COLLECTOR_SHA256:
        raise ValueError("executing original collector differs from frozen runtime source")
    _reference(root, SOURCE0_PROTOCOL, SOURCE0_PROTOCOL_SHA256)
    _reference(root, SOURCE0_RECEIPT, SOURCE0_RECEIPT_SHA256)
    _reference(root, CATALOG_PATH, CATALOG_SHA256)
    _reference(root, HISTORICAL_PATH, HISTORICAL_SHA256)


def preflight(protocol_path: Path, protocol_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if reused._relative(root, protocol_path) != PROTOCOL_PATH:
        raise ValueError("source1 protocol path is not the single fixed TRAIN protocol")
    protocol = reused._json(reused._pinned(root, PROTOCOL_PATH, protocol_sha256, kind="protocol"))
    if (set(protocol) != {"format", "study_id", "purpose", "freshness_claim", "source0_protocol",
                          "source0_teacher_receipt", "r6_protocol", "catalog", "historical_train_summary",
                          "source_sha256", "cells",
                          "episode_schedule", "frame_skip", "max_steps", "budgets", "source_actor",
                          "spec_fingerprints"}
            or protocol["format"] != FORMAT or protocol["study_id"] != STUDY_ID
            or protocol["purpose"] != reused.PURPOSE or protocol["freshness_claim"] != reused.LIMITATION
            or protocol["source0_protocol"] != {"path": SOURCE0_PROTOCOL, "sha256": SOURCE0_PROTOCOL_SHA256}
            or protocol["source0_teacher_receipt"] != {"path": SOURCE0_RECEIPT, "sha256": SOURCE0_RECEIPT_SHA256}
            or protocol["r6_protocol"] != {"path": reused.R6_PATH, "sha256": reused.R6_SHA256}
            or protocol["catalog"] != {"path": CATALOG_PATH, "sha256": CATALOG_SHA256}
            or protocol["historical_train_summary"] != {"path": HISTORICAL_PATH, "sha256": HISTORICAL_SHA256}):
        raise ValueError("source1 protocol must pin exact original and catalog references")
    source0 = reused._json(reused._pinned(root, SOURCE0_PROTOCOL, SOURCE0_PROTOCOL_SHA256, kind="protocol"))
    receipt = reused._json(_reference(root, SOURCE0_RECEIPT, SOURCE0_RECEIPT_SHA256))
    r6 = reused._json(reused._pinned(root, reused.R6_PATH, reused.R6_SHA256, kind="r6"))
    if (source0.get("study_id") != "dreamerv3-reused-train-diversity-v1"
            or source0.get("format") != reused.FORMAT or source0.get("r6_protocol") != protocol["r6_protocol"]
            or source0.get("frame_skip") != 4 or source0.get("max_steps") != 2000
            or source0.get("episode_schedule") != list(range(12))
            or source0.get("budgets") != {"random_decision_cap": 24000, "teacher_decision_cap": 24000}
            or source0.get("source_actor", {}).get("source_id") != "drq-source-0"
            or source0["source_actor"].get("learner_seed") != 0):
        raise ValueError("source0 frozen protocol does not establish the original schedule")
    cells = [{"track_id": 1, "geometry_seed": seed} for seed in FAMILIES]
    if (source0.get("cells") != cells or protocol["cells"] != cells
            or protocol["episode_schedule"] != list(range(12))
            or type(protocol["frame_skip"]) is not int or protocol["frame_skip"] != 4
            or type(protocol["max_steps"]) is not int or protocol["max_steps"] != 2000
            or protocol["budgets"] != {"teacher_decision_cap": 24000}):
        raise ValueError("source1 must attempt the same twelve unique source0 TRAIN cells once")
    if (not isinstance(r6.get("training_pool"), dict) or r6["training_pool"].get("partition") != "TRAIN"
            or not set(FAMILIES).issubset(set(reused._seeds(r6["training_pool"].get("geometry_seeds"), "r6 TRAIN")))
            or not isinstance(r6.get("diagnostic_pool"), dict)
            or r6["diagnostic_pool"].get("partition") != "TRAIN-DIAGNOSTIC"
            or set(FAMILIES) & set(reused._seeds(r6["diagnostic_pool"].get("geometry_seeds"), "r6 diagnostic"))
            or r6.get("environment", {}).get("partition") != "TRAIN"
            or r6["environment"].get("frame_skip") != 4 or r6["environment"].get("max_steps") != 2000
            or r6["environment"].get("reward_shaping") is not False
            or r6["environment"].get("obstacles") is not True
            or r6.get("learner", {}).get("padding") != 4
            or r6["learner"].get("source_environment_steps") != 131072):
        raise ValueError("r6 does not establish the fixed unshaped TRAIN environment")
    if (receipt.get("status") != "completed" or receipt.get("arm") != "teacher"
            or receipt.get("study_id") != source0["study_id"]
            or receipt.get("protocol_sha256") != SOURCE0_PROTOCOL_SHA256
            or receipt.get("r6_protocol_sha256") != reused.R6_SHA256
            or receipt.get("source_id") != "drq-source-0"
            or receipt.get("source_actor_sha256") != source0["source_actor"]["actor_sha256"]
            or receipt.get("source_checkpoint_sha256") != source0["source_actor"]["checkpoint_sha256"]
            or receipt.get("allowed_cells") != cells or receipt.get("decision_cap") != 24000
            or receipt.get("schedule_attempts") != 12 or receipt.get("complete_episode_count") != 12
            or receipt.get("schedule_exhausted") is not True or receipt.get("partial_decisions") != 0
            or receipt.get("unresolved_decision_calls") != 0 or not receipt.get("archive_sha256")
            or receipt.get("distinct_finished_geometries") != list(SOURCE0_FINISHES)
            or not isinstance(receipt.get("episode_rows"), list) or len(receipt["episode_rows"]) != 12):
        raise ValueError("source0 teacher reference is not the completed failed-gate collection")
    for attempt, (row, cell) in enumerate(zip(receipt["episode_rows"], cells)):
        if (not isinstance(row, dict) or row.get("attempt") != attempt
                or any(row.get(key) != value for key, value in cell.items())
                or row.get("status") != "complete" or row.get("complete") is not True
                or row.get("terminal") is not True or row.get("episode_id") != attempt
                or type(row.get("decisions")) is not int or not 1 <= row["decisions"] <= 2000
                or row.get("finished") is not (cell["geometry_seed"] in SOURCE0_FINISHES)):
            raise ValueError("source0 teacher rows disagree with the original completed schedule")
    catalog = reused._json(_reference(root, CATALOG_PATH, CATALOG_SHA256))
    train = catalog.get("train")
    if catalog.get("format") != "haic-drq-training-geometry-catalog-v1" or not isinstance(train, list):
        raise ValueError("pinned catalog has no TRAIN family inventory")
    found: dict[int, str] = {}
    for item in train:
        if isinstance(item, dict) and item.get("geometry_seed") in FAMILIES:
            seed = item["geometry_seed"]
            if seed in found:
                raise ValueError("duplicate TRAIN geometry in catalog")
            found[seed] = item.get("family")
    if found != FAMILIES or len({found[seed] for seed in SOURCE0_FINISHES}) != 2:
        raise ValueError("catalog TRAIN families disagree with the predeclared twelve roads")
    historical = reused._json(_reference(root, HISTORICAL_PATH, HISTORICAL_SHA256))
    geometry = historical.get("geometry")
    if (historical.get("format") != "haic-drq-training-geometry-diagnostic-summary-v1"
            or historical.get("catalog_sha256") != CATALOG_SHA256 or not isinstance(geometry, list)):
        raise ValueError("historical TRAIN diagnostic summary is not the pinned catalog's summary")
    prior: dict[int, bool] = {}
    for item in geometry:
        if isinstance(item, dict) and item.get("geometry_seed") in FAMILIES:
            seed = item["geometry_seed"]
            actor = item.get("by_source", {}).get("1")
            if (seed in prior or item.get("partition") != "train" or item.get("family") != FAMILIES[seed]
                    or not isinstance(actor, dict) or type(actor.get("finished")) is not bool):
                raise ValueError("historical source1 TRAIN road identity is inconsistent")
            prior[seed] = actor["finished"]
    if (set(prior) != set(FAMILIES)
            or sorted(seed for seed, finished in prior.items() if finished) != list(HISTORICAL_SOURCE1_FINISHES)):
        raise ValueError("historical source1 finishes differ from pinned prior TRAIN diagnostic")
    sources = protocol["source_sha256"]
    if (not isinstance(source0.get("source_sha256"), dict)
            or set(source0["source_sha256"]) != reused.REQUIRED_SOURCES
            or not isinstance(sources, dict)
            or set(sources) != reused.REQUIRED_SOURCES | {WRAPPER_PATH}
            or sources.get(reused.COLLECTOR_PATH) != ORIGINAL_COLLECTOR_SHA256
            or any(sources[name] != digest for name, digest in source0["source_sha256"].items())):
        raise ValueError("source1 must pin the exact original runtime source map plus its wrapper")
    _check_extra_pins(root, reused._hash(sources[WRAPPER_PATH], "wrapper SHA"))
    if protocol["spec_fingerprints"] != {"action": ActionSpec().fingerprint,
                                          "observation": ObservationSpec().fingerprint}:
        raise ValueError("source1 action/observation fingerprint mismatch")
    actors = r6.get("source_actors")
    if (not isinstance(actors, list) or len(actors) != 2
            or actors[0].get("source_seed") != 0 or actors[1].get("source_seed") != 1
            or actors[1].get("weight_only_fork") is not True
            or actors[1].get("source_revision") != SOURCE_REVISION
            or actors[1].get("source_checkpoint_sha256") != CHECKPOINT_SHA256
            or actors[1].get("source_actor_sha256") != ACTOR_SHA256
            or actors[1].get("source_actor_weights_sha256") != WEIGHTS_SHA256):
        raise ValueError("r6 does not identify the frozen pad-4 source1 actor")
    source = protocol["source_actor"]
    expected_source = {
        "source_id": "drq-source-1", "learner_seed": 1, "source_revision": SOURCE_REVISION,
        "checkpoint_path": actors[1]["source_checkpoint_path"], "checkpoint_sha256": CHECKPOINT_SHA256,
        "actor_path": actors[1]["source_actor_path"], "actor_sha256": ACTOR_SHA256,
        "actor_weights_sha256": WEIGHTS_SHA256,
    }
    if source != expected_source or source["actor_sha256"] == source0["source_actor"]["actor_sha256"]:
        raise ValueError("source1 identity differs from the pinned r6 seed1 actor")
    checked = {"protocol": {**protocol, "source_sha256": source0["source_sha256"]},
               "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha256,
               "r6_protocol_sha256": reused.R6_SHA256,
               "excluded_seeds": sorted(reused._seeds(r6["diagnostic_pool"]["geometry_seeds"], "r6 diagnostic"))}
    reused._check_pins(root, checked)
    pair = reused.audit_source_actor_pair(
        root / source["checkpoint_path"], root / source["actor_path"], learner_seed=1,
        source_revision=SOURCE_REVISION, expected_checkpoint_sha256=CHECKPOINT_SHA256,
        expected_actor_sha256=ACTOR_SHA256,
    )
    for key, expected in (("learner_seed", 1), ("source_revision", SOURCE_REVISION),
                          ("checkpoint_sha256", CHECKPOINT_SHA256), ("actor_sha256", ACTOR_SHA256),
                          ("actor_weights_sha256", WEIGHTS_SHA256),
                          ("action_fingerprint", protocol["spec_fingerprints"]["action"]),
                          ("observation_fingerprint", protocol["spec_fingerprints"]["observation"]),
                          ("cpu_smoke_observations", 3)):
        if pair.get(key) != expected:
            raise ValueError(f"source1 CPU actor/checkpoint parity failed: {key}")
    checked.update(source_pair_audit=pair, wrapper_sha256=sources[WRAPPER_PATH])
    return checked


def collect(protocol_path: Path, protocol_sha256: str, output_dir: Path, *,
            repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if reused._relative(root, output_dir) != OUTPUT_PATH:
        raise ValueError("output must be the single new source1-only TRAIN collection directory")
    output = root / OUTPUT_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (not output.parent.is_dir() or any((root / part).is_symlink() for part in
            ("runs", "runs/20260926-dreamerv3-reused-train-source1-v1"))):
        raise ValueError("output parent must exist without symlinks")
    checked = preflight(protocol_path, protocol_sha256, repo_root=root)
    source = checked["protocol"]["source_actor"]
    actor, adapter, observation_spec = reused.load_exported_actor(root / source["actor_path"], device="cpu")
    if (adapter.spec.fingerprint != checked["protocol"]["spec_fingerprints"]["action"]
            or observation_spec.fingerprint != checked["protocol"]["spec_fingerprints"]["observation"]):
        raise ValueError("loaded source1 actor contract differs from audited CPU source")
    reused._check_pins(root, checked)
    _check_extra_pins(root, checked["wrapper_sha256"])
    output.mkdir(exist_ok=False)
    arm: dict[str, Any] | None = None
    builder = reused.build_env

    def guarded_builder(*args: Any, **kwargs: Any) -> Any:
        # _collect_arm rechecks all original runtime/actor pins before this call.
        _check_extra_pins(root, checked["wrapper_sha256"])
        return builder(*args, **kwargs)

    try:
        with patch.object(reused, "build_env", guarded_builder):
            arm = reused._collect_arm(root, checked, output, "teacher", actor, adapter, observation_spec)
        reused._check_pins(root, checked)
        _check_extra_pins(root, checked["wrapper_sha256"])
        if (arm["status"] != "completed" or arm["source_id"] != "drq-source-1"
                or arm["schedule_attempts"] != 12 or arm["schedule_exhausted"] is not True
                or arm["complete_episode_count"] != 12 or arm["partial_decisions"] != 0
                or arm["unresolved_decision_calls"] != 0 or arm["stored_decisions"] != arm["decisions_spent"]
                or not arm["archive_sha256"] or not arm["dataset_digest"]
                or any(row["status"] != "complete" or row["complete"] is not True or row["terminal"] is not True
                       for row in arm["episode_rows"])):
            raise ValueError("source1 collection has incomplete attempts or archive integrity")
        finished = arm["distinct_finished_geometries"]
        new = sorted(set(finished) - set(SOURCE0_FINISHES))
        union = sorted(set(finished) | set(SOURCE0_FINISHES))
        families = sorted({FAMILIES[seed] for seed in union})
        result = {
            "format": "haic-dreamerv3-reused-train-source1-result-v1", "study_id": STUDY_ID,
            "purpose": reused.PURPOSE, "freshness_claim": reused.LIMITATION,
            "protocol_sha256": protocol_sha256, "r6_protocol_sha256": reused.R6_SHA256,
            "catalog": {"path": CATALOG_PATH, "sha256": CATALOG_SHA256},
            "historical_source1_reference": {
                "path": HISTORICAL_PATH, "sha256": HISTORICAL_SHA256,
                "finished_geometries_on_same_reused_train_roads": list(HISTORICAL_SOURCE1_FINISHES),
                "interpretation": "earlier single attempts; this collection is replication, not unseen-road discovery",
            },
            "source_id": "drq-source-1", "source_actor_sha256": ACTOR_SHA256,
            "source_checkpoint_sha256": CHECKPOINT_SHA256, "source_pair_audit": checked["source_pair_audit"],
            "source0_reference": {"protocol_path": SOURCE0_PROTOCOL, "protocol_sha256": SOURCE0_PROTOCOL_SHA256,
                                  "teacher_receipt_path": SOURCE0_RECEIPT, "teacher_receipt_sha256": SOURCE0_RECEIPT_SHA256,
                                  "finished_geometries": list(SOURCE0_FINISHES),
                                  "original_learner_gate_passed": False},
            "source1_finished_geometries": finished, "new_finished_geometries": new,
            "new_finished_geometries_relative_to": "the pinned source0 teacher receipt, not historical source1",
            "union_finished_geometries": union, "union_finished_families": families,
            "support_gate": {"min_new_finished_roads": 1, "min_union_finished_roads": 3,
                             "min_union_shape_families": 2, "requires_complete_twelve_attempts": True,
                             "passed": bool(new and len(union) >= 3 and len(families) >= 2),
                             "permits": "design of a separately pinned multi-source model-data protocol only",
                             "authorizes_training": False, "merges_archives": False},
            "status": "completed", "allowed_cells": checked["protocol"]["cells"],
            "teacher": arm,
        }
        reused._write_json(output / "collection-result.json", result)
        return result
    except BaseException as exc:
        receipt_path = output / "teacher/collection-result.json"
        if arm is None and receipt_path.is_file():
            try:
                arm = reused._json(receipt_path)
            except (OSError, ValueError):
                pass  # A damaged receipt cannot certify how many decisions were consumed.
        reused._write_json(output / "abort-receipt.json", {
            "format": "haic-dreamerv3-reused-train-source1-abort-v1", "study_id": STUDY_ID,
            "source_id": "drq-source-1", "protocol_sha256": protocol_sha256,
            "source0_protocol_sha256": SOURCE0_PROTOCOL_SHA256,
            "source0_teacher_receipt_sha256": SOURCE0_RECEIPT_SHA256,
            "failed_arm_receipt_path": "teacher/collection-result.json" if receipt_path.is_file() else None,
            "consumption_known": arm is not None,
            "decision_calls": arm["decision_calls"] if arm else None,
            "decisions_spent": arm["decisions_spent"] if arm else None,
            "unresolved_decision_calls": arm["unresolved_decision_calls"] if arm else None,
            "schedule_attempts": arm["schedule_attempts"] if arm else None,
            "stored_decisions": arm["stored_decisions"] if arm else None,
            "archive_sha256": arm["archive_sha256"] if arm else None,
            "reason": f"{type(exc).__name__}: {exc}",
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
