"""Freeze a blind-safe, failure-led training-road search before any new reset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from scripts.audit_drq_training_seeds import audit_training_seeds
from scripts.generate_drq_training_geometry import _protocol as validate_catalog_protocol


STUDY_ID = "drqv2-geometry-augmentation-v1"
ANALYSIS = "experiments/drqv2-geometry-augmentation-v1-analysis.json"
PROTOCOL = "experiments/drqv2-geometry-augmentation-v1.json"
RUN_ROOT = "runs/20260925-drqv2-geometry-augmentation-v1"
SEED_START = 3910800001
SEED_COUNT = 512
PROTOCOL_SOURCES = (
    "experiments/drqv2-teacher-replay-v1-r3.json",
    "experiments/drqv2-augmentation-pad-v1.json",
    "experiments/drqv2-steering-logit-v1.json",
    "experiments/drqv2-promotion-v1.json",
    "experiments/pixel-rlpd-long-horizon-followup-v1.json",
    "experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json",
)
PRIOR_AUDIT_SOURCES = ("experiments/drqv2-teacher-replay-v1-r3-geometry-audit.json",)
TRAINING_LEDGERS = tuple(
    f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}/episodes.jsonl"
    for seed in (0, 1)
)
SOURCE_PATHS = (
    "core/vendor/car_racing.py", "core/track_variables.py", "core/finish_line.py",
    "common_adapter.py", "env_wrapper.py", "train.py", "drq_v2.py", "agent.py",
    "haic/algorithms/drq_v2/geometry_features.py",
    "scripts/analyze_drq_geometry_failures.py", "scripts/audit_drq_training_seeds.py",
    "scripts/generate_drq_training_geometry.py", "scripts/freeze_drq_training_geometry.py",
    "scripts/diagnose_drq_training_geometry.py",
)
CONSUMED_SHAPE_SEEDS = (
    *range(31001, 31009), *range(32101, 32109), *range(33101, 33109),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _exclusive(path: Path, data: Any) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"freeze parent does not exist: {path.parent}")
    with path.open("xb") as destination:
        destination.write(_json_bytes(data))


def family_rules() -> list[dict[str, Any]]:
    """Six supported *measured* shape neighborhoods, not fake generator knobs."""
    return [
        {
            "name": "opening-short-entry-left-turn",
            "direction_source": "early",
            "thresholds": {
                "first_bend_start_m": {"min": 35, "max": 90},
                "first_bend_abs_turn_rad": {"min": 1.15, "max": 2.2},
            },
            "targeted_failure_mode": "source-specific collision-free failures after the first short-entry left bend",
        },
        {
            "name": "opening-delayed-high-turn",
            "direction_source": "early",
            "thresholds": {
                "first_bend_start_m": {"min": 105, "max": 240},
                "first_bend_abs_turn_rad": {"min": 1.35},
            },
            "targeted_failure_mode": "test whether longer acceleration before the first large turn changes entry failure",
        },
        {
            "name": "easy-curvature-anchor",
            "direction_source": "early",
            "thresholds": {
                "first_bend_abs_turn_rad": {"min": 0.3, "max": 1.15},
                "abs_curvature_p90_per_m": {"max": 0.072},
                "mid_reversals_65m": {"max": 2},
            },
            "targeted_failure_mode": "retain lower curvature and reversal exposure alongside boundary roads",
        },
        {
            "name": "mid-road-left-right-reversal",
            "direction_source": "mid_s",
            "thresholds": {"mid_reversals_65m": {"min": 2}},
            "targeted_failure_mode": "short-gap opposite turns and delayed steering/throttle recovery",
        },
        {
            "name": "mid-road-sustained-or-same-turn",
            "direction_source": "mid_curve",
            "thresholds": {
                "consecutive_same_sign_65m": {"min": 1},
                "mid_bend_count": {"min": 2},
            },
            "targeted_failure_mode": "repeated same-sign bends with intervening partial recovery",
        },
        {
            "name": "finish-approach-turn",
            "direction_source": "late",
            "thresholds": {
                "late_strongest_abs_turn_rad": {"min": 1.25},
                "late_bend_count": {"min": 1},
            },
            "targeted_failure_mode": "late bend before directed 95%-qualified finish-line crossing",
        },
    ]


def _source_mapping(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    return {path: _sha(root / path) for path in paths}


def _excluded_ids(root: Path) -> dict[str, list[int]]:
    previous = json.loads((root / PRIOR_AUDIT_SOURCES[0]).read_text(encoding="utf-8"))
    if previous.get("passed") is not True or previous.get("parse_errors") != []:
        raise ValueError("previous documented seed audit did not pass")
    reserved = set(previous["known_excluded_geometry_seeds"]) | set(previous["candidate_seeds"])
    heldout: set[int] = set()
    blind: set[int] = set()
    for relative in PROTOCOL_SOURCES:
        protocol = json.loads((root / relative).read_text(encoding="utf-8"))
        for key in ("reserved_training_seeds", "training_geometry_seeds", "known_excluded_geometry_seeds"):
            reserved.update(protocol.get(key, []))
        for name, partition in protocol.get("partitions", {}).items():
            seeds = set(partition["seeds"])
            reserved.update(seeds)
            if name == "blind":
                blind.update(seeds)
            elif name in ("screen", "confirmation"):
                heldout.update(seeds)
        for pool in protocol.get("training_pools", {}).values():
            reserved.update(pool["seeds"])
    return {
        "reserved": sorted(reserved),
        "heldout": sorted(heldout),
        "blind": sorted(blind),
    }


def _reference_rows(analysis: dict[str, Any], analysis_sha: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roads = analysis.get("road_geometry")
    if not isinstance(roads, dict):
        raise ValueError("analysis lacks consumed-only road descriptors")
    consumed: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    receipt = {"path": ANALYSIS, "sha256": analysis_sha}
    for seed in CONSUMED_SHAPE_SEEDS:
        road = roads.get(str(seed))
        if not isinstance(road, dict) or road.get("signature") is None:
            raise ValueError(f"historical consumed screen/confirmation road missing: {seed}")
        consumed.append({
            "geometry_seed": seed,
            "partition": "screen" if seed in range(31001, 31009) else "confirmation",
            "consumed": True,
            "road_coordinate_sha256": road["road_coordinate_sha256"],
            "signature": road["signature"],
            "evidence": receipt,
        })
    for seed in range(4260100001, 4260100009):
        road = roads.get(str(seed))
        if not isinstance(road, dict) or road.get("signature") is None:
            raise ValueError(f"consumed r3 teacher training road missing: {seed}")
        training.append({
            "seed": seed,
            "origin": "training",
            "road_coordinate_sha256": road["road_coordinate_sha256"],
            "signature_sha256": road["signature_sha256"],
            "signature": road["signature"],
        })
    return consumed, training


def freeze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    output = root / PROTOCOL
    study_dir = root / RUN_ROOT
    if output.exists() or study_dir.exists():
        raise FileExistsError("a frozen geometry protocol or study run root already exists")
    analysis_path = root / ANALYSIS
    analysis_sha = _sha(analysis_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("format") != "haic-drq-consumed-geometry-failures-v1" or analysis.get("status") != "complete":
        raise ValueError("frozen consumed-only analysis did not complete")
    if any("blind" in name.lower() for name in analysis.get("inputs_sha256_exact_allowlist", {})):
        raise ValueError("analysis input inventory includes a blind path")
    sources = {
        "protocol_sources": _source_mapping(root, PROTOCOL_SOURCES),
        "prior_audit_sources": _source_mapping(root, PRIOR_AUDIT_SOURCES),
        "training_ledgers": _source_mapping(root, TRAINING_LEDGERS),
    }
    candidates = list(range(SEED_START, SEED_START + SEED_COUNT))
    audit = audit_training_seeds(candidates, repo_root=root, **sources)
    if audit.get("passed") is not True or audit.get("matched_collisions") != []:
        raise ValueError("training seed audit did not pass")
    exclusions = _excluded_ids(root)
    if set(candidates) & set(exclusions["reserved"]):
        raise ValueError("candidate seeds overlap exclusion-only protocols")
    consumed, training = _reference_rows(analysis, analysis_sha)
    study_dir.mkdir(parents=True)
    seed_audit_path = study_dir / "candidate-seed-audit.json"
    corpus_path = study_dir / "training-corpus-signatures.json"
    _exclusive(seed_audit_path, audit)
    _exclusive(corpus_path, {
        "format": "haic-drq-training-corpus-signatures-v1",
        "origin": "already consumed r3 teacher-training roads",
        "analysis_sha256": analysis_sha,
        "training_roads": training,
    })
    corpus_receipt = {"path": corpus_path.relative_to(root).as_posix(), "sha256": _sha(corpus_path)}
    for row in training:
        row["evidence"] = corpus_receipt
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              check=True, capture_output=True, text=True).stdout.strip()
    protocol = {
        "format": "haic-drq-training-geometry-protocol-v1",
        "study_id": STUDY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_sha256": _source_mapping(root, SOURCE_PATHS),
        "generator_version": "official-participants-1c11db8-local-vendor",
        "generator_control": {
            "geometry": "uint32 seed only",
            "track_id": "obstacle arrangement only; road geometry unchanged",
            "checkpoint_count": 12,
            "road_half_width_m": 40.0 / 6.0,
            "no_independent_curvature_or_width_knobs": True,
        },
        "source_actors": [
            {
                "learner_seed": actor["learner_seed"],
                "actor_path": actor["actor_path"],
                "actor_sha256": actor["actor_sha256"],
                "actor_weights_sha256": actor["source_pair_audit"]["actor_weights_sha256"],
            }
            for actor in json.loads((root / PROTOCOL_SOURCES[0]).read_text(encoding="utf-8"))["source_actors"]
        ],
        "analysis_receipt": {"path": ANALYSIS, "sha256": analysis_sha},
        "seed_audit": sources,
        "seed_audit_receipt": {
            "path": seed_audit_path.relative_to(root).as_posix(), "sha256": _sha(seed_audit_path),
        },
        "candidate_seeds": candidates,
        "catalog_selection_seed": SEED_START,
        "train_target": 120,
        "diagnostic_target": 16,
        "selection_stages": ["representative_6x10", "variant_6x10", "train_diagnostic_16"],
        "family_rules": family_rules(),
        "near_duplicate_threshold": 0.06,
        "near_duplicate_allow_mirror": True,
        "exclusion_seed_ids": exclusions,
        "structural_reference_seeds": consumed,
        "training_corpus_signatures": training,
        "no_score_selection": True,
        "blind_scope": "excluded seed IDs only; never reset or inspect blind geometry/outcomes",
        "historical_freshness_scope": audit["freshness_claim"],
        "runtime": {
            "geometry_interpreter": "python3.11 with existing Box2D/Gymnasium",
            "diagnostic_interpreter": "/tmp/kilo/haic-cpu21/bin/python",
            "diagnostic_device": "cpu",
        },
    }
    _exclusive(output, protocol)
    validate_catalog_protocol(output, root)
    return {
        "study_id": STUDY_ID,
        "protocol_path": PROTOCOL,
        "protocol_sha256": _sha(output),
        "candidate_count": len(candidates),
        "blind_data_access": audit["blind_data_access"],
        "seed_audit_path": seed_audit_path.relative_to(root).as_posix(),
        "seed_audit_sha256": _sha(seed_audit_path),
        "training_reference_count": len(training),
        "consumed_heldout_reference_count": len(consumed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        result = freeze(args.repo_root)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
