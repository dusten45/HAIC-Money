"""Freeze matched DrQ-v2 training distributions over the accepted road catalog."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from haic.algorithms.drq_v2.geometry_sampler import CATALOG_PATH, CATALOG_SHA256, PROTOCOL_PATH as CATALOG_PROTOCOL_PATH, PROTOCOL_SHA256 as CATALOG_PROTOCOL_SHA256
from scripts.validate_drq_teacher_protocol import _agent_drq_compatibility_record, _adapter_compatibility_record


STUDY_ID = "drqv2-geometry-mix-v1-r6"
PROTOCOL_PATH = Path("experiments/drqv2-geometry-mix-v1-r6.json")
RUN_ROOT = Path("runs/20260925-drqv2-geometry-mix-v1-r6")
SUPERSEDED_PROTOCOL = Path("experiments/drqv2-geometry-mix-v1-r5.json")
SUPERSESSION_RECEIPT = Path("runs/20260925-drqv2-geometry-mix-v1-r5/protocol-superseded-after-partial.json")
SOURCE_PROTOCOL = Path("experiments/drqv2-teacher-replay-v1-r3.json")
SOURCE_PROTOCOL_SHA256 = "7b73cab4fb46ca2fb8e34b99a8667022f3027640e1582343b0b91aa4df4e0e14"
ANALYSIS_PATH = Path("experiments/drqv2-geometry-augmentation-v1-analysis.json")
ANALYSIS_SHA256 = "5fe93686e4f4cd7a68eddab8127e71809ac6b81a5baa27e8a4dc774ee1a53cf7"
CATALOG_RESULT_PATH = Path("experiments/drqv2-geometry-augmentation-v1-catalog-result.json")
TRAINER_SCRIPT = "scripts/train_drq_geometry_mix.py"
DIAGNOSTIC_SCRIPT = "scripts/diagnose_drq_geometry_mix.py"

_RUNTIME_FILES = {
    "agent.py", "action_representation.py", "action_smoothing.py", "common_adapter.py",
    "damage.py", "drq_v2.py", "env_wrapper.py", "train.py", "tracking.py",
    "requirements.txt",
    "haic/algorithms/drq_v2/__init__.py", "haic/algorithms/drq_v2/teacher_replay.py",
    "haic/algorithms/drq_v2/teacher_study.py", "haic/algorithms/drq_v2/geometry_sampler.py",
    "haic/algorithms/drq_v2/geometry_features.py",
    "scripts/train_drq_geometry_mix.py", "scripts/diagnose_drq_geometry_mix.py",
    "scripts/summarize_drq_training_geometry.py", "scripts/annotate_drq_training_geometry.py",
    "scripts/probe_drq_training_finish.py",
}
_DOCUMENT_FILES = {
    "AGENTS.md", "docs/context/current-state.md", "docs/experiments/INDEX.md",
    "docs/plans/active/drqv2-geometry-mix-plan.md",
    "docs/plans/drqv2-teacher-replay-plan.md", "docs/architecture/overview.md",
    "docs/evaluation/protocol.md", "docs/evaluation/generalization-policy.md",
    "docs/experiments/drqv2-geometry-augmentation-v1.md",
}
_TEST_FILES = {
    "tests/test_drq_geometry_features.py", "tests/test_drq_geometry_sampler.py",
    "tests/test_drq_geometry_mix_training.py", "tests/test_drq_geometry_mix_diagnostics.py",
    "tests/test_freeze_drq_geometry_mix.py",
}

VARIANTS = [
    {
        "name": "uniform",
        "hypothesis": "Equal exposure to all six selected shape families preserves the broadest initial road mixture.",
        "family_tiers": {
            "opening-short-entry-left-turn": "boundary",
            "opening-delayed-high-turn": "boundary",
            "easy-curvature-anchor": "boundary",
            "mid-road-left-right-reversal": "boundary",
            "mid-road-sustained-or-same-turn": "boundary",
            "finish-approach-turn": "boundary",
        },
        "tier_weights": {"easy": 0.0, "boundary": 1.0, "difficult": 0.0},
    },
    {
        "name": "failure_weighted",
        "hypothesis": "Emphasize the measured short-entry and mid-reversal neighborhoods while retaining easier and late-turn roads.",
        "family_tiers": {
            "opening-short-entry-left-turn": "difficult",
            "opening-delayed-high-turn": "difficult",
            "easy-curvature-anchor": "easy",
            "mid-road-left-right-reversal": "difficult",
            "mid-road-sustained-or-same-turn": "boundary",
            "finish-approach-turn": "boundary",
        },
        "tier_weights": {"easy": 0.20, "boundary": 0.25, "difficult": 0.55},
    },
    {
        "name": "easy_retention",
        "hypothesis": "Retain prior easy-road competence while adding moderate opening/mid roads and a smaller late-approach dose.",
        "family_tiers": {
            "opening-short-entry-left-turn": "boundary",
            "opening-delayed-high-turn": "boundary",
            "easy-curvature-anchor": "easy",
            "mid-road-left-right-reversal": "boundary",
            "mid-road-sustained-or-same-turn": "boundary",
            "finish-approach-turn": "difficult",
        },
        "tier_weights": {"easy": 0.45, "boundary": 0.40, "difficult": 0.15},
    },
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exclusive(path: Path, value: Any) -> None:
    if path.exists() or not path.parent.is_dir():
        raise FileExistsError(f"freeze output exists or parent is absent: {path}")
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as target:
        target.write(raw)


def _source_files(root: Path, fixed_files: set[str], *, include_core: bool = False) -> dict[str, str]:
    expected = set(fixed_files)
    if include_core:
        expected.update(path.relative_to(root).as_posix() for path in (root / "core").rglob("*.py"))
    missing = sorted(name for name in expected if not (root / name).is_file())
    if missing:
        raise FileNotFoundError(f"geometry mix source snapshot missing: {missing}")
    return {name: _sha(root / name) for name in sorted(expected)}


def freeze(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    output_path = root / PROTOCOL_PATH
    run_root = root / RUN_ROOT
    if output_path.exists() or run_root.exists() or (root / "experiments/drqv2-geometry-mix-v1-diagnostic-protocol.json").exists():
        raise FileExistsError("geometry mix protocol/run root already exists")
    if not (root / SUPERSEDED_PROTOCOL).is_file() or not (root / SUPERSESSION_RECEIPT).is_file():
        raise FileNotFoundError("preinteraction r1 supersession receipt is missing")
    supersession = json.loads((root / SUPERSESSION_RECEIPT).read_text(encoding="utf-8"))
    superseded_sha = _sha(root / SUPERSEDED_PROTOCOL)
    if (supersession.get("superseded_protocol_sha256") != superseded_sha
            or supersession.get("study_id") != "drqv2-geometry-mix-v1-r5"
            or supersession.get("online_decisions") != 16384
            or supersession.get("learner_updates") != 6384
            or supersession.get("full_checkpoint_written") is not False
            or supersession.get("resume_allowed") is not False
            or supersession.get("evaluation_receipts") != []):
        raise ValueError("r4 partial arm cannot be superseded as the reported precheckpoint interruption")
    catalog_path = root / CATALOG_PATH
    catalog_protocol_path = root / CATALOG_PROTOCOL_PATH
    catalog_result_path = root / CATALOG_RESULT_PATH
    source_protocol_path = root / SOURCE_PROTOCOL
    analysis_path = root / ANALYSIS_PATH
    for path in (catalog_path, catalog_protocol_path, catalog_result_path, source_protocol_path, analysis_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha(catalog_path) != CATALOG_SHA256 or _sha(catalog_protocol_path) != CATALOG_PROTOCOL_SHA256:
        raise ValueError("geometry catalog or generation-protocol SHA changed")
    if _sha(source_protocol_path) != SOURCE_PROTOCOL_SHA256 or _sha(analysis_path) != ANALYSIS_SHA256:
        raise ValueError("consumed analysis/source geometry lineage changed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_protocol = json.loads(catalog_protocol_path.read_text(encoding="utf-8"))
    catalog_result = json.loads(catalog_result_path.read_text(encoding="utf-8"))
    if (catalog_result.get("status") != "success" or catalog_result.get("catalog_sha256") != CATALOG_SHA256
            or catalog.get("protocol_sha256") != CATALOG_PROTOCOL_SHA256
            or catalog_protocol.get("analysis_receipt") != {"path": ANALYSIS_PATH.as_posix(), "sha256": ANALYSIS_SHA256}):
        raise ValueError("only the fully frozen, successful training-only catalog is eligible")
    train_rows = catalog.get("train", [])
    diagnostic_rows = catalog.get("train_diagnostic", [])
    train_seeds = [row["geometry_seed"] for row in train_rows]
    diagnostic_seeds = [row["geometry_seed"] for row in diagnostic_rows]
    if len(train_seeds) != 120 or len(diagnostic_seeds) != 16 or set(train_seeds) & set(diagnostic_seeds):
        raise ValueError("catalog partition counts or road IDs differ from its frozen targets")
    family_names = [rule["name"] for rule in catalog_protocol["family_rules"]]
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    actors = []
    for actor in source_protocol["source_actors"]:
        if actor["learner_seed"] not in (0, 1):
            raise ValueError("unexpected frozen DrQ source learner seed")
        actor_path, checkpoint_path = root / actor["actor_path"], root / actor["checkpoint_path"]
        if _sha(actor_path) != actor["actor_sha256"] or _sha(checkpoint_path) != actor["checkpoint_sha256"]:
            raise ValueError("frozen pad-4 source actor or checkpoint changed")
        actors.append({
            "source_seed": actor["learner_seed"],
            "learner_seed": actor["learner_seed"],
            "source_revision": actor["source_revision"],
            "checkpoint_path": actor["checkpoint_path"],
            "checkpoint_sha256": actor["checkpoint_sha256"],
            "actor_path": actor["actor_path"],
            "actor_sha256": actor["actor_sha256"],
            "source_checkpoint_path": actor["checkpoint_path"],
            "source_checkpoint_sha256": actor["checkpoint_sha256"],
            "source_actor_path": actor["actor_path"],
            "source_actor_sha256": actor["actor_sha256"],
            "checkpoint_manifest_path": actor["checkpoint_manifest_path"],
            "checkpoint_manifest_sha256": actor["checkpoint_manifest_sha256"],
            "source_actor_weights_sha256": actor["source_pair_audit"]["actor_weights_sha256"],
            "weight_only_fork": True,
        })
    if len(actors) != 2:
        raise ValueError("exactly two matched source actor seeds are required")
    for variant in VARIANTS:
        if set(variant["family_tiers"]) != set(family_names) or set(variant["tier_weights"]) != {
            "easy", "boundary", "difficult",
        } or abs(sum(variant["tier_weights"].values()) - 1.0) > 1e-9:
            raise ValueError(f"distribution {variant['name']} does not cover frozen families or weights")
        active_tiers = {variant["family_tiers"][family] for family in family_names}
        if any(variant["tier_weights"][tier] > 0 and tier not in active_tiers for tier in ("easy", "boundary", "difficult")):
            raise ValueError(f"distribution {variant['name']} assigns mass to an empty tier")
    current_source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    variants = {variant["name"]: {
        "hypothesis": variant["hypothesis"],
        "family_tiers": variant["family_tiers"],
        "tier_weights": variant["tier_weights"],
        "score_selection": False,
    } for variant in VARIANTS}
    seeds = [
        {"learner_seed": source_seed, "study_seed": 101000 + source_seed,
         "track_seed": 102000 + source_seed,
         "geometry_seed": 103000 + source_seed,
         "actor_rng_seed": 104000 + source_seed,
         "replay_rng_seed": 105000 + source_seed,
         "target_noise_seed": 106000 + source_seed,
         "update_rng_seed": 107000 + source_seed}
        for source_seed in (0, 1)
    ]
    run_records = [
        {"source_seed": seed["learner_seed"], **seed, "variant": variant,
         "run_dir": f"{RUN_ROOT.as_posix()}/learner-{seed['learner_seed']}-{variant}"}
        for seed in seeds for variant in variants
    ]
    source_hashes = _source_files(root, _RUNTIME_FILES, include_core=True)
    document_hashes = _source_files(root, _DOCUMENT_FILES)
    test_hashes = _source_files(root, _TEST_FILES)
    runtime_names = ("torch", "numpy", "gymnasium", "opencv-python", "pygame", "box2d-py")
    runtime_versions = {name: importlib.metadata.version(name) for name in runtime_names}
    runtime_versions["python"] = platform.python_version()
    protocol = {
        "format": "haic-drq-geometry-mix-study-v1",
        "study_id": STUDY_ID,
        "run_root": RUN_ROOT.as_posix(),
        "supersedes": {
            "protocol_path": SUPERSEDED_PROTOCOL.as_posix(),
            "protocol_sha256": superseded_sha,
            "receipt_path": SUPERSESSION_RECEIPT.as_posix(),
            "receipt_sha256": _sha(root / SUPERSESSION_RECEIPT),
            "reason": "r5 stopped at its first checkpoint because the source map included mutable docs; r6 freezes executable runtime hashes separately",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": current_source_revision,
        "source_sha256": source_hashes,
        "source_documents_sha256": document_hashes,
        "source_tests_sha256": test_hashes,
        "protocol_builder_sha256": _sha(Path(__file__).resolve()),
        "runtime": {
            "training_executable": sys.executable,
            "training_device": "cuda:0",
            "training_dependencies": runtime_versions,
            "evaluation_executable": "/tmp/kilo/haic-cpu21/bin/python",
            "verified_interpreter": "/tmp/kilo/haic-cpu21/bin/python",
        },
        "catalog": {
            "path": CATALOG_PATH, "sha256": CATALOG_SHA256,
            "protocol_path": CATALOG_PROTOCOL_PATH,
            "protocol_sha256": CATALOG_PROTOCOL_SHA256,
            "catalog_result_path": CATALOG_RESULT_PATH.as_posix(),
        },
        "catalog_path": CATALOG_PATH,
        "catalog_sha256": CATALOG_SHA256,
        "catalog_protocol_path": CATALOG_PROTOCOL_PATH,
        "catalog_protocol_sha256": CATALOG_PROTOCOL_SHA256,
        "analysis_receipt": {"path": ANALYSIS_PATH.as_posix(), "sha256": ANALYSIS_SHA256},
        "consumed_teacher_replay_data_use": "none",
        "source_actors": actors,
        "environment": {
            "partition": "TRAIN", "geometry_seeds": train_seeds,
            "track_ids": [1, 2, 3, 4], "frame_skip": 4,
            "max_steps": 2000, "obstacles": True,
            "reward_shaping": False, "reward_normalization": False,
        },
        "training_pool": {
            "partition": "TRAIN",
            "geometry_seeds": train_seeds,
            "track_ids": [1, 2, 3, 4],
        },
        "diagnostic_pool": {
            "partition": "TRAIN-DIAGNOSTIC", "geometry_seeds": diagnostic_seeds,
            "track_ids": sorted({row["track_id"] for row in diagnostic_rows}),
            "repeats": 2, "max_steps": 1200, "frame_skip": 4,
            "role": "development diagnostic only; reused historical diagnostic geometry; not fresh confirmation",
            "selected_for_retraining": False,
        },
        "variants": variants,
        "seeds_by_source": seeds,
        "runs": run_records,
        "learner": {
            "algorithm": "DrQ-v2", "source_environment_steps": 131072,
            "observation_shape": [4, 84, 84], "action_dim": 3,
            "feature_dim": 256, "hidden_dim": 256,
            "padding": 4, "actor_lr": 0.0001, "critic_lr": 0.0001,
            "tau": 0.01, "actor_update_frequency": 2, "target_update_frequency": 2,
            "target_noise_std": 0.2, "target_noise_clip": 0.5,
            "steering_logit_l2": 0.0,
            "exploration_initial_std": 0.2, "exploration_final_std": 0.05,
            "exploration_duration": 100000,
            "batch_size": 64,
        },
        "budgets": {
            "additional_online_steps": 32768,
            "warmup_steps": 10000,
            "expected_gradient_updates": 22768,
            "replay_capacity": 100000,
            "n_step": 3, "gamma": 0.99,
            "critic_updates_per_learning_decision": 1,
            "checkpoint_steps": [16384, 32768],
            "batch_size": 64,
        },
        "replay": {"mode": "online-only", "online_rows": 64, "teacher_rows": 0},
        "diagnostics": {
            "format": "haic-drq-geometry-mix-diagnostic-v1",
            "role": "training_diagnostic",
            "partition": "TRAIN-DIAGNOSTIC",
            "geometry_seeds": diagnostic_seeds,
            "track_ids": sorted({row["track_id"] for row in diagnostic_rows}),
            "repeats": 2,
            "max_steps": 1200,
            "frame_skip": 4,
            "raw_reward": True,
            "cpu_device": True,
            "ranked": False,
            "source_actors_included": True,
            "no_official_performance_claim": True,
            "candidate_runs": [
                {"source_seed": run["source_seed"], "variant": run["variant"],
                 "run_dir": run["run_dir"]}
                for run in run_records
            ],
            "source_actors": actors,
            "runtime": {"verified_interpreter": "/tmp/kilo/haic-cpu21/bin/python"},
        },
        "diagnostic_runtime": {"python": "/tmp/kilo/haic-cpu21/bin/python"},
        "selection": {
            "development_only": True,
            "official_ranked": False,
            "do_not_retrain_or_select_new_weights_after_diagnostic": True,
            "future_fresh_confirmation_required": True,
        },
        "excluded_protocol_context": {
            "teacher_replay_a3_status": "r3 inconclusive; no dataset reuse",
            "screen_confirmation_blind": "never opened by this study; reserved IDs remain excluded",
            "blind_access": "none",
        },
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    _exclusive(output_path, protocol)
    return {
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": _sha(output_path),
        "catalog_sha256": CATALOG_SHA256,
        "geometry_count": len(train_seeds),
        "diagnostic_geometry_count": len(diagnostic_seeds),
        "training_run_count": len(protocol["runs"]),
        "official_score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        print(json.dumps(freeze(args.repo_root), sort_keys=True, indent=2))
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
