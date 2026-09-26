"""Audit and freeze the pre-interaction pixel-RLPD pilot protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.rlpd_common import (
    ROOT,
    audit_candidate_seed_tokens,
    canonical_sha256,
    geometry_seeds_from_ledger,
    runtime_metadata,
    sha256_file,
    write_json,
)
# Match the executable collector/trainer import graph before locking the effective
# installed-distribution inventory.
from common_adapter import EpisodeCollector as _episode_collector
from drq_v2 import load_exported_actor as _load_teacher_actor
from train import build_env as _build_training_environment


TEACHER_RUN = Path("runs/20260922-drq-augmentation-pad-v1-restart/control-seed1")
TEACHER_ACTOR = TEACHER_RUN / "checkpoints/step-000131072/actor.pt"
TEACHER_CHECKPOINT = TEACHER_RUN / "checkpoints/step-000131072/checkpoint.pt"
TEACHER_ACTOR_SHA256 = "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954"
TEACHER_CHECKPOINT_SHA256 = "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4"
TEACHER_PROTOCOL_SHA256 = "4a383ec6b83c501a814b0a18c22a39b851d23b22093d56244aa6540b977f160b"

SOURCE_PATHS = (
    "action_representation.py",
    "action_smoothing.py",
    "agent.py",
    "common_adapter.py",
    "core/__init__.py",
    "core/finish_line.py",
    "core/obstacle_contacts.py",
    "core/track_variables.py",
    "core/vendor/__init__.py",
    "core/vendor/car_dynamics.py",
    "core/vendor/car_racing.py",
    "damage.py",
    "docs/competition/restrictions.md",
    "docs/evaluation/generalization-policy.md",
    "docs/evaluation/protocol.md",
    "docs/plans/pixel-rlpd-offpolicy-plan.md",
    "drq_v2.py",
    "env_wrapper.py",
    "evaluate_policy.py",
    "haic/algorithms/rlpd/agent.py",
    "haic/algorithms/rlpd/augment.py",
    "haic/algorithms/rlpd/model.py",
    "haic/algorithms/rlpd/replay.py",
    "package_submission.py",
    "requirements.txt",
    "scripts/collect_rlpd_prior.py",
    "scripts/freeze_rlpd_pilot.py",
    "scripts/rlpd_common.py",
    "scripts/run_rlpd_pilot.py",
    "scripts/train_rlpd.py",
    "tests/test_evaluate_policy.py",
    "tests/test_rlpd_augmentation.py",
    "tests/test_rlpd_model.py",
    "tests/test_rlpd_replay.py",
    "tests/test_rlpd_run.py",
    "tests/test_submission_package.py",
    "tests/test_submission_policy.py",
    "tracking.py",
    "train.py",
)

TRAINING_SEEDS = list(range(4_000_004_001, 4_000_004_033))
PILOT_SCREEN_SEEDS = list(range(4_000_006_001, 4_000_006_005))
PILOT_CONFIRMATION_SEEDS = list(range(4_000_006_011, 4_000_006_019))
PILOT_BLIND_SEEDS = list(range(4_000_006_021, 4_000_006_029))
FULL_TRAINING_SEEDS = list(range(4_000_005_001, 4_000_005_065))
FULL_SCREEN_SEEDS = list(range(4_000_006_031, 4_000_006_039))
FULL_CONFIRMATION_SEEDS = list(range(4_000_006_041, 4_000_006_049))
FULL_BLIND_SEEDS = list(range(4_000_006_051, 4_000_006_059))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/pixel-rlpd-offpolicy-pilot-v2.json",
    )
    return parser.parse_args()


def source_training_ledgers():
    entries = []
    for training_seed in (0, 1):
        run_dir = Path(
            f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{training_seed}"
        )
        ledger = run_dir / "episodes.jsonl"
        if not ledger.is_file():
            raise FileNotFoundError(ledger)
        geometry_seeds = geometry_seeds_from_ledger(ROOT / ledger)
        entries.append({
            "source_training_seed": training_seed,
            "path": str(ledger),
            "sha256": sha256_file(ROOT / ledger),
            "reset_geometry_seed_count": len(geometry_seeds),
            "geometry_seed_set_sha256": canonical_sha256(sorted(geometry_seeds)),
            "geometry_seeds": sorted(geometry_seeds),
        })
    return entries


def historical_reservations(source_ledgers):
    reserved = set()
    for experiment in (ROOT / "experiments").glob("*.json"):
        try:
            document = json.loads(experiment.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        values = document.get("reserved_training_seeds", [])
        if isinstance(values, list):
            reserved.update(value for value in values if type(value) is int)
        for partition in document.get("partitions", {}).values():
            values = partition.get("seeds", []) if isinstance(partition, dict) else []
            if isinstance(values, list):
                reserved.update(value for value in values if type(value) is int)
    for ledger in source_ledgers:
        reserved.update(ledger["geometry_seeds"])
    # Previously documented residual-option teacher diagnostics are consumed data,
    # not a fresh RLPD training or evaluation allocation.
    reserved.update(range(4_000_000_001, 4_000_000_009))
    reserved.update(range(4_200_000_001, 4_200_000_005))
    return sorted(reserved)


def training_cells():
    return [
        {
            "track_id": 1 + index % 4,
            "geometry_seed": seed,
            "obstacles": True,
        }
        for index, seed in enumerate(TRAINING_SEEDS)
    ]


def source_hashes():
    hashes = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source missing at freeze: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def build_protocol():
    actor_path = ROOT / TEACHER_ACTOR
    checkpoint_path = ROOT / TEACHER_CHECKPOINT
    teacher_protocol_path = ROOT / (TEACHER_RUN / "protocol.json")
    teacher_config_path = ROOT / (TEACHER_RUN / "config.json")
    if sha256_file(actor_path) != TEACHER_ACTOR_SHA256:
        raise ValueError("nominated source actor changed from the run-ledger identity")
    if sha256_file(checkpoint_path) != TEACHER_CHECKPOINT_SHA256:
        raise ValueError("nominated source learner checkpoint changed from the run-ledger identity")
    if sha256_file(teacher_protocol_path) != TEACHER_PROTOCOL_SHA256:
        raise ValueError("nominated source protocol changed from the run-ledger identity")
    teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
    run_config = teacher_config.get("config", {})
    drq_config = run_config.get("drq_config", {})
    if (
        run_config.get("algorithm") != "drq-v2"
        or run_config.get("total_steps") != 131072
        or run_config.get("seed") != 1
        or drq_config.get("augmentation_pad") != 4
    ):
        raise ValueError("nominated source run is not the original pad-4 DrQ seed-1 control")

    source_ledgers = source_training_ledgers()
    pilot = sorted(set(PILOT_SCREEN_SEEDS + PILOT_CONFIRMATION_SEEDS + PILOT_BLIND_SEEDS))
    full = sorted(set(FULL_SCREEN_SEEDS + FULL_CONFIRMATION_SEEDS + FULL_BLIND_SEEDS))
    candidate_seeds = sorted(set(TRAINING_SEEDS + FULL_TRAINING_SEEDS + pilot + full))
    token_audit = audit_candidate_seed_tokens(candidate_seeds)
    if token_audit["known_recorded_uses"]:
        collisions = token_audit["known_recorded_uses"]
        raise ValueError(f"proposed candidate geometry has recorded uses: {collisions[:10]}")

    pilot_train_tracks = [1, 2, 3, 4]
    old_reserved = historical_reservations(source_ledgers)
    all_evaluation = pilot + full
    all_reserved = sorted(set(old_reserved + all_evaluation))
    partitions = {
        "screen": {"track_ids": [101, 102, 103], "seeds": PILOT_SCREEN_SEEDS, "repeats": 2},
        "confirmation": {
            "track_ids": [311, 312, 313, 314],
            "seeds": PILOT_CONFIRMATION_SEEDS,
            "repeats": 2,
        },
        "blind": {"track_ids": [321, 322, 323], "seeds": PILOT_BLIND_SEEDS, "repeats": 2},
    }
    return {
        "format": "haic-pixel-rlpd-study-v1",
        "schema_version": 1,
        "name": "pixel-rlpd-offpolicy-pilot-v2",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "internal-local-study-only; no official confirmation, model confirmation, or submission",
        "hypothesis": "At matched online student decisions and learner updates, a frozen prior dataset sampled 50:50 with online replay may improve screen completion over the same pixel SAC without offline replay.",
        "comparison": "RLPD prior-data arm versus the matched online-only ten-Q pixel SAC control; not standard two-Q SAC or DrQ algorithm superiority.",
        "source": {
            "rlpd_repository": "https://github.com/ikostrikov/rlpd",
            "rlpd_commit": "c90fd4baf28c9c9ef40a81460a2e395092844f88",
            "port": "independent PyTorch implementation; only pinned pixel algorithm contracts are ported",
        },
        "teacher": {
            "algorithm": "DrQ-v2 pad-4 control, original training seed 1",
            "actor_path": str(TEACHER_ACTOR),
            "actor_sha256": TEACHER_ACTOR_SHA256,
            "checkpoint_path": str(TEACHER_CHECKPOINT),
            "checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "source_protocol_path": str(TEACHER_RUN / "protocol.json"),
            "source_protocol_sha256": TEACHER_PROTOCOL_SHA256,
            "source_config_path": str(TEACHER_RUN / "config.json"),
            "source_config_sha256": sha256_file(teacher_config_path),
            "source_training_seed": 1,
            "source_checkpoint_environment_decisions": 131072,
            "reuse_scope": "frozen action generator only; no original trajectory, screen/confirmation/blind path, critic, replay, optimizer, or checkpoint state is imported",
            "prior_use": "The same actor previously served as a frozen base in DrQ residual-option development; its old trajectory is not imported.",
        },
        "frame_skip": 4,
        "max_steps": 2000,
        "training_track_ids": pilot_train_tracks,
        "training_geometry_seeds": TRAINING_SEEDS,
        "teacher_data_cells": training_cells(),
        "teacher_data_budget": {
            "decisions": 8192,
            "minimum_distinct_finishes": 2,
            "partial_episode": "drop without fabricating terminal; count all consumed decisions",
            "source_repeats": "one attempt per frozen training geometry cell",
        },
        "student_training": {
            "arms": ["rlpd", "sac-online-only-control"],
            "learner_seeds": [0, 1],
            "steps_per_run": 16384,
            "first_update_step": 1000,
            "random_decisions": 1000,
            "policy_takeover_step": 2000,
            "updates_per_decision": 1,
            "expected_gradient_steps_per_run": 15384,
            "batch_size": 64,
            "offline_batch_size": 32,
            "online_batch_size": 32,
            "backup_entropy": False,
            "replay_capacity": 100000,
            "offline_capacity": 8192,
            "training_sampler": "independent default_rng(seed+0x5A11) samples uniform track/geometry pairs from the frozen training pool; matched arms share the same indexed stream",
            "learner": {
                "actor_lr": 0.0003,
                "critic_lr": 0.0003,
                "temperature_lr": 0.0003,
                "gamma": 0.99,
                "tau": 0.005,
                "num_qs": 10,
                "num_min_qs": 1,
                "target_entropy": -1.5,
                "initial_alpha": 0.1,
                "augmentation_pad": 4,
                "n_step": 1,
                "weight_decay": 0.0,
            },
            "temperature_fidelity_warning": "The pinned author loss with entropy=-log_pi and target_entropy=-1.5 has a positive dL/d(log_alpha) for ordinary positive differential entropy, so this port can reduce alpha throughout training. It is retained verbatim for source fidelity; do not describe -1.5 as a positive entropy target.",
        },
        "partitions": partitions,
        "future_full_reservation": {
            "status": "reserved_not_opened; requires new full protocol and successful pilot gate",
            "training_track_ids": [1, 2, 3, 4],
            "training_geometry_seeds": FULL_TRAINING_SEEDS,
            "teacher_data_cells": [
                {
                    "track_id": 1 + index % 4,
                    "geometry_seed": seed,
                    "obstacles": True,
                }
                for index, seed in enumerate(FULL_TRAINING_SEEDS)
            ],
            "teacher_data_budget": {"decisions": 16384, "minimum_distinct_finishes": 4},
            "screen": {"track_ids": [131, 132, 133], "seeds": FULL_SCREEN_SEEDS, "repeats": 2},
            "confirmation": {"track_ids": [141, 142, 143, 144], "seeds": FULL_CONFIRMATION_SEEDS, "repeats": 2},
            "blind": {"track_ids": [151, 152, 153], "seeds": FULL_BLIND_SEEDS, "repeats": 2},
        },
        "reserved_training_seeds": all_reserved,
        "reservation_contract": {
            "meaning": "cross-track uint32 exclusion list; every pilot/full evaluation geometry and all prior documented reservations are excluded from teacher/student training",
            "training_sampler": "This implementation samples only the explicitly frozen training_geometry_seeds subset.",
            "historical_reserved_count": len(old_reserved),
            "total_reserved_count": len(all_reserved),
        },
        "geometry_audit": {
            "teacher_source_ledgers": source_ledgers,
            "candidate_seed_token_audit": {
                **token_audit,
                "candidate_seed_count": len(candidate_seeds),
                "candidate_seeds_sha256": canonical_sha256(candidate_seeds),
                "newly_scanned_seed_use_count": len(token_audit["known_recorded_uses"]),
            },
            "freshness_statement": "No known recorded use in searchable text artifacts or either original DrQ training ledger. Older pilot and sampled-training schedules are incomplete; this is not a global historical non-use proof. Candidate cells are frozen, not yet consumed.",
            "cross_lane_coordination": "No RLPD geometry is borrowed from the independent DrQ teacher-replay study. Candidate ranges were posted for collision coordination; the available prior message reported no frozen new cell allocation at audit time.",
        },
        "evaluation": {
            "purpose": "custom internal screen; confirmation/blind seeds are reserved but not opened by the pilot",
            "screen_candidate_steps": [8192, 16384],
            "cells_per_candidate": 12,
            "repeats": 2,
            "selection_order": ["canonical_finish_rate", "avg_progress", "-avg_lap_time_ms"],
            "tie_break": "earlier checkpoint",
            "pilot_gate": "both RLPD learner seeds must have an eligible deterministic actor with at least one canonical screen finish; each selected RLPD seed's finish count must be no lower than its matched online-only SAC seed",
            "stop_on_failure": "Record failed/ineligible; do not expand this pilot's collection/training/evaluation budget or open confirmation/blind. Any new hypothesis requires its own source/protocol and fresh allocation.",
            "official_score_claim": False,
        },
        "environment": {
            "track_ids": pilot_train_tracks,
            "obstacles": True,
            "reward_shaping": False,
            "reward_normalization": False,
            "collision_penalty": 0.0,
            "action_smoothing": None,
            "action_control": None,
            "observation": {"dtype": "float32", "shape": [4, 84, 84], "range": [0.0, 1.0]},
            "action": {"native": "symmetric three-dimensional [-1,1]", "executed_q_action": True},
        },
        "runtime": {
            "training": runtime_metadata(device="cuda"),
            "teacher_collection": runtime_metadata(device="cpu"),
            "cpu_evaluation_contract": {
                "python": "3.11",
                "torch": "2.1.0+cpu",
                "numpy": "1.26.0",
                "gymnasium": "0.29.1",
                "opencv": "4.8.1.78",
                "verified_interpreter": "/tmp/kilo/haic-cpu21/bin/python",
                "source": "existing local CPU21 runtime, verified at protocol freeze; not modified official environment",
            },
        },
        "source_hashes": source_hashes(),
        "execution": {
            "collector": "python -m scripts.collect_rlpd_prior --protocol experiments/pixel-rlpd-offpolicy-pilot-v2.json --output runs/20260924-pixel-rlpd-pilot-v2/prior-data",
            "trainer_template": "python -m scripts.train_rlpd --protocol experiments/pixel-rlpd-offpolicy-pilot-v2.json --offline-dataset runs/20260924-pixel-rlpd-pilot-v2/prior-data --run-dir runs/20260924-pixel-rlpd-pilot-v2/{arm}-seed{seed} --arm {rlpd|sac} --seed {0|1} --device cuda",
            "screen": "custom evaluate_policy protocol-file, `screen` partition only; use the verified CPU21 interpreter. Do not execute reserved confirmation/blind cells.",
        },
        "limitations": [
            "Freshness is no-known-recorded-use under an incomplete historical record, not global proof.",
            "Two student seeds and the small correlated screen are internal descriptive evidence, not broad statistical confirmation.",
            "The online-only control matches the pixel learner, not total environment cost including the prior teacher dataset.",
            "A favorable pilot screen alone is not a model promotion, official result, submission, or authorization to reuse holdouts.",
        ],
    }


def main():
    args = parse_args()
    output = args.output.resolve()
    protocol = build_protocol()
    if output.exists():
        raise FileExistsError(output)
    write_json(output, protocol, exclusive=True)
    print(json.dumps({
        "protocol": str(output.relative_to(ROOT)),
        "protocol_sha256": sha256_file(output),
        "source_file_count": len(protocol["source_hashes"]),
        "reserved_training_seed_count": len(protocol["reserved_training_seeds"]),
        "candidate_seed_count": protocol["geometry_audit"]["candidate_seed_token_audit"]["candidate_seed_count"],
        "known_recorded_seed_uses": protocol["geometry_audit"]["candidate_seed_token_audit"]["known_recorded_uses"],
        "training_runtime_sha256": protocol["runtime"]["training"]["installed_distributions_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
