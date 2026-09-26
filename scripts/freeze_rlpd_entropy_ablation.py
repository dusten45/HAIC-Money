"""Audit and freeze the one-factor pixel-RLPD entropy-target ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from common_adapter import EpisodeCollector as _episode_collector
from drq_v2 import load_exported_actor as _load_teacher_actor
from train import build_env as _build_training_environment
from scripts.freeze_rlpd_followup import SOURCE_PATHS as FOLLOWUP_SOURCES, _previous_reservations, _source_ledgers
from scripts.rlpd_common import ROOT, audit_candidate_seed_tokens, canonical_sha256, runtime_metadata, sha256_file, write_json
from scripts.rlpd_entropy_common import EXPECTED_ARM_SPECS


TRAINING_SEEDS = list(range(4_000_030_001, 4_000_030_065))
SCREEN_SEEDS = list(range(4_000_031_001, 4_000_031_009))
CONFIRMATION_SEEDS = list(range(4_000_031_011, 4_000_031_019))
BLIND_SEEDS = list(range(4_000_031_021, 4_000_031_029))
SOURCE_PATHS = sorted(set(FOLLOWUP_SOURCES) | {
    "scripts/freeze_rlpd_entropy_ablation.py",
    "scripts/project_rlpd_entropy_screen_receipts.py",
    "scripts/rlpd_entropy_common.py",
    "scripts/run_rlpd_entropy_ablation.py",
    "scripts/train_rlpd_entropy_ablation.py",
    "tests/test_rlpd_entropy_ablation.py",
})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v5.json",
    )
    return parser.parse_args()


def _cells(seeds):
    return [
        {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
        for index, seed in enumerate(seeds)
    ]


def _source_hashes():
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen ablation source missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def build_protocol():
    v3_path = ROOT / "experiments/pixel-rlpd-long-horizon-followup-v1.json"
    v3_result_path = ROOT / "experiments/pixel-rlpd-long-horizon-followup-v1-result.json"
    v4_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v1.json"
    v4_preflight_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v1-preflight.json"
    v2_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v2.json"
    v2_preflight_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v2-preflight.json"
    v3_entropy_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v3.json"
    v3_entropy_preflight_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v3-preflight.json"
    v4_entropy_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v4.json"
    v4_entropy_result_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v4-result.json"
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v3_result = json.loads(v3_result_path.read_text(encoding="utf-8"))
    v4_preflight = json.loads(v4_preflight_path.read_text(encoding="utf-8"))
    v2_preflight = json.loads(v2_preflight_path.read_text(encoding="utf-8"))
    v3_entropy_preflight = json.loads(v3_entropy_preflight_path.read_text(encoding="utf-8"))
    v4_entropy_result = json.loads(v4_entropy_result_path.read_text(encoding="utf-8"))
    if (
        v3_result.get("blind_opened") is not True
        or v3_result.get("blind", {}).get("canonical_finishes", 0) <= 0
        or v3_result.get("confirmation_gate") != "pass"
    ):
        raise ValueError("V4 ablation requires completed V3 confirmation and blind evidence")
    if v4_preflight.get("environment_interactions") != 0 or v4_preflight.get("cell_allocation_consumed") is not False:
        raise ValueError("V4 seed allocation must be retired only after verifying its zero-interaction preflight")
    if v2_preflight.get("environment_interactions") != 0 or v2_preflight.get("cell_allocation_consumed") is not False:
        raise ValueError("V2 seed allocation must be retired only after verifying its zero-interaction preflight")
    if v3_entropy_preflight.get("environment_interactions") != 0 or v3_entropy_preflight.get("cell_allocation_consumed") is not False:
        raise ValueError("V3 entropy seed allocation must be retired only after verifying its zero-interaction preflight")
    v4_evaluation = v4_entropy_result.get("evaluation", {})
    if (
        v4_entropy_result.get("status") != "aborted_before_screen_source_hash_gate"
        or v4_evaluation.get("screen_cells_started") != 0
        or v4_evaluation.get("confirmation_cells_started") != 0
        or v4_evaluation.get("blind_cells_started") != 0
    ):
        raise ValueError("V4 source-drift stop and zero evaluation-cell evidence must be verified before V5")
    teacher_run = Path("runs/20260922-drq-augmentation-pad-v1-restart/control-seed1")
    teacher_actor = teacher_run / "checkpoints/step-000131072/actor.pt"
    teacher_checkpoint = teacher_run / "checkpoints/step-000131072/checkpoint.pt"
    teacher_protocol = teacher_run / "protocol.json"
    teacher_config = teacher_run / "config.json"
    teacher_identity = {
        "actor_sha256": "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
        "checkpoint_sha256": "c806c0998d4a9241917dd368581b9625e293c16ff45e0e1fb7bd5759dd20b3d4",
        "source_protocol_sha256": "4a383ec6b83c501a814b0a18c22a39b851d23b22093d56244aa6540b977f160b",
    }
    if (
        sha256_file(ROOT / teacher_actor) != teacher_identity["actor_sha256"]
        or sha256_file(ROOT / teacher_checkpoint) != teacher_identity["checkpoint_sha256"]
        or sha256_file(ROOT / teacher_protocol) != teacher_identity["source_protocol_sha256"]
    ):
        raise ValueError("V4 nominated teacher differs from frozen DrQ pad-4 seed-1 bytes")
    teacher_run_config = json.loads((ROOT / teacher_config).read_text(encoding="utf-8")).get("config", {})
    if (
        teacher_run_config.get("algorithm") != "drq-v2"
        or teacher_run_config.get("seed") != 1
        or teacher_run_config.get("total_steps") != 131072
        or teacher_run_config.get("drq_config", {}).get("augmentation_pad") != 4
    ):
        raise ValueError("V4 teacher run config is not the original pad-4 seed-1 actor")

    ledgers = _source_ledgers()
    history = _previous_reservations(ledgers)
    evaluation_seeds = SCREEN_SEEDS + CONFIRMATION_SEEDS + BLIND_SEEDS
    reserved = sorted(set(history + evaluation_seeds))
    candidate_seeds = sorted(set(TRAINING_SEEDS + evaluation_seeds))
    token_audit = audit_candidate_seed_tokens(candidate_seeds)
    if token_audit["known_recorded_uses"]:
        raise ValueError(f"V5 fresh allocation collision: {token_audit['known_recorded_uses'][:10]}")
    if set(TRAINING_SEEDS).intersection(reserved):
        raise ValueError("V5 training pool intersects a previous/reserved uint32 geometry")
    if set(candidate_seeds).intersection(v3["training_geometry_seeds"]):
        raise ValueError("V5 geometry reuses V3 student or teacher prior-data experience")
    if set(candidate_seeds).intersection(
        seed for partition in v3["partitions"].values() for seed in partition["seeds"]
    ):
        raise ValueError("V5 evaluation geometry reuses any V3 consumed/reserved cell")

    partitions = {
        "screen": {"track_ids": [291, 292, 293], "seeds": SCREEN_SEEDS, "repeats": 2},
        "confirmation": {"track_ids": [301, 302, 303, 304], "seeds": CONFIRMATION_SEEDS, "repeats": 2},
        "blind": {"track_ids": [311, 312, 313], "seeds": BLIND_SEEDS, "repeats": 2},
    }
    learner = {
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
    }
    return {
        "format": "haic-pixel-rlpd-study-v1",
        "schema_version": 1,
        "name": "pixel-rlpd-entropy-target-ablation-v4",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "fresh one-factor entropy-target study after V3 and two zero-interaction preflight aborts; no earlier transition, checkpoint, or consumed/reserved cell is reused",
        "hypothesis": "The pinned alpha objective behaves differently at target_entropy=-1.5 versus +1.5; under matched fresh-data training, the alternative may change entropy/alpha trajectories and held-out finish rate.",
        "comparison": "two from-scratch RLPD variants only. Architecture, data bytes/mix, optimizer, initial alpha, loss, update schedule, seeds, and geometry stream are identical; target_entropy is the only treatment factor.",
        "prior_study": {
            "v3_protocol": str(v3_path.relative_to(ROOT)),
            "v3_protocol_sha256": sha256_file(v3_path),
            "v3_result": str(v3_result_path.relative_to(ROOT)),
            "v3_result_sha256": sha256_file(v3_result_path),
            "v3_blind_actor_sha256": v3_result["blind_finalist"]["actor_sha256"],
            "v3_prior_dataset_sha256": v3_result["prior_dataset_sha256"],
            "entropy_v2_protocol": str(v2_path.relative_to(ROOT)),
            "entropy_v2_protocol_sha256": sha256_file(v2_path),
            "entropy_v2_preflight": str(v2_preflight_path.relative_to(ROOT)),
            "entropy_v2_preflight_sha256": sha256_file(v2_preflight_path),
            "entropy_v2_environment_interactions": 0,
            "entropy_v3_protocol": str(v3_entropy_path.relative_to(ROOT)),
            "entropy_v3_protocol_sha256": sha256_file(v3_entropy_path),
            "entropy_v3_preflight": str(v3_entropy_preflight_path.relative_to(ROOT)),
            "entropy_v3_preflight_sha256": sha256_file(v3_entropy_preflight_path),
            "entropy_v3_environment_interactions": 0,
            "v4_protocol": str(v4_path.relative_to(ROOT)),
            "v4_protocol_sha256": sha256_file(v4_path),
            "v4_preflight": str(v4_preflight_path.relative_to(ROOT)),
            "v4_preflight_sha256": sha256_file(v4_preflight_path),
            "v4_environment_interactions": 0,
            "reuse_scope": "prior metrics only; V3 weights/transitions/checkpoints/screen/confirmation/blind and all retired V1/V2/V3 entropy-ablation allocations are excluded",
        },
        "teacher": {
            "algorithm": "DrQ-v2 pad-4 control, original training seed 1",
            "actor_path": str(teacher_actor),
            **teacher_identity,
            "checkpoint_path": str(teacher_checkpoint),
            "source_protocol_path": str(teacher_protocol),
            "source_config_path": str(teacher_config),
            "source_config_sha256": sha256_file(ROOT / teacher_config),
            "source_training_seed": 1,
            "source_checkpoint_environment_decisions": 131072,
            "reuse_scope": "frozen action generator only; collect all V5 transitions on fresh training-only geometry",
        },
        "frame_skip": 4,
        "max_steps": 2000,
        "training_track_ids": [1, 2, 3, 4],
        "training_geometry_seeds": TRAINING_SEEDS,
        "teacher_data_cells": _cells(TRAINING_SEEDS),
        "teacher_data_budget": {"decisions": 16384, "minimum_distinct_finishes": 4, "partial_episode": "drop without synthetic terminal; charge every decision", "source_repeats": "one attempt per frozen fresh geometry cell"},
        "student_training": {
            "arms": list(EXPECTED_ARM_SPECS),
            "arm_specs": EXPECTED_ARM_SPECS,
            "learner_seeds": [40, 41],
            "steps_per_run": 131072,
            "candidate_steps": [65536, 131072],
            "first_update_step": 1000,
            "random_decisions": 1000,
            "policy_takeover_step": 2000,
            "updates_per_decision": 1,
            "expected_gradient_steps_per_run": 130072,
            "batch_size": 64,
            "offline_batch_size": 32,
            "online_batch_size": 32,
            "backup_entropy": False,
            "replay_capacity": 100000,
            "offline_capacity": 16384,
            "training_sampler": "For each learner seed both target variants share identical fresh initial weights and default_rng(seed+0x5A11) indexed geometry choice. The offline dataset bytes/hash and 32:32 offline/online ratio are common.",
            "learner": learner,
        },
        "partitions": partitions,
        "reserved_training_seeds": reserved,
        "reservation_contract": {
            "meaning": "cross-track exclusions include all previous RLPD v1/v2/v3 training, consumed screen/confirmation/blind, unopened reservations, retired entropy-ablation V1/V2/V3 pools, all prior experiments, and the complete V4 evaluation grids",
            "training_sampler": "use only explicit V4 training_geometry_seeds; never draw from reserved_training_seeds",
            "historical_reserved_count": len(history),
            "total_reserved_count": len(reserved),
        },
        "geometry_audit": {
            "teacher_source_ledgers": ledgers,
            "candidate_seed_token_audit": {
                **token_audit,
                "candidate_seed_count": len(candidate_seeds),
                "candidate_seeds_sha256": canonical_sha256(candidate_seeds),
            },
            "freshness_statement": "No known exact token overlap in recorded text/ledger artifacts; the independent audit excluded V1/V2/V3 consumed and reserved sets. Incomplete historical schedules preclude a global non-use claim.",
        },
        "evaluation": {
            "purpose": "internal entropy-target one-factor ablation, with screen selection, paired confirmation comparison, and one blind finalist",
            "screen_candidate_steps": [65536, 131072],
            "cells_per_candidate": 24,
            "repeats": 2,
            "selection_order": ["canonical_finish_rate", "avg_progress", "-avg_lap_time_ms", "-environment_steps"],
            "screen_gate": "both target arms and both seeds require an eligible selected actor with at least one screen finish before confirmation",
            "confirmation_gate": "confirm all four per-arm/seed screen winners on identical fresh cells; each remains eligible and finishes at least once. Declare a target winner only if its per-seed confirmation finishes are no lower than the other target and strictly higher in at least one seed.",
            "blind_finalist": "if one target dominates paired confirmation results, select the blind actor within that target by confirmation finishes/progress/lap/checkpoint/seed and evaluate exactly one model once",
            "stop_rule": "Any screen/confirmation failure, tie/crossed per-seed result, instability/non-finite alpha/loss/gradient stops before blind. A later hypothesis needs another independently audited protocol and fresh cell allocation.",
            "alpha_numeric_audit": "Log alpha/beta, differential entropy, temperature gradient, actor log_std range, losses, and finite status at every interval. Keep the temperature-loss equation identical; no silent arm-specific clamp or retuning.",
            "official_score_claim": False,
        },
        "environment": {
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
            "cpu_evaluation_contract": {"python": "3.11", "torch": "2.1.0+cpu", "numpy": "1.26.0", "gymnasium": "0.29.1", "opencv": "4.8.1.78", "verified_interpreter": "/tmp/kilo/haic-cpu21/bin/python"},
        },
        "source_hashes": _source_hashes(),
        "result_path": "experiments/pixel-rlpd-entropy-target-ablation-v4-result.json",
        "execution": {
            "collector": "python -m scripts.collect_rlpd_prior --protocol experiments/pixel-rlpd-entropy-target-ablation-v4.json --output runs/20260925-pixel-rlpd-entropy-target-ablation-v4/prior-data",
            "trainer_template": "python -m scripts.train_rlpd_entropy_ablation --protocol experiments/pixel-rlpd-entropy-target-ablation-v4.json --offline-dataset runs/20260925-pixel-rlpd-entropy-target-ablation-v4/prior-data --run-dir runs/20260925-pixel-rlpd-entropy-target-ablation-v4/{arm}-seed{40|41} --arm {rlpd-author-target|rlpd-positive-target} --seed {40|41} --device cuda",
            "runner": "python -m scripts.run_rlpd_entropy_ablation --protocol experiments/pixel-rlpd-entropy-target-ablation-v4.json --run-root runs/20260925-pixel-rlpd-entropy-target-ablation-v4 --workers 2",
        },
        "limitations": [
            "The target comparison is limited to two learner RNG seeds and 24-cell screen/confirmation/blind partitions.",
            "The V3 candidate is prior hypothesis evidence only; no V3 dataset, checkpoint, geometry or outcome is reused as V4 data.",
            "`+1.5` is an explicit higher differential-entropy target, not a verified correction to the pinned author's `-1.5` convention.",
            "CarRacing reward/progress/finish metrics are internal proxies, not official competition results.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v4.json")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = build_protocol()
    if args.preflight_only:
        from evaluate_policy import load_protocol_spec
        from scripts.rlpd_entropy_common import read_entropy_protocol

        with tempfile.TemporaryDirectory(prefix="haic-rlpd-entropy-v3-preflight-") as temp:
            temporary_protocol = Path(temp) / "protocol.json"
            write_json(temporary_protocol, protocol, exclusive=True)
            validated = read_entropy_protocol(temporary_protocol, verify_sources=True)
            spec = load_protocol_spec(temporary_protocol)
            if not (
                spec["partitions"]["screen"]["seeds"] == protocol["partitions"]["screen"]["seeds"]
                and validated["student_training"]["steps_per_run"] == 131072
            ):
                raise ValueError("generated entropy protocol didn't round-trip unchanged")
        print(json.dumps({
            "status": "preflight_passed_without_environment_interaction",
            "name": protocol["name"],
            "source_count": len(protocol["source_hashes"]),
            "candidate_seed_count": protocol["geometry_audit"]["candidate_seed_token_audit"]["candidate_seed_count"],
            "partition_cells": {
                name: len(partition["track_ids"]) * len(partition["seeds"])
                for name, partition in protocol["partitions"].items()
            },
            "protocol_output_exists": output.exists(),
        }, sort_keys=True), flush=True)
        return
    if output.exists():
        raise FileExistsError(output)
    write_json(output, protocol, exclusive=True)
    print(json.dumps({
        "protocol": str(output.relative_to(ROOT)),
        "protocol_sha256": sha256_file(output),
        "source_count": len(protocol["source_hashes"]),
        "training_geometry_count": len(protocol["training_geometry_seeds"]),
        "evaluation_geometry_count": sum(len(part["seeds"]) for part in protocol["partitions"].values()),
        "seed_token_match_count": len(protocol["geometry_audit"]["candidate_seed_token_audit"]["known_recorded_uses"]),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
