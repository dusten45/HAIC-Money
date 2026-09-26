"""Audit and freeze the fresh V5 one-factor RLPD entropy-target ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

from evaluate_policy import load_protocol_spec
from scripts.freeze_rlpd_followup import SOURCE_PATHS as FOLLOWUP_SOURCE_PATHS, _previous_reservations, _source_ledgers
from scripts.rlpd_common import (
    ROOT,
    audit_candidate_seed_tokens,
    canonical_sha256,
    runtime_metadata,
    sha256_file,
    write_json,
)
from scripts.rlpd_entropy_common import EXPECTED_ARM_SPECS
from scripts.rlpd_entropy_v5_common import read_entropy_v5_protocol


TRAINING_SEEDS = list(range(4_000_032_001, 4_000_032_065))
SCREEN_SEEDS = list(range(4_000_033_001, 4_000_033_009))
CONFIRMATION_SEEDS = list(range(4_000_033_011, 4_000_033_019))
BLIND_SEEDS = list(range(4_000_033_021, 4_000_033_029))
SOURCE_PATHS = sorted(set(FOLLOWUP_SOURCE_PATHS) | {
    "scripts/freeze_rlpd_entropy_ablation.py",
    "scripts/freeze_rlpd_entropy_ablation_v5.py",
    "scripts/project_rlpd_entropy_screen_receipts.py",
    "scripts/rlpd_entropy_common.py",
    "scripts/rlpd_entropy_v5_common.py",
    "scripts/run_rlpd_entropy_ablation.py",
    "scripts/run_rlpd_entropy_v5.py",
    "scripts/train_rlpd_entropy_ablation.py",
    "scripts/train_rlpd_entropy_v5.py",
    "tests/test_rlpd_entropy_ablation.py",
    "tests/test_rlpd_entropy_v5.py",
})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v5.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _teacher_cells(seeds):
    return [
        {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
        for index, seed in enumerate(seeds)
    ]


def _source_hashes():
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V5 source is missing before protocol freeze: {relative}")
        result[relative] = sha256_file(path)
    return result


def build_protocol():
    v3_path = ROOT / "experiments/pixel-rlpd-long-horizon-followup-v1.json"
    v3_result_path = ROOT / "experiments/pixel-rlpd-long-horizon-followup-v1-result.json"
    v4_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v4.json"
    v4_result_path = ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v4-result.json"
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v3_result = json.loads(v3_result_path.read_text(encoding="utf-8"))
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v4_result = json.loads(v4_result_path.read_text(encoding="utf-8"))
    if (
        v3_result.get("blind_opened") is not True
        or v3_result.get("confirmation_gate") != "pass"
        or v4_result.get("status") != "aborted_before_screen_source_hash_gate"
        or v4_result.get("evaluation", {}).get("screen_cells_started") != 0
        or v4_result.get("evaluation", {}).get("confirmation_cells_started") != 0
        or v4_result.get("evaluation", {}).get("blind_cells_started") != 0
    ):
        raise ValueError("V5 requires completed V3 and the audited zero-screen V4 stop")
    candidate_seeds = sorted(set(
        TRAINING_SEEDS + SCREEN_SEEDS + CONFIRMATION_SEEDS + BLIND_SEEDS
    ))
    if len(candidate_seeds) != 88:
        raise ValueError("V5 geometry must contain 88 unique seed values")
    audit = audit_candidate_seed_tokens(candidate_seeds)
    if audit["known_recorded_uses"]:
        raise ValueError(f"V5 has an exact recorded seed-token collision: {audit['known_recorded_uses'][:10]}")
    ledgers = _source_ledgers()
    history = _previous_reservations(ledgers)
    evaluation_seeds = SCREEN_SEEDS + CONFIRMATION_SEEDS + BLIND_SEEDS
    reserved = sorted(set(history + evaluation_seeds))
    if set(TRAINING_SEEDS).intersection(reserved):
        raise ValueError("V5 training pool overlaps consumed/reserved geometry")
    if set(candidate_seeds).intersection(v3["training_geometry_seeds"]):
        raise ValueError("V5 reuses V3 long-horizon training geometry")
    if set(candidate_seeds).intersection(
        seed for partition in v3["partitions"].values() for seed in partition["seeds"]
    ):
        raise ValueError("V5 reuses V3 screen/confirmation/blind geometry")
    if set(candidate_seeds).intersection(v4["training_geometry_seeds"]):
        raise ValueError("V5 reuses V4 consumed teacher/student geometry")
    if set(candidate_seeds).intersection(
        seed for partition in v4["partitions"].values() for seed in partition["seeds"]
    ):
        raise ValueError("V5 reuses retired V4 evaluation geometry")

    source_hashes = _source_hashes()
    v4_prior_record = {
        "status": v4_result["status"],
        "teacher_dataset_sha256": v4_result["teacher_data"]["sha256"],
        "teacher_data_consumed": True,
        "four_student_runs_consumed": True,
        "evaluation_cells_executed": 0,
        "source_mismatch": v4_result["failure"],
    }
    protocol = {
        "format": "haic-pixel-rlpd-study-v1",
        "schema_version": 1,
        "name": "pixel-rlpd-entropy-target-ablation-v5",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "fresh one-factor entropy-target study after V4's consumed training/source-drift stop; no prior data, weights, or evaluation cell is reused",
        "hypothesis": "At identical fresh data, initialization, offline/online ratio, pixel networks, optimizer and student budget, the pinned target -1.5 and the explicit +1.5 alternative produce different alpha/entropy trajectories and held-out completion.",
        "comparison": "RLPD with author target_entropy=-1.5 versus the same RLPD implementation with target_entropy=+1.5; only the target scalar changes.",
        "prior_studies": {
            "v3_protocol": str(v3_path.relative_to(ROOT)),
            "v3_protocol_sha256": sha256_file(v3_path),
            "v3_result": str(v3_result_path.relative_to(ROOT)),
            "v3_result_sha256": sha256_file(v3_result_path),
            "v3_blind_actor_sha256": v3_result["blind_finalist"]["actor_sha256"],
            "v3_prior_dataset_sha256": v3_result["prior_dataset_sha256"],
            "v4_protocol": str(v4_path.relative_to(ROOT)),
            "v4_protocol_sha256": sha256_file(v4_path),
            "v4_result": str(v4_result_path.relative_to(ROOT)),
            "v4_result_sha256": sha256_file(v4_result_path),
            "v4_preflight_status": v4_prior_record,
            "v4_training_data_and_checkpoints_reused": False,
        },
        "teacher": {
            "algorithm": "DrQ-v2 pad-4 control, original training seed 1",
            "actor_path": v4["teacher"]["actor_path"],
            "actor_sha256": v4["teacher"]["actor_sha256"],
            "checkpoint_path": v4["teacher"]["checkpoint_path"],
            "checkpoint_sha256": v4["teacher"]["checkpoint_sha256"],
            "source_protocol_path": v4["teacher"]["source_protocol_path"],
            "source_protocol_sha256": v4["teacher"]["source_protocol_sha256"],
            "source_config_path": v4["teacher"]["source_config_path"],
            "source_config_sha256": v4["teacher"]["source_config_sha256"],
            "source_training_seed": 1,
            "source_checkpoint_environment_decisions": 131072,
            "reuse_scope": "frozen action generator only; all V5 prior trajectories are collected from scratch on the new teacher-only pool",
        },
        "frame_skip": 4,
        "max_steps": 2000,
        "training_track_ids": [1, 2, 3, 4],
        "training_geometry_seeds": TRAINING_SEEDS,
        "teacher_data_cells": [
            {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
            for index, seed in enumerate(TRAINING_SEEDS)
        ],
        "teacher_data_budget": {
            "decisions": 16384,
            "minimum_distinct_finishes": 4,
            "partial_episode": "drop without synthetic terminal and count decisions",
            "source_repeats": "one attempt per frozen fresh geometry cell",
        },
        "student_training": {
            "arms": list(EXPECTED_ARM_SPECS),
            "arm_specs": EXPECTED_ARM_SPECS,
            "learner_seeds": [50, 51],
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
            "training_sampler": "For each RNG seed both entropy variants start from identical weights, Adam state, Torch/NumPy RNG and the same indexed geometry pool; both use the same immutable offline dataset bytes and 32:32 batch split.",
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
        },
        "partitions": {
            "screen": {"track_ids": [351, 352, 353], "seeds": SCREEN_SEEDS, "repeats": 2},
            "confirmation": {"track_ids": [361, 362, 363, 364], "seeds": CONFIRMATION_SEEDS, "repeats": 2},
            "blind": {"track_ids": [371, 372, 373], "seeds": BLIND_SEEDS, "repeats": 2},
        },
        "reserved_training_seeds": reserved,
        "reservation_contract": {
            "meaning": "cross-track uint32 exclusions include all v1-v4 RLPD prior training/data, consumed screen/confirmation/blind cells and unopened reserves plus V5 evaluation partitions",
            "training_sampler": "train only on this V5 training pool; never sample reserved_training_seeds",
            "historical_reserved_count": len(history),
            "total_reserved_count": len(reserved),
        },
        "geometry_audit": {
            "teacher_source_ledgers": _source_ledgers(),
            "candidate_seed_token_audit": {
                **audit,
                "candidate_seed_count": len(candidate_seeds),
                "candidate_seeds_sha256": canonical_sha256(candidate_seeds),
            },
            "freshness_statement": "No known exact seed-token overlap in searchable prior artifacts/source-ledger seeds; all v1-v4 allocations, including aborts, are excluded. Historical use is incomplete and cannot be globally disproven.",
        },
        "evaluation": {
            "purpose": "fresh internal one-factor entropy-target ablation",
            "screen_candidate_steps": [65536, 131072],
            "cells_per_candidate": 24,
            "repeats": 2,
            "selection_order": ["canonical_finish_rate", "avg_progress", "-avg_lap_time_ms", "-environment_steps"],
            "screen_gate": "both entropy targets and both learner seeds require an eligible selected actor with at least one canonical screen finish",
            "confirmation_gate": "confirm all four per-target/per-seed screen selections on identical fresh cells; all must be operationally eligible, deterministic, reload-identical and finish at least once",
            "target_winner_gate": "one target must be no lower in canonical confirmation finishes for both matched seeds and strictly higher for at least one seed; crossed/tied results stop without blind",
            "blind_finalist": "after confirmation target dominance, choose a screen-selected actor within the winning target using confirmation finish/progress/lap/checkpoint/seed order; evaluate exactly one actor once",
            "stop_rule": "Non-finite alpha/loss/gradient, failed screen eligibility, missing confirmation finishes, or no per-seed target dominance stops the study before any later partition. A further hypothesis requires a new protocol and new cells.",
            "numeric_observations": "Record alpha/beta, entropy=-log_pi, temperature gradient, log_std, losses and finite state; preserve the same loss equation and no target-specific clamp.",
            "official_score_claim": False,
        },
        "environment": v4["environment"],
        "runtime": v4["runtime"],
        "source_hashes": _source_hashes(),
        "result_path": "experiments/pixel-rlpd-entropy-target-ablation-v5-result.json",
        "execution": {
            "collector": "python -m scripts.collect_rlpd_prior --protocol experiments/pixel-rlpd-entropy-target-ablation-v5.json --output runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data",
            "trainer_template": "python -m scripts.train_rlpd_entropy_v5 --protocol experiments/pixel-rlpd-entropy-target-ablation-v5.json --offline-dataset runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data --run-dir runs/20260925-pixel-rlpd-entropy-target-ablation-v5/{arm}-seed{50|51} --arm {rlpd-author-target|rlpd-positive-target} --seed {50|51} --device cuda",
            "runner": "python -m scripts.run_rlpd_entropy_v5 --protocol experiments/pixel-rlpd-entropy-target-ablation-v5.json --run-root runs/20260925-pixel-rlpd-entropy-target-ablation-v5 --workers 2",
        },
        "limitations": [
            "Only two RNG seeds are used per entropy treatment.",
            "+1.5 is a high differential-entropy target, not a verified correction to the author implementation.",
            "V1-v4 data, actor weights, checkpoints, source/training episodes, and screen/confirmation/blind cells are excluded.",
            "All CarRacing finish/progress/reward metrics remain internal proxies rather than official HAIC results.",
        ],
    }
    return protocol


def _source_hashes():
    result = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V5 source missing before freeze: {relative}")
        result[relative] = sha256_file(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/pixel-rlpd-entropy-target-ablation-v5.json")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = build_protocol()
    if args.preflight_only:
        from evaluate_policy import load_protocol_spec
        from scripts.rlpd_entropy_v5_common import read_entropy_v5_protocol
        with tempfile.TemporaryDirectory(prefix="haic-rlpd-entropy-v5-preflight-") as temp:
            temporary = Path(temp) / "protocol.json"
            write_json(temporary, protocol, exclusive=True)
            checked = read_entropy_v5_protocol(temporary, verify_sources=True)
            spec = load_protocol_spec(temporary)
            if checked["student_training"]["learner_seeds"] != [50, 51] or len(spec["partitions"]["screen"]["seeds"]) != 8:
                raise ValueError("generated V5 protocol did not round-trip its declared arm/cell schema")
        print(json.dumps({
            "status": "preflight_passed_zero_environment_interactions",
            "name": protocol["name"],
            "source_count": len(protocol["source_hashes"]),
            "candidate_seed_count": protocol["geometry_audit"]["candidate_seed_token_audit"]["candidate_seed_count"],
            "partition_cells": {key: len(value["track_ids"]) * len(value["seeds"]) for key, value in protocol["partitions"].items()},
            "existing_output": output.exists(),
        }, sort_keys=True), flush=True)
        return
    if output.exists():
        raise FileExistsError(output)
    write_json(output, protocol, exclusive=True)
    print(json.dumps({
        "protocol": str(output.relative_to(ROOT)),
        "protocol_sha256": sha256_file(output),
        "source_count": len(protocol["source_hashes"]),
        "teacher_seed_count": len(protocol["training_geometry_seeds"]),
        "evaluation_seed_count": sum(len(value["seeds"]) for value in protocol["partitions"].values()),
        "reserved_exclusion_count": len(protocol["reserved_training_seeds"]),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
