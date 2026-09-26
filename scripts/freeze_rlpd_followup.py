"""Audit and freeze the long-horizon follow-up to the failed RLPD v2 screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.freeze_rlpd_pilot import (
    SOURCE_PATHS as PILOT_SOURCE_PATHS,
    TEACHER_ACTOR,
    TEACHER_ACTOR_SHA256,
    TEACHER_CHECKPOINT,
    TEACHER_CHECKPOINT_SHA256,
    TEACHER_PROTOCOL_SHA256,
    TEACHER_RUN,
)
from scripts.rlpd_common import (
    ROOT,
    audit_candidate_seed_tokens,
    canonical_sha256,
    geometry_seeds_from_ledger,
    runtime_metadata,
    sha256_file,
    write_json,
)


TRAINING_SEEDS = list(range(4_000_012_001, 4_000_012_065))
SCREEN_SEEDS = list(range(4_000_013_001, 4_000_013_009))
CONFIRMATION_SEEDS = list(range(4_000_013_011, 4_000_013_019))
BLIND_SEEDS = list(range(4_000_013_021, 4_000_013_029))
SOURCE_PATHS = sorted(set(PILOT_SOURCE_PATHS) | {
    "scripts/freeze_rlpd_followup.py",
    "scripts/run_rlpd_followup.py",
})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/pixel-rlpd-long-horizon-followup-v1.json",
    )
    return parser.parse_args()


def _source_ledgers():
    ledgers = []
    for seed in (0, 1):
        path = ROOT / f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}/episodes.jsonl"
        observed = geometry_seeds_from_ledger(path)
        ledgers.append({
            "source_training_seed": seed,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "reset_geometry_seed_count": len(observed),
            "geometry_seed_set_sha256": canonical_sha256(sorted(observed)),
            "geometry_seeds": sorted(observed),
        })
    return ledgers


def _previous_reservations(ledgers):
    reserved = set()
    for path in (ROOT / "experiments").glob("*.json"):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        values = artifact.get("reserved_training_seeds", [])
        if isinstance(values, list):
            reserved.update(value for value in values if type(value) is int)
        for partition in artifact.get("partitions", {}).values():
            if isinstance(partition, dict) and isinstance(partition.get("seeds"), list):
                reserved.update(value for value in partition["seeds"] if type(value) is int)
        if artifact.get("format") == "haic-pixel-rlpd-study-v1":
            reserved.update(
                value for value in artifact.get("training_geometry_seeds", [])
                if type(value) is int
            )
            reserved.update(
                cell.get("geometry_seed")
                for cell in artifact.get("teacher_data_cells", [])
                if type(cell.get("geometry_seed")) is int
            )
            full = artifact.get("future_full_reservation", {})
            reserved.update(
                value for value in full.get("training_geometry_seeds", [])
                if type(value) is int
            )
            reserved.update(
                cell.get("geometry_seed")
                for cell in full.get("teacher_data_cells", [])
                if type(cell.get("geometry_seed")) is int
            )
            for partition in full.values():
                if isinstance(partition, dict) and isinstance(partition.get("seeds"), list):
                    reserved.update(value for value in partition["seeds"] if type(value) is int)
    for ledger in ledgers:
        reserved.update(ledger["geometry_seeds"])
    return sorted(reserved)


def _cells(seeds):
    return [
        {"track_id": 1 + index % 4, "geometry_seed": seed, "obstacles": True}
        for index, seed in enumerate(seeds)
    ]


def _source_hashes():
    hashes = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source missing at freeze: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def build_protocol():
    v2_protocol_path = ROOT / "experiments/pixel-rlpd-offpolicy-pilot-v2.json"
    v2_result_path = ROOT / "experiments/pixel-rlpd-offpolicy-pilot-v2-result.json"
    v2_protocol = json.loads(v2_protocol_path.read_text(encoding="utf-8"))
    v2_result = json.loads(v2_result_path.read_text(encoding="utf-8"))
    if v2_result.get("status") != "stop_hold_failure" or v2_result.get("screen", {}).get("gate") != "fail_stop_hold":
        raise ValueError("follow-up requires the completed, failed v2 screen gate")
    if v2_result.get("confirmation_opened") or v2_result.get("blind_opened"):
        raise ValueError("cannot create follow-up if v2 confirmation or blind was opened")

    teacher_actor_path = ROOT / TEACHER_ACTOR
    teacher_checkpoint_path = ROOT / TEACHER_CHECKPOINT
    teacher_protocol_path = ROOT / (TEACHER_RUN / "protocol.json")
    teacher_config_path = ROOT / (TEACHER_RUN / "config.json")
    if (
        sha256_file(teacher_actor_path) != TEACHER_ACTOR_SHA256
        or sha256_file(teacher_checkpoint_path) != TEACHER_CHECKPOINT_SHA256
        or sha256_file(teacher_protocol_path) != TEACHER_PROTOCOL_SHA256
    ):
        raise ValueError("frozen teacher artifacts differ from the approved pad-4 identity")
    teacher_config = json.loads(teacher_config_path.read_text(encoding="utf-8"))
    run_config = teacher_config.get("config", {})
    drq_config = run_config.get("drq_config", {})
    if (
        run_config.get("algorithm") != "drq-v2"
        or run_config.get("seed") != 1
        or run_config.get("total_steps") != 131072
        or drq_config.get("augmentation_pad") != 4
    ):
        raise ValueError("teacher source config is not the frozen pad-4 seed-1 control")

    screen_seeds = SCREEN_SEEDS
    confirmation_seeds = CONFIRMATION_SEEDS
    blind_seeds = BLIND_SEEDS
    candidate_seeds = sorted(set(TRAINING_SEEDS + screen_seeds + confirmation_seeds + blind_seeds))
    token_audit = audit_candidate_seed_tokens(candidate_seeds)
    if token_audit["known_recorded_uses"]:
        raise ValueError(f"follow-up cell has a recorded token collision: {token_audit['known_recorded_uses'][:10]}")
    ledgers = _source_ledgers()
    history = _previous_reservations(ledgers)
    evaluation_seeds = screen_seeds + confirmation_seeds + blind_seeds
    reserved = sorted(set(history + evaluation_seeds))
    if set(TRAINING_SEEDS).intersection(reserved):
        raise ValueError("new teacher/student training pool overlaps consumed or reserved geometry")

    partitions = {
        "screen": {"track_ids": [201, 202, 203], "seeds": screen_seeds, "repeats": 2},
        "confirmation": {
            "track_ids": [211, 212, 213, 214],
            "seeds": confirmation_seeds,
            "repeats": 2,
        },
        "blind": {"track_ids": [221, 222, 223], "seeds": blind_seeds, "repeats": 2},
    }
    return {
        "format": "haic-pixel-rlpd-study-v1",
        "schema_version": 1,
        "name": "pixel-rlpd-long-horizon-followup-v1",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "independent extended-budget follow-up after the v2 pilot stop gate; no v2 data, policy, or evaluation cell is reused",
        "hypothesis": "The v2 short student horizon and small prior dataset may have limited stable new-road completion. A larger teacher prior and matched long-horizon student runs are a fresh test of the combined recipe, not a causal diagnosis.",
        "comparison": "RLPD with a new 50:50 frozen prior dataset versus matched online-only pixel SAC; same ten-Q learner and 131,072 student decisions; not standard two-Q SAC/DrQ superiority.",
        "prior_study": {
            "protocol_path": str(v2_protocol_path.relative_to(ROOT)),
            "protocol_sha256": sha256_file(v2_protocol_path),
            "result_path": str(v2_result_path.relative_to(ROOT)),
            "result_sha256": sha256_file(v2_result_path),
            "gate": "fail_stop_hold",
            "consumed_data_and_cells_excluded": True,
            "confirmation_or_blind_opened": False,
            "pilot_full_reservation_reused": False,
        },
        "source": {
            "rlpd_repository": "https://github.com/ikostrikov/rlpd",
            "rlpd_commit": "c90fd4baf28c9c9ef40a81460a2e395092844f88",
            "port": "same independent pixel implementation as v2; no model weights or transitions resumed",
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
            "reuse_scope": "frozen action generator only; collect new trajectories on new training geometry; no old teacher/student trajectory, V2 prior transition, critic, replay, optimizer or checkpoint is imported",
        },
        "frame_skip": 4,
        "max_steps": 2000,
        "training_track_ids": [1, 2, 3, 4],
        "training_geometry_seeds": TRAINING_SEEDS,
        "teacher_data_cells": _cells(TRAINING_SEEDS),
        "teacher_data_budget": {
            "decisions": 16384,
            "minimum_distinct_finishes": 4,
            "partial_episode": "drop without fabricating terminal; count all decisions spent",
            "source_repeats": "one attempt per frozen training geometry cell",
        },
        "student_training": {
            "arms": ["rlpd", "sac-online-only-control"],
            "learner_seeds": [10, 11],
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
            "training_sampler": "default_rng(seed+0x5A11), uniform over the frozen fresh training track/geometry pool; matched RLPD/SAC arms share the indexed stream",
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
            "temperature_fidelity_warning": "Keep the pinned author target_entropy=-1.5/loss convention unchanged here. It is not assumed to be an algebraically corrected positive-entropy target; a sign convention ablation requires its own one-factor protocol.",
        },
        "partitions": partitions,
        "reserved_training_seeds": reserved,
        "reservation_contract": {
            "meaning": "cross-track uint32 exclusions include all previously documented reservations, all v2 training/data/screen/held-out/full-reservation seeds, and every v1 screen/held-out allocation; plus all V3 screen/confirmation/blind seeds",
            "training_sampler": "sample only this study's explicit training_geometry_seeds; never sample the exclusion list",
            "historical_reserved_count": len(history),
            "total_reserved_count": len(reserved),
            "v2_train_and_collection_seeds_added": v2_protocol["training_geometry_seeds"],
            "v2_future_training_reservation_added": v2_protocol["future_full_reservation"]["training_geometry_seeds"],
        },
        "geometry_audit": {
            "teacher_source_ledgers": ledgers,
            "candidate_seed_token_audit": {
                **token_audit,
                "candidate_seed_count": len(candidate_seeds),
                "candidate_seeds_sha256": canonical_sha256(candidate_seeds),
            },
            "freshness_statement": "No known exact token use in searchable text or either original DrQ training ledger. V1/V2 allocations, including unopened V2 reservations, were explicitly excluded. This is not a global historical non-use proof.",
        },
        "evaluation": {
            "purpose": "internal extended-recipe study; screen first, matched confirmation only after screen gate, one screen-preselected treatment actor for blind only after confirmation gate",
            "screen_candidate_steps": [65536, 131072],
            "cells_per_candidate": 24,
            "repeats": 2,
            "selection_order": ["canonical_finish_rate", "avg_progress", "-avg_lap_time_ms", "-environment_steps"],
            "screen_gate": "both RLPD learner seeds must have an eligible actor with at least one canonical finish, and each RLPD seed's finishes must be no lower than its matched SAC seed",
            "confirmation_gate": "after screen pass, the preselected per-seed RLPD and matched SAC screen actors use identical fresh confirmation cells; each RLPD must finish at least once and strictly exceed its SAC control; a zero-screen SAC uses diagnostic-only confirmation",
            "blind_finalist": "before confirmation, select one RLPD actor from the per-seed screen finalists using screen finish/progress/lap/checkpoint ordering; lower learner seed breaks a full tie; only this frozen actor may enter blind after every confirmation gate passes",
            "stop_rule": "Any failed screen or confirmation gate stops this study at that stage; do not open later partitions, tune on their cells, or extend budgets. A further iteration requires another source-hashed protocol and fresh cells.",
            "official_score_claim": False,
        },
        "environment": {
            "track_ids": [1, 2, 3, 4],
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
            },
        },
        "source_hashes": _source_hashes(),
        "result_path": "experiments/pixel-rlpd-long-horizon-followup-v1-result.json",
        "execution": {
            "collector": "python -m scripts.collect_rlpd_prior --protocol experiments/pixel-rlpd-long-horizon-followup-v1.json --output runs/20260924-pixel-rlpd-long-horizon-followup-v1/prior-data",
            "trainer_template": "python -m scripts.train_rlpd --protocol experiments/pixel-rlpd-long-horizon-followup-v1.json --offline-dataset runs/20260924-pixel-rlpd-long-horizon-followup-v1/prior-data --run-dir runs/20260924-pixel-rlpd-long-horizon-followup-v1/{arm}-seed{seed} --arm {rlpd|sac} --seed {10|11} --device cuda",
            "runner": "python -m scripts.run_rlpd_followup --protocol experiments/pixel-rlpd-long-horizon-followup-v1.json --run-root runs/20260924-pixel-rlpd-long-horizon-followup-v1 --workers 2",
        },
        "limitations": [
            "The V2 v2-to-follow-up horizon/data expansion is a combined recipe change; no single mechanism is isolated.",
            "Only two fresh student RNG seeds are planned; screen repeats do not create more independent geometry.",
            "The same pre-existing actor hash is used only as an action generator; data are newly collected on fresh geometry.",
            "All rewards/progress/finishes are internal CarRacing proxies, not official HAIC scores.",
        ],
    }


def _source_hashes():
    hashes = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source missing at follow-up freeze: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


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
        "training_seed_count": len(protocol["training_geometry_seeds"]),
        "evaluation_seed_count": sum(len(partition["seeds"]) for partition in protocol["partitions"].values()),
        "reserved_exclusion_count": len(protocol["reserved_training_seeds"]),
        "candidate_seed_audit_count": protocol["geometry_audit"]["candidate_seed_token_audit"]["candidate_seed_count"],
        "recorded_collisions": protocol["geometry_audit"]["candidate_seed_token_audit"]["known_recorded_uses"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
