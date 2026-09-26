"""Validation helpers for the isolated one-factor pixel-RLPD entropy ablation."""

from __future__ import annotations

from pathlib import Path

from scripts.rlpd_common import read_protocol


STUDY_NAME = "pixel-rlpd-entropy-target-ablation-v4"
EXPECTED_ARM_SPECS = {
    "rlpd-author-target": {"algorithm": "rlpd", "use_offline": True, "target_entropy": -1.5},
    "rlpd-positive-target": {"algorithm": "rlpd", "use_offline": True, "target_entropy": 1.5},
}


def read_entropy_protocol(path: str | Path, *, verify_sources: bool = True) -> dict:
    protocol = read_protocol(path, verify_sources=verify_sources)
    if protocol.get("name") != STUDY_NAME:
        raise ValueError("unsupported entropy-ablation protocol name")
    student = protocol["student_training"]
    if student.get("arms") != list(EXPECTED_ARM_SPECS):
        raise ValueError("entropy ablation requires the two predeclared target arms in order")
    if student.get("arm_specs") != EXPECTED_ARM_SPECS:
        raise ValueError("ablation arms must differ only in the frozen target-entropy value")
    if (
        student.get("learner_seeds") != [40, 41]
        or student.get("steps_per_run") != 131072
        or student.get("candidate_steps") != [65536, 131072]
        or student.get("expected_gradient_steps_per_run") != 130072
        or student.get("first_update_step") != 1000
        or student.get("policy_takeover_step") != 2000
        or student.get("batch_size") != 64
        or student.get("offline_batch_size") != 32
        or student.get("online_batch_size") != 32
        or student.get("backup_entropy") is not False
        or student.get("offline_capacity") != 16384
    ):
        raise ValueError("entropy ablation learner budget/mixing differs from its frozen design")
    data_budget = protocol["teacher_data_budget"]
    if (
        data_budget.get("decisions") != 16384
        or data_budget.get("minimum_distinct_finishes") != 4
        or len(protocol["training_geometry_seeds"]) != 64
        or protocol["student_training"]["learner"]["target_entropy"] != -1.5
    ):
        raise ValueError("fresh-data/target baseline contract is invalid")
    expected_cells = {"screen": (3, 8), "confirmation": (4, 8), "blind": (3, 8)}
    for partition, (track_count, seed_count) in expected_cells.items():
        matrix = protocol["partitions"][partition]
        if (
            len(matrix["track_ids"]) != track_count
            or len(matrix["seeds"]) != seed_count
            or matrix["repeats"] != 2
        ):
            raise ValueError(f"{partition} grid does not match its fresh preregistered ablation matrix")
    return protocol
