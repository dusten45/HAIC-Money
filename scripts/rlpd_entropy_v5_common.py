"""Protocol validator for the zero-reuse V5 entropy-target ablation."""

from __future__ import annotations

from pathlib import Path

from scripts.rlpd_common import read_protocol
from scripts.rlpd_entropy_common import EXPECTED_ARM_SPECS


STUDY_NAME = "pixel-rlpd-entropy-target-ablation-v5"


def read_entropy_v5_protocol(path: str | Path, *, verify_sources: bool = True) -> dict:
    protocol = read_protocol(path, verify_sources=verify_sources)
    if protocol.get("name") != STUDY_NAME:
        raise ValueError("entropy V5 protocol name mismatch")
    student = protocol["student_training"]
    if (
        student.get("arms") != list(EXPECTED_ARM_SPECS)
        or student.get("arm_specs") != EXPECTED_ARM_SPECS
        or student.get("learner_seeds") != [50, 51]
        or student.get("steps_per_run") != 131072
        or student.get("candidate_steps") != [65536, 131072]
        or student.get("expected_gradient_steps_per_run") != 130072
        or student.get("offline_capacity") != 16384
        or protocol.get("teacher_data_budget", {}).get("decisions") != 16384
        or protocol.get("teacher_data_budget", {}).get("minimum_distinct_finishes") != 4
    ):
        raise ValueError("V5 fresh-data, matched-target, student-budget contract mismatch")
    return protocol
