"""Run a matched PPO comparison of linear versus squared obstacle urgency."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

from research.hazard_urgency import rescale_quadratic_hazard_risk


FROZEN_SOURCE_ROOT = Path(r"C:\Users\koi\.codex\worktrees\paired-speed-screen\HAIC")
SPLIT_MANIFEST = FROZEN_SOURCE_ROOT / "training" / "maps" / "site" / "site_map_split.json"
EXPERIMENT_NAME = "ppo-hazard-urgency-square-vs-linear-recovery-gate0p5-kl0p5-u8-v1"
ARM_EXPONENTS = {"urgency_squared": 2.0, "urgency_linear": 1.0}
RECOVERY_GATE = {"urgency_squared": 0.5, "urgency_linear": 0.5}
REFERENCE_KL = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _now_stamp() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).strftime("%Y%m%dT%H%M%SKST")


def configure_frozen_runner() -> tuple[Any, Any, Any, Any]:
    if not FROZEN_SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"frozen PPO source is missing: {FROZEN_SOURCE_ROOT}")
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(f"frozen map split is missing: {SPLIT_MANIFEST}")
    # Import the prior experiment stack only after the frozen source leads sys.path.
    sys.path.insert(0, str(FROZEN_SOURCE_ROOT))
    from training import evaluate_bc_reference_kl as evaluator
    from training import paired_bc_reference_kl as experiment
    from training import paired_recovery_gate as recovery_gate
    from training import paired_speed_target as paired
    from training import train_policy

    recovery_gate.EXPERIMENT_NAME = EXPERIMENT_NAME
    recovery_gate.THRESHOLDS = dict(RECOVERY_GATE)
    recovery_gate.configure()

    experiment.ARMS = {arm: REFERENCE_KL for arm in ARM_EXPONENTS}
    experiment.EXPERIMENT_NAME = EXPERIMENT_NAME
    protocol = dict(experiment.PROTOCOL)
    protocol.update(
        {
            "experiment": EXPERIMENT_NAME,
            "paired_variable": "visible_hazard_urgency_exponent",
            "hazard_urgency_exponent_by_arm": dict(ARM_EXPONENTS),
            "recovery_max_hazard_risk_by_arm": dict(RECOVERY_GATE),
            "reference_kl_arms": dict(experiment.ARMS),
            "reference_kl_coefficient": REFERENCE_KL,
            "selection": "fixed_update_8_no_tune_selection",
            "experiment_runner_path": str(Path(__file__).resolve()),
            "experiment_runner_sha256": _sha256(Path(__file__).resolve()),
            "risk_transform_helper_path": str(
                (Path(__file__).resolve().parent / "hazard_urgency.py").resolve()
            ),
            "risk_transform_helper_sha256": _sha256(
                Path(__file__).resolve().parent / "hazard_urgency.py"
            ),
            "tune_geometry_note": "Two TUNE seeds repeat one geometry; paired outcomes are reported by actor seed and map seed.",
        }
    )
    experiment.PROTOCOL = protocol

    base_risk_function = train_policy.visible_hazard_risk
    original_run_one_target = paired._run_one_target

    def run_one_target_with_urgency(
        *args: Any,
        arm_label: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if arm_label not in ARM_EXPONENTS:
            raise ValueError(f"unknown urgency arm: {arm_label}")
        exponent = ARM_EXPONENTS[arm_label]

        def adjusted_risk(features: np.ndarray) -> float:
            values = np.asarray(features, dtype=np.float32)
            if values.shape != (7,):
                raise ValueError("visual feature vector must have width seven")
            quadratic = base_risk_function(values)
            return rescale_quadratic_hazard_risk(
                quadratic,
                float(np.clip(values[6], 0.0, 1.0)),
                exponent,
            )

        train_policy.visible_hazard_risk = adjusted_risk
        try:
            result = original_run_one_target(*args, arm_label=arm_label, **kwargs)
        finally:
            train_policy.visible_hazard_risk = base_risk_function

        manifest_path = Path(result["run_directory"]) / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["paired_variable"] = "visible_hazard_urgency_exponent"
        manifest["hazard_urgency_exponent"] = exponent
        manifest["recovery_max_hazard_risk"] = RECOVERY_GATE[arm_label]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    paired._run_one_target = run_one_target_with_urgency

    def validate_manifests(experiment_directory: Path) -> None:
        status = json.loads(
            (experiment_directory / "training-status.json").read_text(encoding="utf-8")
        )
        if status.get("status") != "completed" or status.get("completed_runs") != 4:
            raise RuntimeError("training is incomplete; refusing to load TUNE geometry")
        for seed in experiment.base.PAIRED_SEEDS:
            for arm, exponent in ARM_EXPONENTS.items():
                path = experiment_directory / f"seed{seed}" / arm / "run-manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("hazard_urgency_exponent") != exponent:
                    raise RuntimeError(f"urgency exponent mismatch for seed {seed}, arm {arm}")
                if manifest.get("recovery_max_hazard_risk") != RECOVERY_GATE[arm]:
                    raise RuntimeError(f"recovery gate mismatch for seed {seed}, arm {arm}")
                if manifest.get("paired_variable") != "visible_hazard_urgency_exponent":
                    raise RuntimeError(f"paired variable mismatch for seed {seed}, arm {arm}")
                if manifest.get("reference_kl_coefficient") != REFERENCE_KL:
                    raise RuntimeError(f"reference KL changed for seed {seed}, arm {arm}")

    recovery_gate._validate_recovery_manifests = validate_manifests

    def compare_urgency_arms(
        episodes_by_arm: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        control_arm = "urgency_squared"
        treatment_arm = "urgency_linear"
        control = {
            str((row["training_seed"], row["episode"]["map_id"], row["episode"]["seed"])): row
            for row in episodes_by_arm[control_arm]
        }
        treatment = {
            str((row["training_seed"], row["episode"]["map_id"], row["episode"]["seed"])): row
            for row in episodes_by_arm[treatment_arm]
        }
        common = sorted(set(control) & set(treatment))
        matched = [
            {"cell": key, control_arm: control[key], treatment_arm: treatment[key]}
            for key in common
        ]
        control_summary = evaluator.speed_eval._summarize_episode_metrics(
            episodes_by_arm[control_arm]
        )
        treatment_summary = evaluator.speed_eval._summarize_episode_metrics(
            episodes_by_arm[treatment_arm]
        )
        safety = {
            "finish_count_not_lower": treatment_summary["finish_count"]
            >= control_summary["finish_count"],
            "collision_decisions_not_worse_per_cell": all(
                treatment[key]["collision_decisions"] <= control[key]["collision_decisions"]
                for key in common
            ),
            "final_damage_not_worse_per_cell": all(
                treatment[key]["final_damage"] <= control[key]["final_damage"] + 1e-9
                for key in common
            ),
            "max_damage_not_worse_per_cell": all(
                treatment[key]["max_damage"] <= control[key]["max_damage"] + 1e-9
                for key in common
            ),
        }
        faster_finished_lap = (
            treatment_summary["median_finished_lap_time_s"] is not None
            and (
                control_summary["median_finished_lap_time_s"] is None
                or treatment_summary["median_finished_lap_time_s"]
                < control_summary["median_finished_lap_time_s"]
            )
        )
        dnf_progress_gain = (
            treatment_summary["mean_dnf_progress"] is not None
            and control_summary["mean_dnf_progress"] is not None
            and treatment_summary["mean_dnf_progress"]
            > control_summary["mean_dnf_progress"]
        )
        gain = (
            treatment_summary["finish_count"] > control_summary["finish_count"]
            or treatment_summary["valid_under_13_count"]
            > control_summary["valid_under_13_count"]
            or faster_finished_lap
            or dnf_progress_gain
        )
        all_matched = len(common) == len(control) == len(treatment)
        supported = all(safety.values()) and gain and all_matched
        return {
            "decision": "supports_linear_urgency" if supported else (
                "rejects_on_safety" if not all(safety.values()) else "inconclusive_no_robust_gain"
            ),
            "control_arm": control_arm,
            "treatment_arm": treatment_arm,
            "control_urgency_exponent": ARM_EXPONENTS[control_arm],
            "treatment_urgency_exponent": ARM_EXPONENTS[treatment_arm],
            "control_recovery_max_hazard_risk": RECOVERY_GATE[control_arm],
            "treatment_recovery_max_hazard_risk": RECOVERY_GATE[treatment_arm],
            "safety_gates": safety,
            "performance_gain": gain,
            "all_cells_matched": all_matched,
            "matched_cell_count": len(common),
            "control": control_summary,
            "treatment": treatment_summary,
            "matched_cells": matched,
            "limitations": [
                "the two TUNE seed labels repeat one geometry",
                "this TUNE screen cannot promote a model to SOTA",
                "training seeds measure optimization variability, not map generalization",
            ],
        }

    evaluator._paired_comparison = compare_urgency_arms
    return experiment, evaluator, recovery_gate, paired


def _cue_response_diagnostics(experiment_directory: Path) -> list[dict[str, Any]]:
    path = experiment_directory / "evaluation-u8" / "decision-traces.jsonl"
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["arm"], int(row["training_seed"]), int(row["episode"]["seed"]))
        groups.setdefault(key, []).append(row)
    results: list[dict[str, Any]] = []
    for (arm, training_seed, map_seed), rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["decision"]))
        cues = [
            row
            for row in rows
            if row["actor_visual_features"][4] >= 0.5
            and row["actor_visual_features"][6] > 0.0
        ]
        cue = cues[0] if cues else None
        brake = next(
            (
                row
                for row in rows
                if cue is not None
                and int(row["decision"]) >= int(cue["decision"])
                and float(row["brake"]) >= 0.05
            ),
            None,
        )
        collisions = [row for row in rows if row.get("collision")]
        final = rows[-1]
        results.append(
            {
                "arm": arm,
                "training_seed": training_seed,
                "map_seed": map_seed,
                "finish": bool(final.get("finished")),
                "retire_reason": final.get("retire_reason"),
                "max_progress": max(float(row["progress"]) for row in rows),
                "mean_speed_m_s": float(np.mean([row["speed"] for row in rows])),
                "first_visible_cue": None
                if cue is None
                else {
                    "decision": int(cue["decision"]),
                    "progress": float(cue["progress"]),
                    "speed_m_s": float(cue["speed"]),
                    "urgency": float(cue["actor_visual_features"][6]),
                },
                "first_brake_ge_0p05_after_cue": None
                if brake is None
                else {
                    "decision": int(brake["decision"]),
                    "delay_decisions": int(brake["decision"] - cue["decision"]),
                    "progress": float(brake["progress"]),
                    "speed_m_s": float(brake["speed"]),
                    "brake": float(brake["brake"]),
                },
                "first_collision": None
                if not collisions
                else {
                    "decision": int(collisions[0]["decision"]),
                    "progress": float(collisions[0]["progress"]),
                    "speed_m_s": float(collisions[0]["speed"]),
                },
            }
        )
    return results


def run_experiment(output_directory: Path) -> dict[str, Any]:
    experiment, evaluator, recovery_gate, _paired = configure_frozen_runner()
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"experiment output already contains files: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    frozen = experiment.build_frozen_protocol_record(split_manifest=SPLIT_MANIFEST)
    protocol_path = output_directory / "protocol.json"
    _write_json(protocol_path, frozen)
    protocol_sha256 = frozen["protocol_sha256"]

    training_result = experiment.run_training(
        output_directory / "run",
        frozen_protocol_path=protocol_path,
        expected_protocol_sha256=protocol_sha256,
        split_manifest=SPLIT_MANIFEST,
    )
    recovery_gate._validate_recovery_manifests(output_directory / "run")
    evaluation_result = evaluator.run_evaluation(
        output_directory / "run",
        split_manifest=SPLIT_MANIFEST,
    )
    evaluation_result["paired_variable"] = "visible_hazard_urgency_exponent"
    evaluation_result["hazard_urgency_exponent_by_arm"] = dict(ARM_EXPONENTS)
    evaluation_result["recovery_max_hazard_risk_by_arm"] = dict(RECOVERY_GATE)
    evaluation_result["recovery_stall_diagnostics"] = recovery_gate._recovery_stall_diagnostics(
        output_directory / "run"
    )
    evaluation_result["stall_risk_metric"] = "shared squared-urgency definition for both arms"
    evaluation_result["cue_response_diagnostics"] = _cue_response_diagnostics(
        output_directory / "run"
    )
    _write_json(output_directory / "run" / "evaluation-u8" / "summary.json", evaluation_result)
    result = {"training": training_result, "evaluation": evaluation_result}
    _write_json(output_directory / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/haic")
        / f"ppo-hazard-urgency-square-vs-linear-kl0p5-u8-{_now_stamp()}",
    )
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.output), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
