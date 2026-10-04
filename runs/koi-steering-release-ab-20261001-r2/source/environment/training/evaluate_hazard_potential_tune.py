"""Run the locked TUNE evaluation for the paired pixel-risk PPO screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from training.evaluate_closed_loop import make_agent, run_episode
from training.evaluate_lagrangian_tune import (
    PLAN_BUDGET_SECONDS,
    VALID_LAP_LIMIT_MS,
    arm_summary,
    build_tune_episodes,
    collision_onsets,
    sha256,
)
from training.preflight_hazard_potential_screen import _source_hashes
from training.train_policy import visible_hazard_risk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = Path(
    "artifacts/haic/ppo-hazard-potential-2seed-u8-20260925"
)
ARMS = tuple(
    (f"seed{seed}-beta-{scale:.2f}", seed, scale)
    for seed in (8104, 8105)
    for scale in (0.0, 0.10)
)
MAX_DECISIONS = 2_000


class _RiskTraceAgent:
    """Collect actor-input risk values without changing its chosen actions."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.risks: list[float | None] = []

    @property
    def policy(self):
        return self.agent.policy

    def reset(self, observation) -> None:
        self.risks.clear()
        self.agent.reset(observation)

    def act(self, observation):
        action = self.agent.act(observation)
        features = getattr(self.agent.policy, "last_visual_features", None)
        if isinstance(features, torch.Tensor):
            values = features.detach().cpu().numpy().reshape(-1)
            self.risks.append(visible_hazard_risk(values))
        else:
            self.risks.append(None)
        return action


def risk_trace_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (row, row.get("next_visible_hazard_risk"))
        for record in records
        for row in (record.get("decision_trace") or [])
        if row.get("visible_hazard_risk") is not None
        and row.get("next_visible_hazard_risk") is not None
    ]
    exposed = [item for item in pairs if float(item[0]["visible_hazard_risk"]) > 0.0]
    urgent = [item for item in pairs if float(item[0]["visible_hazard_risk"]) >= 0.8]

    def decrease_fraction(items: list[tuple[dict[str, Any], Any]]) -> float | None:
        if not items:
            return None
        return sum(
            float(next_risk) < float(row["visible_hazard_risk"])
            for row, next_risk in items
        ) / len(items)

    urgent_deltas = [
        float(next_risk) - float(row["visible_hazard_risk"])
        for row, next_risk in urgent
    ]
    return {
        "risk_transition_count": len(pairs),
        "risk_exposed_transition_count": len(exposed),
        "risk_decreased_fraction": decrease_fraction(exposed),
        "urgent_transition_count": len(urgent),
        "urgent_risk_decreased_fraction": decrease_fraction(urgent),
        "mean_urgent_risk_delta": (
            statistics.mean(urgent_deltas) if urgent_deltas else None
        ),
        "urgent_collision_count": sum(
            bool(row.get("collision")) for row, _ in urgent
        ),
    }


def _validate_frozen_checkpoints(
    *,
    experiment_dir: Path,
    preflight_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    if report.get("preflight_kind") != "paired_pixel_hazard_potential_ppo_fresh_actor":
        raise ValueError("evaluation requires the hazard-potential preflight")
    if report.get("passed") is not True or report.get("execution_started") is not False:
        raise ValueError("hazard-potential preflight is not a passing frozen plan")
    if report.get("source_sha256") != _source_hashes():
        raise ValueError("training or evaluator source changed after preflight")
    resolved_manifest = manifest_path.resolve()
    if str(resolved_manifest) != report.get("manifest"):
        raise ValueError("evaluation manifest differs from the training preflight")
    if sha256(resolved_manifest) != report.get("manifest_sha256"):
        raise ValueError("evaluation manifest changed after the training preflight")
    if report.get("loaded_groups") != ["train"] or report.get("tune_or_held_out_opened") is not False:
        raise ValueError("preflight did not preserve TRAIN-only training access")
    run_directory = (experiment_dir / "runs").resolve()
    if run_directory != Path(report["run_directory"]).resolve():
        raise ValueError("evaluation run directory differs from the preflight")

    report_hash = sha256(preflight_path)
    checkpoints: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for arm, model_seed, scale in ARMS:
        checkpoint = experiment_dir / "runs" / arm / "policy.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen checkpoint for {arm}: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(saved, dict):
            raise ValueError(f"{arm} checkpoint has an unsupported layout")
        metadata = saved.get("metadata", {})
        expected_initial = report["initial_actor_hashes"][str(model_seed)][
            f"beta-{scale:.2f}"
        ]
        if int(saved.get("step", -1)) != 8192:
            raise ValueError(f"{arm} is not the fixed 8,192-step checkpoint")
        if metadata.get("updates_completed") != 8:
            raise ValueError(f"{arm} did not complete all eight PPO updates")
        if metadata.get("hazard_potential_scale") != scale:
            raise ValueError(f"{arm} checkpoint coefficient does not match its label")
        if metadata.get("initial_actor_sha256") != expected_initial:
            raise ValueError(f"{arm} actor lineage differs from the paired preflight")
        if metadata.get("preflight_report_sha256") != report_hash:
            raise ValueError(f"{arm} was not trained against this preflight report")
        if metadata.get("tune_selection_enabled") is not False:
            raise ValueError(f"{arm} used TUNE for checkpoint selection")
        if metadata.get("use_visual_features") is not True:
            raise ValueError(f"{arm} did not use the registered pixel feature input")
        checkpoints[arm] = checkpoint
        hashes[arm] = sha256(checkpoint)
    return checkpoints, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("training/maps/site/site_map_split.json")
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_EXPERIMENT / "preflight.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT / "tune-evaluation.json",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    experiment_dir = args.experiment_dir.resolve()
    preflight_path = args.preflight.resolve()
    output_path = args.output.resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to repeat or overwrite TUNE evaluation: {output_path}")

    checkpoints, checkpoint_hashes = _validate_frozen_checkpoints(
        experiment_dir=experiment_dir,
        preflight_path=preflight_path,
        manifest_path=manifest_path,
    )
    episodes, tune_map_evidence = build_tune_episodes(manifest_path)
    if len(episodes) != 2:
        raise ValueError(f"locked TUNE protocol expects 2 episodes; got {len(episodes)}")

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("x", encoding="utf-8") as partial:
        for arm, model_seed, scale in ARMS:
            for episode_index, episode in enumerate(episodes, start=1):
                checkpoint = checkpoints[arm]
                try:
                    agent = _RiskTraceAgent(
                        make_agent(
                            policy_checkpoint=checkpoint,
                            dynamics_checkpoint=None,
                            plan_budget_seconds=PLAN_BUDGET_SECONDS,
                            planner_settings={},
                        )
                    )
                    torch.manual_seed(int(episode.seed))
                    record = run_episode(
                        mode="ppo_only",
                        episode=episode,
                        agent=agent,
                        max_decisions=MAX_DECISIONS,
                        plan_budget_seconds=PLAN_BUDGET_SECONDS,
                        capture_trace=True,
                    )
                    trace = record.get("decision_trace") or []
                    if len(trace) != len(agent.risks):
                        raise ValueError("actor-input risk trace is not aligned with actions")
                    for index, row in enumerate(trace):
                        row["visible_hazard_risk"] = agent.risks[index]
                        row["next_visible_hazard_risk"] = (
                            agent.risks[index + 1]
                            if index + 1 < len(agent.risks)
                            else None
                        )
                except Exception as error:
                    record = {
                        "mode": "ppo_only",
                        "seed": int(episode.seed),
                        "map_id": episode.map_id,
                        "map_kind": episode.site_map.map_kind,
                        "obstacle_mode": episode.site_map.obstacle_mode,
                        "site_obstacle_count": len(episode.site_map.obstacles),
                        "obstacle_count": len(episode.site_map.obstacles),
                        "completed": False,
                        "lapTimeMs": None,
                        "progress": 0.0,
                        "damage": 0.0,
                        "collisions": 0,
                        "collision_onsets": [],
                        "retire_reason": "agent_setup_error",
                        "error": f"{type(error).__name__}: {error}",
                        "decision_trace": None,
                    }
                record["mode"] = arm
                record["arm"] = arm
                record["model_seed"] = model_seed
                record["hazard_potential_scale"] = scale
                record["checkpoint_sha256"] = checkpoint_hashes[arm]
                record["collision_onsets"] = collision_onsets(record)
                records.append(record)
                partial.write(json.dumps(record, sort_keys=True) + "\n")
                partial.flush()
                print(
                    json.dumps(
                        {
                            "event": "tune_episode_complete",
                            "arm": arm,
                            "episode": episode_index,
                            "seed": int(episode.seed),
                            "completed": bool(record.get("completed")),
                            "lapTimeMs": record.get("lapTimeMs"),
                            "progress": record.get("progress"),
                            "collision_onsets": len(record["collision_onsets"]),
                            "damage": record.get("damage"),
                            "retire_reason": record.get("retire_reason"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    summary_by_arm = {
        arm: arm_summary([item for item in records if item["arm"] == arm])
        for arm, _, _ in ARMS
    }
    mechanism_by_arm = {
        arm: risk_trace_summary([item for item in records if item["arm"] == arm])
        for arm, _, _ in ARMS
    }
    pairwise = {}
    for model_seed in (8104, 8105):
        control_name = f"seed{model_seed}-beta-0.00"
        treatment_name = f"seed{model_seed}-beta-0.10"
        control = summary_by_arm[control_name]
        treatment = summary_by_arm[treatment_name]
        pairwise[str(model_seed)] = {
            "paired_geometry_note": "Both TUNE seed labels use the same registered custom map geometry.",
            "control_completion": control["completed"],
            "treatment_completion": treatment["completed"],
            "control_valid_under_13": control["valid_under_13_seconds"],
            "treatment_valid_under_13": treatment["valid_under_13_seconds"],
            "control_collision_onsets": control["collision_onsets"],
            "treatment_collision_onsets": treatment["collision_onsets"],
            "control_mean_damage": control["mean_final_damage"],
            "treatment_mean_damage": treatment["mean_final_damage"],
            "control_mean_progress": control["mean_progress"],
            "treatment_mean_progress": treatment["mean_progress"],
            "control_median_lap_ms": control["median_completed_lap_ms"],
            "treatment_median_lap_ms": treatment["median_completed_lap_ms"],
            "control_risk_mechanism": mechanism_by_arm[control_name],
            "treatment_risk_mechanism": mechanism_by_arm[treatment_name],
        }

    report = {
        "experiment": "paired pixel-only hazard-potential PPO screen",
        "evaluation_status": "complete",
        "promotion_status": "diagnostic-only-no-sota-promotion",
        "split": "tune",
        "opened_split_groups": ["tune"],
        "held_out_or_official_maps_opened": False,
        "training_or_checkpoint_updates_during_evaluation": False,
        "tune_evaluations_per_arm": len(episodes),
        "tune_geometry_count": len({episode.map_id for episode in episodes}),
        "valid_lap_threshold_ms_exclusive": VALID_LAP_LIMIT_MS,
        "protocol": {
            "mode": "ppo_only",
            "planner_enabled": False,
            "plan_budget_seconds": PLAN_BUDGET_SECONDS,
            "max_decisions": MAX_DECISIONS,
            "capture_decision_trace": True,
            "capture_actor_input_hazard_risk": True,
            "same_tune_episodes_for_all_arms": True,
            "each_arm_episode_evaluated_once": True,
            "tune_used_for_checkpoint_selection_or_coefficient_tuning": False,
        },
        "source_evidence": {
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(manifest_path),
            "preflight": preflight_path.relative_to(ROOT).as_posix(),
            "preflight_sha256": sha256(preflight_path),
            "tune_maps": tune_map_evidence,
            "checkpoints": {
                arm: {
                "path": checkpoint.relative_to(ROOT).as_posix(),
                    "sha256": checkpoint_hashes[arm],
                }
                for arm, checkpoint in checkpoints.items()
            },
        },
        "arm_summaries": summary_by_arm,
        "visible_risk_mechanism_by_arm": mechanism_by_arm,
        "paired_beta_zero_vs_beta_point_one": pairwise,
        "wall_time_seconds": time.monotonic() - started,
        "episodes": records,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_path.unlink()
    print(
        json.dumps(
            {
                "event": "tune_evaluation_complete",
                "output": output_path.relative_to(ROOT).as_posix(),
                "wall_time_seconds": report["wall_time_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
