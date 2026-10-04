"""Run one locked PPO-only evaluation of the four frozen Lagrangian arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from training.evaluate_closed_loop import aggregate_episode_results, make_agent, run_episode
from training.site_maps import SiteMapEpisode, load_site_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = Path("artifacts/haic/lagrangian-fresh-runtime-architecture-20260925")
ARMS = (
    ("seed8104-off", 8104, "off"),
    ("seed8104-adaptive", 8104, "adaptive"),
    ("seed8105-off", 8105, "off"),
    ("seed8105-adaptive", 8105, "adaptive"),
)
PLAN_BUDGET_SECONDS = 4.5
MAX_DECISIONS = 2_000
VALID_LAP_LIMIT_MS = 13_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collision_onsets(record: dict[str, Any]) -> list[dict[str, Any]]:
    trace = record.get("decision_trace") or []
    events: list[dict[str, Any]] = []
    previous_collision = False
    for item in trace:
        colliding = bool(item.get("collision", False))
        if colliding and not previous_collision:
            events.append(
                {
                    key: item.get(key)
                    for key in (
                        "step",
                        "progress",
                        "speed",
                        "speed_change_per_second",
                        "steer",
                        "gas",
                        "brake",
                        "damage",
                        "nearest_obstacle_distance",
                    )
                }
            )
        previous_collision = colliding
    return events


def arm_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("completed")]
    valid = [
        record
        for record in completed
        if record.get("lapTimeMs") is not None
        and float(record["lapTimeMs"]) < VALID_LAP_LIMIT_MS
    ]
    laps = [float(record["lapTimeMs"]) for record in completed if record.get("lapTimeMs") is not None]
    return {
        "episodes": len(records),
        "completed": len(completed),
        "completion_rate": len(completed) / len(records) if records else 0.0,
        "valid_under_13_seconds": len(valid),
        "lap_times_ms": laps,
        "median_completed_lap_ms": statistics.median(laps) if laps else None,
        "mean_progress": statistics.mean(float(item.get("progress", 0.0)) for item in records) if records else 0.0,
        "collision_frames": sum(int(item.get("collisions", 0)) for item in records),
        "collision_onsets": sum(len(item.get("collision_onsets", [])) for item in records),
        "episodes_with_collision": sum(bool(item.get("collision_onsets")) for item in records),
        "mean_final_damage": statistics.mean(float(item.get("damage", 0.0)) for item in records) if records else 0.0,
        "mean_speed_mps": statistics.mean(float(item.get("mean_speed", 0.0)) for item in records) if records else 0.0,
        "max_speed_mps": max((float(item.get("max_speed", 0.0)) for item in records), default=0.0),
        "mean_non_collision_abs_acceleration_mps2": statistics.mean(
            float(item["non_collision_mean_abs_acceleration"])
            for item in records
            if item.get("non_collision_mean_abs_acceleration") is not None
        ) if any(item.get("non_collision_mean_abs_acceleration") is not None for item in records) else None,
        "mean_non_collision_peak_deceleration_mps2": statistics.mean(
            float(item["non_collision_peak_deceleration"])
            for item in records
            if item.get("non_collision_peak_deceleration") is not None
        ) if any(item.get("non_collision_peak_deceleration") is not None for item in records) else None,
        "mean_act_p95_ms": statistics.mean(
            float(item["act_p95_ms"]) for item in records if item.get("act_p95_ms") is not None
        ) if any(item.get("act_p95_ms") is not None for item in records) else None,
        "episodes_detail": [
            {
                "seed": int(item["seed"]),
                "completed": bool(item.get("completed")),
                "lapTimeMs": item.get("lapTimeMs"),
                "valid_under_13_seconds": bool(
                    item.get("completed")
                    and item.get("lapTimeMs") is not None
                    and float(item["lapTimeMs"]) < VALID_LAP_LIMIT_MS
                ),
                "progress": item.get("progress"),
                "collisions": item.get("collisions"),
                "collision_onsets": item.get("collision_onsets", []),
                "damage": item.get("damage"),
                "mean_speed_mps": item.get("mean_speed"),
                "max_speed_mps": item.get("max_speed"),
                "non_collision_peak_deceleration_mps2": item.get("non_collision_peak_deceleration"),
                "retire_reason": item.get("retire_reason"),
            }
            for item in records
        ],
    }


def build_tune_episodes(manifest_path: Path) -> tuple[list[SiteMapEpisode], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported site-map manifest")
    tune_entries = manifest.get("tune")
    if not isinstance(tune_entries, list) or not tune_entries:
        raise ValueError("manifest has no TUNE maps")

    episodes: list[SiteMapEpisode] = []
    map_evidence: list[dict[str, Any]] = []
    base_directory = manifest_path.parent.resolve()
    seen: set[tuple[str, int]] = set()
    for entry in tune_entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("map"), str):
            raise ValueError("invalid TUNE map entry")
        map_path = (base_directory / entry["map"]).resolve()
        map_path.relative_to(base_directory)
        site_map = load_site_map(map_path)
        seeds = entry.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("TUNE map entry has no seeds")
        map_evidence.append(
            {
                "path": map_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(map_path),
                "map_id": site_map.map_id,
                "map_kind": site_map.map_kind,
                "obstacle_mode": site_map.obstacle_mode,
                "obstacle_count": len(site_map.obstacles),
                "seeds": [int(seed) for seed in seeds],
            }
        )
        for seed in seeds:
            identity = (site_map.map_id, int(seed))
            if identity in seen:
                raise ValueError(f"duplicate TUNE episode: {identity}")
            seen.add(identity)
            episodes.append(SiteMapEpisode(site_map=site_map, seed=int(seed), source_path=map_path))
    return episodes, map_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("training/maps/site/site_map_split.json"))
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/lagrangian-fresh-runtime-architecture-20260925/tune-evaluation.json"))
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"refusing to repeat or overwrite TUNE evaluation: {output_path}")

    experiment_dir = args.experiment_dir.resolve()
    checkpoints: dict[str, Path] = {}
    for arm, _, _ in ARMS:
        checkpoint = experiment_dir / arm / "policy.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen checkpoint for {arm}: {checkpoint}")
        checkpoints[arm] = checkpoint

    episodes, tune_map_evidence = build_tune_episodes(manifest_path)
    if len(episodes) != 2:
        raise ValueError(f"locked TUNE protocol expects 2 episodes; got {len(episodes)}")

    records: list[dict[str, Any]] = []
    started = time.monotonic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with partial_path.open("x", encoding="utf-8") as partial:
        for arm, model_seed, cost_mode in ARMS:
            for episode_index, episode in enumerate(episodes, start=1):
                checkpoint = checkpoints[arm]
                try:
                    agent = make_agent(
                        policy_checkpoint=checkpoint,
                        dynamics_checkpoint=None,
                        plan_budget_seconds=PLAN_BUDGET_SECONDS,
                        planner_settings={},
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
                record["cost_mode"] = cost_mode
                record["checkpoint_sha256"] = sha256(checkpoint)
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
    pairwise = {}
    for model_seed in (8104, 8105):
        off_name = f"seed{model_seed}-off"
        adaptive_name = f"seed{model_seed}-adaptive"
        off = summary_by_arm[off_name]
        adaptive = summary_by_arm[adaptive_name]
        pairwise[str(model_seed)] = {
            "paired_geometry_note": "Both TUNE seed labels use the same registered custom map geometry.",
            "off_completion": off["completed"],
            "adaptive_completion": adaptive["completed"],
            "off_valid_under_13": off["valid_under_13_seconds"],
            "adaptive_valid_under_13": adaptive["valid_under_13_seconds"],
            "off_collision_onsets": off["collision_onsets"],
            "adaptive_collision_onsets": adaptive["collision_onsets"],
            "off_mean_damage": off["mean_final_damage"],
            "adaptive_mean_damage": adaptive["mean_final_damage"],
            "off_mean_progress": off["mean_progress"],
            "adaptive_mean_progress": adaptive["mean_progress"],
            "off_median_lap_ms": off["median_completed_lap_ms"],
            "adaptive_median_lap_ms": adaptive["median_completed_lap_ms"],
        }

    report = {
        "experiment": "fresh PPO Lagrangian safety-cost paired screen",
        "evaluation_status": "complete",
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
            "same_tune_episodes_for_all_arms": True,
            "each_arm_episode_evaluated_once": True,
        },
        "source_evidence": {
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256(manifest_path),
            "tune_maps": tune_map_evidence,
            "checkpoints": {
                arm: {
                    "path": checkpoint.relative_to(ROOT).as_posix(),
                    "sha256": sha256(checkpoint),
                }
                for arm, checkpoint in checkpoints.items()
            },
        },
        "arm_summaries": summary_by_arm,
        "paired_off_vs_adaptive": pairwise,
        "runtime_aggregates": aggregate_episode_results(records),
        "wall_time_seconds": time.monotonic() - started,
        "episodes": records,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial_path.unlink()
    print(json.dumps({"event": "tune_evaluation_complete", "output": output_path.relative_to(ROOT).as_posix(), "wall_time_seconds": report["wall_time_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
