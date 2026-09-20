"""Evaluate matched pixel-only PPO and PPO+CEM agents in the local simulator.

The tune split chooses whether the planner is eligible for submission.  The
held-out manifest is only a final comparison and is never fed back into that
choice.  Every attempted episode, including errors and DNF results, is saved.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from agent import Agent
from haic_agent.planner import CEMPlanner
from training.env_factory import create_training_environment


DEFAULT_TRAIN_EPISODES = ((1, 101), (2, 102))
DEFAULT_TUNE_EPISODES = ((3, 201), (3, 202))
DEFAULT_HELD_OUT_EPISODES = tuple([(4, seed) for seed in range(301, 306)] + [(5, seed) for seed in range(401, 406)])
DEFAULT_PLAN_BUDGET_SECONDS = 4.5
DEFAULT_PLANNER_SETTINGS = {
    "horizon": 4,
    "population": 16,
    "iterations": 2,
    "candidate_batch_size": 8,
    "uncertainty_cost": 1.0,
}
DEFAULT_PLANNER_CANDIDATES = (
    {"horizon": 3, "population": 8, "iterations": 1, "candidate_batch_size": 8, "uncertainty_cost": 0.5},
    {"horizon": 4, "population": 16, "iterations": 2, "candidate_batch_size": 8, "uncertainty_cost": 0.75},
    {"horizon": 6, "population": 24, "iterations": 2, "candidate_batch_size": 8, "uncertainty_cost": 1.5},
)


def validate_episode_splits(
    train: Iterable[tuple[int, int]], tune: Iterable[tuple[int, int]], held_out: Iterable[tuple[int, int]]
) -> None:
    """Reject seed reuse before any train, tuning, or final run starts."""
    groups = {
        "train": tuple((int(track), int(seed)) for track, seed in train),
        "tune": tuple((int(track), int(seed)) for track, seed in tune),
        "held_out": tuple((int(track), int(seed)) for track, seed in held_out),
    }
    if len(groups["held_out"]) < 10:
        raise ValueError("held_out must contain at least 10 episodes")
    for name, values in groups.items():
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate episodes")
    for left, right in (("train", "tune"), ("train", "held_out"), ("tune", "held_out")):
        if set(groups[left]) & set(groups[right]):
            raise ValueError(f"{left} and {right} episode sets overlap")


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def aggregate_episode_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all outcomes without filtering failed or unfinished episodes."""
    records = list(results)
    modes = sorted({str(record["mode"]) for record in records})
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        completed = [record for record in selected if bool(record.get("completed"))]
        finished_laps = [float(record["lapTimeMs"]) for record in completed if record.get("lapTimeMs") is not None]
        progress = [float(record.get("progress", 0.0)) for record in selected]
        latencies = [float(value) for record in selected for value in record.get("act_latency_ms", [])]
        by_mode[mode] = {
            "episodes": len(selected),
            "completed_episodes": len(completed),
            "completion_rate": float(len(completed) / len(selected)) if selected else 0.0,
            "median_finished_lap_time_ms": _percentile(finished_laps, 50),
            "p90_finished_lap_time_ms": _percentile(finished_laps, 90),
            "mean_progress": float(np.mean(progress)) if progress else 0.0,
            "act_latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "max": max(latencies) if latencies else None,
            },
            "invalid_actions": int(sum(int(record.get("invalid_actions", 0)) for record in selected)),
        }
    return {"episodes": len(records), "by_mode": by_mode}


def _rank(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    median = metrics.get("median_finished_lap_time_ms")
    p90 = metrics.get("p90_finished_lap_time_ms")
    return (
        float(metrics.get("completion_rate", 0.0)),
        -float(median) if median is not None else float("-inf"),
        -float(p90) if p90 is not None else float("-inf"),
        float(metrics.get("mean_progress", 0.0)),
    )


def choose_planner_from_tune(ppo_only: dict[str, Any], ppo_cem: dict[str, Any]) -> bool:
    """Enable CEM only when tune metrics strictly improve the rank ordering."""
    return _rank(ppo_cem) > _rank(ppo_only)


def _safe_action(agent: Any, observation: np.ndarray) -> tuple[np.ndarray, bool, float]:
    started = time.monotonic()
    try:
        action = np.asarray(agent.act(observation), dtype=np.float32).reshape(-1)
    except Exception:
        return np.zeros(3, dtype=np.float32), False, (time.monotonic() - started) * 1000.0
    elapsed = (time.monotonic() - started) * 1000.0
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        return np.zeros(3, dtype=np.float32), False, elapsed
    return np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]), True, elapsed


def run_episode(
    *,
    mode: str,
    track_id: int,
    seed: int,
    agent: Any,
    max_decisions: int,
    plan_budget_seconds: float,
    environment_factory: Callable[..., Any] = create_training_environment,
) -> dict[str, Any]:
    """Run one full local episode or record its error/DNF with timing evidence."""
    started = time.monotonic()
    environment = None
    latencies: list[float] = []
    invalid_actions = 0
    consecutive_invalid_actions = 0
    collisions = 0
    info: dict[str, Any] = {}
    steps = 0
    terminated = False
    truncated = False
    retire_reason: str | None = None
    start_simulation_time: float | None = None
    try:
        environment = environment_factory(track_id=track_id, seed=seed, max_decisions=max_decisions)
        observation, info = environment.reset()
        start_simulation_time = float(getattr(environment.unwrapped, "t", 0.0))
        reset_started = time.monotonic()
        agent.reset(observation)
        reset_ms = (time.monotonic() - reset_started) * 1000.0
        if reset_ms > 5_000:
            retire_reason = "reset_timeout"
        while retire_reason is None and steps < max_decisions:
            action, valid, latency_ms = _safe_action(agent, observation)
            latencies.append(latency_ms)
            if latency_ms > 5_000:
                retire_reason = "act_timeout"
                break
            if not valid:
                invalid_actions += 1
                consecutive_invalid_actions += 1
                if consecutive_invalid_actions >= 10:
                    retire_reason = "invalid_action"
                    break
            else:
                consecutive_invalid_actions = 0
            observation, _, terminated, truncated, info = environment.step(action)
            collisions += int(bool(info.get("collision", False)))
            steps += 1
            if terminated or truncated:
                break
        finish_time = getattr(environment.unwrapped, "finish_time_s", None)
        completed = finish_time is not None
        lap_time_ms = (
            round((float(finish_time) - float(start_simulation_time)) * 1000.0)
            if completed and start_simulation_time is not None
            else None
        )
        if not completed and retire_reason is None:
            retire_reason = info.get("retire_reason")
            if retire_reason is None:
                retire_reason = "off_track" if terminated else "max_steps" if truncated or steps >= max_decisions else "unknown"
        progress = float(info.get("progress", getattr(environment.environment, "_calculate_progress", lambda: 0.0)()))
        return {
            "mode": mode,
            "track_id": int(track_id),
            "seed": int(seed),
            "completed": bool(completed),
            "lapTimeMs": lap_time_ms,
            "progress": progress,
            "damage": float(info.get("damage", 0.0)),
            "collisions": collisions,
            "retire_reason": retire_reason,
            "invalid_actions": invalid_actions,
            "consecutive_invalid_actions": consecutive_invalid_actions,
            "steps": steps,
            "act_latency_ms": latencies,
            "act_p50_ms": _percentile(latencies, 50),
            "act_p95_ms": _percentile(latencies, 95),
            "act_max_ms": max(latencies) if latencies else None,
            "wall_time_s": time.monotonic() - started,
            "per_call_limit_s": 5.0,
            "plan_budget_s": float(plan_budget_seconds),
            "torch_seed": int(seed),
        }
    except Exception as error:
        return {
            "mode": mode,
            "track_id": int(track_id),
            "seed": int(seed),
            "completed": False,
            "lapTimeMs": None,
            "progress": float(info.get("progress", 0.0)),
            "damage": float(info.get("damage", 0.0)),
            "collisions": collisions,
            "retire_reason": "evaluation_error",
            "error": f"{type(error).__name__}: {error}",
            "invalid_actions": invalid_actions,
            "consecutive_invalid_actions": consecutive_invalid_actions,
            "steps": steps,
            "act_latency_ms": latencies,
            "act_p50_ms": _percentile(latencies, 50),
            "act_p95_ms": _percentile(latencies, 95),
            "act_max_ms": max(latencies) if latencies else None,
            "wall_time_s": time.monotonic() - started,
            "per_call_limit_s": 5.0,
            "plan_budget_s": float(plan_budget_seconds),
            "torch_seed": int(seed),
        }
    finally:
        if environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            else:
                environment.environment.close()


def make_agent(
    *, policy_checkpoint: Path, dynamics_checkpoint: Path | None, plan_budget_seconds: float, planner_settings: dict[str, Any]
) -> Agent:
    """Construct matched agents; PPO-only passes no dynamics to disable CEM."""
    planner_enabled = dynamics_checkpoint is not None
    planner = CEMPlanner(**planner_settings) if planner_enabled else None
    return Agent(
        policy_checkpoint=str(policy_checkpoint),
        dynamics_checkpoint=str(dynamics_checkpoint) if dynamics_checkpoint is not None else None,
        planner=planner,
        planner_enabled=planner_enabled,
        strict_checkpoint_loading=True,
        plan_budget=plan_budget_seconds,
    )


def evaluate_mode(
    *,
    mode: str,
    episodes: Iterable[tuple[int, int]],
    policy_checkpoint: Path,
    dynamics_checkpoint: Path,
    plan_budget_seconds: float,
    max_decisions: int,
    planner_settings: dict[str, Any],
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    use_planner = mode != "ppo_only"
    results = []
    for track_id, seed in episodes:
        try:
            agent = make_agent(
                policy_checkpoint=policy_checkpoint,
                dynamics_checkpoint=dynamics_checkpoint if use_planner else None,
                plan_budget_seconds=plan_budget_seconds,
                planner_settings=planner_settings,
            )
            torch.manual_seed(int(seed))
            record = run_episode(
                    mode=mode,
                    track_id=track_id,
                    seed=seed,
                    agent=agent,
                    max_decisions=max_decisions,
                    plan_budget_seconds=plan_budget_seconds,
                )
        except Exception as error:
            record = {
                "mode": mode, "track_id": int(track_id), "seed": int(seed), "completed": False,
                "lapTimeMs": None, "progress": 0.0, "damage": 0.0, "collisions": 0,
                "retire_reason": "agent_setup_error", "error": f"{type(error).__name__}: {error}",
                "invalid_actions": 0, "consecutive_invalid_actions": 0, "steps": 0,
                "act_latency_ms": [], "act_p50_ms": None, "act_p95_ms": None, "act_max_ms": None,
                "wall_time_s": 0.0, "per_call_limit_s": 5.0, "plan_budget_s": float(plan_budget_seconds),
                "torch_seed": int(seed),
            }
        results.append(record)
        if on_record is not None:
            on_record(record)
    return results


def record_single_episode(
    *,
    output_path: Path,
    mode: str,
    track_id: int,
    seed: int,
    policy_checkpoint: Path,
    dynamics_checkpoint: Path,
    plan_budget_seconds: float = DEFAULT_PLAN_BUDGET_SECONDS,
    max_decisions: int = 2_000,
    planner_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist one full-cap default-budget episode separately from fast sweeps."""
    settings = dict(DEFAULT_PLANNER_SETTINGS if planner_settings is None else planner_settings)
    use_planner = mode == "ppo_cem"
    agent = make_agent(
        policy_checkpoint=policy_checkpoint,
        dynamics_checkpoint=dynamics_checkpoint if use_planner else None,
        plan_budget_seconds=plan_budget_seconds,
        planner_settings=settings,
    )
    torch.manual_seed(int(seed))
    record = run_episode(
        mode=mode,
        track_id=track_id,
        seed=seed,
        agent=agent,
        max_decisions=max_decisions,
        plan_budget_seconds=plan_budget_seconds,
    )
    payload = {
        "budget_class": "default_4.5s" if plan_budget_seconds == DEFAULT_PLAN_BUDGET_SECONDS else "fast_budget",
        "max_decisions": max_decisions,
        "record": record,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def record_deployment_decision(
    summary_path: Path, *, planner_enabled: bool, reason: str
) -> dict[str, Any]:
    """Keep tune selection and conservative package selection as separate facts."""
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    report["tune_selected_planner_enabled"] = bool(
        report.pop("planner_enabled_for_submission", report.get("tune_selected_planner_enabled", False))
    )
    report["package_planner_enabled"] = bool(planner_enabled)
    report["package_selection_reason"] = str(reason)
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_evaluation(
    *,
    output_directory: Path,
    policy_checkpoint: Path,
    dynamics_checkpoint: Path,
    plan_budget_seconds: float = DEFAULT_PLAN_BUDGET_SECONDS,
    max_decisions: int = 2_000,
    train_episodes: tuple[tuple[int, int], ...] = DEFAULT_TRAIN_EPISODES,
    tune_episodes: tuple[tuple[int, int], ...] = DEFAULT_TUNE_EPISODES,
    held_out_episodes: tuple[tuple[int, int], ...] = DEFAULT_HELD_OUT_EPISODES,
    planner_settings: dict[str, Any] | None = None,
    planner_candidates: tuple[dict[str, Any], ...] | None = None,
    package_planner_enabled: bool | None = None,
    package_selection_reason: str | None = None,
) -> dict[str, Any]:
    """Tune CEM on tune episodes then make a final matched held-out report."""
    validate_episode_splits(train_episodes, tune_episodes, held_out_episodes)
    output_directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_directory / "episodes.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    def persist(record: dict[str, Any]) -> None:
        with jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    candidates = (
        (dict(planner_settings),)
        if planner_settings is not None
        else tuple(dict(settings) for settings in (planner_candidates or DEFAULT_PLANNER_CANDIDATES))
    )
    tune_records = evaluate_mode(mode="ppo_only", episodes=tune_episodes, policy_checkpoint=policy_checkpoint,
                                 dynamics_checkpoint=dynamics_checkpoint, plan_budget_seconds=plan_budget_seconds,
                                 max_decisions=max_decisions, planner_settings=candidates[0], on_record=persist)
    tune_summary = aggregate_episode_results(tune_records)
    ppo_only_metrics = tune_summary["by_mode"]["ppo_only"]
    candidate_reports: list[dict[str, Any]] = []
    for index, settings in enumerate(candidates):
        mode = f"ppo_cem_candidate_{index}"
        records = evaluate_mode(mode=mode, episodes=tune_episodes, policy_checkpoint=policy_checkpoint,
                                dynamics_checkpoint=dynamics_checkpoint, plan_budget_seconds=plan_budget_seconds,
                                max_decisions=max_decisions, planner_settings=settings, on_record=persist)
        tune_records += records
        metrics = aggregate_episode_results(records)["by_mode"][mode]
        candidate_reports.append({"index": index, "settings": settings, "metrics": metrics})
    improving = [candidate for candidate in candidate_reports if choose_planner_from_tune(ppo_only_metrics, candidate["metrics"])]
    selected = max(improving, key=lambda candidate: _rank(candidate["metrics"])) if improving else None
    selected_settings = dict(selected["settings"] if selected is not None else candidates[0])
    planner_enabled = selected is not None
    tune_summary = aggregate_episode_results(tune_records)
    held_out_records = evaluate_mode(mode="ppo_only", episodes=held_out_episodes, policy_checkpoint=policy_checkpoint,
                                     dynamics_checkpoint=dynamics_checkpoint, plan_budget_seconds=plan_budget_seconds,
                                     max_decisions=max_decisions, planner_settings=selected_settings, on_record=persist)
    held_out_records += evaluate_mode(mode="ppo_cem", episodes=held_out_episodes, policy_checkpoint=policy_checkpoint,
                                      dynamics_checkpoint=dynamics_checkpoint, plan_budget_seconds=plan_budget_seconds,
                                      max_decisions=max_decisions, planner_settings=selected_settings, on_record=persist)
    held_out_summary = aggregate_episode_results(held_out_records)
    report = {
        "schema_version": 1,
        "budget_class": "default_4.5s" if plan_budget_seconds == DEFAULT_PLAN_BUDGET_SECONDS else "fast_budget",
        "plan_budget_seconds": float(plan_budget_seconds),
        "max_decisions": int(max_decisions),
        "episode_manifests": {"train": train_episodes, "tune": tune_episodes, "held_out": held_out_episodes},
        "planner_candidate_tune_results": candidate_reports,
        "selected_planner_settings": selected_settings,
        "tune_selected_planner_enabled": planner_enabled,
        "package_planner_enabled": planner_enabled if package_planner_enabled is None else bool(package_planner_enabled),
        "package_selection_reason": package_selection_reason or "tune rank selection",
        "planner_selection_source": "tune_only",
        "tune": tune_summary,
        "held_out_final_comparison": held_out_summary,
        "episode_jsonl": jsonl_path.name,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"episodes_path": str(jsonl_path), "summary_path": str(summary_path), **report}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/haic/evaluation"))
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--plan-budget", type=float, default=DEFAULT_PLAN_BUDGET_SECONDS)
    parser.add_argument("--max-decisions", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        output_directory=args.output,
        policy_checkpoint=args.policy_checkpoint,
        dynamics_checkpoint=args.dynamics_checkpoint,
        plan_budget_seconds=args.plan_budget,
        max_decisions=args.max_decisions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
