"""Isolated, from-scratch damage-target TD-MPC2 TRAIN operator.

Run from the repository root with ``python -m scripts.train_tdmpc2_damage``.
An externally frozen, SHA-pinned experiments/*.json protocol is mandatory;
--preflight is file/resource-only and --run never resumes a partial attempt.
The official environment and all primary driving metrics retain RAW reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from haic.algorithms.tdmpc2.haic_env import (
    environment_action, episode_boundary, make_training_env, model_observation,
)
from haic.algorithms.tdmpc2.reward import DAMAGE_COST, DamageIncrementReward
from scripts import train_tdmpc2_long as base


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-damage-train-v1"
SOURCE_PATHS = base.SOURCE_PATHS | {
    "haic/algorithms/tdmpc2/reward.py", "scripts/train_tdmpc2_damage.py",
}
BASELINE = {
    "baseline_training_protocol": ("experiments/tdmpc2-long-reused-train-v2.json",
                                   "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"),
    "baseline_training_result": ("experiments/tdmpc2-long-reused-train-v2-100k-result.json",
                                 "bacf4c7e7a53794ecec7797746388695bac01096287f7aff09b5feafc4e3b515"),
    "baseline_full_eval_protocol": ("experiments/tdmpc2-full-consumed-train-v1.json",
                                     "874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd"),
    "baseline_full_eval_result": ("experiments/tdmpc2-full-consumed-train-v1-result.json",
                                   "3bc41dd2a35bfd488a6cd13a2778d9344f6929873edd77e701b067e31fd1366b"),
}
REWARD_TARGET = {"kind": "cumulative_damage_increment", "coefficient": 5.0,
                 "initial_damage": 0.0, "maximum_damage": 1.0,
                 "negative_delta_tolerance": 1e-6, "replay_reward": "training_reward",
                 "environment_reward": "raw", "checkpoint_reward_prediction": "training_reward"}
FIELDS = {"format", "purpose", "upstream_revision", "source_sha256", *BASELINE,
          "selection", "cells", "episode_schedule", "environment", "run_dir", "training",
          "checkpoint_targets", "resources", "seed_schedule", "reward_target"}


def _exact(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        return value.keys() == expected.keys() and all(
            _exact(value[key], item) for key, item in expected.items())
    if isinstance(expected, list):
        assert isinstance(value, list)
        return len(value) == len(expected) and all(
            _exact(actual, item) for actual, item in zip(value, expected))
    return value == expected


def _check_sources(root: Path, sources: dict, baseline_sources: dict) -> None:
    if (not isinstance(sources, dict) or set(sources) != SOURCE_PATHS
            or {name: sources[name] for name in baseline_sources} != baseline_sources):
        raise ValueError("damage runner requires all unchanged, pinned baseline source bytes")
    for name, expected in sources.items():
        if base.digest(base._inside(root, name)) != base._sha(expected):
            raise ValueError(f"source hash mismatch: {name}")


def _runtime_sources(root: Path, sources: dict, protocol_path: Path, protocol_sha256: str) -> None:
    if base.digest(base._inside(root, protocol_path)) != protocol_sha256:
        raise ValueError("damage protocol changed during run")
    name, expected = BASELINE["baseline_training_protocol"]
    if base.digest(base._inside(root, name)) != expected:
        raise ValueError("baseline training protocol changed during run")
    _check_sources(root, sources, base._json(root / name)["source_sha256"])


def _evidence(root: Path, p: dict) -> dict:
    refs = {key: base._reference(root, p[key], path) for key, (path, _) in BASELINE.items()}
    prior = refs["baseline_training_protocol"]
    result = refs["baseline_training_result"]
    eval_protocol = refs["baseline_full_eval_protocol"]
    evaluated = refs["baseline_full_eval_result"]
    # The frozen runner verifies the original four-road lineage and resources without a reset.
    base.preflight(root / BASELINE["baseline_training_protocol"][0],
                   BASELINE["baseline_training_protocol"][1], root / prior["run_dir"], root=root)
    if (result.get("training_protocol_sha256") != BASELINE["baseline_training_protocol"][1]
            or result.get("run_status") != "completed_boundary_at_least_100k"
            or (result.get("decisions"), result.get("updates"), result.get("episodes")) != (100354, 100354, 307)
            or result.get("checkpoint_target") != 100000
            or evaluated.get("status") != "complete_valid_full_episode_finish_comparison"
            or evaluated.get("evaluation_protocol_sha256") != BASELINE["baseline_full_eval_protocol"][1]
            or evaluated.get("source_model_sha256") != result.get("checkpoint_sha256")
            or evaluated.get("source_training_result_sha256") != result.get("run_result_sha256")
            or evaluated.get("denominators") != {
                "distinct_training_roads": 4, "repeats_per_road_per_mode": 2,
                "prior_episodes": 8, "mppi_episodes": 8, "total_episodes": 16,
                "max_decisions_per_episode": 2000,
            } or any(evaluated.get(mode, {}).get("finishes") != 0
                     or evaluated[mode].get("episodes") != 8 for mode in ("prior", "mppi"))
            or eval_protocol.get("cells") != base.CELLS
            or eval_protocol.get("environment") != base.ENVIRONMENT
            or eval_protocol.get("source", {}).get("protocol") != p["baseline_training_protocol"]):
        raise ValueError("frozen 100k/full-episode baseline evidence differs")
    for name, expected in (
        (result["run_result"], result["run_result_sha256"]),
        (result["checkpoint"], result["checkpoint_sha256"]),
        (str(Path(result["run"]) / "training.jsonl"), result["training_ledger_sha256"]),
        (str(Path(result["run"]) / "steps.jsonl"), result["step_ledger_sha256"]),
        (evaluated["primary_result"], evaluated["primary_result_sha256"]),
        (evaluated["episode_ledger"], evaluated["episode_ledger_sha256"]),
    ):
        if base.digest(base._inside(root, name)) != base._sha(expected):
            raise ValueError(f"baseline artifact hash mismatch: {name}")
    primary = base._json(base._inside(root, evaluated["primary_result"]))
    if (primary.get("status") != "complete" or primary.get("full_episode_finish_comparison_valid") is not True
            or primary.get("episodes_sha256") != evaluated["episode_ledger_sha256"]
            or primary.get("protocol_sha256") != p["baseline_full_eval_protocol"]["sha256"]):
        raise ValueError("baseline full-episode receipt differs")
    return prior


def preflight(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT) -> dict:
    """Validate externally frozen schema, evidence, source bytes and raw resources; zero resets."""
    root = root.resolve(strict=True)
    path = base._inside(root, protocol_path)
    if (path.parent != root / "experiments" or path.suffix != ".json"
            or base.digest(path) != base._sha(protocol_sha256)):
        raise ValueError("frozen damage experiments/*.json protocol SHA mismatch")
    p = base._json(path)
    output = base._inside(root, run_dir, required=False)
    if (set(p) != FIELDS or p["format"] != FORMAT
            or p["purpose"] != "consumed-TRAIN-development"
            or p["upstream_revision"] != base.UPSTREAM
            or p["seed_schedule"] != base.SEED_SCHEDULE
            or output.parent != root / "runs" or not output.name.startswith("tdmpc2-damage-")
            or output.exists() or p["run_dir"] != output.relative_to(root).as_posix()
            or not _exact(p["selection"], {"arm": "independent_3d", "action_dim": 3})
            or not _exact(p["cells"], base.CELLS) or not _exact(p["episode_schedule"], [0, 1, 2, 3])
            or not _exact(p["environment"], base.ENVIRONMENT)
            or not _exact(p["checkpoint_targets"], base.TARGETS)
            or not _exact(p["reward_target"], REWARD_TARGET)
            or DAMAGE_COST != 5.0):
        raise ValueError("invalid damage-only consumed TRAIN protocol or output path")
    for key, (name, sha) in BASELINE.items():
        if p[key] != {"path": name, "sha256": sha}:
            raise ValueError(f"baseline pin differs: {key}")
    prior = _evidence(root, p)
    if (not _exact(p["training"], prior["training"]) or not _exact(p["resources"], prior["resources"])
            or prior["selection"] != p["selection"] or prior["cells"] != p["cells"]
            or prior["checkpoint_targets"] != p["checkpoint_targets"]):
        raise ValueError("training, action, checkpoint or resource settings differ from raw H3 baseline")
    _check_sources(root, p["source_sha256"], prior["source_sha256"])
    base._resources(output, p)
    return p


def _diagnostic(learner, probe: dict, seed: int) -> dict:
    """Reuse the identical in-replay probe, but label every reward as SHAPED."""
    metrics = base._diagnostic(learner, probe, seed)
    renamed = {
        "reward_min": "training_reward_min", "reward_max": "training_reward_max",
        "reward_mae": "training_reward_mae",
        "constant_reward_mae": "constant_training_reward_mae",
        "h3_raw_return_min": "h3_training_return_min",
        "h3_raw_return_max": "h3_training_return_max",
        "h3_discounted_return_min": "h3_discounted_training_return_min",
        "h3_discounted_return_max": "h3_discounted_training_return_max",
        "h3_unique_raw_returns_at_1e_minus_6": "h3_unique_training_returns_at_1e_minus_6",
    }
    return {renamed.get(key, key): value for key, value in metrics.items()}


def _checkpoint(path: Path, learner, replay, probe, decisions: int, updates: int,
                episodes: int, protocol_sha256: str, sources: dict, target: int) -> str:
    if replay.active_length or path.exists():
        raise ValueError("damage checkpoint requires a new whole-episode boundary")
    state = {"format": FORMAT, "protocol_sha256": protocol_sha256, "source_sha256": sources,
             "reward_target": REWARD_TARGET, "target": target, "decisions": decisions,
             "updates": updates, "episodes": episodes, "action_dim": replay.action_dim,
             "learner": learner.state_dict(), "optim": learner.optim.state_dict(),
             "pi_optim": learner.pi_optim.state_dict(), "replay": replay.state_dict(),
             "probe": probe, "rng": base._rng_state(), "resume_supported": False}
    with path.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return base.digest(path)


def run(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT,
        env_factory=make_training_env, clock=time.monotonic) -> dict:
    """Never constructs an environment before the complete read-only preflight."""
    p = preflight(protocol_path, protocol_sha256, run_dir, root=root)
    root = root.resolve(strict=True)
    output = base._inside(root, run_dir, required=False)
    output.mkdir(exist_ok=False)
    ledger, steps = output / "training.jsonl", output / "steps.jsonl"
    decisions = updates = episodes = environment_steps_applied = 0
    reset_intents = 0
    started = started_cpu = runtime = None
    start_journaled = False
    pending_step = None
    env = replay = None
    try:
        cfg = p["training"]
        started, started_cpu = clock(), time.process_time()
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "gymnasium": gym.__version__,
                   "torch_cuda": torch.version.cuda, "device": cfg["device"],
                   "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
                   if cfg["device"] == "cuda" else None}
        base._journal(ledger, {"event": "start", "protocol_sha256": protocol_sha256,
                               "source_sha256": p["source_sha256"], "reward_target": REWARD_TARGET,
                               "seed_schedule": base.SEED_SCHEDULE, "runtime": runtime,
                               "resume_supported": False})
        start_journaled = True
        probe = latest_delta = None
        rolling_sum: dict[str, float] = {}
        rolling_count = delta_count = 0
        delta_sum = np.zeros(3, dtype=np.float64)
        delta_abs_sum = np.zeros(3, dtype=np.float64)
        native_delta_abs_sum = np.zeros(3, dtype=np.float64)
        delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
        interval_episodes: list[dict] = []
        checkpoints: list[dict] = []
        base._resources(output, p)
        _runtime_sources(root, p["source_sha256"], protocol_path, protocol_sha256)
        base._seed(cfg["seed"])
        learner, replay, planner = base._components(p)
        shaper = DamageIncrementReward()
        while decisions < base.TARGETS[-1]:
            if cfg["decision_cap"] - decisions < cfg["max_steps"]:
                raise RuntimeError("decision cap prevents required 100k whole-episode target")
            if clock() - started >= p["resources"]["max_wall_seconds"]:
                raise TimeoutError("wall budget expired before final TRAIN milestone")
            base._resources(output, p)
            _runtime_sources(root, p["source_sha256"], protocol_path, protocol_sha256)
            cell = p["cells"][p["episode_schedule"][episodes % 4]]
            reset_intents += 1
            base._journal(ledger, {"event": "reset_intent", "episode": episodes,
                                   "decisions": decisions, **cell})
            if env is None:
                env = env_factory(cfg["max_steps"])
            obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
            if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                    or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                raise ValueError("reset did not honor pinned consumed TRAIN road")
            pixels = model_observation(obs)
            replay.start_episode(pixels)
            planner.reset()
            shaper.reset()
            base._journal(ledger, {"event": "reset", "episode": episodes,
                                   "decisions": decisions, **cell})
            length = 0
            raw_total = training_total = 0.0
            done = False
            progress = damage = 0.0
            info: dict = {}
            terminated = truncated = terminal = False
            action_trace, native_trace = hashlib.sha256(), hashlib.sha256()
            while length < cfg["max_steps"]:
                if clock() - started >= p["resources"]["max_wall_seconds"]:
                    raise TimeoutError("partial TRAIN episode: wall budget exhausted")
                index = decisions
                if index < cfg["seed_steps"]:
                    action = np.random.uniform(-1, 1, size=3).astype(np.float32)
                else:
                    if updates < cfg["pretrain_updates"]:
                        raise RuntimeError("seed pretraining incomplete before planned action")
                    action, latest_delta = base._planned_action(
                        learner.model, planner, pixels, cfg["device"], t0=(length == 0))
                native = environment_action(action)
                if latest_delta is not None and index >= cfg["seed_steps"]:
                    delta = np.asarray(latest_delta["delta"], dtype=np.float64)
                    prior = environment_action(np.asarray(latest_delta["prior_mean"], dtype=np.float32))
                    mean = environment_action(np.asarray(
                        latest_delta["mppi_weighted_elite_mean"], dtype=np.float32))
                    delta_count += 1
                    delta_sum += delta
                    delta_abs_sum += np.abs(delta)
                    native_delta_abs_sum += np.abs(mean.astype(np.float64) - prior)
                    delta_l2_sum += latest_delta["delta_l2"]
                    delta_l2_max = max(delta_l2_max, latest_delta["delta_l2"])
                    applied_delta_l2_sum += latest_delta["applied_delta_l2"]
                action_trace.update(action.tobytes())
                native_trace.update(native.tobytes())
                pending_step = {"episode": episodes, "decision": decisions + 1, **cell,
                                "action_f32_hex": action.tobytes().hex(),
                                "native_action_f32_hex": native.tobytes().hex()}
                obs, raw_reward, terminated, truncated, info = env.step(native)
                environment_steps_applied += 1
                pending_step.update(raw_reward_repr=repr(raw_reward),
                                    damage_repr=repr(info.get("damage")),
                                    terminated=bool(terminated), truncated=bool(truncated))
                if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                        or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                    raise ValueError("TRAIN road changed during episode")
                following = model_observation(obs)
                done, terminal = episode_boundary(terminated, truncated, info)
                training_reward, damage_delta = shaper.step(raw_reward, info["damage"])
                raw_reward = float(raw_reward)
                progress, damage = float(info["progress"]), float(info["damage"])
                if not math.isfinite(progress):
                    raise ValueError("nonfinite raw TRAIN progress")
                if not done:
                    replay.add_step(following, action, training_reward, terminal=False)
                decisions += 1
                length += 1
                raw_total += raw_reward
                training_total += training_reward
                base._journal(steps, {**pending_step, "reward": raw_reward,
                                      "training_reward": training_reward, "damage_delta": damage_delta,
                                      "damage": damage, "progress": progress,
                                      "finished": bool(info.get("finished", False)),
                                      "terminated": bool(terminated), "truncated": bool(truncated),
                                      "terminal": terminal})
                pending_step = None
                if index == cfg["seed_steps"] - 1 and not replay.eligible_windows():
                    raise RuntimeError("no completed H+1 replay window at seed boundary")
                count = cfg["pretrain_updates"] if index == cfg["seed_steps"] - 1 else int(index >= cfg["seed_steps"])
                if count and probe is None:
                    probe = base._freeze_probe(replay, cfg["batch_size"])
                for _ in range(count):
                    if updates >= cfg["update_cap"]:
                        raise RuntimeError("update cap reached before 100k boundary")
                    if clock() - started >= p["resources"]["max_wall_seconds"]:
                        raise TimeoutError("partial learner updates: wall budget exhausted")
                    if updates % 256 == 0:
                        base._resources(output, p)
                    metrics = learner.update(replay)
                    if any(not math.isfinite(float(value)) for value in metrics.values()):
                        raise FloatingPointError("nonfinite learner update")
                    updates += 1
                    rolling_count += 1
                    for key, value in metrics.items():
                        label = "training_reward_loss" if key == "reward_loss" else key
                        rolling_sum[label] = rolling_sum.get(label, 0.0) + float(value)
                # Keep the frozen runner's update-before-just-ended-episode admission.
                if done:
                    replay.add_step(following, action, training_reward,
                                    terminated=bool(terminated), truncated=bool(truncated), terminal=terminal)
                pixels = following
                if done:
                    break
            if replay.active_length or not done:
                raise RuntimeError("environment failed to close its full TRAIN episode")
            row = {"event": "episode", "episode": episodes, **cell,
                   "decisions": decisions, "updates": updates, "length": length,
                   "return": raw_total, "training_return": training_total,
                   "progress": progress, "damage": damage,
                   "finished": bool(info.get("finished", False)),
                   "terminated": bool(terminated), "truncated": bool(truncated),
                   "terminal": terminal, "finish_time_s": info.get("finish_time_s"),
                   "action_trace_sha256": action_trace.hexdigest(),
                   "native_action_trace_sha256": native_trace.hexdigest()}
            base._journal(ledger, row)
            interval_episodes.append({key: row[key] for key in (
                "episode", "track_id", "geometry_seed", "decisions", "length", "return",
                "training_return", "progress", "damage", "finished")})
            episodes += 1
            if len(checkpoints) < len(base.TARGETS) and decisions >= base.TARGETS[len(checkpoints)]:
                target = base.TARGETS[len(checkpoints)]
                if probe is None or rolling_count == 0 or latest_delta is None or delta_count == 0:
                    raise RuntimeError("checkpoint lacks frozen TRAIN probe, updates or planner gap")
                base._resources(output, p)
                _runtime_sources(root, p["source_sha256"], protocol_path, protocol_sha256)
                diagnostic = _diagnostic(learner, probe, cfg["seed"] + 101)
                report = {"target": target, "decisions": decisions, "updates": updates,
                          "episodes": episodes, "reward_prediction_target": "training_reward",
                          "rolling_update_count": rolling_count,
                          "rolling_update_means": {key: value / rolling_count for key, value in rolling_sum.items()},
                          "train_episodes_since_previous": interval_episodes,
                          "q_scale": float(learner.q_scale.item()),
                          "policy_entropy_rolling_mean": rolling_sum["pi_entropy"] / rolling_count,
                          "resource_usage": base._cost(clock, started, started_cpu),
                          "same_observation_prior_vs_mppi_interval": {
                              "planned_actions": delta_count,
                              "mean_signed_model_delta": (delta_sum / delta_count).tolist(),
                              "mean_absolute_model_delta": (delta_abs_sum / delta_count).tolist(),
                              "mean_absolute_native_delta": (native_delta_abs_sum / delta_count).tolist(),
                              "mean_model_delta_l2": delta_l2_sum / delta_count,
                              "max_model_delta_l2": delta_l2_max,
                              "mean_applied_exploration_noised_delta_l2": applied_delta_l2_sum / delta_count},
                          "same_observation_prior_vs_mppi": latest_delta,
                          "frozen_train_probe": diagnostic, "step_ledger_sha256": base.digest(steps)}
                name = f"checkpoint-at-least-{target:06d}-step-{decisions:06d}.pt"
                checksum = _checkpoint(output / name, learner, replay, probe, decisions, updates,
                                       episodes, protocol_sha256, p["source_sha256"], target)
                report.update(path=name, sha256=checksum)
                base._journal(ledger, {"event": "checkpoint", **report})
                checkpoints.append(report)
                rolling_sum, rolling_count = {}, 0
                interval_episodes = []
                delta_count = 0
                delta_sum.fill(0)
                delta_abs_sum.fill(0)
                native_delta_abs_sum.fill(0)
                delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
        result = {"status": "completed_boundary_at_least_100k", "protocol_sha256": protocol_sha256,
                  "source_sha256": p["source_sha256"], "reward_target": REWARD_TARGET,
                  "decisions": decisions, "updates": updates, "pretrain_updates": cfg["pretrain_updates"],
                  "seed_schedule": base.SEED_SCHEDULE, "episodes": episodes, "action_dim": 3,
                  "checkpoints": checkpoints, "runtime": runtime,
                  "resource_usage": base._cost(clock, started, started_cpu),
                  "training_ledger_sha256": base.digest(ledger), "step_ledger_sha256": base.digest(steps),
                  "reused_train_only": True, "evaluation": None, "resume_supported": False}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
    except BaseException as exc:
        cost = None
        if started is not None and started_cpu is not None:
            try:
                cost = base._cost(clock, started, started_cpu)
            except BaseException:
                pass
        partial = {"event": "partial", "reason": type(exc).__name__,
                   "protocol_sha256": protocol_sha256, "source_sha256": p["source_sha256"],
                   "reward_target": REWARD_TARGET, "runtime": runtime,
                   "episodes": episodes, "decisions": decisions, "updates": updates,
                   "reset_intents": reset_intents,
                   "environment_resets": 0 if reset_intents == 0 else None,
                   "environment_steps_applied": environment_steps_applied,
                   "pending_step": pending_step, "resource_usage": cost,
                   "active_length": replay.active_length if replay is not None else None,
                   "resume_supported": False}
        try:
            if not start_journaled:
                raise OSError("start journal not completed")
            base._journal(ledger, partial)
        except BaseException:
            # If the journal itself fails, independently preserve the unusable run directory.
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(partial, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            directory = os.open(output, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        raise
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="read-only; zero environment resets")
    mode.add_argument("--run", action="store_true", help="new from-scratch TRAIN run, never resume")
    args = parser.parse_args()
    if args.preflight:
        p = preflight(args.protocol, args.protocol_sha256, args.run_dir)
        print(json.dumps({"status": "preflight_only", "protocol_sha256": args.protocol_sha256,
                          "cells": p["cells"], "reward_target": p["reward_target"],
                          "environment_resets": 0}, sort_keys=True))
    else:
        print(json.dumps(run(args.protocol, args.protocol_sha256, args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
