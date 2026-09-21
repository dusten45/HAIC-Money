import argparse
import hashlib
import json
import os
import pickle
import platform
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
# Set process-level limits before importing Torch so every worker uses the same
# CPU-only inference path as the competition submission.
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
if __name__ == "__main__":
    os.environ.update(CPU_ENVIRONMENT)

import numpy as np
import torch
from stable_baselines3 import PPO

from action_smoothing import (
    action_control_fingerprint,
    action_smoothing_fingerprint,
    normalize_action_smoothing,
    normalize_action_control,
    read_embedded_action_smoothing,
)
from action_representation import (
    action_representation_fingerprint,
    normalize_action_representation,
)
from train import build_env, evaluate, find_vecnormalize_path, summarize


TRAIN_SEEDS = frozenset((42, 1337, 2024, 777))
SELECTION_SEEDS = frozenset(range(10001, 10009))
PROTOCOLS = {
    "checkpoint-v1-screen": {
        "track_ids": (1, 2, 3, 4),
        "seeds": (20001, 20002, 20003, 20004),
        "repeats": 1,
    },
    "checkpoint-v1-confirmation": {
        "track_ids": (1, 2, 3, 4),
        "seeds": tuple(range(20101, 20109)),
        "repeats": 2,
    },
    "checkpoint-v1-blind": {
        "track_ids": (7, 8, 9),
        "seeds": tuple(range(20201, 20209)),
        "repeats": 2,
    },
}
ENVIRONMENT_SOURCES = (
    Path("env_wrapper.py"),
    Path("damage.py"),
    Path("core/finish_line.py"),
    Path("core/track_variables.py"),
    Path("core/obstacle_contacts.py"),
    Path("core/vendor/car_racing.py"),
    Path("core/vendor/car_dynamics.py"),
)
RUNTIME_SOURCES = (
    Path("evaluate_policy.py"),
    Path("agent.py"),
    Path("train.py"),
    Path("tracking.py"),
    Path("action_smoothing.py"),
    Path("action_representation.py"),
    *ENVIRONMENT_SOURCES,
)
MAX_PROCESS_RSS_BYTES = 1024 * 1024 * 1024
MAX_INIT_SECONDS = 10.0
MAX_ACTION_SECONDS = 5.0


def parse_int_list(value: str, name: str, minimum: int) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < minimum for item in values):
        raise ValueError(f"{name} values must be at least {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def load_protocol_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("protocol must be a JSON object")
    name = spec.get("name")
    if not isinstance(name, str) or not name or any(
        not (character.isalnum() or character in "-_") for character in name
    ):
        raise ValueError("protocol name must contain only letters, numbers, '-' or '_'")
    if type(spec.get("frame_skip")) is not int or spec["frame_skip"] != 4:
        raise ValueError("protocol frame_skip must be 4")
    if type(spec.get("max_steps")) is not int or spec["max_steps"] <= 0:
        raise ValueError("protocol max_steps must be positive")
    partitions = spec.get("partitions")
    expected_partitions = (
        {"screen"} if spec.get("purpose") == "harness-smoke"
        else {"screen", "confirmation", "blind"}
    )
    if not isinstance(partitions, dict) or set(partitions) != expected_partitions:
        raise ValueError("protocol requires screen, confirmation and blind partitions")
    reserved = set()
    for partition, matrix in partitions.items():
        if not isinstance(matrix, dict):
            raise ValueError(f"invalid {partition} matrix")
        for key, minimum in (("track_ids", 1), ("seeds", 0)):
            values = matrix.get(key)
            if (
                not isinstance(values, list) or not values
                or any(type(value) is not int or value < minimum for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(f"{partition} {key} must be unique bounded integers")
        if any(seed >= 2**32 for seed in matrix["seeds"]):
            raise ValueError(f"{partition} seeds must be uint32 values")
        if type(matrix.get("repeats")) is not int or matrix["repeats"] < 2:
            raise ValueError(f"{partition} requires at least two independent reload repeats")
        if reserved.intersection(matrix["seeds"]):
            raise ValueError("protocol partition seeds must be disjoint, regardless of track ID")
        reserved.update(matrix["seeds"])
    return spec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_archive_member(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read(name)
        except KeyError as error:
            raise ValueError(f"{name} is missing from {path}") from error
    return hashlib.sha256(payload).hexdigest()


def candidate_run_dir(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "checkpoints" else path.parent


def candidate_metadata(path: Path, run_dir: Path | None = None) -> dict:
    explicit_run = run_dir is not None
    if explicit_run and not path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("candidate must belong to the explicit run directory")
    export_metadata = None
    archive_sha256 = sha256_file(path)
    if path.suffix == ".pt":
        from agent import Agent, DRQ_ACTOR_FORMAT

        if path.stat().st_size > 500 * 1024 * 1024:
            raise ValueError("evaluate an exported actor, not a full training checkpoint")
        agent = Agent(model_path=path)
        if agent.format != DRQ_ACTOR_FORMAT:
            raise ValueError(".pt evaluation requires a DrQ exported actor")
        export_metadata = agent.export_metadata
        vecnormalize = None
        policy_sha256 = archive_sha256
    else:
        vecnormalize = find_vecnormalize_path(path, "")
        if vecnormalize is None:
            raise ValueError(f"checkpoint lacks matching VecNormalize state: {path}")
        with vecnormalize.open("rb") as handle:
            normalizer = pickle.load(handle)
        if normalizer.norm_obs:
            raise ValueError(f"observation-normalized checkpoints are not supported: {path}")
        policy_sha256 = sha256_archive_member(path, "policy.pth")
    run_dir = run_dir or candidate_run_dir(path)
    run_config_path = run_dir / "config.json"
    if explicit_run and not run_config_path.is_file():
        raise ValueError("explicit run directory requires config.json provenance")
    action_smoothing = normalize_action_smoothing()
    action_control = normalize_action_control()
    action_representation = normalize_action_representation()
    recorded_config = {}
    run_config_sha256 = None
    action_smoothing_present = False
    action_control_present = False
    action_representation_present = False
    if run_config_path.is_file():
        try:
            config_bytes = run_config_path.read_bytes()
            recorded = json.loads(config_bytes)
            if not isinstance(recorded, dict) or not isinstance(recorded.get("config"), dict):
                raise ValueError("run config must contain a config object")
            run_config_sha256 = hashlib.sha256(config_bytes).hexdigest()
            recorded_config = recorded.get("config", {})
            action_smoothing_present = "action_smoothing" in recorded_config
            action_smoothing = normalize_action_smoothing(
                recorded_config.get("action_smoothing")
            )
            action_control_present = "action_control" in recorded_config
            action_control = normalize_action_control(recorded_config.get("action_control"))
            action_representation_present = "action_representation" in recorded_config
            action_representation = normalize_action_representation(
                recorded_config.get("action_representation")
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid run action smoothing config: {run_config_path}") from error
    if export_metadata is not None:
        if (
            action_smoothing != normalize_action_smoothing()
            or action_control != normalize_action_control()
            or action_representation != normalize_action_representation()
            or recorded_config.get("frame_skip", 4) != 4
        ):
            raise ValueError("DrQ run config conflicts with its embedded frozen contract")
    if explicit_run and (
        type(recorded_config.get("max_steps")) is not int or recorded_config["max_steps"] <= 0
        or type(recorded_config.get("frame_skip")) is not int or recorded_config["frame_skip"] <= 0
    ):
        raise ValueError("explicit run config requires max_steps and frame_skip provenance")
    return {
        "algorithm": "drq-v2" if export_metadata is not None else "ppo",
        "export_metadata": export_metadata,
        "export_spec_fingerprints": {
            key: hashlib.sha256(json.dumps(
                export_metadata[key], sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            for key in ("observation_spec", "action_spec")
        } if export_metadata is not None else None,
        "source_path": str(path),
        "archive_sha256": archive_sha256,
        "policy_sha256": policy_sha256,
        "vecnormalize_path": str(vecnormalize) if vecnormalize is not None else None,
        "vecnormalize_sha256": sha256_file(vecnormalize) if vecnormalize is not None else None,
        "norm_obs": False,
        "run_config_path": str(run_config_path) if run_config_path.is_file() else None,
        "run_config_sha256": run_config_sha256,
        "action_smoothing": action_smoothing,
        "action_smoothing_present": action_smoothing_present,
        "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
        "action_control": action_control,
        "action_control_present": action_control_present,
        "action_control_fingerprint": action_control_fingerprint(action_control),
        "action_representation": action_representation,
        "action_representation_present": action_representation_present,
        "action_representation_fingerprint": action_representation_fingerprint(
            action_representation
        ),
        "run_frame_skip": 4 if export_metadata is not None else recorded_config.get("frame_skip"),
        "run_max_steps": recorded_config.get("max_steps"),
    }


def resolve_action_smoothing(model_path: Path, external_config=None):
    """Prefer embedded provenance and reject disagreement with external config."""

    embedded = read_embedded_action_smoothing(model_path)
    if external_config is None:
        return embedded or normalize_action_smoothing()
    external = normalize_action_smoothing(external_config)
    if (
        embedded is not None
        and action_smoothing_fingerprint(embedded)
        != action_smoothing_fingerprint(external)
    ):
        raise ValueError(
            f"checkpoint and external action smoothing configs do not match: {model_path}"
        )
    return embedded or external


def discover_candidates(paths, run_dir: Path | None = None) -> list[dict]:
    unique = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = candidate_metadata(path, run_dir if path.suffix == ".pt" else None)
        policy_sha256 = metadata["policy_sha256"]
        dedup_key = (
            policy_sha256,
            metadata["action_smoothing_fingerprint"],
            metadata["action_control_fingerprint"],
            metadata["action_representation_fingerprint"],
        )
        existing = unique.get(dedup_key)
        if existing is None:
            metadata["candidate_id"] = (
                f"{policy_sha256[:16]}-{metadata['action_smoothing_fingerprint'][:8]}-"
                f"{metadata['action_control_fingerprint'][:8]}"
                f"-{metadata['action_representation_fingerprint'][:8]}"
            )
            metadata["aliases"] = [str(path)]
            unique[dedup_key] = metadata
        else:
            existing["aliases"].append(str(path))
    return sorted(unique.values(), key=lambda candidate: candidate["source_path"])


def checkpoint_paths(run_dir: Path, legacy_model: Path | None) -> list[Path]:
    paths = sorted((run_dir / "checkpoints").glob("*.zip"))
    paths.extend(path for path in (run_dir / "model.zip", run_dir / "best_model.zip") if path.is_file())
    if legacy_model is not None and legacy_model.is_file():
        paths.append(legacy_model)
    if not paths:
        raise ValueError(f"no SB3 checkpoints found under {run_dir}")
    return paths


def evaluate_model(
    model_path: Path,
    track_ids,
    seeds,
    max_steps,
    frame_skip,
    action_smoothing=None,
    action_control=None,
    action_representation=None,
):
    """Ad-hoc evaluator retained for small, explicitly non-protocol comparisons."""
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    action_smoothing = resolve_action_smoothing(model_path, action_smoothing)

    model = PPO.load(str(model_path), device="cpu")
    episodes = []
    by_track = {}
    for track_id in track_ids:
        track_episodes = []
        for seed in seeds:
            track_episodes.extend(
                evaluate(
                    model,
                    track_id,
                    seed,
                    max_steps,
                    frame_skip,
                    episodes=1,
                    action_smoothing=action_smoothing,
                    action_control=action_control,
                    action_representation=action_representation,
                )
            )
        episodes.extend(track_episodes)
        by_track[str(track_id)] = summarize(track_episodes)

    return {
        "model_path": str(model_path),
        "action_smoothing": normalize_action_smoothing(action_smoothing),
        "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
        "action_control": normalize_action_control(action_control),
        "action_control_fingerprint": action_control_fingerprint(action_control),
        "action_representation": normalize_action_representation(action_representation),
        "action_representation_fingerprint": action_representation_fingerprint(
            action_representation
        ),
        "summary": summarize(episodes),
        "by_track": by_track,
        "episodes": episodes,
    }


def terminal_class(finished, terminated, truncated, info):
    if finished:
        return "finished"
    if info.get("retire_reason"):
        return str(info["retire_reason"])
    if terminated:
        return "out_of_bounds"
    if truncated:
        return "max_steps"
    return "unknown"


def action_trace_digest(actions) -> str:
    digest = hashlib.sha256()
    for action in actions:
        digest.update(np.asarray(action, dtype=np.float32).tobytes())
    return digest.hexdigest()


def timed_policy_call(function, *arguments, seconds=MAX_ACTION_SECONDS):
    def timeout(_signum, _frame):
        raise TimeoutError(f"policy call exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.perf_counter()
    try:
        result = function(*arguments)
        elapsed = time.perf_counter() - started
        if elapsed > seconds:
            raise TimeoutError(f"policy call exceeded {seconds} seconds")
        return result, elapsed
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def peak_rss_bytes():
    # ru_maxrss can inherit the launcher's pre-exec high-water mark on Linux.
    if sys.platform == "linux":
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("cannot read process-local peak RSS")
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * multiplier)


def evaluate_cell(
    model_path: Path,
    track_id: int,
    seed: int,
    max_steps: int,
    frame_skip: int,
    action_smoothing=None,
    action_control=None,
    action_representation=None,
    worker_started=None,
) -> dict:
    torch.set_num_threads(1)
    load_started = time.perf_counter()
    is_drq = model_path.suffix == ".pt"
    if is_drq:
        from agent import Agent, DRQ_ACTOR_FORMAT

        model, _ = timed_policy_call(Agent, model_path, seconds=MAX_INIT_SECONDS)
        if model.format != DRQ_ACTOR_FORMAT:
            raise ValueError("expected a DrQ exported actor")
        if frame_skip != model.export_metadata["action_spec"]["frame_skip"]:
            raise ValueError("DrQ frame_skip does not match exported action spec")
        if (
            normalize_action_smoothing(action_smoothing) != normalize_action_smoothing()
            or normalize_action_control(action_control) != normalize_action_control()
            or normalize_action_representation(action_representation) != normalize_action_representation()
        ):
            raise ValueError("DrQ evaluation cannot override the exported action contract")
    else:
        model = PPO.load(str(model_path), device="cpu")
    load_seconds = time.perf_counter() - load_started
    init_seconds = time.perf_counter() - (worker_started or load_started)
    if is_drq and init_seconds > MAX_INIT_SECONDS:
        raise TimeoutError("CPU worker import and actor construction exceeded 10 seconds")
    actor_peak_rss = peak_rss_bytes()
    action_smoothing = resolve_action_smoothing(model_path, action_smoothing)
    episode_started = time.perf_counter()
    env = build_env(
        track_id,
        seed,
        max_steps,
        frame_skip,
        reward_shaping=False,
        action_smoothing=action_smoothing,
        action_control=action_control,
        action_representation=action_representation,
    )
    action_seconds = []
    actions = []
    collision_actions = 0
    try:
        reset_started = time.perf_counter()
        observation, _ = env.reset()
        reset_seconds = time.perf_counter() - reset_started
        agent_reset_seconds = 0.0
        if is_drq:
            _, agent_reset_seconds = timed_policy_call(model.reset, observation)
        start_time_s = env.unwrapped.t
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        steps = 0
        while not (terminated or truncated):
            if peak_rss_bytes() > MAX_PROCESS_RSS_BYTES and is_drq:
                raise MemoryError("CPU evaluation worker exceeded 1,024 MiB peak RSS")
            if is_drq:
                action, elapsed = timed_policy_call(model.act, observation)
            else:
                action_started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                elapsed = time.perf_counter() - action_started
            action_seconds.append(elapsed)
            representation = normalize_action_representation(action_representation)
            action = np.asarray(action)
            if representation["method"] == "continuous_box":
                action = action.astype(np.float32)
                valid_action = action.shape == (3,) and np.isfinite(action).all()
                if valid_action:
                    action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            else:
                valid_action = (
                    action.shape == (2,)
                    and np.issubdtype(action.dtype, np.integer)
                    and 0 <= int(action[0]) < representation["nvec"][0]
                    and 0 <= int(action[1]) < representation["nvec"][1]
                )
                action = action.astype(np.int64)
            if not valid_action:
                raise ValueError(f"invalid policy action: {action!r}")
            observation, reward, terminated, truncated, info = env.step(action)
            executed_action = getattr(env, "last_action", None)
            if executed_action is None:
                executed_action = action
            actions.append(np.asarray(executed_action, dtype=np.float32).copy())
            total_reward += reward
            steps += 1
            collision_actions += int(bool(info.get("collision", False)))

        finish_time_s = info.get("finish_time_s")
        finished = bool(info.get("finished", False))
        lap_time_ms = (
            round((finish_time_s - start_time_s) * 1000)
            if finished and finish_time_s is not None else None
        )
        peak_rss = peak_rss_bytes()
        return {
            "status": "resource_limit" if is_drq and peak_rss > MAX_PROCESS_RSS_BYTES else "ok",
            "track_id": track_id,
            "seed": seed,
            "steps": steps,
            "reward": total_reward,
            "progress": float(info.get("progress", 0.0)),
            "finished": finished,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "retire_reason": info.get("retire_reason"),
            "finish_qualified": bool(info.get("finish_qualified", False)),
            "finish_time_s": finish_time_s,
            "lap_time_ms": lap_time_ms,
            "damage": float(info.get("damage", 0.0)),
            "termination_class": terminal_class(finished, terminated, truncated, info),
            "collision_actions": collision_actions,
            "steering_delta_abs_mean": (
                float(np.mean(np.abs(np.diff(np.asarray(actions)[:, 0]))))
                if len(actions) > 1
                else 0.0
            ),
            "action_smoothing": action_smoothing,
            "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
            "action_control": normalize_action_control(action_control),
            "action_control_fingerprint": action_control_fingerprint(action_control),
            "action_representation": representation,
            "action_representation_fingerprint": action_representation_fingerprint(
                representation
            ),
            "raw_time_s": float(env.unwrapped.t - start_time_s),
            "action_trace_sha256": action_trace_digest(actions),
            "actions": [action.tolist() for action in actions],
            "model_load_seconds": load_seconds,
            "process_initialization_seconds": init_seconds,
            "reset_seconds": reset_seconds,
            "agent_reset_seconds": agent_reset_seconds,
            "first_action_seconds": action_seconds[0] if action_seconds else 0.0,
            "mean_action_seconds": float(np.mean(action_seconds)) if action_seconds else 0.0,
            "p95_action_seconds": float(np.percentile(action_seconds, 95)) if action_seconds else 0.0,
            "max_action_seconds": float(np.max(action_seconds)) if action_seconds else 0.0,
            "actor_load_peak_rss_bytes": actor_peak_rss,
            "peak_rss_bytes": peak_rss,
            "rss_scope": "whole isolated evaluator, including environment and harness imports",
            "episode_wall_seconds": time.perf_counter() - episode_started,
        }
    finally:
        env.close()


def worker_result(args) -> int:
    runtime = runtime_metadata(Path(__file__).resolve().parent)
    try:
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        runtime["torch_threads"] = torch.get_num_threads()
        runtime["torch_interop_threads"] = torch.get_num_interop_threads()
        if args.model[0].suffix == ".pt":
            validate_cpu_runtime(runtime)
        model_sha256 = sha256_file(args.model[0])
        if args.worker_model_sha256 and model_sha256 != args.worker_model_sha256:
            raise ValueError("worker actor hash differs from immutable candidate")
        action_smoothing = json.loads(args.worker_action_smoothing)
        action_control = json.loads(args.worker_action_control)
        action_representation = json.loads(args.worker_action_representation)
        result = evaluate_cell(
            args.model[0], args.worker_track_id, args.worker_seed,
            args.max_steps, args.frame_skip,
            action_smoothing=action_smoothing,
            action_control=action_control,
            action_representation=action_representation,
            worker_started=PROCESS_STARTED,
        )
        result["loaded_archive_sha256"] = model_sha256
    except Exception as error:
        result = {
            "status": "exception",
            "error_type": type(error).__name__,
            "error": str(error),
            "track_id": args.worker_track_id,
            "seed": args.worker_seed,
        }
    result["runtime"] = runtime
    result["worker_pid"] = os.getpid()
    result["peak_rss_bytes"] = peak_rss_bytes()
    if args.model[0].suffix == ".pt" and result["status"] == "ok" and result["peak_rss_bytes"] > MAX_PROCESS_RSS_BYTES:
        result["status"] = "resource_limit"
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_worker_output(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
            if isinstance(result, dict) and result.get("status") in {
                "ok", "exception", "runtime_timeout", "resource_limit",
            }:
                if result["status"] == "ok" and not all(key in result for key in (
                    "steps", "finished", "progress", "reward", "damage",
                    "lap_time_ms", "termination_class", "action_trace_sha256",
                )):
                    continue
                return result
        except json.JSONDecodeError:
            continue
    return {"status": "exception", "error_type": "WorkerOutputError", "error": stdout[-1000:]}


def run_isolated_cell(
    candidate: dict,
    track_id: int,
    seed: int,
    repeat: int,
    args,
    worker_script: Path,
) -> dict:
    python = getattr(args, "python", None) or sys.executable
    if not Path(python).is_absolute() and "/" in python:
        python = str(Path.cwd() / python)
    command = [
        python, "-B", str(worker_script.resolve()), "--worker",
        "--model", candidate["_worker_model_path"],
        "--worker-model-sha256", candidate["archive_sha256"],
        "--worker-track-id", str(track_id),
        "--worker-seed", str(seed),
        "--max-steps", str(args.max_steps),
        "--frame-skip", str(args.frame_skip),
        "--worker-action-smoothing", json.dumps(
            candidate["action_smoothing"] if candidate["action_smoothing_present"] else None,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--worker-action-control", json.dumps(
            candidate["action_control"] if candidate["action_control_present"] else None,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "--worker-action-representation", json.dumps(
            candidate["action_representation"]
            if candidate["action_representation_present"]
            else None,
            sort_keys=True,
            separators=(",", ":"),
        ),
    ]
    started = time.perf_counter()
    environment = dict(os.environ, **CPU_ENVIRONMENT)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
            env=environment,
            cwd=worker_script.resolve().parent,
        )
        result = parse_worker_output(completed.stdout)
        if completed.returncode != 0:
            result = {
                "status": "exception",
                "error_type": "WorkerExitError",
                "error": completed.stderr[-2000:],
            }
        elif result["status"] == "exception" and result.get("error_type") == "WorkerOutputError":
            result["error"] = completed.stderr[-2000:] or result["error"]
    except subprocess.TimeoutExpired:
        result = {"status": "runtime_timeout", "error": f"exceeded {args.timeout_seconds} seconds"}
    except OSError as error:
        result = {"status": "exception", "error_type": type(error).__name__, "error": str(error)}
    result.update(
        candidate_id=candidate["candidate_id"],
        source_path=candidate["source_path"],
        track_id=track_id,
        seed=seed,
        repeat=repeat,
        parent_wall_seconds=time.perf_counter() - started,
    )
    return result


def result_signature(result: dict):
    return tuple(
        result.get(key) for key in (
            "status", "finished", "termination_class", "steps", "lap_time_ms",
            "progress", "damage", "collision_actions", "action_trace_sha256",
            "steering_delta_abs_mean", "action_smoothing_fingerprint",
            "action_control_fingerprint", "action_representation_fingerprint",
            "terminated", "truncated", "retire_reason", "loaded_archive_sha256",
        )
    )


def determinism_audit(episodes) -> tuple[list[dict], set[str], set[str]]:
    groups = {}
    for episode in episodes:
        groups.setdefault((episode["candidate_id"], episode["track_id"], episode["seed"]), []).append(episode)
    audits = []
    non_reproducible = set()
    unaudited = set()
    for (candidate_id, track_id, seed), repeats in sorted(groups.items()):
        repeats.sort(key=lambda result: result["repeat"])
        baseline = result_signature(repeats[0])
        repeat_ids = [result["repeat"] for result in repeats]
        audited = len(repeats) >= 2 and repeat_ids == list(range(len(repeats)))
        matches = all(result_signature(result) == baseline for result in repeats[1:]) if audited else None
        if audited and not matches:
            non_reproducible.add(candidate_id)
        if not audited:
            unaudited.add(candidate_id)
        audits.append({
            "candidate_id": candidate_id,
            "track_id": track_id,
            "seed": seed,
            "repeats": len(repeats),
            "audited": audited,
            "matches_canonical": matches,
        })
    return audits, non_reproducible, unaudited


def candidate_summary(candidate: dict, episodes, non_reproducible: set[str], unaudited: set[str]) -> dict:
    canonical = [episode for episode in episodes if episode["repeat"] == 0]
    successful = [episode for episode in canonical if episode["status"] == "ok"]
    grouped = {}
    for track_id in sorted({episode["track_id"] for episode in canonical}):
        grouped[str(track_id)] = summarize([
            {
                "finished": episode.get("finished", False),
                "progress": episode.get("progress", 0.0),
                "reward": episode.get("reward", 0.0),
                "steps": episode.get("steps", 0),
                "lap_time_ms": episode.get("lap_time_ms"),
                "damage": episode.get("damage", 0.0),
                "retire_reason": episode.get("termination_class"),
                "steering_delta_abs_mean": episode.get("steering_delta_abs_mean", 0.0),
            }
            for episode in canonical if episode["track_id"] == track_id and episode["status"] == "ok"
        ])
    limits = {
        "process_initialization_seconds": MAX_INIT_SECONDS,
        "agent_reset_seconds": MAX_ACTION_SECONDS,
        "max_action_seconds": MAX_ACTION_SECONDS,
        "peak_rss_bytes": MAX_PROCESS_RSS_BYTES,
    }
    failures = [
        episode for episode in episodes if episode["status"] != "ok" or (
            candidate.get("algorithm") == "drq-v2" and any(
                type(episode.get(key)) not in (int, float)
                or not np.isfinite(episode[key]) or not 0 <= episode[key] <= limit
                for key, limit in limits.items()
            )
        )
    ]
    complete = (
        bool(canonical)
        and len(canonical) == candidate.get("expected_cells", len(canonical))
        and len(episodes) == candidate.get("expected_results", len(episodes))
    )
    audited = complete and candidate["candidate_id"] not in unaudited
    reload_matches = audited and not failures and candidate["candidate_id"] not in non_reproducible
    return {
        **{key: value for key, value in candidate.items() if not key.startswith("_")},
        "eligible": (
            complete and not failures
            and not candidate.get("is_comparator", False)
            and candidate["candidate_id"] not in non_reproducible
            and candidate["candidate_id"] not in unaudited
        ),
        "determinism_audited": audited,
        "cpu_reload_matches": reload_matches,
        "canonical_episodes": len(canonical),
        "operational_failures": len(failures),
        "resources": {
            key: max((episode[key] for episode in episodes
                      if type(episode.get(key)) in (int, float) and np.isfinite(episode[key])), default=0.0)
            for key in limits
        },
        "summary": summarize([
            {
                "finished": episode.get("finished", False),
                "progress": episode.get("progress", 0.0),
                "reward": episode.get("reward", 0.0),
                "steps": episode.get("steps", 0),
                "lap_time_ms": episode.get("lap_time_ms"),
                "damage": episode.get("damage", 0.0),
                "retire_reason": episode.get("termination_class"),
                "steering_delta_abs_mean": episode.get("steering_delta_abs_mean", 0.0),
            }
            for episode in successful
        ]),
        "by_track": grouped,
    }


def ranking_key(summary: dict) -> tuple:
    metrics = summary["summary"]
    # Results without a finish have no lap metric and rank below any finisher.
    lap_time = metrics["avg_lap_time_ms"]
    return (
        summary["eligible"],
        metrics["finish_rate"],
        metrics["avg_progress"],
        -lap_time if lap_time is not None else float("-inf"),
        -summary["operational_failures"],
    )


def runtime_metadata(source_root: Path) -> dict:
    packages = ("gymnasium", "numpy", "opencv-python", "stable-baselines3", "torch")
    return {
        "python": sys.version,
        "python_version": list(sys.version_info[:2]),
        "sys_platform": sys.platform,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cpu_environment": CPU_ENVIRONMENT,
        "runtime_sources": {
            str(path): sha256_file(source_root / path)
            for path in RUNTIME_SOURCES if (source_root / path).is_file()
        },
    }


def validate_cpu_runtime(runtime: dict) -> None:
    packages = runtime["packages"]
    if (
        runtime["python_version"] != [3, 11] or runtime["sys_platform"] != "linux"
        or packages["torch"].split("+")[0] != "2.1.0"
        or packages["numpy"] != "1.26.0" or packages["gymnasium"] != "0.29.1"
        or packages["opencv-python"] != "4.8.1.78"
        or runtime["torch_cuda"] is not None or runtime["cuda_available"]
        or runtime["torch_threads"] != 1 or runtime["torch_interop_threads"] != 1
    ):
        raise ValueError("DrQ selection requires the pinned Python 3.11 Linux Torch 2.1 CPU runtime")


def git_metadata() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_protocol_report(path: Path, protocol: str, ranked) -> None:
    lines = ["# Checkpoint Evaluation", "", f"Protocol: `{protocol}`", ""]
    if any(candidate.get("diagnostic_only", False) for candidate in ranked):
        lines.extend([
            "**DIAGNOSTIC ONLY: NON-PROMOTING. diagnostic_only=true.**",
            "A failed screen remains failed even if this diagnostic finishes; this result cannot authorize blind evaluation or promotion.",
            "",
        ])
    lines.extend(["## Ranking", ""])
    for rank, candidate in enumerate(ranked, start=1):
        summary = candidate["summary"]
        lines.append(
            f"{rank}. `{candidate['candidate_id']}` eligible={candidate['eligible']} "
            f"finish_rate={summary['finish_rate']:.3f} progress={summary['avg_progress']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def protocol_output_dir(root: Path, protocol: str) -> tuple[Path, Path]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final_dir = root / f"{stamp}_{protocol}"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = Path(tempfile.mkdtemp(prefix=f".pending-{stamp}-", dir=root))
    return temporary, final_dir


def snapshot_runtime(temporary_dir: Path) -> Path:
    source_root = Path(__file__).resolve().parent
    runtime_dir = temporary_dir / "runtime"
    runtime_dir.mkdir()
    for source in (
        Path("evaluate_policy.py"),
        Path("agent.py"),
        Path("train.py"),
        Path("tracking.py"),
        Path("action_smoothing.py"),
        Path("action_representation.py"),
        Path("env_wrapper.py"),
        Path("damage.py"),
    ):
        shutil.copy2(source_root / source, runtime_dir / source.name)
    shutil.copytree(source_root / "core", runtime_dir / "core", ignore=shutil.ignore_patterns("__pycache__"))
    return runtime_dir / "evaluate_policy.py"


def snapshot_candidates(temporary_dir: Path, candidates) -> None:
    candidate_dir = temporary_dir / "candidates"
    candidate_dir.mkdir()
    for candidate in candidates:
        suffix = ".pt" if candidate.get("algorithm") == "drq-v2" else ".zip"
        archive = candidate_dir / f"{candidate['candidate_id']}{suffix}"
        shutil.copy2(candidate["source_path"], archive)
        snapshot_sha256 = sha256_file(archive)
        if snapshot_sha256 != candidate["archive_sha256"]:
            raise RuntimeError(f"checkpoint changed while snapshotting: {candidate['source_path']}")
        candidate["evaluation_archive_path"] = archive.relative_to(temporary_dir).as_posix()
        candidate["evaluation_archive_sha256"] = snapshot_sha256
        if candidate.get("vecnormalize_path"):
            vecnormalize_source = Path(candidate["vecnormalize_path"])
            vecnormalize_archive = candidate_dir / f"{candidate['candidate_id']}.vecnormalize.pkl"
            shutil.copy2(vecnormalize_source, vecnormalize_archive)
            vecnormalize_snapshot_sha256 = sha256_file(vecnormalize_archive)
            if vecnormalize_snapshot_sha256 != candidate["vecnormalize_sha256"]:
                raise RuntimeError(
                    f"VecNormalize changed while snapshotting: {vecnormalize_source}"
                )
            candidate["evaluation_vecnormalize_path"] = (
                vecnormalize_archive.relative_to(temporary_dir).as_posix()
            )
            candidate["evaluation_vecnormalize_sha256"] = vecnormalize_snapshot_sha256
        if candidate.get("run_config_path"):
            config_snapshot = candidate_dir / f"{candidate['candidate_id']}.config.json"
            shutil.copy2(candidate["run_config_path"], config_snapshot)
            if sha256_file(config_snapshot) != candidate["run_config_sha256"]:
                raise RuntimeError("run config changed while snapshotting")
            candidate["evaluation_run_config_path"] = str(config_snapshot.relative_to(temporary_dir))
        candidate["_worker_model_path"] = str(archive.resolve())


def serializable_candidates(candidates) -> list[dict]:
    return [
        {key: value for key, value in candidate.items() if not key.startswith("_")}
        for candidate in candidates
    ]


def previous_evaluation_metadata(
    path: Path, partition: str, protocol_sha256: str, candidate: dict, *, diagnostic_confirmation=False,
) -> dict:
    if diagnostic_confirmation and partition != "confirmation":
        raise ValueError("diagnostic confirmation cannot authorize another partition")
    pointer_bytes = path.read_bytes()
    pointer = json.loads(pointer_bytes)
    previous_dir = Path(pointer["evaluation_dir"])
    if not previous_dir.is_absolute():
        raise ValueError("previous evaluation directory must be absolute")
    summary_path = previous_dir / "summary.json"
    manifest_path = previous_dir / "manifest.json"
    summary_bytes = summary_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    summary = json.loads(summary_bytes)
    manifest = json.loads(manifest_bytes)
    expected_partition = "screen" if partition == "confirmation" else "confirmation"
    if (
        pointer.get("ranked") != summary or not isinstance(summary, list) or len(summary) != 1
        or any(record.get("protocol_sha256") != protocol_sha256
               or record.get("partition") != expected_partition for record in (pointer, manifest))
    ):
        raise ValueError("previous evaluation must match the immutable preceding partition and protocol")
    selected = summary[0]
    if any(record.get("diagnostic_only", False) for record in (pointer, manifest, selected)):
        raise ValueError("diagnostic-only evaluations cannot authorize blind evaluation or promotion")
    finish_rate = selected.get("summary", {}).get("finish_rate")
    if (
        selected.get("archive_sha256") != candidate["archive_sha256"]
        or selected.get("eligible") is not True or selected.get("determinism_audited") is not True
        or selected.get("cpu_reload_matches") is not True
        or selected.get("operational_failures") != 0
        or type(finish_rate) not in (int, float) or not 0 <= finish_rate <= 1.0
        or (not diagnostic_confirmation and finish_rate == 0.0)
    ):
        required = "operational success" if diagnostic_confirmation else "nonzero completion"
        raise ValueError(f"previous evaluation must establish repeated {required} for this exact actor")
    previous_actor = (previous_dir / selected["evaluation_archive_path"]).resolve()
    if not previous_actor.is_relative_to(previous_dir.resolve()) or sha256_file(previous_actor) != candidate["archive_sha256"]:
        raise ValueError("previous immutable actor does not match the selected actor")
    return {
        "evaluation_dir": str(previous_dir),
        "partition": expected_partition,
        "actor_sha256": candidate["archive_sha256"],
        "files": {
            name: {"source_path": str(source.resolve()), "sha256": hashlib.sha256(contents).hexdigest()}
            for name, source, contents in (
                ("previous_evaluation.json", path, pointer_bytes),
                ("previous_summary.json", summary_path, summary_bytes),
                ("previous_manifest.json", manifest_path, manifest_bytes),
            )
        },
    }


def run_checkpoint_protocol(args, protocol_name: str) -> Path:
    diagnostic_only = bool(getattr(args, "diagnostic_confirmation", False))
    if diagnostic_only and (
        getattr(args, "protocol_file", None) is None or args.partition != "confirmation"
    ):
        raise ValueError("--diagnostic-confirmation requires custom partition=confirmation")
    spec = None
    protocol_sha256 = None
    if getattr(args, "protocol_file", None) is not None:
        protocol_sha256 = sha256_file(args.protocol_file)
        spec = load_protocol_spec(args.protocol_file)
        if sha256_file(args.protocol_file) != protocol_sha256:
            raise RuntimeError("protocol changed while reading")
        if args.frame_skip != spec["frame_skip"] or args.max_steps != spec["max_steps"]:
            raise ValueError("requested horizon/frame_skip does not match frozen protocol")
        if args.partition not in spec["partitions"]:
            raise ValueError("requested partition is not present in this protocol")
        if args.partition != "screen" and (len(args.model) != 1 or args.legacy_model is not None):
            raise ValueError("confirmation and blind require one already-selected actor")
        protocol = spec["partitions"][args.partition]
        protocol_name = f"{spec['name']}-{args.partition}"
    else:
        protocol = PROTOCOLS[protocol_name]
    output = getattr(args, "output", None)
    if output is not None and output.exists():
        raise FileExistsError(output)
    candidate_paths = list(args.model or checkpoint_paths(args.run_dir, None))
    if args.legacy_model is not None:
        candidate_paths.append(args.legacy_model)
    candidates = discover_candidates(candidate_paths, args.run_dir if spec is not None else None)
    if spec is None and any(candidate["algorithm"] == "drq-v2" for candidate in candidates):
        raise ValueError("DrQ actors require an explicit --protocol-file")
    for candidate in candidates:
        candidate["diagnostic_only"] = diagnostic_only
        candidate["expected_cells"] = len(protocol["track_ids"]) * len(protocol["seeds"])
        candidate["expected_results"] = candidate["expected_cells"] * protocol["repeats"]
        if (
            candidate.get("run_frame_skip") is not None
            and candidate["run_frame_skip"] != args.frame_skip
        ):
            raise ValueError(
                f"candidate frame_skip {candidate['run_frame_skip']} does not match "
                f"protocol frame_skip {args.frame_skip}: {candidate['source_path']}"
            )
        if (
            candidate.get("run_max_steps") is not None
            and candidate["run_max_steps"] != args.max_steps
        ):
            raise ValueError(
                f"candidate max_steps {candidate['run_max_steps']} does not match "
                f"protocol max_steps {args.max_steps}: {candidate['source_path']}"
            )
        candidate["is_comparator"] = (
            args.legacy_model is not None
            and str(args.legacy_model) in candidate["aliases"]
        )
    previous_evaluation = None
    if spec is not None and args.partition != "screen":
        if getattr(args, "previous_evaluation", None) is None:
            raise ValueError("confirmation and blind require --previous-evaluation")
        previous_evaluation = previous_evaluation_metadata(
            args.previous_evaluation, args.partition, protocol_sha256, candidates[0],
            diagnostic_confirmation=diagnostic_only,
        )
    temporary_dir, final_dir = protocol_output_dir(args.evaluations_dir, protocol_name)
    started_at = datetime.now(timezone.utc)
    try:
        worker_script = snapshot_runtime(temporary_dir)
        snapshot_candidates(temporary_dir, candidates)
        if spec is not None:
            spec_snapshot = temporary_dir / "protocol_spec.json"
            shutil.copy2(args.protocol_file, spec_snapshot)
            if sha256_file(spec_snapshot) != protocol_sha256:
                raise RuntimeError("protocol changed while snapshotting")
        if previous_evaluation is not None:
            for filename, provenance in previous_evaluation["files"].items():
                receipt = temporary_dir / filename
                shutil.copy2(provenance["source_path"], receipt)
                if sha256_file(receipt) != provenance["sha256"]:
                    raise RuntimeError("previous evaluation changed while snapshotting")
                provenance["snapshot_path"] = filename
        smoothing_fingerprints = sorted({
            candidate["action_smoothing_fingerprint"] for candidate in candidates
        })
        manifest = {
            "schema_version": 2,
            "protocol": protocol_name,
            "protocol_sha256": protocol_sha256,
            "partition": getattr(args, "partition", None),
            "diagnostic_only": diagnostic_only,
            "run_dir": str(args.run_dir.resolve()),
            "previous_evaluation": previous_evaluation,
            "started_at_utc": started_at.isoformat().replace("+00:00", "Z"),
            "cell_matrix": protocol,
            "frame_skip": args.frame_skip,
            "max_steps": args.max_steps,
            "action_smoothing": (
                candidates[0]["action_smoothing"] if len(smoothing_fingerprints) == 1 else None
            ),
            "action_smoothing_fingerprints": smoothing_fingerprints,
            "action_smoothing_by_candidate": {
                candidate["candidate_id"]: candidate["action_smoothing"]
                for candidate in candidates
            },
            "timeout_seconds": args.timeout_seconds,
            "workers": getattr(args, "workers", 1),
            "limits": {
                "initialization_seconds": MAX_INIT_SECONDS,
                "reset_seconds": MAX_ACTION_SECONDS,
                "action_seconds": MAX_ACTION_SECONDS,
                "peak_rss_bytes": MAX_PROCESS_RSS_BYTES,
            },
            "git": git_metadata(),
            "coordinator_runtime": runtime_metadata(worker_script.parent),
        }
        cells = [
            (candidate, track_id, seed, repeat)
            for candidate in candidates
            for track_id in protocol["track_ids"]
            for seed in protocol["seeds"]
            for repeat in range(protocol["repeats"])
        ]
        with ThreadPoolExecutor(max_workers=getattr(args, "workers", 1)) as pool:
            episodes = list(pool.map(
                lambda cell: run_isolated_cell(*cell, args=args, worker_script=worker_script), cells,
            ))
        audits, non_reproducible, unaudited = determinism_audit(episodes)
        per_candidate = [
            candidate_summary(
                candidate,
                [episode for episode in episodes if episode["candidate_id"] == candidate["candidate_id"]],
                non_reproducible,
                unaudited,
            )
            for candidate in candidates
        ]
        ranked = sorted(per_candidate, key=ranking_key, reverse=True)
        worker_runtimes = {
            json.dumps(episode["runtime"], sort_keys=True): episode["runtime"]
            for episode in episodes if "runtime" in episode
        }
        manifest["worker_runtime"] = list(worker_runtimes.values())
        manifest["runtime"] = next(iter(worker_runtimes.values()), None)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        write_json(temporary_dir / "manifest.json", manifest)
        write_json(temporary_dir / "protocol.json", protocol)
        write_json(temporary_dir / "candidates.json", serializable_candidates(candidates))
        with (temporary_dir / "episodes.jsonl").open("w") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, sort_keys=True) + "\n")
        write_json(temporary_dir / "summary.json", ranked)
        write_json(temporary_dir / "determinism.json", audits)
        write_protocol_report(temporary_dir / "report.md", protocol_name, ranked)
        temporary_dir.replace(final_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump({
                "evaluation_dir": str(final_dir),
                "protocol_name": protocol_name,
                "protocol_sha256": protocol_sha256,
                "partition": getattr(args, "partition", None),
                "diagnostic_only": diagnostic_only,
                "ranked": ranked,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return final_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic PPO or exported DrQ policies on fixed HAIC tracks"
    )
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--track-ids", default="1")
    parser.add_argument("--seeds")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--output", type=Path, help="ad-hoc report or immutable protocol result pointer")
    parser.add_argument("--protocol", choices=("ad-hoc", *PROTOCOLS), default="ad-hoc")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--protocol-file", type=Path, help="frozen screen/confirmation/blind JSON spec")
    parser.add_argument("--partition", choices=("screen", "confirmation", "blind"))
    parser.add_argument("--previous-evaluation", type=Path, help="immutable preceding partition result pointer")
    parser.add_argument(
        "--diagnostic-confirmation", action="store_true",
        help="non-promoting confirmation after an operationally valid zero-finish screen",
    )
    parser.add_argument("--python", help="CPU worker interpreter; default is the current interpreter")
    parser.add_argument("--check-runtime", action="store_true", help="validate this interpreter before training")
    parser.add_argument(
        "--legacy-model", type=Path,
        help="optional historical comparator; omitted from normal screens by default",
    )
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1, help="concurrent isolated CPU cells")
    parser.add_argument(
        "--action-smoothing-config",
        help="JSON smoothing config for ad-hoc evaluation; protocol mode reads run provenance",
    )
    parser.add_argument(
        "--action-control-config",
        help="JSON action-control config for ad-hoc evaluation; protocol mode reads run provenance",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-track-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model-sha256", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-action-smoothing", default="{}", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-action-control", default="{}", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-action-representation", default="{}", help=argparse.SUPPRESS
    )
    return parser.parse_args()


def validate_protocol_request(args) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    custom = args.protocol_file is not None
    if args.diagnostic_confirmation and (not custom or args.partition != "confirmation"):
        raise ValueError("--diagnostic-confirmation requires custom partition=confirmation")
    if (args.protocol != "ad-hoc" or custom) and args.run_dir is None:
        raise ValueError("--run-dir is required for a checkpoint protocol")
    if custom:
        if args.protocol != "ad-hoc" or args.partition is None or not args.model:
            raise ValueError("custom protocols require --model and --partition, without --protocol")
        if any("_latest" in path.parts for path in (args.run_dir, *args.model)):
            raise ValueError("custom protocols require explicit paths, never runs/_latest")
        if not args.run_dir.is_dir():
            raise ValueError("--run-dir must identify an existing run directory")
        if args.partition != "screen":
            if len(args.model) != 1 or args.legacy_model is not None:
                raise ValueError("confirmation and blind require one already-selected actor")
            if args.previous_evaluation is None:
                raise ValueError("confirmation and blind require --previous-evaluation")
        elif args.previous_evaluation is not None:
            raise ValueError("screen does not consume a previous partition")
        spec = load_protocol_spec(args.protocol_file)
        if args.partition not in spec["partitions"]:
            raise ValueError("requested partition is not present in this protocol")
        return
    if args.partition is not None:
        raise ValueError("--partition requires --protocol-file")
    if args.previous_evaluation is not None:
        raise ValueError("--previous-evaluation requires --protocol-file")
    if args.model and any(path.suffix == ".pt" for path in args.model):
        raise ValueError("DrQ actors require --protocol-file for isolated CPU evaluation")
    if args.protocol == "ad-hoc":
        if not args.model:
            raise ValueError("--model is required for ad-hoc evaluation")
        if not args.seeds:
            raise ValueError("--seeds is required for ad-hoc evaluation")


def main():
    args = parse_args()
    if args.check_runtime:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        runtime = runtime_metadata(Path(__file__).resolve().parent)
        try:
            validate_cpu_runtime(runtime)
            if args.protocol_file is not None:
                load_protocol_spec(args.protocol_file)
        except (OSError, ValueError) as error:
            print(json.dumps({"status": "exception", "error": str(error), "runtime": runtime}, sort_keys=True))
            return 1
        print(json.dumps({"status": "ok", "runtime": runtime}, sort_keys=True))
        return 0
    if args.worker:
        return worker_result(args)
    validate_protocol_request(args)
    if args.protocol != "ad-hoc" or args.protocol_file is not None:
        output_dir = run_checkpoint_protocol(args, args.protocol)
        print(f"wrote immutable checkpoint evaluation: {output_dir}")
        return 0

    track_ids = parse_int_list(args.track_ids, "--track-ids", minimum=1)
    seeds = parse_int_list(args.seeds, "--seeds", minimum=0)
    action_smoothing_override = (
        normalize_action_smoothing(json.loads(args.action_smoothing_config))
        if args.action_smoothing_config
        else None
    )
    action_control_override = (
        normalize_action_control(json.loads(args.action_control_config))
        if args.action_control_config
        else None
    )
    started = time.perf_counter()
    models = []
    for path in args.model:
        metadata = candidate_metadata(path)
        action_smoothing = action_smoothing_override
        if action_smoothing is None and metadata["action_smoothing_present"]:
            action_smoothing = metadata["action_smoothing"]
        action_control = action_control_override
        if action_control is None and metadata["action_control_present"]:
            action_control = metadata["action_control"]
        models.append(
            evaluate_model(
                path,
                track_ids,
                seeds,
                args.max_steps,
                args.frame_skip,
                action_smoothing=action_smoothing,
                action_control=action_control,
            )
        )
    report = {
        "protocol": {
            "name": "ad-hoc",
            "deterministic": True,
            "track_ids": track_ids,
            "seeds": seeds,
            "max_steps": args.max_steps,
            "frame_skip": args.frame_skip,
            "episodes_per_model": len(track_ids) * len(seeds),
        },
        "models": models,
        "wall_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"wrote ad-hoc evaluation report: {args.output}")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
