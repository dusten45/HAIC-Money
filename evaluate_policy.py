import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

# Set process-level limits before importing Torch so every worker uses the same
# CPU-only inference path as the competition submission.
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
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
    Path("train.py"),
    Path("tracking.py"),
    Path("action_smoothing.py"),
    Path("action_representation.py"),
    *ENVIRONMENT_SOURCES,
)


def parse_int_list(value: str, name: str, minimum: int) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < minimum for item in values):
        raise ValueError(f"{name} values must be at least {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


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


def candidate_metadata(path: Path) -> dict:
    vecnormalize = find_vecnormalize_path(path, "")
    if vecnormalize is None:
        raise ValueError(f"checkpoint lacks matching VecNormalize state: {path}")
    with vecnormalize.open("rb") as handle:
        normalizer = pickle.load(handle)
    if normalizer.norm_obs:
        raise ValueError(f"observation-normalized checkpoints are not supported: {path}")
    run_dir = candidate_run_dir(path)
    run_config_path = run_dir / "config.json"
    action_smoothing = normalize_action_smoothing()
    action_control = normalize_action_control()
    action_representation = normalize_action_representation()
    recorded_config = {}
    action_smoothing_present = False
    action_control_present = False
    action_representation_present = False
    if run_config_path.is_file():
        try:
            recorded = json.loads(run_config_path.read_text())
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
    return {
        "source_path": str(path),
        "archive_sha256": sha256_file(path),
        "policy_sha256": sha256_archive_member(path, "policy.pth"),
        "vecnormalize_path": str(vecnormalize),
        "vecnormalize_sha256": sha256_file(vecnormalize),
        "norm_obs": False,
        "run_config_path": str(run_config_path) if run_config_path.is_file() else None,
        "run_config_sha256": (
            sha256_file(run_config_path) if run_config_path.is_file() else None
        ),
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
        "run_frame_skip": recorded_config.get("frame_skip"),
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


def discover_candidates(paths) -> list[dict]:
    unique = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = candidate_metadata(path)
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


def evaluate_cell(
    model_path: Path,
    track_id: int,
    seed: int,
    max_steps: int,
    frame_skip: int,
    action_smoothing=None,
    action_control=None,
    action_representation=None,
) -> dict:
    torch.set_num_threads(1)
    load_started = time.perf_counter()
    model = PPO.load(str(model_path), device="cpu")
    load_seconds = time.perf_counter() - load_started
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
        start_time_s = env.unwrapped.t
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        steps = 0
        while not (terminated or truncated):
            action_started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            action_seconds.append(time.perf_counter() - action_started)
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
            round((finish_time_s - start_time_s) * 1000) if finish_time_s is not None else None
        )
        return {
            "status": "ok",
            "track_id": track_id,
            "seed": seed,
            "steps": steps,
            "reward": total_reward,
            "progress": float(info.get("progress", 0.0)),
            "finished": finished,
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
            "model_load_seconds": load_seconds,
            "reset_seconds": reset_seconds,
            "mean_action_seconds": float(np.mean(action_seconds)) if action_seconds else 0.0,
            "max_action_seconds": float(np.max(action_seconds)) if action_seconds else 0.0,
            "episode_wall_seconds": time.perf_counter() - episode_started,
        }
    finally:
        env.close()


def worker_result(args) -> int:
    try:
        action_smoothing = json.loads(args.worker_action_smoothing)
        action_control = json.loads(args.worker_action_control)
        action_representation = json.loads(args.worker_action_representation)
        result = evaluate_cell(
            args.model[0], args.worker_track_id, args.worker_seed,
            args.max_steps, args.frame_skip,
            action_smoothing=action_smoothing,
            action_control=action_control,
            action_representation=action_representation,
        )
    except Exception as error:
        result = {
            "status": "exception",
            "error_type": type(error).__name__,
            "error": str(error),
            "track_id": args.worker_track_id,
            "seed": args.worker_seed,
        }
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_worker_output(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            return json.loads(line)
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
    command = [
        sys.executable, str(worker_script), "--worker",
        "--model", candidate["_worker_model_path"],
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
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            check=False,
            env=dict(os.environ, **CPU_ENVIRONMENT),
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
        audited = len(repeats) >= 2
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
    failures = [episode for episode in canonical if episode["status"] != "ok"]
    return {
        **{key: value for key, value in candidate.items() if not key.startswith("_")},
        "eligible": (
            not failures
            and not candidate.get("is_comparator", False)
            and candidate["candidate_id"] not in non_reproducible
            and candidate["candidate_id"] not in unaudited
        ),
        "determinism_audited": candidate["candidate_id"] not in unaudited,
        "canonical_episodes": len(canonical),
        "operational_failures": len(failures),
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
    unseen = [metrics for track_id, metrics in summary["by_track"].items() if int(track_id) > 1]
    unseen_finishes = [metrics["finish_rate"] * metrics["n_episodes"] for metrics in unseen]
    metrics = summary["summary"]
    # Results without a finish have no lap metric and rank below any finisher.
    lap_time = metrics["avg_lap_time_ms"]
    return (
        summary["eligible"],
        min(unseen_finishes, default=0),
        sum(unseen_finishes),
        metrics["finish_rate"] * metrics["n_episodes"],
        metrics["avg_progress"],
        -lap_time if lap_time is not None else float("-inf"),
        -summary["operational_failures"],
    )


def runtime_metadata(source_root: Path) -> dict:
    packages = ("gymnasium", "numpy", "opencv-python", "stable-baselines3", "torch")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
        "torch_cuda": torch.version.cuda,
        "cpu_environment": CPU_ENVIRONMENT,
        "runtime_sources": {
            str(path): sha256_file(source_root / path)
            for path in RUNTIME_SOURCES if (source_root / path).is_file()
        },
    }


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
    lines = ["# Checkpoint Evaluation", "", f"Protocol: `{protocol}`", "", "## Ranking", ""]
    for rank, candidate in enumerate(ranked, start=1):
        summary = candidate["summary"]
        lines.append(
            f"{rank}. `{candidate['candidate_id']}` eligible={candidate['eligible']} "
            f"finish_rate={summary['finish_rate']:.3f} progress={summary['avg_progress']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n")


def protocol_output_dir(root: Path, protocol: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final_dir = root / f"{stamp}_{protocol}"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    temporary = Path(tempfile.mkdtemp(prefix=f".pending-{stamp}-", dir=root))
    return temporary, final_dir


def snapshot_runtime(temporary_dir: Path) -> Path:
    runtime_dir = temporary_dir / "runtime"
    runtime_dir.mkdir()
    for source in (
        Path("evaluate_policy.py"),
        Path("train.py"),
        Path("tracking.py"),
        Path("action_smoothing.py"),
        Path("action_representation.py"),
        Path("env_wrapper.py"),
        Path("damage.py"),
    ):
        shutil.copy2(source, runtime_dir / source.name)
    shutil.copytree("core", runtime_dir / "core")
    return runtime_dir / "evaluate_policy.py"


def snapshot_candidates(temporary_dir: Path, candidates) -> None:
    candidate_dir = temporary_dir / "candidates"
    candidate_dir.mkdir()
    for candidate in candidates:
        archive = candidate_dir / f"{candidate['candidate_id']}.zip"
        shutil.copy2(candidate["source_path"], archive)
        snapshot_sha256 = sha256_file(archive)
        if snapshot_sha256 != candidate["archive_sha256"]:
            raise RuntimeError(f"checkpoint changed while snapshotting: {candidate['source_path']}")
        candidate["evaluation_archive_path"] = str(archive.relative_to(temporary_dir))
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
            candidate["evaluation_vecnormalize_path"] = str(
                vecnormalize_archive.relative_to(temporary_dir)
            )
            candidate["evaluation_vecnormalize_sha256"] = vecnormalize_snapshot_sha256
        candidate["_worker_model_path"] = str(archive)


def serializable_candidates(candidates) -> list[dict]:
    return [
        {key: value for key, value in candidate.items() if not key.startswith("_")}
        for candidate in candidates
    ]


def run_checkpoint_protocol(args, protocol_name: str) -> Path:
    protocol = PROTOCOLS[protocol_name]
    candidate_paths = args.model or checkpoint_paths(args.run_dir, None)
    if args.legacy_model is not None:
        candidate_paths.append(args.legacy_model)
    candidates = discover_candidates(candidate_paths)
    for candidate in candidates:
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
    temporary_dir, final_dir = protocol_output_dir(args.evaluations_dir, protocol_name)
    started_at = datetime.now(timezone.utc)
    try:
        worker_script = snapshot_runtime(temporary_dir)
        snapshot_candidates(temporary_dir, candidates)
        smoothing_fingerprints = sorted({
            candidate["action_smoothing_fingerprint"] for candidate in candidates
        })
        manifest = {
            "schema_version": 1,
            "protocol": protocol_name,
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
            "git": git_metadata(),
            "runtime": runtime_metadata(worker_script.parent),
        }
        episodes = []
        for candidate in candidates:
            for track_id in protocol["track_ids"]:
                for seed in protocol["seeds"]:
                    for repeat in range(protocol["repeats"]):
                        episodes.append(
                            run_isolated_cell(
                                candidate, track_id, seed, repeat, args, worker_script
                            )
                        )
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
    return final_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate deterministic PPO policies on fixed HAIC tracks"
    )
    parser.add_argument("--model", type=Path, action="append")
    parser.add_argument("--track-ids", default="1")
    parser.add_argument("--seeds")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--output", type=Path, help="optional JSON path for an ad-hoc report")
    parser.add_argument("--protocol", choices=("ad-hoc", *PROTOCOLS), default="ad-hoc")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--legacy-model", type=Path,
        help="optional historical comparator; omitted from normal screens by default",
    )
    parser.add_argument("--evaluations-dir", type=Path, default=Path("evaluations"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
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
    if args.protocol != "ad-hoc" and args.run_dir is None:
        raise ValueError("--run-dir is required for a checkpoint protocol")
    if args.protocol == "ad-hoc":
        if not args.model:
            raise ValueError("--model is required for ad-hoc evaluation")
        if not args.seeds:
            raise ValueError("--seeds is required for ad-hoc evaluation")


def main():
    args = parse_args()
    if args.worker:
        return worker_result(args)
    validate_protocol_request(args)
    if args.protocol != "ad-hoc":
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
