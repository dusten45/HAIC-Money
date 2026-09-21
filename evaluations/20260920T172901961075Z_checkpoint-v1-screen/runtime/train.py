import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import tracking
from action_smoothing import (
    action_control_fingerprint,
    action_smoothing_fingerprint,
    append_action_control_plane,
    build_action_smoother,
    canonical_action_smoothing,
    normalize_action_smoothing,
    normalize_action_control,
    read_embedded_action_smoothing,
    write_embedded_action_smoothing,
    validate_action_control,
)
from core.vendor.car_racing import CarRacing
from env_wrapper import CarEnvironment

RAW_FRAME_BUDGET_MARGIN = 200
FRAME_SKIP = 4
DEFAULT_TRACK_IDS = (1,)
GAME_VARIABLES_VERSION = "variables-6"
CHECKPOINT_PREFIX = "ppo_baseline"
VECNORM_FILENAME = "model_vecnormalize.pkl"
BEST_MODEL_FILENAME = "best_model"
BEST_VECNORM_FILENAME = "best_model_vecnormalize.pkl"
UINT32_SEED_UPPER = 2**32


class HaicTrack(gym.Wrapper):
    def __init__(self, track_id, seed, max_steps, frame_skip=FRAME_SKIP, obstacles=True):
        raw_frame_budget = max_steps * frame_skip + RAW_FRAME_BUDGET_MARGIN
        inner = TimeLimit(
            CarRacing(continuous=True, render_mode=None),
            max_episode_steps=raw_frame_budget,
        )
        wrapped = CarEnvironment(inner, skip_frames=frame_skip)
        super().__init__(wrapped)
        self.track_id = track_id
        self.seed = seed
        self.obstacles = obstacles

    def reset(self, *, seed=None, options=None):
        reset_options = {"track_id": self.track_id} if self.obstacles else None
        return self.env.reset(seed=self.seed, options=reset_options)


def sample_worker_seeds(master_seed: int, n_envs: int) -> list[int]:
    sequences = np.random.SeedSequence(master_seed).spawn(n_envs)
    return [int(sequence.generate_state(1, dtype=np.uint32)[0]) for sequence in sequences]


def sample_track_seed(rng, excluded_seeds: frozenset[int]) -> int:
    while True:
        seed = int(rng.integers(0, UINT32_SEED_UPPER, dtype=np.uint64))
        if seed not in excluded_seeds:
            return seed


class SampledHaicTrack(HaicTrack):
    """Training environment that draws a fresh official track for every episode."""

    def __init__(
        self,
        track_ids,
        sampler_seed,
        max_steps,
        frame_skip=FRAME_SKIP,
        excluded_seeds=(),
        obstacles=True,
    ):
        if not track_ids:
            raise ValueError("track_ids must contain at least one value")
        super().__init__(int(track_ids[0]), 0, max_steps, frame_skip, obstacles)
        self._track_ids = tuple(int(track_id) for track_id in track_ids)
        self._rng = np.random.default_rng(sampler_seed)
        self._excluded_seeds = frozenset(int(seed) for seed in excluded_seeds)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.track_id = int(self._rng.choice(self._track_ids))
        self.seed = sample_track_seed(self._rng, self._excluded_seeds)
        return super().reset()


class FinishBonus(gym.Wrapper):
    FINISH_BONUS = 100.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info.get("finished"):
            reward += self.FINISH_BONUS
        return observation, reward, terminated, truncated, info


class CollisionPenalty(gym.Wrapper):
    def __init__(self, env, penalty):
        super().__init__(env)
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("collision penalty must be finite and non-negative")
        self.penalty = float(penalty)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info.get("collision", False):
            reward -= self.penalty
        return observation, reward, terminated, truncated, info


class ActionSmoothing(gym.Wrapper):
    """Apply one stateful smoother per environment action step."""

    def __init__(self, env, config, action_control=None):
        super().__init__(env)
        self.action_smoothing = normalize_action_smoothing(config)
        self.action_control = normalize_action_control(action_control)
        validate_action_control(self.action_smoothing, self.action_control)
        self.smoother = build_action_smoother(self.action_smoothing)
        self.last_action = None
        if self.action_control["input_channels"] == 5:
            shape = self.observation_space.shape
            if len(shape) != 3 or shape[0] != 4:
                raise ValueError("action control expects 4-channel image observations")
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(5, shape[1], shape[2]),
                dtype=np.float32,
            )

    def _augment_observation(self, observation):
        return append_action_control_plane(
            observation,
            self.action_control,
            self.smoother.last_action,
        )

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        self.smoother.reset(initial_action=self.action_smoothing["initial_action"])
        self.last_action = np.asarray(self.smoother.last_action, dtype=np.float32)
        observation, info = result
        return self._augment_observation(observation), info

    def step(self, action):
        smoothed = np.asarray(self.smoother.smooth(action), dtype=np.float32)
        self.last_action = smoothed.copy()
        observation, reward, terminated, truncated, info = self.env.step(smoothed)
        return self._augment_observation(observation), reward, terminated, truncated, info


def action_smoothing_from_args(method: str, alpha: float):
    if method == "none":
        return normalize_action_smoothing()
    if method == "steering-ema":
        return canonical_action_smoothing("alpha", [float(alpha), 1.0, 1.0])
    raise ValueError(f"unsupported action smoothing method: {method}")


def action_control_from_args(method: str):
    return normalize_action_control(method and {"method": method})


def build_env(
    track_id,
    seed,
    max_steps,
    frame_skip,
    reward_shaping,
    obstacles=True,
    collision_penalty=0.0,
    action_smoothing=None,
    action_control=None,
):
    env = HaicTrack(track_id, seed, max_steps, frame_skip, obstacles)
    env = TimeLimit(env, max_episode_steps=max_steps)
    if collision_penalty:
        env = CollisionPenalty(env, collision_penalty)
    if reward_shaping:
        env = FinishBonus(env)
    if action_smoothing is not None or action_control is not None:
        env = ActionSmoothing(env, action_smoothing, action_control)
    return env


def build_sampled_env(
    track_ids,
    sampler_seed,
    max_steps,
    frame_skip,
    reward_shaping,
    excluded_seeds,
    obstacles,
    collision_penalty=0.0,
    action_smoothing=None,
    action_control=None,
):
    env = SampledHaicTrack(
        track_ids,
        sampler_seed,
        max_steps,
        frame_skip,
        excluded_seeds,
        obstacles,
    )
    env = TimeLimit(env, max_episode_steps=max_steps)
    if collision_penalty:
        env = CollisionPenalty(env, collision_penalty)
    if reward_shaping:
        env = FinishBonus(env)
    if action_smoothing is not None or action_control is not None:
        env = ActionSmoothing(env, action_smoothing, action_control)
    return env


def make_vec_env(
    track_ids: Sequence[int],
    seeds: Sequence[int],
    n_envs: int,
    max_steps: int,
    frame_skip: int,
    reward_shaping: bool,
    vec_type: str = "subproc",
    training_track_mode: str = "sampled",
    track_sampler_seed: int = 917,
    excluded_seeds=(),
    training_obstacles: str = "official",
    collision_penalty: float = 0.0,
    action_smoothing=None,
    action_control=None,
):
    if training_track_mode not in {"fixed", "sampled"}:
        raise ValueError(f"unsupported training_track_mode: {training_track_mode}")
    if training_obstacles not in {"official", "none"}:
        raise ValueError(f"unsupported training_obstacles: {training_obstacles}")
    if not np.isfinite(collision_penalty) or collision_penalty < 0:
        raise ValueError("collision_penalty must be finite and non-negative")
    obstacles = training_obstacles == "official"

    def env_fns():
        worker_seeds = sample_worker_seeds(track_sampler_seed, n_envs)
        for i in range(n_envs):
            track_id = int(track_ids[i % len(track_ids)])
            seed = int(seeds[i % len(seeds)])
            if training_track_mode == "sampled":
                yield lambda sampler_seed=worker_seeds[i]: build_sampled_env(
                    track_ids,
                    sampler_seed,
                    max_steps,
                    frame_skip,
                    reward_shaping,
                    excluded_seeds,
                    obstacles,
                    collision_penalty,
                    action_smoothing,
                    action_control,
                )
            else:
                yield lambda tid=track_id, sd=seed: build_env(
                    tid,
                    sd,
                    max_steps,
                    frame_skip,
                    reward_shaping,
                    obstacles,
                    collision_penalty,
                    action_smoothing,
                    action_control,
                )

    env_fn_list = list(env_fns())
    if vec_type == "subproc":
        return SubprocVecEnv(env_fn_list)
    return DummyVecEnv(env_fn_list)


def evaluate(
    model,
    track_id,
    seed,
    max_steps,
    frame_skip,
    episodes=1,
    action_smoothing=None,
    action_control=None,
):
    results = []
    for _ in range(episodes):
        env = build_env(
            track_id,
            seed,
            max_steps,
            frame_skip,
            reward_shaping=False,
            obstacles=True,
            collision_penalty=0.0,
            action_smoothing=action_smoothing,
            action_control=action_control,
        )
        observation, _ = env.reset()
        start_time_s = env.unwrapped.t
        total_reward = 0.0
        done = False
        steps = 0
        executed_actions = []
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            executed_action = getattr(env, "last_action", None)
            if executed_action is None:
                executed_action = np.asarray(action, dtype=np.float32)
            executed_actions.append(np.asarray(executed_action, dtype=np.float32).copy())
            total_reward += reward
            steps += 1
            done = terminated or truncated
        finish_time_s = info.get("finish_time_s")
        lap_time_ms = (
            round((finish_time_s - start_time_s) * 1000) if finish_time_s is not None else None
        )
        results.append(
            {
                "track_id": track_id,
                "seed": seed,
                "steps": steps,
                "reward": total_reward,
                "progress": info.get("progress", 0.0),
                "finished": info.get("finished", False),
                "finish_time_s": finish_time_s,
                "lap_time_ms": lap_time_ms,
                "damage": info.get("damage", 0.0),
                "retire_reason": info.get("retire_reason"),
                "steering_delta_abs_mean": (
                    float(np.mean(np.abs(np.diff(np.asarray(executed_actions)[:, 0]))))
                    if len(executed_actions) > 1
                    else 0.0
                ),
            }
        )
        env.close()
    return results


def summarize(results):
    total = len(results)
    finished = [r for r in results if r["finished"]]
    lap_times = [r["lap_time_ms"] for r in results if r.get("lap_time_ms") is not None]
    termination_reasons = Counter(
        "finished" if result["finished"] else result.get("retire_reason") or "time_limit"
        for result in results
    )
    return {
        "n_episodes": total,
        "finish_rate": len(finished) / total if total else 0.0,
        "avg_progress": float(np.mean([r["progress"] for r in results])) if total else 0.0,
        "avg_reward": float(np.mean([r["reward"] for r in results])) if total else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in results])) if total else 0.0,
        "avg_lap_time_ms": float(np.mean(lap_times)) if lap_times else None,
        "best_lap_time_ms": float(np.min(lap_times)) if lap_times else None,
        "avg_damage": float(np.mean([r.get("damage", 0.0) for r in results])) if total else 0.0,
        "avg_steering_delta_abs_mean": (
            float(np.mean([r.get("steering_delta_abs_mean", 0.0) for r in results]))
            if total else 0.0
        ),
        "termination_reasons": dict(sorted(termination_reasons.items())),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_vecnormalize_path(resume_path, explicit_path):
    if explicit_path:
        return Path(explicit_path)
    checkpoint = Path(resume_path)
    candidates = [checkpoint.with_name(VECNORM_FILENAME)]
    stem = checkpoint.stem
    if stem == BEST_MODEL_FILENAME:
        candidates.insert(0, checkpoint.with_name(BEST_VECNORM_FILENAME))
    if stem.startswith(CHECKPOINT_PREFIX) and stem.endswith("_steps"):
        candidates.insert(
            0,
            checkpoint.with_name(
                stem.replace(CHECKPOINT_PREFIX, CHECKPOINT_PREFIX + "_vecnormalize", 1)
                + ".pkl"
            ),
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resume_run_config(resume_path):
    checkpoint = Path(resume_path)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return {}
    try:
        recorded = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid resume run config: {config_path}") from error
    config = recorded.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"invalid resume run config payload: {config_path}")
    return config


def validate_action_smoothing_resume(source_config, requested_config, allow_change):
    source = normalize_action_smoothing(source_config)
    requested = normalize_action_smoothing(requested_config)
    changed = action_smoothing_fingerprint(source) != action_smoothing_fingerprint(requested)
    if changed and not allow_change:
        raise ValueError("--action-smoothing must match the resumed checkpoint transform")
    return changed


def validate_action_control_resume(source_config, requested_config, allow_change):
    source = normalize_action_control(source_config)
    requested = normalize_action_control(requested_config)
    changed = action_control_fingerprint(source) != action_control_fingerprint(requested)
    if changed and not allow_change:
        raise ValueError("--action-control must match the resumed checkpoint representation")
    return changed


class HoldoutEvaluator:
    def __init__(
        self,
        eval_track_ids,
        seen_eval_seeds,
        holdout_eval_seeds,
        max_steps,
        frame_skip,
        eval_episodes,
        run_dir,
        action_smoothing=None,
        action_control=None,
    ):
        self.eval_track_ids = eval_track_ids
        self.seen_eval_seeds = seen_eval_seeds
        self.holdout_eval_seeds = holdout_eval_seeds
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.eval_episodes = eval_episodes
        self.run_dir = run_dir
        self.action_smoothing = normalize_action_smoothing(action_smoothing)
        self.action_control = normalize_action_control(action_control)
        self.best_score = None
        self.best_step = None

    def _evaluate_seeds(self, model, seeds):
        results = []
        for tid in self.eval_track_ids:
            for seed in seeds:
                results.extend(
                    evaluate(
                        model, tid, seed, self.max_steps, self.frame_skip,
                        episodes=self.eval_episodes,
                        action_smoothing=self.action_smoothing,
                        action_control=self.action_control,
                    )
                )
        return results

    @staticmethod
    def _selection_score(metrics):
        lap_time_ms = metrics["avg_lap_time_ms"]
        return (
            metrics["finish_rate"],
            metrics["avg_progress"],
            -lap_time_ms if lap_time_ms is not None else float("-inf"),
        )

    def _save_best_model(self, model, metrics):
        best_path = self.run_dir / BEST_MODEL_FILENAME
        model.save(str(best_path))
        best_archive = Path(f"{best_path}.zip")
        write_embedded_action_smoothing(
            best_archive,
            getattr(model, "haic_action_smoothing", self.action_smoothing),
        )
        vecnormalize = model.get_vec_normalize_env()
        if vecnormalize is not None:
            vecnormalize.save(str(self.run_dir / BEST_VECNORM_FILENAME))
        return best_archive

    def _load_cpu_snapshot(self, model):
        snapshot = self.run_dir / ".evaluation_snapshot"
        archive = snapshot.with_suffix(".zip")
        model.save(str(snapshot))
        write_embedded_action_smoothing(
            archive,
            getattr(model, "haic_action_smoothing", self.action_smoothing),
        )
        try:
            return PPO.load(str(archive), device="cpu"), file_sha256(archive)
        finally:
            archive.unlink(missing_ok=True)

    def evaluate(self, model, is_final=False):
        step = int(model.num_timesteps)
        cpu_model, evaluation_checkpoint_sha256 = self._load_cpu_snapshot(model)
        seen_episodes = self._evaluate_seeds(cpu_model, self.seen_eval_seeds)
        holdout_episodes = self._evaluate_seeds(cpu_model, self.holdout_eval_seeds)
        seen_metrics = summarize(seen_episodes)
        holdout_metrics = summarize(holdout_episodes)
        score = self._selection_score(holdout_metrics)
        is_best = self.best_score is None or score > self.best_score
        if is_best:
            self.best_score = score
            self.best_step = step

        metrics = {
            "step": step,
            "is_final": is_final,
            "seen": seen_metrics,
            "holdout": holdout_metrics,
            "seen_episodes": seen_episodes,
            "holdout_episodes": holdout_episodes,
            "action_smoothing": self.action_smoothing,
            "action_smoothing_fingerprint": action_smoothing_fingerprint(
                self.action_smoothing
            ),
            "action_control": self.action_control,
            "action_control_fingerprint": action_control_fingerprint(self.action_control),
            "evaluation_checkpoint_sha256": evaluation_checkpoint_sha256,
            "best_model_updated": is_best,
            "best_model_step": self.best_step,
        }
        if is_best:
            best_archive = self._save_best_model(model, metrics)
            metrics["best_model_sha256"] = file_sha256(best_archive)
            cpu_best = PPO.load(str(best_archive), device="cpu")
            reloaded_seen_episodes = self._evaluate_seeds(cpu_best, self.seen_eval_seeds)
            reloaded_holdout_episodes = self._evaluate_seeds(
                cpu_best, self.holdout_eval_seeds
            )
            metrics["cpu_reload_seen_matches_selection"] = (
                reloaded_seen_episodes == seen_episodes
            )
            metrics["cpu_reload_holdout_matches_selection"] = (
                reloaded_holdout_episodes == holdout_episodes
            )
            if not (
                metrics["cpu_reload_seen_matches_selection"]
                and metrics["cpu_reload_holdout_matches_selection"]
            ):
                raise RuntimeError("persisted best model disagrees with CPU selection")
            tracking.write_json(self.run_dir / "best_model_metrics.json", metrics)
        tracking.log_metrics(self.run_dir, metrics)
        print(
            f"[eval @ {metrics['step']}] "
            f"seen finish_rate={seen_metrics['finish_rate']:.2f} "
            f"progress={seen_metrics['avg_progress']:.3f} | "
            f"holdout finish_rate={holdout_metrics['finish_rate']:.2f} "
            f"progress={holdout_metrics['avg_progress']:.3f} "
            f"best={is_best}"
        )
        return metrics


def checkpoint_paths(checkpoint_dir: Path, step: int):
    stem = f"{CHECKPOINT_PREFIX}_{step}_steps"
    return (
        checkpoint_dir / stem,
        checkpoint_dir / f"{CHECKPOINT_PREFIX}_vecnormalize_{step}_steps.pkl",
    )


def save_checkpoint(model, checkpoint_dir: Path, step: int):
    model_path, vecnormalize_path = checkpoint_paths(checkpoint_dir, step)
    model.save(str(model_path))
    model_archive = Path(f"{model_path}.zip") if model_path.suffix != ".zip" else model_path
    write_embedded_action_smoothing(
        model_archive,
        getattr(model, "haic_action_smoothing", normalize_action_smoothing()),
    )
    vecnormalize = model.get_vec_normalize_env()
    if vecnormalize is not None:
        vecnormalize.save(str(vecnormalize_path))
    print(f"saved checkpoint: {model_path}.zip")


def next_interval_boundary(step: int, interval: int):
    remainder = step % interval
    return step + interval if remainder == 0 else step + interval - remainder


def train_in_segments(
    model,
    total_timesteps,
    save_freq,
    eval_freq,
    on_checkpoint,
    on_evaluate,
    evaluate_final,
):
    current_step = int(model.num_timesteps)
    target_step = current_step + total_timesteps
    next_checkpoint = (
        next_interval_boundary(current_step, save_freq) if save_freq > 0 else None
    )
    next_evaluation = (
        next_interval_boundary(current_step, eval_freq) if eval_freq > 0 else None
    )
    last_evaluation_step = None

    while current_step < target_step:
        segment_targets = [target_step]
        segment_targets.extend(
            step for step in (next_checkpoint, next_evaluation) if step is not None
        )
        segment_target = min(segment_targets)
        model.learn(
            total_timesteps=segment_target - current_step,
            reset_num_timesteps=False,
        )
        updated_step = int(model.num_timesteps)
        if updated_step <= current_step:
            raise RuntimeError("PPO training segment did not advance num_timesteps")
        current_step = updated_step

        if next_checkpoint is not None and current_step >= next_checkpoint:
            on_checkpoint(current_step)
            next_checkpoint = next_interval_boundary(current_step, save_freq)
        if next_evaluation is not None and current_step >= next_evaluation:
            on_evaluate(is_final=current_step >= target_step)
            last_evaluation_step = current_step
            next_evaluation = next_interval_boundary(current_step, eval_freq)

    if evaluate_final and last_evaluation_step != current_step:
        on_evaluate(is_final=True)


class Tee:
    def __init__(self, primary, mirror_path):
        self.primary = primary
        self.mirror = open(mirror_path, "a", buffering=1)

    def write(self, data):
        try:
            self.primary.write(data)
        except Exception:
            pass
        self.mirror.write(data)
        return len(data)

    def flush(self):
        try:
            self.primary.flush()
        except Exception:
            pass
        self.mirror.flush()

    def __getattr__(self, name):
        return getattr(self.primary, name)


def parse_args():
    parser = argparse.ArgumentParser(description="PPO + CNN 학습 (HAIC CarRacing)")
    parser.add_argument("--track-ids", type=str, default="1")
    parser.add_argument("--seeds", type=str, default="42,1337,2024,777")
    parser.add_argument(
        "--training-track-mode", choices=["sampled", "fixed"], default="sampled",
        help="sample fresh (track_id, seed) pairs each episode, or retain fixed worker tracks",
    )
    parser.add_argument(
        "--track-sampler-seed", type=int, default=917,
        help="master RNG seed for reproducible sampled training tracks",
    )
    parser.add_argument(
        "--training-obstacles", choices=["official", "none"], default="official",
        help="official six-obstacle training or obstacle-free curriculum warm start",
    )
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-type", type=str, choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--no-shaping", action="store_true", help="완주 보너스 비활성화")
    parser.add_argument(
        "--collision-penalty", type=float, default=0.0,
        help="training-only native reward penalty per aggregated collision action",
    )
    parser.add_argument(
        "--action-smoothing",
        choices=["none", "steering-ema"],
        default="none",
        help="stateful action intervention shared by training, evaluation, and submission",
    )
    parser.add_argument(
        "--action-smoothing-alpha",
        type=float,
        default=0.35,
        help="steering EMA alpha (smaller values smooth more strongly)",
    )
    parser.add_argument(
        "--allow-action-smoothing-change",
        action="store_true",
        help="allow an explicit action-transform change when resuming a checkpoint",
    )
    parser.add_argument(
        "--action-control",
        choices=["none", "constant-plane", "previous-steering-plane"],
        default="none",
        help="policy observation representation for action-transform state",
    )
    parser.add_argument("--no-norm-reward", action="store_true", help="보상 정규화 비활성화")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo-cnn-baseline1-1", help="실험 이름 (runs/<timestamp>_<name>/ 에 기록)")
    parser.add_argument("--save-path", type=str, default="model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=65536, help="N 스텝마다 체크포인트 저장 (0=비활성)")
    parser.add_argument("--resume", type=str, default="", help="이어서 학습할 체크포인트 zip 경로 (비워두면 새로 시작)")
    parser.add_argument("--resume-vecnormalize", type=str, default="", help="보상 정규화 통계 pkl (기본: resume 경로에서 자동 탐색)")
    parser.add_argument("--eval-freq", type=int, default=65536, help="N 스텝마다 평가 로그 (0=비활성)")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument(
        "--eval",
        action="store_true",
        help="호환용 옵션: 종료 평가는 기본적으로 실행됨",
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help="throughput 측정 등에서 학습 종료 평가를 생략",
    )
    parser.add_argument("--eval-track-ids", type=str, default="")
    parser.add_argument(
        "--seen-eval-seeds",
        "--eval-seeds",
        dest="seen_eval_seeds",
        type=str,
        default="",
        help="학습 seed 평가 목록 (기본: --seeds)",
    )
    parser.add_argument(
        "--holdout-eval-seeds",
        type=str,
        default="10001,10002,10003,10004,10005,10006,10007,10008",
        help="best model 선택용 미사용 seed 목록",
    )
    return parser.parse_args()


def ppo_training_kwargs(args):
    return {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
    }


def validate_training_options(args, seeds, holdout_eval_seeds):
    if not holdout_eval_seeds:
        raise ValueError("--holdout-eval-seeds must contain at least one seed")
    overlapping_holdout_seeds = sorted(set(seeds) & set(holdout_eval_seeds))
    if overlapping_holdout_seeds:
        raise ValueError(
            "--holdout-eval-seeds must not overlap --seeds: "
            f"{overlapping_holdout_seeds}"
        )
    if args.eval and args.skip_final_eval:
        raise ValueError("--eval and --skip-final-eval cannot be used together")
    if Path(args.save_path).stem == BEST_MODEL_FILENAME:
        raise ValueError(f"--save-path '{BEST_MODEL_FILENAME}' is reserved")
    if args.save_freq < 0 or args.eval_freq < 0:
        raise ValueError("--save-freq and --eval-freq must be non-negative")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")
    if not np.isfinite(args.collision_penalty) or args.collision_penalty < 0:
        raise ValueError("--collision-penalty must be finite and non-negative")
    if not np.isfinite(args.action_smoothing_alpha) or not 0 < args.action_smoothing_alpha <= 1:
        raise ValueError("--action-smoothing-alpha must be finite and in (0, 1]")


def main():
    args = parse_args()
    track_ids = [int(t) for t in args.track_ids.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    reward_shaping = not args.no_shaping
    norm_reward = not args.no_norm_reward
    action_smoothing = action_smoothing_from_args(
        args.action_smoothing, args.action_smoothing_alpha
    )
    action_control = action_control_from_args(args.action_control)
    validate_action_control(action_smoothing, action_control)
    source_resume_config = resume_run_config(args.resume) if args.resume else {}
    source_resume_frame_skip = source_resume_config.get("frame_skip")
    if source_resume_frame_skip is not None and source_resume_frame_skip != args.frame_skip:
        raise ValueError(
            f"resume frame_skip {source_resume_frame_skip} does not match "
            f"requested frame_skip {args.frame_skip}"
        )
    embedded_resume_smoothing = (
        read_embedded_action_smoothing(args.resume) if args.resume else None
    )
    recorded_resume_smoothing = (
        normalize_action_smoothing(source_resume_config["action_smoothing"])
        if "action_smoothing" in source_resume_config
        else None
    )
    if (
        embedded_resume_smoothing is not None
        and recorded_resume_smoothing is not None
        and action_smoothing_fingerprint(embedded_resume_smoothing)
        != action_smoothing_fingerprint(recorded_resume_smoothing)
    ):
        raise ValueError("resume checkpoint and run config action smoothing do not match")
    source_resume_smoothing = (
        embedded_resume_smoothing
        or recorded_resume_smoothing
        or normalize_action_smoothing()
    )
    source_resume_control = normalize_action_control(
        source_resume_config.get("action_control")
    )

    eval_track_ids = (
        [int(t) for t in args.eval_track_ids.split(",") if t.strip()] or track_ids
    )
    seen_eval_seeds = (
        [int(s) for s in args.seen_eval_seeds.split(",") if s.strip()] or seeds
    )
    holdout_eval_seeds = [
        int(s) for s in args.holdout_eval_seeds.split(",") if s.strip()
    ]
    validate_training_options(args, seeds, holdout_eval_seeds)

    if args.resume:
        action_smoothing_changed = validate_action_smoothing_resume(
            source_resume_smoothing,
            action_smoothing,
            args.allow_action_smoothing_change,
        )
        action_control_changed = validate_action_control_resume(
            source_resume_control,
            action_control,
            args.allow_action_smoothing_change,
        )
    else:
        action_smoothing_changed = False
        action_control_changed = False

    config = {
        "variables_version": GAME_VARIABLES_VERSION,
        "algorithm": "PPO",
        "policy": "CnnPolicy",
        "track_ids": track_ids,
        "seeds": seeds,
        "training_track_mode": args.training_track_mode,
        "track_sampler_seed": args.track_sampler_seed,
        "sampled_seed_range": [0, UINT32_SEED_UPPER - 1],
        "training_obstacles": args.training_obstacles,
        "evaluation_obstacles": "official",
        "n_envs": args.n_envs,
        "vec_type": args.vec_type,
        "total_timesteps": args.total_timesteps,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "reward_shaping": reward_shaping,
        "collision_penalty": args.collision_penalty,
        "action_smoothing": action_smoothing,
        "action_smoothing_fingerprint": action_smoothing_fingerprint(action_smoothing),
        "action_smoothing_changed_on_resume": action_smoothing_changed,
        "action_control": action_control,
        "action_control_fingerprint": action_control_fingerprint(action_control),
        "action_control_changed_on_resume": action_control_changed,
        "norm_reward": norm_reward,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
        "seed": args.seed,
        "resume_from": args.resume or None,
        "resume_vecnormalize": args.resume_vecnormalize or None,
        "resume_frame_skip": source_resume_frame_skip,
        "resume_action_smoothing": source_resume_smoothing if args.resume else None,
        "resume_action_control": source_resume_control if args.resume else None,
        "eval_track_ids": eval_track_ids,
        "seen_eval_seeds": seen_eval_seeds,
        "holdout_eval_seeds": holdout_eval_seeds,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "eval_requested": args.eval,
        "skip_final_eval": args.skip_final_eval,
    }

    run_dir = tracking.new_run(args.name, config, command_line=" ".join(sys.argv))
    train_log = run_dir / "train.log"
    train_log.write_text("")
    sys.stdout = Tee(sys.stdout, train_log)
    sys.stderr = Tee(sys.stderr, train_log)
    print(f"run_dir: {run_dir}")

    checkpoint_dir = run_dir / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = run_dir / args.save_path

    vecnormalize_stats_path = None
    if args.resume:
        vecnormalize_stats_path = find_vecnormalize_path(
            args.resume, args.resume_vecnormalize
        )
        if norm_reward and vecnormalize_stats_path is None:
            print(
                "[warn] 보상 정규화 통계(.pkl)를 찾지 못했습니다. "
                "보상 정규화를 처음부터 다시 적응합니다."
            )

    print(f"=== HAIC PPO 학습 시작 ===")
    print(f"variables version: {GAME_VARIABLES_VERSION}")
    print(f"track_ids: {track_ids}")
    print(f"seeds: {seeds}")
    print(
        f"training track mode: {args.training_track_mode} "
        f"sampler_seed: {args.track_sampler_seed} obstacles: {args.training_obstacles}"
    )
    print(f"n_envs: {args.n_envs}  total_timesteps: {args.total_timesteps}")
    print(
        f"reward_shaping: {reward_shaping}  norm_reward: {norm_reward} "
        f"collision_penalty: {args.collision_penalty}"
    )
    print(
        f"action smoothing: {args.action_smoothing} "
        f"config={action_smoothing}"
    )
    print(f"action control: {args.action_control} config={action_control}")
    print(f"target_kl: {args.target_kl}")
    print(f"seen eval seeds: {seen_eval_seeds}")
    print(f"holdout eval seeds: {holdout_eval_seeds}")
    if args.resume:
        print(f"resume from: {args.resume}")
        print(f"resume vecnormalize: {vecnormalize_stats_path}")
        if action_smoothing_changed:
            print(
                "[info] resume action smoothing differs from requested config; "
                "recording this as an explicit intervention change"
            )

    raw_env = make_vec_env(
        track_ids,
        seeds,
        args.n_envs,
        args.max_steps,
        args.frame_skip,
        reward_shaping=reward_shaping,
        vec_type=args.vec_type,
        training_track_mode=args.training_track_mode,
        track_sampler_seed=args.track_sampler_seed,
        excluded_seeds=set(seen_eval_seeds) | set(holdout_eval_seeds),
        training_obstacles=args.training_obstacles,
        collision_penalty=args.collision_penalty,
        action_smoothing=action_smoothing,
        action_control=action_control,
    )
    if norm_reward:
        if args.resume and vecnormalize_stats_path is not None:
            vec_env = VecNormalize.load(str(vecnormalize_stats_path), raw_env)
        else:
            vec_env = VecNormalize(
                raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0
            )
    else:
        vec_env = raw_env

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec_env,
            **ppo_training_kwargs(args),
            seed=args.seed,
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            **ppo_training_kwargs(args),
            seed=args.seed,
            verbose=1,
            device="auto",
            policy_kwargs={"normalize_images": False},
        )
    model.haic_action_smoothing = action_smoothing
    model.haic_action_control = action_control

    evaluator = HoldoutEvaluator(
        eval_track_ids,
        seen_eval_seeds,
        holdout_eval_seeds,
        args.max_steps,
        args.frame_skip,
        args.eval_episodes,
        run_dir,
        action_smoothing=action_smoothing,
        action_control=action_control,
    )
    train_in_segments(
        model,
        args.total_timesteps,
        args.save_freq,
        args.eval_freq,
        on_checkpoint=lambda step: save_checkpoint(model, checkpoint_dir, step),
        on_evaluate=lambda is_final: evaluator.evaluate(model, is_final),
        evaluate_final=not args.skip_final_eval,
    )

    model.save(str(save_path))
    final_archive = (
        save_path if save_path.suffix == ".zip" else Path(f"{save_path}.zip")
    )
    write_embedded_action_smoothing(
        final_archive,
        getattr(model, "haic_action_smoothing", action_smoothing),
    )
    print(f"saved model: {save_path}.zip")
    if model.get_vec_normalize_env() is not None:
        vecnorm_path = run_dir / VECNORM_FILENAME
        model.get_vec_normalize_env().save(str(vecnorm_path))
        print(f"saved vecnormalize: {vecnorm_path}")

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
