import argparse
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import tracking
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


class HaicTrack(gym.Wrapper):
    def __init__(self, track_id, seed, max_steps, frame_skip=FRAME_SKIP):
        raw_frame_budget = max_steps * frame_skip + RAW_FRAME_BUDGET_MARGIN
        inner = TimeLimit(
            CarRacing(continuous=True, render_mode=None),
            max_episode_steps=raw_frame_budget,
        )
        wrapped = CarEnvironment(inner, skip_frames=frame_skip)
        super().__init__(wrapped)
        self.track_id = track_id
        self.seed = seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self.seed, options={"track_id": self.track_id})


class FinishBonus(gym.Wrapper):
    FINISH_BONUS = 100.0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if info.get("finished"):
            reward += self.FINISH_BONUS
        return observation, reward, terminated, truncated, info


def build_env(track_id, seed, max_steps, frame_skip, reward_shaping):
    env = HaicTrack(track_id, seed, max_steps, frame_skip)
    env = TimeLimit(env, max_episode_steps=max_steps)
    if reward_shaping:
        env = FinishBonus(env)
    return env


def make_vec_env(
    track_ids: Sequence[int],
    seeds: Sequence[int],
    n_envs: int,
    max_steps: int,
    frame_skip: int,
    reward_shaping: bool,
    vec_type: str = "subproc",
):
    def env_fns():
        for i in range(n_envs):
            track_id = int(track_ids[i % len(track_ids)])
            seed = int(seeds[i % len(seeds)])
            yield lambda tid=track_id, sd=seed: build_env(
                tid, sd, max_steps, frame_skip, reward_shaping
            )

    env_fn_list = list(env_fns())
    if vec_type == "subproc":
        return SubprocVecEnv(env_fn_list)
    return DummyVecEnv(env_fn_list)


def evaluate(model, track_id, seed, max_steps, frame_skip, episodes=1):
    results = []
    for _ in range(episodes):
        env = build_env(track_id, seed, max_steps, frame_skip, reward_shaping=False)
        observation, _ = env.reset()
        start_time_s = env.unwrapped.t
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
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
            }
        )
        env.close()
    return results


def summarize(results):
    total = len(results)
    finished = [r for r in results if r["finished"]]
    lap_times = [r["lap_time_ms"] for r in results if r.get("lap_time_ms") is not None]
    return {
        "n_episodes": total,
        "finish_rate": len(finished) / total if total else 0.0,
        "avg_progress": float(np.mean([r["progress"] for r in results])) if total else 0.0,
        "avg_reward": float(np.mean([r["reward"] for r in results])) if total else 0.0,
        "avg_steps": float(np.mean([r["steps"] for r in results])) if total else 0.0,
        "avg_lap_time_ms": float(np.mean(lap_times)) if lap_times else None,
        "best_lap_time_ms": float(np.min(lap_times)) if lap_times else None,
    }


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
    ):
        self.eval_track_ids = eval_track_ids
        self.seen_eval_seeds = seen_eval_seeds
        self.holdout_eval_seeds = holdout_eval_seeds
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.eval_episodes = eval_episodes
        self.run_dir = run_dir
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
        vecnormalize = model.get_vec_normalize_env()
        if vecnormalize is not None:
            vecnormalize.save(str(self.run_dir / BEST_VECNORM_FILENAME))
        tracking.write_json(self.run_dir / "best_model_metrics.json", metrics)

    def evaluate(self, model, is_final=False):
        step = int(model.num_timesteps)
        seen_metrics = summarize(self._evaluate_seeds(model, self.seen_eval_seeds))
        holdout_metrics = summarize(
            self._evaluate_seeds(model, self.holdout_eval_seeds)
        )
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
            "best_model_updated": is_best,
            "best_model_step": self.best_step,
        }
        if is_best:
            self._save_best_model(model, metrics)
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
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-type", type=str, choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--no-shaping", action="store_true", help="완주 보너스 비활성화")
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


def main():
    args = parse_args()
    track_ids = [int(t) for t in args.track_ids.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    reward_shaping = not args.no_shaping
    norm_reward = not args.no_norm_reward

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

    config = {
        "variables_version": GAME_VARIABLES_VERSION,
        "algorithm": "PPO",
        "policy": "CnnPolicy",
        "track_ids": track_ids,
        "seeds": seeds,
        "n_envs": args.n_envs,
        "vec_type": args.vec_type,
        "total_timesteps": args.total_timesteps,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "reward_shaping": reward_shaping,
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
    print(f"n_envs: {args.n_envs}  total_timesteps: {args.total_timesteps}")
    print(f"reward_shaping: {reward_shaping}  norm_reward: {norm_reward}")
    print(f"target_kl: {args.target_kl}")
    print(f"seen eval seeds: {seen_eval_seeds}")
    print(f"holdout eval seeds: {holdout_eval_seeds}")
    if args.resume:
        print(f"resume from: {args.resume}")
        print(f"resume vecnormalize: {vecnormalize_stats_path}")

    raw_env = make_vec_env(
        track_ids,
        seeds,
        args.n_envs,
        args.max_steps,
        args.frame_skip,
        reward_shaping=reward_shaping,
        vec_type=args.vec_type,
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

    evaluator = HoldoutEvaluator(
        eval_track_ids,
        seen_eval_seeds,
        holdout_eval_seeds,
        args.max_steps,
        args.frame_skip,
        args.eval_episodes,
        run_dir,
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
    print(f"saved model: {save_path}.zip")
    if model.get_vec_normalize_env() is not None:
        vecnorm_path = run_dir / VECNORM_FILENAME
        model.get_vec_normalize_env().save(str(vecnorm_path))
        print(f"saved vecnormalize: {vecnorm_path}")

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
