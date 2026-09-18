import argparse
import sys
from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
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


class EvalLogger(BaseCallback):
    def __init__(
        self,
        eval_track_ids,
        eval_seeds,
        max_steps,
        frame_skip,
        eval_freq,
        eval_episodes,
        run_dir,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_track_ids = eval_track_ids
        self.eval_seeds = eval_seeds
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.run_dir = run_dir
        self.last_eval = 0

    def _init_callback(self):
        super()._init_callback()
        self.last_eval = self.model.num_timesteps

    def _on_step(self):
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        results = []
        for tid in self.eval_track_ids:
            for seed in self.eval_seeds:
                results.extend(
                    evaluate(
                        self.model, tid, seed, self.max_steps, self.frame_skip,
                        episodes=self.eval_episodes,
                    )
                )
        metrics = summarize(results)
        metrics["step"] = int(self.num_timesteps)
        tracking.log_metrics(self.run_dir, metrics)
        print(
            f"[eval @ {metrics['step']}] finish_rate={metrics['finish_rate']:.2f} "
            f"avg_progress={metrics['avg_progress']:.3f} "
            f"avg_reward={metrics['avg_reward']:.1f} "
            f"best_lap_ms={metrics['best_lap_time_ms']}"
        )
        return True


class Tee:
    def __init__(self, primary, mirror_path):
        self.primary = primary
        self.mirror = open(mirror_path, "w", buffering=1)

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
    parser.add_argument("--seeds", type=str, default="42,1337,2024,777,123")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--vec-type", type=str, choices=["dummy", "subproc"], default="subproc")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--no-shaping", action="store_true", help="완주 보너스 비활성화")
    parser.add_argument("--no-norm-reward", action="store_true", help="보상 정규화 비활성화")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="ppo-cnn", help="실험 이름 (runs/<timestamp>_<name>/ 에 기록)")
    parser.add_argument("--save-path", type=str, default="model")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=0, help="N 스텝마다 체크포인트 저장 (0=비활성)")
    parser.add_argument("--resume", type=str, default="", help="이어서 학습할 체크포인트 zip 경로 (비워두면 새로 시작)")
    parser.add_argument("--resume-vecnormalize", type=str, default="", help="보상 정규화 통계 pkl (기본: resume 경로에서 자동 탐색)")
    parser.add_argument("--eval", action="store_true", help="학습 종료 후 완주 평가")
    parser.add_argument("--eval-freq", type=int, default=65536, help="N 스텝마다 평가 로그 (0=비활성)")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--eval-track-ids", type=str, default="")
    parser.add_argument("--eval-seeds", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    track_ids = [int(t) for t in args.track_ids.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    reward_shaping = not args.no_shaping
    norm_reward = not args.no_norm_reward

    eval_track_ids = (
        [int(t) for t in args.eval_track_ids.split(",") if t.strip()] or track_ids
    )
    eval_seeds = (
        [int(s) for s in args.eval_seeds.split(",") if s.strip()] or seeds[:2]
    )

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
        "ent_coef": args.ent_coef,
        "seed": args.seed,
        "resume_from": args.resume or None,
        "resume_vecnormalize": args.resume_vecnormalize or None,
        "eval_track_ids": eval_track_ids,
        "eval_seeds": eval_seeds,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
    }

    run_dir = tracking.new_run(args.name, config, command_line=" ".join(sys.argv))
    sys.stdout = Tee(sys.stdout, run_dir / "train.log")
    sys.stderr = Tee(sys.stderr, run_dir / "train.log")
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
        model = PPO.load(args.resume, env=vec_env)
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            seed=args.seed,
            verbose=1,
            device="auto",
            policy_kwargs={"normalize_images": False},
        )

    callbacks = []
    if args.save_freq and args.save_freq > 0:
        checkpoint_save_freq = max(args.save_freq // args.n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_save_freq,
                save_path=str(checkpoint_dir),
                name_prefix=CHECKPOINT_PREFIX,
                save_vecnormalize=True,
                verbose=2,
            )
        )
    callbacks.append(
        EvalLogger(
            eval_track_ids,
            eval_seeds,
            args.max_steps,
            args.frame_skip,
            args.eval_freq,
            args.eval_episodes,
            run_dir,
        )
    )

    reset_num_timesteps = not bool(args.resume)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(str(save_path))
    print(f"saved model: {save_path}.zip")
    if model.get_vec_normalize_env() is not None:
        vecnorm_path = run_dir / VECNORM_FILENAME
        model.get_vec_normalize_env().save(str(vecnorm_path))
        print(f"saved vecnormalize: {vecnorm_path}")

    if args.eval:
        print("=== 평가 시작 ===")
        for tid in eval_track_ids:
            for s in eval_seeds:
                results = evaluate(model, tid, s, args.max_steps, args.frame_skip)
                for r in results:
                    status = "FINISHED" if r["finished"] else "DNF"
                    print(
                        f"track:{tid} seed:{s} steps:{r['steps']} "
                        f"reward:{r['reward']:.2f} progress:{r['progress']:.4f} "
                        f"lap_time_ms:{r['lap_time_ms']} {status}"
                    )

    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
