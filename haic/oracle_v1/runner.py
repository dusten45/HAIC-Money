"""Run the local-only oracle against the repository's HAIC environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit

from core.vendor.car_racing import CarRacing, FPS
from env_wrapper import CarEnvironment

from .controller import OracleController


def run_episode(args, track_id, seed):
    raw_budget = args.max_steps * args.frame_skip + 200
    base = CarRacing(continuous=True, render_mode="rgb_array")
    env = CarEnvironment(TimeLimit(base, max_episode_steps=raw_budget),
                         skip_frames=args.frame_skip, no_operation=args.warmup)
    try:
        env.reset(seed=seed, options={"track_id": track_id})
        controller = OracleController(base, target_speed=args.target_speed,
                                      avoid_obstacles=args.avoid_obstacles)
        start_t = base.t
        start_wall = time.monotonic()
        action_hash = hashlib.sha256()
        collision_actions = 0
        steps = 0
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        while steps < args.max_steps:
            action, _ = controller.act()
            action = np.asarray(action, dtype=np.float32)
            if action.shape != (3,) or not env.action_space.contains(action):
                raise ValueError(f"Invalid action: {action}")
            _, reward, terminated, truncated, info = env.step(action)
            action_hash.update(action.tobytes(order="C"))
            collision_actions += bool(info.get("collision"))
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break

        finish_time = base.finish_time_s
        finished = finish_time is not None
        return {
            "track_id": track_id,
            "geometry_seed": seed,
            "finished": finished,
            "finish_qualified": base.finish_qualified_time_s is not None,
            "finish_time_s": finish_time,
            "lap_time_ms": (round((finish_time - start_t) * 1000)
                            if finished else None),
            "progress": env._calculate_progress(),
            "damage": env.damage.damage,
            "steps": steps,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
            "reason": ("finished" if finished else info.get("retire_reason")
                       or ("environment_terminated" if terminated
                           else "environment_truncated" if truncated
                           else "runner_max_steps")),
            "collision_positive_actions": collision_actions,
            "action_trace_sha256": action_hash.hexdigest(),
            "elapsed_raw_frames": round((base.t - start_t) * FPS),
            "wall_seconds": time.monotonic() - start_wall,
        }
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--target-speed", type=float, default=12.0)
    parser.add_argument("--avoid-obstacles", action="store_true")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; its parent must already exist")
    args = parser.parse_args(argv)
    if min(args.max_steps, args.frame_skip) < 1 or args.warmup < 0:
        parser.error("max-steps/frame-skip must be positive; warmup must be nonnegative")
    if not np.isfinite(args.target_speed) or args.target_speed <= 0:
        parser.error("target-speed must be finite and positive")
    if any(track_id not in (1, 2, 3, 4, 5) for track_id in args.track_ids):
        parser.error("track-ids must be 1..5")
    if any(not 0 <= seed <= 0xFFFFFFFF for seed in args.seeds):
        parser.error("seeds must be 0..4294967295")
    if (len(set(args.track_ids)) != len(args.track_ids)
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error("track-ids and seeds must be unique")
    if not args.output.parent.is_dir() or args.output.exists():
        parser.error("output must not exist and its parent directory must already exist")
    args.output.mkdir()

    metadata = {
        "args": vars(args) | {"output": str(args.output)},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "domain_randomize": False,
        "fps": FPS,
        "episode_limit": "runner max_steps does not set environment truncation",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, allow_nan=False, sort_keys=True) + "\n"
    )
    episodes = [run_episode(args, track_id, seed)
                for track_id in args.track_ids for seed in args.seeds]
    summary = {
        "episode_count": len(episodes),
        "finish_count": sum(episode["finished"] for episode in episodes),
        "road_count": len({(episode["track_id"], episode["geometry_seed"])
                            for episode in episodes}),
        "episodes": episodes,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True) + "\n"
    )
    for episode in episodes:
        print(f"track{episode['track_id']}_seed{episode['geometry_seed']}: "
              f"finish={episode['finished']} progress={episode['progress']:.4f} "
              f"damage={episode['damage']:.2f} steps={episode['steps']} "
              f"reason={episode['reason']}", flush=True)


if __name__ == "__main__":
    main()
