import argparse
import json
from pathlib import Path

import numpy as np

import tracking
from common_adapter import CPUActorAdapter, EpisodeCollector
from drq_v2 import DrQv2Agent, DrQv2Config, load_exported_actor
from train import build_env, build_sampled_env, summarize


def parse_ints(value):
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Run a native DrQ-v2 HAIC smoke")
    parser.add_argument("--name", required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--track-ids", default="1,2,3")
    parser.add_argument("--track-sampler-seed", type=int, default=917)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-track-ids", default="4,5,6")
    parser.add_argument("--eval-seeds", default="14001,14002,14003,14004")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.total_steps <= 0 or args.updates_per_step <= 0:
        raise ValueError("total steps and updates per step must be positive")
    track_ids = parse_ints(args.track_ids)
    eval_track_ids = parse_ints(args.eval_track_ids)
    eval_seeds = parse_ints(args.eval_seeds)
    config = DrQv2Config(
        device=args.device,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
    )
    run_config = {
        "algorithm": "drq-v2",
        "total_steps": args.total_steps,
        "track_ids": track_ids,
        "track_sampler_seed": args.track_sampler_seed,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "seed": args.seed,
        "drq_config": config.__dict__,
        "reward_contract": {"reward_shaping": False, "norm_reward": False},
    }
    run_dir = tracking.new_run(args.name, run_config)
    environment = build_sampled_env(
        track_ids,
        args.track_sampler_seed,
        args.max_steps,
        args.frame_skip,
        reward_shaping=False,
        excluded_seeds=(),
        obstacles=True,
    )
    agent = DrQv2Agent(config, seed=args.seed)
    collector = EpisodeCollector(environment, action_adapter=agent.action_adapter, gamma=config.gamma)
    observation, _ = collector.reset()
    rng = np.random.default_rng(args.seed)
    latest_metrics = {}
    for step in range(args.total_steps):
        if step < args.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        else:
            action = agent.act(observation, deterministic=False)
        transition = collector.step(action)
        agent.observe(transition)
        for _ in range(args.updates_per_step):
            latest_metrics = agent.update()
        if transition.done:
            observation, _ = collector.reset()
        else:
            observation = transition.next_observation
        if (step + 1) % 1000 == 0 or step + 1 == args.total_steps:
            record = {"step": step + 1, **latest_metrics}
            tracking.log_metrics(run_dir, record)
            print(json.dumps(record, sort_keys=True))
    source_paths = ["common_adapter.py", "drq_v2.py", "train.py", "env_wrapper.py", "damage.py"]
    checkpoint = agent.save_checkpoint(
        run_dir / "checkpoint.pt",
        source_paths=source_paths,
        run_metadata={**run_config, "seeds": [args.seed]},
    )
    restored = DrQv2Agent(config, seed=args.seed)
    restored.load_checkpoint(checkpoint)
    before = agent.act(observation, deterministic=True)
    after = restored.act(observation, deterministic=True)
    if not np.allclose(before, after, atol=1e-6):
        raise RuntimeError("checkpoint restore action parity failed")
    export_path = agent.export_actor(run_dir / "actor.pt")
    exported_actor, exported_adapter, _spec = load_exported_actor(export_path, device="cpu")
    cpu_policy = CPUActorAdapter(exported_actor, exported_adapter)
    evaluation = []
    for track_id in eval_track_ids:
        for seed in eval_seeds:
            eval_env = build_env(track_id, seed, args.max_steps, args.frame_skip, False)
            eval_collector = EpisodeCollector(eval_env, action_adapter=exported_adapter, gamma=config.gamma)
            eval_observation, _ = eval_collector.reset()
            total_reward = 0.0
            transition = None
            while transition is None or not transition.done:
                transition = eval_collector.step(
                    cpu_policy.native_action(eval_observation, deterministic=True)
                )
                total_reward += transition.reward
                eval_observation = transition.next_observation
            info = transition.info
            evaluation.append({
                "track_id": track_id,
                "seed": seed,
                "steps": transition.step + 1,
                "reward": total_reward,
                "progress": float(info.get("progress", 0.0)),
                "finished": bool(info.get("finished", False)),
                "lap_time_ms": info.get("lap_time_ms"),
                "damage": float(info.get("damage", 0.0)),
                "retire_reason": info.get("retire_reason"),
            })
            eval_env.close()
    tracking.write_json(run_dir / "smoke_result.json", {
        "checkpoint": str(checkpoint),
        "actor": str(export_path),
        "cpu_action_parity": True,
        "environment_steps": agent.environment_steps,
        "gradient_steps": agent.gradient_steps,
        "cpu_export_evaluation": summarize(evaluation),
    })
    environment.close()
    print(f"DrQ-v2 smoke complete: {run_dir}")


if __name__ == "__main__":
    main()
