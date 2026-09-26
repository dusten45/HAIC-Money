import argparse
import copy
from dataclasses import asdict
import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

try:
    import resource
except ImportError:
    resource = None

import tracking
from action_smoothing import normalize_action_smoothing
from common_adapter import EpisodeCollector
from dreamer_v3 import (
    DreamerV3Agent,
    DreamerV3Config,
    ExportedDreamerV3Actor,
    load_exported_actor,
)
from train import build_sampled_env, file_sha256

ROOT = Path(__file__).resolve().parent


def parse_ints(value):
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train DreamerV3 with exported CPU checkpoint selection")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", type=Path, help="new, explicit output directory; never overwritten")
    parser.add_argument("--resume", type=Path, help="explicit checkpoint with full trainer state")
    parser.add_argument("--total-steps", type=int, required=True, help="absolute target environment-step count")
    parser.add_argument("--track-ids", default="1,2,3,4")
    parser.add_argument("--track-sampler-seed", type=int, default=917)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--policy-start-step", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--burnin-steps", type=int, default=8)
    parser.add_argument("--terminal-window-fraction", type=float, default=0.0)
    parser.add_argument("--reset-start-fraction", type=float, default=0.0)
    parser.add_argument("--short-episode-fraction", type=float, default=0.0)
    parser.add_argument("--observation-loss-scale", type=float, default=1.0)
    parser.add_argument("--kl-free-nats", type=float, default=1.0)
    parser.add_argument("--overshoot-horizon", type=int, default=1)
    parser.add_argument("--overshoot-kl-weight", type=float, default=0.0)
    parser.add_argument("--overshoot-free-nats", type=float, default=0.0)
    parser.add_argument("--continue-positive-weight", type=float, default=1.0)
    parser.add_argument("--replay-pretrain-updates", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=32768)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--protocol-file", type=Path)
    parser.add_argument("--evaluations-dir", type=Path, default=ROOT / "evaluations")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-screen", action="store_true", help="pretraining-only run; do not evaluate a policy")
    parser.add_argument("--eval-track-ids", default="4,5,6")
    parser.add_argument("--eval-seeds", default="14001,14002,14003,14004")
    return parser.parse_args()


def training_protocol(args):
    if args.protocol_file:
        protocol = json.loads(args.protocol_file.read_text())
    else:
        protocol = {
            "name": "dreamerv3-smoke",
            "purpose": "harness-smoke",
            "frame_skip": args.frame_skip,
            "max_steps": args.max_steps,
            "partitions": {"screen": {
                "track_ids": parse_ints(args.eval_track_ids),
                "seeds": parse_ints(args.eval_seeds),
                "repeats": 2,
            }},
        }
    if protocol.get("frame_skip") != args.frame_skip or protocol.get("max_steps") != args.max_steps:
        raise ValueError("protocol horizon/frame skip must match training")
    source_hashes = protocol.get("source_sha256")
    if args.protocol_file and not source_hashes:
        raise ValueError("frozen protocol must provide non-empty source_sha256")
    if source_hashes is not None:
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError("source_sha256 must be a non-empty path-to-hash object")
        for relative_path, expected in source_hashes.items():
            if not isinstance(relative_path, str) or not isinstance(expected, str) or len(expected) != 64:
                raise ValueError("source_sha256 entries must use relative paths and SHA-256 hex digests")
            source_path = (ROOT / relative_path).resolve()
            try:
                source_path.relative_to(ROOT.resolve())
            except ValueError as error:
                raise ValueError("protocol source paths must remain inside the repository") from error
            if not source_path.is_file() or file_sha256(source_path) != expected:
                raise ValueError(f"frozen source hash mismatch: {relative_path}")
    if "training_track_ids" in protocol and protocol["training_track_ids"] != parse_ints(args.track_ids):
        raise ValueError("protocol training_track_ids must match --track-ids")
    learner_seeds = protocol.get("learner_seeds")
    if learner_seeds is not None:
        if (
            not isinstance(learner_seeds, list)
            or not learner_seeds
            or any(type(seed) is not int or seed < 0 for seed in learner_seeds)
            or len(set(learner_seeds)) != len(learner_seeds)
        ):
            raise ValueError("protocol learner_seeds must be unique nonnegative integers")
        if getattr(args, "seed", None) is not None and args.seed not in learner_seeds:
            raise ValueError("--seed must be declared in protocol learner_seeds")
    partitions = protocol["partitions"]
    if "screen" not in partitions:
        raise ValueError("protocol must contain a screen partition")
    reserved = set()
    for partition in partitions.values():
        seeds, tracks = partition["seeds"], partition["track_ids"]
        if not seeds or not tracks or partition["repeats"] < 2:
            raise ValueError("each partition needs cells and at least two CPU reload repeats")
        if not isinstance(partition["repeats"], int) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
            raise ValueError("protocol repeats and geometry seeds must be valid integers")
        if any(type(track) is not int or track < 1 for track in tracks):
            raise ValueError("protocol track IDs must be positive integers")
        if len(set(seeds)) != len(seeds) or len(set(tracks)) != len(tracks):
            raise ValueError("duplicate protocol cells")
        if reserved.intersection(seeds):
            raise ValueError("partition seeds must be disjoint, not just track/seed pairs")
        reserved.update(seeds)
    extra_reserved = protocol.get("reserved_training_seeds", [])
    if not isinstance(extra_reserved, list) or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in extra_reserved):
        raise ValueError("reserved_training_seeds must be a list of valid uint32 integers")
    if len(set(extra_reserved)) != len(extra_reserved):
        raise ValueError("reserved_training_seeds must not contain duplicates")
    reserved.update(extra_reserved)

    development = protocol.get("training_development")
    if development is not None:
        if not isinstance(development, dict):
            raise ValueError("training_development must be an object")
        dev_seeds = development.get("seeds", [])
        dev_tracks = development.get("track_ids", [])
        dev_cells = development.get("cells", [])
        if (
            not isinstance(dev_seeds, list)
            or not isinstance(dev_tracks, list)
            or not isinstance(dev_cells, list)
            or not dev_seeds
            or not dev_tracks
            or not dev_cells
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in dev_seeds)
            or any(type(track) is not int or track < 1 for track in dev_tracks)
            or len(set(dev_seeds)) != len(dev_seeds)
            or len(set(dev_tracks)) != len(dev_tracks)
        ):
            raise ValueError("training_development requires unique positive tracks and uint32 seeds")
        cell_pairs = []
        for cell in dev_cells:
            if not isinstance(cell, dict):
                raise ValueError("training_development cells must be objects")
            track, seed = cell.get("track_id"), cell.get("seed")
            if type(track) is not int or track not in dev_tracks:
                raise ValueError("training_development cell uses an undeclared track ID")
            if type(seed) is not int or seed not in dev_seeds:
                raise ValueError("training_development cell uses an undeclared seed")
            cell_pairs.append((track, seed))
        if (
            len(set(cell_pairs)) != len(cell_pairs)
            or len({seed for _, seed in cell_pairs}) != len(cell_pairs)
            or set(dev_seeds) != {seed for _, seed in cell_pairs}
            or set(dev_tracks) != {track for track, _ in cell_pairs}
        ):
            raise ValueError("training_development cells must cover each declared track and seed once")
        if set(dev_seeds).intersection(reserved - set(extra_reserved)):
            raise ValueError("training_development seeds must be disjoint from evaluator partitions")
        if not set(dev_seeds).issubset(extra_reserved):
            raise ValueError("all training_development seeds must be in reserved_training_seeds")

    budget = {
        "total_environment_decisions": getattr(args, "total_steps", None),
        "random_prefill_decisions": getattr(args, "warmup_steps", None),
        "policy_start_step": getattr(args, "policy_start_step", None),
        "replay_pretrain_updates": getattr(args, "replay_pretrain_updates", None),
        "online_updates_per_decision": getattr(args, "updates_per_step", None),
        "sequence_length": getattr(args, "seq_len", None),
        "burnin_steps": getattr(args, "burnin_steps", None),
        "terminal_window_fraction": getattr(args, "terminal_window_fraction", None),
        "reset_start_fraction": getattr(args, "reset_start_fraction", 0.0),
        "short_episode_fraction": getattr(args, "short_episode_fraction", 0.0),
        "observation_loss_scale": getattr(args, "observation_loss_scale", None),
        "kl_free_nats": getattr(args, "kl_free_nats", None),
        "overshoot_horizon": getattr(args, "overshoot_horizon", None),
        "overshoot_kl_weight": getattr(args, "overshoot_kl_weight", None),
        "overshoot_free_nats": getattr(args, "overshoot_free_nats", None),
        "continue_positive_weight": getattr(args, "continue_positive_weight", None),
        "batch_size": getattr(args, "batch_size", None),
        "replay_capacity": getattr(args, "replay_capacity", None),
        "track_sampler_seed": getattr(args, "track_sampler_seed", None),
        "device": getattr(args, "device", None),
    }
    if all(value is not None for value in budget.values()):
        recorded_budget = protocol.get("training_budget")
        if args.protocol_file and recorded_budget != budget:
            raise ValueError("protocol training_budget must match the requested training command")
        protocol["training_budget"] = budget
    return protocol, sorted(reserved)


def selection_score(result):
    if not result["eligible"] or not result["determinism_audited"]:
        raise RuntimeError("CPU export failed operational or deterministic reload gate")
    metrics = result["summary"]
    lap = metrics["avg_lap_time_ms"]
    values = [metrics["finish_rate"], metrics["avg_progress"]]
    if not np.isfinite(values).all() or (lap is not None and not np.isfinite(lap)):
        raise ValueError("non-finite CPU selection metrics")
    return (*values, -lap if lap is not None else float("-inf"))


def verify_checkpoint(agent, checkpoint, actor_path, observation):
    """Check actual restore/export actions without consuming the training RNG stream."""
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    online_state = (
        agent._online_h.clone(),
        agent._online_z.clone(),
        agent._online_prev_a.clone(),
        agent._online_first,
        agent._online_observation_pending,
    )
    policy_stats = (
        None
        if agent._last_policy_stats is None
        else {key: value.copy() for key, value in agent._last_policy_stats.items()}
    )
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            restored = DreamerV3Agent(agent.config)
            restored.load_checkpoint(checkpoint)
            actor = load_exported_actor(actor_path, device="cpu")
            observations = [observation, np.zeros_like(observation), np.ones_like(observation)]
            max_restore_error = max_cpu_error = 0.0
            for _ in range(2):
                agent.reset_episode()
                restored.reset_episode()
                actor.reset_episode()
                for value in observations:
                    expected = agent.act(value, deterministic=True)
                    actual = restored.act(value, deterministic=True)
                    obs_t = torch.as_tensor(np.ascontiguousarray(value)).float().unsqueeze(0)
                    exported = actor.act(obs_t, deterministic=True)
                    max_restore_error = max(max_restore_error, float(np.max(np.abs(expected - actual))))
                    max_cpu_error = max(max_cpu_error, float(np.max(np.abs(expected - exported))))
                    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
                    np.testing.assert_allclose(exported, expected, atol=1e-4, rtol=0)
            return {
                "restore_max_abs_error": max_restore_error,
                "cpu_export_max_abs_error": max_cpu_error,
                "observations_per_reset": len(observations),
                "resets": 2,
            }
    finally:
        with torch.no_grad():
            agent._online_h.copy_(online_state[0])
            agent._online_z.copy_(online_state[1])
            agent._online_prev_a.copy_(online_state[2])
        agent._online_first = online_state[3]
        agent._online_observation_pending = online_state[4]
        agent._last_policy_stats = policy_stats
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def evaluate_checkpoint(actor_path, protocol_path, output, args):
    command = [
        str(args.eval_python), str(ROOT / "evaluate_policy.py"),
        "--model", str(actor_path.resolve()),
        "--run-dir", str(protocol_path.parent.resolve()),
        "--protocol-file", str(protocol_path.resolve()), "--partition", "screen",
        "--output", str(output.resolve()),
        "--max-steps", str(args.max_steps), "--frame-skip", str(args.frame_skip),
        "--evaluations-dir", str(args.evaluations_dir.resolve()),
        "--workers", str(args.eval_workers),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    report = json.loads(output.read_text())
    if report.get("partition") != "screen" or report.get("protocol_sha256") != file_sha256(protocol_path):
        raise RuntimeError("CPU evaluation protocol does not match the frozen selection screen")
    if len(report["ranked"]) != 1:
        raise RuntimeError("checkpoint evaluation must describe exactly one exported actor")
    result = report["ranked"][0]
    if result["archive_sha256"] != file_sha256(actor_path):
        raise RuntimeError("evaluated CPU actor does not match saved checkpoint export")
    selection_score(result)
    return report


def check_evaluation_runtime(python, protocol):
    with tempfile.TemporaryDirectory(prefix="haic-dreamer-preflight-") as directory:
        path = Path(directory) / "protocol.json"
        tracking.write_json(path, protocol)
        completed = subprocess.run(
            [str(python), str(ROOT / "evaluate_policy.py"), "--check-runtime", "--protocol-file", str(path)],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
        )
    return json.loads(completed.stdout)


def training_memory():
    fields = {}
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                fields[line.split(":")[0]] = int(line.split()[1]) / 1024
    peak_rss_mib = fields.get("VmHWM")
    if peak_rss_mib is None:
        if resource is not None:
            peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        else:
            from evaluate_policy import peak_rss_bytes
            peak_rss_mib = peak_rss_bytes() / (1024 * 1024)
    return {"rss_mib": fields.get("VmRSS"), "rss_peak_mib": peak_rss_mib}


def save_and_select(agent, observation, run_dir, run_config, args, best, trainer_state):
    step = agent.environment_steps
    directory = run_dir / "checkpoints" / f"step-{step:09d}"
    directory.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / name for name in (
        "common_adapter.py", "dreamer_v3.py", "train_dreamerv3.py", "train.py", "tracking.py",
        "agent.py", "evaluate_policy.py", "env_wrapper.py", "damage.py", "requirements.txt",
    )] + sorted((ROOT / "core").rglob("*.py"))
    checkpoint = agent.save_checkpoint(
        directory / "checkpoint.pt", source_paths=sources,
        run_metadata={**run_config, "seeds": [args.seed],
                      "restore_scope": "learner/replay/RNG plus recurrent sequence state"},
        trainer_state=trainer_state,
    )
    actor_path = agent.export_actor(directory / "actor.pt")
    parity = verify_checkpoint(agent, checkpoint, actor_path, observation)
    report = evaluate_checkpoint(actor_path, run_dir / "protocol.json", directory / "evaluation.json", args)
    result = report["ranked"][0]
    record = {
        "step": step, "gradient_steps": agent.gradient_steps, "replay_size": agent.replay.size,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "actor": str(actor_path.resolve()), "actor_sha256": file_sha256(actor_path),
        "evaluation_dir": report["evaluation_dir"], "cpu_result": result, "parity": parity,
        "selection_order": ["unseen_finish_rate", "progress", "completed_lap_time"],
        "protocol_sha256": file_sha256(run_dir / "protocol.json"),
    }
    record["selected"] = best is None or selection_score(result) > selection_score(best["cpu_result"])
    tracking.write_json(directory / "selection.json", record)
    tracking.log_metrics(run_dir, record)
    if record["selected"]:
        tracking.write_json(run_dir / "selection.json", record)
        best = record
    print(json.dumps({"checkpoint_step": step, "selected": record["selected"],
                      "cpu_summary": result["summary"]}), flush=True)
    return best


def main():
    args = parse_args()
    if min(args.total_steps, args.updates_per_step, args.max_steps, args.torch_threads, args.eval_workers) <= 0 or args.eval_freq < 0:
        raise ValueError("invalid step, update, horizon, thread or evaluation interval")
    if args.frame_skip != 4:
        raise ValueError("the first algorithm comparison freezes frame_skip=4")
    if any(path and "_latest" in path.parts for path in (args.resume, args.run_dir, args.protocol_file, args.evaluations_dir)):
        raise ValueError("DreamerV3 training requires explicit paths, not _latest")
    if args.run_dir and args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    track_ids = parse_ints(args.track_ids)
    if min(track_ids) < 1 or len(set(track_ids)) != len(track_ids):
        raise ValueError("training track IDs must be positive and unique")
    if args.burnin_steps < 0 or args.seq_len <= 0 or args.batch_size <= 0:
        raise ValueError("batch size and sequence length must be positive; burn-in cannot be negative")
    if not 0.0 <= args.terminal_window_fraction <= 1.0:
        raise ValueError("terminal_window_fraction must be within [0,1]")
    if not 0.0 <= args.reset_start_fraction <= 1.0:
        raise ValueError("reset_start_fraction must be within [0,1]")
    if not 0.0 <= args.short_episode_fraction <= 1.0:
        raise ValueError("short_episode_fraction must be within [0,1]")
    if args.observation_loss_scale <= 0.0:
        raise ValueError("observation_loss_scale must be positive")
    if args.kl_free_nats < 0.0:
        raise ValueError("kl_free_nats cannot be negative")
    if args.overshoot_horizon < 1 or args.overshoot_kl_weight < 0.0 or args.overshoot_free_nats < 0.0 or (
        args.overshoot_kl_weight > 0.0 and args.overshoot_horizon < 2
    ):
        raise ValueError("overshooting requires horizon >=2 and nonnegative weight")
    if args.continue_positive_weight <= 0.0:
        raise ValueError("continue_positive_weight must be positive")
    if args.warmup_steps < 0 or args.replay_pretrain_updates < 0 or (
        args.replay_pretrain_updates
        and (args.warmup_steps <= 0 or args.warmup_steps > args.total_steps)
    ):
        raise ValueError(
            "warmup must be nonnegative; replay pretraining requires warmup within the run and nonnegative updates"
        )
    if args.policy_start_step is None:
        args.policy_start_step = args.warmup_steps
    if args.policy_start_step < args.warmup_steps:
        raise ValueError("policy_start_step cannot precede random prefill")
    if args.skip_screen and args.policy_start_step <= args.total_steps:
        raise ValueError("--skip-screen is only allowed when no policy action is trained in this run")
    if args.replay_capacity < max(
        args.batch_size * (args.seq_len + args.burnin_steps), args.warmup_steps
    ):
        raise ValueError("replay capacity must reach the batch and warmup thresholds")

    protocol, excluded_seeds = training_protocol(args)
    evaluation_runtime = (
        {"status": "not_checked", "reason": "pretraining-only run skips CPU screen evaluation"}
        if args.skip_screen
        else check_evaluation_runtime(args.eval_python, protocol)
    )

    config = DreamerV3Config(
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        burnin_steps=args.burnin_steps,
        terminal_window_fraction=args.terminal_window_fraction,
        reset_start_fraction=args.reset_start_fraction,
        short_episode_fraction=args.short_episode_fraction,
        observation_loss_scale=args.observation_loss_scale,
        kl_free_nats=args.kl_free_nats,
        overshoot_horizon=args.overshoot_horizon,
        overshoot_kl_weight=args.overshoot_kl_weight,
        overshoot_free_nats=args.overshoot_free_nats,
        continue_positive_weight=args.continue_positive_weight,
        warmup_steps=args.warmup_steps,
        replay_capacity=args.replay_capacity,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
    )
    run_config = {
        "algorithm": "dreamerv3",
        "name": args.name,
        "seed": args.seed,
        "device": args.device,
        "track_ids": track_ids,
        "track_sampler_seed": args.track_sampler_seed,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "policy_start_step": args.policy_start_step,
        "skip_screen": args.skip_screen,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "burnin_steps": args.burnin_steps,
        "terminal_window_fraction": args.terminal_window_fraction,
        "reset_start_fraction": args.reset_start_fraction,
        "short_episode_fraction": args.short_episode_fraction,
        "observation_loss_scale": args.observation_loss_scale,
        "kl_free_nats": args.kl_free_nats,
        "overshoot_horizon": args.overshoot_horizon,
        "overshoot_kl_weight": args.overshoot_kl_weight,
        "overshoot_free_nats": args.overshoot_free_nats,
        "continue_positive_weight": args.continue_positive_weight,
        "replay_pretrain_updates": args.replay_pretrain_updates,
        "replay_capacity": args.replay_capacity,
        "dreamer_config": asdict(config),
        "updates_per_step": args.updates_per_step,
        "eval_freq": args.eval_freq,
        "eval_python": str(Path(args.eval_python).absolute()),
        "eval_workers": args.eval_workers,
        "evaluation_runtime": evaluation_runtime,
        "protocol": protocol,
        "excluded_training_seeds": excluded_seeds,
        "training_track_mode": "sampled", "training_obstacles": "official",
        "reward_contract": {"reward_shaping": False, "norm_reward": False, "collision_penalty": 0.0},
        "action_smoothing": normalize_action_smoothing(), "observation_channels": 4,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                    "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark},
    }

    if args.run_dir:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        tracking.write_json(run_dir / "config.json", {
            "name": args.name, "config": run_config, "git": tracking.git_info(),
            "command_line": shlex.join(sys.argv), "pip_freeze": tracking.pip_freeze(),
        })
    else:
        run_dir = tracking.new_run(args.name, run_config, command_line=shlex.join(sys.argv)).resolve()

    if args.protocol_file:
        frozen_bytes = args.protocol_file.read_bytes()
        (run_dir / "protocol.json").write_bytes(frozen_bytes)
    else:
        tracking.write_json(run_dir / "protocol.json", protocol)
    print(f"run_dir: {run_dir}", flush=True)

    environment = build_sampled_env(
        track_ids,
        args.track_sampler_seed,
        args.max_steps,
        args.frame_skip,
        reward_shaping=False,
        excluded_seeds=excluded_seeds,
        obstacles=True,
    )
    agent = DreamerV3Agent(config, seed=args.seed)
    collector = EpisodeCollector(environment, action_adapter=agent.action_adapter, gamma=config.gamma)
    observation, reset_info = collector.reset()
    agent.reset_episode()

    latest_metrics, best = {}, None
    started = time.perf_counter()
    episode_reward = 0.0
    episode_actions, episode_proposed_actions, episode_policy_stats = [], [], []

    try:
        with (run_dir / "episodes.jsonl").open("x") as episodes:
            def log_episode(value):
                episodes.write(json.dumps(value, sort_keys=True) + "\n")
                episodes.flush()

            log_episode({"event": "reset", "episode_id": collector.episode_id, **reset_info})
            for step in range(agent.environment_steps, args.total_steps):
                if step < args.policy_start_step:
                    action = np.random.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    policy_stats = None
                else:
                    action = agent.act(observation, deterministic=False)
                    policy_stats = agent._last_policy_stats

                transition = collector.step(action)
                agent.observe(transition)
                episode_reward += transition.reward
                episode_actions.append(transition.action)
                episode_proposed_actions.append(np.asarray(action, dtype=np.float32).copy())
                episode_policy_stats.append(
                    None if policy_stats is None else {
                        key: value.copy() for key, value in policy_stats.items()
                    }
                )

                if step + 1 == args.warmup_steps:
                    for pretrain_step in range(args.replay_pretrain_updates):
                        pretrain_metrics = agent.update(model_only=True)
                        if not pretrain_metrics:
                            raise RuntimeError(
                                "replay-only pretraining found no valid same-episode sequence"
                            )
                        pretrain_record = {
                            "step": step + 1,
                            "replay_pretrain_update": pretrain_step + 1,
                            **pretrain_metrics,
                            **training_memory(),
                        }
                        tracking.log_metrics(run_dir, pretrain_record)
                        print(json.dumps(pretrain_record, sort_keys=True), flush=True)

                for _ in range(args.updates_per_step):
                    completed_decisions = step + 1
                    if completed_decisions <= args.warmup_steps:
                        latest_metrics = {}
                    elif completed_decisions < args.policy_start_step:
                        latest_metrics = agent.update(model_only=True)
                    else:
                        latest_metrics = agent.update()

                if transition.done:
                    actions = np.asarray(episode_actions)
                    proposed_actions = np.asarray(episode_proposed_actions)
                    policy_stats = [stats for stats in episode_policy_stats if stats is not None]
                    actor_stats = {
                        f"policy_{key}_mean": np.stack([stats[key] for stats in policy_stats])
                        .mean(axis=0).tolist()
                        for key in ("mean", "std", "entropy")
                    } if policy_stats else {
                        f"policy_{key}_mean": None for key in ("mean", "std", "entropy")
                    }
                    log_episode({
                        "event": "end", "episode_id": transition.episode_id, "global_step": step + 1,
                        "track_id": transition.info["track_id"], "seed": transition.info["seed"],
                        "steps": transition.step + 1, "reward": episode_reward,
                        "terminated": transition.terminated, "truncated": transition.truncated,
                        "terminal": transition.terminal,
                        **{key: transition.info.get(key) for key in ("finished", "progress", "damage", "retire_reason")},
                        "native_action_mean": actions.mean(axis=0).tolist(),
                        "native_action_std": actions.std(axis=0).tolist(),
                        "native_saturation_fraction": (np.abs(actions) >= .99).mean(axis=0).tolist(),
                        "proposed_action_mean": proposed_actions.mean(axis=0).tolist(),
                        "proposed_action_std": proposed_actions.std(axis=0).tolist(),
                        "proposed_out_of_bounds_fraction": (np.abs(proposed_actions) > 1.0).mean(axis=0).tolist(),
                        **actor_stats,
                    })
                    episode_reward = 0.0
                    episode_actions, episode_proposed_actions, episode_policy_stats = [], [], []
                    observation, reset_info = collector.reset()
                    agent.reset_episode()
                    log_episode({"event": "reset", "episode_id": collector.episode_id, **reset_info})
                else:
                    observation = transition.next_observation

                if (step + 1) % 1000 == 0 or step + 1 == args.total_steps:
                    record = {
                        "step": step + 1, **latest_metrics,
                        "replay_bytes": agent.replay.memory_bytes,
                        **training_memory(),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    tracking.log_metrics(run_dir, record)
                    print(json.dumps(record, sort_keys=True), flush=True)

                if (args.eval_freq and (step + 1) % args.eval_freq == 0) or step + 1 == args.total_steps:
                    if args.skip_screen:
                        if step + 1 == args.total_steps:
                            checkpoint_path = run_dir / "pretraining-checkpoint.pt"
                            actor_path = run_dir / "pretraining-actor.pt"
                            run_metadata = {
                                "run_config": run_config,
                                "protocol_sha256": file_sha256(run_dir / "protocol.json"),
                            }
                            agent.save_checkpoint(
                                checkpoint_path,
                                run_metadata=run_metadata,
                            )
                            agent.export_actor(actor_path)
                            parity = verify_checkpoint(
                                agent, checkpoint_path, actor_path, observation
                            )
                            result_record = {
                                "format": "haic-dreamerv3-pretraining-v1",
                                "status": "pretraining_only",
                                "screen_evaluation": "not_run_by_design",
                                "environment_steps": agent.environment_steps,
                                "gradient_steps": agent.gradient_steps,
                                "checkpoint_sha256": file_sha256(checkpoint_path),
                                "actor_sha256": file_sha256(actor_path),
                                "checkpoint_cpu_parity": parity,
                                "runtime_eligibility": evaluation_runtime,
                            }
                            tracking.write_json(run_dir / "pretraining-result.json", result_record)
                        continue
                    trainer_state = {
                        "format": "haic-dreamerv3-trainer-v1", "run_config": run_config,
                        "episode_id": collector.episode_id, "reset_info": reset_info,
                        "episode_actions": episode_actions, "episode_reward": episode_reward,
                        "observation": observation, "selected_checkpoint": best,
                    }
                    best = save_and_select(agent, observation, run_dir, run_config, args, best, trainer_state)

        tracking.write_json(run_dir / "result.json", {
            "environment_steps": agent.environment_steps, "gradient_steps": agent.gradient_steps,
            "selected_checkpoint": best, "wall_seconds": time.perf_counter() - started,
        })
    finally:
        environment.close()
    print(f"DreamerV3 training complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
