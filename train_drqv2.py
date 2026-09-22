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
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None

import tracking
from action_smoothing import normalize_action_smoothing
from common_adapter import CPUActorAdapter, EpisodeCollector
from drq_v2 import DrQv2Agent, DrQv2Config, load_exported_actor
from train import build_sampled_env, file_sha256


ROOT = Path(__file__).resolve().parent


def parse_ints(value):
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description="Train DrQ-v2 with exported CPU checkpoint selection")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-dir", type=Path, help="new, explicit output directory; never overwritten")
    parser.add_argument("--resume", type=Path, help="explicit checkpoint with full trainer state; legacy checkpoints are evaluation-only")
    parser.add_argument("--total-steps", type=int, required=True, help="absolute target environment-step count, including restored steps")
    parser.add_argument("--track-ids", default="1,2,3,4")
    parser.add_argument("--track-sampler-seed", type=int, default=917)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--steering-logit-l2", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=32768)
    parser.add_argument("--eval-python", default=sys.executable)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--protocol-file", type=Path)
    parser.add_argument("--evaluations-dir", type=Path, default=ROOT / "evaluations")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--eval-track-ids", default="4,5,6")
    parser.add_argument("--eval-seeds", default="14001,14002,14003,14004")
    return parser.parse_args()


def training_protocol(args):
    if args.protocol_file:
        protocol = json.loads(args.protocol_file.read_text())
    else:
        # The small legacy CLI grid is a harness screen, never a promotion protocol.
        protocol = {
            "name": "drqv2-smoke",
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
    if "training_track_ids" in protocol and protocol["training_track_ids"] != parse_ints(args.track_ids):
        raise ValueError("protocol training_track_ids must match --track-ids")
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
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            restored = DrQv2Agent(agent.config)
            restored.load_checkpoint(checkpoint)
            cpu_reference = copy.deepcopy(restored.actor).cpu().eval()
            actor, adapter, spec = load_exported_actor(actor_path, device="cpu")
            policy = CPUActorAdapter(actor, adapter, spec)
            observations = [observation, np.zeros_like(observation), np.ones_like(observation)]
            max_restore_error = max_cpu_error = max_cross_device_error = 0.0
            for _ in range(2):
                policy.reset_episode()
                for value in observations:
                    expected = agent.act(value, deterministic=True)
                    actual = restored.act(value, deterministic=True)
                    exported = policy.native_action(value, deterministic=True)
                    with torch.inference_mode():
                        cpu_expected = cpu_reference(torch.as_tensor(value).unsqueeze(0)).squeeze(0).numpy()
                    max_restore_error = max(max_restore_error, float(np.max(np.abs(expected - actual))))
                    max_cpu_error = max(max_cpu_error, float(np.max(np.abs(cpu_expected - exported))))
                    max_cross_device_error = max(max_cross_device_error, float(np.max(np.abs(expected - exported))))
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
                    # Different CUDA/CPU kernels need not agree bitwise; CPU weights are authoritative.
                    np.testing.assert_allclose(exported, cpu_expected, atol=1e-6, rtol=0)
            return {"restore_max_abs_error": max_restore_error,
                    "cpu_export_max_abs_error": max_cpu_error,
                    "training_device_vs_cpu_max_abs_error": max_cross_device_error,
                    "observations_per_reset": len(observations), "resets": 2}
    finally:
        random.setstate(python_rng)


def evaluate_checkpoint(actor_path, protocol_path, output, args):
    # Selection must execute in the pinned CPU interpreter, never the CUDA trainer.
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
    with tempfile.TemporaryDirectory(prefix="haic-drq-preflight-") as directory:
        path = Path(directory) / "protocol.json"
        tracking.write_json(path, protocol)
        completed = subprocess.run(
            [str(python), str(ROOT / "evaluate_policy.py"), "--check-runtime", "--protocol-file", str(path)],
            cwd=ROOT, check=True, capture_output=True, text=True, timeout=60,
        )
    return json.loads(completed.stdout)


def training_memory():
    # Linux rusage can inherit a parent's high-water mark across exec.
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
    return {"rss_mib": fields.get("VmRSS"),
            "rss_peak_mib": peak_rss_mib}


def save_and_select(agent, observation, run_dir, run_config, args, best, trainer_state):
    step = agent.environment_steps
    directory = run_dir / "checkpoints" / f"step-{step:09d}"
    directory.mkdir(parents=True, exist_ok=False)
    sources = [ROOT / name for name in (
        "common_adapter.py", "drq_v2.py", "train_drqv2.py", "train.py", "tracking.py",
        "agent.py", "evaluate_policy.py", "env_wrapper.py", "damage.py", "requirements.txt",
    )] + sorted((ROOT / "core").rglob("*.py"))
    checkpoint = agent.save_checkpoint(
        directory / "checkpoint.pt", source_paths=sources,
        run_metadata={**run_config, "seeds": [args.seed],
                      "restore_scope": "learner/replay/RNG plus verified deterministic current-episode reconstruction"},
        trainer_state=trainer_state,
    )
    actor_path = agent.export_actor(directory / "actor.pt")
    parity = verify_checkpoint(agent, checkpoint, actor_path, observation)
    report = evaluate_checkpoint(actor_path, run_dir / "protocol.json", directory / "evaluation.json", args)
    result = report["ranked"][0]
    record = {
        "step": step, "checkpoint": str(checkpoint.resolve()),
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
        # This index is run-local; immutable checkpoint/evaluation records are authoritative.
        tracking.write_json(run_dir / "selection.json", record)
        best = record
    print(json.dumps({"checkpoint_step": step, "selected": record["selected"],
                      "cpu_summary": result["summary"]}), flush=True)
    return best


def restore_collector(collector, sampler_rng, warmup_rng, state, run_config):
    if state is None or state.get("format") != "haic-drq-trainer-v1":
        raise ValueError("training resume requires full trainer state; historical checkpoints are evaluation-only")
    for key in ("track_ids", "track_sampler_seed", "max_steps", "frame_skip", "seed",
                "updates_per_step", "excluded_training_seeds", "protocol", "reward_contract",
                "runtime", "training_source_sha256"):
        if state["run_config"][key] != run_config[key]:
            raise ValueError(f"resume training contract mismatch: {key}")
    sampler_rng.bit_generator.state = copy.deepcopy(state["sampler_before_reset"])
    collector.episode_id = state["episode_id"] - 1
    observation, info = collector.reset()
    if (info["track_id"], info["seed"]) != (state["reset_info"]["track_id"], state["reset_info"]["seed"]):
        raise RuntimeError("resume sampled-track reconstruction mismatch")
    for action in state["episode_actions"]:
        transition = collector.step(action)
        if transition.done:
            raise RuntimeError("resume action prefix unexpectedly ended the episode")
        observation = transition.next_observation
    if not np.array_equal(observation, state["observation"]):
        raise RuntimeError("resume physics/observation reconstruction mismatch")
    warmup_rng.bit_generator.state = copy.deepcopy(state["warmup_rng"])
    return observation, info


def resume_selection(checkpoint, state, protocol_path):
    completed = checkpoint.parent / "selection.json"
    if not completed.is_file():
        raise ValueError("resume requires the checkpoint's completed immutable CPU selection record")
    record = json.loads(completed.read_text())
    if record["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("resume selection checkpoint hash mismatch")
    candidates = []
    if state.get("selected_checkpoint") is not None:
        candidates.append(state["selected_checkpoint"])
    candidates.append(record)
    for candidate in candidates:
        if candidate["protocol_sha256"] != file_sha256(protocol_path):
            raise ValueError("resume selection protocol hash mismatch")
        if candidate["actor_sha256"] != file_sha256(Path(candidate["actor"])):
            raise ValueError("resume selected actor hash mismatch")
        if candidate["cpu_result"]["archive_sha256"] != candidate["actor_sha256"]:
            raise ValueError("resume CPU evaluation actor hash mismatch")
    return max(candidates, key=lambda candidate: selection_score(candidate["cpu_result"]))


def main():
    args = parse_args()
    if min(args.total_steps, args.updates_per_step, args.max_steps, args.torch_threads, args.eval_workers) <= 0 or args.eval_freq < 0:
        raise ValueError("invalid step, update, horizon, thread or evaluation interval")
    if args.frame_skip != 4:
        raise ValueError("the first algorithm comparison freezes frame_skip=4")
    if any(path and "_latest" in path.parts for path in (args.resume, args.run_dir, args.protocol_file, args.evaluations_dir)):
        raise ValueError("DrQ training requires explicit paths, not _latest")
    if args.run_dir and args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    track_ids = parse_ints(args.track_ids)
    if min(track_ids) < 1 or len(set(track_ids)) != len(track_ids):
        raise ValueError("training track IDs must be positive and unique")
    if args.replay_capacity < max(args.batch_size, args.warmup_steps):
        raise ValueError("replay capacity must reach the batch and warmup thresholds")
    protocol, excluded_seeds = training_protocol(args)
    evaluation_runtime = check_evaluation_runtime(args.eval_python, protocol)
    config = DrQv2Config(
        device=args.device,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        steering_logit_l2=args.steering_logit_l2,
    )
    run_config = {
        "algorithm": "drq-v2",
        "total_steps": args.total_steps,
        "track_ids": track_ids,
        "track_sampler_seed": args.track_sampler_seed,
        "max_steps": args.max_steps,
        "frame_skip": args.frame_skip,
        "seed": args.seed,
        "drq_config": asdict(config),
        "updates_per_step": args.updates_per_step,
        "eval_freq": args.eval_freq,
        "eval_workers": args.eval_workers,
        "eval_python": str(Path(args.eval_python).absolute()),
        "evaluation_runtime": evaluation_runtime,
        "protocol": protocol,
        "excluded_training_seeds": excluded_seeds,
        "training_track_mode": "sampled", "training_obstacles": "official",
        "track_schedule": "independent track and uint32 seed draws; identical episode-index stream across training seeds",
        "reward_contract": {"reward_shaping": False, "norm_reward": False, "collision_penalty": 0.0},
        "action_smoothing": normalize_action_smoothing(), "observation_channels": 4,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
                    "torch_threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark},
        "resume_from": str(args.resume.resolve()) if args.resume else None,
        "resume_sha256": file_sha256(args.resume) if args.resume else None,
        "training_source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in [ROOT / name for name in (
                "train_drqv2.py", "drq_v2.py", "common_adapter.py", "train.py", "env_wrapper.py", "damage.py",
            )] + sorted((ROOT / "core").rglob("*.py"))
        },
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
        if json.loads(frozen_bytes) != protocol:
            raise RuntimeError("protocol changed while initializing the run")
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
    agent = DrQv2Agent(config, seed=args.seed)
    collector = EpisodeCollector(environment, action_adapter=agent.action_adapter, gamma=config.gamma)
    rng = np.random.default_rng(args.seed)
    sampler_rng = environment.get_wrapper_attr("_rng")
    sampler_before_reset = copy.deepcopy(sampler_rng.bit_generator.state)
    episode_reward, episode_actions = 0.0, []
    latest_metrics, best, last_actor_metrics = {}, None, {}
    if args.resume:
        state = agent.load_checkpoint(args.resume)
        observation, reset_info = restore_collector(collector, sampler_rng, rng, state, run_config)
        sampler_before_reset = state["sampler_before_reset"]
        episode_reward = state["episode_reward"]
        episode_actions = list(state["episode_actions"])
        best = resume_selection(args.resume, state, run_dir / "protocol.json")
        last_actor_metrics = state.get("last_actor_metrics", {})
        tracking.write_json(run_dir / "selection.json", best)
    else:
        observation, reset_info = collector.reset()
    if args.total_steps <= agent.environment_steps:
        environment.close()
        raise ValueError("total-steps must exceed the restored environment-step count")
    started = time.perf_counter()
    try:
        with (run_dir / "episodes.jsonl").open("x") as episodes:
            def log_episode(value):
                episodes.write(json.dumps(value, sort_keys=True) + "\n")
                episodes.flush()

            log_episode({"event": "resume" if args.resume else "reset", "episode_id": collector.episode_id,
                         "global_step": agent.environment_steps, **reset_info})
            for step in range(agent.environment_steps, args.total_steps):
                if step < args.warmup_steps:
                    action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                else:
                    action = agent.act(observation, deterministic=False)
                transition = collector.step(action)
                agent.observe(transition)
                episode_reward += transition.reward
                episode_actions.append(transition.action)
                for _ in range(args.updates_per_step):
                    previous_updates = agent.gradient_steps
                    latest_metrics = agent.update()
                    if agent.gradient_steps > previous_updates and agent.gradient_steps % config.actor_update_frequency == 0:
                        last_actor_metrics = {
                            key: value for key, value in latest_metrics.items()
                            if key.startswith(("actor_", "steering_"))
                        }
                if transition.done:
                    actions = np.asarray(episode_actions)
                    log_episode({
                        "event": "end", "episode_id": transition.episode_id, "global_step": step + 1,
                        "track_id": transition.info["track_id"], "seed": transition.info["seed"],
                        "steps": transition.step + 1, "reward": episode_reward,
                        "terminated": transition.terminated, "truncated": transition.truncated,
                        "terminal": transition.terminal,
                        **{key: transition.info.get(key) for key in ("finished", "progress", "damage", "retire_reason")},
                        "native_action_mean": actions.mean(axis=0).tolist(),
                        "native_saturation_fraction": (np.abs(actions) >= .99).mean(axis=0).tolist(),
                        "steering_abs_ge_0_46_fraction": float((np.abs(actions[:, 0]) >= .46).mean()),
                    })
                    episode_reward, episode_actions = 0.0, []
                    sampler_before_reset = copy.deepcopy(sampler_rng.bit_generator.state)
                    observation, reset_info = collector.reset()
                    log_episode({"event": "reset", "episode_id": collector.episode_id, **reset_info})
                else:
                    observation = transition.next_observation
                if (step + 1) % 1000 == 0 or step + 1 == args.total_steps:
                    record = {"step": step + 1, **latest_metrics,
                              "exploration_std": agent.exploration_std(),
                              "last_updated_actor_loss": last_actor_metrics.get("actor_loss"),
                              "last_actor_update": last_actor_metrics,
                              "replay_bytes": agent.replay.memory_bytes,
                              **training_memory(),
                              "elapsed_seconds": time.perf_counter() - started}
                    tracking.log_metrics(run_dir, record)
                    print(json.dumps(record, sort_keys=True), flush=True)
                if (args.eval_freq and (step + 1) % args.eval_freq == 0) or step + 1 == args.total_steps:
                    trainer_state = {
                        "format": "haic-drq-trainer-v1", "run_config": run_config,
                        "sampler_before_reset": sampler_before_reset,
                        "warmup_rng": copy.deepcopy(rng.bit_generator.state),
                        "episode_id": collector.episode_id, "reset_info": reset_info,
                        "episode_actions": episode_actions, "episode_reward": episode_reward,
                        "observation": observation,
                        "selected_checkpoint": best,
                        "last_actor_metrics": last_actor_metrics,
                    }
                    best = save_and_select(agent, observation, run_dir, run_config, args, best, trainer_state)
        tracking.write_json(run_dir / "result.json", {
            "environment_steps": agent.environment_steps, "gradient_steps": agent.gradient_steps,
            "selected_checkpoint": best, "wall_seconds": time.perf_counter() - started,
            "promotion": "not determined by selection; requires fresh confirmation",
        })
    finally:
        environment.close()
    print(f"DrQ-v2 training complete: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
