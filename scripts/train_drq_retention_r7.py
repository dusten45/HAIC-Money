"""Train one separately frozen DrQ-v2 r7 source-retention arm on TRAIN only."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from common_adapter import EpisodeCollector
from drq_v2 import DrQCritic, Uint8Replay, load_exported_actor
from haic.algorithms.drq_v2.retention import RetentionReplaySampler, update_retained
from haic.algorithms.drq_v2.teacher_study import audit_source_actor_pair, fork_from_source
from scripts import train_drq_geometry_mix as r6


STUDY_ID = "drqv2-retention-r7"
RUN_ROOT = Path("runs/20260926-drqv2-retention-r7")
R6_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
DIAGNOSTIC_UPDATES = (0, 1, 2000, 6384, 11384, 14576, 22768)
SOURCE_FIELDS = ("frames", "actions", "rewards", "terminated", "truncated", "terminal",
                 "episode_ids", "episode_steps", "sequence_ids")


def sha(path: Path) -> str:
    return r6._sha256(path)


def pinned(root: Path, path: str, digest: str) -> Path:
    resolved = r6._repo_file(root, path, "pinned input")
    if sha(resolved) != digest:
        raise ValueError(f"pinned input SHA-256 mismatch: {path}")
    return resolved


def load_protocol(root: Path, path: Path) -> tuple[dict, dict, str]:
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("r7 protocol must remain in repository")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != "haic-drq-retention-study-v1" or protocol.get("study_id") != STUDY_ID:
        raise ValueError("not the separately frozen r7 protocol")
    if protocol.get("r6_protocol_sha256") != R6_SHA:
        raise ValueError("r7 reference is not the frozen r6 control")
    r6_path = pinned(root, protocol["r6_protocol_path"], R6_SHA)
    reference = json.loads(r6_path.read_text(encoding="utf-8"))
    r6._validate_protocol(reference)
    r6._validate_catalog(reference, root)
    r6._checkpoint_source_paths(root, reference)
    if protocol.get("run_root") != RUN_ROOT.as_posix():
        raise ValueError("r7 run_root differs from its frozen separate root")
    if protocol.get("r6_contract") != "inherit-learner-budget-geometry-environment-verbatim" or any(
        key in protocol for key in ("budgets", "learner", "variants", "environment")
    ):
        raise ValueError("r7 must inherit the pinned r6 learner, budget, variants and environment without overrides")
    if protocol.get("catalog_sha256") != reference["catalog"]["sha256"]:
        raise ValueError("r7 TRAIN catalog differs from r6")
    if protocol.get("diagnostic_manifest_sha256") != "fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380":
        raise ValueError("r7 TRAIN-DIAGNOSTIC manifest differs from r6")
    diag = protocol["diagnostic_cache"]
    pinned(root, diag["path"], diag["sha256"])
    pinned(root, diag["hybrid_result_path"], diag["hybrid_result_sha256"])
    gradient = protocol["gradient_probe"]
    receipt = json.loads(pinned(root, gradient["path"], gradient["sha256"]).read_text(encoding="utf-8"))
    if (receipt.get("lambda_preserve") != protocol.get("lambda_preserve")
            or receipt.get("environment_decisions") != 0
            or receipt.get("learner_updates") != 0
            or receipt.get("train_diagnostic_used") is not False):
        raise ValueError("lambda differs from TRAIN-only frozen gradient-scale receipt")
    if not isinstance(protocol.get("lambda_preserve"), (int, float)) or protocol["lambda_preserve"] <= 0:
        raise ValueError("positive single frozen lambda required")
    if protocol.get("diagnostic_updates") != list(DIAGNOSTIC_UPDATES):
        raise ValueError("policy drift schedule changed")
    expected = {(seed, variant, condition) for seed in (0, 1)
                for variant in r6.VARIANTS for condition in ("r7a", "r7b")}
    rows = protocol.get("runs")
    if not isinstance(rows, list) or len(rows) != 12 or {
        (row["source_seed"], row["variant"], row["condition"]) for row in rows
    } != expected:
        raise ValueError("r7 requires exactly two treatments for all six r6 arms")
    for row in rows:
        seed, variant, condition = row["source_seed"], row["variant"], row["condition"]
        if row.get("run_dir") != f"{RUN_ROOT}/learner-{seed}-{variant}-{condition}":
            raise ValueError("r7 run path differs from frozen 12-arm matrix")
        if row.get("rng_seeds") != {key: r6._arm(reference, seed, variant)[key] for key in r6._RNG_SEED_NAMES}:
            raise ValueError("r7 learner/geometry RNG seeds differ from r6")
    if set(protocol.get("source_replay", {})) != {"0", "1"}:
        raise ValueError("source replay must bind both original seed checkpoints")
    for seed in (0, 1):
        source = next(row for row in reference["source_actors"] if row["learner_seed"] == seed)
        replay = protocol["source_replay"][str(seed)]
        if (replay.get("checkpoint_path"), replay.get("checkpoint_sha256")) != (
            source["checkpoint_path"], source["checkpoint_sha256"]
        ):
            raise ValueError("source replay must be the original source-training checkpoint")
        ledger_path = f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}/episodes.jsonl"
        if replay.get("episode_ledger_path") != ledger_path:
            raise ValueError("source replay ledger does not match the original training run")
        pinned(root, ledger_path, replay["episode_ledger_sha256"])
    if not isinstance(protocol.get("code_sha256"), dict) or not protocol["code_sha256"]:
        raise ValueError("r7 executable source hashes are required")
    for relative, digest in protocol["code_sha256"].items():
        pinned(root, relative, digest)
    return protocol, reference, sha(path)


def load_source_replay(root: Path, source: dict, *, expected_sha: str, expected_seed: int,
                       excluded_roads: set[int]) -> Uint8Replay:
    path = pinned(root, source["checkpoint_path"], expected_sha)
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if payload.get("format") != "haic-drq-v2-checkpoint-v1" or payload.get("environment_steps") != 131072:
        raise ValueError("source replay checkpoint has wrong learner lineage")
    state = payload["replay"]
    replay = Uint8Replay(capacity=100000, action_dim=3, n_step=3, gamma=.99)
    replay.load_state_dict(state)
    if replay.size != 100000 or replay.newest_sequence != 131071 or replay.oldest_sequence != 31072:
        raise ValueError("source replay does not contain the expected original rolling window")
    if not np.array_equal(np.sort(replay.sequence_ids), np.arange(31072, 131072)):
        raise ValueError("source replay sequence ring is incomplete or overwritten")
    episode_path = root / f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{expected_seed}/episodes.jsonl"
    if not episode_path.is_file():
        raise ValueError("source episode provenance ledger missing")
    episodes = {}
    for line in episode_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "reset":
            episodes[int(row["episode_id"])] = int(row["seed"])
    if not episodes or set(map(int, np.unique(replay.episode_ids))) - set(episodes):
        raise ValueError("source replay contains a transition with no source episode reset")
    if set(episodes.values()) & excluded_roads:
        raise ValueError("source checkpoint ledger overlaps TRAIN-DIAGNOSTIC roads")
    for field in SOURCE_FIELDS:
        getattr(replay, field).flags.writeable = False
    del payload
    return replay


def _read_diagnostic_cache(path: Path, seed: int, families: dict[int, str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if "observations" not in archive or "source_seed" not in archive:
            raise ValueError("drift cache must contain observation pixels and source seed")
        selected = np.asarray(archive["source_seed"]) == seed
        images = np.asarray(archive["observations"])[selected]
        roads = np.asarray(archive["geometry_seed"])[selected]
        if images.ndim == 5 and images.shape[2:] == (4, 84, 84):
            steps_per_road = images.shape[1]
            roads = np.repeat(roads, steps_per_road)
            images = images.reshape(-1, 4, 84, 84)
        pixels = images
        if pixels.ndim != 4 or pixels.shape[1:] != (4, 84, 84) or pixels.dtype != np.float32 or len(pixels) < 20:
            raise ValueError("source-success drift cache has invalid pixel shape or no samples")
        if not set(map(int, roads)).issubset(families):
            raise ValueError("drift cache includes a road outside frozen TRAIN-DIAGNOSTIC")
        metadata = {"geometry_seed": roads, "family": np.asarray([families[int(road)] for road in roads])}
        if "decision" in archive:
            metadata["decision"] = np.asarray(archive["decision"])[selected].reshape(-1)
        return pixels.copy(), metadata


@torch.inference_mode()
def drift_snapshot(agent, teacher, source_critic, pixels: np.ndarray, metadata: dict,
                   *, update: int) -> dict:
    current_actions, source_actions = [], []
    cosines, current_q_source, current_q_current = [], [], []
    frozen_q_source, frozen_q_current = [], []
    for start in range(0, len(pixels), 64):
        obs = torch.as_tensor(pixels[start:start + 64], device=agent.device)
        original = teacher(obs)
        current = agent.actor(obs)
        source_actions.append(original.cpu().numpy())
        current_actions.append(current.cpu().numpy())
        cosines.extend(F.cosine_similarity(teacher.encoder(obs), agent.actor.encoder(obs)).cpu().tolist())
        current_q_source.extend(agent.critic_one(obs, original).cpu().tolist())
        current_q_current.extend(agent.critic_one(obs, current).cpu().tolist())
        frozen_q_source.extend(source_critic(obs, original).cpu().tolist())
        frozen_q_current.extend(source_critic(obs, current).cpu().tolist())
    source_action, action = np.concatenate(source_actions), np.concatenate(current_actions)
    difference = np.abs(action - source_action)
    return {
        "gradient_update": update,
        "source_seed": int(metadata["source_seed"]),
        "count": len(pixels),
        "steering_abs_mean": float(difference[:, 0].mean()),
        "brake_abs_mean": float(difference[:, 2].mean()),
        "native_l1_mean": float(difference.sum(axis=1).mean()),
        "native_l2_mean": float(np.linalg.norm(difference, axis=1).mean()),
        "axis_ge_0_10_fraction": float((difference >= .10).mean()),
        "any_axis_ge_0_10_fraction": float((difference >= .10).any(axis=1).mean()),
        "actor_feature_cosine_mean": float(np.mean(cosines)),
        "current_q_source_action_mean": float(np.mean(current_q_source)),
        "current_q_current_action_mean": float(np.mean(current_q_current)),
        "frozen_q_source_action_mean": float(np.mean(frozen_q_source)),
        "frozen_q_current_action_mean": float(np.mean(frozen_q_current)),
        "by_family": {str(family): {
            "count": int(np.count_nonzero(indices)),
            "steering_abs_mean": float(difference[indices, 0].mean()),
            "brake_abs_mean": float(difference[indices, 2].mean()),
            "any_axis_ge_0_10_fraction": float((difference[indices] >= .10).any(axis=1).mean()),
        } for family in np.unique(metadata["family"])
            for indices in [metadata["family"] == family]},
    }


def train(root: Path, protocol_path: Path, seed: int, variant: str, condition: str) -> dict:
    protocol, reference, protocol_sha = load_protocol(root, protocol_path)
    if (seed, variant, condition) not in {(r["source_seed"], r["variant"], r["condition"])
                                          for r in protocol["runs"]}:
        raise ValueError("r7 arm is not frozen in protocol")
    arm = r6._arm(reference, seed, variant)
    source = next(row for row in reference["source_actors"] if row["learner_seed"] == seed)
    source_path = pinned(root, source["checkpoint_path"], source["checkpoint_sha256"])
    actor_path = pinned(root, source["actor_path"], source["actor_sha256"])
    audit_source_actor_pair(source_path, actor_path, learner_seed=seed,
                            source_revision=source["source_revision"],
                            expected_checkpoint_sha256=source["checkpoint_sha256"],
                            expected_actor_sha256=source["actor_sha256"])
    if not torch.cuda.is_available():
        raise ValueError("r6 frozen CUDA training device is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    agent, fork = fork_from_source(source_path, actor_path, r6._agent_config(reference),
                                   learner_seed=seed, study_seed=arm["actor_rng_seed"],
                                   expected_checkpoint_sha256=source["checkpoint_sha256"],
                                   expected_actor_sha256=source["actor_sha256"])
    r6._assert_online_only_fork(agent)
    streams = r6._make_update_rngs(agent, arm)
    catalog = json.loads(pinned(root, reference["catalog"]["path"], reference["catalog"]["sha256"]).read_text())
    excluded = {int(row["geometry_seed"]) for row in catalog["train_diagnostic"]}
    source_replay = load_source_replay(root, source,
                                       expected_sha=protocol["source_replay"][str(seed)]["checkpoint_sha256"],
                                       expected_seed=seed, excluded_roads=excluded)
    sampler = RetentionReplaySampler(agent.replay, source_replay, seed=arm["replay_rng_seed"])
    teacher, _, _ = load_exported_actor(actor_path, device=str(agent.device))
    teacher.eval().requires_grad_(False)
    source_payload = torch.load(str(source_path), map_location="cpu", mmap=True, weights_only=False)
    source_critic = DrQCritic().to(agent.device)
    source_critic.load_state_dict(source_payload["critic_one"])
    source_critic.eval().requires_grad_(False)
    del source_payload
    pixels, diagnostic_metadata = _read_diagnostic_cache(
        pinned(root, protocol["diagnostic_cache"]["path"], protocol["diagnostic_cache"]["sha256"]),
        seed, {int(row["geometry_seed"]): row["family"] for row in catalog["train_diagnostic"]}
    )
    diagnostic_metadata["source_seed"] = seed
    run_dir = root / RUN_ROOT / f"learner-{seed}-{variant}-{condition}"
    if run_dir.exists():
        raise FileExistsError(f"r7 run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "study_protocol.json").write_bytes(protocol_path.read_bytes())
    initial = run_dir / "initial-weights.pt"
    torch.save({"actor": agent.actor.state_dict(), "critic_one": agent.critic_one.state_dict(),
                "critic_two": agent.critic_two.state_dict(), "target_one": agent.target_one.state_dict(),
                "target_two": agent.target_two.state_dict(), "source_checkpoint_sha256": source["checkpoint_sha256"]}, initial)
    config = {"format": "haic-drq-retention-run-config-v1", "protocol_sha256": protocol_sha,
              "source_seed": seed, "variant": variant, "condition": condition,
              "source_actor_sha256": source["actor_sha256"], "source_replay_sha256": sha(source_path),
              "source_episode_ledger_sha256": sha(root / f"runs/20260922-drq-augmentation-pad-v1-restart/control-seed{seed}/episodes.jsonl"),
              "initial_checkpoint_sha256": sha(initial), "weight_only_fork": fork,
              "rng_seeds": {name: arm[name] for name in r6._RNG_SEED_NAMES},
              "lambda_preserve": protocol["lambda_preserve"] if condition == "r7b" else 0.0,
              "source_rows_per_batch": 32, "online_rows_per_batch": 32,
              "optimizer": "Adam", "actor_lr": agent.config.actor_learning_rate,
              "critic_lr": agent.config.critic_learning_rate, "encoder_update": True,
              "diagnostic_cache_sha256": protocol["diagnostic_cache"]["sha256"]}
    r6._atomic_json(run_dir / "run-config.json", config)
    env = r6._build_training_env(reference, root, variant, arm)
    collector = EpisodeCollector(env, action_adapter=agent.action_adapter, gamma=agent.config.gamma)
    online_seeds, _ = r6._validate_catalog(reference, root)
    trace = {key: np.empty((22768, 64), dtype=dtype) for key, dtype in (
        ("source", np.uint8), ("source_indices", np.int64), ("episode_id", np.int64))}
    episodes, source_samples, online_samples = [], 0, 0
    episode_actions: list[np.ndarray] = []
    episode_reward = 0.0
    checkpoint_records = []
    started = time.monotonic()
    try:
        observation, reset_info = collector.reset()
        if str(reset_info["geometry_seed"]) not in online_seeds:
            raise ValueError("non-TRAIN initial catalog road")
        episodes.append({"event": "reset", "additional_online_step": 0, "episode_id": collector.episode_id, **reset_info})
        with (run_dir / "drift.jsonl").open("x", encoding="utf-8") as drift, (
            run_dir / "step-metrics.jsonl").open("x", encoding="utf-8") as step_log:
            drift.write(json.dumps(drift_snapshot(agent, teacher, source_critic, pixels,
                                                 diagnostic_metadata, update=0), sort_keys=True) + "\n")
            drift.flush()
            for step in range(1, 32769):
                action = agent.act(observation, deterministic=False)
                transition = collector.step(action)
                agent.observe(transition)
                episode_actions.append(transition.action.copy())
                episode_reward += transition.reward
                metrics = {}
                if step > 10000:
                    batch = sampler.sample(64)
                    offset = step - 10001
                    for key in trace:
                        trace[key][offset] = batch[key]
                    source_samples += int(np.count_nonzero(batch["source"] == 1))
                    online_samples += int(np.count_nonzero(batch["source"] == 0))
                    metrics = update_retained(agent, batch, streams,
                                              teacher=teacher if condition == "r7b" else None,
                                              lambda_preserve=config["lambda_preserve"])
                    if agent.gradient_steps != step - 10000:
                        raise RuntimeError("r7 budget departed from one update per decision")
                elif agent.gradient_steps:
                    raise RuntimeError("r7 warmup must have no updates")
                row = {"additional_online_step": step, "gradient_steps": agent.gradient_steps,
                       "episode_id": transition.episode_id, "geometry_seed": transition.info["geometry_seed"],
                       "track_id": transition.info["track_id"], "geometry_family": transition.info["geometry_family"],
                       "source_samples": source_samples, "online_samples": online_samples,
                       "update_metrics": metrics}
                step_log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                if transition.done:
                    episodes.append(r6._episode_event(transition, episode_actions,
                                                      episode_reward, seed, variant, step))
                    episode_actions = []
                    episode_reward = 0.0
                if agent.gradient_steps in DIAGNOSTIC_UPDATES[1:] and (step > 10000):
                    drift.write(json.dumps(drift_snapshot(agent, teacher, source_critic, pixels,
                                                         diagnostic_metadata, update=agent.gradient_steps), sort_keys=True) + "\n")
                    drift.flush()
                if step in (16384, 32768):
                    offset = step - 10000
                    archive = run_dir / f"replay-sample-trace-step-{step:09d}.npz"
                    packed = io.BytesIO()
                    np.savez_compressed(packed, **{key: value[:offset] for key, value in trace.items()},
                                        protocol_sha256=np.frombuffer(protocol_sha.encode(), dtype=np.uint8))
                    r6._atomic_bytes(archive, packed.getvalue())
                    checkpoint_dir = run_dir / "checkpoints" / f"step-{step:09d}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=False)
                    state = {"format": "haic-drq-retention-trainer-v1", "study_protocol_sha256": protocol_sha,
                             "source_seed": seed, "variant": variant, "condition": condition,
                             "source_replay_sha256": config["source_replay_sha256"],
                             "source_samples": source_samples, "online_samples": online_samples,
                             "sample_trace_sha256": sha(archive), "sample_trace_rows": offset,
                             "additional_online_steps": step, "study_gradient_steps": agent.gradient_steps,
                             "sampler_state": sampler.state_dict(), "catalog_sampler_state": r6._find_geometry_sampler(env).state_dict(),
                             "rng_streams": streams.state_dict(), "resume_allowed": False}
                    model = agent.save_checkpoint(checkpoint_dir / "checkpoint.pt",
                                                  source_paths=[root / key for key in protocol["code_sha256"]],
                                                  run_metadata={"study_id": STUDY_ID,
                                                                "study_protocol_sha256": protocol_sha,
                                                                "source_checkpoint_sha256": source["checkpoint_sha256"],
                                                                "source_replay_sha256": config["source_replay_sha256"],
                                                                "catalog_sha256": reference["catalog"]["sha256"],
                                                                "max_steps": 2000, "reward_contract": {"raw_reward": True}},
                                                  trainer_state=state)
                    actor = agent.export_actor(checkpoint_dir / "actor.pt")
                    checkpoint_records.append({"checkpoint_online_step": step,
                                               "checkpoint_path": model.relative_to(root).as_posix(),
                                               "checkpoint_sha256": sha(model),
                                               "actor_path": actor.relative_to(root).as_posix(),
                                               "actor_sha256": sha(actor),
                                               "study_gradient_steps": agent.gradient_steps,
                                               "sample_trace_path": archive.relative_to(root).as_posix(),
                                               "sample_trace_sha256": sha(archive)})
                if step % 1000 == 0:
                    step_log.flush()
                    print(json.dumps({"run": run_dir.name, "decisions": step,
                                      "updates": agent.gradient_steps, "source_samples": source_samples,
                                      "online_samples": online_samples, "elapsed_seconds": time.monotonic() - started}), flush=True)
                if transition.done and step < 32768:
                    observation, reset_info = collector.reset()
                    if str(reset_info["geometry_seed"]) not in online_seeds:
                        raise ValueError("non-TRAIN catalog road")
                    episodes.append({"event": "reset", "additional_online_step": step,
                                     "episode_id": collector.episode_id, **reset_info})
                elif not transition.done:
                    observation = transition.next_observation
    finally:
        env.close()
    if agent.gradient_steps != 22768 or source_samples != 22768 * 32 or online_samples != 22768 * 32:
        raise RuntimeError("r7 frozen update/sample budget failed")
    with (run_dir / "episodes.jsonl").open("x", encoding="utf-8") as output:
        for event in episodes:
            output.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    result = {"format": "haic-drq-retention-run-result-v1", "study_id": STUDY_ID,
              "study_protocol_sha256": protocol_sha, "source_seed": seed, "variant": variant,
              "condition": condition, "additional_online_steps": 32768, "study_gradient_steps": 22768,
              "source_samples": source_samples, "online_samples": online_samples,
              "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
              "source_checkpoint_sha256": source["checkpoint_sha256"],
              "source_actor_sha256": source["actor_sha256"], "source_replay_sha256": config["source_replay_sha256"],
              "diagnostic_trace_sha256": sha(run_dir / "drift.jsonl"),
              "online_replay_manifest_sha256": sha(run_dir / "step-metrics.jsonl"),
              "candidates": checkpoint_records, "completed": True,
              "elapsed_seconds": time.monotonic() - started}
    r6._atomic_json(run_dir / "checkpoint-catalog.json", {
        "format": "haic-drq-retention-checkpoint-catalog-v1", "study_protocol_sha256": protocol_sha,
        "source_seed": seed, "variant": variant, "condition": condition,
        "candidates": checkpoint_records})
    r6._atomic_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/drqv2-retention-r7.json"))
    parser.add_argument("--source-seed", type=int, choices=(0, 1))
    parser.add_argument("--variant", choices=r6.VARIANTS)
    parser.add_argument("--condition", choices=("r7a", "r7b"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.preflight_only:
        protocol, reference, digest = load_protocol(root, path)
        catalog = json.loads(pinned(root, reference["catalog"]["path"],
                                    reference["catalog"]["sha256"]).read_text())
        families = {int(row["geometry_seed"]): row["family"] for row in catalog["train_diagnostic"]}
        cache = pinned(root, protocol["diagnostic_cache"]["path"], protocol["diagnostic_cache"]["sha256"])
        for seed in (0, 1):
            source = next(item for item in reference["source_actors"] if item["learner_seed"] == seed)
            replay = load_source_replay(root, source, expected_sha=source["checkpoint_sha256"],
                                        expected_seed=seed, excluded_roads=set(families))
            if len(replay.valid_indices()) < 32:
                raise ValueError("source checkpoint has fewer than 32 usable transitions")
            pixels, _ = _read_diagnostic_cache(cache, seed, families)
            print(json.dumps({"protocol_sha256": digest, "source_seed": seed,
                              "source_replay_rows": replay.size, "drift_observations": len(pixels),
                              "environment_decisions": 0, "learner_updates": 0}), flush=True)
        return 0
    if args.source_seed is None or args.variant is None or args.condition is None:
        parser.error("--source-seed, --variant and --condition are required for training")
    print(json.dumps(train(root, path, args.source_seed, args.variant, args.condition), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
