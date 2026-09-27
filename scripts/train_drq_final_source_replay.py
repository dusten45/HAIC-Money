"""Train one final-source replay arm with unchanged r7b learner mechanics."""

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

from common_adapter import ActionAdapter, EpisodeCollector
from drq_v2 import DrQCritic, Uint8Replay, load_exported_actor
from haic.algorithms.drq_v2.retention import RetentionReplaySampler, update_retained
from haic.algorithms.drq_v2.teacher_study import audit_source_actor_pair, fork_from_source
from scripts import collect_drq_final_source_replay as collection
from scripts import train_drq_geometry_mix as r6
from scripts import train_drq_retention_r7 as r7


STUDY_ID = collection.STUDY_ID
RUN_ROOT = collection.RUN_ROOT
FORMAT = "haic-drq-final-source-replay-study-v1"
PROTOCOL_PATH = Path("experiments/drqv2-final-source-replay-v1.json")
R7_SHA = "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9"


def _source(seed: int, reference: dict) -> dict:
    return next(row for row in reference["source_actors"] if row["learner_seed"] == seed)


def _receipt(root: Path, relative: dict, seed: int, original: dict, catalog_sha: str,
             collection_sha: str, schedule_sha: str) -> dict:
    expected = f"{RUN_ROOT}/collection/seed{seed}"
    collection.require(relative.get("pool_path") == f"{expected}/pool.pt"
                       and relative.get("receipt_path") == f"{expected}/receipt.json",
                       "source pool must be the separately sealed seed-specific output")
    path = r7.pinned(root, relative["receipt_path"], relative["receipt_sha256"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mandatory = {"format": "haic-drq-final-source-pool-v1", "completed": True,
                 "source_seed": seed, "source_actor_sha256": original["actor_sha256"],
                 "source_checkpoint_sha256": original["checkpoint_sha256"],
                 "pool_path": relative["pool_path"], "pool_sha256": relative["pool_sha256"],
                 "partition": "TRAIN", "excluded_diagnostic_roads": True,
                 "decisions": 100000, "capacity": 100000,
                 "collection_noise_std": .05, "catalog_sha256": catalog_sha,
                 "collection_protocol_sha256": collection_sha, "schedule_sha256": schedule_sha}
    collection.require(all(receipt.get(key) == value for key, value in mandatory.items()),
                       "sealed final source receipt identity/budget differs from protocol")
    for key, filename in (("episode_ledger_path", "episodes.jsonl"),
                          ("step_ledger_path", "steps.jsonl")):
        collection.require(receipt.get(key) == f"{expected}/{filename}", "source ledger path differs")
        r7.pinned(root, receipt[key], receipt[key.replace("path", "sha256")])
    collection.require(type(receipt.get("valid_n_step_starts")) is int
                       and receipt["valid_n_step_starts"] >= 32,
                       "source replay receipt has insufficient terminal-safe starts")
    r7.pinned(root, relative["pool_path"], relative["pool_sha256"])
    return receipt


def load_protocol(root: Path, path: Path, *, arm: tuple[int, str] | None = None
                  ) -> tuple[dict, dict, dict, dict, dict[int, list[dict]], str]:
    collection.require(path.resolve(strict=True).is_relative_to(root), "training protocol must be repository-local")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    collection.require(protocol.get("format") == FORMAT and protocol.get("study_id") == STUDY_ID
                       and protocol.get("run_root") == RUN_ROOT.as_posix()
                       and protocol.get("r6_protocol_sha256") == r7.R6_SHA
                       and protocol.get("r7_protocol_sha256") == R7_SHA
                       and protocol.get("lambda_preserve") == .5,
                       "not the frozen final-source replay r7b-only study")
    old_path = r7.pinned(root, protocol["r7_protocol_path"], R7_SHA)
    old, reference, _ = r7.load_protocol(root, old_path)
    collection.require(protocol.get("r6_protocol_path") == old["r6_protocol_path"]
                       and protocol.get("catalog_sha256") == reference["catalog"]["sha256"]
                       and protocol.get("diagnostic_cache") == old["diagnostic_cache"]
                       and protocol.get("diagnostic_updates") == list(r7.DIAGNOSTIC_UPDATES)
                       and protocol.get("replay_contract") == {
                           "batch_size": 64, "source_rows": 32, "online_rows": 32,
                           "sample_without_replacement_per_pool": True,
                           "source_kind": "final_source_policy"}
                       and not any(key in protocol for key in ("budgets", "learner", "variants", "environment")),
                       "r7b budget, drift cache, replay ratio or inherited r6 settings changed")
    collection.require(protocol.get("collection_protocol_path") == collection.PROTOCOL_PATH.as_posix(),
                       "training must bind the separate frozen collection protocol")
    collection_path = r7.pinned(root, protocol["collection_protocol_path"],
                                protocol["collection_protocol_sha256"])
    collector_protocol, _, schedules, collector_sha = collection.load_protocol(
        root, collection_path, check_outputs=False)
    collection.require(collector_protocol["r7_protocol_sha256"] == R7_SHA,
                       "source collection is not paired with original r7b")
    expected = {(seed, variant, "final_source") for seed in (0, 1) for variant in r6.VARIANTS}
    rows = protocol.get("runs")
    collection.require(isinstance(rows, list) and len(rows) == 6
                       and {(row["source_seed"], row["variant"], row["condition"]) for row in rows} == expected,
                       "exactly six r6 seed/mixture final-source arms required")
    for row in rows:
        seed, variant = row["source_seed"], row["variant"]
        collection.require(row.get("run_dir") == f"{RUN_ROOT}/learner-{seed}-{variant}-final_source"
                           and row.get("rng_seeds") == {
                               key: r6._arm(reference, seed, variant)[key] for key in r6._RNG_SEED_NAMES},
                           "final source learner run path or r6 RNG streams changed")
    collection.require(set(protocol.get("source_replay", {})) == {"0", "1"},
                       "both SHA-bound final source replay pools are required")
    receipts = {}
    for seed in (0, 1):
        original = _source(seed, reference)
        receipts[seed] = _receipt(root, protocol["source_replay"][str(seed)], seed, original,
                                  reference["catalog"]["sha256"], collector_sha,
                                  collector_protocol["sources"][str(seed)]["schedule_sha256"])
        collection.require(receipts[seed]["original_ledger_sha256"] ==
                           old["source_replay"][str(seed)]["episode_ledger_sha256"]
                           and receipts[seed]["collection_rng_seeds"] ==
                           collector_protocol["sources"][str(seed)]["collection_rng_seeds"],
                           "final source receipt lost original ledger/RNG lineage")
    hashes = protocol.get("code_sha256")
    collection.require(isinstance(hashes, dict)
                       and {"scripts/train_drq_final_source_replay.py",
                            "scripts/collect_drq_final_source_replay.py",
                            "haic/algorithms/drq_v2/retention.py",
                            "scripts/train_drq_retention_r7.py"}.issubset(hashes),
                       "training executable source map is incomplete")
    for name, digest in hashes.items():
        r7.pinned(root, name, digest)
    for row in rows:
        if arm is None or (row["source_seed"], row["variant"]) == arm:
            collection.require(not (root / row["run_dir"]).exists(),
                               "final-source arm exists; refusing overwrite")
    return protocol, reference, old, receipts, schedules, r7.sha(path)


def check_pool_lineage(replay: Uint8Replay, episode_path: Path, step_path: Path,
                       schedule: list[dict], *, count: int = 100000, seed: int | None = None) -> None:
    """Independently join every sealed transition to its reset and step ledgers."""
    resets: dict[int, dict] = {}
    ends: dict[int, dict] = {}
    last_reset = -1
    for line in episode_path.open("r", encoding="utf-8"):
        row = json.loads(line)
        ep = row.get("episode_id")
        collection.require(type(ep) is int and 0 <= ep < len(schedule), "unknown source episode ID")
        road = schedule[ep]
        collection.require(row.get("schedule_index") == ep
                           and (seed is None or row.get("source_seed") == seed)
                           and row.get("original_episode_id") == road["original_episode_id"]
                           and row.get("track_id") == road["track_id"]
                           and row.get("seed") == road["geometry_seed"]
                           and (row.get("event") != "reset" or row.get("partition") == "TRAIN"),
                           "source episode ledger has foreign road lineage")
        if row.get("event") == "reset":
            collection.require(ep == last_reset + 1 and ep not in resets
                               and (ep == 0 or ep - 1 in ends),
                               "source episode resets are not ordered after completed ends")
            last_reset = ep
            resets[ep] = row
        elif row.get("event") in ("end", "capped_partial"):
            collection.require(ep in resets and ep not in ends, "duplicate or orphan source end")
            ends[ep] = row
        else:
            raise ValueError("unknown source episode event")
    collection.require(bool(resets) and set(resets) == set(ends), "source episode ledger is incomplete")
    position = -1
    previous_episode, previous_step = -1, -1
    episode_rewards: dict[int, float] = {ep: 0. for ep in resets}
    for position, line in enumerate(step_path.open("r", encoding="utf-8")):
        row = json.loads(line)
        collection.require(position < count and row.get("decision") == position + 1
                           and row.get("sequence_id") == position,
                           "source step ledger sequence is incomplete or reordered")
        ep = row.get("episode_id")
        collection.require(type(ep) is int and ep in resets and ep in ends
                           and (seed is None or row.get("source_seed") == seed)
                           and ep == int(replay.episode_ids[position])
                           and row.get("episode_step") == int(replay.episode_steps[position])
                           and row.get("schedule_index") == ep
                           and row.get("original_episode_id") == schedule[ep]["original_episode_id"]
                           and row.get("partition") == "TRAIN"
                           and row.get("track_id") == schedule[ep]["track_id"]
                           and row.get("geometry_seed") == schedule[ep]["geometry_seed"],
                           "source step belongs to an unknown, diagnostic or non-TRAIN road")
        collection.require((ep == previous_episode and row["episode_step"] == previous_step + 1)
                           or (ep == previous_episode + 1 and row["episode_step"] == 0
                               and (previous_episode < 0 or ends[previous_episode]["event"] == "end")),
                           "source transitions skip or reorder episodes")
        previous_episode, previous_step = ep, row["episode_step"]
        action = np.asarray(row.get("native_action"), dtype=np.float32)
        noised = np.asarray(row.get("noised_native_action"), dtype=np.float64)
        original = np.asarray(row.get("unnoised_native_action"), dtype=np.float64)
        noise = np.asarray(row.get("noise"), dtype=np.float64)
        official = np.asarray(row.get("official_action"), dtype=np.float32)
        pre_adapter = np.asarray(row.get("pre_adapter_native_action"), dtype=np.float32)
        collection.require(action.shape == original.shape == noised.shape == noise.shape == official.shape == (3,)
                           and pre_adapter.shape == (3,)
                           and all(np.isfinite(value).all() for value in (action, original, noised, noise, official))
                           and np.allclose(noised, original + noise, rtol=0, atol=1e-12)
                           and np.array_equal(pre_adapter, np.clip(noised, -1, 1).astype(np.float32))
                           and np.max(np.abs(action - pre_adapter)) <= 1e-6
                           and np.array_equal(action, replay.actions[position])
                           and np.array_equal(official, ActionAdapter().to_official(pre_adapter))
                           and np.array_equal(action, ActionAdapter().to_native(official))
                           and float(replay.rewards[position]) == float(np.float32(row.get("reward")))
                           and all(bool(row.get(name)) == bool(getattr(replay, name)[position])
                                   for name in ("terminated", "truncated", "terminal")),
                           "sealed source action/reward/terminal differs from step ledger")
        episode_rewards[ep] += row["reward"]
        if row["episode_step"] == 0:
            collection.require(resets[ep]["collection_step"] == position, "reset step offset differs")
        if bool(replay.terminated[position] or replay.truncated[position]):
            collection.require(ends[ep]["event"] == "end"
                               and ends[ep]["collection_step"] == position + 1
                               and ends[ep]["steps"] == row["episode_step"] + 1,
                               "terminal boundary does not join episode ledger")
    collection.require(position + 1 == count and ends[last_reset]["collection_step"] == count,
                       "source replay step ledger does not cover entire 100k window")
    collection.require(all(abs(episode_rewards[ep] - end["reward"]) < 1e-5
                           for ep, end in ends.items()), "episode reward differs from source step ledger")
    collection.require(ends[last_reset]["event"] == "end" or (
        ends[last_reset]["event"] == "capped_partial" and not bool(replay.terminated[count - 1]
                                                               or replay.truncated[count - 1])),
        "source cap was not a genuine unfinished episode")


def load_source_pool(root: Path, source: dict, receipt: dict, schedule: list[dict], seed: int) -> Uint8Replay:
    path = r7.pinned(root, source["pool_path"], source["pool_sha256"])
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    collection.require(payload.get("format") == "haic-drq-final-source-replay-v1"
                       and payload.get("source_seed") == seed
                       and payload.get("collection_protocol_sha256") == receipt["collection_protocol_sha256"]
                       and payload.get("source_checkpoint_sha256") == receipt["source_checkpoint_sha256"]
                       and payload.get("source_actor_sha256") == receipt["source_actor_sha256"],
                       "source replay pool has incorrect frozen actor/original checkpoint lineage")
    state = payload["replay"]
    expected_arrays = {"frames": ((100000, 84, 84), np.uint8),
                       "actions": ((100000, 3), np.float32),
                       "rewards": ((100000,), np.float32),
                       "episode_ids": ((100000,), np.int64),
                       "episode_steps": ((100000,), np.int64),
                       "sequence_ids": ((100000,), np.int64),
                       **{key: ((100000,), np.bool_) for key in ("terminal", "terminated", "truncated")}}
    collection.require(isinstance(state, dict) and all(
        isinstance(state.get(key), np.ndarray)
        and state[key].shape == shape and state[key].dtype == dtype
        for key, (shape, dtype) in expected_arrays.items()),
        "source replay stored arrays do not match original Uint8Replay contract")
    replay = Uint8Replay(capacity=100000, action_dim=3, n_step=3, gamma=.99)
    replay.load_state_dict(state)
    del payload
    collection.require(replay.size == 100000 and replay.oldest_sequence == 0 and replay.newest_sequence == 99999
                       and np.array_equal(replay.sequence_ids, np.arange(100000)),
                       "source replay is not the complete seq0..99999 window")
    check_pool_lineage(replay,
                       r7.pinned(root, receipt["episode_ledger_path"], receipt["episode_ledger_sha256"]),
                       r7.pinned(root, receipt["step_ledger_path"], receipt["step_ledger_sha256"]),
                       schedule, seed=seed)
    collection.require(len(replay.valid_indices()) == receipt["valid_n_step_starts"]
                       and receipt["valid_n_step_starts"] >= 32,
                       "sealed source replay has fewer than 32 validated terminal-safe n-step starts")
    for field in r7.SOURCE_FIELDS:
        getattr(replay, field).flags.writeable = False
    for pixels in replay._boundary_observations.values():
        pixels.flags.writeable = False
    collection.require(r7.sha(path) == source["pool_sha256"], "source replay changed during loading")
    return replay


def train(root: Path, protocol_path: Path, seed: int, variant: str) -> dict:
    collection.require(seed in (0, 1) and variant in r6.VARIANTS, "arm is not a frozen six-run cell")
    protocol, reference, old, receipts, schedules, protocol_sha = load_protocol(
        root, protocol_path, arm=(seed, variant))
    arm = r6._arm(reference, seed, variant)
    source = _source(seed, reference)
    pool_ref = protocol["source_replay"][str(seed)]
    source_path = r7.pinned(root, source["checkpoint_path"], source["checkpoint_sha256"])
    actor_path = r7.pinned(root, source["actor_path"], source["actor_sha256"])
    audit_source_actor_pair(source_path, actor_path, learner_seed=seed,
                            source_revision=source["source_revision"],
                            expected_checkpoint_sha256=source["checkpoint_sha256"],
                            expected_actor_sha256=source["actor_sha256"])
    pool = load_source_pool(root, pool_ref, receipts[seed], schedules[seed], seed)
    collection.require(torch.cuda.is_available(), "r6 frozen CUDA training device is unavailable")
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
    sampler = RetentionReplaySampler(agent.replay, pool, seed=arm["replay_rng_seed"])
    teacher, _, _ = load_exported_actor(actor_path, device=str(agent.device))
    teacher.eval().requires_grad_(False)
    source_payload = torch.load(str(source_path), map_location="cpu", mmap=True, weights_only=False)
    source_critic = DrQCritic().to(agent.device)
    source_critic.load_state_dict(source_payload["critic_one"])
    source_critic.eval().requires_grad_(False)
    del source_payload
    catalog = json.loads(r7.pinned(root, reference["catalog"]["path"],
                                   reference["catalog"]["sha256"]).read_text(encoding="utf-8"))
    pixels, diagnostic_metadata = r7._read_diagnostic_cache(
        r7.pinned(root, old["diagnostic_cache"]["path"], old["diagnostic_cache"]["sha256"]),
        seed, {int(row["geometry_seed"]): row["family"] for row in catalog["train_diagnostic"]})
    diagnostic_metadata["source_seed"] = seed
    run_dir = root / RUN_ROOT / f"learner-{seed}-{variant}-final_source"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "study_protocol.json").write_bytes(protocol_path.read_bytes())
    initial = run_dir / "initial-weights.pt"
    torch.save({"actor": agent.actor.state_dict(), "critic_one": agent.critic_one.state_dict(),
                "critic_two": agent.critic_two.state_dict(), "target_one": agent.target_one.state_dict(),
                "target_two": agent.target_two.state_dict(), "source_checkpoint_sha256": source["checkpoint_sha256"]}, initial)
    config = {"format": "haic-drq-retention-run-config-v1", "protocol_sha256": protocol_sha,
              "source_seed": seed, "variant": variant, "condition": "final_source",
              "source_actor_sha256": source["actor_sha256"], "source_replay_sha256": pool_ref["pool_sha256"],
              "source_checkpoint_sha256": source["checkpoint_sha256"],
              "source_episode_ledger_sha256": receipts[seed]["episode_ledger_sha256"],
              "source_pool_receipt_sha256": pool_ref["receipt_sha256"],
              "initial_checkpoint_sha256": r7.sha(initial), "weight_only_fork": fork,
              "rng_seeds": {name: arm[name] for name in r6._RNG_SEED_NAMES}, "lambda_preserve": .5,
              "source_rows_per_batch": 32, "online_rows_per_batch": 32,
              "optimizer": "Adam", "actor_lr": agent.config.actor_learning_rate,
              "critic_lr": agent.config.critic_learning_rate, "encoder_update": True,
              "diagnostic_cache_sha256": old["diagnostic_cache"]["sha256"]}
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
        collection.require(str(reset_info["geometry_seed"]) in online_seeds,
                           "non-TRAIN initial catalog road")
        episodes.append({"event": "reset", "additional_online_step": 0,
                         "episode_id": collector.episode_id, **reset_info})
        with (run_dir / "drift.jsonl").open("x", encoding="utf-8") as drift, (
            run_dir / "step-metrics.jsonl").open("x", encoding="utf-8") as step_log:
            drift.write(json.dumps(r7.drift_snapshot(agent, teacher, source_critic, pixels,
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
                    metrics = update_retained(agent, batch, streams, teacher=teacher, lambda_preserve=.5)
                    collection.require(agent.gradient_steps == step - 10000,
                                       "r7b one-update-per-decision budget changed")
                else:
                    collection.require(agent.gradient_steps == 0, "r7b warmup unexpectedly updated")
                step_log.write(json.dumps({"additional_online_step": step, "gradient_steps": agent.gradient_steps,
                                           "episode_id": transition.episode_id,
                                           "geometry_seed": transition.info["geometry_seed"],
                                           "track_id": transition.info["track_id"],
                                           "geometry_family": transition.info["geometry_family"],
                                           "source_samples": source_samples, "online_samples": online_samples,
                                           "update_metrics": metrics}, sort_keys=True, allow_nan=False) + "\n")
                if transition.done:
                    episodes.append(r6._episode_event(transition, episode_actions,
                                                      episode_reward, seed, variant, step))
                    episode_actions, episode_reward = [], 0.0
                if agent.gradient_steps in r7.DIAGNOSTIC_UPDATES[1:] and step > 10000:
                    drift.write(json.dumps(r7.drift_snapshot(agent, teacher, source_critic, pixels,
                                                             diagnostic_metadata,
                                                             update=agent.gradient_steps), sort_keys=True) + "\n")
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
                             "source_seed": seed, "variant": variant, "condition": "final_source",
                             "source_replay_sha256": config["source_replay_sha256"],
                             "source_samples": source_samples, "online_samples": online_samples,
                             "sample_trace_sha256": r7.sha(archive), "sample_trace_rows": offset,
                             "additional_online_steps": step, "study_gradient_steps": agent.gradient_steps,
                             "sampler_state": sampler.state_dict(),
                             "catalog_sampler_state": r6._find_geometry_sampler(env).state_dict(),
                             "rng_streams": streams.state_dict(), "resume_allowed": False}
                    model = agent.save_checkpoint(
                        checkpoint_dir / "checkpoint.pt",
                        source_paths=[root / key for key in protocol["code_sha256"]],
                        run_metadata={"study_id": STUDY_ID, "study_protocol_sha256": protocol_sha,
                                      "source_checkpoint_sha256": source["checkpoint_sha256"],
                                      "source_replay_sha256": config["source_replay_sha256"],
                                      "catalog_sha256": reference["catalog"]["sha256"],
                                      "max_steps": 2000, "reward_contract": {"raw_reward": True}},
                        trainer_state=state)
                    actor = agent.export_actor(checkpoint_dir / "actor.pt")
                    checkpoint_records.append({"checkpoint_online_step": step,
                                               "checkpoint_path": model.relative_to(root).as_posix(),
                                               "checkpoint_sha256": r7.sha(model),
                                               "actor_path": actor.relative_to(root).as_posix(),
                                               "actor_sha256": r7.sha(actor),
                                               "study_gradient_steps": agent.gradient_steps,
                                               "sample_trace_path": archive.relative_to(root).as_posix(),
                                               "sample_trace_sha256": r7.sha(archive)})
                if step % 1000 == 0:
                    step_log.flush()
                    print(json.dumps({"run": run_dir.name, "decisions": step,
                                      "updates": agent.gradient_steps, "source_samples": source_samples,
                                      "online_samples": online_samples,
                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
                if transition.done and step < 32768:
                    observation, reset_info = collector.reset()
                    collection.require(str(reset_info["geometry_seed"]) in online_seeds,
                                       "non-TRAIN catalog road")
                    episodes.append({"event": "reset", "additional_online_step": step,
                                     "episode_id": collector.episode_id, **reset_info})
                elif not transition.done:
                    observation = transition.next_observation
    finally:
        env.close()
        collection.require(r7.sha(root / pool_ref["pool_path"]) == pool_ref["pool_sha256"]
                           and r7.sha(root / pool_ref["receipt_path"]) == pool_ref["receipt_sha256"],
                           "sealed source replay pool or receipt changed during training")
    collection.require(agent.gradient_steps == 22768
                       and source_samples == online_samples == 22768 * 32,
                       "r7b budget or 32:32 sample quota failed")
    with (run_dir / "episodes.jsonl").open("x", encoding="utf-8") as output:
        for event in episodes:
            output.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
    result = {"format": "haic-drq-retention-run-result-v1", "study_id": STUDY_ID,
              "study_protocol_sha256": protocol_sha, "source_seed": seed, "variant": variant,
              "condition": "final_source", "additional_online_steps": 32768,
              "study_gradient_steps": 22768, "source_samples": source_samples,
              "online_samples": online_samples, "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
              "source_checkpoint_sha256": source["checkpoint_sha256"],
              "source_actor_sha256": source["actor_sha256"],
              "source_replay_sha256": pool_ref["pool_sha256"],
              "diagnostic_trace_sha256": r7.sha(run_dir / "drift.jsonl"),
              "online_replay_manifest_sha256": r7.sha(run_dir / "step-metrics.jsonl"),
              "candidates": checkpoint_records, "completed": True,
              "elapsed_seconds": time.monotonic() - started}
    r6._atomic_json(run_dir / "checkpoint-catalog.json", {
        "format": "haic-drq-retention-checkpoint-catalog-v1", "study_protocol_sha256": protocol_sha,
        "source_seed": seed, "variant": variant, "condition": "final_source",
        "candidates": checkpoint_records})
    r6._atomic_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-seed", type=int, choices=(0, 1))
    parser.add_argument("--variant", choices=r6.VARIANTS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.preflight_only:
        arm = (args.source_seed, args.variant) if args.source_seed is not None and args.variant is not None else None
        protocol, _, _, receipts, schedules, digest = load_protocol(root, path, arm=arm)
        for seed in (0, 1):
            load_source_pool(root, protocol["source_replay"][str(seed)], receipts[seed], schedules[seed], seed)
        print(json.dumps({"study_protocol_sha256": digest, "arms": 6,
                          "environment_decisions": 0, "learner_updates": 0}, sort_keys=True))
        return 0
    if args.source_seed is None or args.variant is None:
        parser.error("--source-seed and --variant are required for training")
    print(json.dumps(train(root, path, args.source_seed, args.variant), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
