"""Freeze and run a new-host RLPD learner on already-consumed G0 TRAIN roads.

This is a new study, not a resume of V5 and not G1 or protected evaluation.
The historical V5 protocol is validated at its original hash; only its obsolete
source/runtime pins are bypassed, then the new protocol pins current sources.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from common_adapter import EpisodeCollector, ObservationSpec
from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig, sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay
from scripts.rlpd_common import canonical_sha256, load_offline_replay, runtime_metadata, sha256_file
from scripts.rlpd_entropy_v5_common import read_entropy_v5_protocol
from scripts.train_rlpd import array_sha256, seed_everything, transition_to_replay, vm_hwm_kib
from train import build_env


ROOT = Path(__file__).resolve().parents[1]
V5_PROTOCOL = "experiments/pixel-rlpd-entropy-target-ablation-v5.json"
V5_SHA = "2e7df4152e2282bcd34a6d2a160737cc13c6f1a03d769bd69832e1fc51326546"
PRIOR_DIR = "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data"
PRIOR_MANIFEST_SHA = "2dad0686e8c7c84954f096375f31495673bbad25c988034613b4145cb97fd770"
PRIOR_DATASET_SHA = "815563d490c2aa9903f29baee34fc7c8217a1f762e42324b121a0a11a523d77c"
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_SHA = "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b"
G0_MANIFEST = "runs/20260926-rlpd-g0-completion-v1/manifest.json"
G0_MANIFEST_SHA = "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358"
G0_LEDGER = "runs/20260926-rlpd-g0-completion-v1/cells.jsonl"
G0_LEDGER_SHA = "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559"
NEW_SOURCES = frozenset({
    "haic/algorithms/rlpd/agent.py", "haic/algorithms/rlpd/augment.py",
    "haic/algorithms/rlpd/replay.py", "scripts/rlpd_common.py",
    "scripts/rlpd_entropy_common.py", "scripts/rlpd_entropy_v5_common.py",
    "scripts/train_rlpd.py", "scripts/train_rlpd_newhost.py",
})
FORMAT = "haic-rlpd-newhost-reused-train-v1"


def located(relative: str, prefix: str) -> Path:
    value = Path(relative)
    if (not isinstance(relative, str) or value.is_absolute() or len(value.parts) < 2
            or value.parts[0] != prefix or value.as_posix() != relative
            or any(part in (".", "..") for part in value.parts)):
        raise ValueError(f"expected a safe repository-relative {prefix}/ path: {relative}")
    current = ROOT
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked path: {relative}")
    return current


def pinned(relative: str, expected: str, prefix: str) -> Path:
    path = located(relative, prefix)
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"missing or changed frozen evidence: {relative}")
    return path


def source_path(relative: str) -> Path:
    value = Path(relative)
    if (value.is_absolute() or not value.parts or value.as_posix() != relative
            or any(part in (".", "..") for part in value.parts)):
        raise ValueError(f"unsafe source path: {relative}")
    current = ROOT
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked source path: {relative}")
    if not current.is_file():
        raise ValueError(f"missing source: {relative}")
    return current


def historical_evidence() -> tuple[dict, dict, list[dict]]:
    v5_path = pinned(V5_PROTOCOL, V5_SHA, "experiments")
    # read_protocol's *old* source map includes mutable competition documents.
    # Its structural V5 contract and original protocol bytes are still checked.
    v5 = read_entropy_v5_protocol(v5_path, verify_sources=False)
    manifest = json.loads(pinned(f"{PRIOR_DIR}/manifest.json", PRIOR_MANIFEST_SHA, "runs").read_text())
    if (manifest.get("study_protocol_sha256") != V5_SHA
            or manifest.get("dataset_sha256") != PRIOR_DATASET_SHA
            or manifest.get("study_id") != v5["name"]):
        raise ValueError("original V5 prior manifest does not bind the original protocol")
    original = json.loads(pinned(G0_PROTOCOL, G0_SHA, "experiments").read_text())
    g0_manifest = json.loads(pinned(G0_MANIFEST, G0_MANIFEST_SHA, "runs").read_text())
    ledger = [json.loads(line) for line in pinned(G0_LEDGER, G0_LEDGER_SHA, "runs").read_text().splitlines()]
    if (original.get("format") != "haic-rlpd-g0-diagnostic-v1"
            or original.get("status") != "frozen" or original.get("partition") != "TRAIN"
            or original.get("frame_skip") != 4 or original.get("max_steps") != 2000
            or g0_manifest.get("protocol_sha256") != G0_SHA
            or g0_manifest.get("cells_sha256") != G0_LEDGER_SHA
            or g0_manifest.get("cell_count") != 24 or len(ledger) != 24
            or len(original.get("cells", [])) != 12 or len(original.get("actors", [])) != 2):
        raise ValueError("G0 protocol/manifest/ledger chain is invalid")
    expected_actors = {actor["id"]: actor["sha256"] for actor in original["actors"]}
    if len(expected_actors) != 2:
        raise ValueError("G0 actor identities are not unique")
    for index, cell in enumerate(original["cells"]):
        if (cell.get("track_id") != 1 or cell.get("obstacles") is not True
                or cell.get("partition") != "TRAIN" or type(cell.get("geometry_seed")) is not int):
            raise ValueError("G0 catalog contains an ineligible TRAIN road")
        for offset, (actor_id, actor_sha) in enumerate(expected_actors.items()):
            row = ledger[index * 2 + offset]
            if (row.get("track_id") != cell["track_id"]
                    or row.get("geometry_seed") != cell["geometry_seed"]
                    or row.get("partition") != "TRAIN" or row.get("actor_id") != actor_id
                    or row.get("actor_sha256") != actor_sha
                    or type(row.get("steps")) is not int or row["steps"] <= 0):
                raise ValueError("G0 ledger does not attest both consumed actor/road slots")
    if len({row["geometry_seed"] for row in original["cells"]}) != 12:
        raise ValueError("G0 road catalog contains duplicates")
    for relative, expected in original["source_hashes"].items():
        if sha256_file(source_path(relative)) != expected:
            raise ValueError(f"original G0 source changed: {relative}")
    return v5, original, original["cells"]


def config_from_v5(v5: dict) -> RLPDConfig:
    student = v5["student_training"]
    learner = student["learner"]
    arm = student["arm_specs"]["rlpd-author-target"]
    if (arm.get("use_offline") is not True or arm.get("target_entropy") != -1.5
            or student["batch_size"] != 64 or student["offline_batch_size"] != 32
            or student["online_batch_size"] != 32 or student["replay_capacity"] != 100000
            or student["first_update_step"] != 1000 or student["policy_takeover_step"] != 2000):
        raise ValueError("original V5 RLPD author arm does not match the intended learner")
    return RLPDConfig(
        actor_lr=learner["actor_lr"], critic_lr=learner["critic_lr"],
        temperature_lr=learner["temperature_lr"], gamma=learner["gamma"], tau=learner["tau"],
        batch_size=student["batch_size"], num_qs=learner["num_qs"],
        num_min_qs=learner["num_min_qs"], target_entropy=arm["target_entropy"],
        initial_alpha=learner["initial_alpha"], backup_entropy=student["backup_entropy"],
        augmentation_pad=learner["augmentation_pad"],
    )


def current_runtime() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("this new-host learner requires CUDA; no CPU training fallback")
    runtime = runtime_metadata(device="cuda")
    if (runtime["torch"] != "2.1.0+cu121"
            or runtime["gpu"]["name"] != "NVIDIA GeForce RTX 4060 Ti"
            or runtime["gpu"]["compute_capability"] != [8, 9]):
        raise RuntimeError("new-host Torch/GPU identity differs from RTX 4060 Ti/Torch 2.1")
    return runtime


def freeze(output: str, *, seed: int, steps: int) -> dict:
    target = located(output, "experiments")
    if target.exists():
        raise FileExistsError(target)
    if type(seed) is not int or seed < 0 or type(steps) is not int or steps <= 2000:
        raise ValueError("learner seed must be nonnegative and steps must exceed the 2,000-step warmup")
    v5, g0, cells = historical_evidence()
    config = config_from_v5(v5)
    offline, digest, _ = load_offline_replay(ROOT / PRIOR_DIR, v5, seed=seed + 0x0FF1)
    if digest != PRIOR_DATASET_SHA or offline.valid_count != len(offline):
        raise ValueError("original V5 prior replay did not reconstruct exactly")
    sources = set(g0["source_hashes"]) | NEW_SOURCES
    hashes = {relative: sha256_file(source_path(relative)) for relative in sorted(sources)}
    if any(hashes[relative] != digest for relative, digest in g0["source_hashes"].items()):
        raise ValueError("original G0 source drifted during freeze")
    protocol = {
        "format": FORMAT, "status": "frozen", "role": "reused-TRAIN-online-RLPD-not-G1-or-evaluation",
        "v5_protocol_sha256": V5_SHA, "v5_prior_manifest_sha256": PRIOR_MANIFEST_SHA,
        "v5_prior_dataset_sha256": PRIOR_DATASET_SHA, "g0_protocol_sha256": G0_SHA,
        "g0_manifest_sha256": G0_MANIFEST_SHA, "g0_ledger_sha256": G0_LEDGER_SHA,
        "cells": cells, "seed": seed, "total_steps": steps,
        "candidate_steps": sorted({steps // 2, steps}), "first_update_step": 1000,
        "policy_takeover_step": 2000, "batch_size": 64, "offline_batch_size": 32,
        "online_batch_size": 32, "replay_capacity": 100000,
        "frame_skip": 4, "max_steps": 2000, "obstacles": True, "reward_shaping": False,
        "learner_config": asdict(config), "runtime": current_runtime(),
        "source_hashes": hashes, "core_hour_cap": None, "official_performance_claim": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {"protocol": output, "sha256": sha256_file(target)}


def preflight(protocol_path: str, expected_sha: str) -> tuple[dict, RLPDConfig, FrameStackReplay]:
    path = pinned(protocol_path, expected_sha, "experiments")
    protocol = json.loads(path.read_text())
    v5, g0, cells = historical_evidence()
    config = config_from_v5(v5)
    if (protocol.get("format") != FORMAT or protocol.get("status") != "frozen"
            or protocol.get("role") != "reused-TRAIN-online-RLPD-not-G1-or-evaluation"
            or protocol.get("v5_protocol_sha256") != V5_SHA
            or protocol.get("v5_prior_manifest_sha256") != PRIOR_MANIFEST_SHA
            or protocol.get("v5_prior_dataset_sha256") != PRIOR_DATASET_SHA
            or protocol.get("g0_protocol_sha256") != G0_SHA
            or protocol.get("g0_manifest_sha256") != G0_MANIFEST_SHA
            or protocol.get("g0_ledger_sha256") != G0_LEDGER_SHA
            or protocol.get("cells") != cells or protocol.get("learner_config") != asdict(config)
            or type(protocol.get("seed")) is not int or protocol["seed"] < 0
            or type(protocol.get("total_steps")) is not int or protocol["total_steps"] <= 2000
            or protocol.get("candidate_steps") != sorted({protocol["total_steps"] // 2, protocol["total_steps"]})
            or protocol.get("first_update_step") != 1000 or protocol.get("policy_takeover_step") != 2000
            or protocol.get("batch_size") != 64 or protocol.get("offline_batch_size") != 32
            or protocol.get("online_batch_size") != 32 or protocol.get("replay_capacity") != 100000
            or protocol.get("frame_skip") != 4 or protocol.get("max_steps") != 2000
            or protocol.get("obstacles") is not True or protocol.get("reward_shaping") is not False
            or protocol.get("core_hour_cap") is not None
            or protocol.get("official_performance_claim") is not False):
        raise ValueError("new-host protocol differs from the frozen V5/G0 reused-TRAIN contract")
    sources = set(g0["source_hashes"]) | NEW_SOURCES
    hashes = protocol.get("source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != sources:
        raise ValueError("new-host source closure is incomplete")
    for relative, expected in hashes.items():
        if sha256_file(source_path(relative)) != expected:
            raise ValueError(f"new-host executable source drifted: {relative}")
    if protocol.get("runtime") != current_runtime():
        raise ValueError("new-host runtime/dependency inventory changed after freeze")
    offline, digest, _ = load_offline_replay(ROOT / PRIOR_DIR, v5, seed=protocol["seed"] + 0x0FF1)
    if digest != PRIOR_DATASET_SHA or offline.valid_count != len(offline):
        raise ValueError("original V5 prior replay failed validation")
    return protocol, config, offline


def journal(handle, row: dict, *, sync: bool = False) -> None:
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def write_receipt(path: Path, row: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        journal(handle, row, sync=True)


def save_checkpoint(output: Path, protocol: dict, protocol_sha: str, agent: PixelRLPDAgent,
                    offline: FrameStackReplay, online: FrameStackReplay, cell: dict,
                    episode_id: int, episode_steps: int) -> dict:
    directory = output / "checkpoints" / f"step-{agent.environment_steps:09d}"
    directory.mkdir(parents=True, exist_ok=False)
    contract = {"partition": "TRAIN", "reused_g0_cells": protocol["cells"],
                "v5_prior_dataset_sha256": PRIOR_DATASET_SHA, "obstacles": True,
                "reward_shaping": False, "frame_skip": 4, "max_steps": 2000}
    actor = agent.export_actor(directory / "actor.pt", source_sha256=canonical_sha256(protocol["source_hashes"]),
                               protocol_sha256=protocol_sha, training_seed=protocol["seed"],
                               environment_contract=contract)
    state = agent.checkpoint_state(offline_replay=offline, online_replay=online,
                                   trainer_state={"current_cell": cell, "episode_id": episode_id,
                                                  "episode_steps": episode_steps,
                                                  "exact_resume_supported": False})
    state["newhost_lineage"] = {"protocol_sha256": protocol_sha, "v5_protocol_sha256": V5_SHA,
                                "v5_prior_dataset_sha256": PRIOR_DATASET_SHA,
                                "g0_ledger_sha256": G0_LEDGER_SHA}
    temporary = directory / "checkpoint.pt.tmp"
    with temporary.open("xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    path = directory / "checkpoint.pt"
    temporary.replace(path)
    record = {"environment_steps": agent.environment_steps, "gradient_steps": agent.gradient_steps,
              "actor_sha256": sha256_file(actor), "checkpoint_sha256": sha256_file(path),
              "protocol_sha256": protocol_sha, "v5_prior_dataset_sha256": PRIOR_DATASET_SHA,
              "actor_path": str(actor.relative_to(output)), "checkpoint_path": str(path.relative_to(output)),
              "episode_boundary": episode_steps == 0, "exact_resume_supported": False}
    write_receipt(directory / "candidate.json", record)
    return record


def train_loop(output: Path, protocol: dict, protocol_sha: str, config: RLPDConfig,
               offline: FrameStackReplay, *, env_factory=None, agent_factory=None,
               online_capacity: int | None = None) -> dict:
    env_factory = env_factory or build_env
    agent_factory = agent_factory or PixelRLPDAgent
    seed = protocol["seed"]
    seed_everything(seed)
    agent = agent_factory(config, seed=seed, device="cuda")
    online = FrameStackReplay(online_capacity or protocol["replay_capacity"],
                              seed=seed + 0x0A11, source="online")
    cell_rng = np.random.default_rng(seed + 0x5A11)
    action_rng = np.random.default_rng(seed + 0xAC71)
    started = time.perf_counter()
    completed = 0
    last_metrics: dict = {}
    candidates: list[dict] = []
    episode_id = -1
    episode_steps = 0
    episode_reward = 0.0
    cell: dict = {}
    env = None
    collector = None
    observation = None
    with (output / "attempts.jsonl").open("x", encoding="utf-8") as attempts, (output / "steps.jsonl").open("x", encoding="utf-8") as steps, (output / "metrics.jsonl").open("x", encoding="utf-8") as metrics:
        try:
            for index in range(protocol["total_steps"]):
                if collector is None:
                    episode_id += 1
                    cell = protocol["cells"][int(cell_rng.integers(0, len(protocol["cells"])))]
                    if cell not in protocol["cells"] or cell["track_id"] != 1 or cell["obstacles"] is not True:
                        raise ValueError("attempted a road outside the consumed G0 catalog")
                    journal(attempts, {"event": "reset_intent", "episode_id": episode_id,
                                       "global_step": completed, "cell": cell}, sync=True)
                    env = env_factory(cell["track_id"], cell["geometry_seed"], 2000, 4,
                                      reward_shaping=False, obstacles=True)
                    collector = EpisodeCollector(env)
                    observation, reset_info = collector.reset()
                    journal(attempts, {"event": "reset", "episode_id": episode_id,
                                       "global_step": completed, "cell": cell,
                                       "reset_observation_sha256": array_sha256(ObservationSpec().to_uint8(observation)),
                                       "reset_info": reset_info}, sync=True)
                    episode_steps = 0
                    episode_reward = 0.0
                assert collector is not None and observation is not None and env is not None
                if index < protocol["policy_takeover_step"]:
                    action = action_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    behavior = "random"
                else:
                    action, _, _ = agent.act_with_details(observation, deterministic=False)
                    behavior = "policy"
                transition = collector.step(action)
                assert transition.applied_action is not None
                completed = index + 1
                episode_steps += 1
                episode_reward += transition.reward
                journal(steps, {"event": "step", "global_step": completed,
                                "episode_id": episode_id, "episode_step": episode_steps,
                                "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                                "behavior": behavior, "proposed_action": np.asarray(action).tolist(),
                                "executed_action": transition.action.tolist(),
                                "applied_action": transition.applied_action.tolist(),
                                "reward": transition.reward, "terminated": transition.terminated,
                                "truncated": transition.truncated, "terminal": transition.terminal,
                                "observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.observation)),
                                "next_observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.next_observation))})
                online.add(transition_to_replay(transition, action, episode_id, cell["geometry_seed"]))
                if index >= protocol["first_update_step"]:
                    batch = sample_balanced_batch(offline, online, batch_size=64, offline_count=32)
                    last_metrics = agent.update(batch)
                    if (last_metrics.get("offline_samples"), last_metrics.get("online_samples")) != (32, 32):
                        raise RuntimeError("learner update did not consume exactly 32:32 samples")
                agent.environment_steps = completed
                if transition.done:
                    journal(attempts, {"event": "end", "episode_id": episode_id,
                                       "global_step": completed, "cell": cell, "steps": episode_steps,
                                       "reward": episode_reward, "finished": bool(transition.info.get("finished", False)),
                                       "progress": float(transition.info.get("progress", 0.0)),
                                       "damage": float(transition.info.get("damage", 0.0)),
                                       "retire_reason": transition.info.get("retire_reason"),
                                       "terminated": transition.terminated,
                                       "truncated": transition.truncated}, sync=True)
                    env.close()
                    env = None
                    collector = None
                    observation = None
                    episode_steps = 0
                else:
                    observation = transition.next_observation
                if completed % 1000 == 0 or completed in protocol["candidate_steps"]:
                    row = {"environment_steps": completed, "gradient_steps": agent.gradient_steps,
                           "online_replay_size": len(online), "online_replay_valid": online.valid_count,
                           "offline_replay_size": len(offline), "v5_prior_dataset_sha256": PRIOR_DATASET_SHA,
                           "vm_hwm_kib": vm_hwm_kib(), "elapsed_seconds": time.perf_counter() - started,
                           "gpu_allocated_bytes": (torch.cuda.memory_allocated(agent.device)
                                                   if agent.device.type == "cuda" else 0), **last_metrics}
                    journal(metrics, row, sync=True)
                    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
                if completed in protocol["candidate_steps"]:
                    candidate = save_checkpoint(output, protocol, protocol_sha, agent, offline, online,
                                                cell, episode_id, episode_steps)
                    candidates.append(candidate)
                    journal(attempts, {"event": "checkpoint", **candidate}, sync=True)
            if (completed != protocol["total_steps"]
                    or agent.gradient_steps != completed - protocol["first_update_step"]):
                raise RuntimeError("learner did not complete its frozen decision/update budget")
            result = {"format": "haic-rlpd-newhost-result-v1", "status": "complete",
                      "protocol_sha256": protocol_sha, "environment_steps": completed,
                      "gradient_steps": agent.gradient_steps, "completed_episodes": episode_id + int(episode_steps == 0),
                      "open_episode_steps": episode_steps, "candidates": candidates,
                      "v5_protocol_sha256": V5_SHA, "v5_prior_dataset_sha256": PRIOR_DATASET_SHA,
                      "g0_ledger_sha256": G0_LEDGER_SHA, "reused_train_only": True,
                      "official_performance_claim": False, "wall_seconds": time.perf_counter() - started,
                      "attempts_sha256": sha256_file(output / "attempts.jsonl"),
                      "steps_sha256": sha256_file(output / "steps.jsonl"),
                      "metrics_sha256": sha256_file(output / "metrics.jsonl")}
            write_receipt(output / "result.json", result)
            return result
        except BaseException as exc:
            # Attempt/step ledgers survive even if a secondary failure receipt cannot fit.
            for handle in (attempts, steps, metrics):
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except OSError:
                    pass
            failure = {"format": "haic-rlpd-newhost-failure-v1", "status": "partial-not-exactly-resumable",
                       "protocol_sha256": protocol_sha, "completed_decisions": completed,
                       "gradient_steps": agent.gradient_steps, "episode_id": episode_id,
                       "episode_steps": episode_steps, "current_cell": cell,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "candidates": candidates}
            try:
                write_receipt(output / "failure.json", failure)
            except OSError as receipt_error:
                print(f"failure receipt unavailable ({receipt_error}); preserve attempt/step ledgers", file=sys.stderr)
            raise
        finally:
            if env is not None:
                env.close()


def run(protocol_path: str, expected_sha: str, run_dir: str) -> dict:
    protocol, config, offline = preflight(protocol_path, expected_sha)
    output = located(run_dir, "runs")
    output.mkdir(parents=False, exist_ok=False)
    write_receipt(output / "attempt.json", {"format": FORMAT, "status": "started",
                                            "protocol_sha256": expected_sha, "cells": protocol["cells"],
                                            "v5_prior_dataset_sha256": PRIOR_DATASET_SHA,
                                            "started_at_utc": datetime.now(timezone.utc).isoformat()})
    try:
        (output / "source").mkdir(exist_ok=False)
        for relative, expected in protocol["source_hashes"].items():
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(relative), target)
            if sha256_file(target) != expected:
                raise ValueError(f"source snapshot changed during copy: {relative}")
        shutil.copyfile(located(protocol_path, "experiments"), output / "study_protocol.json")
        if sha256_file(output / "study_protocol.json") != expected_sha:
            raise ValueError("protocol changed during snapshot")
    except BaseException as exc:
        write_receipt(output / "failure.json", {"status": "pre-reset-setup-failed", "completed_decisions": 0,
                                                "error_type": type(exc).__name__, "error": str(exc)})
        raise
    try:
        return train_loop(output, protocol, expected_sha, config, offline)
    except BaseException as exc:
        if not (output / "failure.json").exists():
            try:
                write_receipt(output / "failure.json", {
                    "status": "pre-reset-or-unjournaled-failure", "protocol_sha256": expected_sha,
                    "completed_decisions": None, "error_type": type(exc).__name__, "error": str(exc),
                    "attempts_path": "attempts.jsonl" if (output / "attempts.jsonl").exists() else None,
                    "steps_path": "steps.jsonl" if (output / "steps.jsonl").exists() else None,
                })
            except OSError as receipt_error:
                print(f"failure receipt unavailable ({receipt_error}); preserve run directory", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--output", required=True)
    freeze_parser.add_argument("--seed", type=int, required=True)
    freeze_parser.add_argument("--steps", type=int, default=131072)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", required=True)
        command.add_argument("--protocol-sha256", required=True)
        if name == "run":
            command.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(args.output, seed=args.seed, steps=args.steps)
    elif args.operation == "preflight":
        protocol, _, offline = preflight(args.protocol, args.protocol_sha256)
        result = {"status": "passed-no-environment-reset", "protocol_sha256": args.protocol_sha256,
                  "seed": protocol["seed"], "total_steps": protocol["total_steps"],
                  "prior_transitions": len(offline), "reused_g0_roads": len(protocol["cells"])}
    else:
        result = run(args.protocol, args.protocol_sha256, args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
