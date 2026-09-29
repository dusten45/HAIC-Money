"""Isolated V5-seed50 learning-state fine-tune; control versus recovery replay.

Freeze/preflight perform zero environment interaction. Run is a separate,
explicit operator action, restricted to the twelve already-consumed G0 roads.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable

import numpy as np
import torch

from common_adapter import EpisodeCollector, ObservationSpec
from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig
from haic.algorithms.rlpd.recovery import (
    FINISH_POLICY, import_learning_state, load_recovery_replay, sample_recovery_batch,
)
from haic.algorithms.rlpd.replay import FrameStackReplay
from scripts.rlpd_common import canonical_sha256, load_offline_replay, runtime_metadata, sha256_file
from scripts.rlpd_entropy_v5_common import read_entropy_v5_protocol
from scripts.train_rlpd import array_sha256, seed_everything, transition_to_replay


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
FORMAT = "haic-rlpd-recovery-learning-v1"
CHECKPOINT = ("runs/20260925-pixel-rlpd-entropy-target-ablation-v5/"
              "rlpd-author-target-seed50/checkpoints/step-000131072/checkpoint.pt")
CHECKPOINT_SHA = "2564f28eeb894d1db9d2b38f367f97f6d63b3c41cba5059243925ce351a5e0e2"
SOURCE_FILES = frozenset({
    "common_adapter.py", "train.py", "tracking.py", "env_wrapper.py", "damage.py",
    "action_representation.py", "action_smoothing.py", "requirements.txt",
    "core/__init__.py", "core/finish_line.py", "core/obstacle_contacts.py",
    "core/track_variables.py", "core/vendor/__init__.py",
    "core/vendor/car_dynamics.py", "core/vendor/car_racing.py",
    "haic/__init__.py", "haic/algorithms/__init__.py",
    "haic/algorithms/rlpd/agent.py", "haic/algorithms/rlpd/augment.py",
    "haic/algorithms/rlpd/model.py", "haic/algorithms/rlpd/replay.py",
    "haic/algorithms/rlpd/recovery.py", "scripts/__init__.py",
    "scripts/rlpd_common.py", "scripts/rlpd_entropy_common.py",
    "scripts/rlpd_entropy_v5_common.py", "scripts/train_rlpd.py",
    "scripts/train_rlpd_recovery.py",
})


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
    # Preserve the original protocol bytes/structure, not obsolete mutable source pins.
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
        raise RuntimeError("this recovery learner requires CUDA; no CPU training fallback")
    runtime = runtime_metadata(device="cuda")
    if (runtime["torch"] != "2.1.0+cu121"
            or runtime["gpu"]["name"] != "NVIDIA GeForce RTX 4060 Ti"
            or runtime["gpu"]["compute_capability"] != [8, 9]):
        raise RuntimeError("recovery Torch/GPU identity differs from RTX 4060 Ti/Torch 2.1")
    return runtime


def journal(handle, row: dict, *, sync: bool = False) -> None:
    handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    if sync:
        os.fsync(handle.fileno())


def write_receipt(path: Path, row: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        journal(handle, row, sync=True)


def source_checkpoint() -> Path:
    path = pinned(CHECKPOINT, CHECKPOINT_SHA, "runs")
    candidate = json.loads(path.with_name("candidate.json").read_text())
    if (candidate.get("checkpoint_sha256") != CHECKPOINT_SHA
            or candidate.get("training_seed") != 50
            or candidate.get("arm") != "rlpd-author-target"
            or candidate.get("protocol_sha256") != V5_SHA
            or candidate.get("offline_dataset_sha256") != PRIOR_DATASET_SHA
            or candidate.get("environment_steps") != 131072):
        raise ValueError("source checkpoint is not the V5 author-target seed50 final model")
    return path


def freeze(output: str, *, recovery_dataset: str, recovery_manifest_sha256: str,
           seed: int = 60, steps: int = 8192) -> dict:
    if type(seed) is not int or seed < 0 or steps not in (8192, 16384):
        raise ValueError("require nonnegative seed and bounded 8192/16384 decisions")
    target = located(output, "experiments")
    if target.exists():
        raise FileExistsError(target)
    v5, _, cells = historical_evidence()
    config = config_from_v5(v5)
    checkpoint = source_checkpoint()
    offline, digest, _ = load_offline_replay(ROOT / PRIOR_DIR, v5, seed=seed + 0x0FF1)
    if digest != PRIOR_DATASET_SHA or offline.valid_count != len(offline):
        raise ValueError("ordinary prior reconstruction/lineage failed")
    recovery, receipt = load_recovery_replay(
        located(recovery_dataset, "runs"), manifest_sha256=recovery_manifest_sha256,
        allowed_cells=cells, seed=seed + 0x0EC0, required_policy=FINISH_POLICY,
    )
    hashes = {name: sha256_file(source_path(name)) for name in sorted(SOURCE_FILES)}
    protocol = {
        "format": FORMAT, "status": "frozen", "partition": "TRAIN",
        "cells": cells, "seed": seed, "total_steps": steps,
        "source_checkpoint": CHECKPOINT, "source_checkpoint_sha256": sha256_file(checkpoint),
        "v5_protocol_sha256": V5_SHA, "prior_dataset_sha256": PRIOR_DATASET_SHA,
        "g0_ledger_sha256": G0_LEDGER_SHA,
        "recovery_dataset": recovery_dataset, "recovery_receipt": receipt,
        "minimum_transitions": 128, "minimum_geometries": 3,
        "gate_scope": "unique-qualified-failure-stack-executed-action",
        "eligibility_policy": FINISH_POLICY,
        "arms": {"control": {"online": 32, "prior": 32},
                 "treatment": {"online": 32, "prior": 16, "recovery": 16}},
        "learner_config": asdict(config), "first_update_step": 2,
        "online_capacity": steps, "policy_from_step": 1,
        "frame_skip": 4, "max_steps": 2000, "obstacles": True,
        "reward_shaping": False, "behavior_cloning": False,
        "fresh_online_replay": True, "reset_rng": True, "exact_continuation": False,
        "runtime": current_runtime(), "source_hashes": hashes,
        "official_performance_claim": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    del offline, recovery
    write_receipt(target, protocol)
    return {"protocol": output, "sha256": sha256_file(target)}


def preflight(protocol_path: str, expected_sha: str):
    path = pinned(protocol_path, expected_sha, "experiments")
    protocol = json.loads(path.read_text())
    v5, _, cells = historical_evidence()
    config = config_from_v5(v5)
    expected = {
        "format": FORMAT, "status": "frozen", "partition": "TRAIN", "cells": cells,
        "source_checkpoint": CHECKPOINT, "source_checkpoint_sha256": CHECKPOINT_SHA,
        "v5_protocol_sha256": V5_SHA, "prior_dataset_sha256": PRIOR_DATASET_SHA,
        "g0_ledger_sha256": G0_LEDGER_SHA,
        "minimum_transitions": 128, "minimum_geometries": 3,
        "gate_scope": "unique-qualified-failure-stack-executed-action",
        "eligibility_policy": FINISH_POLICY,
        "arms": {"control": {"online": 32, "prior": 32},
                 "treatment": {"online": 32, "prior": 16, "recovery": 16}},
        "learner_config": asdict(config), "first_update_step": 2, "policy_from_step": 1,
        "frame_skip": 4, "max_steps": 2000, "obstacles": True,
        "reward_shaping": False, "behavior_cloning": False,
        "fresh_online_replay": True, "reset_rng": True, "exact_continuation": False,
        "official_performance_claim": False,
    }
    if (any(protocol.get(key) != value for key, value in expected.items())
            or type(protocol.get("seed")) is not int or protocol["seed"] < 0
            or type(protocol.get("total_steps")) is not int
            or protocol["total_steps"] not in (8192, 16384)
            or protocol.get("online_capacity") != protocol["total_steps"]):
        raise ValueError("recovery protocol differs from bounded matched TRAIN contract")
    if set(protocol.get("source_hashes", {})) != SOURCE_FILES:
        raise ValueError("recovery source closure is incomplete")
    for name, digest in protocol["source_hashes"].items():
        if sha256_file(source_path(name)) != digest:
            raise ValueError(f"recovery source changed: {name}")
    if protocol.get("runtime") != current_runtime():
        raise ValueError("runtime changed after freeze")
    source_checkpoint()
    prior, digest, _ = load_offline_replay(ROOT / PRIOR_DIR, v5, seed=protocol["seed"] + 0x0FF1)
    if digest != PRIOR_DATASET_SHA or prior.valid_count != len(prior):
        raise ValueError("ordinary prior failed verification")
    recovery, receipt = load_recovery_replay(
        located(protocol["recovery_dataset"], "runs"),
        manifest_sha256=protocol["recovery_receipt"]["manifest_sha256"],
        allowed_cells=cells, seed=protocol["seed"] + 0x0EC0, required_policy=FINISH_POLICY,
    )
    if receipt != protocol["recovery_receipt"]:
        raise ValueError("recovery source counts/lineage changed")
    return protocol, config, prior, recovery


def train_loop(output: Path, protocol: dict, protocol_sha: str, config, prior, recovery,
               *, arm: str, env_factory=None, agent_factory: Callable = PixelRLPDAgent,
               learning_state: dict | None = None) -> dict:
    if arm not in protocol["arms"]:
        raise ValueError("unknown arm")
    if env_factory is None:
        from train import build_env
        env_factory = build_env
    seed = protocol["seed"]
    seed_everything(seed)
    agent = agent_factory(config, seed=seed, device="cuda")
    if learning_state is None:
        # Trusted local SHA-pinned checkpoint, not an untrusted model upload.
        learning_state = torch.load(source_checkpoint(), map_location="cpu", weights_only=False)
    assert isinstance(learning_state, dict)
    lineage = import_learning_state(agent, learning_state, seed=seed)
    del learning_state
    online = FrameStackReplay(protocol["online_capacity"], seed=seed + 0x0A11, source="online")
    cell_rng = np.random.default_rng(seed + 0x5A11)
    env = None
    collector = None
    observation = None
    episode_id = -1
    episode_steps = 0
    completed = 0
    cell = None
    started = time.perf_counter()
    with (output / "attempts.jsonl").open("x") as attempts, \
            (output / "steps.jsonl").open("x") as steps, \
            (output / "metrics.jsonl").open("x") as metrics:
        try:
            for index in range(protocol["total_steps"]):
                if collector is None:
                    episode_id += 1
                    cell = protocol["cells"][int(cell_rng.integers(len(protocol["cells"])))]
                    journal(attempts, {"event": "reset_intent", "cell": cell,
                                       "episode_id": episode_id, "global_step": completed}, sync=True)
                    env = env_factory(cell["track_id"], cell["geometry_seed"], 2000, 4,
                                      reward_shaping=False, obstacles=True)
                    collector = EpisodeCollector(env)
                    observation, _ = collector.reset()
                    episode_steps = 0
                    journal(attempts, {"event": "reset", "cell": cell, "episode_id": episode_id,
                                       "observation_sha256": array_sha256(ObservationSpec().to_uint8(observation))}, sync=True)
                assert observation is not None and cell is not None and env is not None
                action = agent.act(observation, deterministic=False)
                transition = collector.step(action)
                assert transition.applied_action is not None
                completed = index + 1
                episode_steps += 1
                online.add(transition_to_replay(transition, action, episode_id, cell["geometry_seed"]))
                agent.environment_steps = completed
                journal(steps, {"global_step": completed, "episode_id": episode_id,
                                "episode_step": transition.step, "cell": cell,
                                "proposed_action": action.tolist(), "executed_action": transition.action.tolist(),
                                "applied_action": transition.applied_action.tolist(), "reward": transition.reward,
                                "terminated": transition.terminated, "truncated": transition.truncated,
                                "terminal": transition.terminal,
                                "observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.observation)),
                                "next_observation_sha256": array_sha256(ObservationSpec().to_uint8(transition.next_observation))})
                if completed >= protocol["first_update_step"]:
                    batch = sample_recovery_batch(online, prior, recovery, arm=arm)
                    if batch["source_counts"] != protocol["arms"][arm]:
                        raise RuntimeError("fixed source counts changed")
                    values = agent.update(batch)
                    journal(metrics, {**values, "global_step": completed,
                                      "source_counts": batch["source_counts"],
                                      "sample_rows": {name: batch[name].tolist() for name in (
                                          "source", "indices", "episode_id", "step", "track_id", "geometry_seed")}})
                if transition.done:
                    journal(attempts, {"event": "end", "cell": cell, "episode_id": episode_id,
                                       "global_step": completed, "episode_steps": episode_steps,
                                       "finished": bool(transition.info.get("finished", False)),
                                       "progress": transition.info.get("progress", 0),
                                       "damage": transition.info.get("damage", 0),
                                       "terminated": transition.terminated, "truncated": transition.truncated}, sync=True)
                    env.close()
                    env = collector = observation = None
                    episode_steps = 0
                else:
                    observation = transition.next_observation
            if agent.gradient_steps != completed - protocol["first_update_step"] + 1:
                raise RuntimeError("bounded update count mismatch")
            contract = {"partition": "TRAIN", "cells": protocol["cells"], "frame_skip": 4,
                        "max_steps": 2000, "obstacles": True, "reward_shaping": False,
                        "arm": arm, "source_counts": protocol["arms"][arm],
                        "recovery_receipt": protocol["recovery_receipt"], "lineage": lineage}
            actor = agent.export_actor(output / "actor.pt", source_sha256=canonical_sha256(protocol["source_hashes"]),
                                       protocol_sha256=protocol_sha, training_seed=seed, environment_contract=contract)
            state = agent.checkpoint_state(offline_replay=prior, online_replay=online,
                                           trainer_state={"episode_id": episode_id, "open_episode_steps": episode_steps,
                                                          "exact_resume_supported": False})
            state["recovery_lineage"] = {**contract, "protocol_sha256": protocol_sha,
                                         "source_checkpoint_sha256": CHECKPOINT_SHA}
            checkpoint = output / "checkpoint.pt"
            with checkpoint.open("xb") as handle:
                torch.save(state, handle)
            result = {"format": FORMAT, "status": "complete", "arm": arm,
                      "protocol_sha256": protocol_sha, "environment_steps": completed,
                      "gradient_steps": agent.gradient_steps, "lineage": lineage,
                      "open_episode_steps": episode_steps, "episode_boundary": episode_steps == 0,
                      "exact_resume_supported": False, "recovery_receipt": protocol["recovery_receipt"],
                      "prior_dataset_sha256": PRIOR_DATASET_SHA, "source_checkpoint_sha256": CHECKPOINT_SHA,
                      "actor_sha256": sha256_file(actor), "checkpoint_sha256": sha256_file(checkpoint),
                      "wall_seconds": time.perf_counter() - started, "official_performance_claim": False,
                      "ledger_hashes": {name: sha256_file(output / name)
                                        for name in ("attempts.jsonl", "steps.jsonl", "metrics.jsonl")}}
            write_receipt(output / "result.json", result)
            return result
        except BaseException as exc:
            write_receipt(output / "failure.json", {
                "status": "partial-not-exactly-resumable", "arm": arm, "protocol_sha256": protocol_sha,
                "completed_decisions": completed, "gradient_steps": agent.gradient_steps,
                "open_episode_steps": episode_steps, "cell": cell,
                "error_type": type(exc).__name__, "error": str(exc), "lineage": lineage,
            })
            raise
        finally:
            if env is not None:
                env.close()


def run(protocol_path: str, expected_sha: str, run_dir: str, *, arm: str) -> dict:
    protocol, config, prior, recovery = preflight(protocol_path, expected_sha)
    if arm not in protocol["arms"]:
        raise ValueError("unknown arm")
    output = located(run_dir, "runs")
    output.mkdir(exist_ok=False)
    write_receipt(output / "attempt.json", {"format": FORMAT, "arm": arm, "protocol_sha256": expected_sha,
                                            "status": "started", "cells": protocol["cells"],
                                            "source_checkpoint_sha256": CHECKPOINT_SHA})
    try:
        for name, digest in protocol["source_hashes"].items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path(name), target)
            if sha256_file(target) != digest:
                raise ValueError("source snapshot changed")
        shutil.copyfile(pinned(protocol_path, expected_sha, "experiments"), output / "study_protocol.json")
        if sha256_file(output / "study_protocol.json") != expected_sha:
            raise ValueError("protocol snapshot changed")
        return train_loop(output, protocol, expected_sha, config, prior, recovery, arm=arm)
    except BaseException as exc:
        if not (output / "failure.json").exists():
            write_receipt(output / "failure.json", {"status": "pre-reset-or-unjournaled-failure",
                                                     "error_type": type(exc).__name__, "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--output", required=True)
    command.add_argument("--recovery-dataset", required=True)
    command.add_argument("--recovery-manifest-sha256", required=True)
    command.add_argument("--seed", type=int, default=60)
    command.add_argument("--steps", type=int, choices=(8192, 16384), default=8192)
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--protocol", required=True)
        command.add_argument("--protocol-sha256", required=True)
        if name == "run":
            command.add_argument("--run-dir", required=True)
            command.add_argument("--arm", choices=("control", "treatment"), required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(args.output, recovery_dataset=args.recovery_dataset,
                        recovery_manifest_sha256=args.recovery_manifest_sha256, seed=args.seed, steps=args.steps)
    elif args.operation == "preflight":
        protocol, _, prior, recovery = preflight(args.protocol, args.protocol_sha256)
        result = {"status": "passed-zero-reset", "prior_rows": prior.valid_count,
                  "recovery_rows": recovery.valid_count, "total_steps": protocol["total_steps"]}
    else:
        result = run(args.protocol, args.protocol_sha256, args.run_dir, arm=args.arm)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
