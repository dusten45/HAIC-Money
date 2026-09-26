"""Collect a frozen DrQ teacher dataset for the pixel-RLPD pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from common_adapter import EpisodeCollector, ObservationSpec
from drq_v2 import load_exported_actor
from scripts.rlpd_common import (
    DATASET_FORMAT,
    ROOT,
    read_protocol,
    runtime_metadata,
    sha256_file,
    snapshot_sources,
    write_json,
)
from train import build_env


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-actor", type=Path)
    parser.add_argument("--max-decisions", type=int)
    return parser.parse_args()


def write_episode(
    path: Path,
    episode_id: int,
    cell: dict,
    transitions: list,
    proposed_actions: list[np.ndarray],
    initial_observation,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    latest_frames = np.stack([
        ObservationSpec().to_uint8(transition.observation)[-1]
        for transition in transitions
    ])
    final_observation = ObservationSpec().to_uint8(transitions[-1].next_observation)
    arrays = {
        "frames": latest_frames,
        "initial_observation": ObservationSpec().to_uint8(initial_observation),
        "final_observation": final_observation,
        "proposed_actions": np.stack(proposed_actions),
        "executed_actions": np.stack([transition.action.copy() for transition in transitions]),
        "applied_actions": np.stack([transition.applied_action.copy() for transition in transitions]),
        "rewards": np.asarray([transition.reward for transition in transitions], dtype=np.float32),
        "terminated": np.asarray([transition.terminated for transition in transitions], dtype=np.bool_),
        "truncated": np.asarray([transition.truncated for transition in transitions], dtype=np.bool_),
        "terminal": np.asarray([transition.terminal for transition in transitions], dtype=np.bool_),
    }
    np.savez_compressed(path, **arrays)
    terminal = transitions[-1]
    info = terminal.info
    return {
        "episode_id": episode_id,
        "path": str(path.name),
        "sha256": sha256_file(path),
        "track_id": int(cell["track_id"]),
        "geometry_seed": int(cell["geometry_seed"]),
        "steps": len(transitions),
        "reward": float(sum(item.reward for item in transitions)),
        "terminated": terminal.terminated,
        "truncated": terminal.truncated,
        "terminal": terminal.terminal,
        "finished": bool(info.get("finished", False)),
        "retire_reason": info.get("retire_reason"),
        "progress": float(info.get("progress", 0.0)),
        "teacher_actor_sha256": None,
    }


def collect(protocol_path: Path, output: Path, teacher_actor_path: Path | None = None, max_decisions=None):
    protocol_path = Path(protocol_path).resolve()
    output = Path(output).resolve()
    protocol = read_protocol(protocol_path)
    protocol_sha = sha256_file(protocol_path)
    actual_runtime = runtime_metadata(device="cpu")
    expected_runtime = protocol["runtime"]["teacher_collection"]
    if (
        actual_runtime["installed_distributions_sha256"]
        != expected_runtime["installed_distributions_sha256"]
        or actual_runtime["torch"] != expected_runtime["torch"]
        or actual_runtime["numpy"] != expected_runtime["numpy"]
        or actual_runtime.get("gpu") != expected_runtime.get("gpu")
    ):
        raise RuntimeError("teacher collection runtime differs from the frozen dependency inventory")
    teacher = protocol["teacher"]
    actor_path = Path(teacher_actor_path or ROOT / teacher["actor_path"]).resolve()
    if not actor_path.is_file() or sha256_file(actor_path) != teacher["actor_sha256"]:
        raise ValueError("frozen teacher actor is missing or has a different content hash")
    checkpoint_path = (ROOT / teacher["checkpoint_path"]).resolve()
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != teacher["checkpoint_sha256"]:
        raise ValueError("frozen teacher learner checkpoint hash does not match the protocol")
    source_protocol_path = (ROOT / teacher["source_protocol_path"]).resolve()
    if (
        not source_protocol_path.is_file()
        or sha256_file(source_protocol_path) != teacher["source_protocol_sha256"]
    ):
        raise ValueError("teacher source-run protocol changed after study freeze")
    source_config_path = (ROOT / teacher["source_config_path"]).resolve()
    if (
        not source_config_path.is_file()
        or sha256_file(source_config_path) != teacher["source_config_sha256"]
    ):
        raise ValueError("teacher source-run config changed after study freeze")
    if max_decisions is None:
        max_decisions = protocol["teacher_data_budget"]["decisions"]
    if (
        type(max_decisions) is not int
        or max_decisions != protocol["teacher_data_budget"]["decisions"]
    ):
        raise ValueError("teacher collection must use the exact frozen decision budget")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir()
    snapshot_sources(protocol, output / "source")
    (output / "study_protocol.json").write_bytes(protocol_path.read_bytes())
    write_json(output / "run_config.json", {
        "study_id": protocol["name"],
        "protocol_sha256": protocol_sha,
        "teacher_actor_sha256": teacher["actor_sha256"],
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "teacher_source_config_sha256": teacher["source_config_sha256"],
        "max_decisions": max_decisions,
        "frame_skip": protocol["frame_skip"],
        "max_steps": protocol["max_steps"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
    }, exclusive=True)
    write_json(output / "runtime.json", actual_runtime, exclusive=True)
    actor, action_adapter, observation_spec = load_exported_actor(actor_path, device="cpu")
    if observation_spec.fingerprint != ObservationSpec().fingerprint:
        raise ValueError("teacher actor observation contract differs from the HAIC contract")
    if action_adapter.spec.frame_skip != protocol["frame_skip"]:
        raise ValueError("teacher actor frame-skip contract differs from the frozen protocol")

    spent = 0
    stored = 0
    finish_cells = set()
    episodes = []
    episode_log = output / "collection.jsonl"
    episode_counter = 0
    started = time.perf_counter()
    with episode_log.open("x", encoding="utf-8") as log:
        for cell_index, cell in enumerate(protocol["teacher_data_cells"]):
            if spent >= max_decisions:
                break
            environment = build_env(
                int(cell["track_id"]),
                int(cell["geometry_seed"]),
                protocol["max_steps"],
                protocol["frame_skip"],
                reward_shaping=False,
                obstacles=True,
            )
            collector = EpisodeCollector(environment, action_adapter=action_adapter)
            transitions = []
            proposed_actions = []
            try:
                observation, reset_info = collector.reset()
                log.write(json.dumps({
                    "event": "reset",
                    "cell_index": cell_index,
                    "episode_id": episode_counter,
                    "track_id": reset_info["track_id"],
                    "geometry_seed": reset_info["seed"],
                    "obstacles": True,
                }, sort_keys=True) + "\n")
                log.flush()
                while spent < max_decisions:
                    proposed_action = actor.act(observation, deterministic=True)
                    transition = collector.step(proposed_action)
                    transitions.append(transition)
                    proposed_actions.append(np.asarray(proposed_action, dtype=np.float32).copy())
                    spent += 1
                    if transition.done:
                        relative_name = f"episode-{episode_counter:04d}.npz"
                        metadata = write_episode(
                            episodes_dir / relative_name,
                            episode_counter,
                            cell,
                            transitions,
                            proposed_actions,
                            transitions[0].observation,
                        )
                        metadata["path"] = f"episodes/{relative_name}"
                        metadata["teacher_actor_sha256"] = teacher["actor_sha256"]
                        episodes.append(metadata)
                        stored += len(transitions)
                        if metadata["finished"]:
                            finish_cells.add((metadata["track_id"], metadata["geometry_seed"]))
                        log.write(json.dumps({"event": "stored_episode", **metadata}, sort_keys=True) + "\n")
                        log.flush()
                        episode_counter += 1
                        break
                    observation = transition.next_observation
                else:
                    pass
                if transitions and not transitions[-1].done:
                    log.write(json.dumps({
                        "event": "discarded_incomplete_episode",
                        "episode_id": episode_counter,
                        "track_id": cell["track_id"],
                        "geometry_seed": cell["geometry_seed"],
                        "decisions_spent": len(transitions),
                        "reason": "teacher decision cap reached; no synthetic terminal inserted",
                    }, sort_keys=True) + "\n")
                    log.flush()
            finally:
                environment.close()

    manifest = {
        "format": DATASET_FORMAT,
        "study_id": protocol["name"],
        "study_protocol_sha256": protocol_sha,
        "teacher_actor_sha256": teacher["actor_sha256"],
        "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "teacher_source_protocol_sha256": teacher["source_protocol_sha256"],
        "teacher_source_config_sha256": teacher["source_config_sha256"],
        "decisions_spent": spent,
        "stored_decisions": stored,
        "discarded_decisions": spent - stored,
        "minimum_distinct_finishes": protocol["teacher_data_budget"]["minimum_distinct_finishes"],
        "distinct_finish_geometries": len(finish_cells),
        "episodes": episodes,
        "wall_seconds": time.perf_counter() - started,
        "python": sys.version,
        "torch": torch.__version__,
        "device": "cpu",
    }
    from scripts.rlpd_common import canonical_sha256

    manifest["dataset_sha256"] = canonical_sha256(manifest)
    write_json(output / "manifest.json", manifest, exclusive=True)
    result = {
        "study_id": protocol["name"],
        "dataset_sha256": manifest["dataset_sha256"],
        "decisions_spent": spent,
        "stored_decisions": stored,
        "discarded_decisions": spent - stored,
        "complete_episodes": len(episodes),
        "distinct_finish_geometries": len(finish_cells),
        "minimum_distinct_finishes": manifest["minimum_distinct_finishes"],
        "eligible": len(finish_cells) >= manifest["minimum_distinct_finishes"],
        "dataset_dir": str(output),
    }
    write_json(output / "result.json", result, exclusive=True)
    if not result["eligible"]:
        raise RuntimeError("teacher data eligibility gate failed; do not start student training")
    return result


def main():
    args = parse_args()
    result = collect(args.protocol, args.output, args.teacher_actor, args.max_decisions)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
