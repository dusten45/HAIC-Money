"""Collect a frozen Dreamer-only, training-excluded open-loop dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from common_adapter import ActionAdapter, EpisodeCollector, ObservationSpec
from train import build_env


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protocol_sources(protocol: dict) -> None:
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("protocol must freeze source_sha256 before development collection")
    for relative_path, expected in source_hashes.items():
        if not isinstance(relative_path, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("source_sha256 entries must use relative paths and SHA-256 hex digests")
        source_path = (ROOT / relative_path).resolve()
        try:
            source_path.relative_to(ROOT.resolve())
        except ValueError as error:
            raise ValueError("protocol source paths must remain inside the repository") from error
        if not source_path.is_file() or sha256(source_path) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative_path}")


def load_development_cells(protocol: dict) -> list[tuple[int, int]]:
    development = protocol.get("training_development")
    if not isinstance(development, dict):
        raise ValueError("protocol must define training_development")
    cells = development.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("training_development.cells must be an explicit non-empty list")

    tracks = development.get("track_ids")
    seeds = development.get("seeds")
    if not isinstance(tracks, list) or not isinstance(seeds, list):
        raise ValueError("training_development must declare track_ids and seeds")
    if any(type(track) is not int or track < 1 for track in tracks):
        raise ValueError("development track IDs must be positive integers")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("development geometry seeds must be uint32 integers")

    cells_out = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each development cell must be an object")
        track, seed = cell.get("track_id"), cell.get("seed")
        if type(track) is not int or track not in tracks:
            raise ValueError("development cell has an undeclared track ID")
        if type(seed) is not int or seed not in seeds:
            raise ValueError("development cell has an undeclared geometry seed")
        cells_out.append((track, seed))

    if len(set(cells_out)) != len(cells_out):
        raise ValueError("duplicate training-development cells")
    if len({seed for _, seed in cells_out}) != len(cells_out):
        raise ValueError("each development cell must use a distinct geometry seed")
    if set(seeds) != {seed for _, seed in cells_out}:
        raise ValueError("declared development seeds must exactly match the cells")

    evaluator_seeds = {
        seed
        for partition in protocol.get("partitions", {}).values()
        for seed in partition.get("seeds", [])
    }
    if evaluator_seeds.intersection(seeds):
        raise ValueError("development seeds must be disjoint from screen/confirmation/blind")
    if not set(seeds).issubset(set(protocol.get("reserved_training_seeds", []))):
        raise ValueError("all development seeds must be excluded from every training track")
    return cells_out


def collect(protocol_path: Path, output_path: Path) -> Path:
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    verify_protocol_sources(protocol)
    if protocol.get("frame_skip") != 4:
        raise ValueError("Dreamer development collection freezes frame_skip=4")
    max_steps = protocol.get("max_steps")
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("protocol max_steps must be a positive integer")
    development = protocol["training_development"]
    if development.get("collector") != "uniform_native_random":
        raise ValueError("only the preregistered uniform_native_random collector is supported")
    collector_seed = development.get("collector_seed")
    if type(collector_seed) is not int or not 0 <= collector_seed < 2**32:
        raise ValueError("training_development.collector_seed must be a uint32 integer")
    cells = load_development_cells(protocol)
    if output_path.exists():
        raise FileExistsError(output_path)

    rng = np.random.default_rng(collector_seed)
    observation_spec = ObservationSpec()
    action_adapter = ActionAdapter()
    episode_observations = []
    episode_actions = []
    episode_applied_actions = []
    episode_rewards = []
    episode_terminated = []
    episode_truncated = []
    episode_terminal = []
    observation_offsets = [0]
    transition_offsets = [0]
    episode_rows = []

    for cell_index, (track_id, seed) in enumerate(cells):
        env = build_env(
            track_id,
            seed,
            max_steps,
            protocol["frame_skip"],
            reward_shaping=False,
            obstacles=True,
        )
        collector = EpisodeCollector(
            env,
            observation_spec=observation_spec,
            action_adapter=action_adapter,
            gamma=1.0,
        )
        try:
            observation, reset_info = collector.reset()
            if reset_info.get("track_id") != track_id or reset_info.get("seed") != seed:
                raise RuntimeError("local environment reset did not honor its frozen cell")

            transitions = []
            for _ in range(max_steps):
                proposed_action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                transition = collector.step(proposed_action)
                transitions.append(transition)
                if transition.done:
                    break
                observation = transition.next_observation

            if not transitions:
                raise RuntimeError(f"development cell produced no decisions: {(track_id, seed)}")
            last = transitions[-1]
            obs_frames = [t.observation for t in transitions] + [last.next_observation]
            episode_observations.extend(observation_spec.to_uint8(frame) for frame in obs_frames)
            episode_actions.extend(t.action for t in transitions)
            episode_applied_actions.extend(t.applied_action for t in transitions)
            episode_rewards.extend(float(t.reward) for t in transitions)
            episode_terminated.extend(bool(t.terminated) for t in transitions)
            episode_truncated.extend(bool(t.truncated) for t in transitions)
            episode_terminal.extend(bool(t.terminal) for t in transitions)
            observation_offsets.append(observation_offsets[-1] + len(transitions) + 1)
            transition_offsets.append(transition_offsets[-1] + len(transitions))
            episode_rows.append({
                "episode_id": cell_index,
                "track_id": track_id,
                "geometry_seed": seed,
                "decisions": len(transitions),
                "collector": "uniform_native_random",
                "finished": bool(last.info.get("finished", False)),
                "retire_reason": last.info.get("retire_reason"),
                "progress": float(last.info.get("progress", 0.0)),
                "damage": float(last.info.get("damage", 0.0)),
                "terminated": bool(last.terminated),
                "truncated": bool(last.truncated),
                "terminal": bool(last.terminal),
                "raw_reward_sum": float(sum(t.reward for t in transitions)),
            })
        finally:
            env.close()

    source_files = (
        ROOT / "common_adapter.py",
        ROOT / "train.py",
        ROOT / "core/vendor/car_racing.py",
        ROOT / "env_wrapper.py",
        Path(__file__).resolve(),
    )
    metadata = {
        "format": "haic-dreamerv3-development-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "collector_seed": collector_seed,
        "frame_skip": protocol["frame_skip"],
        "max_steps": max_steps,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
        "episodes": episode_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        with temp_path.open("wb") as archive:
            np.savez_compressed(
                archive,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                observation_offsets=np.asarray(observation_offsets, dtype=np.int64),
                transition_offsets=np.asarray(transition_offsets, dtype=np.int64),
                observations=np.stack(episode_observations).astype(np.uint8, copy=False),
                actions=np.asarray(episode_actions, dtype=np.float32),
                applied_actions=np.asarray(episode_applied_actions, dtype=np.float32),
                rewards=np.asarray(episode_rewards, dtype=np.float32),
                terminated=np.asarray(episode_terminated, dtype=bool),
                truncated=np.asarray(episode_truncated, dtype=bool),
                terminal=np.asarray(episode_terminal, dtype=bool),
            )
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = collect(args.protocol.resolve(), args.output.resolve())
    print(path)


if __name__ == "__main__":
    main()
