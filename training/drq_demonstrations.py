"""Collect and serialize training-only demonstrations for native DrQ-v2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, EpisodeCollector, ObservationSpec
from drq_v2 import Uint8Replay
from training.vision_teacher import VisionCorridorAgent


DRQ_DEMONSTRATION_FORMAT = "haic-drq-demonstrations-v1"


def _close_environment(environment: Any) -> None:
    close = getattr(environment, "close", None)
    if callable(close):
        close()
    else:
        environment.environment.close()


def collect_drq_teacher_demonstrations(
    episodes: Iterable[Any],
    *,
    max_decisions: int,
    n_step: int = 3,
    gamma: float = 0.99,
    seed: int = 0,
    teacher_factory: Callable[[], Any] = VisionCorridorAgent,
    environment_factory: Callable[..., Any] | None = None,
) -> tuple[Uint8Replay, dict[str, Any]]:
    """Collect full DrQ transitions from an official-action visual teacher.

    The returned replay is separate from online experience.  Teacher actions are
    converted through the same symmetric native action adapter used by DrQ, while
    the environment receives the corresponding official HAIC action.
    """

    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("at least one explicit training episode is required")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    capacity = len(episode_list) * int(max_decisions)
    observation_spec = ObservationSpec()
    action_adapter = ActionAdapter(ActionSpec())
    if environment_factory is None:
        from training.env_factory import create_episode_environment

        environment_factory = create_episode_environment
    replay = Uint8Replay(
        capacity=capacity,
        observation_spec=observation_spec,
        action_dim=3,
        n_step=n_step,
        gamma=gamma,
        seed=seed,
    )
    episode_records: list[dict[str, Any]] = []
    for episode_id, episode in enumerate(episode_list):
        environment = environment_factory(episode, max_decisions=max_decisions)
        teacher = teacher_factory()
        collector = EpisodeCollector(
            environment,
            observation_spec=observation_spec,
            action_adapter=action_adapter,
            gamma=gamma,
        )
        steps = 0
        try:
            observation, _ = collector.reset()
            reset = getattr(teacher, "reset", None)
            if callable(reset):
                reset(observation)
            for step in range(max_decisions):
                official_action = np.asarray(teacher.act(observation), dtype=np.float32)
                native_action = action_adapter.to_native(official_action, clip=False)
                transition = collector.step(native_action)
                transition.episode_id = episode_id
                transition.step = step
                replay.add(transition)
                steps += 1
                observation = transition.next_observation
                if transition.done:
                    break
        finally:
            _close_environment(environment)
        if isinstance(episode, tuple) and len(episode) == 2:
            identity = {"track_id": int(episode[0]), "seed": int(episode[1])}
        else:
            from training.env_factory import describe_episode

            identity = describe_episode(episode)
        episode_records.append({**identity, "steps": steps})
    if replay.size < 1:
        raise RuntimeError("teacher collection produced no transitions")
    metadata = {
        "teacher": f"{teacher_factory.__module__}.{teacher_factory.__qualname__}",
        "teacher_role": "vision_corridor_training_only",
        "source_action_space": "official-[steer,gas,brake]",
        "stored_action_space": "symmetric-native-3d",
        "reward_contract": {
            "reward_shaping": False,
            "norm_reward": False,
            "collision_penalty": 0.0,
        },
        "episodes": episode_records,
        "episode_count": len(episode_records),
        "transition_count": replay.size,
        "max_decisions": int(max_decisions),
        "n_step": int(n_step),
        "gamma": float(gamma),
        "seed": int(seed),
    }
    return replay, metadata


def save_drq_demonstrations(
    path: str | Path,
    replay: Uint8Replay,
    *,
    metadata: dict[str, Any] | None = None,
    action_spec: ActionSpec | None = None,
) -> Path:
    """Write an immutable-input demonstration artifact for later training."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if replay.size < 1:
        raise ValueError("demonstration replay must not be empty")
    payload = {
        "format": DRQ_DEMONSTRATION_FORMAT,
        "observation_spec": asdict(replay.observation_spec),
        "action_spec": asdict(action_spec or ActionSpec()),
        "replay": replay.state_dict(),
        "metadata": dict(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_drq_demonstrations(
    path: str | Path,
    *,
    observation_spec: ObservationSpec | None = None,
    action_spec: ActionSpec | None = None,
) -> tuple[Uint8Replay, dict[str, Any]]:
    """Load and validate a native-action DrQ demonstration replay."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != DRQ_DEMONSTRATION_FORMAT:
        raise ValueError("unsupported DrQ demonstration format")
    saved_observation = ObservationSpec(**payload["observation_spec"])
    saved_action = ActionSpec(**payload["action_spec"])
    expected_observation = observation_spec or ObservationSpec()
    expected_action = action_spec or ActionSpec()
    if saved_observation.fingerprint != expected_observation.fingerprint:
        raise ValueError("demonstration observation contract does not match DrQ")
    if saved_action.fingerprint != expected_action.fingerprint:
        raise ValueError("demonstration action contract does not match DrQ")
    state = payload.get("replay")
    if not isinstance(state, dict):
        raise ValueError("demonstration artifact lacks replay state")
    replay = Uint8Replay(
        capacity=int(state["capacity"]),
        observation_spec=saved_observation,
        action_dim=int(state["action_dim"]),
        n_step=int(state["n_step"]),
        gamma=float(state["gamma"]),
        seed=0,
    )
    replay.load_state_dict(state)
    if replay.size < 1:
        raise ValueError("demonstration replay must not be empty")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("demonstration metadata must be a mapping")
    return replay, dict(metadata)


def _parse_episodes(value: str) -> tuple[tuple[int, int], ...]:
    episodes = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError("episodes must use track_id:seed pairs")
        track_id, seed = map(int, fields)
        if track_id < 1 or not 0 <= seed < 2**32:
            raise argparse.ArgumentTypeError("episode track IDs and seeds are out of range")
        episodes.append((track_id, seed))
    if not episodes or len(set(episodes)) != len(episodes):
        raise argparse.ArgumentTypeError("episodes must be non-empty and unique")
    return tuple(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True, type=_parse_episodes)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    replay, metadata = collect_drq_teacher_demonstrations(
        args.episodes,
        max_decisions=args.max_decisions,
        n_step=args.n_step,
        gamma=args.gamma,
        seed=args.seed,
    )
    output = save_drq_demonstrations(args.output, replay, metadata=metadata)
    print(f"wrote {replay.size} DrQ demonstration transitions: {output}")


if __name__ == "__main__":
    main()
