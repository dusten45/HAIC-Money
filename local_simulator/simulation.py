from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .environment import create_environment, reset_environment, snapshot_track
from .policies import Policy
from .schema import MapSpec, SCHEMA_VERSION, map_to_dict
from .simulation_types import RunLog, RunStep


NO_OP_ACTION = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
ACTION_LOW = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)


def _safe_action(policy: Policy, observation: np.ndarray) -> np.ndarray:
    try:
        action = np.asarray(policy.act(observation), dtype=np.float32).reshape(-1)
    except Exception:
        return NO_OP_ACTION.copy()
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        return NO_OP_ACTION.copy()
    return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)


def _call_policy_reset(policy: Policy, observation: np.ndarray) -> None:
    reset = getattr(policy, "reset", None)
    if reset is not None:
        reset(observation)


def _vec2(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _encode_frame(frame: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("failed to encode a simulation frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _summary(
    environment: Any,
    steps: tuple[RunStep, ...],
    start_time_s: float,
    last_info: dict[str, Any],
    max_steps: int,
    local_retire_reason: str | None = None,
) -> dict[str, Any]:
    raw_environment = environment.unwrapped
    finish_time_s = getattr(raw_environment, "finish_time_s", None)
    finished = finish_time_s is not None
    progress = float(last_info.get("progress", 0.0))
    damage = float(last_info.get("damage", 0.0))
    collision_count = sum(1 for step in steps if step.collision)
    retire_reason = local_retire_reason or last_info.get("retire_reason")
    if not finished and retire_reason is None:
        if (steps and steps[-1].truncated) or len(steps) >= max_steps:
            retire_reason = "max_steps"
        elif steps and steps[-1].terminated:
            retire_reason = "terminated"
    lap_time_ms = (
        round((float(finish_time_s) - start_time_s) * 1000)
        if finished
        else None
    )
    return {
        "lap_time_ms": lap_time_ms,
        "finished": finished,
        "progress": progress,
        "damage": damage,
        "collision_count": collision_count,
        "retire_reason": retire_reason,
    }


def run_episode(
    spec: MapSpec,
    policy: Policy,
    record_frames: bool = False,
) -> RunLog:
    environment, _raw_environment = create_environment(
        spec,
        render_mode="rgb_array" if record_frames else None,
    )
    steps: list[RunStep] = []
    frames: list[str] = []
    last_info: dict[str, Any] = {}
    start_time_s = 0.0
    try:
        observation, _reset_info = reset_environment(environment, spec)
        start_time_s = float(environment.unwrapped.t)
        _call_policy_reset(policy, observation)
        track = snapshot_track(environment)

        for step_index in range(spec.max_steps):
            action = _safe_action(policy, observation)
            observation, _reward, terminated, truncated, info = environment.step(action)
            last_info = dict(info)
            raw_car = environment.unwrapped.car
            hull = raw_car.hull
            step_record = RunStep(
                step=step_index,
                sim_time_s=float(environment.unwrapped.t) - start_time_s,
                action=tuple(float(value) for value in action),
                position=_vec2(hull.position),
                angle=float(hull.angle),
                velocity=_vec2(hull.linearVelocity),
                progress=float(info.get("progress", 0.0)),
                damage=float(info.get("damage", 0.0)),
                collision=bool(info.get("collision", False)),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            steps.append(step_record)
            if record_frames:
                frame = environment.unwrapped.render()
                if frame is None:
                    raise RuntimeError("the environment did not return an RGB frame")
                frames.append(_encode_frame(frame))
            if terminated or truncated:
                break

        frozen_steps = tuple(steps)
        run_metadata = {
            "run_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "map": map_to_dict(spec),
            "policy": str(getattr(policy, "name", policy.__class__.__name__)),
            "control_hz": 50.0 / spec.frame_skip,
        }
        return RunLog(
            schema_version=SCHEMA_VERSION,
            run=run_metadata,
            track=track,
            steps=frozen_steps,
            summary=_summary(
                environment,
                frozen_steps,
                start_time_s,
                last_info,
                spec.max_steps,
            ),
            frames=tuple(frames) if record_frames else None,
        )
    finally:
        environment.close()
