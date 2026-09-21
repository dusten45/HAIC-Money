from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

import cv2
import numpy as np

from .environment import create_environment, reset_environment, snapshot_track
from .policies import Policy
from .schema import (
    CustomMapSpec,
    MapDocument,
    RUN_SCHEMA_VERSION,
    map_fingerprint,
    map_to_dict,
)
from .simulation_types import RunLog, RunStep


NO_OP_ACTION = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
ACTION_LOW = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)


def normalize_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    try:
        result = np.asarray(action, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return NO_OP_ACTION.copy()
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        return NO_OP_ACTION.copy()
    return np.clip(result, ACTION_LOW, ACTION_HIGH).astype(np.float32)


def safe_policy_action(policy: Policy, observation: np.ndarray) -> np.ndarray:
    try:
        return normalize_action(policy.act(observation))
    except Exception:
        return NO_OP_ACTION.copy()


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


class SimulationSession:
    def __init__(
        self,
        document: MapDocument,
        policy: Policy,
        record_frames: bool = False,
    ) -> None:
        self.document = document
        self.policy = policy
        self.record_frames = record_frames
        self.environment, self.raw_environment = create_environment(
            document,
            render_mode="rgb_array" if record_frames else None,
        )
        try:
            self.observation, _reset_info = reset_environment(
                self.environment,
                document,
            )
            _call_policy_reset(policy, self.observation)
            self.track = snapshot_track(self.environment)
            self.start_time_s = float(self.raw_environment.t)
        except Exception:
            self.environment.close()
            raise
        self.steps: list[RunStep] = []
        self.frames: list[str] = []
        self.last_info: dict[str, Any] = {}
        self.done = False
        self.closed = False
        self._result: RunLog | None = None

    @classmethod
    def start(
        cls,
        document: MapDocument,
        policy: Policy,
        record_frames: bool = False,
    ) -> "SimulationSession":
        return cls(document, policy, record_frames=record_frames)

    def step(self, action: Sequence[float] | np.ndarray | None = None) -> RunStep:
        if self.closed:
            raise RuntimeError("simulation session is closed")
        if self.done:
            raise RuntimeError("simulation session has already finished")
        selected_action = (
            safe_policy_action(self.policy, self.observation)
            if action is None
            else normalize_action(action)
        )
        observation, _reward, terminated, truncated, info = self.environment.step(
            selected_action
        )
        self.observation = observation
        self.last_info = dict(info)
        raw_car = self.environment.unwrapped.car
        hull = raw_car.hull
        step_record = RunStep(
            step=len(self.steps),
            sim_time_s=float(self.raw_environment.t) - self.start_time_s,
            action=tuple(float(value) for value in selected_action),
            position=_vec2(hull.position),
            angle=float(hull.angle),
            velocity=_vec2(hull.linearVelocity),
            progress=float(info.get("progress", 0.0)),
            damage=float(info.get("damage", 0.0)),
            collision=bool(info.get("collision", False)),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self.steps.append(step_record)
        if self.record_frames:
            frame = self.raw_environment.render()
            if frame is None:
                raise RuntimeError("the environment did not return an RGB frame")
            self.frames.append(_encode_frame(frame))
        self.done = bool(terminated or truncated)
        return step_record

    def finish(self, reason: str | None = None) -> RunLog:
        if self._result is not None:
            return self._result
        frozen_steps = tuple(self.steps)
        policy_name = str(getattr(self.policy, "name", self.policy.__class__.__name__))
        map_payload = map_to_dict(self.document)
        map_id = self.document.map_id
        map_kind = self.document.map_kind
        self._result = RunLog(
            schema_version=RUN_SCHEMA_VERSION,
            run={
                "run_id": uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "map": map_payload,
                "map_ref": {
                    "map_id": map_id,
                    "map_kind": map_kind,
                    "fingerprint": map_fingerprint(self.document),
                },
                "environment": "custom-v1" if isinstance(self.document, CustomMapSpec) else "official-v1",
                "policy": {"kind": policy_name, "name": policy_name},
                "control_hz": 50.0 / self.document.frame_skip,
            },
            track=self.track,
            steps=frozen_steps,
            summary=_summary(
                self.environment,
                frozen_steps,
                self.start_time_s,
                self.last_info,
                self.document.max_steps,
                local_retire_reason=reason,
            ),
            frames=tuple(self.frames) if self.record_frames else None,
        )
        self.close()
        return self._result

    def close(self) -> None:
        if self.closed:
            return
        self.environment.close()
        self.closed = True


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "NO_OP_ACTION",
    "SimulationSession",
    "normalize_action",
    "safe_policy_action",
]
