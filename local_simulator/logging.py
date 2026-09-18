from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .environment import TrackSnapshot
from .schema import LEGACY_SCHEMA_VERSION, RUN_SCHEMA_VERSION
from .simulation_types import RunLog, RunStep


def _track_to_dict(track: TrackSnapshot) -> dict[str, Any]:
    return {
        "points": [list(point) for point in track.points],
        "width": track.width,
    }


def _track_from_dict(payload: object) -> TrackSnapshot:
    if not isinstance(payload, Mapping):
        raise ValueError("track must be an object")
    raw_points = payload.get("points")
    if not isinstance(raw_points, (list, tuple)):
        raise ValueError("track.points must be an array")
    points: list[tuple[float, float, float, float]] = []
    for index, raw_point in enumerate(raw_points):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 4:
            raise ValueError(f"track.points[{index}] must contain four values")
        try:
            points.append(tuple(float(value) for value in raw_point))  # type: ignore[arg-type]
        except (TypeError, ValueError) as error:
            raise ValueError(f"track.points[{index}] must contain numbers") from error
    try:
        width = float(payload["width"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("track.width must be a number") from error
    return TrackSnapshot(points=tuple(points), width=width)


def _step_to_dict(step: RunStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "sim_time_s": step.sim_time_s,
        "action": list(step.action),
        "position": list(step.position),
        "angle": step.angle,
        "velocity": list(step.velocity),
        "progress": step.progress,
        "damage": step.damage,
        "collision": step.collision,
        "terminated": step.terminated,
        "truncated": step.truncated,
    }


def _step_from_dict(payload: object) -> RunStep:
    if not isinstance(payload, Mapping):
        raise ValueError("each step must be an object")

    def vector(name: str, size: int) -> tuple[float, ...]:
        raw = payload.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != size:
            raise ValueError(f"steps[].{name} must contain {size} values")
        try:
            return tuple(float(value) for value in raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"steps[].{name} must contain numbers") from error

    try:
        return RunStep(
            step=int(payload["step"]),
            sim_time_s=float(payload["sim_time_s"]),
            action=vector("action", 3),
            position=vector("position", 2),
            angle=float(payload["angle"]),
            velocity=vector("velocity", 2),
            progress=float(payload["progress"]),
            damage=float(payload["damage"]),
            collision=bool(payload["collision"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid run step") from error


def run_log_to_dict(run_log: RunLog) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run": run_log.run,
        "track": _track_to_dict(run_log.track),
        "steps": [_step_to_dict(step) for step in run_log.steps],
        "summary": run_log.summary,
    }
    if run_log.frames is not None:
        result["frames"] = list(run_log.frames)
    return result


def run_log_from_dict(payload: object) -> RunLog:
    if not isinstance(payload, Mapping):
        raise ValueError("run log must be an object")
    if payload.get("schema_version") not in (
        LEGACY_SCHEMA_VERSION,
        RUN_SCHEMA_VERSION,
    ):
        raise ValueError(
            f"unsupported schema_version: {payload.get('schema_version')}"
        )
    raw_run = payload.get("run")
    raw_summary = payload.get("summary")
    raw_steps = payload.get("steps")
    if not isinstance(raw_run, Mapping):
        raise ValueError("run must be an object")
    if not isinstance(raw_summary, Mapping):
        raise ValueError("summary must be an object")
    if not isinstance(raw_steps, (list, tuple)):
        raise ValueError("steps must be an array")
    raw_frames = payload.get("frames")
    if raw_frames is not None and not isinstance(raw_frames, (list, tuple)):
        raise ValueError("frames must be an array when present")
    return RunLog(
        schema_version=RUN_SCHEMA_VERSION,
        run=dict(raw_run),
        track=_track_from_dict(payload.get("track")),
        steps=tuple(_step_from_dict(step) for step in raw_steps),
        summary=dict(raw_summary),
        frames=None if raw_frames is None else tuple(str(frame) for frame in raw_frames),
    )


def save_run_log(run_log: RunLog, output_path: Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run_log_to_dict(run_log), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_run_log(input_path: Path) -> RunLog:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return run_log_from_dict(payload)
