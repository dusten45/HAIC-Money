"""Export a recorded local-simulator run log as an annotated MP4 video.

The simulator stores optional camera frames as base64 encoded JPEGs in a run
log.  This module keeps video generation separate from the physics runner so
that exporting a video never re-runs an agent or changes a simulation result.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .logging import load_run_log
from .map import default_artifact_root
from .simulation_types import RunLog, RunStep


def _decode_frame(encoded: str, index: int) -> np.ndarray:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frames[{index}] is not valid base64 JPEG data") from error
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"frames[{index}] is not a decodable image")
    return image


def _overlay_step(image: np.ndarray, step: RunStep) -> np.ndarray:
    """Add compact telemetry without changing the underlying camera image."""

    output = image.copy()
    height, width = output.shape[:2]
    if step.collision:
        cv2.rectangle(output, (0, 0), (width - 1, height - 1), (40, 40, 255), 7)

    action = ", ".join(f"{value:+.2f}" for value in step.action)
    velocity = float(np.hypot(step.velocity[0], step.velocity[1]))
    lines = (
        f"step {step.step + 1}  t={step.sim_time_s:.2f}s",
        f"action [steer, gas, brake] = [{action}]",
        f"progress {step.progress:.3f}  speed {velocity:.2f}  damage {step.damage:.3f}",
    )
    if step.collision:
        lines += ("COLLISION",)
    if step.terminated or step.truncated:
        endpoint = "TERMINATED" if step.terminated else "TRUNCATED"
        lines += (endpoint,)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.8, width / 1500.0))
    line_height = max(20, int(round(27 * font_scale / 0.6)))
    margin = max(8, int(round(width / 160.0)))
    box_height = margin * 2 + line_height * len(lines)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, box_height), (8, 12, 20), -1)
    cv2.addWeighted(overlay, 0.78, output, 0.22, 0, output)
    for line_index, text in enumerate(lines):
        color = (70, 110, 255) if text in {"COLLISION", "TERMINATED"} else (235, 245, 250)
        if text == "TRUNCATED":
            color = (100, 210, 255)
        baseline = margin + line_height * (line_index + 1) - 5
        cv2.putText(
            output,
            text,
            (margin, baseline),
            font,
            font_scale,
            color,
            1 if width < 1000 else 2,
            cv2.LINE_AA,
        )
    return output


def export_run_video(
    run_log: RunLog,
    output_path: Path,
    *,
    fps: float | None = None,
) -> Path:
    """Export embedded run frames as an annotated MP4 and return its path.

    ``run_log`` must have been generated with ``record_frames=True``.  The
    frame and step counts must match so that telemetry cannot silently drift
    away from the camera image it describes.
    """

    frames = run_log.frames
    if not frames:
        raise ValueError(
            "run log has no embedded frames; rerun local_simulator.run with --record-frames"
        )
    if len(frames) != len(run_log.steps):
        raise ValueError(
            "run log frame count does not match step count "
            f"({len(frames)} frames for {len(run_log.steps)} steps)"
        )

    first = _decode_frame(frames[0], 0)
    height, width = first.shape[:2]
    if fps is None:
        raw_control_hz = run_log.run.get("control_hz", 12.5)
        try:
            fps = float(raw_control_hz)
        except (TypeError, ValueError) as error:
            raise ValueError("run.control_hz must be numeric when fps is omitted") from error
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer for {output}")
    try:
        for index, (encoded, step) in enumerate(zip(frames, run_log.steps)):
            image = first if index == 0 else _decode_frame(encoded, index)
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"frames[{index}] has size {image.shape[1]}x{image.shape[0]}, "
                    f"expected {width}x{height}"
                )
            writer.write(_overlay_step(image, step))
    finally:
        writer.release()
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a recorded local-simulator run log as an annotated MP4."
    )
    parser.add_argument("--run", dest="run_path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_artifact_root() / "videos" / "run.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="video frame rate; defaults to run.control_hz",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        output = export_run_video(load_run_log(args.run_path), args.output, fps=args.fps)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["export_run_video"]
