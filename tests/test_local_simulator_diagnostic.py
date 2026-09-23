import base64
from pathlib import Path

import cv2
import numpy as np

from local_simulator.diagnostic import write_mp4, write_report
from local_simulator.environment import TrackSnapshot
from local_simulator.simulation_types import RunLog, RunStep


def _frame(value: int) -> str:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _run_log() -> RunLog:
    steps = tuple(
        RunStep(
            step=index,
            sim_time_s=index / 12.5,
            action=(0.2, 0.7, 0.0),
            position=(float(index), 0.0),
            angle=0.0,
            velocity=(1.0, 0.0),
            progress=index / 2,
            damage=0.1 if index else 0.0,
            collision=index == 1,
            terminated=index == 2,
            truncated=False,
        )
        for index in range(3)
    )
    return RunLog(
        schema_version=2,
        run={"control_hz": 12.5, "policy": {"name": "agent"}},
        track=TrackSnapshot(
            points=((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 2.0, 0.0)),
            width=6.67,
        ),
        steps=steps,
        summary={"progress": 1.0, "damage": 0.1, "collision_count": 1},
        frames=tuple(_frame(index * 30) for index in range(3)),
    )


def test_diagnostic_report_embeds_replay_and_diagnostic_controls(tmp_path: Path):
    output = write_report(_run_log(), tmp_path / "report.html")
    content = output.read_text(encoding="utf-8")
    assert "HAIC agent diagnostic replay" in content
    assert "run-data" in content
    assert "Trajectory and current action" in content
    assert "steering saturation" in content
    assert "data:image/jpeg;base64" in content


def test_diagnostic_mp4_is_decodable_without_rerunning_agent(tmp_path: Path):
    output = write_mp4(_run_log(), tmp_path / "replay.mp4")
    assert output.is_file()
    capture = cv2.VideoCapture(str(output))
    try:
        decoded = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    assert decoded == 3
