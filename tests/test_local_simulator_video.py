import base64
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from local_simulator.environment import TrackSnapshot
from local_simulator.simulation_types import RunLog, RunStep
from local_simulator.video import export_run_video


def _encoded_frame(color: tuple[int, int, int]) -> str:
    image = np.full((48, 64, 3), color, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise AssertionError("test JPEG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _run_log(*, frames: tuple[str, ...] | None) -> RunLog:
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
        run={"control_hz": 12.5, "run_id": "test"},
        track=TrackSnapshot(points=((0.0, 0.0, 0.0, 0.0),), width=6.67),
        steps=steps,
        summary={},
        frames=frames,
    )


class TestLocalSimulatorVideo(unittest.TestCase):
    def test_export_requires_recorded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no embedded frames"):
                export_run_video(_run_log(frames=None), Path(directory) / "run.mp4")

    def test_export_rejects_mismatched_frame_count(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "frame count"):
                export_run_video(
                    _run_log(frames=(_encoded_frame((0, 0, 0)),)),
                    Path(directory) / "run.mp4",
                )

    def test_export_writes_decodable_mp4_with_one_frame_per_step(self):
        frames = tuple(_encoded_frame((index * 30, 20, 10)) for index in range(3))
        with tempfile.TemporaryDirectory() as directory:
            output = export_run_video(_run_log(frames=frames), Path(directory) / "run.mp4")
            self.assertTrue(output.is_file())
            capture = cv2.VideoCapture(str(output))
            try:
                decoded = 0
                while True:
                    success, _frame = capture.read()
                    if not success:
                        break
                    decoded += 1
            finally:
                capture.release()
            self.assertEqual(decoded, 3)


if __name__ == "__main__":
    unittest.main()
