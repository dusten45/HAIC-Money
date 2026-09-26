from __future__ import annotations

import unittest

from core.finish_line import FinishLineTracker
from scripts.probe_drq_training_finish import forward_crossing_probe


class TrainingRoadFinishProbeTests(unittest.TestCase):
    def test_valid_centerline_path_reaches_forward_finish_without_mutating_source(self) -> None:
        tracker = FinishLineTracker(
            center=(2.0, -3.0), forward=(0.6, 0.8), half_width=40.0 / 6.0,
            half_depth=0.875,
        )
        result = forward_crossing_probe(tracker)
        self.assertTrue(result["qualified_forward_centerline_crossing_accepted"])
        self.assertFalse(result["physical_car_reachability_proven"])
        self.assertTrue(result["original_tracker_unmodified"])
        self.assertIsNone(tracker.finish_time_s)

    def test_zero_width_or_existing_finish_is_rejected(self) -> None:
        tracker = FinishLineTracker(center=(0.0, 0.0), forward=(1.0, 0.0),
                                    half_width=0.0, half_depth=0.875)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            forward_crossing_probe(tracker)
        tracker.half_width = 40.0 / 6.0
        tracker.finish_time_s = 1.0
        with self.assertRaisesRegex(ValueError, "reset state"):
            forward_crossing_probe(tracker)


if __name__ == "__main__":
    unittest.main()
