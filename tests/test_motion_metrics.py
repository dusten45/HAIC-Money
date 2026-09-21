import unittest


class TestMotionMetrics(unittest.TestCase):
    def test_reports_speed_and_signed_acceleration_statistics(self):
        from training.evaluate_closed_loop import summarize_motion_metrics

        metrics = summarize_motion_metrics(
            speeds=[10.0, 20.0, 30.0, 40.0],
            accelerations=[-10.0, 0.0, 10.0, 30.0],
        )

        self.assertEqual(metrics["mean_speed"], 25.0)
        self.assertEqual(metrics["max_speed"], 40.0)
        self.assertEqual(metrics["mean_abs_acceleration"], 12.5)
        self.assertEqual(metrics["peak_acceleration"], 30.0)
        self.assertEqual(metrics["peak_deceleration"], -10.0)
        self.assertAlmostEqual(metrics["p95_abs_acceleration"], 27.0)

    def test_separates_collision_speed_drops_from_braking_metrics(self):
        from training.evaluate_closed_loop import summarize_motion_metrics

        metrics = summarize_motion_metrics(
            speeds=[40.0, 38.0, 5.0, 4.0],
            accelerations=[-25.0, -412.5, -12.5],
            non_collision_accelerations=[-25.0, -12.5],
            collision_accelerations=[-412.5],
        )

        self.assertEqual(metrics["peak_deceleration"], -412.5)
        self.assertEqual(metrics["non_collision_peak_deceleration"], -25.0)
        self.assertEqual(metrics["collision_peak_deceleration"], -412.5)

    def test_empty_motion_trace_reports_unavailable_values(self):
        from training.evaluate_closed_loop import summarize_motion_metrics

        metrics = summarize_motion_metrics(speeds=[], accelerations=[])

        self.assertTrue(all(value is None for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
