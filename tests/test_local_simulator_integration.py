import unittest

from local_simulator.environment import build_map_bundle
from local_simulator.policies import BaselinePolicy
from local_simulator.schema import CustomObstacle, MapSpec
from local_simulator.simulation import run_episode


class TestLocalSimulatorIntegration(unittest.TestCase):
    def test_map_run_log_contains_replay_data(self):
        spec = MapSpec(
            1,
            42,
            "official_plus_custom",
            (CustomObstacle(0.5, 0.0, 1.2),),
            2,
            4,
        )
        map_bundle = build_map_bundle(spec)
        run_log = run_episode(spec, BaselinePolicy(), record_frames=False)

        self.assertGreater(len(map_bundle.track.points), 0)
        self.assertEqual(len(run_log.steps), 2)
        self.assertIn("lap_time_ms", run_log.summary)
        self.assertEqual(run_log.run["map"]["obstacle_mode"], "official_plus_custom")
        self.assertEqual(run_log.steps[0].action, (0.0, 1.0, 0.0))

    def test_custom_map_can_be_reset_and_run(self):
        from local_simulator.track_generator import generate_custom_map

        document = generate_custom_map("custom-track-0004", 31, "oval")
        run_log = run_episode(document, BaselinePolicy(), record_frames=False)

        self.assertEqual(run_log.run["map"]["map_id"], "custom-track-0004")
        self.assertGreater(len(run_log.track.points), 0)


if __name__ == "__main__":
    unittest.main()
