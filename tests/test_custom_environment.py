import unittest

import numpy as np


class TestCustomEnvironment(unittest.TestCase):
    def test_custom_environment_has_track_and_expected_observation(self):
        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.track_generator import generate_custom_map

        document = generate_custom_map("custom-track-0001", 90421, "hairpin")
        environment, raw = create_environment(document, render_mode=None)
        try:
            observation, _ = reset_environment(environment, document)
            self.assertEqual(observation.shape, (4, 84, 84))
            self.assertEqual(len(raw.track), len(document.geometry.centerline))
            next_observation, *_ = environment.step(
                np.array([0.0, 1.0, 0.0], dtype=np.float32)
            )
            self.assertEqual(next_observation.shape, (4, 84, 84))
        finally:
            environment.close()

    def test_custom_map_snapshot_is_deterministic(self):
        from local_simulator.environment import (
            create_environment,
            reset_environment,
            snapshot_track,
        )
        from local_simulator.track_generator import generate_custom_map

        document = generate_custom_map("custom-track-0002", 17, "s_curve")
        first, _ = create_environment(document, render_mode=None)
        second, _ = create_environment(document, render_mode=None)
        try:
            reset_environment(first, document)
            reset_environment(second, document)
            self.assertEqual(snapshot_track(first), snapshot_track(second))
        finally:
            first.close()
            second.close()

    def test_custom_obstacle_is_attached_without_official_obstacles(self):
        from dataclasses import replace

        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.schema import CustomObstacle
        from local_simulator.track_generator import generate_custom_map

        generated = generate_custom_map("custom-track-0003", 21, "chicane")
        document = replace(
            generated,
            obstacles=(CustomObstacle(0.5, 0.0, 1.2),),
        )
        environment, raw = create_environment(document, render_mode=None)
        try:
            reset_environment(environment, document)
            self.assertEqual(len(raw.obstacles), 1)
        finally:
            environment.close()

    def test_every_generated_template_runs_in_custom_environment(self):
        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        for template in sorted(TEMPLATES):
            document = generate_custom_map(f"custom-track-env-{template}", 73, template)
            environment, _raw = create_environment(document, render_mode=None)
            try:
                observation, _info = reset_environment(environment, document)
                self.assertEqual(observation.shape, (4, 84, 84))
                next_observation, *_ = environment.step(
                    np.array([0.0, 1.0, 0.0], dtype=np.float32)
                )
                self.assertEqual(next_observation.shape, (4, 84, 84))
            finally:
                environment.close()


if __name__ == "__main__":
    unittest.main()
