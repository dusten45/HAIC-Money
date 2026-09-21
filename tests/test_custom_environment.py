import unittest

import numpy as np


class TestCustomEnvironment(unittest.TestCase):
    def test_custom_map_start_and_direction_match_preview_and_simulation(self):
        from dataclasses import replace

        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.preview import custom_track_snapshot
        from local_simulator.track_generator import generate_custom_map

        generated = generate_custom_map("custom-track-oriented", 29, "s_curve")
        geometry = replace(generated.geometry, start_index=3, direction=-1)
        document = replace(generated, geometry=geometry)
        count = len(geometry.centerline)
        expected_centers = tuple(
            geometry.centerline[(geometry.start_index + geometry.direction * offset) % count]
            for offset in range(count)
        )
        preview = custom_track_snapshot(document)
        environment, raw = create_environment(document, render_mode=None)
        try:
            reset_environment(environment, document)
            preview_centers = tuple((point[2], point[3]) for point in preview.points)
            simulation_centers = tuple((point[2], point[3]) for point in raw.track)
            self.assertEqual(preview_centers, expected_centers)
            self.assertEqual(simulation_centers, expected_centers)
        finally:
            environment.close()

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

    def test_max_supported_generated_width_stays_within_the_playfield(self):
        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        for template in sorted(TEMPLATES):
            document = generate_custom_map(
                f"custom-track-wide-boundary-{template}",
                73,
                template,
                width=9.0,
            )
            environment, _raw = create_environment(document, render_mode=None)
            try:
                reset_environment(environment, document)
                _observation, _reward, terminated, truncated, _info = environment.step(
                    np.array([0.0, 1.0, 0.0], dtype=np.float32)
                )
                self.assertFalse(terminated or truncated)
            finally:
                environment.close()

    def test_extreme_generation_width_is_rejected_instead_of_creating_an_undrivable_map(self):
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        for template in sorted(TEMPLATES):
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "width"):
                generate_custom_map(
                    f"custom-track-too-wide-{template}",
                    73,
                    template,
                    width=100.0,
                )

    def test_extreme_corner_count_variants_reset_and_step(self):
        from local_simulator.environment import create_environment, reset_environment
        from local_simulator.track_generator import generate_custom_map

        observed_counts = set()
        for seed in (0, 42, 73):
            document = generate_custom_map(f"custom-track-env-{seed}", seed, "extreme_technical")
            observed_counts.add(dict(document.generator)["corner_count"])
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

        self.assertEqual(observed_counts, {12, 14, 16})

    def test_map_payload_can_be_inspected_but_unsafe_geometry_cannot_run(self):
        from local_simulator.environment import create_environment
        from local_simulator.schema import map_from_dict, map_to_dict
        from local_simulator.track_generator import generate_custom_map

        payload = map_to_dict(generate_custom_map("custom-track-inspection", 73, "oval"))
        payload["geometry"]["centerline"] = [
            [-30, -30], [-15, -30], [0, -30], [0, 0], [3, 0], [3, 3],
            [30, 3], [30, 15], [30, 30], [0, 30], [-30, 30], [-30, 0],
        ]
        document = map_from_dict(payload)
        with self.assertRaisesRegex(ValueError, "turn radius"):
            create_environment(document, render_mode=None)


if __name__ == "__main__":
    unittest.main()
