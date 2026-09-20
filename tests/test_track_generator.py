import unittest


class TestTrackGenerator(unittest.TestCase):
    def test_same_design_seed_has_same_geometry_and_fingerprint(self):
        from local_simulator.track_generator import (
            custom_geometry_fingerprint,
            generate_custom_map,
        )

        first = generate_custom_map("custom-track-0001", 90421, "hairpin")
        second = generate_custom_map("custom-track-0001", 90421, "hairpin")

        self.assertEqual(first, second)
        self.assertEqual(
            custom_geometry_fingerprint(first.geometry),
            custom_geometry_fingerprint(second.geometry),
        )

    def test_different_design_seed_changes_geometry(self):
        from local_simulator.track_generator import generate_custom_map

        first = generate_custom_map("custom-track-0001", 1, "oval")
        second = generate_custom_map("custom-track-0002", 2, "oval")

        self.assertNotEqual(first.geometry.centerline, second.geometry.centerline)

    def test_self_intersection_is_rejected(self):
        from local_simulator.track_generator import validate_custom_geometry
        from local_simulator.track_model import CustomTrackGeometry

        geometry = CustomTrackGeometry(
            centerline=(
                (0.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (30.0, 10.0),
                (30.0, 0.0),
                (40.0, 0.0),
                (40.0, 10.0),
                (50.0, 10.0),
                (50.0, 0.0),
            ),
            width=8.0,
        )

        with self.assertRaisesRegex(ValueError, "self-intersection"):
            validate_custom_geometry(geometry)

    def test_batch_generation_uses_distinct_ids(self):
        from local_simulator.track_generator import generate_custom_maps

        maps = generate_custom_maps(20, 3, "chicane")

        self.assertEqual(
            [item.map_id for item in maps],
            ["custom-track-0020", "custom-track-0021", "custom-track-0022"],
        )
        self.assertEqual(len({item.geometry.centerline for item in maps}), 3)

    def test_all_templates_generate_valid_maps(self):
        from local_simulator.track_generator import generate_custom_map

        for template in ("oval", "s_curve", "hairpin", "chicane"):
            with self.subTest(template=template):
                result = generate_custom_map(
                    f"custom-track-{template}",
                    90421,
                    template,
                )
                self.assertGreaterEqual(len(result.geometry.centerline), 12)

    def test_generation_v2_records_distinct_corner_profiles(self):
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        self.assertEqual(
            TEMPLATES,
            frozenset({"oval", "s_curve", "hairpin", "chicane", "technical"}),
        )
        profiles = {}
        expected_ranges = {
            "oval": (4, 4),
            "s_curve": (6, 8),
            "hairpin": (6, 9),
            "chicane": (7, 10),
            "technical": (9, 12),
        }
        for template in sorted(TEMPLATES):
            first = generate_custom_map(f"custom-track-{template}", 90421, template)
            second = generate_custom_map(f"custom-track-{template}", 90421, template)
            first_meta = dict(first.generator)
            second_meta = dict(second.generator)
            self.assertEqual(first, second)
            self.assertEqual(first_meta["generator_version"], 2)
            self.assertEqual(first_meta["corner_count"], len(first_meta["corner_sequence"]))
            self.assertGreaterEqual(first_meta["corner_count"], expected_ranges[template][0])
            self.assertLessEqual(first_meta["corner_count"], expected_ranges[template][1])
            sequence = first_meta["corner_sequence"]
            self.assertEqual(len(sequence), first_meta["corner_count"])
            self.assertTrue(all(token.split(":", 1)[0] in {"left", "right"} for token in sequence))
            if template == "s_curve":
                self.assertTrue(
                    any(
                        sequence[index].split(":", 1)[0]
                        != sequence[index + 1].split(":", 1)[0]
                        for index in range(len(sequence) - 1)
                    )
                )
            elif template == "hairpin":
                self.assertGreaterEqual(sum(token.endswith(":hairpin") for token in sequence), 2)
            elif template == "chicane":
                switches = sum(
                    sequence[index].split(":", 1)[0]
                    != sequence[index + 1].split(":", 1)[0]
                    for index in range(len(sequence) - 1)
                )
                self.assertGreaterEqual(switches, 4)
            elif template == "technical":
                self.assertGreaterEqual(len({token.split(":", 1)[1] for token in sequence}), 3)
            profiles[template] = tuple(first_meta["corner_sequence"])
        self.assertNotEqual(profiles["oval"], profiles["technical"])

    def test_boundary_seeds_are_repeatable_and_each_template_varies(self):
        from local_simulator.track_generator import generate_custom_map

        for seed in (0, 4294967295):
            first = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
            second = generate_custom_map(f"custom-track-boundary-{seed}", seed, "technical")
            self.assertEqual(first, second)

        for template in ("oval", "s_curve", "hairpin", "chicane", "technical"):
            geometries = {
                generate_custom_map(f"custom-track-{template}-{seed}", seed, template).geometry.centerline
                for seed in range(16)
            }
            with self.subTest(template=template):
                self.assertGreaterEqual(len(geometries), 8)

    def test_sampled_centerline_respects_segment_limit(self):
        import math
        from local_simulator.track_generator import TEMPLATES, generate_custom_map

        for template in sorted(TEMPLATES):
            geometry = generate_custom_map(f"custom-track-spacing-{template}", 42, template).geometry
            for index, point in enumerate(geometry.centerline):
                following = geometry.centerline[(index + 1) % len(geometry.centerline)]
                self.assertLessEqual(math.dist(point, following), 4.00001)

    def test_invalid_design_seeds_are_rejected(self):
        from local_simulator.track_generator import generate_custom_map

        for seed in (True, -1, 4294967296, 1.5):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "design_seed"):
                    generate_custom_map("custom-track-invalid-seed", seed, "oval")

    def test_unknown_template_is_rejected(self):
        from local_simulator.track_generator import generate_custom_map

        with self.assertRaisesRegex(ValueError, "template"):
            generate_custom_map("custom-track-invalid", 1, "unknown")


if __name__ == "__main__":
    unittest.main()
