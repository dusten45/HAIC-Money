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


if __name__ == "__main__":
    unittest.main()
