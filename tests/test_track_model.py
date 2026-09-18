import unittest


class TestTrackModel(unittest.TestCase):
    def test_custom_geometry_rejects_non_finite_values(self):
        from local_simulator.track_model import CustomTrackGeometry

        with self.assertRaises(ValueError):
            CustomTrackGeometry(
                centerline=((0.0, float("nan")), (1.0, 1.0)),
                width=8.0,
            )

    def test_custom_map_rejects_non_custom_identifier(self):
        from local_simulator.track_model import CustomMapSpec, CustomTrackGeometry

        with self.assertRaises(ValueError):
            CustomMapSpec(
                map_id="track-1",
                geometry=CustomTrackGeometry(
                    centerline=((0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (0.0, 12.0)),
                    width=8.0,
                ),
                obstacles=(),
                max_steps=2000,
                frame_skip=4,
            )


if __name__ == "__main__":
    unittest.main()
