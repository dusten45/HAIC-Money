import unittest


class TestMapSchema(unittest.TestCase):
    def _schema_api(self):
        try:
            from local_simulator.schema import (
                CustomObstacle,
                MapSpec,
                map_from_dict,
                map_to_dict,
                validate_map_payload,
            )
        except ImportError as error:
            self.fail(f"map schema API is missing: {error}")
        return (
            CustomObstacle,
            MapSpec,
            map_from_dict,
            map_to_dict,
            validate_map_payload,
        )

    def test_round_trip_preserves_map(self):
        CustomObstacle, MapSpec, map_from_dict, map_to_dict, _ = self._schema_api()
        spec = MapSpec(
            track_id=2,
            seed=123,
            obstacle_mode="official_plus_custom",
            obstacles=(CustomObstacle(0.42, -0.25, 1.2),),
            max_steps=2000,
            frame_skip=4,
        )

        self.assertEqual(map_from_dict(map_to_dict(spec)), spec)

    def test_rejects_invalid_seed_and_obstacle(self):
        _, _, _, _, validate_map_payload = self._schema_api()

        with self.assertRaises(ValueError):
            validate_map_payload({"track_id": 1, "seed": -1})

        with self.assertRaises(ValueError):
            validate_map_payload(
                {
                    "track_id": 1,
                    "seed": 1,
                    "obstacle_mode": "official",
                    "obstacles": [
                        {"progress": 1.2, "lateral": 0, "radius": 1.2}
                    ],
                }
            )

    def test_custom_map_round_trip_preserves_geometry(self):
        from local_simulator.schema import map_from_dict, map_to_dict
        from local_simulator.track_model import CustomMapSpec, CustomTrackGeometry

        original = CustomMapSpec(
            map_id="custom-track-0001",
            geometry=CustomTrackGeometry(
                centerline=((0.0, 0.0), (12.0, 0.0), (12.0, 12.0), (0.0, 12.0)),
                width=8.0,
            ),
            obstacles=(),
            max_steps=2000,
            frame_skip=4,
            generator=(("template", "oval"), ("design_seed", 7)),
        )

        self.assertEqual(map_from_dict(map_to_dict(original)), original)

    def test_schema_one_payload_is_read_as_official_map(self):
        result = self._schema_api()[2]({"track_id": 1, "seed": 42})

        self.assertEqual(result.track_id, 1)
        self.assertEqual(result.seed, 42)
        self.assertEqual(result.map_kind, "official")


if __name__ == "__main__":
    unittest.main()
