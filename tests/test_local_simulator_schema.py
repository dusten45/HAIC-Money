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


if __name__ == "__main__":
    unittest.main()
