import unittest
from types import SimpleNamespace

from local_simulator.schema import CustomObstacle, MapSpec
from local_simulator.track_model import CustomMapSpec, CustomTrackGeometry


class TestEnvironmentSetup(unittest.TestCase):
    def _environment_api(self):
        try:
            from local_simulator.environment import (
                attach_custom_obstacles,
                create_environment,
                map_custom_obstacles,
                snapshot_track,
            )
        except ImportError as error:
            self.fail(f"environment API is missing: {error}")
        return (
            attach_custom_obstacles,
            create_environment,
            map_custom_obstacles,
            snapshot_track,
        )

    def test_same_map_has_same_track_snapshot(self):
        _, create_environment, _, snapshot_track = self._environment_api()
        spec = MapSpec(1, 42, "official_plus_custom", (), 20, 4)
        first, _ = create_environment(spec, render_mode=None)
        second, _ = create_environment(spec, render_mode=None)
        try:
            first.reset(seed=spec.seed, options={"track_id": spec.track_id})
            second.reset(seed=spec.seed, options={"track_id": spec.track_id})
            self.assertEqual(snapshot_track(first), snapshot_track(second))
        finally:
            first.close()
            second.close()

    def test_official_mode_has_six_obstacles(self):
        _, create_environment, _, _ = self._environment_api()
        spec = MapSpec(1, 42, "official", (), 20, 4)
        environment, _ = create_environment(spec, render_mode=None)
        try:
            environment.reset(seed=spec.seed, options={"track_id": spec.track_id})
            self.assertEqual(len(environment.unwrapped.obstacles), 6)
        finally:
            environment.close()

    def test_custom_obstacle_is_attached_and_counted(self):
        attach_custom_obstacles, create_environment, map_custom_obstacles, _ = (
            self._environment_api()
        )
        spec = MapSpec(
            1,
            42,
            "custom_only",
            (CustomObstacle(0.50, 0.0, 1.2),),
            20,
            4,
        )
        environment, _ = create_environment(spec, render_mode=None)
        try:
            environment.reset(seed=spec.seed)
            custom = map_custom_obstacles(environment, spec)
            attach_custom_obstacles(environment, custom)
            self.assertEqual(len(environment.unwrapped.obstacles), 1)
        finally:
            environment.close()

    def test_custom_obstacle_radius_cannot_exceed_custom_road_width(self):
        _, _, map_custom_obstacles, _ = self._environment_api()
        from local_simulator.track_model import CustomObstacle as TrackObstacle
        from local_simulator.preview import build_map_preview
        from training.site_environment import site_obstacle_specs
        from training.site_maps import load_site_map_payload

        spec = CustomMapSpec(
            map_id="custom-track-narrow",
            geometry=CustomTrackGeometry(
                centerline=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
                width=0.5,
            ),
            obstacles=(TrackObstacle(progress=0.5, lateral=1.0, radius=0.6),),
            max_steps=20,
            frame_skip=4,
        )
        environment = SimpleNamespace(
            unwrapped=SimpleNamespace(track=[(0.0, 0.0, 0.0, 0.0)])
        )

        with self.assertRaisesRegex(ValueError, "radius.*road width"):
            map_custom_obstacles(environment, spec)
        with self.assertRaisesRegex(ValueError, "radius.*road width"):
            build_map_preview(spec)

        site_map = load_site_map_payload(
            {
                "schema_version": 2,
                "map_kind": "custom",
                "map_id": "custom-track-narrow",
                "geometry": {
                    "centerline": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    "width": 0.5,
                },
                "obstacles": [{"progress": 0.5, "lateral": 1.0, "radius": 0.6}],
            }
        )
        track = [(0.0, 0.0, 0.0, 0.0)]
        with self.assertRaisesRegex(ValueError, "radius.*road width"):
            site_obstacle_specs(site_map, track)


if __name__ == "__main__":
    unittest.main()
