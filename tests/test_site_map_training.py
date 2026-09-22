from dataclasses import replace
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_MAPS = REPO_ROOT / "training" / "maps" / "site"


class TestSiteMapLoader(unittest.TestCase):
    def test_loads_track_lab_custom_map_and_preserves_five_obstacles(self):
        from training.site_maps import load_site_map

        spec = load_site_map(SITE_MAPS / "custom-track-haic-obstacles-20260920.json")

        self.assertEqual(spec.map_id, "custom-track-haic-obstacles-20260920")
        self.assertEqual(spec.map_kind, "custom")
        self.assertEqual(spec.generator_template, "technical")
        self.assertEqual(spec.design_seed, 20260920)
        self.assertEqual(spec.geometry.width, 8.0)
        self.assertEqual(len(spec.geometry.centerline), 246)
        self.assertEqual(len(spec.obstacles), 5)
        self.assertTrue(all(obstacle.radius == 1.2 for obstacle in spec.obstacles))

    def test_rejects_custom_map_schema_v1(self):
        from training.site_maps import load_site_map_payload

        with self.assertRaisesRegex(ValueError, "custom maps require schema_version 2"):
            load_site_map_payload(
                {
                    "schema_version": 1,
                    "map_kind": "custom",
                    "map_id": "custom-track-invalid",
                    "geometry": {"centerline": [[0, 0], [1, 0], [1, 1]], "width": 8},
                }
            )

    def test_accepts_official_site_map_schema_v1_and_v2(self):
        from training.site_maps import load_site_map_payload

        for schema_version in (1, 2):
            with self.subTest(schema_version=schema_version):
                spec = load_site_map_payload(
                    {
                        "schema_version": schema_version,
                        "map_kind": "official",
                        "map_id": f"official-track-3-seed-91-v{schema_version}",
                        "track_id": 3,
                        "seed": 91,
                        "obstacle_mode": "official_plus_custom",
                        "obstacles": [{"progress": 0.5, "lateral": 0.1, "radius": 1.0}],
                    }
                )
                self.assertEqual(spec.track_id, 3)
                self.assertEqual(spec.obstacle_mode, "official_plus_custom")
                self.assertEqual(len(spec.obstacles), 1)


class TestSiteMapSplit(unittest.TestCase):
    def test_manifest_expands_seeds_and_keeps_map_designs_disjoint(self):
        from training.site_maps import load_site_map_split

        split = load_site_map_split(SITE_MAPS / "site_map_split.json")

        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.tune), 2)
        self.assertEqual(len(split.held_out), 5)
        train_ids = {episode.site_map.map_id for episode in split.train}
        tune_ids = {episode.site_map.map_id for episode in split.tune}
        held_out_ids = {episode.site_map.map_id for episode in split.held_out}
        self.assertEqual(
            train_ids,
            {
                "custom-track-haic-obstacles-20260920",
                "custom-track-haic-train-20260921",
            },
        )
        self.assertTrue(train_ids.isdisjoint(tune_ids))
        self.assertTrue(train_ids.isdisjoint(held_out_ids))
        self.assertTrue(tune_ids.isdisjoint(held_out_ids))

    def test_rejects_a_site_map_id_reused_across_splits(self):
        from training.site_maps import load_site_map_split_payload

        map_name = "custom-track-haic-obstacles-20260920.json"
        with self.assertRaisesRegex(ValueError, "train and tune map IDs overlap"):
            load_site_map_split_payload(
                {
                    "schema_version": 1,
                    "train": [{"map": map_name, "seeds": [1]}],
                    "tune": [{"map": map_name, "seeds": [2]}],
                    "held_out": [
                        {"map": "custom-track-haic-heldout-20260923.json", "seeds": [3]}
                    ],
                },
                base_directory=SITE_MAPS,
            )

    def test_rejects_manifest_path_escaping_its_map_directory(self):
        from training.site_maps import load_site_map_split_payload

        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "must stay inside the manifest directory"):
                load_site_map_split_payload(
                    {
                        "schema_version": 1,
                        "train": [{"map": "../outside.json", "seeds": [1]}],
                        "tune": [{"map": "tune.json", "seeds": [2]}],
                        "held_out": [{"map": "heldout.json", "seeds": [3]}],
                    },
                    base_directory=Path(temporary_directory),
                )


class TestSiteMapEnvironment(unittest.TestCase):
    def test_custom_site_map_builds_track_and_five_physical_obstacles(self):
        from training.env_factory import create_training_environment
        from training.site_maps import load_site_map

        spec = load_site_map(SITE_MAPS / "custom-track-haic-obstacles-20260920.json")
        environment = create_training_environment(
            track_id=None,
            seed=20260924,
            max_decisions=3,
            site_map=spec,
        )
        try:
            observation, info = environment.reset()

            self.assertEqual(observation.shape, (4, 84, 84))
            self.assertEqual(observation.dtype.name, "float32")
            self.assertEqual(len(environment.unwrapped.track), 246)
            self.assertEqual(len(environment.unwrapped.obstacles), 5)
            self.assertAlmostEqual(environment.unwrapped.track[0][2], spec.geometry.centerline[0][0])
            self.assertAlmostEqual(environment.unwrapped.track[0][3], spec.geometry.centerline[0][1])
            self.assertEqual(info["site_map_id"], spec.map_id)
            self.assertEqual(info["site_map_kind"], "custom")
            self.assertEqual(info["site_obstacle_count"], 5)

            environment.reset()
            self.assertEqual(len(environment.unwrapped.obstacles), 5)
        finally:
            environment.close()

    def test_closed_loop_episode_records_site_map_identity_and_obstacles(self):
        from training.evaluate_closed_loop import run_episode
        from training.site_maps import load_site_map_split

        class _Agent:
            def reset(self, observation):
                self.observation = observation

            def act(self, observation):
                return [0.0, 0.6, 0.0]

        episode = load_site_map_split(SITE_MAPS / "site_map_split.json").train[0]
        result = run_episode(
            mode="ppo_only",
            episode=episode,
            agent=_Agent(),
            max_decisions=1,
            plan_budget_seconds=0.1,
        )

        self.assertEqual(result["map_id"], episode.map_id)
        self.assertEqual(result["map_kind"], "custom")
        self.assertEqual(result["obstacle_mode"], "custom_only")
        self.assertEqual(result["site_obstacle_count"], 5)
        self.assertEqual(result["obstacle_count"], 5)

    def test_custom_site_map_start_index_and_direction_control_track_order(self):
        from training.env_factory import create_training_environment
        from training.site_maps import load_site_map

        spec = load_site_map(SITE_MAPS / "custom-track-haic-obstacles-20260920.json")
        start_index = 41
        geometry = replace(spec.geometry, start_index=start_index, direction=-1)
        spec = replace(spec, geometry=geometry)
        environment = create_training_environment(
            track_id=None,
            seed=20260924,
            max_decisions=2,
            site_map=spec,
        )
        try:
            environment.reset()
            track = environment.unwrapped.track
            centerline = geometry.centerline
            self.assertAlmostEqual(track[0][2], centerline[start_index][0])
            self.assertAlmostEqual(track[0][3], centerline[start_index][1])
            self.assertAlmostEqual(track[1][2], centerline[start_index - 1][0])
            self.assertAlmostEqual(track[1][3], centerline[start_index - 1][1])
        finally:
            environment.close()

    def test_site_obstacle_contact_reaches_existing_damage_system(self):
        import numpy as np

        from training.env_factory import create_training_environment
        from training.site_maps import load_site_map

        spec = load_site_map(SITE_MAPS / "custom-track-haic-obstacles-20260920.json")
        environment = create_training_environment(
            track_id=None, seed=20260924, max_decisions=3, site_map=spec
        )
        try:
            environment.reset()
            raw = environment.unwrapped
            obstacle = raw.obstacles[0]
            target = (float(obstacle.position.x), float(obstacle.position.y))
            for body in (raw.car.hull, *raw.car.wheels):
                body.position = target
                body.linearVelocity = (0.0, 0.0)
                body.angularVelocity = 0.0

            _, _, _, _, info = environment.step(np.zeros(3, dtype=np.float32))

            self.assertTrue(info["collision"])
            self.assertAlmostEqual(info["damage"], 0.2)
        finally:
            environment.close()

    def test_site_map_episode_yields_normal_training_transition_and_metadata(self):
        import numpy as np

        from training.env_factory import create_episode_environment
        from training.site_maps import load_site_map_split

        episode = load_site_map_split(SITE_MAPS / "site_map_split.json").train[0]
        environment = create_episode_environment(episode, max_decisions=2)
        try:
            observation, info = environment.reset()
            transition = environment.step_transition(np.array([0.0, 0.6, 0.0], dtype=np.float32))

            self.assertEqual(observation.shape, (4, 84, 84))
            self.assertEqual(transition.observation.shape, (4, 84, 84))
            self.assertEqual(transition.next_observation.shape, (4, 84, 84))
            self.assertEqual(transition.label_timing, "next_decision")
            self.assertEqual(info["site_map_id"], episode.map_id)
            self.assertEqual(info["site_obstacle_count"], 5)
            self.assertEqual(transition.labels.road_half_width, episode.site_map.geometry.width)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
