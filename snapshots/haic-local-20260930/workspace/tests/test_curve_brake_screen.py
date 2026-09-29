import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


class TestCustomTrainTuneLoading(unittest.TestCase):
    def test_split_loader_never_materializes_held_out_map_entries(self):
        from training.site_maps import SiteMapSpec
        from training.curve_brake_screen import load_custom_train_tune_split

        payload = {
            "train": [
                {"map": "train-a.json", "seeds": [11, 12]},
                {"map": "train-b.json", "seeds": [13, 14]},
            ],
            "tune": [{"map": "tune.json", "seeds": [21, 22]}],
            "held_out": [{"map": "must-not-load.json", "seeds": [31]}],
        }
        loaded_names = []

        def load_map(path):
            loaded_names.append(path.name)
            return SiteMapSpec(
                map_id=path.stem,
                map_kind="custom",
                track_id=1,
                seed=1,
                obstacle_mode="custom_only",
                obstacles=(),
                max_steps=100,
                frame_skip=4,
            )

        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "site_map_split.json"
            split_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("training.curve_brake_screen.load_site_map", side_effect=load_map):
                split = load_custom_train_tune_split(split_path)

        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.tune), 2)
        self.assertEqual(split.held_out, ())
        self.assertEqual(
            loaded_names,
            ["train-a.json", "train-b.json", "tune.json"],
        )
        self.assertEqual([episode.seed for episode in split.train], [11, 12, 13, 14])
        self.assertEqual([episode.seed for episode in split.tune], [21, 22])
        self.assertNotIn("must-not-load.json", loaded_names)
        self.assertTrue(
            all(episode.site_map.map_kind == "custom" for episode in (*split.train, *split.tune))
        )


class TestFixedObservationAndDecisionTraces(unittest.TestCase):
    def test_evaluator_captures_exact_pre_action_pixels_and_raw_decision_outcomes(self):
        from training.curve_brake_screen import evaluate_policy_with_traces
        from training.site_maps import SiteMapEpisode, SiteMapSpec

        observations = [np.full((4, 84, 84), index / 10.0, dtype=np.float32) for index in range(3)]
        features = [
            [0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.7, 1.0, 0.0, 0.9],
            [0.5, 0.0, 0.0, 0.2, 1.0, 0.0, 0.9],
        ]
        labels = [
            SimpleNamespace(
                speed=30.0, tile_progress=0.1, collision=False, damage=0.0,
                off_track=False, finished=False,
            ),
            SimpleNamespace(
                speed=25.0, tile_progress=0.2, collision=True, damage=0.2,
                off_track=False, finished=False,
            ),
            SimpleNamespace(
                speed=20.0, tile_progress=1.0, collision=False, damage=0.3,
                off_track=False, finished=True,
            ),
        ]

        class _Environment:
            def __init__(self):
                self.index = 0
                self.t = 0.0
                self.finish_time_s = None

            @property
            def unwrapped(self):
                return self

            def reset(self):
                return observations[0].copy(), {}

            def step_transition(self, action):
                index = self.index
                self.index += 1
                if index == 2:
                    self.finish_time_s = 12.5
                return SimpleNamespace(
                    next_observation=(observations[index + 1].copy() if index < 2 else observations[index].copy()),
                    labels=labels[index],
                    terminated=index == 2,
                    truncated=False,
                    action=action,
                )

            def close(self):
                pass

        class _Model:
            def __init__(self):
                self.index = 0
                self.last_visual_features = None

            def __call__(self, observation):
                self.last_visual_features = torch.tensor([features[self.index]])
                self.index += 1
                return object()

            def deterministic_actions(self, _output):
                return torch.tensor([[0.1, 0.02, 0.14]])

        episode = SiteMapEpisode(
            site_map=SiteMapSpec(
                map_id="custom-trace-test",
                map_kind="custom",
                track_id=1,
                seed=21,
                obstacle_mode="custom_only",
                obstacles=(),
                max_steps=100,
                frame_skip=4,
            ),
            seed=21,
            source_path=Path("custom-trace-test.json"),
        )
        with patch(
            "training.curve_brake_screen.create_episode_environment",
            return_value=_Environment(),
        ):
            result = evaluate_policy_with_traces(
                _Model(), [episode], max_decisions=3, capture_observations=True
            )

        record = result["episodes"][0]
        self.assertTrue(record["completed"])
        self.assertEqual(record["lap_time_ms"], 12500.0)
        self.assertEqual(record["collision_onsets"], 1)
        self.assertEqual(record["final_progress"], 1.0)
        self.assertAlmostEqual(record["final_damage"], 0.3)
        self.assertEqual(record["high_curvature_actions"]["decisions"], 2)
        self.assertEqual(record["pre_visible_hazard_braking"]["visible_onsets"], 1)
        self.assertAlmostEqual(
            record["pre_visible_hazard_braking"]["immediate_pre_onset"]["mean_brake"],
            0.14,
        )
        self.assertEqual(len(record["decision_trace"]), 3)
        self.assertEqual(record["decision_trace"][1]["collision_onset"], True)
        self.assertAlmostEqual(record["decision_trace"][1]["road_curve_magnitude"], 0.7)
        self.assertEqual(result["observations"].dtype, np.float32)
        np.testing.assert_array_equal(result["observations"], np.stack(observations))
        self.assertEqual(len(result["observation_metadata"]), 3)

    def test_actor_replay_uses_same_images_and_reports_feature_group_counts(self):
        from training.curve_brake_screen import replay_actor_on_observation_bank

        observations = np.stack(
            [np.full((4, 84, 84), index / 10.0, dtype=np.float32) for index in range(4)]
        )
        metadata = [
            {"road_curve_magnitude": 0.6, "obstacle_present": 1.0, "obstacle_urgency": 0.9},
            {"road_curve_magnitude": 0.7, "obstacle_present": 1.0, "obstacle_urgency": 0.3},
            {"road_curve_magnitude": 0.2, "obstacle_present": 0.0, "obstacle_urgency": 0.0},
            {"road_curve_magnitude": 0.8, "obstacle_present": 1.0, "obstacle_urgency": 0.95},
        ]

        class _Model:
            def __call__(self, batch):
                self.last_visual_features = torch.tensor(
                    [
                        [0.0, 0.0, 0.0, row["road_curve_magnitude"], row["obstacle_present"], 0.0, row["obstacle_urgency"]]
                        for row in metadata
                    ]
                )
                self.last_batch = batch.detach().cpu().numpy().copy()
                return object()

            def deterministic_actions(self, _output):
                return torch.tensor([[0.0, 0.01 + index / 100.0, index / 10.0] for index in range(4)])

        model = _Model()
        result = replay_actor_on_observation_bank(model, observations, metadata)

        np.testing.assert_array_equal(model.last_batch, observations)
        self.assertEqual(result["groups"]["urgent_visible"]["decisions"], 2)
        self.assertEqual(result["groups"]["high_curve"]["decisions"], 3)
        self.assertEqual(result["groups"]["intersection"]["decisions"], 2)
        self.assertIsNone(result["groups"]["urgent_visible"]["mean_speed"])
        self.assertAlmostEqual(result["groups"]["urgent_visible"]["mean_gas"], 0.025)
        self.assertAlmostEqual(result["groups"]["urgent_visible"]["mean_brake"], 0.15)
        self.assertAlmostEqual(result["rows"][3]["brake"], 0.3)


if __name__ == "__main__":
    unittest.main()
