import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch


class TestHazardExposureRecorder(unittest.TestCase):
    def test_records_map_progress_episode_and_update_aggregates_without_step_traces(self):
        from training.hazard_exposure import HazardExposureRecorder

        recorder = HazardExposureRecorder()
        episode = {
            "map_id": "custom-track-train-a",
            "map_kind": "custom",
            "seed": 11,
            "source_map": "train-a.json",
        }
        recorder.begin_episode(update_number=3, episode_number=1, episode=episode)
        recorder.record_decision(
            progress=0.62,
            damage_delta=0.25,
            features=np.asarray([0.5, 0.0, 0.0, 0.0, 1.0, -0.2, 0.8], dtype=np.float32),
            risk=0.512,
            action=np.asarray([0.4, 0.2, 0.4], dtype=np.float32),
            next_collision=True,
            next_progress=0.63,
            pre_speed=10.0,
            next_speed=9.0,
            next_damage=0.5,
        )
        recorder.record_decision(
            progress=0.76,
            damage_delta=0.0,
            features=np.asarray([0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.9], dtype=np.float32),
            risk=0.0,
            action=np.asarray([-0.5, 0.0, 0.0], dtype=np.float32),
            next_collision=False,
            next_progress=0.77,
            pre_speed=9.0,
            next_speed=9.0,
            next_damage=0.5,
        )
        recorder.end_episode(
            boundary="rollout_cutoff",
            end_progress=0.77,
            terminated=False,
            truncated=True,
            finished=False,
            off_track=False,
        )

        payload = recorder.to_dict()
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(len(payload["episodes"]), 1)
        recorded = payload["episodes"][0]
        self.assertEqual(recorded["update_number"], 3)
        self.assertEqual(recorded["episode_number"], 1)
        self.assertEqual(recorded["boundary"], "rollout_cutoff")
        self.assertEqual(recorded["decision_count"], 2)
        self.assertEqual(len(recorded["bins"]), 20)
        by_index = {item["bin_index"]: item for item in recorded["bins"]}
        self.assertEqual(by_index[12]["all_decisions"], 1)
        self.assertEqual(by_index[12]["risk_positive_count"], 1)
        self.assertAlmostEqual(by_index[12]["risk_positive_mean"], 0.512)
        self.assertEqual(by_index[12]["next_collision_count"], 1)
        urgent_band = by_index[12]["urgency_bands"][3]
        self.assertEqual(urgent_band["obstacle_observations"], 1)
        self.assertEqual(urgent_band["collision_next_count"], 1)
        self.assertAlmostEqual(urgent_band["signed_obstacle_offset_mean"], -0.2)
        self.assertAlmostEqual(urgent_band["steer_away_alignment_mean"], 0.4)
        self.assertEqual(urgent_band["steer_away_aligned_count"], 1)
        self.assertAlmostEqual(by_index[12]["damage_delta_sum"], 0.25)
        self.assertEqual(by_index[15]["all_decisions"], 1)
        self.assertEqual(by_index[15]["urgency_bands"][3]["steer_away_aligned_count"], 1)
        self.assertEqual(by_index[15]["brake_active_count"], 0)
        self.assertAlmostEqual(by_index[15]["risk_sum"], 0.0)
        self.assertEqual(recorded["collision_followups"][0]["origin_bin_index"], 12)
        self.assertEqual(recorded["collision_followups"][0]["next_bin_index"], 15)
        self.assertEqual(recorded["collision_followups"][0]["count"], 1)
        self.assertAlmostEqual(
            recorded["collision_followups"][0]["followup_progress_change_mean"], 0.13
        )
        self.assertEqual(by_index[11]["all_decisions"], 0)
        self.assertEqual(by_index[11]["risk_mean"], 0.0)
        self.assertEqual(recorded["coverage"]["reached_bin_count"], 2)
        self.assertTrue(recorded["coverage"]["target_bins"]["0.60-0.65"]["reached"])
        self.assertTrue(recorded["coverage"]["target_bins"]["0.75-0.80"]["reached"])
        def all_keys(value):
            if isinstance(value, dict):
                return set(value).union(*(all_keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(all_keys(item) for item in value))
            return set()

        keys = all_keys(payload)
        self.assertNotIn("observation", keys)
        self.assertNotIn("pixels", keys)
        self.assertNotIn("step_trace", keys)

    def test_marks_missing_perception_as_unavailable_instead_of_clear_road(self):
        from training.hazard_exposure import HazardExposureRecorder

        recorder = HazardExposureRecorder()
        recorder.begin_episode(
            update_number=1,
            episode_number=0,
            episode={"map_id": "m", "map_kind": "custom", "seed": 7},
        )
        recorder.record_decision(
            progress=0.1,
            damage_delta=0.0,
            features=None,
            risk=None,
            action=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            next_collision=False,
            next_progress=0.1,
            pre_speed=0.0,
            next_speed=0.0,
            next_damage=0.0,
        )
        recorder.end_episode(
            boundary="max_decisions",
            end_progress=0.1,
            terminated=False,
            truncated=True,
            finished=False,
            off_track=False,
        )

        bin_data = recorder.to_dict()["episodes"][0]["bins"][2]
        self.assertEqual(bin_data["all_decisions"], 1)
        self.assertEqual(bin_data["feature_observations"], 0)
        self.assertEqual(bin_data["perception_unavailable_count"], 1)
        self.assertIsNone(bin_data["risk_mean"])

    def test_steer_alignment_uses_car_relative_offset_and_excludes_neutral_positions(self):
        from training.hazard_exposure import HazardExposureRecorder

        recorder = HazardExposureRecorder()
        recorder.begin_episode(
            update_number=1,
            episode_number=0,
            episode={"map_id": "train-map", "map_kind": "custom", "seed": 9},
        )
        cases = (
            # Same obstacle and steer: with the car centered, it is left of us.
            (0.0, 0.6, 0.288),
            # Moving the car right of the road center puts that same obstacle right of us.
            (0.2, 0.8, 0.32),
            # Offset cancels the car position: do not score a left/right alignment.
            (2.4 / 42.0, 0.8, 0.64),
        )
        for index, (road_offset, urgency, risk) in enumerate(cases):
            recorder.record_decision(
                progress=0.1 + index * 0.001,
                damage_delta=0.0,
                features=np.asarray(
                    [0.5, road_offset, 0.0, 0.0, 1.0, -0.2, urgency],
                    dtype=np.float32,
                ),
                risk=risk,
                action=np.asarray([0.4, 0.0, 0.0], dtype=np.float32),
                next_collision=False,
                next_progress=0.1 + index * 0.001,
                pre_speed=10.0,
                next_speed=10.0,
                next_damage=0.0,
            )
        recorder.end_episode(
            boundary="max_decisions",
            end_progress=0.1,
            terminated=False,
            truncated=True,
            finished=False,
            off_track=False,
        )

        progress_bin = recorder.to_dict()["episodes"][0]["bins"][2]
        centered = progress_bin["urgency_bands"][2]
        offset_car = progress_bin["urgency_bands"][3]
        self.assertAlmostEqual(centered["car_relative_obstacle_offset_px_mean"], -2.4)
        self.assertEqual(centered["steer_away_aligned_count"], 1)
        self.assertAlmostEqual(centered["steer_away_alignment_mean"], 0.4)
        self.assertAlmostEqual(offset_car["car_relative_obstacle_offset_px_mean"], 3.0)
        self.assertEqual(offset_car["steer_away_misaligned_count"], 1)
        self.assertEqual(offset_car["relative_offset_neutral_count"], 1)
        self.assertEqual(offset_car["steer_away_alignment_observations"], 1)
        self.assertAlmostEqual(offset_car["steer_away_alignment_mean"], -0.4)


class TestTrainOnlySiteMapLoading(unittest.TestCase):
    def test_loads_train_group_maps_without_opening_tune_or_held_out_maps(self):
        from training.site_maps import load_train_site_map_episodes

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "train.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "map_kind": "custom",
                        "map_id": "custom-track-train",
                        "max_steps": 20,
                        "frame_skip": 4,
                        "geometry": {
                            "centerline": [[0, 0], [1, 1], [2, 0]],
                            "width": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "schema_version": 1,
                "train": [{"map": "train.json", "seeds": [5, 6]}],
                "tune": [{"map": "must-not-open-tune.json", "seeds": [7]}],
                "held_out": [{"map": "must-not-open-heldout.json", "seeds": [8]}],
            }
            manifest_path = root / "split.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            episodes = load_train_site_map_episodes(manifest_path)

        self.assertEqual([episode.seed for episode in episodes], [5, 6])
        self.assertEqual({episode.map_id for episode in episodes}, {"custom-track-train"})


class TestRolloutHazardInstrumentation(unittest.TestCase):
    def test_uses_pre_action_cached_features_and_actual_transition_outcomes(self):
        from training.hazard_exposure import HazardExposureRecorder
        from training.train_policy import collect_rollout

        class Labels:
            def __init__(self, progress, damage, collision=False):
                self.speed = 0.0
                self.wheel_omega = (0.0, 0.0, 0.0, 0.0)
                self.steering_angle = 0.0
                self.yaw_rate = 0.0
                self.lateral_error = 0.0
                self.road_half_width = 1.0
                self.heading_error = 0.0
                self.tile_progress = progress
                self.damage = damage
                self.collision = collision
                self.off_track = False
                self.finished = False

        class Model:
            throttle_limit = 1.0
            brake_limit = 1.0

            def __init__(self):
                self.last_visual_features = None
                self.forward_ids = []

            def __call__(self, observations):
                from haic_agent.networks import PolicyOutput

                identifier = int(observations[0, 0, 0, 0].item())
                self.forward_ids.append(identifier)
                feature = torch.tensor(
                    [[0.5, 0.0, 0.0, 0.0, 1.0, -0.2 if identifier == 0 else 0.0, 0.8 if identifier == 0 else 0.0]]
                )
                self.last_visual_features = feature
                return PolicyOutput(
                    latent=torch.zeros((1, 128)),
                    action_mean=torch.zeros((1, 2)),
                    action_log_std=torch.zeros((1, 2)),
                    value=torch.zeros(1),
                    auxiliary_predictions=torch.zeros((1, 10)),
                )

            def sample_actions_with_pretransform(self, output):
                del output
                return (
                    torch.tensor([[0.0, 0.1, 0.0]]),
                    torch.tensor([0.0]),
                    torch.zeros((1, 2)),
                )

        class Environment:
            def reset(self):
                self.step_index = 0
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step_transition(self, action):
                from training.env_factory import CollectedTransition

                del action
                current_index = self.step_index
                self.step_index += 1
                current_progress = 0.63 + current_index * 0.01
                collision = current_index == 0
                return CollectedTransition(
                    observation=np.full((4, 84, 84), current_index, dtype=np.float32),
                    action=np.asarray([0.4, 0.3, 0.7], dtype=np.float32),
                    next_observation=np.full((4, 84, 84), self.step_index, dtype=np.float32),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    labels=Labels(
                        progress=current_progress + (0.01 if current_index == 0 else 0.02),
                        damage=0.4,
                        collision=collision,
                    ),
                    hud_features=None,
                    observation_labels=(
                        Labels(
                            progress=current_progress,
                            damage=0.1 if current_index == 0 else 0.4,
                        )
                        if current_index == 0
                        else None
                    ),
                )

            def close(self):
                pass

        recorder = HazardExposureRecorder()
        model = Model()
        from training.site_maps import (
            CustomTrackGeometry,
            SiteMapEpisode,
            SiteMapSpec,
        )

        site_map = SiteMapSpec(
            map_id="custom-track-train",
            map_kind="custom",
            track_id=1,
            seed=5,
            obstacle_mode="custom_only",
            obstacles=(),
            max_steps=20,
            frame_skip=4,
            geometry=CustomTrackGeometry(((0.0, 0.0), (1.0, 1.0), (2.0, 0.0)), 4.0),
        )
        episode = SiteMapEpisode(site_map, 5, Path("train.json"))
        with patch(
            "training.train_policy.interleave_training_episodes",
            return_value=(episode,),
        ), patch(
            "training.train_policy._create_environment_for_episode",
            return_value=Environment(),
        ), patch("training.train_policy.shape_transition_reward", return_value=0.0):
            rollout = collect_rollout(
                model,
                [episode],
                total_steps=2,
                max_decisions=5,
                hazard_exposure_recorder=recorder,
                update_number=4,
            )

        plain_model = Model()
        with patch(
            "training.train_policy.interleave_training_episodes",
            return_value=(episode,),
        ), patch(
            "training.train_policy._create_environment_for_episode",
            return_value=Environment(),
        ), patch("training.train_policy.shape_transition_reward", return_value=0.0):
            plain_rollout = collect_rollout(
                plain_model,
                [episode],
                total_steps=2,
                max_decisions=5,
            )

        for recorded_step, plain_step in zip(rollout.steps, plain_rollout.steps, strict=True):
            for field in (
                "observation",
                "action",
                "pretransform_action",
                "auxiliary_targets",
            ):
                torch.testing.assert_close(
                    getattr(recorded_step, field), getattr(plain_step, field)
                )
            for field in (
                "log_probability",
                "value",
                "next_value",
                "reward",
                "advantage",
                "return_",
            ):
                self.assertEqual(getattr(recorded_step, field), getattr(plain_step, field))
            self.assertEqual(recorded_step.terminated, plain_step.terminated)
            self.assertEqual(recorded_step.truncated, plain_step.truncated)

        self.assertEqual(len(rollout), 2)
        self.assertEqual(model.forward_ids, [0, 1, 2])
        result = recorder.to_dict()["episodes"][0]
        self.assertEqual(result["update_number"], 4)
        self.assertEqual(result["bins"][12]["all_decisions"], 2)
        self.assertEqual(result["bins"][12]["urgent_count"], 1)
        self.assertEqual(result["bins"][12]["next_collision_count"], 1)
        self.assertAlmostEqual(result["bins"][12]["damage_delta_sum"], 0.3)
        self.assertAlmostEqual(result["bins"][12]["gas_mean"], 0.3)
        self.assertAlmostEqual(result["bins"][12]["brake_mean"], 0.7)
        self.assertEqual(result["bins"][12]["risk_positive_brake_active_count"], 1)
        urgent_band = result["bins"][12]["urgency_bands"][3]
        self.assertAlmostEqual(urgent_band["signed_obstacle_offset_mean"], -0.2)
        self.assertAlmostEqual(urgent_band["steer_away_alignment_mean"], 0.4)
        self.assertEqual(urgent_band["steer_away_aligned_count"], 1)
        self.assertEqual(len(result["collision_followups"]), 1)
        followup = result["collision_followups"][0]
        self.assertEqual(followup["origin_bin_index"], 12)
        self.assertEqual(followup["next_bin_index"], 12)
        self.assertEqual(followup["count"], 1)
        self.assertAlmostEqual(followup["followup_progress_change_mean"], 0.0)
        self.assertEqual(result["bins"][0]["all_decisions"], 0)
        self.assertEqual(result["boundary"], "rollout_cutoff")


if __name__ == "__main__":
    unittest.main()
