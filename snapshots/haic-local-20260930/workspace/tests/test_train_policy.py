import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _cache_probe_observation(identifier):
    return torch.full((4, 84, 84), float(identifier), dtype=torch.float32).numpy()


def _cache_probe_labels():
    from training.labels import TrainingLabels

    return TrainingLabels(
        speed=0.0,
        wheel_omega=(0.0, 0.0, 0.0, 0.0),
        steering_angle=0.0,
        yaw_rate=0.0,
        tile_progress=0.0,
        collision=False,
        damage=0.0,
        off_track=False,
        finished=False,
    )


class _CacheProbeModel:
    def __init__(self):
        self.forward_observation_ids = []
        self.last_visual_features = None

    def __call__(self, observations):
        from haic_agent.networks import PolicyOutput

        identifier = int(observations[0, 0, 0, 0].item())
        self.forward_observation_ids.append(identifier)
        self.last_visual_features = torch.full((1, 7), float(identifier))
        return PolicyOutput(
            latent=torch.full((1, 128), float(identifier)),
            action_mean=torch.zeros((1, 2)),
            action_log_std=torch.zeros((1, 2)),
            value=torch.tensor([identifier + 0.5], dtype=torch.float32),
            auxiliary_predictions=torch.full((1, 10), float(identifier)),
        )

    def sample_actions_with_pretransform(self, output):
        identifier = output.latent[0, 0]
        action = torch.stack((identifier / 100.0, torch.tensor(0.05), torch.tensor(0.0))).unsqueeze(0)
        log_probability = torch.tensor([-0.25]) - identifier
        pretransform_action = torch.stack((identifier, identifier + 0.5)).unsqueeze(0)
        return action, log_probability, pretransform_action


class _CacheProbeEnvironment:
    def __init__(self, initial_id, next_ids, *, terminated=(), truncated=()):
        self.initial_id = initial_id
        self.next_ids = tuple(next_ids)
        self.terminated_steps = set(terminated)
        self.truncated_steps = set(truncated)
        self.step_index = 0

    def reset(self):
        self.current_id = self.initial_id
        return _cache_probe_observation(self.current_id), {}

    def step_transition(self, action):
        from training.env_factory import CollectedTransition

        del action
        current_id = self.current_id
        next_id = self.next_ids[self.step_index]
        self.step_index += 1
        self.current_id = next_id
        return CollectedTransition(
            observation=_cache_probe_observation(current_id),
            action=torch.zeros(3).numpy(),
            next_observation=_cache_probe_observation(next_id),
            reward=0.0,
            terminated=self.step_index in self.terminated_steps,
            truncated=self.step_index in self.truncated_steps,
            labels=_cache_probe_labels(),
            hud_features=None,
        )

    def close(self):
        pass


class TestRolloutCollection(unittest.TestCase):
    def test_reuses_next_state_policy_output_and_visual_features_for_the_next_action(self):
        from training.train_policy import collect_rollout

        model = _CacheProbeModel()
        received_visual_features = []
        received_next_visual_features = []

        def capture_reward(
            transition,
            previous_progress,
            *,
            visual_features=None,
            next_visual_features=None,
            **_,
        ):
            del transition, previous_progress
            received_visual_features.append(visual_features.clone())
            received_next_visual_features.append(next_visual_features.clone())
            return 0.0

        with (
            patch(
                "training.train_policy._create_environment_for_episode",
                return_value=_CacheProbeEnvironment(0, (1, 2)),
            ),
            patch("training.train_policy.shape_transition_reward", side_effect=capture_reward),
        ):
            rollout = collect_rollout(model, [(1, 1)], total_steps=2, max_decisions=2)

        self.assertEqual(model.forward_observation_ids, [0, 1, 2])
        self.assertEqual([step.value for step in rollout.steps], [0.5, 1.5])
        self.assertEqual([step.next_value for step in rollout.steps], [1.5, 2.5])
        torch.testing.assert_close(rollout.steps[0].action, torch.tensor([0.0, 0.05, 0.0]))
        torch.testing.assert_close(rollout.steps[1].action, torch.tensor([0.01, 0.05, 0.0]))
        self.assertEqual([step.log_probability for step in rollout.steps], [-0.25, -1.25])
        torch.testing.assert_close(rollout.steps[1].pretransform_action, torch.tensor([1.0, 1.5]))
        torch.testing.assert_close(received_visual_features[0], torch.zeros(7))
        torch.testing.assert_close(received_next_visual_features[0], torch.ones(7))
        torch.testing.assert_close(received_visual_features[1], torch.ones(7))
        torch.testing.assert_close(received_next_visual_features[1], torch.full((7,), 2.0))

    def test_passes_hazard_potential_scale_and_discount_to_reward_shaper(self):
        from training.train_policy import collect_rollout

        received = []

        def capture_reward(
            transition,
            previous_progress,
            *,
            gamma=None,
            hazard_potential_scale=0.0,
            **_,
        ):
            del transition, previous_progress
            received.append((gamma, hazard_potential_scale))
            return 0.0

        with (
            patch(
                "training.train_policy._create_environment_for_episode",
                return_value=_CacheProbeEnvironment(0, (1,)),
            ),
            patch("training.train_policy.shape_transition_reward", side_effect=capture_reward),
        ):
            collect_rollout(
                _CacheProbeModel(),
                [(1, 1)],
                total_steps=1,
                max_decisions=1,
                gamma=0.91,
                hazard_potential_scale=0.17,
            )

        self.assertEqual(received, [(0.91, 0.17)])

    def test_summarizes_hazard_risk_reduction_and_collision_exposure(self):
        from training.train_policy import summarize_hazard_potential_transitions

        summary = summarize_hazard_potential_transitions(
            [
                {"current_risk": 0.90, "next_risk": 0.60, "shaping_reward": 0.02, "collision": False},
                {"current_risk": 0.85, "next_risk": 0.95, "shaping_reward": -0.01, "collision": True},
                {"current_risk": 0.30, "next_risk": 0.10, "shaping_reward": 0.01, "collision": False},
                {"current_risk": 0.00, "next_risk": 0.00, "shaping_reward": 0.00, "collision": False},
            ]
        )

        self.assertEqual(summary["transition_count"], 4)
        self.assertEqual(summary["risk_exposed_transition_count"], 3)
        self.assertAlmostEqual(summary["risk_decreased_fraction"], 2.0 / 3.0)
        self.assertEqual(summary["urgent_transition_count"], 2)
        self.assertAlmostEqual(summary["urgent_risk_decreased_fraction"], 0.5)
        self.assertAlmostEqual(summary["mean_urgent_risk_delta"], -0.10)
        self.assertEqual(summary["urgent_collision_count"], 1)

    def test_terminal_transition_does_not_reuse_its_next_output_after_environment_reset(self):
        from training.train_policy import collect_rollout

        model = _CacheProbeModel()
        environments = [
            _CacheProbeEnvironment(10, (99,), terminated=(1,)),
            _CacheProbeEnvironment(20, (21,)),
        ]
        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=environments,
        ):
            rollout = collect_rollout(model, [(1, 1), (1, 2)], total_steps=2, max_decisions=1)

        self.assertEqual(model.forward_observation_ids, [10, 99, 20, 21])
        self.assertEqual([step.value for step in rollout.steps], [10.5, 20.5])
        torch.testing.assert_close(rollout.steps[1].action, torch.tensor([0.2, 0.05, 0.0]))

    def test_cache_does_not_cross_rollout_update_boundary(self):
        from training.train_policy import collect_rollout

        model = _CacheProbeModel()
        environments = [
            _CacheProbeEnvironment(3, (4,)),
            _CacheProbeEnvironment(9, (10,)),
        ]
        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=environments,
        ):
            before_update = collect_rollout(model, [(1, 1)], total_steps=1, max_decisions=4)
            after_update = collect_rollout(model, [(1, 2)], total_steps=1, max_decisions=4)

        self.assertEqual(model.forward_observation_ids, [3, 4, 9, 10])
        self.assertEqual(before_update.steps[0].next_value, 4.5)
        self.assertEqual(after_update.steps[0].value, 9.5)
        torch.testing.assert_close(after_update.steps[0].action, torch.tensor([0.09, 0.05, 0.0]))

    def test_truncation_keeps_next_value_for_bootstrap(self):
        from training.train_policy import collect_rollout

        model = _CacheProbeModel()
        with (
            patch(
                "training.train_policy._create_environment_for_episode",
                return_value=_CacheProbeEnvironment(5, (7,), truncated=(1,)),
            ),
            patch("training.train_policy.shape_transition_reward", return_value=1.25),
        ):
            rollout = collect_rollout(
                model,
                [(1, 1)],
                total_steps=1,
                max_decisions=4,
                gamma=0.9,
            )

        self.assertEqual(model.forward_observation_ids, [5, 7])
        self.assertFalse(rollout.steps[0].terminated)
        self.assertTrue(rollout.steps[0].truncated)
        self.assertEqual(rollout.steps[0].next_value, 7.5)
        self.assertAlmostEqual(rollout.steps[0].return_, 8.0)

    def test_cache_can_be_disabled_as_an_exact_collection_reference(self):
        from training.train_policy import collect_rollout

        cached_model = _CacheProbeModel()
        uncached_model = _CacheProbeModel()
        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=[
                _CacheProbeEnvironment(0, (1, 2)),
                _CacheProbeEnvironment(0, (1, 2)),
            ],
        ), patch("training.train_policy.shape_transition_reward", return_value=0.75):
            cached = collect_rollout(
                cached_model, [(1, 1)], total_steps=2, max_decisions=2
            )
            uncached = collect_rollout(
                uncached_model,
                [(1, 1)],
                total_steps=2,
                max_decisions=2,
                reuse_next_policy_output=False,
            )

        self.assertEqual(cached_model.forward_observation_ids, [0, 1, 2])
        self.assertEqual(uncached_model.forward_observation_ids, [0, 1, 1, 2])
        for cached_step, uncached_step in zip(cached.steps, uncached.steps, strict=True):
            torch.testing.assert_close(cached_step.observation, uncached_step.observation)
            torch.testing.assert_close(cached_step.action, uncached_step.action)
            torch.testing.assert_close(cached_step.pretransform_action, uncached_step.pretransform_action)
            self.assertEqual(cached_step.log_probability, uncached_step.log_probability)
            self.assertEqual(cached_step.value, uncached_step.value)
            self.assertEqual(cached_step.next_value, uncached_step.next_value)
            self.assertEqual(cached_step.reward, uncached_step.reward)
            self.assertEqual(cached_step.return_, uncached_step.return_)

    def test_manual_episode_and_rollout_limits_mark_gae_boundaries(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import collect_rollout

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )

        class _Environment:
            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return CollectedTransition(
                    observation=pixels,
                    action=action,
                    next_observation=pixels,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    labels=labels,
                    hud_features=None,
                )

            def close(self):
                pass

        class _Output:
            value = torch.zeros(1)

        class _Model:
            def __call__(self, observations):
                return _Output()

            def sample_actions_with_pretransform(self, output):
                return torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 2)

        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=lambda episode, max_decisions: _Environment(),
        ):
            rollout = collect_rollout(
                _Model(), [(1, 1)], total_steps=3, max_decisions=2
            )

        self.assertEqual(len(rollout), 3)
        self.assertFalse(rollout.steps[0].truncated)
        self.assertTrue(rollout.steps[1].truncated)
        self.assertTrue(rollout.steps[2].truncated)

    def test_rollout_passes_current_observation_features_into_the_reward(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import collect_rollout

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )

        class _Environment:
            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return CollectedTransition(
                    observation=pixels,
                    action=action,
                    next_observation=pixels,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    labels=labels,
                    hud_features=None,
                )

            def close(self):
                pass

        class _Output:
            value = torch.zeros(1)

        class _Model:
            def __init__(self):
                self.calls = 0
                self.last_visual_features = None

            def __call__(self, observations):
                self.calls += 1
                self.last_visual_features = torch.full((1, 7), float(self.calls))
                return _Output()

            def sample_actions_with_pretransform(self, output):
                return torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 2)

        received_features = []
        received_next_features = []

        def capture_reward(
            transition,
            previous_progress,
            *,
            visual_features=None,
            next_visual_features=None,
            **_,
        ):
            del transition, previous_progress
            received_features.append(visual_features)
            received_next_features.append(next_visual_features)
            return 0.0

        with (
            patch(
                "training.train_policy._create_environment_for_episode",
                return_value=_Environment(),
            ),
            patch("training.train_policy.shape_transition_reward", side_effect=capture_reward),
        ):
            collect_rollout(_Model(), [(1, 1)], total_steps=1, max_decisions=1)

        torch.testing.assert_close(received_features[0], torch.ones(7))
        torch.testing.assert_close(received_next_features[0], torch.full((7,), 2.0))

    def test_mixed_domain_rollout_round_robins_each_official_track_with_custom_maps(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.site_maps import SiteMapEpisode, SiteMapSpec
        from training.train_policy import collect_rollout

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )

        class _Environment:
            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return CollectedTransition(
                    observation=pixels,
                    action=action,
                    next_observation=pixels,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    labels=labels,
                    hud_features=None,
                )

            def close(self):
                pass

        class _Output:
            value = torch.zeros(1)

        class _Model:
            def __call__(self, observations):
                return _Output()

            def sample_actions_with_pretransform(self, output):
                return torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 2)

        custom_episodes = tuple(
            SiteMapEpisode(
                site_map=SiteMapSpec(
                    map_id=f"custom-track-sample-{index}",
                    map_kind="custom",
                    track_id=1,
                    seed=index,
                    obstacle_mode="custom_only",
                    obstacles=(),
                    max_steps=4,
                    frame_skip=4,
                ),
                seed=index,
                source_path=Path(f"custom-{index}.json"),
            )
            for index in range(14)
        )
        official_episodes = tuple((1, 100 + index) for index in range(4)) + tuple(
            (2, 200 + index) for index in range(4)
        )
        episodes = custom_episodes + official_episodes
        selected = []

        with patch(
            "training.train_policy._create_environment_for_episode",
            side_effect=lambda episode, max_decisions: (selected.append(episode) or _Environment()),
        ):
            collect_rollout(_Model(), episodes, total_steps=6, max_decisions=1)

        domains = [
            "custom" if isinstance(episode, SiteMapEpisode) else f"track{episode[0]}"
            for episode in selected
        ]
        self.assertEqual(domains, ["custom", "track1", "track2"] * 2)


class TestRepeatedPPOTraining(unittest.TestCase):
    def test_training_distributes_total_steps_across_multiple_on_policy_updates(self):
        from training.train_policy import train

        metrics = {
            "finish_rate": 0.0,
            "median_finished_lap_time_s": None,
            "p90_finished_lap_time_s": None,
            "mean_progress": 0.1,
        }
        action_stats = {"steer_mean": -0.1, "brake_fraction": 0.08}

        class _Updater:
            def __init__(self, model, config):
                self.update_sizes = []
                self.config = config
                updater_instances.append(self)

            def update(self, rollout):
                self.update_sizes.append(len(rollout))
                return {"policy_loss": 0.0}

        updater_instances = []
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_policy.collect_rollout", side_effect=[range(5), range(5)]) as collect,
                patch("training.train_policy.rollout_action_metrics", return_value=action_stats) as action_metrics,
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy", return_value=metrics) as evaluate,
                patch("training.train_policy.hud_ablation", return_value={"hud_enabled": metrics}),
                patch("training.train_policy.save_best_checkpoint", return_value=True) as save_best,
            ):
                result = train(
                    output_directory=Path(directory),
                    total_steps=10,
                    max_decisions=20,
                    evaluation_max_decisions=7,
                    seed=42,
                    updates=2,
                    learning_rate=1e-4,
                    teacher_warmup_epochs=0,
                    use_hud=False,
                    use_visual_features=True,
                )

        self.assertEqual(collect.call_count, 2)
        self.assertEqual(action_metrics.call_count, 2)
        self.assertEqual([call.kwargs["total_steps"] for call in collect.call_args_list], [5, 5])
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(
            [call.kwargs["max_decisions"] for call in evaluate.call_args_list],
            [7, 7, 20],
        )
        self.assertEqual(save_best.call_count, 2)
        self.assertEqual(result["training_steps"], 10)
        self.assertEqual(result["updates_completed"], 2)
        self.assertEqual([item["steps"] for item in result["updates"]], [5, 5])
        self.assertEqual([item["action_metrics"] for item in result["updates"]], [action_stats, action_stats])
        self.assertTrue(all(not call.args[0].use_hud for call in collect.call_args_list))
        self.assertTrue(all(call.args[0].use_visual_features for call in collect.call_args_list))
        self.assertTrue(all(call.kwargs["metadata"]["use_hud"] is False for call in save_best.call_args_list))
        self.assertTrue(
            all(call.kwargs["metadata"]["use_visual_features"] is True for call in save_best.call_args_list)
        )
        self.assertTrue(result["use_visual_features"])
        self.assertIn("full_tune_eval_s", result["timings_s"])
        self.assertAlmostEqual(updater_instances[0].config.learning_rate, 1e-4)

    def test_teacher_warmup_runs_before_ppo_and_receives_only_train_episodes(self):
        from training.env_factory import split_track_seeds
        from training.train_policy import train

        metrics = {
            "finish_rate": 0.0,
            "median_finished_lap_time_s": None,
            "p90_finished_lap_time_s": None,
            "mean_progress": 0.1,
        }
        events = []
        split = split_track_seeds(
            train=((1, 101),), tune=((2, 201),), held_out=((3, 301),)
        )

        class _Updater:
            def __init__(self, model, config):
                del model, config

            def update(self, rollout):
                del rollout
                events.append("ppo_update")
                return {"policy_loss": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "training.train_policy.collect_teacher_demonstrations",
                    side_effect=lambda episodes, **kwargs: (
                        events.append(("collect_teacher", tuple(episodes.train), kwargs["max_decisions"]))
                        or "demo-data"
                    ),
                ),
                patch(
                    "training.train_policy.behavioral_cloning_warmup",
                    side_effect=lambda model, demonstrations, **kwargs: (
                        events.append(("teacher_warmup", demonstrations))
                        or {"enabled": True, "final_action_mse": 0.01}
                    ),
                ),
                patch(
                    "training.train_policy.collect_rollout",
                    side_effect=lambda *args, **kwargs: (events.append("ppo_rollout") or range(2)),
                ),
                patch(
                    "training.train_policy.rollout_action_metrics",
                    return_value={"steer_mean": 0.0},
                ),
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy", return_value=metrics),
                patch("training.train_policy.hud_ablation", return_value={"hud_enabled": metrics}),
                patch("training.train_policy.save_best_checkpoint", return_value=True),
            ):
                result = train(
                    output_directory=Path(directory),
                    total_steps=2,
                    max_decisions=4,
                    seed=42,
                    updates=1,
                    split=split,
                    teacher_warmup_epochs=1,
                )

        self.assertEqual(
            events,
            [
                ("collect_teacher", split.train, 4),
                ("teacher_warmup", "demo-data"),
                "ppo_rollout",
                "ppo_update",
            ],
        )
        self.assertTrue(result["teacher_warmup"]["enabled"])
        self.assertEqual(result["teacher_warmup"]["final_action_mse"], 0.01)
        self.assertIn("teacher_warmup_s", result["timings_s"])


class TestPolicyThreadConsistency(unittest.TestCase):
    def test_training_uses_the_same_single_cpu_thread_as_agent_runtime(self):
        from training.train_policy import train

        metrics = {
            "finish_rate": 0.0,
            "median_finished_lap_time_s": None,
            "p90_finished_lap_time_s": None,
            "mean_progress": 0.0,
        }

        class _Updater:
            def __init__(self, model, config):
                del model, config

            def update(self, rollout):
                del rollout
                return {}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_policy.torch.set_num_threads") as set_threads,
                patch("training.train_policy.collect_rollout", return_value=range(1)),
                patch("training.train_policy.rollout_action_metrics", return_value={"steer_mean": 0.0}),
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy", return_value=metrics),
                patch("training.train_policy.hud_ablation", return_value={"hud_enabled": metrics}),
                patch("training.train_policy.save_best_checkpoint", return_value=True),
            ):
                train(
                    output_directory=Path(directory),
                    total_steps=1,
                    max_decisions=1,
                    seed=42,
                    updates=1,
                    learning_rate=1e-5,
                    teacher_warmup_epochs=0,
                    use_hud=False,
                    use_visual_features=True,
                )

        set_threads.assert_called_once_with(1)

    def test_standalone_policy_evaluation_uses_the_agent_cpu_thread_count(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import evaluate_policy

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=1.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=True,
        )

        class _Environment:
            t = 1.25
            finish_time_s = 12.0

            @property
            def unwrapped(self):
                return self

            def reset(self):
                return pixels, {}

            def step_transition(self, action):
                return CollectedTransition(
                    observation=pixels,
                    action=action,
                    next_observation=pixels,
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    labels=labels,
                    hud_features=None,
                )

            def close(self):
                pass

        class _Output:
            auxiliary_predictions = torch.zeros(1, 10)

        class _Model:
            def __call__(self, observations):
                return _Output()

            def deterministic_actions(self, output):
                return torch.zeros(1, 3)

        with (
            patch("training.train_policy.torch.set_num_threads") as set_threads,
            patch("training.train_policy._create_environment_for_episode", return_value=_Environment()),
        ):
            metrics = evaluate_policy(_Model(), [(1, 1)], max_decisions=1)

        set_threads.assert_called_once_with(1)
        self.assertEqual(metrics["finish_rate"], 1.0)
        self.assertEqual(metrics["median_finished_lap_time_s"], 10.75)
        self.assertEqual(metrics["p90_finished_lap_time_s"], 10.75)


class TestEvaluationBudget(unittest.TestCase):
    def test_hud_ablation_is_skipped_for_an_actor_without_a_hud_branch(self):
        # Break caught: running the sweep on a full-frame-only actor burned
        # 14-22% of a continuation's wall clock while both arms returned
        # byte-identical metrics, taking that time away from rollout.
        from unittest.mock import patch

        from training.train_policy import hud_ablation

        model = SimpleNamespace(use_hud=False)
        with patch("training.train_policy.evaluate_policy") as evaluate:
            report = hud_ablation(model, [(1, 1)], max_decisions=4)

        evaluate.assert_not_called()
        self.assertTrue(report["skipped"])
        self.assertIn("HUD", report["reason"])

    def test_hud_ablation_still_runs_both_arms_when_the_branch_is_active(self):
        # Break caught: skipping unconditionally would silently drop the
        # measurement that justifies keeping the HUD branch at all.
        from unittest.mock import patch

        from training.train_policy import hud_ablation

        model = SimpleNamespace(use_hud=True)
        with patch(
            "training.train_policy.evaluate_policy", return_value={"finish_rate": 1.0}
        ) as evaluate:
            report = hud_ablation(model, [(1, 1)], max_decisions=4)

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(set(report), {"hud_enabled", "hud_disabled"})
        self.assertTrue(model.use_hud)


class TestTrainingSignalScaling(unittest.TestCase):
    def test_hazard_potential_reward_is_added_after_base_reward_scaling(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=70.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.zeros(3, dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        current = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        next_features = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )

        base = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=current,
            next_visual_features=next_features,
            gamma=0.99,
            hazard_potential_scale=0.0,
        )
        shaped = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=current,
            next_visual_features=next_features,
            gamma=0.99,
            hazard_potential_scale=0.10,
        )

        # beta * (gamma * -0.5 - -1.0), added after REWARD_SCALE.
        self.assertAlmostEqual(shaped - base, 0.0505)

    def test_collision_and_terminal_crash_receive_strong_learning_penalties(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=20.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=True,
            damage=1.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=True,
            truncated=False,
            labels=labels,
            hud_features=None,
        )

        reward = shape_transition_reward(transition, previous_progress=0.0)

        self.assertAlmostEqual(reward, -12.015)

    def test_safe_speed_is_rewarded_but_speed_reward_is_gated_after_collision_or_off_track(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()

        def shaped_reward(speed, *, collision=False, off_track=False):
            labels = TrainingLabels(
                speed=speed,
                wheel_omega=(0.0, 0.0, 0.0, 0.0),
                steering_angle=0.0,
                yaw_rate=0.0,
                tile_progress=0.0,
                collision=collision,
                damage=0.0,
                off_track=off_track,
                finished=False,
            )
            transition = CollectedTransition(
                observation=pixels,
                action=torch.zeros(3).numpy(),
                next_observation=pixels,
                reward=0.0,
                terminated=False,
                truncated=False,
                labels=labels,
                hud_features=None,
            )
            return shape_transition_reward(transition, previous_progress=0.0)

        safe_slow = shaped_reward(20.0)
        safe_fast = shaped_reward(40.0)
        self.assertGreater(safe_fast - safe_slow, 0.0039)

        for failure in ({"collision": True}, {"off_track": True}):
            with self.subTest(failure=failure):
                self.assertAlmostEqual(
                    shaped_reward(20.0, **failure), shaped_reward(50.0, **failure)
                )

    def test_visible_centered_obstacle_penalizes_throttle_but_not_coasting(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=40.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        clear_visual = np.asarray([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        centered_hazard = np.asarray([0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        def shaped_reward(gas, visual_features):
            transition = CollectedTransition(
                observation=pixels,
                action=np.asarray([0.0, gas, 0.0], dtype=np.float32),
                next_observation=pixels,
                reward=0.0,
                terminated=False,
                truncated=False,
                labels=labels,
                hud_features=None,
            )
            return shape_transition_reward(
                transition, previous_progress=0.0, visual_features=visual_features
            )

        clear_throttle = shaped_reward(0.06, clear_visual)
        hazard_throttle = shaped_reward(0.06, centered_hazard)
        hazard_coast = shaped_reward(0.0, centered_hazard)

        from training.train_policy import (
            HAZARD_SPEED_RELIEF,
            REWARD_SCALE,
            SPEED_SHORTFALL_PENALTY,
            SPEED_TARGET,
        )

        # Comparing the two hazard cases isolates the throttle penalty from the
        # cruise-speed charge, which is identical at identical speed.
        self.assertAlmostEqual(hazard_coast - hazard_throttle, 0.4)
        # A visible hazard lowers the cruise-speed target, so this speed sits at
        # or above target in front of one but short of it on clear road.
        hazard_target = SPEED_TARGET * (1.0 - HAZARD_SPEED_RELIEF)
        self.assertGreaterEqual(40.0, hazard_target)
        clear_charge = (
            SPEED_SHORTFALL_PENALTY * (SPEED_TARGET - 40.0) / SPEED_TARGET * REWARD_SCALE
        )
        self.assertAlmostEqual(hazard_coast - clear_throttle, clear_charge)
        self.assertAlmostEqual(clear_throttle - hazard_throttle, 0.4 - clear_charge)

    def test_recovery_bonus_is_disabled_while_visible_hazard_risk_is_high(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import REWARD_SCALE, shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        clear_visual = np.asarray(
            [0.02, -0.063, -0.226, 0.389, 0.0, 0.0, 0.0], dtype=np.float32
        )
        centered_hazard = np.asarray(
            [0.02, -0.063, -0.226, 0.389, 1.0, 0.18, 1.0], dtype=np.float32
        )

        def reward(damage, visual_features):
            labels = TrainingLabels(
                speed=0.1,
                wheel_omega=(0.0, 0.0, 0.0, 0.0),
                steering_angle=0.0,
                yaw_rate=0.0,
                tile_progress=0.251,
                collision=False,
                damage=damage,
                off_track=False,
                finished=False,
            )
            transition = CollectedTransition(
                observation=pixels,
                action=np.asarray([0.135, 0.138, 0.0], dtype=np.float32),
                next_observation=pixels,
                reward=0.0,
                terminated=False,
                truncated=False,
                labels=labels,
                hud_features=None,
            )
            return shape_transition_reward(
                transition,
                previous_progress=0.251,
                previous_speed=0.1,
                visual_features=visual_features,
                throttle_limit=0.24,
                brake_limit=0.28,
            )

        high_risk_damage_delta = reward(0.2, centered_hazard) - reward(
            0.0, centered_hazard
        )
        clear_road_damage_delta = reward(0.2, clear_visual) - reward(
            0.0, clear_visual
        )

        # The only remaining effect of damage at a high-risk stall should be
        # the normal -0.1 * damage term. Recovery rewards remain available on
        # clear road, where they help the policy resume after damage.
        self.assertAlmostEqual(high_risk_damage_delta, -0.1 * 0.2 * REWARD_SCALE)
        self.assertGreater(clear_road_damage_delta, 0.0)

    def test_auxiliary_sensor_targets_are_normalized_by_physical_scale(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import auxiliary_targets

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=20.0,
            wheel_omega=(25.0, -50.0, 100.0, 0.0),
            steering_angle=0.2,
            yaw_rate=-1.5,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
            lateral_error=4.0,
            road_half_width=8.0,
            heading_error=torch.pi / 2,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )

        targets = auxiliary_targets(transition)

        torch.testing.assert_close(
            targets,
            torch.tensor([0.5, 0.25, -0.5, 1.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0]),
        )

    def test_auxiliary_visual_targets_match_the_observation_used_for_the_action(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import auxiliary_targets

        pixels = torch.zeros(4, 84, 84).numpy()
        observation_labels = TrainingLabels(
            speed=12.0,
            wheel_omega=(10.0, 20.0, 30.0, 40.0),
            steering_angle=0.1,
            yaw_rate=0.2,
            tile_progress=0.1,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
            lateral_error=-2.0,
            road_half_width=8.0,
            heading_error=-torch.pi / 2,
        )
        next_labels = TrainingLabels(
            speed=40.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.2,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=next_labels,
            hud_features=None,
            observation_labels=observation_labels,
        )

        targets = auxiliary_targets(transition)

        torch.testing.assert_close(
            targets,
            torch.tensor([0.3, 0.1, 0.2, 0.3, 0.4, 0.25, 0.2 / 3.0, -0.25, -1.0, 0.0]),
        )

    def test_shaped_rewards_are_scaled_before_critic_returns(self):
        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=0.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.3,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=torch.zeros(3).numpy(),
            next_observation=pixels,
            reward=1.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )

        reward = shape_transition_reward(transition, previous_progress=0.1)

        # 0.235 = (1.0 env + 2.0 progress - 0.05 time - 0.6 speed shortfall at a
        # standstill) * 0.1 reward scale.
        self.assertAlmostEqual(reward, 0.235)

    def test_route_deviation_and_heading_error_add_dense_training_penalties(self):
        import math
        from types import SimpleNamespace

        from training.train_policy import shape_transition_reward

        def make_transition(lateral_error, heading_error):
            labels = SimpleNamespace(
                speed=0.0,
                tile_progress=0.0,
                finished=False,
                collision=False,
                off_track=False,
                damage=0.0,
                lateral_error=lateral_error,
                road_half_width=8.0,
                heading_error=heading_error,
            )
            return SimpleNamespace(reward=0.0, labels=labels)

        centered = shape_transition_reward(make_transition(0.0, 0.0), 0.0)
        edge_and_misaligned = shape_transition_reward(
            make_transition(8.0, math.pi / 2.0), 0.0
        )

        # Both cases sit at a standstill, so both carry the same -0.06 scaled
        # speed shortfall charge on top of the route-tracking penalty.
        self.assertAlmostEqual(centered, -0.065)
        self.assertAlmostEqual(edge_and_misaligned, -0.165)


class TestCurveBrakeScreenTraining(unittest.TestCase):
    def test_recovery_clearance_reward_credits_only_positive_visible_risk_reduction(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=4.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.4,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.zeros(3, dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        current = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        next_features = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )

        control = shape_transition_reward(
            transition,
            previous_progress=0.0,
            previous_speed=4.0,
            visual_features=current,
            next_visual_features=next_features,
            recovery_clearance_reward=0.0,
        )
        treatment = shape_transition_reward(
            transition,
            previous_progress=0.0,
            previous_speed=4.0,
            visual_features=current,
            next_visual_features=next_features,
            recovery_clearance_reward=6.0,
        )

        self.assertAlmostEqual(treatment - control, 0.3)

    def test_recovery_clearance_reward_requires_safe_damaged_low_speed_high_risk_state(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        current = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        lower_risk = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )
        subthreshold_risk = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.sqrt(0.49)], dtype=np.float32
        )
        quarter_risk = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5], dtype=np.float32
        )
        higher_risk = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        def make_transition(*, damage=0.4, speed=4.0, collision=False, off_track=False):
            labels = TrainingLabels(
                speed=speed,
                wheel_omega=(0.0, 0.0, 0.0, 0.0),
                steering_angle=0.0,
                yaw_rate=0.0,
                tile_progress=0.0,
                collision=collision,
                damage=damage,
                off_track=off_track,
                finished=False,
            )
            return CollectedTransition(
                observation=pixels,
                action=np.zeros(3, dtype=np.float32),
                next_observation=pixels,
                reward=0.0,
                terminated=False,
                truncated=False,
                labels=labels,
                hud_features=None,
            )

        cases = (
            ("no_change", make_transition(), current, current, 4.0),
            ("risk_increase", make_transition(), lower_risk, higher_risk, 4.0),
            ("collision", make_transition(collision=True), current, lower_risk, 4.0),
            ("off_track", make_transition(off_track=True), current, lower_risk, 4.0),
            ("insufficient_damage", make_transition(damage=0.1), current, lower_risk, 4.0),
            ("too_fast", make_transition(speed=9.0), current, lower_risk, 9.0),
            ("risk_below_gate", make_transition(), subthreshold_risk, quarter_risk, 4.0),
        )
        for name, transition, before, after, previous_speed in cases:
            with self.subTest(name=name):
                without_bonus = shape_transition_reward(
                    transition,
                    previous_progress=0.0,
                    previous_speed=previous_speed,
                    visual_features=before,
                    next_visual_features=after,
                    recovery_clearance_reward=0.0,
                )
                with_bonus = shape_transition_reward(
                    transition,
                    previous_progress=0.0,
                    previous_speed=previous_speed,
                    visual_features=before,
                    next_visual_features=after,
                    recovery_clearance_reward=6.0,
                )
                self.assertAlmostEqual(with_bonus, without_bonus)

        no_next_features = shape_transition_reward(
            make_transition(),
            previous_progress=0.0,
            previous_speed=4.0,
            visual_features=current,
            next_visual_features=None,
            recovery_clearance_reward=6.0,
        )
        no_signal = shape_transition_reward(
            make_transition(),
            previous_progress=0.0,
            previous_speed=4.0,
            visual_features=current,
            next_visual_features=None,
            recovery_clearance_reward=0.0,
        )
        self.assertAlmostEqual(no_next_features, no_signal)

    def test_recovery_clearance_reward_rejects_nonfinite_and_negative_values(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=4.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.4,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.zeros(3, dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        visual = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        next_features = np.asarray(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, np.sqrt(0.5)], dtype=np.float32
        )

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                shape_transition_reward(
                    transition,
                    previous_progress=0.0,
                    previous_speed=4.0,
                    visual_features=visual,
                    next_visual_features=next_features,
                    recovery_clearance_reward=value,
                )

    def test_curve_brake_reward_adds_only_the_registered_scaled_term(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=70.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.asarray([0.0, 0.0, 0.14], dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        visual = np.asarray([1.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0], dtype=np.float32)

        control = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=visual,
            curve_brake_reward=0.0,
        )
        treatment = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=visual,
            curve_brake_reward=6.0,
        )

        self.assertAlmostEqual(treatment - control, 0.225)

    def test_obstacle_brake_reward_is_overridable_with_existing_default(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=70.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.asarray([0.0, 0.0, 0.14], dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        visual = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        default = shape_transition_reward(
            transition, previous_progress=0.0, visual_features=visual
        )
        explicit_default = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=visual,
            obstacle_brake_reward=6.0,
        )
        doubled = shape_transition_reward(
            transition,
            previous_progress=0.0,
            visual_features=visual,
            obstacle_brake_reward=12.0,
        )

        self.assertAlmostEqual(default, explicit_default)
        self.assertAlmostEqual(doubled - explicit_default, 0.3)

    def test_obstacle_brake_reward_rejects_nonfinite_and_negative_values(self):
        import numpy as np

        from training.env_factory import CollectedTransition
        from training.labels import TrainingLabels
        from training.train_policy import shape_transition_reward

        pixels = torch.zeros(4, 84, 84).numpy()
        labels = TrainingLabels(
            speed=70.0,
            wheel_omega=(0.0, 0.0, 0.0, 0.0),
            steering_angle=0.0,
            yaw_rate=0.0,
            tile_progress=0.0,
            collision=False,
            damage=0.0,
            off_track=False,
            finished=False,
        )
        transition = CollectedTransition(
            observation=pixels,
            action=np.zeros(3, dtype=np.float32),
            next_observation=pixels,
            reward=0.0,
            terminated=False,
            truncated=False,
            labels=labels,
            hud_features=None,
        )
        visual = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)

        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                shape_transition_reward(
                    transition,
                    previous_progress=0.0,
                    visual_features=visual,
                    obstacle_brake_reward=value,
                )

    def test_actor_only_initialization_does_not_restore_optimizer_state(self):
        from haic_agent.networks import VisualActorCritic
        from training.ppo import PPOConfig, PPOUpdater
        from training.train_policy import load_actor_weights

        source = VisualActorCritic(use_hud=False, use_visual_features=True)
        expected_state = {
            name: value.detach().clone() for name, value in source.state_dict().items()
        }
        target = VisualActorCritic(use_hud=False, use_visual_features=True)
        updater = PPOUpdater(target, PPOConfig(learning_rate=2e-6))
        self.assertEqual(updater.optimizer.state, {})

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor-only-source.pt"
            torch.save(
                {
                    "model_state": expected_state,
                    "optimizer_state": {"must_not_be_loaded": True},
                    "step": 19456,
                    "metadata": {
                        "use_hud": False,
                        "use_visual_features": True,
                        "use_temporal_features": False,
                    },
                },
                checkpoint,
            )

            load_actor_weights(checkpoint, target)

        self.assertEqual(updater.optimizer.state, {})
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, expected_state[name])

    def test_deferred_tune_mode_saves_final_update_without_evaluating_tune(self):
        from training.train_policy import train

        class _Optimizer:
            state = {}

        class _Updater:
            captured_configs = []

            def __init__(self, model, config):
                del model
                self.captured_configs.append(config)
                self.optimizer = _Optimizer()

            def update(self, rollout):
                del rollout
                return {"policy_loss": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_policy.collect_rollout", side_effect=[range(2), range(2)]) as collect,
                patch("training.train_policy.rollout_action_metrics", return_value={"brake_fraction": 0.0}),
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy") as evaluate,
                patch("training.train_policy.hud_ablation") as ablation,
                patch("training.train_policy.save_best_checkpoint") as save_best,
                patch("training.train_policy.save_checkpoint") as save_final,
            ):
                result = train(
                    output_directory=Path(directory),
                    total_steps=4,
                    max_decisions=4,
                    seed=8104,
                    updates=2,
                    learning_rate=2e-6,
                    policy_mean_learning_rate=1e-5,
                    teacher_warmup_epochs=0,
                    use_hud=False,
                    use_visual_features=True,
                    obstacle_brake_reward=12.0,
                    curve_brake_reward=6.0,
                    tune_selection=False,
                )

        evaluate.assert_not_called()
        self.assertEqual(_Updater.captured_configs[0].policy_mean_learning_rate, 1e-5)
        self.assertTrue(
            all(call.kwargs["curve_brake_reward"] == 6.0 for call in collect.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["obstacle_brake_reward"] == 12.0 for call in collect.call_args_list)
        )
        ablation.assert_not_called()
        save_best.assert_not_called()
        save_final.assert_called_once()
        self.assertEqual(save_final.call_args.kwargs["step"], 4)
        final_metadata = save_final.call_args.kwargs["metadata"]
        self.assertEqual(final_metadata["updates_completed"], 2)
        self.assertEqual(final_metadata["policy_mean_learning_rate"], 1e-5)
        self.assertEqual(final_metadata["obstacle_brake_reward"], 12.0)
        self.assertEqual(final_metadata["reward_config"]["obstacle_brake_reward"], 12.0)
        self.assertEqual(final_metadata["curve_brake_reward"], 6.0)
        self.assertIsNone(final_metadata["tune_metrics"])
        self.assertEqual(result["training_steps"], 4)
        self.assertEqual(result["policy_mean_learning_rate"], 1e-5)
        self.assertEqual(result["obstacle_brake_reward"], 12.0)
        self.assertFalse(result["tune_selection_enabled"])
        self.assertIsNone(result["full_tune_metrics"])

    def test_actor_snapshots_are_written_at_requested_update_boundaries(self):
        from training.train_policy import train

        class _Optimizer:
            state = {}

            def state_dict(self):
                return {"state": {}, "param_groups": []}

        class _Updater:
            def __init__(self, model, config):
                del model, config
                self.optimizer = _Optimizer()

            def update(self, rollout):
                del rollout
                return {"policy_loss": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("training.train_policy.collect_rollout", side_effect=[range(2), range(2)]),
                patch("training.train_policy.rollout_action_metrics", return_value={"brake_fraction": 0.0}),
                patch("training.train_policy.PPOUpdater", _Updater),
                patch("training.train_policy.evaluate_policy") as evaluate,
                patch("training.train_policy.hud_ablation") as ablation,
            ):
                train(
                    output_directory=Path(directory),
                    total_steps=4,
                    max_decisions=4,
                    seed=8104,
                    updates=2,
                    learning_rate=2e-6,
                    teacher_warmup_epochs=0,
                    use_hud=False,
                    use_visual_features=True,
                    tune_selection=False,
                    actor_snapshot_steps=(2, 4),
                )

            evaluate.assert_not_called()
            ablation.assert_not_called()
            for step in (2, 4):
                snapshot = torch.load(
                    Path(directory) / f"actor-step-{step}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(snapshot["step"], step)
                self.assertEqual(set(snapshot), {"model_state", "step", "metadata"})
                self.assertTrue(snapshot["metadata"]["actor_snapshot_only"])


class TestRolloutActionMetrics(unittest.TestCase):
    def test_rollout_metrics_expose_steering_and_mutually_exclusive_pedals(self):
        from training.rollout import RolloutStorage
        from training.train_policy import rollout_action_metrics

        storage = RolloutStorage()
        for action in (
            torch.tensor([-0.2, 0.01, 0.0]),
            torch.tensor([0.4, 0.0, 0.3]),
            torch.tensor([0.1, 0.02, 0.0]),
        ):
            storage.add(
                observation=torch.zeros(4, 84, 84),
                action=action,
                pretransform_action=torch.zeros(2),
                log_probability=0.0,
                value=0.0,
                next_value=0.0,
                reward=0.0,
                terminated=False,
                truncated=False,
                auxiliary_targets=torch.zeros(10),
            )

        metrics = rollout_action_metrics(storage)

        self.assertAlmostEqual(metrics["brake_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["gas_mean"], 0.01)
        self.assertAlmostEqual(metrics["gas_max"], 0.02)
        self.assertEqual(metrics["pedal_overlap_fraction"], 0.0)
