import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


class TestTeacherDemonstrations(unittest.TestCase):
    def test_only_training_episodes_are_collected_and_actions_map_to_actor_coordinates(self):
        from haic_agent.networks import VisualActorCritic
        from training.env_factory import split_track_seeds
        from training.imitation import collect_teacher_demonstrations

        split = split_track_seeds(
            train=((1, 10), (2, 20)), tune=((3, 30),), held_out=((4, 40),)
        )
        created_episodes = []
        environments = []

        class _Environment:
            def __init__(self):
                self.steps = 0
                self.closed = False

            def reset(self):
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step_transition(self, action):
                self.steps += 1
                return SimpleNamespace(
                    next_observation=np.full(
                        (4, 84, 84), self.steps / 10.0, dtype=np.float32
                    ),
                    terminated=self.steps >= 2,
                    truncated=False,
                )

            def close(self):
                self.closed = True

        class _Teacher:
            def reset(self, observation=None):
                del observation

            def act(self, observation):
                del observation
                return np.asarray([0.2, 0.04, 0.0], dtype=np.float32)

        def create_environment(episode, *, max_decisions, render_mode=None):
            del max_decisions, render_mode
            created_episodes.append(episode)
            environment = _Environment()
            environments.append(environment)
            return environment

        with patch(
            "training.imitation.create_episode_environment",
            side_effect=create_environment,
        ):
            demonstrations = collect_teacher_demonstrations(
                split, max_decisions=5, teacher_factory=_Teacher
            )

        self.assertEqual(created_episodes, list(split.train))
        self.assertEqual(demonstrations.episodes, 2)
        self.assertEqual(demonstrations.steps, 4)
        self.assertEqual(demonstrations.observations.shape, (4, 4, 84, 84))
        self.assertEqual(demonstrations.observations.dtype, torch.uint8)
        self.assertEqual(demonstrations.pretransform_actions.shape, (4, 2))
        self.assertTrue(all(environment.closed for environment in environments))
        torch.testing.assert_close(
            demonstrations.observations[:, 0, 0, 0],
            torch.tensor([0, 26, 0, 26], dtype=torch.uint8),
        )
        expected = torch.tensor([0.2, 0.005, 0.0]).repeat(4, 1)
        decoded = VisualActorCritic._bound_actions(
            demonstrations.pretransform_actions
        )
        torch.testing.assert_close(decoded, expected, atol=1e-6, rtol=0.0)

    def test_teacher_pedals_are_rescaled_before_labels_and_environment_steps(self):
        from haic_agent.networks import VisualActorCritic
        from training.env_factory import split_track_seeds
        from training.imitation import collect_teacher_demonstrations

        split = split_track_seeds(train=((1, 10),), tune=((2, 20),), held_out=((3, 30),))
        executed_actions = []

        class _Environment:
            def reset(self):
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step_transition(self, action):
                executed_actions.append(action.copy())
                done = len(executed_actions) == 3
                return SimpleNamespace(
                    next_observation=np.zeros((4, 84, 84), dtype=np.float32),
                    terminated=done,
                    truncated=False,
                )

            def close(self):
                pass

        class _Teacher:
            def reset(self, observation=None):
                del observation
                self.actions = iter(
                    ((0.2, 0.12, 0.0), (-0.3, 0.0, 0.28), (-0.2, 0.06, 0.0))
                )

            def act(self, observation):
                del observation
                return np.asarray(next(self.actions), dtype=np.float32)

        with patch("training.imitation.create_episode_environment", return_value=_Environment()):
            demonstrations = collect_teacher_demonstrations(
                split, max_decisions=5, teacher_factory=_Teacher
            )

        expected_policy_actions = np.asarray(
            ((0.2, 0.015, 0.0), (-0.3, 0.0, 0.0225), (-0.2, 0.0075, 0.0)),
            dtype=np.float32,
        )
        np.testing.assert_allclose(executed_actions, expected_policy_actions, atol=1e-6)
        decoded_labels = VisualActorCritic._bound_actions(demonstrations.pretransform_actions)
        torch.testing.assert_close(
            decoded_labels, torch.from_numpy(expected_policy_actions), atol=1e-6, rtol=0.0
        )

    def test_empty_training_split_is_rejected(self):
        from training.env_factory import split_track_seeds
        from training.imitation import collect_teacher_demonstrations

        split = split_track_seeds(train=(), tune=((3, 30),), held_out=((4, 40),))

        with self.assertRaisesRegex(ValueError, "at least one train episode"):
            collect_teacher_demonstrations(split, max_decisions=5)

    def test_dagger_labels_learner_visited_states_but_collects_only_from_train(self):
        from haic_agent.networks import VisualActorCritic
        from training.env_factory import split_track_seeds
        from training.imitation import collect_dagger_demonstrations

        split = split_track_seeds(
            train=((1, 10),), tune=((3, 30),), held_out=((4, 40),)
        )
        created_episodes = []
        executed_actions = []

        class _Environment:
            def __init__(self):
                self.steps = 0

            def reset(self):
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step_transition(self, action):
                executed_actions.append(action.copy())
                self.steps += 1
                return SimpleNamespace(
                    next_observation=np.full(
                        (4, 84, 84), self.steps / 10.0, dtype=np.float32
                    ),
                    terminated=self.steps >= 2,
                    truncated=False,
                )

            def close(self):
                pass

        class _Teacher:
            def reset(self, observation=None):
                del observation

            def act(self, observation):
                del observation
                return np.asarray([0.2, 0.04, 0.0], dtype=np.float32)

        def create_environment(episode, *, max_decisions, render_mode=None):
            del max_decisions, render_mode
            created_episodes.append(episode)
            return _Environment()

        torch.manual_seed(19)
        torch.set_num_threads(1)
        policy = VisualActorCritic()
        with torch.no_grad():
            policy.policy_mean.weight.zero_()
            policy.policy_mean.bias[0] = torch.atanh(torch.tensor(-0.5))
            policy.policy_mean.bias[1] = torch.atanh(torch.tensor(0.4))
        expected_policy_action = policy.deterministic_actions(
            policy(torch.zeros(1, 4, 84, 84))
        )[0]

        with patch(
            "training.imitation.create_episode_environment",
            side_effect=create_environment,
        ):
            demonstrations = collect_dagger_demonstrations(
                policy,
                split,
                max_decisions=5,
                teacher_action_probability=0.0,
                seed=7,
                teacher_factory=_Teacher,
            )

        self.assertEqual(created_episodes, list(split.train))
        self.assertEqual(demonstrations.collection_method, "dagger")
        self.assertEqual(demonstrations.teacher_action_fraction, 0.0)
        self.assertEqual(demonstrations.steps, 2)
        self.assertEqual(len(executed_actions), 2)
        for action in executed_actions:
            torch.testing.assert_close(torch.from_numpy(action), expected_policy_action)
        decoded_labels = VisualActorCritic._bound_actions(
            demonstrations.pretransform_actions
        )
        torch.testing.assert_close(
            decoded_labels,
            torch.tensor([[0.2, 0.005, 0.0]]).repeat(2, 1),
            atol=1e-6,
            rtol=0.0,
        )

    def test_demonstrations_can_be_aggregated_without_losing_train_only_labels(self):
        from training.imitation import TeacherDemonstrations, combine_demonstrations

        first = TeacherDemonstrations(
            observations=torch.zeros(2, 4, 84, 84, dtype=torch.uint8),
            pretransform_actions=torch.zeros(2, 2),
            episodes=1,
            steps=2,
        )
        second = TeacherDemonstrations(
            observations=torch.ones(3, 4, 84, 84, dtype=torch.uint8),
            pretransform_actions=torch.ones(3, 2),
            episodes=2,
            steps=3,
            collection_method="dagger",
            teacher_action_fraction=0.5,
        )

        combined = combine_demonstrations((first, second))

        self.assertEqual(combined.observations.shape, (5, 4, 84, 84))
        self.assertEqual(combined.observations.dtype, torch.uint8)
        self.assertEqual(combined.steps, 5)
        self.assertEqual(combined.episodes, 3)
        self.assertEqual(combined.collection_method, "aggregated")
        self.assertEqual(combined.teacher_action_fraction, 0.7)


class TestBehavioralCloningWarmup(unittest.TestCase):
    def test_actor_warmup_reduces_teacher_action_mse_with_finite_metrics(self):
        from haic_agent.networks import VisualActorCritic
        from training.imitation import TeacherDemonstrations, behavioral_cloning_warmup

        torch.manual_seed(12)
        torch.set_num_threads(1)
        model = VisualActorCritic()
        demonstrations = TeacherDemonstrations(
            observations=torch.zeros(4, 4, 84, 84),
            pretransform_actions=torch.tensor([[-0.4, 0.25]]).repeat(4, 1),
            episodes=1,
            steps=4,
        )

        metrics = behavioral_cloning_warmup(
            model,
            demonstrations,
            epochs=8,
            batch_size=4,
            learning_rate=0.01,
        )

        self.assertEqual(metrics["epochs"], 8)
        self.assertEqual(metrics["demonstration_steps"], 4)
        self.assertEqual(metrics["observation_storage_dtype"], "torch.float32")
        self.assertTrue(np.isfinite(metrics["initial_action_mse"]))
        self.assertTrue(np.isfinite(metrics["final_action_mse"]))
        self.assertEqual(metrics["teacher_policy_cap_fraction"], 0.75)
        self.assertLess(metrics["final_action_mse"], metrics["initial_action_mse"])

    def test_zero_warmup_epochs_are_rejected_by_the_training_helper(self):
        from haic_agent.networks import VisualActorCritic
        from training.imitation import TeacherDemonstrations, behavioral_cloning_warmup

        demonstrations = TeacherDemonstrations(
            observations=torch.zeros(1, 4, 84, 84),
            pretransform_actions=torch.zeros(1, 2),
            episodes=1,
            steps=1,
        )

        with self.assertRaisesRegex(ValueError, "epochs must be positive"):
            behavioral_cloning_warmup(
                VisualActorCritic(), demonstrations, epochs=0, batch_size=1
            )


if __name__ == "__main__":
    unittest.main()
