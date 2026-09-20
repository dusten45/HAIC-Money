import unittest

import gymnasium as gym
import numpy as np


class _Joint:
    def __init__(self, angle):
        self.angle = angle


class _Wheel:
    def __init__(self, index):
        self.omega = float(index + 1)
        self.joint = _Joint(0.1 * (index + 1))


class _Hull:
    linearVelocity = (3.0, 4.0)
    angularVelocity = 0.25


class _Car:
    def __init__(self):
        self.hull = _Hull()
        self.wheels = [_Wheel(index) for index in range(4)]

    def set_damage_effects(self, grip, engine, steering):
        self.damage_effects = (grip, engine, steering)


class _RawEnvironment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(0, 255, (96, 96, 3), np.uint8)
        self.car = _Car()
        self.track = [None] * 100
        self.tile_visited_count = 0
        self.raw_steps = 0
        self.actions = []
        self.finish_time_s = None
        self.finish_qualified_time_s = None

    def _image(self):
        return np.full((96, 96, 3), self.raw_steps, dtype=np.uint8)

    def reset(self, *, seed=None, options=None):
        self.raw_steps = 0
        self.actions.clear()
        return self._image(), {"reset_seed": seed, "track_id": options["track_id"]}

    def step(self, action):
        self.raw_steps += 1
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.tile_visited_count = self.raw_steps
        self.car.hull.linearVelocity = (float(self.raw_steps), 0.0)
        self.car.hull.angularVelocity = float(self.raw_steps) / 10.0
        for index, wheel in enumerate(self.car.wheels):
            wheel.omega = float(self.raw_steps + index)
            wheel.joint.angle = float(self.raw_steps + index) / 100.0
        return self._image(), float(self.raw_steps), False, False, {"collision": False}


class TestTrainingCollector(unittest.TestCase):
    def test_transition_holds_one_action_for_four_ticks_after_fifty_tick_warmup(self):
        # Break caught: changing the collector cadence or emitting labels from an
        # intermediate raw tick disconnects a training transition from evaluation.
        from training.env_factory import make_collecting_environment

        raw_environment = _RawEnvironment()
        environment = make_collecting_environment(raw_environment)

        observation, _ = environment.reset(seed=17, options={"track_id": 3})
        transition = environment.step_transition(np.array([0.25, 0.5, 0.0]))

        self.assertEqual(raw_environment.raw_steps, 54)
        self.assertEqual(observation.shape, (4, 84, 84))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(observation == observation[-1]))
        self.assertGreaterEqual(float(observation.min()), 0.0)
        self.assertLessEqual(float(observation.max()), 1.0)
        self.assertEqual(len(raw_environment.actions), 54)
        for action in raw_environment.actions[-4:]:
            np.testing.assert_array_equal(action, [0.25, 0.5, 0.0])
        self.assertEqual(transition.reward, 210.0)
        self.assertEqual(transition.labels.speed, 54.0)
        self.assertEqual(transition.labels.tile_progress, 0.54)
        self.assertEqual(transition.label_timing, "next_decision")
        self.assertTrue(np.all(transition.hud_features.full_frame == 54.0 / 255.0))

    def test_transition_labels_are_read_after_the_fourth_action_tick(self):
        # Break caught: reading labels before the action or after an
        # intermediate raw tick misaligns auxiliary targets with next_observation.
        from training.env_factory import make_collecting_environment

        raw_environment = _RawEnvironment()
        environment = make_collecting_environment(raw_environment)
        environment.reset(seed=17, options={"track_id": 3})

        transition = environment.step_transition(np.array([0.25, 0.5, 0.0]))

        self.assertEqual(raw_environment.raw_steps, 54)
        self.assertEqual(transition.labels.speed, 54.0)
        self.assertEqual(transition.labels.wheel_omega, (54.0, 55.0, 56.0, 57.0))
        self.assertAlmostEqual(transition.labels.steering_angle, 0.545)
        self.assertEqual(transition.labels.yaw_rate, 5.4)
        self.assertEqual(transition.labels.tile_progress, 0.54)

    def test_split_rejects_an_episode_present_in_training_and_holdout(self):
        # Break caught: a future edit permits a held-out track/seed pair to
        # leak into training and makes validation metrics optimistic.
        from training.env_factory import split_track_seeds

        with self.assertRaisesRegex(ValueError, "train and held_out"):
            split_track_seeds(train=[(1, 7)], tune=[(2, 8)], held_out=[(1, 7)])

    def test_split_preserves_the_explicit_deterministic_episode_order(self):
        from training.env_factory import split_track_seeds

        split = split_track_seeds(
            train=[(3, 20), (1, 10)], tune=[(2, 30)], held_out=[(4, 40)]
        )

        self.assertEqual(split.train, ((3, 20), (1, 10)))
        self.assertEqual(split.tune, ((2, 30),))
        self.assertEqual(split.held_out, ((4, 40),))


if __name__ == "__main__":
    unittest.main()
