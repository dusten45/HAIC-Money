import time
import unittest

import gymnasium as gym
import numpy as np

from damage import CollisionDamage
from env_wrapper import CarEnvironment, image_preprocessing
from local_runner import safe_act, safe_reset
from core.finish_line import FinishLineTracker


class TestAgentContract(unittest.TestCase):
    def test_safe_act_accepts_and_clips_valid_action(self):
        class Agent:
            def act(self, observation):
                return [2.0, -1.0, 0.5]

        action, valid = safe_act(Agent(), None)

        self.assertTrue(valid)
        np.testing.assert_array_equal(action, [1.0, 0.0, 0.5])

    def test_safe_act_replaces_invalid_results(self):
        agents = [
            type("MissingAct", (), {})(),
            type("BadShape", (), {"act": lambda self, observation: [0.0, 1.0]})(),
            type("NotFinite", (), {"act": lambda self, observation: [0.0, np.nan, 0.0]})(),
        ]

        for agent in agents:
            with self.subTest(agent=type(agent).__name__):
                action, valid = safe_act(agent, None)
                self.assertFalse(valid)
                np.testing.assert_array_equal(action, [0.0, 0.0, 0.0])

    def test_safe_act_times_out(self):
        class SlowAgent:
            def act(self, observation):
                time.sleep(0.1)
                return [0.0, 0.0, 0.0]

        action, valid = safe_act(SlowAgent(), None, timeout_sec=0.01)

        self.assertFalse(valid)
        np.testing.assert_array_equal(action, [0.0, 0.0, 0.0])

    def test_safe_reset_is_optional_and_propagates_errors(self):
        safe_reset(object(), None)

        class BrokenReset:
            def reset(self, observation):
                raise RuntimeError("reset failed")

        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            safe_reset(BrokenReset(), None)


class _DummyCar:
    def set_damage_effects(self, grip, engine, steering):
        self.effects = (grip, engine, steering)


class _DummyEnvironment(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(96, 96, 3),
            dtype=np.uint8,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.car = _DummyCar()
        self.track = [None] * 10
        self.tile_visited_count = 0
        self.raw_steps = 0
        self.finish_on_raw_step = None
        self.finish_qualified_time_s = None
        self.finish_time_s = None

    def reset(self, *, seed=None, options=None):
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action):
        self.raw_steps += 1
        finished = self.raw_steps == self.finish_on_raw_step
        if finished:
            self.finish_qualified_time_s = 0.02
            self.finish_time_s = 0.04
        return np.full((96, 96, 3), 255, dtype=np.uint8), 0.0, False, finished, {
            "collision": False,
            "finished": finished,
            "finish_time_s": self.finish_time_s,
        }


class TestEnvironmentContract(unittest.TestCase):
    def test_preprocessing_returns_float32_unit_range(self):
        image = np.full((96, 96, 3), 255, dtype=np.uint8)

        observation = image_preprocessing(image)

        self.assertEqual(observation.shape, (84, 84))
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(float(observation.min()), 1.0)
        self.assertEqual(float(observation.max()), 1.0)

    def test_wrapper_observation_matches_declared_space(self):
        environment = CarEnvironment(_DummyEnvironment(), no_operation=0)

        observation, _info = environment.reset()

        self.assertEqual(environment.observation_space.shape, (4, 84, 84))
        self.assertEqual(environment.observation_space.dtype, np.float32)
        self.assertTrue(environment.observation_space.contains(observation))

    def test_fifth_collision_retires(self):
        damage = CollisionDamage()

        for _ in range(4):
            self.assertFalse(damage.update(True))
        self.assertTrue(damage.update(True))
        self.assertEqual(damage.damage, 1.0)

    def test_frame_skip_stops_on_second_raw_finish_tick(self):
        raw_environment = _DummyEnvironment()
        raw_environment.finish_on_raw_step = 2
        environment = CarEnvironment(raw_environment, no_operation=0, skip_frames=4)
        environment.reset()

        _, _, terminated, truncated, info = environment.step([0.0, 0.0, 0.0])

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(raw_environment.raw_steps, 2)
        self.assertEqual(info["finish_time_s"], 0.04)
        self.assertTrue(info["finished"])


class TestFinishLineTracker(unittest.TestCase):
    def tracker(self):
        tracker = FinishLineTracker((0.0, 0.0), (1.0, 0.0), 2.0, 0.5)
        tracker.update((-1.0, 0.0), (1.0, 0.0), 0.0, 0.02)
        return tracker

    def cross(self, tracker, progress=0.95, velocity=(1.0, 0.0)):
        tracker.update((-0.4, 0.0), velocity, progress, 0.20)
        tracker.update((0.1, 0.0), velocity, progress, 0.23)
        return tracker.update((0.6, 0.0), velocity, progress, 0.28)

    def test_below_qualification_does_not_finish(self):
        tracker = self.tracker()
        self.assertFalse(self.cross(tracker, 0.94))
        self.assertIsNone(tracker.finish_time_s)

    def test_qualification_without_crossing_does_not_finish(self):
        tracker = self.tracker()
        tracker.update((-1.0, 0.0), (1.0, 0.0), 0.95, 0.20)
        self.assertIsNotNone(tracker.qualified_time_s)
        self.assertIsNone(tracker.finish_time_s)

    def test_forward_crossing_finishes_at_center_tick(self):
        tracker = self.tracker()
        self.assertTrue(self.cross(tracker))
        self.assertEqual(tracker.qualified_time_s, 0.20)
        self.assertEqual(tracker.finish_time_s, 0.23)

    def test_reverse_crossing_does_not_finish(self):
        tracker = self.tracker()
        self.assertFalse(self.cross(tracker, velocity=(-1.0, 0.0)))
        self.assertIsNone(tracker.finish_time_s)

    def test_initial_contact_does_not_finish(self):
        tracker = FinishLineTracker((0.0, 0.0), (1.0, 0.0), 2.0, 0.5)
        tracker.update((0.1, 0.0), (1.0, 0.0), 0.95, 0.02)
        tracker.update((0.6, 0.0), (1.0, 0.0), 0.95, 0.04)
        self.assertIsNone(tracker.finish_time_s)

    def test_lateral_departure_and_stationary_crossing_do_not_finish(self):
        lateral = self.tracker()
        lateral.update((-0.4, 0.0), (1.0, 5.0), 0.95, 0.20)
        lateral.update((0.1, 0.0), (1.0, 5.0), 0.95, 0.23)
        lateral.update((0.6, 3.0), (1.0, 5.0), 0.95, 0.28)
        self.assertIsNone(lateral.finish_time_s)

        stationary = self.tracker()
        self.assertFalse(self.cross(stationary, velocity=(0.0, 0.0)))
        self.assertIsNone(stationary.finish_time_s)

    def test_finish_time_is_immutable(self):
        tracker = self.tracker()
        self.assertTrue(self.cross(tracker))
        tracker.update((1.0, 0.0), (1.0, 0.0), 1.0, 1.0)
        self.assertEqual(tracker.finish_time_s, 0.23)


if __name__ == "__main__":
    unittest.main()
