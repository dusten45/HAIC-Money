import unittest

import numpy as np

from common_adapter import ActionAdapter
from haic.algorithms.drq_v2.speed_control import release_light_brake


class DrQSpeedControlTests(unittest.TestCase):
    def test_releases_only_light_brake_on_near_straights(self):
        action = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        modified = release_light_brake(action)
        expected = action.copy()
        expected[2] *= np.float32(0.25)
        np.testing.assert_array_equal(modified, expected)
        self.assertEqual(modified.dtype, np.float32)
        self.assertFalse(np.shares_memory(action, modified))
        np.testing.assert_array_equal(action, np.array([0.1, 0.9, 0.2], dtype=np.float32))

    def test_preserves_intentional_braking_turns_and_low_throttle(self):
        for action in ([0.21, 0.9, 0.2], [0.1, 0.79, 0.2], [0.1, 0.9, 0.31],
                       [0.1, 0.9, 0.0], [-1.0, 0.0, 1.0]):
            with self.subTest(action=action):
                expected = np.asarray(action, dtype=np.float32)
                np.testing.assert_array_equal(release_light_brake(expected), expected)

    def test_accepts_official_pedals_after_exactly_one_mapping(self):
        official = ActionAdapter().to_official(np.array([0.1, 0.8, -0.6], dtype=np.float32))
        modified = release_light_brake(official)
        np.testing.assert_array_equal(modified[:2], official[:2])
        self.assertAlmostEqual(float(modified[2]), float(official[2] * 0.25), places=7)

    def test_rejects_invalid_actions_without_clipping(self):
        for action in ([0.0, 0.5], [0.0, np.nan, 0.2], [np.inf, 0.8, 0.2],
                       [0.0, 1.01, 0.2], [0.0, 0.8, -0.01], [1.01, 0.8, 0.2]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                release_light_brake(action)

    def test_repeat_and_episode_reset_are_stateless(self):
        action = np.array([0.2, 0.8, 0.3], dtype=np.float32)
        first = release_light_brake(action)
        np.testing.assert_array_equal(release_light_brake(action), first)
        self.assertEqual(float(first[2]), float(action[2] * np.float32(0.25)))


if __name__ == "__main__":
    unittest.main()
