"""An early bend cue must supplement, not replace, the stable actor."""

import unittest

import numpy as np

from haic_agent.anticipatory_bend_runtime import AnticipatoryBendAgent


class _Base:
    def reset(self, observation):
        del observation

    def act(self, observation):
        del observation
        return np.asarray([0.0, 0.12, 0.0], dtype=np.float32)


class _ObstacleBase(_Base):
    def __init__(self):
        self.corridor = self

    def act(self, observation):
        del observation
        return np.asarray([0.3, 0.0, 0.08], dtype=np.float32)

    def last_step_diagnostics(self):
        return {"obstacle_y": 44.0}


class _AlreadyTurningBase(_Base):
    def act(self, observation):
        del observation
        return np.asarray([-0.25, 0.12, 0.0], dtype=np.float32)


class _BeginningTurnBase(_Base):
    def act(self, observation):
        del observation
        return np.asarray([-0.08, 0.12, 0.0], dtype=np.float32)


def _observation(far_center, near_center=42):
    pixels = np.zeros((4, 84, 84), dtype=np.float32)
    if far_center is None:
        return pixels
    for index, row in enumerate((54, 50, 46, 42, 38, 34, 30)):
        center = round(near_center + (far_center - near_center) * index / 6)
        pixels[:, row, center - 4:center + 5] = 0.34
    return pixels


class AnticipatoryBendTests(unittest.TestCase):
    def test_visible_left_bend_steers_before_road_loss_without_pedal_change(self):
        agent = AnticipatoryBendAgent(_Base())
        action = agent.act(_observation(30))
        self.assertLess(float(action[0]), -0.05)
        self.assertAlmostEqual(float(action[1]), 0.12, places=6)
        self.assertEqual(float(action[2]), 0.0)

    def test_visible_right_bend_steers_right(self):
        agent = AnticipatoryBendAgent(_Base())
        action = agent.act(_observation(54))
        self.assertGreater(float(action[0]), 0.05)

    def test_straight_or_missing_road_keeps_base_action(self):
        agent = AnticipatoryBendAgent(_Base())
        np.testing.assert_array_equal(agent.act(_observation(42)),
                                      np.asarray([0.0, 0.12, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(agent.act(_observation(None)),
                                      np.asarray([0.0, 0.12, 0.0], dtype=np.float32))

    def test_preview_does_not_override_active_obstacle_avoidance(self):
        agent = AnticipatoryBendAgent(_ObstacleBase())
        np.testing.assert_array_equal(agent.act(_observation(30)),
                                      np.asarray([0.3, 0.0, 0.08], dtype=np.float32))

    def test_preview_activation_is_counted_and_reset(self):
        agent = AnticipatoryBendAgent(_Base())
        agent.act(_observation(42))
        agent.act(_observation(30))
        self.assertEqual(agent.preview_count, 1)
        agent.reset(_observation(42))
        self.assertEqual(agent.preview_count, 0)

    def test_preview_does_not_double_existing_bend_steering(self):
        agent = AnticipatoryBendAgent(_AlreadyTurningBase())
        action = agent.act(_observation(30))
        self.assertAlmostEqual(float(action[0]), -0.25, places=6)
        self.assertAlmostEqual(float(action[1]), 0.12, places=6)

    def test_ambiguous_preview_does_not_reverse_committed_steering(self):
        agent = AnticipatoryBendAgent(_AlreadyTurningBase())
        action = agent.act(_observation(54))
        self.assertAlmostEqual(float(action[0]), -0.25, places=6)

    def test_preview_starts_at_first_far_road_cue(self):
        agent = AnticipatoryBendAgent(_Base())
        self.assertLess(float(agent.act(_observation(36))[0]), -0.05)

    def test_preview_hands_back_when_actor_begins_turning(self):
        agent = AnticipatoryBendAgent(_BeginningTurnBase())
        self.assertAlmostEqual(float(agent.act(_observation(30))[0]), -0.08, places=6)

    def test_preview_waits_when_near_road_is_laterally_displaced(self):
        agent = AnticipatoryBendAgent(_Base())
        np.testing.assert_array_equal(agent.act(_observation(46, near_center=36)),
                                      np.asarray([0.0, 0.12, 0.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
