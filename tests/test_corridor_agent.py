import unittest

import numpy as np


def _observation(*, obstacle_x=None, curve=0.0, speed=0.0):
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    for y in range(20, 63):
        center = 42.0 + curve * (54 - y)
        left = int(round(center - 11))
        right = int(round(center + 11))
        frame[y, left:right] = 0.4
    if obstacle_x is not None:
        frame[37:41, obstacle_x : obstacle_x + 3] = 0.68
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.tile(frame[None, :, :], (4, 1, 1))


class TestVisionCorridorAgent(unittest.TestCase):
    def test_corridor_controller_accelerates_on_a_straight_road(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        action = VisionCorridorAgent().act(_observation())

        self.assertLess(abs(float(action[0])), 0.02)
        self.assertAlmostEqual(float(action[1]), 0.12, places=7)
        self.assertEqual(float(action[2]), 0.0)

    def test_corridor_controller_turns_with_the_visible_centerline(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        action = VisionCorridorAgent().act(_observation(curve=-0.25, speed=55.0))

        self.assertLess(action[0], -0.05)
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(action[2], 0.0)

    def test_obstacle_causes_steering_away_and_latches_the_selected_side(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        obstacle_left = VisionCorridorAgent().act(_observation(obstacle_x=35))
        obstacle_right = VisionCorridorAgent().act(_observation(obstacle_x=46))

        self.assertGreater(obstacle_left[0], 0.1)
        self.assertLess(obstacle_right[0], -0.1)
        for action in (obstacle_left, obstacle_right):
            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.all(np.isfinite(action)))
            self.assertEqual(float(action[2]), 0.0)

        controller = VisionCorridorAgent()
        controller.act(_observation(obstacle_x=35))
        shifted_obstacle = controller.act(_observation(obstacle_x=46))
        self.assertGreater(shifted_obstacle[0], 0.0)

    def test_hud_speed_estimate_tracks_the_rendered_speed_bar(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        frame = _observation(speed=48.0)[-1]

        self.assertAlmostEqual(VisionCorridorAgent._estimate_speed(frame), 48.0, delta=0.1)

    def test_fast_speed_is_braked_before_a_visible_obstacle(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        action = VisionCorridorAgent().act(_observation(obstacle_x=35, speed=58.0))

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)

    def test_missing_checkpoint_uses_the_deterministic_corridor_controller(self):
        from agent import Agent

        agent = Agent(
            policy_checkpoint="missing-policy.pt",
            planner_enabled=False,
            strict_checkpoint_loading=False,
        )

        self.assertEqual(agent.controller_mode, "corridor")
        self.assertIsNotNone(agent.corridor_controller)
        self.assertIsNone(agent.policy)
        self.assertFalse(agent.planner_enabled)

    def test_strict_runtime_rejects_a_missing_trained_policy_checkpoint(self):
        from agent import Agent

        with self.assertRaisesRegex(RuntimeError, "failed to load required policy checkpoint"):
            Agent(
                policy_checkpoint="missing-policy.pt",
                planner_enabled=False,
                strict_checkpoint_loading=True,
            )

    def test_explicit_corridor_mode_skips_checkpoint_loading(self):
        from agent import Agent

        agent = Agent(
            controller_mode="corridor",
            policy_checkpoint="missing-policy.pt",
            strict_checkpoint_loading=True,
        )

        self.assertEqual(agent.controller_mode, "corridor")
        self.assertFalse(agent.planner_enabled)
        self.assertTrue(np.all(np.isfinite(agent.act(_observation()))))


if __name__ == "__main__":
    unittest.main()
