import unittest

import numpy as np


def _observation(
    *, obstacle_x=None, obstacle_y=52, curve=0.0, speed=0.0, road_half_width=11
):
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    for y in range(20, 63):
        center = 42.0 + curve * (54 - y)
        left = int(round(center - road_half_width))
        right = int(round(center + road_half_width))
        frame[y, left:right] = 0.4
    if obstacle_x is not None:
        frame[obstacle_y : obstacle_y + 4, obstacle_x : obstacle_x + 3] = 0.68
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.tile(frame[None, :, :], (4, 1, 1))


class TestVisionCorridorAgent(unittest.TestCase):
    def test_training_pipeline_uses_the_speed_aware_corridor_teacher(self):
        from training.vision_teacher import VisionCorridorAgent

        action = VisionCorridorAgent().act(_observation())

        self.assertAlmostEqual(float(action[1]), 0.12, places=7)
        self.assertEqual(float(action[2]), 0.0)

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

    def test_detects_obstacle_inside_wide_road_outside_centerline_window(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        controller = VisionCorridorAgent()
        controller.act(
            _observation(obstacle_x=25, obstacle_y=38, road_half_width=20)
        )

        diagnostics = controller.last_step_diagnostics()
        self.assertIsNotNone(diagnostics["obstacle_y"])
        self.assertEqual(controller.diagnostics()["obstacle_encounters"], 1)

    def test_obstacle_side_is_rechecked_while_distant_then_latched_nearby(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        controller = VisionCorridorAgent()
        controller._obstacle_side = 1.0
        far_obstacle = controller.act(_observation(obstacle_x=46, obstacle_y=30))
        close_shifted_obstacle = controller.act(
            _observation(obstacle_x=35, obstacle_y=52)
        )

        self.assertLess(float(far_obstacle[0]), -0.1)
        self.assertLess(float(controller.last_step_diagnostics()["obstacle_steer_side"]), 0.0)
        self.assertLess(float(close_shifted_obstacle[0]), 0.0)

    def test_obstacle_side_predicts_a_perspective_crossing_before_close_pass(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        controller = VisionCorridorAgent()
        controller.act(
            _observation(obstacle_x=22, obstacle_y=37, curve=-0.9)
        )
        controller.act(
            _observation(obstacle_x=27, obstacle_y=37, curve=-0.9)
        )

        self.assertLess(
            float(controller.last_step_diagnostics()["obstacle_steer_side"]), 0.0
        )

    def test_hud_speed_estimate_tracks_the_rendered_speed_bar(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        frame = _observation(speed=48.0)[-1]

        self.assertAlmostEqual(VisionCorridorAgent._estimate_speed(frame), 48.0, delta=0.1)

    def test_fast_speed_is_braked_before_a_visible_obstacle(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        action = VisionCorridorAgent().act(_observation(obstacle_x=35, speed=58.0))

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)

    def test_controller_profiles_raise_straight_road_throttle_and_report_obstacle_reads(self):
        from haic_agent.corridor_agent import VisionCorridorAgent

        safe = VisionCorridorAgent()
        race = VisionCorridorAgent(cruise_speed=76.0, curve_speed_penalty=1.5, max_gas=0.24)
        safe_action = safe.act(_observation())
        race_action = race.act(_observation())
        race.act(_observation(obstacle_x=35))

        self.assertAlmostEqual(float(safe_action[1]), 0.12, places=7)
        self.assertAlmostEqual(float(race_action[1]), 0.24, places=7)
        diagnostics = race.diagnostics()
        self.assertEqual(diagnostics["obstacle_encounters"], 1)
        self.assertEqual(diagnostics["obstacle_detection_frames"], 1)
        self.assertEqual(diagnostics["gas_frames"], 2)

    def test_missing_checkpoint_uses_visual_actor_in_development_without_corridor_fallback(self):
        from agent import Agent
        from haic_agent.networks import VisualActorCritic

        agent = Agent(
            policy_checkpoint="missing-policy.pt",
            planner_enabled=False,
            strict_checkpoint_loading=False,
        )

        self.assertIsInstance(agent.policy, VisualActorCritic)
        self.assertFalse(hasattr(agent, "corridor_controller"))
        action = agent.act(np.zeros((4, 84, 84), dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_strict_runtime_rejects_a_missing_trained_policy_checkpoint(self):
        from agent import Agent

        with self.assertRaisesRegex(RuntimeError, "failed to load required policy checkpoint"):
            Agent(
                policy_checkpoint="missing-policy.pt",
                planner_enabled=False,
                strict_checkpoint_loading=True,
            )

    def test_runtime_api_does_not_accept_corridor_controller_mode(self):
        from agent import Agent

        with self.assertRaises(TypeError):
            Agent(controller_mode="corridor", policy_checkpoint="missing-policy.pt")


if __name__ == "__main__":
    unittest.main()
