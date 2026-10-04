"""Synthetic behavioral tests; no simulator or track reset."""
import unittest
import numpy as np
from agents.apex_2026.line_agent import Agent


def scene(bend=0.0, obstacle=False):
    f = np.full((84, 84), .65, dtype=np.float32)
    for y in range(74):
        center = 42 + bend * max(58-y, 0)**2 / 150
        lo, hi = max(0, int(center-12)), min(84, int(center+13))
        f[y, lo:hi] = .4
    f[74:] = 0
    if obstacle:
        f[30:36, 43:48] = .95
    return np.repeat(f[None], 4, axis=0)


class LineAgentTests(unittest.TestCase):
    def test_straight_road_accelerates_without_turning(self):
        a = Agent()
        action = a.act(scene())
        self.assertLess(abs(action[0]), .08)
        self.assertGreater(action[1], .5)
        self.assertEqual(action.dtype, np.float32)

    def test_mirrored_bends_produce_opposite_steering(self):
        right = scene(bend=3.)
        left = np.roll(right[:, :, ::-1], 1, axis=2).copy()
        ar, al = Agent().act(right), Agent().act(left)
        self.assertGreater(ar[0], .05)
        self.assertLess(al[0], -.05)
        self.assertLess(abs(ar[0]+al[0]), .15)

    def test_planned_path_goes_around_obstacle(self):
        a = Agent()
        a.act(scene(obstacle=True))
        path = np.asarray(a.diagnostics['path'])
        crossing = path[(path[:, 1] >= 29) & (path[:, 1] <= 37), 0]
        self.assertGreater(len(crossing), 0)
        self.assertTrue(np.all(crossing < 41) or np.all(crossing > 50), crossing)

    def test_near_obstacle_changes_tracking_action_not_only_planned_path(self):
        obs = scene(bend=1.5)
        obs[:, 47:53, 43:48] = .95
        agent = Agent({'lookahead': 26, 'pursuit_gain': 3.5})
        action = agent.act(obs)
        self.assertLess(action[0], -.04)

    def test_curve_schedule_slower_than_straight(self):
        straight, curved = Agent(), Agent()
        straight.act(scene())
        curved.act(scene(bend=1.7))
        self.assertLess(curved.diagnostics['target_speed'], straight.diagnostics['target_speed'])

    def test_overspeed_brakes_and_never_accelerates_simultaneously(self):
        obs = scene(bend=1.7)
        obs[:, 77:83, 10:13] = .9
        action = Agent().act(obs)
        self.assertEqual(action[1], 0.)
        self.assertGreater(action[2], 0.)

    def test_motion_preview_counteracts_existing_turn_and_sideslip(self):
        a = Agent({'motion_preview': .12})
        straight = a._preview_curvature(.02, 50., 0., 0., 30.)
        moving_right = a._preview_curvature(.02, 50., .8, .1, 30.)
        self.assertLess(moving_right, straight)
        self.assertAlmostEqual(a._preview_curvature(-.02, 50., -.8, -.1, 30.), -moving_right)

    def test_visual_motion_recovers_known_camera_rotation(self):
        import cv2
        rng = np.random.default_rng(4)
        previous = cv2.GaussianBlur(rng.random((84,84)).astype(np.float32), (3,3), 0)
        scale = np.diag([1.3608, 1.701])
        angle = -.04
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        matrix = scale @ rotation @ np.linalg.inv(scale)
        center = np.array([42.,63.])
        transform = np.column_stack((matrix, center-matrix@center))
        current = cv2.warpAffine(previous, transform, (84,84))
        yaw, slip, valid = Agent()._motion(previous, current)
        self.assertTrue(valid)
        self.assertAlmostEqual(yaw, .5, delta=.12)

    def test_traction_budget_coasts_when_turning_at_speed(self):
        a = Agent({'traction_accel': 180.})
        self.assertAlmostEqual(a._traction_gas_limit(60., 0.), 1.)
        self.assertLessEqual(a._traction_gas_limit(60., .1), .5)
        self.assertGreater(a._traction_gas_limit(60., .1), 0.)
        self.assertEqual(a._traction_gas_limit(80., .1), 0.)
        self.assertLessEqual(a._traction_gas_limit(40., .2), .5)
        self.assertGreater(a._traction_gas_limit(10., .1), .9)
        self.assertEqual(a._traction_gas_limit(60., -.1), a._traction_gas_limit(60., .1))

    def test_yaw_observer_survives_missing_texture_and_reset(self):
        a = Agent({'motion_observer': True})
        a._observe_motion(40., .7, .1, True)
        yaw, slip = a._observe_motion(40., 0., 0., False)
        self.assertGreater(yaw, 0.)
        self.assertGreater(slip, 0.)
        a.reset()
        self.assertEqual(a._observe_motion(40., 0., 0., False), (0., 0.))

    def test_continuous_path_removes_staircase_curvature(self):
        rows = np.arange(58, 7, -2, dtype=np.float32)
        path = np.column_stack((42.+np.round(.2*(58.-rows)), rows))
        a = Agent({'path_smoothing': 200.})
        refined = a._smooth_path(path, np.ones((84,84), np.uint8))
        self.assertLess(np.max(np.abs(np.diff(refined[:,0], n=2))), .15)
        self.assertLess(np.max(np.abs(refined[:,0]-path[:,0])), 1.5)

    def test_continuous_path_keeps_obstacle_bypass_clear(self):
        a = Agent({'path_smoothing': 200., 'lookahead': 26})
        obs = scene(obstacle=True)
        a.act(obs)
        path = np.asarray(a.diagnostics['path'])
        free, _, _ = a._free_space(obs[-1])
        x, y = np.rint(path).astype(int).T
        self.assertTrue(np.all(free[y,x] > 0))
        crossing = path[(path[:,1] >= 29) & (path[:,1] <= 37),0]
        self.assertTrue(np.all(crossing < 41) or np.all(crossing > 50))

    def test_hud_dynamics_decodes_current_rotation_with_policy_sign(self):
        from agents.apex_2026.diagnostics.hud_dynamics_calibration import render_hud
        a = Agent({'hud_dynamics': True})
        for raw_yaw, raw_wheel in [(1.5, -.15), (-2.3, .27), (0., 0.)]:
            yaw, wheel = a._hud_dynamics(render_hud(raw_yaw, raw_wheel, 70., 150.))
            self.assertAlmostEqual(yaw, -raw_yaw, delta=.065)
            self.assertAlmostEqual(wheel, -raw_wheel, delta=.005)

    def test_reset_clears_temporal_state(self):
        a = Agent()
        a.act(scene(bend=1.))
        a.reset(scene())
        np.testing.assert_array_equal(a.act(scene()), Agent().act(scene()))

    def test_invalid_image_fails_closed(self):
        for obs in (np.zeros((2, 3)), np.full((4,84,84), np.nan)):
            np.testing.assert_array_equal(Agent().act(obs), np.array([0., 0., .3], np.float32))


if __name__ == '__main__':
    unittest.main()
