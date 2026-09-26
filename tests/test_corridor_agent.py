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


def _green_shoulder_observation(*, side="both", near_width=6, speed=48.0):
    """Render a road whose near edge is visibly touching the green shoulder."""
    frame = _observation(speed=speed)[-1].copy()
    left = int(round(42.0 - near_width / 2.0))
    right = left + int(near_width)
    for row in range(48, 63):
        if side == "both":
            frame[row, :] = 0.1
            frame[row, left:right] = 0.4
        elif side == "right":
            frame[row, right:] = 0.1
        elif side == "left":
            frame[row, :left] = 0.1
        else:
            raise ValueError(f"unsupported shoulder side: {side}")
    return np.tile(frame[None, :, :], (4, 1, 1))


def _near_corridor_dropout_observation(*, far_center=53.0, speed=24.0):
    """Render only the distant road, leaving the near camera view green."""
    frame = _observation(speed=speed)[-1].copy()
    frame[48:63, :] = 0.1
    for row in range(20, 48):
        center = int(round(far_center))
        left = max(0, center - 11)
        right = min(frame.shape[1], center + 12)
        frame[row, left:right] = 0.4
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

        # The low-inertia curve gain intentionally makes the first correction
        # smaller than the legacy controller while preserving its direction.
        self.assertLess(action[0], -0.02)
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

    def test_forward_controller_detects_a_wider_projected_obstacle_shape(self):
        from agent import _RacingLineController

        frame = _observation(speed=48.0)[-1].copy()
        # A near obstacle can occupy substantially more pixels than the
        # smallest sprite.  It remains inside the road band but is deliberately
        # wider/taller than the old component filter accepted.
        frame[34:48, 35:50] = 0.68
        observation = np.tile(frame[None, :, :], (4, 1, 1))
        controller = _RacingLineController()
        centers, spans = controller._road_geometry(frame)

        obstacle = controller._nearest_obstacle(frame, centers, spans)
        action = controller.act(observation)

        self.assertIsNotNone(obstacle)
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)

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

    def test_forward_controller_holds_a_straight_road_without_weaving(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        actions = [controller.act(_observation()) for _ in range(8)]

        for action in actions:
            self.assertLess(abs(float(action[0])), 0.02)
            self.assertGreater(float(action[1]), 0.0)
            self.assertEqual(float(action[2]), 0.0)

    def test_forward_controller_limits_direction_reversal_and_resets(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        actions = [
            controller.act(_observation(curve=0.9)),
            controller.act(_observation(curve=-0.9)),
            controller.act(_observation(curve=0.9)),
        ]
        steers = [float(action[0]) for action in actions]
        self.assertTrue(all(abs(steer) <= controller.MAX_STEER for steer in steers))
        self.assertTrue(
            all(
                abs(current - previous) <= controller.MAX_STEER_STEP + 1e-6
                for previous, current in zip(steers, steers[1:])
            )
        )
        self.assertTrue(
            all(previous * current >= -1e-6 for previous, current in zip(steers, steers[1:]))
        )

        controller.reset()
        straight = controller.act(_observation())
        self.assertLess(abs(float(straight[0])), 0.02)

    def test_forward_controller_clips_preview_to_vehicle_safe_track_envelope(self):
        from agent import _ForwardCorridorController

        spans = {54: (50.0, 72.0), 42: (48.0, 70.0)}
        centers = {54: 50.0, 42: 49.0}

        safe_near = _ForwardCorridorController._safe_center_at(54.0, centers, spans)
        outward_left = _ForwardCorridorController._track_bound_steering(-0.2, spans)
        outward_right = _ForwardCorridorController._track_bound_steering(
            0.2, {54: (10.0, 32.0), 50: (10.0, 32.0)}
        )

        self.assertAlmostEqual(safe_near, 57.0, places=5)
        self.assertGreater(outward_left, 0.0)
        self.assertLess(outward_right, 0.0)

    def test_forward_controller_does_not_countersteer_before_a_low_curvature_turn(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        # A small, noisy left-looking bend immediately before a clear right
        # bend must not create a visible countersteer.  That pre-turn wiggle
        # is what starts the drift observed in visual replays; the controller
        # should hold the corridor until the intended turn is unambiguous.
        approach = controller.act(_observation(curve=-0.08, speed=20.0))
        right_turn = controller.act(_observation(curve=0.65, speed=20.0))

        self.assertGreaterEqual(
            float(approach[0]),
            -0.01,
            "a small opposite-looking pre-turn correction must not induce drift",
        )
        self.assertGreater(
            float(right_turn[0]),
            float(approach[0]),
            "the intended turn should begin in the requested direction",
        )
        self.assertTrue(np.all(np.isfinite(approach)))
        self.assertTrue(np.all(np.isfinite(right_turn)))

    def test_forward_controller_does_not_let_noisy_near_edge_flip_turn_direction(self):
        from agent import _ForwardCorridorController

        # The distant road points slightly right of the image center, while a
        # noisy near edge is rendered farther right.  The old heading term
        # dominated this view and commanded a left counter-steer.  The
        # look-ahead center should remain the authoritative direction.
        frame = np.full((84, 84), 0.1, dtype=np.float32)
        for row in range(20, 63):
            center = 50.0 if row >= 50 else 42.0
            left = int(round(center - 11.0))
            right = int(round(center + 11.0))
            frame[row, left:right] = 0.4
        frame[77:83, 10:13] = 0.27 / 18.0

        action = _ForwardCorridorController().act(
            np.tile(frame[None, :, :], (4, 1, 1))
        )

        self.assertGreaterEqual(float(action[0]), 0.0)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_brakes_when_the_corridor_disappears_after_tracking(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        controller.act(_observation())
        recovery = controller.act(np.zeros((4, 84, 84), dtype=np.float32))

        self.assertEqual(float(recovery[1]), 0.0)
        self.assertGreater(float(recovery[2]), 0.0)
        self.assertLessEqual(abs(float(recovery[0])), controller.MAX_STEER)

    def test_forward_controller_uses_distant_road_to_recover_near_dropout(self):
        from agent import _ForwardCorridorController

        # The close road samples can be occluded by the green shoulder while
        # the distant track is still visible.  The recovery must follow that
        # remaining centerline instead of declaring a straight road and
        # driving onward in the previous direction.
        for far_center, expected_sign in ((53.0, 1.0), (30.0, -1.0)):
            controller = _ForwardCorridorController()
            action = controller.act(
                _near_corridor_dropout_observation(far_center=far_center)
            )

            self.assertTrue(controller.road_visible)
            self.assertGreater(
                expected_sign * float(action[0]),
                0.0,
                msg=f"far road at x={far_center} must guide recovery steering",
            )
            self.assertLessEqual(abs(float(action[0])), controller.MAX_STEER)
            self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_wide_search_recovers_a_road_outside_local_window(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        controller.act(_observation())
        frame = np.full((84, 84), 0.1, dtype=np.float32)
        frame[20:63, 70:84] = 0.4
        frame[77:83, 10:13] = 0.27 / 18.0
        shifted = np.tile(frame[None, :, :], (4, 1, 1))

        action = controller.act(shifted)

        self.assertGreater(float(action[0]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.RECOVERY_MAX_STEER)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_recovers_forward_crawl_after_persistent_dropout(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        controller.act(_observation(speed=32.0))
        dropout = np.zeros((4, 84, 84), dtype=np.float32)
        actions = [controller.act(dropout) for _ in range(6)]

        # Losing the visual corridor warrants a short recovery brake, but a
        # persistent camera dropout must still produce a nonzero forward
        # crawl rather than parking the car for the rest of the episode.
        self.assertTrue(
            any(float(action[1]) > 0.0 and float(action[2]) == 0.0 for action in actions[2:]),
            "persistent visual dropout must recover to forward crawl",
        )
        self.assertGreater(float(actions[-1][1]), 0.0)
        self.assertEqual(float(actions[-1][2]), 0.0)
        self.assertTrue(all(np.all(np.isfinite(action)) for action in actions))

    def test_forward_controller_brakes_when_green_shoulder_touches_both_sides(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        action = controller.act(_green_shoulder_observation(near_width=6))

        # A centered car can still be unsafe when the road has narrowed to the
        # point that green is visible immediately beside it.  Treat that view
        # as a hazard, rather than accelerating through the narrow corridor.
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_brakes_before_the_vehicle_envelope_reaches_green(self):
        from agent import _ForwardCorridorController

        frame = _observation(speed=48.0)[-1].copy()
        # Keep the distant road centered, but move the near asphalt edge close
        # to the camera center.  The center pixel is still on asphalt; the
        # enlarged safety margin must nevertheless classify this as green
        # contact risk before throttle is allowed.
        frame[48:63, :] = 0.1
        frame[48:63, 37:59] = 0.4
        action = _ForwardCorridorController().act(
            np.tile(frame[None, :, :], (4, 1, 1))
        )

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)

    def test_forward_controller_keeps_green_contact_braking_priority_for_five_frames(self):
        from agent import _ForwardCorridorController

        frame = _green_shoulder_observation(side="right", near_width=12)
        controller = _ForwardCorridorController()
        actions = [controller.act(frame) for _ in range(7)]

        self.assertTrue(all(float(action[1]) == 0.0 for action in actions[:5]))
        self.assertTrue(all(float(action[2]) > 0.0 for action in actions[:5]))
        self.assertTrue(any(float(action[1]) > 0.0 for action in actions[5:]))

    def test_forward_controller_releases_a_shoulder_brake_after_a_bounded_hold(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        actions = [
            controller.act(_green_shoulder_observation(side="right", near_width=12))
            for _ in range(10)
        ]

        # A visible one-sided shoulder is a warning to steer back toward the
        # asphalt, not a reason to hold the car on the brake forever.  The
        # first frames may brake while the controller makes that correction,
        # but a bounded hold must eventually leave a small forward command.
        self.assertTrue(
            any(float(action[1]) > 0.0 and float(action[2]) == 0.0 for action in actions[3:]),
            "persistent visible corridor hazard must recover to forward motion",
        )
        self.assertTrue(all(np.all(np.isfinite(action)) for action in actions))

    def test_forward_controller_recovers_throttle_after_a_transient_hazard(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        hazard = controller.act(_green_shoulder_observation(near_width=6))
        self.assertEqual(float(hazard[1]), 0.0)
        self.assertGreater(float(hazard[2]), 0.0)

        # Once the road is visible again, the hazard response must not leave
        # a stale low target speed braking the car indefinitely.  This keeps
        # the recovery bounded without prescribing an episode-level score.
        recovery = [
            controller.act(_observation(speed=48.0))
            for _ in range(12)
        ]
        self.assertTrue(
            any(float(action[1]) > 0.0 and float(action[2]) == 0.0 for action in recovery),
            "clear corridor must regain mutually-exclusive forward throttle",
        )
        self.assertGreater(float(recovery[-1][1]), 0.0)
        self.assertEqual(float(recovery[-1][2]), 0.0)

    def test_forward_controller_speed_brake_watchdog_preserves_forward_progress(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        actions = [controller.act(_observation(speed=80.0)) for _ in range(12)]

        # Ordinary speed regulation may brake initially, but a stale/high HUD
        # reading must not hold the car on the brake forever.
        self.assertTrue(
            any(float(action[1]) > 0.0 and float(action[2]) == 0.0 for action in actions[9:]),
            "speed-control braking must eventually hand back a forward command",
        )
        self.assertTrue(all(np.all(np.isfinite(action)) for action in actions))

    def test_forward_controller_steers_away_from_green_shoulder(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        right_shoulder = controller.act(
            _green_shoulder_observation(side="right", near_width=12)
        )
        controller.reset()
        left_shoulder = controller.act(
            _green_shoulder_observation(side="left", near_width=12)
        )

        # Positive steering moves away from a left-side shoulder and negative
        # steering moves away from a right-side shoulder in the track frame.
        self.assertLess(float(right_shoulder[0]), 0.0)
        self.assertGreater(float(left_shoulder[0]), 0.0)
        for action in (right_shoulder, left_shoulder):
            self.assertEqual(float(action[1]), 0.0)
            self.assertGreater(float(action[2]), 0.0)
            self.assertLessEqual(abs(float(action[0])), controller.MAX_STEER)

    def test_forward_controller_brakes_before_a_center_obstacle(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        action = controller.act(_observation(obstacle_x=40, speed=48.0))

        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.MAX_STEER)
        self.assertLessEqual(float(action[2]), controller.MAX_BRAKE)

    def test_forward_controller_cuts_throttle_for_a_distant_obstacle(self):
        from agent import _ForwardCorridorController

        action = _ForwardCorridorController().act(
            _observation(obstacle_x=40, obstacle_y=30, speed=20.0)
        )

        # A distant object may not yet narrow the near corridor, but it is
        # still a collision risk.  Remove gas immediately to buy time for the
        # already bounded escape steering.
        self.assertEqual(float(action[1]), 0.0)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_uses_ttc_like_limit_for_a_close_obstacle(self):
        from agent import _ForwardCorridorController

        # At low speed a close obstacle used to remove gas but could leave no
        # longitudinal margin for the escape steer.  The distance-aware limit
        # must request a small brake before the object reaches the envelope.
        action = _ForwardCorridorController().act(
            _observation(obstacle_x=40, obstacle_y=52, speed=20.0)
        )

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertTrue(np.all(np.isfinite(action)))

    def test_forward_controller_interpolates_obstacle_speed_limits(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        limits = [controller._obstacle_speed_limit(row) for row in (30, 41, 47, 54)]

        self.assertEqual(limits[0], controller.OBSTACLE_FAR_SPEED)
        self.assertGreater(limits[1], limits[2])
        self.assertGreater(limits[2], limits[3])
        self.assertEqual(limits[3], controller.OBSTACLE_CLOSE_SPEED)

    def test_forward_controller_treats_bright_nonroad_intrusion_as_hazard(self):
        from agent import _ForwardCorridorController

        frame = _observation(speed=48.0)[-1].copy()
        # Grayscale preprocessing makes green shoulder and orange obstacles
        # bright; put that same signal inside the visible asphalt corridor.
        frame[48:58, 39:46] = 0.66
        controller = _ForwardCorridorController()
        action = controller.act(np.tile(frame[None, :, :], (4, 1, 1)))

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.MAX_STEER)

    def test_forward_controller_raises_safe_forward_speed_but_brakes_for_hazards(self):
        from agent import _ForwardCorridorController

        straight = _ForwardCorridorController().act(_observation(speed=20.0))
        low_curvature = _ForwardCorridorController().act(
            _observation(curve=0.05, speed=20.0)
        )
        hazard = _ForwardCorridorController().act(
            _observation(curve=0.05, obstacle_x=40, speed=20.0)
        )

        # The previous legacy cap was 0.08 gas, which made clear-road
        # progress unnecessarily slow.  The safe speed increase applies only
        # when the corridor is clear; a visible obstacle still takes priority
        # and receives mutually-exclusive braking.
        self.assertGreater(float(straight[1]), 0.08)
        self.assertGreater(float(low_curvature[1]), 0.08)
        self.assertEqual(float(straight[2]), 0.0)
        self.assertEqual(float(low_curvature[2]), 0.0)
        self.assertEqual(float(hazard[1]), 0.0)
        self.assertGreater(float(hazard[2]), 0.0)
        self.assertTrue(all(np.all(np.isfinite(action)) for action in (straight, low_curvature, hazard)))

    def test_forward_controller_reduces_speed_before_a_previewed_curve(self):
        from agent import _ForwardCorridorController

        straight_controller = _ForwardCorridorController()
        straight = straight_controller.act(_observation(speed=52.0))

        curve_controller = _ForwardCorridorController()
        curve = curve_controller.act(_observation(curve=0.25, speed=52.0))

        # The curve is visible in the look-ahead rows before the near edge
        # becomes unsafe.  Speed planning should therefore lower its target
        # and relinquish throttle before the steering command becomes large.
        self.assertLess(curve_controller._target_speed, straight_controller._target_speed)
        self.assertLessEqual(float(curve[1]), float(straight[1]))
        self.assertEqual(float(curve[1]) * float(curve[2]), 0.0)
        self.assertTrue(np.all(np.isfinite(curve)))

    def test_forward_controller_enters_a_strong_curve_with_a_bounded_lateral_command(self):
        from agent import _ForwardCorridorController

        controller = _ForwardCorridorController()
        action = controller.act(_observation(curve=0.5, speed=52.0))
        high_speed_moderate_curve = _ForwardCorridorController().act(
            _observation(curve=0.25, speed=52.0)
        )
        low_speed_moderate_curve = _ForwardCorridorController().act(
            _observation(curve=0.25, speed=20.0)
        )

        # A strong previewed bend must shed speed before lateral force grows;
        # the curve mode also keeps the first steering command well below the
        # general maximum so the car remains inside the visible road band.
        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.CURVE_MAX_STEER)
        self.assertLess(
            abs(float(high_speed_moderate_curve[0])),
            abs(float(low_speed_moderate_curve[0])),
        )
        self.assertLessEqual(float(action[2]), controller.MAX_BRAKE)

    def test_reset_racing_line_controller_is_fast_on_clear_straights(self):
        from agent import _RacingLineController

        controller = _RacingLineController()
        action = controller.act(_observation(speed=20.0))

        self.assertGreater(float(action[1]), 0.14)
        self.assertEqual(float(action[2]), 0.0)
        self.assertLess(abs(float(action[0])), 0.02)
        self.assertEqual(controller._target_speed, controller.cruise_speed)

    def test_reset_racing_line_controller_brakes_for_close_obstacles(self):
        from agent import _RacingLineController

        controller = _RacingLineController()
        action = controller.act(_observation(obstacle_x=40, obstacle_y=52, speed=48.0))

        self.assertEqual(float(action[1]), 0.0)
        self.assertGreater(float(action[2]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.MAX_STEER)
        controller.reset()
        self.assertIsNone(controller._target_speed)
        self.assertEqual(controller._last_steer, 0.0)

    def test_racing_line_controller_prepositions_to_a_safe_edge_for_a_distant_obstacle(self):
        from agent import _RacingLineController

        action = _RacingLineController().act(
            _observation(obstacle_x=35, obstacle_y=30, speed=20.0)
        )

        # The obstacle is still distant, but the first command should already
        # move toward the wider right-side envelope rather than waiting for a
        # close, high-urgency avoidance turn.
        self.assertGreater(float(action[0]), 0.04)
        self.assertEqual(float(action[1]), 0.0)
        self.assertEqual(float(action[2]), 0.0)

    def test_racing_line_controller_uses_an_early_brake_for_a_sharp_curve(self):
        from agent import _RacingLineController

        controller = _RacingLineController()
        action = controller.act(_observation(curve=0.75, speed=60.0))

        self.assertGreater(float(action[2]), 0.0)
        self.assertEqual(float(action[1]), 0.0)
        self.assertLessEqual(abs(float(action[0])), controller.SHARP_CURVE_STEER_LIMIT)

    def test_racing_line_controller_suppresses_final_outward_edge_command(self):
        from agent import _RacingLineController

        outward_left = _RacingLineController._suppress_outward_steering(
            -0.1, {54: (50.0, 72.0)}
        )
        outward_right = _RacingLineController._suppress_outward_steering(
            0.1, {54: (10.0, 32.0)}
        )

        self.assertEqual(outward_left, 0.0)
        self.assertEqual(outward_right, 0.0)

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
