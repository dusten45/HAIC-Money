from types import SimpleNamespace
import unittest

import numpy as np

from haic.algorithms.koi import minimum_clearance as model


def frame(box=(41, 36, 3, 3), left=27, right=58):
    image = np.full((84, 84), .1, dtype=np.float32)
    image[5:73, left:right] = .4
    if box is not None:
        x, y, width, height = box
        image[y:y + height, x:x + width] = .7
    return image


CENTERS = {y: 42. for y in (30, 34, 38, 42, 46, 50, 54)}


class Driver:
    def __init__(self, box=(41, 36, 3, 3), side=-1., **info):
        self.box = box
        self.steps = 11
        self.impact_left = 0
        self.speed_history = [44., 44.]
        self.last_command = 0.
        self.base = SimpleNamespace(_last_obstacle=None, _obstacle_side=side, _frame=lambda observation: observation)
        self.base._estimate_speed = lambda image: float(np.clip((image[77:83, 10:13].sum() - .27) / .085, 0, 80))
        self.last = dict(road_centers=dict(CENTERS), pixel_speed=44., geometry_repaired=False,
                         impact_proxy_trigger=False, contact_proxy=False, correction=0.,
                         projected_obstacle_x=None, contact_active=False, target_speed=44.)
        self.last.update(info)
        self.action = np.asarray([side * .34, .08, .12], dtype=np.float32)

    def act(self, observation):
        self.steps += 1
        self.base._last_obstacle = None if self.box is None else (
            self.box[1] + (self.box[3] - 1) / 2,
            self.box[0] + (self.box[2] - 1) / 2, 42.)
        return self.action.copy()

    def reset(self, observation=None):
        self.steps = 0


class GeometryTests(unittest.TestCase):
    def test_physical_dimensions_include_steered_front_wheels(self):
        expected = (55 + 14 * np.cos(.4) + 27 * np.sin(.4)) * .02
        self.assertAlmostEqual(model.NOMINAL_HALF_WIDTH_M, expected)
        self.assertAlmostEqual(model.HALF_WIDTH_M, expected + .01)
        self.assertGreater(model.HALF_WIDTH_M, 60 * .02)
        self.assertAlmostEqual(model.X_SCALE, 2.7 * 6 * 84 / 1000)
        self.assertAlmostEqual(model.Y_SCALE, 2.7 * 6 * 84 / 800)
        self.assertEqual(model.HALF_LENGTH_M, 130 * .02 + .01)

    def test_full_component_bounds_match_baseline_centroid(self):
        for width in range(2, 10):
            for height in range(2, 9):
                box = (38, 35, width, height)
                obj = (35 + (height - 1) / 2, 38 + (width - 1) / 2, 42.)
                self.assertEqual(model.component_bounds(frame(box), obj), box)

    def test_clipped_or_oversize_components_are_rejected(self):
        for box in ((40, 21, 3, 3), (40, 60, 3, 3), (40, 35, 10, 3), (40, 35, 3, 11), (0, 35, 3, 3)):
            with self.subTest(box=box):
                obj = (box[1] + (box[3] - 1) / 2, box[0] + (box[2] - 1) / 2, 42.)
                self.assertIsNone(model.component_bounds(frame(box), obj))

    def test_bounding_box_not_centroid_controls_clearance(self):
        small = model.minimum_command(frame((41, 32, 3, 3)), CENTERS, (41, 32, 3, 3), 0., -1.)
        wide = model.minimum_command(frame((39, 32, 7, 3)), CENTERS, (39, 32, 7, 3), 0., -1.)
        assert small is not None and wide is not None
        self.assertGreater(abs(wide['displacement']), abs(small['displacement']))

    def test_first_grid_feasible_displacement_is_chosen(self):
        box = (41, 36, 3, 3)
        image = frame(box)
        proposal = model.minimum_command(image, CENTERS, box, 0., -1.)
        assert proposal is not None
        previous = proposal['displacement'] + model.DISPLACEMENT_RESOLUTION_PX
        valid, _, _ = model.sweep_clearance([previous, proposal['displacement']], CENTERS, box,
                                           model.supported_boundaries(image, CENTERS, box))
        self.assertEqual(valid.tolist(), [False, True])
        self.assertGreaterEqual(proposal['obstacle_margin'], 0.)
        self.assertGreaterEqual(proposal['road_margin'], 0.)

    def test_both_flanks_have_symmetric_minimum_targets(self):
        box = (41, 36, 3, 3)
        left = model.minimum_command(frame(box), CENTERS, box, 0., -1.)
        right = model.minimum_command(frame(box), CENTERS, box, 0., 1.)
        assert left is not None and right is not None
        self.assertAlmostEqual(left['displacement'], -right['displacement'])
        self.assertAlmostEqual(left['steer'], -right['steer'])

    def test_visible_obstacle_can_need_no_offset_but_front_rear_are_checked(self):
        box = (33, 38, 3, 3)
        proposal = model.minimum_command(frame(box), CENTERS, box, 0., 1.)
        assert proposal is not None
        self.assertEqual(proposal['displacement'], 0.)
        self.assertTrue(proposal['road_clear'])
        # A tall centered near box is not declared passed just because projection is off.
        box = (41, 48, 3, 10)
        self.assertIsNone(model.minimum_command(frame(box), CENTERS, box, 0., -1.))

    def test_narrow_unknown_search_clipped_and_image_clipped_roads_fail_closed(self):
        box = (41, 36, 3, 3)
        for left, right in ((37, 48), (0, 84), (24, 60), (0, 51)):
            with self.subTest(edges=(left, right)):
                self.assertIsNone(model.minimum_command(frame(box, left, right), CENTERS, box, 0., -1.))

    def test_other_dark_or_bright_holes_do_not_become_supported_road(self):
        box = (41, 36, 3, 3)
        for value in (.1, .7):
            image = frame(box)
            image[45, 33] = value
            self.assertNotIn(45, model.supported_boundaries(image, CENTERS, box))
            self.assertIsNone(model.minimum_command(image, CENTERS, box, 0., -1.))

    def test_visible_approach_row56_is_required_not_hidden_by_car_occlusion(self):
        box = (41, 38, 3, 3)
        image = frame(box)
        self.assertIsNotNone(model.minimum_command(image, CENTERS, box, 0., -1.))
        image[56] = .1
        self.assertFalse(model.car_occlusion_mask()[56].any())
        self.assertIsNone(model.minimum_command(image, CENTERS, box, 0., -1.))

    def test_known_current_car_can_occlude_only_its_own_raster_silhouette(self):
        box = (33, 38, 3, 3)
        image = frame(box)
        mask = model.car_occlusion_mask()
        image[mask] = .1
        self.assertIsNotNone(model.minimum_command(image, CENTERS, box, 0., 1.))
        image[58, 31] = .1
        self.assertIsNone(model.minimum_command(image, CENTERS, box, 0., 1.))

    def test_actual_hold_guard_does_not_assume_a_constant_command_until_passage(self):
        box = (41, 36, 3, 3)
        image = frame(box)
        proposal = model.minimum_command(image, CENTERS, box, 0., -1.)
        assert proposal is not None
        self.assertFalse(model.commanded_path_clear(image, CENTERS, box, proposal['steer']))
        self.assertTrue(model.commanded_path_clear(image, CENTERS, box, proposal['steer'], 44. * .08))
        image[56] = .1
        self.assertFalse(model.commanded_path_clear(image, CENTERS, box, proposal['steer'], 44. * .08))

    def test_hold_guard_still_checks_imminent_obstacle_overlap(self):
        box = (41, 55, 3, 3)
        self.assertFalse(model.commanded_path_clear(frame(box), CENTERS, box, 0., 60. * .08))
        for distance in (0., -1., np.inf, np.nan):
            self.assertFalse(model.commanded_path_clear(frame(), CENTERS, (41, 36, 3, 3), 0., distance))


class AgentTests(unittest.TestCase):
    def test_steering_only_and_target_is_geometric_not_scaled(self):
        driver = Driver(box=(44, 30, 3, 3))
        agent = model.MinimumClearanceAgent(driver)
        action = agent.act(frame(driver.box))
        self.assertEqual(action.dtype, np.float32)
        self.assertEqual(action.shape, (3,))
        np.testing.assert_array_equal(action[1:], driver.action[1:])
        self.assertEqual(driver.last['target_speed'], 44.)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'minimum_clearance')
        self.assertGreater(agent.last['minimum_clearance_target_x'], 27.)
        self.assertLess(abs(action[0]), abs(driver.action[0]))
        self.assertEqual(driver.last_command, float(action[0]))

    def test_applied_command_includes_damping_and_vetoes_reference_only_return(self):
        driver = Driver(box=(48, 38, 3, 3), correction=.1375)
        agent = model.MinimumClearanceAgent(driver)
        action = agent.act(frame(driver.box))
        self.assertEqual(agent.last['minimum_clearance_displacement_px'], 0.)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'applied_command_guard')
        self.assertFalse(agent.last['return_centerline_active'])
        np.testing.assert_array_equal(action, driver.action)

    def test_centered_reference_does_not_override_when_applied_arc_needs_countersteer(self):
        driver = Driver()
        agent = model.MinimumClearanceAgent(driver)
        action = agent.act(frame(driver.box))
        self.assertEqual(agent.last['minimum_clearance_reason'], 'applied_command_guard')
        np.testing.assert_array_equal(action, driver.action)

    def test_hold_variant_preserves_full_reference_and_changes_only_steering(self):
        driver = Driver()
        agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08)
        action = agent.act(frame(driver.box))
        self.assertEqual(agent.last['minimum_clearance_reason'], 'minimum_clearance')
        self.assertEqual(agent.last['minimum_clearance_command_distance_m'], 44. * .08)
        self.assertGreaterEqual(agent.last['minimum_clearance_road_margin_px'], 0.)
        self.assertGreaterEqual(agent.last['minimum_clearance_obstacle_margin_px'], 0.)
        np.testing.assert_array_equal(action[1:], driver.action[1:])
        self.assertEqual(driver.last['target_speed'], 44.)
        image = frame(driver.box)
        image[45, 33] = .1
        np.testing.assert_array_equal(agent.act(image), driver.action)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'unsupported_sweep')

    def test_hold_speed_uses_recent_maximum_and_fails_on_saturation(self):
        driver = Driver(pixel_speed=52.)
        driver.speed_history = [60., 56., 53.]
        agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08)
        agent.act(frame(driver.box))
        self.assertEqual(agent.last['minimum_clearance_command_distance_m'], 60. * .08)
        driver.speed_history = [80.]
        driver.last['pixel_speed'] = 80.
        np.testing.assert_array_equal(agent.act(frame(driver.box)), driver.action)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'applied_command_guard')

    def test_hold_configuration_is_bound_to_actual_action_repeat(self):
        for value in (0., .04, .16, np.inf, np.nan):
            with self.assertRaises(ValueError):
                model.MinimumClearanceAgent(Driver(), command_hold_seconds=value)

    def test_hold_rejects_raw_hud_saturation_hidden_by_baseline_average(self):
        driver = Driver(pixel_speed=75.)
        image = frame(driver.box)
        image[77:83, 10:13] = (.27 + .085 * 90) / 18
        action = model.MinimumClearanceAgent(driver, command_hold_seconds=.08).act(image)
        np.testing.assert_array_equal(action, driver.action)
        self.assertEqual(driver.last['minimum_clearance_reason'], 'applied_command_guard')
        self.assertIsNone(driver.last['minimum_clearance_command_speed_bound_mps'])

    def test_hold_validates_all_history_inputs_before_maximum(self):
        for bad in (np.nan, np.inf, -1.):
            driver = Driver()
            driver.speed_history = [44., bad]
            agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08)
            np.testing.assert_array_equal(agent.act(frame(driver.box)), driver.action)
            self.assertIsNone(agent.last['minimum_clearance_command_speed_bound_mps'])

    def test_projection_footprint_gate_rejects_observed_r2_active_crossing(self):
        driver = Driver(box=(40, 30, 3, 4), side=1., projected_obstacle_x=38.63380281690142,
                        contact_active=True)
        agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08, require_projection_clearance=True)
        np.testing.assert_array_equal(agent.act(frame(driver.box)), driver.action)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'projection_footprint_guard')
        self.assertGreater(agent.last['minimum_clearance_required_projection_gap_px'], abs(38.63380281690142 - 42))

    def test_projection_gate_requires_known_separation_on_correct_flank(self):
        for projected in (None, 42., 34., np.inf, np.nan):
            driver = Driver(projected_obstacle_x=projected)
            agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08, require_projection_clearance=True)
            np.testing.assert_array_equal(agent.act(frame(driver.box)), driver.action)
            self.assertEqual(agent.last['minimum_clearance_reason'], 'projection_footprint_guard')
        driver = Driver(projected_obstacle_x=50.)
        agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08, require_projection_clearance=True)
        driver.base._obstacle_side = -1.
        agent.act(frame(driver.box))
        self.assertEqual(agent.last['minimum_clearance_reason'], 'minimum_clearance')

    def test_projection_gap_uses_obstacle_width_and_unchanged_footprint_margin(self):
        for box, expected in (((41, 32, 3, 3), 'minimum_clearance'),
                              ((39, 32, 7, 3), 'projection_footprint_guard')):
            driver = Driver(box=box, projected_obstacle_x=48.)
            agent = model.MinimumClearanceAgent(driver, command_hold_seconds=.08, require_projection_clearance=True)
            agent.act(frame(box))
            self.assertEqual(agent.last['minimum_clearance_reason'], expected)
            np.testing.assert_array_equal(agent.last['baseline_pedals'], driver.action[1:])

    def test_prefix_no_obstacle_and_uncertain_cases_preserve_all_action_values(self):
        cases = [dict(steps=8), dict(box=None), dict(impact_left=1), dict(speed_history=[65.]),
                 dict(info={'impact_proxy_trigger': True}), dict(info={'contact_proxy': True}),
                 dict(info={'road_centers': {54: 42., 42: 42.}}),
                 dict(info={'geometry_repaired': True}), dict(side=0.), dict(box=(41, 57, 3, 3)),
                 dict(info={'road_centers': {**CENTERS, 54: 50.}}), dict(info={'correction': .3})]
        for case in cases:
            with self.subTest(case=case):
                driver = Driver(box=case.get('box', (41, 36, 3, 3)), side=case.get('side', -1.), **case.get('info', {}))
                for name in ('steps', 'impact_left', 'speed_history'):
                    if name in case:
                        setattr(driver, name, case[name])
                action = model.MinimumClearanceAgent(driver).act(frame(driver.box))
                np.testing.assert_array_equal(action, driver.action)

    def test_projection_off_does_not_release_gap_when_road_path_still_intersects(self):
        driver = Driver(contact_active=False)
        agent = model.MinimumClearanceAgent(driver)
        agent.act(frame(driver.box))
        self.assertFalse(agent.last['baseline_projection_active'])
        self.assertTrue(agent.last['avoidance_active'])
        self.assertFalse(agent.last['return_centerline_active'])
        self.assertNotEqual(agent.last['minimum_clearance_displacement_px'], 0.)

    def test_available_projection_clear_is_distinct_from_branch_off(self):
        for value, available, clear in ((None, False, False), (42., True, False), (49., True, True)):
            with self.subTest(projected=value):
                driver = Driver(projected_obstacle_x=value)
                agent = model.MinimumClearanceAgent(driver)
                agent.act(frame(driver.box))
                self.assertEqual(agent.last['baseline_projection_available'], available)
                self.assertEqual(agent.last['baseline_projection_clear'], clear)

    def test_projection_off_returns_only_on_supported_full_footprint(self):
        driver = Driver(box=(33, 38, 3, 3), side=1.)
        agent = model.MinimumClearanceAgent(driver)
        action = agent.act(frame(driver.box))
        self.assertEqual(action[0], 0.)
        self.assertTrue(agent.last['baseline_avoidance_active'])
        self.assertFalse(agent.last['avoidance_active'])
        self.assertTrue(agent.last['return_centerline_active'])
        self.assertEqual(agent.last['minimum_clearance_displacement_px'], 0.)

    def test_temporal_projection_contradiction_vetoes_return(self):
        driver = Driver(box=(33, 38, 3, 3), side=1., projected_obstacle_x=42., contact_active=True)
        agent = model.MinimumClearanceAgent(driver)
        action = agent.act(frame(driver.box))
        np.testing.assert_array_equal(action, driver.action)
        self.assertEqual(agent.last['minimum_clearance_reason'], 'return_projection_guard')
        self.assertFalse(agent.last['return_centerline_active'])

    def test_reset_clears_only_isolated_state_and_calls_baseline_reset(self):
        driver = Driver()
        agent = model.MinimumClearanceAgent(driver)
        agent.act(frame(driver.box))
        agent.reset(frame())
        self.assertEqual(agent.last_step_diagnostics(), {})
        self.assertEqual(driver.steps, 0)

    def test_diagnostics_are_a_copy_and_json_serializable(self):
        import json
        driver = Driver()
        agent = model.MinimumClearanceAgent(driver)
        agent.act(frame(driver.box))
        info = agent.last_step_diagnostics()
        json.dumps(info, allow_nan=False)
        info['minimum_clearance_reason'] = 'external'
        self.assertNotEqual(agent.last['minimum_clearance_reason'], 'external')


if __name__ == '__main__':
    unittest.main()
