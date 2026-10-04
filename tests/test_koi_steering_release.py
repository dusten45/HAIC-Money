from types import SimpleNamespace
from typing import Any

import numpy as np

from haic.algorithms.koi import steering_release as model


def pixels(y: int | None = 38, x=33, speed=45.):
    image = np.full((84, 84), .1, dtype=np.float32)
    image[20:70, 27:58] = .4
    image[77:83, 10:13] = (.27 + .085 * speed) / 18
    if y is not None:
        image[y - 1:y + 2, x - 1:x + 2] = .98
    return image


class Driver:
    def __init__(self, correction=.02, projected: float | None = 25.):
        self.steps, self.impact_left = 10, 0
        self.speed_history = [45.] * 4
        self.correction, self.projected = correction, projected
        self.base = SimpleNamespace(_obstacle_side=1.,
                                    _frame=lambda observation: observation[-1] if observation.ndim == 3 else observation,
                                    _estimate_speed=lambda image: float(np.clip((image[77:83, 10:13].sum() - .27) / .085, 0, 80)))
        self.last: dict[str, Any] = {}
        self.object: tuple[float, float, float] | None = (38., 33., 9.)

    def act(self, observation):
        self.steps += 1
        obj = self.object
        near = self.base._obstacle_side * .34 * float(np.clip((obj[0] - 22) / 18, 0, 1)) if obj is not None else 0.
        inherited = np.float32(np.clip(near, -.7, .7))
        steer = np.float32(np.clip(inherited + self.correction, -.7, .7))
        damping = float(steer)
        if self.impact_left > 0:
            steer = np.float32(np.clip(self.correction, -.7, .7))
            self.impact_left -= 1
        self.action = np.array([steer, .35, .12], dtype=np.float32)
        self.last = dict(near_object=obj, road_centers={42: 42., 54: 42.}, correction=self.correction,
                         inherited_steer=float(inherited), damping_steer=damping, parent_action=self.action.tolist(),
                         geometry_repaired=False, far_mode='arrival_speed', contact_active=False,
                         contact_mode='crossing_projection',
                         projected_obstacle_x=self.projected, braking_proxy_veto=False, target_speed=44.,
                         impact_proxy_trigger=False, impact_steps_remaining=self.impact_left, contact_proxy=False)
        return self.action

    def reset(self, observation):
        self.steps, self.impact_left = 0, 0

    def last_step_diagnostics(self):
        return dict(self.last)


def advance(agent, y=42, x=31):
    observed_y = .5 * (max(22, y - 1) + min(61, y + 1))
    agent.driver.object = (float(observed_y), float(x), 9.)
    return agent.act(pixels(y, x))


def test_near_only_release_leaves_targets_pedals_and_source_data_intact():
    driver = Driver()
    agent = model.SteeringReleaseAgent(driver)
    np.testing.assert_array_equal(agent.act(pixels()), driver.action)
    action = advance(agent)
    assert agent.last['steering_release_reason'] == 'near_release', agent.last['baseline_steering_terms']
    assert agent.last['steering_release_alpha'] == 0.
    assert action[0] == np.float32(.02)
    np.testing.assert_array_equal(action[1:], driver.action[1:])
    assert agent.last['target_speed'] == 44.
    assert agent.last['steering_terms']['avoidance_component'] == 0.
    assert agent.last['steering_terms']['reconstruction_context'] == 'actual_postrelease'
    assert agent.last['baseline_steering_terms']['avoidance_component'] > .3
    assert driver.last['inherited_steer'] != action[0]


def test_initial_prefix_and_no_object_are_complete_baseline_actions():
    driver = Driver()
    driver.steps = 0
    agent = model.SteeringReleaseAgent(driver)
    for _ in range(10):
        np.testing.assert_array_equal(agent.act(pixels()), driver.action)
        assert agent.last['steering_release_reason'] == 'launch_prefix'
    driver.object = None
    np.testing.assert_array_equal(agent.act(pixels(None)), driver.action)
    assert not agent.last['steering_release_changed']


def test_unknown_projection_is_not_initial_clearance():
    agent = model.SteeringReleaseAgent(Driver(projected=None))
    agent.act(pixels())
    np.testing.assert_array_equal(advance(agent), agent.driver.action)
    assert agent.last['steering_release_reason'] == 'projection_not_cleared'


def test_valid_same_object_clearance_survives_late_projection_unavailability():
    agent = model.SteeringReleaseAgent(Driver())
    agent.act(pixels())
    advance(agent)
    agent.driver.projected = None
    action = advance(agent, 46, 29)
    assert agent.last['steering_release_reason'] == 'near_release'
    assert not agent.last['steering_release_projection_available']
    assert agent.last['steering_release_same_object']
    np.testing.assert_array_equal(action[1:], agent.driver.action[1:])


def test_reacquisition_rearms_and_never_uses_previous_object_clearance():
    agent = model.SteeringReleaseAgent(Driver())
    agent.act(pixels())
    advance(agent)
    agent.driver.projected = None
    np.testing.assert_array_equal(advance(agent, 30, 31), agent.driver.action)
    assert not agent.last['steering_release_projection_cleared']
    assert agent.last['steering_release_reason'] == 'projection_not_cleared'


def test_actual_current_reentry_invalidates_latch_until_fresh_valid_projection():
    agent = model.SteeringReleaseAgent(Driver())
    agent.act(pixels())
    advance(agent)
    agent.driver.projected = None
    np.testing.assert_array_equal(advance(agent, 57, 40), agent.driver.action)
    assert agent.last['steering_release_reason'] == 'current_reentry'
    assert not agent.last['steering_release_projection_cleared']
    np.testing.assert_array_equal(advance(agent, 61, 31), agent.driver.action)
    assert agent.last['steering_release_reason'] == 'projection_not_cleared'
    agent.driver.projected = 25.
    advance(agent, 61, 31)
    assert agent.last['steering_release_reason'] == 'near_release'


def test_reentry_rearms_even_during_a_protected_impact_replacement():
    agent = model.SteeringReleaseAgent(Driver())
    agent.act(pixels())
    advance(agent)
    agent.driver.projected = None
    agent.driver.impact_left = 1
    np.testing.assert_array_equal(advance(agent, 57, 40), agent.driver.action)
    assert agent.last['steering_release_reason'] == 'protected_replacement'
    assert not agent.last['steering_release_projection_cleared']
    np.testing.assert_array_equal(advance(agent, 61, 31), agent.driver.action)
    assert agent.last['steering_release_reason'] == 'projection_not_cleared'


def test_near_term_decay_does_not_admit_a_future_road_steer_reentry():
    agent = model.SteeringReleaseAgent(Driver(correction=-.3))
    agent.act(pixels())
    action = advance(agent, 42, 33)
    assert not model.release_safe((32, 41, 3, 3), (42., 33., 9.), 1., -.3, .04, 45.)
    assert agent.last['steering_release_alpha'] >= .5
    assert action[0] > -.14


def test_hud_raw_saturation_and_nonfirst_history_nan_fail_closed():
    for saturated in (True, False):
        driver = Driver()
        agent = model.SteeringReleaseAgent(driver)
        agent.act(pixels())
        driver.object = (42., 31., 9.)
        image = pixels(42, 31, speed=90. if saturated else 45.)
        if not saturated:
            driver.speed_history = [45., np.nan]
        np.testing.assert_array_equal(agent.act(image), driver.action)
        assert agent.last['steering_release_reason'] == 'hud_uncertainty'


def test_crossing_or_flip_rearms_clearance_without_replacing_projection():
    driver = Driver()
    agent = model.SteeringReleaseAgent(driver)
    agent.act(pixels())
    advance(agent)
    driver.projected = 40.
    np.testing.assert_array_equal(advance(agent, 46, 29), driver.action)
    assert not agent.last['steering_release_projection_cleared']
    driver.projected = 25.
    driver.base._obstacle_side = -1.
    np.testing.assert_array_equal(advance(agent, 50, 27), driver.action)
    assert not agent.last['steering_release_projection_cleared']


def test_reset_clears_release_history():
    agent = model.SteeringReleaseAgent(Driver())
    agent.act(pixels())
    advance(agent)
    agent.reset(pixels(None))
    assert agent.last_step_diagnostics() == {}
    assert agent._previous is None and not agent._projection_cleared


def test_box_guard_includes_full_footprint_and_unchanged_projection_band():
    assert model.PROJECTION_BAND == 6.
    assert model.HALF_WIDTH > 1.57
    assert not model.guarded_arc((41, 55, 3, 3), (56., 42., 9.), 1., 0., 5., require_rear_clear=False)
    assert not model.guarded_arc((35, 55, 3, 3), (56., 36., 9.), 1., 0., 5., require_rear_clear=False)
    assert model.guarded_arc((30, 55, 3, 3), (56., 31., 9.), 1., 0., 5., require_rear_clear=False)


def test_bounds_use_original_brightness_connectivity_and_exact_component_centroid():
    image = pixels(None)
    image[38:40, 31:33] = .55
    image[40:42, 33:35] = .55
    assert model.detected_bounds(image, (39.5, 32.5, 42.)) == (31, 38, 4, 4)
    assert model.detected_bounds(image, (39.5, 33., 42.)) is None


class MotionDriver(Driver):
    def __init__(self):
        super().__init__(projected=None)
        self.base._last_obstacle_side_offset = None
        self.base._road_centers = lambda image: {30: 42., 42: 42., 54: 42.}
        self.base._nearest_bright_object = lambda image, centers: self.object

    def act(self, observation):
        obj = self.object
        if obj is not None:
            offset = obj[1] - obj[2]
            candidate = 1. if offset < 0 else -1.
            old = self.base._last_obstacle_side_offset
            if self.base._obstacle_side == 0 or (obj[0] < 52 and abs(offset) > 1):
                self.base._obstacle_side = candidate
            elif obj[0] < 52 and old is not None:
                motion = offset - old
                if self.base._obstacle_side > 0 and offset > -3 and motion > 2:
                    self.base._obstacle_side = -1.
                elif self.base._obstacle_side < 0 and offset < 3 and motion < -2:
                    self.base._obstacle_side = 1.
            self.base._last_obstacle_side_offset = offset
        return super().act(observation)


def test_generation_stabilizes_only_observed_ambiguous_same_component_motion():
    for enabled, expected in ((False, -1.), (True, 1.)):
        driver = MotionDriver()
        agent = model.SteeringReleaseAgent(driver, stabilize_ambiguous_flank=enabled)
        driver.object = (32., 31., 35.5)
        agent.act(pixels(32, 31))
        driver.object = (36., 33., 33.675)
        action = agent.act(pixels(36, 33))
        assert driver.base._obstacle_side == expected
        assert agent.last['steering_generation_ambiguous_motion_ignored'] is enabled
        assert np.isclose(driver.base._last_obstacle_side_offset, -.675, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(action[1:], driver.action[1:])
        assert agent.last['target_speed'] == 44.
        assert agent.last['steering_terms']['reconstruction_valid']


def test_generation_retains_unambiguous_choice_and_resets_with_identity():
    driver = MotionDriver()
    agent = model.SteeringReleaseAgent(driver, stabilize_ambiguous_flank=True)
    driver.object = (32., 31., 35.5)
    agent.act(pixels(32, 31))
    driver.object = (36., 33., 31.)
    agent.act(pixels(36, 33))
    assert driver.base._obstacle_side == -1.
    assert not agent.last['steering_generation_ambiguous_motion_ignored']
    driver.base._obstacle_side = 1.
    driver.base._last_obstacle_side_offset = -4.5
    driver.object = (26., 33., 33.675)
    agent.act(pixels(26, 33))
    assert driver.base._obstacle_side == -1.
    assert not agent.last['steering_generation_ambiguous_motion_ignored']


def test_generation_preserves_motion_choice_when_offset_already_favors_new_side():
    driver = MotionDriver()
    agent = model.SteeringReleaseAgent(driver, stabilize_ambiguous_flank=True)
    driver.object = (32., 31., 35.5)
    agent.act(pixels(32, 31))
    driver.object = (36., 33., 32.325)
    agent.act(pixels(36, 33))
    assert driver.base._obstacle_side == -1.
    assert not agent.last['steering_generation_ambiguous_motion_ignored']


def test_generation_suppression_is_mirrored_for_negative_committed_flank():
    driver = MotionDriver()
    agent = model.SteeringReleaseAgent(driver, stabilize_ambiguous_flank=True)
    driver.object = (32., 35., 30.5)
    agent.act(pixels(32, 35))
    assert driver.base._obstacle_side == -1.
    driver.object = (36., 33., 32.325)
    agent.act(pixels(36, 33))
    assert driver.base._obstacle_side == -1.
    assert agent.last['steering_generation_ambiguous_motion_ignored']
