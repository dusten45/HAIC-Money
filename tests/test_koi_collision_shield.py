from types import SimpleNamespace

import numpy as np
import pytest

from haic.algorithms.koi import collision_shield as shield


def image(obstacle=True):
    frame = np.full((84, 84), .1, np.float32)
    frame[23:72, 30:55] = .4
    if obstacle:
        frame[43:47, 40:44] = .7
    return np.tile(frame[None], (4, 1, 1))


class Crossing:
    contact_mode = 'crossing_projection'

    def __init__(self):
        self.steps = 11
        self.base = SimpleNamespace(_frame=lambda obs: obs[-1], _estimate_speed=lambda frame: self.speed)
        self.action = np.array([.12, .6, .07], np.float32)
        self.speed = 10.
        self.last_command = ('unchanged',)

    def act(self, observation):
        return self.action.copy()

    def reset(self, observation=None):
        pass

    def last_step_diagnostics(self):
        return dict(pixel_speed=self.speed, road_centers={30: 42., 42: 42., 54: 42.})


def controlled_gaps(monkeypatch, baseline=-1., safe_index: int | None = 18):
    def gaps(paths, boxes):
        values = np.full(paths.shape[0], -1.)
        values[0] = baseline
        if safe_index is not None and safe_index < len(values):
            values[safe_index] = 1.
        return values
    monkeypatch.setattr(shield, 'footprint_gaps', gaps)


def test_requires_crossing_and_unwraps_exact_agent():
    with pytest.raises(ValueError):
        shield.CollisionShieldAgent(SimpleNamespace())
    driver = Crossing()
    assert shield.CollisionShieldAgent(SimpleNamespace(driver=driver)).driver is driver


def test_clear_baseline_is_exact_even_when_smaller_steering_is_safe(monkeypatch):
    controlled_gaps(monkeypatch, baseline=1.)
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    for _ in range(20):
        np.testing.assert_array_equal(agent.act(image()), driver.action)
        assert not agent.last_shield['active']
    assert driver.last_command == ('unchanged',)


def test_override_pedals_unchanged_and_budget_cannot_reset_on_flicker(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    for count in range(1, 7):
        action = agent.act(image())
        assert agent.last_shield['active']
        assert agent.last_shield['encounter_actions'] == count
        np.testing.assert_array_equal(action[1:], driver.action[1:])
    for _ in range(300):
        np.testing.assert_array_equal(agent.act(image()), driver.action)
        assert agent.last_shield['reason'] == 'encounter_budget'
    controlled_gaps(monkeypatch, baseline=1.)
    for _ in range(2):
        np.testing.assert_array_equal(agent.act(image()), driver.action)
    controlled_gaps(monkeypatch)
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['encounter_id'] == 1
    controlled_gaps(monkeypatch, baseline=1.)
    for _ in range(3):
        agent.act(image())
    controlled_gaps(monkeypatch)
    agent.act(image())
    assert agent.last_shield['active'] and agent.last_shield['encounter_id'] == 2
    assert agent.last_shield['encounter_actions'] == 1


def test_immediate_handback_no_heading_recovery(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    agent.act(image())
    controlled_gaps(monkeypatch, baseline=1.)
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['threat_free_decisions'] == 1
    assert not agent.last_shield['active']


def test_no_safe_candidate_never_forces_colliding_action(monkeypatch):
    controlled_gaps(monkeypatch, safe_index=None)
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['reason'] == 'no_collision_free_candidate'


def test_minimum_action_change_and_soft_road_cost(monkeypatch):
    def gaps(paths, boxes):
        values = np.ones(paths.shape[0])
        values[0] = -1.
        return values
    monkeypatch.setattr(shield, 'footprint_gaps', gaps)
    monkeypatch.setattr(shield, 'road_cost', lambda paths, *args: np.zeros(paths.shape[0]))
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    assert agent.act(image())[0] == pytest.approx(.125)
    # Even arbitrarily bad road departure is not a hard rejection.
    controlled_gaps(monkeypatch, safe_index=1)
    monkeypatch.setattr(shield, 'road_cost', lambda paths, *args: np.full(paths.shape[0], 1000.))
    assert agent.act(image())[0] == pytest.approx(-.4)


def test_box_dropout_expires_and_does_not_rearm(monkeypatch):
    driver = Crossing()
    driver.speed = .1
    agent = shield.CollisionShieldAgent(driver)
    controlled_gaps(monkeypatch)
    for _ in range(6):
        agent.act(image())
    controlled_gaps(monkeypatch, baseline=1.)
    for _ in range(20):
        agent.act(image(False))
        assert agent.last_shield['threat_free_decisions'] == 0
        assert agent.last_shield['encounter_actions'] == 6
    assert agent.last_shield['tracked_boxes'] == 0
    agent.reset()
    assert agent.encounter_actions == 0 and agent.wheel_angle == 0


def test_full_detector_retains_multiple_near_and_rear_objects_not_clipped():
    frame = image(False)[-1]
    frame[40:44, 40:44] = .7
    frame[65:69, 50:54] = .7
    frame[17:21, 40:44] = .7
    frame[45:49, :3] = .7
    assert shield.detect_boxes(frame).shape == (3, 4)


def test_projection_sign_lag_and_actual_hold_then_baseline():
    paths, wheels, hold = shield.project_paths([.4, -.4, 0], 0., 40., 0.)
    assert paths[0, hold, 0] > 0 and paths[1, hold, 0] < 0
    assert 0 < wheels[0] <= .24 + 1e-9
    assert wheels[1] == pytest.approx(-wheels[0])
    np.testing.assert_allclose(paths[2, :, 0], 0)
    longer, _, _ = shield.project_paths([.4], .4, 40., 0.)
    assert longer[0, -1, 2] > paths[0, -1, 2]
    assert np.max(np.linalg.norm(np.diff(paths[:, :, :2], axis=1), axis=2)) * shield.Y_SCALE < .5


def test_sat_clearance_checks_all_objects_wheels_and_rotation():
    paths = np.zeros((1, 1, 3))
    clear = np.array([[4., 0., 5., 1.]])
    wheel_hit = np.array([[1.5, -1.8, 1.6, -1.4]])
    assert shield.footprint_gaps(paths, clear)[0] > 0
    assert shield.footprint_gaps(paths, np.concatenate((clear, wheel_hit)))[0] < 0
    paths[0, 0, 2] = np.pi / 2
    assert shield.footprint_gaps(paths, np.array([[2.4, -.1, 2.5, .1]]))[0] < 0


def test_memory_rotates_as_well_as_translates():
    boxes = np.array([[0., 5., 1., 6.]])
    advanced = shield.advance_boxes(boxes, (0., 1., np.pi / 2))
    np.testing.assert_allclose(advanced, [[-5., 0., -4., 1.]], atol=1e-12)


def test_invalid_projection_and_launch_are_exact(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    agent = shield.CollisionShieldAgent(driver)
    driver.steps = 10
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['reason'] == 'launch_prefix'
    driver.steps = 11
    driver.speed = float('nan')
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['reason'] == 'invalid_projection'


def test_saturated_raw_hud_not_hidden_by_average(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    driver.base._estimate_speed = lambda frame: 80.
    agent = shield.CollisionShieldAgent(driver)
    np.testing.assert_array_equal(agent.act(image()), driver.action)
    assert agent.last_shield['reason'] == 'invalid_projection'


def test_unrelated_visible_object_does_not_rearm_lost_threat(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    driver.speed = .1
    agent = shield.CollisionShieldAgent(driver)
    for _ in range(6):
        agent.act(image())
    other = image(False)
    other[:, 30:34, 65:69] = .7
    controlled_gaps(monkeypatch, baseline=1.)
    for _ in range(10):
        np.testing.assert_array_equal(agent.act(other), driver.action)
        assert agent.last_shield['threat_free_decisions'] == 0
    assert agent.last_shield['rearm_blocked']
    controlled_gaps(monkeypatch)
    np.testing.assert_array_equal(agent.act(other), driver.action)
    assert agent.last_shield['encounter_actions'] == 6


def test_occluded_shrunken_fragment_cannot_refresh_full_threat(monkeypatch):
    controlled_gaps(monkeypatch)
    driver = Crossing()
    driver.speed = .1
    agent = shield.CollisionShieldAgent(driver)
    agent.act(image())
    fragment = image(False)
    fragment[:, 43:45, 41:43] = .7
    controlled_gaps(monkeypatch, baseline=1.)
    for _ in range(4):
        agent.act(fragment)
        assert not agent.last_shield['baseline_clear_observed']
    assert agent.last_shield['rearm_blocked']


def test_real_pixel_geometry_overrides_colliding_turn_without_road_veto():
    driver = Crossing()
    driver.action[0] = .65
    driver.speed = 40.
    agent = shield.CollisionShieldAgent(driver)
    agent.wheel_angle = .2
    observation = image(False)
    observation[:, 48:51, 52:55] = .7
    action = agent.act(observation)
    assert agent.last_shield['active']
    assert agent.last_shield['baseline_clearance'] <= 0
    assert agent.last_shield['chosen_clearance'] > 0
    assert action[0] < driver.action[0]
    np.testing.assert_array_equal(action[1:], driver.action[1:])
