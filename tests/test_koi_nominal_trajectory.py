from types import SimpleNamespace
from pathlib import Path
import importlib

import numpy as np
import pytest

from haic.algorithms.koi import nominal_trajectory as trajectory
from haic.algorithms.koi.collision_shield import CollisionShieldAgent


def fixture(center=42., obstacle_x=34):
    frame = np.full((84, 84), .1, np.float32)
    frame[5:73, 27:58] = .4
    frame[38:41, obstacle_x:obstacle_x + 3] = .7
    centers = {row: center for row in (30, 34, 38, 42, 46, 50, 54)}
    return frame, centers, (obstacle_x, 38, 3, 3)


def test_complete_bilateral_path_has_clear_footprint_and_rejoins():
    frame, centers, box = fixture()
    proposal = trajectory.select_trajectory(frame, centers, box, 35., 0., 0.)
    assert proposal is not None
    assert proposal['candidate_count'] >= 2
    assert proposal['road_margin_px'] >= 0
    assert proposal['obstacle_margin_px'] > 0
    assert proposal['all_object_gap_m'] > 0
    assert proposal['terminal_offset_px'] == 0
    assert abs(proposal['steer']) <= float(np.float32(.4))


def test_unknown_road_and_infeasible_near_object_fail_closed():
    frame, centers, box = fixture()
    frame[10:30] = .1
    assert trajectory.select_trajectory(frame, centers, box, 35., 0., 0.) is None
    frame, centers, box = fixture()
    assert trajectory.select_trajectory(frame, centers, (41, 53, 3, 3), 35., 0., 0.) is None


def test_other_visible_obstacle_is_checked(monkeypatch):
    frame, centers, box = fixture()
    original = trajectory.footprint_gaps
    calls = []

    def gaps(paths, boxes):
        calls.append(boxes.copy())
        return original(paths, boxes)

    frame[19:22, 43:46] = .7
    monkeypatch.setattr(trajectory, 'footprint_gaps', gaps)
    trajectory.select_trajectory(frame, centers, box, 35., 0., 0.)
    assert calls and len(calls[0]) == 2


class Nominal:
    contact_mode = 'crossing_projection'

    def __init__(self):
        self.steps = 11
        self.impact_left = 0
        self.calls = 0
        self.frame, self.centers, self.box = fixture()
        self.action = np.array([.34, .6, .07], np.float32)
        self.base = SimpleNamespace(_last_obstacle=(39., 35., 42.),
                                    _frame=lambda obs: obs[-1], _estimate_speed=lambda frame: 35.)

    def act(self, observation):
        self.calls += 1
        return self.action.copy()

    def reset(self, observation=None):
        self.calls = 0

    def last_step_diagnostics(self):
        return dict(pixel_speed=35., road_centers=self.centers, geometry_repaired=False,
                    impact_proxy_trigger=False, contact_proxy=False)


def test_nominal_calls_once_preserves_pedals_and_shield_does_not_unwrap():
    nominal = Nominal()
    candidate = trajectory.NominalTrajectoryAgent(nominal)
    shield = CollisionShieldAgent(candidate)
    assert shield.driver is candidate
    observation = np.tile(nominal.frame[None], (4, 1, 1))
    action = shield.act(observation)
    assert nominal.calls == 1
    np.testing.assert_array_equal(action[1:], nominal.action[1:])
    assert candidate.last['nominal_reason'] == 'nominal_trajectory'
    assert candidate.last['nominal_changed']
    assert candidate.last_nominal_action == nominal.action.tolist()
    assert action.dtype == np.float32
    candidate.observe_executed_action(action)
    assert candidate.previous_steer == float(action[0])
    candidate.reset()
    assert candidate.previous_steer == candidate.wheel_angle == 0.


@pytest.mark.parametrize('guard', ['prefix', 'impact', 'unknown', 'no_object'])
def test_inactive_action_is_bit_exact(guard):
    nominal = Nominal()
    candidate = trajectory.NominalTrajectoryAgent(nominal)
    if guard == 'prefix':
        nominal.steps = 10
    elif guard == 'impact':
        nominal.impact_left = 1
    elif guard == 'unknown':
        nominal.centers = {}
    else:
        nominal.base._last_obstacle = None
    action = candidate.act(np.tile(nominal.frame[None], (4, 1, 1)))
    np.testing.assert_array_equal(action, nominal.action)
    assert nominal.calls == 1
    assert not candidate.last['nominal_changed']


def test_complete_tracking_validation_can_veto_reference(monkeypatch):
    frame, centers, box = fixture()
    actual = trajectory.rollout_trajectories

    def overlapping(*args, **kwargs):
        paths, commands, variation = actual(*args, **kwargs)
        paths[:, :, :2] = [(35. - 42.) / trajectory.X_SCALE, (63. - 39.) / trajectory.Y_SCALE]
        return paths, commands, variation

    monkeypatch.setattr(trajectory, 'rollout_trajectories', overlapping)
    assert trajectory.select_trajectory(frame, centers, box, 35., 0., 0.) is None


def test_selected_rejoin_reference_continues_without_detector_release():
    nominal = Nominal()
    candidate = trajectory.NominalTrajectoryAgent(nominal)
    observation = np.tile(nominal.frame[None], (4, 1, 1))
    action = candidate.act(observation)
    assert candidate.reference is not None
    previous_end = candidate.reference[-1, 1]
    candidate.observe_executed_action(action)
    assert candidate.reference[-1, 1] < previous_end
    nominal.base._last_obstacle = None
    nominal.frame[38:41, 34:37] = .4
    action = candidate.act(np.tile(nominal.frame[None], (4, 1, 1)))
    assert candidate.last['nominal_reason'] == 'nominal_trajectory_continue'
    np.testing.assert_array_equal(action[1:], nominal.action[1:])
    assert nominal.calls == 2


def test_unknown_remaining_path_discards_plan_and_is_exact_fallback():
    nominal = Nominal()
    candidate = trajectory.NominalTrajectoryAgent(nominal)
    action = candidate.act(np.tile(nominal.frame[None], (4, 1, 1)))
    candidate.observe_executed_action(action)
    nominal.centers = {}
    action = candidate.act(np.tile(nominal.frame[None], (4, 1, 1)))
    assert candidate.reference is None
    np.testing.assert_array_equal(action, nominal.action)


def test_full_first_hold_guard_extends_beyond_short_reference(monkeypatch):
    frame, centers, box = fixture()
    reference = np.array([[[0., 0., 0.], [0., 1., 0.]]])
    original = trajectory.project_paths

    def leaves_road(*args, **kwargs):
        paths, wheels, hold = original(*args, **kwargs)
        paths[:, -1, 0] = 30.
        return paths, wheels, hold

    monkeypatch.setattr(trajectory, 'project_paths', leaves_road)
    result = trajectory.checked_rollouts(reference, frame, centers, box, 35., 0., 3.)
    assert not result[3][0]


def test_complete_rejoin_must_fit_actual_remaining_plan_budget():
    frame, centers, box = fixture()
    reference = np.array([[[0., 0., 0.], [0., 20., 0.]]])
    result = trajectory.checked_rollouts(reference, frame, centers, box, 15., 0., 3.)
    assert not result[3][0]
    result = trajectory.checked_rollouts(reference, frame, centers, box, 35., 0., 3., remaining_actions=1)
    assert not result[3][0]


def test_guarded_feedback_still_matches_final_shield_actuator_model():
    nominal = Nominal()
    nominal.action[0] = .12
    nominal.centers = {}
    candidate = trajectory.NominalTrajectoryAgent(nominal)
    shield = CollisionShieldAgent(candidate)
    observation = np.tile(nominal.frame[None], (4, 1, 1))
    for _ in range(3):
        action = shield.act(observation)
        candidate.observe_executed_action(action)
        assert candidate.speed == shield.last_shield['projection_speed']
        assert candidate.wheel_angle == shield.wheel_angle


def frozen_corridor(monkeypatch):
    source = Path(__file__).resolve().parents[1] / 'submissions' / '20261002-crossing-projection-collision-shield-v1-baseline' / 'source'
    monkeypatch.syspath_prepend(str(source))
    cls = importlib.import_module('haic_agent.corridor_agent').VisionCorridorAgent
    agent = cls()
    frame, centers, _ = fixture()
    monkeypatch.setattr(agent, '_frame', lambda obs: frame)
    monkeypatch.setattr(agent, '_road_centers', lambda frame: centers)
    monkeypatch.setattr(agent, '_observation_speed', lambda obs, frame: 35.)
    return agent


def test_frozen_avoidance_demand_does_not_scale_with_required_displacement(monkeypatch):
    agent = frozen_corridor(monkeypatch)
    actions = []
    for x in (34., 42.):
        agent.reset()
        monkeypatch.setattr(agent, '_nearest_bright_object', lambda frame, centers: (40., x, 42.))
        actions.append(float(agent.act(None)[0]))
    assert actions[0] == pytest.approx(.34)
    assert actions[1] == pytest.approx(-.34)


def test_frozen_motion_branch_reverses_flank_without_offset_sign_crossing(monkeypatch):
    agent = frozen_corridor(monkeypatch)
    actions = []
    for y, offset in ((40., -3.238887), (44., -.667105)):
        monkeypatch.setattr(agent, '_nearest_bright_object',
                            lambda frame, centers: (y, 42. + offset, 42.))
        actions.append(float(agent.act(None)[0]))
    assert actions == pytest.approx([.34, -.34])
