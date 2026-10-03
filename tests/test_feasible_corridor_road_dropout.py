"""Deterministic checks for diagnostic no-corridor road-row recovery."""

import numpy as np
import pytest

import agent


def _candidate():
    candidate = getattr(agent, "_FeasibleCorridorRoadDropoutController", None)
    assert candidate is not None
    return candidate()


def _dropout_observation(*, mirror=False, obstacle=True, row42=False):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    # The sampled road is exactly {46: 22.5, 50: 28, 54: 34}. The near
    # obstacle is surrounded by asphalt but too near for the checked planner.
    # Account for the tracker's +/-17-pixel search as it walks from near
    # row54 to far row46; some asphalt pixels lie outside that local window.
    frame[46, 10:35] = 0.4
    frame[50, 16:40] = 0.4
    frame[54:61, 22:44] = 0.4
    if row42:
        frame[42, 4:32] = 0.4
    if obstacle:
        frame[55:59, 23:26] = 0.65
    if mirror:
        frame = frame[:, ::-1].copy()
    frame[77:83, 10:13] = (0.27 + 0.085 * 29.0) / 18.0
    return np.repeat(frame[None], 4, axis=0)


def _planned_observation():
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    frame[22:62, 29:55] = 0.4
    frame[38:42, 40:43] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * 29.0) / 18.0
    return np.repeat(frame[None], 4, axis=0)


@pytest.mark.parametrize("mirror", [False, True])
def test_near_obstacle_row42_dropout_keeps_observed_bend(mirror):
    observation = _dropout_observation(mirror=mirror)
    control = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()
    direction = 1.0 if mirror else -1.0
    control._last_steer = candidate._last_steer = direction * 0.203

    control_action = control.act(observation)
    candidate_action = candidate.act(observation)

    assert control._corridor_plan is None
    assert candidate._corridor_plan is None
    assert 42 not in candidate._corridor_centers
    assert candidate._carry_steer_request == pytest.approx(direction * 0.442)
    assert control_action[0] == pytest.approx(direction * 0.133)
    assert candidate_action[0] == pytest.approx(direction * 0.221)
    np.testing.assert_array_equal(candidate_action[1:], control_action[1:])


def test_dropout_slew_keeps_turn_until_road_row_recovers():
    observation = _dropout_observation()
    control = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()
    control._last_steer = candidate._last_steer = -0.203

    control_steering = [float(control.act(observation)[0]) for _ in range(3)]
    candidate_steering = [float(candidate.act(observation)[0]) for _ in range(3)]

    assert control_steering == pytest.approx([-0.133, -0.063, 0.007])
    assert candidate_steering == pytest.approx([-0.221, -0.221, -0.221])


@pytest.mark.parametrize("row42,obstacle", [(True, True), (False, False)])
def test_existing_reference_or_no_obstacle_remains_v2_exact(row42, obstacle):
    observation = _dropout_observation(row42=row42, obstacle=obstacle)
    control = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()
    control._last_steer = candidate._last_steer = -0.203

    np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
    assert candidate._corridor_plan is None


@pytest.mark.parametrize("centers", [
    {50: 28.0, 46: 22.5, 38: 16.0},
    {54: 34.0, 50: 28.0, 42: 20.0},
    {54: 34.0, 50: 28.0},
    {54: 34.0, 34: 24.0, 30: 20.0},
])
def test_ineligible_road_geometry_remains_v2_exact(centers):
    control = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()
    obstacle = (55.5, 24.0, 34.0)
    raw = 0.09

    assert candidate._adjust_road_steering(
        steering=raw, straight=False, centers=centers, obstacle=obstacle,
    ) == control._adjust_road_steering(
        steering=raw, straight=False, centers=centers, obstacle=obstacle,
    )


def test_feasible_corridor_action_remains_v2_exact():
    observation = _planned_observation()
    control = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()

    np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
    assert control._corridor_plan is not None
    assert candidate._corridor_plan is not None


def test_reset_and_invalid_observation_preserve_v2_contract():
    candidate = _candidate()
    control = agent._FeasibleCorridorFallbackSpeedController()
    observation = _dropout_observation()
    candidate.act(observation)
    control.act(observation)
    candidate.reset()
    control.reset()

    np.testing.assert_array_equal(candidate.act(None), control.act(None))
    np.testing.assert_array_equal(candidate.act(observation), _candidate().act(observation))
    assert candidate._corridor_plan is None
