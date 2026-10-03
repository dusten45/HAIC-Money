"""Behavior checks for the diagnostic near-center obstacle-side latch."""

import numpy as np
import pytest

import agent


def _candidate():
    candidate_type = getattr(agent, "_FeasibleCorridorTemporalSideController", None)
    assert candidate_type is not None
    return candidate_type()


def _observation(*, obstacle_x=None, obstacle_y=40, road=(36, 47)):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    left, right = road
    frame[22:62, left : right + 1] = 0.4
    if obstacle_x is not None:
        frame[obstacle_y - 2 : obstacle_y + 2, obstacle_x - 1 : obstacle_x + 2] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * 30.0) / 18.0
    return np.repeat(frame[None], 4, axis=0)


def test_same_obstacle_near_center_sign_reversal_keeps_prior_side():
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    left = _observation(obstacle_x=41)
    right = _observation(obstacle_x=42)

    np.testing.assert_array_equal(candidate.act(left), control.act(left))
    assert candidate._obstacle_side == control._obstacle_side == 1.0
    assert candidate._corridor_plan is None

    control_action = control.act(right)
    candidate_action = candidate.act(right)

    assert control._obstacle_side == -1.0
    assert candidate._obstacle_side == 1.0
    assert candidate._corridor_plan is None
    assert candidate_action[0] > control_action[0]
    np.testing.assert_array_equal(candidate_action[1:], control_action[1:])


@pytest.mark.parametrize("second_x", [40, 41, 43])
def test_no_near_center_sign_reversal_keeps_parent_action(second_x):
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    first = _observation(obstacle_x=41)
    second = _observation(obstacle_x=second_x)

    np.testing.assert_array_equal(candidate.act(first), control.act(first))
    np.testing.assert_array_equal(candidate.act(second), control.act(second))
    assert candidate._obstacle_side == control._obstacle_side


def test_first_sighting_keeps_parent_action():
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    observation = _observation(obstacle_x=42)

    np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
    assert candidate._obstacle_side == control._obstacle_side == -1.0


@pytest.mark.parametrize("second_x,road,expected_side", [
    (49, (43, 54), 1.0),  # Same object at the inclusive 8-pixel tracking bound.
    (50, (44, 55), -1.0),  # A 9-pixel jump starts a different object.
])
def test_tracking_distance_decides_whether_near_center_switch_is_vetoed(
    second_x, road, expected_side,
):
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    first = _observation(obstacle_x=41)
    second = _observation(obstacle_x=second_x, road=road)

    np.testing.assert_array_equal(candidate.act(first), control.act(first))
    control_action = control.act(second)
    candidate_action = candidate.act(second)

    assert control._obstacle_side == -1.0
    assert candidate._obstacle_side == expected_side
    if second_x == 50:
        np.testing.assert_array_equal(candidate_action, control_action)


@pytest.mark.parametrize("break_observation", [
    None,
    _observation(obstacle_x=None),
])
def test_road_loss_or_clear_road_forgets_previous_obstacle(break_observation):
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    first = _observation(obstacle_x=41)
    second = _observation(obstacle_x=42)

    np.testing.assert_array_equal(candidate.act(first), control.act(first))
    np.testing.assert_array_equal(
        candidate.act(break_observation), control.act(break_observation),
    )
    np.testing.assert_array_equal(candidate.act(second), control.act(second))
    assert candidate._obstacle_side == control._obstacle_side == -1.0


def test_reset_forgets_previous_obstacle():
    candidate = _candidate()
    candidate.act(_observation(obstacle_x=41))
    candidate.reset()

    fresh = _candidate()
    observation = _observation(obstacle_x=42)
    np.testing.assert_array_equal(candidate.act(observation), fresh.act(observation))
    assert candidate._obstacle_side == fresh._obstacle_side == -1.0


def test_fully_checked_pass_keeps_parent_actions():
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    for obstacle_x in (41, 42):
        observation = _observation(obstacle_x=obstacle_x, road=(29, 54))
        np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
        assert control._corridor_plan is not None
        assert candidate._corridor_plan is not None


def test_no_corridor_inherits_18_speed_cap():
    candidate = _candidate()
    control = agent._FeasibleCorridorRoadDropoutController()
    observation = _observation(obstacle_x=41)

    np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
    assert candidate._pace_effective_target == pytest.approx(18.0)
