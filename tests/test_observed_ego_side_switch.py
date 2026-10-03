"""A nearby obstacle may reverse a committed pass only on the ego's safe side."""

from unittest.mock import patch

import numpy as np
import pytest

import agent


FRAME = np.full((4, 84, 84), 0.1, dtype=np.float32)
FLAT_CENTERS = {row: 41.5 for row in (54, 50, 46, 42, 38, 34, 30)}


def _controller():
    controller_class = getattr(agent, "_ObservedEgoSideSwitchController", None)
    assert controller_class is not None
    return controller_class()


def _act(controller, obstacle, centers=FLAT_CENTERS):
    with patch.object(controller, "_road_centers", return_value=centers), patch.object(
        controller, "_nearest_obstacle", return_value=obstacle
    ):
        return controller.act(FRAME)


def _start_on_side(controller, side):
    obstacle_x = 30.0 if side > 0 else 50.0
    _act(controller, (26.0, obstacle_x, 41.5))
    assert controller._obstacle_side == side


@pytest.mark.parametrize(
    "row,obstacle_x,road_center,prior_side,expected_side",
    [
        (37.999, 28.2, 27.1, 1.0, -1.0),
        (38.0, 28.2, 27.1, 1.0, 1.0),
        (44.5, 41.2, 41.3125, -1.0, 1.0),
        (51.999, 41.2, 41.3125, -1.0, 1.0),
        (52.0, 41.2, 41.3125, -1.0, -1.0),
        (44.5, 41.8, 41.7, 1.0, -1.0),
        (44.5, 55.0, 56.0, -1.0, -1.0),
        (44.5, 41.5, 41.6, -1.0, -1.0),
    ],
)
def test_switch_uses_distance_and_ego_relative_side(
    row, obstacle_x, road_center, prior_side, expected_side
):
    controller = _controller()
    _start_on_side(controller, prior_side)
    action = _act(controller, (row, obstacle_x, road_center))
    assert controller._obstacle_side == expected_side
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.all(action >= [-1.0, 0.0, 0.0])
    assert np.all(action <= [1.0, 1.0, 1.0])


def test_consumed_seed42_centerline_crossing_does_not_reverse_pass():
    controller = _controller()
    observations = [
        ((27.8, 25.0, 28.10714340209961),
         {54: 40.0, 50: 41.5, 46: 39.35, 42: 37.0, 38: 33.5,
          34: 31.0, 30: 28.107142857142858}),
        ((34.2, 26.0, 26.706999492645267),
         {54: 40.0, 50: 37.5, 46: 34.5, 42: 32.0, 38: 29.5,
          34: 26.56, 30: 23.0}),
        ((38.5, 28.2, 27.100000000000001),
         {54: 38.0, 50: 34.5, 46: 32.0, 42: 29.5, 38: 26.76,
          34: 23.5, 30: 19.0}),
    ]
    for obstacle, centers in observations:
        action = _act(controller, obstacle, centers)
    assert controller._obstacle_side == 1.0
    assert action[0] == pytest.approx(-0.147, abs=1e-6)


def test_screen_centered_obstacle_can_reverse_a_late_pass():
    controller = _controller()
    controller._obstacle_side = -1.0
    action = _act(controller, (44.5, 41.2, 41.3125))
    assert controller._obstacle_side == 1.0
    assert action[0] > 0.0


def test_near_first_detection_misses_and_reset_preserve_latch_contract():
    controller = _controller()
    _act(controller, (55.0, 30.0, 41.5))
    assert controller._obstacle_side == 1.0
    for miss in range(1, 5):
        _act(controller, None)
        assert controller._obstacle_side == 1.0
        assert controller._obstacle_missing == miss
    _act(controller, (44.5, 28.2, 27.1))
    assert controller._obstacle_side == 1.0
    assert controller._obstacle_missing == 0
    for _ in range(5):
        _act(controller, None)
    assert controller._obstacle_side == 0.0
    _act(controller, (55.0, 50.0, 41.5))
    assert controller._obstacle_side == -1.0
    controller.reset()
    assert controller._obstacle_side == 0.0
    _act(controller, (55.0, 30.0, 41.5))
    assert controller._obstacle_side == 1.0


@pytest.mark.parametrize(
    "controller_class,expected_steer,expected_gas,expected_side",
    [
        (agent._CompoundClearingBrakeCarryController,
         [0.07, 0.0, 0.0, 0.07, 0.14], [0.08, 0.08, 0.11, 0.08, 0.08],
         [1.0, -1.0, -1.0, 1.0, 1.0]),
        (agent._ObservedMarginArbitrationController,
         [0.07, 0.14, 0.07, 0.14, 0.21], [0.08, 0.08, 0.08, 0.08, 0.08],
         [1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_existing_controllers_keep_frozen_side_and_action_sequence(
    controller_class, expected_steer, expected_gas, expected_side
):
    controller = controller_class()
    sequence = [
        (26.0, 30.0, 41.5),
        (44.5, 50.0, 41.5),
        None,
        (51.9, 30.0, 41.5),
        (52.0, 50.0, 41.5),
    ]
    actions = []
    sides = []
    for obstacle in sequence:
        action = _act(controller, obstacle)
        actions.append(action)
        sides.append(controller._obstacle_side)
    np.testing.assert_array_equal(
        np.asarray(actions),
        np.asarray([[steer, gas, 0.0] for steer, gas in zip(expected_steer, expected_gas)],
                   dtype=np.float32),
    )
    assert sides == expected_side
