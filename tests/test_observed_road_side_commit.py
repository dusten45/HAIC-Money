from unittest.mock import patch

import numpy as np
import pytest

from agent import (
    _CompoundClearingBrakeCarryController,
    _ObservedRoadSideCommitController,
)


def _act(controller, obstacle):
    centers = {row: 41.5 for row in (54, 50, 46, 42, 38, 34, 30)}
    observation = np.full((4, 84, 84), 0.1, dtype=np.float32)
    with patch.object(controller, "_road_centers", return_value=centers), patch.object(
        controller, "_nearest_obstacle", return_value=obstacle
    ):
        action = controller.act(observation)
    assert np.all(np.isfinite(action))
    assert np.all(action >= [-1, 0, 0])
    assert np.all(action <= [1, 1, 1])
    assert action[1] * action[2] == 0
    return action


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("row,expected", [(43.999, -1), (44.0, 1), (51.45, 1), (52.0, 1)])
def test_near_side_is_committed_and_far_can_reselect(direction, row, expected):
    controller = _ObservedRoadSideCommitController()
    _act(controller, (26.0, 41.5 - direction * 8, 41.5))
    assert controller._obstacle_side == direction
    _act(controller, (row, 41.5 + direction * 8, 41.5))
    assert controller._obstacle_side == expected * direction


@pytest.mark.parametrize("row,expected", [(44.0, -1), (51.999, -1), (52.0, 1)])
def test_original_controller_retains_row52_boundary(row, expected):
    controller = _CompoundClearingBrakeCarryController()
    _act(controller, (26.0, 30.0, 41.5))
    _act(controller, (row, 50.0, 41.5))
    assert controller._obstacle_side == expected


def test_near_first_detection_and_latch_reset():
    controller = _ObservedRoadSideCommitController()
    _act(controller, (55.0, 30.0, 41.5))
    assert controller._obstacle_side == 1
    for miss in range(1, 5):
        _act(controller, None)
        assert controller._obstacle_missing == miss
        assert controller._obstacle_side == 1
    _act(controller, (55.0, 50.0, 41.5))
    assert controller._obstacle_side == 1
    assert controller._obstacle_missing == 0
    for _ in range(5):
        _act(controller, None)
    assert controller._obstacle_side == 0
    assert controller._obstacle_missing == 0
    _act(controller, (55.0, 50.0, 41.5))
    assert controller._obstacle_side == -1
    controller.reset()
    fresh = _ObservedRoadSideCommitController()
    assert controller.__dict__ == fresh.__dict__
    np.testing.assert_array_equal(
        _act(controller, (55.0, 30.0, 41.5)),
        _act(fresh, (55.0, 30.0, 41.5)),
    )
