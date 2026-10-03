import numpy as np
import pytest

import agent


def _controller():
    controller = getattr(agent, "_ObservedMarginArbitrationController", None)
    assert controller is not None
    return controller()


def _frame(left, right, obstacle_x, obstacle_y):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    frame[22:62, left:right + 1] = 0.4
    frame[obstacle_y - 2:obstacle_y + 2, obstacle_x - 1:obstacle_x + 2] = 0.65
    return frame


def _request(controller, frame, centers, base, bias):
    obstacle = controller._nearest_obstacle(frame, centers)
    assert obstacle is not None
    controller._adjust_road_steering(
        steering=base, straight=False, centers=centers, obstacle=obstacle
    )
    return controller._adjust_obstacle_steering(
        base_steering=base, obstacle_bias=bias, straight=False
    )


def test_near_obstacle_with_insufficient_right_road_room_keeps_curve():
    controller = _controller()
    centers = {54: 35.75, 50: 36.0, 30: 42.5}
    frame = _frame(27, 44, obstacle_x=33, obstacle_y=52)
    assert _request(controller, frame, centers, -0.015, 0.24) == pytest.approx(-0.0075)


def test_clear_right_road_room_starts_and_retains_avoidance():
    controller = _controller()
    centers = {54: 40.0, 34: 42.5, 30: 43.6666666667}
    wide = _frame(32, 50, obstacle_x=39, obstacle_y=32)
    assert _request(controller, wide, centers, -0.05, 0.2) == pytest.approx(0.15)
    narrow = _frame(27, 44, obstacle_x=33, obstacle_y=52)
    assert _request(controller, narrow, centers, -0.05, 0.2) == pytest.approx(0.15)
    controller._adjust_road_steering(
        steering=-0.05, straight=False, centers=centers, obstacle=None
    )
    assert _request(controller, narrow, centers, -0.05, 0.2) == pytest.approx(-0.025)


def test_missing_ego_road_and_left_side_are_handled_symmetrically():
    controller = _controller()
    centers = {54: 43.0, 34: 40.0, 30: 39.0}
    left_room = _frame(32, 50, obstacle_x=45, obstacle_y=32)
    assert _request(controller, left_room, centers, 0.05, -0.2) == pytest.approx(-0.15)
    controller.reset()
    no_ego_road = _frame(44, 60, obstacle_x=50, obstacle_y=32)
    assert _request(controller, no_ego_road, centers, 0.05, -0.2) == pytest.approx(0.025)


@pytest.mark.parametrize("right,expected", [(48, -0.025), (49, 0.15)])
def test_passage_margin_boundary(right, expected):
    controller = _controller()
    frame = _frame(32, right, obstacle_x=39, obstacle_y=32)
    centers = {54: 40.0, 34: 42.5, 30: 43.6666666667}
    assert _request(controller, frame, centers, -0.05, 0.2) == pytest.approx(expected)


def test_full_camera_action_is_finite_bounded_and_exclusive():
    controller = _controller()
    frame = _frame(32, 50, obstacle_x=39, obstacle_y=32)
    observation = np.repeat(frame[None, :, :], 4, axis=0)
    action = controller.act(observation)
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.all(action >= [-1.0, 0.0, 0.0])
    assert np.all(action <= [1.0, 1.0, 1.0])
    assert action[1] * action[2] == 0.0
