import numpy as np
import pytest

import agent


def _controller():
    controller = getattr(agent, "_ObservedCenterlineArbitrationController", None)
    assert controller is not None
    return controller()


def _request(near, far, base, bias):
    controller = _controller()
    controller._adjust_road_steering(
        steering=base,
        straight=False,
        centers={54: near, 30: far},
        obstacle=(41.5, 31.2, 34.125),
    )
    return controller._adjust_obstacle_steering(
        base_steering=base, obstacle_bias=bias, straight=False
    )


@pytest.mark.parametrize("mirror", [-1.0, 1.0])
@pytest.mark.parametrize("near,far,base,bias,expected", [
    (35.0, 35.5, -0.130, 0.240, -0.065),
    (38.5, 38.5, -0.024, 0.216, -0.012),
    (40.0, 43.6666666667, -0.024, 0.1333333333, 0.1093333333),
])
def test_obstacle_arbitration_requires_observed_road_crossing(
    mirror, near, far, base, bias, expected
):
    center = 41.5
    result = _request(
        center + mirror * (near - center),
        center + mirror * (far - center),
        mirror * base,
        mirror * bias,
    )
    assert result == pytest.approx(mirror * expected)


def test_one_pixel_bend_and_crossing_are_required():
    assert _request(40.7, 41.6, -0.05, 0.2) == pytest.approx(-0.025)
    assert _request(41.4, 42.6, -0.05, 0.2) == pytest.approx(0.15)


def test_no_observed_far_row_keeps_curve_retention():
    controller = _controller()
    result = controller._adjust_obstacle_steering(
        base_steering=-0.05, obstacle_bias=0.2, straight=False
    )
    assert result == pytest.approx(-0.025)


@pytest.mark.parametrize("mirror", [-1.0, 1.0])
def test_observed_supported_choice_remains_during_continuous_obstacle(mirror):
    controller = _controller()
    obstacle = (42.0, 40.0, 41.0)

    def observe(near, far, visible=True):
        controller._adjust_road_steering(
            steering=-0.024 * mirror,
            straight=False,
            centers={54: 41.5 + mirror * (near - 41.5),
                     30: 41.5 + mirror * (far - 41.5)},
            obstacle=obstacle if visible else None,
        )

    def action(bias=0.1333333333):
        return controller._adjust_obstacle_steering(
            base_steering=-0.024 * mirror,
            obstacle_bias=bias * mirror,
            straight=False,
        )

    observe(40.0, 43.6666666667)
    assert action() == pytest.approx(0.1093333333 * mirror)
    observe(38.5, 38.5)
    assert action() == pytest.approx(0.1093333333 * mirror)
    observe(38.5, 38.5, visible=False)
    observe(38.5, 38.5)
    assert action() == pytest.approx(-0.012 * mirror)
    controller.reset()
    observe(38.5, 38.5)
    assert action() == pytest.approx(-0.012 * mirror)


def test_side_reversal_releases_previous_centerline_choice():
    controller = _controller()
    controller._adjust_road_steering(
        steering=-0.05, straight=False,
        centers={54: 40.0, 30: 44.0}, obstacle=(42.0, 40.0, 41.0),
    )
    assert controller._adjust_obstacle_steering(
        base_steering=-0.05, obstacle_bias=0.2, straight=False
    ) == pytest.approx(0.15)
    controller._adjust_road_steering(
        steering=0.05, straight=False,
        centers={54: 43.0, 30: 43.0}, obstacle=(43.0, 44.0, 42.0),
    )
    assert controller._adjust_obstacle_steering(
        base_steering=0.05, obstacle_bias=-0.2, straight=False
    ) == pytest.approx(0.025)


def test_lost_road_clears_choice_and_actions_remain_bounded():
    controller = _controller()
    controller._adjust_road_steering(
        steering=-0.05, straight=False,
        centers={54: 40.0, 30: 44.0}, obstacle=(42.0, 40.0, 41.0),
    )
    assert controller._adjust_obstacle_steering(
        base_steering=-0.05, obstacle_bias=0.2, straight=False
    ) == pytest.approx(0.15)
    observation = np.full((4, 84, 84), 0.7, dtype=np.float32)
    observation[:, 30:55, 30:50] = 0.4
    observation[:, 40:44, 34:37] = 0.65
    first_action = controller.act(observation)
    assert controller._centerline_override_side == 1.0
    for action in (first_action, controller.act(None)):
        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert np.all(action >= [-1.0, 0.0, 0.0])
        assert np.all(action <= [1.0, 1.0, 1.0])
        assert action[1] * action[2] == 0.0
    assert controller.road_visible is False
    controller._adjust_road_steering(
        steering=-0.05, straight=False,
        centers={54: 38.5, 30: 38.5}, obstacle=(42.0, 38.0, 38.5),
    )
    assert controller._adjust_obstacle_steering(
        base_steering=-0.05, obstacle_bias=0.2, straight=False
    ) == pytest.approx(-0.025)
