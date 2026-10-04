"""Camera-sequence contracts for faster aligned clear-road acceleration."""
import numpy as np
import pytest

import agent


def camera(*, speed=40.0, shift=0.0, bend=1.0, obstacle=False):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    for row in range(22, 62):
        left = int(31 + shift + bend * (54 - row) / 4)
        frame[row, left:left+22] = 0.35
    frame[77:83, 10:13] = (speed * 0.085 + 0.27) / 18.0
    if obstacle:
        frame[50:54, 40:44] = 0.9
    return np.repeat(frame[None], 4, axis=0)


def candidate():
    return agent._SpeedOptimizedController()


def test_accelerates_an_aligned_clear_curve_without_changing_steering():
    observation = camera()
    old = agent._ClearRoadRow42DropoutController().act(observation)
    upgraded = candidate().act(observation)
    assert upgraded[0] == old[0]
    assert upgraded[1] > old[1] + 0.06
    assert upgraded[2] == old[2] == 0.0


@pytest.mark.parametrize("changes", [
    {"shift": 9}, {"bend": 3}, {"bend": 0}, {"obstacle": True}, {"speed": 70}, {"speed": 54},
])
def test_uncertain_curve_hazard_or_overspeed_keeps_validated_action(changes):
    observation = camera(**changes)
    old = agent._ClearRoadRow42DropoutController().act(observation)
    upgraded = candidate().act(observation)
    np.testing.assert_array_equal(upgraded, old)


def test_recent_obstacle_and_reset_do_not_leak_clear_curve_boost():
    runtime = candidate()
    old = agent._ClearRoadRow42DropoutController()
    for observation in (camera(obstacle=True), camera(), camera(), camera()):
        np.testing.assert_array_equal(runtime.act(observation), old.act(observation))
    runtime.reset()
    fresh = candidate().act(camera())
    np.testing.assert_array_equal(runtime.act(camera()), fresh)


def test_invalid_current_frame_abstains_and_reacquisition_is_repeatable():
    runtime = candidate()
    runtime.act(camera())
    invalid = camera()
    invalid[-1, 40, 40] = np.nan
    action = runtime.act(invalid)
    assert action[1] == 0.0
    assert action.dtype == np.float32 and action.shape == (3,)
    assert np.isfinite(action).all()
    runtime.reset()
    first = runtime.act(camera())
    runtime.reset()
    np.testing.assert_array_equal(runtime.act(camera()), first)


def test_sustained_boost_remains_finite_bounded_and_slew_limited():
    runtime = candidate()
    previous = 0.0
    for observation in [camera(speed=s) for s in (15, 25, 35, 45, 55, 65)]:
        action = runtime.act(observation)
        assert action.shape == (3,) and action.dtype == np.float32
        assert np.isfinite(action).all()
        assert -1 <= action[0] <= 1 and np.all((action[1:] >= 0) & (action[1:] <= 1))
        assert abs(float(action[0]) - previous) <= 0.070001
        previous = float(action[0])
