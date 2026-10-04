"""A missing current-frame road reference must not reverse a clear-road bend."""

import numpy as np
import pytest

import agent


def _candidate(*, hud_speed=None):
    controller_type = getattr(agent, "_ClearRoadRow42DropoutController", None)
    assert controller_type is not None
    controller = controller_type()
    controller._row42_current_hud_speed = hud_speed
    return controller


def _request(controller, centers, *, steering, straight=False, obstacle=None):
    return controller._adjust_road_steering(
        steering=steering, straight=straight, centers=centers,
        obstacle=obstacle,
    )


@pytest.mark.parametrize(
    "centers, inherited, corrected",
    [
        ({54: 35.0, 50: 31.0, 46: 27.0}, 0.084, -0.328),
        ({54: 48.0, 50: 52.0, 46: 56.0}, -0.072, 0.328),
    ],
)
def test_clear_road_missing_row42_keeps_measured_bend(
    centers, inherited, corrected,
):
    controller = _candidate(hud_speed=30.0)
    original = centers.copy()

    assert _request(controller, centers, steering=inherited) == pytest.approx(corrected)
    assert centers == original


def test_obstacle_preserves_parent_road_request(monkeypatch):
    controller = _candidate()
    centers = {54: 35.0, 50: 31.0, 46: 27.0}
    # The inherited obstacle-road hook can independently repair a dropout.
    # Isolate the new subclass's input to that hook to verify abstention.
    monkeypatch.setattr(
        agent._BoundedSideHoldController,
        "_adjust_road_steering",
        lambda self, *, steering, straight, centers, obstacle: steering,
    )

    assert _request(
        controller, centers, steering=0.084,
        obstacle=(33.0, 40.0, 41.5),
    ) == pytest.approx(0.084)


def test_low_speed_without_recent_obstacle_keeps_parent_request():
    centers = {54: 35.0, 50: 31.0, 46: 27.0}
    candidate = _candidate(hud_speed=29.0)
    parent = agent._BoundedSideHoldController()

    assert _request(candidate, centers, steering=0.084) == _request(
        parent, centers, steering=0.084,
    )


@pytest.mark.parametrize(
    "missing, expected", [(0, -0.328), (2, -0.328), (3, 0.084)],
)
def test_recent_obstacle_memory_qualifies_only_two_missing_decisions(
    missing, expected,
):
    candidate = _candidate(hud_speed=29.0)
    candidate._obstacle_side = 1.0
    candidate._obstacle_missing = missing

    assert _request(
        candidate, {54: 35.0, 50: 31.0, 46: 27.0}, steering=0.084,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "centers, straight",
    [
        ({54: 35.0, 50: 31.0, 46: 27.0, 42: 25.0}, False),
        ({54: 35.0, 50: 31.0}, False),
        ({54: 35.0, 46: 27.0}, False),
        ({54: 35.0, 50: 31.0, 46: 27.0}, True),
    ],
)
def test_present_reference_missing_near_row_or_straight_road_is_parent_exact(
    centers, straight,
):
    candidate = _candidate()
    parent = agent._BoundedSideHoldController()

    assert _request(
        candidate, centers, steering=0.084, straight=straight,
    ) == _request(parent, centers, steering=0.084, straight=straight)


def _clear_road_dropout_observation(speed=29.0):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    frame[46, 10:35] = 0.4
    frame[50, 16:40] = 0.4
    frame[54:61, 22:44] = 0.4
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.repeat(frame[None], 4, axis=0)


def test_high_camera_speed_enables_dropout_correction_with_inherited_steer_slew():
    observation = _clear_road_dropout_observation(speed=31.0)
    candidate = _candidate()
    parent = agent._BoundedSideHoldController()
    candidate._last_steer = parent._last_steer = -0.203

    candidate_action = candidate.act(observation)
    parent_action = parent.act(observation)

    assert candidate_action[0] == pytest.approx(-0.273)
    assert parent_action[0] == pytest.approx(-0.133)
    assert candidate_action[0] - (-0.203) == pytest.approx(-candidate.MAX_STEER_STEP)
    assert candidate.road_visible and parent.road_visible


def test_low_camera_speed_abstains_on_act_without_obstacle_memory():
    observation = _clear_road_dropout_observation(speed=29.0)
    candidate = _candidate()
    parent = agent._BoundedSideHoldController()
    candidate._last_steer = parent._last_steer = -0.203

    assert candidate.act(observation)[0] == pytest.approx(parent.act(observation)[0])


def test_current_frame_speed_replaces_prior_high_speed_qualification():
    candidate = _candidate()
    centers = {54: 35.0, 50: 31.0, 46: 27.0}

    candidate.act(_clear_road_dropout_observation(speed=31.0))
    assert _request(candidate, centers, steering=0.084) == pytest.approx(-0.328)

    candidate.act(_clear_road_dropout_observation(speed=29.0))
    assert _request(candidate, centers, steering=0.084) == pytest.approx(0.084)


def test_reset_clears_obstacle_context_before_low_speed_act():
    observation = _clear_road_dropout_observation(speed=29.0)
    candidate = _candidate()
    parent = agent._BoundedSideHoldController()
    candidate._obstacle_side = 1.0
    candidate._obstacle_missing = 1
    candidate.reset()
    candidate._last_steer = parent._last_steer = -0.203

    assert candidate.act(observation)[0] == pytest.approx(parent.act(observation)[0])
