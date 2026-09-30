from unittest.mock import patch

import numpy as np
import pytest

from agent import _ObservedCurveArbitrationController, _ObservedRoadSideCommitController


def _evidence(controller, centers):
    controller._adjust_road_steering(
        steering=0.0, straight=False, centers=centers, obstacle=(32.0, 40.0, 42.0)
    )


@pytest.mark.parametrize("mirror", [-1, 1])
@pytest.mark.parametrize("base,bias,near,far", [
    (-0.024, 0.13333333333333333, 40.0, 43.666666666666664),
    (-0.022, 0.224, 41.0, 44.0),
])
def test_observed_bend_overrules_opposing_recentering(mirror, base, bias, near, far):
    candidate = _ObservedCurveArbitrationController()
    centers = {54: 41.5 + mirror * (near - 41.5), 30: 41.5 + mirror * (far - 41.5)}
    _evidence(candidate, centers)
    result = candidate._adjust_obstacle_steering(
        base_steering=mirror * base, obstacle_bias=mirror * bias, straight=False
    )
    assert result == pytest.approx(mirror * (base + bias))
    assert candidate._pace_straight is False


@pytest.mark.parametrize("base,bias,straight,centers", [
    (-0.2, 0.24, False, {54: 40.0, 30: 30.0}),
    (0.2, -0.24, False, {54: 40.0, 30: 50.0}),
    (-0.2, -0.24, False, {54: 40.0, 30: 50.0}),
    (-0.2, 0.24, True, {54: 40.0, 30: 50.0}),
    (-0.2, 0.24, False, {}),
])
def test_supported_or_excluded_cases_remain_parent_exact(base, bias, straight, centers):
    candidate = _ObservedCurveArbitrationController()
    parent = _ObservedRoadSideCommitController()
    _evidence(candidate, centers)
    args = dict(base_steering=base, obstacle_bias=bias, straight=straight)
    assert candidate._adjust_obstacle_steering(**args) == parent._adjust_obstacle_steering(**args)


def test_flat_evidence_is_additive_and_previous_preview_guard_remains():
    candidate = _ObservedCurveArbitrationController()
    _evidence(candidate, {54: 40.0, 30: 40.0})
    candidate._last_steer = -0.05
    candidate._preview_transition_pending = True
    result = candidate._adjust_obstacle_steering(
        base_steering=-0.024, obstacle_bias=0.13, straight=False
    )
    assert result == pytest.approx(0.106)
    assert candidate._last_steer == 0.0


def test_evidence_is_current_frame_only_and_reset_clears_it():
    candidate = _ObservedCurveArbitrationController()
    _evidence(candidate, {54: 40.0, 30: 44.0})
    assert candidate._observed_bend_displacement == 4.0
    _evidence(candidate, {})
    assert candidate._observed_bend_displacement is None
    _evidence(candidate, {54: 40.0, 30: 44.0})
    candidate.act(None)
    assert candidate._observed_bend_displacement is None
    _evidence(candidate, {54: 40.0, 30: 44.0})
    candidate.reset()
    assert candidate._observed_bend_displacement is None
    assert candidate.__dict__ == _ObservedCurveArbitrationController().__dict__


def test_full_action_with_unsupported_recentering_remains_bounded():
    candidate = _ObservedCurveArbitrationController()
    centers = {54: 40.0, 50: 40.0, 46: 41.0, 42: 40.0, 38: 41.5, 34: 41.0, 30: 43.6666666667}
    observation = np.full((4, 84, 84), 0.1, dtype=np.float32)
    with patch.object(candidate, "_road_centers", return_value=centers), patch.object(
        candidate, "_nearest_obstacle", return_value=(32.0, 40.0, 42.333333)
    ):
        action = candidate.act(observation)
    assert action[0] > 0.0
    assert np.all(np.isfinite(action))
    assert np.all(action >= [-1, 0, 0])
    assert np.all(action <= [1, 1, 1])
    assert action[1] * action[2] == 0
