from unittest.mock import patch

import numpy as np
import pytest

from agent import _CompoundClearingBrakeCarryController, _ObservedRoadTargetController


def _request(controller, centers, *, straight=False, obstacle=None):
    far = centers.get(42, controller.IMAGE_CENTER)
    near = centers.get(54, controller.IMAGE_CENTER)
    raw = 0.016 * (far - controller.IMAGE_CENTER) + 0.012 * (far - near)
    return controller._adjust_road_steering(
        steering=raw, straight=straight, centers=centers, obstacle=obstacle
    )


@pytest.mark.parametrize("near,mid,far", [(34.0, 28.5, 24.0), (34.5, 29.0, 24.5)])
@pytest.mark.parametrize("mirror", [False, True])
def test_measured_dropout_preserves_turn_before_obstacle_arbitration(near, mid, far, mirror):
    centers = {54: near, 50: mid, 46: far}
    direction = 1.0 if mirror else -1.0
    if mirror:
        centers = {row: 83.0 - center for row, center in centers.items()}
    original = centers.copy()
    control = _CompoundClearingBrakeCarryController()
    candidate = _ObservedRoadTargetController()
    obstacle = (54.2, 61.0 if mirror else 22.0, centers[54])
    old = _request(control, centers, obstacle=obstacle)
    new = _request(candidate, centers, obstacle=obstacle)
    old_final = control._adjust_obstacle_steering(
        base_steering=old, obstacle_bias=-direction * 0.24, straight=False
    )
    new_final = candidate._adjust_obstacle_steering(
        base_steering=new, obstacle_bias=-direction * 0.24, straight=False
    )
    assert old_final == pytest.approx(-direction * (0.33 if near == 34 else 0.324))
    assert new_final == pytest.approx(direction * (0.2 if near == 34 else 0.196))
    assert candidate._carry_steer_request == pytest.approx(new)
    assert centers == original
    assert 42 not in centers


@pytest.mark.parametrize("centers,straight", [
    ({54: 34.0, 50: 28.5, 46: 24.0, 42: 20.0}, False),
    ({50: 28.5, 46: 24.0, 38: 18.0}, False),
    ({54: 34.0, 46: 24.0}, False),
    ({54: 34.0, 50: 28.5, 46: 24.0}, True),
    ({54: 34.0, 52: 32.0, 50: 28.5}, False),
    ({}, False),
])
def test_excluded_geometry_remains_control_exact(centers, straight):
    control = _CompoundClearingBrakeCarryController()
    candidate = _ObservedRoadTargetController()
    assert _request(candidate, centers, straight=straight) == _request(
        control, centers, straight=straight
    )
    assert candidate.__dict__ == control.__dict__


def test_bracketed_missing_reference_uses_interpolation_for_nonmonotonic_road():
    centers = {54: 40.0, 46: 30.0, 38: 38.0, 30: 29.0}
    candidate = _ObservedRoadTargetController()
    result = _request(candidate, centers)
    assert result == pytest.approx(0.016 * (34.0 - 41.5) + 0.012 * (34.0 - 40.0))


def _observation():
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    frame[77:83, 10:13] = (0.27 + 0.085 * 32.0) / 18.0
    return np.tile(frame[None], (4, 1, 1))


def _act_with_geometry(controller, centers, obstacle):
    with patch.object(controller, "_road_centers", return_value=centers), patch.object(
        controller, "_nearest_obstacle", return_value=obstacle
    ):
        return controller.act(_observation())


def test_stateful_normal_dropout_road_loss_and_reset_contract():
    normal = {54: 34.0, 50: 28.5, 46: 24.0, 42: 20.0}
    dropout = {54: 34.0, 50: 28.5, 46: 24.0}
    obstacle = (54.2, 22.0, 34.0)
    control = _CompoundClearingBrakeCarryController()
    candidate = _ObservedRoadTargetController()
    for _ in range(5):
        np.testing.assert_array_equal(
            _act_with_geometry(candidate, normal, obstacle),
            _act_with_geometry(control, normal, obstacle),
        )
    original = dropout.copy()
    candidate_action = _act_with_geometry(candidate, dropout, obstacle)
    control_action = _act_with_geometry(control, dropout, obstacle)
    # The inherited slew can mask the first corrected request; the next
    # decision retains the bend while the old controller keeps unwinding.
    candidate_action = _act_with_geometry(candidate, dropout, obstacle)
    control_action = _act_with_geometry(control, dropout, obstacle)
    assert candidate_action[0] < control_action[0] < 0.0
    assert dropout == original
    for centers, current_obstacle in [(dropout, None), ({54: 34.0}, None), (normal, None)]:
        action = _act_with_geometry(candidate, centers, current_obstacle)
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))
        assert -1 <= action[0] <= 1
        assert np.all((action[1:] >= 0) & (action[1:] <= 1))
        assert action[1] * action[2] == 0
    candidate.reset()
    fresh = _ObservedRoadTargetController()
    assert candidate.__dict__ == fresh.__dict__
    np.testing.assert_array_equal(candidate.act(None), np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(
        _act_with_geometry(candidate, dropout, obstacle),
        _act_with_geometry(fresh, dropout, obstacle),
    )
    assert np.all(np.isfinite(candidate.act(None)))
