"""Causal timing and geometric regressions for a research-only memory helper."""
import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "research/speed_20261005/camera_circle_memory_study.py"


def module():
    if not SOURCE.exists():
        return None
    spec = importlib.util.spec_from_file_location("circle_memory_study_test", SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def timing(mode, previous, current):
    study = module()
    if study is None:
        # Existing missing-pass transport uses the current HUD sample only.
        return current
    return study.motion_samples(previous, current, mode)


def test_current_memory_interval_uses_both_available_hud_endpoints():
    previous, current = np.array([60., -3.]), np.array([50., -1.])
    speed, yaw = timing("endpoint_mean", previous, current)
    assert speed == 55. and yaw == -2.


def test_next_forecast_uses_past_slope_without_future_observation():
    previous, current = np.array([60., -3.]), np.array([50., -1.])
    speed, yaw = timing("past_slope_forecast", previous, current)
    assert speed == 45. and yaw == 0.


def test_right_turn_transport_sends_stationary_ahead_point_left():
    study = module()
    if study is None:
        return
    shifted = study.transform_points(np.array([[0., 10.]]), .2, np.zeros(2))
    assert shifted[0, 0] < -1.9 and shifted[0, 1] > 9.7


def test_front_edge_and_body_interior_collisions_do_not_depend_on_corners():
    study = module()
    if study is None:
        return
    edge_circle = np.array([0., 3.])
    corners = np.array([[-1.6, -2.4], [1.6, -2.4], [-1.6, 2.6], [1.6, 2.6]])
    assert np.min(np.linalg.norm(corners-edge_circle, axis=1)) > 1.2
    assert study.rectangle_clearance(edge_circle, np.zeros(2), 0.) < 0.
    assert study.rectangle_clearance(np.zeros(2), np.zeros(2), 0.) < 0.
    assert study.rectangle_clearance(np.array([3., 0.]), np.zeros(2), np.pi/2) < 0.


def test_one_new_detection_does_not_erase_a_different_missing_circle():
    study = module()
    if study is None:
        return
    memory = study.CircleMemory()
    memory.update(np.array([[0., 20.], [0., 30.]]), 0., np.zeros(2), .1)
    memory.update(np.array([[0., 22.]]), 0., np.array([0., 8.]), .1)
    assert len(memory.slots) == 2
    np.testing.assert_allclose(sorted(s["center"][1] for s in memory.slots), [12., 22.])
    assert sorted(s["missed"] for s in memory.slots) == [0, 1]


def test_ambiguous_association_keeps_union_and_reports_uncertainty():
    study = module()
    if study is None:
        return
    memory = study.CircleMemory()
    memory.update(np.array([[-.3, 20.], [.3, 20.]]), 0., np.zeros(2), .1)
    memory.update(np.array([[0., 20.]]), 0., np.zeros(2), .1)
    assert memory.ambiguous and len(memory.slots) == 3
    assert sum(s["missed"] > 0 for s in memory.slots) == 2


def test_missing_age_grows_uncertainty_without_timed_hazard_erasure():
    study = module()
    if study is None:
        return
    memory = study.CircleMemory()
    memory.update(np.array([[0., 20.]]), 0., np.zeros(2), .1)
    for _ in range(6):
        memory.update(np.empty((0, 2)), 0., np.zeros(2), .1)
    assert len(memory.slots) == 1 and memory.slots[0]["missed"] == 6
    assert memory.slots[0]["uncertainty"] > .6
    memory.reset()
    assert memory.slots == []


def test_true_arc_projection_and_fold_ambiguity_are_preserved():
    study = module()
    if study is None:
        return
    path = np.array([[0., 0.], [0., 10.], [10., 10.]])
    projections = study.path_projections(path, np.array([8., 10.]), .1)
    assert len(projections) == 1 and abs(projections[0]["arc"]-18.) < 1e-12
    assert projections[0]["arc"] > 10.
    folded = np.array([[0., 0.], [0., 10.], [1., 10.], [1., 0.]])
    projections = study.path_projections(folded, np.array([.5, 5.]), .1)
    assert len(projections) == 2
    assert abs(projections[0]["arc"]-projections[1]["arc"]) > 10.
