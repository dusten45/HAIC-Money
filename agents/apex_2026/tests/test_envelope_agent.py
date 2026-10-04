"""Physical curvature and quantization checks for the heading envelope lane."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "envelope_agent.py"


def _module():
    spec = importlib.util.spec_from_file_location("apex_envelope_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("slope", [0.0, 0.10, 0.25, -0.35])
def test_straight_pixel_staircase_keeps_cruise_speed(slope):
    metric = _module().Agent()._metric
    forward = np.arange(-3.5, 35.0, 2.0 / 1.701)
    true_side = slope * forward + 0.27
    quantized_side = np.round(true_side * metric.PX_X) / metric.PX_X
    assert np.max(abs(quantized_side - true_side)) <= 0.5 / metric.PX_X + 1e-12
    assert 95.0 <= metric._curve_speed(forward, quantized_side, speed=80.0) <= 100.0


@pytest.mark.parametrize("heading", [0.0, 0.35, 0.60])
def test_rotated_circular_arc_has_consistent_physical_corner_speed(heading):
    metric = _module().Agent()._metric
    radius = 30.0
    angles = np.linspace(-0.12, 0.82, 100)
    x = radius * (1.0 - np.cos(angles))
    y = radius * np.sin(angles)
    side = x * np.cos(heading) + y * np.sin(heading)
    forward = y * np.cos(heading) - x * np.sin(heading)
    target = metric._curve_speed(forward, side, speed=80.0)
    expected = np.sqrt(metric.lateral_accel * radius)
    assert 0.80 * expected <= target <= 1.20 * expected


def test_same_curve_requires_lower_speed_when_near_and_at_higher_current_speed():
    metric = _module().Agent()._metric
    forward = np.arange(-3.5, 35.0, 2.0 / 1.701)
    near = 0.035 * np.maximum(forward, 0.0) ** 2
    far = 0.035 * np.maximum(forward - 18.0, 0.0) ** 2
    near_target = metric._curve_speed(forward, near, speed=80.0)
    far_target = metric._curve_speed(forward, far, speed=80.0)
    assert 35.0 < near_target < 65.0
    assert far_target > near_target + 15.0
    assert metric._curve_speed(forward, far, speed=0.0) > far_target + 3.0


def test_sparse_curve_falls_back_to_finite_conservative_global_fit():
    metric = _module().Agent()._metric
    forward = np.asarray([0.0, 8.0, 16.0, 24.0, 32.0])
    assert 30.0 < metric._curve_speed(forward, 0.035 * forward ** 2) < 70.0
    assert metric._curve_speed(forward, np.zeros(5)) == 100.0


def test_nearly_identical_heading_supports_still_use_conservative_fallback():
    metric = _module().Agent()._metric
    forward = np.linspace(0.0, 10.1, 100)
    assert 30.0 < metric._curve_speed(forward, 0.035 * forward ** 2) < 70.0


def test_straight_camera_accelerates_and_invalid_camera_recovers_with_reset():
    agent = _module().Agent()
    frame = np.full((84, 84), 0.63, dtype=np.float32)
    frame[:73, 33:52] = 0.40
    frame[73:] = 0.0
    observation = np.repeat(frame[None], 4, axis=0)
    action = agent.act(observation)
    assert agent.mode == "metric"
    assert action[1] > 0.8 and action[2] == 0.0
    for _ in range(6):
        action = agent.act(np.full_like(observation, np.nan))
        assert np.isfinite(action).all()
        assert -1.0 <= action[0] <= 1.0
        assert 0.0 <= action[1] <= 1.0 and 0.0 <= action[2] <= 1.0
    agent.reset()
    assert agent.last_speed == agent.last_steer == 0.0
    assert agent._metric.lost_frames == 0
    assert agent._metric.last_target == 100.0


@pytest.mark.parametrize("parameter,value", [
    ("braking_accel", 0.0), ("braking_accel", np.nan),
    ("braking_accel", np.inf), ("heading_window", 5.0),
    ("heading_window", 20.0), ("heading_step", 0.0001),
    ("heading_step", 10.0),
])
def test_geometry_parameters_are_finite_and_bounded(parameter, value):
    with pytest.raises(ValueError):
        _module().Agent(**{parameter: value})
