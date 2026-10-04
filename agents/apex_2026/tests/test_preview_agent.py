"""Distance-aware braking tests for the isolated camera preview candidate."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "preview_agent.py"


def _module():
    spec = importlib.util.spec_from_file_location("apex_preview_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bend(onset):
    ahead = np.arange(-3.0, 36.0, 1.0)
    side = 0.035 * np.maximum(ahead - onset, 0.0) ** 2
    return ahead, side


def _straight_camera(speed=30.0, center=42):
    frame = np.full((84, 84), 0.63, dtype=np.float32)
    frame[:73, center - 9:center + 10] = 0.40
    frame[73:] = 0.0
    top, bottom = 81.9 - 0.042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0.0, min(col + 1.0, 12.6) - max(float(col), 10.5))
            height = max(0.0, min(row + 1.0, bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_same_bend_allows_more_approach_speed_when_farther_away():
    metric = _module().Agent()._metric
    near = metric._curve_speed(*_bend(0.0))
    far = metric._curve_speed(*_bend(18.0))
    assert 35.0 < near < 50.0
    assert far > near + 15.0
    assert far <= metric.cruise_speed


def test_near_curve_still_limits_speed_and_stronger_brakes_help_far_approach():
    module = _module()
    weak = module.Agent(braking_accel=60.0)._metric
    strong = module.Agent(braking_accel=140.0)._metric
    ahead, side = _bend(0.0)
    expected_limit = np.sqrt(145.0 / 0.07)
    assert weak._curve_speed(ahead, side) <= expected_limit + 1.0
    assert strong._curve_speed(ahead, side) == pytest.approx(weak._curve_speed(ahead, side))
    assert strong._curve_speed(*_bend(18.0)) > weak._curve_speed(*_bend(18.0))


def test_constant_heading_straight_reaches_cruise_and_circle_is_rotation_stable():
    metric = _module().Agent()._metric
    ahead = np.arange(-3.0, 36.0, 1.0)
    assert metric._curve_speed(ahead, 1.5 * ahead + 3.0) == pytest.approx(100.0)
    # The same circle has constant physical curvature even when its tangent
    # is rotated in the camera; heading normalization avoids slope artifacts.
    radius = 30.0
    targets = []
    for heading in (0.0, 0.45):
        angles = np.linspace(-0.10, 0.90, 80)
        x = radius * (1.0 - np.cos(angles))
        y = radius * np.sin(angles)
        side = x * np.cos(heading) + y * np.sin(heading)
        forward = y * np.cos(heading) - x * np.sin(heading)
        targets.append(metric._curve_speed(forward, side))
    expected = np.sqrt(145.0 * radius)
    assert all(0.8 * expected < target < 1.15 * expected for target in targets)
    assert abs(targets[0] - targets[1]) < 8.0


def test_straight_action_invalid_recovery_and_reset_are_finite_and_bounded():
    agent = _module().Agent()
    action = agent.act(_straight_camera())
    assert agent.mode == "metric"
    assert agent.last_target == pytest.approx(100.0)
    assert action[1] > 0.8
    for _ in range(6):
        action = agent.act(np.full((4, 84, 84), np.nan, dtype=np.float32))
        assert np.isfinite(action).all()
        assert -1.0 <= action[0] <= 1.0
        assert 0.0 <= action[1] <= 1.0 and 0.0 <= action[2] <= 1.0
    agent.reset()
    assert agent.last_speed == agent.last_steer == 0.0
    assert agent._metric.last_target == 100.0
    assert agent._metric.lost_frames == 0
    assert agent.mode == "robust"


def test_high_speed_small_steer_respects_lateral_demand_gates():
    agent = _module().Agent(cruise_speed=125.0)
    action = agent.act(_straight_camera(speed=100.0, center=46))
    assert agent.mode == "metric"
    assert 0.0 < abs(float(action[0])) < 0.04
    demand = agent.last_speed ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert 0.50 < demand < 0.75
    assert action[1] <= 0.25
    assert action[2] == 0.0


def test_guided_mode_uses_configured_gas_demand_gates():
    agent = _module().Agent(low_gas_gate=0.20, high_gas_gate=0.40)
    # Keep a significant lateral demand below its physical speed ceiling.
    action = agent._steering_pedals([0.0, 1.0, 0.0], steer=0.1, speed=40.0,
        lateral_accel=145.0, low_gas_gate=agent._metric.low_gas_gate,
        high_gas_gate=agent._metric.high_gas_gate)
    demand = 40.0 ** 2 * np.tan(0.1) / (3.24 * 145.0)
    assert 0.20 < demand < 0.40
    assert action[1] == 0.25
    assert action[2] == 0.0


def test_single_pixel_road_row_quantization_does_not_brake_on_straight():
    module = _module()
    baseline = module.Agent()
    quantized = module.Agent()
    observation = _straight_camera(speed=80.0)
    reference = baseline.act(observation)
    # A one-pixel edge quantization on a sampled camera row changes its
    # inferred center by one pixel; it does not represent a road bend.
    observation[:, 59, :] = 0.63
    observation[:, 59, 34:53] = 0.40
    action = quantized.act(observation)
    assert baseline.mode == quantized.mode == "metric"
    assert reference[1] > 0.8 and reference[2] == 0.0
    assert quantized.last_target > 90.0
    assert action[1] > 0.8 and action[2] == 0.0


def test_sparse_visible_curve_still_imposes_conservative_speed_constraint():
    metric = _module().Agent()._metric
    ahead = np.asarray([0.0, 8.0, 16.0, 24.0, 32.0])
    side = 0.035 * ahead ** 2
    assert 30.0 < metric._curve_speed(ahead, side) < 70.0


def test_braking_reserve_includes_distance_travelled_during_one_decision():
    metric = _module().Agent()._metric
    road = _bend(18.0)
    stationary = metric._curve_speed(*road, speed=0.0)
    moving = metric._curve_speed(*road, speed=80.0)
    assert moving < stationary - 3.0
    assert moving > metric._curve_speed(*_bend(0.0), speed=80.0)


@pytest.mark.parametrize("parameter,value", [
    ("braking_accel", 0.0), ("braking_accel", np.inf),
    ("braking_accel", np.nan), ("curve_window", 3.0),
    ("curve_window", 40.0), ("curve_step", 0.0),
    ("curve_step", 0.0001), ("curve_step", 20.0),
])
def test_invalid_geometry_parameters_are_rejected(parameter, value):
    with pytest.raises(ValueError):
        _module().Agent(**{parameter: value})
