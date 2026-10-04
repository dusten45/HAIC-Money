"""Camera and hazard regressions for the guarded envelope revision."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "guarded_envelope_agent.py"


def _module():
    spec = importlib.util.spec_from_file_location("apex_guarded_envelope_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _camera(*, sparse=False, bend=True, speed=70.0):
    frame = np.full((84, 84), 0.63, dtype=np.float32)
    rows = (69, 68, 55, 54, 43, 42, 31, 30, 19, 18, 9, 8) if sparse else range(4, 73)
    for row in rows:
        ahead = (63.0 - row) / 1.701
        side = 0.035 * max(ahead - 18.0, 0.0) ** 2 if bend else 0.0
        middle = 42.0 + 1.3608 * side
        lo, hi = max(0, round(middle - 9)), min(84, round(middle + 9) + 1)
        frame[row, lo:hi] = 0.40
    frame[73:] = 0.0
    top, bottom = 81.9 - 0.042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0.0, min(col + 1.0, 12.6) - max(float(col), 10.5))
            height = max(0.0, min(row + 1.0, bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


@pytest.mark.parametrize("sparse", [True, False])
def test_visible_localized_bend_brakes_with_sparse_and_dense_camera_support(sparse):
    agent = _module().Agent(cruise_speed=95.0, preview_time=0.23, braking_accel=60.0)
    action = agent.act(_camera(sparse=sparse))
    assert agent.mode == "metric"
    assert agent.last_speed > 65.0
    # The near-straight approach cannot erase the visible sharp bend. Its
    # local curvature and decision reserve must lower the approach target.
    assert agent.last_target < 60.0
    assert action[1] == 0.0
    assert action[2] > 0.10


def test_sparse_bend_reserves_more_braking_distance_at_higher_speed():
    metric = _module().Agent(cruise_speed=95.0, braking_accel=60.0)._metric
    observation = _camera(sparse=True)
    road = metric._road(observation[-1])
    at_rest = metric._curve_speed(road[0], road[1], speed=0.0)
    moving = metric._curve_speed(road[0], road[1], speed=70.0)
    assert moving < at_rest - 3.0


def _planned_hazard(agent, *, speed, target, robust_action):
    def metric(_observation):
        agent._metric.last_speed = speed
        agent._metric.last_target = 95.0
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def robust(_observation):
        agent._robust.road_visible = True
        agent._robust._obstacle_side = 1.0
        agent._robust._corridor_bbox = (36.0, 40.0, 40.0, 44.0)
        agent._robust._corridor_plan = {"target_speed": target}
        agent._robust._temporal_track = None
        agent._robust._temporal_assessment = None
        agent._robust._pace_speed = speed
        agent._robust._pace_effective_target = target
        return np.asarray(robust_action, dtype=np.float32)

    agent._metric.act = metric
    agent._metric._speed = lambda _frame: speed
    agent._robust.act = robust


def test_guided_hazard_keeps_robust_brake_and_reports_actual_speed_target():
    agent = _module().Agent()
    _planned_hazard(agent, speed=40.0, target=18.0, robust_action=(0.03, 0.0, 0.30))
    action = agent.act(_camera(bend=False))
    assert agent.mode == "guided"
    assert action[1] == 0.0 and action[2] >= 0.30
    assert agent.last_target == 18.0


def test_guided_hazard_target_brakes_even_without_explicit_robust_brake():
    agent = _module().Agent()
    _planned_hazard(agent, speed=40.0, target=18.0, robust_action=(0.03, 1.0, 0.0))
    action = agent.act(_camera(bend=False))
    assert action[1] == 0.0 and action[2] > 0.35
    assert agent.last_target == 18.0


def test_guided_corridor_retains_crawl_pedals_and_selected_steering():
    agent = _module().Agent()
    _planned_hazard(agent, speed=10.0, target=18.0, robust_action=(0.30, 0.035, 0.0))
    action = agent.act(_camera(bend=False))
    assert action[1] <= 0.035 + 1e-6 and action[2] == 0.0
    assert action[0] == np.float32(0.30)
    assert agent._metric.last_steer == agent._robust._last_steer == float(action[0])


def test_visible_horizon_uses_configured_braking_and_decision_reserve():
    metric = _module().Agent(cruise_speed=95.0, braking_accel=60.0)._metric
    observation = _camera(bend=False)
    forward = np.asarray([0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    metric._road = lambda _frame: (forward, np.zeros(6), 63.0 - forward * 1.701,
                                  np.full(6, 33.0), np.full(6, 51.0))
    metric.act(observation)
    reserve = 2.6 + 0.08 * metric.last_speed
    expected = np.sqrt(38.0 ** 2 + 2.0 * 60.0 * (20.0 - reserve))
    assert metric.last_target <= expected + 1e-6


def test_embedded_defaults_match_selected_explicit_parameters():
    agent = _module().Agent()
    assert agent._metric.cruise_speed == 95.0
    assert agent._metric.preview_time == 0.23
    assert agent._metric.braking_accel == 60.0


@pytest.mark.parametrize("observation", [
    [[0.0] * 84, [0.0] * 83], np.full((4, 84, 84), "bad"),
])
def test_ragged_or_nonnumeric_camera_recovers_with_finite_bounded_action(observation):
    action = _module().Agent().act(observation)
    assert np.isfinite(action).all()
    assert np.all(action >= [-1.0, 0.0, 0.0])
    assert np.all(action <= [1.0, 1.0, 1.0])


def test_dim_small_orange_core_cannot_shrink_the_physical_obstacle_radius():
    robust = _module().Agent()._robust
    robust._corridor_edges = {row: (30.0, 55.0) for row in range(22, 59)}
    bbox = (44, 36, 45, 37)
    row = 36.5 + 2.4 * 1.701
    # The official broad front hull bar extends laterally1.2m. At this
    # pose it intersects the fixed1.2m obstacle despite a dim2x2 bright core.
    lateral_separation = (44.5 - 41.5) / 1.3608
    assert lateral_separation - 1.2 - 1.2 < 0.0
    assert robust._corridor_hull_margin(row, 41.5, 39.0, 26, bbox) is None
