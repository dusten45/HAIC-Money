"""Camera-level checks for the isolated clear-road pace candidate."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "pace_agent.py"


def _agent_type():
    spec = importlib.util.spec_from_file_location("apex_pace_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Agent


def _camera(*, center, speed):
    frame = np.full((84, 84), 0.63, dtype=np.float32)
    for row in range(73):
        middle = int(round(center))
        frame[row, middle - 9:middle + 10] = 0.40
    frame[73:] = 0.0
    top, bottom = 81.9 - 0.042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0.0, min(col + 1.0, 12.6) - max(float(col), 10.5))
            height = max(0.0, min(row + 1.0, bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_below_target_mild_turn_keeps_useful_acceleration():
    agent = _agent_type()()
    action = agent.act(_camera(center=46, speed=30))
    assert agent.mode == "metric"
    assert abs(action[0]) > 0.04
    assert agent.last_target > agent.last_speed + 3.0
    assert action[1] > 0.5
    assert action[2] == 0.0


def test_near_grip_limit_turn_keeps_throttle_bounded():
    agent = _agent_type()()
    action = agent.act(_camera(center=47, speed=90))
    assert agent.mode == "metric"
    demand = agent.last_speed ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert demand > 0.75
    assert agent.last_target > agent.last_speed
    assert action[1] <= 0.16 + 1e-6
    assert action[2] == 0.0


def test_gas_demand_gates_are_configurable_on_public_agent():
    agent = _agent_type()(low_gas_gate=0.20, high_gas_gate=0.40)
    action = agent.act(_camera(center=46, speed=30))
    assert agent.mode == "metric"
    demand = agent.last_speed ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert 0.20 < demand < 0.40
    assert 0.16 < action[1] <= 0.25
    assert action[2] == 0.0


def test_high_demand_below_old_steer_cutoff_caps_gas():
    agent = _agent_type()(cruise_speed=130)
    action = agent.act(_camera(center=47, speed=102))
    demand = agent.last_speed ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert agent.mode == "metric"
    assert 0.0 < abs(action[0]) < 0.04
    assert demand > 0.75
    assert agent.last_target > agent.last_speed
    assert action[1] <= 0.16 + 1e-6


def test_middle_demand_below_old_steer_cutoff_caps_gas():
    agent = _agent_type()(cruise_speed=130)
    action = agent.act(_camera(center=46, speed=100))
    demand = agent.last_speed ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert agent.mode == "metric"
    assert 0.0 < abs(action[0]) < 0.04
    assert 0.50 < demand < 0.75
    assert agent.last_target > agent.last_speed
    assert 0.16 < action[1] <= 0.25


def test_guided_turn_uses_configured_gas_gates_for_emitted_steering():
    agent = _agent_type()(cruise_speed=130, low_gas_gate=0.20,
                          high_gas_gate=0.40)

    def planned(_observation):
        robust = agent._robust
        robust.road_visible = True
        robust._obstacle_side = 1.0
        robust._corridor_bbox = (36, 40, 40, 44)
        robust._corridor_plan = {"side": 1.0}
        robust._temporal_track = None
        robust._temporal_assessment = None
        robust._pace_speed = 50.0
        robust._pace_effective_target = 100.0
        return np.asarray([0.10, 0.0, 0.0], dtype=np.float32)

    agent._robust.act = planned
    action = agent.act(_camera(center=42, speed=50))
    demand = 50.0 ** 2 * abs(np.tan(float(action[0]))) / (3.24 * 145.0)
    assert agent.mode == "guided"
    assert 0.40 < demand < 0.75
    assert action[1] <= 0.16 + 1e-6
    assert action[2] == 0.0


def test_reset_replays_first_clear_road_action():
    agent_type = _agent_type()
    value = _camera(center=46, speed=30)
    agent = agent_type()
    first = agent.act(value)
    agent.act(_camera(center=47, speed=90))
    agent.reset(value)
    np.testing.assert_array_equal(agent.act(value), first)


def test_invalid_frame_keeps_action_finite_and_bounded():
    agent = _agent_type()()
    value = _camera(center=46, speed=30)
    value[-1, 0, 0] = np.nan
    action = agent.act(value)
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= [-1.0, 0.0, 0.0])
    assert np.all(action <= [1.0, 1.0, 1.0])
