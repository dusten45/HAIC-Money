"""Standalone hybrid camera routing and regression contracts."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "agents/apex_2026/hybrid_agent.py"


def module():
    spec = importlib.util.spec_from_file_location("apex_hybrid_test", SOURCE)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def observation(obstacle=False):
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[74:] = 0
    if obstacle:
        frame[43:48, 39:43] = .686
    return np.tile(frame, (4, 1, 1))


def test_clear_road_accelerates_using_metric_control():
    agent = module().Agent()
    action = agent.act(observation())
    assert agent.mode == "metric"
    assert action.dtype == np.float32
    assert action[1] > .8


def test_obstacle_route_has_only_robust_memory_and_ages_out():
    agent = module().Agent()
    action = agent.act(observation(True))
    assert np.isfinite(action).all()
    assert agent._metric.pass_side == 0.0
    assert agent._robust._obstacle_side != 0.0
    for _ in range(14):
        agent.act(observation())
    assert agent.mode == "metric"
    assert agent._robust._obstacle_side == 0.0
    assert agent._robust._temporal_track is None


def test_clear_metric_cannot_keep_false_obstacle_pass_bias():
    metric = module()._MetricClearAgent()
    metric.act(observation(True))
    for _ in range(3):
        action = metric.act(observation())
        assert action[0] == 0.0
        assert metric.pass_side == 0.0


def test_both_controllers_age_and_track_actual_emitted_steering():
    agent = module().Agent()
    counts = [0, 0]
    for index, controller in enumerate((agent._metric, agent._robust)):
        original = controller.act

        def counted(value, original=original, index=index):
            counts[index] += 1
            return original(value)

        controller.act = counted
    for value in (observation(), observation(True), observation()):
        action = agent.act(value)
        assert agent._metric.last_steer == float(action[0])
        assert agent._robust._last_steer == float(action[0])
    assert counts == [3, 3]


def test_emitted_steering_lateral_ceiling_brakes_excess_speed():
    action = module().Agent._steering_pedals(np.asarray([0, 1, 0], np.float32),
        steer=.25, speed=90., lateral_accel=145.)
    assert action[0] == np.float32(.25)
    assert action[1] == 0.0
    assert action[2] >= .65


def test_low_lateral_demand_keeps_clear_road_acceleration():
    action = module().Agent._steering_pedals(np.asarray([0, .9, 0], np.float32),
        steer=.03, speed=20., lateral_accel=145.)
    assert action[1] == np.float32(.9)
    assert action[2] == 0.0


def test_uncertain_corridor_preserves_full_robust_pedals():
    agent = module().Agent()

    def uncertain(_observation):
        agent._robust.road_visible = True
        agent._robust._obstacle_side = 1.
        agent._robust._corridor_bbox = (36., 40., 40., 44.)
        agent._robust._corridor_plan = None
        agent._robust._temporal_assessment = None
        return np.asarray([.15, .02, .3], np.float32)

    agent._robust.act = uncertain
    action = agent.act(observation())
    np.testing.assert_array_equal(action, np.asarray([.15, .02, .3], np.float32))
    assert agent.mode == "robust"


def test_lost_road_recovery_is_bounded_and_reset_clears_both():
    agent = module().Agent()
    agent.act(observation())
    empty = np.full((4, 84, 84), .63, np.float32)
    empty[:, 74:] = 0
    for _ in range(5):
        action = agent.act(empty)
    assert 0 < action[1] <= .10
    assert action[2] == 0.0
    agent.reset()
    assert agent._metric.last_steer == 0.0
    assert agent._robust._last_steer == 0.0
    assert agent._robust._temporal_track is None


def test_invalid_image_produces_finite_bounded_actions():
    agent = module().Agent()
    invalid = observation()
    invalid[-1, 0, 0] = np.nan
    action = agent.act(invalid)
    assert action.shape == (3,)
    assert action.dtype == np.float32
    assert np.isfinite(action).all()
    assert np.all(action >= [-1, 0, 0])
    assert np.all(action <= [1, 1, 1])
