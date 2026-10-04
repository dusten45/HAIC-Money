"""Safety routing contracts for the independent standalone lane."""

import importlib.util
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "safety_agent.py"


def module():
    spec = importlib.util.spec_from_file_location("apex_safety_test", SOURCE)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def observation():
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[74:] = 0
    return np.tile(frame, (4, 1, 1))


def planned_controller(agent, *, hazard=True, speed=40., target=18.,
                       robust_action=(.03, 0., .3)):
    """A visible, valid corridor can still require a slow/braking approach."""
    def metric(_observation):
        agent._metric.last_speed = speed
        agent._metric.last_target = 100.
        return np.asarray([0., 1., 0.], np.float32)

    def robust(_observation):
        agent._robust.road_visible = True
        agent._robust._obstacle_side = 1. if hazard else 0.
        agent._robust._corridor_bbox = (36., 40., 40., 44.) if hazard else None
        agent._robust._corridor_plan = {"target_speed": target} if hazard else None
        agent._robust._temporal_track = None
        agent._robust._temporal_assessment = None
        agent._robust._pace_speed = speed
        agent._robust._pace_effective_target = target
        return np.asarray(robust_action, np.float32)

    agent._metric.act = metric
    agent._metric._speed = lambda _frame: speed
    agent._robust.act = robust


def test_planned_hazard_preserves_braking_and_reports_bounded_target():
    agent = module().Agent()
    planned_controller(agent)
    action = agent.act(observation())
    assert agent.mode == "guided"
    assert action[1] == 0.
    assert action[2] >= .3
    assert agent.last_target == 18.


def test_planned_hazard_target_limits_speed_without_robust_brake():
    agent = module().Agent()
    planned_controller(agent, robust_action=(.03, 1., 0.))
    action = agent.act(observation())
    assert action[1] == 0.
    assert action[2] > .35
    assert agent.last_target == 18.


def test_hazard_margin_is_explicit_and_does_not_discard_existing_brake():
    agent = module().Agent(hazard_speed_margin=30.)
    planned_controller(agent)
    action = agent.act(observation())
    assert action[1] == 0.
    assert action[2] >= .3
    assert agent.last_target == 48.


def test_absent_hazard_retains_fast_metric_pedals():
    agent = module().Agent()
    planned_controller(agent, hazard=False)
    action = agent.act(observation())
    assert agent.mode == "metric"
    assert action[1] == 1.
    assert action[2] == 0.
    assert agent.last_target == 100.


def test_uncertain_corridor_keeps_original_robust_action():
    agent = module().Agent()
    planned_controller(agent, robust_action=(.15, .02, .3))
    original = agent._robust.act

    def uncertain(value):
        action = original(value)
        agent._robust._corridor_plan = None
        return action

    agent._robust.act = uncertain
    action = agent.act(observation())
    assert agent.mode == "robust"
    np.testing.assert_array_equal(action, np.asarray([.15, .02, .3], np.float32))


def test_real_obstacle_memory_ages_out_and_clear_route_resumes():
    agent = module().Agent()
    value = observation()
    value[:, 43:48, 39:43] = .686
    agent.act(value)
    assert agent._robust._obstacle_side != 0.
    assert agent._metric.pass_side == 0.
    for _ in range(14):
        action = agent.act(observation())
    assert agent.mode == "metric"
    assert agent._robust._obstacle_side == 0.
    assert agent._robust._temporal_track is None
    assert action[1] > .8


def test_hazard_crawl_gas_cannot_be_replaced_by_fast_metric_pedals():
    agent = module().Agent()
    planned_controller(agent, speed=10., target=18., robust_action=(.03, .035, 0.))
    action = agent.act(observation())
    assert agent.mode == "guided"
    assert action[1] <= .035 + 1e-6
    assert action[2] == 0.


def test_planned_hazard_preserves_corridor_steering_and_syncs_memories():
    agent = module().Agent(transition_steer_step=.08)
    planned_controller(agent, speed=20., target=30., robust_action=(.3, 1., 0.))
    agent.mode = "metric"
    agent.last_steer = -.3
    agent._metric.last_steer = -.3
    agent._robust._last_steer = -.3
    action = agent.act(observation())
    # An opposite-side transitional angle has not been checked by the corridor
    # planner. Keep its selected command instead of averaging across routes.
    assert action[0] == np.float32(.3)
    assert agent._metric.last_steer == float(action[0])
    assert agent._robust._last_steer == float(action[0])


def test_clear_route_transition_still_has_configured_steering_slew():
    agent = module().Agent(transition_steer_step=.08)
    planned_controller(agent, hazard=False, speed=20., target=30.,
                       robust_action=(.3, 1., 0.))
    agent._robust._road_sweep = lambda _centers: 20.
    agent.mode = "metric"
    agent.last_steer = -.3
    agent._metric.last_steer = -.3
    agent._robust._last_steer = -.3
    action = agent.act(observation())
    assert agent.mode == "guided"
    assert np.isclose(action[0], -.22)
    assert agent._metric.last_steer == float(action[0])
    assert agent._robust._last_steer == float(action[0])


def test_lost_road_recovery_and_reset_remain_bounded():
    agent = module().Agent()
    agent.act(observation())
    empty = np.full((4, 84, 84), .63, np.float32)
    empty[:, 74:] = 0
    for _ in range(5):
        action = agent.act(empty)
    assert 0. < action[1] <= .1
    assert action[2] == 0.
    agent.reset()
    assert agent._metric.last_steer == 0.
    assert agent._robust._last_steer == 0.
    assert agent._robust._temporal_track is None


def test_invalid_frame_returns_finite_bounded_action():
    agent = module().Agent()
    value = observation()
    value[-1, 0, 0] = np.nan
    action = agent.act(value)
    assert action.dtype == np.float32
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert np.all(action >= [-1., 0., 0.])
    assert np.all(action <= [1., 1., 1.])
