"""Behavioral geometry and actuator tests for the small navigation candidate."""

import importlib.util
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "navigation_agent.py"


def agent_type():
    assert SOURCE.is_file(), "small camera navigation candidate is missing"
    spec = importlib.util.spec_from_file_location("apex_navigation_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def camera(speed=35.0, center=42.0, curvature=0.0, obstacle=None, half_width=9):
    frame = np.full((84, 84), 0.63, dtype=np.float32)
    for row in range(73):
        ahead = (63.0 - row) / 1.701
        middle = center + 1.3608 * 0.5 * curvature * max(ahead, 0.0) ** 2
        left, right = int(round(middle - half_width)), int(round(middle + half_width))
        frame[row, max(0, left):min(84, right + 1)] = 0.40
    if obstacle is not None:
        x, row = obstacle
        frame[row - 2:row + 3, x - 2:x + 3] = 0.686
    frame[73:] = 0.0
    top, bottom = 81.9 - 0.042 * speed, 81.9
    for row in range(73, 84):
        for col in range(10, 14):
            width = max(0.0, min(col + 1.0, 12.6) - max(float(col), 10.5))
            height = max(0.0, min(row + 1.0, bottom) - max(float(row), top))
            frame[row, col] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_curved_obstacle_route_follows_road_after_the_pass():
    agent = agent_type()()
    frame = camera(curvature=0.018, obstacle=(45, 40))[-1]
    road = agent._road(frame)
    obstacles = agent._obstacles(frame, road)
    assert len(obstacles) == 1
    plan = agent._plan_path(road, obstacles, speed=35.0)
    assert not plan["blocked"]
    assert plan["side"] == -1
    # A straight ego-to-pass ray would cut across the curved exit. The local
    # route must return to the measured centerline beyond the obstacle.
    at_pass = np.interp(obstacles[0][1], plan["forward"], plan["lateral"])
    assert at_pass < obstacles[0][0] - 2.4
    beyond = obstacles[0][1] + 13.0
    assert np.interp(beyond, plan["forward"], plan["lateral"]) > at_pass + 4.0


def test_only_local_path_rows_are_needed_to_certify_a_pass():
    agent = agent_type()()
    frame = camera(obstacle=(45, 46))[-1]
    road = agent._road(frame)
    # Remove distant road beyond the relevant obstacle exit, keeping the
    # entire local approach/pass. Missing distant rows cannot block it.
    keep = road[0] < 24.0
    road = tuple(values[keep] for values in road)
    plan = agent._plan_path(road, agent._obstacles(frame, road), speed=30.0)
    assert not plan["blocked"]
    assert plan["side"] == -1


def test_two_sides_without_hull_clearance_brake():
    agent = agent_type()()
    frame = camera(speed=45.0, obstacle=(42, 43), half_width=5)[-1]
    road = agent._road(frame)
    plan = agent._plan_path(road, [(0.0, 12.0, 1.2)], speed=45.0)
    assert plan["blocked"]
    action = agent.act(np.repeat(frame[None], 4, axis=0))
    assert action[1] == 0.0
    assert action[2] > 0.0


def test_low_speed_turn_accelerates_but_high_lateral_demand_brakes():
    low = agent_type()().act(camera(speed=25.0, center=47.0))
    high = agent_type()().act(camera(speed=100.0, center=50.0))
    assert low[0] > 0.03
    assert low[1] > 0.5 and low[2] == 0.0
    assert high[1] == 0.0 and high[2] > 0.0


def test_bend_steering_cannot_cut_back_across_the_selected_obstacle_side():
    agent = agent_type()()
    obstacle = (-3.49, 16.46, 1.2)
    steer = agent._avoidance_steer(-0.068, obstacle, side=1)
    # The front reaches the obstacle before the body center does. A command
    # that follows the near bend into the circle is not a feasible approach.
    entry = obstacle[1]-agent.HULL_HALF_LENGTH-obstacle[2]
    curvature = np.tan(steer)/agent.WHEELBASE
    predicted = 0.5*curvature*entry**2
    required = obstacle[2]+agent.HULL_HALF_WIDTH+agent.clearance
    assert predicted-obstacle[0] >= required-0.05


def test_near_pass_cannot_teleport_the_planned_hull_out_of_the_circle():
    agent = agent_type()()
    road = agent._road(camera()[-1])
    plan = agent._plan_path(road, [(0.0, 5.29, 1.2)], speed=30.0)
    assert abs(plan["lateral"][0]) < 0.01
    assert plan["blocked"]


def test_far_pass_starts_at_the_actual_pose_and_heading():
    agent = agent_type()()
    road = agent._road(camera(obstacle=(45, 40))[-1])
    plan = agent._plan_path(road, [(2.2, 13.52, 1.2)], speed=30.0)
    assert not plan["blocked"]
    assert abs(plan["lateral"][0]) < 0.01
    assert abs((plan["lateral"][1]-plan["lateral"][0])/
               (plan["forward"][1]-plan["forward"][0])) < 0.04


def test_insufficient_forward_road_uses_bounded_recovery():
    agent = agent_type()()
    observation = camera(speed=20.0)
    observation[:, :61, :] = 0.63
    action = agent.act(observation)
    assert agent.lost_frames == 1
    assert action[1] == 0.0 and action[2] > 0.0


def test_invalid_road_loss_actions_are_bounded_and_reset_clears_memory():
    agent = agent_type()()
    agent.act(camera(speed=40.0, obstacle=(45, 45)))
    for frame in (np.full((4, 84, 84), np.nan),
                  np.full((4, 84, 84), 0.63)):
        for _ in range(8):
            action = agent.act(frame)
            assert action.shape == (3,) and np.isfinite(action).all()
            assert -1.0 <= action[0] <= 1.0
            assert 0.0 <= action[1] <= 1.0 and 0.0 <= action[2] <= 1.0
    assert action[1] <= 0.090001 and action[2] == 0.0
    agent.reset()
    assert agent.pass_side == 0 and agent.lost_frames == 0
    np.testing.assert_array_equal(agent.act(camera()), agent_type()().act(camera()))
