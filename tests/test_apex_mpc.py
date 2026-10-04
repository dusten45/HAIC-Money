import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    assert path.is_file(), "trajectory planner is not implemented"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup():
    controller = load(ROOT / "agents/apex_2026/mpc_agent.py", "mpc_candidate").Agent()
    image = load(ROOT / "tests/test_apex_vision.py", "camera_helpers").camera()
    return controller, image


def test_vehicle_envelope_rejects_a_center_point_that_overhangs_the_edge():
    agent, observation = setup()
    road = agent._road(observation[-1])
    assert agent._trajectory_clear(np.array([0.0]), np.array([8.0]), np.array([0.0]), road, None)
    assert not agent._trajectory_clear(np.array([5.8]), np.array([8.0]), np.array([0.0]), road, None)


def test_vehicle_front_contact_with_expanded_obstacle_is_rejected():
    agent, observation = setup()
    road = agent._road(observation[-1])
    obstacle = (0.0, 18.0, 1.2)
    assert not agent._trajectory_clear(np.array([0.0]), np.array([14.5]), np.array([0.0]), road, obstacle)
    assert agent._trajectory_clear(np.array([-4.0]), np.array([14.5]), np.array([0.0]), road, obstacle)


def test_planner_accelerates_on_a_visible_clear_straight():
    agent, observation = setup()
    action = agent.act(observation)
    assert abs(action[0]) < 0.06
    assert action[1] > 0.25
    assert action[2] == 0


def test_road_edge_rollout_steers_toward_the_available_asphalt():
    agent, _ = setup()
    camera = load(ROOT / "tests/test_apex_vision.py", "camera_edge").camera
    action = agent.act(camera(center=48, speed=60))
    assert action[0] > 0
    assert np.isfinite(action).all()


def test_obstacle_rollout_keeps_the_road_and_selects_a_clear_side():
    agent, _ = setup()
    camera = load(ROOT / "tests/test_apex_vision.py", "camera_obstacle").camera
    action = agent.act(camera(speed=60, obstacle=(45, 30)))
    assert action[0] < 0
    assert action[1] == 0 or agent.last_plan_feasible


def test_reset_discards_previous_planner_obstacle():
    agent, observation = setup()
    camera = load(ROOT / "tests/test_apex_vision.py", "camera_reset").camera
    agent.act(camera(speed=60, obstacle=(45, 30)))
    agent.reset(observation)
    assert agent._planner_obstacle is None
    np.testing.assert_array_equal(agent.act(observation), setup()[0].act(observation))


def test_lane_change_can_pass_an_object_and_return_inside_the_straight():
    agent, observation = setup()
    agent.pass_side = -1.0
    steering, acceleration, feasible, _, _ = agent._rollouts(
        agent._road(observation[-1]), 60.0, -0.12, 0.0, 0.0, (0.0, 24.0, 1.2))
    assert feasible
    assert acceleration >= 0.0
    assert steering < 0.0


def test_rotated_front_corner_can_clear_the_actual_round_obstacle():
    agent, observation = setup()
    assert agent._trajectory_clear(np.array([-0.85]), np.array([3.5]),
                                   np.array([-0.48]), agent._road(observation[-1]),
                                   (0.55, 7.05, 1.2))


def test_an_immediately_blocked_body_does_not_throttle_into_the_object():
    agent, _ = setup()
    camera = load(ROOT / "tests/test_apex_vision.py", "camera_blocked").camera
    for _ in range(4):
        action = agent.act(camera(speed=40, obstacle=(42, 61)))
        assert action[1] == 0.0
        assert action[2] > 0.0
