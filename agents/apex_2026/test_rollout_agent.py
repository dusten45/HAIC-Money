"""Behavior checks using rendered-like pixel observations only."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np


def load_agent():
    path = Path(__file__).with_name('rollout_agent.py')
    assert path.exists(), 'Independent rollout controller has not been implemented'
    spec = importlib.util.spec_from_file_location('independent_rollout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent()


def observation(points=((42, 83), (42, 0)), speed=0.0, obstacle=None):
    frame = np.full((84, 84), 0.58, dtype=np.float32)
    cv2.polylines(frame, [np.array(points, np.int32)], False, 0.4, 21)
    if obstacle is not None:
        cv2.circle(frame, obstacle, 4, 0.92, -1)
    frame[59:68, 40:44] = 0.24
    frame[74:] = 0
    # Fractional coverage approximates antialiased speed bar.
    height = speed * 0.042
    for row in range(74, 82):
        frame[row, 11:13] = np.clip(80.0 - row, 0, height) - np.clip(79.0 - row, 0, height)
    return np.stack([frame] * 4)


def test_straight_road_accelerates_with_finite_controls():
    action = np.asarray(load_agent().act(observation()))
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert abs(action[0]) < 0.15
    assert action[1] > 0.5
    assert action[2] < 0.1


def test_left_and_right_bends_receive_opposite_steering():
    left = load_agent().act(observation(((42, 83), (42, 63), (18, 20), (5, 0)), speed=25))
    right = load_agent().act(observation(((42, 83), (42, 63), (66, 20), (79, 0)), speed=25))
    assert left[0] < -0.1
    assert right[0] > 0.1


def test_obstacle_changes_the_trajectory_before_contact():
    clear = load_agent().act(observation(speed=28))
    blocked = load_agent().act(observation(speed=28, obstacle=(42, 43)))
    assert abs(blocked[0]) > abs(clear[0]) + 0.1 or blocked[2] > clear[2] + 0.15


def test_fast_sharp_bend_brakes_more_than_straight():
    agent = load_agent()
    sharp = observation(((42, 83), (42, 57), (7, 44), (0, 44)), speed=70)
    turn = agent.act(sharp)
    straight = load_agent().act(observation(speed=70))
    assert turn[2] > straight[2] + 0.1
    assert turn[1] < straight[1]


def test_invalid_input_produces_finite_bounded_stop():
    action = np.asarray(load_agent().act(np.full((4, 84, 84), np.nan, np.float32)))
    assert np.isfinite(action).all()
    assert -1 <= action[0] <= 1
    assert 0 <= action[1] <= 1
    assert 0 <= action[2] <= 1
    assert action[1] == 0


def test_official_reset_accepts_initial_observation():
    agent = load_agent()
    agent.act(observation(((42, 83), (42, 63), (18, 20), (5, 0)), speed=25))
    agent.reset(observation())
    reset_action = agent.act(observation())
    np.testing.assert_array_equal(reset_action, load_agent().act(observation()))
