"""Pixel-only behaviors required of the independent Apex controller."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).parents[1] / "agents" / "apex_2026" / "agent.py"


@pytest.fixture
def agent_type():
    assert SOURCE.is_file(), "independent camera controller is not implemented"
    spec = importlib.util.spec_from_file_location("apex_test_controller", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def camera(center=42.0, curvature=0.0, speed=0.0, obstacle=None):
    frame = np.full((84, 84), 0.65, dtype=np.float32)
    for y in range(73):
        ahead = (63.0 - y) / 1.701
        middle = center + 1.3608 * 0.5 * curvature * max(0.0, ahead) ** 2
        left, right = int(round(middle - 9)), int(round(middle + 9))
        frame[y, max(0, left):min(84, right + 1)] = 0.40
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2:y + 3, x - 2:x + 3] = 0.686
    frame[73:] = 0.0
    # Area-preserving antialiasing of the official speed indicator.
    top, bottom = 81.9 - 0.042 * speed, 81.9
    for y in range(73, 84):
        for x in range(10, 14):
            width = max(0.0, min(x + 1.0, 12.6) - max(float(x), 10.5))
            height = max(0.0, min(y + 1.0, bottom) - max(float(y), top))
            frame[y, x] = width * height
    return np.repeat(frame[None], 4, axis=0)


def test_speed_gauge_keeps_range_above_eighty(agent_type):
    low = agent_type()._speed(camera(speed=80)[-1])
    high = agent_type()._speed(camera(speed=120)[-1])
    assert 70 < low < 90
    assert high > 110


def test_centered_road_accelerates_without_steering(agent_type):
    action = agent_type().act(camera())
    assert abs(action[0]) < 0.02
    assert action[1] > 0.25
    assert action[2] == 0


def test_right_offset_road_steers_right(agent_type):
    action = agent_type().act(camera(center=47))
    assert action[0] > 0.03


def test_far_bend_brakes_before_road_disappears(agent_type):
    action = agent_type().act(camera(curvature=0.028, speed=110))
    assert action[1] == 0
    assert action[2] > 0


def test_obstacle_on_right_selects_left_pass(agent_type):
    clear = agent_type().act(camera(speed=40))
    obstacle = agent_type().act(camera(speed=40, obstacle=(45, 38)))
    assert obstacle[0] < clear[0] - 0.02


def test_near_obstacle_keeps_the_pass_through_hull_contact_zone(agent_type):
    action = agent_type().act(camera(speed=40, obstacle=(45, 62)))
    assert action[0] < -0.03


def test_pass_stays_selected_when_obstacle_crosses_front_and_rear_rows(agent_type):
    agent = agent_type()
    for row in (50, 58, 62, 66, 69):
        action = agent.act(camera(speed=40, obstacle=(45, row)))
        assert action[0] < -0.02, f"pass released at image row {row}"


def test_a_new_distant_obstacle_gets_its_own_pass_side(agent_type):
    agent = agent_type()
    assert agent.act(camera(speed=40, obstacle=(45, 50)))[0] < 0
    # A new pass may need two decisions to respect the steering slew bound.
    agent.act(camera(speed=40, obstacle=(38, 20)))
    action = agent.act(camera(speed=40, obstacle=(38, 20)))
    assert action[0] > 0.02


def test_long_curb_fragment_is_not_a_compact_obstacle(agent_type):
    observation = camera(speed=40)
    observation[:, 27:51, 44:47] = 0.686
    clear = agent_type().act(camera(speed=40))
    fragmented = agent_type().act(observation)
    np.testing.assert_array_equal(fragmented, clear)


def test_high_speed_lateral_correction_brakes_for_required_acceleration(agent_type):
    action = agent_type().act(camera(center=50, speed=90))
    assert action[1] == 0
    assert action[2] > 0


def test_lost_road_is_finite_and_eventually_crawls(agent_type):
    agent = agent_type()
    agent.act(camera(curvature=0.01))
    missing = np.full((4, 84, 84), 0.65, dtype=np.float32)
    actions = [agent.act(missing) for _ in range(15)]
    assert all(np.isfinite(action).all() for action in actions)
    assert actions[-1][1] > 0
    assert actions[-1][2] == 0


def test_reset_erases_previous_episode_commands(agent_type):
    agent = agent_type()
    agent.act(camera(center=50, speed=80))
    agent.reset(camera())
    np.testing.assert_array_equal(agent.act(camera()), agent_type().act(camera()))


def test_invalid_pixels_produce_valid_bounded_action(agent_type):
    action = agent_type().act(np.full((4, 84, 84), np.nan, dtype=np.float32))
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert -1 <= action[0] <= 1
    assert 0 <= action[1] <= 1
    assert 0 <= action[2] <= 1
