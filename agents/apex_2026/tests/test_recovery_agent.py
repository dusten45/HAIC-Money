"""Camera regressions for compact obstacles and a stationary blocked nose."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'recovery_agent.py'


def module():
    spec = importlib.util.spec_from_file_location('apex_recovery_test', SOURCE)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def stationary_observation():
    frame = np.full((84, 84), .63, np.float32)
    for row in range(74):
        left = int(round(38 - .25 * (row - 22)))
        frame[row, max(0, left):min(84, left + 20)] = .4
    frame[55:58, 40:44] = .671
    frame[74:] = 0
    return np.tile(frame, (4, 1, 1))


def test_round_obstacle_survives_a_nearer_elongated_curb_fragment():
    agent = module().Agent()
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[43:47, 39:42] = .671
    frame[60:62, 37:46] = .60
    frame[74:] = 0
    centers = agent._robust._road_centers(frame)
    obstacle = agent._robust._nearest_obstacle(frame, centers)
    assert obstacle is not None
    assert obstacle[0] < 50


def test_stationary_hazard_uses_the_open_left_side():
    agent = module().Agent()
    action = None
    for _ in range(12):
        action = agent.act(stationary_observation())
    assert agent.mode == 'recovery'
    assert action[0] <= -.25
    assert .10 <= action[1] <= .25
    assert action[2] == 0
    assert agent._metric.last_steer == float(action[0])
    assert agent._robust._last_steer == float(action[0])


def test_missing_local_clearance_does_not_start_stall_recovery():
    agent = module().Agent()
    observation = stationary_observation()
    # Missing asphalt is below the obstacle threshold, so the real hazard
    # remains detectable while its immediate road clearance is unknown.
    observation[:, 52:59, :39] = .53
    observation[:, 52:59, 44:] = .53
    for _ in range(12):
        agent.act(observation)
    assert agent._robust._corridor_bbox is not None
    assert agent.mode != 'recovery'


def test_recovery_has_a_finite_budget_and_resets():
    agent = module().Agent()
    modes = []
    for _ in range(45):
        action = agent.act(stationary_observation())
        modes.append(agent.mode)
        assert action.shape == (3,)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0])
        assert np.all(action <= [1, 1, 1])
    assert 'recovery' in modes
    assert any(mode != 'recovery' for mode in modes[30:40])
    agent.reset()
    assert agent._stall_frames == 0
    assert agent._recovery_remaining == 0
    assert agent._recovery_side == 0
    assert agent._recovery_cooldown == 0
    assert agent._metric.last_steer == 0
    assert agent._robust._last_steer == 0


def test_invalid_camera_during_recovery_returns_finite_bounded_action():
    agent = module().Agent()
    for _ in range(12):
        agent.act(stationary_observation())
    observation = stationary_observation()
    observation[-1, 0, 0] = np.nan
    action = agent.act(observation)
    assert np.isfinite(action).all()
    assert np.all(action >= [-1, 0, 0])
    assert np.all(action <= [1, 1, 1])
