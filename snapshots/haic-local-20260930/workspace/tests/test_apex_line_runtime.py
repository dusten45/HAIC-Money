import numpy as np

from haic_agent.apex_line_runtime import ApexLineAgent


class FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def reset(self, observation):
        pass

    def act(self, observation):
        return self.action.copy()


def observation(*, bend=0, obstacle=False):
    frames = np.zeros((4, 84, 84), dtype=np.float32)
    for index, row in enumerate((54, 50, 46, 42, 38, 34, 30)):
        center = 42 + bend * index
        frames[:, row, center - 10:center + 11] = 0.4
    if obstacle:
        frames[-1, 39:42, 43:47] = 0.9
    return frames


def test_apex_line_shifts_toward_visible_turn_with_fast_pedal():
    frames = observation(bend=-2)
    agent = ApexLineAgent(FixedAgent([0.0, 0.30, 0.0]), FixedAgent([0.0, 0.08, 0.0]))
    agent.reset(frames)

    action = agent.act(frames)

    assert action[0] < -0.05
    assert action[1] == np.float32(0.30)
    assert action[2] == 0.0
    assert agent.apex_decisions == 1


def test_apex_line_leaves_straight_and_obstacle_actions_alone():
    agent = ApexLineAgent(FixedAgent([0.0, 0.30, 0.0]), FixedAgent([0.0, 0.08, 0.0]))
    for frames in (observation(), observation(bend=-2, obstacle=True)):
        agent.reset(frames)
        action = agent.act(frames)
        assert action[0] == 0.0
        assert agent.apex_decisions == 0
