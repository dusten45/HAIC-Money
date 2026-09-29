import numpy as np

from haic_agent.fast_pedal_stable_steer_runtime import FastPedalStableSteerAgent


class FixedAgent:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def reset(self, observation):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return self.action.copy()


def test_clear_road_uses_fast_pedal_but_stable_steering():
    road = np.full((4, 84, 84), 0.3, dtype=np.float32)
    road[:, 77:83, 10:13] = 0.0
    fast = FixedAgent([-0.2, 0.30, 0.0])
    stable = FixedAgent([0.15, 0.08, 0.0])
    agent = FastPedalStableSteerAgent(fast, stable)
    agent.reset(road)
    assert np.allclose(agent.act(road), [0.15, 0.30, 0.0])
    assert fast.calls == stable.calls == 1


def test_missing_road_keeps_stable_action():
    no_road = np.zeros((4, 84, 84), dtype=np.float32)
    fast = FixedAgent([-0.2, 0.30, 0.0])
    stable = FixedAgent([0.15, 0.08, 0.0])
    agent = FastPedalStableSteerAgent(fast, stable)
    agent.reset(no_road)
    assert np.allclose(agent.act(no_road), stable.action)
