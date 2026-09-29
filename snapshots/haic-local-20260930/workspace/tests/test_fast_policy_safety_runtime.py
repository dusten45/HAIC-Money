import numpy as np

from haic_agent.fast_policy_safety_runtime import FastPolicySafetyAgent


class _Dummy:
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def reset(self, observation):
        self.calls = 0

    def act(self, observation):
        self.calls += 1
        return self.action


def test_road_loss_switches_to_stable_and_brakes(monkeypatch):
    agent = FastPolicySafetyAgent(_Dummy((0.0, 0.8, 0.0)), _Dummy((0.2, 0.1, 0.0)))
    observation = np.zeros((4, 84, 84), dtype=np.float32)
    agent.reset(observation)
    monkeypatch.setattr("haic_agent.fast_policy_safety_runtime.road_centers", lambda frame: {})
    monkeypatch.setattr("haic_agent.fast_policy_safety_runtime.estimate_observation_speed",
                        lambda observation, frame: 60.0)
    action = agent.act(observation)
    assert np.allclose(action, (0.2, 0.0, 0.2))
    assert agent.protection_entries == 1
    assert agent.protected_decisions == 1
