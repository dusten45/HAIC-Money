import numpy as np

from haic_agent.predictive_corridor_runtime import PredictiveCorridorAgent


def test_dropout_preserves_turn_and_brakes_when_road_disappears(monkeypatch):
    agent = PredictiveCorridorAgent()
    agent._last_visible_steer = 0.4
    agent._missing_frames = 0
    monkeypatch.setattr("haic_agent.predictive_corridor_runtime.SpeedCoupledCorridorAgent.act",
                        lambda self, observation: np.array([0.0, 0.5, 0.0], dtype=np.float32))
    monkeypatch.setattr("haic_agent.predictive_corridor_runtime.current_frame",
                        lambda observation: np.zeros((84, 84), dtype=np.float32))
    monkeypatch.setattr("haic_agent.predictive_corridor_runtime.road_centers",
                        lambda frame: {})
    monkeypatch.setattr("haic_agent.predictive_corridor_runtime.PredictiveCorridorAgent.last_step_diagnostics",
                        lambda self: {"obstacle_y": None})
    action = agent.act(None)
    assert action[0] > 0.0
    assert action[1] == 0.0
    assert action[2] >= 0.2
    assert agent.dropout_count == 1
