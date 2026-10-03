"""The v3 diagnostic must observe side-switch decisions without changing actions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools import trace_feasible_corridor_dev_v3 as trace


def test_side_switch_probe_records_pixels_and_preserves_action(monkeypatch) -> None:
    class Controller:
        SIDE_VETO_ROW_RADIUS = 13.0
        SIDE_VETO_ROAD_ROWS = 3

        def __init__(self):
            self._obstacle_side = -1.0
            self._corridor_bbox = (48, 42, 52, 49)
            self._corridor_edges = {43: (32.0, 61.0), 44: (32.0, 62.0), 45: (33.0, 62.0)}

        def _allow_obstacle_side_switch(self, *, obstacle_y, obstacle_x, candidate_side):
            return candidate_side > 0.0

    agent = SimpleNamespace(_forward_controller=Controller())
    expected_action = np.asarray([0.12, 0.0, 0.1], dtype=np.float32)

    def fake_act(actual_agent, observation):
        assert actual_agent is agent
        assert observation == "observed"
        assert actual_agent._forward_controller._allow_obstacle_side_switch(
            obstacle_y=44.0, obstacle_x=50.0, candidate_side=1.0,
        )
        return expected_action, {"road_centers": {"42": 41.0}}

    monkeypatch.setattr(trace, "BASE_ACT_WITH_DIAGNOSTICS", fake_act)
    original_method = agent._forward_controller._allow_obstacle_side_switch
    action, detail = trace.act_with_side_switches(agent, "observed")
    assert action is expected_action
    assert detail["side_switch_decisions"] == [{
        "obstacle_y": 44.0, "obstacle_x": 50.0,
        "candidate_side": 1.0, "current_side": -1.0,
        "allowed": True, "bbox": [48, 42, 52, 49],
        "nearest_edge_rows": [
            {"row": 44, "left": 32.0, "right": 62.0},
            {"row": 43, "left": 32.0, "right": 61.0},
            {"row": 45, "left": 33.0, "right": 62.0},
        ],
    }]
    assert "_allow_obstacle_side_switch" not in vars(agent._forward_controller)
    assert agent._forward_controller._allow_obstacle_side_switch.__func__ is original_method.__func__
