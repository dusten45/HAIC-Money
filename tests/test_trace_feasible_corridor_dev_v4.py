"""The v4 diagnostic observes temporal decisions without changing Agent actions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools import trace_feasible_corridor_dev_v4 as trace


def test_temporal_probe_records_visible_geometry_and_preserves_action(monkeypatch) -> None:
    class Controller:
        def __init__(self):
            self._obstacle_side = -1.0
            self._temporal_obstacle = (50.0, 40.0, -0.5)
            self._corridor_detection = (43.0, 50.5, 50.0)
            self._corridor_bbox = (48, 42, 53, 49)
            self._corridor_edges = {42: (32.0, 61.0), 43: (33.0, 61.0), 44: (33.0, 62.0)}

        def _allow_obstacle_side_switch(self, *, obstacle_y, obstacle_x, candidate_side):
            return False

    controller = Controller()
    agent = SimpleNamespace(_forward_controller=controller)
    expected_action = np.asarray([0.12, 0.0, 0.1], dtype=np.float32)

    def fake_act(actual_agent, observation):
        assert actual_agent is agent
        assert observation == "observed"
        assert not controller._allow_obstacle_side_switch(
            obstacle_y=43.0, obstacle_x=50.5, candidate_side=1.0,
        )
        return expected_action, {"road_centers": {"42": 41.0}}

    monkeypatch.setattr(trace, "BASE_ACT_WITH_DIAGNOSTICS", fake_act)
    original_method = controller._allow_obstacle_side_switch
    action, detail = trace.act_with_temporal_switches(agent, "observed")
    assert action is expected_action
    assert detail["temporal_side_switch_decisions"] == [{
        "obstacle_y": 43.0,
        "obstacle_x": 50.5,
        "candidate_side": 1.0,
        "current_side": -1.0,
        "allowed": False,
        "previous_obstacle": [50.0, 40.0, -0.5],
        "current_obstacle": [43.0, 50.5, 50.0],
        "bbox": [48, 42, 53, 49],
        "nearest_edge_rows": [
            {"row": 43, "left": 33.0, "right": 61.0},
            {"row": 42, "left": 32.0, "right": 61.0},
            {"row": 44, "left": 33.0, "right": 62.0},
        ],
    }]
    assert "_allow_obstacle_side_switch" not in vars(controller)
    assert controller._allow_obstacle_side_switch.__func__ is original_method.__func__
