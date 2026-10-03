"""The fresh-screen regression probe may only replay completed, consumed cells."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from tools import evaluate_bare_generalization as evaluator
from tools import trace_ego_screen_regressions as trace


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/observed-ego-side-switch-generalization-v1.json").read_text())


def test_scope_accepts_exactly_the_two_consumed_regressions() -> None:
    trace.require_consumed_cell(PROTOCOL, 1, 1190129265)
    trace.require_consumed_cell(PROTOCOL, 1, 583088201)
    for track, seed in ((1, 2938666309), (2, 1190129265), (1, 352898933), (1, 595620866)):
        with pytest.raises(ValueError, match="two consumed screen cells"):
            trace.require_consumed_cell(PROTOCOL, track, seed)
    with pytest.raises(ValueError, match="exact integers"):
        trace.require_consumed_cell(PROTOCOL, True, 1190129265)
    altered = deepcopy(PROTOCOL)
    altered["partitions"]["screen"]["seeds"].append(352898933)
    with pytest.raises(ValueError, match="screen grid differs"):
        trace.require_consumed_cell(altered, 1, 1190129265)


def test_complete_screen_gate_requires_all_valid_receipts_and_matching_summary(tmp_path: Path) -> None:
    freeze = {"protocol_sha256": "test-protocol", "source_sha256": {"candidate": "test-candidate"}}
    for track, seed, repeat in evaluator.expected_cells(PROTOCOL, "screen"):
        for arm in evaluator.ARMS:
            evaluator.record_cell(tmp_path, freeze, {
                "partition": "screen", "arm": arm, "track_id": track,
                "seed": seed, "repeat": repeat, "finished": False,
                "progress": 0.5, "collision_count": 0, "damage": 0.0,
                "lap_time_ms": None, "retire_reason": "max_steps",
                "offtrack_samples": 0, "partial_offtrack_samples": 0,
                "action_trace_sha256": "a" * 64, "error": None,
            })
    summary = evaluator.report_phase(tmp_path, freeze, PROTOCOL, "screen")
    references = trace.require_completed_screen(tmp_path, freeze, PROTOCOL, summary, 1, 1190129265)
    assert set(references) == {"control", "candidate"}
    assert summary["canonical_cells"] == 32
    missing = evaluator.cell_path(tmp_path, "screen", "candidate", 2, 1190129265, 0)
    missing.unlink()
    with pytest.raises(ValueError, match="all completed screen receipts"):
        trace.require_completed_screen(tmp_path, freeze, PROTOCOL, summary, 1, 1190129265)
    evaluator.record_cell(tmp_path, freeze, {
        "partition": "screen", "arm": "candidate", "track_id": 2,
        "seed": 1190129265, "repeat": 0, "finished": False,
        "progress": 0.5, "collision_count": 0, "damage": 0.0,
        "lap_time_ms": None, "retire_reason": "max_steps",
        "offtrack_samples": 0, "partial_offtrack_samples": 0,
        "action_trace_sha256": "a" * 64, "error": None,
    })
    doctored = dict(summary, candidate_finishes=1)
    with pytest.raises(ValueError, match="screen summary differs"):
        trace.require_completed_screen(tmp_path, freeze, PROTOCOL, doctored, 1, 1190129265)


def test_diagnostic_capture_preserves_methods_and_records_branch_arguments() -> None:
    class Controller:
        IMAGE_CENTER = 41.5
        OBSTACLE_LOW = 0.54
        _obstacle_side = -1.0
        _observed_side_margins = (10.5, 8.5)

        def _road_centers(self, frame):
            return {42: 41.5, 54: 41.5}

        def _nearest_obstacle(self, frame, centers):
            return (43.5, 41.5, 41.5)

        def _adjust_road_steering(self, *, steering, straight, centers, obstacle):
            return steering + 0.01

        def _allow_obstacle_side_switch(self, *, obstacle_y, obstacle_x, candidate_side):
            return True

        def _adjust_obstacle_steering(self, *, base_steering, obstacle_bias, straight):
            return base_steering + obstacle_bias

    class Agent:
        def __init__(self):
            self._forward_controller = Controller()

        def act(self, observation):
            controller = self._forward_controller
            frame = observation[-1]
            centers = controller._road_centers(frame)
            obstacle = controller._nearest_obstacle(frame, centers)
            road = controller._adjust_road_steering(
                steering=0.02, straight=False, centers=centers, obstacle=obstacle,
            )
            controller._allow_obstacle_side_switch(
                obstacle_y=obstacle[0], obstacle_x=obstacle[1], candidate_side=1.0,
            )
            steer = controller._adjust_obstacle_steering(
                base_steering=road, obstacle_bias=0.12, straight=False,
            )
            return np.asarray([steer, 0.08, 0.0], dtype=np.float32)

    agent = Agent()
    observation = np.zeros((4, 84, 84), dtype=np.float32)
    observation[-1, 42:46, 40:44] = 0.8
    action, detail = trace.act_with_diagnostics(agent, observation)
    assert action.tolist() == pytest.approx([0.15, 0.08, 0.0])
    assert detail["obstacle"] == [43.5, 41.5, 41.5]
    assert detail["obstacle_box"] == {"bbox_inclusive": [40, 42, 43, 45], "area": 16}
    assert detail["road_centers"] == {"42": 41.5, "54": 41.5}
    assert detail["side_switch"] == {"obstacle_y": 43.5, "obstacle_x": 41.5, "candidate_side": 1.0, "allowed": True}
    assert detail["obstacle_steering"] == {"base": 0.03, "bias": 0.12, "straight": False, "output": pytest.approx(0.15)}
    for name in trace.INSTRUMENTED_METHODS:
        assert name not in vars(agent._forward_controller)


def test_diagnostic_capture_restores_methods_when_agent_fails() -> None:
    class Controller:
        OBSTACLE_LOW = 0.54

        def _road_centers(self, frame):
            return {}

        def _nearest_obstacle(self, frame, centers):
            return None

        def _adjust_road_steering(self, **kwargs):
            return 0.0

        def _allow_obstacle_side_switch(self, **kwargs):
            return False

        def _adjust_obstacle_steering(self, **kwargs):
            return 0.0

    class Agent:
        def __init__(self):
            self._forward_controller = Controller()

        def act(self, observation):
            self._forward_controller._road_centers(observation[-1])
            raise RuntimeError("synthetic policy failure")

    agent = Agent()
    with pytest.raises(RuntimeError, match="synthetic policy failure"):
        trace.act_with_diagnostics(agent, np.zeros((4, 84, 84), dtype=np.float32))
    assert all(name not in vars(agent._forward_controller) for name in trace.INSTRUMENTED_METHODS)
