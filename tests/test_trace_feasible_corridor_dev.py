"""Diagnostic replay must preserve its frozen consumed-cell boundary."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import compare_feasible_corridor_dev as compare
from tools import trace_feasible_corridor_dev as trace


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/feasible-corridor-dev-v1.json").read_text())


def test_scope_accepts_only_preconsumed_mechanism_cells() -> None:
    trace.require_consumed_cell(PROTOCOL, 3, 3857792434)
    trace.require_consumed_cell(PROTOCOL, 1, 3857792434)
    trace.require_consumed_cell(PROTOCOL, 1, 17)
    trace.require_consumed_cell(PROTOCOL, 1, 1190129265)
    for track, seed in ((2, 3857792434), (3, 2786341258),
                        (3, 4257225385), (3, 3654224474)):
        with pytest.raises(ValueError, match="outside"):
            trace.require_consumed_cell(PROTOCOL, track, seed)
    with pytest.raises(ValueError, match="exact integers"):
        trace.require_consumed_cell(PROTOCOL, True, 3857792434)
    edited = deepcopy(PROTOCOL)
    edited["mechanism_cells"] = [[3, 3857792434]]
    with pytest.raises(ValueError, match="frozen mechanism grid"):
        trace.require_consumed_cell(edited, 3, 3857792434)


def test_both_frozen_worker_receipts_are_required(tmp_path: Path) -> None:
    identity = {"protocol_sha256": "fixed", "source_sha256": {"baseline": "b", "candidate": "c"}}
    for arm in compare.ARMS:
        compare.record_cell(tmp_path, identity, {
            "arm": arm, "track_id": 3, "seed": 3857792434,
            "receipt_origin": "COLD_WORKER", "error": None,
            "action_trace_sha256": "a" * 64,
        })
    rows, receipt_hashes = trace.require_frozen_receipts(tmp_path, identity, 3, 3857792434)
    assert set(rows) == {"baseline", "candidate"}
    assert set(receipt_hashes) == {"baseline", "candidate"}
    assert all(len(value) == 64 for value in receipt_hashes.values())
    compare.cell_path(tmp_path, "candidate", 3, 3857792434).unlink()
    with pytest.raises(ValueError, match="both completed development receipts"):
        trace.require_frozen_receipts(tmp_path, identity, 3, 3857792434)


def test_receipt_gate_rejects_reused_or_failed_worker(tmp_path: Path) -> None:
    identity = {"protocol_sha256": "fixed"}
    for arm, origin in (("baseline", "EGO_SCREEN_CONTROL"), ("candidate", "COLD_WORKER")):
        compare.record_cell(tmp_path, identity, {
            "arm": arm, "track_id": 3, "seed": 3857792434,
            "receipt_origin": origin, "error": None,
            "action_trace_sha256": "b" * 64,
        })
    with pytest.raises(ValueError, match="original cold worker"):
        trace.require_frozen_receipts(tmp_path, identity, 3, 3857792434)


def test_capture_records_actual_candidate_plan_and_restores_methods() -> None:
    class Controller:
        OBSTACLE_LOW = 0.54
        _corridor_bbox = None
        _corridor_plan = None
        _corridor_previous = None
        _corridor_misses = 0
        _pace_sweep = 3.0
        _target_speed = 20.0

        def _road_centers(self, frame):
            return {42: 41.5, 54: 40.0}

        def _nearest_obstacle(self, frame, centers, spans=None):
            self._corridor_bbox = (40, 42, 43, 45)
            return (43.5, 41.5, 41.5)

        def _corridor_candidate(self, side):
            return (36.0, 1.25, 47) if side < 0 else None

        def _adjust_road_steering(self, *, steering, straight, centers, obstacle):
            return steering + 0.01

        def _adjust_obstacle_steering(self, *, base_steering, obstacle_bias, straight):
            return base_steering + obstacle_bias

    class Agent:
        def __init__(self):
            self._forward_controller = Controller()

        def act(self, observation):
            controller = self._forward_controller
            centers = controller._road_centers(observation[-1])
            controller._nearest_obstacle(observation[-1], {}, None)
            controller._corridor_candidate(-1.0)
            controller._corridor_candidate(1.0)
            road = controller._adjust_road_steering(
                steering=-0.2, straight=False, centers={}, obstacle=(43.5, 41.5, 41.5),
            )
            steering = controller._adjust_obstacle_steering(
                base_steering=road, obstacle_bias=0.07, straight=False,
            )
            controller._corridor_plan = {
                "side": -1.0, "target_x": 36.0, "clearance_px": 1.25,
                "bbox": (40, 42, 43, 45),
                "control_waypoints": ((54, 39.5), (47, 36.0)),
                "edges": {42: (30.0, 53.0)},
            }
            return np.asarray([steering, 0.08, 0.0], dtype=np.float32)

    agent = Agent()
    action, detail = trace.act_with_diagnostics(agent, np.zeros((4, 84, 84), dtype=np.float32))
    assert action.tolist() == pytest.approx([-0.12, 0.08, 0.0])
    assert detail["obstacle"] == [43.5, 41.5, 41.5]
    assert detail["obstacle_bbox"] == [40, 42, 43, 45]
    assert detail["corridor_candidates"] == [
        {"side": -1.0, "target_x": 36.0, "clearance_px": 1.25, "near_y": 47.0},
        {"side": 1.0, "target_x": None, "clearance_px": None, "near_y": None},
    ]
    assert detail["corridor_plan"]["control_waypoints"] == [[54.0, 39.5], [47.0, 36.0]]
    assert detail["corridor_plan"]["edges"] == {"42": [30.0, 53.0]}
    assert detail["road_centers"] == {"42": 41.5, "54": 40.0}
    assert detail["road_sweep"] == 3.0
    assert detail["road_steering"] == {"input": -0.2, "straight": False, "output": pytest.approx(-0.19)}
    assert detail["obstacle_steering"] == {
        "base": pytest.approx(-0.19), "bias": 0.07,
        "straight": False, "output": pytest.approx(-0.12),
    }
    assert "_nearest_obstacle" not in vars(agent._forward_controller)
    assert "_corridor_candidate" not in vars(agent._forward_controller)
    assert "_adjust_road_steering" not in vars(agent._forward_controller)
    assert "_adjust_obstacle_steering" not in vars(agent._forward_controller)
    assert "_road_centers" not in vars(agent._forward_controller)


def test_exact_replay_comparison_rejects_changed_action_hash_or_outcome() -> None:
    action = np.asarray([-0.1, 0.08, 0.0], dtype=np.float32)
    digest = hashlib.sha256(action.tobytes()).hexdigest()
    receipt = {
        "finished": False, "progress": 0.5, "lap_time_ms": None,
        "collision_count": 1, "damage": 0.2, "retire_reason": "off_track",
        "steps": 1, "action_trace_sha256": digest,
        "offtrack_samples": 0, "partial_offtrack_samples": 0,
    }
    episode = {
        "summary": {key: receipt[key] for key in (
            "finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason")},
        "action_trace_sha256": digest,
        "steps": [{"wheels_on_road": 4}],
    }
    trace.verify_exact_replay(episode, receipt)
    changed = deepcopy(episode)
    changed["action_trace_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="action trace"):
        trace.verify_exact_replay(changed, receipt)
    changed = deepcopy(episode)
    changed["summary"]["collision_count"] = 2
    with pytest.raises(RuntimeError, match="collision_count"):
        trace.verify_exact_replay(changed, receipt)
