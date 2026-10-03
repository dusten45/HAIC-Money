"""The telemetry probe must stay inside already consumed screen cells."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from local_simulator.session import SimulationSession
from local_simulator.simulation_types import RunStep
from tools import trace_consumed_screen as trace
from tools import evaluate_bare_generalization as evaluator


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/observed-margin-generalization-v1.json").read_text())


def test_scope_accepts_only_original_screen_grid() -> None:
    trace.require_consumed_screen_cell(PROTOCOL, 1, 3857792434)
    trace.require_consumed_screen_cell(PROTOCOL, 3, 3857792434)
    with pytest.raises(ValueError, match="consumed screen"):
        trace.require_consumed_screen_cell(PROTOCOL, 1, 725991793)  # sealed confirmation
    with pytest.raises(ValueError, match="consumed screen"):
        trace.require_consumed_screen_cell(PROTOCOL, 1, 1994844988)  # sealed blind
    with pytest.raises(ValueError, match="consumed screen"):
        trace.require_consumed_screen_cell(PROTOCOL, 5, 3857792434)
    with pytest.raises(ValueError, match="exact integer"):
        trace.require_consumed_screen_cell(PROTOCOL, True, 3857792434)


def test_scope_rejects_a_doctored_protocol_expanding_screen() -> None:
    altered = deepcopy(PROTOCOL)
    altered["partitions"]["screen"]["seeds"].append(725991793)
    with pytest.raises(ValueError, match="frozen screen grid"):
        trace.require_consumed_screen_cell(altered, 1, 725991793)


def test_receipt_gate_requires_both_completed_original_arms_and_valid_digests(tmp_path: Path) -> None:
    freeze = {"protocol_sha256": "frozen-test"}
    rows = {
        arm: {
            "partition": "screen", "arm": arm, "track_id": 1, "seed": 3857792434,
            "repeat": 0, "finished": False, "progress": 0.4, "error": None,
        }
        for arm in ("control", "candidate")
    }
    with pytest.raises(ValueError, match="both completed"):
        trace.require_consumed_receipts(tmp_path, freeze, 1, 3857792434)
    evaluator.record_cell(tmp_path, freeze, rows["control"])
    with pytest.raises(ValueError, match="both completed"):
        trace.require_consumed_receipts(tmp_path, freeze, 1, 3857792434)
    evaluator.record_cell(tmp_path, freeze, rows["candidate"])
    assert trace.require_consumed_receipts(tmp_path, freeze, 1, 3857792434) == {
        "baseline": rows["control"], "margin": rows["candidate"],
    }
    tampered = evaluator.cell_path(tmp_path, "screen", "candidate", 1, 3857792434, 0)
    envelope = json.loads(tampered.read_text())
    envelope["row"]["progress"] = 0.9
    tampered.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        trace.require_consumed_receipts(tmp_path, freeze, 1, 3857792434)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.calls = 0
        self.actions = []

    def step(self, action):
        self.calls += 1
        self.actions.append(tuple(float(value) for value in action))
        return np.zeros((4, 84, 84), dtype=np.float32), 2.75, False, False, {
            "progress": 0.25, "damage": 0.0, "collision": False,
        }


def _synthetic_session() -> SimulationSession:
    class Hull:
        position = (1.0, -2.0)
        linearVelocity = (3.0, 4.0)
        angle = 0.5

    class Car:
        hull = Hull()

    class Raw:
        t = 0.08
        car = Car()

    session = SimulationSession.__new__(SimulationSession)
    session.environment = _FakeEnvironment()
    session.raw_environment = Raw()
    session.environment.unwrapped = session.raw_environment
    session.start_time_s = 0.0
    session.observation = np.zeros((4, 84, 84), dtype=np.float32)
    session.steps = []
    session.record_frames = False
    session.done = False
    session.closed = False
    session.last_info = {}
    return session


def test_reward_capture_calls_real_session_step_once_without_changing_result() -> None:
    action = np.asarray([0.25, 0.5, 0.0], dtype=np.float32)
    ordinary = _synthetic_session()
    expected_step = ordinary.step(action)
    diagnostic = _synthetic_session()
    original_method = diagnostic.environment.step.__func__
    actual_step, reward = trace.step_with_reward(diagnostic, action)

    assert actual_step == expected_step
    assert reward == 2.75
    assert diagnostic.environment.calls == 1
    assert diagnostic.environment.actions == [(0.25, 0.5, 0.0)]
    assert "step" not in diagnostic.environment.__dict__
    assert diagnostic.environment.step.__func__ is original_method


def test_obstacle_capture_calls_detector_once_and_restores_it() -> None:
    class Controller:
        road_visible = False
        _obstacle_side = -1.0

        def __init__(self) -> None:
            self.calls = 0

        def _nearest_obstacle(self, frame, centers):
            self.calls += 1
            return (44.5, 41.2, 41.3125)

    class Agent:
        def __init__(self) -> None:
            self._forward_controller = Controller()

        def act(self, observation):
            assert self._forward_controller._nearest_obstacle(observation, {}) == (
                44.5, 41.2, 41.3125
            )
            self._forward_controller.road_visible = True
            self._forward_controller._obstacle_side = 1.0
            return np.asarray([0.085, 0.1, 0.0], dtype=np.float32)

    agent = Agent()
    detector_method = agent._forward_controller._nearest_obstacle.__func__
    action, obstacle = trace.act_with_obstacle(agent, np.zeros((4, 84, 84), dtype=np.float32))

    assert tuple(action) == pytest.approx((0.085, 0.1, 0.0))
    assert obstacle == (44.5, 41.2, 41.3125)
    assert agent._forward_controller.calls == 1
    assert agent._forward_controller.road_visible is True
    assert agent._forward_controller._obstacle_side == 1.0
    assert "_nearest_obstacle" not in agent._forward_controller.__dict__
    assert agent._forward_controller._nearest_obstacle.__func__ is detector_method


def test_telemetry_row_distinguishes_new_tile_reward_and_physical_road_loss() -> None:
    step = RunStep(
        step=7, sim_time_s=0.64, action=(0.2, 0.1, 0.0),
        position=(1.5, -2.5), angle=0.3, velocity=(3.0, 4.0),
        progress=0.3, damage=0.2, collision=True, terminated=False, truncated=False,
    )
    row = trace.make_step_row(
        step=step, observation_sha256="a" * 64, act_ms=1.5,
        obstacle=(44.5, 41.2, 41.3125), side_before=-1.0,
        controller_state={"road_visible": True, "obstacle_side": 1.0},
        tiles_before=2, tiles_after=3, track_tiles=10,
        reward=99.6, wheel_on_road=[False, False, False, False],
        offtrack_counter=4,
    )
    assert row["progress_before"] == 0.2
    assert row["progress_delta"] == pytest.approx(0.1)
    assert row["new_tiles"] == 1
    assert row["new_tile_reward"] == 100.0
    assert row["official_reward"] == 99.6
    assert row["wheels_on_road"] == 0
    assert row["velocity"] == [3.0, 4.0]
    assert row["speed_world"] == 5.0
    assert row["detected_obstacle"] == [44.5, 41.2, 41.3125]
    assert row["obstacle_side_before"] == -1.0
    assert row["controller"] == {"road_visible": True, "obstacle_side": 1.0}
