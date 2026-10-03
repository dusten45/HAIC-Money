"""Impact recovery is brief, object-bound, and respects the steering slew."""

from types import SimpleNamespace

import numpy as np
import pytest

import agent


def _run_pair(monkeypatch, *, second_identity=1, first_brake=0.04,
              second_width=3.1, speed_drop=5.0):
    controller = agent._ImpactAwareSparseRoadController()
    speeds = iter((19.5, 19.5 - speed_drop))
    monkeypatch.setattr(controller, "_frame", lambda observation: np.zeros((84, 84)))
    monkeypatch.setattr(controller, "_estimate_speed", lambda frame: next(speeds))
    observations = iter((
        ((37, 54, 40, 57), 1, [-0.13, 0.05, first_brake]),
        ((37, 57, 39, 59), second_identity, [-0.16, 0.05, 0.0]),
    ))

    def parent_act(self, observation):
        bbox, identity, action = next(observations)
        self._corridor_bbox = bbox
        self._temporal_track = SimpleNamespace(identity=identity, observations=10)
        self._temporal_assessment = SimpleNamespace(
            road_rows=(56, 57, 58), left_width_px=3.1,
            right_width_px=second_width, selected_side=1.0,
        )
        self._obstacle_side = 1.0
        result = np.asarray(action, dtype=np.float32)
        self._last_steer = float(result[0])
        return result

    monkeypatch.setattr(agent._TemporalReachabilityController, "act", parent_act)
    first = controller.act(None)
    second = controller.act(None)
    return controller, first, second


def test_confirmed_impact_escape_respects_inherited_slew(monkeypatch):
    controller, first, second = _run_pair(monkeypatch)

    assert first[0] == pytest.approx(-0.13)
    assert second[0] > first[0]
    assert second[0] - first[0] <= controller.MAX_STEER_STEP + 1e-6
    assert second[1:] == pytest.approx([0.05, 0.0])
    assert controller._last_steer == pytest.approx(float(second[0]))


@pytest.mark.parametrize("changes", [
    {"second_identity": 2},
    {"first_brake": 0.08},
    {"second_width": 1.9},
    {"speed_drop": 4.49},
])
def test_impact_escape_requires_consistent_near_obstacle(monkeypatch, changes):
    _, _, second = _run_pair(monkeypatch, **changes)
    assert second[0] == pytest.approx(-0.16)


def test_impact_state_clears_on_reset():
    controller = agent._ImpactAwareSparseRoadController()
    controller._impact_escape_frames = 2
    controller._impact_previous_hud = 19.5
    controller._impact_previous_track_identity = 1

    controller.reset()

    assert controller._impact_escape_frames == 0
    assert controller._impact_previous_hud is None
    assert controller._impact_previous_track_identity is None


@pytest.mark.parametrize("armed_side, side, selected_side, plan, previous_steer, expected", [
    (1.0, -1.0, -1.0, None, -0.13, -0.16),
    (1.0, 1.0, -1.0, None, -0.13, -0.16),
    (1.0, 1.0, 1.0, {"side": 1.0}, -0.13, -0.16),
    (-1.0, -1.0, -1.0, None, 0.48, 0.32),
])
def test_armed_escape_yields_to_side_or_path_change(
    monkeypatch, armed_side, side, selected_side, plan, previous_steer, expected,
):
    controller = agent._ImpactAwareSparseRoadController()
    controller._impact_escape_frames = 1
    controller._impact_escape_identity = 1
    controller._impact_escape_side = armed_side
    controller._last_steer = previous_steer
    monkeypatch.setattr(controller, "_frame", lambda observation: np.zeros((84, 84)))
    monkeypatch.setattr(controller, "_estimate_speed", lambda frame: 20.0)

    def parent_act(self, observation):
        self._corridor_bbox = (37, 57, 39, 59)
        self._temporal_track = SimpleNamespace(identity=1, observations=10)
        self._temporal_assessment = SimpleNamespace(
            road_rows=(56, 57, 58), left_width_px=3.1,
            right_width_px=3.1, selected_side=selected_side,
        )
        self._obstacle_side = side
        self._corridor_plan = plan
        action = np.asarray([0.32 if previous_steer > 0.4 else -0.16, 0.05, 0.0],
                            dtype=np.float32)
        self._last_steer = float(action[0])
        return action

    monkeypatch.setattr(agent._TemporalReachabilityController, "act", parent_act)
    action = controller.act(None)

    assert action[0] == pytest.approx(expected)
    assert controller._impact_escape_frames == 0


def test_escape_is_one_shot_until_obstacle_tracking_expires(monkeypatch):
    controller, _, _ = _run_pair(monkeypatch)
    assert controller._impact_recovered_identity == 1
    controller._impact_escape_frames = 0
    controller._impact_previous_hud = 19.5
    controller._impact_previous_brake = 0.0
    controller._impact_previous_box = (37, 54, 40, 57)
    monkeypatch.setattr(controller, "_estimate_speed", lambda frame: 14.5)

    def parent_act(self, observation):
        self._corridor_bbox = (37, 57, 39, 59)
        self._temporal_track = SimpleNamespace(identity=1, observations=10)
        self._temporal_assessment = SimpleNamespace(
            road_rows=(56, 57, 58), left_width_px=3.1,
            right_width_px=3.1, selected_side=1.0,
        )
        self._obstacle_side = 1.0
        self._corridor_plan = None
        self._last_steer = -0.16
        return np.asarray([-0.16, 0.05, 0.0], dtype=np.float32)

    monkeypatch.setattr(agent._TemporalReachabilityController, "act", parent_act)
    action = controller.act(None)
    assert action[0] == pytest.approx(-0.16)
    assert controller._impact_recovered_identity == 1

    def tracking_lost(self, observation):
        self._temporal_track = None
        self._corridor_bbox = None
        return np.zeros(3, dtype=np.float32)

    monkeypatch.setattr(agent._TemporalReachabilityController, "act", tracking_lost)
    controller.act(None)
    assert controller._impact_recovered_identity is None
    assert controller._impact_escape_frames == 0
    assert controller._impact_previous_track_identity is None
