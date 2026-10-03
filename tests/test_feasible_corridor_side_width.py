"""Visible-road side veto layered on the frozen temporal corridor controller."""

import numpy as np
import pytest

import agent


def _candidate():
    candidate_type = getattr(agent, "_FeasibleCorridorSideWidthController", None)
    assert candidate_type is not None
    return candidate_type()


def _switch(controller, *, previous, current, bbox, edges, side=-1.0, proposed=1.0):
    controller._obstacle_side = side
    controller._temporal_obstacle = previous
    controller._corridor_detection = current
    controller._corridor_bbox = bbox
    controller._corridor_edges = dict(edges)
    return controller._allow_obstacle_side_switch(
        obstacle_y=current[0], obstacle_x=current[1], candidate_side=proposed,
    )


STEP_237 = {
    "previous": (48.3, 38.7, -0.35769109725952575),
    "current": (41.8, 48.0, 48.201001739501955),
    "bbox": (47, 40, 49, 43),
    "edges": {52: (32.0, 50.0), 53: (32.0, 51.0), 54: (31.0, 48.0)},
}


def test_exact_v4_step_237_vetoes_switch_into_narrow_right_side():
    parent = agent._FeasibleCorridorTemporalSideController()
    candidate = _candidate()

    assert _switch(parent, **STEP_237) is True
    assert _switch(candidate, **STEP_237) is False
    assert candidate._obstacle_side == -1.0


@pytest.mark.parametrize("override", [
    {"edges": {52: (32.0, 50.0), 54: (31.0, 48.0)}},
    {"edges": {52: (32.0, 50.0), 54: (31.0, 48.0), 56: (31.0, 48.0)}},
    {"edges": {54: (31.0, 48.0), 55: (31.0, 48.0), 56: (31.0, 48.0)}},
    {"edges": {52: (32.0, 61.0), 53: (32.0, 61.0), 54: (31.0, 61.0)}},
    {"previous": (20.0, 38.7, -0.35769109725952575)},
])
def test_missing_distant_open_or_untracked_edge_evidence_keeps_parent_switch(override):
    conditions = {**STEP_237, **override}
    parent = agent._FeasibleCorridorTemporalSideController()
    candidate = _candidate()

    assert _switch(parent, **conditions) is True
    assert _switch(candidate, **conditions) is True


def test_temporal_sign_flip_veto_remains_in_force():
    conditions = {
        **STEP_237,
        "previous": (48.3, 38.7, -0.35769109725952575),
        "current": (41.8, 48.0, 47.8),
    }
    parent = agent._FeasibleCorridorTemporalSideController()
    candidate = _candidate()

    assert _switch(parent, **conditions) is False
    assert _switch(candidate, **conditions) is False


def _observation(*, obstacle=None, road=(29, 54)):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    left, right = road
    frame[22:62, left : right + 1] = 0.4
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2 : y + 2, x - 1 : x + 2] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * 30.0) / 18.0
    return np.repeat(frame[None], 4, axis=0)


@pytest.mark.parametrize("obstacle,road", [
    (None, (29, 54)),
    ((41, 40), (29, 54)),
    ((41, 40), (36, 47)),
])
def test_clear_road_proven_corridor_and_first_obstacle_keep_v4_action(obstacle, road):
    parent = agent._FeasibleCorridorTemporalSideController()
    candidate = _candidate()
    observation = _observation(obstacle=obstacle, road=road)

    np.testing.assert_array_equal(candidate.act(observation), parent.act(observation))
    assert candidate._obstacle_side == parent._obstacle_side
    assert candidate._pace_effective_target == parent._pace_effective_target


def test_row42_dropout_recovery_keeps_v4_road_steering_and_speed():
    # This is the sparse road pattern that motivated v3/v4's row42 fallback.
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    frame[46, 10:35] = 0.4
    frame[50, 16:40] = 0.4
    frame[54:61, 22:44] = 0.4
    frame[55:59, 23:26] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * 29.0) / 18.0
    observation = np.repeat(frame[None], 4, axis=0)
    parent = agent._FeasibleCorridorTemporalSideController()
    candidate = _candidate()
    parent._last_steer = candidate._last_steer = -0.203

    np.testing.assert_array_equal(candidate.act(observation), parent.act(observation))
    assert candidate._corridor_plan is None
    assert 42 not in candidate._corridor_centers
    assert candidate._pace_effective_target == pytest.approx(18.0)
