"""Synthetic checks for the diagnostic local side veto and pace cap."""

import numpy as np
import pytest

import agent


def _candidate():
    candidate = getattr(agent, "_FeasibleCorridorSideVetoPaceController", None)
    assert candidate is not None
    return candidate()


def _side_switch(controller, *, current, proposed, bbox, edges, x=39.0, y=40.0):
    controller._obstacle_side = current
    controller._corridor_bbox = bbox
    controller._corridor_edges = dict(edges)
    controller._corridor_detection = (y, x, 41.5)
    return controller._allow_obstacle_side_switch(
        obstacle_y=y, obstacle_x=x, candidate_side=proposed,
    )


def _observation(road, *, obstacle=None, speed=30.0):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    for row in range(22, 62):
        left, right = road(row)
        frame[row, left : right + 1] = 0.4
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2 : y + 2, x - 1 : x + 2] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.repeat(frame[None], 4, axis=0)


@pytest.mark.parametrize("current,proposed,bbox,edges,x", [
    (-1.0, 1.0, (38, 38, 40, 42), {39: (25, 45), 40: (25, 45), 41: (25, 45)}, 39.0),
    (1.0, -1.0, (43, 38, 45, 42), {39: (38, 58), 40: (38, 58), 41: (38, 58)}, 44.0),
])
def test_three_nearest_observed_rows_veto_switch_to_blocked_side(
    current, proposed, bbox, edges, x,
):
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()

    assert _side_switch(
        control, current=current, proposed=proposed, bbox=bbox, edges=edges, x=x,
    ) is True
    assert _side_switch(
        candidate, current=current, proposed=proposed, bbox=bbox, edges=edges, x=x,
    ) is False
    assert candidate._obstacle_side == current


@pytest.mark.parametrize("y,bbox,edges", [
    (30.545454545454547, (47, 29, 50, 32),
     {41: (32, 51), 42: (32, 52), 43: (32, 49)}),
    (38.8, (47, 37, 49, 40),
     {49: (32, 50), 50: (32, 51), 51: (32, 49)}),
])
def test_recorded_far_edge_rows_veto_wrong_side_switch(y, bbox, edges):
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()
    kwargs = dict(current=-1.0, proposed=1.0, bbox=bbox,
                  edges=edges, x=48.0, y=y)

    assert _side_switch(control, **kwargs) is True
    assert _side_switch(candidate, **kwargs) is False


@pytest.mark.parametrize("edges", [
    {39: (25, 58), 40: (25, 58), 41: (25, 58)},
    {39: (25, 45), 40: (25, 45)},
    {39: (25, 45), 40: (25, 45), 53: (25, 45)},
    {39: (38, 45), 40: (38, 45), 41: (38, 45)},
    {39: (25, 45), 40: (25, 45), 41: (25, 58)},
])
def test_feasible_or_incomplete_edge_evidence_keeps_inherited_switch(edges):
    candidate = _candidate()
    assert _side_switch(
        candidate, current=-1.0, proposed=1.0,
        bbox=(38, 38, 40, 42), edges=edges,
    ) is True


def test_missing_bbox_or_unlatched_side_keeps_inherited_switch():
    candidate = _candidate()
    assert _side_switch(
        candidate, current=-1.0, proposed=1.0,
        bbox=None, edges={39: (25, 45), 40: (25, 45), 41: (25, 45)},
    ) is True
    assert _side_switch(
        candidate, current=0.0, proposed=1.0,
        bbox=(38, 38, 40, 42),
        edges={39: (25, 45), 40: (25, 45), 41: (25, 45)},
    ) is True


def test_no_corridor_target_is_twenty_without_changing_steering():
    def narrow_bend(row):
        shift = (54 - row) // 4
        return 36 + shift, 48 + shift

    observation = _observation(narrow_bend, obstacle=(45, 40))
    v2 = agent._FeasibleCorridorFallbackSpeedController()
    candidate = _candidate()

    v2_action = v2.act(observation)
    candidate_action = candidate.act(observation)

    assert v2._corridor_plan is None
    assert candidate._corridor_plan is None
    assert v2._pace_effective_target == pytest.approx(18.0)
    assert candidate._pace_effective_target == pytest.approx(20.0)
    assert candidate_action[0] == v2_action[0]
    assert candidate_action[1] == 0.0 and candidate_action[2] > 0.0


@pytest.mark.parametrize("obstacle,expected_plan", [(None, False), ((41, 40), True)])
def test_clear_road_and_fully_checked_pass_remain_parent_exact(obstacle, expected_plan):
    observation = _observation(lambda row: (29, 54), obstacle=obstacle)
    control = agent._FeasibleCorridorRoadDropoutController()
    candidate = _candidate()

    np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
    assert (candidate._corridor_plan is not None) is expected_plan


def test_reset_clears_obstacle_and_corridor_state():
    candidate = _candidate()
    _side_switch(
        candidate, current=-1.0, proposed=1.0,
        bbox=(38, 38, 40, 42),
        edges={39: (25, 45), 40: (25, 45), 41: (25, 45)},
    )
    candidate.reset()
    fresh = _candidate()

    assert candidate.__dict__ == fresh.__dict__
    np.testing.assert_array_equal(candidate.act(None), fresh.act(None))
