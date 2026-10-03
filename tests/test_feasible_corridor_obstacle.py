"""Synthetic image-space cases for the diagnostic feasible-corridor planner."""

import numpy as np

import agent


def _controller():
    candidate = getattr(agent, "_FeasibleCorridorObstacleController", None)
    assert candidate is not None
    return candidate()


def _observation(road, *, obstacle=None):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    for row in range(22, 62):
        left, right = road(row)
        frame[row, left : right + 1] = 0.4
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2 : y + 2, x - 1 : x + 2] = 0.65
    return np.repeat(frame[None, :, :], 4, axis=0)


def test_bent_approach_rejects_sign_chosen_pass_that_crosses_near_edge():
    def road(row):
        return (28, 45) if row >= 48 else (30, 50)

    observation = _observation(road, obstacle=(39, 40))
    baseline = agent._CompoundClearingBrakeCarryController()
    old_action = baseline.act(observation)
    assert baseline._obstacle_side == 1.0

    candidate = _controller()
    new_action = candidate.act(observation)
    assert candidate._obstacle_side == -1.0
    assert new_action[0] < 0.0 < old_action[0]
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["bbox"] == (38, 38, 40, 41)


def test_centered_obstacle_pixel_jitter_does_not_reverse_feasible_pass():
    road = lambda row: (29, 54)
    baseline = agent._CompoundClearingBrakeCarryController()
    candidate = _controller()
    for x in (41, 42):
        observation = _observation(road, obstacle=(x, 40))
        baseline.act(observation)
        candidate.act(observation)
    assert baseline._obstacle_side == -1.0
    assert candidate._obstacle_side == 1.0
    assert candidate._corridor_plan is not None


def test_obstacle_occlusion_preserves_two_sided_road_edge_evidence():
    observation = _observation(lambda row: (30, 53), obstacle=(41, 40))
    candidate = _controller()
    candidate.act(observation)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["edges"][40] == (30.0, 53.0)


def test_far_obstacle_gets_a_plan_before_it_enters_middle_road_rows():
    observation = _observation(lambda row: (29, 54), obstacle=(41, 24))
    candidate = _controller()
    candidate.act(observation)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["bbox"] == (40, 22, 42, 25)
    assert 22 in candidate._corridor_plan["edges"]


def test_far_obstacle_without_visible_outer_edges_keeps_original_action():
    observation = _observation(lambda row: (29, 54), obstacle=(41, 24))
    observation[:, 22:28, :] = 0.1
    observation[:, 22:26, 40:43] = 0.65
    baseline = agent._CompoundClearingBrakeCarryController()
    candidate = _controller()
    np.testing.assert_array_equal(candidate.act(observation), baseline.act(observation))
    assert candidate._corridor_plan is None


def test_obstacle_gap_cannot_expand_road_into_unrelated_asphalt_island():
    observation = _observation(lambda row: (29, 45), obstacle=(40, 40))
    observation[:, 38:42, 37:43] = 0.1
    observation[:, 38:42, 43:49] = 0.4
    observation[:, 38:42, 39:42] = 0.65
    candidate = _controller()
    candidate.act(observation)
    assert candidate._corridor_edges[40] == (29.0, 45.0)
    if candidate._corridor_plan is not None:
        assert candidate._corridor_plan["target_x"] <= 45.0


def test_no_vehicle_width_corridor_falls_back_to_original_obstacle_action():
    observation = _observation(lambda row: (36, 48), obstacle=(41, 40))
    baseline = agent._CompoundClearingBrakeCarryController()
    candidate = _controller()
    np.testing.assert_array_equal(candidate.act(observation), baseline.act(observation))
    assert candidate._corridor_plan is None


def test_clear_track_actions_and_reset_remain_exactly_baseline():
    baseline = agent._CompoundClearingBrakeCarryController()
    candidate = _controller()
    for road in (
        lambda row: (30, 53),
        lambda row: (30 + (54 - row) // 6, 53 + (54 - row) // 6),
    ):
        observation = _observation(road)
        np.testing.assert_array_equal(candidate.act(observation), baseline.act(observation))
        assert candidate._corridor_plan is None
    baseline.reset()
    candidate.reset()
    observation = _observation(lambda row: (30, 53))
    np.testing.assert_array_equal(candidate.act(observation), baseline.act(observation))


def test_lost_road_discards_prior_obstacle_corridor():
    candidate = _controller()
    candidate.act(_observation(lambda row: (29, 54), obstacle=(41, 40)))
    assert candidate._corridor_previous is not None
    candidate.act(np.full((4, 84, 84), 0.7, dtype=np.float32))
    assert candidate._corridor_previous is None
    assert candidate._corridor_plan is None
