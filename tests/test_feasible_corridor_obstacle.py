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


def test_pass_target_right_of_ego_does_not_command_left_turn_on_right_bend():
    def road(row):
        left = 35 if row >= 48 else min(42, 35 + 48 - row)
        return left, left + 21

    observation = _observation(road, obstacle=(50, 40))
    candidate = _controller()
    action = candidate.act(observation)

    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == -1.0
    assert candidate._corridor_plan["target_x"] > candidate.IMAGE_CENTER
    assert action[0] > 0.0


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


def test_small_road_drift_keeps_feasible_pass_side_for_same_obstacle():
    candidate = _controller()
    first = _observation(lambda row: (26, 54), obstacle=(41, 35))
    second = _observation(lambda row: (29, 57), obstacle=(41, 36))

    candidate.act(first)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == -1.0
    candidate.act(second)

    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == -1.0


def test_same_obstacle_can_switch_when_previous_pass_is_infeasible():
    candidate = _controller()
    first = _observation(lambda row: (26, 54), obstacle=(41, 27))
    second = _observation(lambda row: (36, 60), obstacle=(41, 28))

    candidate.act(first)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == -1.0
    candidate.act(second)

    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == 1.0


def test_far_obstacle_can_switch_to_materially_safer_feasible_side():
    candidate = _controller()
    first = _observation(lambda row: (26, 54), obstacle=(41, 27))
    second = _observation(lambda row: (30, 58), obstacle=(41, 28))

    candidate.act(first)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == -1.0
    candidate.act(second)

    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["side"] == 1.0


def test_obstacle_occlusion_preserves_two_sided_road_edge_evidence():
    observation = _observation(lambda row: (30, 53), obstacle=(41, 40))
    candidate = _controller()
    candidate.act(observation)
    assert candidate._corridor_plan is not None
    assert candidate._corridor_plan["edges"][40] == (30.0, 53.0)


def test_passage_score_uses_narrowest_checked_approach_clearance():
    observation = _observation(lambda row: (29, 54), obstacle=(41, 35))
    observation[:, 50, 49:55] = 0.7
    candidate = _controller()

    candidate.act(observation)

    assert candidate._corridor_plan is not None
    right = candidate._corridor_candidate(1.0)
    assert right is not None
    assert right[1] < 1.0
    assert candidate._corridor_plan["side"] == -1.0
    assert candidate._corridor_plan["clearance_px"] > right[1]


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


def test_near_obstacle_requires_checked_inflated_rows_before_plan():
    observation = _observation(lambda row: (37, 60), obstacle=(45, 59))
    baseline = agent._CompoundClearingBrakeCarryController()
    candidate = _controller()

    baseline_action = baseline.act(observation)
    candidate_action = candidate.act(observation)

    assert candidate._corridor_bbox == (44, 57, 46, 60)
    assert candidate._corridor_plan is None
    np.testing.assert_array_equal(candidate_action, baseline_action)


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
