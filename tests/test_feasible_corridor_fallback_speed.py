"""Synthetic checks for the diagnostic visible no-corridor speed cap."""

import numpy as np

import agent


def _observation(road, *, obstacle=None, speed=30.0):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    for row in range(22, 62):
        left, right = road(row)
        frame[row, left : right + 1] = 0.4
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2 : y + 2, x - 1 : x + 2] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * speed) / 18.0
    return np.repeat(frame[None, :, :], 4, axis=0)


def _narrow_bend(row):
    shift = (54 - row) // 4
    return 36 + shift, 48 + shift


def _fallback_controller():
    candidate = getattr(agent, "_FeasibleCorridorFallbackSpeedController", None)
    assert candidate is not None
    return candidate()


def test_visible_no_corridor_brakes_instead_of_releasing_at_speed_thirty():
    observation = _observation(_narrow_bend, obstacle=(45, 40))
    control = agent._FeasibleCorridorObstacleController()
    candidate = _fallback_controller()

    control_action = control.act(observation)
    candidate_action = candidate.act(observation)

    assert control._corridor_plan is None
    assert candidate._corridor_plan is None
    assert abs(candidate._estimate_speed(observation[-1]) - 30.0) < 0.01
    assert control._pace_command_target == control.COMPOUND_TARGET_SPEED
    assert control_action[1] > 0.0 and control_action[2] == 0.0
    assert candidate_action[0] == control_action[0]
    assert candidate_action[1] == 0.0 and candidate_action[2] > 0.0


def test_verified_obstacle_corridor_and_clear_road_remain_v1_exact():
    control = agent._FeasibleCorridorObstacleController()
    candidate = _fallback_controller()
    road = lambda row: (29, 54)
    for observation, expected_plan in (
        (_observation(road, obstacle=(41, 40)), True),
        (_observation(road), False),
    ):
        np.testing.assert_array_equal(candidate.act(observation), control.act(observation))
        assert (candidate._corridor_plan is not None) is expected_plan


def test_visible_no_corridor_cap_does_not_latch_after_obstacle_disappears():
    control = agent._FeasibleCorridorObstacleController()
    candidate = _fallback_controller()
    control.act(_observation(_narrow_bend, obstacle=(45, 40)))
    candidate.act(_observation(_narrow_bend, obstacle=(45, 40)))
    clear = _observation(_narrow_bend)

    np.testing.assert_array_equal(candidate.act(clear), control.act(clear))
    assert candidate._corridor_plan is None


def test_invalid_road_recovery_is_v1_exact():
    control = agent._FeasibleCorridorObstacleController()
    candidate = _fallback_controller()
    clear = _observation(lambda row: (29, 54))
    control.act(clear)
    candidate.act(clear)
    lost = np.full((4, 84, 84), 0.7, dtype=np.float32)

    np.testing.assert_array_equal(candidate.act(lost), control.act(lost))
    assert candidate._corridor_plan is None


def test_nonvisual_metadata_fields_do_not_change_fallback_action():
    observation = _observation(_narrow_bend, obstacle=(45, 40))
    first = _fallback_controller()
    second = _fallback_controller()
    first.track_id, first.seed, first.progress, first.damage = 1, 17, 0.9, 0.0
    second.track_id, second.seed, second.progress, second.damage = 4, 999, 0.1, 0.8

    np.testing.assert_array_equal(first.act(observation), second.act(observation))
