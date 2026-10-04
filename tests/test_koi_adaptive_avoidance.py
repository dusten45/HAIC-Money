"""Synthetic governor tests; no simulator, driving, or training is used."""

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from haic.algorithms.koi.adaptive_avoidance import (
    AdaptiveAvoidanceAgent,
    commanded_passage_clear,
)


ROAD_ROWS = (30, 34, 38, 42, 46, 50, 54)


def synthetic_scene(*, left=34, top=48, width=3, height=4,
                    road_left=27, road_right=58):
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    frame[:, road_left:road_right] = 0.4
    frame[top:top + height, left:left + width] = 0.7
    # Same arithmetic centroid as connectedComponentsWithStats, not a rounded pixel.
    obj = (top + (height - 1) / 2, left + (width - 1) / 2, 42.0)
    return frame, obj


class FakeCrossingDriver:
    """Emulates only the state contract consumed by the outer governor."""

    def __init__(self, *, steps=10, speed=60.0, side=1.0, steer=0.2,
                 centers=None, obj=None, history_veto=False):
        self.steps = steps
        self.impact_left = 0
        self.speed_history = deque([speed] * 4, maxlen=6)
        self.brake_history = deque([0.0] * 4, maxlen=6)
        self.base = SimpleNamespace(_obstacle_side=side)
        self.speed = speed
        self.centers: dict[int, float] = dict.fromkeys(ROAD_ROWS, 42.0) if centers is None else centers
        self.obj = synthetic_scene()[1] if obj is None else obj
        self.far_object: tuple[float, float, float] | None = None
        self.action = np.array([steer, 0.0, 0.6], dtype=np.float32)
        self.impact_after: int | None = None
        self.impact_proxy_trigger = False
        self.contact_proxy = False
        self.history_veto = history_veto
        self.baseline_target = 44.0
        self.brakes_seen = []
        self.returned = self.action.copy()
        self.last = {}
        self.reset_observation = None
        self.reset_count = 0

    def act(self, observation):
        self.steps += 1
        self.brakes_seen.append(tuple(self.brake_history))
        veto = self.history_veto and any(b > 0.01 for b in list(self.brake_history)[-4:])
        if self.impact_after is not None:
            self.impact_left = self.impact_after
        elif self.impact_left > 0:
            self.impact_left -= 1
        if self.history_veto and self.impact_proxy_trigger:
            self.impact_left = 0 if veto else 12
        self.returned = self.action.copy()
        if self.history_veto and self.impact_proxy_trigger:
            self.returned[0] = 0.15 if veto else -0.35
        self.speed_history.append(self.speed)
        self.brake_history.append(float(self.returned[2]))
        self.last = {
            "road_centers": dict(self.centers),
            "pixel_speed": self.speed,
            "near_object": self.obj,
            "far_object": self.far_object,
            "impact_proxy_trigger": self.impact_proxy_trigger,
            "contact_proxy": self.contact_proxy,
            "braking_proxy_veto": veto,
            "target_speed": 60.0 if self.steps <= 10 or self.obj is None else self.baseline_target,
            "damping_steer": 0.05,
        }
        return self.returned

    def reset(self, observation=None):
        self.reset_observation = observation
        self.reset_count += 1
        self.steps = 0
        self.impact_left = 0
        self.speed_history.clear()
        self.brake_history.clear()
        self.last = {}


def make_agent(**kwargs):
    driver = FakeCrossingDriver(**kwargs)
    return AdaptiveAvoidanceAgent(driver=driver), driver


def reach_steady_budget(agent, frame):
    action = agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    for _ in range(19):
        action = agent.act(frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
    return action


def assert_baseline(agent, driver, frame, reason):
    action = agent.act(frame)
    np.testing.assert_array_equal(action, driver.returned)
    assert action is not driver.returned
    assert agent.last["adaptive_speed_reason"] == reason
    assert agent.last["adaptive_speed_changed"] is False
    assert agent.previous_pass_speed == 44.0
    expected_target = driver.last["target_speed"] if driver.obj is not None and driver.steps > 10 else None
    assert agent.previous_speed_target == expected_target
    assert driver.brake_history[-1] == float(action[2])
    return action


@pytest.mark.parametrize("prior_steps", range(10))
def test_first_ten_actions_are_baseline_exact(prior_steps):
    frame, _ = synthetic_scene()
    agent, driver = make_agent(steps=prior_steps)
    assert_baseline(agent, driver, frame, "launch_prefix")
    assert driver.steps == prior_steps + 1


def test_eleventh_action_can_adapt_and_does_not_mutate_driver_action_or_pixels():
    frame, _ = synthetic_scene()
    before = frame.copy()
    agent, driver = make_agent()
    action = agent.act(frame)
    assert driver.steps == 11
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == 45.0
    assert agent.last["adaptive_speed_target"] == 45.0
    assert action[0] == driver.returned[0]
    np.testing.assert_array_equal(driver.returned, driver.action)
    np.testing.assert_array_equal(frame, before)


def test_launch_target60_does_not_leak_into_first_eligible_near_target():
    frame, _ = synthetic_scene()
    agent, driver = make_agent(steps=0)
    for _ in range(10):
        assert_baseline(agent, driver, frame, "launch_prefix")
        assert driver.last["target_speed"] == 60.0
        assert agent.previous_speed_target is None
    agent.act(frame)
    assert driver.steps == 11
    assert driver.last["target_speed"] == 44.0
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == 45.0
    assert agent.last["adaptive_speed_target"] == 45.0


@pytest.mark.parametrize("far_only", [False, True])
def test_no_near_obstacle_including_far_only_retains_exact_baseline(far_only):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.obj = None
    driver.far_object = (25.0, 35.0, 42.0) if far_only else None
    driver.action = np.array([-0.325, 0.45, 0.03], dtype=np.float32)
    assert_baseline(agent, driver, frame, "no_near_obstacle")


@pytest.mark.parametrize("missing_row", ROAD_ROWS)
def test_any_missing_road_row_prevents_relaxing_baseline(missing_row):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    del driver.centers[missing_row]
    assert_baseline(agent, driver, frame, "missing_road_rows")


def test_extra_road_row_also_fails_exact_row_reliability_contract():
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.centers[58] = 42.0
    assert_baseline(agent, driver, frame, "missing_road_rows")


@pytest.mark.parametrize("guard", [
    "impact_before_expiring", "impact_before_remaining", "impact_after",
    "impact_proxy_trigger", "contact_proxy", "prior_speed_drop",
])
def test_each_impact_guard_retains_baseline(guard):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    if guard == "impact_before_expiring":
        driver.impact_left = 1
    elif guard == "impact_before_remaining":
        driver.impact_left = 3
    elif guard == "impact_after":
        driver.impact_after = 3
    elif guard == "prior_speed_drop":
        # Maxlen eviction during the call must not lose this pre-call speed peak.
        driver.speed_history = deque([71.0, 60.0, 60.0, 60.0], maxlen=4)
    else:
        setattr(driver, guard, True)
    assert_baseline(agent, driver, frame, "impact_guard")
    if guard == "impact_before_expiring":
        assert driver.impact_left == 0
    if guard == "prior_speed_drop":
        assert max(driver.speed_history) == 60.0


@pytest.mark.parametrize("history", [[], [70.0, 60.0]])
def test_empty_history_and_exact_ten_speed_drop_do_not_trigger_guard(history):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.speed_history = deque(history, maxlen=6)
    agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"


@pytest.mark.parametrize("side,steer", [
    (None, 0.2), (0.0, 0.2), (0.5, 0.2), (2.0, 0.2),
    (1.0, 0.0), (-1.0, 0.0), (1.0, -0.2), (-1.0, 0.2),
    (1.0, 0.5), (-1.0, -0.5), (1.0, 0.7), (-1.0, -0.7),
])
def test_invalid_side_wrong_direction_or_large_current_steer_guards(side, steer):
    frame, _ = synthetic_scene()
    agent, driver = make_agent(side=side, steer=steer)
    assert_baseline(agent, driver, frame, "steering_guard")


@pytest.mark.parametrize("row54", [38.0, 46.0, 37.0, 47.0])
def test_alignment_boundary_and_outside_remain_baseline(row54):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.centers[54] = row54
    assert_baseline(agent, driver, frame, "alignment_guard")


@pytest.mark.parametrize("spread", [20.0, 30.0])
def test_curve_budget_at_or_below_fixed44_cannot_be_relaxed(spread):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.centers[30] = 42.0 + spread
    assert_baseline(agent, driver, frame, "curve_limited")
    assert agent.last["adaptive_road_target"] <= 44.0


@pytest.mark.parametrize("side,left,steer", [(1.0, 34, 0.2), (-1.0, 48, -0.2)])
def test_clearance_is_measured_on_the_chosen_left_or_right_side(side, left, steer):
    frame, obj = synthetic_scene(left=left)
    agent, driver = make_agent(side=side, steer=steer, obj=obj)
    # Cramp the unchosen side; its width must not lower the chosen budget.
    if side > 0:
        frame[:, :left] = 0.1
    else:
        frame[:, left + 3:] = 0.1
    action = reach_steady_budget(agent, frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_passage_width_px"] == 21
    assert agent.last["adaptive_pass_speed"] == pytest.approx(60.0)
    assert action[0] == driver.action[0]


@pytest.mark.parametrize("side,left,road_left,road_right", [
    (1.0, 34, 27, 46), (-1.0, 48, 39, 58),
])
def test_nine_pixel_passage_does_not_relax_fixed44(side, left, road_left, road_right):
    frame, obj = synthetic_scene(left=left, road_left=road_left, road_right=road_right)
    agent, driver = make_agent(side=side, steer=side * 0.2, obj=obj)
    assert_baseline(agent, driver, frame, "narrow_passage")
    assert agent.last["adaptive_passage_width_px"] == 9
    assert agent.last["adaptive_pass_speed"] == 44.0


def test_disconnected_asphalt_is_not_added_to_passage_width():
    frame, _ = synthetic_scene()
    frame[50, 42] = 0.1
    agent, driver = make_agent()
    assert_baseline(agent, driver, frame, "narrow_passage")
    assert agent.last["adaptive_passage_width_px"] == 5


@pytest.mark.parametrize("row", [47, 48, 49, 50, 51, 52])
def test_one_restricted_obstacle_or_immediately_adjacent_row_vetoes_wide_others(row):
    frame, _ = synthetic_scene()
    frame[row, 46:] = 0.1
    agent, driver = make_agent()
    assert_baseline(agent, driver, frame, "narrow_passage")
    assert agent.last["adaptive_passage_width_px"] == 9


@pytest.mark.parametrize("row,reason", [(46, "adaptive"), (53, "swept_path_guard")])
def test_rows_outside_immediate_obstacle_neighbors_do_not_reduce_local_width(row, reason):
    frame, _ = synthetic_scene()
    frame[row, 37:] = 0.1
    agent, _ = make_agent()
    agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == reason
    assert agent.last["adaptive_passage_width_px"] == 21


@pytest.mark.parametrize("side,steer,reason", [
    (1.0, 0.0, "steering_guard"),
    (1.0, 0.001, "swept_path_guard"),
    (-1.0, -0.001, "swept_path_guard"),
])
def test_zero_or_near_zero_current_steer_cannot_relax_through_centered_obstacle(side, steer, reason):
    frame, obj = synthetic_scene(left=41)
    agent, driver = make_agent(side=side, steer=steer, obj=obj)
    # A contradictory inherited diagnostic must not replace the current command.
    assert_baseline(agent, driver, frame, reason)
    assert driver.last["damping_steer"] != driver.returned[0]


def test_source_case_road_avoidance_cancellation_uses_swept_path_guard():
    frame = np.full((84, 84), 0.1, dtype=np.float32)
    for row in range(84):
        shift = int(max(4 - abs(row - 42) / 3, 0))
        frame[row, 27 + shift:58 + shift] = 0.4
    frame[42, 30:62] = 0.4
    frame[29:32, 41:44] = 0.7
    centers = {54: 42.0, 50: 43.0, 46: 44.0, 42: 45.5,
               38: 44.0, 34: 43.0, 30: 42.0}
    agent, driver = make_agent(speed=50.0, side=-1.0, steer=-0.011111,
                               centers=centers, obj=(30.0, 42.0, 42.0))
    driver.action[2] = 0.08
    assert_baseline(agent, driver, frame, "swept_path_guard")
    assert agent.last["adaptive_passage_width_px"] == 14


def test_passage_exists_but_current_command_footprint_leaves_road():
    frame, obj = synthetic_scene(road_right=47)
    agent, driver = make_agent(obj=obj)
    assert_baseline(agent, driver, frame, "swept_path_guard")
    assert agent.last["adaptive_passage_width_px"] == 10
    assert agent.last["adaptive_pass_speed"] > 44.0
    assert not commanded_passage_clear(frame, float(driver.action[0]), (34, 48, 3, 4), 1.0)


@pytest.mark.parametrize("row,col", [(54, 40), (54, 46), (56, 40), (56, 46)])
def test_entire_three_by_seven_approach_footprint_must_be_asphalt(row, col):
    frame, _ = synthetic_scene()
    assert commanded_passage_clear(frame, 0.2, (34, 48, 3, 4), 1.0)
    frame[row, col] = 0.1
    agent, driver = make_agent()
    assert_baseline(agent, driver, frame, "swept_path_guard")
    assert agent.last["adaptive_passage_width_px"] == 21


@pytest.mark.parametrize("top", [56, 57])
def test_late_bounding_box_below_row55_falls_back_despite_observed_passage(top):
    frame, obj = synthetic_scene(top=top)
    agent, driver = make_agent(obj=obj)
    assert_baseline(agent, driver, frame, "swept_path_guard")
    assert agent.last["adaptive_passage_width_px"] == 21


def test_constant_command_cannot_reach_forward_row_after_turn_becomes_unreachable():
    frame = np.full((84, 84), 0.4, dtype=np.float32)
    frame[47:51, 34:37] = 0.7
    assert not commanded_passage_clear(frame, 0.35, (34, 47, 3, 4), 1.0)


@pytest.mark.parametrize("side,left,steer", [(1.0, 34, 0.2), (-1.0, 48, -0.2)])
def test_reachable_command_can_pass_on_either_selected_side(side, left, steer):
    frame, _ = synthetic_scene(left=left)
    assert commanded_passage_clear(frame, steer, (left, 48, 3, 4), side)
    assert not commanded_passage_clear(frame, steer, (left, 48, 3, 4), -side)


@pytest.mark.parametrize("layer", ["helper", "agent"])
def test_nearer_part_of_tall_obstacle_cannot_be_skipped_by_swept_guard(layer):
    frame, obj = synthetic_scene(left=45, top=54, height=7)
    # At row60 this command still rounds to x42: its seven-wide footprint
    # includes bright column45, even though the row54..55 path is clear.
    assert frame[60, 45] == pytest.approx(0.7)
    if layer == "helper":
        assert not commanded_passage_clear(frame, -0.332286, (45, 54, 3, 7), -1.0)
    else:
        agent, driver = make_agent(speed=44.5, side=-1.0, steer=-0.332286, obj=obj)
        driver.action[2] = 0.02
        assert_baseline(agent, driver, frame, "swept_path_guard")


@pytest.mark.parametrize("side,left,road_left,road_right", [
    (1.0, 34, 27, 60), (-1.0, 48, 24, 58),
    (1.0, 72, 60, 84), (-1.0, 9, 0, 24),
])
def test_search_or_image_clipped_road_is_not_an_observed_boundary(
        side, left, road_left, road_right):
    frame, obj = synthetic_scene(left=left, road_left=road_left, road_right=road_right)
    if left > 60:
        obj = (obj[0], obj[1], 76.0)
    elif left < 20:
        obj = (obj[0], obj[1], 10.0)
    agent, driver = make_agent(side=side, steer=side * 0.2, obj=obj)
    assert_baseline(agent, driver, frame, "passage_unobserved")
    assert agent.last["adaptive_passage_width_px"] is None


@pytest.mark.parametrize("left,top,width,height", [
    (34, 48, 1, 4), (34, 48, 3, 1), (34, 48, 10, 2),
    (34, 48, 3, 11), (30, 48, 9, 9), (34, 22, 3, 4), (34, 58, 3, 4),
])
def test_unreliable_component_dimensions_area_or_vertical_boundaries_fallback(
        left, top, width, height):
    frame, obj = synthetic_scene(left=left, top=top, width=width, height=height)
    agent, driver = make_agent(obj=obj)
    assert_baseline(agent, driver, frame, "passage_unobserved")


@pytest.mark.parametrize("change", ["missing", "centroid_mismatch", "ambiguous"])
def test_component_must_exist_and_match_exactly_one_centroid(change):
    frame, obj = synthetic_scene()
    agent, driver = make_agent(obj=obj)
    if change == "missing":
        frame[48:52, 34:37] = 0.4
    elif change == "centroid_mismatch":
        driver.obj = (obj[0] + 0.02, obj[1], obj[2])
    else:
        frame[46:53, 32:39] = 0.7
        frame[47:52, 33:38] = 0.4
        frame[48:51, 34:37] = 0.7
        driver.obj = (49.0, 35.0, 42.0)
    assert_baseline(agent, driver, frame, "passage_unobserved")


@pytest.mark.parametrize("left,top,width,height", [
    (34, 48, 2, 2), (30, 44, 8, 10),
])
def test_reliable_component_at_minimum_or_maximum_area_is_eligible(left, top, width, height):
    frame, obj = synthetic_scene(left=left, top=top, width=width, height=height)
    agent, _ = make_agent(obj=obj)
    agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"


@pytest.mark.parametrize("spread", [0.0, 5.0, 10.0, 18.0])
def test_adaptive_target_never_exceeds_non_obstacle_road_budget(spread):
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.centers[30] = 42.0 + spread
    reach_steady_budget(agent, frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    road_budget = max(38.0, 60.0 - 0.8 * spread)
    assert agent.last["adaptive_road_target"] == pytest.approx(road_budget)
    assert agent.last["adaptive_speed_target"] == pytest.approx(road_budget)
    assert agent.last["adaptive_speed_target"] <= road_budget
    assert agent.last["adaptive_speed_target"] <= agent.last["adaptive_arrival_cap"]


def test_clearance_influence_is_continuous_not_just_wide_or_narrow():
    budgets = []
    for width in (10, 11, 12, 13, 14, 15):
        frame, obj = synthetic_scene(top=55, road_right=37 + width)
        agent, _ = make_agent(obj=obj)
        reach_steady_budget(agent, frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
        assert agent.last["adaptive_passage_width_px"] == width
        budgets.append(agent.last["adaptive_pass_speed"])
    np.testing.assert_allclose(np.diff(budgets), 16.0 / 6.0, atol=1e-6)
    assert budgets[0] > 44.0
    assert budgets[-1] == pytest.approx(60.0)


def test_current_steering_demand_continuously_lowers_speed_budget():
    budgets = []
    for steer in (0.2, 0.3, 0.4, 0.49):
        frame, obj = synthetic_scene(top=55)
        agent, driver = make_agent(steer=steer, obj=obj)
        action = reach_steady_budget(agent, frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
        assert action[0] == driver.returned[0]
        budgets.append(agent.last["adaptive_pass_speed"])
    assert all(a > b for a, b in zip(budgets, budgets[1:]))
    assert budgets[0] == pytest.approx(60.0)
    assert 44.0 < budgets[-1] < 45.0


def test_alignment_continuously_lowers_speed_budget_before_guard_boundary():
    budgets = []
    for offset in (0.0, 1.0, 2.0, 3.0):
        frame, _ = synthetic_scene()
        agent, driver = make_agent()
        driver.centers[54] += offset
        reach_steady_budget(agent, frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
        budgets.append(agent.last["adaptive_pass_speed"])
    np.testing.assert_allclose(budgets, [60.0, 56.0, 52.0, 48.0], atol=1e-6)


def test_eligible_speed60_replaces_both_fixed44_brakes_instead_of_retaining_max():
    frame, obj = synthetic_scene()
    agent, driver = make_agent(speed=60.0)
    old_arrival_cap = np.sqrt(44.0 ** 2 + 110.0 * max((63.0 - obj[0]) / 1.701 - 7.0, 0.0))
    old_brake = max(0.28, min(0.6, 0.04 * (60.0 - old_arrival_cap)))
    driver.action[2] = old_brake
    action = reach_steady_budget(agent, frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_speed_target"] == pytest.approx(60.0)
    assert action[2] == 0.0
    assert action[2] < driver.returned[2]
    assert action[1] == pytest.approx(0.12)
    assert agent.last["baseline_pedals"] == driver.returned[1:].tolist()
    assert agent.last["adaptive_speed_changed"] is True


@pytest.mark.parametrize("speed", [0.0, 30.0, 44.0, 58.0, 60.0, 61.0, 62.1, 65.0, 80.0])
@pytest.mark.parametrize("steer", [0.2, 0.35, 0.49])
def test_new_pedals_are_finite_bounded_exclusive_and_current_steer_is_unchanged(speed, steer):
    frame, obj = synthetic_scene(top=55)
    agent, driver = make_agent(speed=speed, steer=steer, obj=obj)
    for _ in range(20):
        action = agent.act(frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
        assert action.dtype == np.float32
        assert action.shape == (3,)
        assert np.isfinite(action).all()
        assert action[0] == driver.returned[0]
        assert 0.0 <= action[1] <= 0.6 + 1e-7
        assert 0.0 <= action[2] <= 0.6 + 1e-7
        assert action[1] == 0.0 or action[2] == 0.0
        assert driver.brake_history[-1] == float(action[2])
        assert agent.last["final_gas"] == float(action[1])
        assert agent.last["final_brake"] == float(action[2])


def test_arrival_cap_tracks_distance_and_can_bind_below_road_budget():
    frame, obj = synthetic_scene(top=55, road_right=47)
    agent, _ = make_agent(obj=obj)
    action = reach_steady_budget(agent, frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_arrival_cap"] == agent.last["adaptive_pass_speed"]
    assert agent.last["adaptive_speed_target"] == agent.last["adaptive_arrival_cap"]
    assert agent.last["adaptive_speed_target"] < agent.last["adaptive_road_target"]
    assert action[1] == 0.0
    assert action[2] > 0.0


def test_pass_and_final_target_rise_at_most_one_per_act_when_clearance_improves():
    narrow, obj = synthetic_scene(top=55, road_right=47)
    wide, _ = synthetic_scene(top=55)
    agent, _ = make_agent(obj=obj)
    agent.act(narrow)
    previous_pass = agent.last["adaptive_pass_speed"]
    previous_target = agent.last["adaptive_speed_target"]
    for _ in range(4):
        agent.act(wide)
        current_pass = agent.last["adaptive_pass_speed"]
        current_target = agent.last["adaptive_speed_target"]
        assert 0.0 < current_pass - previous_pass <= 1.0 + 1e-7
        assert 0.0 < current_target - previous_target <= 1.0 + 1e-7
        previous_pass, previous_target = current_pass, current_target


def test_final_target_rise_is_limited_when_road_budget_recovers():
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    driver.centers[30] = 60.0
    reach_steady_budget(agent, frame)
    previous_target = agent.last["adaptive_speed_target"]
    assert previous_target == pytest.approx(45.6)
    assert agent.previous_pass_speed == pytest.approx(60.0)
    driver.centers[30] = 42.0
    agent.act(frame)
    assert agent.last["adaptive_speed_target"] <= previous_target + 1.0


def test_final_target_rise_is_limited_when_distance_envelope_increases():
    near, obj = synthetic_scene(top=55, road_right=51)
    far, far_obj = synthetic_scene(top=40, road_right=51)
    agent, driver = make_agent(steer=0.1, obj=obj)
    reach_steady_budget(agent, near)
    previous_target = agent.last["adaptive_speed_target"]
    previous_pass = agent.last["adaptive_pass_speed"]
    previous_cap = agent.last["adaptive_arrival_cap"]
    driver.obj = far_obj
    agent.act(far)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == previous_pass
    assert agent.last["adaptive_arrival_cap"] > previous_cap + 1.0
    assert previous_target < agent.last["adaptive_speed_target"] <= previous_target + 1.0


def test_cold_entry_pass_and_target_increase_by_at_most_one_from_fixed44():
    frame, _ = synthetic_scene()
    agent, _ = make_agent()
    previous_pass = 44.0
    previous_target = 44.0
    assert agent.previous_pass_speed == previous_pass
    assert agent.previous_speed_target is None
    for _ in range(20):
        agent.act(frame)
        assert agent.last["adaptive_speed_reason"] == "adaptive"
        current_pass = agent.last["adaptive_pass_speed"]
        current_target = agent.last["adaptive_speed_target"]
        assert 0.0 <= current_pass - previous_pass <= 1.0
        assert 0.0 <= current_target - previous_target <= 1.0
        previous_pass, previous_target = current_pass, current_target
    assert previous_pass == pytest.approx(60.0)
    assert previous_target == pytest.approx(60.0)


def test_higher_risk_lowers_pass_and_target_immediately_not_by_one():
    wide, obj = synthetic_scene(top=55)
    narrow, _ = synthetic_scene(top=55, road_right=47)
    agent, driver = make_agent(obj=obj)
    reach_steady_budget(agent, wide)
    previous_target = agent.last["adaptive_speed_target"]
    agent.act(narrow)
    assert agent.last["adaptive_pass_speed"] == pytest.approx(44.0 + 16.0 / 6.0)
    assert agent.last["adaptive_speed_target"] < previous_target - 1.0
    assert driver.brake_history[-1] == agent.last["final_brake"]


def test_tighter_road_budget_drops_target_immediately_even_with_clear_passage():
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    reach_steady_budget(agent, frame)
    assert agent.last["adaptive_speed_target"] == pytest.approx(60.0)
    driver.centers[30] = 60.0
    agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_speed_target"] == pytest.approx(45.6)


def test_increased_steering_risk_drops_budget_immediately_and_keeps_current_steer():
    frame, obj = synthetic_scene(top=55)
    agent, driver = make_agent(obj=obj)
    reach_steady_budget(agent, frame)
    assert agent.last["adaptive_pass_speed"] == pytest.approx(60.0)
    driver.action[0] = 0.49
    action = agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == pytest.approx(44.0 + 16.0 / 30.0)
    assert agent.last["adaptive_speed_target"] == agent.last["adaptive_pass_speed"]
    assert action[0] == driver.action[0]
    assert action[1] == 0.0
    assert action[2] > 0.0


@pytest.mark.parametrize("dropout", ["no_near", "impact", "narrow", "missing_rows"])
def test_dropout_resets_to_fixed44_and_reentry_is_rate_limited(dropout):
    wide, obj = synthetic_scene(top=55)
    agent, driver = make_agent(obj=obj)
    reach_steady_budget(agent, wide)
    assert agent.previous_pass_speed == pytest.approx(60.0)
    assert agent.previous_speed_target == pytest.approx(60.0)
    dropout_frame = wide
    if dropout == "no_near":
        driver.obj = None
    elif dropout == "impact":
        driver.contact_proxy = True
    elif dropout == "missing_rows":
        del driver.centers[34]
    else:
        dropout_frame, _ = synthetic_scene(top=55, road_right=46)
    agent.act(dropout_frame)
    assert agent.last["adaptive_speed_reason"] != "adaptive"
    assert agent.previous_pass_speed == 44.0
    assert agent.previous_speed_target == (None if dropout == "no_near" else 44.0)
    driver.obj = obj
    driver.contact_proxy = False
    driver.centers[34] = 42.0
    agent.act(wide)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == pytest.approx(45.0)
    assert agent.last["adaptive_speed_target"] == pytest.approx(45.0)


def test_near_fallback_remembers_current_baseline_target_not_old_adaptive_target():
    frame, _ = synthetic_scene()
    agent, driver = make_agent()
    reach_steady_budget(agent, frame)
    driver.baseline_target = 38.0
    driver.contact_proxy = True
    assert_baseline(agent, driver, frame, "impact_guard")
    assert agent.previous_speed_target == 38.0
    driver.contact_proxy = False
    driver.baseline_target = 44.0
    agent.act(frame)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    assert agent.last["adaptive_pass_speed"] == 45.0
    assert agent.last["adaptive_speed_target"] == 39.0


def test_final_brake_writeback_changes_next_stateful_fake_veto_and_preserves_current_steer():
    frame, _ = synthetic_scene()
    agent, driver = make_agent(history_veto=True)
    first = reach_steady_budget(agent, frame)
    assert driver.returned[2] == pytest.approx(0.6)
    assert first[2] == 0.0
    assert driver.brake_history[-1] == 0.0
    driver.impact_proxy_trigger = True
    second = agent.act(frame)
    assert driver.brakes_seen[-1][-1] == 0.0
    assert driver.last["braking_proxy_veto"] is False
    assert driver.impact_left == 12
    assert second[0] == driver.returned[0] == pytest.approx(-0.35)
    assert agent.last["adaptive_speed_reason"] == "impact_guard"
    assert driver.brake_history[-1] == float(second[2])

    # Counterfactual baseline discarded brake would veto and use a different steer.
    control = FakeCrossingDriver(history_veto=True)
    control.act(frame)
    control.impact_proxy_trigger = True
    control.act(frame)
    assert control.last["braking_proxy_veto"] is True
    assert control.impact_left == 0
    assert control.returned[0] == pytest.approx(0.15)


def test_nonzero_adaptive_brake_is_written_back_and_seen_by_next_veto():
    frame, _ = synthetic_scene()
    agent, driver = make_agent(speed=80.0, history_veto=True)
    first = agent.act(frame)
    assert first[2] > 0.01
    assert driver.brake_history[-1] == float(first[2])
    driver.impact_proxy_trigger = True
    second = agent.act(frame)
    assert driver.brakes_seen[-1][-1] == float(first[2])
    assert driver.last["braking_proxy_veto"] is True
    assert second[0] == driver.returned[0] == pytest.approx(0.15)
    assert driver.brake_history[-1] == float(second[2])


def test_reset_clears_governor_state_forwards_observation_and_repeats_sequence():
    narrow, obj = synthetic_scene(top=55, road_right=47)
    wide, _ = synthetic_scene(top=55)
    agent, driver = make_agent(obj=obj)

    def sequence():
        actions = []
        diagnostics = []
        for frame in [wide] * 10 + [narrow, wide, wide]:
            actions.append(agent.act(frame).copy())
            diagnostics.append(agent.last_step_diagnostics())
        return np.stack(actions), diagnostics

    agent.reset(narrow)
    assert driver.reset_observation is narrow
    assert driver.reset_count == 1
    first_actions, first_diagnostics = sequence()
    assert agent.previous_pass_speed > 44.0
    agent.reset(wide)
    assert driver.reset_observation is wide
    assert driver.reset_count == 2
    assert agent.previous_pass_speed == 44.0
    assert agent.previous_speed_target is None
    assert agent.last == {}
    assert driver.steps == 0
    assert not driver.brake_history
    assert not driver.speed_history
    second_actions, second_diagnostics = sequence()
    np.testing.assert_array_equal(first_actions, second_actions)
    assert first_diagnostics == second_diagnostics


def test_stack_uses_newest_frame_and_diagnostics_returns_independent_mapping():
    frame, _ = synthetic_scene()
    stack = np.stack([np.full_like(frame, 0.1), frame])
    agent, _ = make_agent()
    agent.act(stack)
    assert agent.last["adaptive_speed_reason"] == "adaptive"
    diagnostics = agent.last_step_diagnostics()
    diagnostics["adaptive_speed_reason"] = "external_change"
    assert agent.last["adaptive_speed_reason"] == "adaptive"
