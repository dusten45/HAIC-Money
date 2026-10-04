"""A checked pass side may persist briefly when its full path disappears."""

from types import SimpleNamespace

import pytest

import agent


def _state(controller, *, plan=True, identity=1, bottom=24,
           own_width=2.07, other_width=1.07):
    controller._corridor_plan = (
        {"side": 1.0, "clearance_px": 1.035} if plan else None
    )
    controller._corridor_bbox = (38, bottom - 2, 41, bottom)
    controller._temporal_track = SimpleNamespace(
        identity=identity, observations=1 if plan else 2,
    )
    controller._temporal_assessment = SimpleNamespace(
        brake_required=True, selected_target_x=None,
        road_rows=(bottom + 1, bottom + 2, bottom + 3),
        left_width_px=other_width, right_width_px=own_width,
    )


def test_checked_side_holds_for_at_most_three_unverified_decisions(monkeypatch):
    controller = agent._BoundedSideHoldController()
    proposed = [0.02]
    monkeypatch.setattr(
        agent._HighSpeedBendPriorityController, "_adjust_obstacle_steering",
        lambda self, **kwargs: proposed[0],
    )
    _state(controller)
    assert controller._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=0.12, straight=True,
    ) == pytest.approx(0.02)

    proposed[0] = -0.03
    _state(controller, plan=False, bottom=40)
    for _ in range(3):
        assert controller._adjust_obstacle_steering(
            base_steering=-0.03, obstacle_bias=0.12, straight=False,
        ) == pytest.approx(0.07)
    assert controller._adjust_obstacle_steering(
        base_steering=-0.03, obstacle_bias=0.12, straight=False,
    ) == pytest.approx(-0.03)


@pytest.mark.parametrize("change", [
    {"identity": 2},
    {"bottom": 53},
    {"own_width": 1.99},
    {"other_width": 1.5},
])
def test_side_hold_abstains_when_original_side_is_not_visibly_open(
    monkeypatch, change,
):
    controller = agent._BoundedSideHoldController()
    monkeypatch.setattr(
        agent._HighSpeedBendPriorityController, "_adjust_obstacle_steering",
        lambda self, **kwargs: -0.03,
    )
    _state(controller)
    controller._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=0.12, straight=True,
    )
    follow_up = {"plan": False, "bottom": 40, **change}
    _state(controller, **follow_up)
    assert controller._adjust_obstacle_steering(
        base_steering=-0.03, obstacle_bias=0.12, straight=False,
    ) == pytest.approx(-0.03)


def test_hold_state_clears_on_reset(monkeypatch):
    controller = agent._BoundedSideHoldController()
    monkeypatch.setattr(
        agent._HighSpeedBendPriorityController, "_adjust_obstacle_steering",
        lambda self, **kwargs: 0.01,
    )
    _state(controller)
    controller._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=0.12, straight=True,
    )
    assert controller._side_hold_identity == 1
    controller.reset()
    assert controller._side_hold_identity is None
    assert controller._side_hold_side == 0.0
    assert controller._side_hold_decisions == 0


def test_no_arm_without_a_checked_early_spatial_path(monkeypatch):
    controller = agent._BoundedSideHoldController()
    monkeypatch.setattr(
        agent._HighSpeedBendPriorityController, "_adjust_obstacle_steering",
        lambda self, **kwargs: -0.03,
    )
    _state(controller, plan=False, bottom=40)
    assert controller._adjust_obstacle_steering(
        base_steering=-0.03, obstacle_bias=0.12, straight=False,
    ) == pytest.approx(-0.03)
    assert controller._side_hold_identity is None
