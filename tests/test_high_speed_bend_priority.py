"""A high-speed blind pass may not cancel a clearly visible road bend."""

from types import SimpleNamespace

import pytest

import agent


@pytest.mark.parametrize(
    "road, base, bias, inherited",
    [
        ({30: 32.0, 54: 40.5}, -0.058, 0.203, 0.034),
        ({30: 51.0, 54: 42.0}, 0.065, -0.19, -0.02),
    ],
)
def test_fast_uncertified_pass_preserves_bend_steering(
    monkeypatch, road, base, bias, inherited,
):
    controller = agent._HighSpeedBendPriorityController()
    controller._impact_previous_hud = 56.0
    controller._temporal_assessment = SimpleNamespace(brake_required=True)
    controller._corridor_bbox = (34, 36, 37, 39)
    controller._corridor_centers = road
    controller._corridor_plan = None
    calls = []

    def parent(self, *, base_steering, obstacle_bias, straight):
        calls.append((base_steering, obstacle_bias, straight))
        return inherited

    monkeypatch.setattr(agent._ImpactAwareSparseRoadController,
                        "_adjust_obstacle_steering", parent)
    result = controller._adjust_obstacle_steering(
        base_steering=base, obstacle_bias=bias, straight=False,
    )

    assert result == pytest.approx(base)
    assert calls == [(base, bias, False)]


@pytest.mark.parametrize(
    "change",
    [
        {"hud": 44.9},
        {"assessment": SimpleNamespace(brake_required=False)},
        {"bbox": (34, 30, 37, 35)},
        {"road": {30: 36.0, 54: 40.5}},
        {"plan": {"side": 1.0}},
        {"inherited": -0.025},
    ],
)
def test_bend_priority_abstains_without_high_speed_conflict(monkeypatch, change):
    controller = agent._HighSpeedBendPriorityController()
    controller._impact_previous_hud = change.get("hud", 56.0)
    controller._temporal_assessment = change.get(
        "assessment", SimpleNamespace(brake_required=True),
    )
    controller._corridor_bbox = change.get("bbox", (34, 36, 37, 39))
    controller._corridor_centers = change.get("road", {30: 32.0, 54: 40.5})
    controller._corridor_plan = change.get("plan")
    inherited = change.get("inherited", 0.034)

    monkeypatch.setattr(
        agent._ImpactAwareSparseRoadController, "_adjust_obstacle_steering",
        lambda self, **kwargs: inherited,
    )
    result = controller._adjust_obstacle_steering(
        base_steering=-0.058, obstacle_bias=0.203, straight=False,
    )

    assert result == pytest.approx(inherited)
