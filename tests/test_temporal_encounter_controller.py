"""Checks for the diagnostic corridor margin and persistent near encounter."""

from types import SimpleNamespace

import pytest

import agent


def _near(controller, bottom, *, speed=15.0, width=4.0, identity=7):
    controller._encounter_speed = speed
    controller._corridor_bbox = (39, bottom - 3, 42, bottom)
    controller._temporal_track = SimpleNamespace(identity=identity, misses=0)
    controller._temporal_assessment = SimpleNamespace(
        brake_required=True, selected_side=-1,
        left_width_px=width, right_width_px=0.5,
        road_rows=(bottom - 3, bottom - 2, bottom - 1),
    )
    return controller._adjust_pedals(
        gas=0.12, brake=0.0, straight=False,
        obstacle=(float(bottom - 1), 40.0, 41.0),
    )


def test_marginal_full_corridor_does_not_override_temporal_warning(monkeypatch):
    controller = agent._TemporalCorridorMarginController()
    controller._temporal_track = SimpleNamespace(uncertainty_px=1.0, observations=3)
    controller._temporal_assessment = SimpleNamespace(
        brake_required=True, selected_side=-1, selected_target_x=None,
    )
    monkeypatch.setattr(
        agent._TemporalReachabilityController, "_corridor_candidate",
        lambda self, side: (44.135, 0.235, 28.0) if side > 0 else None,
    )

    assert controller._corridor_candidate(1.0) is None


def test_healthy_full_corridor_remains_available(monkeypatch):
    controller = agent._TemporalCorridorMarginController()
    controller._temporal_track = SimpleNamespace(uncertainty_px=1.0, observations=3)
    controller._temporal_assessment = SimpleNamespace(
        brake_required=False, selected_side=1, selected_target_x=47.0,
    )
    monkeypatch.setattr(
        agent._TemporalReachabilityController, "_corridor_candidate",
        lambda self, side: (47.0, 2.0, 37.0),
    )

    assert controller._corridor_candidate(1.0) == (47.0, 2.0, 37.0)


def test_full_plan_steering_uses_unscaled_geometric_command(monkeypatch):
    controller = agent._TemporalFullSteerController()
    monkeypatch.setattr(
        agent._TemporalReachabilityController, "_adjust_obstacle_steering",
        lambda self, **kwargs: setattr(self, "_corridor_plan", {"side": 1.0}) or 0.035,
    )

    assert controller._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=0.12, straight=True,
    ) == pytest.approx(0.07)


def test_near_box_crossing_58_does_not_rearm_brake():
    controller = agent._TemporalEncounterController()

    assert _near(controller, 56)[1] >= 0.12
    assert _near(controller, 57)[1] >= 0.12
    assert _near(controller, 59)[1] < 0.12
    assert _near(controller, 57)[1] < 0.12
    assert controller._encounter_brake_spent == 2


def test_near_occlusion_preserves_encounter_caution():
    controller = agent._TemporalEncounterController()
    _near(controller, 56)
    _near(controller, 57)
    controller._encounter_speed = 14.0
    controller._temporal_track = SimpleNamespace(identity=7, misses=1)
    controller._temporal_assessment = None
    controller._corridor_bbox = None

    gas, brake = controller._adjust_pedals(
        gas=0.12, brake=0.0, straight=False, obstacle=None,
    )

    assert gas == 0.0
    assert brake > 0.0
    assert controller._encounter_brake_spent == 2


def test_disappearance_after_near_box_passes_does_not_hold_brake():
    controller = agent._TemporalEncounterController()
    _near(controller, 59)
    _near(controller, 61)
    controller._encounter_speed = 17.0
    controller._temporal_assessment = None
    controller._corridor_bbox = None

    gas, brake = controller._adjust_pedals(
        gas=0.04, brake=0.0, straight=False, obstacle=None,
    )

    assert gas == 0.04
    assert brake == 0.0


def test_fast_disappearance_after_two_brake_frames_keeps_caution():
    controller = agent._TemporalEncounterController()
    _near(controller, 59, speed=27.0)
    _near(controller, 61, speed=27.0)
    controller._encounter_speed = 27.0
    controller._temporal_assessment = None
    controller._corridor_bbox = None

    gas, brake = controller._adjust_pedals(
        gas=0.04, brake=0.0, straight=False, obstacle=None,
    )

    assert gas == 0.0
    assert brake >= 0.04


def test_single_sighting_passed_box_releases_caution():
    controller = agent._TemporalEncounterController()
    _near(controller, 61, speed=32.0)
    controller._encounter_speed = 32.0
    controller._temporal_track = SimpleNamespace(identity=7, observations=1, misses=1)
    controller._temporal_assessment = None
    controller._corridor_bbox = None

    gas, brake = controller._adjust_pedals(
        gas=0.04, brake=0.0, straight=False, obstacle=None,
    )

    assert gas == 0.04
    assert brake == 0.0


def test_new_far_obstacle_gets_fresh_brake_budget():
    controller = agent._TemporalEncounterController()
    _near(controller, 56)
    _near(controller, 57)
    _near(controller, 59)

    assert _near(controller, 47, identity=8)[1] >= 0.12
    assert controller._encounter_brake_spent == 1


def test_stall_escape_requires_visible_width_and_stops_when_moving():
    controller = agent._TemporalEncounterController()
    _near(controller, 56)
    _near(controller, 57)
    _near(controller, 57, speed=1.0, width=4.0)
    _near(controller, 57, speed=0.4, width=4.0)
    gas, brake = _near(controller, 57, speed=0.2, width=4.0)
    assert gas >= 0.08 and brake == 0.0

    gas, _ = _near(controller, 57, speed=7.0, width=4.0)
    assert gas < 0.08
    _near(controller, 57, speed=0.5, width=2.0)
    gas, _ = _near(controller, 57, speed=0.2, width=2.0)
    assert gas < 0.08


def test_lost_road_clears_encounter_state():
    controller = agent._TemporalEncounterController()
    _near(controller, 56)

    controller._lost_road_action()

    assert controller._encounter_brake_spent == 0
    assert controller._encounter_last_box is None


def test_near_bend_does_not_lock_side_from_narrow_one_frame_opening(monkeypatch):
    controller = agent._TemporalEncounterController()
    controller._obstacle_side = 1.0
    controller._corridor_bbox = (26, 37, 29, 39)
    controller._corridor_detection = (38.0, 27.2, 30.4)
    controller._temporal_track = SimpleNamespace(observations=3, uncertainty_px=1.0)
    controller._temporal_assessment = SimpleNamespace(
        left_width_px=2.07, right_width_px=-2.93,
        selected_side=1, selected_target_x=None,
        road_rows=(30, 31, 32), brake_required=True,
    )
    monkeypatch.setattr(controller, "_corridor_candidate", lambda side: None)
    monkeypatch.setattr(
        agent._TemporalCombinedCorridorController, "_adjust_obstacle_steering",
        lambda self, **kwargs: kwargs["obstacle_bias"],
    )

    command = controller._adjust_obstacle_steering(
        base_steering=-0.1, obstacle_bias=0.24, straight=False,
    )

    assert command > 0.0
    assert controller._obstacle_side == 1.0
    assert controller._early_escape_side == 0.0


def test_first_sighting_chooses_strongly_open_side(monkeypatch):
    controller = agent._TemporalEncounterController()
    controller._obstacle_side = -1.0
    controller._corridor_bbox = (45, 22, 47, 25)
    controller._corridor_detection = (23.5, 46.0, 45.0)
    controller._temporal_track = SimpleNamespace(observations=1, uncertainty_px=1.0)
    controller._temporal_assessment = SimpleNamespace(
        left_width_px=0.07, right_width_px=3.07,
        selected_side=-1, selected_target_x=None,
        road_rows=(22, 23, 24), brake_required=True,
    )
    monkeypatch.setattr(controller, "_corridor_candidate", lambda side: None)
    monkeypatch.setattr(
        agent._TemporalCombinedCorridorController, "_adjust_obstacle_steering",
        lambda self, **kwargs: kwargs["obstacle_bias"],
    )

    command = controller._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=-0.12, straight=True,
    )

    assert command > 0.0
    assert controller._obstacle_side == 1.0
