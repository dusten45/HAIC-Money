"""Sparse-road curve preview must not override an observed pass."""

from types import SimpleNamespace

import numpy as np
import pytest

import agent


@pytest.mark.parametrize("bend", [-10.5, 10.5])
def test_preview_arms_only_without_local_road_rows(monkeypatch, bend):
    monkeypatch.setattr(
        agent._TemporalReachabilityController,
        "_adjust_obstacle_steering", lambda self, **kwargs: 0.0,
    )
    inputs = dict(base_steering=0.0, obstacle_bias=0.12, straight=False)

    def configured(rows):
        controller = agent._SparseRoadBendPreviewController()
        controller._corridor_bbox = (
            (24, 25, 26, 27) if bend < 0 else (46, 25, 48, 27)
        )
        controller._corridor_centers = {30: 42.0 + bend, 54: 42.0}
        far_edge = (23.0, 51.0) if bend < 0 else (33.0, 61.0)
        controller._corridor_edges = {
            30: far_edge, 31: far_edge, 32: far_edge,
            54: (33.0, 51.0),
        }
        controller._temporal_track = SimpleNamespace(identity=1)
        controller._temporal_assessment = SimpleNamespace(road_rows=rows)
        controller._corridor_plan = None
        return controller

    sparse = configured(())
    observed = configured((30, 31, 32))
    assert sparse._adjust_obstacle_steering(**inputs) == pytest.approx(
        -0.28 if bend < 0 else 0.28
    )
    assert sparse._preview_armed
    assert observed._adjust_obstacle_steering(**inputs) == 0.0
    assert not observed._preview_armed

    # An early sparse-road decision remains latched for this obstacle. A
    # few newly visible local rows are insufficient to certify a full pass.
    sparse._temporal_assessment.road_rows = (30, 31, 32)
    assert sparse._adjust_obstacle_steering(**inputs) == pytest.approx(
        -0.28 if bend < 0 else 0.28
    )
    sparse._corridor_plan = {"side": 1.0}
    assert sparse._adjust_obstacle_steering(**inputs) == 0.0


def test_preview_clears_on_reset_and_new_obstacle(monkeypatch):
    monkeypatch.setattr(
        agent._TemporalReachabilityController,
        "_adjust_obstacle_steering", lambda self, **kwargs: 0.0,
    )
    controller = agent._SparseRoadBendPreviewController()
    controller._corridor_bbox = (24, 25, 26, 27)
    controller._corridor_centers = {30: 31.5, 54: 42.0}
    controller._corridor_edges = {54: (33.0, 51.0)}
    controller._temporal_track = SimpleNamespace(identity=1)
    controller._temporal_assessment = SimpleNamespace(road_rows=())
    controller._corridor_plan = None
    inputs = dict(base_steering=0.0, obstacle_bias=0.12, straight=False)
    assert controller._adjust_obstacle_steering(**inputs) == pytest.approx(-0.28)

    controller._temporal_track = SimpleNamespace(identity=2)
    controller._temporal_assessment.road_rows = (30, 31, 32)
    assert controller._adjust_obstacle_steering(**inputs) == 0.0
    assert not controller._preview_armed
    controller.reset()
    assert controller._preview_track_identity is None
    assert not controller._preview_armed


def test_preview_expires_when_tracking_is_lost(monkeypatch):
    controller = agent._SparseRoadBendPreviewController()
    controller._preview_track_identity = 1
    controller._preview_armed = True
    controller._temporal_track = None
    monkeypatch.setattr(
        agent._TemporalReachabilityController,
        "act", lambda self, observation: np.zeros(3, dtype=np.float32),
    )

    controller.act(None)

    assert controller._preview_track_identity is None
    assert not controller._preview_armed
