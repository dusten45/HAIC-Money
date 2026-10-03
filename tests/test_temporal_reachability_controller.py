"""Integration checks for the camera-only temporal reachability controller."""

from types import SimpleNamespace

import numpy as np
import pytest

import agent


def _controller():
    return agent._TemporalReachabilityController()


def _observation(*, obstacle=None):
    frame = np.full((84, 84), 0.7, dtype=np.float32)
    frame[22:62, 29:55] = 0.4
    if obstacle is not None:
        x, y = obstacle
        frame[y - 2:y + 2, x - 1:x + 2] = 0.65
    frame[77:83, 10:13] = (0.27 + 0.085 * 30.0) / 18.0
    return np.repeat(frame[None], 4, axis=0)


def test_clear_road_matches_previous_controller_and_reset_clears_track():
    previous = agent._FeasibleCorridorSideWidthController()
    candidate = _controller()
    clear = _observation()

    np.testing.assert_array_equal(candidate.act(clear), previous.act(clear))
    assert candidate._temporal_track is None
    assert candidate._temporal_assessment is None
    candidate.act(_observation(obstacle=(41, 40)))
    assert candidate._temporal_track is not None
    assert candidate._temporal_track.observations == 1

    candidate.reset()
    assert candidate._temporal_track is None
    assert candidate._temporal_assessment is None


def test_first_obstacle_sighting_uses_stack_motion_once():
    candidate = _controller()
    stack = np.stack([
        _observation(obstacle=(41, row))[-1]
        for row in (28, 33, 38, 43)
    ])

    candidate.act(stack)

    assert candidate._temporal_track is not None
    assert candidate._temporal_track.observations == 4
    assert candidate._temporal_track.vy == pytest.approx(5.0)


def test_reachability_assessment_blocks_late_switch_even_if_v5_width_gate_opens():
    previous = agent._FeasibleCorridorSideWidthController()
    candidate = _controller()
    for controller in (previous, candidate):
        controller._obstacle_side = -1.0
        controller._temporal_obstacle = (46.0, 47.2, -1.94)
        controller._corridor_detection = (49.5, 45.2, 47.45)
        controller._corridor_bbox = (44, 48, 46, 51)
        controller._corridor_edges = {
            53: (32.0, 59.0), 54: (31.8, 56.8), 55: (31.6, 54.6),
            56: (31.4, 52.4), 57: (31.2, 50.2), 58: (31.0, 48.0),
        }
    track = agent._track_obstacle_stack(
        None,
        ((47, 29, 50, 32), (47, 37, 50, 40),
         (45, 46, 47, 49), (44, 48, 46, 51)),
    )
    candidate._temporal_assessment = agent._assess_temporal_pass(
        track, candidate._corridor_edges,
        committed_side=-1, last_steer=0.11484,
    )
    candidate._temporal_track = SimpleNamespace(observations=8)

    assert previous._allow_obstacle_side_switch(
        obstacle_y=49.5, obstacle_x=45.2, candidate_side=1.0,
    )
    assert candidate._allow_obstacle_side_switch(
        obstacle_y=49.5, obstacle_x=45.2, candidate_side=1.0,
    ) is False
    requested = candidate._adjust_obstacle_steering(
        base_steering=0.24, obstacle_bias=-0.24, straight=False,
    )
    assert requested < 0.0


def test_unverified_far_track_allows_existing_early_curve_side_change():
    candidate = _controller()
    candidate._obstacle_side = 1.0
    candidate._temporal_obstacle = (32.4, 25.5, -2.6)
    candidate._corridor_detection = (32.5, 32.6, 31.5)
    candidate._corridor_bbox = (31, 31, 34, 34)
    candidate._corridor_edges = {}
    candidate._temporal_track = SimpleNamespace(observations=2)
    candidate._temporal_assessment = SimpleNamespace(
        selected_side=1, selected_target_x=None, brake_required=True,
    )

    assert candidate._allow_obstacle_side_switch(
        obstacle_y=32.5, obstacle_x=32.6, candidate_side=-1.0,
    ) is True


def test_no_corridor_speed_relief_requires_reachable_observed_path():
    candidate = _controller()
    obstacle = (36.0, 40.0, 41.0)
    inputs = dict(
        target_speed=45.0, steering=0.1, straight=False,
        obstacle=obstacle,
    )
    candidate._temporal_assessment = SimpleNamespace(
        selected_side=-1, selected_target_x=34.0,
        selected_path_margin_px=2.0, brake_required=False,
        ttc_decisions=8.0, slew_decisions=2.0,
        road_rows=(36, 37, 38, 39),
    )
    candidate._temporal_track = SimpleNamespace(observations=3)

    relieved = candidate._adjust_target_speed_for_steering(**inputs)
    assert 18.0 < relieved <= 45.0

    candidate._temporal_assessment.brake_required = True
    assert candidate._adjust_target_speed_for_steering(**inputs) == pytest.approx(18.0)
    candidate._temporal_assessment = None
    assert candidate._adjust_target_speed_for_steering(**inputs) == pytest.approx(18.0)


def test_occluded_near_obstacle_keeps_a_bounded_speed_target():
    candidate = _controller()
    candidate._temporal_track = SimpleNamespace(
        misses=1, last_seen_bbox=(38, 42, 41, 47),
    )

    occluded = candidate._adjust_target_speed_for_steering(
        target_speed=45.0, steering=0.1, straight=False, obstacle=None,
    )

    assert 18.0 <= occluded <= 24.0
    candidate._temporal_track = None
    assert candidate._adjust_target_speed_for_steering(
        target_speed=45.0, steering=0.1, straight=False, obstacle=None,
    ) == pytest.approx(45.0)


@pytest.mark.parametrize("bottom", [49, 55])
def test_unreachable_near_obstacle_brakes_briefly_then_crawls(bottom):
    candidate = _controller()
    candidate._temporal_assessment = SimpleNamespace(brake_required=True)
    candidate._corridor_bbox = (38, bottom - 5, 41, bottom)
    candidate._temporal_brake_frames = 0
    inputs = dict(gas=0.12, brake=0.0, straight=False, obstacle=(47.0, 39.5, 41.0))

    first = candidate._adjust_pedals(**inputs)
    second = candidate._adjust_pedals(**inputs)
    third = candidate._adjust_pedals(**inputs)

    assert first[0] == second[0] == 0.0
    assert first[1] > 0.0 and second[1] > 0.0
    assert 0.0 < third[0] <= 0.04 and third[1] == 0.0


def test_lost_road_resets_uncertain_brake_window():
    candidate = _controller()
    candidate._temporal_brake_frames = 2
    candidate._temporal_track = SimpleNamespace(misses=1)
    candidate._temporal_assessment = SimpleNamespace(brake_required=True)

    candidate._lost_road_action()

    assert candidate._temporal_brake_frames == 0
    assert candidate._temporal_track is None
    assert candidate._temporal_assessment is None


def test_one_frame_occlusion_does_not_restart_brake_window():
    candidate = _controller()
    candidate._temporal_track = SimpleNamespace(identity=7, misses=0)
    candidate._temporal_assessment = SimpleNamespace(brake_required=True)
    candidate._corridor_bbox = (38, 44, 41, 49)
    visible = dict(gas=0.12, brake=0.0, straight=False,
                   obstacle=(47.0, 39.5, 41.0))

    assert candidate._adjust_pedals(**visible)[0] == 0.0
    candidate._temporal_track = SimpleNamespace(identity=7, misses=1)
    candidate._temporal_assessment = None
    candidate._corridor_bbox = None
    candidate._adjust_pedals(gas=0.12, brake=0.0, straight=False, obstacle=None)
    candidate._temporal_track = SimpleNamespace(identity=7, misses=0)
    candidate._temporal_assessment = SimpleNamespace(brake_required=True)
    candidate._corridor_bbox = (38, 44, 41, 49)

    assert candidate._adjust_pedals(**visible)[0] == 0.0
    assert candidate._adjust_pedals(**visible)[0] > 0.0


def test_local_pass_carries_side_into_next_full_corridor_plan():
    candidate = _controller()
    candidate._corridor_detection = (38.0, 44.0, 42.0)
    candidate._corridor_bbox = (43, 36, 45, 40)
    candidate._corridor_edges = {}
    candidate._temporal_assessment = SimpleNamespace(
        selected_side=-1, selected_target_x=35.0,
    )

    candidate._adjust_obstacle_steering(
        base_steering=0.0, obstacle_bias=0.12, straight=True,
    )

    assert candidate._obstacle_side == -1.0
    assert candidate._corridor_previous == (44.0, 38.0, -1.0)


def test_visible_single_side_guides_steer_when_far_corridor_rows_are_missing():
    # Consumed 2/408 at step234: the right side is closed and a left approach
    # is visible near the car, although the far full-corridor proof is absent.
    candidate = _controller()
    candidate._obstacle_side = -1.0
    candidate._corridor_detection = (30.5, 48.6, 49.3)
    candidate._corridor_bbox = (47, 29, 50, 32)
    candidate._corridor_edges = {row: (32.0, 52.0) for row in range(41, 59)}
    track = agent._track_obstacle_stack(
        None, (None, None, (47, 24, 49, 27), (47, 29, 50, 32)),
    )
    candidate._temporal_track = track
    candidate._temporal_assessment = agent._assess_temporal_pass(
        track, candidate._corridor_edges,
        committed_side=-1, last_steer=-0.045,
    )
    assert candidate._temporal_assessment.selected_target_x is None
    assert candidate._temporal_assessment.left_width_px > 3.0
    assert candidate._temporal_assessment.right_width_px < 0.0

    requested = candidate._adjust_obstacle_steering(
        base_steering=0.1, obstacle_bias=-0.12, straight=False,
    )

    assert requested < 0.0
    assert candidate._obstacle_side == -1.0


def test_partial_target_rejects_visible_pinched_obstacle_row():
    candidate = _controller()
    bbox = (47, 29, 50, 32)
    candidate._corridor_bbox = bbox
    candidate._corridor_edges = {row: (32.0, 52.0) for row in range(29, 59)}
    candidate._corridor_edges[32] = (40.0, 52.0)
    track = agent._track_obstacle_stack(
        None, (None, None, (47, 24, 49, 27), bbox),
    )
    candidate._temporal_track = track
    assessment = agent._assess_temporal_pass(
        track, candidate._corridor_edges,
        committed_side=-1, last_steer=-0.045,
    )

    assert assessment.selected_target_x is None
    assert candidate._partial_near_target(assessment, bbox) is None


def test_partial_target_rejects_a_gap_in_near_car_road_rows():
    candidate = _controller()
    bbox = (47, 29, 50, 32)
    candidate._corridor_bbox = bbox
    candidate._corridor_edges = {
        row: (32.0, 52.0) for row in (*range(29, 55), 58)
    }
    track = agent._track_obstacle_stack(
        None, (None, None, (47, 24, 49, 27), bbox),
    )
    candidate._temporal_track = track
    assessment = agent._assess_temporal_pass(
        track, candidate._corridor_edges,
        committed_side=-1, last_steer=-0.045,
    )

    assert candidate._partial_near_target(assessment, bbox) is None


def test_partial_target_keeps_guiding_through_bottom_row_52():
    candidate = _controller()
    bbox = (47, 49, 50, 52)
    candidate._corridor_bbox = bbox
    candidate._corridor_edges = {row: (32.0, 60.0) for row in range(49, 59)}
    track = agent._track_obstacle_stack(
        None,
        ((47, 38, 50, 41), (47, 42, 50, 45),
         (47, 46, 50, 49), bbox),
    )
    candidate._temporal_track = track
    assessment = agent._assess_temporal_pass(
        track, candidate._corridor_edges,
        committed_side=-1, last_steer=-0.08,
    )

    assert assessment.selected_target_x is None
    assert candidate._partial_near_target(assessment, bbox) is not None


def test_ordinary_agent_route_remains_the_validated_baseline():
    runtime = agent.Agent("model.pt")

    assert isinstance(runtime._forward_controller, agent._CompoundClearingBrakeCarryController)
    assert not isinstance(runtime._forward_controller, agent._TemporalReachabilityController)
