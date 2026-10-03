"""Camera-only temporal obstacle and passing-reachability primitives."""

import pytest

import agent


# Source-bound diagnostic bbox pixels, in (left, top, right, bottom) order.
_V3_TRACK_1_119_BBOXES = (
    (37, 28, 39, 31),  # steps 119..126
    (36, 33, 39, 36),
    (37, 37, 39, 40),
    (38, 40, 40, 43),
    (39, 44, 41, 46),
    (40, 47, 43, 50),
    (41, 50, 43, 53),
    (41, 53, 43, 55),
)
_V4_TRACK_2_408_BBOXES = (
    (47, 24, 49, 27),  # steps 233..244
    (47, 29, 50, 32),
    (48, 34, 50, 37),
    (47, 37, 50, 40),
    (47, 40, 49, 43),
    (46, 43, 48, 46),
    (44, 46, 47, 48),
    (43, 48, 45, 51),
    (41, 50, 44, 53),
    (40, 52, 42, 55),
    (39, 54, 41, 57),
    (39, 54, 41, 57),
)


def _latest(bbox):
    return (None, None, None, bbox)


def test_v4_four_frame_stack_estimates_closing_motion():
    # Removing stack history would leave this first sighting without a TTC.
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])

    assert track.bbox == _V4_TRACK_2_408_BBOXES[3]
    assert track.observations == 4
    assert track.vy == pytest.approx(5.0)
    assert track.closing_rate_upper == pytest.approx(5.0)
    assert track.misses == 0


def test_reset_tiled_frame_is_one_observation_not_four_motion_samples():
    # env_wrapper.reset() repeats one rendered frame across the full stack.
    track = agent._track_obstacle_stack(None, (_V3_TRACK_1_119_BBOXES[0],) * 4)

    assert track.observations == 1
    assert track.vy == 0.0
    assert track.closing_rate_upper == 0.0


def test_two_missing_frames_predict_and_reassociate_same_obstacle():
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])
    original_id = track.identity

    track = agent._track_obstacle_stack(track, _latest(None))
    assert track.misses == 1
    assert track.bbox[3] > 40.0
    track = agent._track_obstacle_stack(track, _latest(None))
    assert track.misses == 2

    track = agent._track_obstacle_stack(track, _latest(_V4_TRACK_2_408_BBOXES[6]))
    assert track.identity == original_id
    assert track.misses == 0
    assert track.observations == 5
    assert track.bbox == _V4_TRACK_2_408_BBOXES[6]


def test_distant_new_detection_does_not_inherit_obstacle_identity():
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])

    other = agent._track_obstacle_stack(track, _latest((8, 38, 11, 41)))

    assert other.identity != track.identity
    assert other.observations == 1
    assert other.vy == 0.0


def test_three_missing_frames_expire_a_track():
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])

    for _ in range(3):
        track = agent._track_obstacle_stack(track, _latest(None))

    assert track is None


def test_v3_step_120_keeps_the_observed_wide_right_side():
    # Actual v3 1/1190129265 pixels: a centroid-only fallback switched left.
    track = agent._track_obstacle_stack(
        None, (None, None, *_V3_TRACK_1_119_BBOXES[:2]),
    )
    assessment = agent._assess_temporal_pass(
        track,
        {
            38: (31.0, 49.0), 39: (30.0, 49.0), 40: (30.0, 49.0),
            41: (32.0, 49.0), 42: (33.0, 49.0), 43: (33.0, 49.0),
            44: (32.0, 49.0), 45: (32.0, 49.0), 46: (32.0, 49.0),
            47: (32.0, 49.0), 48: (32.0, 49.0), 49: (32.0, 49.0),
            50: (32.0, 49.0), 51: (32.0, 49.0), 52: (32.0, 49.0),
            53: (32.0, 49.0), 54: (32.0, 50.0), 55: (32.0, 50.0),
            56: (32.0, 50.0), 57: (32.0, 50.0), 58: (32.0, 50.0),
        },
        committed_side=1,
        last_steer=0.02108,
    )

    assert assessment.road_rows == (38, 39, 40)
    assert assessment.left_width_px < 0.0
    assert assessment.right_width_px > 3.0
    assert assessment.selected_side == 1
    assert assessment.selected_target_x is not None
    assert assessment.selected_path_margin_px > 0.0
    assert not assessment.brake_required


def test_v3_step_123_reversal_has_too_few_decisions_left():
    track = agent._track_obstacle_stack(None, _V3_TRACK_1_119_BBOXES[:4])
    track = agent._track_obstacle_stack(track, _latest(_V3_TRACK_1_119_BBOXES[4]))
    assessment = agent._assess_temporal_pass(
        track,
        {
            43: (30.8, 50.2), 44: (31.6, 50.4), 45: (33.0, 50.6),
            48: (33.0, 51.0), 49: (32.0, 51.0), 50: (32.0, 51.0),
            51: (34.0, 51.0), 52: (34.0, 51.0), 53: (34.0, 51.0),
            54: (34.0, 51.0), 55: (33.0, 51.0), 56: (33.0, 51.0),
            57: (33.0, 51.0), 58: (33.0, 51.0),
        },
        committed_side=-1,
        last_steer=-0.18892,
    )

    assert assessment.right_width_px > 3.0
    assert assessment.ttc_decisions < assessment.slew_decisions + 2
    assert assessment.selected_side == -1
    assert assessment.brake_required


def test_v4_step_237_rejects_the_visibly_pinched_right_switch():
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])
    track = agent._track_obstacle_stack(track, _latest(_V4_TRACK_2_408_BBOXES[4]))
    assessment = agent._assess_temporal_pass(
        track,
        {52: (32.0, 50.0), 53: (32.0, 51.0), 54: (31.0, 48.0)},
        committed_side=-1,
        last_steer=0.07,
    )

    assert assessment.left_width_px > 9.0
    assert assessment.right_width_px < 0.0
    assert assessment.ttc_decisions <= 2.0
    assert assessment.selected_side == -1
    assert assessment.brake_required


def test_v4_step_244_cannot_claim_a_new_pass_after_contact():
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])
    for bbox in _V4_TRACK_2_408_BBOXES[4:]:
        track = agent._track_obstacle_stack(track, _latest(bbox))
    assessment = agent._assess_temporal_pass(
        track,
        {54: (30.0, 54.5), 55: (30.0, 54.0), 56: (30.0, 54.0)},
        committed_side=-1,
        last_steer=0.32,
    )

    assert assessment.ttc_decisions == 0.0
    assert assessment.selected_side == -1
    assert assessment.brake_required


def test_v5_step_240_new_right_opening_has_no_reachable_approach():
    # V5 2/4089604952: rows 53..55 suddenly open, but row 58 and the
    # 10-row steering approach still block the right side before contact 243.
    track = agent._track_obstacle_stack(None, _V4_TRACK_2_408_BBOXES[:4])
    for bbox in (
        (47, 40, 49, 43), (46, 43, 48, 46),
        (45, 46, 47, 49), (44, 48, 46, 51),
    ):
        track = agent._track_obstacle_stack(track, _latest(bbox))
    assessment = agent._assess_temporal_pass(
        track,
        {
            53: (32.0, 59.0), 54: (31.8, 56.8), 55: (31.6, 54.6),
            56: (31.4, 52.4), 57: (31.2, 50.2), 58: (31.0, 48.0),
        },
        committed_side=-1,
        last_steer=0.11484,
    )

    assert assessment.right_width_px == pytest.approx(2.67)
    assert assessment.right_shift_per_row > 0.78
    assert assessment.right_approach_clearance_px < 0.0
    assert assessment.selected_side == -1
    assert assessment.brake_required


def test_missing_nearby_road_rows_cannot_certify_a_pass():
    track = agent._track_obstacle_stack(
        None, (None, None, *_V3_TRACK_1_119_BBOXES[:2]),
    )
    assessment = agent._assess_temporal_pass(
        track,
        {22: (16.0, 55.0), 58: (20.0, 60.0)},
        committed_side=1,
        last_steer=0.02,
    )

    assert assessment.road_rows == ()
    assert assessment.left_width_px is None
    assert assessment.right_width_px is None
    assert assessment.selected_side == 1
    assert assessment.brake_required


def test_slew_constraint_blocks_a_late_switch_even_when_both_sides_fit():
    track = agent._track_obstacle_stack(
        None,
        ((38, 30, 40, 33), (38, 35, 40, 38),
         (38, 40, 40, 43), (38, 45, 40, 48)),
    )
    assessment = agent._assess_temporal_pass(
        track,
        {row: (22.0, 59.0) for row in range(50, 59)},
        committed_side=-1,
        last_steer=-0.14,
    )

    assert assessment.left_width_px > 0.75
    assert assessment.right_width_px > 0.75
    assert assessment.ttc_decisions < assessment.slew_decisions + 2
    assert assessment.selected_side == -1
    assert assessment.brake_required
