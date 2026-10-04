"""Measured curved passage and obstacle footprint regressions."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'corridor_agent.py'


def module():
    spec = importlib.util.spec_from_file_location('apex_corridor_test', SOURCE)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def curve_center(row):
    distance = 63.0 - row
    return 41.5 + .25 * distance + .003 * distance * distance


def curved_controller():
    controller = module().Agent()._robust
    # An asphalt bend of the official 6.67m halfwidth. Projected camera
    # width increases with heading; these are directly measured outer edges.
    edges = {}
    for row in range(22, 59):
        slope = .25 + .006 * (63.0 - row)
        halfwidth = 6.6666666667 * 1.3608 * np.sqrt(1 + (1.25 * slope) ** 2)
        center = curve_center(row)
        edges[row] = (center - halfwidth, center + halfwidth)
    controller._corridor_edges = edges
    controller._corridor_bbox = (49, 36, 51, 39)
    controller._corridor_detection = (37.5, 50., curve_center(37.5))
    controller._corridor_centers = {row: curve_center(row)
                                    for row in (54, 50, 46, 42, 38, 34, 30)}
    return controller


def test_measured_curved_pass_is_not_rejected_for_the_road_turn():
    controller = curved_controller()
    passage = controller._corridor_candidate(1)
    assert passage is not None
    target, clearance, near_row = passage
    assert target > 54.3
    assert clearance >= 0
    assert near_row <= 58


def test_unneeded_distant_rows_do_not_reject_a_complete_pass_horizon():
    controller = curved_controller()
    controller._corridor_edges = {row: edge for row, edge in
                                  controller._corridor_edges.items() if row >= 26}
    assert controller._corridor_candidate(1) is not None


def test_missing_required_near_road_row_rejects_pass():
    controller = curved_controller()
    del controller._corridor_edges[55]
    assert controller._corridor_candidate(1) is None


def test_a_pass_that_leaves_the_observed_curve_is_rejected():
    controller = curved_controller()
    center = curve_center(34)
    controller._corridor_edges[34] = (center - 3.0, center + 3.0)
    assert controller._corridor_candidate(1) is None


def test_a_sharp_exit_kink_with_insufficient_hull_room_is_rejected():
    controller = curved_controller()
    assert controller._corridor_candidate(1) is not None
    # Center points still have the ordinary side margin. A long rotated
    # hull spans neighboring rows and cannot fit around this observed kink.
    for row in range(30, 35):
        left, right = controller._corridor_edges[row]
        shift = 4.0 * (1.0 - abs(row - 32) / 3.0)
        controller._corridor_edges[row] = (left + shift, right + shift)
    assert controller._corridor_candidate(1) is None


def test_road_intruding_between_hull_corners_is_rejected():
    controller = curved_controller()
    for row, shift in ((28, 1.5), (29, 3.0), (30, 1.5)):
        left, right = controller._corridor_edges[row]
        controller._corridor_edges[row] = (left + shift, right + shift)
    assert controller._corridor_candidate(1) is None


def test_partial_pass_with_an_unobserved_curve_anchor_rejects_cleanly():
    value = module()
    controller = value.Agent()._robust
    bbox = (36, 45, 38, 48)
    controller._corridor_edges = {row: (32., 53.) for row in range(53, 59)}
    controller._temporal_track = value._TemporalObstacleTrack(
        bbox, bbox, (), (), 0, 3, 1)
    assessment = value._TemporalPassAssessment(
        1, 0., 9., None, None, (53, 54, 55), True,
        None, None, .8, .3, None, None)
    assert controller._partial_near_target(assessment, bbox) is None


def test_vertical_curb_fragment_does_not_replace_the_round_obstacle():
    agent = module().Agent()
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[43:47, 39:42] = .671
    frame[50:59, 44:46] = .60
    frame[74:] = 0
    obstacle = agent._robust._nearest_obstacle(
        frame, agent._robust._road_centers(frame))
    assert obstacle is not None
    assert obstacle[0] < 50


def test_reset_discards_camera_hazard_and_actions_stay_bounded():
    agent = module().Agent()
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[43:47, 39:42] = .671
    frame[74:] = 0
    observation = np.tile(frame, (4, 1, 1))
    for _ in range(8):
        action = agent.act(observation)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
        assert np.all(action >= [-1, 0, 0])
        assert np.all(action <= [1, 1, 1])
    assert agent._robust._temporal_track is not None
    agent.reset()
    assert agent._robust._temporal_track is None
    assert agent._robust._corridor_plan is None
    assert agent._robust._obstacle_side == 0
    assert agent._metric.last_steer == 0


def test_full_visible_bend_span_survives_the_old_search_window():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    for row in range(22, 59):
        left = max(0, int(round(1 + 3 * (row - 54))))
        right = int(round(45 + .5 * (row - 54)))
        frame[row, left:right + 1] = .4
    bbox = (21, 44, 23, 47)
    frame[44:48, 21:24] = .671
    centers = {row: 35. for row in (54, 50, 46, 42, 38, 34, 30)}
    spans = controller._visible_edges(frame, centers, bbox)
    assert spans[54] == (1., 45.)
    assert spans[50] == (0., 43.)


def test_a_disconnected_road_is_not_merged_into_the_near_component():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[22:59, 10:28] = .4
    frame[22:59, 53:72] = .4
    frame[40:44, 17:20] = .671
    centers = {row: 20. for row in (54, 50, 46, 42, 38, 34, 30)}
    spans = controller._visible_edges(frame, centers, (17, 40, 19, 43))
    assert spans[54] == (10., 27.)
    assert spans[42] == (10., 27.)


def test_missing_near_asphalt_is_not_filled_from_neighboring_rows():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[22:59, 30:52] = .4
    frame[55] = .63
    centers = {row: 41. for row in (54, 50, 46, 42, 38, 34, 30)}
    spans = controller._visible_edges(frame, centers, (40, 40, 42, 43))
    assert 54 in spans and 56 in spans
    assert 55 not in spans


def test_border_clipped_asphalt_can_certify_an_interior_pass():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[22:74, :63] = .4
    frame[36:40, 25:28] = .671
    centers = controller._road_centers(frame)
    assert controller._nearest_obstacle(frame, centers) is not None
    assert controller._corridor_edges[54] == (0., 62.)
    assert controller._corridor_candidate(1) is not None


def test_hull_crossing_the_viewport_is_rejected_even_with_its_center_inside():
    controller = module().Agent()._robust
    controller._corridor_edges = {row: (0., 83.) for row in range(22, 59)}
    controller._corridor_edges[57] = (0., 80.)
    # The visible bend rotates the long hull at a near pose; its center at
    # x80 is in the image but a front corner reaches beyond x83.
    assert controller._corridor_hull_margin(
        58., 80., 58., 26, (15, 30, 17, 33)) is None


def test_only_the_measured_obstacle_can_bridge_an_asphalt_gap():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[22:59, 30:52] = .4
    frame[40:44, 40:43] = .671
    frame[42, 35:37] = .63
    centers = {row: 41.5 for row in (54, 50, 46, 42, 38, 34, 30)}
    spans = controller._visible_edges(frame, centers, (40, 40, 42, 43))
    assert spans[42] == (37., 51.)


def test_compact_white_curb_does_not_replace_the_orange_circle():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[43:47, 39:42] = 171 / 255
    # Saved target frame 81 had a compact curb fragment, not an elongated
    # component: its .855 peak distinguishes it from the orange circle.
    frame[60:62, 44:46] = 218 / 255
    obstacle = controller._nearest_obstacle(frame, controller._road_centers(frame))
    assert obstacle is not None
    assert obstacle[0] < 50


def test_a_compact_non_orange_component_is_not_an_obstacle():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[:74, 32:53] = .4
    frame[43:47, 39:42] = .60
    assert controller._nearest_obstacle(frame, controller._road_centers(frame)) is None


def fringe_fixture():
    controller = module().Agent()._robust
    frame = np.full((84, 84), .63, np.float32)
    frame[22:59, :42] = .4
    # Actual obstacle patch from target frame 80, x11..14/y43..46.
    # Pixel x11/y44 is a measured .537 halo outside the bright core bbox.
    frame[43:47, 11:15] = np.asarray([
        [117, 158, 167, 127],
        [137, 168, 171, 157],
        [132, 167, 171, 151],
        [112, 144, 151, 117],
    ], np.float32) / 255
    centers = {row: 32. for row in (54, 50, 46, 42, 38, 34, 30)}
    return controller, frame, centers, (12, 43, 14, 46)


def test_actual_one_pixel_orange_fringe_preserves_the_asphalt_topology():
    controller, frame, centers, bbox = fringe_fixture()
    assert controller._visible_edges(frame, centers, bbox)[44] == (0., 41.)


def test_a_second_fringe_pixel_does_not_bridge_an_unmeasured_gap():
    controller, frame, centers, bbox = fringe_fixture()
    frame[44, 10] = 137 / 255
    assert controller._visible_edges(frame, centers, bbox)[44] == (15., 41.)


def test_grass_adjacent_to_the_circle_does_not_count_as_occlusion():
    controller, frame, centers, bbox = fringe_fixture()
    frame[44, 11] = .63
    assert controller._visible_edges(frame, centers, bbox)[44] == (15., 41.)
