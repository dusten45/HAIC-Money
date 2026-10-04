"""Full observed-circle support extends perception without erasing hazards."""
import importlib.util
from pathlib import Path

import numpy as np

from agents.apex_2026.fast_rear_clear_agent import Agent as Reference


ROOT = Path(__file__).resolve().parents[1]


def candidate_type():
    source = ROOT / "fast_early_far_agent.py"
    if not source.exists():
        return Reference
    spec = importlib.util.spec_from_file_location("early_far_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent


def saved(step):
    fixture = np.load(Path(__file__).parent / "fixtures" / "early-far-track2.npz")
    return fixture["frames"][int(np.flatnonzero(fixture["steps"] == step)[0])]


def circles(kind, frame):
    agent = kind()
    return agent._circles(frame, agent._road(frame))


def camera():
    frame = np.full((84, 84), .63, np.float32)
    frame[:73, 33:52] = .4
    frame[73:] = 0.
    return frame


def test_saved_fully_visible_circle_is_seen_one_action_earlier():
    frame = saved(198)
    assert circles(Reference, frame) == []
    detected = circles(candidate_type(), frame)
    assert len(detected) == 1
    np.testing.assert_allclose(detected[0], [32.33392122281011, -.36743092298647856])
    next_detected = circles(Reference, saved(199))
    assert detected[0][0] - next_detected[0][0] > 7.


def test_saved_two_by_four_circle_survives_quantized_aspect():
    frame = saved(166)
    assert circles(Reference, frame) == []
    detected = circles(candidate_type(), frame)
    assert len(detected) == 1
    np.testing.assert_allclose(detected[0], [32.50188964474678, 2.51952632905014])


def test_component_crossing_old_roi_has_one_unclipped_centroid():
    detected = circles(candidate_type(), saved(180))
    assert len(detected) == 1
    np.testing.assert_allclose(detected[0], [31.15814226925338, 22.41328630217519])


def test_earlier_circle_preserves_existing_near_hazard():
    frame = camera()
    frame[7:10, 41:43] = .67
    frame[30:33, 41:43] = .67
    old = circles(Reference, frame)
    detected = circles(candidate_type(), frame)
    assert len(old) == 1 and old[0] in detected
    assert len(detected) == 2


def test_partial_top_circle_and_unobserved_road_support_are_rejected():
    frame = camera()
    frame[:3, 41:43] = .67
    assert circles(candidate_type(), frame) == circles(Reference, frame) == []
    frame = camera()
    frame[5:8, 41:43] = .67
    assert circles(candidate_type(), frame) == circles(Reference, frame) == []


def test_circle_with_radius_outside_observed_road_is_rejected():
    frame = camera()
    frame[7:10, 34:36] = .67
    assert circles(candidate_type(), frame) == circles(Reference, frame) == []


def test_grass_white_curb_and_long_texture_do_not_add_far_circles():
    for kind in ("grass", "curb", "texture"):
        frame = camera()
        if kind == "grass":
            frame[7:10, 41:43] = .63
        elif kind == "curb":
            frame[7:10, 41:43] = .67
            frame[8, 43] = .85
        else:
            frame[5:14, 41:43] = .67
        assert circles(candidate_type(), frame) == circles(Reference, frame) == []


def test_clear_action_defaults_reset_and_invalid_input_match_parent():
    controller, reference = candidate_type()(), Reference()
    for observation in (np.repeat(camera()[None], 4, axis=0),
                        np.full((4, 84, 84), np.nan)):
        np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
    controller.reset()
    reference.reset()
    for key in ("cruise_speed", "lateral_accel", "preview_time", "yaw_gain",
                "braking_accel", "obstacle_margin", "pass_side", "pass_missing"):
        assert getattr(controller, key) == getattr(reference, key)
    observation = np.repeat(camera()[None], 4, axis=0)
    np.testing.assert_array_equal(controller.act(observation), reference.act(observation))
