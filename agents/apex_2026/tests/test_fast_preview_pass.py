"""Current-ego curved road and official obstacle geometry regressions."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "fast_preview_pass_agent.py"


def _module():
    spec = importlib.util.spec_from_file_location("apex_fast_preview_pass_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _camera(*, width=20/3, obstacle_distance=24, obstacle_side=1.6, speed=58, bend=True):
    frame = np.full((84, 84), .63, dtype=np.float32)
    for row in range(73):
        y = (63 - row) / 1.701
        center = -.0105*y*y + .252*y - .412 if bend else 0.0
        slope = -.021*y + .252 if bend else 0.0
        middle = 42 + 1.3608*center
        half = 1.3608*width*np.sqrt(1+slope*slope)
        frame[row, max(0, round(middle-half)):min(84, round(middle+half)+1)] = .4
    y0, x0 = 63 - 1.701*obstacle_distance, 42 + 1.3608*obstacle_side
    yy, xx = np.indices((84,84))
    obstacle = ((xx-x0)/1.3608)**2 + ((yy-y0)/1.701)**2 <= 1.2**2
    frame[obstacle] = .67
    frame[73:] = 0.0
    top, bottom = 81.9 - .042*speed, 81.9
    for row in range(73,84):
        for col in range(10,14):
            w = max(0.0, min(col+1.0,12.6)-max(float(col),10.5))
            h = max(0.0, min(row+1.0,bottom)-max(float(row),top))
            frame[row,col] = w*h
    return np.repeat(frame[None],4,axis=0)


def test_curved_pass_starts_at_actual_ego_and_avoids_wrong_way_fallback():
    agent = _module().Agent()
    action = agent.act(_camera())
    plan = getattr(agent, "last_pass_plan", None)
    assert plan is not None
    assert agent.mode == "curved_pass"
    assert action[0] < -.008
    assert agent.last_target > 45
    assert plan["x"][0] == pytest.approx(0.0, abs=1e-9)
    assert plan["heading"][0] == pytest.approx(0.0, abs=1e-9)
    bbox = agent._robust._corridor_bbox
    obstacle_x = (.5*(bbox[0]+bbox[2])-42)/1.3608
    obstacle_y = (63-.5*(bbox[1]+bbox[3]))/1.701
    separation = np.hypot(np.asarray(plan["x"])-obstacle_x,
                          np.asarray(plan["y"])-obstacle_y)
    assert separation.min() >= 3.5
    assert plan["road_margin"] >= .30
    assert plan["obstacle_margin"] >= .75


def test_narrow_road_cannot_be_certified_by_centerline_alone():
    agent = _module().Agent()
    action = agent.act(_camera(width=3.8, obstacle_side=0.0, bend=False))
    assert getattr(agent, "last_pass_plan", None) is None
    assert agent.mode != "curved_pass"
    assert np.isfinite(action).all()


def test_imminent_centered_obstacle_cannot_teleport_hull_to_passing_lane():
    agent = _module().Agent()
    action = agent.act(_camera(obstacle_distance=5, obstacle_side=0.0, speed=30, bend=False))
    assert getattr(agent, "last_pass_plan", None) is None
    assert agent.mode != "curved_pass"
    assert action[1] <= .3


def test_curved_pass_reset_repeats_action_and_invalid_frame_is_bounded():
    agent = _module().Agent()
    observation = _camera()
    first = agent.act(observation)
    agent.act(_camera(obstacle_distance=19))
    agent.reset()
    np.testing.assert_array_equal(agent.act(observation), first)
    invalid = agent.act(np.full((4,84,84),np.nan,dtype=np.float32))
    assert invalid.shape == (3,) and invalid.dtype == np.float32
    assert np.isfinite(invalid).all()
    assert np.all(invalid >= [-1,0,0]) and np.all(invalid <= [1,1,1])


def test_pass_steering_slew_uses_previous_emitted_action():
    agent = _module().Agent()
    agent.last_steer = .40
    agent._metric.last_steer = .40
    agent._robust._last_steer = .40
    action = agent.act(_camera())
    assert agent.mode == "curved_pass"
    assert abs(float(action[0])-.40) <= .24+1e-6


def test_steered_front_wheel_cannot_fit_inside_only_the_hull_margin():
    # Official front wheel center1.1m plus the rotated .28x.54m rectangle
    # reaches1.568m at the permitted .4-rad angle. A1.51m asphalt edge misses
    # that wheel although a1.2m body hull reports .31m margin.
    wheel_edge = 1.1+.28*np.cos(.4)+.54*np.sin(.4)
    assert wheel_edge > 1.51
    ahead = np.asarray([-4.0,0.0,4.0])
    left,right = np.full(3,-1.51),np.full(3,1.51)
    margin = _module().Agent._hull_road_margin(0.0,0.0,0.0,ahead,left,right)
    assert margin is None or margin < .30


@pytest.mark.parametrize("parameters", [{"pass_separation":3}, {"pass_cruise":float("nan")}, {"pass_cruise":130}])
def test_invalid_pass_geometry_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        _module().Agent(**parameters)


@pytest.mark.parametrize("behavior,arguments", [
    ("test_quantized_aligned_circle_accelerates_below_pursuit_grip_capacity", ()),
    ("test_pursuit_retains_sharp_corner_braking_and_physical_steering_bound", (-1,)),
    ("test_pursuit_retains_sharp_corner_braking_and_physical_steering_bound", (1,)),
    ("test_shorter_preview_corrects_visible_straight_offset_promptly", ()),
    ("test_fast_small_steering_leaves_rear_wheel_force_for_lateral_grip", ()),
    ("test_fast_straight_keeps_full_acceleration", ()),
    ("test_accelerating_tighter_bend_caps_gas_before_rear_wheel_spin", ()),
    ("test_near_force_limit_keeps_declared_gas_gate", ()),
    ("test_real_camera_obstacle_retains_guarded_robust_pedals", ()),
    ("test_reset_and_invalid_camera_preserve_finite_bounded_actions", ()),
    ("test_invalid_physical_controller_parameters_are_rejected", ({"preview_time":float("nan")},)),
    ("test_invalid_physical_controller_parameters_are_rejected", ({"lateral_accel":0},)),
    ("test_invalid_physical_controller_parameters_are_rejected", ({"high_gas_gate":2},)),
])
def test_previous_clear_road_camera_behaviors(behavior, arguments):
    # Reuse the real V1 camera behaviors against this new standalone source.
    # The prior tests and inference file remain unchanged.
    spec = importlib.util.spec_from_file_location("apex_pass_previous_tests",
        Path(__file__).with_name("test_fast_preview.py"))
    previous_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous_tests)
    previous_tests.SOURCE = SOURCE
    getattr(previous_tests, behavior)(*arguments)
