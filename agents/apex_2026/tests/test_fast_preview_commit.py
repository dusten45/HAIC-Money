"""Camera-only commitment, motion and frozen V2 safety regressions."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "fast_preview_commit_agent.py"
FIXTURE = Path(__file__).with_name("fixtures") / "preview_v2_commit_camera.npz"


def _module():
    spec = importlib.util.spec_from_file_location("apex_preview_commit_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _previous_tests():
    spec = importlib.util.spec_from_file_location("apex_commit_previous_tests",
        Path(__file__).with_name("test_fast_preview_pass.py"))
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    tests.SOURCE = SOURCE
    return tests


def _yaw_frame(yaw):
    frame = np.zeros((84, 84), dtype=np.float32)
    start, end = sorted((63.0, 63.0+1.68*yaw))
    for row in range(75,81):
        h = max(0.0,min(row+1.0,79.8)-max(float(row),75.6))
        for col in range(51,83):
            w = max(0.0,min(col+1.0,end)-max(float(col),start))
            frame[row,col] = .299*h*w
    return frame


def test_real_camera_pass_keeps_reference_instead_of_tightening_to_joint_limit():
    observations = np.load(FIXTURE)["observations"]
    agent = _module().Agent()
    peaks = []
    for observation in observations:
        action = agent.act(observation)
        plan = agent.last_pass_plan
        assert plan is not None, "step112 must keep the original feasible curve"
        peaks.append(plan["max_curvature"])
        assert plan["road_margin"] >= .30
        assert plan["obstacle_margin"] >= .75
        assert plan["min_separation"] >= 3.5
        assert np.isfinite(action).all()
    assert agent.mode == "committed_pass"
    assert max(peaks) < .03, "same world curve cannot tighten 6x under camera motion"
    assert agent.last_target > 50


@pytest.mark.parametrize("yaw", [-5.8,-1.5,-.5,0.,.5,1.5,5.8])
def test_signed_yaw_uses_complete_bar_area_without_saturating(yaw):
    reader = getattr(_module().Agent,"_camera_yaw",None)
    assert reader is not None, "commit odometry requires a calibrated legal HUD reader"
    assert reader(_yaw_frame(yaw)) == pytest.approx(yaw,abs=2e-6)


def test_hud_boundary_row_does_not_create_rotation():
    reader = getattr(_module().Agent,"_camera_yaw",None)
    assert reader is not None
    frame = _yaw_frame(1.5)
    frame[73,:] = .5
    assert reader(frame) == pytest.approx(1.5,abs=2e-6)


def test_rigid_camera_motion_preserves_arc_and_rotates_forward_coordinate():
    transform = getattr(_module().Agent,"_camera_transform",None)
    assert transform is not None
    # Positive rightward ego yaw: a previous ahead point appears to the left;
    # the former rightward point also moves into positive current forward Y.
    x,y = transform(np.array([0.,10.]),np.array([10.,0.]),np.pi/2,0.,0.)
    np.testing.assert_allclose(x,[-10.,0.],atol=1e-9)
    np.testing.assert_allclose(y,[0.,10.],atol=1e-9)
    t=np.linspace(0.,.7,30)
    ox,oy=40*(1-np.cos(t)),40*np.sin(t)
    nx,ny=transform(ox,oy,.21,1.2,4.8)
    # Circle radius and all chord lengths must survive translation/rotation.
    np.testing.assert_allclose(np.hypot(np.diff(nx),np.diff(ny)),
                               np.hypot(np.diff(ox),np.diff(oy)),atol=1e-10)


def test_missing_current_circle_cannot_keep_an_unchecked_committed_pass():
    tests = _previous_tests()
    agent = _module().Agent()
    agent.act(tests._camera())
    assert agent.last_pass_plan is not None
    observation = tests._camera(obstacle_distance=45)
    agent.act(observation)
    assert agent.mode != "committed_pass"


def test_committed_curve_cannot_override_new_narrow_road():
    tests = _previous_tests()
    agent = _module().Agent()
    agent.act(tests._camera())
    assert agent.last_pass_plan is not None
    agent.act(tests._camera(width=2.8,obstacle_distance=20))
    assert agent.last_pass_plan is None
    assert agent.mode != "committed_pass"


def test_reset_discards_world_curve_and_repeats_original_action():
    observations = np.load(FIXTURE)["observations"]
    agent = _module().Agent()
    first = agent.act(observations[0])
    agent.act(observations[1])
    agent.reset()
    np.testing.assert_array_equal(agent.act(observations[0]),first)
    assert agent.mode == "curved_pass"


def test_stationary_circle_at_fast_hud_speed_invalidates_motion_association():
    observation = _previous_tests()._camera()
    agent = _module().Agent()
    agent.act(observation)
    # Repeating an unchanged circle while the HUD reports ~58m/s is not a
    # plausible next physical camera; a stale association must not be used.
    agent.act(observation)
    assert agent.mode != "committed_pass"
    assert agent.last_commit_rejection == "circle association disagrees with HUD motion"


def test_actual_ego_departure_cannot_keep_a_geometrically_clear_reference():
    tests = _previous_tests()
    agent = _module().Agent()
    first = tests._camera(bend=False,obstacle_side=1.6,speed=30)
    agent.act(first)
    assert agent.last_pass_plan is not None
    # At30m/s predicted motion is2.4m. A current same-circle displacement
    # error1.8m still fits association but shifts the path too far from ego.
    road = agent._metric._road(agent._metric._frame(first))
    previous = agent._commit_circle.copy()
    current = previous+np.asarray([1.8,-.08*agent._camera_previous_speed])
    bbox=(42+1.3608*current[0]-1,63-1.701*current[1]-1,
          42+1.3608*current[0]+1,63-1.701*current[1]+1)
    assert agent._transport_reference(road,bbox,agent._camera_previous_speed,0.) is None
    assert agent.last_commit_rejection == "actual ego departed the committed reference"


def test_extreme_heading_change_discards_inconsistent_camera_odometry():
    tests = _previous_tests()
    agent = _module().Agent()
    first=tests._camera()
    agent.act(first)
    road=agent._metric._road(agent._metric._frame(first))
    assert agent._transport_reference(road,agent._robust._corridor_bbox,58.,8.) is None
    # A single unrealistic yaw transition may be rejected by motion or circle
    # association; either must prevent acceptance of the old reference.
    assert agent.last_commit_rejection in {
        "inconsistent HUD motion","circle association disagrees with HUD motion"}


def test_current_branch_jump_invalidates_commit_even_with_wide_edge_margins():
    fixture=np.load(FIXTURE)
    observations=fixture["observations"]
    agent=_module().Agent()
    agent.act(observations[0])
    frame=agent._metric._frame(observations[1])
    road=agent._metric._road(frame)
    ahead,side,rows,lefts,rights=road
    # A camera branch-selection jump inside still-wide asphalt spans cannot
    # establish the single road model used to admit the original V2 pass.
    side=side.copy()
    side[(ahead>7.) & (ahead<12.)]+=3.5
    assert np.all((42+1.3608*side >= lefts) & (42+1.3608*side <= rights))
    ambiguous=(ahead,side,rows,lefts,rights)
    plan=agent._transport_reference(ambiguous,fixture["bboxes"][1],
        agent._metric._speed(frame),agent._camera_yaw(frame))
    assert plan is None
    assert agent.last_commit_rejection == "ambiguous current road model"


@pytest.mark.parametrize("behavior,arguments", [
    ("test_curved_pass_starts_at_actual_ego_and_avoids_wrong_way_fallback",()),
    ("test_narrow_road_cannot_be_certified_by_centerline_alone",()),
    ("test_imminent_centered_obstacle_cannot_teleport_hull_to_passing_lane",()),
    ("test_curved_pass_reset_repeats_action_and_invalid_frame_is_bounded",()),
    ("test_pass_steering_slew_uses_previous_emitted_action",()),
    ("test_steered_front_wheel_cannot_fit_inside_only_the_hull_margin",()),
    ("test_invalid_pass_geometry_parameters_are_rejected",({"pass_separation":3},)),
    ("test_invalid_pass_geometry_parameters_are_rejected",({"pass_cruise":float("nan")},)),
    ("test_invalid_pass_geometry_parameters_are_rejected",({"pass_cruise":130},)),
])
def test_previous_pass_safety_behaviors(behavior,arguments):
    getattr(_previous_tests(),behavior)(*arguments)


@pytest.mark.parametrize("behavior,arguments", [
    ("test_quantized_aligned_circle_accelerates_below_pursuit_grip_capacity",()),
    ("test_pursuit_retains_sharp_corner_braking_and_physical_steering_bound",(-1,)),
    ("test_pursuit_retains_sharp_corner_braking_and_physical_steering_bound",(1,)),
    ("test_shorter_preview_corrects_visible_straight_offset_promptly",()),
    ("test_fast_small_steering_leaves_rear_wheel_force_for_lateral_grip",()),
    ("test_fast_straight_keeps_full_acceleration",()),
    ("test_accelerating_tighter_bend_caps_gas_before_rear_wheel_spin",()),
    ("test_near_force_limit_keeps_declared_gas_gate",()),
    ("test_real_camera_obstacle_retains_guarded_robust_pedals",()),
    ("test_reset_and_invalid_camera_preserve_finite_bounded_actions",()),
    ("test_invalid_physical_controller_parameters_are_rejected",({"preview_time":float("nan")},)),
    ("test_invalid_physical_controller_parameters_are_rejected",({"lateral_accel":0},)),
    ("test_invalid_physical_controller_parameters_are_rejected",({"high_gas_gate":2},)),
])
def test_previous_clear_road_behaviors(behavior,arguments):
    _previous_tests().test_previous_clear_road_camera_behaviors(behavior,arguments)
