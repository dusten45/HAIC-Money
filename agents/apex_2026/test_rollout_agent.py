"""Behavior checks using rendered-like pixel observations only."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np


def load_agent():
    path = Path(__file__).with_name('rollout_agent.py')
    assert path.exists(), 'Independent rollout controller has not been implemented'
    spec = importlib.util.spec_from_file_location('independent_rollout', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent()


def rendered_hud(speed):
    # Invoke the unchanged public renderer without creating a world or reset.
    from types import SimpleNamespace
    import pygame
    from core.vendor.car_racing import CarRacing, WINDOW_W, WINDOW_H, STATE_W, STATE_H
    from env_wrapper import image_preprocessing
    wheels = [SimpleNamespace(omega=0, joint=SimpleNamespace(angle=0)) for _ in range(4)]
    fake = SimpleNamespace(surf=pygame.Surface((WINDOW_W, WINDOW_H)),
                           car=SimpleNamespace(hull=SimpleNamespace(linearVelocity=(speed, 0),
                                                                   angularVelocity=0), wheels=wheels))
    CarRacing._render_indicators(fake, WINDOW_W, WINDOW_H)
    return image_preprocessing(CarRacing._create_image_array(fake, fake.surf, (STATE_W, STATE_H)))


def observation(points=((42, 83), (42, 0)), speed=0.0, obstacle=None):
    frame = np.full((84, 84), 0.58, dtype=np.float32)
    cv2.polylines(frame, [np.array(points, np.int32)], False, 0.4, 21)
    if obstacle is not None:
        cv2.circle(frame, obstacle, 4, 0.92, -1)
    frame[59:68, 40:44] = 0.24
    frame[74:] = 0
    frame[74:] = rendered_hud(speed)[74:]
    return np.stack([frame] * 4)


def test_straight_road_accelerates_with_finite_controls():
    action = np.asarray(load_agent().act(observation()))
    assert action.shape == (3,)
    assert np.isfinite(action).all()
    assert abs(action[0]) < 0.15
    assert action[1] > 0.5
    assert action[2] < 0.1


def test_left_and_right_bends_receive_opposite_steering():
    left = load_agent().act(observation(((42, 83), (42, 63), (18, 20), (5, 0)), speed=25))
    right = load_agent().act(observation(((42, 83), (42, 63), (66, 20), (79, 0)), speed=25))
    assert left[0] < -0.1
    assert right[0] > 0.1


def test_obstacle_changes_the_trajectory_before_contact():
    clear = load_agent().act(observation(speed=28))
    blocked = load_agent().act(observation(speed=28, obstacle=(42, 43)))
    assert abs(blocked[0]) > abs(clear[0]) + 0.1 or blocked[2] > clear[2] + 0.15


def test_fast_sharp_bend_brakes_more_than_straight():
    agent = load_agent()
    sharp = observation(((42, 83), (42, 57), (7, 44), (0, 44)), speed=70)
    turn = agent.act(sharp)
    straight = load_agent().act(observation(speed=70))
    assert turn[2] > straight[2] + 0.1
    assert turn[1] < straight[1]


def test_invalid_input_produces_finite_bounded_stop():
    action = np.asarray(load_agent().act(np.full((4, 84, 84), np.nan, np.float32)))
    assert np.isfinite(action).all()
    assert -1 <= action[0] <= 1
    assert 0 <= action[1] <= 1
    assert 0 <= action[2] <= 1
    assert action[1] == 0


def test_official_reset_accepts_initial_observation():
    agent = load_agent()
    agent.act(observation(((42, 83), (42, 63), (18, 20), (5, 0)), speed=25))
    agent.reset(observation())
    reset_action = agent.act(observation())
    np.testing.assert_array_equal(reset_action, load_agent().act(observation()))


def test_speed_decoder_matches_public_renderer_without_environment_reset():
    agent = load_agent()
    for speed in (0, 5, 10, 20, 30, 50, 70, 90, 100):
        assert abs(agent._speed(rendered_hud(speed)) - speed) < 1.5


def test_longitudinal_rollout_matches_observed_straight_dynamics():
    agent = load_agent()
    assert hasattr(agent, '_acceleration'), 'Rollout dynamics must be calibrated independently'
    # Wide intervals around required-cell measured before/after velocities;
    # these prevent nonphysical deceleration under full gas at ordinary speed.
    assert 35 < agent._acceleration(50.0, 1.0, 0.0) < 50
    assert -15 < agent._acceleration(50.0, 0.0, 0.0) < -5
    assert -130 < agent._acceleration(50.0, 0.0, 0.35) < -70
    assert -220 < agent._acceleration(50.0, 0.0, 0.8) < -150


def test_steering_rollout_respects_public_joint_target_and_limit():
    agent = load_agent()
    assert hasattr(agent, '_curvature'), 'Steering model must use physical joint limits'
    # Wheelbase3.24 and direct steering target are public car construction facts.
    assert np.isclose(agent._curvature(0.1), np.tan(0.1) / 3.24)
    assert np.isclose(agent._curvature(1.0), agent._curvature(0.4))
    assert np.isclose(agent._curvature(-0.1), -agent._curvature(0.1))


def test_small_orange_obstacle_survives_road_mask_smoothing():
    frame = observation()[-1]
    cv2.circle(frame, (42, 43), 1, (0.299 * 255 + 0.587 * 165) / 255, -1)
    fields = load_agent()._fields(frame)
    assert fields[0][43, 42] == 0, 'Morphological smoothing must not erase small obstacles'


def test_pixel_motion_recovers_rotation_and_lateral_slip():
    agent = load_agent()
    assert hasattr(agent, '_motion'), 'Stateful rollouts need pixel motion estimates'
    rng = np.random.default_rng(182)
    previous = cv2.GaussianBlur(rng.uniform(0.25, 0.72, (84, 84)).astype(np.float32), (3, 3), 0)
    center = np.array([42.0, 63.0])
    theta = -0.08
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    scale = np.diag([agent.PIXELS_X, agent.PIXELS_Y])
    linear = scale @ rotation @ np.linalg.inv(scale)
    translation = center - linear @ center + np.array([-8 * agent.PIXELS_X * 0.08, 30 * agent.PIXELS_Y * 0.08])
    affine = np.column_stack([linear, translation]).astype(np.float32)
    current = cv2.warpAffine(previous, affine, (84, 84))
    estimate = agent._motion(previous, current)
    assert estimate[2]
    assert abs(estimate[0] - 1.0) < 0.15
    assert abs(estimate[1] - np.arctan2(8.0, 30.0)) < 0.12


def test_pixel_motion_without_texture_falls_back_safely():
    agent = load_agent()
    assert hasattr(agent, '_motion'), 'Motion estimation needs a no-texture fallback'
    empty = np.full((84, 84), 0.4, np.float32)
    estimate = agent._motion(empty, empty)
    assert np.isfinite(estimate[:2]).all()
    assert not estimate[2]


def test_incoming_yaw_changes_planned_steering_on_same_current_road():
    current = observation(speed=35)[-1]
    for y in range(8, 60, 13):
        for x in (7, 19, 64, 76):
            current[y:y + 5, x:x + 5] = 0.69
    matrix = cv2.getRotationMatrix2D((42, 63), -5, 1)
    previous = cv2.warpAffine(current, matrix, (84, 84), borderMode=cv2.BORDER_REPLICATE)
    stationary = load_agent().act(np.stack([current] * 4))
    spinning = load_agent().act(np.stack([previous, previous, previous, current]))
    assert spinning[0] < stationary[0] - 0.05


def test_stopped_car_can_plan_around_small_near_obstacle():
    frame = observation()[-1]
    cv2.circle(frame, (42, 52), 1, 0.68, -1)
    action = load_agent().act(np.stack([frame] * 4))
    assert action[1] >= 0.09, 'A slow feasible trajectory should escape a stationary local optimum'
    assert abs(action[0]) > 0.1


def test_high_incoming_yaw_avoids_power_oversteer_at_low_speed():
    current = observation(speed=20)[-1]
    for y in range(8, 60, 13):
        for x in (7, 19, 64, 76):
            current[y:y + 5, x:x + 5] = 0.69
    previous = cv2.warpAffine(current, cv2.getRotationMatrix2D((42, 63), -15, 1),
                             (84, 84), borderMode=cv2.BORDER_REPLICATE)
    agent = load_agent()
    action = agent.act(np.stack([previous, previous, previous, current]))
    assert agent.last_diagnostics['yaw_rate'] > 2.0
    assert action[1] <= 0.36
