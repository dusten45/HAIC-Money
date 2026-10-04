import numpy as np
import pytest

from haic.algorithms.koi import collision_shield as shield
from haic.algorithms.koi import trajectory_rollout as model
from haic.algorithms.koi.trajectory_rollout import MAX_ACTIONS, rollout_trajectories


def references(offsets=(0.,), end=40.):
    forward = np.linspace(0., end, 161)
    x = np.asarray(offsets)[:, None] * np.sin(np.pi * forward / end)**2
    yaw = np.arctan(np.gradient(x, forward, axis=1))
    yaw[:, [0, -1]] = 0.
    return np.stack((x, np.broadcast_to(forward, x.shape), yaw), axis=2)


def test_straight_has_zero_command_and_exact_terminal():
    paths, commands, variation = rollout_trajectories(references(), 44., 0., 3.)
    assert commands.dtype == np.float32
    assert paths.dtype == variation.dtype == np.float64
    assert commands.shape == variation.shape == (1,)
    np.testing.assert_array_equal(paths[:, :, (0, 2)], 0.)
    np.testing.assert_array_equal(commands, [0.])
    np.testing.assert_array_equal(variation, [0.])
    assert paths[0, -1, 1] == pytest.approx(40.)


def test_bilateral_sign_lag_and_countersteer_rejoin():
    paths, commands, variation = rollout_trajectories(references((3., -3.)), 44., 0., 5.28)
    assert commands[0] > 0 and commands[1] == -commands[0]
    np.testing.assert_allclose(paths[0, :, 0], -paths[1, :, 0], atol=1e-12)
    np.testing.assert_allclose(paths[0, :, 1], paths[1, :, 1], atol=1e-12)
    np.testing.assert_allclose(paths[0, :, 2], -paths[1, :, 2], atol=1e-12)
    assert paths[0, :, 0].max() > 2.
    assert np.any(paths[0, :, 2] < 0.)
    assert abs(paths[0, -1, 0]) < .3
    assert variation[0] > .1 and variation[0] == pytest.approx(variation[1])


def test_tracking_failure_is_returned_for_caller_terminal_rejection():
    paths, _, _ = rollout_trajectories(references((3.,)), 44., 0., 3.)
    assert abs(paths[0, -1, 0]) > 1.


def test_each_lane_stops_at_its_own_terminal_and_is_padded_without_movement():
    reference = np.concatenate((references(end=20.), references(end=40.)))
    paths, _, variation = rollout_trajectories(reference, 44., 0., 3.)
    assert paths[0, -1, 1] == pytest.approx(20.)
    assert paths[1, -1, 1] == pytest.approx(40.)
    assert np.count_nonzero(np.diff(paths[0, :, 1]) == 0) > 0
    np.testing.assert_array_equal(variation, 0.)


def test_first_hold_uses_exact_returned_float32_command_and_frozen_model():
    paths, commands, _ = rollout_trajectories(references((3., -3.)), 44., .1, 3.)
    expected, _, hold = shield.project_paths(commands, 0., 44., .1, actions=1)
    np.testing.assert_array_equal(paths[:, :hold + 1], expected)
    assert paths.shape[1] > hold + 1


def test_next_command_tracks_actual_pose_and_variation_sums_issued_jumps(monkeypatch):
    reference = references((3.,))
    commands = []
    original = np.arctan

    def capture(value):
        result = original(value)
        commands.append(np.clip(result, -.4, .4).astype(np.float32))
        return result

    monkeypatch.setattr(model.np, 'arctan', capture)
    paths, first, variation = rollout_trajectories(reference, 44., .1, 5.28)
    np.testing.assert_array_equal(first, commands[0])
    expected_variation = np.abs(np.diff(np.asarray(commands, dtype=np.float64), axis=0)).sum(axis=0)
    np.testing.assert_array_equal(variation, expected_variation)
    _, _, hold = shield.project_paths(first, 0., 44., .1, actions=1)
    x, y, yaw = paths[0, hold]
    target_y = min(y + 5.28, reference[0, -1, 1])
    dx = np.interp(target_y, reference[0, :, 1], reference[0, :, 0]) - x
    dy = target_y - y
    lateral = dx * np.cos(yaw) - dy * np.sin(yaw)
    expected = np.float32(np.clip(original(2 * shield.WHEELBASE * lateral / (dx * dx + dy * dy)), -.4, .4))
    assert commands[1][0] == expected


def test_opposite_initial_wheel_is_not_instantly_replaced_by_positive_command():
    paths, commands, _ = rollout_trajectories(references((8.,)), 44., -.2, 5.)
    assert commands[0] > 0
    assert paths[0, 1, 0] < 0 and paths[0, 1, 2] < 0


def test_every_body_corner_sample_moves_at_most_half_vertical_pixel():
    paths, _, _ = rollout_trajectories(references((5., -5., 0.)), 79., -.3, 3.)
    sine, cosine = np.sin(paths[:, :, 2]), np.cos(paths[:, :, 2])
    for x in (-shield.HALF_WIDTH, shield.HALF_WIDTH):
        for y in (-shield.REAR, shield.FRONT):
            corners = paths[:, :, :2] + np.stack(
                (x * cosine + y * sine, -x * sine + y * cosine), axis=2)
            travel = np.linalg.norm(np.diff(corners, axis=1), axis=2)
            assert travel.max() * shield.Y_SCALE <= .5 + 1e-12


def test_determinism_bank_independence_and_no_input_mutation():
    reference = references((-3., 0., 3.))
    original = reference.copy()
    first = rollout_trajectories(reference, 44., .1, 3.)
    rollout_trajectories(references((7.,)), 20., -.3, 5.)
    second = rollout_trajectories(reference, 44., .1, 3.)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    single = rollout_trajectories(reference[:1], 44., .1, 3.)
    np.testing.assert_array_equal(single[1], first[1][:1])
    np.testing.assert_array_equal(single[2], first[2][:1])
    np.testing.assert_array_equal(single[0], first[0][:1, :single[0].shape[1]])
    np.testing.assert_array_equal(reference, original)


@pytest.mark.parametrize('speed', [None, np.nan, np.inf, -1., 0., 80.])
def test_unknown_invalid_and_saturated_speed_fail(speed):
    with pytest.raises(ValueError):
        rollout_trajectories(references(), speed, 0., 3.)


@pytest.mark.parametrize('wheel,preview', [(np.nan, 3.), (.41, 3.), (0., 0.),
                                          (0., -1.), (0., np.inf)])
def test_invalid_actuator_and_preview_fail(wheel, preview):
    with pytest.raises(ValueError):
        rollout_trajectories(references(), 44., wheel, preview)


def test_invalid_reference_shape_origin_and_forward_order_fail():
    reference = references()
    shifted = reference.copy()
    shifted[0, 0, 0] = 1.
    reverse = reference.copy()
    reverse[0, 5, 1] = reverse[0, 4, 1]
    nonfinite = reference.copy()
    nonfinite[0, 5, 2] = np.nan
    for bad in (reference[0], reference[:0], reference[:, :1], shifted, reverse, nonfinite):
        with pytest.raises(ValueError):
            rollout_trajectories(bad, 44., 0., 3.)


@pytest.mark.parametrize('speed', [.001, 1e-300])
def test_tiny_speed_work_is_bounded_and_incomplete_terminal_remains_visible(speed):
    paths, _, _ = rollout_trajectories(references(), speed, 0., 3.)
    assert paths.shape == (1, MAX_ACTIONS * 8 + 1, 3)
    assert paths[0, -1, 1] < 40.
    assert paths[0, -1, 1] == pytest.approx(MAX_ACTIONS * shield.ACTION_SECONDS * speed)
