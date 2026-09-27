import numpy as np
import pytest

from haic.algorithms.tdmpc2.haic_env import (
    environment_action,
    episode_boundary,
    make_training_env,
    model_action,
    model_observation,
)


def test_observation_retains_order_and_uint8_pixel_contract():
    frames = np.stack([np.full((84, 84), v / 255, np.float32) for v in (0, 30, 100, 255)])
    result = model_observation(frames)
    assert result.shape == (4, 64, 64) and result.dtype == np.uint8
    assert list(result[:, 0, 0]) == [0, 30, 100, 255]
    with pytest.raises(ValueError):
        model_observation(np.ones((4, 84, 84), np.float64))


@pytest.mark.parametrize(
    "model,env",
    [
        ([-1, -1, -1], [-1, 0, 0]),
        ([0, 0, 0], [0, 0.5, 0.5]),
        ([1, 1, 1], [1, 1, 1]),
        ([0.2, 1, 1], [0.2, 1, 1]),
    ],
)
def test_action_mapping_is_bijective_with_independent_pedals(model, env):
    np.testing.assert_allclose(environment_action(model), env, atol=1e-7)
    np.testing.assert_allclose(model_action(env), model, atol=1e-7)


def test_action_validation_fails_closed():
    with pytest.raises(ValueError):
        environment_action([0, 2, 0])
    with pytest.raises(ValueError):
        model_action([0, 0, np.nan])


def test_finish_is_terminal_but_plain_time_limit_bootstraps():
    assert episode_boundary(True, False, {}) == (True, True)
    assert episode_boundary(False, True, {"finished": True}) == (True, True)
    assert episode_boundary(False, True, {"finished": False}) == (True, False)
    assert episode_boundary(False, False, {}) == (False, False)


def test_training_wrapper_preserves_official_raw_and_decision_limits_without_reset():
    env = make_training_env(2000)
    try:
        np.testing.assert_array_equal(env.action_space.low, [-1, 0, 0])
        np.testing.assert_array_equal(env.action_space.high, [1, 1, 1])
        assert env.observation_space.shape == (4, 84, 84)
        assert env._max_episode_steps == 2000
        assert env.env._skip_frames == 4 and env.env._no_operation == 50
        assert env.env.env._max_episode_steps == 2000 * 4 + 200
    finally:
        env.close()
