import numpy as np
import pytest

from haic.algorithms.tdmpc2.action_2d import environment_action, model_action


@pytest.mark.parametrize("model,native", [
    ([-1, -1], [-1, 0, 1]), ([0, 0], [0, 0, 0]), ([1, 1], [1, 1, 0]),
    ([.3, -.25], [.3, 0, .25]),
])
def test_exclusive_action_roundtrip(model, native):
    np.testing.assert_allclose(environment_action(model), native)
    np.testing.assert_allclose(model_action(native), model)


def test_invalid_or_simultaneous_pedals_rejected():
    for action in ([0, .1, .1], [0, 2, 0], [0, 0, float("nan")]):
        with pytest.raises(ValueError):
            model_action(np.asarray(action, dtype=np.float32))
    for action in ([0, 0, 0], [0, 2], [0, float("inf")]):
        with pytest.raises(ValueError):
            environment_action(np.asarray(action, dtype=np.float32))
