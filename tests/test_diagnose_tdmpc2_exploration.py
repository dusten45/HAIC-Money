import numpy as np
import pytest

from scripts import diagnose_tdmpc2_exploration as diagnosis
from scripts.diagnose_tdmpc2_exploration import action_for


def test_matched_draw_and_exclusive_pedal_mapping():
    draw = np.array([0.25, -0.6, 0.75], dtype=np.float32)
    model3, applied3 = action_for(draw, "independent_3d")
    model2, applied2 = action_for(draw, "exclusive_2d")
    np.testing.assert_allclose(model3, draw)
    np.testing.assert_allclose(model2, draw[:2])
    np.testing.assert_allclose(applied3, [0.25, 0.2, 0.875])
    np.testing.assert_allclose(applied2, [0.25, 0, 0.6])
    np.testing.assert_allclose(action_for(np.array([0, 0.7, -0.4], np.float32),
                                          "exclusive_2d")[1], [0, 0.7, 0])


def test_exclusive_actions_never_use_both_pedals():
    rng = np.random.default_rng(42)
    for _ in range(500):
        model, applied = action_for(rng.uniform(-1, 1, 3).astype(np.float32), "exclusive_2d")
        assert model.shape == (2,) and applied.shape == (3,)
        assert applied[1] * applied[2] == 0
        assert -1 <= applied[0] <= 1 and 0 <= applied[1] <= 1 and 0 <= applied[2] <= 1


def test_invalid_draw_and_arm_fail_closed():
    with pytest.raises(ValueError):
        action_for(np.zeros(2, dtype=np.float32), "exclusive_2d")
    with pytest.raises(ValueError):
        action_for(np.array([0, 0, 2], np.float32), "independent_3d")
    with pytest.raises(ValueError):
        action_for(np.zeros(3, dtype=np.float32), "unknown")


def test_collector_uses_underlying_road_identity_not_empty_reset_info(tmp_path, monkeypatch):
    class Hull:
        position = (0, 0)
        linearVelocity = (0, 0)
        angle = 0

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.car = type("Car", (), {"hull": Hull()})()
            self.track_id = None
            self.track_seed = None

        def reset(self, *, seed, options):
            self.track_id, self.track_seed = options["track_id"], seed
            return np.zeros((4, 84, 84), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((4, 84, 84), dtype=np.float32), 1.0, True, False, {
                "progress": .125, "damage": 0, "finished": False, "retire_reason": "off_track",
            }

        def close(self):
            pass

    monkeypatch.setattr(diagnosis, "make_training_env", lambda max_steps: Env())
    monkeypatch.setattr(diagnosis, "validate", lambda protocol, path: "f" * 64)
    protocol = {"repeats": 1, "cells": [{"track_id": 1, "geometry_seed": 3910800001}],
                "rng_seed": 42, "max_steps": 1, "max_decisions_per_arm": 2,
                "partition": "consumed-TRAIN-development"}
    result = diagnosis.collect(protocol, tmp_path / "run", "f" * 64)
    assert result["episodes"] == 2
    assert all(arm["decisions"] == 1 and arm["mean_progress"] == .125
               for arm in result["arms"].values())
    assert result["arms"]["exclusive_2d"]["finishes"] == 0
