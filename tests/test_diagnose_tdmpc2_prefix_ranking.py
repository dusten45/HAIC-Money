"""Synthetic-only tests: no real TD-MPC2 artifact is loaded or environment reset."""

from copy import deepcopy
import hashlib
import math

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.replay import EpisodeReplay
from scripts.diagnose_tdmpc2_prefix_ranking import (
    DISCOUNT, PINS, PIXELS, PROTOCOL, ROADS, RUN, LoggedWindow,
    _native_action_trace, bind_logged_windows, rank_logged_windows, score_frozen,
)


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = TDMPC2ModelConfig(num_bins=0)

    def encode(self, obs, task=None):
        return obs[:, 0, 0, 0].float()[:, None]

    def reward(self, z, action, task=None):
        return z + action[:, :1]

    def next(self, z, action, task=None):
        return z + action[:, :1]


def sample():
    """Four distinct consumed TRAIN starts; one short episode is not H=3."""
    protocol = {
        "format": "haic-tdmpc2-reused-train-pilot-v1",
        "purpose": "reused-TRAIN-engineering-pilot", "run_dir": RUN,
        "source_sha256": {"haic/algorithms/tdmpc2/model.py": "0" * 64},
        "cells": [{"track_id": 1, "geometry_seed": seed} for seed in ROADS],
        "episode_schedule": [0, 1, 2, 3],
        "environment": {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"},
        "training": {"horizon": 3, "discount": DISCOUNT, "observation_shape": list(PIXELS),
                     "episodic": True, "replay_capacity": 14000, "max_steps": 2000},
    }
    replay = EpisodeReplay(capacity=14000, horizon=3)
    rows = [{"event": "start", "protocol_sha256": PINS[PROTOCOL],
             "source_sha256": protocol["source_sha256"]}]
    decisions = 0
    for episode_id, (initial, reward, length) in enumerate(((10, 3., 3), (20, 1., 4),
                                                              (30, 5., 3), (40, 2., 2))):
        cell = protocol["cells"][episode_id]
        image = np.full(PIXELS, initial, np.uint8)
        replay.start_episode(image)
        rows.extend(({"event": event, "episode": episode_id, "decisions": decisions, **cell}
                     for event in ("reset_intent", "reset")))
        actions = []
        for step in range(length):
            action = np.array([0, -.5, 1], np.float32)
            actions.append(action)
            finished = step == length - 1
            replay.add_step(image, action, reward, terminated=finished)
            decisions += 1
            rows.append({"event": "step", "episode": episode_id, "decisions": decisions,
                         "reward": reward, "terminated": finished, "truncated": False,
                         "terminal": finished})
        stacked = np.stack(actions)
        rows.append({"event": "episode", "episode": episode_id, "decisions": decisions,
                     "length": length, "reward": length * reward, "terminated": True,
                     "truncated": False, "terminal": True, **cell,
                     "action_trace_sha256": hashlib.sha256(stacked.tobytes()).hexdigest(),
                     "native_action_trace_sha256": _native_action_trace(stacked),
                     "updates": episode_id, "pretrain_updates": 0})
        rows.append({"event": "checkpoint", "episodes": episode_id + 1,
                     "decisions": decisions, "updates": episode_id, "pretrain_updates": 0,
                     "sha256": PINS[f"{RUN}/boundary.pt"] if episode_id == 3 else "a" * 64})
    checkpoint = {"format": protocol["format"], "protocol_sha256": PINS[PROTOCOL],
                  "source_sha256": protocol["source_sha256"], "episodes": 4, "decisions": decisions,
                  "updates": 3, "pretrain_updates": 0, "replay": replay.state_dict()}
    return protocol, checkpoint, rows


def test_full_ledger_binding_and_observational_rank_only():
    windows, skipped = bind_logged_windows(*sample())
    assert (len(windows), skipped) == (3, 1)
    assert [w.geometry_seed for w in windows] == list(ROADS[:3])
    assert [w.terminal_on_last_step for w in windows] == [True, False, True]
    before = torch.get_rng_state().clone()
    result = rank_logged_windows(SmallModel(), windows)
    torch.testing.assert_close(torch.get_rng_state(), before)
    assert result["scope"] == "observational_logged_replay_only"
    assert result["same_state_counterfactual_ranking"] is False
    assert result["exact_prefix_parity_verified"] is False
    assert result["environment_resets"] == 0
    assert result["episodes_scored"] == 3
    assert result["pairs_across_distinct_episode_starts"] == {
        "concordant": 2, "discordant": 1, "predicted_tie": 0, "logged_tie": 0,
    }
    first = result["logged_windows"][0]
    assert first["predicted_reward_return"] == pytest.approx(10 * (1 + DISCOUNT + DISCOUNT**2))
    assert first["logged_raw_reward_return"] == pytest.approx(3 * (1 + DISCOUNT + DISCOUNT**2))


def test_open_loop_action_order_and_negative_rewards_without_q_bootstrap():
    actions = np.array([[1., 0., 0.], [0., 0., 0.], [-1., 0., 0.]], dtype=np.float32)
    window = LoggedWindow(5, 1, ROADS[0], np.full(PIXELS, 10, np.uint8), actions,
                          (-1., -2., -3.), True)
    result = rank_logged_windows(SmallModel(), [window])
    row = result["logged_windows"][0]
    assert row["predicted_reward_return"] == pytest.approx(11 + 11 * DISCOUNT + 10 * DISCOUNT**2)
    assert row["logged_raw_reward_return"] == pytest.approx(-1 - 2 * DISCOUNT - 3 * DISCOUNT**2)
    assert result["pairs_across_distinct_episode_starts"] == {
        "concordant": 0, "discordant": 0, "predicted_tie": 0, "logged_tie": 0,
    }


def test_roundoff_only_logged_returns_are_not_ranked_as_signal():
    base = np.zeros(PIXELS, dtype=np.uint8)
    actions = np.zeros((3, 3), dtype=np.float32)
    windows = [LoggedWindow(i, 1, ROADS[i % 4], base + i, actions.copy(),
                            (-.4, -.4, -.39401 + i * 1e-12), False)
               for i in range(4)]
    result = rank_logged_windows(SmallModel(), windows)
    assert result["ranking_identifiable"] is False
    assert result["concordance_on_comparable_pairs"] is None
    assert result["pairs_across_distinct_episode_starts"]["logged_tie"] == 6


def test_actual_reward_head_contract_on_synthetic_pixels():
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": PIXELS}, episodic=True))
    window = bind_logged_windows(*sample())[0][0]
    first = rank_logged_windows(model, [window])
    second = rank_logged_windows(model, [window])
    assert first == second
    assert math.isfinite(first["logged_windows"][0]["predicted_reward_return"])


@pytest.mark.parametrize("corrupt,reason", [
    (lambda p, c, r: c["replay"]["episodes"][1]["actions"].__setitem__((0, 0), .5), "action hash"),
    (lambda p, c, r: r[4].__setitem__("reward", 500.), "step/reward"),
    (lambda p, c, r: r[1].__setitem__("geometry_seed", ROADS[1]), "consumed TRAIN cell"),
    (lambda p, c, r: c["replay"]["episodes"][0]["terminated"].__setitem__(0, True), "reset boundary"),
    (lambda p, c, r: c["replay"]["episodes"][0].__setitem__("start_step", 1), "trimmed prefix"),
    (lambda p, c, r: c["replay"].__setitem__("active", {}), "boundary differs"),
    (lambda p, c, r: r[-1].__setitem__("sha256", "0" * 64), "last boundary"),
    (lambda p, c, r: r.append({"event": "partial"}), "last boundary"),
    (lambda p, c, r: p["cells"][0].__setitem__("geometry_seed", 123), "four-cell"),
])
def test_fail_closed_on_replay_ledger_road_and_boundary_drift(corrupt, reason):
    protocol, checkpoint, rows = sample()
    corrupt(protocol, checkpoint, rows)
    with pytest.raises(ValueError, match=reason):
        bind_logged_windows(protocol, checkpoint, rows)


def test_per_step_float32_reward_can_bind_original_ledger_precision():
    protocol, checkpoint, rows = sample()
    steps = [row for row in rows if row["event"] == "step" and row["episode"] == 0]
    for row in steps:
        row["reward"] = 3.00000000001  # Stored replay float32 is still exactly 3.
    windows, _ = bind_logged_windows(protocol, checkpoint, rows)
    assert windows[0].rewards == (3.00000000001,) * 3
    assert math.isfinite(rank_logged_windows(SmallModel(), windows)["logged_windows"][0]["logged_raw_reward_return"])


def test_nonfinite_predictions_and_malformed_actions_fail_closed():
    window = bind_logged_windows(*sample())[0][0]
    invalid = deepcopy(window)
    invalid.actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid logged"):
        rank_logged_windows(SmallModel(), [invalid])

    class NonfiniteModel(SmallModel):
        def next(self, z, action, task=None):
            return torch.full_like(z, float("nan"))

    with pytest.raises(ValueError, match="predicted latent"):
        rank_logged_windows(NonfiniteModel(), [window])


def test_fixed_file_entry_cannot_use_unpinned_synthetic_directory(tmp_path):
    with pytest.raises(ValueError, match="missing or changed frozen artifact"):
        score_frozen(tmp_path)
