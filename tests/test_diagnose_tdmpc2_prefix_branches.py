"""Synthetic-only checks; never construct or reset the actual HAIC environment."""

import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.rlpd import prefix_parity as parity
from scripts import diagnose_tdmpc2_prefix_branches as probe


def _protocol():
    sources = {name: "0" * 64 for name in probe.V2_SOURCES}
    v2 = {
        "format": "haic-tdmpc2-reused-train-pilot-v1",
        "purpose": "reused-TRAIN-engineering-pilot", "run_dir": probe.RUN,
        "source_sha256": sources,
        "cells": [{"track_id": 1, "geometry_seed": seed} for seed in probe.ROADS],
        "episode_schedule": [0, 1, 2, 3],
        "environment": {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"},
        "training": {"max_steps": 2000, "horizon": 3, "discount": .995,
                     "observation_shape": [4, 64, 64], "replay_capacity": 14000,
                     "episodic": True},
    }
    spec = {
        "format": "haic-tdmpc2-v2-prefix-branches-v1",
        "purpose": "consumed-TRAIN-same-state-diagnostic",
        "v2_sha256": probe.V2_PINS.copy(),
        "source_sha256": {**sources, **{name: "1" * 64 for name in probe.NEW_SOURCES}},
        "runtime": {"python": "synthetic"},
        "anchors": [{"episode_id": ep, "start_step": step} for ep, step in probe.ANCHORS],
        "candidates": {"logged": None, **probe.FIXED_SUFFIXES},
        "environment": {"track_id": 1, "geometry_seeds": list(probe.ROADS),
                        "max_steps": 2000, "frame_skip": 4, "obstacles": True,
                        "reward_shaping": False},
        "horizon": 3, "discount": .995, "tie_tolerance": 1e-6,
        "augmentation_seed": 73301,
        "max_resets": len(probe.ANCHORS) * (2 + len(probe.FIXED_SUFFIXES)),
    }
    return spec, v2


@pytest.mark.parametrize("change", [
    lambda s, v: s["v2_sha256"].__setitem__(probe.V2_PROTOCOL, "0" * 64),
    lambda s, v: s["source_sha256"].__setitem__("core/vendor/car_racing.py", "f" * 64),
    lambda s, v: s["anchors"].pop(),
    lambda s, v: s["candidates"].__setitem__("gas", [[0, 0, 0]] * 3),
    lambda s, v: s.__setitem__("max_resets", 73),
    lambda s, v: v["episode_schedule"].reverse(),
    lambda s, v: s["environment"].__setitem__("reward_shaping", True),
])
def test_protocol_fails_closed_on_v2_schedule_source_candidates_and_budget(monkeypatch, change):
    monkeypatch.setattr(probe, "runtime_identity", lambda: {"python": "synthetic"})
    spec, v2 = _protocol()
    probe.check_protocol(spec, v2)
    change(spec, v2)
    with pytest.raises(ValueError, match="fixed source-bound"):
        probe.check_protocol(spec, v2)


def _small_replay(monkeypatch):
    monkeypatch.setattr(probe, "ANCHORS", ((0, 2), (1, 2)))
    monkeypatch.setattr(probe, "EXPECTED_COUNTS", {
        "decisions": 28, "episodes": 4, "updates": 6, "pretrain_updates": 0,
    })
    _, v2 = _protocol()
    episodes = []
    rows = [{"event": "start", "protocol_sha256": probe.V2_PINS[probe.V2_PROTOCOL],
             "source_sha256": v2["source_sha256"]}]
    decisions = 0
    for ep in range(4):
        cell = v2["cells"][ep]
        rows.extend({"event": event, "episode": ep, "decisions": decisions, **cell}
                    for event in ("reset_intent", "reset"))
        actions = np.tile(np.array([0., -1., 0.], np.float32), (7, 1))
        obs = np.full((8, *probe.PIXELS), ep, np.uint8)
        term = np.array([False] * 6 + [True], np.bool_)
        trunc = np.zeros(7, np.bool_)
        reward = np.full(7, .5, np.float32)
        episodes.append({"episode_id": ep, "start_step": 0, "actions": actions,
                         "observations": obs, "rewards": reward,
                         "terminated": term, "truncated": trunc, "terminal": term.copy()})
        for i in range(7):
            decisions += 1
            rows.append({"event": "step", "episode": ep, "decisions": decisions,
                         "reward": .5, "terminated": bool(term[i]), "truncated": False,
                         "terminal": bool(term[i])})
        rows.append({"event": "episode", "episode": ep, "decisions": decisions,
                     "length": 7, "reward": 3.5, "terminated": True, "truncated": False,
                     "terminal": True, **cell,
                     "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                     "native_action_trace_sha256": probe._native_trace(actions),
                     "updates": ep + 3, "pretrain_updates": 0})
        rows.append({"event": "checkpoint", "episodes": ep + 1, "decisions": decisions,
                     "updates": ep + 3, "pretrain_updates": 0,
                     "sha256": probe.V2_PINS[f"{probe.RUN}/boundary.pt"] if ep == 3 else "f" * 64})
    checkpoint = {"format": v2["format"], "protocol_sha256": probe.V2_PINS[probe.V2_PROTOCOL],
                  "source_sha256": v2["source_sha256"], "decisions": decisions,
                  "episodes": 4, "updates": 6, "pretrain_updates": 0,
                  "replay": {"format": "haic-tdmpc2-episode-replay-v1",
                             "capacity": 14000, "horizon": 3, "action_dim": 3,
                             "observation_shape": probe.PIXELS,
                             "size": decisions, "next_episode_id": 4,
                             "active": None, "episodes": episodes}}
    return v2, checkpoint, rows


def test_full_replay_and_ledger_binding_including_later_nonterminal_anchors(monkeypatch):
    v2, checkpoint, rows = _small_replay(monkeypatch)
    anchors = probe.bind_anchors(v2, checkpoint, rows)
    assert [(a.episode_id, a.seed, a.step) for a in anchors] == [
        (0, probe.ROADS[0], 2), (1, probe.ROADS[1], 2)]
    assert anchors[0].logged_observations.shape == (4, *probe.PIXELS)
    assert anchors[0].rewards == (.5, .5)
    assert anchors[0].logged_flags == ((False, False, False),) * 3


@pytest.mark.parametrize("change,reason", [
    (lambda c, r: c["replay"]["episodes"][0]["actions"].__setitem__((0, 0), .5), "action/native hash"),
    (lambda c, r: r[4].__setitem__("reward", 9.), "reward/flags"),
    (lambda c, r: r[1].__setitem__("geometry_seed", 5), "cell/reset"),
    (lambda c, r: c["replay"]["episodes"][0].__setitem__("start_step", 1), "boundary"),
    (lambda c, r: c["replay"]["episodes"][1]["terminal"].__setitem__(1, True), "terminal boundary"),
    (lambda c, r: c["replay"].__setitem__("active", {}), "identity"),
    (lambda c, r: r[-1].__setitem__("sha256", "a" * 64), "identity"),
    (lambda c, r: r.append({"event": "partial"}), "identity"),
])
def test_all_steps_and_hashes_block_corrupt_source(monkeypatch, change, reason):
    v2, checkpoint, rows = _small_replay(monkeypatch)
    change(checkpoint, rows)
    with pytest.raises(ValueError, match=reason):
        probe.bind_anchors(v2, checkpoint, rows)


class TinyRewardModel:
    cfg = SimpleNamespace(num_bins=0)

    def eval(self):
        return self

    def encode(self, pixels, task):
        return torch.zeros((1, 1))

    def reward(self, z, action, task):
        return action[:, 1:2] + z

    def next(self, z, action, task):
        return z + action[:, :1]


def test_reward_only_open_loop_and_pairwise_tie_terminal_accounting():
    base = np.zeros(probe.PIXELS, np.uint8)
    candidates = {"a": np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0]], np.float32),
                  "b": np.array([[0, 0, 0]] * 3, np.float32)}
    rng = torch.get_rng_state().clone()
    values = probe.predicted_returns(TinyRewardModel(), base, candidates, 9)
    torch.testing.assert_close(rng, torch.get_rng_state())
    assert values["a"] == pytest.approx(2 * .995)
    assert values["b"] == 0
    rows = [dict(predicted_reward_return=p, real_discounted_raw_return=r,
                 full_h3=True, terminal_excluded=t, ranking_eligible=not t)
            for p, r, t in ((4., 10., False), (1., 1., False),
                            (3., 10. + 1e-10, False), (10., 100., True))]
    result = probe.rank(rows)
    assert result["pairs"] == {"concordant": 2, "discordant": 0,
                               "real_tie": 1, "predicted_tie": 0,
                               "terminal_or_short_excluded": 3}
    assert result["concordance"] == 1
    for row in rows[:3]:
        row["real_discounted_raw_return"] = 1.0
    assert probe.rank(rows)["concordance"] is None


class FakeEnv:
    def __init__(self, *, drift=False, terminal=False):
        self.clock = 0
        self.drift = drift
        self.terminal = terminal
        self.closed = False
        self.steps = 0
        self.state_drift = False

    def reset(self):
        self.clock = 0
        return np.zeros((4, 84, 84), np.float32), {"track_id": 1, "seed": probe.ROADS[0]}

    def step(self, action):
        self.clock += 1
        self.steps += 1
        obs = np.full((4, 84, 84), float(self.clock) / 255, np.float32)
        if self.drift and self.steps == 1:
            obs.fill(.9)
        reward = .5 + float(action[1]) / 10 + float(action[0]) / 100
        done = self.terminal and self.clock == 2
        return obs, reward, done, False, {"finished": False}

    def close(self):
        self.closed = True


def _fake_signature(env, obs, *, track_id, seed, info, action=None, prefix_sha256=parity.INITIAL_HASH,
                    reward=None, terminated=None, truncated=None, raw_commands=None):
    assert track_id == 1 and seed == probe.ROADS[0]
    action_data = b"" if action is None else action.tobytes()
    state = (("hull.position", int(bool(getattr(env, "state_drift", False)) and env.clock >= 1)),
             ("outer.elapsed_steps", env.clock), ("road_sha256", "r" * 64))
    return parity.Signature(hashlib.sha256(action_data).hexdigest(), prefix_sha256,
                            hashlib.sha256(b"".join(raw_commands or ())).hexdigest(),
                            len(raw_commands or ()), "pixel", "raw", reward,
                            terminated, truncated, tuple(sorted(info.items())), state)


def _fake_tap(env, action):
    obs, reward, term, trunc, info = env.step(action)
    commands = (action.astype(np.float64).tobytes(),) * 4
    return obs, reward, term, trunc, info, commands


def _anchor():
    # Executed symmetric [0,-1,0] means native [0,0,.5] and reward .5.
    action = np.array([[0, -1, 0]], np.float32)
    logged = np.tile(action, (3, 1))
    obs = [np.full((4, 84, 84), i / 255, np.float32) for i in range(5)]
    pixels = np.stack([probe.model_observation(o) for o in obs])
    return probe.BoundAnchor(0, 1, probe.ROADS[0], 1, pixels[:2], action, (.5,),
                             logged, pixels[1:], (.5,) * 3,
                             ((False, False, False),) * 3)


def _mock_parity(monkeypatch):
    monkeypatch.setattr(parity, "_chain", lambda env: (None,) * 5)
    monkeypatch.setattr(parity, "snapshot", _fake_signature)
    monkeypatch.setattr(parity, "_step_with_raw_tap", _fake_tap)


def _fake_compare(env, expected):
    obs, info = env.reset()
    parity._assert_same(expected.initial, parity.snapshot(env, obs, track_id=1,
                       seed=expected.seed, info=info), "initial")
    for index, data in enumerate(expected.actions):
        action = np.frombuffer(data, np.float32).copy()
        obs, reward, term, trunc, info, raw = parity._step_with_raw_tap(env, action)
        actual = parity.snapshot(env, obs, track_id=1, seed=expected.seed, info=info,
                                 action=action, prefix_sha256=expected.steps[index].prefix_sha256,
                                 reward=reward, terminated=term, truncated=trunc, raw_commands=raw)
        parity._assert_same(expected.steps[index], actual, f"step[{index}]")
    return obs


def test_pixel_or_reward_mismatch_stops_baseline_on_first_step(monkeypatch):
    _mock_parity(monkeypatch)
    for drift, raw in ((True, .5), (False, .6)):
        env = FakeEnv(drift=drift)
        anchor = _anchor()
        anchor = probe.BoundAnchor(anchor.episode_id, anchor.track_id, anchor.seed,
                                   anchor.step, anchor.observations, anchor.actions,
                                   (raw,), anchor.logged_actions, anchor.logged_observations,
                                   anchor.logged_rewards, anchor.logged_flags)
        with pytest.raises(parity.ParityError, match="model_observation|raw reward"):
            probe.capture_bound_prefix(env, anchor)
        assert env.steps == 1


def test_logged_suffix_mismatch_and_terminal_are_detected_before_ranking(monkeypatch):
    _mock_parity(monkeypatch)
    anchor = _anchor()
    env = FakeEnv(drift=True)
    env.clock = anchor.step
    with pytest.raises(parity.ParityError, match="model_observation"):
        probe.execute_suffix(env, anchor, "logged", anchor.logged_actions, "r" * 64)
    env = FakeEnv(terminal=True)
    env.clock = anchor.step
    row = probe.execute_suffix(env, anchor, "coast", probe.candidate_actions(anchor)["coast"], "r" * 64)
    assert row["steps"] == 1 and row["terminal_excluded"] is True
    assert row["ranking_eligible"] is False


def test_identical_pixels_but_different_accessible_state_prevents_branch(monkeypatch):
    _mock_parity(monkeypatch)
    anchor = _anchor()
    captured, _ = probe.capture_bound_prefix(FakeEnv(), anchor)
    branch = FakeEnv()
    branch.state_drift = True
    with pytest.raises(parity.ParityError, match="step\\[0\\].state.hull.position"):
        _fake_compare(branch, captured)
    assert branch.steps == anchor.step  # No action suffix may be dispatched.


def test_run_is_explicit_and_preserves_failure_before_first_branch(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    anchor = _anchor()
    spec = {"source_sha256": {}, "runtime": {}, "augmentation_seed": 1, "max_resets": 6}
    monkeypatch.setattr(probe, "preflight", lambda *args: (spec, (anchor,), TinyRewardModel()))
    checked = []
    monkeypatch.setattr(probe, "_recheck_runtime_sources", lambda *args: checked.append(True))
    _mock_parity(monkeypatch)
    instances = []

    def make_env(*args, **kwargs):
        assert args == (1, probe.ROADS[0], 2000, 4)
        assert kwargs == {"reward_shaping": False, "obstacles": True}
        env = FakeEnv(drift=len(instances) == 1)
        instances.append(env)
        return env

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=make_env))
    output = tmp_path / "runs" / "branch-result.json"
    pre = probe.run(tmp_path, tmp_path / "unused.json", "a" * 64)
    assert pre["status"] == "preflight_only" and not instances and not output.exists()
    monkeypatch.setattr(parity, "_compare_replay_prefix", _fake_compare)
    with pytest.raises(parity.ParityError, match="model_observation"):
        probe.run(tmp_path, tmp_path / "unused.json", "a" * 64, execute=True, output=output)
    recorded = json.loads(output.read_text())
    assert recorded["status"] == "stopped_parity_or_execution_failure"
    assert recorded["environment_resets_attempted"] == 2
    assert len(checked) == 2
    assert recorded["anchors"][0]["anchor_model_observation_hex"] == anchor.observations[-1].tobytes().hex()
    assert recorded["same_state_counterfactual_ranking"] is False
    assert recorded["anchors"][0]["candidates"] == []
    assert all(env.closed for env in instances)


def test_all_synthetic_branches_match_anchor_and_preserve_exact_bytes(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    anchor = _anchor()
    spec = {"source_sha256": {}, "runtime": {}, "augmentation_seed": 1, "max_resets": 6}
    monkeypatch.setattr(probe, "preflight", lambda *args: (spec, (anchor,), TinyRewardModel()))
    monkeypatch.setattr(probe, "_recheck_runtime_sources", lambda *args: None)
    _mock_parity(monkeypatch)
    monkeypatch.setattr(parity, "_compare_replay_prefix", _fake_compare)
    instances = []

    def make_env(*args, **kwargs):
        env = FakeEnv()
        instances.append(env)
        return env

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=make_env))
    output = tmp_path / "runs" / "result.json"
    result = probe.run(tmp_path, tmp_path / "unused.json", "a" * 64,
                       execute=True, output=output)
    assert result == json.loads(output.read_text())
    assert result["status"] == "complete"
    assert result["environment_resets_attempted"] == 6
    entry = result["anchors"][0]
    assert entry["anchor_model_observation_sha256"] == hashlib.sha256(anchor.observations[-1].tobytes()).hexdigest()
    assert [row["candidate"] for row in entry["candidates"]] == list(probe.candidate_actions(anchor))
    for row in entry["candidates"]:
        assert row["steps"] == 3 and row["ranking_eligible"]
        assert row["model_action_bytes_hex"] == probe.candidate_actions(anchor)[row["candidate"]].tobytes().hex()
        assert math.isfinite(row["real_discounted_raw_return"])
    assert sum(result["aggregate_pair_counts"].values()) == 10
    assert all(env.closed for env in instances)


def test_changed_physics_source_fails_before_reset(monkeypatch, tmp_path):
    (tmp_path / "train.py").write_text("frozen source bytes")
    monkeypatch.setattr(probe, "runtime_identity", lambda: {"python": "synthetic"})
    with pytest.raises(ValueError, match="missing or changed frozen source"):
        probe._recheck_runtime_sources(tmp_path, {
            "source_sha256": {"train.py": "f" * 64}, "runtime": {"python": "synthetic"},
        })


def test_invalid_protocol_sha_blocks_before_any_pickle_or_reset(monkeypatch, tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "branch.json").write_text("{}")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("must not unpickle"))
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=lambda *a, **kw: pytest.fail("must not build")))
    with pytest.raises(ValueError, match="missing or changed frozen source"):
        probe.run(tmp_path, tmp_path / "experiments" / "branch.json", "0" * 64,
                  execute=True, output=tmp_path / "runs" / "x.json")
