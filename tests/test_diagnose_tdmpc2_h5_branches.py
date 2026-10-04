"""Synthetic-only RAW100k H5 branch checks; no actual HAIC env construction/reset."""

import copy
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.rlpd import prefix_parity as parity
from scripts import diagnose_tdmpc2_h5_branches as branch


def _protocol(monkeypatch):
    monkeypatch.setattr(branch, "runtime_identity", lambda: {"python": "synthetic"})
    sources = {"core/vendor/car_racing.py": "3" * 64}
    logged_sources = {"scripts/diagnose_tdmpc2_h5_logged.py": "4" * 64,
                      branch.HELPER: branch.HELPER_SHA}
    raw = {"format": "haic-tdmpc2-long-train-v1", "source_sha256": sources,
           "cells": [{"track_id": 1, "geometry_seed": seed} for seed in branch.ROADS],
           "episode_schedule": [0, 1, 2, 3],
           "selection": {"arm": "independent_3d", "action_dim": 3},
           "training": {"horizon": 3, "discount": branch.DISCOUNT}}
    spec = {
        "format": "haic-tdmpc2-raw100k-h5-branches-v1",
        "purpose": "consumed-TRAIN-same-reconstructed-accessible-state-H5-raw-reward",
        "source": {"protocol": {"path": "experiments/tdmpc2-long-reused-train-v2.json",
                                "sha256": branch.RAW_PROTOCOL_SHA},
                   "result": {"path": f"{branch.RUN}/result.json", "sha256": branch.RAW_RESULT_SHA},
                   "checkpoint": {"path": branch.CHECKPOINT, "sha256": branch.CHECKPOINT_SHA},
                   "training_ledger": {"path": f"{branch.RUN}/training.jsonl", "sha256": branch.TRAIN_SHA},
                   "step_ledger": {"path": f"{branch.RUN}/steps.jsonl", "sha256": branch.STEPS_SHA}},
        "logged_quality": {"protocol": {"path": branch.LOGGED_PROTOCOL, "sha256": branch.LOGGED_PROTOCOL_SHA},
                           "result": {"path": branch.LOGGED_RESULT, "sha256": branch.LOGGED_RESULT_SHA}},
        "source_sha256": {**sources, **{name: logged_sources.get(name, "5" * 64)
                                           for name in branch.EXTRA_SOURCES}},
        "runtime": {"python": "synthetic"},
        "anchors": [{"episode_id": ep, "start_step": step} for ep, step in branch.ANCHORS],
        "candidates": {"logged": None, **branch.FIXED_SUFFIXES},
        "environment": copy.deepcopy(branch.ENVIRONMENT), "horizon": 5, "discount": .995,
        "tie_tolerance": 1e-6, "augmentation_seed": 834, "max_resets": 72,
        "quality_gate": branch.GATE.copy(),
    }
    logged = {"format": "haic-tdmpc2-h5-logged-v1", "source_sha256": logged_sources,
               "environment_resets": 0, "selection": {"window_count": 256,
               "manifest_sha256": branch.LOGGED_MANIFEST_SHA,
               "horizons": [3, 5], "discount": .995, "constant_reward_per_step": 0.0}}
    for key, name in (("source_protocol", "protocol"), ("source_result", "result"),
                      ("checkpoint", "checkpoint"), ("training_ledger", "training_ledger"),
                      ("step_ledger", "step_ledger")):
        logged[key] = spec["source"][name].copy()
    return spec, raw, logged


@pytest.mark.parametrize("tamper", ["candidate", "old_h3", "gate", "cap", "road", "source",
                                    "logged", "resealed_result", "resealed_protocol", "manifest"])
def test_contract_requires_exact_raw_source_quality_roads_and_five_actions(monkeypatch, tamper):
    spec, raw, logged = _protocol(monkeypatch)
    branch.check_protocol(spec, raw, logged)
    if tamper == "candidate":
        spec["candidates"]["gas"] = [[0., 1., -1.]] * 3
    elif tamper == "old_h3":
        spec["format"] = "haic-tdmpc2-v2-prefix-branches-v1"
    elif tamper == "gate":
        spec["quality_gate"]["min_informative_real_pairs"] = 24
    elif tamper == "cap":
        spec["max_resets"] = 73
    elif tamper == "road":
        spec["environment"]["geometry_seeds"][0] = 99
    elif tamper == "source":
        spec["source_sha256"][branch.HELPER] = "a" * 64
    elif tamper == "logged":
        spec["logged_quality"]["result"]["path"] = "runs/tdmpc2-prefix-branches-20260928-v1.json"
    elif tamper == "resealed_result":
        spec["logged_quality"]["result"] = {
            "path": "runs/tdmpc2-long-v2-h5-logged-resealed.json", "sha256": "b" * 64}
    elif tamper == "resealed_protocol":
        spec["logged_quality"]["protocol"]["sha256"] = "c" * 64
    else:
        logged["selection"]["manifest_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="fixed RAW100k H5"):
        branch.check_protocol(spec, raw, logged)


def _quality(monkeypatch):
    spec, _, logged = _protocol(monkeypatch)
    windows = []
    for index in range(256):
        positive = index < 32
        term = [False] * 4 + [positive]
        windows.append({"episode_id": index, "start_step": 0, "track_id": 1,
                        "geometry_seed": branch.ROADS[index % 4],
                        "actions_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                        "actual_rewards": [1. + index % 3] * 5,
                        "predicted_rewards": [.9 * (1. + index % 3)] * 5,
                        "terminated": term, "truncated": [False] * 5, "terminal": term,
                        "predicted_terminal_probability": [.1] * 4 + [.9 if positive else .1]})
    manifest = [{key: row[key] for key in ("episode_id", "start_step", "actions_sha256")}
                for row in windows]
    logged["selection"]["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(branch, "LOGGED_MANIFEST_SHA", logged["selection"]["manifest_sha256"])
    weights = [branch.DISCOUNT**t for t in range(5)]
    mae = sum(.1 * row["actual_rewards"][0] * sum(weights) for row in windows) / 256
    base = sum(row["actual_rewards"][0] * sum(weights) for row in windows) / 256
    receipt = {"format": "haic-tdmpc2-h5-logged-result-v1", "quality_gate_passed": True,
               "scope": "in_replay_observational_fit_only", "same_state_counterfactual_ranking": False,
               "environment_resets": 0, "optimizer_steps": 0, "protocol_path": branch.LOGGED_PROTOCOL,
               "protocol_sha256": spec["logged_quality"]["protocol"]["sha256"],
               "source_protocol_sha256": branch.RAW_PROTOCOL_SHA,
               "source_result_sha256": branch.RAW_RESULT_SHA,
               "source_checkpoint_sha256": branch.CHECKPOINT_SHA,
               "training_ledger_sha256": branch.TRAIN_SHA,
               "step_ledger_sha256": branch.STEPS_SHA,
               "source_sha256": logged["source_sha256"],
               "window_manifest_sha256": logged["selection"]["manifest_sha256"],
               "window_count": 256, "road_window_counts": {str(seed): 64 for seed in branch.ROADS},
               "horizons": {"h5": {"windows": 256, "reward_transitions": 1280,
                                   "semantic_terminal_positive_transitions": 32,
                                   "semantic_terminal_negative_transitions": 1248,
                                   "terminal_true_positive": 32, "terminal_false_positive": 0,
                                   "mae_per_discounted_step": mae / sum(weights),
                                   "constant_mae_per_discounted_step": base / sum(weights)}},
               "windows": windows}
    return receipt, logged, spec


@pytest.mark.parametrize("tamper", ["old_h3", "bad_mae", "trivial_terminal", "different_manifest",
                                    "bad_timeout", "cross_reset", "false_pass_flag", "missing_pass_flag"])
def test_frozen_logged_quality_independently_recomputed_and_fails_closed(monkeypatch, tamper):
    receipt, logged, spec = _quality(monkeypatch)
    result = branch.check_logged_quality(receipt, logged, spec)
    assert result["windows"] == 256 and result["positive_transitions"] == 32
    if tamper == "old_h3":
        receipt["format"] = "haic-tdmpc2-v2-prefix-branches-result-v1"
    elif tamper == "bad_mae":
        for row in receipt["windows"]:
            row["predicted_rewards"] = [-1.] * 5
    elif tamper == "trivial_terminal":
        for row in receipt["windows"]:
            row["predicted_terminal_probability"] = [.1] * 5
    elif tamper == "different_manifest":
        receipt["windows"][0]["actions_sha256"] = "a" * 64
    elif tamper == "bad_timeout":
        receipt["windows"][0]["terminal"][4] = False
    elif tamper == "false_pass_flag":
        receipt["quality_gate_passed"] = False
    elif tamper == "missing_pass_flag":
        receipt.pop("quality_gate_passed")
    else:
        receipt["windows"][1]["truncated"][1] = True
    with pytest.raises(ValueError, match="receipt|gate|manifest|terminal"):
        branch.check_logged_quality(receipt, logged, spec)


def test_resealed_same_prefix_receipt_cannot_self_assert_quality(monkeypatch):
    receipt, logged, spec = _quality(monkeypatch)
    receipt["windows"][0]["actual_rewards"] = [500.] * 5
    receipt["windows"][0]["terminal"][-1] = False
    assert receipt["quality_gate_passed"] is True
    spec["logged_quality"]["result"] = {
        "path": "runs/tdmpc2-long-v2-h5-logged-resealed.json", "sha256": "b" * 64}
    _, raw, _ = _protocol(monkeypatch)
    with pytest.raises(ValueError, match="fixed RAW100k H5"):
        branch.check_protocol(spec, raw, logged)
    with pytest.raises(ValueError, match="receipt is missing or unbound"):
        branch.check_logged_quality(receipt, logged, spec)


def _replay_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(branch, "FINAL", {**branch.FINAL, "decisions": 440, "episodes": 4, "updates": 440})
    spec, raw, _ = _protocol(monkeypatch)
    episodes, stored, rows = [], [], []
    decision = 0
    for eid in range(4):
        length = 110
        actions = np.tile(np.array([0., -1., 0.], np.float32), (length, 1))
        obs = np.full((length + 1, *branch.PIXELS), eid, np.uint8)
        rewards = np.full(length, .5, np.float32)
        term = np.zeros(length, np.bool_)
        trunc = np.zeros(length, np.bool_)
        terminal = np.zeros(length, np.bool_)
        if eid % 2:
            trunc[-1] = True
            terminal[-1] = (eid == 3)  # A finish is semantic terminal; a timeout is not.
        else:
            term[-1] = terminal[-1] = True
        for offset, action in enumerate(actions):
            decision += 1
            rows.append({"episode": eid, "decision": decision, "track_id": 1,
                         "geometry_seed": branch.ROADS[eid],
                         "action_f32_hex": action.tobytes().hex(),
                         "native_action_f32_hex": branch.environment_action(action).tobytes().hex(),
                         "reward": .5, "terminated": bool(term[offset]),
                         "truncated": bool(trunc[offset]), "terminal": bool(terminal[offset])})
        native = b"".join(branch.environment_action(a).tobytes() for a in actions)
        episodes.append({"length": length, "episode": eid, "track_id": 1,
                         "geometry_seed": branch.ROADS[eid], "decisions": decision, "return": 55.,
                         "terminated": bool(term[-1]), "truncated": bool(trunc[-1]),
                         "terminal": bool(terminal[-1]),
                         "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                         "native_action_trace_sha256": hashlib.sha256(native).hexdigest()})
        stored.append({"episode_id": eid, "start_step": 0, "observations": obs,
                       "actions": actions, "rewards": rewards, "terminated": term,
                       "truncated": trunc, "terminal": terminal})
    path = tmp_path / "steps.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    replay = {"format": "haic-tdmpc2-episode-replay-v1", "capacity": 120000, "horizon": 3,
              "action_dim": 3, "augmentation_pad": 3, "include_partial": False,
              "bootstrap_on_truncation": True, "observation_shape": branch.PIXELS,
              "next_episode_id": 4, "size": 440, "active": None, "episodes": stored}
    state = {"format": raw["format"], "protocol_sha256": branch.RAW_PROTOCOL_SHA,
             "source_sha256": raw["source_sha256"], "learner": {"q_scale": torch.tensor(1.)},
             "replay": replay, **branch.FINAL}
    return state, raw, episodes, rows, path


def test_complete_h3_trained_replay_binds_all_440_steps_to_five_action_suffixes(monkeypatch, tmp_path):
    state, raw, episodes, _, path = _replay_fixture(monkeypatch, tmp_path)
    anchors = branch.bind_replay(state, raw, episodes, path)
    assert [(anchor.episode_id, anchor.step) for anchor in anchors] == list(branch.ANCHORS)
    assert len(anchors) == 12 and all(anchor.logged_actions.shape == (5, 3) for anchor in anchors)
    assert anchors[0].logged_rewards == (.5,) * 5
    assert len(candidate := branch.candidate_actions(anchors[0])) == 5
    assert candidate["coast"].tobytes() != candidate["logged"].tobytes()
    assert all(a.logged_observations.shape == (6, *branch.PIXELS) for a in anchors)


@pytest.mark.parametrize("tamper", ["model_action", "native_action", "reward", "flag", "road", "pixel",
                                    "extra_step", "h3_as_h5"])
def test_full_source_binding_detects_corrupt_step_or_replay_before_reset(monkeypatch, tmp_path, tamper):
    state, raw, episodes, rows, path = _replay_fixture(monkeypatch, tmp_path)
    if tamper == "model_action":
        state["replay"]["episodes"][2]["actions"][5, 0] = .25
    elif tamper == "native_action":
        rows[5]["native_action_f32_hex"] = "0" * 24
    elif tamper == "reward":
        rows[5]["reward"] = 9.
    elif tamper == "flag":
        rows[5]["terminal"] = True
    elif tamper == "road":
        rows[5]["geometry_seed"] += 1
    elif tamper == "pixel":
        state["replay"]["episodes"][1]["observations"] = np.zeros((110, *branch.PIXELS), np.uint8)
    elif tamper == "extra_step":
        rows.append(rows[-1])
    else:
        state["replay"]["episodes"][0]["actions"] = state["replay"]["episodes"][0]["actions"][:3]
    if tamper not in ("model_action", "pixel", "h3_as_h5"):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="lineage|shape|cursor"):
        branch.bind_replay(state, raw, episodes, path)


class TinyRewardModel:
    cfg = SimpleNamespace(num_bins=0)

    def eval(self):
        return self

    def encode(self, observation, task):
        return torch.zeros((1, 1))

    def reward(self, z, action, task):
        return z + action[:, 1:2]

    def next(self, z, action, task):
        return z + action[:, :1]

    def termination(self, z, task):
        return torch.zeros((z.shape[0], 1))

    def pi(self, z, task):
        return torch.zeros((z.shape[0], 3)), {}

    def Q(self, z, action, task, *, return_type):
        assert return_type == "avg"
        return torch.zeros((z.shape[0], 1))


def test_open_loop_world_model_H5_reward_only_no_q_or_bootstrap_and_native_mapping():
    model = TinyRewardModel()
    pixels = np.zeros(branch.PIXELS, np.uint8)
    candidates = {"x": np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0],
                                  [0, 0, 0], [0, 0, 0]], np.float32)}
    rng = torch.get_rng_state().clone()
    actual = branch.predicted_returns(model, pixels, candidates, 42)
    assert actual["x"] == pytest.approx(sum(branch.DISCOUNT**t * r
                                             for t, r in enumerate((0, 2, 0, 1, 1))))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    five = {name: actions.copy() for name, actions in {
        "x": candidates["x"], **{k: np.asarray(v, np.float32) for k, v in branch.FIXED_SUFFIXES.items()}}.items()}
    matched = branch.matched_model_scores(model, pixels, five, 42)
    assert matched["x"]["h5_reward"] == pytest.approx(actual["x"])
    assert matched["x"]["h3_reward"] == pytest.approx(sum(branch.DISCOUNT**t * r
                                                          for t, r in enumerate((0, 2, 0))))
    assert matched["x"]["h5_planner"] == pytest.approx(actual["x"])
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for key, native in (("coast", [0, 0, 0]), ("gas", [0, 1, 0]),
                        ("brake", [0, 0, 1]), ("left_gas", [-.5, 1, 0])):
        action = np.asarray(branch.FIXED_SUFFIXES[key][0], np.float32)
        np.testing.assert_array_equal(branch.environment_action(action), native)
    np.testing.assert_array_equal(branch.environment_action(np.zeros(3, np.float32)), [0, .5, .5])


class FakeEnv:
    def __init__(self, seed=branch.ROADS[0], *, drift=False, pixel_drift=False,
                 end_at=None, finish=False, timeout=False):
        self.clock = 0
        self.seed = seed
        self.drift = drift
        self.pixel_drift = pixel_drift
        self.end_at = end_at
        self.finish = finish
        self.timeout = timeout
        self.closed = False
        self.steps = 0

    def reset(self):
        self.clock = 0
        return np.zeros((4, 84, 84), np.float32), {"track_id": 1, "seed": self.seed}

    def step(self, native):
        self.clock += 1
        self.steps += 1
        obs = np.full((4, 84, 84), self.clock / 255, np.float32)
        if self.pixel_drift and self.clock > 0:
            obs.fill(.9)
        reward = .5 + float(native[1]) / 10 + float(native[0]) / 100
        done = self.end_at == self.clock
        return (obs, reward, done and not (self.finish or self.timeout),
                done and (self.finish or self.timeout), {"finished": done and self.finish})

    def close(self):
        self.closed = True


def _pixel(obs):
    return np.full(branch.PIXELS, int(round(float(obs[0, 0, 0]) * 255)), np.uint8)


def _fake_snapshot(env, obs, *, track_id, seed, info, action=None,
                   prefix_sha256=parity.INITIAL_HASH, reward=None, terminated=None,
                   truncated=None, raw_commands=None):
    assert track_id == 1 and seed == env.seed
    action_bytes = b"" if action is None else action.tobytes()
    state = (("hull.position", int(env.drift and env.clock > 0)),
             ("outer.elapsed_steps", env.clock), ("road_sha256", "f" * 64))
    return parity.Signature(hashlib.sha256(action_bytes).hexdigest(), prefix_sha256,
                            hashlib.sha256(b"".join(raw_commands or ())).hexdigest(), len(raw_commands or ()),
                            hashlib.sha256(obs.tobytes()).hexdigest(), "raw_pixel", reward, terminated,
                            truncated, tuple(sorted(info.items())), state)


def _fake_tap(env, native):
    obs, reward, term, trunc, info = env.step(native)
    commands = (native.astype(np.float64).tobytes(),) * 4
    return obs, reward, term, trunc, info, commands


def _mock_parity(monkeypatch):
    monkeypatch.setattr(parity, "_chain", lambda env: (None,) * 5)
    monkeypatch.setattr(parity, "snapshot", _fake_snapshot)
    monkeypatch.setattr(parity, "_step_with_raw_tap", _fake_tap)
    monkeypatch.setattr(branch, "model_observation", _pixel)


def _anchor(ep=0, step=16):
    action = np.array([0., -1., 0.], np.float32)
    obs = np.stack([_pixel(np.full((4, 84, 84), t / 255, np.float32)) for t in range(step + 6)])
    return branch.BoundAnchor(ep, 1, branch.ROADS[ep], step, obs[:step + 1],
                              np.tile(action, (step, 1)), (.5,) * step,
                              np.tile(action, (5, 1)), obs[step:step + 6], (.5,) * 5,
                              ((False, False, False),) * 5)


def test_fifth_terminal_is_full_but_early_terminal_or_timeout_is_excluded(monkeypatch):
    _mock_parity(monkeypatch)
    anchor = _anchor(step=16)
    actions = branch.candidate_actions(anchor)["coast"]
    terminal = FakeEnv(end_at=21)
    terminal.clock = 16
    row = branch.execute_suffix(terminal, anchor, "coast", actions, "f" * 64)
    assert row["full_h5"] and row["ranking_eligible"] and row["ending"] == "terminated"
    assert row["real_discounted_raw_return"] == pytest.approx(sum(.5 * .995**t for t in range(5)))
    assert row["real_h3_prefix_raw_return"] == pytest.approx(sum(.5 * .995**t for t in range(3)))
    for finish, timeout_flag, ending, semantic_terminal in (
        (True, False, "finished", True), (False, True, "timeout", False)
    ):
        env = FakeEnv(end_at=21, finish=finish, timeout=timeout_flag)
        env.clock = 16
        full = branch.execute_suffix(env, anchor, "coast", actions, "f" * 64)
        assert full["steps"] == env.steps == 5
        assert full["raw_rewards"] == [.5] * 5
        assert full["real_discounted_raw_return"] == pytest.approx(sum(.5 * .995**t for t in range(5)))
        assert full["full_h5"] and full["ranking_eligible"] and full["exclusion_reason"] is None
        assert full["ending"] == ending
        assert branch.episode_boundary(False, True, {"finished": finish}) == (True, semantic_terminal)
    timeout = FakeEnv(end_at=18, finish=True)
    timeout.clock = 16
    row = branch.execute_suffix(timeout, anchor, "coast", actions, "f" * 64)
    assert row["steps"] == 2 and not row["ranking_eligible"]
    assert row["real_h3_prefix_raw_return"] is None
    assert row["exclusion_reason"] == "finished_before_fifth"
    timeout = FakeEnv(end_at=19, timeout=True)
    timeout.clock = 16
    row = branch.execute_suffix(timeout, anchor, "coast", actions, "f" * 64)
    assert row["steps"] == 3 and not row["ranking_eligible"]
    assert row["ending"] == "timeout" and row["exclusion_reason"] == "timeout_before_fifth"
    with pytest.raises(ValueError, match="suffix action bytes"):
        branch.execute_suffix(FakeEnv(), anchor, "coast", actions[:3], "f" * 64)


def test_logged_h5_suffix_cannot_pad_original_h3_or_mask_timeout(monkeypatch):
    _mock_parity(monkeypatch)
    anchor = _anchor()
    original = FakeEnv(end_at=18, finish=True)
    original.clock = 16
    with pytest.raises(parity.ParityError, match="RAW raw reward/terminal/timeout flags"):
        branch.execute_suffix(original, anchor, "logged", anchor.logged_actions, "f" * 64)
    caught = FakeEnv(pixel_drift=True)
    with pytest.raises(parity.ParityError, match="model_observation"):
        branch.capture_bound_prefix(caught, anchor)
    assert caught.steps == 1


def test_within_anchor_ties_predicted_ties_and_road_denominators(monkeypatch):
    rows = [{"candidate": str(i), "full_h5": True, "ranking_eligible": True,
             "exclusion_reason": None, "real_discounted_raw_return": actual,
             "predicted_reward_return": predicted}
            for i, (actual, predicted) in enumerate(((4., 1.), (3., 1.), (2., 2.),
                                                      (4. + 1e-10, 2.), (10., 10.)))]
    rows[-1]["full_h5"] = rows[-1]["ranking_eligible"] = False
    rows[-1]["exclusion_reason"] = "timeout_before_fifth"
    result = branch.rank(rows)
    assert result["pairs"] == {"concordant": 1, "discordant": 2, "real_tie": 1,
                               "predicted_tie": 2, "terminal_or_short_excluded": 4}
    assert result["informative_real_pairs"] == 5
    assert result["concordance_on_informative_real_pairs"] == .2
    assert result["excluded_candidates_by_reason"]["timeout_before_fifth"] == 1
    entries = [{"geometry_seed": branch.ROADS[ep], "ranking": result} for ep, _ in branch.ANCHORS]
    summary = branch.summarize(entries)
    assert summary["by_road"][str(branch.ROADS[0])]["informative_real_pairs"] == 15
    assert summary["informative_real_pairs"] == 60 and summary["informative_roads"] == 4
    assert summary["excluded_candidates_by_reason"]["timeout_before_fifth"] == 12
    assert summary["full_h5_candidates"] == 48
    assert not summary["quality_gate_passed"]
    for row in rows:
        row["predicted_reward_return"] = row["real_discounted_raw_return"]
    match = branch.rank(rows)
    assert match["pairs"]["concordant"] == 5
    assert branch.summarize([{"geometry_seed": branch.ROADS[ep], "ranking": match}
                              for ep, _ in branch.ANCHORS])["quality_gate_passed"]


def test_same_five_actions_h3_h5_choice_and_planner_q_are_separate():
    rows = []
    for i, (h3, h5, actual) in enumerate(((10., 0., 0.), (0., 8., 8.),
                                          (1., 1., 1.), (2., 2., 2.), (3., 3., 3.))):
        rows.append({"candidate": str(i), "ranking_eligible": True,
                     "real_discounted_raw_return": actual,
                     "real_h3_prefix_raw_return": h3,
                     "h3_predicted_reward_return": h3,
                     "predicted_reward_return": h5,
                     "h3_predicted_planner_score": h3,
                     "h5_predicted_planner_score": -h5})
    comparison = branch.matched_action_quality(rows)
    assert comparison["rankings"]["h3_reward_vs_real_h3_prefix"]["pairs"]["discordant"] == 0
    assert comparison["selected"]["h3_reward"]["candidate"] == "0"
    assert comparison["selected"]["h3_reward"]["real_h5_regret"] == 8.
    assert comparison["selected"]["h5_reward"]["best_real_h5_tie"]
    assert comparison["selected"]["h5_planner"]["candidate"] == "0"
    rows[0]["ranking_eligible"] = False
    assert branch.matched_action_quality(rows)["eligible_candidates"] == 4


def test_gate_exactly_40_real_informative_pairs_three_roads_requires_more_than_60_percent():
    entries = []
    for ep, step in branch.ANCHORS:
        informative = (ep == 0 and step != 100) or (ep in (1, 2) and step == 16)
        counts = {"concordant": 6 if informative else 0, "discordant": 4 if informative else 0,
                  "predicted_tie": 0, "real_tie": 0 if informative else 10,
                  "terminal_or_short_excluded": 0}
        entries.append({"geometry_seed": branch.ROADS[ep], "ranking": {
            "pairs": counts, "informative_real_pairs": 10 if informative else 0,
            "full_h5_candidates": 5, "excluded_candidates_by_reason": {
                reason: 0 for reason in ("terminated_before_fifth", "finished_before_fifth",
                                         "timeout_before_fifth", "short_without_boundary")}}})
    exact = branch.summarize(entries)
    assert exact["informative_real_pairs"] == 40 and exact["informative_roads"] == 3
    assert exact["pairs"]["concordant"] == 24
    assert exact["concordance_on_informative_real_pairs"] == .6
    assert not exact["quality_gate_passed"]

    entries[3]["ranking"]["pairs"]["concordant"] += 1
    entries[3]["ranking"]["pairs"]["discordant"] -= 1
    above = branch.summarize(entries)
    assert above["pairs"]["concordant"] == 25
    assert above["informative_real_pairs"] == 40 and above["informative_roads"] == 3
    assert above["concordance_on_informative_real_pairs"] == .625 and above["quality_gate_passed"]

    entries[3]["ranking"]["pairs"]["concordant"] -= 1
    entries[3]["ranking"]["pairs"]["predicted_tie"] += 1
    tied_prediction = branch.summarize(entries)
    assert tied_prediction["informative_real_pairs"] == 40
    assert tied_prediction["pairs"]["predicted_tie"] == 1
    assert tied_prediction["pairs"]["concordant"] == 24
    assert not tied_prediction["quality_gate_passed"]

    entries[3]["ranking"]["pairs"]["concordant"] += 1
    entries[3]["ranking"]["pairs"]["predicted_tie"] -= 1
    entries[1]["ranking"]["pairs"] = {"concordant": 0, "discordant": 0,
                                        "predicted_tie": 0, "real_tie": 10,
                                        "terminal_or_short_excluded": 0}
    entries[1]["ranking"]["informative_real_pairs"] = 0
    insufficient = branch.summarize(entries)
    assert insufficient["informative_real_pairs"] == 30
    assert insufficient["informative_roads"] == 3
    assert not insufficient["quality_gate_passed"]

    entries[1]["ranking"]["pairs"] = {"concordant": 6, "discordant": 4,
                                        "predicted_tie": 0, "real_tie": 0,
                                        "terminal_or_short_excluded": 0}
    entries[1]["ranking"]["informative_real_pairs"] = 10
    entries[2]["ranking"]["pairs"] = dict(entries[6]["ranking"]["pairs"])
    entries[2]["ranking"]["informative_real_pairs"] = 10
    entries[6]["ranking"]["pairs"] = {"concordant": 0, "discordant": 0, "predicted_tie": 0,
                                        "real_tie": 10, "terminal_or_short_excluded": 0}
    entries[6]["ranking"]["informative_real_pairs"] = 0
    two_roads = branch.summarize(entries)
    assert two_roads["informative_real_pairs"] == 40
    assert two_roads["informative_roads"] == 2
    assert not two_roads["quality_gate_passed"]


def _run_bundle(monkeypatch, anchors, tmp_path):
    source_run = tmp_path / branch.RUN
    source_run.mkdir(parents=True, exist_ok=True)
    (source_run / "steps.jsonl").write_bytes(b"synthetic step ledger")
    (source_run / "training.jsonl").write_bytes(b"synthetic episode ledger")
    monkeypatch.setattr(branch, "STEPS_SHA", branch.digest(source_run / "steps.jsonl"))
    monkeypatch.setattr(branch, "TRAIN_SHA", branch.digest(source_run / "training.jsonl"))
    spec = {"source": {}, "logged_quality": {}, "source_sha256": {}, "runtime": {},
            "augmentation_seed": 834, "max_resets": 72}
    bundle = {"spec": spec, "quality": {}, "protocol_sha256": "a" * 64}
    monkeypatch.setattr(branch, "preflight", lambda *args: bundle)
    monkeypatch.setattr(branch, "load_bound_model", lambda *args: (anchors, TinyRewardModel()))
    monkeypatch.setattr(branch, "_recheck", lambda *args, **kwargs: None)
    _mock_parity(monkeypatch)
    return bundle


def test_default_preflight_no_load_or_reset_and_execute_exact_12_by_5_cap(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    anchors = tuple(_anchor(ep, step) for ep, step in branch.ANCHORS)
    _run_bundle(monkeypatch, anchors, tmp_path)
    instances = []

    def factory(*args, **kwargs):
        assert args[0] == 1 and args[2:] == (2000, 4)
        assert kwargs == {"reward_shaping": False, "obstacles": True}
        env = FakeEnv(args[1])
        instances.append(env)
        return env

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=factory))
    output = tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-synthetic.json"
    preview = branch.run(tmp_path, "a" * 64)
    assert preview["status"] == "preflight_only" and not instances and not output.exists()
    report = branch.run(tmp_path, "a" * 64, execute=True, output=output)
    assert report == json.loads(output.read_text())
    assert report["body_sha256"] == hashlib.sha256(json.dumps(
        {k: v for k, v in report.items() if k != "body_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert report["environment_resets_attempted"] == 72 == report["maximum_resets"]
    assert report["replay_bound_before_first_reset"] is True
    assert len(report["anchors"]) == 12
    assert all(len(e["candidates"]) == 5 and sum(e["ranking"]["pairs"].values()) == 10
               for e in report["anchors"])
    assert sum(report["summary"]["pairs"].values()) == 120
    assert report["summary"]["matched_action_quality"]["eligible_anchors"] == 12
    assert all(entry["matched_action_quality"]["eligible_candidates"] == 5 for entry in report["anchors"])
    assert all(row["steps"] == 5 for e in report["anchors"] for row in e["candidates"])
    assert all(env.closed for env in instances)
    with pytest.raises(ValueError, match="unused"):
        branch.run(tmp_path, "a" * 64, execute=True, output=output)  # Never exact-resume.


def test_partial_receipt_on_first_binder_failure_before_reset(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    _run_bundle(monkeypatch, (_anchor(),), tmp_path)
    monkeypatch.setattr(branch, "load_bound_model", lambda *a: (_ for _ in ()).throw(
        ValueError("full RAW lineage differs")))
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=lambda *a, **kw: pytest.fail("built env")))
    output = tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-binder.json"
    with pytest.raises(ValueError, match="full RAW lineage differs"):
        branch.run(tmp_path, "a" * 64, execute=True, output=output)
    report = json.loads(output.read_text())
    assert report["environment_resets_attempted"] == 0
    assert report["status"] == "stopped_binder_parity_or_execution_failure"
    assert report["anchors"] == []


@pytest.mark.parametrize("tamper", ["selected_action", "selected_road"])
def test_run_rejects_changed_selected_RAW_step_before_fake_env_construction(monkeypatch, tmp_path, tamper):
    (tmp_path / "runs").mkdir()
    state, raw, episodes, rows, _ = _replay_fixture(monkeypatch, tmp_path)
    _run_bundle(monkeypatch, (_anchor(),), tmp_path)
    if tamper == "selected_action":
        rows[16]["action_f32_hex"] = "0" * 24
    else:
        rows[16]["geometry_seed"] = branch.ROADS[1]
    steps = tmp_path / branch.RUN / "steps.jsonl"
    steps.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(branch, "STEPS_SHA", branch.digest(steps))
    monkeypatch.setattr(branch, "load_bound_model", lambda *args: (
        branch.bind_replay(state, raw, episodes, steps), TinyRewardModel()))
    constructed = []

    def forbidden_factory(*args, **kwargs):
        constructed.append((args, kwargs))
        pytest.fail("constructed fake env before complete RAW binding")

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=forbidden_factory))
    output = tmp_path / "runs" / f"tdmpc2-raw100k-h5-branches-{tamper}.json"
    with pytest.raises(ValueError, match="lineage differs"):
        branch.run(tmp_path, "a" * 64, execute=True, output=output)
    report = json.loads(output.read_text())
    assert not constructed and report["environment_resets_attempted"] == 0
    assert report["anchors"] == [] and not report["same_state_counterfactual_ranking"]
    assert report["status"] == "stopped_binder_parity_or_execution_failure"


def test_reset_budget_stops_before_construction_of_extra_branch(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    bundle = _run_bundle(monkeypatch, (_anchor(),), tmp_path)
    bundle["spec"]["max_resets"] = 3
    instances = []

    def factory(*args, **kwargs):
        env = FakeEnv(args[1])
        instances.append(env)
        return env

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=factory))
    output = tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-cap.json"
    with pytest.raises(ValueError, match="reset cap exceeded"):
        branch.run(tmp_path, "a" * 64, execute=True, output=output)
    result = json.loads(output.read_text())
    assert result["environment_resets_attempted"] == len(instances) == 3
    assert len(result["anchors"][0]["candidates"]) == 2
    assert result["status"] == "stopped_binder_parity_or_execution_failure"


def test_partial_receipt_on_first_branch_accessible_state_mismatch(monkeypatch, tmp_path):
    (tmp_path / "runs").mkdir()
    _run_bundle(monkeypatch, (_anchor(),), tmp_path)
    instances = []

    def factory(*args, **kwargs):
        env = FakeEnv(args[1], drift=len(instances) == 1)
        instances.append(env)
        return env

    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=factory))
    output = tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-parity.json"
    with pytest.raises(parity.ParityError, match="state.hull.position"):
        branch.run(tmp_path, "a" * 64, execute=True, output=output)
    report = json.loads(output.read_text())
    assert report["environment_resets_attempted"] == 2
    assert len(report["anchors"]) == 1 and not report["anchors"][0]["candidates"]
    assert report["anchors"][0]["candidate_action_bytes_hex"]["logged"] == _anchor().logged_actions.tobytes().hex()
    assert not report["same_state_counterfactual_ranking"]
    assert all(env.closed for env in instances)


def test_source_tamper_before_reset_and_untrusted_protocol_before_torch_load(monkeypatch, tmp_path):
    actual_recheck = branch._recheck
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / branch.PROTOCOL).write_text("{}")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted torch.load"))
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=lambda *a, **kw: pytest.fail("built env")))
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        branch.run(tmp_path, "0" * 64, execute=True,
                   output=tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-noreset.json")
    assert not list((tmp_path / "runs").iterdir())
    source = tmp_path / "physics.py"
    source.write_bytes(b"untampered")
    expected = branch.digest(source)
    source.write_bytes(b"changed")
    bundle = _run_bundle(monkeypatch, (_anchor(),), tmp_path)
    bundle["spec"]["source_sha256"] = {"physics.py": expected}
    bundle["protocol_sha256"] = branch.digest(tmp_path / branch.PROTOCOL)
    monkeypatch.setattr(branch, "_recheck", actual_recheck)  # Explicit real source check at first reset.
    monkeypatch.setattr(branch, "runtime_identity", lambda: {})
    output = tmp_path / "runs" / "tdmpc2-raw100k-h5-branches-tamper.json"
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        branch.run(tmp_path, bundle["protocol_sha256"], execute=True, output=output)
    report = json.loads(output.read_text())
    assert report["environment_resets_attempted"] == 0
    assert report["status"] == "stopped_binder_parity_or_execution_failure"


def test_missing_separate_quality_receipt_fails_before_pickle_or_env(monkeypatch, tmp_path):
    (tmp_path / "experiments").mkdir()
    protocol = tmp_path / branch.PROTOCOL
    protocol.write_text(json.dumps({"logged_quality": {}}))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded checkpoint"))
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(build_env=lambda *a, **kw: pytest.fail("built env")))
    with pytest.raises(ValueError, match="missing separately frozen H5 logged quality"):
        branch.preflight(tmp_path, branch.digest(protocol))


def test_source_drift_after_preflight_rechecked_before_torch_load(monkeypatch, tmp_path):
    (tmp_path / "experiments").mkdir()
    protocol = tmp_path / branch.PROTOCOL
    protocol.write_bytes(b"frozen protocol body")
    source = tmp_path / "physics.py"
    source.write_bytes(b"expected")
    sha = branch.digest(source)
    source.write_bytes(b"tampered")
    bundle = {"checkpoint": tmp_path / "untrusted.pt", "protocol_sha256": branch.digest(protocol),
              "spec": {"source": {}, "logged_quality": {}, "source_sha256": {"physics.py": sha},
                       "runtime": {}}}
    monkeypatch.setattr(branch, "runtime_identity", lambda: {})
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("untrusted torch.load"))
    with pytest.raises(ValueError, match="missing or changed pinned source"):
        branch.load_bound_model(bundle, tmp_path)


def test_checkpoint_hash_and_torch_load_use_same_open_descriptor_after_path_swap(monkeypatch, tmp_path):
    checkpoint = tmp_path / "trusted.pt"
    replacement = tmp_path / "replacement.pt"
    checkpoint.write_bytes(b"trusted fixture checkpoint bytes")
    replacement.write_bytes(b"path swap attacker bytes")
    monkeypatch.setattr(branch, "CHECKPOINT_SHA", branch.digest(checkpoint))
    monkeypatch.setattr(branch, "_recheck", lambda *args, **kwargs: None)
    monkeypatch.setattr(branch, "bind_replay", lambda *args: (_anchor(),))

    class FakeWorldModel:
        def load_state_dict(self, weights, *, strict):
            assert strict and set(weights) == {"fixture"}

        def eval(self):
            return self

    monkeypatch.setattr(branch, "WorldModel", lambda cfg: FakeWorldModel())
    loaded = []

    def load_from_descriptor(stream, *, map_location, weights_only):
        assert map_location == "cpu" and weights_only is False
        assert hasattr(stream, "read") and stream.tell() == 0
        replacement.replace(checkpoint)  # The pathname now references different bytes.
        loaded.append(stream.read())
        return {"learner": {"model.fixture": torch.tensor(1.)}}

    monkeypatch.setattr(torch, "load", load_from_descriptor)
    anchors, model = branch.load_bound_model(
        {"checkpoint": checkpoint, "raw": {}, "episodes": []}, tmp_path)
    assert len(anchors) == 1 and isinstance(model, FakeWorldModel)
    assert loaded == [b"trusted fixture checkpoint bytes"]
    assert checkpoint.read_bytes() == b"path swap attacker bytes"
