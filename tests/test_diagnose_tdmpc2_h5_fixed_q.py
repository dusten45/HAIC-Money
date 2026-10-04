"""Synthetic-only fixed-Q diagnostics; never construct or reset an HAIC environment."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from haic.algorithms.tdmpc2.model import two_hot_inv
from scripts import diagnose_tdmpc2_h5_branches as branch
from scripts import diagnose_tdmpc2_h5_fixed_q as fixed


class TinyFiveCritics:
    cfg = SimpleNamespace(num_bins=101, num_q=5, vmin=-10.0, vmax=10.0)

    def __init__(self):
        self.policy_draws = []

    def encode(self, pixels, task):
        return torch.zeros((1, 1))

    def reward(self, z, action, task):
        bins = 50 + (action[:, 1] * 2).long()
        logits = torch.full((z.shape[0], 101), -100.)
        logits.scatter_(1, bins[:, None], 100.)
        return logits

    def next(self, z, action, task):
        return z + action[:, :1]

    def termination(self, z, task):
        return torch.where(z >= 1, torch.full_like(z, .9), torch.full_like(z, .1))

    def pi(self, z, task):
        actions = torch.randn((z.shape[0], 3)).clamp(-1, 1)
        self.policy_draws.append(actions.clone())
        return actions, {}

    def Q(self, z, action, task, *, return_type):
        assert return_type == "all"
        bins = 50 + torch.arange(5)[:, None] + (action[:, 0] > 0).long()[None, :]
        logits = torch.full((5, z.shape[0], 101), -100.)
        logits.scatter_(2, bins[:, :, None].expand(5, z.shape[0], 1), 100.)
        return logits


def _actions():
    return {"logged": np.tile(np.array([1., 0., 0.], np.float32), (5, 1)),
            **{name: np.asarray(values, np.float32) for name, values in branch.FIXED_SUFFIXES.items()}}


def test_fixed_same_pair_horizons_policy_rng_and_terminal_reward(monkeypatch):
    model = TinyFiveCritics()
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, "randperm", lambda *a, **kw: pytest.fail("random critic heads selected"))
    scored = fixed._score(model, np.zeros(branch.PIXELS, np.uint8), _actions(), 834)
    assert [tuple(row["pair"]) for row in scored["fixed_pairs"]] == list(fixed.PAIRS)
    assert scored["bootstrap_seed"] == 100_834
    assert len(model.policy_draws) == 20
    for sample in model.policy_draws[1:]:
        torch.testing.assert_close(sample, model.policy_draws[0], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for pair in scored["fixed_pairs"]:
        assert pair["scores"]["h3"]["logged"] == pytest.approx(0, abs=1e-5)
        assert pair["scores"]["h5"]["logged"] == pytest.approx(0, abs=1e-5)
    z = torch.zeros((5, 1))
    policy_action = torch.zeros((5, 3))
    logits = model.Q(z, policy_action, None, return_type="all")
    for pair in fixed.PAIRS:
        got = fixed._FixedPair(model, pair).Q(z, policy_action, None, return_type="avg")
        torch.testing.assert_close(got, two_hot_inv(logits[list(pair)], model.cfg).mean(0))
    assert scored["fixed_pairs"][0]["scores"]["h3"]["gas"] != scored["fixed_pairs"][-1]["scores"]["h3"]["gas"]


@pytest.mark.parametrize("change", ["four_actions", "six_actions", "three_steps", "six_critics"])
def test_model_scoring_rejects_wrong_input_count_before_encoding(change):
    model, actions = TinyFiveCritics(), _actions()
    if change == "four_actions":
        actions.pop("brake")
    elif change == "six_actions":
        actions["extra"] = np.zeros((5, 3), np.float32)
    elif change == "three_steps":
        actions["logged"] = actions["logged"][:3]
    else:
        model.cfg = SimpleNamespace(num_bins=101, num_q=6)
    with pytest.raises(ValueError, match="exactly five H5 actions/five critics"):
        fixed._score(model, np.zeros(branch.PIXELS, np.uint8), actions, 42)
    assert not model.policy_draws


def _reseal(receipt):
    body = {k: v for k, v in receipt.items() if k != "body_sha256"}
    receipt["body_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                                    allow_nan=False).encode()).hexdigest()
    return receipt


def _fixture():
    """Twelve synthetic anchors, 120 real pairs, exactly 29 real ties."""
    spec = {"source": {"checkpoint": {"sha256": "a" * 64}},
            "logged_quality": {"result": {"sha256": "b" * 64}},
            "source_sha256": {"scripts/diagnose_tdmpc2_h5_branches.py": "c" * 64},
            "runtime": {"python": "synthetic"}, "augmentation_seed": 834,
            "candidates": {"logged": None, **branch.FIXED_SUFFIXES}}
    receipt = {"format": "haic-tdmpc2-raw100k-h5-branches-result-v1", "status": "complete",
               "protocol_sha256": fixed.PROTOCOL_SHA, "source": spec["source"],
               "logged_quality": spec["logged_quality"], "source_sha256": spec["source_sha256"],
               "runtime": spec["runtime"],
               "scope": "four_reused_consumed_TRAIN_roads_same_reconstructed_accessible_state_only",
               "environment_resets_attempted": 72, "maximum_resets": 72,
               "replay_bound_before_first_reset": True, "same_state_counterfactual_ranking": True,
               "historical_hidden_box2d_state_proven": False, "fresh_or_official_score": False,
               "horizon": 5, "discount": .995, "tie_tolerance": 1e-6, "anchors": []}
    pixel = np.zeros(branch.PIXELS, np.uint8)
    actions = _actions()
    for index, (episode, step) in enumerate(branch.ANCHORS):
        values = [0., 0., 0., 1., 2.] if index < 5 else [0., 0., 1., 1., 2.]
        real: dict[str, float] = dict(zip(fixed.NAMES, values))
        reward_h3 = {name: (-value if index < 4 else value) for name, value in real.items()}
        reward_h5 = {name: (-value if index < 5 else value) for name, value in real.items()}
        predicted = {"h3_reward": reward_h3, "h5_reward": reward_h5,
                     "h3_planner": real, "h5_planner": {name: -value for name, value in real.items()}}
        qualities = {label: fixed._quality(scores, real) for label, scores in predicted.items()}
        rows = []
        for name, value in real.items():
            action = actions[name]
            rows.append({"candidate": name, "model_action_bytes_hex": action.tobytes().hex(),
                         "native_action_bytes_hex": b"".join(branch.environment_action(a).tobytes()
                                                              for a in action).hex(),
                         "raw_rewards": [value, 0., 0., 0., 0.], "steps": 5,
                         "full_h5": True, "ranking_eligible": True, "exclusion_reason": None,
                         "real_discounted_raw_return": value, "real_h3_prefix_raw_return": value,
                         "h3_predicted_reward_return": reward_h3[name],
                         "predicted_reward_return": reward_h5[name],
                         "h3_predicted_planner_score": real[name],
                         "h5_predicted_planner_score": -real[name]})
        receipt["anchors"].append({"episode_id": episode, "start_step": step, "track_id": 1,
                                    "geometry_seed": branch.ROADS[episode],
                                    "anchor_model_observation_hex": pixel.tobytes().hex(),
                                    "anchor_model_observation_sha256": hashlib.sha256(pixel.tobytes()).hexdigest(),
                                    "candidate_action_bytes_hex": {name: action.tobytes().hex()
                                                                   for name, action in actions.items()},
                                    "candidates": rows,
                                    "matched_action_quality": {
                                        "rankings": {f"{label}_vs_real_h5": {"pairs": q["pairs"]}
                                                     for label, q in qualities.items()},
                                        "selected": {label: {"candidate": q["selected_candidate"],
                                                             "best_real_h5_tie": q["best_real_h5_tie"],
                                                             "real_h5_regret": q["real_h5_regret"]}
                                                     for label, q in qualities.items()}}})
    totals = {label: fixed._aggregate([
        fixed._quality({row["candidate"]: row[field] for row in anchor["candidates"]},
                       {row["candidate"]: row["real_discounted_raw_return"]
                        for row in anchor["candidates"]}) for anchor in receipt["anchors"]])
        for label, field in (("h3_reward", "h3_predicted_reward_return"),
                             ("h5_reward", "predicted_reward_return"),
                             ("h3_planner", "h3_predicted_planner_score"),
                             ("h5_planner", "h5_predicted_planner_score"))}
    receipt["summary"] = {"full_h5_candidates": 60, "informative_real_pairs": 91,
                          "quality_gate_passed": True,
                          "pairs": {**totals["h5_reward"]["pairs"], "terminal_or_short_excluded": 0},
                          "matched_action_quality": {
                              "rankings": {f"{label}_vs_real_h5": {"pairs": total["pairs"],
                                                                     "informative_real_pairs": 91}
                                           for label, total in totals.items()},
                              "selected": {label: {"best_real_h5_ties": total["selected_best_real_h5_ties"],
                                                   "mean_real_h5_regret": total["mean_real_h5_regret"]}
                                           for label, total in totals.items()}}}
    return _reseal(receipt), spec


@pytest.mark.parametrize("change", ["body", "anchors", "sixth_action", "action_bytes",
                                     "raw_reward", "real_ties", "summary"])
def test_resealed_tamper_and_input_counts_fail_closed(change):
    receipt, spec = _fixture()
    assert len(fixed._verify_receipt(receipt, spec)) == 12
    if change == "body":
        receipt["body_sha256"] = "0" * 64
    elif change == "anchors":
        receipt["anchors"].pop()
    elif change == "sixth_action":
        receipt["anchors"][0]["candidates"].append(copy.deepcopy(receipt["anchors"][0]["candidates"][0]))
    elif change == "action_bytes":
        receipt["anchors"][0]["candidates"][0]["model_action_bytes_hex"] = "00" * 60
    elif change == "raw_reward":
        receipt["anchors"][0]["candidates"][0]["raw_rewards"][0] = 200.
    elif change == "real_ties":
        row = receipt["anchors"][0]["candidates"][1]
        row["raw_rewards"][0] = row["real_discounted_raw_return"] = row["real_h3_prefix_raw_return"] = .5
    else:
        receipt["summary"]["matched_action_quality"]["selected"]["h3_reward"]["best_real_h5_ties"] = 12
    if change != "body":
        _reseal(receipt)
    with pytest.raises(ValueError, match="receipt|anchors|five|action|return|ranking|summary"):
        fixed._verify_receipt(receipt, spec)


def test_no_reset_stdout_only_and_frozen_sha_gate_before_load(monkeypatch):
    receipt, spec = _fixture()
    receipt_bytes = json.dumps(receipt).encode()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    original = receipt["summary"]["matched_action_quality"]
    summary = {"format": "haic-tdmpc2-raw100k-h5-branches-summary-v1",
               "status": "complete_weak_reward_gate_passed_planner_gate_held",
               "protocol": branch.PROTOCOL, "operator": "scripts/diagnose_tdmpc2_h5_branches.py",
               "primary_receipt": fixed.RECEIPT, "primary_receipt_sha256": receipt_sha,
               "source_model_trained_horizon": 3, "preflight_environment_resets": 0,
               "execution_environment_resets_attempted": 72, "anchors": 12,
               "full_h5_candidate_suffixes": 60, "informative_h5_pairs": 91, "real_h5_ties_at_1e-6": 29,
               "matched_same_five_candidates_vs_real_h5": {
                   (f"{label}_only" if label.endswith("reward") else
                    f"{label.replace('planner', 'terminal_q_planner_score')}"): {
                       "concordant": original["rankings"][f"{label}_vs_real_h5"]["pairs"]["concordant"],
                       "discordant": original["rankings"][f"{label}_vs_real_h5"]["pairs"]["discordant"],
                       "informative_pairs": 91,
                       "best_action_tied_at_1e-6": original["selected"][label]["best_real_h5_ties"],
                       "mean_real_h5_regret": original["selected"][label]["mean_real_h5_regret"]}
                   for label in ("h3_reward", "h5_reward", "h3_planner", "h5_planner")},
               "protocol_sha256": fixed.PROTOCOL_SHA,
               "operator_sha256": spec["source_sha256"]["scripts/diagnose_tdmpc2_h5_branches.py"],
               "source_model_sha256": spec["source"]["checkpoint"]["sha256"]}
    summary_bytes = json.dumps(summary).encode()
    summary_sha = hashlib.sha256(summary_bytes).hexdigest()
    monkeypatch.setattr(fixed, "RECEIPT_SHA", receipt_sha)
    monkeypatch.setattr(fixed, "RESULT_SHA", summary_sha)
    records = {fixed.RECEIPT: receipt_bytes, fixed.RESULT: summary_bytes}

    def pinned(root, name, expected):
        data = records[name]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("pinned source/receipt SHA drift")
        return SimpleNamespace(read_bytes=lambda: data)

    monkeypatch.setattr(branch, "pinned", pinned)
    bundle = {"spec": spec, "protocol_sha256": fixed.PROTOCOL_SHA}
    monkeypatch.setattr(branch, "preflight", lambda *a: bundle)
    monkeypatch.setattr(branch, "_recheck", lambda *a, **kw: None)
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(
        build_env=lambda *a, **kw: pytest.fail("constructed HAIC environment")))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded checkpoint before SHA gate"))
    monkeypatch.setattr(branch, "load_bound_model", lambda *a: pytest.fail("loaded after bad SHA"))
    records[fixed.RECEIPT] = receipt_bytes + b" "
    with pytest.raises(ValueError, match="SHA drift"):
        fixed.run(Path("."))
    records[fixed.RECEIPT] = receipt_bytes
    monkeypatch.setattr(branch, "preflight", lambda *a: (_ for _ in ()).throw(ValueError("source SHA drift")))
    with pytest.raises(ValueError, match="source SHA drift"):
        fixed.run(Path("."))
    monkeypatch.setattr(branch, "preflight", lambda *a: bundle)

    pixels = np.zeros(branch.PIXELS, np.uint8)
    anchors = [SimpleNamespace(episode_id=e, step=s, seed=branch.ROADS[e],
                               observations=np.stack([pixels]), logged_actions=_actions()["logged"])
               for e, s in branch.ANCHORS]
    monkeypatch.setattr(branch, "load_bound_model", lambda *a: (anchors, SimpleNamespace()))
    monkeypatch.setattr(branch, "candidate_actions", lambda anchor: _actions())

    def synthetic_score(model, pixels, actions, seed):
        index = (seed - 834) // 1000 * 3 + [16, 50, 100].index((seed - 834) % 1000)
        rows = receipt["anchors"][index]["candidates"]
        reward = {"h3": {r["candidate"]: r["h3_predicted_reward_return"] for r in rows},
                  "h5": {r["candidate"]: r["predicted_reward_return"] for r in rows}}
        return {"reward_only": reward, "bootstrap_seed": seed + 100_000,
                "fixed_pairs": [{"pair": list(pair), "scores": {"h3": reward["h3"].copy(),
                                                                  "h5": reward["h5"].copy()}}
                                for pair in fixed.PAIRS]}

    monkeypatch.setattr(fixed, "_score", synthetic_score)
    result = fixed.run(Path("."))
    assert result["environment_resets"] == result["optimizer_steps"] == 0
    assert result["original_stochastic_MPPI_reproduced"] is False
    assert result["primary_receipt_sha256"] == receipt_sha
    assert result["summary"]["reward_only"]["h3"]["pairs"]["concordant"] == 63
    assert result["summary"]["reward_only"]["h5"]["pairs"]["concordant"] == 56
    assert len(result["anchors"]) == 12 and len(result["summary"]["fixed_pairs"]) == 10
    assert result["summary"]["fixed_pairs"][0]["h3"]["informative_real_pairs"] == 91
