"""Synthetic-only all-five-Q controls; no HAIC construction/reset or writes."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from haic.algorithms.tdmpc2.model import two_hot_inv
from scripts import diagnose_tdmpc2_h3_all_q as all_q
from scripts import diagnose_tdmpc2_h5_fixed_q as fixed


def test_all_five_decoded_q_is_mean_of_ten_unordered_pairwise_means():
    cfg = SimpleNamespace(num_bins=101, vmin=-10., vmax=10.)
    generator = torch.Generator().manual_seed(93)
    logits = torch.randn((5, 5, 101), generator=generator, dtype=torch.float64)
    decoded = two_hot_inv(logits, cfg)
    pair_average = torch.stack([two_hot_inv(logits[list(pair)], cfg).mean(0)
                                for pair in fixed.PAIRS]).mean(0)
    direct = decoded.mean(0)
    assert all(sum(head in pair for pair in fixed.PAIRS) == 4 for head in range(5))
    torch.testing.assert_close(pair_average, direct, rtol=0, atol=1e-12)
    # Reward, discount and terminal mask are unchanged by Q-head aggregation.
    reward = torch.tensor([[.2], [1.], [-.3], [3.], [-1.]], dtype=torch.float64)
    alive = torch.tensor([[True], [False], [True], [True], [False]])
    torch.testing.assert_close(reward + .995**3 * alive * pair_average,
                               reward + .995**3 * alive * direct, rtol=0, atol=1e-12)


def _parent():
    """Synthetic 12-anchor/91-informative parent, not HAIC outcomes."""
    anchors = []
    for index, (episode, step) in enumerate(fixed.branch.ANCHORS):
        values = [0., 0., 0., 1., 2.] if index < 5 else [0., 0., 1., 1., 2.]
        real: dict[str, float] = dict(zip(fixed.NAMES, values))
        pairs = []
        for pair in fixed.PAIRS:
            scores = dict(real)
            pairs.append({"pair": list(pair), "scores": {
                "h3": {"scores": scores, "quality": fixed._quality(scores, real)}}})
        anchors.append({"episode_id": episode, "start_step": step,
                        "geometry_seed": fixed.branch.ROADS[episode],
                        "bootstrap_seed": 834 + episode * 1000 + step + 100_000,
                        "anchor_model_observation_sha256": "a" * 64,
                        "candidate_action_sha256": {name: "b" * 64 for name in fixed.NAMES},
                        "real_h5_return": real, "fixed_pairs": pairs})
    parent = {"format": "haic-tdmpc2-h5-fixed-q-read-only-v1",
              "scope": "fixed_pair_observational_reused_TRAIN_not_original_stochastic_MPPI",
              "environment_resets": 0, "optimizer_steps": 0, "fresh_cells": False,
              "policy_benefit_established": False, "original_stochastic_MPPI_reproduced": False,
              "operator_sha256": all_q.PARENT_OPERATOR_SHA,
              "branch_protocol_sha256": fixed.PROTOCOL_SHA,
              "branch_result_sha256": fixed.RESULT_SHA,
              "primary_receipt_sha256": fixed.RECEIPT_SHA,
              "primary_body_sha256": "c" * 64, "bootstrap_rng": "synthetic common policy draws",
              "source": {"checkpoint": {"path": "runs/synthetic.pt", "sha256": "d" * 64}},
              "source_sha256": {"haic/algorithms/tdmpc2/model.py": "e" * 64},
              "anchors": anchors, "summary": {"anchors": 12, "candidate_suffixes": 60,
                                             "actual_h5_pairs": 120, "real_h5_ties": 29,
                                             "informative_real_h5_pairs": 91,
                                             "original_stochastic_planner_concordant_reference": {"h3": 65},
                                             "reward_only": {"h3": {"pairs": {"concordant": 63}},
                                                             "h5": {"pairs": {"concordant": 56}}},
                                             "fixed_pairs": []}}
    for index, pair in enumerate(fixed.PAIRS):
        aggregate = fixed._aggregate([row["fixed_pairs"][index]["scores"]["h3"]["quality"]
                                     for row in anchors])
        parent["summary"]["fixed_pairs"].append({"pair": list(pair), "h3": aggregate})
    parent["body_sha256"] = all_q._body_sha(parent)
    return parent


def _summary_bytes(parent):
    frozen = {"format": "haic-tdmpc2-h5-fixed-q-read-only-summary-v1",
              "status": "complete_observational_consumed_train_only", "operator": all_q.PARENT_OPERATOR,
              "operator_sha256": all_q.PARENT_OPERATOR_SHA,
              "complete_cli_output_body_sha256": parent["body_sha256"],
              "source_h5_branch_protocol_sha256": fixed.PROTOCOL_SHA,
              "source_h5_branch_summary_sha256": fixed.RESULT_SHA,
              "source_h5_primary_receipt_sha256": fixed.RECEIPT_SHA,
              "environment_resets": 0, "optimizer_steps": 0,
              "anchors": 12, "informative_real_h5_pairs": 91}
    return json.dumps(frozen).encode()


def test_ten_pair_scores_per_candidate_on_same_twelve_anchors():
    parent = _parent()
    anchors, metrics = all_q._all_five(parent)
    assert len(anchors) == 12 and metrics["informative_real_pairs"] == 91
    assert metrics["pairs"] == {"concordant": 91, "discordant": 0, "predicted_tie": 0, "real_tie": 29}
    assert all(row["h3_all_five_q_scores"] == row["real_h5_return"] for row in anchors)
    assert all_q._decision(metrics)["passed"] is True
    assert metrics["selected_best_real_h5_ties"] == 12 and metrics["mean_real_h5_regret"] == 0


@pytest.mark.parametrize("concordant,ties,regret,passes", [
    (66, 9, .7953492685094497, True),
    (66, 9, all_q.BASELINE["mean_real_h5_regret"], False),
    (65, 9, .1, False),
    (66, 8, .1, False),
    (66, 9, .7953492685094499, False),
])
def test_all_three_predeclared_gates_including_strict_regret(concordant, ties, regret, passes):
    metrics = {"informative_real_pairs": 91, "eligible_anchors": 12,
               "pairs": {"concordant": concordant, "real_tie": 29},
               "selected_best_real_h5_ties": ties, "mean_real_h5_regret": regret}
    screen = all_q._decision(metrics)
    assert screen["passed"] is passes
    assert screen["checks"] == {
        "concordant_at_least_66_of_91": concordant >= 66,
        "best_choice_at_least_9_of_12": ties >= 9,
        "mean_regret_strictly_below_original": regret < all_q.BASELINE["mean_real_h5_regret"]}


@pytest.mark.parametrize("change", ["one_less_anchor", "missing_pair", "duplicate_pair",
                                     "missing_candidate", "reordered_candidate", "real_pair_tie",
                                     "changed_score_without_quality", "changed_pair_total"])
def test_resealed_parent_shape_or_score_tamper_fails_closed(change):
    parent = _parent()
    if change == "one_less_anchor":
        parent["anchors"].pop()
    elif change == "missing_pair":
        parent["anchors"][0]["fixed_pairs"].pop()
    elif change == "duplicate_pair":
        parent["anchors"][0]["fixed_pairs"][1]["pair"] = [0, 1]
    elif change == "missing_candidate":
        parent["anchors"][0]["fixed_pairs"][0]["scores"]["h3"]["scores"].pop("gas")
    elif change == "reordered_candidate":
        row = parent["anchors"][0]
        row["real_h5_return"] = dict(reversed(list(row["real_h5_return"].items())))
    elif change == "real_pair_tie":
        parent["anchors"][0]["real_h5_return"]["gas"] = 99.
    elif change == "changed_score_without_quality":
        parent["anchors"][0]["fixed_pairs"][0]["scores"]["h3"]["scores"]["gas"] = 99.
    else:
        parent["summary"]["fixed_pairs"][0]["h3"]["pairs"]["concordant"] += 1
    parent["body_sha256"] = all_q._body_sha(parent)
    with pytest.raises(ValueError, match="twelve|pairs|pair|candidate|score|quality|summary|five|same"):
        all_q._all_five(parent)


def test_source_tamper_before_inherited_checkpoint_load_and_no_reset(monkeypatch):
    parent = _parent()
    parent_summary = _summary_bytes(parent)
    original = {"matched_same_five_candidates_vs_real_h5": {
        "h3_terminal_q_planner_score": {"concordant": 65, "informative_pairs": 91,
                                        "best_action_tied_at_1e-6": 8,
                                        "mean_real_h5_regret": all_q.BASELINE["mean_real_h5_regret"]}}}
    records = {all_q.PARENT_OPERATOR: b"synthetic operator bytes",
               all_q.PARENT_SUMMARY: parent_summary,
               fixed.RESULT: json.dumps(original).encode(),
               fixed.RECEIPT: b"synthetic receipt", fixed.branch.PROTOCOL: b"synthetic protocol",
               "haic/algorithms/tdmpc2/model.py": b"synthetic model", "runs/synthetic.pt": b"synthetic checkpoint"}
    monkeypatch.setattr(all_q, "PARENT_OPERATOR_SHA", hashlib.sha256(records[all_q.PARENT_OPERATOR]).hexdigest())
    parent["operator_sha256"] = all_q.PARENT_OPERATOR_SHA
    parent["body_sha256"] = all_q._body_sha(parent)
    records[all_q.PARENT_SUMMARY] = _summary_bytes(parent)
    monkeypatch.setattr(all_q, "PARENT_SUMMARY_SHA", hashlib.sha256(records[all_q.PARENT_SUMMARY]).hexdigest())
    monkeypatch.setattr(fixed, "RESULT_SHA", hashlib.sha256(records[fixed.RESULT]).hexdigest())
    monkeypatch.setattr(fixed, "RECEIPT_SHA", hashlib.sha256(records[fixed.RECEIPT]).hexdigest())
    monkeypatch.setattr(fixed, "PROTOCOL_SHA", hashlib.sha256(records[fixed.branch.PROTOCOL]).hexdigest())
    parent["branch_result_sha256"] = fixed.RESULT_SHA
    parent["primary_receipt_sha256"] = fixed.RECEIPT_SHA
    parent["branch_protocol_sha256"] = fixed.PROTOCOL_SHA
    parent["source"]["checkpoint"]["sha256"] = hashlib.sha256(records["runs/synthetic.pt"]).hexdigest()
    parent["source_sha256"]["haic/algorithms/tdmpc2/model.py"] = hashlib.sha256(
        records["haic/algorithms/tdmpc2/model.py"]).hexdigest()
    parent["body_sha256"] = all_q._body_sha(parent)
    records[all_q.PARENT_SUMMARY] = _summary_bytes(parent)
    monkeypatch.setattr(all_q, "PARENT_SUMMARY_SHA", hashlib.sha256(records[all_q.PARENT_SUMMARY]).hexdigest())

    def pinned(root, name, expected):
        data = records[name]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("source/receipt SHA drift")
        return SimpleNamespace(read_bytes=lambda: data)

    monkeypatch.setattr(fixed.branch, "pinned", pinned)
    monkeypatch.setitem(sys.modules, "train", SimpleNamespace(
        build_env=lambda *a, **kw: pytest.fail("constructed HAIC environment")))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("loaded checkpoint before SHA gate"))
    monkeypatch.setattr(fixed, "run", lambda *a: pytest.fail("inherited scorer called with bad source"))
    records[all_q.PARENT_OPERATOR] = b"tampered bytes"
    with pytest.raises(ValueError, match="SHA drift"):
        all_q.run(Path("."))
    records[all_q.PARENT_OPERATOR] = b"synthetic operator bytes"
    monkeypatch.setattr(fixed, "run", lambda *a: (_ for _ in ()).throw(ValueError("inherited receipt SHA drift")))
    with pytest.raises(ValueError, match="inherited receipt SHA drift"):
        all_q.run(Path("."))
    monkeypatch.setattr(fixed, "run", lambda *a: copy.deepcopy(parent))
    report = all_q.run(Path("."))
    assert report["environment_resets"] == report["optimizer_steps"] == 0
    assert report["summary"]["decision_screen"]["passed"] is True
    assert report["body_sha256"] == all_q._body_sha(report)
    assert len(report["anchors"]) == 12 and report["parent"]["operator_sha256"] == all_q.PARENT_OPERATOR_SHA
