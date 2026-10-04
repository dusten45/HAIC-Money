"""Read-only H3 all-five-Q score on the frozen five-action H5 branch outcomes.

This averages all ten fixed-critic-pair H3 scores at each archived anchor. It
never constructs an environment, trains, runs full MPPI, or writes a receipt.
Only a separately source-frozen study could test any subsequent policy effect.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scripts import diagnose_tdmpc2_h5_fixed_q as fixed


ROOT = Path(__file__).resolve().parents[1]
PARENT_OPERATOR = "scripts/diagnose_tdmpc2_h5_fixed_q.py"
PARENT_OPERATOR_SHA = "c8034289ecc644f21066eed1e5ac10c794d91ad826ae12950f14a683af663630"
PARENT_SUMMARY = "experiments/tdmpc2-h5-fixed-q-v1-result.json"
PARENT_SUMMARY_SHA = "6b6bf566973c8a3b57b732343f4f6a74dbf8ec449f66d19c7e70a4af9f2e6219"
BASELINE = {"concordant": 65, "informative_real_pairs": 91,
            "selected_best_real_h5_ties": 8,
            "mean_real_h5_regret": 0.7953492685094498}
THRESHOLDS = {"min_concordant": 66, "min_selected_best_real_h5_ties": 9,
              "max_mean_real_h5_regret_exclusive": BASELINE["mean_real_h5_regret"]}


def _body_sha(record: dict) -> str:
    body = {key: value for key, value in record.items() if key != "body_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _decision(metrics: dict) -> dict:
    if (metrics.get("informative_real_pairs") != 91 or metrics.get("eligible_anchors") != 12
            or metrics.get("pairs", {}).get("real_tie") != 29):
        raise ValueError("not the fixed 12-anchor/91-informative H5 decision cohort")
    concordant = metrics["pairs"]["concordant"]
    ties = metrics["selected_best_real_h5_ties"]
    regret = fixed._finite(metrics["mean_real_h5_regret"])
    checks = {"concordant_at_least_66_of_91": concordant >= THRESHOLDS["min_concordant"],
              "best_choice_at_least_9_of_12": ties >= THRESHOLDS["min_selected_best_real_h5_ties"],
              "mean_regret_strictly_below_original": regret < THRESHOLDS["max_mean_real_h5_regret_exclusive"]}
    return {"thresholds": THRESHOLDS.copy(), "checks": checks, "passed": all(checks.values())}


def _validate_parent(root: Path, parent: dict) -> None:
    if (parent.get("body_sha256") != _body_sha(parent)
            or parent.get("format") != "haic-tdmpc2-h5-fixed-q-read-only-v1"
            or parent.get("scope") != "fixed_pair_observational_reused_TRAIN_not_original_stochastic_MPPI"
            or parent.get("environment_resets") != 0 or parent.get("optimizer_steps") != 0
            or parent.get("fresh_cells") is not False
            or parent.get("policy_benefit_established") is not False
            or parent.get("original_stochastic_MPPI_reproduced") is not False
            or parent.get("operator_sha256") != PARENT_OPERATOR_SHA
            or parent.get("branch_protocol_sha256") != fixed.PROTOCOL_SHA
            or parent.get("branch_result_sha256") != fixed.RESULT_SHA
            or parent.get("primary_receipt_sha256") != fixed.RECEIPT_SHA):
        raise ValueError("fixed-pair parent/source/body/zero-reset contract differs")
    source_summary = fixed.branch._json(fixed.branch.pinned(
        root, PARENT_SUMMARY, PARENT_SUMMARY_SHA).read_bytes())
    if (source_summary.get("format") != "haic-tdmpc2-h5-fixed-q-read-only-summary-v1"
            or source_summary.get("status") != "complete_observational_consumed_train_only"
            or source_summary.get("operator") != PARENT_OPERATOR
            or source_summary.get("operator_sha256") != PARENT_OPERATOR_SHA
            or source_summary.get("complete_cli_output_body_sha256") != parent["body_sha256"]
            or source_summary.get("source_h5_branch_protocol_sha256") != fixed.PROTOCOL_SHA
            or source_summary.get("source_h5_branch_summary_sha256") != fixed.RESULT_SHA
            or source_summary.get("source_h5_primary_receipt_sha256") != fixed.RECEIPT_SHA
            or source_summary.get("environment_resets") != 0 or source_summary.get("optimizer_steps") != 0
            or source_summary.get("anchors") != 12 or source_summary.get("informative_real_h5_pairs") != 91):
        raise ValueError("frozen fixed-pair result summary does not bind parent computation")
    original = fixed.branch._json(fixed.branch.pinned(root, fixed.RESULT, fixed.RESULT_SHA).read_bytes())
    stochastic = original.get("matched_same_five_candidates_vs_real_h5", {}).get(
        "h3_terminal_q_planner_score", {})
    if (stochastic.get("concordant") != BASELINE["concordant"]
            or stochastic.get("informative_pairs") != BASELINE["informative_real_pairs"]
            or stochastic.get("best_action_tied_at_1e-6") != BASELINE["selected_best_real_h5_ties"]
            or stochastic.get("mean_real_h5_regret") != BASELINE["mean_real_h5_regret"]):
        raise ValueError("original stochastic H3 comparator differs from primary result")


def _all_five(parent: dict) -> tuple[list[dict], dict]:
    summary = parent.get("summary", {})
    if (summary.get("anchors") != 12 or summary.get("candidate_suffixes") != 60
            or summary.get("actual_h5_pairs") != 120 or summary.get("real_h5_ties") != 29
            or summary.get("informative_real_h5_pairs") != 91
            or summary.get("original_stochastic_planner_concordant_reference", {}).get("h3") != 65
            or summary.get("reward_only", {}).get("h3", {}).get("pairs", {}).get("concordant") != 63
            or summary.get("reward_only", {}).get("h5", {}).get("pairs", {}).get("concordant") != 56):
        raise ValueError("parent five-action H3/H5 source denominators changed")
    anchors = parent.get("anchors")
    pair_totals = summary.get("fixed_pairs")
    if (not isinstance(anchors, list) or len(anchors) != 12
            or not isinstance(pair_totals, list) or len(pair_totals) != 10):
        raise ValueError("all-five-Q requires twelve anchors and ten fixed Q pairs")
    rows, qualities = [], []
    for entry, (episode, start) in zip(anchors, fixed.branch.ANCHORS):
        if (not isinstance(entry, dict) or entry.get("episode_id") != episode
                or entry.get("start_step") != start or entry.get("geometry_seed") != fixed.branch.ROADS[episode]
                or entry.get("bootstrap_seed") != 834 + episode * 1000 + start + 100_000):
            raise ValueError("parent anchor/RNG identity differs")
        real = entry.get("real_h5_return")
        if not isinstance(real, dict) or tuple(real) != fixed.NAMES:
            raise ValueError("not the same five actual H5-return candidates")
        pairs = entry.get("fixed_pairs")
        if not isinstance(pairs, list) or len(pairs) != 10:
            raise ValueError("missing unordered fixed-Q pairs")
        for index, (record, pair) in enumerate(zip(pairs, fixed.PAIRS)):
            if record.get("pair") != list(pair) or pair_totals[index].get("pair") != list(pair):
                raise ValueError("fixed-Q head pair/order differs")
            predicted = record.get("scores", {}).get("h3", {})
            scores = predicted.get("scores")
            if not isinstance(scores, dict) or tuple(scores) != fixed.NAMES:
                raise ValueError("fixed-Q pair has missing/extra/reordered action scores")
            quality = fixed._quality(scores, real)
            if quality != predicted.get("quality"):
                raise ValueError("fixed-Q per-anchor score/choice quality differs")
        scores = {name: math.fsum(fixed._finite(pair["scores"]["h3"]["scores"][name])
                                       for pair in pairs) / 10 for name in fixed.NAMES}
        quality = fixed._quality(scores, real)
        qualities.append(quality)
        rows.append({"episode_id": episode, "start_step": start, "geometry_seed": entry["geometry_seed"],
                     "anchor_model_observation_sha256": entry["anchor_model_observation_sha256"],
                     "candidate_action_sha256": entry["candidate_action_sha256"],
                     "bootstrap_seed": entry["bootstrap_seed"], "real_h5_return": real,
                     "h3_all_five_q_scores": scores, "quality": quality})
    for index, pair in enumerate(fixed.PAIRS):
        computed = fixed._aggregate([entry["fixed_pairs"][index]["scores"]["h3"]["quality"]
                                     for entry in anchors])
        if computed != pair_totals[index]["h3"]:
            raise ValueError("fixed-Q 12-anchor pair summary differs from rows")
    return rows, fixed._aggregate(qualities)


def run(root: Path = ROOT) -> dict:
    root = Path(root).resolve(strict=True)
    fixed.branch.pinned(root, PARENT_OPERATOR, PARENT_OPERATOR_SHA)
    fixed.branch.pinned(root, PARENT_SUMMARY, PARENT_SUMMARY_SHA)
    # The parent verifies original protocol/result/body, operator, RAW source,
    # checkpoint, full ledgers, and archived replay pixels/actions before load.
    parent = fixed.run(root)
    _validate_parent(root, parent)
    anchors, metrics = _all_five(parent)
    # Fail closed if the inherited source or evidence changes before stdout.
    for name, sha in parent["source_sha256"].items():
        fixed.branch.pinned(root, name, sha)
    for ref in parent["source"].values():
        fixed.branch.pinned(root, ref["path"], ref["sha256"])
    for name, sha in ((PARENT_OPERATOR, PARENT_OPERATOR_SHA), (PARENT_SUMMARY, PARENT_SUMMARY_SHA),
                      (fixed.RESULT, fixed.RESULT_SHA), (fixed.RECEIPT, fixed.RECEIPT_SHA),
                      (fixed.branch.PROTOCOL, fixed.PROTOCOL_SHA)):
        fixed.branch.pinned(root, name, sha)
    report = {"format": "haic-tdmpc2-h3-all-q-read-only-v1",
              "scope": "five_fixed_suffixes_four_consumed_TRAIN_roads_not_full_MPPI_or_policy_benefit",
              "environment_resets": 0, "optimizer_steps": 0, "fresh_cells": False,
              "policy_benefit_established": False, "original_stochastic_MPPI_reproduced": False,
              "operator_sha256": fixed.branch.digest(Path(__file__)),
              "parent": {"operator": PARENT_OPERATOR, "operator_sha256": PARENT_OPERATOR_SHA,
                         "summary": PARENT_SUMMARY, "summary_sha256": PARENT_SUMMARY_SHA,
                         "complete_body_sha256": parent["body_sha256"],
                         "branch_protocol_sha256": fixed.PROTOCOL_SHA,
                         "branch_result_sha256": fixed.RESULT_SHA,
                         "primary_receipt_sha256": fixed.RECEIPT_SHA,
                         "primary_body_sha256": parent["primary_body_sha256"]},
              "source": parent["source"], "source_sha256": parent["source_sha256"],
              "bootstrap_rng": parent["bootstrap_rng"],
              "aggregation": "mean of 10 fixed pairwise H3 scores; each of 5 decoded Q heads appears in 4 pairs",
              "original_stochastic_h3": BASELINE.copy(),
              "anchors": anchors,
              "summary": {"anchors": 12, "candidate_suffixes": 60, "real_h5_pairs": 120,
                          "real_h5_ties": 29, "informative_real_h5_pairs": 91,
                          "h3_all_five_q": metrics, "decision_screen": _decision(metrics)}}
    report["body_sha256"] = _body_sha(report)
    return report


def main() -> None:
    print(json.dumps(run(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
