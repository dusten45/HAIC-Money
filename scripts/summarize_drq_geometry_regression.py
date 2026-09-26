"""Aggregate fixed r6 source-success failure markers from offline evidence.

Descriptive markers cannot establish why an alternative closed-loop policy failed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from scripts.diagnose_drq_geometry_regression import ROOT, RUN, VARIANTS, digest


PAIRED_SHA = "29b6688413738f5a0ae7faaa950d202c53c73c1c14735f9e205d3ec01ed2cdd7"
REPLAY_SHA = "2fbfcef1cb7c19bc9ae6aa334df75f0e6c57d490485cfc8576ce99ded4f25135"
SAME_STATE_SHA = "2c1e1a3ce21137c00bc2042d069a9a410b77f92eb76bac3334c2c548888c07cc"
EARLY_ACTION = .10
REPRESENTATION_COSINE = .90
Q_ROAD_MAJORITY = 10
HIGH_PROGRESS = .90


def markers(pair: dict, early: list[dict]) -> dict:
    assert len(early) == 20 and {state["decision"] for state in early} == set(range(1, 21))
    assert all(state["source_seed"] == pair["source_seed"] and
               state["geometry_seed"] == pair["geometry_seed"] and
               state["variant"] == pair["variant"] for state in early)
    q1_reversal = sum(s["source_q1_ranking_variant_minus_source"] < 0 <
                      s["variant_q1_ranking_variant_minus_source"] for s in early)
    qmin_reversal = sum(s["source_q_min_ranking_variant_minus_source"] < 0 <
                        s["variant_q_min_ranking_variant_minus_source"] for s in early)
    representation = sum(s["source_actor_encoder_cosine"] < REPRESENTATION_COSINE for s in early)
    first_action = early[0]["action_linf_difference"]
    assert abs(first_action - pair["divergence"]["initial_native_linf_difference"]) < 1e-5
    is_loss = pair["transition"] == "lost"
    labels = []
    if is_loss:
        if first_action >= EARLY_ACTION:
            labels.append("early_policy_drift_marker")
        if q1_reversal >= Q_ROAD_MAJORITY:
            labels.append("critic_q1_reversal_marker")
        if representation >= Q_ROAD_MAJORITY:
            labels.append("actor_encoder_shift_marker")
        if pair["candidate_max_progress"] >= HIGH_PROGRESS:
            labels.append("finish_or_late_stage_marker")
        if not labels:
            labels.append("other_or_indeterminate")
    elif pair["transition"] == "both_failed":
        labels.append("source_also_failed_adaptation_not_established")
    return {
        "variant": pair["variant"], "source_seed": pair["source_seed"],
        "geometry_seed": pair["geometry_seed"], "family": pair["family"],
        "transition": pair["transition"], "labels": labels,
        "source_steps": pair["source_steps"], "candidate_steps": pair["candidate_steps"],
        "source_max_progress": pair["source_max_progress"],
        "candidate_max_progress": pair["candidate_max_progress"],
        "first_sustained_action_decision": pair["divergence"]["first_sustained_action_decision"],
        "initial_same_state_native_action_linf": first_action,
        "first20_source_critic_q1_prefers_source_r6_critic_q1_prefers_r6": q1_reversal,
        "first20_twin_min_reversal": qmin_reversal,
        "first20_actor_encoder_cosine_below_0_90": representation,
        "first20_mean_action_linf": statistics.mean(x["action_linf_difference"] for x in early),
        "first20_median_encoder_cosine": statistics.median(x["source_actor_encoder_cosine"] for x in early),
        "first20_median_source_q1_change_for_same_source_action": statistics.median(
            x["variant_critic"]["q1"]["source_action"] - x["source_critic"]["q1"]["source_action"]
            for x in early),
        "first20_median_source_q2_change_for_same_source_action": statistics.median(
            x["variant_critic"]["q2"]["source_action"] - x["source_critic"]["q2"]["source_action"]
            for x in early),
    }


def analyze(root: Path) -> dict:
    base = root / RUN
    files = {"paired": ("paired-regression-v1.json", PAIRED_SHA),
             "replay": ("replay-distribution-v1.json", REPLAY_SHA),
             "source_state": ("source-state-replay-v2/result.json", SAME_STATE_SHA)}
    for _, (relative, sha) in files.items():
        assert digest(base / relative) == sha
    paired = json.loads((base / files["paired"][0]).read_text())
    replay = json.loads((base / files["replay"][0]).read_text())
    same = json.loads((base / files["source_state"][0]).read_text())
    assert same["driven_decisions"] == 5998 and same["new_policy_rollouts"] == 0
    assert len(paired["pairs"]) == 96 and len(replay["runs"]) == 6 and len(same["states"]) == 1998
    early = defaultdict(list)
    all_states = defaultdict(list)
    for row in same["states"]:
        key = (row["variant"], row["source_seed"], row["geometry_seed"])
        all_states[key].append(row)
        if row["decision"] <= 20:
            early[key].append(row)
    assert len(early) == 33
    by_pair = []
    for pair in paired["pairs"]:
        key = (pair["variant"], pair["source_seed"], pair["geometry_seed"])
        if pair["source_finished"]:
            by_pair.append(markers(pair, sorted(early[key], key=lambda r: r["decision"])))
        else:
            assert not early[key]
            by_pair.append({"variant": pair["variant"], "source_seed": pair["source_seed"],
                            "geometry_seed": pair["geometry_seed"], "family": pair["family"],
                            "transition": pair["transition"], "labels":
                            ["source_also_failed_adaptation_not_established"] if pair["transition"] == "both_failed" else [],
                            "source_max_progress": pair["source_max_progress"],
                            "candidate_max_progress": pair["candidate_max_progress"]})
    outcomes = {}
    for variant in VARIANTS:
        rows = [r for r in by_pair if r["variant"] == variant]
        assert len(rows) == 32
        losses = [r for r in rows if r["transition"] == "lost"]
        label_count = Counter(label for r in losses for label in r["labels"])
        states = [s for key, items in all_states.items() if key[0] == variant for s in items]
        retained = [r for r in rows if r["transition"] == "both_finished"]
        outcomes[variant] = {
            "transitions": paired["variant_transitions"][variant], "lost_source_success_road_count": len(losses),
            "lost_marker_counts": dict(sorted(label_count.items())),
            "both_failed_source_also_failed_count": sum(r["transition"] == "both_failed" for r in rows),
            "retained_source_success_road_count": len(retained),
            "source_success_state_records": len(states),
            "first20_Q1_reversal_by_road": {f"seed{r['source_seed']}-road{r['geometry_seed']}":
                                            r["first20_source_critic_q1_prefers_source_r6_critic_q1_prefers_r6"]
                                            for r in rows if r["transition"] in ("lost", "both_finished")},
            "source_success_state_actor_encoder_cosine_median": statistics.median(
                s["source_actor_encoder_cosine"] for s in states),
            "source_success_state_actor_action_linf_median": statistics.median(
                s["action_linf_difference"] for s in states),
        }
    return {
        "scope": "source-success regression markers on reused TRAIN-DIAGNOSTIC; NOT causal attribution or fresh/generalization evidence",
        "input_sha256": {key: sha for key, (_, sha) in files.items()},
        "definitions": {"early_policy_drift_marker": "lost and same-initial-observation native action Linf >=.10",
                        "critic_q1_reversal_marker": "lost and >=10/20 source states with source Q1 preferring source AND r6 Q1 preferring r6",
                        "actor_encoder_shift_marker": "lost and >=10/20 source states with feature cosine <.90; descriptive only",
                        "finish_or_late_stage_marker": "lost with observed max visited-tile progress >=.90; may still be pre-finish",
                        "insufficient_source_state_retention": "not identifiable per TRAIN-DIAGNOSTIC cell: TRAIN road pool disjoint and replay has no images",
                        "source_also_failed_adaptation_not_established": "both_failed, not an observed regression of a source success",
                        "other_or_indeterminate": "lost with none of the four positive markers; absence of markers is not absence of mechanisms",
                        "caveat": "labels nonexclusive; threshold fixed before Q/encoder numeric inspection; serially correlated states and reused roads"},
        "variants": outcomes, "pairs": by_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to((ROOT / RUN).resolve())
    assert not out.exists() and out.parent.is_dir()
    result = analyze(ROOT)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(out), "sha256": digest(out),
                      "lost_markers": {key: value["lost_marker_counts"] for key, value in result["variants"].items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
