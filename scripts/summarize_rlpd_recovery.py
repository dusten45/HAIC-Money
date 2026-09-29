"""Audit coupled branch artifacts and report multi-second recovery trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.branch_oracle_policy_action import _sha, _write


def summarize(dataset: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("format") != "haic-rlpd-recovery-dataset-v1" or manifest.get("status") != "complete":
        raise ValueError("a completed coupled recovery dataset is required")
    rows = []
    unique = {"failure": set(), "finish-control": set()}
    groups = {}
    for episode in manifest["episodes"]:
        path = (dataset / episode["path"]).resolve()
        if not path.is_relative_to(dataset.resolve()) or _sha(path) != episode["sha256"]:
            raise ValueError("changed or unsafe branch trace")
        with np.load(path, allow_pickle=False) as trace:
            n, anchor, horizon = episode["steps"], episode["anchor_step"], episode["horizon"]
            mask = trace["recovery_mask"]
            if len(trace["reward"]) != n or int(mask.sum()) != episode["accepted_transitions"]:
                raise ValueError("branch transition accounting mismatch")
            if (trace["frames"].shape != (n + 1, 84, 84)
                    or np.any(mask & (trace["role"] != "oracle"))):
                raise ValueError("branch frame or action-role accounting mismatch")
            for step in np.flatnonzero(mask):
                stack = trace["frames"][np.maximum(np.arange(step - 3, step + 1), 0)]
                unique[episode["stratum"]].add((stack.tobytes(), trace["executed_action"][step].tobytes()))
            snapshots = {}
            for label, step in (("anchor", anchor), ("handoff", anchor + horizon),
                                ("handoff_plus_1s", anchor + horizon + 13),
                                ("handoff_plus_3s", anchor + horizon + 38),
                                ("handoff_plus_5s", anchor + horizon + 63),
                                ("anchor_plus_75_decisions", anchor + 75),
                                ("anchor_plus_88_decisions", anchor + 88)):
                snapshots[label] = (None if step > n else {
                    "step": step, "speed_m_s": float(trace["speed"][step]),
                    "abs_lateral_error_m": float(abs(trace["center_error"][step])),
                    "damage": float(trace["damage"][step]),
                    "progress": float(trace["progress"][step]),
                })
            row = {key: episode[key] for key in (
                "policy_id", "geometry_seed", "stratum", "horizon", "steps", "finished",
                "censored", "local_recovery_qualified", "accepted_transitions",
            )}
            row["snapshots"] = snapshots
            row["anchor_still_overspeed"] = episode.get("current_anchor_curve_entry_overspeed")
            row["trace_sha256"] = episode["sha256"]
            rows.append(row)
            groups.setdefault((episode["policy_id"], episode["geometry_seed"]), {})[horizon] = row
    finish_tables = {}
    for horizon in (12, 25):
        table: dict[str, Any] = {"kept": 0, "lost": 0, "gained": 0, "neither": 0, "unknown": 0,
                 "failure_anchors_qualified": 0, "finish_controls_qualified": 0}
        deltas = []
        for branches in groups.values():
            control, treatment = branches[0], branches[horizon]
            if control["censored"] or treatment["censored"]:
                table["unknown"] += 1
            else:
                key = {(True, True): "kept", (True, False): "lost",
                       (False, True): "gained", (False, False): "neither"}[
                           control["finished"], treatment["finished"]]
                table[key] += 1
            if treatment["local_recovery_qualified"]:
                table["failure_anchors_qualified" if treatment["stratum"] == "failure"
                      else "finish_controls_qualified"] += 1
            a = control["snapshots"][f"anchor_plus_{horizon + 63}_decisions"]
            b = treatment["snapshots"]["handoff_plus_5s"]
            if a is not None and b is not None:
                if a["step"] != b["step"]:
                    raise ValueError("counterfactual telemetry times differ")
                deltas.append({key: b[key] - a[key] for key in (
                    "speed_m_s", "abs_lateral_error_m", "damage", "progress")})
        table["anchors"] = len(groups)
        table["net_finish_rescues"] = table["gained"] - table["lost"]
        table["same_time_5s_followup_delta_medians"] = ({key: float(np.median([r[key] for r in deltas]))
            for key in deltas[0]} if deltas else {})
        table["delta_pair_count"] = len(deltas)
        finish_tables[str(horizon)] = table
    return {"format": "haic-rlpd-coupled-recovery-audit-v1",
            "manifest_sha256": _sha(dataset / "manifest.json"), "verified_episode_hashes": len(rows),
            "unique_accepted_transitions_by_stratum": {key: len(value) for key, value in unique.items()},
            "local_support_gate_not_training_gate": {
                          "minimum_unique_failure_transitions": 128, "minimum_failure_geometries": 3,
                          "unique_failure_transitions": len(unique["failure"]),
                          "failure_geometries": len({r["geometry_seed"] for r in rows
                                                     if r["stratum"] == "failure" and r["accepted_transitions"]}),
                          "passed": len(unique["failure"]) >= 128 and len({r["geometry_seed"] for r in rows
                              if r["stratum"] == "failure" and r["accepted_transitions"]}) >= 3},
            "finish_tables_by_horizon": finish_tables, "branches": rows,
            "training_authorized": False,
            "limitations": ["Repeated TRAIN upper bound, not a deployable privileged controller.",
                             "Local five-second recovery is not full-episode completion or the prepared paired-finish training gate.",
                             "Telemetry delta medians exclude pairs ending before the common assessment time.",
                             "Menu branches and repeated actor/road pairs are correlated, not independent geometries."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.dataset)
    _write(args.output, result, exclusive=True)
    print(json.dumps({key: result[key] for key in (
        "verified_episode_hashes", "local_support_gate_not_training_gate", "finish_tables_by_horizon")}, sort_keys=True))


if __name__ == "__main__":
    main()
