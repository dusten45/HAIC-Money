"""Emit per-state metrics at rule-selected Oracle comparison precursors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "runs/oracle-policy-diagnosis-v1/result.json"


def summarize(policy_id: str | None = None, seed: int | None = None) -> dict:
    result = json.loads(RESULT.read_text())
    rows = [row for row in result["comparisons"]
            if row["study"] == "rlpd-g0"
            and not row["replay_finished"]
            and row["first_failure_precursor"]["mode"] == "curve-entry-overspeed"]
    if policy_id is not None:
        rows = [row for row in rows if row["policy_id"] == policy_id]
    if seed is not None:
        rows = [row for row in rows if row["geometry_seed"] == seed]
    details = []
    for row in rows:
        stem = f"{row['policy_id']}-track{row['track_id']}-seed{row['geometry_seed']}"
        policy_path = ROOT / "runs/oracle-policy-diagnosis-v1/policies" / f"{stem}.npz"
        oracle_path = ROOT / "runs/oracle-policy-diagnosis-v1/oracle" / (
            f"oracle-track{row['track_id']}-seed{row['geometry_seed']}-cap2000.npz"
        )
        with np.load(policy_path, allow_pickle=False) as policy, np.load(oracle_path, allow_pickle=False) as oracle:
            step = int(row["first_failure_precursor"]["step"])
            policy_state = {
                "policy_action": policy["policy_or_oracle_action"][step].astype(float).tolist(),
                "oracle_action_on_same_policy_state": policy["oracle_action_at_state"][step].astype(float).tolist(),
                "speed_m_s": float(policy["pre_speed"][step]),
                "curvature": float(policy["curvature"][step]),
                "center_error_m": float(policy["center_error"][step]),
                "heading_error_rad": float(policy["heading_error"][step]),
                "progress_after_action": float(policy["progress"][step]),
                "damage_after_action": float(policy["damage"][step]),
                "position_m": policy["pre_position"][step].astype(float).tolist(),
            }
            oracle_state = {
                "oracle_action": oracle["policy_or_oracle_action"][step].astype(float).tolist(),
                "speed_m_s": float(oracle["pre_speed"][step]),
                "curvature": float(oracle["curvature"][step]),
                "center_error_m": float(oracle["center_error"][step]),
                "heading_error_rad": float(oracle["heading_error"][step]),
                "progress_after_action": float(oracle["progress"][step]),
                "damage_after_action": float(oracle["damage"][step]),
                "position_m": oracle["pre_position"][step].astype(float).tolist(),
            }
            step_window = []
            for index in range(max(0, step - 3), min(len(policy["step"]), step + 4)):
                step_window.append({
                    "step": index,
                    "policy_action": policy["policy_or_oracle_action"][index].astype(float).tolist(),
                    "oracle_action_on_policy_state": policy["oracle_action_at_state"][index].astype(float).tolist(),
                    "policy_speed_m_s": float(policy["pre_speed"][index]),
                    "policy_center_error_m": float(policy["center_error"][index]),
                    "policy_heading_error_rad": float(policy["heading_error"][index]),
                    "policy_progress": float(policy["progress"][index]),
                    "policy_damage": float(policy["damage"][index]),
                    "oracle_trajectory_speed_m_s": float(oracle["pre_speed"][index]),
                    "oracle_trajectory_center_error_m": float(oracle["center_error"][index]),
                    "oracle_trajectory_heading_error_rad": float(oracle["heading_error"][index]),
                    "oracle_trajectory_progress": float(oracle["progress"][index]),
                    "oracle_trajectory_damage": float(oracle["damage"][index]),
                })
        details.append({
            "policy_id": row["policy_id"],
            "geometry_seed": row["geometry_seed"],
            "archived_finish": row["original_finished"],
            "current_replay_finish": row["replay_finished"],
            "oracle_finish_on_cell": row["oracle_finished"],
            "first_precursor_step": step,
            "precursor_rule": row["first_failure_precursor"]["evidence"],
            "policy_state": policy_state,
            "oracle_action_on_same_policy_state": policy_state["oracle_action_on_same_policy_state"],
            "oracle_separate_trajectory_state_same_decision": oracle_state,
            "time_aligned_window_same_elapsed_decisions_not_same_pose": step_window,
            "source_trace_sha256": row["source_trace_sha256"],
        })
    return {"selected_examples": len(details), "examples": details}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-id", choices=("rlpd-seed11", "rlpd-seed50"))
    parser.add_argument("--geometry-seed", type=int)
    args = parser.parse_args()
    print(json.dumps(summarize(args.policy_id, args.geometry_seed), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
