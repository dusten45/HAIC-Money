"""Summarize immutable benchmark receipts and legal action trace statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    rows = []
    for original in receipt["rows"]:
        trace_path = args.traces/f"track-{original['track_id']}-seed-{original['seed']}.json"
        trace = json.loads(trace_path.read_text())
        actions = np.asarray([r["action"] for r in trace], dtype=np.float32)
        speeds = np.asarray([r["speed"] for r in trace])
        targets = np.asarray([r["debug"]["last_target"] for r in trace])
        assert len(trace) == original["steps"]
        assert np.isfinite(actions).all() and np.isfinite(speeds).all() and np.isfinite(targets).all()
        assert np.all(actions >= [-1,0,0]) and np.all(actions <= [1,1,1])
        assert hashlib.sha256(actions.tobytes()).hexdigest() == original["action_trace_sha256"]
        rows.append({**original, "trace_file_sha256": sha(trace_path),
                     "diagnostics": {"mean_actual_speed": float(speeds.mean()),
                                     "mean_target_speed": float(targets.mean()),
                                     "mean_gas": float(actions[:,1].mean()),
                                     "braking_fraction": float((actions[:,2]>0).mean()),
                                     "all_actions_finite_and_bounded": True}})
    report = {"evidence_type": "mandatory development benchmark; no holdout",
              "input_receipt": str(args.receipt), "input_receipt_sha256": sha(args.receipt),
              "freeze": receipt["freeze"], "rows": rows,
              "finishes": sum(r["finished"] for r in rows),
              "original_13_second_goal_met": all(r["finished"] and r["lap_time_ms"]<=13000 for r in rows),
              "official_container_certified": False,
              "limits": "Road-relative references and margins are heuristic; no formal trajectory reachability or collision proof."}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
