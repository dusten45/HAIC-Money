"""Summarize preserved source-bound driving traces; this runs no simulation."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path("agents/apex_2026")
    results = root / "results/speed-20261005"
    receipt_path = results / "rear-clear-v1.json"
    receipt = json.loads(receipt_path.read_text())
    lineage = json.loads((results / "rear-clear-v1-lineage.json").read_text())
    source = root / "fast_rear_clear_agent.py"
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source) == lineage["source_sha256"]
    assert digest(receipt_path) == lineage["receipt_sha256"]
    rows = []
    for outcome in lineage["outcomes"]:
        name = f"track-{outcome['track_id']}-seed-{outcome['seed']}.json"
        path = Path(".haic-artifacts/apex-speed-20261005/rear-clear-v1-traces") / name
        trace = json.loads(path.read_text())
        actions = np.asarray([x["action"] for x in trace], np.float32)
        assert hashlib.sha256(actions.tobytes()).hexdigest() == outcome["action_trace_sha256"]
        speed = np.asarray([x["speed"] for x in trace])
        targets = np.asarray([x["debug"]["last_target"] for x in trace])
        bands = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 95), (95, 200))
        rows.append({
            "track_id": outcome["track_id"], "seed": outcome["seed"],
            "official_lap_time_ms": outcome["lap_time_ms"],
            "source_sha256": lineage["source_sha256"],
            "action_trace_sha256": outcome["action_trace_sha256"],
            "trace_file_sha256": digest(path), "action_count": len(trace),
            "post_action_speed_min_m_per_s": float(speed.min()),
            "post_action_speed_max_m_per_s": float(speed.max()),
            "sample_duration_seconds_by_speed_band": {
                f"{lo}-{hi}": round(.08 * int(np.sum((speed >= lo) & (speed < hi))), 2)
                for lo, hi in bands},
            "sample_duration_brake_positive_seconds": round(.08 * int(np.sum(actions[:, 2] > 0)), 2),
            "target_below60_action_count": int(np.sum(targets < 60)),
            "slow_actions_after40": [int(i) for i in np.flatnonzero((speed < 20) & (np.arange(len(speed)) > 40))],
        })
    output = results / "rear-clear-speed-budget.json"
    report = {
        "classification": "analysis_of_preserved_traces_not_new_validation",
        "helper_sha256": digest(Path(__file__)), "receipt_sha256": digest(receipt_path),
        "new_simulation_episodes": 0, "new_holdout_opened": False,
        "timing_caveat": "Each action contributes .08s to sampled bands; the final action can end early. Band durations are diagnostic and do not replace official lap times.",
        "rows": rows,
    }
    with output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
