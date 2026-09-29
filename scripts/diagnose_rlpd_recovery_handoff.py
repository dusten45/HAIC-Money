"""Zero-reset temporal diagnostics after full-vector recovery hands control back."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts.branch_oracle_policy_action import _sha, _write
from scripts.evaluate_rlpd_recovery import first


def events(arrays: dict, anchor: int, horizon: int) -> dict:
    n = len(arrays["applied_action"])
    handoff = anchor + horizon
    if not 0 <= anchor <= handoff <= n:
        raise ValueError("invalid handoff bounds")
    speed = arrays["speed"][:-1]
    lateral = np.abs(arrays["center_error"][:-1])
    action = arrays["applied_action"]
    teacher = arrays["oracle_action"][:-1]
    if any(not np.isfinite(v).all() for v in (speed, lateral, action, teacher)):
        raise ValueError("nonfinite handoff telemetry")
    predicates = {
        "curve-entry-overspeed": ((np.abs(arrays["curvature"][:-1]) >= .025)
            & (speed >= 14) & (action[:, 1] >= .35) & (action[:, 2] < .05)
            & (teacher[:, 2] >= .1), 1),
        "lateral-excursion": (lateral >= 6, 2),
        "added-damage": (arrays["damage"] > arrays["damage"][anchor] + 1e-9, 1),
        "steering-opposition": ((speed >= 2.5) & (np.abs(teacher[:, 0]) >= .25)
            & (action[:, 0] * teacher[:, 0] < 0)
            & (np.abs(action[:, 0] - teacher[:, 0]) >= .6), 1),
    }
    found = {}
    for name, (active, sustained) in predicates.items():
        index = first(active[handoff:], sustained)
        found[name] = None if index is None else {
            "decision": handoff + index, "seconds_after_handoff": index * .08,
            "beyond_five_second_followup": index >= 63,
            "speed_m_s": float(arrays["speed"][handoff + index]),
            "abs_lateral_m": float(abs(arrays["center_error"][handoff + index])),
            "progress": float(arrays["progress"][handoff + index]),
            "damage": float(arrays["damage"][handoff + index]),
        }
    times = [(v["decision"], key) for key, v in found.items() if v is not None]
    window_end = min(n, handoff + 63)
    maximum = (float(np.max(speed[handoff:window_end])) if window_end > handoff else None)
    return {"events_after_handoff": found,
            "first_type": min(times)[1] if times else "unclassified",
            "handoff_speed_m_s": float(arrays["speed"][handoff]),
            "first_five_seconds_max_speed_m_s": maximum,
            "reacceleration_delta_m_s": None if maximum is None else maximum - float(arrays["speed"][handoff]),
            "post_step_damage_event_time_is_approximate": True,
            "terminal_damage_observation_included": True,
            "causal": False}


def diagnose(dataset: Path, manifest_sha256: str) -> dict:
    if _sha(dataset / "manifest.json") != manifest_sha256:
        raise ValueError("source manifest changed")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest.get("status") != "complete" or len(manifest.get("episodes", [])) != 42:
        raise ValueError("full 42-branch census is required")
    rows = []
    for episode in manifest["episodes"]:
        path = (dataset / episode["path"]).resolve()
        if not path.is_relative_to(dataset.resolve()) or _sha(path) != episode["sha256"]:
            raise ValueError("source branch changed")
        with np.load(path, allow_pickle=False) as trace:
            row = {k: episode[k] for k in (
                "policy_id", "geometry_seed", "horizon", "stratum", "finished", "censored",
                "local_recovery_qualified", "steps")}
            row.update(events(dict(trace), episode["anchor_step"], episode["horizon"]))
            rows.append(row)
    groups = {}
    for horizon in (0, 12, 25):
        for stratum in ("failure", "finish-control"):
            for outcome in ("finished", "nonfinish", "censored"):
                selected = [r for r in rows if r["horizon"] == horizon and r["stratum"] == stratum
                    and ("censored" if r["censored"] else "finished" if r["finished"] else "nonfinish") == outcome]
                if not selected:
                    continue
                deltas = [r["reacceleration_delta_m_s"] for r in selected
                          if r["reacceleration_delta_m_s"] is not None]
                groups[f"h{horizon}/{stratum}/{outcome}"] = {
                    "episodes": len(selected),
                    "first_types": dict(Counter(r["first_type"] for r in selected)),
                    "event_counts": dict(Counter(k for r in selected for k, v in
                                                 r["events_after_handoff"].items() if v is not None)),
                    "first_events_beyond_followup": dict(Counter(k for r in selected for k, v in
                        r["events_after_handoff"].items() if v is not None and v["beyond_five_second_followup"])),
                    "median_reacceleration_delta_m_s": float(np.median(deltas)) if deltas else None,
                }
    return {"format": "haic-rlpd-recovery-handoff-diagnosis-v1", "environment_resets": 0,
            "manifest_sha256": manifest_sha256, "source_code_sha256": _sha(Path(__file__)),
            "episodes": rows, "groups": groups,
            "limitations": ["Post-handoff events are descriptive and may also occur on finishes; not a causal failure classifier.",
                             "Repeated menu/actor-road observations are correlated and all roads are consumed TRAIN.",
                             "Five-second maximum speeds use only observations before episode end and are not always a complete followup window."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = diagnose(args.dataset, args.manifest_sha256)
    _write(args.output, result, exclusive=True)
    print(json.dumps(result["groups"], sort_keys=True))


if __name__ == "__main__":
    main()
