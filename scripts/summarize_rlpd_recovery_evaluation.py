"""Verify full consumed-TRAIN evaluation receipts and seal a compact result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_rlpd_recovery import SEEDS, sha, terminal_curve_association, transition_table, write_json


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("evaluation is not complete")
    for name, digest in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError(f"changed evaluation artifact: {name}")
    rows = result["episodes"]
    if len(rows) != manifest["episodes_complete"]:
        raise ValueError("episode count differs from manifest")
    actors = sorted({row["actor_id"] for row in rows})
    per_actor = {}
    for actor in actors:
        selected = [row for row in rows if row["actor_id"] == actor]
        if len(selected) != 12 or {row["geometry_seed"] for row in selected} != set(SEEDS):
            raise ValueError("incomplete or duplicate actor-road census")
        for row in selected:
            receipt = json.loads((root / f"{actor}-seed-{row['geometry_seed']}-receipt.json").read_text())
            if row != receipt or sha(root / row["trace_path"]) != row["trace_sha256"]:
                raise ValueError("episode receipt or trace binding differs")
            with np.load(root / row["trace_path"], allow_pickle=False) as trace:
                if len(trace["step"]) != row["steps"]:
                    raise ValueError("trace decision count differs")
                if terminal_curve_association(dict(trace), row) != row["terminal_curve_association"]:
                    raise ValueError("terminal-curve classification differs")
        per_actor[actor] = {
            "finished": sum(row["finished"] for row in selected), "denominator": 12,
            "censored": sum(row["censored"] for row in selected),
            "curve_entry_associated_terminal_failures": sum(row["terminal_curve_association"][
                "associated_terminal_failure"] for row in selected),
            "finished_geometries": [row["geometry_seed"] for row in selected if row["finished"]],
            "actor_sha256": selected[0]["actor_sha256"],
            "decisions": sum(row["steps"] for row in selected),
            "mean_final_progress": float(np.mean([row["progress"] for row in selected])),
            "mean_final_damage": float(np.mean([row["damage"] for row in selected])),
        }
        claimed = result["per_actor"][actor]
        if (per_actor[actor]["finished"] != claimed["finish_count"] or per_actor[actor]["censored"] != claimed["censored"]
                or per_actor[actor]["curve_entry_associated_terminal_failures"] != claimed["curve_entry_associated_terminal_failures"]):
            raise ValueError("published actor aggregate differs")
    pairs = {}
    for name, claimed in result["contemporaneous_pairs"].items():
        a, b = name.split("->")
        actual = transition_table([r for r in rows if r["actor_id"] == a], [r for r in rows if r["actor_id"] == b])
        if actual != claimed:
            raise ValueError("paired finish table differs")
        pairs[name] = actual
    return {"format": "haic-rlpd-recovery-evaluation-audit-v1", "status": "complete",
            "primary_directory": str(root), "manifest_sha256": sha(root / "manifest.json"),
            "result_sha256": sha(root / "result.json"), "verified_artifact_hashes": len(manifest["files_sha256"]),
            "episodes": len(rows), "per_actor": per_actor, "pairs": pairs,
            "source_code_sha256": sha(Path(__file__)), "environment_resets": 0,
            "limitations": ["Consumed TRAIN and one learner seed; no fresh generalization or official score.",
                             "Curve-associated terminal failure is a prospective temporal proxy, not causal attribution.",
                             "Retained earlier failed evaluation attempts are additional interaction cost, not fresh cells."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evaluation)
    write_json(args.output, result)
    print(json.dumps({"per_actor": result["per_actor"], "pairs": result["pairs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
