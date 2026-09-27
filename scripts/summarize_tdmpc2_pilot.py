"""Read-only, denominator-aware summary of a frozen TD-MPC2 TRAIN pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _group(rows: list[dict], *, phase: str | None = None) -> dict:
    if phase is not None:
        rows = [row for row in rows if row.get("phase") == phase]
    if not rows:
        return {"episodes": 0, "decisions": 0, "finishes": 0,
                "mean_progress": None, "mean_raw_reward": None}
    return {"episodes": len(rows), "decisions": sum(row["length"] for row in rows),
            "finishes": sum(bool(row["finished"]) for row in rows),
            "mean_progress": statistics.mean(row["progress"] for row in rows),
            "mean_raw_reward": statistics.mean(row["reward"] for row in rows)}


def summarize(protocol_path: Path, run_dir: Path, expected_sha: str) -> dict:
    if digest(protocol_path) != expected_sha:
        raise ValueError("protocol SHA does not match frozen input")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["run_dir"] != run_dir.as_posix():
        raise ValueError("run directory differs from frozen protocol")
    ledger = run_dir / "training.jsonl"
    if not ledger.is_file():
        raise FileNotFoundError("training ledger missing")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[0].get("event") != "start" or rows[0]["protocol_sha256"] != expected_sha:
        raise ValueError("training ledger does not begin with frozen protocol")
    if rows[0]["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("training source map differs from protocol")
    steps = [row for row in rows if row.get("event") == "step"]
    if [row["decisions"] for row in steps] != list(range(1, len(steps) + 1)):
        raise ValueError("training decisions are not consecutive")
    if len(steps) > protocol["training"]["decision_cap"]:
        raise ValueError("decision cap exceeded")
    episodes = [row.copy() for row in rows if row.get("event") == "episode"]
    seed_last = protocol["training"]["seed_steps"] + 1
    previous = 0
    for row in episodes:
        row["phase"] = ("random" if row["decisions"] <= seed_last else
                        "planned" if previous >= seed_last else "mixed")
        if row["decisions"] - previous != row["length"]:
            raise ValueError("episode lengths do not sum to decisions")
        previous = row["decisions"]
    partials = [row for row in rows if row.get("event") == "partial"]
    checkpoints = [row for row in rows if row.get("event") == "checkpoint"]
    checkpoint = run_dir / "boundary.pt"
    latest_checkpoint_valid = bool(checkpoints and checkpoint.is_file()
                                   and digest(checkpoint) == checkpoints[-1]["sha256"])
    last_is_boundary = rows[-1].get("event") == "checkpoint" and latest_checkpoint_valid
    if checkpoints and not latest_checkpoint_valid:
        raise ValueError("latest boundary checkpoint hash mismatch")
    result_path, eval_path = run_dir / "result.json", run_dir / "evaluation-result.json"
    training_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    evaluation_result = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else None
    if training_result is not None and training_result["protocol_sha256"] != expected_sha:
        raise ValueError("training result protocol mismatch")
    if evaluation_result is not None and (evaluation_result["protocol_sha256"] != expected_sha
                                          or evaluation_result["source_sha256"] != protocol["source_sha256"]):
        raise ValueError("evaluation result protocol/source mismatch")
    by_mode = None
    if evaluation_result is not None:
        pairs = defaultdict(dict)
        eval_rows = [json.loads(line) for line in (run_dir / "train-evaluation.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in eval_rows:
            if row.get("event") == "episode":
                pairs[(row["repeat"], row["geometry_seed"])][row["mode"]] = row
        if len(pairs) != 4 * protocol["evaluation"]["repeats"] or any(set(p) != {"prior", "mppi"} for p in pairs.values()):
            raise ValueError("prior/MPPI evaluation pairs incomplete")
        by_mode = {"pair_count": len(pairs), "per_mode": evaluation_result["per_mode"],
                   "mppi_minus_prior_mean_progress": statistics.mean(
                       arm["mppi"]["progress"] - arm["prior"]["progress"] for arm in pairs.values()),
                   "mppi_minus_prior_mean_raw_reward": statistics.mean(
                       arm["mppi"]["reward"] - arm["prior"]["reward"] for arm in pairs.values()),
                   "finish_comparison_valid": False, "fresh_holdout": False}
    probe_rows = [row for row in rows if row.get("event") == "diagnostic"]
    recorded_probes = {row["stage"]: {key: value for key, value in row["metrics"].items()
                                      if key not in ("episode_ids", "start_steps")}
                       for row in probe_rows}
    return {"protocol_sha256": expected_sha, "reused_train_only": True,
            "first_recorded_cell": protocol["cells"][0], "recorded_step_count": len(steps),
            "planned_decision_count": max(0, len(steps) - seed_last),
            "complete_episode_count": len(episodes), "partial_records": len(partials),
            "checkpoint_count": len(checkpoints), "last_ledger_event": rows[-1]["event"],
            "latest_checkpoint_hash_valid": latest_checkpoint_valid,
            "exact_boundary_resume_allowed": last_is_boundary,
            "latest_checkpoint_sha256": checkpoints[-1]["sha256"] if checkpoints else None,
            "all_episodes": _group(episodes),
            "random_episodes": _group(episodes, phase="random"),
            "mixed_episodes": _group(episodes, phase="mixed"),
            "planned_episodes": _group(episodes, phase="planned"),
            "training_status": training_result["status"] if training_result else None,
            "updates": training_result["updates"] if training_result else
            checkpoints[-1]["updates"] if checkpoints else 0,
            "pretrain_updates": training_result["pretrain_updates"] if training_result else
            checkpoints[-1]["pretrain_updates"] if checkpoints else 0,
            "world_model_probe": recorded_probes,
            "paired_development": by_mode}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.protocol, args.run_dir, args.protocol_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
