"""Execute the frozen teacher, matched training, and screen-only pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.collect_rlpd_prior import collect
from scripts.rlpd_common import (
    ROOT,
    canonical_sha256,
    read_protocol,
    runtime_metadata,
    sha256_file,
    snapshot_sources,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _summarize_screen(evaluation_dir: Path):
    candidates = json.loads((evaluation_dir / "candidates.json").read_text())
    summaries = json.loads((evaluation_dir / "summary.json").read_text())
    episode_counts = {}
    with (evaluation_dir / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode["repeat"] == 0 and episode["status"] == "ok" and episode.get("finished"):
                episode_counts[episode["candidate_id"]] = episode_counts.get(episode["candidate_id"], 0) + 1
    return candidates, summaries, episode_counts


def _rank_for_run(
    candidates,
    summaries,
    episode_counts,
    expected,
    candidate_identity,
    *,
    expected_canonical_episodes: int = 12,
):
    summaries_by_id = {item["candidate_id"]: item for item in summaries}
    eligible = []
    for candidate in candidates:
        identity = candidate_identity.get(candidate["archive_sha256"])
        if identity is None or any(identity[key] != value for key, value in expected.items()):
            continue
        summary = summaries_by_id[candidate["candidate_id"]]
        if (
            summary.get("eligible") is not True
            or summary.get("determinism_audited") is not True
            or summary.get("cpu_reload_matches") is not True
            or summary.get("operational_failures") != 0
            or summary.get("canonical_episodes") != expected_canonical_episodes
        ):
            continue
        metrics = summary["summary"]
        lap = metrics.get("avg_lap_time_ms")
        environment_steps = int(identity["environment_steps"])
        eligible.append((
            episode_counts.get(candidate["candidate_id"], 0),
            metrics.get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -environment_steps,
            candidate,
            summary,
        ))
    if not eligible:
        return None
    count, progress, negative_lap, _, candidate, summary = max(eligible, key=lambda row: row[:4])
    return {
        "arm": expected["arm"],
        "training_seed": expected["seed"],
        "environment_steps": candidate_identity[candidate["archive_sha256"]]["environment_steps"],
        "actor_sha256": candidate["archive_sha256"],
        "learner_checkpoint_sha256": candidate_identity[candidate["archive_sha256"]]["checkpoint_sha256"],
        "candidate_id": candidate["candidate_id"],
        "eligible": summary["eligible"],
        "determinism_audited": summary["determinism_audited"],
        "cpu_reload_matches": summary["cpu_reload_matches"],
        "operational_failures": summary["operational_failures"],
        "canonical_finishes": count,
        "avg_progress": progress,
        "avg_lap_time_ms": None if negative_lap == float("-inf") else -negative_lap,
        "summary": summary["summary"],
        "source_path": candidate["source_path"],
    }


def run(protocol_path: Path, run_root: Path, *, workers: int = 2):
    protocol_path = protocol_path.resolve()
    protocol = read_protocol(protocol_path)
    if protocol["name"] != "pixel-rlpd-offpolicy-pilot-v2":
        raise ValueError("this runner only executes the predeclared RLPD pilot")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha256_file(protocol_path)
    snapshot_sources(protocol, run_root / "source")
    (run_root / "study_protocol.json").write_bytes(protocol_path.read_bytes())
    runtime = runtime_metadata(device="cuda")
    write_json(run_root / "config.json", {
        "format": "haic-rlpd-pilot-root-config-v1",
        "study_id": protocol["name"],
        "protocol_sha256": protocol_sha,
        "source_hashes_sha256": canonical_sha256(protocol["source_hashes"]),
        "frame_skip": protocol["frame_skip"],
        "max_steps": protocol["max_steps"],
        "environment": protocol["environment"],
        "runtime_inventory_sha256": runtime["installed_distributions_sha256"],
    }, exclusive=True)
    write_json(run_root / "runtime.json", runtime, exclusive=True)
    write_json(run_root / "execution.json", {
        "format": "haic-rlpd-pilot-execution-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha,
        "commands": [],
        "worker_count": workers,
        "status": "running",
    }, exclusive=True)

    data_dir = run_root / "prior-data"
    prior = collect(protocol_path, data_dir)
    execution_path = run_root / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["teacher_dataset_sha256"] = prior["dataset_sha256"]
    execution["teacher_data_result"] = prior
    write_json(execution_path, execution)

    identity_by_hash = {}
    train_commands = []
    started = time.perf_counter()
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            run_dir = run_root / f"{arm}-seed{seed}"
            command = [
                sys.executable,
                "-m",
                "scripts.train_rlpd",
                "--protocol",
                str(protocol_path),
                "--offline-dataset",
                str(data_dir),
                "--run-dir",
                str(run_dir),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--device",
                "cuda",
            ]
            execution = json.loads(execution_path.read_text())
            execution["commands"].append(command)
            write_json(execution_path, execution)
            train_commands.append(command)
            print(json.dumps({"event": "training_start", "arm": arm, "seed": seed, "command": command}), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            candidates = json.loads((run_dir / "frozen_candidates.json").read_text())["candidates"]
            for candidate in candidates:
                identity_by_hash[candidate["actor_sha256"]] = {
                    "arm": arm,
                    "seed": seed,
                    "environment_steps": candidate["environment_steps"],
                    "checkpoint_sha256": candidate["checkpoint_sha256"],
                }
    execution = json.loads(execution_path.read_text())
    execution["training_wall_seconds"] = time.perf_counter() - started
    write_json(execution_path, execution)

    actor_paths = []
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            run_dir = run_root / f"{arm}-seed{seed}"
            records = json.loads((run_dir / "frozen_candidates.json").read_text())["candidates"]
            steps = {record["environment_steps"] for record in records}
            if steps != set(protocol["evaluation"]["screen_candidate_steps"]):
                raise RuntimeError(f"{arm}-seed{seed} did not produce both frozen screen checkpoints")
            for record in sorted(records, key=lambda item: item["environment_steps"]):
                actor_paths.append((run_dir / record["actor_path"]).resolve())

    cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
    read_protocol(protocol_path)
    evaluation_pointer = run_root / "screen-evaluation.json"
    command = [
        cpu_python,
        "-m",
        "evaluate_policy",
        "--run-dir",
        str(run_root),
        "--protocol-file",
        str(protocol_path),
        "--partition",
        "screen",
        "--python",
        cpu_python,
        "--workers",
        str(workers),
        "--evaluations-dir",
        str(ROOT / "evaluations"),
        "--output",
        str(evaluation_pointer),
    ]
    for path in actor_paths:
        command.extend(("--model", str(path)))
    print(json.dumps({"event": "screen_start", "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    evaluation_receipt = json.loads(evaluation_pointer.read_text())
    evaluation_dir = Path(evaluation_receipt["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    selected = {}
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            identity = {"arm": arm, "seed": seed}
            selected[f"{arm}-seed{seed}"] = _rank_for_run(
                candidates,
                summaries,
                finish_counts,
                identity,
                identity_by_hash,
            )
    seed_gates = {}
    for seed in protocol["student_training"]["learner_seeds"]:
        treatment = selected[f"rlpd-seed{seed}"]
        control = selected[f"sac-seed{seed}"]
        seed_gates[str(seed)] = {
            "treatment_selected": treatment is not None,
            "control_selected": control is not None,
            "treatment_finishes": None if treatment is None else treatment["canonical_finishes"],
            "control_finishes": None if control is None else control["canonical_finishes"],
            "pass": bool(
                treatment is not None
                and control is not None
                and treatment["canonical_finishes"] >= 1
                and treatment["canonical_finishes"] >= control["canonical_finishes"]
            ),
        }
    passed = all(item["pass"] for item in seed_gates.values())
    result = {
        "format": "haic-pixel-rlpd-pilot-result-v1",
        "study_id": protocol["name"],
        "protocol_path": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": protocol_sha,
        "teacher_dataset_sha256": prior["dataset_sha256"],
        "evaluation_pointer": str(evaluation_pointer.relative_to(ROOT)),
        "evaluation_pointer_sha256": sha256_file(evaluation_pointer),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "evaluation_manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "screen_candidates": selected,
        "seed_gates": seed_gates,
        "pilot_screen_gate": "pass" if passed else "stop_hold_failure",
        "confirmation_opened": False,
        "blind_opened": False,
        "official_performance_claim": False,
    }
    write_json(run_root / "pilot-screen-gate.json", result, exclusive=True)
    experiment_result = ROOT / "experiments/pixel-rlpd-offpolicy-pilot-v2-result.json"
    write_json(experiment_result, result, exclusive=True)
    execution = json.loads(execution_path.read_text())
    execution["status"] = result["pilot_screen_gate"]
    execution["result_path"] = str(experiment_result.relative_to(ROOT))
    execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(execution_path, execution)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    args = parse_args()
    run(args.protocol, args.run_root, workers=args.workers)


if __name__ == "__main__":
    main()
