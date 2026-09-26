"""Execute a frozen target-entropy ablation through screen, confirmation and blind gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.collect_rlpd_prior import collect
from scripts.project_rlpd_entropy_screen_receipts import project as project_screen
from scripts.rlpd_common import ROOT, canonical_sha256, runtime_metadata, sha256_file, snapshot_sources, write_json
from scripts.rlpd_entropy_common import STUDY_NAME, read_entropy_protocol
from scripts.run_rlpd_pilot import _rank_for_run, _summarize_screen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _run_evaluator(*, protocol_path, run_root, partition, model_paths, output, cpu_python, workers, previous=None):
    command = [
        cpu_python,
        "-m",
        "evaluate_policy",
        "--run-dir",
        str(run_root),
        "--protocol-file",
        str(protocol_path),
        "--partition",
        partition,
        "--python",
        cpu_python,
        "--workers",
        str(workers),
        "--evaluations-dir",
        str(ROOT / "evaluations"),
        "--output",
        str(output),
    ]
    if previous is not None:
        command.extend(("--previous-evaluation", str(previous)))
    for path in model_paths:
        command.extend(("--model", str(path)))
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = json.loads(Path(output).read_text())
    evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
    candidates, summaries, finishes = _summarize_screen(evaluation_dir)
    return receipt, evaluation_dir, candidates, summaries, finishes, command


def _single_summary(evaluation_dir, actor_sha, partition):
    candidates, summaries, finishes = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError(f"{partition} result must have one candidate")
    candidate, summary = candidates[0], summaries[0]
    if candidate["archive_sha256"] != actor_sha:
        raise ValueError(f"{partition} evaluated an unexpected target variant/actor")
    matrix = None
    return {
        "pointer": None,
        "actor_sha256": actor_sha,
        "canonical_episodes": summary.get("canonical_episodes"),
        "canonical_finishes": finishes.get(candidate["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def _winner_by_confirmation(protocol, confirmation):
    author, positive = "rlpd-author-target", "rlpd-positive-target"
    seeds = protocol["student_training"]["learner_seeds"]
    for seed in seeds:
        for arm in (author, positive):
            row = confirmation[f"{arm}-seed{seed}"]
            if (
                row["canonical_episodes"] != 32
                or not row["eligible"]
                or not row["determinism_audited"]
                or not row["cpu_reload_matches"]
                or row["operational_failures"] != 0
                or row["canonical_finishes"] < 1
            ):
                return None
    author_scores = [confirmation[f"{author}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    positive_scores = [confirmation[f"{positive}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    author_dominates = all(a >= p for a, p in zip(author_scores, positive_scores)) and any(
        a > p for a, p in zip(author_scores, positive_scores)
    )
    positive_dominates = all(p >= a for a, p in zip(author_scores, positive_scores)) and any(
        p > a for a, p in zip(author_scores, positive_scores)
    )
    if author_dominates == positive_dominates:
        return None
    return author if author_dominates else positive


def _choose_target_finalist(protocol, target, confirmation, screen_selection):
    rows = []
    for seed in protocol["student_training"]["learner_seeds"]:
        key = f"{target}-seed{seed}"
        confirm = confirmation[key]
        screen = screen_selection["selected"][key]
        lap = confirm["summary"].get("avg_lap_time_ms")
        rows.append((
            confirm["canonical_finishes"],
            confirm["summary"].get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -screen["environment_steps"],
            -seed,
            {**screen, "confirmation_finishes": confirm["canonical_finishes"]},
        ))
    return max(rows, key=lambda row: row[:5])[-1]


def run(protocol_path: Path, run_root: Path, *, workers: int = 2):
    protocol_path = protocol_path.resolve()
    protocol = read_entropy_protocol(protocol_path)
    if protocol["name"] != STUDY_NAME:
        raise ValueError("runner only executes the frozen entropy-target ablation")
    if type(workers) is not int or workers <= 0:
        raise ValueError("worker count must be positive")
    run_root = Path(run_root).resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    protocol_sha = sha256_file(protocol_path)
    runtime = runtime_metadata(device="cuda")
    expected = protocol["runtime"]["training"]
    if (
        runtime["installed_distributions_sha256"] != expected["installed_distributions_sha256"]
        or runtime["torch"] != expected["torch"]
        or runtime["numpy"] != expected["numpy"]
        or runtime.get("gpu") != expected.get("gpu")
    ):
        raise RuntimeError("parent training runtime differs from frozen dependency/device inventory")
    snapshot_sources(protocol, run_root / "source")
    (run_root / "study_protocol.json").write_bytes(protocol_path.read_bytes())
    write_json(run_root / "config.json", {
        "format": "haic-rlpd-entropy-ablation-root-config-v1",
        "study_id": protocol["name"],
        "protocol_sha256": protocol_sha,
        "source_hashes_sha256": canonical_sha256(protocol["source_hashes"]),
        "frame_skip": protocol["frame_skip"],
        "max_steps": protocol["max_steps"],
        "config": {
            "algorithm": "rlpd",
            "frame_skip": protocol["frame_skip"],
            "max_steps": protocol["max_steps"],
            "action_smoothing": None,
            "action_control": None,
            "action_representation": None,
        },
        "arm_specs": protocol["student_training"]["arm_specs"],
        "environment": protocol["environment"],
        "runtime_inventory_sha256": runtime["installed_distributions_sha256"],
    }, exclusive=True)
    write_json(run_root / "runtime.json", runtime, exclusive=True)
    execution_path = run_root / "execution.json"
    write_json(execution_path, {
        "format": "haic-rlpd-entropy-ablation-execution-v1",
        "study_id": protocol["name"],
        "protocol_sha256": protocol_sha,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": [],
        "stages": [],
        "workers": workers,
        "confirmation_opened": False,
        "blind_opened": False,
        "status": "running",
    }, exclusive=True)
    try:
        dataset_dir = run_root / "prior-data"
        prior = collect(protocol_path, dataset_dir)
        execution = json.loads(execution_path.read_text())
        execution["teacher_dataset_sha256"] = prior["dataset_sha256"]
        execution["stages"].append({"name": "teacher_collection", "result": prior})
        write_json(execution_path, execution)

        train_started = time.perf_counter()
        candidate_by_hash = {}
        for arm in protocol["student_training"]["arms"]:
            for seed in protocol["student_training"]["learner_seeds"]:
                run_dir = run_root / f"{arm}-seed{seed}"
                command = [
                    sys.executable,
                    "-m",
                    "scripts.train_rlpd_entropy_ablation",
                    "--protocol",
                    str(protocol_path),
                    "--offline-dataset",
                    str(dataset_dir),
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
                subprocess.run(command, cwd=ROOT, check=True)
                result = json.loads((run_dir / "result.json").read_text())
                if (
                    result["environment_steps"] != protocol["student_training"]["steps_per_run"]
                    or result["gradient_steps"] != protocol["student_training"]["expected_gradient_steps_per_run"]
                    or result["target_entropy"] != protocol["student_training"]["arm_specs"][arm]["target_entropy"]
                ):
                    raise RuntimeError(f"run config/budget/target mismatch for {arm}-seed{seed}")
                rows = json.loads((run_dir / "frozen_candidates.json").read_text())["candidates"]
                if {row["environment_steps"] for row in rows} != set(protocol["student_training"]["candidate_steps"]):
                    raise RuntimeError(f"candidate checkpoints differ from predeclared steps for {arm}-seed{seed}")
                for row in rows:
                    candidate_by_hash[row["actor_sha256"]] = {
                        **row,
                        "arm": arm,
                        "seed": seed,
                        "path": str((run_dir / row["actor_path"]).resolve()),
                    }
        execution = json.loads(execution_path.read_text())
        execution["stages"].append({
            "name": "matched_entropy_training",
            "wall_seconds": time.perf_counter() - train_started,
            "candidate_actor_hashes": sorted(candidate_by_hash),
        })
        write_json(execution_path, execution)

        protocol = read_entropy_protocol(protocol_path)
        cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
        screen_pointer = run_root / "screen-evaluation.json"
        screen_command = [
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
            str(screen_pointer),
        ]
        for row in candidate_by_hash.values():
            screen_command.extend(("--model", row["path"]))
        execution = json.loads(execution_path.read_text())
        execution["commands"].append(screen_command)
        write_json(execution_path, execution)
        subprocess.run(screen_command, cwd=ROOT, check=True)
        screen_receipt = json.loads(screen_pointer.read_text())
        screen_dir = Path(screen_receipt["evaluation_dir"]).resolve()
        screen_candidates, screen_summaries, screen_finishes = _summarize_screen(screen_dir)
        summary_by_id = {row["candidate_id"]: row for row in screen_summaries}
        screen_selected = {}
        expected_cells = len(protocol["partitions"]["screen"]["track_ids"]) * len(protocol["partitions"]["screen"]["seeds"])
        for arm in protocol["student_training"]["arms"]:
            for seed in protocol["student_training"]["learner_seeds"]:
                screen_selected[f"{arm}-seed{seed}"] = _rank_for_run(
                    screen_candidates,
                    screen_summaries,
                    screen_finishes,
                    {"arm": arm, "seed": seed},
                    candidate_by_hash,
                    expected_canonical_episodes=expected_cells,
                )
        screen_gates = {
            key: bool(value is not None and value["canonical_finishes"] >= 1)
            for key, value in screen_selected.items()
        }
        screen_pass = all(screen_gates.values())
        execution = json.loads(execution_path.read_text())
        execution["stages"].append({
            "name": "screen",
            "evaluation_dir": str(screen_dir.relative_to(ROOT)),
            "pointer": str(screen_pointer.relative_to(ROOT)),
            "canonical_episodes": len(screen_selected) * expected_cells,
            "canonical_finishes": sum(screen_finishes.values()),
            "seed_arm_gates": screen_gates,
            "passed": screen_pass,
        })
        write_json(execution_path, execution)

        result = {
            "format": "haic-rlpd-entropy-target-ablation-result-v1",
            "study_id": protocol["name"],
            "protocol_path": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": protocol_sha,
            "teacher_dataset_sha256": prior["dataset_sha256"],
            "screen_pointer": str(screen_pointer.relative_to(ROOT)),
            "screen_pointer_sha256": sha256_file(screen_pointer),
            "screen_manifest_sha256": sha256_file(screen_dir / "manifest.json"),
            "screen_selected_by_arm_seed": screen_selected,
            "screen_seed_arm_gates": screen_gates,
            "screen_gate": "pass" if screen_pass else "stop_hold_failure",
            "confirmation_opened": False,
            "blind_opened": False,
            "official_performance_claim": False,
        }
        if not screen_pass:
            result["next_gate"] = "screen gate failed; confirmation/blind remain closed"
            _write_final_result(protocol, run_root, result, status=result["screen_gate"])
            return result

        projection_path = project_screen(protocol_path, run_root)
        projection = json.loads(projection_path.read_text())
        result["screen_selection_path"] = str(projection_path.relative_to(ROOT))
        result["screen_selection_sha256"] = sha256_file(projection_path)
        result["screen_selection_projections"] = projection["projections"]

        confirmation = {}
        for arm in protocol["student_training"]["arms"]:
            for seed in protocol["student_training"]["learner_seeds"]:
                key = f"{arm}-seed{seed}"
                selected = screen_selected[key]
                candidate = candidate_by_hash[selected["actor_sha256"]]
                previous = Path(projection["projections"][key]["previous_screen_pointer"])
                output = run_root / f"confirmation-{key}.json"
                command = [
                    cpu_python,
                    "-m",
                    "evaluate_policy",
                    "--run-dir",
                    str(run_root),
                    "--protocol-file",
                    str(protocol_path),
                    "--partition",
                    "confirmation",
                    "--previous-evaluation",
                    str(previous),
                    "--python",
                    cpu_python,
                    "--workers",
                    str(workers),
                    "--evaluations-dir",
                    str(ROOT / "evaluations"),
                    "--output",
                    str(output),
                    "--model",
                    candidate["path"],
                ]
                execution = json.loads(execution_path.read_text())
                execution["commands"].append(command)
                execution["confirmation_opened"] = True
                write_json(execution_path, execution)
                subprocess.run(command, cwd=ROOT, check=True)
                receipt = json.loads(output.read_text())
                evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
                rows, summaries, finishes = _summarize_screen(evaluation_dir)
                if len(rows) != 1 or len(summaries) != 1 or rows[0]["archive_sha256"] != selected["actor_sha256"]:
                    raise ValueError(f"confirmation candidate mismatch for {key}")
                summary = summaries[0]
                confirmation[key] = {
                    "arm": arm,
                    "training_seed": seed,
                    "target_entropy": protocol["student_training"]["arm_specs"][arm]["target_entropy"],
                    "actor_sha256": selected["actor_sha256"],
                    "pointer": str(output.relative_to(ROOT)),
                    "pointer_sha256": sha256_file(output),
                    "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
                    "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
                    "canonical_episodes": summary["canonical_episodes"],
                    "canonical_finishes": finishes.get(rows[0]["candidate_id"], 0),
                    "eligible": summary["eligible"],
                    "determinism_audited": summary["determinism_audited"],
                    "cpu_reload_matches": summary["cpu_reload_matches"],
                    "operational_failures": summary["operational_failures"],
                    "summary": summary["summary"],
                }
        target_winner = _paired_entropy_winner(confirmation, protocol["student_training"]["learner_seeds"])
        result["confirmation"] = confirmation
        result["confirmation_target_winner"] = target_winner
        result["confirmation_opened"] = True
        result["confirmation_gate"] = "pass" if target_winner is not None else "stop_hold_failure"
        if target_winner is None:
            result["blind_opened"] = False
            result["next_gate"] = "target settings did not show a strict per-seed confirmation dominance; blind remains closed"
            _write_final_result(protocol, run_root, result, status=result["confirmation_gate"])
            return result

        # Within the screen-selected winners for the confirmed target, confirmation
        # picks one learner/checkpoint; blind remains a single untouched dataset.
        finalist = max(
            (
                row for row in confirmation.values()
                if row["arm"] == target_winner
            ),
            key=lambda row: (
                row["canonical_finishes"],
                row["summary"]["avg_progress"],
                -(row["summary"]["avg_lap_time_ms"] or float("inf")),
                -screen_selected[f'{target_winner}-seed{row["training_seed"]}']["environment_steps"],
                -row["training_seed"],
            ),
        )
        result["blind_finalist_preselected"] = {
            "arm": finalist["arm"],
            "training_seed": finalist["training_seed"],
            "target_entropy": finalist["target_entropy"],
            "actor_sha256": finalist["actor_sha256"],
        }
        candidate = candidate_by_hash[finalist["actor_sha256"]]
        blind_output = run_root / "blind-finalist.json"
        blind_command = [
            cpu_python,
            "-m",
            "evaluate_policy",
            "--run-dir",
            str(run_root),
            "--protocol-file",
            str(protocol_path),
            "--partition",
            "blind",
            "--previous-evaluation",
            str(ROOT / finalist["pointer"]),
            "--python",
            cpu_python,
            "--workers",
            str(workers),
            "--evaluations-dir",
            str(ROOT / "evaluations"),
            "--output",
            str(blind_output),
            "--model",
            candidate["path"],
        ]
        execution = json.loads(execution_path.read_text())
        execution["commands"].append(blind_command)
        execution["blind_opened"] = True
        write_json(execution_path, execution)
        subprocess.run(blind_command, cwd=ROOT, check=True)
        receipt = json.loads(blind_output.read_text())
        evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
        rows, summaries, finishes = _summarize_screen(evaluation_dir)
        if len(rows) != 1 or len(summaries) != 1 or rows[0]["archive_sha256"] != finalist["actor_sha256"]:
            raise ValueError("blind cell results don't bind the frozen winner actor")
        result["blind"] = {
            "pointer": str(blind_output.relative_to(ROOT)),
            "pointer_sha256": sha256_file(blind_output),
            "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
            "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
            "actor_sha256": finalist["actor_sha256"],
            "canonical_episodes": summaries[0]["canonical_episodes"],
            "canonical_finishes": finishes.get(rows[0]["candidate_id"], 0),
            "eligible": summaries[0]["eligible"],
            "summary": summaries[0]["summary"],
        }
        result["blind_opened"] = True
        result["blind_gate"] = "recorded_single_internal_generalization_result"
        _write_final_result(protocol, run_root, result, status="completed_internal_blind")
        return result
    except BaseException as exc:
        execution = json.loads(execution_path.read_text())
        execution["status"] = "failed_entropy_ablation_stage"
        execution["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        execution["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(execution_path, execution)
        raise


def _paired_entropy_winner(confirmation, learner_seeds):
    specs = ("rlpd-author-target", "rlpd-positive-target")
    for arm in specs:
        for seed in learner_seeds:
            key = f"{arm}-seed{seed}"
            row = confirmation[key]
            if (
                row["canonical_episodes"] != 32
                or not row["eligible"]
                or not row["determinism_audited"]
                or not row["cpu_reload_matches"]
                or row["operational_failures"] != 0
                or row["canonical_finishes"] < 1
            ):
                return None
    author = [confirmation[f"rlpd-author-target-seed{seed}"]["canonical_finishes"] for seed in learner_seeds]
    positive = [confirmation[f"rlpd-positive-target-seed{seed}"]["canonical_finishes"] for seed in learner_seeds]
    author_dominates = all(a >= p for a, p in zip(author, positive)) and any(a > p for a, p in zip(author, positive))
    positive_dominates = all(p >= a for a, p in zip(author, positive)) and any(p > a for a, p in zip(author, positive))
    if author_dominates == positive_dominates:
        return None
    return "rlpd-author-target" if author_dominates else "rlpd-positive-target"


def _write_final_result(protocol, run_root, result, *, status):
    result["protocol_sha256"] = sha256_file(protocol["_path"])
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    output = ROOT / protocol["result_path"]
    write_json(output, result, exclusive=True)
    write_json(run_root / "result.json", result, exclusive=True)
    execution = json.loads((run_root / "execution.json").read_text())
    execution["status"] = status
    execution["result_path"] = str(output.relative_to(ROOT))
    execution["finished_at_utc"] = result["finished_at_utc"]
    write_json(run_root / "execution.json", execution)


def main():
    args = parse_args()
    result = run(args.protocol, args.run_root, workers=args.workers)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
