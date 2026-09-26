"""Run a separately frozen long-horizon RLPD/SAC study through held-out gates."""

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
from scripts.run_rlpd_pilot import _rank_for_run, _summarize_screen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def _screen_selection(protocol, run_root, candidate_by_hash, workers, cpu_python):
    model_paths = [record["path"] for record in candidate_by_hash.values()]
    evaluation_pointer = run_root / "screen-evaluation.json"
    command = [
        cpu_python,
        "-m",
        "evaluate_policy",
        "--run-dir",
        str(run_root),
        "--protocol-file",
        str(protocol["_path"]),
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
    for path in model_paths:
        command.extend(("--model", str(path)))
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = json.loads(evaluation_pointer.read_text())
    evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    selected = {}
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            selected[f"{arm}-seed{seed}"] = _rank_for_run(
                candidates,
                summaries,
                finish_counts,
                {"arm": arm, "seed": seed},
                candidate_by_hash,
                expected_canonical_episodes=(
                    len(protocol["partitions"]["screen"]["track_ids"])
                    * len(protocol["partitions"]["screen"]["seeds"])
                ),
            )
    gates = {}
    for seed in protocol["student_training"]["learner_seeds"]:
        treatment, control = selected[f"rlpd-seed{seed}"], selected[f"sac-seed{seed}"]
        gates[str(seed)] = {
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
    return {
        "pointer": evaluation_pointer,
        "receipt": receipt,
        "evaluation_dir": evaluation_dir,
        "selected": selected,
        "seed_gates": gates,
        "passed": all(gate["pass"] for gate in gates.values()),
        "summaries": summaries,
        "finish_counts": finish_counts,
    }


def _evaluate_selected_actor(
    *,
    protocol_path,
    protocol,
    run_root,
    actor_path,
    previous_pointer,
    output_pointer,
    partition,
    diagnostic,
    workers,
    cpu_python,
):
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
        "--previous-evaluation",
        str(previous_pointer),
        "--python",
        cpu_python,
        "--workers",
        str(workers),
        "--evaluations-dir",
        str(ROOT / "evaluations"),
        "--output",
        str(output_pointer),
        "--model",
        str(actor_path),
    ]
    if diagnostic:
        command.append("--diagnostic-confirmation")
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = json.loads(output_pointer.read_text())
    evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError(f"{partition} evaluation must contain exactly one bound candidate")
    summary = summaries[0]
    return {
        "pointer": str(output_pointer),
        "pointer_sha256": sha256_file(output_pointer),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "candidate_id": candidates[0]["candidate_id"],
        "actor_sha256": candidates[0]["archive_sha256"],
        "canonical_finishes": finish_counts.get(candidates[0]["candidate_id"], 0),
        "canonical_episodes": summary.get("canonical_episodes"),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "diagnostic_only": diagnostic,
        "summary": summary.get("summary"),
    }


def _finalist_screen_rank(selected, training_seeds):
    finalists = []
    for seed in training_seeds:
        candidate = selected[f"rlpd-seed{seed}"]
        if candidate is None:
            raise ValueError("cannot rank a blind finalist without an eligible screen actor")
        lap_time = candidate["avg_lap_time_ms"]
        finalists.append((
            candidate["canonical_finishes"],
            candidate["avg_progress"],
            -lap_time if lap_time is not None else float("-inf"),
            -candidate["environment_steps"],
            -seed,
            candidate,
        ))
    return max(finalists, key=lambda item: item[:5])[-1]


def _update_execution(run_root, *, status, result_path=None, failure=None):
    path = run_root / "execution.json"
    execution = json.loads(path.read_text())
    execution["status"] = status
    execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if result_path is not None:
        execution["result_path"] = str(Path(result_path).relative_to(ROOT))
    if failure is not None:
        execution["failure"] = {"type": type(failure).__name__, "message": str(failure)}
    write_json(path, execution)


def run(protocol_path: Path, run_root: Path, *, workers: int = 2):
    protocol_path = protocol_path.resolve()
    protocol = read_protocol(protocol_path)
    if protocol["name"] != "pixel-rlpd-long-horizon-followup-v1":
        raise ValueError("this runner only executes the separately frozen follow-up study")
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
        "format": "haic-rlpd-followup-root-config-v1",
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
        "environment": protocol["environment"],
        "runtime_inventory_sha256": runtime["installed_distributions_sha256"],
    }, exclusive=True)
    write_json(run_root / "runtime.json", runtime, exclusive=True)
    write_json(run_root / "execution.json", {
        "format": "haic-rlpd-followup-execution-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha,
        "commands": [],
        "stages": [],
        "worker_count": workers,
        "confirmation_opened": False,
        "blind_opened": False,
        "status": "running",
    }, exclusive=True)

    try:
        dataset_dir = run_root / "prior-data"
        prior = collect(protocol_path, dataset_dir)
        execution = json.loads((run_root / "execution.json").read_text())
        execution["teacher_dataset_sha256"] = prior["dataset_sha256"]
        execution["stages"].append({"name": "teacher_collection", "result": prior})
        write_json(run_root / "execution.json", execution)

        start = time.perf_counter()
        candidate_by_hash = {}
        for arm in ("rlpd", "sac"):
            for seed in protocol["student_training"]["learner_seeds"]:
                run_dir = run_root / f"{arm}-seed{seed}"
                command = [
                    sys.executable, "-m", "scripts.train_rlpd",
                    "--protocol", str(protocol_path),
                    "--offline-dataset", str(dataset_dir),
                    "--run-dir", str(run_dir),
                    "--arm", arm,
                    "--seed", str(seed),
                    "--device", "cuda",
                ]
                execution = json.loads((run_root / "execution.json").read_text())
                execution["commands"].append(command)
                write_json(run_root / "execution.json", execution)
                subprocess.run(command, cwd=ROOT, check=True)
                result = json.loads((run_dir / "result.json").read_text())
                if (
                    result["environment_steps"] != protocol["student_training"]["steps_per_run"]
                    or result["gradient_steps"] != protocol["student_training"]["expected_gradient_steps_per_run"]
                ):
                    raise RuntimeError(f"{arm}-seed{seed} failed the exact training-budget gate")
                candidate_records = json.loads((run_dir / "frozen_candidates.json").read_text())["candidates"]
                if {row["environment_steps"] for row in candidate_records} != set(protocol["student_training"]["candidate_steps"]):
                    raise RuntimeError(f"{arm}-seed{seed} checkpoint set differs from the frozen screen plan")
                for record in candidate_records:
                    candidate_by_hash[record["actor_sha256"]] = {
                        **record,
                        "arm": arm,
                        "seed": seed,
                        "path": str((run_dir / record["actor_path"]).resolve()),
                    }
        execution = json.loads((run_root / "execution.json").read_text())
        execution["stages"].append({
            "name": "matched_student_training",
            "wall_seconds": time.perf_counter() - start,
            "candidate_actor_hashes": sorted(candidate_by_hash),
        })
        write_json(run_root / "execution.json", execution)

        # Re-hash code immediately before the first screen cell.
        protocol = read_protocol(protocol_path)
        cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
        screen = _screen_selection(protocol, run_root, candidate_by_hash, workers, cpu_python)
        execution = json.loads((run_root / "execution.json").read_text())
        execution["stages"].append({
            "name": "screen",
            "evaluation_dir": str(screen["evaluation_dir"].relative_to(ROOT)),
            "evaluation_pointer": str(screen["pointer"].relative_to(ROOT)),
            "result_count": sum(item.get("expected_results", 0) for item in screen["summaries"]),
            "canonical_finishes": sum(screen["finish_counts"].values()),
            "seed_gates": screen["seed_gates"],
        })
        execution["confirmation_opened"] = False
        execution["blind_opened"] = False
        write_json(run_root / "execution.json", execution)
        result = {
            "format": "haic-rlpd-long-horizon-followup-result-v1",
            "study_id": protocol["name"],
            "protocol_path": str(protocol_path.relative_to(ROOT)),
            "protocol_sha256": protocol_sha,
            "teacher_dataset_sha256": prior["dataset_sha256"],
            "screen_pointer": str(screen["pointer"].relative_to(ROOT)),
            "screen_pointer_sha256": sha256_file(screen["pointer"]),
            "screen_manifest_sha256": sha256_file(screen["evaluation_dir"] / "manifest.json"),
            "selected_screen_actors": screen["selected"],
            "screen_seed_gates": screen["seed_gates"],
            "screen_gate": "pass" if screen["passed"] else "stop_hold_failure",
            "confirmation_opened": False,
            "blind_opened": False,
            "official_performance_claim": False,
        }
        if not screen["passed"]:
            result["next_gate"] = "hard stop; do not open confirmation/blind or extend this study"
            output = ROOT / protocol["result_path"]
            write_json(output, result, exclusive=True)
            write_json(run_root / "result.json", result, exclusive=True)
            _update_execution(run_root, status=result["screen_gate"], result_path=output)
            return result

        # Select the blind subject from screen-only information before confirmation.
        protocol = read_protocol(protocol_path)
        finalist = _finalist_screen_rank(
            screen["selected"], protocol["student_training"]["learner_seeds"]
        )
        result["blind_finalist_preselected_from_screen"] = {
            "actor_sha256": finalist["actor_sha256"],
            "arm": finalist["arm"],
            "training_seed": finalist["training_seed"],
            "environment_steps": finalist["environment_steps"],
        }
        confirmation = {}
        for seed in protocol["student_training"]["learner_seeds"]:
            for arm in ("rlpd", "sac"):
                selected = screen["selected"][f"{arm}-seed{seed}"]
                actor_record = candidate_by_hash[selected["actor_sha256"]]
                run_config = json.loads((run_root / f"{arm}-seed{seed}" / "config.json").read_text())
                # A zero-screen control is diagnostic only; it cannot open blind or promote.
                diagnostic = arm == "sac" and selected["canonical_finishes"] == 0
                pointer = run_root / f"confirmation-{arm}-seed{seed}.json"
                record = _evaluate_selected_actor(
                    protocol_path=protocol_path,
                    protocol=protocol,
                    run_root=run_root,
                    actor_path=Path(actor_record["path"]),
                    previous_pointer=screen["pointer"],
                    output_pointer=pointer,
                    partition="confirmation",
                    diagnostic=diagnostic,
                    workers=workers,
                    cpu_python=cpu_python,
                )
                confirmation[f"{arm}-seed{seed}"] = record
                execution = json.loads((run_root / "execution.json").read_text())
                execution["stages"].append({
                    "name": "confirmation_actor",
                    "arm": arm,
                    "training_seed": seed,
                    "diagnostic_only": diagnostic,
                    "result": record,
                })
                execution["confirmation_opened"] = True
                write_json(run_root / "execution.json", execution)
        confirmation_gates = {
            str(seed): {
                "rlpd_finishes": confirmation[f"rlpd-seed{seed}"]["canonical_finishes"],
                "sac_finishes": confirmation[f"sac-seed{seed}"]["canonical_finishes"],
                "pass": bool(
                    confirmation[f"rlpd-seed{seed}"]["eligible"]
                    and confirmation[f"rlpd-seed{seed}"]["determinism_audited"]
                    and confirmation[f"rlpd-seed{seed}"]["cpu_reload_matches"]
                    and confirmation[f"rlpd-seed{seed}"]["operational_failures"] == 0
                    and confirmation[f"sac-seed{seed}"]["eligible"]
                    and confirmation[f"sac-seed{seed}"]["determinism_audited"]
                    and confirmation[f"sac-seed{seed}"]["cpu_reload_matches"]
                    and confirmation[f"sac-seed{seed}"]["operational_failures"] == 0
                    and confirmation[f"rlpd-seed{seed}"]["canonical_finishes"] >= 1
                    and confirmation[f"rlpd-seed{seed}"]["canonical_finishes"]
                    > confirmation[f"sac-seed{seed}"]["canonical_finishes"]
                ),
            }
            for seed in protocol["student_training"]["learner_seeds"]
        }
        result["confirmation"] = confirmation
        result["confirmation_seed_gates"] = confirmation_gates
        result["confirmation_opened"] = True
        if not all(gate["pass"] for gate in confirmation_gates.values()):
            result["confirmation_gate"] = "stop_hold_failure"
            result["blind_opened"] = False
            result["next_gate"] = "confirmation failed; blind remains closed"
            output = ROOT / protocol["result_path"]
            write_json(output, result, exclusive=True)
            write_json(run_root / "result.json", result, exclusive=True)
            _update_execution(run_root, status=result["confirmation_gate"], result_path=output)
            return result

        protocol = read_protocol(protocol_path)
        blind_seed = int(finalist["training_seed"])
        blind_confirmation = confirmation[f"rlpd-seed{blind_seed}"]
        blind_pointer = run_root / "blind-finalist.json"
        blind = _evaluate_selected_actor(
            protocol_path=protocol_path,
            protocol=protocol,
            run_root=run_root,
            actor_path=Path(candidate_by_hash[finalist["actor_sha256"]]["path"]),
            previous_pointer=Path(blind_confirmation["pointer"]),
            output_pointer=blind_pointer,
            partition="blind",
            diagnostic=False,
            workers=workers,
            cpu_python=cpu_python,
        )
        result["blind"] = blind
        result["blind_opened"] = True
        result["blind_gate"] = "recorded_final_internal_generalization_result"
        execution = json.loads((run_root / "execution.json").read_text())
        execution["stages"].append({"name": "blind_finalist", "result": blind})
        execution["blind_opened"] = True
        write_json(run_root / "execution.json", execution)
        output = ROOT / protocol["result_path"]
        write_json(output, result, exclusive=True)
        write_json(run_root / "result.json", result, exclusive=True)
        _update_execution(run_root, status="completed_internal_blind", result_path=output)
        return result
    except BaseException as exc:
        _update_execution(run_root, status="failed_execution", failure=exc)
        raise


def main():
    args = parse_args()
    result = run(args.protocol, args.run_root, workers=args.workers)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
