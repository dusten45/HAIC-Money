"""Resume V5 confirmation lineage after a selection-key error; never replay screen cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, read_protocol, sha256_file, write_json
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol
from scripts.run_rlpd_pilot import _summarize_screen


RUN_ROOT = ROOT / "runs/20260925-pixel-rlpd-entropy-target-ablation-v5"
RECOVERY_FORMAT = "haic-rlpd-entropy-v5-recovery-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-confirmation-and-gated-blind", action="store_true")
    return parser.parse_args()


def _eval_command(protocol_path, run_root, partition, output, model, previous, workers, cpu_python):
    return [
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
        str(model),
    ]


def _summarize_one(pointer_path: Path, actor_sha: str, partition: str, expected_cells: int):
    receipt = json.loads(pointer_path.read_text())
    evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError(f"{partition} receipt must contain exactly one target candidate")
    candidate, summary = candidates[0], summaries[0]
    if candidate["archive_sha256"] != actor_sha:
        raise ValueError(f"{partition} result actor doesn't match frozen screen selection")
    if summary.get("canonical_episodes") != expected_cells:
        raise ValueError(f"{partition} canonical episode count doesn't match its geometry matrix")
    return {
        "status": "complete",
        "partition": partition,
        "pointer": str(pointer_path.relative_to(ROOT)),
        "pointer_sha256": sha256_file(pointer_path),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "actor_sha256": actor_sha,
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finish_counts.get(candidate["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def preflight(protocol_path: Path, run_root: Path, workers: int):
    protocol_path = Path(protocol_path).resolve()
    run_root = Path(run_root).resolve()
    protocol = read_entropy_v5_protocol(protocol_path)
    if protocol["name"] != STUDY_NAME or run_root != RUN_ROOT.resolve():
        raise ValueError("recovery must match the frozen V5 experiment/run path")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be positive")
    execution = json.loads((run_root / "execution.json").read_text())
    if (
        execution.get("status") != "failed_entropy_ablation_stage"
        or execution.get("failure", {}).get("type") != "KeyError"
        or execution.get("failure", {}).get("message") != "'projections'"
        or execution.get("confirmation_opened") is not False
        or execution.get("blind_opened") is not False
        or len(execution.get("stages", [])) != 3
        or execution["stages"][-1].get("name") != "screen"
        or execution["stages"][-1].get("passed") is not True
    ):
        raise ValueError("original V5 run doesn't prove the expected post-screen/zero-confirmation recovery point")

    selection_path = run_root / "screen-selection-projections.json"
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("protocol_sha256") != sha256_file(protocol_path)
        or selection.get("environment_interactions") != 0
        or selection.get("cells_replayed") is not False
        or selection.get("screen_gates") != {
            arm: {str(seed): True for seed in protocol["student_training"]["learner_seeds"]}
            for arm in protocol["student_training"]["arms"]
        }
    ):
        raise ValueError("screen projection/gates don't match the frozen V5 protocol")
    if not all(
        selection.get("confirmation_receipts", {}).get(key, {}).get("previous_evaluation_preflight")
        for key in (
            f"{arm}-seed{seed}"
            for arm in protocol["student_training"]["arms"]
            for seed in protocol["student_training"]["learner_seeds"]
        )
    ):
        raise ValueError("all four exact screen actor predecessor receipts must be preflighted")

    for arm in protocol["student_training"]["arms"]:
        for seed in protocol["student_training"]["learner_seeds"]:
            if (run_root / f"confirmation-{arm}-seed{seed}.json").exists():
                raise FileExistsError("confirmation result already exists; audit its consumed cells first")
    if (run_root / "blind-finalist.json").exists():
        raise FileExistsError("blind result already exists; do not run a second blind")
    if (run_root / "entropy-recovery.json").exists():
        raise FileExistsError("entropy recovery record already exists; do not rerun it")
    blind_finalist = protocol["evaluation"].get("blind_finalist")
    return {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "run_root": run_root,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": sha256_file(selection_path),
        "confirmation_cells": len(protocol["partitions"]["confirmation"]["track_ids"])
        * len(protocol["partitions"]["confirmation"]["seeds"]),
        "blind_cells": len(protocol["partitions"]["blind"]["track_ids"])
        * len(protocol["partitions"]["blind"]["seeds"]),
        "cpu_python": protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"],
        "blind_selection_rule": blind_finalist,
    }


def _confirmation_target_winner(protocol, confirmations):
    arms = protocol["student_training"]["arms"]
    seeds = protocol["student_training"]["learner_seeds"]
    for arm in arms:
        for seed in seeds:
            row = confirmations[f"{arm}-seed{seed}"]
            if (
                row["canonical_episodes"] != 32
                or not row["eligible"]
                or not row["determinism_audited"]
                or not row["cpu_reload_matches"]
                or row["operational_failures"] != 0
                or row["canonical_finishes"] < 1
            ):
                return None
    author, positive = arms
    author_finishes = [confirmations[f"{author}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    positive_finishes = [confirmations[f"{positive}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    author_dominates = all(a >= p for a, p in zip(author_finishes, positive_finishes)) and any(
        a > p for a, p in zip(author_finishes, positive_finishes)
    )
    positive_dominates = all(p >= a for a, p in zip(author_finishes, positive_finishes)) and any(
        p > a for a, p in zip(author_finishes, positive_finishes)
    )
    if author_dominates == positive_dominates:
        return None
    return author if author_dominates else positive


def _blind_actor(protocol, winner, confirmations, selection):
    seeds = protocol["student_training"]["learner_seeds"]
    candidates = []
    for seed in seeds:
        key = f"{winner}-seed{seed}"
        confirmation = confirmations[key]
        summary = confirmation["summary"]
        screen = selection["selected"][key]
        lap = summary.get("avg_lap_time_ms")
        candidates.append((
            confirmation["canonical_finishes"],
            summary.get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -screen["environment_steps"],
            -seed,
            key,
        ))
    return max(candidates, key=lambda item: item[:5])[-1]


def execute(plan, *, workers: int = 2):
    protocol, run_root = plan["protocol"], plan["run_root"]
    recovery_path = run_root / "entropy-recovery.json"
    script_path = Path(__file__).resolve()
    recovery = {
        "format": "haic-rlpd-entropy-v5-recovery-v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running_confirmation",
        "protocol_sha256": plan["protocol_sha256"],
        "screen_selection_path": str(plan["selection_path"].relative_to(ROOT)),
        "screen_selection_sha256": plan["selection_sha256"],
        "recovery_script_sha256": sha256_file(script_path),
        "screen_cells_reexecuted": 0,
        "confirmation_cells_per_candidate": plan["confirmation_cells"],
        "blind_cells_one_candidate": plan["blind_cells"],
        "confirmation_results": {},
        "blind_opened": False,
        "blind_status": "closed_until_confirmation_gate",
    }
    write_json(recovery_path, recovery, exclusive=True)
    try:
        confirmation = {}
        for arm in protocol["student_training"]["arms"]:
            for seed in protocol["student_training"]["learner_seeds"]:
                key = f"{arm}-seed{seed}"
                selected = plan["selection"]["selected"][key]
                actor_path = Path(selected["source_path"]).resolve()
                if sha256_file(actor_path) != selected["actor_sha256"]:
                    raise ValueError(f"selected V5 actor changed before confirmation: {key}")
                previous = Path(
                    plan["selection"]["confirmation_receipts"][key]["previous_screen_pointer"]
                ).resolve()
                output = run_root / f"confirmation-{key}.json"
                command = [
                    plan["cpu_python"],
                    "-m",
                    "evaluate_policy",
                    "--run-dir",
                    str(run_root),
                    "--protocol-file",
                    str(plan["protocol_path"]),
                    "--partition",
                    "confirmation",
                    "--previous-evaluation",
                    str(previous),
                    "--python",
                    plan["cpu_python"],
                    "--workers",
                    str(workers),
                    "--evaluations-dir",
                    str(ROOT / "evaluations"),
                    "--output",
                    str(output),
                    "--model",
                    str(actor_path),
                ]
                recovery.setdefault("confirmation_attempts", {})[key] = {
                    "status": "running",
                    "actor_sha256": selected["actor_sha256"],
                    "previous_screen_pointer": str(previous),
                    "command": command,
                    "canonical_cells": plan["confirmation_cells"],
                }
                write_json(recovery_path, recovery)
                subprocess.run(command, cwd=ROOT, check=True)
                record = json.loads(output.read_text())
                evaluation_dir = Path(record["evaluation_dir"]).resolve()
                rows, summaries, finishes = _summarize_screen(evaluation_dir)
                if len(rows) != 1 or len(summaries) != 1 or rows[0]["archive_sha256"] != selected["actor_sha256"]:
                    raise ValueError(f"confirmation actor-binding mismatch for {key}")
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
                recovery["confirmation_results"][key] = confirmation[key]
                recovery["confirmation_attempts"][key]["status"] = "complete"
                write_json(recovery_path, recovery)
        winner = _confirmation_target_winner(protocol, confirmation)
        recovery["confirmation_gate"] = "pass" if winner is not None else "stop_hold_failure"
        recovery["confirmation_target_winner"] = winner
        if winner is None:
            recovery["status"] = "confirmation_gate_failed_blind_closed"
            recovery["blind_opened"] = False
            recovery["blind_status"] = "closed_after_failed_paired_target_gate"
            recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(recovery_path, recovery)
            result = _build_result(plan, recovery)
            _save_result(plan, result, status=recovery["status"])
            return result

        finalist_key = _blind_actor(protocol, winner, confirmation, plan["selection"])
        finalist = confirmation[finalist_key]
        selected = plan["selection"]["selected"][finalist_key]
        actor_path = Path(selected["source_path"]).resolve()
        previous = (ROOT / finalist["pointer"]).resolve()
        blind_output = run_root / "blind-finalist.json"
        blind_command = [
            plan["cpu_python"],
            "-m",
            "evaluate_policy",
            "--run-dir",
            str(run_root),
            "--protocol-file",
            str(plan["protocol_path"]),
            "--partition",
            "blind",
            "--previous-evaluation",
            str(previous),
            "--python",
            plan["cpu_python"],
            "--workers",
            str(workers),
            "--evaluations-dir",
            str(ROOT / "evaluations"),
            "--output",
            str(blind_output),
            "--model",
            str(actor_path),
        ]
        recovery["blind_opened"] = True
        recovery["blind_status"] = "running_single_confirmation_selected_actor"
        recovery["blind_finalist"] = {
            "arm": winner,
            "training_seed": finalist["training_seed"],
            "actor_sha256": selected["actor_sha256"],
            "target_entropy": finalist["target_entropy"],
        }
        recovery["blind_command"] = blind_command
        write_json(recovery_path, recovery)
        subprocess.run(blind_command, cwd=ROOT, check=True)
        pointer = json.loads(blind_output.read_text())
        evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
        rows, summaries, finishes = _summarize_screen(evaluation_dir)
        if len(rows) != 1 or len(summaries) != 1 or rows[0]["archive_sha256"] != selected["actor_sha256"]:
            raise ValueError("blind evaluator did not bind the frozen target winner")
        summary = summaries[0]
        recovery["blind_result"] = {
            "pointer": str(blind_output.relative_to(ROOT)),
            "pointer_sha256": sha256_file(blind_output),
            "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
            "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
            "actor_sha256": selected["actor_sha256"],
            "target_entropy": finalist["target_entropy"],
            "canonical_episodes": summary["canonical_episodes"],
            "canonical_finishes": finishes.get(rows[0]["candidate_id"], 0),
            "eligible": summary["eligible"],
            "determinism_audited": summary["determinism_audited"],
            "cpu_reload_matches": summary["cpu_reload_matches"],
            "operational_failures": summary["operational_failures"],
            "summary": summary["summary"],
        }
        recovery["blind_status"] = "complete_internal_final_result"
        recovery["status"] = "completed_internal_blind"
        recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        result = _build_result(plan, recovery)
        _save_result(plan, result, status=recovery["status"])
        return result
    except BaseException as exc:
        recovery["status"] = "failed_evaluator_after_dispatch_audit_cells_before_retry"
        recovery["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recovery["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        raise


def _build_result(plan, recovery):
    protocol = plan["protocol"]
    dataset = json.loads((plan["run_root"] / "prior-data" / "manifest.json").read_text())
    screen = json.loads((plan["run_root"] / "screen-selection-projections.json").read_text())
    return {
        "format": "haic-rlpd-entropy-target-ablation-result-v2",
        "study_id": protocol["name"],
        "protocol_path": str(Path(protocol["_path"]).relative_to(ROOT)),
        "protocol_sha256": plan["protocol_sha256"],
        "v5_prior_dataset": {
            "sha256": dataset["dataset_sha256"],
            "decisions_spent": dataset["decisions_spent"],
            "stored_decisions": dataset["stored_decisions"],
            "episodes": len(dataset["episodes"]),
            "distinct_finish_geometries": dataset["distinct_finish_geometries"],
        },
        "screen": {
            "parent_pointer": str((plan["run_root"] / "screen-evaluation.json").relative_to(ROOT)),
            "parent_pointer_sha256": sha256_file(plan["run_root"] / "screen-evaluation.json"),
            "parent_manifest_sha256": screen["parent_screen_manifest_sha256"],
            "parent_summary_sha256": screen["parent_screen_summary_sha256"],
            "selected_by_arm_seed": screen["selected"],
            "screen_gates": screen["screen_gates"],
            "cells_replayed": False,
        },
        "confirmation": recovery.get("confirmation_results"),
        "confirmation_target_winner": recovery.get("confirmation_target_winner"),
        "confirmation_gate": recovery.get("confirmation_gate"),
        "blind_opened": recovery.get("blind_opened"),
        "blind_finalist": recovery.get("blind_finalist"),
        "blind": recovery.get("blind_result"),
        "recovery_receipt_path": str((plan["run_root"] / "entropy-recovery.json").relative_to(ROOT)),
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "confirmation_cells_replayed": 0,
        "official_performance_claim": False,
    }


def _save_result(plan, result, *, status):
    path = ROOT / plan["protocol"]["result_path"]
    write_json(path, result, exclusive=True)
    write_json(plan["run_root"] / "result.json", result, exclusive=True)
    execution_path = plan["run_root"] / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["status"] = status
    execution["result_path"] = str(path.relative_to(ROOT))
    execution["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(execution_path, execution)


def run(protocol_path: Path, run_root: Path, workers: int = 2):
    protocol_path = Path(protocol_path).resolve()
    run_root = Path(run_root).resolve()
    protocol = read_entropy_v5_protocol(protocol_path)
    if run_root != (ROOT / "runs/20260925-pixel-rlpd-entropy-target-ablation-v5").resolve():
        raise ValueError("V5 recovery may only operate on the frozen V5 run root")
    execution = json.loads((run_root / "execution.json").read_text())
    if execution.get("status") != "failed_entropy_ablation_stage" or execution.get("failure") != {
        "type": "KeyError",
        "message": "'projections'",
    }:
        raise ValueError("original run failure does not match the audited post-screen projection-key error")
    selection_path = run_root / "screen-selection-projections.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("protocol_sha256") != sha256_file(protocol_path) or not all(
        all(values.values()) for values in selection.get("screen_gates", {}).values()
    ):
        raise ValueError("screen-selection artifact does not pass the frozen V5 per-arm/per-seed gate")
    if not all(
        record.get("previous_evaluation_preflight")
        for record in selection.get("confirmation_receipts", {}).values()
    ):
        raise ValueError("not every single-actor confirmation receipt passed preflight")
    if (run_root / "entropy-recovery.json").exists():
        raise FileExistsError("V5 recovery record exists; do not retry or replay cells")
    for arm in protocol["student_training"]["arms"]:
        for seed in protocol["student_training"]["learner_seeds"]:
            if (run_root / f"confirmation-{arm}-seed{seed}.json").exists():
                raise FileExistsError("V5 confirmation pointer exists; audit exact cell consumption")
    if (run_root / "blind-finalist.json").exists():
        raise FileExistsError("V5 blind pointer exists; never rerun blind")
    return {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "run_root": run_root,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": sha256_file(selection_path),
        "confirmation_cells": (
            len(protocol["partitions"]["confirmation"]["track_ids"])
            * len(protocol["partitions"]["confirmation"]["seeds"])
        ),
        "blind_cells": (
            len(protocol["partitions"]["blind"]["track_ids"])
            * len(protocol["partitions"]["blind"]["seeds"])
        ),
        "cpu_python": protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-confirmation-and-gated-blind", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute_confirmation_and_gated_blind:
        raise ValueError("choose exactly one preflight-only or execute mode")
    plan = run(args.protocol, args.run_root, workers=args.workers)
    if args.preflight_only:
        print(json.dumps({
            "status": "confirmation_lineage_preflight_passed_no_environment_work",
            "protocol_sha256": plan["protocol_sha256"],
            "selection_sha256": plan["selection_sha256"],
            "confirmation_actor_count": 4,
            "blind_candidate_count": 1,
        }, sort_keys=True), flush=True)
        return
    result = execute(plan, workers=args.workers)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
