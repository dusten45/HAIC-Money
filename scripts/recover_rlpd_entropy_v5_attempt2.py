"""Retry only V5 confirmation after a no-cell command-construction failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, sha256_file, write_json
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol
from scripts.run_rlpd_pilot import _summarize_screen


RUN_ROOT = ROOT / "runs/20260925-pixel-rlpd-entropy-target-ablation-v5"
RECOVERY_PATH = RUN_ROOT / "entropy-recovery-attempt2.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-confirmation-and-gated-blind", action="store_true")
    return parser.parse_args()


def _eval_command(protocol_path, run_root, partition, model_path, previous, output, cpu_python, workers):
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
        str(model_path),
    ]


def _bind_screen_actor(protocol, run_root, key, selected, projection):
    actor_path = Path(selected["source_path"]).resolve()
    if sha256_file(actor_path) != selected["actor_sha256"]:
        raise ValueError(f"screen actor hash changed before confirmation: {key}")
    pointer = Path(projection["previous_screen_pointer"]).resolve()
    if sha256_file(pointer) != projection["previous_screen_pointer_sha256"]:
        raise ValueError(f"screen selection pointer changed for {key}")
    actor = candidate_metadata(actor_path, run_dir=run_root)
    prior = previous_evaluation_metadata(
        pointer,
        "confirmation",
        sha256_file(protocol["_path"]),
        actor,
    )
    return {
        "actor_path": str(actor_path),
        "actor_sha256": selected["actor_sha256"],
        "environment_steps": selected["environment_steps"],
        "canonical_screen_finishes": selected["canonical_finishes"],
        "previous_screen_pointer": str(pointer),
        "previous_screen_pointer_sha256": sha256_file(pointer),
        "previous_evaluation_metadata": prior,
        "arm": selected["arm"],
        "training_seed": selected["training_seed"],
    }


def preflight(protocol_path, run_root, workers):
    protocol_path, run_root = Path(protocol_path).resolve(), Path(run_root).resolve()
    protocol = read_entropy_v5_protocol(protocol_path)
    if protocol["name"] != STUDY_NAME or run_root != RUN_ROOT.resolve():
        raise ValueError("recovery scope must match the V5 frozen study/run root")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be positive")
    first_attempt = json.loads((run_root / "entropy-recovery.json").read_text())
    if (
        first_attempt.get("status") != "failed_evaluator_after_dispatch_audit_cells_before_retry"
        or first_attempt.get("failure", {}).get("message") != "'protocol_path'"
        or first_attempt.get("screen_cells_reexecuted") != 0
        or first_attempt.get("confirmation_results") != {}
        or first_attempt.get("blind_opened") is not False
    ):
        raise ValueError("only the audited no-dispatch V5 recovery attempt can be continued")
    if RECOVERY_PATH.exists() or (run_root / "blind-finalist.json").exists():
        raise FileExistsError("V5 recovery attempt 2 or blind output already exists; inspect before any retry")

    selection_path = run_root / "screen-selection-projections.json"
    selection = json.loads(selection_path.read_text())
    if (
        selection.get("protocol_sha256") != sha256_file(protocol_path)
        or selection.get("cells_replayed") is not False
        or selection.get("environment_interactions") != 0
    ):
        raise ValueError("selection projections are not bound to V5 source screen without replay")
    screen_gates = selection.get("screen_gates", {})
    if set(screen_gates) != set(protocol["student_training"]["arms"]) or any(
        set(screen_gates[arm]) != {str(seed) for seed in protocol["student_training"]["learner_seeds"]}
        or not all(screen_gates[arm].values())
        for arm in protocol["student_training"]["arms"]
    ):
        raise ValueError("both target arms and both learner seeds must pass the frozen screen gate")
    candidates = selection.get("selected", {})
    projections = selection.get("confirmation_receipts", {})
    expected_keys = {
        f"{arm}-seed{seed}"
        for arm in protocol["student_training"]["arms"]
        for seed in protocol["student_training"]["learner_seeds"]
    }
    if set(candidates) != expected_keys or set(projections) != expected_keys:
        raise ValueError("V5 screen selection doesn't contain exactly its four arm/seed winners")
    bindings = {
        key: _bind_screen_actor(protocol, run_root, key, candidates[key], projections[key])
        for key in sorted(expected_keys)
    }
    for arm in protocol["student_training"]["arms"]:
        for seed in protocol["student_training"]["learner_seeds"]:
            if (run_root / f"confirmation-{arm}-seed{seed}.json").exists():
                raise FileExistsError(f"confirmation output exists; do not repeat cells: {arm}-seed{seed}")

    return {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "run_root": run_root,
        "selection_path": selection_path,
        "selection_sha256": sha256_file(selection_path),
        "selection": selection,
        "bindings": bindings,
        "cpu_python": protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"],
        "workers": workers,
        "confirmation_cells": len(protocol["partitions"]["confirmation"]["track_ids"])
        * len(protocol["partitions"]["confirmation"]["seeds"]),
        "blind_cells": len(protocol["partitions"]["blind"]["track_ids"])
        * len(protocol["partitions"]["blind"]["seeds"]),
    }


def _one_candidate_result(output, expected_sha, cells):
    pointer = json.loads(output.read_text())
    evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
    candidates, summaries, finishes = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1 or candidates[0]["archive_sha256"] != expected_sha:
        raise ValueError("confirmation/blind receipt isn't bound to its exact screen candidate")
    summary = summaries[0]
    if summary.get("canonical_episodes") != cells:
        raise ValueError("confirmation/blind canonical episode count differs from protocol")
    return {
        "pointer": str(output.relative_to(ROOT)),
        "pointer_sha256": sha256_file(output),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "actor_sha256": expected_sha,
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finishes.get(candidates[0]["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def _target_winner(protocol, confirmations):
    author, positive = protocol["student_training"]["arms"]
    seeds = protocol["student_training"]["learner_seeds"]
    for arm in (author, positive):
        for seed in seeds:
            result = confirmations[f"{arm}-seed{seed}"]
            if (
                result["canonical_episodes"] != 32
                or not result["eligible"]
                or not result["determinism_audited"]
                or not result["cpu_reload_matches"]
                or result["operational_failures"] != 0
                or result["canonical_finishes"] < 1
            ):
                return None
    author_finishes = [confirmations[f"{author}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    positive_finishes = [confirmations[f"{positive}-seed{seed}"]["canonical_finishes"] for seed in seeds]
    author_wins = all(a >= p for a, p in zip(author_finishes, positive_finishes)) and any(
        a > p for a, p in zip(author_finishes, positive_finishes)
    )
    positive_wins = all(p >= a for a, p in zip(author_finishes, positive_finishes)) and any(
        p > a for a, p in zip(author_finishes, positive_finishes)
    )
    if author_wins == positive_wins:
        return None
    return author if author_wins else positive


def _blind_seed(protocol, winner, confirmations, bindings):
    candidates = []
    for seed in protocol["student_training"]["learner_seeds"]:
        key = f"{winner}-seed{seed}"
        confirm = confirmations[key]
        summary = confirm["summary"]
        screen_step = bindings[key]["environment_steps"]
        lap = summary.get("avg_lap_time_ms")
        candidates.append((
            confirm["canonical_finishes"],
            summary.get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -screen_step,
            -seed,
            key,
        ))
    return max(candidates, key=lambda item: item[:5])[-1]


def _save_result(plan, recovery, *, status):
    result_path = ROOT / plan["protocol"]["result_path"]
    result = {
        "format": "haic-rlpd-entropy-target-ablation-result-v5",
        "study_id": plan["protocol"]["name"],
        "protocol_sha256": plan["protocol_sha256"],
        "recovery": str((plan["run_root"] / "entropy-recovery-attempt2.json").relative_to(ROOT)),
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "teacher_dataset_sha256": json.loads((plan["run_root"] / "prior-data/manifest.json").read_text())["dataset_sha256"],
        "screen_selection_path": str(plan["selection_path"].relative_to(ROOT)),
        "screen_selection_sha256": plan["selection_sha256"],
        "screen_gate": "pass",
        "screen_selected_by_arm_seed": plan["selection"]["selected"],
        "confirmation": recovery.get("confirmation_results"),
        "confirmation_target_winner": recovery.get("confirmation_target_winner"),
        "confirmation_gate": recovery.get("confirmation_gate"),
        "blind_opened": recovery.get("blind_opened", False),
        "blind_finalist": recovery.get("blind_finalist"),
        "blind": recovery.get("blind_result"),
        "status": status,
        "official_performance_claim": False,
    }
    write_json(result_path, result, exclusive=True)
    write_json(plan["run_root"] / "result.json", result, exclusive=True)
    execution_path = plan["run_root"] / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["recovery_path"] = str((plan["run_root"] / "entropy-recovery-attempt2.json").relative_to(ROOT))
    execution["final_status"] = status
    execution["result_path"] = str(result_path.relative_to(ROOT))
    execution["finished_at_utc"] = recovery["finished_at_utc"]
    write_json(execution_path, execution)
    return result


def execute(plan):
    recovery_path = plan["run_root"] / "entropy-recovery-attempt2.json"
    if recovery_path.exists():
        raise FileExistsError("V5 recovery attempt 2 exists; inspect before any continuation")
    recovery = {
        "format": "haic-rlpd-entropy-v5-recovery-attempt2-v1",
        "status": "running_confirmation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": plan["protocol_sha256"],
        "prior_no_dispatch_attempt": str((plan["run_root"] / "entropy-recovery.json").relative_to(ROOT)),
        "prior_screen_pointer": str((plan["run_root"] / "screen-evaluation.json").relative_to(ROOT)),
        "screen_selection_path": str(plan["selection_path"].relative_to(ROOT)),
        "screen_cells_reexecuted": 0,
        "confirmation_cells_per_candidate": 32,
        "blind_cells_one_finalist": 24,
        "confirmation_attempts": {},
        "confirmation_results": {},
        "blind_opened": False,
    }
    write_json(recovery_path, recovery, exclusive=True)
    try:
        for arm in plan["protocol"]["student_training"]["arms"]:
            for seed in plan["protocol"]["student_training"]["learner_seeds"]:
                key = f"{arm}-seed{seed}"
                bound = plan["bindings"][key]
                output = plan["run_root"] / f"confirmation-{key}.json"
                command = _eval_command(
                    plan["protocol_path"],
                    plan["run_root"],
                    "confirmation",
                    bound["actor_path"],
                    bound["previous_screen_pointer"],
                    output,
                    plan["cpu_python"],
                    plan["workers"],
                )
                recovery["confirmation_attempts"][key] = {
                    "status": "running",
                    "actor_sha256": bound["actor_sha256"],
                    "previous_screen_pointer_sha256": bound["previous_screen_pointer_sha256"],
                    "canonical_cells": 32,
                    "command": command,
                }
                write_json(recovery_path, recovery)
                subprocess.run(command, cwd=ROOT, check=True)
                result = _one_candidate_result(output, bound["actor_sha256"], 32)
                result.update({"arm": arm, "training_seed": seed, "target_entropy": bound["target_entropy"]})
                recovery["confirmation_attempts"][key]["status"] = "complete"
                recovery["confirmation_results"][key] = result
                write_json(recovery_path, recovery)

        winner = _target_winner(plan["protocol"], recovery["confirmation_results"])
        recovery["confirmation_target_winner"] = winner
        recovery["confirmation_gate"] = "pass" if winner is not None else "stop_hold_failure"
        recovery["confirmation_status"] = "completed_all_four_candidates"
        if winner is None:
            recovery["status"] = "confirmation_failed; blind_closed"
            recovery["blind_opened"] = False
            recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(recovery_path, recovery)
            return _save_result(plan, recovery, status=recovery["status"])

        finalist_key = _blind_seed(plan["protocol"], winner, recovery["confirmation_results"], plan["bindings"])
        bound = plan["bindings"][finalist_key]
        previous = plan["run_root"] / f"confirmation-{finalist_key}.json"
        blind_output = plan["run_root"] / "blind-finalist.json"
        command = _eval_command(
            plan["protocol_path"],
            plan["run_root"],
            "blind",
            bound["actor_path"],
            previous,
            blind_output,
            plan["cpu_python"],
            plan["workers"],
        )
        recovery["blind_opened"] = True
        recovery["blind_finalist"] = {
            "arm": winner,
            "training_seed": bound["training_seed"],
            "target_entropy": bound["target_entropy"],
            "actor_sha256": bound["actor_sha256"],
        }
        recovery["blind_status"] = "running_one_confirmation_selected_actor"
        recovery["blind_command"] = command
        write_json(recovery_path, recovery)
        subprocess.run(command, cwd=ROOT, check=True)
        blind_result = _one_candidate_result(blind_output, bound["actor_sha256"], 24)
        blind_result.update({"arm": winner, "training_seed": bound["training_seed"], "target_entropy": bound["target_entropy"]})
        recovery["blind_result"] = blind_result
        recovery["blind_status"] = "complete_internal_final_blind"
        recovery["status"] = "completed_internal_blind"
        recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        return _save_result(plan, recovery, status=recovery["status"])
    except BaseException as exc:
        recovery["status"] = "failed_confirmation_or_blind; audit exact cell consumption before any retry"
        recovery["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recovery["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-confirmation-and-gated-blind", action="store_true")
    args = parser.parse_args()
    if args.preflight_only == args.execute_confirmation_and_gated_blind:
        raise ValueError("choose exactly one preflight or execute mode")
    plan = preflight(args.protocol, args.run_root, args.workers)
    if args.preflight_only:
        print(json.dumps({
            "status": "V5 confirmation lineages passed zero-environment preflight",
            "protocol_sha256": plan["protocol_sha256"],
            "selection_sha256": plan["selection_sha256"],
            "confirmation_actors": 4,
            "confirmation_cells_per_actor": plan["confirmation_cells"],
            "blind_cells_if_gate_passes": plan["blind_cells"],
        }, sort_keys=True), flush=True)
        return
    result = execute(plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
