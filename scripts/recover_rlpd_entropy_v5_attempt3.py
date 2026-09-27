"""Resume V5 from one completed confirmation without repeating that actor's cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluate_policy import candidate_metadata
from scripts.rlpd_common import ROOT, read_protocol, sha256_file, write_json
from scripts.run_rlpd_pilot import _summarize_screen


RUN_ROOT = ROOT / "runs/20260925-pixel-rlpd-entropy-target-ablation-v5"
V5_NAME = "pixel-rlpd-entropy-target-ablation-v5"
CONFIRM_KEYS = (
    "rlpd-author-target-seed50",
    "rlpd-author-target-seed51",
    "rlpd-positive-target-seed50",
    "rlpd-positive-target-seed51",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-remaining-confirmation-and-gated-blind", action="store_true")
    return parser.parse_args()


def _candidate_binding(protocol, run_root, selection, key):
    selected = selection["selected"][key]
    receipt = selection["confirmation_receipts"][key]
    actor_path = Path(selected["source_path"]).resolve()
    if sha256_file(actor_path) != selected["actor_sha256"]:
        raise ValueError(f"screen-selected actor content hash changed: {key}")
    previous_pointer = Path(receipt["previous_screen_pointer"]).resolve()
    if sha256_file(previous_pointer) != receipt["previous_screen_pointer_sha256"]:
        raise ValueError(f"actor-specific screen predecessor changed: {key}")
    candidate = candidate_metadata(actor_path, run_dir=run_root)
    if candidate.get("archive_sha256") != selected["actor_sha256"] or candidate.get("algorithm") != "rlpd":
        raise ValueError(f"screen actor metadata no longer matches its fixed selection: {key}")
    return {
        "key": key,
        "arm": selected["arm"],
        "training_seed": selected["training_seed"],
        "screen_step": selected["environment_steps"],
        "target_entropy": protocol["student_training"]["arm_specs"][selected["arm"]]["target_entropy"],
        "target_entropy":  -1.5 if selected["arm"] == "rlpd-author-target" else 1.5,
        "actor_path": str(actor_path),
        "actor_sha256": selected["actor_sha256"],
        "learner_checkpoint_sha256": selected["learner_checkpoint_sha256"],
        "previous_screen_pointer": str(previous_pointer),
        "previous_screen_pointer_sha256": sha256_file(previous_pointer),
        "screen_finishes": selected["canonical_finishes"],
    }


def _read_single_confirmation(protocol_path, run_root, key, binding):
    pointer_path = run_root / f"confirmation-{key}.json"
    pointer = json.loads(pointer_path.read_text())
    if (
        pointer.get("protocol_sha256") != sha256_file(protocol_path)
        or pointer.get("partition") != "confirmation"
        or pointer.get("diagnostic_only") is not False
    ):
        raise ValueError(f"existing confirmation pointer has wrong V5 identity: {key}")
    evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError("consumed confirmation must be a single candidate receipt")
    candidate, summary = candidates[0], summaries[0]
    if (
        candidate["archive_sha256"] != binding["actor_sha256"]
        or summary.get("canonical_episodes") != 32
        or summary.get("eligible") is not True
        or summary.get("determinism_audited") is not True
        or summary.get("cpu_reload_matches") is not True
        or summary.get("operational_failures") != 0
    ):
        raise ValueError("consumed confirmation doesn't satisfy its exact actor/runtime contract")
    result = {
        "arm": binding["arm"],
        "training_seed": binding["training_seed"],
        "target_entropy": binding["target_entropy"],
        "actor_sha256": binding["actor_sha256"],
        "pointer": str(pointer_path.relative_to(ROOT)),
        "pointer_sha256": sha256_file(pointer_path),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finish_counts.get(candidate["candidate_id"], 0),
        "eligible": True,
        "determinism_audited": True,
        "cpu_reload_matches": True,
        "operational_failures": 0,
        "summary": summary["summary"],
        "evidence_status": "completed_in_confirmation_attempt_2_before_post-evaluation bookkeeping failure",
    }
    return result


def preflight(protocol_path, run_root, workers):
    protocol_path, run_root = Path(protocol_path).resolve(), Path(run_root).resolve()
    protocol = read_protocol(protocol_path)
    if protocol.get("name") != V5_NAME or run_root != RUN_ROOT.resolve():
        raise ValueError("recovery scope must match the frozen V5 protocol/run root")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be positive")
    previous_recovery = json.loads((run_root / "entropy-recovery-attempt2.json").read_text())
    first_confirmation_key = "rlpd-author-target-seed50"
    first_confirmation = previous_recovery.get("confirmation_attempts", {}).get(first_confirmation_key, {})
    first_pointer = run_root / f"confirmation-{first_confirmation_key}.json"
    if (
        previous_recovery.get("status") != "failed_confirmation_or_blind; audit exact cell consumption before any retry"
        or previous_recovery.get("screen_cells_reexecuted") != 0
        or previous_recovery.get("blind_opened") is not False
        or first_confirmation.get("status") != "running"
        or not first_pointer.is_file()
    ):
        raise ValueError("attempt2 does not prove exactly one completed confirmation and no other actor evaluation")
    for key in CONFIRM_KEYS[1:]:
        if (run_root / f"confirmation-{key}.json").exists():
            raise FileExistsError(f"confirmation already exists; never repeat its cells: {key}")
    blind_pointer = run_root / "blind-finalist.json"
    if blind_pointer.exists() or (run_root / "entropy-recovery-attempt3.json").exists():
        raise FileExistsError("V5 partial-recovery attempt already exists; inspect its cells before proceeding")
    selection = json.loads((run_root / "screen-selection-projections.json").read_text())
    if (
        selection.get("protocol_sha256") != sha256_file(protocol_path)
        or selection.get("environment_interactions") != 0
        or selection.get("cells_replayed") is not False
    ):
        raise ValueError("V5 selected-screen actor projections have changed")
    bindings = {key: _candidate_binding(protocol, run_root, selection, key) for key in CONFIRM_KEYS}
    if not all(binding["screen_finishes"] >= 1 for binding in bindings.values()):
        raise ValueError("screen actor selection doesn't satisfy the frozen two-target/two-seed gate")
    consumed = _read_single_confirmation(protocol_path, run_root, first_confirmation_key, bindings[first_confirmation_key])
    pending = {key: bindings[key] for key in CONFIRM_KEYS[1:]}
    commands = {}
    cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
    for key, binding in pending.items():
        output = run_root / f"confirmation-{key}.json"
        commands[key] = [
            cpu_python, "-m", "evaluate_policy",
            "--run-dir", str(run_root),
            "--protocol-file", str(protocol_path),
            "--partition", "confirmation",
            "--previous-evaluation", binding["previous_screen_pointer"],
            "--python", cpu_python,
            "--workers", str(workers),
            "--evaluations-dir", str(ROOT / "evaluations"),
            "--output", str(output),
            "--model", binding["actor_path"],
        ]
    return {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": sha256_file(protocol_path),
        "run_root": run_root,
        "screen_selection": selection,
        "screen_selection_sha256": sha256_file(run_root / "screen-selection-projections.json"),
        "bindings": bindings,
        "consumed_confirmation": consumed,
        "pending_confirmation_keys": list(pending),
        "commands": commands,
        "workers": workers,
        "cpu_python": cpu_python,
        "screen_results_replayed": 0,
        "confirmation_results_to_evaluate": 3,
        "blind_finalist_limit": 1,
    }


def _evaluate_one(plan, key):
    output = plan["run_root"] / f"confirmation-{key}.json"
    command = plan["commands"][key]
    subprocess.run(command, cwd=ROOT, check=True)
    pointer = json.loads(output.read_text())
    if (
        pointer.get("protocol_sha256") != plan["protocol_sha256"]
        or pointer.get("partition") != "confirmation"
        or pointer.get("diagnostic_only") is not False
    ):
        raise ValueError(f"new confirmation receipt mismatch for {key}")
    evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
    candidates, summaries, finishes = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError(f"confirmation should have one actor for {key}")
    candidate, summary = candidates[0], summaries[0]
    binding = plan["bindings"][key]
    if (
        candidate["archive_sha256"] != binding["actor_sha256"]
        or summary.get("canonical_episodes") != 32
    ):
        raise ValueError(f"confirmation result differs from exact frozen actor/cell contract for {key}")
    return {
        "arm": binding["arm"],
        "training_seed": binding["training_seed"],
        "target_entropy": binding["target_entropy"],
        "actor_sha256": binding["actor_sha256"],
        "pointer": str(output.relative_to(ROOT)),
        "pointer_sha256": sha256_file(output),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finishes.get(candidate["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def _paired_winner(protocol, confirmation_results):
    arms, seeds = protocol["student_training"]["arms"], protocol["student_training"]["learner_seeds"]
    for arm in arms:
        for seed in seeds:
            row = confirmation_results[f"{arm}-seed{seed}"]
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
    a_scores = [confirmation_results[f"{author}-seed{s}"]["canonical_finishes"] for s in seeds]
    p_scores = [confirmation_results[f"{positive}-seed{s}"]["canonical_finishes"] for s in seeds]
    a_wins = all(a >= p for a, p in zip(a_scores, p_scores)) and any(a > p for a, p in zip(a_scores, p_scores))
    p_wins = all(p >= a for a, p in zip(a_scores, p_scores)) and any(p > a for a, p in zip(a_scores, p_scores))
    if a_wins == p_wins:
        return None
    return author if a_wins else positive


def _choose_blind_key(protocol, winner, confirmation_results, bindings):
    seeds = protocol["student_training"]["learner_seeds"]
    items = []
    for seed in seeds:
        key = f"{winner}-seed{seed}"
        result = confirmation_results[key]
        summary = result["summary"]
        lap = summary.get("avg_lap_time_ms")
        items.append((
            result["canonical_finishes"],
            summary.get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -bindings[key]["environment_steps"],
            -seed,
            key,
        ))
    return max(items, key=lambda row: row[:5])[-1]


def _blind_command(plan, key):
    binding = plan["bindings"][key]
    previous = plan["run_root"] / f"confirmation-{key}.json"
    output = plan["run_root"] / "blind-finalist.json"
    cpu = plan["cpu_python"]
    return [
        cpu, "-m", "evaluate_policy",
        "--run-dir", str(plan["run_root"]),
        "--protocol-file", str(plan["protocol_path"]),
        "--partition", "blind",
        "--previous-evaluation", str(previous),
        "--python", cpu,
        "--workers", str(plan["workers"]),
        "--evaluations-dir", str(ROOT / "evaluations"),
        "--output", str(output),
        "--model", binding["actor_path"],
    ]


def _read_blind_result(plan, key):
    binding = plan["bindings"][key]
    pointer_path = plan["run_root"] / "blind-finalist.json"
    pointer = json.loads(pointer_path.read_text())
    if (
        pointer.get("protocol_sha256") != plan["protocol_sha256"]
        or pointer.get("partition") != "blind"
        or pointer.get("diagnostic_only") is not False
    ):
        raise ValueError("blind receipt doesn't match frozen V5 protocol/partition")
    evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
    rows, summaries, finishes = _summarize_screen(evaluation_dir)
    if len(rows) != 1 or len(summaries) != 1 or rows[0]["archive_sha256"] != binding["actor_sha256"]:
        raise ValueError("blind receipt doesn't bind the screen/confirmation winner actor")
    summary = summaries[0]
    if summary.get("canonical_episodes") != 24:
        raise ValueError("blind canonical-cell count differs from the frozen V5 partition")
    return {
        "arm": binding["arm"],
        "training_seed": binding["training_seed"],
        "target_entropy": binding["target_entropy"],
        "actor_sha256": binding["actor_sha256"],
        "pointer": str(pointer_path.relative_to(ROOT)),
        "pointer_sha256": sha256_file(pointer_path),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finishes.get(rows[0]["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def _attempt3_result(plan, recovery):
    protocol, run_root = plan["protocol"], plan["run_root"]
    manifest = json.loads((run_root / "prior-data" / "manifest.json").read_text())
    return {
        "format": "haic-rlpd-entropy-target-ablation-result-v5",
        "study_id": protocol["name"],
        "protocol_path": str(plan["protocol_path"].relative_to(ROOT)),
        "protocol_sha256": plan["protocol_sha256"],
        "teacher_dataset": {
            "sha256": manifest["dataset_sha256"],
            "decisions_spent": manifest["decisions_spent"],
            "stored_decisions": manifest["stored_decisions"],
            "episodes": len(manifest["episodes"]),
            "distinct_finish_geometries": manifest["distinct_finish_geometries"],
        },
        "screen": {
            "pointer": str((run_root / "screen-evaluation.json").relative_to(ROOT)),
            "pointer_sha256": sha256_file(run_root / "screen-evaluation.json"),
            "selection": str((run_root / "screen-selection-projections.json").relative_to(ROOT)),
            "selection_sha256": sha256_file(run_root / "screen-selection-projections.json"),
            "selected_by_arm_seed": plan["selection"]["selected"],
            "gates": plan["selection"]["screen_gates"],
            "cells_replayed": 0,
        },
        "confirmation": recovery["confirmation_results"],
        "confirmation_target_winner": recovery.get("confirmation_target_winner"),
        "confirmation_gate": recovery.get("confirmation_gate"),
        "blind_opened": recovery.get("blind_opened", False),
        "blind_finalist": recovery.get("blind_finalist"),
        "blind": recovery.get("blind_result"),
        "initial_confirmation_attempts": {
            "attempt_1": {"confirmation_cells": 0, "error": "KeyError protocol_path before evaluator dispatch"},
            "attempt_2": {
                "consumed_key": "rlpd-author-target-seed50",
                "result_pointer": "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/confirmation-rlpd-author-target-seed50.json",
                "canonical_finishes": recovery["confirmation_results"]["rlpd-author-target-seed50"]["canonical_finishes"],
                "error": "KeyError target_entropy after evaluator completed; no metrics were lost",
            },
        },
        "recovery_execution": str((run_root / "entropy-recovery-attempt3.json").relative_to(ROOT)),
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "official_performance_claim": False,
    }


def _save_attempt3_result(plan, recovery):
    result = _attempt3_result(plan, recovery)
    target = ROOT / plan["protocol"]["result_path"]
    write_json(target, result, exclusive=True)
    write_json(plan["run_root"] / "result.json", result, exclusive=True)
    execution_path = plan["run_root"] / "execution.json"
    execution = json.loads(execution_path.read_text())
    execution["recovery_status"] = recovery["status"]
    execution["recovery_execution"] = result["recovery_execution"]
    execution["result_path"] = str(target.relative_to(ROOT))
    execution["finished_at_utc"] = recovery["finished_at_utc"]
    write_json(execution_path, execution)
    return result


def execute_attempt3(plan):
    recovery_path = plan["run_root"] / "entropy-recovery-attempt3.json"
    if recovery_path.exists():
        raise FileExistsError("V5 attempt-3 recovery exists; inspect it before any retry")
    recovery = {
        "format": "haic-rlpd-entropy-v5-recovery-attempt3-v1",
        "status": "running_three_remaining_confirmations",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": plan["protocol_sha256"],
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "screen_cells_reexecuted": 0,
        "confirmation_cells": 32,
        "confirmation_results": {"rlpd-author-target-seed50": plan["consumed_confirmation"]},
        "confirmation_attempts": {
            "rlpd-author-target-seed50": {"status": "prior_completion_recorded; not rerun", "pointer_sha256": plan["consumed_confirmation"]["pointer_sha256"]}
        },
        "blind_opened": False,
    }
    write_json(recovery_path, recovery, exclusive=True)
    try:
        for key in CONFIRM_KEYS[1:]:
            output = plan["run_root"] / f"confirmation-{key}.json"
            if output.exists():
                raise FileExistsError(f"confirmation output exists; never rerun consumed cells: {key}")
            command = plan["commands"][key]
            recovery["confirmation_attempts"][key] = {
                "status": "running",
                "actor_sha256": plan["bindings"][key]["actor_sha256"],
                "command": command,
                "cells": plan["confirmation_cells"],
            }
            write_json(recovery_path, recovery)
            result = _evaluate_one(plan, key)
            recovery["confirmation_attempts"][key]["status"] = "complete"
            recovery["confirmation_results"][key] = result
            write_json(recovery_path, recovery)

        winner = _paired_winner(plan["protocol"], recovery["confirmation_results"])
        recovery["confirmation_target_winner"] = winner
        recovery["confirmation_gate"] = "pass" if winner is not None else "stop_hold_failure"
        recovery["confirmation_stage"] = "all four exact actor receipts complete"
        if winner is None:
            recovery["status"] = "confirmation_gate_failed_blind_closed"
            recovery["blind_opened"] = False
            recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(recovery_path, recovery)
            return _save_attempt3_result(plan, recovery)

        finalist_key = _choose_blind_key(plan["protocol"], winner, recovery["confirmation_results"], plan["bindings"])
        finalist = plan["bindings"][finalist_key]
        previous_confirmation = plan["run_root"] / f"confirmation-{finalist_key}.json"
        blind_output = plan["run_root"] / "blind-finalist.json"
        if blind_output.exists():
            raise FileExistsError("blind pointer already exists; never rerun blind cells")
        blind_command = [
            plan["cpu_python"], "-m", "evaluate_policy",
            "--run-dir", str(plan["run_root"]),
            "--protocol-file", str(plan["protocol_path"]),
            "--partition", "blind",
            "--previous-evaluation", str(previous_confirmation),
            "--python", plan["cpu_python"],
            "--workers", str(plan["workers"]),
            "--evaluations-dir", str(ROOT / "evaluations"),
            "--output", str(blind_output),
            "--model", finalist["actor_path"],
        ]
        recovery["blind_opened"] = True
        recovery["blind_finalist"] = {"key": finalist_key, "actor_sha256": finalist["actor_sha256"]}
        recovery["blind_command"] = blind_command
        recovery["status"] = "running_one_confirmation_selected_blind"
        write_json(recovery_path, recovery)
        subprocess.run(blind_command, cwd=ROOT, check=True)
        blind = _read_single_evaluation(
            plan["protocol_path"], blind_output, finalist["actor_sha256"], "blind", 24
        )
        blind.update({"arm": finalist["arm"], "training_seed": finalist["training_seed"], "target_entropy": finalist["target_entropy"]})
        recovery["blind_result"] = blind
        recovery["status"] = "completed_internal_blind"
        recovery["blind_status"] = "one final candidate, all previous gates passed"
        recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        return _save_attempt3_result(plan, recovery)
    except BaseException as exc:
        recovery["status"] = "failed_after_dispatch; audit exact consumed cells before any retry"
        recovery["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recovery["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        raise
def main():
    args = parse_args()
    if args.preflight_only == args.execute_remaining_confirmation_and_gated_blind:
        raise ValueError("choose exactly one preflight or execute mode")
    if args.preflight_only:
        plan = preflight(args.protocol, args.run_root, args.workers)
        print(json.dumps({
            "status": "preflight_passed_without_environment_interaction",
            "protocol_sha256": plan["protocol_sha256"],
            "screen_selection_sha256": plan["screen_selection_sha256"],
            "consumed_confirmation_receipt": plan["consumed_confirmation"],
            "pending_confirmation_keys": plan["pending_confirmation_keys"],
            "pending_confirmation_cells": 3 * 32,
            "blind_cells_if_target_gate_passes": 24,
            "screen_cells_replayed": 0,
        }, sort_keys=True), flush=True)
        return
    if not args.execute_remaining_confirmation_and_gated_blind:
        raise ValueError("execution mode must be explicit")
    plan = preflight(args.protocol, args.run_root, args.workers)
    result = execute_attempt3(plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
