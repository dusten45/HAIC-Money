"""Resume only the unconsumed V3 confirmation gate through exact screen selections."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, read_protocol, sha256_file, write_json
from scripts.run_rlpd_pilot import _summarize_screen


RUN_ROOT_RELATIVE = Path("runs/20260924-pixel-rlpd-long-horizon-followup-v1")
RECOVERY_FORMAT = "haic-rlpd-followup-confirmation-recovery-v1"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-confirmation-and-gated-blind", action="store_true")
    return parser.parse_args()


def _cell_count(protocol, partition):
    matrix = protocol["partitions"][partition]
    return len(matrix["track_ids"]) * len(matrix["seeds"])


def _candidate_contract(protocol, run_root, selection):
    contracts = {}
    for key, selected in selection["selected_by_run"].items():
        arm, raw_seed = key.split("-seed", maxsplit=1)
        seed = int(raw_seed)
        model_path = Path(selected["source_path"]).resolve()
        expected_path = (run_root / f"{arm}-seed{seed}" / "checkpoints" / f"step-{selected['environment_steps']:09d}" / "actor.pt").resolve()
        if model_path != expected_path or sha256_file(model_path) != selected["actor_sha256"]:
            raise ValueError(f"selected actor path/hash mismatch for {key}")
        projection = selection["confirm_receipts"].get(key)
        if not isinstance(projection, dict):
            raise ValueError(f"selected screen receipt is missing for {key}")
        previous_pointer = Path(projection["screen_pointer"]).resolve()
        if sha256_file(previous_pointer) != projection["screen_pointer_sha256"]:
            raise ValueError(f"projected screen receipt changed for {key}")
        candidate = candidate_metadata(model_path, run_dir=run_root)
        if candidate.get("algorithm") != "rlpd" or candidate.get("archive_sha256") != selected["actor_sha256"]:
            raise ValueError(f"candidate metadata differs from screen selection for {key}")
        diagnostic = arm == "sac" and int(selected["canonical_finishes"]) == 0
        predecessor = previous_evaluation_metadata(
            previous_pointer,
            "confirmation",
            sha256_file(protocol["_path"]),
            candidate,
            diagnostic_confirmation=diagnostic,
        )
        contracts[key] = {
            "arm": arm,
            "training_seed": seed,
            "environment_steps": selected["environment_steps"],
            "canonical_screen_finishes": selected["canonical_finishes"],
            "actor_path": str(model_path),
            "actor_sha256": selected["actor_sha256"],
            "learner_checkpoint_sha256": selected["learner_checkpoint_sha256"],
            "screen_pointer": str(previous_pointer),
            "screen_pointer_sha256": sha256_file(previous_pointer),
            "previous_evaluation_metadata": predecessor,
            "diagnostic_confirmation": diagnostic,
            "confirmation_cells": _cell_count(protocol, "confirmation"),
        }
    if len(contracts) != 4:
        raise ValueError("exactly four matched actor confirmations must be bound")
    for seed in protocol["student_training"]["learner_seeds"]:
        if (
            contracts[f"rlpd-seed{seed}"]["canonical_screen_finishes"] < 1
            or contracts[f"rlpd-seed{seed}"]["canonical_screen_finishes"]
            < contracts[f"sac-seed{seed}"]["canonical_screen_finishes"]
        ):
            raise ValueError("frozen screen promotion gate failed; confirmation cannot start")
    return contracts


def _evaluation_command(protocol_path, run_root, contract, partition, output, workers, cpu_python):
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
        contract["screen_pointer"] if partition == "confirmation" else contract["confirmation_pointer"],
        "--python",
        cpu_python,
        "--workers",
        str(workers),
        "--evaluations-dir",
        str(ROOT / "evaluations"),
        "--output",
        str(output),
        "--model",
        contract["actor_path"],
    ]
    if partition == "confirmation" and contract["diagnostic_confirmation"]:
        command.append("--diagnostic-confirmation")
    return command


def _run_one_confirmation(protocol_path, protocol, run_root, contract, output, workers, cpu_python):
    command = _evaluation_command(
        protocol_path, run_root, contract, "confirmation", output, workers, cpu_python
    )
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = json.loads(output.read_text())
    evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError("confirmation receipt must bind exactly one candidate")
    candidate, summary = candidates[0], summaries[0]
    if candidate["archive_sha256"] != contract["actor_sha256"]:
        raise ValueError("confirmation evaluated a different actor than its screen-bound receipt")
    if summary.get("canonical_episodes") != _cell_count(protocol, "confirmation"):
        raise ValueError("confirmation canonical-cell count differs from frozen partition")
    return {
        "status": "complete",
        "partition": "confirmation",
        "diagnostic_only": contract["diagnostic_confirmation"],
        "pointer": str(output),
        "pointer_sha256": sha256_file(output),
        "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
        "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
        "actor_sha256": contract["actor_sha256"],
        "canonical_episodes": summary["canonical_episodes"],
        "canonical_finishes": finish_counts.get(candidate["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def preflight(protocol_path: Path, run_root: Path, workers: int):
    protocol_path = protocol_path.resolve()
    run_root = Path(run_root).resolve()
    protocol = read_protocol(protocol_path)
    if protocol["name"] != "pixel-rlpd-long-horizon-followup-v1" or run_root != (ROOT / RUN_ROOT_RELATIVE).resolve():
        raise ValueError("recovery scope must match the frozen V3 study/run root")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be positive")
    original_execution = json.loads((run_root / "execution.json").read_text())
    if (
        original_execution.get("screen_gate")
        or original_execution.get("confirmation_opened")
        or original_execution.get("blind_opened")
        or len(original_execution.get("stages", [])) != 3
        or original_execution["stages"][-1].get("name") != "screen"
    ):
        raise ValueError("original runner state does not establish a screen-complete, zero-confirmation recovery point")
    if original_execution.get("status") != "failed_execution":
        raise ValueError("recovery expects the documented pre-cell confirmation provenance failure")
    if original_execution.get("failure", {}).get("type") != "CalledProcessError":
        raise ValueError("unexpected original runner failure; inspect cells before recovery")
    if original_execution.get("stages", [])[-1].get("canonical_finishes") is None:
        raise ValueError("original screen completion receipt is incomplete")

    selection_path = run_root / "screen-selection-projections.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if (
        selection.get("protocol_sha256") != sha256_file(protocol_path)
        or selection.get("parent_screen_pointer") != str((run_root / "screen-evaluation.json").resolve())
        or selection.get("new_environment_interactions") != 0
        or selection.get("cell_replay") is not False
    ):
        raise ValueError("single-actor selection projections are not bound to the immutable v3 screen")
    if (ROOT / protocol["result_path"]).exists():
        raise FileExistsError("follow-up result already exists; do not overwrite a previous gate record")
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            pointer = run_root / f"confirmation-{arm}-seed{seed}.json"
            if pointer.exists():
                raise FileExistsError(f"confirmation pointer already exists; audit its cell consumption: {pointer}")
    if (run_root / "blind-finalist.json").exists():
        raise FileExistsError("blind pointer already exists; audit blind cell consumption before recovery")
    contracts = _candidate_contract(protocol, run_root, selection)
    cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
    commands = {}
    for key, contract in contracts.items():
        pointer = run_root / f"confirmation-{key}.json"
        if pointer.exists():
            raise FileExistsError(f"confirmation pointer already exists; do not rerun consumed cells: {pointer}")
        commands[key] = _evaluation_command(
            protocol_path, run_root, contract, "confirmation", pointer, workers, cpu_python
        )
    blind_finalist = selection["blind_finalist_preselected_from_screen"]
    finalist_key = f"rlpd-seed{blind_finalist['training_seed']}"
    if contracts[finalist_key]["actor_sha256"] != blind_finalist["actor_sha256"]:
        raise ValueError("preselected blind finalist does not match its screen-bound confirmation actor")
    blind_pointer = run_root / "blind-finalist.json"
    if blind_pointer.exists():
        raise FileExistsError("blind pointer already exists; inspect existing blind evaluation before any action")
    return {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "run_root": run_root,
        "selection": selection,
        "selection_sha256": sha256_file(selection_path),
        "contracts": contracts,
        "commands": commands,
        "blind_finalist_key": finalist_key,
        "cpu_python": cpu_python,
        "confirmation_cells_per_actor": _cell_count(protocol, "confirmation"),
        "blind_cells_for_finalist": _cell_count(protocol, "blind"),
    }


def execute(plan):
    run_root = plan["run_root"]
    recovery_path = run_root / "confirmation-recovery-execution.json"
    if recovery_path.exists():
        raise FileExistsError("confirmation recovery record exists; inspect it, never restart blindly")
    recovery = {
        "format": RECOVERY_FORMAT,
        "status": "running_confirmation",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": plan["protocol_sha256"],
        "screen_selection_path": "screen-selection-projections.json",
        "screen_selection_sha256": plan["selection_sha256"],
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "screen_cells_reexecuted": 0,
        "confirmation_cells_per_actor": plan["confirmation_cells_per_actor"],
        "confirmation_stage": "running",
        "confirmation_results": {},
        "confirmation_attempts": {},
        "blind_opened": False,
        "blind_status": "closed_until_all_confirmation_gates_pass",
        "commands": plan["commands"],
    }
    write_json(recovery_path, recovery, exclusive=True)
    all_confirmations_pass = True
    try:
        for key in ("rlpd-seed10", "rlpd-seed11", "sac-seed10", "sac-seed11"):
            contract = plan["contracts"][key]
            output = plan["run_root"] / f"confirmation-{key}.json"
            recovery["confirmation_attempts"][key] = {
                "status": "running",
                "cells": _cell_count(plan["protocol"], "confirmation"),
                "previous_screen_pointer": contract["screen_pointer"],
                "actor_sha256": contract["actor_sha256"],
            }
            write_json(recovery_path, recovery)
            result = _run_one_confirmation(
                Path(plan["protocol"]["_path"]),
                plan["protocol"],
                plan["run_root"],
                contract,
                output,
                2,
                plan["cpu_python"],
            )
            recovery["confirmation_attempts"][key]["status"] = "complete"
            recovery["confirmation_results"][key] = result
            write_json(recovery_path, recovery)
        recovery["confirmation_stage"] = "complete"
        byseed = {}
        for seed in (10, 11):
            treatment = recovery["confirmation_results"][f"rlpd-seed{seed}"]
            control = recovery["confirmation_results"][f"sac-seed{seed}"]
            byseed[str(seed)] = {
                "rlpd_finishes": treatment["canonical_finishes"],
                "sac_finishes": control["canonical_finishes"],
                "pass": bool(
                    treatment["eligible"]
                    and treatment["determinism_audited"]
                    and treatment["cpu_reload_matches"]
                    and treatment["operational_failures"] == 0
                    and control["eligible"]
                    and control["determinism_audited"]
                    and control["cpu_reload_matches"]
                    and control["operational_failures"] == 0
                    and treatment["canonical_finishes"] >= 1
                    and treatment["canonical_finishes"] > control["canonical_finishes"]
                ),
            }
        recovery["confirmation_seed_gates"] = byseed
        all_confirmations_pass = all(row["pass"] for row in byseed.values())
        recovery["confirmation_gate"] = "pass" if all_confirmations_pass else "stop_hold_failure"
        if not all_confirmations_pass:
            recovery["status"] = "confirmation_gate_stop_hold_failure"
            recovery["confirmation_stage"] = "complete_gate_failed"
            recovery["blind_status"] = "closed_after_confirmation_failure"
            recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(recovery_path, recovery)
            result = _result_from_recovery(plan, recovery)
            output = ROOT / plan["protocol"]["result_path"]
            write_json(output, result, exclusive=True)
            write_json(plan["run_root"] / "result.json", result, exclusive=True)
            return result

        finalist_key = plan["blind_finalist_key"]
        finalist = plan["contracts"][finalist_key]
        previous = Path(recovery["confirmation_results"][finalist_key]["pointer"])
        blind_pointer = plan["run_root"] / "blind-finalist.json"
        recovery["blind_status"] = "running_single_preselected_finalist"
        recovery["blind_opened"] = True
        recovery["status"] = "running_blind_finalist"
        recovery["commands"]["blind-finalist"] = _evaluation_command(
            Path(plan["protocol"]["_path"]),
            plan["run_root"],
            {**finalist, "confirmation_pointer": str(previous)},
            "blind",
            blind_pointer,
            2,
            plan["cpu_python"],
        )
        write_json(recovery_path, recovery)
        subprocess.run(recovery["commands"]["blind-finalist"], cwd=ROOT, check=True)
        receipt = json.loads(blind_pointer.read_text())
        evaluation_dir = Path(receipt["evaluation_dir"]).resolve()
        candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
        if len(candidates) != 1 or len(summaries) != 1 or candidates[0]["archive_sha256"] != finalist["actor_sha256"]:
            raise ValueError("blind result does not bind the preselected RLPD finalist")
        blind_result = {
            "pointer": str(blind_pointer),
            "pointer_sha256": sha256_file(blind_pointer),
            "evaluation_dir": str(evaluation_dir.relative_to(ROOT)),
            "manifest_sha256": sha256_file(evaluation_dir / "manifest.json"),
            "actor_sha256": finalist["actor_sha256"],
            "canonical_finishes": finish_counts.get(candidates[0]["candidate_id"], 0),
            "canonical_episodes": summaries[0]["canonical_episodes"],
            "eligible": summaries[0]["eligible"],
            "summary": summaries[0]["summary"],
        }
        recovery["blind_result"] = blind_result
        recovery["blind_status"] = "complete_final_internal_result"
        recovery["status"] = "completed_internal_blind"
        recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        result = _result_from_recovery(plan, recovery)
        output = ROOT / plan["protocol"]["result_path"]
        write_json(output, result, exclusive=True)
        write_json(plan["run_root"] / "result.json", result, exclusive=True)
        return result
    except BaseException as exc:
        if recovery.get("blind_opened"):
            recovery["status"] = "failed_blind_execution; exact cell consumption requires read-only audit"
        else:
            recovery["status"] = "failed_confirmation_execution; exact cell consumption requires read-only audit"
        recovery["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recovery["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(recovery_path, recovery)
        raise


def _result_from_recovery(plan, recovery):
    return {
        "format": "haic-rlpd-long-horizon-followup-result-v1",
        "study_id": plan["protocol"]["name"],
        "protocol_sha256": plan["protocol_sha256"],
        "prior_dataset_sha256": json.loads((plan["run_root"] / "prior-data" / "manifest.json").read_text())["dataset_sha256"],
        "screen_selection_path": str((plan["run_root"] / "screen-selection-projections.json").relative_to(ROOT)),
        "screen_selection_sha256": plan["selection_sha256"],
        "confirmation_recovery_path": str((plan["run_root"] / "confirmation-recovery-execution.json").relative_to(ROOT)),
        "screen_results": plan["selection"]["selected_by_run"],
        "screen_gate": "pass",
        "confirmation": recovery.get("confirmation_results"),
        "confirmation_seed_gates": recovery.get("confirmation_seed_gates"),
        "confirmation_gate": recovery.get("confirmation_gate"),
        "blind_finalist": plan["selection"]["blind_finalist_preselected_from_screen"],
        "blind": recovery.get("blind_result"),
        "confirmation_opened": bool(recovery.get("confirmation_results")),
        "blind_opened": bool(recovery.get("blind_opened")),
        "derivation_note": "The native custom evaluator requires one candidate in each preceding-partition receipt. V3 screen evaluated eight actors in one immutable receipt; selected summary rows and exact archived actors are projected into labeled one-actor lineage bundles from those existing rows, with parent pointer/summary/manifest hashes. No screen observations were rerun or modified.",
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "official_performance_claim": False,
    }


def main():
    args = parse_args()
    if args.preflight_only == args.execute_confirmation_and_gated_blind:
        raise ValueError("choose exactly one of --preflight-only or --execute-confirmation-and-gated-blind")
    plan = preflight(args.protocol, args.run_root, args.workers)
    confirmation_preflight = {
        key: {
            "actor_sha256": contract["actor_sha256"],
            "previous_screen_pointer": contract["screen_pointer"],
            "previous_evaluation_actor_sha256": contract["previous_evaluation_metadata"]["actor_sha256"],
            "diagnostic_confirmation": contract["diagnostic_confirmation"],
            "confirmation_cells": contract["confirmation_cells"],
        }
        for key, contract in plan["contracts"].items()
    }
    if args.preflight_only:
        print(json.dumps({
            "status": "preflight_passed_no_environment_interaction",
            "protocol_sha256": plan["protocol_sha256"],
            "screen_selection_sha256": plan["selection_sha256"],
            "confirmation_preflight": confirmation_preflight,
            "confirmation_cells_per_actor": plan["confirmation_cells_per_actor"],
            "blind_cells": plan["blind_cells_for_finalist"],
            "blind_finalist_key": plan["blind_finalist_key"],
        }, sort_keys=True), flush=True)
        return
    result = execute(plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
