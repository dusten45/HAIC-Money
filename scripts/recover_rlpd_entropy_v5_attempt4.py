"""Continue V5 after an audited no-dispatch error without replaying consumed cells."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, sha256_file, write_json
from scripts.rlpd_entropy_v5_common import STUDY_NAME, read_entropy_v5_protocol
from scripts.run_rlpd_pilot import _summarize_screen


RUN_ROOT = ROOT / "runs/20260925-pixel-rlpd-entropy-target-ablation-v5"
RECOVERY_PATH = RUN_ROOT / "entropy-recovery-attempt4.json"
PREFLIGHT_PATH = RUN_ROOT / "entropy-recovery-attempt4-preflight.json"
CONFIRM_KEYS = (
    "rlpd-author-target-seed50",
    "rlpd-author-target-seed51",
    "rlpd-positive-target-seed50",
    "rlpd-positive-target-seed51",
)
CONSUMED_KEY = "rlpd-author-target-seed50"
PENDING_KEYS = CONFIRM_KEYS[1:]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-three-remaining-confirmations-and-gated-blind", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(Path(path).read_text())


def _write_exclusive_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_if_absent_or_equal(path, value):
    path = Path(path)
    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"existing result differs from the audited recovery result: {path}")
        return
    try:
        _write_exclusive_json(path, value)
    except FileExistsError:
        if _read_json(path) != value:
            raise ValueError(f"concurrent result differs from the audited recovery result: {path}")


def _assert_actor_identity(actor, selected, arm, seed, protocol_sha256, dataset_sha256, key):
    if actor.get("archive_sha256") != selected["actor_sha256"] or actor.get("algorithm") != "rlpd":
        raise ValueError(f"screen actor metadata no longer matches its fixed selection: {key}")
    export = actor.get("export_metadata", {})
    contract = export.get("environment_contract", {})
    if (
        export.get("format") != "haic-rlpd-pixel-actor-v1"
        or export.get("protocol_sha256") != protocol_sha256
        or export.get("training_seed") != seed
        or export.get("environment_steps") != selected["environment_steps"]
        or contract.get("arm") != arm
        or contract.get("training_seed") != seed
        or contract.get("offline_dataset_sha256") != dataset_sha256
    ):
        raise ValueError(f"actor's embedded V5 arm/seed/data identity mismatches selection: {key}")


def _binding(protocol, protocol_path, run_root, selection, key, dataset_sha256):
    selected = selection["selected"][key]
    receipt = selection["confirmation_receipts"][key]
    arm, seed_text = key.rsplit("-seed", 1)
    seed = int(seed_text)
    if selected.get("arm") != arm or selected.get("training_seed") != seed:
        raise ValueError(f"screen actor metadata doesn't match its arm/seed key: {key}")
    actor_path = Path(selected["source_path"]).resolve()
    if not actor_path.is_file() or sha256_file(actor_path) != selected["actor_sha256"]:
        raise ValueError(f"screen-selected actor content hash changed: {key}")
    actor = candidate_metadata(actor_path, run_dir=run_root)
    _assert_actor_identity(
        actor,
        selected,
        arm,
        seed,
        sha256_file(protocol_path),
        dataset_sha256,
        key,
    )
    previous = Path(receipt["previous_screen_pointer"]).resolve()
    if (
        not previous.is_file()
        or sha256_file(previous) != receipt.get("previous_screen_pointer_sha256")
        or receipt.get("actor_sha256") != selected["actor_sha256"]
    ):
        raise ValueError(f"actor-specific screen predecessor changed: {key}")
    prior = previous_evaluation_metadata(
        previous,
        "confirmation",
        sha256_file(protocol_path),
        actor,
    )
    if prior.get("partition") != "screen":
        raise ValueError(f"actor-specific predecessor is not the frozen screen partition: {key}")
    if receipt.get("previous_evaluation_preflight", {}).get("actor_sha256") != selected["actor_sha256"]:
        raise ValueError(f"screen predecessor's recorded actor preflight changed: {key}")
    return {
        "key": key,
        "arm": arm,
        "training_seed": seed,
        "target_entropy": protocol["student_training"]["arm_specs"][arm]["target_entropy"],
        "actor_path": str(actor_path),
        "actor_sha256": selected["actor_sha256"],
        "learner_checkpoint_sha256": selected["learner_checkpoint_sha256"],
        "environment_steps": selected["environment_steps"],
        "screen_finishes": selected["canonical_finishes"],
        "previous_screen_pointer": str(previous),
        "previous_screen_pointer_sha256": sha256_file(previous),
    }


def _assert_prior_attempts(protocol_sha256, run_root, *, allow_finalized=False):
    execution = _read_json(run_root / "execution.json")
    recovery = _read_json(RECOVERY_PATH) if allow_finalized else None
    original_state = (
        execution.get("status") == "failed_entropy_ablation_stage"
        and execution.get("confirmation_opened") is False
        and execution.get("blind_opened") is False
    )
    finalized_state = (
        allow_finalized
        and recovery.get("status") in {"confirmation_gate_failed_blind_closed", "completed_internal_blind"}
        and execution.get("status") == recovery.get("status")
        and execution.get("confirmation_opened") is True
        and execution.get("blind_opened") is recovery.get("blind_opened")
    )
    if (
        execution.get("failure") != {"type": "KeyError", "message": "'projections'"}
        or not (original_state or finalized_state)
        or len(execution.get("stages", [])) != 3
        or execution["stages"][-1].get("name") != "screen"
        or execution["stages"][-1].get("passed") is not True
    ):
        raise ValueError("original V5 run doesn't prove the audited post-screen, zero-confirmation state")

    attempt1 = _read_json(run_root / "entropy-recovery.json")
    if (
        attempt1.get("protocol_sha256") != protocol_sha256
        or attempt1.get("status") != "failed_evaluator_after_dispatch_audit_cells_before_retry"
        or attempt1.get("failure") != {"type": "KeyError", "message": "'protocol_path'"}
        or attempt1.get("screen_cells_reexecuted") != 0
        or attempt1.get("confirmation_results") != {}
        or attempt1.get("blind_opened") is not False
    ):
        raise ValueError("attempt 1 isn't the audited no-dispatch protocol-path failure")

    attempt2 = _read_json(run_root / "entropy-recovery-attempt2.json")
    attempt2_actor = attempt2.get("confirmation_attempts", {}).get(CONSUMED_KEY, {})
    if (
        attempt2.get("protocol_sha256") != protocol_sha256
        or attempt2.get("status") != "failed_confirmation_or_blind; audit exact cell consumption before any retry"
        or attempt2.get("failure") != {"type": "KeyError", "message": "'target_entropy'"}
        or attempt2.get("screen_cells_reexecuted") != 0
        or attempt2.get("confirmation_results") != {}
        or attempt2.get("blind_opened") is not False
        or set(attempt2.get("confirmation_attempts", {})) != {CONSUMED_KEY}
        or attempt2_actor.get("status") != "running"
        or attempt2_actor.get("canonical_cells") != 32
    ):
        raise ValueError("attempt 2 doesn't isolate the one completed confirmation actor")

    attempt3 = _read_json(run_root / "entropy-recovery-attempt3.json")
    attempt3_actor = attempt3.get("confirmation_attempts", {}).get(CONSUMED_KEY, {})
    consumed_pointer = run_root / f"confirmation-{CONSUMED_KEY}.json"
    consumed_pointer_sha256 = sha256_file(consumed_pointer)
    attempt3_script = ROOT / "scripts/recover_rlpd_entropy_v5_attempt3.py"
    if (
        attempt3.get("protocol_sha256") != protocol_sha256
        or attempt3.get("status") != "failed_after_dispatch; audit exact consumed cells before any retry"
        or attempt3.get("failure") != {"type": "KeyError", "message": "'confirmation_cells'"}
        or attempt3.get("screen_cells_reexecuted") != 0
        or attempt3.get("blind_opened") is not False
        or attempt3.get("confirmation_cells") != 32
        or set(attempt3.get("confirmation_attempts", {})) != {CONSUMED_KEY}
        or set(attempt3.get("confirmation_results", {})) != {CONSUMED_KEY}
        or attempt3_actor.get("status") != "prior_completion_recorded; not rerun"
        or attempt3_actor.get("pointer_sha256") != consumed_pointer_sha256
        or attempt3["confirmation_results"][CONSUMED_KEY].get("pointer_sha256") != consumed_pointer_sha256
        or attempt3.get("recovery_script_sha256") != sha256_file(attempt3_script)
    ):
        raise ValueError("attempt 3 doesn't prove a no-dispatch failure after only the consumed actor")
    return attempt1


def _read_candidate_result(pointer_path, binding, partition, protocol_sha256, expected_cells):
    pointer = _read_json(pointer_path)
    if (
        pointer.get("protocol_sha256") != protocol_sha256
        or pointer.get("partition") != partition
        or pointer.get("diagnostic_only") is not False
    ):
        raise ValueError(f"{partition} receipt has the wrong frozen V5 identity: {binding['key']}")
    evaluation_dir = Path(pointer["evaluation_dir"]).resolve()
    candidates, summaries, finish_counts = _summarize_screen(evaluation_dir)
    if len(candidates) != 1 or len(summaries) != 1:
        raise ValueError(f"{partition} receipt must summarize exactly one actor: {binding['key']}")
    candidate, summary = candidates[0], summaries[0]
    if candidate.get("archive_sha256") != binding["actor_sha256"]:
        raise ValueError(f"{partition} receipt isn't bound to the fixed screen actor: {binding['key']}")
    if summary.get("canonical_episodes") != expected_cells:
        raise ValueError(f"{partition} canonical episode count differs from the frozen V5 contract")
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
        "canonical_finishes": finish_counts.get(candidate["candidate_id"], 0),
        "eligible": summary.get("eligible") is True,
        "determinism_audited": summary.get("determinism_audited") is True,
        "cpu_reload_matches": summary.get("cpu_reload_matches") is True,
        "operational_failures": summary.get("operational_failures"),
        "summary": summary.get("summary"),
    }


def _eval_command(plan, partition, key, output, previous):
    binding = plan["bindings"][key]
    cpu = plan["cpu_python"]
    return [
        cpu,
        "-m",
        "evaluate_policy",
        "--run-dir",
        str(plan["run_root"]),
        "--protocol-file",
        str(plan["protocol_path"]),
        "--partition",
        partition,
        "--previous-evaluation",
        str(previous),
        "--python",
        cpu,
        "--workers",
        str(plan["workers"]),
        "--evaluations-dir",
        str(ROOT / "evaluations"),
        "--output",
        str(output),
        "--model",
        binding["actor_path"],
    ]


def preflight(protocol_path, run_root, workers, *, finalizing=False):
    protocol_path, run_root = Path(protocol_path).resolve(), Path(run_root).resolve()
    protocol = read_entropy_v5_protocol(protocol_path)
    if protocol.get("name") != STUDY_NAME or run_root != RUN_ROOT.resolve():
        raise ValueError("recovery scope must match the frozen V5 protocol/run root")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be positive")
    protocol_sha256 = sha256_file(protocol_path)
    attempt1 = _assert_prior_attempts(protocol_sha256, run_root, allow_finalized=finalizing)

    selection_path = run_root / "screen-selection-projections.json"
    selection = _read_json(selection_path)
    selection_sha256 = sha256_file(selection_path)
    if attempt1.get("screen_selection_sha256") != selection_sha256:
        raise ValueError("screen selection differs from the selection frozen in recovery attempt 1")
    expected_keys = {
        f"{arm}-seed{seed}"
        for arm in protocol["student_training"]["arms"]
        for seed in protocol["student_training"]["learner_seeds"]
    }
    expected_gates = {
        arm: {str(seed): True for seed in protocol["student_training"]["learner_seeds"]}
        for arm in protocol["student_training"]["arms"]
    }
    if (
        selection.get("protocol_sha256") != protocol_sha256
        or selection.get("environment_interactions") != 0
        or selection.get("cells_replayed") is not False
        or selection.get("screen_gates") != expected_gates
        or set(selection.get("selected", {})) != expected_keys
        or set(selection.get("confirmation_receipts", {})) != expected_keys
    ):
        raise ValueError("screen selection doesn't match the four frozen V5 actors without replay")
    screen_pointer_path = run_root / "screen-evaluation.json"
    screen_pointer = _read_json(screen_pointer_path)
    if (
        selection.get("parent_screen_pointer") != str(screen_pointer_path)
        or selection.get("parent_screen_pointer_sha256") != sha256_file(screen_pointer_path)
        or screen_pointer.get("protocol_sha256") != protocol_sha256
        or screen_pointer.get("partition") != "screen"
        or screen_pointer.get("diagnostic_only") is not False
    ):
        raise ValueError("projected selection isn't bound to the frozen screen receipt")

    dataset_manifest = _read_json(run_root / "prior-data" / "manifest.json")
    dataset_sha256 = dataset_manifest.get("dataset_sha256")
    bindings = {
        key: _binding(protocol, protocol_path, run_root, selection, key, dataset_sha256)
        for key in sorted(expected_keys)
    }
    if not all(bindings[key]["screen_finishes"] >= 1 for key in CONFIRM_KEYS):
        raise ValueError("all four frozen screen selections must pass their minimum-finish gates")
    consumed_pointer = run_root / f"confirmation-{CONSUMED_KEY}.json"
    consumed = _read_candidate_result(
        consumed_pointer,
        bindings[CONSUMED_KEY],
        "confirmation",
        protocol_sha256,
        32,
    )
    attempt3_result = _read_json(run_root / "entropy-recovery-attempt3.json")["confirmation_results"][CONSUMED_KEY]
    if (
        consumed["pointer_sha256"] != attempt3_result.get("pointer_sha256")
        or consumed["canonical_finishes"] != attempt3_result.get("canonical_finishes")
        or consumed["actor_sha256"] != attempt3_result.get("actor_sha256")
        or not consumed["eligible"]
        or not consumed["determinism_audited"]
        or not consumed["cpu_reload_matches"]
        or consumed["operational_failures"] != 0
        or consumed["canonical_finishes"] < 1
    ):
        raise ValueError("existing seed-50 receipt no longer matches the exact consumed-cell audit")
    consumed["evidence_status"] = "prior completion verified from immutable receipt; not rerun"

    blind_pointer = run_root / "blind-finalist.json"
    result_path = ROOT / protocol["result_path"]
    recovery = None
    if finalizing:
        if not RECOVERY_PATH.is_file():
            raise FileNotFoundError("result-only finalization requires the completed attempt-4 recovery receipt")
        recovery = _read_json(RECOVERY_PATH)
        if (
            recovery.get("status") not in {"confirmation_gate_failed_blind_closed", "completed_internal_blind"}
            or set(recovery.get("confirmation_results", {})) != set(CONFIRM_KEYS)
        ):
            raise ValueError("attempt 4 lacks a complete confirmation result set for safe finalization")
        for key in PENDING_KEYS:
            if not (run_root / f"confirmation-{key}.json").is_file():
                raise FileNotFoundError(f"completed confirmation pointer is missing for finalization: {key}")
        if recovery.get("blind_opened"):
            if not blind_pointer.is_file():
                raise FileNotFoundError("attempt 4 records blind opened but its receipt is missing")
        elif blind_pointer.exists():
            raise ValueError("blind pointer exists although recovery says the blind gate stayed closed")
    else:
        if (
            RECOVERY_PATH.exists()
            or blind_pointer.exists()
            or result_path.exists()
            or (run_root / "result.json").exists()
        ):
            raise FileExistsError("attempt 4, blind, or V5 result already exists; inspect exact cells before proceeding")
        for key in PENDING_KEYS:
            if (run_root / f"confirmation-{key}.json").exists():
                raise FileExistsError(f"pending confirmation pointer already exists; never rerun its cells: {key}")

    cpu_python = protocol["runtime"]["cpu_evaluation_contract"]["verified_interpreter"]
    if not Path(cpu_python).is_file():
        raise FileNotFoundError(f"frozen CPU evaluator interpreter is missing: {cpu_python}")
    plan = {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha256,
        "run_root": run_root,
        "selection": selection,
        "selection_path": selection_path,
        "selection_sha256": selection_sha256,
        "bindings": bindings,
        "consumed_confirmation": consumed,
        "pending_confirmation_keys": list(PENDING_KEYS),
        "cpu_python": cpu_python,
        "workers": workers,
        "confirmation_cells": 32,
        "confirmation_episode_rows": 64,
        "blind_cells": 24,
        "blind_episode_rows": 48,
        "finalizing": finalizing,
        "existing_recovery": recovery,
    }
    plan["commands"] = {
        key: _eval_command(
            plan,
            "confirmation",
            key,
            run_root / f"confirmation-{key}.json",
            bindings[key]["previous_screen_pointer"],
        )
        for key in PENDING_KEYS
    }
    return plan


def _preflight_record(plan):
    return {
        "format": "haic-rlpd-entropy-v5-recovery-attempt4-preflight-v1",
        "study_id": STUDY_NAME,
        "protocol_sha256": plan["protocol_sha256"],
        "screen_selection_sha256": plan["selection_sha256"],
        "prior_attempts": {
            "attempt1": "no confirmation or blind cells dispatched",
            "attempt2": "one confirmation completed; receipt independently verified",
            "attempt3": "KeyError confirmation_cells before first pending evaluator dispatch",
        },
        "consumed_confirmation": plan["consumed_confirmation"],
        "pending_confirmation_actor_keys": plan["pending_confirmation_keys"],
        "canonical_cells_per_pending_actor": plan["confirmation_cells"],
        "episode_rows_per_pending_actor": plan["confirmation_episode_rows"],
        "blind_canonical_cells_if_paired_gate_passes": plan["blind_cells"],
        "blind_episode_rows_if_paired_gate_passes": plan["blind_episode_rows"],
        "screen_cells_replayed": 0,
        "blind_opened": False,
        "pending_actor_sha256": {
            key: plan["bindings"][key]["actor_sha256"]
            for key in plan["pending_confirmation_keys"]
        },
        "status": "preflight_passed_without_environment_interaction",
        "official_performance_claim": False,
    }


def _save_preflight(plan):
    record = _preflight_record(plan)
    if PREFLIGHT_PATH.exists():
        if _read_json(PREFLIGHT_PATH) != record:
            raise FileExistsError("attempt-4 preflight record differs; inspect before continuing")
    else:
        write_json(PREFLIGHT_PATH, record, exclusive=True)
    return record


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
        p > a for p, a in zip(positive_finishes, author_finishes)
    )
    if author_dominates == positive_dominates:
        return None
    return author if author_dominates else positive


def _blind_actor(protocol, winner, confirmations, bindings):
    candidates = []
    for seed in protocol["student_training"]["learner_seeds"]:
        key = f"{winner}-seed{seed}"
        result = confirmations[key]
        summary = result["summary"]
        lap = summary.get("avg_lap_time_ms")
        candidates.append((
            result["canonical_finishes"],
            summary.get("avg_progress", float("-inf")),
            -lap if lap is not None else float("-inf"),
            -bindings[key]["environment_steps"],
            -seed,
            key,
        ))
    return max(candidates, key=lambda row: row[:5])[-1]


def _build_result(plan, recovery):
    protocol, run_root = plan["protocol"], plan["run_root"]
    dataset = _read_json(run_root / "prior-data" / "manifest.json")
    selection = plan["selection"]
    screen_pointer = run_root / "screen-evaluation.json"
    return {
        "format": "haic-rlpd-entropy-target-ablation-result-v5",
        "study_id": protocol["name"],
        "protocol_path": str(plan["protocol_path"].relative_to(ROOT)),
        "protocol_sha256": plan["protocol_sha256"],
        "teacher_dataset": {
            "sha256": dataset["dataset_sha256"],
            "decisions_spent": dataset["decisions_spent"],
            "stored_decisions": dataset["stored_decisions"],
            "episodes": len(dataset["episodes"]),
            "distinct_finish_geometries": dataset["distinct_finish_geometries"],
        },
        "screen": {
            "pointer": str(screen_pointer.relative_to(ROOT)),
            "pointer_sha256": sha256_file(screen_pointer),
            "selection": str(plan["selection_path"].relative_to(ROOT)),
            "selection_sha256": plan["selection_sha256"],
            "selected_by_arm_seed": selection["selected"],
            "gates": selection["screen_gates"],
            "cells_replayed": 0,
        },
        "confirmation": recovery["confirmation_results"],
        "confirmation_target_winner": recovery.get("confirmation_target_winner"),
        "confirmation_gate": recovery.get("confirmation_gate"),
        "blind_opened": recovery.get("blind_opened", False),
        "blind_finalist": recovery.get("blind_finalist"),
        "blind": recovery.get("blind_result"),
        "recovery_execution": str(RECOVERY_PATH.relative_to(ROOT)),
        "recovery_script_sha256": recovery["recovery_script_sha256"],
        "confirmation_cells_replayed": 0,
        "official_performance_claim": False,
    }


def _save_result(plan, recovery):
    result = _build_result(plan, recovery)
    result_path = ROOT / plan["protocol"]["result_path"]
    _write_json_if_absent_or_equal(result_path, result)
    _write_json_if_absent_or_equal(plan["run_root"] / "result.json", result)
    execution_path = plan["run_root"] / "execution.json"
    execution = _read_json(execution_path)
    execution["status"] = recovery["status"]
    execution["confirmation_opened"] = True
    execution["blind_opened"] = recovery["blind_opened"]
    execution["recovery_path"] = str(RECOVERY_PATH.relative_to(ROOT))
    execution["result_path"] = str(result_path.relative_to(ROOT))
    execution["finished_at_utc"] = recovery["finished_at_utc"]
    write_json(execution_path, execution)
    return result


def _assert_result_row_matches(receipt, verified, key):
    fields = (
        "arm",
        "training_seed",
        "target_entropy",
        "actor_sha256",
        "pointer",
        "pointer_sha256",
        "evaluation_dir",
        "manifest_sha256",
        "canonical_episodes",
        "canonical_finishes",
        "eligible",
        "determinism_audited",
        "cpu_reload_matches",
        "operational_failures",
        "summary",
    )
    if any(receipt.get(field) != verified.get(field) for field in fields):
        raise ValueError(f"attempt-4 recovery row differs from its immutable evaluation receipt: {key}")


def finalize_existing(plan):
    if not plan["finalizing"]:
        raise ValueError("result-only finalization requires a finalizing preflight plan")
    if not PREFLIGHT_PATH.is_file() or _read_json(PREFLIGHT_PATH) != _preflight_record(plan):
        raise ValueError("matching original attempt-4 preflight is required for result-only finalization")
    recovery = plan["existing_recovery"]
    if (
        recovery.get("protocol_sha256") != plan["protocol_sha256"]
        or recovery.get("screen_selection_sha256") != plan["selection_sha256"]
        or recovery.get("screen_cells_reexecuted") != 0
        or set(recovery.get("confirmation_results", {})) != set(CONFIRM_KEYS)
    ):
        raise ValueError("completed recovery receipt does not bind the frozen V5 cells")
    if set(recovery.get("confirmation_attempts", {})) != set(CONFIRM_KEYS):
        raise ValueError("completed recovery attempt map is incomplete or contains unexpected actors")

    for key in CONFIRM_KEYS:
        attempt = recovery["confirmation_attempts"][key]
        if attempt.get("status") not in {"complete", "prior_completion_recorded; not rerun"}:
            raise ValueError(f"confirmation attempt is not complete: {key}")
        pointer = plan["run_root"] / f"confirmation-{key}.json"
        verified = _read_candidate_result(
            pointer,
            plan["bindings"][key],
            "confirmation",
            plan["protocol_sha256"],
            32,
        )
        _assert_result_row_matches(recovery["confirmation_results"][key], verified, key)

    winner = _confirmation_target_winner(plan["protocol"], recovery["confirmation_results"])
    expected_gate = "pass" if winner is not None else "stop_hold_failure"
    if (
        winner != recovery.get("confirmation_target_winner")
        or recovery.get("confirmation_gate") != expected_gate
    ):
        raise ValueError("stored confirmation gate differs from recomputed frozen paired gate")
    blind_pointer = plan["run_root"] / "blind-finalist.json"
    if winner is None:
        if recovery.get("blind_opened") is not False or blind_pointer.exists():
            raise ValueError("blind opened despite a failed paired gate")
        if recovery.get("status") != "confirmation_gate_failed_blind_closed":
            raise ValueError("failed paired gate has an unexpected recovery status")
    else:
        if recovery.get("status") != "completed_internal_blind" or recovery.get("blind_opened") is not True:
            raise ValueError("passing paired gate lacks a completed, recorded blind evaluation")
        finalist_key = _blind_actor(
            plan["protocol"],
            winner,
            recovery["confirmation_results"],
            plan["bindings"],
        )
        finalist = plan["bindings"][finalist_key]
        if (
            recovery.get("blind_finalist", {}).get("key") != finalist_key
            or recovery["blind_finalist"].get("actor_sha256") != finalist["actor_sha256"]
        ):
            raise ValueError("blind finalist differs from the frozen confirmation tie-break")
        blind = _read_candidate_result(
            blind_pointer,
            finalist,
            "blind",
            plan["protocol_sha256"],
            24,
        )
        _assert_result_row_matches(recovery.get("blind_result", {}), blind, "blind finalist")

    return _save_result(plan, recovery)


def execute(plan):
    if not PREFLIGHT_PATH.is_file() or _read_json(PREFLIGHT_PATH) != _preflight_record(plan):
        raise ValueError("matching zero-interaction attempt-4 preflight is required before execution")
    if RECOVERY_PATH.exists():
        raise FileExistsError("attempt-4 recovery record exists; audit exact cell consumption before any retry")

    recovery = {
        "format": "haic-rlpd-entropy-v5-recovery-attempt4-v1",
        "status": "running_three_remaining_confirmations",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": plan["protocol_sha256"],
        "screen_selection_sha256": plan["selection_sha256"],
        "recovery_script_sha256": sha256_file(Path(__file__)),
        "prior_attempt3_path": str((plan["run_root"] / "entropy-recovery-attempt3.json").relative_to(ROOT)),
        "screen_cells_reexecuted": 0,
        "confirmation_cells_per_pending_actor": plan["confirmation_cells"],
        "confirmation_episode_rows_per_pending_actor": plan["confirmation_episode_rows"],
        "confirmation_results": {CONSUMED_KEY: plan["consumed_confirmation"]},
        "confirmation_attempts": {
            CONSUMED_KEY: {
                "status": "prior_completion_recorded; not rerun",
                "pointer_sha256": plan["consumed_confirmation"]["pointer_sha256"],
            }
        },
        "blind_opened": False,
        "blind_cells": plan["blind_cells"],
    }
    _write_exclusive_json(RECOVERY_PATH, recovery)
    try:
        for key in PENDING_KEYS:
            output = plan["run_root"] / f"confirmation-{key}.json"
            if output.exists():
                raise FileExistsError(f"confirmation pointer exists; never repeat consumed cells: {key}")
            binding = plan["bindings"][key]
            command = plan["commands"][key]
            recovery["confirmation_attempts"][key] = {
                "status": "running",
                "actor_sha256": binding["actor_sha256"],
                "previous_screen_pointer_sha256": binding["previous_screen_pointer_sha256"],
                "canonical_cells": plan["confirmation_cells"],
                "episode_rows": plan["confirmation_episode_rows"],
                "command": command,
            }
            write_json(RECOVERY_PATH, recovery)
            subprocess.run(command, cwd=ROOT, check=True)
            result = _read_candidate_result(
                output,
                binding,
                "confirmation",
                plan["protocol_sha256"],
                plan["confirmation_cells"],
            )
            recovery["confirmation_results"][key] = result
            recovery["confirmation_attempts"][key]["status"] = "complete"
            write_json(RECOVERY_PATH, recovery)

        winner = _confirmation_target_winner(plan["protocol"], recovery["confirmation_results"])
        recovery["confirmation_target_winner"] = winner
        recovery["confirmation_gate"] = "pass" if winner is not None else "stop_hold_failure"
        recovery["confirmation_stage"] = "all four exact screen-selected actor receipts complete"
        if winner is None:
            recovery["status"] = "confirmation_gate_failed_blind_closed"
            recovery["blind_opened"] = False
            recovery["blind_status"] = "closed_after_failed_paired_target_gate"
            recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(RECOVERY_PATH, recovery)
            return _save_result(plan, recovery)

        finalist_key = _blind_actor(
            plan["protocol"],
            winner,
            recovery["confirmation_results"],
            plan["bindings"],
        )
        finalist = plan["bindings"][finalist_key]
        blind_output = plan["run_root"] / "blind-finalist.json"
        if blind_output.exists():
            raise FileExistsError("blind pointer already exists; never rerun blind cells")
        previous_confirmation = plan["run_root"] / f"confirmation-{finalist_key}.json"
        blind_command = _eval_command(
            plan,
            "blind",
            finalist_key,
            blind_output,
            previous_confirmation,
        )
        recovery["blind_opened"] = True
        recovery["blind_finalist"] = {
            "key": finalist_key,
            "arm": finalist["arm"],
            "training_seed": finalist["training_seed"],
            "target_entropy": finalist["target_entropy"],
            "actor_sha256": finalist["actor_sha256"],
        }
        recovery["blind_command"] = blind_command
        recovery["blind_status"] = "running_one_confirmation_selected_finalist"
        write_json(RECOVERY_PATH, recovery)
        subprocess.run(blind_command, cwd=ROOT, check=True)
        blind_result = _read_candidate_result(
            blind_output,
            finalist,
            "blind",
            plan["protocol_sha256"],
            plan["blind_cells"],
        )
        recovery["blind_result"] = blind_result
        recovery["blind_status"] = "complete_internal_final_result"
        recovery["status"] = "completed_internal_blind"
        recovery["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(RECOVERY_PATH, recovery)
        return _save_result(plan, recovery)
    except BaseException as exc:
        recovery["status"] = "failed_after_dispatch; audit exact consumed cells before any retry"
        recovery["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        recovery["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(RECOVERY_PATH, recovery)
        raise


def main():
    args = parse_args()
    if sum((args.preflight_only, args.execute_three_remaining_confirmations_and_gated_blind, args.finalize_only)) != 1:
        raise ValueError("choose exactly one preflight, execution, or result-only finalization mode")
    plan = preflight(args.protocol, args.run_root, args.workers, finalizing=args.finalize_only)
    if args.preflight_only:
        record = _save_preflight(plan)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        return
    if args.finalize_only:
        result = finalize_existing(plan)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return
    result = execute(plan)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
