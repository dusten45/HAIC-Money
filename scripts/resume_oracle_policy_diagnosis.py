"""Resume only missing oracle cells from an interrupted diagnosis run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from scripts import diagnose_oracle_policy_failures as diagnosis


ROOT = diagnosis.ROOT
OUTPUT_RELATIVE = "runs/oracle-policy-diagnosis-v1"
OUTPUT = ROOT / OUTPUT_RELATIVE
PROTOCOL_RELATIVE = "experiments/oracle-policy-diagnosis-v1-continuation.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, value: Any, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode) as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_trace(output: Path, summary: dict[str, Any]) -> dict[str, np.ndarray]:
    trace_path = (output / summary["trace_path"]).resolve()
    if not trace_path.is_relative_to(output.resolve()):
        raise diagnosis.DiagnosisError("partial trace escapes the run directory")
    if _sha(trace_path) != summary["trace_sha256"]:
        raise diagnosis.DiagnosisError(f"partial trace hash mismatch: {trace_path}")
    with np.load(trace_path, allow_pickle=False) as archived:
        return {name: archived[name] for name in archived.files}


def _policy_key(summary: dict[str, Any]) -> tuple[str, int, int]:
    return summary["policy_id"], int(summary["track_id"]), int(summary["geometry_seed"])


def _oracle_key(summary: dict[str, Any]) -> tuple[int, int, int]:
    return int(summary["track_id"]), int(summary["geometry_seed"]), int(summary["max_steps"])


def _read_partial(output: Path, cases: list[dict[str, Any]], protocol: dict[str, Any]):
    metadata = diagnosis._read_json(output / "metadata.json")
    parent_protocol = ROOT / "experiments/oracle-policy-diagnosis-v1.json"
    if metadata.get("protocol_sha256") != _sha(parent_protocol):
        raise diagnosis.DiagnosisError("partial run was not created from the frozen parent protocol")
    if metadata.get("runner_sha256") != protocol["source_sha256"]["scripts/diagnose_oracle_policy_failures.py"]:
        raise diagnosis.DiagnosisError("partial run used a different parent collector")

    rows = _jsonl(output / "rollouts.jsonl")
    policies: dict[tuple[str, int, int], dict[str, Any]] = {}
    oracles: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        summary = row["summary"]
        if row["kind"] == "policy":
            key = _policy_key(summary)
            if key in policies:
                raise diagnosis.DiagnosisError(f"duplicate partial policy rollout row: {key}")
            policies[key] = {"summary": summary, "trace": _load_trace(output, summary)}
        elif row["kind"] == "oracle":
            key = _oracle_key(summary)
            if key in oracles:
                raise diagnosis.DiagnosisError(f"duplicate partial oracle rollout row: {key}")
            oracles[key] = {"summary": summary, "trace": _load_trace(output, summary)}
        else:
            raise diagnosis.DiagnosisError(f"unknown partial rollout kind: {row['kind']}")

    expected_policies = {
        (case["policy_id"], case["track_id"], case["geometry_seed"]): case for case in cases
    }
    if policies.keys() != expected_policies.keys():
        raise diagnosis.DiagnosisError(
            f"partial policy rows differ from the 45-case cohort: {len(policies)}"
        )
    expected_cells = {
        (case["track_id"], case["geometry_seed"], case["max_steps"]): case for case in cases
    }
    if not oracles.keys() <= expected_cells.keys():
        raise diagnosis.DiagnosisError("partial oracle rows contain an unselected cell")
    if not oracles:
        raise diagnosis.DiagnosisError("no completed oracle receipts to resume from")
    return policies, oracles, expected_cells


def _load_protocol(output: Path) -> dict[str, Any]:
    protocol_path = ROOT / PROTOCOL_RELATIVE
    protocol = diagnosis._read_json(protocol_path)
    parent_protocol_path = ROOT / "experiments/oracle-policy-diagnosis-v1.json"
    parent_protocol = diagnosis._read_json(parent_protocol_path)
    diagnosis.verify_source_hashes(parent_protocol)
    if protocol.get("parent_protocol_sha256") != _sha(parent_protocol_path):
        raise diagnosis.DiagnosisError("parent protocol differs from the continuation freeze")
    if protocol.get("parent_runner_sha256") != _sha(ROOT / "scripts/diagnose_oracle_policy_failures.py"):
        raise diagnosis.DiagnosisError("parent runner differs from the continuation freeze")
    if protocol.get("bootstrap_rollouts_sha256") != _sha(output / "rollouts.jsonl"):
        marker = output / "continuation.json"
        if not marker.exists():
            raise diagnosis.DiagnosisError("bootstrap progress ledger differs from the frozen partial run")
    if protocol.get("runner_sha256") != _sha(Path(__file__).resolve()):
        raise diagnosis.DiagnosisError("continuation runner differs from its frozen protocol")
    for relative, expected in protocol.get("source_sha256", {}).items():
        if _sha(ROOT / relative) != expected:
            raise diagnosis.DiagnosisError(f"continuation source hash mismatch: {relative}")
    return protocol


def _analysis_result(output: Path, policies, oracles, source_hashes, protocol) -> dict[str, Any]:
    comparisons = []
    for key, policy in sorted(policies.items()):
        summary = policy["summary"]
        oracle_key = (int(summary["track_id"]), int(summary["geometry_seed"]), int(summary["max_steps"]))
        oracle = oracles[oracle_key]
        comparisons.append(diagnosis.analyze_pair(
            summary, policy["trace"], oracle["summary"], oracle["trace"],
        ))
    per_study = {}
    for study in ("drq-r6", "rlpd-g0"):
        rows = [row for row in comparisons if row["study"] == study]
        per_study[study] = {
            "episodes": len(rows),
            "historical_failures": sum(not row["original_finished"] for row in rows),
            "current_replay_failures": sum(not row["replay_finished"] for row in rows),
            "oracle_finishes_on_matched_cells": sum(row["oracle_finished"] for row in rows),
            "archived_outcome_changes": sum(row["replay_outcome_changed_from_archived_episode"] for row in rows),
            "taxonomy_current_failures": dict(sorted(Counter(
                row["first_failure_precursor"]["mode"] for row in rows
                if not row["replay_finished"]
            ).items())),
        }
    taxonomy = Counter(row["first_failure_precursor"]["mode"] for row in comparisons
                       if not row["replay_finished"])
    output_files = {
        entry["summary"]["trace_path"]: entry["summary"]["trace_sha256"]
        for entry in [*policies.values(), *oracles.values()]
    }
    metadata = diagnosis._read_json(output / "metadata.json")
    continuation = diagnosis._read_json(output / "continuation.json")
    parent_protocol = diagnosis._read_json(ROOT / "experiments/oracle-policy-diagnosis-v1.json")
    result = {
        "format": "haic-oracle-policy-diagnosis-result-v1",
        "status": "complete_descriptive_consumed_train_diagnostic",
        "diagnostic_only": True,
        "official_score": False,
        "fresh_generalization_claim": False,
        "policy_episode_count": len(comparisons),
        "historical_failure_count": sum(not row["original_finished"] for row in comparisons),
        "current_replay_failure_count": sum(not row["replay_finished"] for row in comparisons),
        "unique_oracle_cells": len(oracles),
        "completed_policy_decisions": sum(entry["summary"]["steps"] for entry in policies.values()),
        "completed_oracle_decisions": sum(entry["summary"]["steps"] for entry in oracles.values()),
        "per_study": per_study,
        "first_precursor_taxonomy": dict(sorted(taxonomy.items())),
        "cohort_contract": metadata["cohort_contract"],
        "analysis_contract": metadata["analysis_contract"],
        "source_hashes": source_hashes,
        "frozen_source_sha256": parent_protocol["source_sha256"],
        "parent_protocol_sha256": metadata["protocol_sha256"],
        "parent_runner_sha256": metadata["runner_sha256"],
        "continuation_protocol_sha256": _sha(ROOT / PROTOCOL_RELATIVE),
        "continuation_runner_sha256": continuation["runner_sha256"],
        "continuation_completed_oracles": continuation["completed_oracle_count"],
        "rollout_files_sha256": output_files,
        "limit_caveat": continuation["initial_phase_caveat"],
        "limitations": [
            "All diagnostic geometries are previously consumed track-1 TRAIN or TRAIN-DIAGNOSTIC cells.",
            "DrQ r6 source archives lack pixels/dense pose; current-runtime frozen-actor reruns collected these signals.",
            "Some cross-runtime rerun outcomes changed despite near-identical first action and matching reset image/road hash; archived failures and current replays are counted separately.",
            "Separate closed-loop rollouts share reset observation and road coordinates, not simulator hidden state after their first different actions.",
            "Threshold taxonomy labels the earliest measured precursor and is not proof of causality.",
            "The DrQ and RLPD cohorts use different geometry seed pools and are reported stratified, not as a matched policy comparison.",
            "Oracle privileged state and outputs are diagnostic only and are not submission behavior or official performance.",
        ],
        "comparisons": comparisons,
    }
    diagnosis._write_json(output / "result.json", result)
    experiment_result_path = ROOT / "experiments/oracle-policy-diagnosis-v1-result.json"
    if experiment_result_path.exists():
        raise diagnosis.DiagnosisError("experiment result already exists; refusing to overwrite")
    diagnosis._write_json(experiment_result_path, result)
    return result


def resume(output: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    if output.resolve() != OUTPUT.resolve() or not output.is_dir():
        raise diagnosis.DiagnosisError("continuation is restricted to the frozen partial run directory")
    protocol = _load_protocol(output)
    cases, source_hashes = diagnosis.load_cases()
    policies, oracles, expected_cells = _read_partial(output, cases, protocol)
    marker_path = output / "continuation.json"
    if marker_path.exists():
        marker = diagnosis._read_json(marker_path)
        if marker.get("runner_sha256") != protocol["runner_sha256"]:
            raise diagnosis.DiagnosisError("partial continuation was made by another runner")
        if (marker.get("continuation_protocol_sha256") != _sha(ROOT / PROTOCOL_RELATIVE)
                or marker.get("bootstrap_rollouts_sha256") != protocol["bootstrap_rollouts_sha256"]):
            raise diagnosis.DiagnosisError("partial continuation marker differs from its frozen protocol")
    else:
        if len(policies) != 45 or len(oracles) != 7:
            raise diagnosis.DiagnosisError("frozen bootstrap must contain 45 policies and seven oracle receipts")
        marker = {
            "format": "haic-oracle-policy-diagnosis-continuation-v1",
            "continuation_protocol_sha256": _sha(ROOT / PROTOCOL_RELATIVE),
            "runner_sha256": _sha(Path(__file__).resolve()),
            "bootstrap_rollouts_sha256": _sha(output / "rollouts.jsonl"),
            "bootstrap_policy_count": len(policies),
            "bootstrap_oracle_count": len(oracles),
            "phase": "resuming_missing_oracle_cells",
            "initial_phase_caveat": "parent invocation reached its 1200-second command timeout after saving all 45 policy rollouts and seven oracle rollouts; no policy rollout is repeated",
        }
        if marker["bootstrap_rollouts_sha256"] != protocol["bootstrap_rollouts_sha256"]:
            raise diagnosis.DiagnosisError("initial progress ledger differs from the continuation freeze")
        _write(marker_path, marker, exclusive=True)

    if preflight_only:
        return {
            "status": "continuation_preflight_passed_no_environment_resets",
            "saved_policy_rollouts": len(policies),
            "saved_oracle_rollouts": len(oracles),
            "oracle_rollouts_remaining": len(expected_cells) - len(oracles),
        }
    if (output / "result.json").exists():
        raise diagnosis.DiagnosisError("diagnosis result already exists; refusing a second continuation")
    started = time.monotonic()
    remaining = sorted(set(expected_cells) - set(oracles))
    for index, key in enumerate(remaining, 1):
        source_case = expected_cells[key]
        case = {
            "track_id": source_case["track_id"],
            "geometry_seed": source_case["geometry_seed"],
            "max_steps": source_case["max_steps"],
            "source_road_centerline_sha256": source_case.get("source_road_centerline_sha256"),
        }
        summary, trace = diagnosis._collect_episode(case, "oracle")
        name = f"oracle-track{case['track_id']}-seed{case['geometry_seed']}-cap{case['max_steps']}"
        trace_path = output / "oracle" / f"{name}.npz"
        summary["trace_path"] = trace_path.relative_to(output).as_posix()
        summary["trace_sha256"] = diagnosis._write_trace(trace_path, trace)
        _append(output / "rollouts.jsonl", {"kind": "oracle", "summary": summary})
        oracles[key] = {"summary": summary, "trace": trace}
        marker["completed_oracle_count"] = len(oracles)
        marker["last_completed_cell"] = {"track_id": key[0], "geometry_seed": key[1], "max_steps": key[2]}
        _write(marker_path, marker)
        print(f"oracle-resume {index:02d}/{len(remaining)} {name}: "
              f"finish={summary['finished']} steps={summary['steps']}", flush=True)

    if len(oracles) != len(expected_cells):
        raise diagnosis.DiagnosisError("continuation ended without a complete oracle-cell census")
    marker["phase"] = "complete"
    marker["completed_oracle_count"] = len(oracles)
    marker["continuation_wall_seconds"] = time.monotonic() - started
    _write(marker_path, marker)
    result = _analysis_result(output, policies, oracles, source_hashes, protocol)
    return {
        "status": result["status"],
        "saved_policy_rollouts": len(policies),
        "saved_oracle_rollouts": len(oracles),
        "historical_failures": result["historical_failure_count"],
        "current_replay_failures": result["current_replay_failure_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        result = resume(output, preflight_only=args.preflight_only)
    except Exception as error:
        print(f"continuation failed closed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
