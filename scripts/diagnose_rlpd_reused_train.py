"""Drive one already-consumed RLPD TRAIN cell, without a new-road G1 claim."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

from agent import Agent
from haic.algorithms.rlpd.g0_diagnostic import G0EventRules
from scripts.diagnose_rlpd_g0 import run_cell
from train import build_env


ROOT = Path(__file__).resolve().parents[1]
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_PROTOCOL_SHA = "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b"
G0_MANIFEST = "runs/20260926-rlpd-g0-completion-v1/manifest.json"
G0_MANIFEST_SHA = "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358"
G0_CELLS = "runs/20260926-rlpd-g0-completion-v1/cells.jsonl"
G0_CELLS_SHA = "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def located(relative: str, prefix: str) -> Path:
    value = Path(relative)
    if (value.is_absolute() or len(value.parts) < 2 or value.parts[0] != prefix
            or value.as_posix() != relative or any(part in (".", "..") for part in value.parts)):
        raise ValueError(f"expected a repository-relative {prefix}/ path: {relative}")
    current = ROOT
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked study path: {relative}")
    return current


def source_file(relative: str) -> Path:
    value = Path(relative)
    if (value.is_absolute() or not value.parts or value.as_posix() != relative
            or any(part in (".", "..") for part in value.parts)):
        raise ValueError(f"unsafe source path: {relative}")
    current = ROOT
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked source path: {relative}")
    return current


def pinned(relative: str, expected: str, prefix: str) -> Path:
    path = located(relative, prefix)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"missing or changed study evidence: {relative}")
    return path


def evidence() -> tuple[dict, dict, dict]:
    original = json.loads(pinned(G0_PROTOCOL, G0_PROTOCOL_SHA, "experiments").read_text())
    manifest = json.loads(pinned(G0_MANIFEST, G0_MANIFEST_SHA, "runs").read_text())
    cells = pinned(G0_CELLS, G0_CELLS_SHA, "runs").read_text().splitlines()
    if (original.get("format") != "haic-rlpd-g0-diagnostic-v1"
            or original.get("status") != "frozen" or manifest.get("cell_count") != 24
            or manifest.get("protocol_sha256") != G0_PROTOCOL_SHA
            or manifest.get("cells_sha256") != G0_CELLS_SHA or len(cells) != 24):
        raise ValueError("original G0 TRAIN evidence does not match its frozen chain")
    row = original["cells"][0]
    actor = original["actors"][1]
    prior = json.loads(cells[1])
    if (row != {"track_id": 1, "geometry_seed": 4272000001,
                "partition": "TRAIN", "obstacles": True}
            or prior.get("track_id") != row["track_id"]
            or prior.get("geometry_seed") != row["geometry_seed"]
            or prior.get("actor_id") != actor["id"]
            or prior.get("actor_sha256") != actor["sha256"]):
        raise ValueError("selected cell/actor were not consumed together in G0")
    pinned(actor["path"], actor["sha256"], "runs")
    return original, row, actor


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze(protocol_path: str) -> None:
    output = located(protocol_path, "experiments")
    if output.exists():
        raise FileExistsError(output)
    original, row, actor = evidence()
    paths = set(original["source_hashes"]) | {"scripts/diagnose_rlpd_reused_train.py"}
    source_hashes = {relative: sha256(source_file(relative))
                     for relative in sorted(paths)}
    protocol = {
        "format": "haic-rlpd-reused-train-single-cell-v1", "status": "frozen",
        "role": "already-consumed-TRAIN-engineering-diagnostic-not-G1-coverage",
        "g0_protocol_sha256": G0_PROTOCOL_SHA, "g0_manifest_sha256": G0_MANIFEST_SHA,
        "g0_cells_sha256": G0_CELLS_SHA, "cell": row, "actor": actor,
        "event_rules": original["event_rules"],
        "centerline_far_threshold_m": original["centerline_far_threshold_m"],
        "source_hashes": source_hashes,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "device": "cpu"},
        "execution": {"policy_updates": 0, "environment_resets": 1,
                      "max_decisions_per_episode": original["event_rules"]["max_decisions"],
                      "core_hour_cap": None, "new_road_coverage": False},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, protocol)
    print(json.dumps({"protocol": protocol_path, "sha256": sha256(output)}), flush=True)


def drive(protocol_path: str, expected_sha: str, run_dir: str) -> None:
    frozen = json.loads(pinned(protocol_path, expected_sha, "experiments").read_text())
    original, row, actor_info = evidence()
    if (frozen.get("format") != "haic-rlpd-reused-train-single-cell-v1"
            or frozen.get("status") != "frozen"
            or frozen.get("g0_protocol_sha256") != G0_PROTOCOL_SHA
            or frozen.get("g0_manifest_sha256") != G0_MANIFEST_SHA
            or frozen.get("g0_cells_sha256") != G0_CELLS_SHA
            or frozen.get("cell") != row or frozen.get("actor") != actor_info
            or frozen.get("event_rules") != original["event_rules"]
            or frozen.get("centerline_far_threshold_m") != original["centerline_far_threshold_m"]
            or frozen.get("execution") != {"policy_updates": 0, "environment_resets": 1,
                                           "max_decisions_per_episode": 2000,
                                           "core_hour_cap": None, "new_road_coverage": False}
            or frozen.get("runtime") != {"python": platform.python_version(),
                                         "torch": torch.__version__, "numpy": np.__version__,
                                         "device": "cpu"}):
        raise ValueError("reused-TRAIN run differs from the frozen single-cell contract")
    required = set(original["source_hashes"]) | {"scripts/diagnose_rlpd_reused_train.py"}
    hashes = frozen.get("source_hashes")
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError("source closure differs from the frozen run")
    for relative, expected in hashes.items():
        source = source_file(relative)
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"executable source drifted: {relative}")
    output = located(run_dir, "runs")
    output.mkdir(parents=False, exist_ok=False)
    for relative in hashes:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if sha256(destination) != hashes[relative]:
            raise ValueError(f"source snapshot failed: {relative}")
    shutil.copyfile(ROOT / protocol_path, output / "study_protocol.json")
    attempt = {"format": "haic-rlpd-reused-train-attempt-v1", "status": "attempting",
               "partition": "TRAIN", "reused_prior_cell": True, "cell": row,
               "actor_id": actor_info["id"], "actor_sha256": actor_info["sha256"],
               "protocol_sha256": expected_sha,
               "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "attempt.json", attempt)
    print(json.dumps({"status": "attempt_recorded", "run_dir": run_dir, "cell": row}), flush=True)
    state: dict = {}
    try:
        actor_path = pinned(actor_info["path"], actor_info["sha256"], "runs")
        actor = Agent(model_path=str(actor_path))
        if sha256(pinned(protocol_path, expected_sha, "experiments")) != expected_sha:
            raise ValueError("study protocol changed before reset")
        result, trace = run_cell(
            row=row, actor=actor, actor_info=actor_info,
            actor_payload=actor_path.read_bytes(), rules=G0EventRules(**original["event_rules"]),
            centerline_far_threshold_m=original["centerline_far_threshold_m"],
            env_factory=build_env, attempt=state,
        )
        trace_path = output / "trace.npz"
        with trace_path.open("xb") as stream:
            np.savez_compressed(stream, **trace)
        result.update({"format": "haic-rlpd-reused-train-cell-v1",
                       "role": "reused-TRAIN-engineering-only", "protocol_sha256": expected_sha,
                       "trace_path": "trace.npz", "trace_sha256": sha256(trace_path),
                       "collection_status": "complete"})
        write_json(output / "result.json", result)
        print(json.dumps({"status": "complete", "outcome": result["summary"]["outcome"],
                          "steps": result["steps"], "trace_sha256": result["trace_sha256"]}), flush=True)
    except BaseException as exc:
        partial = state.get("partial_arrays")
        receipt = {"format": "haic-rlpd-reused-train-failure-v1",
                   "status": "collection_censored", "phase": state.get("phase", "pre-reset"),
                   "decisions_completed": state.get("decisions_completed", 0),
                   "decision_calls": state.get("decision_calls", 0),
                   "raw_counters": state.get("raw_counters", {}),
                   "error_type": type(exc).__name__, "error": str(exc),
                   "protocol_sha256": expected_sha}
        if partial is not None:
            trace_path = output / "partial-trace.npz"
            arrays = {key: np.asarray(values) for key, values in partial.items()}
            if "initial_stack" in state:
                arrays["initial_stack"] = state["initial_stack"]
            with trace_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            receipt["partial_trace_sha256"] = sha256(trace_path)
        write_json(output / "failure.json", receipt)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--protocol", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--protocol", required=True)
    run_parser.add_argument("--protocol-sha256", required=True)
    run_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze(args.protocol)
    else:
        drive(args.protocol, args.protocol_sha256, args.run_dir)


if __name__ == "__main__":
    main()
