"""Compare a camera corridor candidate on exactly 55 consumed development cells.

This tool never opens confirmation or blind partitions, freezes a finalist, or
claims fresh generalization. Every new episode uses the actual Agent route in
the frozen generalization worker. The completed ego screen's control receipts
are reused only after exact source and environment lineage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import compare_reused_bare_candidate as prior
from tools import evaluate_bare_generalization as fresh

ARMS = ("baseline", "candidate")
CLASSES = {"baseline": "_CompoundClearingBrakeCarryController",
           "candidate": "_FeasibleCorridorObstacleController"}
MARGIN_SCREEN_PROTOCOL = ROOT / "experiments" / "observed-margin-generalization-v1.json"
MARGIN_SCREEN_RUN = ROOT / ".haic-artifacts" / "observed-margin-generalization-v1" / "run"
EGO_SCREEN_PROTOCOL = ROOT / "experiments" / "observed-ego-side-switch-generalization-v1.json"
EGO_SCREEN_RUN = ROOT / ".haic-artifacts" / "observed-ego-side-switch-generalization-v1" / "run"
MECHANISM_CELLS = (
    (1, 3857792434), (3, 3857792434), (1, 3892761381), (1, 42),
    (2, 644062), (4, 2951861974), (1, 17), (3, 2951861974),
    (1, 1190129265), (2, 4089604952),
)


def _screen_cells(protocol: dict, name: str, geometry_count: int) -> tuple[tuple[int, int], ...]:
    if not isinstance(protocol, dict) or protocol.get("name") != name:
        raise ValueError(f"wrong consumed screen protocol: {name}")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError(f"{name} needs the exact three frozen partitions")
    screen = partitions["screen"]
    tracks, seeds = screen.get("track_ids"), screen.get("seeds")
    if (tracks != [1, 2, 3, 4] or not isinstance(seeds, list)
            or len(seeds) != geometry_count or len(set(seeds)) != geometry_count
            or any(type(seed) is not int or not 0 <= seed <= fresh.MAX_SEED for seed in seeds)):
        raise ValueError(f"{name} screen must be four IDs by {geometry_count} distinct seeds")
    cells = tuple((track, seed) for track in tracks for seed in seeds)
    if len(set(cells)) != 4 * geometry_count:
        raise ValueError(f"{name} screen has duplicate cells")
    return cells


def margin_screen_cells(protocol: dict) -> tuple[tuple[int, int], ...]:
    return _screen_cells(protocol, "observed-margin-generalization-v1", 4)


def ego_screen_cells(protocol: dict) -> tuple[tuple[int, int], ...]:
    return _screen_cells(protocol, "observed-ego-side-switch-generalization-v1", 8)


def expected_development_cells(margin: dict, ego: dict) -> tuple[tuple[int, int], ...]:
    cells = prior.LEGACY_CELLS + margin_screen_cells(margin) + ego_screen_cells(ego)
    seeds = [seed for _, seed in prior.LEGACY_CELLS]
    seeds += margin["partitions"]["screen"]["seeds"]
    seeds += ego["partitions"]["screen"]["seeds"]
    if len(cells) != 55 or len(set(cells)) != 55 or len(seeds) != len(set(seeds)):
        raise ValueError("consumed development geometry seeds overlap")
    return cells


def validate_protocol(protocol: dict, margin: dict, ego: dict, margin_sha256: str,
                      ego_sha256: str, sealed_seeds: set[int]) -> tuple[tuple[int, int], ...]:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("development protocol schema_version must be 1")
    if protocol.get("name") != "feasible-corridor-dev-v1" or protocol.get("status") != "PREREGISTERED":
        raise ValueError("wrong or unregistered corridor development protocol")
    if protocol.get("evidence_scope") != "CONSUMED_DEVELOPMENT_ONLY":
        raise ValueError("evidence_scope must be CONSUMED_DEVELOPMENT_ONLY")
    if protocol.get("max_steps") != 2000 or protocol.get("frame_skip") != 4:
        raise ValueError("official evaluation requires max_steps 2000 and frame_skip 4")
    for prefix, screen_hash in (("margin", margin_sha256), ("ego", ego_sha256)):
        for suffix in ("protocol", "freeze", "summary", "receipts"):
            fresh._sha256_string(protocol.get(f"{prefix}_screen_{suffix}_sha256"),
                                 f"{prefix}_screen_{suffix}_sha256")
        if protocol[f"{prefix}_screen_protocol_sha256"] != screen_hash:
            raise ValueError(f"{prefix} screen protocol hash mismatch")
    if protocol.get("controller_classes") != CLASSES:
        raise ValueError("unexpected actual Agent routes")
    source_hashes = protocol.get("agent_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(ARMS):
        raise ValueError("both agent source hashes are required")
    for arm in ARMS:
        fresh._sha256_string(source_hashes[arm], f"agent_sha256.{arm}")
    if source_hashes["baseline"] != ego.get("control_agent_sha256"):
        raise ValueError("baseline must be current ego screen's active control source")
    fresh._sha256_string(protocol.get("model_sha256"), "model_sha256")
    if protocol["model_sha256"] != margin.get("model_sha256") or protocol["model_sha256"] != ego.get("model_sha256"):
        raise ValueError("screen and development model hashes differ")
    helpers = protocol.get("runtime_helper_sha256")
    if not isinstance(helpers, dict) or set(helpers) != {"action_smoothing.py", "action_representation.py"}:
        raise ValueError("both runtime helper hashes are required")
    for name, value in helpers.items():
        fresh._sha256_string(value, name)
    if helpers != margin.get("runtime_helper_sha256") or helpers != ego.get("runtime_helper_sha256"):
        raise ValueError("screen and development helper hashes differ")
    cells = expected_development_cells(margin, ego)
    if protocol.get("development_cells") != [list(cell) for cell in cells]:
        raise ValueError("development_cells must be the exact 55 consumed cells in order")
    if protocol.get("mechanism_cells") != [list(cell) for cell in MECHANISM_CELLS]:
        raise ValueError("mechanism_cells must match the fixed ten-cell triage")
    if not set(MECHANISM_CELLS).issubset(cells):
        raise ValueError("mechanism cells are outside the consumed development grid")
    overlap = {seed for _, seed in cells} & sealed_seeds
    if overlap:
        raise ValueError(f"confirmation/blind geometry seed requested: {sorted(overlap)}")
    gates = protocol.get("gates")
    if (not isinstance(gates, dict) or gates.get("fresh_confirmation") != "SEALED"
            or gates.get("blind") != "SEALED" or gates.get("sota_promotion") is not False):
        raise ValueError("development diagnosis cannot open confirmation/blind or SOTA promotion")
    return cells


def validate_output_root(root: Path) -> Path:
    root = root.resolve()
    artifacts = (ROOT / ".haic-artifacts").resolve()
    if root == artifacts or not root.is_relative_to(artifacts):
        raise ValueError("output root must be a child of .haic-artifacts")
    return root


def _receipt_manifest(protocol: dict, root: Path) -> str:
    entries = {}
    for track, seed, repeat in fresh.expected_cells(protocol, "screen"):
        for arm in fresh.ARMS:
            path = fresh.cell_path(root, "screen", arm, track, seed, repeat)
            if not path.is_file():
                raise ValueError(f"missing consumed screen receipt: {path}")
            entries[str(path.relative_to(root)).replace("\\", "/")] = fresh.digest(path)
    return hashlib.sha256(fresh._canonical(entries)).hexdigest()


def verify_rejected_screen(protocol: dict, root: Path, protocol_sha256: str,
                           reference: dict) -> int:
    """Fail closed unless every frozen screen receipt reproduces its REJECT summary."""
    if not (root / "freeze.json").is_file() or not (root / "screen-summary.json").is_file():
        raise ValueError("completed consumed screen freeze or summary is missing")
    identity = fresh._read_json(root / "freeze.json")
    if identity.get("protocol_sha256") != protocol_sha256:
        raise ValueError("consumed screen protocol hash mismatch")
    if identity.get("source_sha256") != {
        "control": protocol.get("control_agent_sha256"),
        "candidate": protocol.get("candidate_agent_sha256"),
    }:
        raise ValueError("consumed screen source hash mismatch")
    for key, value in reference.items():
        if identity.get(key) != value:
            raise ValueError(f"consumed screen {key} lineage mismatch")
    rows = []
    for track, seed, repeat in fresh.expected_cells(protocol, "screen"):
        for arm in fresh.ARMS:
            row = fresh.load_cell(root, identity, "screen", arm, track, seed, repeat)
            if row is None or row.get("error"):
                raise ValueError(f"unconsumed or failed screen cell: {track}/{seed}/{arm}/{repeat}")
            fresh._sha256_string(row.get("action_trace_sha256"), "action_trace_sha256")
            rows.append(row)
    spot_failures = fresh._verify_spot_checks(rows, protocol, "screen")
    if spot_failures:
        raise ValueError(f"consumed screen spot check failed: {spot_failures}")
    count = len(protocol["partitions"]["screen"]["track_ids"]) * len(
        protocol["partitions"]["screen"]["seeds"])
    summary = fresh.report_phase(root, identity, protocol, "screen")
    if fresh._read_json(root / "screen-summary.json") != summary or summary["decision"] != "REJECT":
        raise ValueError("consumed screen summary or spot checks changed")
    _receipt_manifest(protocol, root)
    for phase in ("confirmation", "blind"):
        if (root / "cells" / phase).exists():
            raise ValueError(f"consumed screen {phase} partition was opened")
    return count


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path],
                   margin: dict, ego: dict) -> dict:
    synthetic = {
        "control_agent_sha256": protocol["agent_sha256"]["baseline"],
        "candidate_agent_sha256": protocol["agent_sha256"]["candidate"],
        "model_sha256": protocol["model_sha256"],
        "runtime_helper_sha256": protocol["runtime_helper_sha256"],
    }
    identity = fresh.build_identity(protocol_path, synthetic, paths["baseline"],
                                    paths["candidate"], paths["model"])
    identity["source_sha256"] = {
        "baseline": identity["source_sha256"]["control"],
        "candidate": identity["source_sha256"]["candidate"],
    }
    reference = {key: identity[key] for key in (
        "model_sha256", "helper_sha256", "harness_sha256", "environment_sha256",
        "platform", "python",
    )}
    for prefix, screen, run in (("margin", margin, MARGIN_SCREEN_RUN),
                                ("ego", ego, EGO_SCREEN_RUN)):
        screen_path = paths[f"{prefix}_screen_protocol"]
        hashes = {
            "protocol": fresh.digest(screen_path),
            "freeze": fresh.digest(run / "freeze.json"),
            "summary": fresh.digest(run / "screen-summary.json"),
        }
        for suffix, observed in hashes.items():
            expected = protocol[f"{prefix}_screen_{suffix}_sha256"]
            if observed != expected:
                raise ValueError(f"{prefix} screen {suffix} hash mismatch")
            identity[f"{prefix}_screen_{suffix}_sha256"] = observed
        verify_rejected_screen(screen, run, hashes["protocol"], reference)
        receipts_hash = _receipt_manifest(screen, run)
        if receipts_hash != protocol[f"{prefix}_screen_receipts_sha256"]:
            raise ValueError(f"{prefix} screen receipts hash mismatch")
        identity[f"{prefix}_screen_receipts_sha256"] = receipts_hash
    identity["comparator_sha256"] = fresh.digest(Path(__file__))
    identity["evidence_scope"] = "CONSUMED_DEVELOPMENT_ONLY"
    return identity


def check_frozen_inputs(identity: dict, paths: dict[str, Path],
                        screens: dict[str, dict]) -> None:
    virtual = {**identity, "source_sha256": {
        "control": identity["source_sha256"]["baseline"],
        "candidate": identity["source_sha256"]["candidate"],
    }}
    fresh.check_frozen_inputs(virtual, {
        "control": paths["baseline"], "candidate": paths["candidate"],
        "model": paths["model"], "protocol": paths["protocol"],
    })
    for prefix, run in (("margin", MARGIN_SCREEN_RUN), ("ego", EGO_SCREEN_RUN)):
        for suffix, path in (("protocol", paths[f"{prefix}_screen_protocol"]),
                             ("freeze", run / "freeze.json"),
                             ("summary", run / "screen-summary.json")):
            if fresh.digest(path) != identity[f"{prefix}_screen_{suffix}_sha256"]:
                raise ValueError(f"{prefix} screen {suffix} changed during development")
        if _receipt_manifest(screens[prefix], run) != identity[f"{prefix}_screen_receipts_sha256"]:
            raise ValueError(f"{prefix} consumed screen receipts changed during development")
    if fresh.digest(Path(__file__)) != identity["comparator_sha256"]:
        raise ValueError("development comparator changed during the run")


def work_plan(cells: tuple[tuple[int, int], ...],
              ego_cells: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], str, str], ...]:
    ego_set = set(ego_cells)
    if len(ego_set) != 32 or len(set(cells)) != len(cells):
        raise ValueError("current screen reuse requires exactly 32 consumed cells")
    return tuple((cell, arm, "reuse" if arm == "baseline" and cell in ego_set else "worker")
                 for cell in cells for arm in ARMS)


def cell_path(root: Path, arm: str, track: int, seed: int) -> Path:
    if arm not in ARMS:
        raise ValueError("unknown development arm")
    return root / "cells" / arm / f"track-{track}-seed-{seed}.json"


def load_cell(root: Path, identity: dict, arm: str, track: int, seed: int) -> dict | None:
    path = cell_path(root, arm, track, seed)
    if not path.exists():
        return None
    envelope = fresh._read_json(path)
    if not isinstance(envelope, dict) or envelope.get("identity") != identity:
        raise ValueError(f"cell identity mismatch: {path}")
    row = envelope.get("row")
    if not isinstance(row, dict) or (row.get("arm"), row.get("track_id"), row.get("seed")) != (arm, track, seed):
        raise ValueError(f"cell coordinates mismatch: {path}")
    if envelope.get("digest") != hashlib.sha256(fresh._canonical({"identity": identity, "row": row})).hexdigest():
        raise ValueError(f"cell digest mismatch: {path}")
    return row


def record_cell(root: Path, identity: dict, row: dict) -> bool:
    path = cell_path(root, row["arm"], row["track_id"], row["seed"])
    if path.exists():
        if load_cell(root, identity, row["arm"], row["track_id"], row["seed"]) != row:
            raise ValueError(f"existing cell differs: {path}")
        return False
    payload = {"identity": identity, "row": row}
    fresh._atomic_json(path, {**payload, "digest": hashlib.sha256(fresh._canonical(payload)).hexdigest()})
    return True


def import_ego_control(screen_root: Path, screen_identity: dict, output_root: Path,
                       dev_identity: dict, track: int, seed: int) -> dict:
    original = fresh.load_cell(screen_root, screen_identity, "screen", "control", track, seed, 0)
    if original is None or original.get("error"):
        raise ValueError(f"missing or failed ego screen control receipt: {track}/{seed}")
    fresh._sha256_string(original.get("action_trace_sha256"), "action_trace_sha256")
    receipt = fresh.cell_path(screen_root, "screen", "control", track, seed, 0)
    imported = {key: value for key, value in original.items()
                if key not in ("partition", "arm", "repeat")}
    imported.update(arm="baseline", receipt_origin="EGO_SCREEN_CONTROL",
                    source_receipt_sha256=fresh.digest(receipt))
    record_cell(output_root, dev_identity, imported)
    return imported


def _row_issue(row: dict) -> str | None:
    if row.get("error"):
        return "operational failure"
    if type(row.get("finished")) is not bool:
        return "malformed numeric or finish result"
    for name in ("progress", "damage"):
        value = row.get(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            return f"non-finite or missing numeric {name}"
        if (name == "progress" and not 0.0 <= value <= 1.0) or (name == "damage" and value < 0):
            return f"out-of-range numeric {name}"
    if type(row.get("collision_count")) is not int or row["collision_count"] < 0:
        return "malformed numeric collision_count"
    if row["finished"]:
        time_ms = row.get("lap_time_ms")
        if type(time_ms) not in (int, float) or not math.isfinite(time_ms) or time_ms < 0:
            return "non-finite or missing numeric lap_time_ms"
    return None


def derive_triage_gate(rows: list[dict], cells: tuple[tuple[int, int], ...],
                       ego_cells: tuple[tuple[int, int], ...]) -> dict:
    """Permit more consumed development only after the fixed mechanism checks."""
    lookup = {(row["track_id"], row["seed"], row["arm"]): row for row in rows}
    ego_set = set(ego_cells)
    reasons = []
    missing = []
    changed = []
    adverse_contacts = []
    adverse_damage = []
    for track, seed in cells:
        baseline = lookup.get((track, seed, "baseline"))
        candidate = lookup.get((track, seed, "candidate"))
        if baseline is None or candidate is None:
            missing.append([track, seed])
            continue
        valid = True
        for arm, row in (("baseline", baseline), ("candidate", candidate)):
            expected_origin = ("EGO_SCREEN_CONTROL" if arm == "baseline" and (track, seed) in ego_set
                               else "COLD_WORKER")
            issue = _row_issue(row)
            if issue:
                reasons.append(f"{track}/{seed}/{arm}: {issue}")
                valid = False
            if row.get("receipt_origin") != expected_origin:
                reasons.append(f"{track}/{seed}/{arm}: receipt provenance failure")
                valid = False
            try:
                fresh._sha256_string(row.get("action_trace_sha256"), "action_trace_sha256")
                if expected_origin == "EGO_SCREEN_CONTROL":
                    fresh._sha256_string(row.get("source_receipt_sha256"), "source_receipt_sha256")
            except ValueError:
                reasons.append(f"{track}/{seed}/{arm}: action or source receipt reproducibility failure")
                valid = False
        if not valid:
            continue
        if baseline["action_trace_sha256"] != candidate["action_trace_sha256"]:
            changed.append([track, seed])
        if baseline["finished"] and not candidate["finished"]:
            reasons.append(f"{track}/{seed}: control finish lost")
        if (not candidate["finished"] and candidate.get("retire_reason") == "crash"
                and (baseline["finished"] or baseline.get("retire_reason") != "crash")):
            reasons.append(f"{track}/{seed}: new crash DNF")
        if (not baseline["finished"] and not candidate["finished"]
                and candidate["progress"] + 1e-9 < baseline["progress"]):
            reasons.append(f"{track}/{seed}: both-DNF progress declined")
        if candidate["collision_count"] > baseline["collision_count"]:
            adverse_contacts.append([track, seed, candidate["collision_count"] - baseline["collision_count"]])
        damage_delta = candidate["damage"] - baseline["damage"]
        if damage_delta > 1e-9:
            adverse_damage.append([track, seed, damage_delta])
    if not missing and not changed:
        reasons.append("candidate action trace unchanged across mechanism cells")
    status = ("INCOMPLETE" if missing else "BLOCK_FULL" if reasons else
              "ALLOW_FULL_CONSUMED_DEVELOPMENT")
    return {"status": status, "reasons": reasons, "missing_cells": missing,
            "changed_action_cells": changed, "adverse_contact_cells": adverse_contacts,
            "adverse_damage_cells": adverse_damage}


def report(root: Path, identity: dict, cells: tuple[tuple[int, int], ...],
           *, mechanism_only: bool, ego_cells: tuple[tuple[int, int], ...]) -> dict:
    rows = []
    for track, seed in cells:
        for arm in ARMS:
            current = load_cell(root, identity, arm, track, seed)
            if current is not None:
                rows.append(current)
    mapped = [{**row, "arm": "control" if row["arm"] == "baseline" else "candidate",
               "repeat": 0, "error": _row_issue(row)} for row in rows]
    comparison = fresh.compare_pairs(mapped, [(track, seed, 0) for track, seed in cells])
    triage_gate = derive_triage_gate(rows, cells, ego_cells) if mechanism_only else None
    if mechanism_only:
        decision = ("INCOMPLETE" if triage_gate["status"] == "INCOMPLETE" else
                    "TRIAGE_BLOCK_FULL" if triage_gate["status"] == "BLOCK_FULL" else
                    "TRIAGE_ALLOW_FULL_COMPARISON_REJECT" if comparison["decision"] == "REJECT" else
                    "MECHANISM_DIAGNOSTIC_ALLOW_FULL")
    else:
        decision = ("REJECT" if comparison["decision"] == "REJECT" else
                    "INCOMPLETE" if comparison["decision"] == "INCOMPLETE" else
                    "RETAIN_DIAGNOSTIC_CANDIDATE" if comparison["decision"] == "RETAIN" else
                    "INCONCLUSIVE")
    ego_set = set(ego_cells)
    reused_expected = sum(cell in ego_set for cell in cells)
    result = {
        "decision": decision, "comparison": comparison,
        "rows_recorded": len(rows), "cells_expected": len(cells),
        "rows_expected": len(cells) * len(ARMS),
        "cold_worker_episodes_expected": len(cells) * len(ARMS) - reused_expected,
        "reused_baseline_expected": reused_expected,
        "reused_baseline_recorded": sum(row.get("receipt_origin") == "EGO_SCREEN_CONTROL" for row in rows),
        "distinct_geometry_seeds": len({seed for _, seed in cells}),
        "mechanism_only": mechanism_only,
        "protocol_sha256": identity["protocol_sha256"],
        "margin_screen_protocol_sha256": identity.get("margin_screen_protocol_sha256"),
        "ego_screen_protocol_sha256": identity.get("ego_screen_protocol_sha256"),
        "candidate_agent_sha256": identity.get("source_sha256", {}).get("candidate"),
        "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
        "fresh_generalization": False, "confirmation_unlocked": False,
        "blind_opened": False, "sota_promotion": False,
    }
    if mechanism_only:
        result["triage_gate"] = {**triage_gate, "comparison_decision": comparison["decision"]}
    return result


def require_triage_gate(root: Path, identity: dict,
                        ego_cells: tuple[tuple[int, int], ...]) -> dict:
    path = root / "mechanism-summary.json"
    if not path.is_file():
        raise ValueError("mechanism triage summary missing; run --mechanism-only first")
    for track, seed in MECHANISM_CELLS:
        for arm in ARMS:
            if load_cell(root, identity, arm, track, seed) is None:
                raise ValueError("mechanism triage incomplete: all 20 rows are required")
    current = report(root, identity, MECHANISM_CELLS,
                     mechanism_only=True, ego_cells=ego_cells)
    if fresh._read_json(path) != current:
        raise ValueError("mechanism triage summary differs from the 20 frozen rows")
    if current["triage_gate"]["status"] != "ALLOW_FULL_CONSUMED_DEVELOPMENT":
        raise ValueError("mechanism triage blocked full development; a new protocol is required")
    return current


def _run_worker(arm: str, track: int, seed: int, protocol: dict,
                paths: dict[str, Path]) -> dict:
    payload = {
        "source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
        "controller_class": protocol["controller_classes"][arm],
        "track_id": track, "seed": seed,
    }
    command = [sys.executable, str(Path(fresh.__file__).resolve()), "--worker",
               json.dumps(payload, separators=(",", ":"))]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                               timeout=300, check=True)
    measured = json.loads(completed.stdout)
    if not isinstance(measured, dict) or measured.get("error") is not None:
        raise ValueError("cold worker returned an invalid result")
    fresh._sha256_string(measured.get("action_trace_sha256"), "action_trace_sha256")
    if type(measured.get("steps")) is not int or not 0 <= measured["steps"] <= 2000:
        raise ValueError("cold worker returned an invalid action count")
    issue = _row_issue(measured)
    if issue:
        raise ValueError(f"cold worker returned {issue}")
    return measured


def run(root: Path, identity: dict, protocol: dict, paths: dict[str, Path],
        screens: dict[str, dict], ego_identity: dict,
        cells: tuple[tuple[int, int], ...], ego_cells: tuple[tuple[int, int], ...],
        *, mechanism_only: bool) -> dict:
    root = validate_output_root(root)
    fresh.prepare_run(root, identity)
    with fresh._run_lock(root):
        if not mechanism_only:
            require_triage_gate(root, identity, ego_cells)
        for (track, seed), arm, origin in work_plan(cells, ego_cells):
            existing = load_cell(root, identity, arm, track, seed)
            if existing is not None:
                if existing.get("error"):
                    raise RuntimeError(f"recorded operational failure: {arm}/{track}/{seed}")
                continue
            check_frozen_inputs(identity, paths, screens)
            if origin == "reuse":
                row = import_ego_control(EGO_SCREEN_RUN, ego_identity, root, identity, track, seed)
                check_frozen_inputs(identity, paths, screens)
            else:
                row = {"arm": arm, "track_id": track, "seed": seed,
                       "receipt_origin": "COLD_WORKER"}
                try:
                    row.update(_run_worker(arm, track, seed, protocol, paths))
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                        json.JSONDecodeError, ValueError) as error:
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    row.update(error=str(error), stderr=stderr[-4000:])
                    record_cell(root, identity, row)
                    failure = report(root, identity, cells, mechanism_only=mechanism_only,
                                     ego_cells=ego_cells)
                    fresh._atomic_json(root / ("mechanism-summary.json" if mechanism_only else "summary.json"), failure)
                    raise RuntimeError(f"worker failed: {arm}/{track}/{seed}") from error
                check_frozen_inputs(identity, paths, screens)
                record_cell(root, identity, row)
            print(json.dumps({"cell": [arm, track, seed], "receipt_origin": row["receipt_origin"],
                              "finished": row["finished"], "progress": row["progress"],
                              "contacts": row["collision_count"]}), flush=True)
    check_frozen_inputs(identity, paths, screens)
    summary = report(root, identity, cells, mechanism_only=mechanism_only,
                     ego_cells=ego_cells)
    summary_path = root / ("mechanism-summary.json" if mechanism_only else "summary.json")
    if summary_path.exists() and fresh._read_json(summary_path) != summary:
        raise ValueError(f"existing consumed development summary differs: {summary_path}")
    if not summary_path.exists():
        fresh._atomic_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--baseline-agent", required=True, type=Path)
    parser.add_argument("--candidate-agent", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--mechanism-only", action="store_true")
    args = parser.parse_args()
    output_root = validate_output_root(args.output_root)
    margin = fresh._read_json(MARGIN_SCREEN_PROTOCOL)
    ego = fresh._read_json(EGO_SCREEN_PROTOCOL)
    margin_hash, ego_hash = fresh.digest(MARGIN_SCREEN_PROTOCOL), fresh.digest(EGO_SCREEN_PROTOCOL)
    protocol = fresh._read_json(args.protocol)
    cells = validate_protocol(protocol, margin, ego, margin_hash, ego_hash,
                              prior.sealed_holdout_seeds(ROOT / "experiments"))
    paths = {"protocol": args.protocol, "baseline": args.baseline_agent,
             "candidate": args.candidate_agent, "model": args.model,
             "margin_screen_protocol": MARGIN_SCREEN_PROTOCOL,
             "ego_screen_protocol": EGO_SCREEN_PROTOCOL}
    identity = build_identity(args.protocol, protocol, paths, margin, ego)
    ego_identity = fresh._read_json(EGO_SCREEN_RUN / "freeze.json")
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "evidence_scope": "CONSUMED_DEVELOPMENT_ONLY",
                          "identity": identity, "development_cells": len(cells),
                          "mechanism_cells": len(MECHANISM_CELLS),
                          "cold_worker_episodes_full": 78,
                          "reused_ego_control_receipts": 32}, indent=2))
        return 0
    selected = MECHANISM_CELLS if args.mechanism_only else cells
    summary = run(output_root, identity, protocol, paths,
                  {"margin": margin, "ego": ego}, ego_identity, selected,
                  ego_screen_cells(ego), mechanism_only=args.mechanism_only)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
