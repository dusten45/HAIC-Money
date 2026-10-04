"""Source-bound, zero-repeat continuation; freeze/preflight never construct an env.

Only --run invokes the unchanged archived worker. A verified natural-boundary
pause is mandatory even when an original operator_error occurred before intent.
Original claims stay original: this is an unexecuted-slot child, not fresh roads.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
import traceback
from types import FunctionType
import zipfile

from scripts import analyze_koi_steering_generalization as analysis
from scripts import audit_koi_steering_generalization as freshness
from scripts import evaluate_koi_steering_generalization as parent
from scripts import finalize_koi_steering_generalization as forensic

ROOT = parent.ROOT
STUDY = "koi-steering-generalization-continuation-v1"
FORMAT = "haic-koi-steering-generalization-continuation-v1"
RUN_PATH = Path("runs") / STUDY
PROTOCOL_PATH = Path("experiments") / f"{STUDY}.json"
LEDGER_NAME = "reset-intents.jsonl"
PARENT_SHA = "ccc6720f676814eb705e853792eceb31c6c5fb2b2aa695b1144641c2befcd7dd"
AUDIT_SHA = "b2efd664347d841b595e3362d1b2bc387ac2b36139475fe038129690c5867fe4"
GUARD_SHA = "745977592b22409efcdb71719c03f746425ebfb2f59ed8aeb36e2cb38279b4a3"
ARMS = parent.ARMS
sha, read_json, read_jsonl = parent.sha, parent.read_json, parent.read_jsonl
COMPACT_KEYS = ("mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage",
                "collisions", "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes")
INHERITED_KEYS = ("analysis_spec", "model_hashes", "model_source_sha256", "candidate_manifest_sha256",
                  "environment_source_sha256", "helper_source_sha256", "root_agent_sha256",
                  "root_agent_source_sha256", "runtime_versions", "python", "frame_skip", "warmup_ticks",
                  "raw_fps", "max_decisions", "steering_terms_schema", "reference_evidence", "freshness",
                  "protected_claims_observed_sha256", "telemetry_passive_not_agent_input", "official_action")
PROTOCOL_KEYS = set(INHERITED_KEYS) | {"format", "study", "scope", "parent_protocol_sha256", "parent_boundary",
    "parent_result", "parent_evidence_sha256", "original_worker_sha256", "operator_sha256", "analyzer_sha256",
    "continuation_source_sha256", "source_copies", "episodes", "schedule", "cells", "resource_forecast",
    "resource_measurement", "reset_ledger_file", "imported_original_slots"}
_PARENT_CACHE = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return row["track_id"], row["seed"], row["mode"]


def cell(row):
    return dict(partition="TRAIN", track_id=row["track_id"], geometry_seed=row["seed"], obstacles=True)


@contextmanager
def claim_lock():
    descriptor = os.open(ROOT / freshness.REGISTRY, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _pin(path, pins):
    path = Path(path).resolve()
    digest = sha(path)
    require(str(path) not in pins or pins[str(path)] == digest, "evidence changed between reads")
    pins[str(path)] = digest
    return path


def _file(run, name, expected, pins):
    path = analysis.prior.physical.verify_file(run, name, expected)
    require(not path.is_symlink(), "symlinked episode evidence")
    return _pin(path, pins)


def validate_record(run, row, protocol, pins, *, compute_analysis=True):
    """Frozen pure episode/stream validators, plus complete process/hash chain."""
    key = identity(row)
    require(set(row) == set(COMPACT_KEYS) | {"status", "file", "sha256", "slot_id", "process_file", "process_sha256"}
            and row["status"] == "completed" and row["slot_id"] == parent.slot_id(*key)
            and type(row["track_id"]) is int and type(row["seed"]) is int and row["mode"] in ARMS,
            "completed event schema/identity differs")
    stem = f"{key[0]}-{key[1]}-{key[2]}"
    require(row["file"] == stem + ".json" and row["process_file"] == stem + ".bound-process.json",
            "episode filename identity differs")
    record = read_json(_file(run, row["file"], row["sha256"], pins))
    require((record["track_id"], record["seed"], record["mode"]) == key, "saved episode identity differs")
    require(all(row[k] == record[k] for k in COMPACT_KEYS), "compact episode fields differ")
    process = read_json(_file(run, row["process_file"], row["process_sha256"], pins))
    require(process["status"] == "completed" and not process["error"]
            and process["slot_id"] == parent.slot_id(*key)
            and process["operator_sha256"] == protocol["operator_sha256"], "bound process identity differs")
    require(process["original_process_file"] == stem + ".process.json", "original process filename differs")
    original = read_json(_file(run, process["original_process_file"], process["original_process_sha256"], pins))
    require(all(process[k] == original[k] for k in ("status", "error", "wall_time_s")), "process chain differs")
    require(type(process["wall_time_s"]) in (int, float) and math.isfinite(process["wall_time_s"])
            and process["wall_time_s"] > 0, "invalid child wall time")
    require(record["raw_trace_file"] == stem + ".raw.jsonl"
            and record["partial_decisions_file"] == stem + ".decisions.jsonl", "trace filenames differ")
    raw, _ = read_jsonl(_file(run, record["raw_trace_file"], record["raw_trace_sha256"], pins))
    decisions, _ = read_jsonl(_file(run, record["partial_decisions_file"], record["partial_decisions_sha256"], pins))
    require(not record["error"] and not record["invalid_actions"]
            and record["retire_reason"] not in analysis.prior.physical.OPERATIONAL_RETIREMENTS
            and record["retire_reason"] != "max_steps", "original completion is operational/censored")
    require(record["versions"] == protocol["runtime_versions"] and record["runtime_module_paths"]
            and record["damage_telemetry_valid"] and record["collision_telemetry_valid"], "invalid episode telemetry")
    analysis.prior.validate_episode(record, raw, key, protocol)
    analysis.validate_streamed_decisions(record, decisions)
    road = hashlib.sha256(json.dumps(record["catalog"]["track"], sort_keys=True).encode()).hexdigest()
    require(record["track_geometry_sha256"] == road
            and road not in protocol["reference_evidence"]["prior_road_sha256"], "road digest is invalid/consumed")
    for suffix in (".stdout.txt", ".stderr.txt"):
        path = run / (stem + suffix)
        if path.exists():
            _pin(path, pins)
    return record, analysis.analyze_episode(record, raw) if compute_analysis else None, process


def validate_pair(previous, record):
    require(previous["geometry_sha256"] == record["geometry_sha256"], "paired layout differs")
    require(previous["initial_state"] == record["initial_state"]
            and previous["initial_observation_sha256"] == record["initial_observation_sha256"], "paired initial state differs")
    keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
    require(min(len(previous["decision_trace"]), len(record["decision_trace"])) >= 10, "paired prefix too short")
    require([[d[k] for k in keys] for d in previous["decision_trace"][:10]]
            == [[d[k] for k in keys] for d in record["decision_trace"][:10]], "paired first-ten prefix differs")


def validate_roads(episodes):
    roads, pairs = {}, {}
    for key, (record, _) in episodes.items():
        track, seed, _ = key
        road = record["track_geometry_sha256"]
        require(seed not in roads or roads[seed] == road, "cross-track same-seed road differs")
        require(all(s == seed or digest != road for s, digest in roads.items()), "distinct seeds replay one road")
        roads[seed] = road
        pair = track, seed
        if pair in pairs:
            validate_pair(pairs.pop(pair), record)
        else:
            pairs[pair] = record
    return roads, pairs


def verified_parent(boundary_path, boundary_sha256):
    """Read-only API for main: verified originals and exactly zero-intent slots."""
    from scripts.pause_koi_steering_generalization import verify_boundary
    boundary = verify_boundary(boundary_path, boundary_sha256, root=ROOT)
    require(boundary["protocol_sha256"] == PARENT_SHA, "pause belongs to a different parent")
    require(boundary["status"] == "paused" and boundary["terminal"]["process_exited"] is True
            and boundary["boundary"]["pending_reset_intents"] == 0
            and boundary["boundary"]["no_live_children"] is True
            and boundary["boundary"]["child_interrupted"] is False, "uncertain original boundary")
    protocol, exposure = parent.validate_frozen(PARENT_SHA)
    require(protocol["freshness"]["audit_receipt"]["sha256"] == AUDIT_SHA, "original claim receipt differs")
    run = (ROOT / parent.RUN_PATH).resolve()
    cache_key = (str(ROOT.resolve()), str(Path(boundary_path).resolve()), boundary_sha256)
    cached = _PARENT_CACHE.get(cache_key)
    if cached is not None:
        # Boundary verification and the original source/model validation above
        # ALWAYS run. Only semantic parsing of byte-identical terminal episodes
        # is reused; new cross-lane metadata is never cached by check_claims.
        require(cached["protocol"] == protocol and cached["exposure"] == exposure
                and cached["boundary"] == boundary, "cached terminal source contract differs")
        inventory = freshness.StableInventory(ROOT)
        for path, expected in cached["source_pins"].items():
            raw = inventory.read(Path(path).relative_to(ROOT).as_posix())
            require(hashlib.sha256(raw).hexdigest() == expected, "cached original evidence changed")
            inventory.raw.clear()
        for row in cached["remaining_slots"]:
            key = identity(row)
            output = run / f"{key[0]}-{key[1]}-{key[2]}.json"
            require(not any(output.with_suffix(s).exists() for s in forensic.SUFFIXES), "cached zero-intent slot acquired evidence")
        return deepcopy(cached)
    pins = {}
    _pin(boundary_path, pins)
    report_path = _pin(run / "episode-report.json", pins)
    report = read_json(report_path)
    require(str(report_path) == boundary["terminal"]["report_path"]
            and sha(report_path) == boundary["terminal"]["report_sha256"], "pause report pin differs")
    require(report["protocol_sha256"] == PARENT_SHA and not report["complete"], "not an incomplete original attempt")
    review_pin = report["review_receipt"]
    review = parent.validate_review(protocol, PARENT_SHA, review_pin["path"], review_pin["sha256"])
    require(read_json(_pin(run / "execution-review.json", pins)) == review_pin, "original execution review differs")
    require(review["evidence_guard_sha256"] == GUARD_SHA
            and review["finalizer_sha256"] == sha(forensic.__file__), "original forensic review differs")
    _pin(review_pin["path"], pins)
    require(sha(forensic.GUARD_PATH) == GUARD_SHA, "original forensic guard differs")
    _pin(forensic.GUARD_PATH, pins)
    forensic.verify_partial_artifacts(run, report)
    ledger_path = _pin(run / "reset-ledger.jsonl", pins)
    require(sha(ledger_path) == report["reset_ledger_sha256"]
            == boundary["terminal"]["reset_ledger_sha256"], "original reset ledger pin differs")
    ledger, _ = read_jsonl(ledger_path)
    require(len(ledger) % 2 == 0, "pending original reset intent")
    count = len(ledger) // 2
    require(count == boundary["boundary"]["completed_slots"]
            and 0 < count < len(protocol["schedule"]), "original boundary denominator differs")
    rows = report["rows"]
    require(len(rows) == len(protocol["schedule"]), "original full schedule missing")
    completed, remaining, child_walls = {}, [], []
    for index, (row, planned) in enumerate(zip(rows, protocol["schedule"])):
        key = identity(planned)
        require(identity(row) == key, "original report schedule/order differs")
        if index < count:
            require(row["status"] == "completed" and ledger[2 * index + 1] == row, "original completion/ledger differs")
            intent = ledger[2 * index]
            require(set(intent) == {"status", "arm", "track", "seed", "slot_id", "time"}
                    and intent["status"] == "reset_intent" and (intent["track"], intent["seed"], intent["arm"]) == key
                    and intent["slot_id"] == parent.slot_id(*key), "original reset-intent identity differs")
            record, measured, process = validate_record(run, row, protocol, pins)
            completed[key] = record, measured
            child_walls.append(process["wall_time_s"])
        else:
            require(row["status"] in {"unrun", "operator_error"}, "original remaining slot has uncertain status")
            require(row["status"] != "operator_error" or index == count, "nonboundary original operator_error")
            require(not row.get("partial_artifacts") and not row.get("file"), "pre-intent slot has artifacts")
            output = run / f"{key[0]}-{key[1]}-{key[2]}.json"
            require(not any(output.with_suffix(s).exists() for s in forensic.SUFFIXES), "zero-intent slot has unclassified evidence")
            remaining.append(deepcopy(planned))
    validate_roads(completed)
    wall = boundary["terminal"]["wall_time_s"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= sum(child_walls), "parent wall evidence invalid")
    if forensic.RESULT_PATH.exists():
        _pin(forensic.RESULT_PATH, pins)
    result = dict(protocol=protocol, exposure=exposure, report=report, completed=completed,
                  remaining_slots=remaining, source_pins=pins, boundary=boundary,
                  parent_wall_time_s=wall, child_wall_time_s=child_walls)
    if forensic.RESULT_PATH.exists():
        _PARENT_CACHE[cache_key] = deepcopy(result)
    return result


def guard_peak_rss_bytes():
    """Process-lifetime high-water RSS includes cached state and transient copies."""
    require(sys.platform == "linux", "ru_maxrss byte conversion is Linux-only")
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    require(type(peak_kib) is int and peak_kib > 0, "invalid own-process guard RSS measurement")
    return peak_kib * 1024


def resource_forecast(verified, measured_audit_s, parent_peak_guard_rss_bytes):
    require(type(measured_audit_s) in (int, float) and math.isfinite(measured_audit_s)
            and measured_audit_s > 0, "full audit timing must be measured")
    require(type(parent_peak_guard_rss_bytes) is int and parent_peak_guard_rss_bytes > 0,
            "guard peak RSS must be a measured positive integer byte count")
    count, remaining = len(verified["completed"]), len(verified["remaining_slots"])
    wall = verified["parent_wall_time_s"]
    observed = wall / count
    # Each slot has a parent audit and another audit immediately before reset.
    # The unchanged worker verifies at setup and reset; the private adapters also
    # verify before delegating. Budget conservatively for eight complete guards.
    # Future guards also verify the growing child ledger. Scale the measured
    # initial completed prefix to the full cohort, not merely the current prefix.
    growth = verified["protocol"]["episodes"] / count
    guarded = max(observed, max(verified["child_wall_time_s"]) + 8 * measured_audit_s * growth)
    forecast = deepcopy(verified["protocol"]["resource_forecast"])
    historical_peak = verified["protocol"]["reference_evidence"]["prior_peak_rss_bytes"]
    reserve = forecast["memory_reserve_bytes"]
    require(type(historical_peak) is int and historical_peak > 0
            and type(reserve) is int and reserve >= 0, "invalid frozen historical memory evidence")
    guard_estimate = math.ceil(parent_peak_guard_rss_bytes * 1.5 * growth)
    child_estimate = math.ceil(historical_peak * 1.5)
    # Legacy admission adds memory_reserve_bytes to peak_child_bytes. Supply the
    # combined coordinator/guard plus one child estimate, not a single old child.
    forecast.update(episodes=remaining, wall_budget_s=guarded * remaining * 1.5 + 2 * measured_audit_s,
                    parent_peak_guard_rss_bytes=parent_peak_guard_rss_bytes,
                    historical_single_child_peak_rss_bytes=historical_peak,
                    estimated_full_cohort_guard_peak_bytes=guard_estimate,
                    estimated_single_child_peak_bytes=child_estimate,
                    peak_child_bytes=guard_estimate + child_estimate,
                    required_memory_bytes=guard_estimate + child_estimate + reserve,
                    resource_formula=dict(schema="koi-continuation-memory-v1",
                        guard_rss_source="freeze process Linux RUSAGE_SELF.ru_maxrss * 1024 after verified_parent/full audit/defensive copies; process-lifetime high water",
                        guard_safety_factor=1.5, historical_child_safety_factor=1.5,
                        guard_growth_factor=growth,
                        required_memory="ceil(parent_peak_guard_rss_bytes * 1.5 * guard_growth_factor) + ceil(historical_single_child_peak_rss_bytes * 1.5) + original memory_reserve_bytes",
                        peak_child_bytes_semantics="combined full-cohort coordinator/guard estimate plus one child, excluding separate reserve",
                        scope="conservative declared peak estimate, not observed future cohort RSS or a single-episode bound"),
                    actual_parent_process_wall_s=wall, original_completed_slots=count,
                    actual_parent_wall_per_slot_s=observed, measured_full_audit_s=measured_audit_s,
                    guarded_reference_per_slot_s=guarded, wall_safety_margin_factor=1.5,
                    child_timeout_s=max(forecast["child_timeout_s"], guarded * 1.5),
                    full_guard_budgeted_passes_per_slot=8,
                    full_cohort_guard_growth_factor=growth,
                    rationale="Separate remaining-only budget: terminal actual process wall/slots and measured full validation/audit scaled to full cohort with eight guard passes and 50% margin. RAM combines measured own-process guard/cache peak scaled to full cohort and historical single-child peak, each at1.5x, plus original reserve; not an observed future-cohort bound. Original budgets untouched; never retry exposed slots.")
    return forecast


def _validate_child_ledger(protocol, path):
    if not path.exists():
        return []
    require(not path.is_symlink(), "symlinked child ledger")
    raw = path.read_bytes()
    require(not raw or raw.endswith(b"\n"), "torn child reset ledger")
    events, _ = read_jsonl(path)
    require(len(events) <= 2 * len(protocol["schedule"]), "child reset ledger exceeds schedule")
    for index, event in enumerate(events):
        planned = protocol["schedule"][index // 2]
        key = identity(planned)
        if index % 2 == 0:
            require(set(event) == {"status", "arm", "track", "seed", "slot_id", "time"}
                    and event["status"] == "reset_intent" and (event["track"], event["seed"], event["arm"]) == key
                    and type(event["track"]) is int and type(event["seed"]) is int
                    and type(event["time"]) in (int, float) and math.isfinite(event["time"])
                    and event["slot_id"] == parent.slot_id(*key), "child reset-intent schema/order differs")
        else:
            require(set(event) == set(COMPACT_KEYS) | {"status", "file", "sha256", "slot_id", "process_file", "process_sha256"}
                    and event["status"] == "completed" and identity(event) == key
                    and event["slot_id"] == parent.slot_id(*key), "child completion schema/order differs")
            validate_record(path.parent, event, protocol, {}, compute_analysis=False)
    return events


def helper_paths():
    return [Path(__file__), ROOT / "scripts/pause_koi_steering_generalization.py", Path(forensic.__file__),
            ROOT / "scripts/finalize_koi_steering_generalization_composite.py"]


def validate_frozen(protocol_sha256):
    destination = (ROOT / RUN_PATH).resolve()
    inventory = freshness.StableInventory(ROOT)
    root_raw = inventory.read(str(PROTOCOL_PATH))
    copied_raw = inventory.read(str(RUN_PATH / "protocol.json"))
    require(hashlib.sha256(root_raw).hexdigest() == protocol_sha256 and root_raw == copied_raw, "child root/copy protocol differs")
    protocol = json.loads(root_raw)
    require(set(protocol) == PROTOCOL_KEYS, "unexpected child protocol schema")
    require(protocol["format"] == FORMAT and protocol["study"] == STUDY, "child typed protocol differs")
    require(protocol["scope"] == "original zero-reset-intent slots only; not new roads or repeated evidence", "child scope differs")
    verified = verified_parent(protocol["parent_boundary"]["path"], protocol["parent_boundary"]["sha256"])
    original = verified["protocol"]
    require(protocol["parent_protocol_sha256"] == PARENT_SHA
            and protocol["original_worker_sha256"] == original["operator_sha256"], "original parent/worker differs")
    require(protocol["schedule"] == verified["remaining_slots"] and protocol["episodes"] == len(verified["remaining_slots"]), "child has repeated/altered slots")
    require(protocol["reset_ledger_file"] == LEDGER_NAME and protocol["imported_original_slots"] == len(verified["completed"]), "child dataset metadata differs")
    expected_cells = [c for c in original["cells"] if any(cell(r) == c for r in protocol["schedule"])]
    require(protocol["cells"] == expected_cells, "child cell conditions differ")
    require(all(protocol[k] == original[k] for k in INHERITED_KEYS), "unchanged model/environment/measurement contract differs")
    require(protocol["operator_sha256"] == sha(__file__) and protocol["analyzer_sha256"] == original["analyzer_sha256"], "child source differs")
    require(protocol["parent_evidence_sha256"] == verified["source_pins"], "immutable original completion pins differ")
    require(protocol["parent_result"] == {"path": str(forensic.RESULT_PATH.resolve()), "sha256": sha(forensic.RESULT_PATH)}, "preserved original partial result differs")
    require(protocol["continuation_source_sha256"] == {str(p): sha(p) for p in helper_paths()}, "child source closure differs")
    forecast = resource_forecast(verified, protocol["resource_forecast"]["measured_full_audit_s"],
                                 protocol["resource_forecast"]["parent_peak_guard_rss_bytes"])
    formula = protocol["resource_forecast"]["resource_formula"]
    expected_formula = forecast["resource_formula"]
    require(type(formula) is dict and set(formula) == set(expected_formula)
            and all(type(formula[k]) is type(value) and formula[k] == value for k, value in expected_formula.items()),
            "typed memory resource formula differs")
    for key in ("parent_peak_guard_rss_bytes", "historical_single_child_peak_rss_bytes",
                "estimated_full_cohort_guard_peak_bytes", "estimated_single_child_peak_bytes",
                "peak_child_bytes", "required_memory_bytes", "memory_reserve_bytes"):
        require(type(protocol["resource_forecast"][key]) is int, "memory forecast byte counts must be integers")
    require(protocol["resource_forecast"] == forecast, "remaining budget changed")
    expected_sources = {str(p) for p in [*parent.helper_paths(), *helper_paths(), ROOT / "agent.py",
                        *map(Path, original["environment_source_sha256"])]}
    expected_sources.update(str(ROOT / parent.RUN_PATH / name) for name in
                            ("candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"))
    require(len(protocol["source_copies"]) == len(expected_sources)
            and {copy["source"] for copy in protocol["source_copies"]} == expected_sources, "incomplete/duplicate child source copies")
    for copy in protocol["source_copies"]:
        require(sha(copy["source"]) == copy["sha256"], "child copied live source differs")
        _file(destination, copy["file"], copy["sha256"], {})
    require(sha(destination / "candidate.zip") == parent.CANDIDATE_SHA
            and sha(destination / "crossing-projection-source-reconstruction.zip") == parent.BASELINE_SHA
            and sha(destination / "candidate.manifest.json") == original["candidate_manifest_sha256"], "child model bytes differ")
    _validate_child_ledger(protocol, destination / LEDGER_NAME)
    return protocol, verified["exposure"]


def check_claims(receipt, protocol_sha256=None, candidate_cell=None, *, boundary_pin=None):
    """Exact typed child allowance; unchanged original self remains fully audited."""
    original, expected_receipt = parent.validate_frozen(PARENT_SHA)
    require(receipt == expected_receipt, "original audit receipt differs")
    freshness.verify_claims(receipt, root=ROOT)
    allowed = set()
    if protocol_sha256 is not None:
        protocol, _ = validate_frozen(protocol_sha256)
        if candidate_cell is not None:
            require(candidate_cell in protocol["cells"], "reset outside child remaining cohort")
        allowed = {str(PROTOCOL_PATH), str(RUN_PATH / "protocol.json"), str(RUN_PATH / LEDGER_NAME)}
        require(boundary_pin is None or boundary_pin == protocol["parent_boundary"], "conflicting boundary allowance")
        boundary_pin = protocol["parent_boundary"]
    else:
        require(not (ROOT / PROTOCOL_PATH).exists() and not (ROOT / RUN_PATH).exists(), "unfrozen child metadata exists")
    require(boundary_pin is not None and set(boundary_pin) == {"path", "sha256"}, "exact terminal boundary allowance required")
    if boundary_pin is None:
        raise ValueError("exact terminal boundary allowance required")
    from scripts.pause_koi_steering_generalization import verify_boundary
    verify_boundary(boundary_pin["path"], boundary_pin["sha256"], root=ROOT)
    boundary_path = Path(boundary_pin["path"]).resolve()
    require(boundary_path == (ROOT / parent.RUN_PATH / "boundary-pause.json").resolve()
            or boundary_path.parent == (ROOT / "experiments").resolve(), "boundary metadata outside canonical receipt locations")
    allowed.add(boundary_path.relative_to(ROOT).as_posix())
    require(sha(freshness.__file__) == original["helper_source_sha256"][str(Path(freshness.__file__))], "archived auditor source differs")
    namespace = dict(freshness.audit.__globals__)
    discover = freshness._discover

    def authenticated_discover(root):
        paths = discover(root)
        # These paths were authenticated independently above. Everything else,
        # including any lookalike protocol/intent, remains in the original audit.
        return [path for path in paths if path not in allowed]

    namespace["_discover"] = authenticated_discover
    audit = FunctionType(freshness.audit.__code__, namespace, freshness.audit.__name__,
                         freshness.audit.__defaults__, freshness.audit.__closure__)
    audit.__kwdefaults__ = freshness.audit.__kwdefaults__
    report = audit(ROOT, authenticated_receipt=receipt, self_protocol_sha256=PARENT_SHA)
    require(report["status"] == "clear", "candidate-relevant freshness HOLD: " + json.dumps(report["collisions"] + report["blockers"]))
    registry = ROOT / freshness.REGISTRY
    protected = {str(registry / f"seed-{s}.json"): sha(registry / f"seed-{s}.json")
                 for s in parent.PROTECTED if (registry / f"seed-{s}.json").exists()}
    require(protected == original["protected_claims_observed_sha256"], "protected claim metadata changed")
    return protected


def validate_review(protocol, protocol_sha256, path, expected_sha256):
    require(sha(path) == expected_sha256, "child review receipt hash differs")
    review = read_json(path)
    require(review["status"] == "passed" and review["environment_resets"] == 0
            and isinstance(review["reviewer"], str) and review["reviewer"].strip(), "child independent review missing")
    fields = dict(protocol_sha256=protocol_sha256, operator_sha256=protocol["operator_sha256"],
                  analyzer_sha256=protocol["analyzer_sha256"], audit_sha256=AUDIT_SHA,
                  parent_protocol_sha256=PARENT_SHA, parent_boundary_sha256=protocol["parent_boundary"]["sha256"],
                  original_worker_sha256=protocol["original_worker_sha256"])
    require(all(review.get(k) == v for k, v in fields.items()), "child exact-source review differs")
    finalizer_path = ROOT / "scripts/finalize_koi_steering_generalization_composite.py"
    require(review.get("composite_finalizer_sha256") == protocol["continuation_source_sha256"][str(finalizer_path)]
            == sha(finalizer_path), "review lacks bound composite finalizer")
    guard_path = ROOT / "experiments/koi-steering-generalization-v1-composite-guard.json"
    require(review.get("composite_guard_sha256") == sha(guard_path), "review lacks bound composite guard")
    guard = read_json(guard_path)
    require(guard["continuation_protocol_sha256"] == protocol_sha256
            and guard["continuation_operator_sha256"] == protocol["operator_sha256"]
            and guard["parent_protocol_sha256"] == PARENT_SHA
            and guard["composite_finalizer_sha256"] == review["composite_finalizer_sha256"], "composite guard source association differs")
    return review


def preflight(protocol_sha256):
    destination = ROOT / RUN_PATH
    require(not any((destination / name).exists() for name in (LEDGER_NAME, "episode-report.json", "execution-review.json"))
            and not list(destination.glob("*.raw.jsonl")), "child consumed/partial; never resume")
    protocol, receipt = validate_frozen(protocol_sha256)
    require(parent.runtime_versions() == protocol["runtime_versions"], "CPU21 runtime differs")
    measurement = parent.resource_measurements(destination)
    parent.check_resources(measurement, protocol["resource_forecast"], protocol["episodes"])
    with claim_lock():
        check_claims(receipt, protocol_sha256)
    return dict(status="passed", protocol_sha256=protocol_sha256, environment_resets=0,
                episodes=protocol["episodes"], resource_measurement=measurement)


def _save_exclusive(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def freeze_protocol(boundary_path, boundary_sha256):
    """Requires terminal proof before writing ANY child metadata; no simulator."""
    verified = verified_parent(boundary_path, boundary_sha256)
    original_result = read_json(forensic.RESULT_PATH)
    require(original_result["protocol_sha256"] == PARENT_SHA
            and original_result["episode_report_sha256"] == sha(ROOT / parent.RUN_PATH / "episode-report.json")
            and original_result["evidence_guard_sha256"] == GUARD_SHA
            and original_result["gate_passed"] is False, "original guarded incomplete result must be preserved before child freeze")
    destination = (ROOT / RUN_PATH).resolve()
    require(not destination.exists() and not (ROOT / PROTOCOL_PATH).exists(), "never overwrite a child freeze")
    original = verified["protocol"]
    with claim_lock():
        started = time.monotonic()
        check_claims(verified["exposure"], boundary_pin=dict(path=str(Path(boundary_path).resolve()), sha256=boundary_sha256))
        # Reverify terminal evidence after the potentially slow inventory read.
        verified = verified_parent(boundary_path, boundary_sha256)
        measured = time.monotonic() - started
        _pin(forensic.RESULT_PATH, verified["source_pins"])
        peak_guard = guard_peak_rss_bytes()
        forecast = resource_forecast(verified, measured, peak_guard)
        measurement = parent.resource_measurements(ROOT / "runs")
        parent.check_resources(measurement, forecast, len(verified["remaining_slots"]))
        destination.mkdir(exist_ok=False)
        copies = []
        paths = [*parent.helper_paths(), *helper_paths(), ROOT / "agent.py",
                 *map(Path, original["environment_source_sha256"])]
        for path in dict.fromkeys(paths):
            relative = Path("source/environment") / path.relative_to(parent.SNAPSHOT) if path.is_relative_to(parent.SNAPSHOT) else Path("source/project") / path.relative_to(ROOT)
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copies.append(dict(file=str(relative), source=str(path), sha256=sha(copied)))
        for name in ("candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"):
            source = ROOT / parent.RUN_PATH / name
            (destination / name).write_bytes(source.read_bytes())
            copies.append(dict(file=name, source=str(source), sha256=sha(source)))
        schedule = verified["remaining_slots"]
        protocol = {k: deepcopy(original[k]) for k in INHERITED_KEYS}
        protocol.update(format=FORMAT, study=STUDY, scope="original zero-reset-intent slots only; not new roads or repeated evidence",
                        parent_protocol_sha256=PARENT_SHA,
                        parent_boundary=dict(path=str(Path(boundary_path).resolve()), sha256=boundary_sha256),
                        parent_result=dict(path=str(forensic.RESULT_PATH.resolve()), sha256=sha(forensic.RESULT_PATH)),
                        parent_evidence_sha256=verified["source_pins"], original_worker_sha256=original["operator_sha256"],
                        operator_sha256=sha(__file__), analyzer_sha256=original["analyzer_sha256"],
                        continuation_source_sha256={str(p): sha(p) for p in helper_paths()},
                        source_copies=copies, episodes=len(schedule), schedule=schedule,
                        cells=[c for c in original["cells"] if any(cell(r) == c for r in schedule)],
                        resource_forecast=forecast, resource_measurement=measurement,
                        reset_ledger_file=LEDGER_NAME, imported_original_slots=len(verified["completed"]))
        _save_exclusive(destination / "protocol.json", protocol)
        with (ROOT / PROTOCOL_PATH).open("xb") as stream:
            stream.write((destination / "protocol.json").read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    result = dict(status="frozen", protocol_sha256=sha(destination / "protocol.json"),
                  environment_resets=0, episodes=len(schedule), run=str(destination))
    validate_frozen(result["protocol_sha256"])
    return result


def _worker_namespace(protocol_sha256):
    """Function-local reuse, never global monkeypatch the archived operator."""
    protocol, _ = validate_frozen(protocol_sha256)
    namespace = dict(parent.worker.__globals__)

    def validate_worker(digest):
        actual, exposure = validate_frozen(digest)
        # The frozen worker checks its own __file__. The child runner/review and
        # process receipts use the NEW source hash; only this private view differs.
        return dict(actual, operator_sha256=actual["original_worker_sha256"]), exposure

    def review_worker(_view, digest, path, expected):
        actual, _ = validate_frozen(digest)
        return validate_review(actual, digest, path, expected)

    namespace.update(ROOT=ROOT, RUN_PATH=RUN_PATH, PROTOCOL_PATH=PROTOCOL_PATH,
                     CELLS=tuple((r["track_id"], r["geometry_seed"]) for r in protocol["cells"]),
                     validate_frozen=validate_worker, check_claims=check_claims, validate_review=review_worker)
    return namespace


def worker(arm, track, seed, output):
    digest = sha(ROOT / PROTOCOL_PATH)
    protocol, _ = validate_frozen(digest)
    events = _validate_child_ledger(protocol, ROOT / RUN_PATH / LEDGER_NAME)
    require(events and len(events) % 2 == 1
            and (events[-1]["track"], events[-1]["seed"], events[-1]["arm"]) == (track, seed, arm), "worker lacks exact pending child intent")
    unchanged = FunctionType(parent.worker.__code__, _worker_namespace(digest), parent.worker.__name__,
                             parent.worker.__defaults__, parent.worker.__closure__)
    return unchanged(arm, track, seed, output)


def dataset_metadata(protocol_sha256):
    """Public integration contract: parent episodes are original-path, never copies."""
    protocol, _ = validate_frozen(protocol_sha256)
    verified = verified_parent(protocol["parent_boundary"]["path"], protocol["parent_boundary"]["sha256"])
    return dict(protocol=protocol, parent_protocol=verified["protocol"], parent_report=verified["report"],
                parent_episodes=verified["completed"], child_slots=deepcopy(protocol["schedule"]),
                parent_source_pins=verified["source_pins"],
                child_run=(ROOT / RUN_PATH).resolve(), child_ledger=(ROOT / RUN_PATH / LEDGER_NAME).resolve(),
                combined_schedule=deepcopy(verified["protocol"]["schedule"]))


def run(protocol_sha256, review_path, review_sha256):
    preflight(protocol_sha256)
    protocol, exposure = validate_frozen(protocol_sha256)
    review_path = Path(review_path).resolve()
    validate_review(protocol, protocol_sha256, review_path, review_sha256)
    destination = (ROOT / RUN_PATH).resolve()
    rows = deepcopy(protocol["schedule"])
    metadata = dataset_metadata(protocol_sha256)
    combined = dict(metadata["parent_episodes"])
    rows_error, current = None, None
    _save_exclusive(destination / "execution-review.json", dict(path=str(review_path), sha256=review_sha256))
    ledger_path = destination / LEDGER_NAME
    started = time.monotonic()
    try:
        with ledger_path.open("x", buffering=1) as ledger, tempfile.TemporaryDirectory(prefix="koi-continuation-", dir="/tmp/kilo") as temporary:
            directories = {}
            for arm, name in zip(ARMS, ("crossing-projection-source-reconstruction.zip", "candidate.zip")):
                directory = Path(temporary) / arm
                directory.mkdir()
                with zipfile.ZipFile(destination / name) as archive:
                    archive.extractall(directory)
                directories[arm] = directory
            for current in rows:
                remaining = protocol["resource_forecast"]["wall_budget_s"] - (time.monotonic() - started)
                require(remaining > protocol["resource_forecast"]["child_timeout_s"], "insufficient frozen budget for a whole child slot")
                validate_review(protocol, protocol_sha256, review_path, review_sha256)
                parent.check_resources(parent.resource_measurements(destination), protocol["resource_forecast"],
                                       sum(r["status"] == "unrun" for r in rows))
                key = identity(current)
                with claim_lock():
                    check_claims(exposure, protocol_sha256, cell(current))
                    # Auditing may be slow. Never publish a reset after budget expiry.
                    remaining = protocol["resource_forecast"]["wall_budget_s"] - (time.monotonic() - started)
                    require(remaining > protocol["resource_forecast"]["child_timeout_s"], "insufficient whole-slot budget before intent")
                    ledger.write(json.dumps(dict(status="reset_intent", arm=key[2], track=key[0], seed=key[1],
                                                 slot_id=parent.slot_id(*key), time=time.time())) + "\n")
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    current["status"] = "reset_intent"
                output = destination / f"{key[0]}-{key[1]}-{key[2]}.json"
                command = f"import sys;sys.path.insert(0,{str(ROOT)!r});from scripts.evaluate_koi_steering_generalization_continuation import worker;worker({key[2]!r},{key[0]},{key[1]},{str(output)!r})"
                parent.run_child([protocol["python"], "-I", "-c", command], directories[key[2]], output,
                                 protocol["resource_forecast"]["child_timeout_s"],
                                 parent.slot_id(*key), protocol["operator_sha256"])
                record = read_json(output)
                process = output.with_suffix(".bound-process.json")
                compact = {k: record[k] for k in COMPACT_KEYS}
                compact.update(status="completed", file=output.name, sha256=sha(output), slot_id=parent.slot_id(*key),
                               process_file=process.name, process_sha256=sha(process))
                current.update(compact)
                ledger.write(json.dumps(compact, allow_nan=False) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
                verified_record, measured, _ = validate_record(destination, current, protocol, {})
                combined[key] = verified_record, measured
                validate_roads(combined)
                print(json.dumps(compact, allow_nan=False), flush=True)
            roads, unmatched = validate_roads(combined)
            require(len(combined) == parent.EPISODES and not unmatched and len(roads) == len(parent.SEEDS), "composite coverage incomplete")
            validate_frozen(protocol_sha256)
    except BaseException as failure:
        rows_error = f"{type(failure).__name__}: {failure}"
        if current is not None and current["status"] != "completed":
            current.update(status="operator_error", error=rows_error)
            output = destination / f"{current['track_id']}-{current['seed']}-{current['mode']}.json"
            current["partial_artifacts"] = [dict(file=p.name, sha256=sha(p), bytes=p.stat().st_size)
                                             for s in forensic.SUFFIXES if (p := output.with_suffix(s)).exists()]
        _save_exclusive(destination / "operator-failure.json", dict(error=rows_error, traceback=traceback.format_exc()))
        raise
    finally:
        _save_exclusive(destination / "episode-report.json", dict(protocol_sha256=protocol_sha256, rows=rows,
                         operator_error=rows_error, reset_ledger_sha256=sha(ledger_path) if ledger_path.exists() else None,
                         review_receipt=dict(path=str(review_path), sha256=review_sha256),
                         complete=rows_error is None and all(r["status"] == "completed" for r in rows),
                         parent_protocol_sha256=PARENT_SHA, parent_boundary=protocol["parent_boundary"],
                         wall_time_s=time.monotonic() - started,
                         freshness_scope="original proven unexecuted slots only; no repeat or partial-exposure relabel"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-protocol", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--boundary-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.freeze_protocol:
        if not args.boundary or not args.boundary_sha256:
            parser.error("--freeze-protocol requires terminal --boundary and --boundary-sha256")
        print(json.dumps(freeze_protocol(args.boundary, args.boundary_sha256)))
    elif args.preflight:
        if not args.protocol_sha256:
            parser.error("--protocol-sha256 required")
        print(json.dumps(preflight(args.protocol_sha256)))
    else:
        if not args.protocol_sha256 or not args.review or not args.review_sha256:
            parser.error("--run requires --protocol-sha256, --review and --review-sha256")
        run(args.protocol_sha256, args.review, args.review_sha256)


if __name__ == "__main__":
    main()
