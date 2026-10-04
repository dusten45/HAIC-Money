"""Join a reviewed boundary continuation without rerunning completed episodes."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts import analyze_koi_steering_generalization as analysis
from scripts import evaluate_koi_steering_generalization as original
from scripts import finalize_koi_steering_generalization as forensic

GUARD_PATH = original.ROOT / "experiments/koi-steering-generalization-v1-composite-guard.json"
RESULT_PATH = original.ROOT / "experiments/koi-steering-generalization-v1-composite-result.json"


def key(row):
    return row["track_id"], row["seed"], row["mode"]


def verify_source_closure(run, protocol, helper_paths, continuation_paths=()):
    if protocol["analysis_spec"] != analysis.ANALYSIS_SPEC or protocol["analyzer_sha256"] != analysis.sha256(analysis.__file__):
        raise ValueError("frozen measurement or gates changed")
    if protocol["model_hashes"] != {analysis.ARMS[0]: analysis.BASELINE_SHA, analysis.ARMS[1]: analysis.CANDIDATE_SHA}:
        raise ValueError("frozen models changed")
    if len(protocol["environment_source_sha256"]) != 142 or set(protocol["helper_source_sha256"]) != {str(p) for p in helper_paths}:
        raise ValueError("transitive source closure differs")
    if set(protocol.get("continuation_source_sha256", {})) != {str(p) for p in continuation_paths}:
        raise ValueError("continuation source closure differs")
    copies = {}
    for row in protocol["source_copies"]:
        analysis.prior.physical.verify_file(run, row["file"], row["sha256"])
        if row["source"] in copies:
            raise ValueError("duplicate source copy")
        copies[row["source"]] = row["sha256"]
    for group in ("environment_source_sha256", "helper_source_sha256", "continuation_source_sha256"):
        for path, pin in protocol.get(group, {}).items():
            if analysis.sha256(path) != pin or copies.get(path) != pin:
                raise ValueError("source/copy differs: " + path)
    for path, pin in protocol["root_agent_source_sha256"].items():
        if analysis.sha256(path) != pin:
            raise ValueError("root Agent changed")
    for path, pin in protocol["reference_evidence"]["source_sha256"].items():
        if analysis.sha256(path) != pin:
            raise ValueError("frozen reference evidence changed")
    for suffix, pin in analysis.prior.physical.FROZEN_SOURCE_SHA256.items():
        if [h for p, h in protocol["environment_source_sha256"].items() if p.endswith("/" + suffix)] != [pin]:
            raise ValueError("environment source differs")
    analysis.prior.verify_models(protocol, run, run / "candidate.zip", run / "candidate.manifest.json")
    runtime = [h for p, h in protocol["helper_source_sha256"].items() if p.endswith("/haic/algorithms/koi/steering_release.py")]
    if runtime != [protocol["model_source_sha256"][analysis.ARMS[1]]["haic_agent/steering_release_runtime.py"]]:
        raise ValueError("candidate runtime source differs")
    for field, value in (("frame_skip", 4), ("warmup_ticks", 50), ("raw_fps", 50), ("max_decisions", 1200)):
        if protocol[field] != value:
            raise ValueError("runtime condition differs")


def load_completed_source(run, protocol, report, ledger_name, schedule, allow_boundary=False):
    """Verify original byte artifacts independently of legacy freshness entrypoints."""
    run = Path(run)
    if report["protocol_sha256"] != analysis.sha256(run / "protocol.json") or protocol["schedule"] != schedule:
        raise ValueError("source schedule/protocol association differs")
    forensic.verify_partial_artifacts(run, report)
    expected = [key(r) for r in schedule]
    if [key(r) for r in report["rows"]] != expected or len(set(expected)) != len(expected):
        raise ValueError("source report schedule differs")
    rows, episodes = {}, {}
    for row in report["rows"]:
        k = key(row)
        rows[k] = dict(row)
        if row["status"] != "completed":
            if not allow_boundary or row["status"] not in ("unrun", "operator_error") or row.get("partial_artifacts") or "file" in row:
                raise ValueError("unresolved interrupted/invalid slot")
            continue
        compact = {"mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage", "collisions", "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes", "status", "file", "sha256", "slot_id", "process_file", "process_sha256"}
        stem = f"{k[0]}-{k[1]}-{k[2]}"
        if set(row) != compact or row["slot_id"] != original.slot_id(*k) or type(row["track_id"]) is not int or type(row["seed"]) is not int or row["file"] != stem + ".json" or row["process_file"] != stem + ".bound-process.json":
            raise ValueError("completed slot schema/identity differs")
        e = analysis.read_json(analysis.prior.physical.verify_file(run, row["file"], row["sha256"]))
        process = analysis.read_json(analysis.prior.physical.verify_file(run, row["process_file"], row["process_sha256"]))
        receipt = analysis.read_json(analysis.prior.physical.verify_file(run, process["original_process_file"], process["original_process_sha256"]))
        if process["status"] != "completed" or process["error"] or process["slot_id"] != original.slot_id(*k) or process["operator_sha256"] != protocol["operator_sha256"]:
            raise ValueError("source process binding differs")
        if any(process[f] != receipt[f] for f in ("status", "error", "wall_time_s")):
            raise ValueError("original process receipt differs")
        if process["original_process_file"] != stem + ".process.json" or e["raw_trace_file"] != stem + ".raw.jsonl" or e["partial_decisions_file"] != stem + ".decisions.jsonl":
            raise ValueError("source trace/process filename differs")
        raw, _ = analysis.read_jsonl(analysis.prior.physical.verify_file(run, e["raw_trace_file"], e["raw_trace_sha256"]))
        stream, _ = analysis.read_jsonl(analysis.prior.physical.verify_file(run, e["partial_decisions_file"], e["partial_decisions_sha256"]))
        analysis.prior.validate_episode(e, raw, k, protocol)
        analysis.validate_streamed_decisions(e, stream)
        if e.get("error") or e.get("invalid_actions") or e.get("retire_reason") in analysis.prior.physical.OPERATIONAL_RETIREMENTS or not e.get("damage_telemetry_valid") or not e.get("collision_telemetry_valid") or e.get("versions") != protocol["runtime_versions"] or not e.get("runtime_module_paths"):
            raise ValueError("operationally invalid source episode")
        for field in ("completed", "lapTimeMs", "progress", "damage", "collisions", "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes"):
            if row[field] != e[field]:
                raise ValueError("compact/primary episode field differs: " + field)
        road = hashlib.sha256(json.dumps(e["catalog"]["track"], sort_keys=True).encode()).hexdigest()
        if e["track_geometry_sha256"] != road or road in protocol["reference_evidence"]["prior_road_sha256"]:
            raise ValueError("road geometry binding/novelty differs")
        measurement = analysis.analyze_episode(e, raw)
        if measurement["planned_censor"]:
            raise ValueError("planned episode censor cannot be repaired by survivors")
        episodes[k] = e, measurement
    ledger, _ = analysis.read_jsonl(analysis.prior.physical.verify_file(run, ledger_name, report["reset_ledger_sha256"]))
    if len(ledger) != len(episodes) * 2:
        raise ValueError("pending intent or duplicate/unknown source reset")
    for index, event in enumerate(ledger):
        k = expected[index // 2]
        if index % 2 == 0:
            if set(event) != {"status", "track", "seed", "arm", "slot_id", "time"} or event.get("status") != "reset_intent" or (event.get("track"), event.get("seed"), event.get("arm")) != k or event.get("slot_id") != original.slot_id(*k) or type(event["track"]) is not int or type(event["seed"]) is not int or type(event["time"]) not in (int, float) or not math.isfinite(event["time"]):
                raise ValueError("reset intent differs")
        elif event != rows[k] or event.get("status") != "completed":
            raise ValueError("reset completion/report differs")
    if set(expected[:len(episodes)]) != set(episodes):
        raise ValueError("source completion is not an exact prefix")
    if report.get("operator_error"):
        if not allow_boundary or not report["operator_error"].startswith("KeyboardInterrupt:"):
            raise ValueError("unexpected operator failure")
    elif not report.get("complete"):
        raise ValueError("missing source terminal state")
    return rows, episodes


def combine(parent_rows, parent_episodes, child_rows, child_episodes, remaining):
    expected = [key(r) for r in original.scheduled_slots()]
    planned = [key(r) for r in remaining]
    completed = [k for k in expected if k in parent_episodes]
    if set(parent_rows) != set(expected) or planned != [k for k in expected if k not in parent_episodes]:
        raise ValueError("continuation is not the exact unexecuted complement")
    if set(child_rows) != set(planned) or set(child_episodes) != set(planned) or set(parent_episodes) & set(child_episodes):
        raise ValueError("duplicated, missing or unexpected continuation episode")
    merged = {**parent_episodes, **child_episodes}
    rows = [parent_rows[k] if k in parent_episodes else child_rows[k] for k in expected]
    if len(merged) != analysis.SLOTS or any(r["status"] != "completed" for r in rows):
        raise ValueError("incomplete combined census")
    return rows, merged, dict(parent_completed=len(completed), continuation_completed=len(planned), completed_unique=len(merged), actual_reset_intents=len(merged), repeated_completed_slots=0)


def freeze_guard(child_sha256):
    from scripts import evaluate_koi_steering_generalization_continuation as child
    protocol, _ = child.validate_frozen(child_sha256)
    guard = dict(study=original.STUDY, purpose="exact unexecuted-slot boundary continuation, unchanged policies/environment/metrics/gates",
                 parent_protocol_sha256=original.sha(original.ROOT / original.PROTOCOL_PATH),
                 continuation_protocol_sha256=child_sha256, composite_finalizer_sha256=analysis.sha256(__file__),
                 continuation_operator_sha256=protocol["operator_sha256"], analyzer_sha256=analysis.sha256(analysis.__file__),
                 original_evidence_guard_sha256=analysis.sha256(forensic.GUARD_PATH),
                 actual_canonical_reset_intents=analysis.SLOTS, repeated_completed_slots=0,
                 policy_updates=0, environment_resets=0,
                 original_attempt_status="preserved incomplete boundary attempt; not silently relabeled passed",
                 blocker_policy="any pending reset, child interruption, invalid telemetry, repeated slot, unmatched model/environment, or incomplete canonical census prevents generalization verdict")
    original.save(GUARD_PATH, guard)
    return dict(path=str(GUARD_PATH), sha256=analysis.sha256(GUARD_PATH), **guard)


def finalize(guard_sha256):
    from scripts import evaluate_koi_steering_generalization_continuation as child
    if analysis.sha256(GUARD_PATH) != guard_sha256:
        raise ValueError("composite guard changed")
    guard = analysis.read_json(GUARD_PATH)
    if guard["composite_finalizer_sha256"] != analysis.sha256(__file__):
        raise ValueError("composite finalizer changed")
    protocol, _ = child.validate_frozen(guard["continuation_protocol_sha256"])
    if protocol["operator_sha256"] != guard["continuation_operator_sha256"]:
        raise ValueError("continuation source changed")
    run = original.ROOT / child.RUN_PATH
    report = analysis.read_json(run / "episode-report.json")
    review_pin = report["review_receipt"]
    review = child.validate_review(protocol, guard["continuation_protocol_sha256"], review_pin["path"], review_pin["sha256"])
    if review.get("composite_finalizer_sha256") != guard["composite_finalizer_sha256"] or review.get("composite_guard_sha256") != guard_sha256:
        raise ValueError("independent review lacks composite guard binding")
    if analysis.read_json(run / "execution-review.json") != review_pin:
        raise ValueError("continuation execution review/report differs")
    parent_run = original.ROOT / original.RUN_PATH
    parent_protocol = analysis.read_json(parent_run / "protocol.json")
    parent_report = analysis.read_json(parent_run / "episode-report.json")
    if analysis.sha256(parent_run / "protocol.json") != guard["parent_protocol_sha256"]:
        raise ValueError("parent protocol changed")
    if analysis.sha256(forensic.GUARD_PATH) != guard["original_evidence_guard_sha256"]:
        raise ValueError("original forensic guard changed")
    parent_review_pin = parent_report["review_receipt"]
    parent_review = original.validate_review(parent_protocol, guard["parent_protocol_sha256"], parent_review_pin["path"], parent_review_pin["sha256"])
    if analysis.read_json(parent_run / "execution-review.json") != parent_review_pin or parent_review.get("evidence_guard_sha256") != guard["original_evidence_guard_sha256"] or parent_review.get("finalizer_sha256") != analysis.sha256(forensic.__file__):
        raise ValueError("original guarded review binding differs")
    verify_source_closure(parent_run, parent_protocol, original.helper_paths())
    verify_source_closure(run, protocol, original.helper_paths(), child.helper_paths())
    for field in ("model_hashes", "model_source_sha256", "candidate_manifest_sha256", "runtime_versions", "environment_source_sha256", "analysis_spec", "root_agent_source_sha256"):
        if parent_protocol[field] != protocol[field]:
            raise ValueError("matched policy/runtime boundary changed: " + field)
    parent_rows, parent_episodes = load_completed_source(parent_run, parent_protocol, parent_report, "reset-ledger.jsonl", original.scheduled_slots(), allow_boundary=True)
    child_rows, child_episodes = load_completed_source(run, protocol, report, "reset-intents.jsonl", protocol["schedule"])
    rows, episodes, counts = combine(parent_rows, parent_episodes, child_rows, child_episodes, protocol["schedule"])
    result = analysis.summarize(rows, episodes)
    result.update(study=original.STUDY, evidence_kind="reviewed boundary-composite, not a passed original interrupted run",
                  analysis_spec=analysis.ANALYSIS_SPEC, model_hashes=parent_protocol["model_hashes"],
                  parent_protocol_sha256=guard["parent_protocol_sha256"], continuation_protocol_sha256=guard["continuation_protocol_sha256"],
                  parent_episode_report_sha256=analysis.sha256(parent_run / "episode-report.json"),
                  continuation_episode_report_sha256=analysis.sha256(run / "episode-report.json"),
                  original_attempt_operator_error=parent_report["operator_error"],
                  original_attempt_result_sha256=analysis.sha256(forensic.RESULT_PATH),
                  composite_finalizer_sha256=guard["composite_finalizer_sha256"], composite_guard_sha256=guard_sha256,
                  source_census=counts, review_receipt=review_pin,
                  passive_analysis=True, environment_resets=0, model_updates=0, official_action=False)
    analysis.prior.finite_tree(result)
    original.save(RESULT_PATH, result)
    return {k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed", "decision", "source_census")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-composite-guard", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--continuation-protocol-sha256")
    parser.add_argument("--guard-sha256")
    args = parser.parse_args()
    if args.freeze_composite_guard:
        if not args.continuation_protocol_sha256:
            parser.error("--continuation-protocol-sha256 required")
        result = freeze_guard(args.continuation_protocol_sha256)
    else:
        if not args.guard_sha256:
            parser.error("--guard-sha256 required")
        result = finalize(args.guard_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
