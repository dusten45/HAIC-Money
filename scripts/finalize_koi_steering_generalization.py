"""Additional forensic guard; never edits frozen sources/models or resets an env."""

import argparse
import json
from pathlib import Path

from scripts import analyze_koi_steering_generalization as analyzer
from scripts import evaluate_koi_steering_generalization as operator

GUARD_PATH = operator.ROOT / "experiments/koi-steering-generalization-v1-evidence-guard.json"
RESULT_PATH = operator.ROOT / "experiments/koi-steering-generalization-v1-result.json"
SUFFIXES = (".json", ".raw.jsonl", ".decisions.jsonl", ".process.json", ".bound-process.json", ".stdout.txt", ".stderr.txt")


def verify_partial_artifacts(run, report):
    run = Path(run).resolve()
    verified = []
    for row in report["rows"]:
        if row["status"] == "completed":
            continue
        output = run / f"{row['track_id']}-{row['seed']}-{row['mode']}.json"
        inventory = {p.name: p for suffix in SUFFIXES if (p := output.with_suffix(suffix)).exists()}
        pins = row.get("partial_artifacts", [])
        if not isinstance(pins, list) or any(not isinstance(p, dict) for p in pins) or len(pins) != len(inventory) or {p.get("file") for p in pins} != set(inventory):
            raise ValueError("partial failure-time artifact inventory differs")
        for pin in pins:
            if set(pin) != {"file", "sha256", "bytes"} or type(pin["bytes"]) is not int:
                raise ValueError("malformed partial artifact pin")
            path = analyzer.prior.physical.verify_file(run, pin["file"], pin["sha256"])
            if path.stat().st_size != pin["bytes"]:
                raise ValueError("partial artifact failure-time byte count differs")
            verified.append(dict(track_id=row["track_id"], seed=row["seed"], mode=row["mode"], **pin))
    return verified


def freeze_guard(protocol_sha256):
    protocol, _ = operator.validate_frozen(protocol_sha256)
    guard = dict(study=operator.STUDY, purpose="additional forensic verification, no measurement/gate/model/cohort change",
                 protocol_sha256=protocol_sha256, operator_sha256=protocol["operator_sha256"],
                 analyzer_sha256=protocol["analyzer_sha256"], finalizer_sha256=analyzer.sha256(__file__),
                 mandatory_partial_artifact_inventory=True, environment_resets=0,
                 limitation="Frozen raw loader does not verify producer failure-time partial_artifacts; only guarded finalization is authoritative.")
    operator.save(GUARD_PATH, guard)
    return dict(file=str(GUARD_PATH), sha256=analyzer.sha256(GUARD_PATH), **guard)


def finalize(guard_sha256):
    if analyzer.sha256(GUARD_PATH) != guard_sha256:
        raise ValueError("evidence guard receipt changed")
    guard = analyzer.read_json(GUARD_PATH)
    if guard["finalizer_sha256"] != analyzer.sha256(__file__):
        raise ValueError("finalizer source changed")
    run = operator.ROOT / operator.RUN_PATH
    protocol, _ = operator.validate_frozen(guard["protocol_sha256"])
    if any(protocol[k] != guard[k] for k in ("operator_sha256", "analyzer_sha256")):
        raise ValueError("guard frozen source binding differs")
    report = analyzer.read_json(run / "episode-report.json")
    review_pin = report["review_receipt"]
    review = operator.validate_review(protocol, guard["protocol_sha256"], review_pin["path"], review_pin["sha256"])
    if review.get("finalizer_sha256") != guard["finalizer_sha256"] or review.get("evidence_guard_sha256") != guard_sha256:
        raise ValueError("independent review lacks evidence guard binding")
    verified = verify_partial_artifacts(run, report)
    protocol, slots, episodes, errors = analyzer.load_run(run, run / "candidate.zip", run / "candidate.manifest.json")
    result = analyzer.summarize(slots, episodes, errors)
    result.update(study=operator.STUDY, analysis_spec=analyzer.ANALYSIS_SPEC,
                  protocol_sha256=guard["protocol_sha256"], model_hashes=protocol["model_hashes"],
                  analyzer_sha256=protocol["analyzer_sha256"], finalizer_sha256=guard["finalizer_sha256"],
                  evidence_guard_sha256=guard_sha256, episode_report_sha256=analyzer.sha256(run / "episode-report.json"),
                  partial_failure_time_artifacts_verified=verified, review_receipt=review_pin,
                  passive_analysis=True, environment_resets=0, model_updates=0, official_action=False)
    analyzer.prior.finite_tree(result)
    operator.save(RESULT_PATH, result)
    return {k: result[k] for k in ("episodes_available", "completion", "gates", "gate_passed", "decision")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-evidence-guard", action="store_true")
    action.add_argument("--finalize", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--guard-sha256")
    args = parser.parse_args()
    if args.freeze_evidence_guard:
        if not args.protocol_sha256:
            parser.error("--protocol-sha256 required")
        result = freeze_guard(args.protocol_sha256)
    else:
        if not args.guard_sha256:
            parser.error("--guard-sha256 required")
        result = finalize(args.guard_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
