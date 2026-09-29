"""Two stateful consumed-TRAIN recovery continuations with fixed speed."""
import argparse
import hashlib
import json
from pathlib import Path

from haic_agent.fixed_high_speed_recovery_v2 import MODES, FixedHighSpeedRecoveryV2
from training import fixed_high_speed_steering as evaluator


def run(output):
    source = Path("artifacts/haic-research-v2/fixed-high-speed-steering-train-20260929/report.json")
    reference = json.loads(source.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluator.ARMS = MODES
    evaluator.FixedHighSpeedAgent = FixedHighSpeedRecoveryV2
    evaluator.run(output)
    result = json.loads(output.read_text())
    controls = {(r["track"], r["seed"]): r for r in reference["rows"] if r["arm"] == "damping"}
    comparisons = []
    for row in result["rows"]:
        control = controls[row["track"], row["seed"]]
        episode = json.loads((output.parent / row["evidence"]).read_text())
        trace = episode["decision_trace"]
        changed = [t for t in trace if t["controller"]["recovery_changed"]]
        collisions = [t for t in trace if t["collision"]]
        first = changed[0] if changed else None
        comparisons.append(dict(arm=row["arm"],track=row["track"],seed=row["seed"],
            same_prefix=row["qualification"]["prefix_hash"] == control["qualification"]["prefix_hash"],
            control_completed=control["completed"], candidate_completed=row["completed"],
            preserved_finish=not control["completed"] or row["completed"],
            effective_interventions=len(changed),first_intervention_step=None if first is None else first["step"],
            progress_after_intervention=None if first is None else episode["progress"]-first["progress"],
            collision_progress=[dict(step=t["step"],progress=t["progress"],
                                     subsequent_progress=episode["progress"]-t["progress"]) for t in collisions],
            impact_proxy_count=sum(t["controller"]["impact_proxy_trigger"] for t in trace)))
    passed = len(comparisons)==12 and all(r["same_prefix"] for r in comparisons)
    summary = {mode:dict(completed=sum(r["candidate_completed"] for r in comparisons if r["arm"]==mode),
                        preserved_original_finishes=sum(r["control_completed"] and r["candidate_completed"] for r in comparisons if r["arm"]==mode),
                        effective_interventions=sum(r["effective_interventions"] for r in comparisons if r["arm"]==mode)) for mode in MODES}
    with (output.parent/"recovery_comparison.json").open("x") as stream:
        json.dump(dict(reference=source.as_posix(),reference_sha256=source_hash,comparisons=comparisons,
                       prefix_valid=passed,summary=summary,consumed_train_only=True,
                       strict_road_qualification_is_diagnostic_only=True),stream,indent=2)
    if not passed or hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError("Frozen damping prefix verification failed")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    run(parser.parse_args().output)
