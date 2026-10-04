"""Repeat the corrected pursuit arm on consumed TRAIN cells, never new score."""
import argparse
import hashlib
import json
from pathlib import Path

from haic_agent.fixed_high_speed_pursuit_repair import FixedHighSpeedPursuitRepair
from training import fixed_high_speed_steering as evaluator


def run(output):
    source = Path("artifacts/haic-research-v2/fixed-high-speed-steering-train-20260929/report.json")
    reference = json.loads(source.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert reference["common_prefix_valid"]
    evaluator.ARMS = ("pursuit",)
    evaluator.FixedHighSpeedAgent = FixedHighSpeedPursuitRepair
    evaluator.run(output)
    result = json.loads(output.read_text())
    controls = {(r["track"], r["seed"]): r for r in reference["rows"] if r["arm"] == "control"}
    comparisons = [dict(track=r["track"], seed=r["seed"],
                        same_prefix=r["qualification"]["prefix_hash"] ==
                        controls[r["track"], r["seed"]]["qualification"]["prefix_hash"])
                   for r in result["rows"]]
    passed = len(comparisons) == 6 and all(row["same_prefix"] for row in comparisons)
    with (output.parent / "paired-prefix-verification.json").open("x") as stream:
        json.dump(dict(reference=source.as_posix(), reference_sha256=source_hash,
                       comparisons=comparisons, passed=passed, consumed_train_only=True), stream, indent=2)
    if not passed or hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Frozen control prefix verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
