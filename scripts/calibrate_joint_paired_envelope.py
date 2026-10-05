"""Fit one declared interval-excess target on OLD CAL; never run a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from haic.algorithms.joint_control import paired_residual as residual


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def run(analysis_path, absolute_path, output):
    analysis_path, absolute_path, output = map(Path, (analysis_path, absolute_path, output))
    absolute = json.loads(absolute_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    for name, expected in absolute["source_pins"].items():
        if sha(name) != expected:
            raise ValueError("original source pin differs: " + name)
    rows = []
    for anchor in analysis["anchors"]:
        stage = anchor["calibrated"]
        predicted, actual = stage["comparison"], stage["actual"]
        rows.append(dict(anchor_id=anchor["anchor_id"], seed=anchor["seed"], role=anchor["split"],
                         supported=bool(all(predicted["cost_supported"]) and actual["common_support"]),
                         predicted_delta=predicted["reference_delta"][1], actual_delta=actual["delta_by_reference"]))
    fit = residual.fit_calibration(rows, absolute)
    fit["evidence_pins"] = {str(p.resolve()): sha(p) for p in (analysis_path, absolute_path)}
    fit["source_pins"] = dict(absolute["source_pins"])
    for p in (Path(__file__), ROOT / "haic/algorithms/joint_control/paired_residual.py"):
        fit["source_pins"][str(p.resolve())] = sha(p)
    diagnostics = []
    for anchor, row in zip(analysis["anchors"], rows):
        record = {k: row[k] for k in ("anchor_id", "seed", "role", "supported")}
        if row["supported"]:
            stage = anchor["calibrated"]["comparison"]
            d = np.asarray(stage["reference_delta"], float)
            interval = residual.expanded_interval(d, lower=fit["performance"]["lower"],
                                                 upper=fit["performance"]["upper"])
            error = residual.envelope_excess(row["predicted_delta"], row["actual_delta"])
            physical = bool(np.asarray(stage["absolute_supported"])[1].all()
                            and not np.asarray(stage["veto"])[1].any())
            sign = -1 if interval[1, 1] < -.05 else 1 if interval[1, 0] > .05 else 0
            record.update(old_interval=stage["delta_interval"][1], new_interval=interval[1],
                          new_sign=sign, physical_pass_unchanged=physical,
                          would_select=physical and sign == -1,
                          excess=error, covered=error["lower"] <= fit["performance"]["lower"]
                          and error["upper"] <= fit["performance"]["upper"])
        diagnostics.append(record)
    report = dict(schema="haic-joint-temporal-envelope-retrospective-v1",
                  qualification="Posthoc development on previously consumed old pairs; not new validation.",
                  original_allowance=absolute["paired_cost_residual"], performance=fit["performance"],
                  rows=diagnostics, absolute_calibration_unchanged=fit["absolute_calibration"] == absolute)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("envelope-calibration.json", "envelope-retrospective.json"):
        if (output / name).exists():
            raise FileExistsError("immutable output already exists: " + name)
    for name, value in (("envelope-calibration.json", fit), ("envelope-retrospective.json", report)):
        with (output / name).open("x") as stream:
            json.dump(clean(value), stream, allow_nan=False, indent=2)
            stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "runs/joint-temporal-interval-v1/paired-analysis.json")
    parser.add_argument("--absolute", type=Path, default=ROOT / "runs/joint-temporal-interval-v1/calibration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/joint-temporal-diagnosis-v1")
    args = parser.parse_args()
    report = run(args.analysis, args.absolute, args.output)
    print(json.dumps(clean(report), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
