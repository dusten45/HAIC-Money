"""Summarize frozen RLPD G0 TRAIN-only outcomes by paired geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from haic.algorithms.rlpd.g0_diagnostic import summarize_g0_pairs
from scripts.diagnose_rlpd_g0 import ROOT, _file, _pinned_json, sha256_file


def summarize_run(root: Path, output_root: str) -> dict:
    root = root.resolve()
    relative = Path(output_root)
    if (not output_root or relative.is_absolute() or relative.parts[0] != "runs"
            or ".." in relative.parts):
        raise ValueError("G0 summary requires a repository-contained runs/ root")
    run_dir = root / relative
    manifest_path = _file(root, (relative / "manifest.json").as_posix(), "runs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("ranked") is not False
            or manifest.get("role") != "TRAIN-only-failure-diagnostic"):
        raise ValueError("G0 manifest is incomplete or is a ranking receipt")
    protocol = _pinned_json(
        _file(root, manifest["protocol_path"], "experiments"),
        manifest["protocol_sha256"],
    )
    if (protocol.get("format") != "haic-rlpd-g0-diagnostic-v1"
            or protocol.get("status") != "frozen"
            or protocol.get("geometry_audit_sha256") != manifest.get("geometry_audit_sha256")):
        raise ValueError("G0 summary protocol and manifest are inconsistent")
    audit = _pinned_json(
        _file(root, protocol["geometry_audit_path"], "experiments"),
        protocol["geometry_audit_sha256"],
    )
    if (audit.get("status") != "no_known_recorded_overlap"
            or audit.get("passed") is not True
            or audit.get("cells") != protocol.get("cells")):
        raise ValueError("G0 summary cannot attribute results to its frozen TRAIN audit")
    cells = [(row["track_id"], row["geometry_seed"]) for row in protocol["cells"]]
    actor_ids = tuple(row["id"] for row in protocol["actors"])
    ledger = _file(root, (relative / "cells.jsonl").as_posix(), "runs")
    if sha256_file(ledger) != manifest.get("cells_sha256"):
        raise ValueError("G0 cells ledger changed after the manifest was written")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if manifest.get("cell_count") != len(rows) or manifest.get("geometry_count") != len(cells):
        raise ValueError("G0 manifest counts disagree with the complete ledger")
    for row in rows:
        trace = row.get("trace_path")
        if (not isinstance(trace, str) or not trace.startswith("traces/")
                or Path(trace).is_absolute() or ".." in Path(trace).parts):
            raise ValueError("G0 trace escaped its frozen run directory")
        trace_path = _file(root, (relative / trace).as_posix(), "runs")
        if not trace_path.is_relative_to(run_dir) or sha256_file(trace_path) != row.get("trace_sha256"):
            raise ValueError("G0 trajectory trace differs from the per-cell ledger")
    if (
        sum(row["steps"] for row in rows) != manifest.get("decisions_spent")
        or sum(row["driven_raw_frames"] for row in rows) != manifest.get("driven_raw_frames")
        or sum(row["reset_initial_raw_frames"] for row in rows) != manifest.get("reset_initial_raw_frames")
        or sum(row["reset_noop_raw_frames"] for row in rows) != manifest.get("reset_noop_raw_frames")
        or manifest["decisions_spent"] > manifest["decision_cap"]
    ):
        raise ValueError("G0 raw/decision/reset frame accounting disagrees with the manifest")
    summary = summarize_g0_pairs(rows, cells=cells, actor_ids=actor_ids)
    return {
        "format": "haic-rlpd-g0-train-summary-v1",
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_sha256": manifest["protocol_sha256"],
        "decisions_spent": manifest["decisions_spent"],
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_run(ROOT, args.run_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
