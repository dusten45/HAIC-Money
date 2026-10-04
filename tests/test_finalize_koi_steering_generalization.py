"""Additional evidence guard tests, without touching the original frozen loader."""

import pytest
from typing import Any

from scripts import finalize_koi_steering_generalization as finalizer


def test_completed_rows_use_existing_full_receipt_validation(tmp_path):
    assert finalizer.verify_partial_artifacts(tmp_path, {"rows": [{"status": "completed"}]}) == []


@pytest.mark.parametrize("defect", [None, "changed", "bytes", "missing", "added", "duplicate", "omitted", "escape", "type"])
def test_original_failure_time_partial_pins_are_mandatory(tmp_path, defect):
    row: dict[str, Any] = dict(track_id=1, seed=3184000001, mode="crossing_projection", status="operator_error")
    path = tmp_path / "1-3184000001-crossing_projection.raw.jsonl"
    path.write_text('{"t":1}\n')
    pin: dict[str, Any] = dict(file=path.name, sha256=finalizer.analyzer.sha256(path), bytes=path.stat().st_size)
    row["partial_artifacts"] = [pin]
    if defect == "changed":
        path.write_text('{"t":2}\n')
    elif defect == "bytes":
        pin["bytes"] += 1
    elif defect == "missing":
        path.unlink()
    elif defect == "added":
        path.with_suffix(".unexpected").write_text("ignored")
        (tmp_path / "1-3184000001-crossing_projection.decisions.jsonl").write_text("new")
    elif defect == "duplicate":
        row["partial_artifacts"].append(dict(pin))
    elif defect == "omitted":
        row.pop("partial_artifacts")
    elif defect == "escape":
        pin["file"] = "../outside.json"
    elif defect == "type":
        pin["bytes"] = str(pin["bytes"])
    if defect:
        with pytest.raises(ValueError):
            finalizer.verify_partial_artifacts(tmp_path, {"rows": [row]})
    else:
        assert len(finalizer.verify_partial_artifacts(tmp_path, {"rows": [row]})) == 1
