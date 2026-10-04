import json
from importlib.metadata import version

import numpy as np
import pytest
import sys

from agents.apex_2026 import evaluate
from agents.apex_2026.evaluate import (
    DEVELOPMENT_SEEDS, MANDATORY_CELLS, assess, end_reason, execute_worker, pace_profile, rejection_streak,
    validate_action,
)


def row(track, seed, *, finished=True, seconds=12.0):
    return {"track_id": track, "seed": seed, "finished": finished,
            "lap_time_ms": round(seconds * 1000) if finished else None,
            "progress": 1.0 if finished else 0.99, "collision_count": 0}


def cells():
    return [row(*cell) for cell in MANDATORY_CELLS], [
        row(track, seed) for track in range(1, 5) for seed in (90001, 90002, 90003, 90004)
    ]


def test_dnf_is_never_treated_as_a_fast_lap():
    mandatory, extra = cells()
    mandatory[0] = row(*MANDATORY_CELLS[0], finished=False)
    verdict = assess(mandatory, extra, limit_seconds=13)
    assert verdict["passed"] is False
    assert verdict["original_target_met"] is False


def test_missing_mandatory_and_duplicate_cells_are_rejected():
    mandatory, extra = cells()
    with pytest.raises(ValueError, match="mandatory"):
        assess(mandatory[:-1], extra, limit_seconds=13)
    with pytest.raises(ValueError, match="duplicate"):
        assess(mandatory, extra + [extra[0]], limit_seconds=13)


def test_relaxation_does_not_relabel_original_target():
    mandatory, extra = cells()
    mandatory[0] = row(*MANDATORY_CELLS[0], seconds=14)
    assert not assess(mandatory, extra, limit_seconds=13)["passed"]
    verdict = assess(mandatory, extra, limit_seconds=15)
    assert verdict["passed"]
    assert not verdict["original_target_met"]
    assert [pace_profile(n) for n in (0, 2, 3, 5, 6, 100)] == [13, 13, 15, 15, 18, 18]


def test_generalization_failure_and_per_track_floor_fail():
    mandatory, extra = cells()
    extra[0] = row(1, 90001, finished=False)
    assert assess(mandatory, extra, limit_seconds=13)["passed"]
    extra[1] = row(1, 90002, finished=False)
    assert not assess(mandatory, extra, limit_seconds=13)["passed"]


@pytest.mark.parametrize("action", ([0, 1.1, 0], [2, 0, 0], [0, 0, -0.1], [0, np.nan, 0], [0, 0]))
def test_invalid_actions_are_rejected_without_clipping(action):
    with pytest.raises(ValueError):
        validate_action(action)


def test_valid_action_is_float32_and_unmodified():
    action = validate_action([-0.4, 0.6, 0.2])
    np.testing.assert_array_equal(action, np.asarray([-0.4, 0.6, 0.2], np.float32))


def development_report(source="candidate"):
    declared = [*MANDATORY_CELLS, *((track, seed) for seed in DEVELOPMENT_SEEDS for track in range(1, 5))]
    rows = [row(*cell) for cell in declared]
    rows[0] = row(*MANDATORY_CELLS[0], finished=False)
    return {"freeze": {"suite": "development", "cells": [list(cell) for cell in declared],
                       "trial_role": "candidate", "source_sha256": source, "parameters": {},
                       "selected_profile_seconds": 13}, "rows": rows}


def test_only_distinct_complete_development_rejections_relax_the_next_trial():
    reports = [development_report(f"candidate-{i}") for i in range(6)]
    for report in reports[3:]:
        report["freeze"]["selected_profile_seconds"] = 15
    reports[0]["rows"].reverse()  # Completion is independent of receipt row order.
    assert rejection_streak(reports[:3]) == 3
    assert pace_profile(rejection_streak(reports[:3])) == 15
    assert rejection_streak(reports) == 6
    assert pace_profile(rejection_streak(reports)) == 18
    with pytest.raises(ValueError, match="repeat"):
        rejection_streak(reports + [reports[0]])
    reports[-1]["freeze"]["selected_profile_seconds"] = 18
    with pytest.raises(ValueError, match="prospective"):
        rejection_streak(reports)


@pytest.mark.parametrize("suite", ("mandatory", "holdout"))
def test_non_development_receipts_cannot_relax_pace(suite):
    report = development_report()
    report["freeze"]["suite"] = suite
    with pytest.raises(ValueError, match="development"):
        rejection_streak([report])


def test_development_rejection_requires_exact_declared_and_measured_cells():
    report = development_report()
    report["freeze"]["cells"].pop()
    with pytest.raises(ValueError, match="development cells"):
        rejection_streak([report])
    report = development_report()
    report["rows"].pop()
    with pytest.raises(ValueError, match="development cells"):
        rejection_streak([report])
    report = development_report()
    report["rows"][-1]["seed"] += 1
    with pytest.raises(ValueError, match="development cells"):
        rejection_streak([report])


def test_operational_error_cannot_count_as_development_rejection():
    report = development_report()
    report["rows"][0]["error"] = "worker timed out"
    with pytest.raises(ValueError, match="operational"):
        rejection_streak([report])


def test_playfield_termination_is_not_a_time_limit():
    assert end_reason({}, finished=False, terminated=True, truncated=False) == "terminated"
    assert end_reason({}, finished=False, terminated=False, truncated=True) == "time_limit"
    assert end_reason({"retire_reason": "crash"}, finished=False, terminated=True, truncated=False) == "crash"
    assert end_reason({}, finished=True, terminated=False, truncated=True) is None


def test_worker_timeout_and_bad_json_are_recorded_as_operational_errors():
    measured, error = execute_worker([sys.executable, "-c", "import time; time.sleep(1)"], timeout=.05)
    assert measured is None and "timed out" in error
    measured, error = execute_worker([sys.executable, "-c", "print('broken receipt')"])
    assert measured is None and "JSON" in error
    measured, error = execute_worker([sys.executable, "-c", "print('{\"finished\":true}')"])
    assert measured == {"finished": True} and error is None


def test_baseline_results_cannot_relax_candidate_criteria():
    report = development_report("control")
    report["freeze"]["trial_role"] = "benchmark"
    with pytest.raises(ValueError, match="benchmark"):
        rejection_streak([report])


def test_track_floor_catches_a_concentrated_failure_with_good_aggregate():
    mandatory, _ = cells()
    extra = [row(track, seed) for track in range(1, 5) for seed in range(92000, 92010)]
    for i in range(3):
        extra[i] = row(1, 92000 + i, finished=False)
    verdict = assess(mandatory, extra, limit_seconds=13)
    assert verdict["extra_fast_fraction"] == .925
    assert not verdict["passed"]


def test_mandatory_cli_creates_fresh_source_bound_receipt(tmp_path, monkeypatch):
    source = tmp_path / "candidate.py"
    source.write_text("class Agent: pass\n", encoding="utf-8")
    output = tmp_path / "mandatory.json"
    calls = []

    def fake_worker(command):
        payload = json.loads(command[-1])
        calls.append((payload["track"], payload["seed"]))
        return {**row(payload["track"], payload["seed"]),
                "source_sha256": evaluate.digest(source), "parameters": {}, "error": None}, None

    monkeypatch.setattr(evaluate, "ENV_FILES", ())
    monkeypatch.setattr(evaluate, "execute_worker", fake_worker)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--source", str(source),
                                      "--suite", "mandatory", "--output", str(output)])
    evaluate.main()
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["freeze"]["source_sha256"] == evaluate.digest(source)
    assert receipt["freeze"]["runtime_versions"]["box2d-py"] == version("box2d-py")
    assert receipt["freeze"]["cells"] == [list(cell) for cell in MANDATORY_CELLS]
    assert {tuple((r["track_id"], r["seed"])) for r in receipt["rows"]} == set(MANDATORY_CELLS)
    assert receipt["screen_passed"] is True and "verdict" not in receipt
    assert len(list(output.with_suffix(".cells").glob("*.json"))) == 4
    assert len(calls) == 4
    original = output.read_bytes()
    with pytest.raises(SystemExit):
        evaluate.main()
    assert output.read_bytes() == original
    assert len(calls) == 4
