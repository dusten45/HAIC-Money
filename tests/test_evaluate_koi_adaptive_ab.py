import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.evaluate_koi_adaptive_ab import SEEDS, run_child, scheduled_slots
from scripts import evaluate_koi_adaptive_ab as operator


def test_all_slots_are_frozen_and_include_both_arms():
    slots = scheduled_slots()
    assert len(slots) == 48
    assert {s["seed"] for s in slots} == set(SEEDS)
    assert len({(s["track_id"], s["seed"], s["mode"]) for s in slots}) == 48
    assert all(s["status"] == "unrun" for s in slots)
    for a, b in zip(slots[::2], slots[1::2]):
        assert (a["track_id"], a["seed"]) == (b["track_id"], b["seed"])
        assert {a["mode"], b["mode"]} == {"crossing_projection", "adaptive_v1"}


@pytest.mark.parametrize("observer", [None, SimpleNamespace(raw_count=0, last={}),
                                     SimpleNamespace(raw_count=0, last={}, initial={"t": 1.02})])
def test_zero_interaction_or_zero_decision_error_telemetry_is_safe(observer):
    result = operator.observer_measurements(observer)
    assert result["raw_ticks"] == 0
    assert result["simulation_end"] is None
    assert result["catalog"] is None
    assert result["simulation_start"] == getattr(observer, "initial", {}).get("t")


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "none"])
def test_child_receipts_preserve_partial_files(tmp_path, monkeypatch, failure):
    output = tmp_path / "episode.json"
    partial = output.with_suffix(".raw.jsonl")
    partial.write_text('partial evidence\n')

    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], b"progress", b"diagnostic")
        return subprocess.CompletedProcess(args[0], 1 if failure == "nonzero" else 0,
                                           "progress", "diagnostic")

    monkeypatch.setattr(subprocess, "run", run)
    if failure == "none":
        run_child(["unused"], tmp_path, output, 90)
    else:
        with pytest.raises(RuntimeError, match="child_"):
            run_child(["unused"], tmp_path, output, 90)
    assert partial.read_text() == 'partial evidence\n'
    assert output.with_suffix(".stdout.txt").read_text() == "progress"
    assert output.with_suffix(".stderr.txt").read_text() == "diagnostic"
    receipt = json.loads(output.with_suffix(".process.json").read_text())
    assert receipt["status"] == ("completed" if failure == "none" else "operator_error")
    assert receipt["wall_time_s"] >= 0


@pytest.mark.parametrize("retire_reason", [None, "act_timeout"])
def test_full_operator_mock_children_no_environment(tmp_path, monkeypatch, retire_reason):
    def child(command, directory, output, timeout):
        track, seed, arm = output.stem.split("-", 2)
        operator.save(output, dict(
            mode=arm, track_id=int(track), seed=int(seed), completed=True,
            lapTimeMs=1000, progress=1.0, damage=0.0, collisions=0,
            retire_reason=retire_reason, error=None, invalid_actions=0, steps=0,
            raw_ticks=0, peak_rss_bytes=0, geometry_sha256="same",
            initial_observation_sha256="same", decision_trace=[],
        ))

    monkeypatch.setattr(operator, "run_child", child)
    destination = tmp_path / "mock-run"
    if retire_reason is None:
        operator.run(destination)
    else:
        with pytest.raises(AssertionError):
            operator.run(destination)
    report = json.loads((destination / "episode-report.json").read_text())
    assert len(report["rows"]) == 48
    if retire_reason is None:
        assert all(row["status"] == "completed" for row in report["rows"])
        assert report["operator_error"] is None
        assert len((destination / "reset-ledger.jsonl").read_text().splitlines()) == 96
    else:
        assert report["rows"][0]["retire_reason"] == "act_timeout"
        assert report["rows"][0]["status"] == "completed"
        assert sum(row["status"] == "unrun" for row in report["rows"]) == 47
        assert report["operator_error"] is not None


def test_failed_operator_keeps_all_slots_no_environment(tmp_path, monkeypatch):
    def child(*args):
        raise RuntimeError("mock timeout")

    monkeypatch.setattr(operator, "run_child", child)
    destination = tmp_path / "mock-failed"
    with pytest.raises(RuntimeError, match="mock timeout"):
        operator.run(destination)
    report = json.loads((destination / "episode-report.json").read_text())
    assert len(report["rows"]) == 48
    assert report["rows"][0]["status"] == "operator_error"
    assert sum(row["status"] == "unrun" for row in report["rows"]) == 47
    assert report["operator_error"] == "RuntimeError: mock timeout"
    assert (destination / "operator-failure.json").exists()
