"""Zero-reset operator helper tests; real worker is never called."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import evaluate_koi_steering_release_ab as operator


def test_exact24_consumed_balanced48_schedule_and_protected_exclusions():
    rows = operator.scheduled_slots()
    assert len(rows) == len({(r["track_id"], r["seed"], r["mode"]) for r in rows}) == 48
    assert {r["seed"] for r in rows} == set(operator.SEEDS)
    assert {r["track_id"] for r in rows} == {1, 2, 3}
    assert sum(r["mode"] == operator.ARMS[0] for r in rows[::2]) == 12
    assert not set(operator.PROTECTED) & {r["seed"] for r in rows}


def test_passive_pre_act_state_retains_expired_recovery_hold_without_mutation():
    inner = SimpleNamespace(base=SimpleNamespace(_obstacle_side=1), steps=20, impact_left=1)
    outer = SimpleNamespace(driver=SimpleNamespace(driver=inner))
    original, before = operator.controller_state(outer)
    assert original is inner
    assert before == dict(obstacle_side=1., steps=20, impact_left=1)
    assert inner.impact_left == 1
    # Emulate a source controller decrement, not a real Agent action.
    inner.impact_left = 0
    inner.steps = 21
    _, after = operator.controller_state(outer)
    assert before["impact_left"] == 1 and after["impact_left"] == 0
    assert before["steps"] == 20 and after["steps"] == 21


def test_import_and_main_help_do_not_import_environment_or_construct_candidate():
    source = ast.parse(Path(operator.__file__).read_text())
    for statement in source.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            module = statement.module if isinstance(statement, ast.ImportFrom) else statement.names[0].name
            assert module is not None and not module.startswith(("training", "gymnasium", "Box2D", "pygame", "haic.algorithms.koi.steering_release"))


def test_skin_inclusive_whole_wheel_lateral_separation_observer_only():
    hull = np.array([[-1., -2.], [1., -2.], [1., 2.], [-1., 2.]])
    wheel = np.array([[1.2, 1.], [1.8, 1.], [1.8, 2.], [1.2, 2.]])
    centers = np.array([[3., 1.5], [0., 50.], [1.8, 1.5]])
    tangents = np.array([[0., 1.]] * 3)
    separated = operator.lateral_separation([hull, wheel], [.01, .02], centers, tangents, np.array([.5] * 3))
    assert separated == pytest.approx([.68, -1.51, -.52])
    # Being far ahead does not create lateral separation for the centered object.
    assert separated[1] < 0


def test_forecast_empirical_fixed_wall_limit_not_arbitrary_hardcoded_memory():
    evidence = dict(prior_peak_rss_bytes=285000000, prior_total_child_wall_s=960,
                    prior_max_child_wall_s=25, prior_max_episode_bytes=1024)
    forecast = operator.resource_forecast(evidence)
    assert forecast["peak_child_bytes"] == 427500000
    assert forecast["wall_budget_s"] == 1800
    assert forecast["approximate_reference_s_per_episode"] == 20
    assert forecast["empirical_reference_peak_decision_s"] == .045
    assert forecast["decision_budget_s"] == 4.5
    assert not forecast["gpu"] and forecast["serial_children"] == 1
    evidence["prior_total_child_wall_s"] = 1300
    with pytest.raises(AssertionError, match="wall forecast"):
        operator.resource_forecast(evidence)


@pytest.mark.parametrize("defect", ["host", "cgroup", "disk", "temp", None])
def test_resource_checks_real_measured_headroom_contract(defect):
    forecast = dict(peak_child_bytes=100, memory_reserve_bytes=25, per_episode_disk_bytes=50, disk_reserve_bytes=20, temp_bytes=100)
    measurement: dict[str, Any] = dict(host_mem_available_bytes=1000, visible_cgroup_ancestors=[dict(raw_headroom_bytes=None), dict(raw_headroom_bytes=1000)],
                       disk=[dict(device=1, free_bytes=1000), dict(device=2, free_bytes=1000)])
    if defect == "host":
        measurement["host_mem_available_bytes"] = 125
    elif defect == "cgroup":
        measurement["visible_cgroup_ancestors"][1]["raw_headroom_bytes"] = 125
    elif defect in ("disk", "temp"):
        measurement["disk"][0 if defect == "disk" else 1]["free_bytes"] = 119
    if defect:
        with pytest.raises(AssertionError):
            operator.check_resources(measurement, forecast, 2)
    else:
        operator.check_resources(measurement, forecast, 2)


def test_existing_protected_claims_untouched_consumed_claim_collision_closed(tmp_path):
    protected = tmp_path / "seed-49300.json"
    protected.write_text('{"protected":true}')
    before = protected.read_bytes()
    pins = operator.check_claims(tmp_path)
    assert pins[str(protected)] == operator.sha(protected)
    assert protected.read_bytes() == before
    (tmp_path / "seed-38301.json").write_text("{}")
    with pytest.raises(AssertionError, match="active claim"):
        operator.check_claims(tmp_path)


@pytest.mark.parametrize("failure", [False, True])
def test_actual_process_receipt_hash_binding_even_timeout(tmp_path, monkeypatch, failure):
    def child(command, directory, output, timeout):
        operator.save(output.with_suffix(".process.json"), dict(status="operator_error" if failure else "completed",
                                                              error="child_timeout" if failure else None, wall_time_s=.01))
        if failure:
            output.with_suffix(".raw.jsonl").write_text('{"partial":')
            raise RuntimeError("child_timeout")

    monkeypatch.setattr(operator.original, "run_child", child)
    output = tmp_path / "episode.json"
    if failure:
        with pytest.raises(RuntimeError):
            operator.run_child([], tmp_path, output, 1, "1:38300:crossing_projection", "sourcehash")
    else:
        operator.run_child([], tmp_path, output, 1, "1:38300:crossing_projection", "sourcehash")
    receipt = json.loads(output.with_suffix(".bound-process.json").read_text())
    assert receipt["slot_id"] == "1:38300:crossing_projection"
    assert receipt["operator_sha256"] == "sourcehash"
    assert receipt["original_process_sha256"] == operator.sha(output.with_suffix(".process.json"))
    if failure:
        assert output.with_suffix(".raw.jsonl").exists()


@pytest.mark.parametrize("failure", [None, "timeout", "short_prefix", "censor"])
def test_full_operator_mock_freezes_source_schedule_and_partial_evidence(tmp_path, monkeypatch, failure):
    baseline_path = operator.ROOT / "runs/koi-minimum-clearance-ab-20260930-r3/crossing-projection-source-reconstruction.zip"
    with zipfile.ZipFile(baseline_path) as archive:
        baseline_files = {n: archive.read(n) for n in archive.namelist()}
    root = tmp_path / "project"
    (root / "experiments/train-seed-claims").mkdir(parents=True)
    (root / "scripts").mkdir()
    helpers = [root / "scripts/analyze_koi_steering_release_ab.py", root / "scripts/evaluate_koi_steering_release_ab.py",
               root / "haic/algorithms/koi/steering_terms.py", root / "haic/algorithms/koi/steering_release.py"]
    for path in helpers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# mocked, never imported\n")
    candidate = root / "candidate.zip"
    files = dict(baseline_files)
    files.update({"haic_agent/steering_terms.py": helpers[2].read_bytes(), "haic_agent/steering_release_runtime.py": helpers[3].read_bytes()})
    with zipfile.ZipFile(candidate, "w") as archive:
        for n, d in files.items():
            archive.writestr(n, d)
    candidate.with_suffix(".manifest.json").write_text(json.dumps(dict(candidate="koi-steering-release-synthetic", candidate_zip_sha256=operator.sha(candidate),
                                                                     baseline_source_sha256={n: hashlib.sha256(d).hexdigest() for n, d in baseline_files.items()},
                                                                     files=[dict(path=n, sha256=hashlib.sha256(d).hexdigest()) for n, d in files.items()])))
    monkeypatch.setattr(operator, "ROOT", root)
    monkeypatch.setattr(operator, "helper_paths", lambda: helpers)
    monkeypatch.setattr(operator, "frozen_baseline_files", lambda: baseline_files)
    monkeypatch.setattr(operator, "consumed_evidence", lambda: dict(source_sha256={}, prior_peak_rss_bytes=285000000,
                                                                  prior_total_child_wall_s=960, prior_max_child_wall_s=25, prior_max_episode_bytes=1024))

    def process(command, **kwargs):
        if command[0] == "git":
            relative = command[-1].split("snapshots/haic-local-20260930/workspace/", 1)[1]
            return subprocess.CompletedProcess(command, 0, stdout=(operator.SNAPSHOT / relative).read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(dict(torch="2.1.0+cpu", synthetic=True)))

    def child(command, directory, output, timeout):
        if failure == "timeout":
            output.with_suffix(".raw.jsonl").write_text('{"partial":')
            operator.save(output.with_suffix(".process.json"), dict(status="operator_error", error="timeout", wall_time_s=.01))
            raise RuntimeError("mock timeout")
        protocol = json.loads((output.parent / "protocol.json").read_text())
        track, seed, arm = output.stem.split("-", 2)
        count = 0 if failure == "short_prefix" else 10
        operator.save(output, dict(mode=arm, track_id=int(track), seed=int(seed), completed=True, lapTimeMs=1000,
                                   progress=1., damage=0., collisions=0, error=None, invalid_actions=0,
                                   retire_reason="max_steps" if failure == "censor" else None,
                                   steps=count, raw_ticks=0, peak_rss_bytes=285000000,
                                   initial_state={}, geometry_sha256="same", initial_observation_sha256="same",
                                   decision_trace=[dict(steer=0, gas=0, brake=0, car_x=0, car_y=0, car_yaw=0, progress=0)] * count,
                                   versions=protocol["runtime_versions"], damage_telemetry_valid=True, collision_telemetry_valid=True))
        operator.save(output.with_suffix(".process.json"), dict(status="completed", error=None, wall_time_s=.01))

    monkeypatch.setattr(operator.subprocess, "run", process)
    monkeypatch.setattr(operator.original, "run_child", child)
    destination = root / "mock-run"
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            operator.run(destination, candidate, study="synthetic-no-resets")
    else:
        operator.run(destination, candidate, study="synthetic-no-resets")
    protocol = json.loads((destination / "protocol.json").read_text())
    report = json.loads((destination / "episode-report.json").read_text())
    assert protocol["schedule"] == operator.scheduled_slots()
    assert len(protocol["environment_source_sha256"]) == 142
    assert protocol["study"] == "synthetic-no-resets"
    assert protocol["resource_forecast"]["wall_budget_s"] == 1800
    assert all(operator.sha(destination / copy["file"]) == copy["sha256"] for copy in protocol["source_copies"])
    assert report["reset_ledger_sha256"] == operator.sha(destination / "reset-ledger.jsonl")
    if failure:
        assert report["operator_error"] and (destination / "operator-failure.json").exists()
        if failure == "timeout":
            assert (destination / "1-38300-crossing_projection.raw.jsonl").exists()
            assert (destination / "1-38300-crossing_projection.bound-process.json").exists()
    else:
        assert all(row["status"] == "completed" for row in report["rows"])
        assert all(operator.sha(destination / row["process_file"]) == row["process_sha256"] for row in report["rows"])
        assert len((destination / "reset-ledger.jsonl").read_text().splitlines()) == 96
