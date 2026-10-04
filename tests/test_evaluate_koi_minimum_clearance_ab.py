"""Operator safety checks without constructing/resetting any environment."""

import hashlib
import json
import zipfile
from typing import Any

import numpy as np
import pytest

from scripts import evaluate_koi_minimum_clearance_ab as operator


def test_exact_consumed_schedule_and_balanced_arm_order():
    rows = operator.scheduled_slots()
    assert len(rows) == 48
    assert len({(r["track_id"], r["seed"], r["mode"]) for r in rows}) == 48
    assert {r["seed"] for r in rows} == set(operator.SEEDS)
    assert {r["track_id"] for r in rows} == {1, 2, 3}
    assert sum(r["mode"] == operator.ARMS[0] for r in rows[::2]) == 12
    assert not {49300, 51300} & {r["seed"] for r in rows}


def test_clearance_includes_polygon_skin_wheels_and_signed_containment():
    hull = np.asarray([[-1., -2.], [1., -2.], [1., 2.], [-1., 2.]])
    wheel = np.asarray([[1.2, 1.], [1.8, 1.], [1.8, 2.], [1.2, 2.]])
    centers = np.asarray([[3., 1.5], [0., 0.], [1.8, 1.5]])
    normals = np.asarray([[0., 1.]] * 3)
    radii = np.asarray([.5, .5, .5])
    clearance, rear, front = operator.footprint_measurements([hull, wheel], [.01, .02], centers, normals, radii)
    assert clearance == pytest.approx([.68, -1.51, -.52])
    assert rear == pytest.approx([-3.51, -2.01, -3.51])
    assert front == pytest.approx([.52, 2.02, .52])


def test_forecast_from_actual_prior_peak_duration_and_bytes():
    forecast = operator.resource_forecast(dict(prior_peak_rss_bytes=284217344, prior_total_child_wall_s=1000,
                                             prior_max_child_wall_s=30, prior_max_episode_bytes=1024))
    assert forecast["peak_child_bytes"] == 426326016
    assert forecast["wall_budget_s"] == 1800
    assert forecast["child_timeout_s"] == 90
    assert forecast["serial_children"] == 1 and not forecast["gpu"]


@pytest.mark.parametrize("defect", [None, "host", "ancestor", "disk", "temp"])
def test_resource_checks_all_finite_ancestors_and_distinct_filesystems(defect):
    forecast = dict(peak_child_bytes=100, memory_reserve_bytes=25, per_episode_disk_bytes=50,
                    disk_reserve_bytes=20, temp_bytes=100)
    measurement: dict[str, Any] = dict(host_mem_available_bytes=1000,
                       visible_cgroup_ancestors=[dict(raw_headroom_bytes=None), dict(raw_headroom_bytes=1000)],
                       disk=[dict(device=1, free_bytes=1000), dict(device=2, free_bytes=1000)])
    if defect == "host":
        measurement["host_mem_available_bytes"] = 125
    elif defect == "ancestor":
        measurement["visible_cgroup_ancestors"][1]["raw_headroom_bytes"] = 124
    elif defect == "disk":
        measurement["disk"][0]["free_bytes"] = 119
    elif defect == "temp":
        measurement["disk"][1]["free_bytes"] = 119
    if defect:
        with pytest.raises(AssertionError):
            operator.check_resources(measurement, forecast, 2)
    else:
        operator.check_resources(measurement, forecast, 2)


def test_claim_checks_fail_closed_for_any_track_seed(tmp_path):
    operator.check_claims(tmp_path)
    (tmp_path / "seed-38301.json").write_text("{}")
    with pytest.raises(AssertionError, match="active claim"):
        operator.check_claims(tmp_path)


@pytest.fixture
def candidate(tmp_path):
    files = dict(operator.frozen_baseline_files())
    baseline = {n: hashlib.sha256(d).hexdigest() for n, d in files.items()}
    files["haic_agent/minimum_clearance_runtime.py"] = b"# synthetic never executed\n"
    path = tmp_path / "candidate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    manifest = dict(candidate="koi-minimum-clearance-v1", candidate_zip_sha256=operator.sha(path),
                    baseline_source_sha256=baseline, files=[dict(path=n, sha256=hashlib.sha256(d).hexdigest()) for n, d in files.items()])
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("failure", [None, "timeout", "operational"])
def test_source_freeze_full_mock_operator_and_partial_receipts(tmp_path, monkeypatch, candidate, failure):
    def child(command, directory, output, timeout):
        if failure == "timeout":
            output.with_suffix(".raw.jsonl").write_text('{"partial":')
            raise RuntimeError("mock_timeout")
        protocol = json.loads((output.parent / "protocol.json").read_text())
        track, seed, arm = output.stem.split("-", 2)
        operator.save(output, dict(mode=arm, track_id=int(track), seed=int(seed), completed=True,
                                  lapTimeMs=1000, progress=1., damage=0., collisions=0,
                                  retire_reason="act_timeout" if failure == "operational" else None,
                                  error=None, invalid_actions=0, steps=10, raw_ticks=0, peak_rss_bytes=0,
                                  geometry_sha256="same", initial_observation_sha256="same", initial_state={},
                                  decision_trace=[dict(steer=0, gas=0, brake=0, car_x=0, car_y=0, car_yaw=0, progress=0)] * 10,
                                  versions=protocol["runtime_versions"], damage_telemetry_valid=True, collision_telemetry_valid=True))

    monkeypatch.setattr(operator.original, "run_child", child)
    destination = tmp_path / "mock-run"
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            operator.run(destination, candidate, study="mock-no-resets")
    else:
        operator.run(destination, candidate, study="mock-no-resets")
    report = json.loads((destination / "episode-report.json").read_text())
    protocol = json.loads((destination / "protocol.json").read_text())
    assert len(report["rows"]) == 48
    assert protocol["study"] == "mock-no-resets"
    assert protocol["scope"].startswith("explicit consumed TRAIN")
    assert len(protocol["environment_source_sha256"]) == 142
    assert len(protocol["helper_source_sha256"]) >= 5
    assert all(operator.sha(destination / r["file"]) == r["sha256"] for r in protocol["source_copies"])
    assert report["reset_ledger_sha256"] == operator.sha(destination / "reset-ledger.jsonl")
    assert protocol["candidate_manifest_sha256"] == operator.sha(candidate.with_suffix(".manifest.json"))
    if failure:
        assert sum(r["status"] == "unrun" for r in report["rows"]) == 47
        assert report["operator_error"]
        assert (destination / "operator-failure.json").exists()
        if failure == "timeout":
            assert (destination / "1-38300-crossing_projection.raw.jsonl").exists()
    else:
        assert all(r["status"] == "completed" for r in report["rows"])
        assert len((destination / "reset-ledger.jsonl").read_text().splitlines()) == 96
        assert report["operator_error"] is None
