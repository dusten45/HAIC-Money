"""Zero-reset tests: real workers, Agents and environment constructors never run."""

import ast
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import evaluate_koi_steering_generalization as operator


def test_fixed_uint32_24_roads_72_cells_144_balanced_slots():
    rows = operator.scheduled_slots()
    assert operator.STUDY == "koi-steering-generalization-v1"
    assert operator.SEEDS == tuple(range(3184000001, 3184000025))
    assert all(0 <= s < 2**32 for s in operator.SEEDS)
    assert len(operator.CELLS) == 72
    assert len(rows) == len({(r["track_id"], r["seed"], r["mode"]) for r in rows}) == 144
    assert sum(r["mode"] == operator.ARMS[0] for r in rows[::2]) == 36
    assert not {s for _, s in operator.CELLS} & (set(operator.PROTECTED) | set(operator.original.SEEDS))
    assert rows == operator.freshness.scheduled_slots()


def test_dynamic_wall_forecast_no_old_48_or_1800_floor():
    evidence = dict(prior_peak_rss_bytes=285000000, prior_total_child_wall_s=960,
                    reference_episodes=48, prior_max_child_wall_s=25, prior_max_episode_bytes=1024)
    forecast = operator.resource_forecast(evidence)
    assert forecast["wall_budget_s"] == 20 * 144 * 1.6
    assert forecast["approximate_reference_s_per_episode"] == 20
    assert forecast["episodes"] == 144 and forecast["serial_children"] == 1 and not forecast["gpu"]
    assert forecast["peak_child_bytes"] == 427500000
    evidence.update(prior_total_child_wall_s=48, reference_episodes=48)
    assert operator.resource_forecast(evidence)["wall_budget_s"] == 144 * 1.6


def test_import_and_help_zero_environment_policy_construction():
    code = "import sys;from scripts import evaluate_koi_steering_generalization;assert not any(n.startswith(('training.','gymnasium','Box2D','pygame')) for n in sys.modules);assert 'agent' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=operator.ROOT, check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "scripts.evaluate_koi_steering_generalization", "--help"],
                            cwd=operator.ROOT, check=True, capture_output=True, text=True)
    assert "--freeze-protocol" in result.stdout and "--preflight" in result.stdout and "--run" in result.stdout
    assert "--candidate" not in result.stdout


def test_read_only_r2_reference_hashes_and_transitive_source_closure():
    evidence = operator.reference_evidence()
    assert evidence["reference_episodes"] == 48
    assert evidence["prior_total_child_wall_s"] > 0
    assert len(evidence["prior_road_sha256"]) == 8
    assert operator.sha(operator.ROOT / operator.R2_PATH / "protocol.json") == operator.R2_PROTOCOL_SHA
    assert operator.sha(operator.ROOT / "agent.py") == operator.ROOT_AGENT_SHA
    assert all(operator.sha(p) == h for p, h in evidence["source_sha256"].items())
    names = {p.relative_to(operator.ROOT).as_posix() for p in operator.helper_paths()}
    assert set(operator.freshness.IMPLEMENTATION) - {"docs/evaluation/generalization-policy.md"} <= names
    assert "scripts/evaluate_koi_steering_release_ab.py" in names
    assert "scripts/analyze_koi_steering_release_ab.py" in names


def test_passive_controller_and_skin_inclusive_fixture_measurement():
    inner = SimpleNamespace(base=SimpleNamespace(_obstacle_side=1), steps=20, impact_left=1)
    driver, state = operator.controller_state(SimpleNamespace(driver=inner))
    assert driver is inner and state == dict(obstacle_side=1., steps=20, impact_left=1)
    assert inner.impact_left == 1
    hull = np.array([[-1., -2.], [1., -2.], [1., 2.], [-1., 2.]])
    wheel = np.array([[1.2, 1.], [1.8, 1.], [1.8, 2.], [1.2, 2.]])
    centers = np.array([[3., 1.5], [0., 50.], [1.8, 1.5]])
    tangents = np.array([[0., 1.]] * 3)
    assert operator.lateral_separation([hull, wheel], [.01, .02], centers, tangents, np.array([.5] * 3)) == pytest.approx([.68, -1.51, -.52])


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    real_root = operator.ROOT
    root = tmp_path / "project"
    (root / "runs").mkdir(parents=True)
    (root / "experiments/train-seed-claims").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "agent.py").write_bytes((real_root / "agent.py").read_bytes())
    r2 = root / operator.R2_PATH
    r2.mkdir()
    for name in ("protocol.json", "candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"):
        (r2 / name).write_bytes((real_root / operator.R2_PATH / name).read_bytes())
    helpers = [root / "scripts/evaluate_koi_steering_generalization.py", root / "scripts/analyze_koi_steering_generalization.py",
               root / "haic/algorithms/koi/steering_terms.py", root / "haic/algorithms/koi/steering_release.py"]
    for path in helpers:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((real_root / path.relative_to(root)).read_bytes())
    exposure_path = root / operator.freshness.AUDIT_PATH
    operator.save(exposure_path, dict(status="clear", synthetic=True))
    exposure = operator.read_json(exposure_path)
    protected = root / "experiments/train-seed-claims/seed-49300.json"
    protected.write_bytes(b'{"synthetic_protected":true}')
    calls = []

    def load_audit(path, expected_sha256, *, root):
        assert Path(path) == exposure_path and operator.sha(path) == expected_sha256
        return exposure

    def verify_claims(receipt, *, root):
        assert receipt == exposure
        calls.append(("verify", None, None))

    def audit(*, root, authenticated_receipt, self_protocol_sha256=None):
        assert authenticated_receipt == exposure
        calls.append(("audit", self_protocol_sha256, None))
        return dict(status="clear")

    def audit_cell(receipt, cell, *, root, self_protocol_sha256):
        assert receipt == exposure and cell in operator.freshness.proposed_cells()
        ledger = root / operator.RUN_PATH / "reset-ledger.jsonl"
        events = ledger.read_text().splitlines() if ledger.exists() else []
        calls.append(("cell", self_protocol_sha256, cell))
        assert len(events) % 2 == 0  # parent audit is before its next intent
        lock = operator.os.open(root / "experiments/train-seed-claims", operator.os.O_RDONLY | operator.os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            operator.os.close(lock)
        return dict(status="clear")

    monkeypatch.setattr(operator, "ROOT", root)
    monkeypatch.setattr(operator, "__file__", str(helpers[0]))
    monkeypatch.setattr(operator, "helper_paths", lambda: helpers)
    monkeypatch.setattr(operator.freshness, "load_audit", load_audit)
    monkeypatch.setattr(operator.freshness, "verify_claims", verify_claims)
    monkeypatch.setattr(operator.freshness, "audit", audit)
    monkeypatch.setattr(operator.freshness, "audit_cell", audit_cell)
    monkeypatch.setattr(operator, "reference_evidence", lambda: dict(source_sha256={str(r2 / n): operator.sha(r2 / n) for n in
                        ("protocol.json", "candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip")},
                        reference_episodes=48, prior_peak_rss_bytes=285000000, prior_total_child_wall_s=960,
                        prior_max_child_wall_s=25, prior_max_episode_bytes=1024, prior_road_sha256=["old-road"]))
    monkeypatch.setattr(operator, "runtime_versions", lambda: dict(torch="2.1.0+cpu", synthetic=True))
    monkeypatch.setattr(operator, "resource_measurements", lambda destination: dict(host_mem_available_bytes=10**12,
                        visible_cgroup_ancestors=[dict(raw_headroom_bytes=10**12)],
                        disk=[dict(device=1, free_bytes=10**12), dict(device=1, free_bytes=10**12)]))
    result = operator.freeze_protocol(exposure_path, operator.sha(exposure_path))
    destination = root / operator.RUN_PATH
    protocol = operator.read_json(destination / "protocol.json")
    review_path = root / "experiments/independent-review.json"
    operator.save(review_path, dict(status="passed", environment_resets=0, reviewer="independent-synthetic-review",
                                  protocol_sha256=result["protocol_sha256"], operator_sha256=protocol["operator_sha256"],
                                  analyzer_sha256=protocol["analyzer_sha256"], audit_sha256=operator.sha(exposure_path)))
    return SimpleNamespace(root=root, destination=destination, protocol=protocol, protocol_sha=result["protocol_sha256"],
                           review_path=review_path, review_sha=operator.sha(review_path), calls=calls,
                           protected=protected, exposure=exposure, helpers=helpers)


def test_freeze_and_preflight_no_worker_byte_identical_models_root_and_protocol(frozen_run, monkeypatch):
    monkeypatch.setattr(operator, "worker", lambda *args: pytest.fail("worker called by zero-reset preflight"))
    run = frozen_run
    assert operator.preflight(run.protocol_sha)["environment_resets"] == 0
    assert (run.root / operator.PROTOCOL_PATH).read_bytes() == (run.destination / "protocol.json").read_bytes()
    for name in ("candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"):
        assert (run.destination / name).read_bytes() == (run.root / operator.R2_PATH / name).read_bytes()
    assert len(run.protocol["environment_source_sha256"]) == 142
    assert run.protocol["episodes"] == 144
    assert run.protocol["resource_forecast"]["wall_budget_s"] == 4608
    assert run.protected.read_bytes() == b'{"synthetic_protected":true}'
    assert not (run.destination / "reset-ledger.jsonl").exists()
    assert all(operator.sha(run.destination / c["file"]) == c["sha256"] for c in run.protocol["source_copies"])
    with pytest.raises(AssertionError, match="overwrite"):
        operator.freeze_protocol(run.root / operator.freshness.AUDIT_PATH, "unused")


@pytest.mark.parametrize("target", ["root_protocol", "candidate", "manifest", "baseline", "helper", "source_copy", "root_agent", "reference"])
def test_preflight_rejects_tampered_source_models_protocol_and_root(frozen_run, target):
    run = frozen_run
    paths = dict(root_protocol=run.root / operator.PROTOCOL_PATH, candidate=run.destination / "candidate.zip",
                 manifest=run.destination / "candidate.manifest.json", baseline=run.destination / "crossing-projection-source-reconstruction.zip",
                 helper=run.helpers[1], source_copy=run.destination / run.protocol["source_copies"][0]["file"],
                 root_agent=run.root / "agent.py", reference=run.root / operator.R2_PATH / "candidate.zip")
    path = paths[target]
    path.write_bytes(path.read_bytes() + b"\n# tampered\n")
    with pytest.raises(AssertionError):
        operator.preflight(run.protocol_sha)


@pytest.mark.parametrize("artifact", ["reset-ledger.jsonl", "partial.raw.jsonl", "episode-report.json", "execution-review.json"])
def test_partial_attempt_is_not_preflighted_as_fresh_or_resumable(frozen_run, artifact):
    (frozen_run.destination / artifact).write_bytes(b"")
    with pytest.raises(AssertionError):
        operator.preflight(frozen_run.protocol_sha)


@pytest.mark.parametrize("key", ["status", "environment_resets", "reviewer", "protocol_sha256", "operator_sha256", "analyzer_sha256", "audit_sha256"])
def test_run_requires_independent_exact_protocol_review_before_any_intent(frozen_run, key):
    run = frozen_run
    review = operator.read_json(run.review_path)
    review[key] = 1 if key == "environment_resets" else ""
    run.review_path.write_text(json.dumps(review))
    with pytest.raises(AssertionError):
        operator.run(run.protocol_sha, run.review_path, operator.sha(run.review_path))
    assert not (run.destination / "reset-ledger.jsonl").exists()


@pytest.mark.parametrize("failure", [None, "timeout", "short_prefix", "censor", "road", "oldroad", "audit"])
def test_mock_serial_run_all_144_slots_or_preserved_partial_never_reset(frozen_run, monkeypatch, failure):
    run = frozen_run
    count = 0

    if failure == "audit":
        def blocked(*args, **kwargs):
            raise AssertionError("fresh collision")
        monkeypatch.setattr(operator.freshness, "audit_cell", blocked)

    def child(command, directory, output, timeout):
        nonlocal count
        count += 1
        assert "scripts.evaluate_koi_steering_generalization" in command[-1]
        assert timeout <= run.protocol["resource_forecast"]["child_timeout_s"]
        events = operator.read_jsonl(output.parent / "reset-ledger.jsonl")[0]
        assert events[-1]["status"] == "reset_intent" and len(events) == count * 2 - 1
        if failure == "timeout":
            output.with_suffix(".raw.jsonl").write_bytes(b'{"partial":')
            output.with_suffix(".decisions.jsonl").write_bytes(b'{"status":"act_intent"}\n')
            operator.save(output.with_suffix(".process.json"), dict(status="operator_error", error="timeout", wall_time_s=.01))
            raise RuntimeError("mock timeout")
        track, seed, arm = output.stem.split("-", 2)
        trace_count = 0 if failure == "short_prefix" else 10
        road = f"road-{seed}" if failure != "road" else "same-all-roads"
        if failure == "oldroad":
            road = "old-road"
        operator.save(output, dict(mode=arm, track_id=int(track), seed=int(seed), completed=True, lapTimeMs=1000,
                                  progress=1., damage=0., collisions=0, error=None, invalid_actions=0,
                                  retire_reason="max_steps" if failure == "censor" else None,
                                  steps=trace_count, raw_ticks=0, peak_rss_bytes=285000000,
                                  initial_state={}, geometry_sha256="same", initial_observation_sha256="same",
                                  track_geometry_sha256=road,
                                  decision_trace=[dict(steer=0, gas=0, brake=0, car_x=0, car_y=0, car_yaw=0, progress=0)] * trace_count,
                                  versions=run.protocol["runtime_versions"], damage_telemetry_valid=True, collision_telemetry_valid=True))
        operator.save(output.with_suffix(".process.json"), dict(status="completed", error=None, wall_time_s=.01))

    monkeypatch.setattr(operator.original, "run_child", child)
    if failure:
        with pytest.raises((AssertionError, RuntimeError)):
            operator.run(run.protocol_sha, run.review_path, run.review_sha)
    else:
        operator.run(run.protocol_sha, run.review_path, run.review_sha)
    report = operator.read_json(run.destination / "episode-report.json")
    ledger, _ = operator.read_jsonl(run.destination / "reset-ledger.jsonl")
    assert report["reset_ledger_sha256"] == operator.sha(run.destination / "reset-ledger.jsonl")
    assert report["protocol_sha256"] == run.protocol_sha
    assert len(report["rows"]) == 144
    assert "partial cells remain consumed" in report["freshness_scope"]
    if failure:
        assert report["operator_error"] and not report["complete"]
        assert (run.destination / "operator-failure.json").exists()
        if failure == "timeout":
            assert count == 1 and len(ledger) == 1
            first = operator.scheduled_slots()[0]
            output = run.destination / f"{first['track_id']}-{first['seed']}-{first['mode']}.json"
            assert output.with_suffix(".raw.jsonl").exists() and output.with_suffix(".decisions.jsonl").exists()
            bound = operator.read_json(output.with_suffix(".bound-process.json"))
            assert bound["original_process_sha256"] == operator.sha(output.with_suffix(".process.json"))
            pins = report["rows"][0]["partial_artifacts"]
            assert len(pins) == 4
            assert all(operator.sha(run.destination / p["file"]) == p["sha256"] and (run.destination / p["file"]).stat().st_size == p["bytes"] for p in pins)
        if failure == "audit":
            assert count == 0 and not ledger
    else:
        assert count == 144 and len(ledger) == 288
        assert report["complete"] and not report["operator_error"]
        assert all(row["status"] == "completed" for row in report["rows"])
        assert len(report["track_geometry_sha256"]) == 24
        assert len([c for c in run.calls if c[0] == "cell"]) == 144
    with pytest.raises(AssertionError):
        operator.preflight(run.protocol_sha)


@pytest.mark.parametrize("guard", ["sources", "claims", "review", "resources", None])
def test_actual_observer_reset_body_runs_all_guards_under_lock_before_fake_reset(frozen_run, monkeypatch, guard):
    """Execute the worker's reset body in isolation, without importing its environment."""
    tree = ast.parse(Path(operator.__file__).read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    observer = next(n for n in worker.body if isinstance(n, ast.ClassDef) and n.name == "Observer")
    reset = next(n for n in observer.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    module = ast.fix_missing_locations(ast.Module(body=[reset], type_ignores=[]))
    run = frozen_run
    operator.save(run.destination / "execution-review.json", dict(path=str(run.review_path), sha256=run.review_sha))
    reached = []

    class FakeResetReached(RuntimeError):
        pass

    def fake_reset():
        reached.append(True)
        fd = operator.os.open(run.root / "experiments/train-seed-claims", operator.os.O_RDONLY | operator.os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            operator.os.close(fd)
        raise FakeResetReached("not an environment reset")

    def validate_frozen(_):
        if guard == "sources":
            raise AssertionError("source copy drift")

    def check_claims(*args):
        if guard == "claims":
            raise AssertionError("new reservation")
        return run.protocol["protected_claims_observed_sha256"]

    def validate_review(*args):
        if guard == "review":
            raise AssertionError("review drift")

    def check_resources(*args):
        if guard == "resources":
            raise AssertionError("headroom")

    monkeypatch.chdir(run.root / operator.R2_PATH)  # model-source check points to a temp extracted inventory
    for name, digest in run.protocol["model_source_sha256"][operator.ARMS[0]].items():
        # The isolated function's SHA implementation verifies frozen expected values, not an imported Agent.
        assert isinstance(digest, str)
    def sha(path):
        path = Path(path)
        relative = str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else None
        if relative in run.protocol["model_source_sha256"][operator.ARMS[0]]:
            return run.protocol["model_source_sha256"][operator.ARMS[0]][relative]
        return operator.sha(path)

    namespace: dict[str, Any] = dict(ROOT=run.root, os=operator.os, fcntl=fcntl, validate_frozen=validate_frozen,
                     check_claims=check_claims, exposure=run.exposure, protocol_sha256=run.protocol_sha,
                     protocol=run.protocol, track=1, seed=operator.SEEDS[0], arm=operator.ARMS[0],
                     sha=sha, Path=Path, output=run.destination / "unused.json", read_json=operator.read_json,
                     PROTOCOL_PATH=operator.PROTOCOL_PATH, validate_review=validate_review,
                     check_resources=check_resources, resource_measurements=operator.resource_measurements)
    exec(compile(module, "isolated-zero-reset-observer", "exec"), namespace)
    self = SimpleNamespace(inner=SimpleNamespace(reset=fake_reset))
    with pytest.raises(AssertionError if guard else FakeResetReached):
        namespace["reset"](self)
    assert bool(reached) == (guard is None)
