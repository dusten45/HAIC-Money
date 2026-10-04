"""Synthetic terminal receipts and subprocess doubles; never create/reset an env."""

from copy import deepcopy
import ast
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from scripts import evaluate_koi_steering_generalization_continuation as child
from scripts import pause_koi_steering_generalization as pause
from tests.test_koi_steering_generalization_loader import six_object_episode, write_json, write_rows

REAL_CHECK_CLAIMS = child.check_claims


def _discover(_root):
    raise AssertionError("audit must use its isolated discovery namespace")


def saved_episode(run, protocol, planned, *, altered_prefix=False):
    key = child.identity(planned)
    stem = f"{key[0]}-{key[1]}-{key[2]}"
    episode, raw, stream = six_object_episode(key[2], key[0], key[1])
    if altered_prefix:
        episode["initial_observation_sha256"] = "different-initial-pixels"
    episode["versions"] = protocol["runtime_versions"]
    raw_path, stream_path = run / (stem + ".raw.jsonl"), run / (stem + ".decisions.jsonl")
    write_rows(raw_path, raw)
    write_rows(stream_path, stream)
    episode.update(raw_trace_file=raw_path.name, raw_trace_sha256=child.sha(raw_path),
                   partial_decisions_file=stream_path.name, partial_decisions_sha256=child.sha(stream_path))
    output, process = run / (stem + ".json"), run / (stem + ".process.json")
    write_json(output, episode)
    receipt = dict(status="completed", error=None, wall_time_s=10.)
    write_json(process, receipt)
    bound = run / (stem + ".bound-process.json")
    write_json(bound, dict(receipt, slot_id=child.parent.slot_id(*key), operator_sha256=protocol["operator_sha256"],
                          original_process_file=process.name, original_process_sha256=child.sha(process)))
    compact = {k: episode[k] for k in child.COMPACT_KEYS}
    compact.update(status="completed", file=output.name, sha256=child.sha(output), slot_id=child.parent.slot_id(*key),
                   process_file=bound.name, process_sha256=child.sha(bound))
    return compact, episode


@pytest.fixture
def original_boundary(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "experiments/train-seed-claims").mkdir(parents=True)
    run = root / child.parent.RUN_PATH
    run.mkdir(parents=True)
    sources = root / "scripts"
    sources.mkdir()
    operator_source = sources / "evaluate_koi_steering_generalization_continuation.py"
    operator_source.write_bytes(Path(child.__file__).read_bytes())
    pause_source = sources / "pause_koi_steering_generalization.py"
    pause_source.write_bytes(Path(pause.__file__).read_bytes())
    finalizer_source = sources / "finalize_koi_steering_generalization.py"
    finalizer_source.write_bytes(Path(child.forensic.__file__).read_bytes())
    composite_source = sources / "finalize_koi_steering_generalization_composite.py"
    composite_source.write_bytes((child.ROOT / "scripts/finalize_koi_steering_generalization_composite.py").read_bytes())
    parent_source = sources / "evaluate_koi_steering_generalization.py"
    parent_source.write_bytes(Path(child.parent.__file__).read_bytes())
    audit_source = sources / "audit_koi_steering_generalization.py"
    audit_source.write_bytes(Path(child.freshness.__file__).read_bytes())
    root_agent = root / "agent.py"
    root_agent.write_bytes((child.ROOT / "agent.py").read_bytes())
    seed = child.parent.SEEDS[0]
    schedule = deepcopy(child.parent.scheduled_slots()[:4])
    real_protocol = child.read_json(child.ROOT / child.parent.RUN_PATH / "protocol.json")
    protocol = {k: deepcopy(real_protocol[k]) for k in child.INHERITED_KEYS}
    protocol.update(study=child.parent.STUDY, episodes=4, schedule=schedule,
                    cells=[child.cell(r) for r in schedule[::2]], source_copies=[],
                    operator_sha256=child.sha(parent_source), analyzer_sha256=real_protocol["analyzer_sha256"],
                    helper_source_sha256={str(audit_source): child.sha(audit_source)}, environment_source_sha256={},
                    reference_evidence=dict(source_sha256={}, prior_road_sha256=[],
                                            prior_peak_rss_bytes=real_protocol["reference_evidence"]["prior_peak_rss_bytes"]),
                    protected_claims_observed_sha256={}, resource_forecast=deepcopy(real_protocol["resource_forecast"]))
    for name in ("candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"):
        (run / name).write_bytes((child.ROOT / child.parent.RUN_PATH / name).read_bytes())
    write_json(run / "protocol.json", protocol)
    exposure = dict(synthetic=True)
    monkeypatch.setattr(child, "ROOT", root)
    monkeypatch.setattr(child, "__file__", str(operator_source))
    monkeypatch.setattr(child.parent, "validate_frozen", lambda digest: (deepcopy(protocol), deepcopy(exposure)))
    monkeypatch.setattr(child.parent, "helper_paths", lambda: [parent_source, audit_source])
    monkeypatch.setattr(child.freshness, "__file__", str(audit_source))
    monkeypatch.setattr(child.forensic, "__file__", str(finalizer_source))
    monkeypatch.setattr(child.forensic, "GUARD_PATH", root / "experiments/original-guard.json")
    monkeypatch.setattr(child.forensic, "RESULT_PATH", root / "experiments/original-result.json")
    monkeypatch.setattr(child, "helper_paths", lambda: [operator_source, pause_source, finalizer_source, composite_source])
    write_json(child.forensic.GUARD_PATH, dict(synthetic=True))
    monkeypatch.setattr(child, "GUARD_SHA", child.sha(child.forensic.GUARD_PATH))
    review_path = root / "experiments/original-review.json"
    write_json(review_path, dict(status="passed", environment_resets=0, reviewer="independent synthetic",
               protocol_sha256=child.PARENT_SHA, operator_sha256=protocol["operator_sha256"],
               analyzer_sha256=protocol["analyzer_sha256"], audit_sha256=child.AUDIT_SHA,
               evidence_guard_sha256=child.GUARD_SHA, finalizer_sha256=child.sha(finalizer_source)))
    review_pin = dict(path=str(review_path), sha256=child.sha(review_path))
    write_json(run / "execution-review.json", review_pin)
    first, episode = saved_episode(run, protocol, schedule[0])
    key = child.identity(first)
    intent = dict(status="reset_intent", arm=key[2], track=key[0], seed=key[1],
                  slot_id=child.parent.slot_id(*key), time=1.)
    ledger = [intent, deepcopy(first)]
    ledger_path = run / "reset-ledger.jsonl"
    write_rows(ledger_path, ledger)
    rows = [first, dict(schedule[1], status="operator_error", error="KeyboardInterrupt: ", partial_artifacts=[]),
            *deepcopy(schedule[2:])]
    report = dict(protocol_sha256=child.PARENT_SHA, rows=rows, operator_error="KeyboardInterrupt: ",
                  reset_ledger_sha256=child.sha(ledger_path), review_receipt=review_pin, complete=False)
    report_path = run / "episode-report.json"
    write_json(report_path, report)
    write_json(child.forensic.RESULT_PATH, dict(protocol_sha256=child.PARENT_SHA,
                episode_report_sha256=child.sha(report_path), evidence_guard_sha256=child.GUARD_SHA, gate_passed=False))
    boundary = dict(status="paused", protocol_sha256=child.PARENT_SHA,
                    boundary=dict(pending_reset_intents=0, completed_slots=1, no_live_children=True, child_interrupted=False),
                    terminal=dict(process_exited=True, report_path=str(report_path), report_sha256=child.sha(report_path),
                                  reset_ledger_sha256=child.sha(ledger_path), wall_time_s=50.))
    boundary_path = root / "experiments/boundary.json"
    write_json(boundary_path, boundary)

    def verify_boundary(path, digest, *, root):
        assert Path(path) == boundary_path and child.sha(path) == digest
        return deepcopy(boundary)

    monkeypatch.setattr(pause, "verify_boundary", verify_boundary)
    monkeypatch.setattr(child.parent, "runtime_versions", lambda: protocol["runtime_versions"])
    monkeypatch.setattr(child.parent, "resource_measurements", lambda _: dict(host_mem_available_bytes=10**12,
                     visible_cgroup_ancestors=[dict(raw_headroom_bytes=10**12)],
                     disk=[dict(device=1, free_bytes=10**12), dict(device=1, free_bytes=10**12)]))
    return SimpleNamespace(root=root, run=run, protocol=protocol, exposure=exposure, boundary=boundary,
                           boundary_path=boundary_path, boundary_sha=child.sha(boundary_path), report=report,
                           ledger=ledger, first=first, episode=episode, sources=[operator_source, pause_source, finalizer_source, composite_source])


def repin_parent(fixture):
    write_rows(fixture.run / "reset-ledger.jsonl", fixture.ledger)
    fixture.report["reset_ledger_sha256"] = child.sha(fixture.run / "reset-ledger.jsonl")
    write_json(fixture.run / "episode-report.json", fixture.report)
    fixture.boundary["terminal"]["report_sha256"] = child.sha(fixture.run / "episode-report.json")
    fixture.boundary["terminal"]["reset_ledger_sha256"] = fixture.report["reset_ledger_sha256"]


def test_import_and_cli_do_not_import_environment_or_construct_policy():
    code = "import sys;from scripts import evaluate_koi_steering_generalization_continuation;assert not any(n.startswith(('training.','gymnasium','Box2D','pygame')) for n in sys.modules);assert 'agent' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=child.ROOT, check=True, capture_output=True)
    result = subprocess.run([sys.executable, "-m", "scripts.evaluate_koi_steering_generalization_continuation", "--help"],
                            cwd=child.ROOT, check=True, capture_output=True, text=True)
    assert "--boundary" in result.stdout and "--run" in result.stdout
    assert "--candidate" not in result.stdout


def test_actual_parent_pinned_sources_models_and_root_remain_byte_identical():
    protocol, exposure = child.parent.validate_frozen(child.PARENT_SHA)
    assert protocol["episodes"] == 144 and len(protocol["cells"]) == 72
    assert protocol["model_hashes"] == child.freshness.MODEL_HASHES
    assert child.sha(child.parent.ROOT / "agent.py") == child.parent.ROOT_AGENT_SHA
    assert exposure["claims"] and len(exposure["claims"]) == 24
    assert all(child.sha(path) == digest for path, digest in protocol["helper_source_sha256"].items())
    assert not any(name.startswith(("training.", "gymnasium", "Box2D", "pygame")) for name in sys.modules)


def test_conventional_boundary_receipt_is_not_a_hidden_allocation_protocol(original_boundary):
    fixture = original_boundary
    write_json(fixture.run / "boundary-pause.json", fixture.boundary)
    paths = child.freshness._discover(fixture.root)
    assert (child.parent.RUN_PATH / "protocol.json").as_posix() in paths
    assert (child.parent.RUN_PATH / "boundary-pause.json").as_posix() not in paths
    # Child intents are metadata, hence require explicit typed authentication.
    destination = fixture.root / child.RUN_PATH
    destination.mkdir()
    (destination / child.LEDGER_NAME).write_bytes(b"")
    assert (child.RUN_PATH / child.LEDGER_NAME).as_posix() in child.freshness._discover(fixture.root)


def test_only_original_zero_intent_slots_including_preintent_error_are_planned(original_boundary):
    fixture = original_boundary
    before = {p: child.sha(p) for p in fixture.run.iterdir() if p.is_file()}
    verified = child.verified_parent(fixture.boundary_path, fixture.boundary_sha)
    assert len(verified["completed"]) == 1 and len(verified["remaining_slots"]) == 3
    assert verified["remaining_slots"] == fixture.protocol["schedule"][1:]
    assert verified["completed"][child.identity(fixture.first)][0] == fixture.episode
    assert before == {p: child.sha(p) for p in before}
    assert not (fixture.root / child.RUN_PATH).exists()


def test_production_144_slot_complement_is_never_reordered_or_repeated(original_boundary):
    fixture = original_boundary
    schedule = child.parent.scheduled_slots()
    assert len(schedule) == 144 and len({child.identity(r) for r in schedule}) == 144
    fixture.protocol.update(schedule=deepcopy(schedule), episodes=144,
                            cells=child.freshness.proposed_cells())
    fixture.report["rows"] = [fixture.first, fixture.report["rows"][1], *deepcopy(schedule[2:])]
    repin_parent(fixture)
    verified = child.verified_parent(fixture.boundary_path, fixture.boundary_sha)
    assert verified["remaining_slots"] == schedule[1:]
    assert len(verified["remaining_slots"]) == 143
    assert not set(verified["completed"]) & {child.identity(r) for r in verified["remaining_slots"]}


@pytest.mark.parametrize("defect", ["pending_intent", "uncertain_boundary", "running_parent", "unexecuted_artifact", "unknown_status", "report_order", "ledger_mismatch", "partial_pin", "wrong_process", "raw_change", "wall"])
def test_parent_uncertainty_always_fails_closed(original_boundary, defect):
    fixture = original_boundary
    next_row = fixture.protocol["schedule"][1]
    if defect == "pending_intent":
        key = child.identity(next_row)
        fixture.ledger.append(dict(status="reset_intent", track=key[0], seed=key[1], arm=key[2],
                                   slot_id=child.parent.slot_id(*key), time=2.))
    elif defect == "uncertain_boundary":
        fixture.boundary["boundary"]["child_interrupted"] = True
    elif defect == "running_parent":
        fixture.boundary["terminal"]["process_exited"] = False
    elif defect == "unexecuted_artifact":
        key = child.identity(next_row)
        (fixture.run / f"{key[0]}-{key[1]}-{key[2]}.raw.jsonl").write_bytes(b"")
    elif defect == "unknown_status":
        fixture.report["rows"][2]["status"] = "reset_pending"
    elif defect == "report_order":
        fixture.report["rows"][2:4] = reversed(fixture.report["rows"][2:4])
    elif defect == "ledger_mismatch":
        fixture.ledger[0]["slot_id"] = "wrong"
    elif defect == "partial_pin":
        fixture.report["rows"][1]["partial_artifacts"] = [dict(file="missing", sha256="wrong", bytes=0)]
    elif defect == "wrong_process":
        path = fixture.run / fixture.first["process_file"]
        process = child.read_json(path)
        process["slot_id"] = "wrong"
        write_json(path, process)
        fixture.first["process_sha256"] = child.sha(path)
        fixture.ledger[1] = deepcopy(fixture.first)
    elif defect == "raw_change":
        (fixture.run / fixture.episode["raw_trace_file"]).write_bytes(b"different")
    else:
        fixture.boundary["terminal"]["wall_time_s"] = 1.
    repin_parent(fixture)
    with pytest.raises((ValueError, AssertionError)):
        child.verified_parent(fixture.boundary_path, fixture.boundary_sha)


@pytest.mark.parametrize("defect", ["slot_id", "extra_key", "bool_track"])
def test_completed_record_requires_exact_typed_event_identity(original_boundary, defect):
    fixture = original_boundary
    row = deepcopy(fixture.first)
    if defect == "slot_id":
        row["slot_id"] = "wrong"
    elif defect == "extra_key":
        row["undeclared_exposure"] = 1
    else:
        row["track_id"] = True
    with pytest.raises(ValueError, match="completed event schema/identity"):
        child.validate_record(fixture.run, row, fixture.protocol, {})


def test_actual_wall_and_full_guards_have_separate_remaining_budget(original_boundary):
    verified = child.verified_parent(original_boundary.boundary_path, original_boundary.boundary_sha)
    forecast = child.resource_forecast(verified, 7., 512 * 1024**2)
    assert forecast["actual_parent_process_wall_s"] == 50.
    assert forecast["actual_parent_wall_per_slot_s"] == 50.
    assert forecast["full_cohort_guard_growth_factor"] == 4.
    assert forecast["guarded_reference_per_slot_s"] == 10. + 8 * 7. * 4.
    assert forecast["wall_budget_s"] == (10. + 8 * 7. * 4.) * 3 * 1.5 + 14.
    assert forecast["child_timeout_s"] >= forecast["guarded_reference_per_slot_s"] * 1.5
    assert original_boundary.protocol["resource_forecast"]["wall_budget_s"] != forecast["wall_budget_s"]
    with pytest.raises(ValueError):
        child.resource_forecast(verified, float("nan"), 512 * 1024**2)


def test_guard_rss_sampler_uses_self_linux_kib_high_water(monkeypatch):
    calls = []

    def usage(who):
        calls.append(who)
        return SimpleNamespace(ru_maxrss=12345)

    monkeypatch.setattr(child.resource, "getrusage", usage)
    monkeypatch.setattr(child.sys, "platform", "linux")
    assert child.guard_peak_rss_bytes() == 12345 * 1024
    assert calls == [child.resource.RUSAGE_SELF]


@pytest.mark.parametrize("platform,peak", [("darwin", 12345), ("linux", 0), ("linux", -1), ("linux", True), ("linux", 1.5)])
def test_guard_rss_sampler_rejects_nonlinux_or_invalid_units(monkeypatch, platform, peak):
    monkeypatch.setattr(child.sys, "platform", platform)
    monkeypatch.setattr(child.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=peak))
    with pytest.raises(ValueError):
        child.guard_peak_rss_bytes()


def test_memory_forecast_adds_guard_growth_child_and_existing_reserve(original_boundary):
    verified = child.verified_parent(original_boundary.boundary_path, original_boundary.boundary_sha)
    guard_peak = 512 * 1024**2 + 1
    forecast = child.resource_forecast(verified, 7., guard_peak)
    original = original_boundary.protocol
    child_peak = original["reference_evidence"]["prior_peak_rss_bytes"]
    reserve = original["resource_forecast"]["memory_reserve_bytes"]
    grown_guard = math.ceil(guard_peak * 1.5 * original["episodes"] / len(verified["completed"]))
    historical_child = math.ceil(child_peak * 1.5)
    assert forecast["parent_peak_guard_rss_bytes"] == guard_peak
    assert forecast["historical_single_child_peak_rss_bytes"] == child_peak
    assert forecast["estimated_full_cohort_guard_peak_bytes"] == grown_guard
    assert forecast["estimated_single_child_peak_bytes"] == historical_child
    assert forecast["peak_child_bytes"] == grown_guard + historical_child
    assert forecast["required_memory_bytes"] == grown_guard + historical_child + reserve
    assert forecast["memory_reserve_bytes"] == reserve
    assert forecast["peak_child_bytes"] > original["resource_forecast"]["peak_child_bytes"]
    assert forecast["resource_formula"]["guard_growth_factor"] == 4.
    assert forecast["resource_formula"]["guard_safety_factor"] == 1.5
    assert forecast["resource_formula"]["historical_child_safety_factor"] == 1.5
    assert original["resource_forecast"].get("parent_peak_guard_rss_bytes") is None


@pytest.mark.parametrize("peak", [None, False, True, 0, -1, 1024., "1024", float("inf"), float("nan")])
def test_forecast_requires_positive_integer_frozen_guard_measurement(original_boundary, peak):
    verified = child.verified_parent(original_boundary.boundary_path, original_boundary.boundary_sha)
    with pytest.raises(ValueError, match="guard peak RSS"):
        child.resource_forecast(verified, 7., peak)


def test_freeze_samples_own_peak_after_validation_audit_and_defensive_copy(original_boundary, monkeypatch):
    fixture = original_boundary
    real_verified = child.verified_parent
    events = []
    recorded_peak = 700 * 1024**2

    def verified(path, digest):
        result = real_verified(path, digest)
        events.append("verified_parent")
        return result

    def audit(*args, **kwargs):
        events.append("full_guard")
        return {}

    def measured_peak():
        assert events == ["verified_parent", "full_guard", "verified_parent"]
        events.append("own_peak_rss")
        return recorded_peak

    monkeypatch.setattr(child, "verified_parent", verified)
    monkeypatch.setattr(child, "check_claims", audit)
    monkeypatch.setattr(child, "guard_peak_rss_bytes", measured_peak)
    result = child.freeze_protocol(fixture.boundary_path, fixture.boundary_sha)
    protocol = child.read_json(fixture.root / child.RUN_PATH / "protocol.json")
    assert protocol["resource_forecast"]["parent_peak_guard_rss_bytes"] == recorded_peak
    assert events.count("own_peak_rss") == 1
    assert result["environment_resets"] == 0


@pytest.mark.parametrize("limit", ["host", "cgroup"])
def test_insufficient_ram_for_guard_cache_refuses_freeze_before_writes(original_boundary, monkeypatch, limit):
    fixture = original_boundary
    old = fixture.protocol["resource_forecast"]
    old_bound = old["peak_child_bytes"] + old["memory_reserve_bytes"]
    measurement = dict(host_mem_available_bytes=old_bound + 1024 if limit == "host" else 10**12,
                       visible_cgroup_ancestors=[dict(raw_headroom_bytes=old_bound + 1024 if limit == "cgroup" else 10**12)],
                       disk=[dict(device=1, free_bytes=10**12), dict(device=1, free_bytes=10**12)])
    child.parent.check_resources(measurement, old, 3)
    monkeypatch.setattr(child, "check_claims", lambda *args, **kwargs: {})
    monkeypatch.setattr(child, "guard_peak_rss_bytes", lambda: 2 * 1024**3)
    monkeypatch.setattr(child.parent, "resource_measurements", lambda _: measurement)
    with pytest.raises(AssertionError, match=f"{limit} memory forecast does not fit"):
        child.freeze_protocol(fixture.boundary_path, fixture.boundary_sha)
    assert not (fixture.root / child.RUN_PATH).exists()
    assert not (fixture.root / child.PROTOCOL_PATH).exists()


def test_parent_cache_reuses_only_semantics_and_defends_against_caller_mutation(original_boundary, monkeypatch):
    fixture = original_boundary
    first = child.verified_parent(fixture.boundary_path, fixture.boundary_sha)
    first["completed"][child.identity(fixture.first)][0]["decision_trace"][0]["steer"] = 99
    first["remaining_slots"][0]["seed"] = 0
    monkeypatch.setattr(child.analysis, "analyze_episode", lambda *args: pytest.fail("unchanged semantics should be reused"))
    second = child.verified_parent(fixture.boundary_path, fixture.boundary_sha)
    assert second["completed"][child.identity(fixture.first)][0] == fixture.episode
    assert second["remaining_slots"] == fixture.protocol["schedule"][1:]


@pytest.mark.parametrize("defect", ["episode", "raw", "stream", "process", "report", "result", "unexecuted", "source_contract", "boundary"])
def test_parent_cache_rechecks_all_exact_pins_and_candidate_ambiguity(original_boundary, defect):
    fixture = original_boundary
    child.verified_parent(fixture.boundary_path, fixture.boundary_sha)
    if defect == "unexecuted":
        key = child.identity(fixture.protocol["schedule"][1])
        (fixture.run / f"{key[0]}-{key[1]}-{key[2]}.raw.jsonl").write_bytes(b"")
    elif defect == "source_contract":
        fixture.protocol["max_decisions"] = 100
    elif defect == "boundary":
        fixture.boundary["boundary"]["child_interrupted"] = True
    else:
        paths = dict(episode=fixture.run / fixture.first["file"], raw=fixture.run / fixture.episode["raw_trace_file"],
                     stream=fixture.run / fixture.episode["partial_decisions_file"], process=fixture.run / fixture.first["process_file"],
                     report=fixture.run / "episode-report.json", result=child.forensic.RESULT_PATH)
        paths[defect].write_bytes(paths[defect].read_bytes() + b"\n")
    with pytest.raises((ValueError, AssertionError)):
        child.verified_parent(fixture.boundary_path, fixture.boundary_sha)


@pytest.fixture
def frozen_child(original_boundary, monkeypatch):
    fixture = original_boundary
    audit_calls = []

    def check_claims(receipt, protocol_sha256=None, candidate_cell=None, *, boundary_pin=None):
        assert receipt == fixture.exposure
        descriptor = child.os.open(fixture.root / child.freshness.REGISTRY, child.os.O_RDONLY | child.os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            child.os.close(descriptor)
        audit_calls.append((protocol_sha256, candidate_cell))
        return {}

    monkeypatch.setattr(child, "check_claims", check_claims)
    result = child.freeze_protocol(fixture.boundary_path, fixture.boundary_sha)
    destination = fixture.root / child.RUN_PATH
    protocol = child.read_json(destination / "protocol.json")
    finalizer = fixture.sources[3]
    guard_path = fixture.root / "experiments/koi-steering-generalization-v1-composite-guard.json"
    write_json(guard_path, dict(continuation_protocol_sha256=result["protocol_sha256"],
               continuation_operator_sha256=protocol["operator_sha256"], parent_protocol_sha256=child.PARENT_SHA,
               composite_finalizer_sha256=child.sha(finalizer)))
    review_path = fixture.root / "experiments/child-review.json"
    write_json(review_path, dict(status="passed", environment_resets=0, reviewer="independent synthetic child review",
                  protocol_sha256=result["protocol_sha256"], operator_sha256=protocol["operator_sha256"],
                  analyzer_sha256=protocol["analyzer_sha256"], audit_sha256=child.AUDIT_SHA,
                  parent_protocol_sha256=child.PARENT_SHA, parent_boundary_sha256=fixture.boundary_sha,
                  original_worker_sha256=protocol["original_worker_sha256"],
                  composite_finalizer_sha256=child.sha(finalizer), composite_guard_sha256=child.sha(guard_path)))
    return SimpleNamespace(original=fixture, destination=destination, protocol=protocol,
                           protocol_sha=result["protocol_sha256"], review_path=review_path,
                           review_sha=child.sha(review_path), audit_calls=audit_calls)


def test_freeze_preserves_original_bytes_and_preflight_never_uses_worker(frozen_child, monkeypatch):
    fixture = frozen_child
    monkeypatch.setattr(child, "worker", lambda *args: pytest.fail("worker forbidden"))
    result = child.preflight(fixture.protocol_sha)
    assert result["environment_resets"] == 0 and result["episodes"] == 3
    assert (fixture.original.root / child.PROTOCOL_PATH).read_bytes() == (fixture.destination / "protocol.json").read_bytes()
    for name in ("candidate.zip", "candidate.manifest.json", "crossing-projection-source-reconstruction.zip"):
        assert (fixture.destination / name).read_bytes() == (fixture.original.run / name).read_bytes()
    assert set(fixture.protocol) == child.PROTOCOL_KEYS
    assert fixture.protocol["schedule"] == fixture.original.protocol["schedule"][1:]
    assert fixture.protocol["parent_result"]["sha256"] == child.sha(child.forensic.RESULT_PATH)
    forecast = fixture.protocol["resource_forecast"]
    assert type(forecast["parent_peak_guard_rss_bytes"]) is int and forecast["parent_peak_guard_rss_bytes"] > 0
    assert forecast["required_memory_bytes"] == forecast["peak_child_bytes"] + forecast["memory_reserve_bytes"]
    assert not (fixture.destination / child.LEDGER_NAME).exists()
    metadata = child.dataset_metadata(fixture.protocol_sha)
    assert len(metadata["parent_episodes"]) == 1 and len(metadata["child_slots"]) == 3
    assert len(metadata["combined_schedule"]) == 4
    assert metadata["child_ledger"].name == "reset-intents.jsonl"


def test_frozen_rss_scalar_is_validated_by_formula_not_current_process_rss(frozen_child, monkeypatch):
    fixture = frozen_child
    before = deepcopy(fixture.protocol["resource_forecast"])
    monkeypatch.setattr(child, "guard_peak_rss_bytes", lambda: pytest.fail("frozen validation must not resample RSS"))
    validated, _ = child.validate_frozen(fixture.protocol_sha)
    assert validated["resource_forecast"] == before
    assert child.preflight(fixture.protocol_sha)["environment_resets"] == 0


@pytest.mark.parametrize("defect", ["scalar", "float_scalar", "formula_factor", "formula_growth_type", "formula_key", "required_memory", "historical_peak", "float_bytes", "lower_peak"])
def test_frozen_memory_formula_and_derived_bytes_are_exact_typed(frozen_child, defect):
    fixture = frozen_child
    protocol = deepcopy(fixture.protocol)
    forecast = protocol["resource_forecast"]
    if defect == "scalar":
        forecast["parent_peak_guard_rss_bytes"] += 1024
    elif defect == "float_scalar":
        forecast["parent_peak_guard_rss_bytes"] = float(forecast["parent_peak_guard_rss_bytes"])
    elif defect == "formula_factor":
        forecast["resource_formula"]["guard_safety_factor"] = 1.
    elif defect == "formula_growth_type":
        forecast["resource_formula"]["guard_growth_factor"] = 4
    elif defect == "formula_key":
        forecast["resource_formula"]["unreviewed_allowance"] = True
    elif defect == "required_memory":
        forecast["required_memory_bytes"] -= 1024
    elif defect == "historical_peak":
        forecast["historical_single_child_peak_rss_bytes"] -= 1024
    elif defect == "float_bytes":
        forecast["peak_child_bytes"] = float(forecast["peak_child_bytes"])
    else:
        forecast["peak_child_bytes"] = fixture.original.protocol["resource_forecast"]["peak_child_bytes"]
    write_json(fixture.destination / "protocol.json", protocol)
    (fixture.original.root / child.PROTOCOL_PATH).write_bytes((fixture.destination / "protocol.json").read_bytes())
    with pytest.raises(ValueError):
        child.validate_frozen(child.sha(fixture.destination / "protocol.json"))


@pytest.mark.parametrize("defect", ["repeat", "cohort", "model", "gates", "wall", "unexpected_key", "source", "sourcecopy", "parent_record", "root_agent"])
def test_child_typed_allowance_rejects_changed_protocol_or_evidence(frozen_child, defect):
    fixture = frozen_child
    protocol = deepcopy(fixture.protocol)
    if defect == "repeat":
        protocol["schedule"][0] = deepcopy(fixture.original.protocol["schedule"][0])
    elif defect == "cohort":
        protocol["cells"][0]["geometry_seed"] += 100
    elif defect == "model":
        protocol["model_hashes"][child.ARMS[1]] = "changed"
    elif defect == "gates":
        protocol["analysis_spec"]["min_eligible_geometry_seeds"] = 0
    elif defect == "wall":
        protocol["resource_forecast"]["wall_budget_s"] *= 10
    elif defect == "unexpected_key":
        protocol["hidden_cohort"] = [1]
    elif defect == "source":
        fixture.original.sources[0].write_bytes(b"changed source")
    elif defect == "sourcecopy":
        (fixture.destination / protocol["source_copies"][0]["file"]).write_bytes(b"changed source copy")
    elif defect == "parent_record":
        (fixture.original.run / fixture.original.first["file"]).write_bytes(b"changed original")
    else:
        protocol["root_agent_sha256"] = "changed-root"
    write_json(fixture.destination / "protocol.json", protocol)
    (fixture.original.root / child.PROTOCOL_PATH).write_bytes((fixture.destination / "protocol.json").read_bytes())
    with pytest.raises((ValueError, AssertionError)):
        child.validate_frozen(child.sha(fixture.destination / "protocol.json"))


@pytest.mark.parametrize("event", ["wrong_identity", "wrong_order", "torn", "old_completed"])
def test_child_reset_ledger_cannot_reclaim_original_or_unknown_slot(frozen_child, event):
    fixture = frozen_child
    planned = fixture.protocol["schedule"][0]
    key = child.identity(planned)
    row = dict(status="reset_intent", arm=key[2], track=key[0], seed=key[1],
               slot_id=child.parent.slot_id(*key), time=1.)
    if event == "wrong_identity":
        row["seed"] = key[1] + 1
    elif event == "wrong_order":
        row["status"] = "completed"
    elif event == "old_completed":
        key = child.identity(fixture.original.first)
        row.update(arm=key[2], track=key[0], seed=key[1], slot_id=child.parent.slot_id(*key))
    path = fixture.destination / child.LEDGER_NAME
    write_rows(path, [row])
    if event == "torn":
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        child.validate_frozen(fixture.protocol_sha)


@pytest.mark.parametrize("failure", [None, "mate", "timeout", "audit"])
def test_mock_run_executes_only_remaining_and_checks_imported_mate(frozen_child, monkeypatch, failure):
    fixture = frozen_child
    before = {p: child.sha(p) for p in fixture.original.run.iterdir() if p.is_file()}
    calls = []
    monkeypatch.setattr(child.parent, "EPISODES", 4)
    monkeypatch.setattr(child.parent, "SEEDS", (fixture.protocol["schedule"][0]["seed"],))
    if failure == "audit":
        original_check = child.check_claims

        def blocked(receipt, digest=None, candidate_cell=None):
            if candidate_cell is not None:
                raise ValueError("fresh candidate collision")
            return original_check(receipt, digest, candidate_cell)

        monkeypatch.setattr(child, "check_claims", blocked)

    def child_process(command, directory, output, timeout, identity, operator_hash):
        index = len(calls)
        calls.append(output.name)
        assert "scripts.evaluate_koi_steering_generalization_continuation" in command[-1]
        assert operator_hash == fixture.protocol["operator_sha256"]
        assert timeout <= fixture.protocol["resource_forecast"]["child_timeout_s"]
        events, _ = child.read_jsonl(fixture.destination / child.LEDGER_NAME)
        assert len(events) == 2 * index + 1
        if failure == "timeout":
            output.with_suffix(".raw.jsonl").write_bytes(b'{"partial":')
            raise RuntimeError("synthetic timeout")
        planned = fixture.protocol["schedule"][index]
        saved_episode(fixture.destination, fixture.protocol, planned, altered_prefix=failure == "mate")

    monkeypatch.setattr(child.parent, "run_child", child_process)
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            child.run(fixture.protocol_sha, fixture.review_path, fixture.review_sha)
    else:
        child.run(fixture.protocol_sha, fixture.review_path, fixture.review_sha)
    report = child.read_json(fixture.destination / "episode-report.json")
    assert report["reset_ledger_sha256"] == child.sha(fixture.destination / child.LEDGER_NAME)
    assert len(report["rows"]) == 3
    assert before == {p: child.sha(p) for p in before}
    assert fixture.original.first["file"] not in calls
    if failure:
        assert not report["complete"] and report["operator_error"]
        if failure == "timeout":
            assert report["rows"][0]["partial_artifacts"][0]["bytes"] > 0
        if failure == "audit":
            assert not calls
    else:
        assert report["complete"] and len(calls) == 3
        assert len(child.read_jsonl(fixture.destination / child.LEDGER_NAME)[0]) == 6
    with pytest.raises(ValueError):
        child.preflight(fixture.protocol_sha)


def test_private_worker_namespace_does_not_mutate_original_globals(frozen_child):
    fixture = frozen_child
    original = dict(child.parent.worker.__globals__)
    namespace = child._worker_namespace(fixture.protocol_sha)
    assert namespace is not child.parent.worker.__globals__
    assert namespace["RUN_PATH"] == child.RUN_PATH and namespace["PROTOCOL_PATH"] == child.PROTOCOL_PATH
    protocol, _ = namespace["validate_frozen"](fixture.protocol_sha)
    assert protocol["operator_sha256"] == fixture.protocol["original_worker_sha256"]
    assert namespace["__file__"] == child.parent.__file__
    assert child.parent.worker.__globals__ == original


@pytest.mark.parametrize("block", [None, "review", "audit", "source"])
def test_unchanged_observer_reset_uses_private_child_guards_under_lock(frozen_child, monkeypatch, block):
    fixture = frozen_child
    planned = fixture.protocol["schedule"][0]
    key = child.identity(planned)
    write_rows(fixture.destination / child.LEDGER_NAME, [dict(status="reset_intent", arm=key[2], track=key[0],
               seed=key[1], slot_id=child.parent.slot_id(*key), time=1.)])
    write_json(fixture.destination / "execution-review.json", dict(path=str(fixture.review_path), sha256=fixture.review_sha))
    namespace = child._worker_namespace(fixture.protocol_sha)
    view, exposure = namespace["validate_frozen"](fixture.protocol_sha)
    namespace.update(protocol=view, protocol_sha256=fixture.protocol_sha, exposure=exposure,
                     output=fixture.destination / f"{key[0]}-{key[1]}-{key[2]}.json", track=key[0], seed=key[1], arm=key[2])
    directory = fixture.original.root / "extracted-policy"
    directory.mkdir()
    with zipfile.ZipFile(fixture.destination / ("candidate.zip" if key[2] == child.ARMS[1] else "crossing-projection-source-reconstruction.zip")) as archive:
        archive.extractall(directory)
    monkeypatch.chdir(directory)
    tree = ast.parse(Path(child.parent.__file__).read_text())
    worker = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "worker")
    observer = next(n for n in worker.body if isinstance(n, ast.ClassDef) and n.name == "Observer")
    reset = next(n for n in observer.body if isinstance(n, ast.FunctionDef) and n.name == "reset")
    module = ast.fix_missing_locations(ast.Module(body=[reset], type_ignores=[]))
    reached = []

    class FakeResetReached(RuntimeError):
        pass

    def fake_reset():
        descriptor = child.os.open(fixture.original.root / child.freshness.REGISTRY,
                                   child.os.O_RDONLY | child.os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            child.os.close(descriptor)
        reached.append(True)
        raise FakeResetReached("not an environment reset")

    if block == "review":
        fixture.review_path.write_bytes(b"changed review")
    elif block == "audit":
        def denied(*args, **kwargs):
            raise ValueError("new candidate-relevant metadata")
        namespace["check_claims"] = denied
    elif block == "source":
        fixture.original.sources[0].write_bytes(b"changed continuation source")
    exec(compile(module, "isolated-unchanged-observer-reset", "exec"), namespace)
    with pytest.raises(FakeResetReached if block is None else (ValueError, AssertionError)):
        namespace["reset"](SimpleNamespace(inner=SimpleNamespace(reset=fake_reset)))
    assert bool(reached) == (block is None)


def test_exact_auditor_overlay_excludes_only_authenticated_own_paths(frozen_child, monkeypatch):
    fixture = frozen_child
    allowed = [str(child.PROTOCOL_PATH), str(child.RUN_PATH / "protocol.json"), str(child.RUN_PATH / child.LEDGER_NAME),
               fixture.original.boundary_path.relative_to(fixture.original.root).as_posix()]
    extra = "runs/lookalike/reset-intents.jsonl"
    discoveries = allowed.copy()
    discovered = []
    monkeypatch.setattr(child.freshness, "verify_claims", lambda *args, **kwargs: None)
    monkeypatch.setattr(child.freshness, "_discover", lambda root: discoveries.copy())

    def audited(root, *, authenticated_receipt, self_protocol_sha256, required_sources=None):
        paths = _discover(root)  # Resolved through the isolated FunctionType namespace.
        discovered.append((paths, self_protocol_sha256))
        return dict(status="clear" if not paths else "HOLD", collisions=paths, blockers=[])

    monkeypatch.setattr(child.freshness, "audit", audited)
    original_discover = child.freshness._discover
    with child.claim_lock():
        assert REAL_CHECK_CLAIMS(fixture.original.exposure, fixture.protocol_sha) == {}
    assert discovered[-1] == ([], child.PARENT_SHA)
    assert child.freshness._discover is original_discover
    discoveries.append(extra)
    with child.claim_lock(), pytest.raises(ValueError, match="freshness HOLD"):
        REAL_CHECK_CLAIMS(fixture.original.exposure, fixture.protocol_sha)
    assert discovered[-1] == ([extra], child.PARENT_SHA)


@pytest.mark.parametrize("field", ["parent_protocol_sha256", "parent_boundary_sha256", "original_worker_sha256", "composite_finalizer_sha256", "composite_guard_sha256", "environment_resets"])
def test_review_must_bind_child_parent_worker_boundary_and_composite(frozen_child, field):
    fixture = frozen_child
    review = child.read_json(fixture.review_path)
    review[field] = 1 if field == "environment_resets" else "wrong"
    write_json(fixture.review_path, review)
    with pytest.raises(ValueError):
        child.run(fixture.protocol_sha, fixture.review_path, child.sha(fixture.review_path))
    assert not (fixture.destination / child.LEDGER_NAME).exists()
