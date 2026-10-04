"""Synthetic composite entrypoints: real receipts, no policies/worlds/signals."""

from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import scripts
from scripts import finalize_koi_steering_generalization_composite as composite
from tests.test_koi_steering_generalization_loader import (
    SyntheticRun, synthetic_run, write_json, write_rows,
)


PRODUCTION_SCHEDULE = deepcopy(composite.original.scheduled_slots())
PRODUCTION_CELLS = deepcopy(composite.analysis.CELLS)
CHILD_MODULE = "scripts.evaluate_koi_steering_generalization_continuation"


@pytest.fixture
def entrypoint(synthetic_run, monkeypatch):
    original, analysis = composite.original, composite.analysis
    root = synthetic_run.path
    child_run = root / "continuation"
    child_run.mkdir()
    monkeypatch.setattr(original, "ROOT", root)
    monkeypatch.setattr(original, "RUN_PATH", Path("."))
    monkeypatch.setattr(original, "PROTOCOL_PATH", Path("protocol.json"))
    monkeypatch.setattr(composite, "GUARD_PATH", root / "composite-guard.json")
    monkeypatch.setattr(composite, "RESULT_PATH", root / "composite-result.json")
    monkeypatch.setattr(composite.forensic, "GUARD_PATH", root / "original-guard.json")
    monkeypatch.setattr(composite.forensic, "RESULT_PATH", root / "original-result.json")
    write_json(composite.forensic.GUARD_PATH, {"environment_resets": 0})
    write_json(composite.forensic.RESULT_PATH, {"gate_passed": False, "decision": "INCOMPLETE"})

    synthetic_run.protocol["candidate_manifest_sha256"] = "synthetic-manifest"
    old_operator = synthetic_run.protocol["operator_sha256"]
    extra_paths = []
    for name in ("child-operator.py", "pause-proof.py", "guard-finalizer.py", "composite-finalizer.py"):
        path = root / name
        path.write_text("# Synthetic authorization source, never imported. " + name + "\n")
        extra_paths.append(path)
    protocol = deepcopy(synthetic_run.protocol)
    protocol.update(schedule=deepcopy(protocol["schedule"][1:]), episodes=1,
                    operator_sha256=analysis.sha256(extra_paths[0]),
                    original_worker_sha256=old_operator,
                    parent_result={"path": str(composite.forensic.RESULT_PATH),
                                   "sha256": analysis.sha256(composite.forensic.RESULT_PATH)},
                    continuation_source_sha256={str(p): analysis.sha256(p) for p in extra_paths})
    for row in protocol["source_copies"]:
        (child_run / row["file"]).write_bytes((root / row["file"]).read_bytes())
    for index, path in enumerate(extra_paths):
        copied = child_run / f"extra-copy-{index}.py"
        copied.write_bytes(path.read_bytes())
        protocol["source_copies"].append(dict(source=str(path), file=copied.name,
                                              sha256=analysis.sha256(path)))
    continuation = SyntheticRun(child_run, protocol)
    original_persist = continuation.persist

    def persist_child():
        original_persist()
        write_rows(child_run / "reset-intents.jsonl", continuation.ledger)

    monkeypatch.setattr(continuation, "persist", persist_child)
    synthetic_run.saved(0)
    synthetic_run.operator_error = "KeyboardInterrupt: "
    synthetic_run.rows[1].update(status="operator_error", error="KeyboardInterrupt: ",
                                 partial_artifacts=[])
    continuation.saved(0)

    fake: Any = ModuleType(CHILD_MODULE)
    fake.__file__ = str(extra_paths[0])
    fake.RUN_PATH = Path("continuation")
    fake.helper_paths = lambda: list(extra_paths)
    state = SimpleNamespace(parent=synthetic_run, child=continuation, root=root,
                            module=fake, boundary_valid=True, calls=[], extra_paths=extra_paths,
                            original_result=composite.forensic.RESULT_PATH.read_bytes(), seals=0)

    def validate_frozen(expected_sha):
        state.calls.append("validate_frozen")
        if not state.boundary_valid:
            raise ValueError("independent terminal boundary invalid")
        if analysis.sha256(child_run / "protocol.json") != expected_sha:
            raise ValueError("child protocol pin differs")
        actual = analysis.read_json(child_run / "protocol.json")
        pin = actual["parent_result"]
        if analysis.sha256(pin["path"]) != pin["sha256"]:
            raise ValueError("preserved parent result changed")
        return actual, {"synthetic_authorization": True}

    def validate_review(actual, expected_sha, path, pin):
        state.calls.append("validate_review")
        review = original.validate_review(actual, expected_sha, path, pin)
        if review["original_worker_sha256"] != actual["original_worker_sha256"]:
            raise ValueError("private worker provenance changed")
        return review

    fake.validate_frozen = validate_frozen
    fake.validate_review = validate_review
    fake.verified_parent = lambda *args: {"boundary": {"independently_verified": state.boundary_valid}}
    fake.worker = lambda *args: pytest.fail("no actual worker permitted")
    monkeypatch.setitem(sys.modules, CHILD_MODULE, fake)
    monkeypatch.setattr(scripts, "evaluate_koi_steering_generalization_continuation", fake, raising=False)

    def seal():
        if state.seals:
            monkeypatch.setattr(composite, "GUARD_PATH", root / f"composite-guard-{state.seals}.json")
        state.seals += 1
        write_json(root / "protocol.json", state.parent.protocol)
        write_json(child_run / "protocol.json", state.child.protocol)
        parent_sha = analysis.sha256(root / "protocol.json")
        child_sha = analysis.sha256(child_run / "protocol.json")
        parent_review = root / "parent-review.json"
        write_json(parent_review, dict(status="passed", environment_resets=0,
                   reviewer="synthetic-independent", protocol_sha256=parent_sha,
                   operator_sha256=state.parent.protocol["operator_sha256"],
                   analyzer_sha256=state.parent.protocol["analyzer_sha256"], audit_sha256="synthetic",
                   finalizer_sha256=analysis.sha256(composite.forensic.__file__),
                   evidence_guard_sha256=analysis.sha256(composite.forensic.GUARD_PATH)))
        state.parent.review_pin = dict(path=str(parent_review), sha256=analysis.sha256(parent_review))
        write_json(root / "execution-review.json", state.parent.review_pin)
        state.parent.persist()
        state.child.persist()
        state.guard = composite.freeze_guard(child_sha)
        state.guard_sha = state.guard["sha256"]
        child_review = root / "child-review.json"
        write_json(child_review, dict(status="passed", environment_resets=0,
                   reviewer="synthetic-independent", protocol_sha256=child_sha,
                   operator_sha256=state.child.protocol["operator_sha256"],
                   analyzer_sha256=state.child.protocol["analyzer_sha256"], audit_sha256="synthetic",
                   original_worker_sha256=state.child.protocol["original_worker_sha256"],
                   composite_finalizer_sha256=state.guard["composite_finalizer_sha256"],
                   composite_guard_sha256=state.guard_sha))
        state.child.review_pin = dict(path=str(child_review), sha256=analysis.sha256(child_review))
        write_json(child_run / "execution-review.json", state.child.review_pin)
        state.child.persist()

    state.seal = seal
    seal()
    return state


def test_production_census_is_144_slots_72_pairs_24_geometries():
    assert len(PRODUCTION_SCHEDULE) == 144
    assert len(PRODUCTION_CELLS) == 72
    assert len({r["seed"] for r in PRODUCTION_SCHEDULE}) == 24
    assert composite.analysis.SLOTS == 144
    assert composite.analysis.ANALYSIS_SPEC["objects_per_arm"] == 432


def test_entrypoint_real_artifacts_and_original_analysis_publish_separate_result(entrypoint, monkeypatch):
    original_summary = composite.analysis.summarize
    observed = []

    def inspect_summary(rows, episodes, *args):
        observed.append((rows, episodes))
        assert len(rows) == len(episodes) == 2
        assert all(len(report["obstacles"]) == 6 for _, report in episodes.values())
        return original_summary(rows, episodes, *args)

    monkeypatch.setattr(composite.analysis, "summarize", inspect_summary)
    result = composite.finalize(entrypoint.guard_sha)
    published = composite.analysis.read_json(composite.RESULT_PATH)
    assert len(observed) == 1
    assert result["source_census"] == dict(parent_completed=1, continuation_completed=1,
                                         completed_unique=2, actual_reset_intents=2,
                                         repeated_completed_slots=0)
    assert published["gates"]["all_slots_valid"]
    assert not published["gate_passed"]  # Two synthetic slots do not meet production coverage gates.
    assert published["analysis_spec"] == composite.analysis.ANALYSIS_SPEC
    assert published["original_attempt_operator_error"].startswith("KeyboardInterrupt:")
    assert published["original_attempt_result_sha256"] == composite.analysis.sha256(composite.forensic.RESULT_PATH)
    assert published["official_action"] is False and published["model_updates"] == published["environment_resets"] == 0
    assert composite.forensic.RESULT_PATH.read_bytes() == entrypoint.original_result
    assert composite.RESULT_PATH != composite.forensic.RESULT_PATH
    assert entrypoint.child.protocol["operator_sha256"] != entrypoint.child.protocol["original_worker_sha256"]
    assert entrypoint.child.protocol["helper_source_sha256"] == entrypoint.parent.protocol["helper_source_sha256"]
    assert set(entrypoint.child.protocol["continuation_source_sha256"]) == {str(p) for p in entrypoint.extra_paths}


@pytest.mark.parametrize("operation", ["freeze", "finalize"])
def test_forged_keyboard_interrupt_cannot_bypass_independent_boundary(entrypoint, monkeypatch, operation):
    entrypoint.boundary_valid = False
    monkeypatch.setattr(composite, "load_completed_source", lambda *args, **kwargs: pytest.fail("boundary must reject before analysis"))
    monkeypatch.setattr(composite.analysis, "summarize", lambda *args: pytest.fail("boundary must reject before summary"))
    with pytest.raises(ValueError, match="boundary invalid"):
        if operation == "freeze":
            composite.freeze_guard(composite.analysis.sha256(entrypoint.child.path / "protocol.json"))
        else:
            composite.finalize(entrypoint.guard_sha)
    assert not composite.RESULT_PATH.exists()


@pytest.mark.parametrize("defect", ["child_review_hash", "child_review_guard", "child_review_source",
                                  "child_review_missing_guard", "child_review_worker",
                                  "child_execution", "parent_review_hash", "parent_review_guard",
                                  "parent_review_missing_guard",
                                  "parent_execution", "composite_guard", "original_guard", "parent_result"])
def test_review_guard_execution_and_preserved_result_pins(entrypoint, monkeypatch, defect):
    analysis = composite.analysis
    if defect in ("child_execution", "parent_execution"):
        run = entrypoint.child if defect == "child_execution" else entrypoint.parent
        write_json(run.path / "execution-review.json", {"path": "wrong", "sha256": "wrong"})
    elif defect in ("composite_guard", "original_guard", "parent_result"):
        path = {"composite_guard": composite.GUARD_PATH, "original_guard": composite.forensic.GUARD_PATH,
                "parent_result": composite.forensic.RESULT_PATH}[defect]
        write_json(path, {"tampered": True})
    else:
        run = entrypoint.parent if defect.startswith("parent") else entrypoint.child
        path = Path(run.review_pin["path"])
        review = analysis.read_json(path)
        field = "evidence_guard_sha256" if defect.startswith("parent_review") else "composite_guard_sha256"
        if defect == "child_review_source":
            field = "operator_sha256"
        elif defect == "child_review_worker":
            field = "original_worker_sha256"
        if "missing" in defect:
            review.pop(field)
        else:
            review[field] = "changed"
        write_json(path, review)
        if not defect.endswith("hash"):
            run.review_pin["sha256"] = analysis.sha256(path)
            write_json(run.path / "execution-review.json", run.review_pin)
            run.persist()
    monkeypatch.setattr(composite.analysis, "summarize", lambda *args: pytest.fail("invalid binding reached summary"))
    with pytest.raises((ValueError, AssertionError)):
        composite.finalize(entrypoint.guard_sha)
    assert not composite.RESULT_PATH.exists()


def test_guard_freeze_is_exclusive_and_does_not_rewrite_parent_attempt(entrypoint):
    guard_bytes = composite.GUARD_PATH.read_bytes()
    with pytest.raises(FileExistsError):
        composite.freeze_guard(composite.analysis.sha256(entrypoint.child.path / "protocol.json"))
    assert composite.GUARD_PATH.read_bytes() == guard_bytes
    assert composite.forensic.RESULT_PATH.read_bytes() == entrypoint.original_result


@pytest.mark.parametrize("field", ["model_hashes", "model_source_sha256", "candidate_manifest_sha256",
                                  "runtime_versions", "environment_source_sha256", "analysis_spec"])
def test_parent_child_model_environment_runtime_and_gate_boundary(entrypoint, monkeypatch, field):
    protocol = entrypoint.child.protocol
    if field == "environment_source_sha256":
        old = next(iter(protocol[field]))
        new = entrypoint.root / "replacement-environment.py"
        new.write_bytes(Path(old).read_bytes())
        protocol[field][str(new)] = protocol[field].pop(old)
        next(row for row in protocol["source_copies"] if row["source"] == old)["source"] = str(new)
    elif isinstance(protocol[field], dict):
        protocol[field]["changed"] = "different"
    else:
        protocol[field] = "different"
    entrypoint.seal()
    monkeypatch.setattr(composite.analysis, "summarize", lambda *args: pytest.fail("unmatched protocol reached summary"))
    with pytest.raises((ValueError, AssertionError)):
        composite.finalize(entrypoint.guard_sha)
    assert not composite.RESULT_PATH.exists()


@pytest.mark.parametrize("defect", ["raw", "stream", "process", "ledger", "completed_id", "extra_helper", "source_copy"])
def test_entrypoint_authenticates_actual_child_artifacts(entrypoint, monkeypatch, defect):
    run, analysis = entrypoint.child, composite.analysis
    row = run.rows[0]
    episode_path = run.path / row["file"]
    episode = analysis.read_json(episode_path)
    if defect in ("raw", "stream"):
        path = run.path / episode["raw_trace_file" if defect == "raw" else "partial_decisions_file"]
        path.write_text(path.read_text() + "{}\n")
    elif defect == "process":
        path = run.path / row["process_file"]
        process = analysis.read_json(path)
        process["operator_sha256"] = run.protocol["original_worker_sha256"]
        write_json(path, process)
        row["process_sha256"] = analysis.sha256(path)
        run.ledger[1]["process_sha256"] = row["process_sha256"]
        run.persist()
    elif defect == "ledger":
        run.ledger[0]["time"] = True
        run.persist()
    elif defect == "completed_id":
        row["slot_id"] = "wrong"
        run.ledger[1]["slot_id"] = "wrong"
        run.persist()
    elif defect == "extra_helper":
        entrypoint.extra_paths[1].write_text("changed source\n")
    else:
        (run.path / run.protocol["source_copies"][0]["file"]).write_text("changed copied bytes\n")
    monkeypatch.setattr(composite.analysis, "summarize", lambda *args: pytest.fail("invalid artifact reached summary"))
    with pytest.raises((ValueError, AssertionError)):
        composite.finalize(entrypoint.guard_sha)
    assert not composite.RESULT_PATH.exists()


@pytest.mark.parametrize("defect", [None, "missing", "repeat", "reorder", "extra", "incomplete"])
def test_full_canonical_union_reaches_original_analysis_entrypoint(entrypoint, monkeypatch, defect):
    schedule = deepcopy(PRODUCTION_SCHEDULE)
    split = 35
    monkeypatch.setattr(composite.original, "scheduled_slots", lambda: deepcopy(schedule))
    monkeypatch.setattr(composite.original, "CELLS", deepcopy(PRODUCTION_CELLS))
    monkeypatch.setattr(composite.analysis, "CELLS", deepcopy(PRODUCTION_CELLS))
    monkeypatch.setattr(composite.analysis, "SLOTS", 144)
    entrypoint.parent.protocol["schedule"] = schedule
    entrypoint.child.protocol["schedule"] = deepcopy(schedule[split:])
    entrypoint.seal()
    parent_rows = {composite.key(row): dict(row, status="completed" if i < split else "unrun")
                   for i, row in enumerate(schedule)}
    child_rows = {composite.key(row): dict(row, status="completed") for row in schedule[split:]}
    parent_episodes = {composite.key(row): (None, None) for row in schedule[:split]}
    child_episodes = {composite.key(row): (None, None) for row in schedule[split:]}
    if defect == "missing":
        child_episodes.pop(next(iter(child_episodes)))
    elif defect == "repeat":
        child_episodes[next(iter(parent_episodes))] = (None, None)
    elif defect == "reorder":
        entrypoint.child.protocol["schedule"].reverse()
        entrypoint.seal()
    elif defect == "extra":
        child_rows[(4, 3184000099, composite.analysis.ARMS[0])] = dict(status="completed")
    elif defect == "incomplete":
        next(iter(child_rows.values()))["status"] = "operator_error"
    calls = []

    def load(run, protocol, report, ledger, expected, allow_boundary=False):
        calls.append((ledger, allow_boundary))
        if allow_boundary:
            assert expected == schedule
            return parent_rows, parent_episodes
        assert expected == entrypoint.child.protocol["schedule"]
        return child_rows, child_episodes

    def numerical_stub(rows, episodes):
        assert defect is None, "incomplete/noncanonical union reached analysis"
        assert [composite.key(row) for row in rows] == [composite.key(row) for row in schedule]
        assert len(episodes) == 144 and len(composite.analysis.CELLS) == 72
        assert len({k[1] for k in episodes}) == 24
        assert parent_episodes.keys().isdisjoint(child_episodes)
        return dict(episodes_available=144, completion={"kept": 72}, gates={"synthetic": True},
                    gate_passed=True, decision="SYNTHETIC_NUMERIC_STUB")

    monkeypatch.setattr(composite, "load_completed_source", load)
    monkeypatch.setattr(composite.analysis, "summarize", numerical_stub)
    if defect:
        with pytest.raises(ValueError):
            composite.finalize(entrypoint.guard_sha)
        assert not composite.RESULT_PATH.exists()
        return
    result = composite.finalize(entrypoint.guard_sha)
    assert calls == [("reset-ledger.jsonl", True), ("reset-intents.jsonl", False)]
    assert result["source_census"]["completed_unique"] == result["source_census"]["actual_reset_intents"] == 144
    assert result["source_census"]["parent_completed"] == split
    assert composite.forensic.RESULT_PATH.read_bytes() == entrypoint.original_result
