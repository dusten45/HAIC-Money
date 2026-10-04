"""No worlds or policies are constructed by these boundary-composition tests."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts import finalize_koi_steering_generalization_composite as composite
from tests.test_koi_steering_generalization_loader import synthetic_run, write_json


def load_fixture(run, allow_boundary=False):
    run.persist()
    return composite.load_completed_source(run.path, run.protocol,
            composite.analysis.read_json(run.path / "episode-report.json"),
            "reset-ledger.jsonl", run.protocol["schedule"], allow_boundary=allow_boundary)


def test_completed_source_validates_primary_six_object_artifacts(synthetic_run):
    synthetic_run.saved(0)
    synthetic_run.saved(1)
    rows, episodes = load_fixture(synthetic_run)
    assert len(rows) == len(episodes) == 2
    assert all(len(r[1]["obstacles"]) == 6 for r in episodes.values())


@pytest.mark.parametrize("defect", [None, "root", "source", "copy", "extra_schema", "extra_source", "model", "measurement"])
def test_separate_child_source_closure_preserves_original_sources(synthetic_run, defect):
    protocol = deepcopy(synthetic_run.protocol)
    helper_paths = [Path(p) for p in protocol["helper_source_sha256"]]
    extra = synthetic_run.path / "separate-continuation-source.py"
    extra.write_text("# Never imported or executed.\n")
    copied = synthetic_run.path / "separate-source-copy.py"
    copied.write_bytes(extra.read_bytes())
    pin = composite.analysis.sha256(extra)
    protocol["continuation_source_sha256"] = {str(extra): pin}
    protocol["source_copies"].append(dict(source=str(extra), file=copied.name, sha256=pin))
    if defect == "root":
        Path(next(iter(protocol["root_agent_source_sha256"]))).write_text("changed root\n")
    elif defect == "source":
        Path(next(iter(protocol["environment_source_sha256"]))).write_text("changed source\n")
    elif defect == "copy":
        (synthetic_run.path / protocol["source_copies"][0]["file"]).write_text("changed copy\n")
    elif defect == "extra_schema":
        protocol["continuation_source_sha256"]["unexpected"] = "pin"
    elif defect == "extra_source":
        extra.write_text("changed continuation\n")
    elif defect == "model":
        protocol["model_hashes"][composite.analysis.ARMS[1]] = "changed model"
    elif defect == "measurement":
        protocol["analysis_spec"]["version"] = "changed measurement"
    if defect:
        with pytest.raises(ValueError):
            composite.verify_source_closure(synthetic_run.path, protocol, helper_paths, [extra])
    else:
        composite.verify_source_closure(synthetic_run.path, protocol, helper_paths, [extra])


def test_only_explicit_benign_boundary_accepts_unreset_complement(synthetic_run):
    synthetic_run.saved(0)
    synthetic_run.operator_error = "KeyboardInterrupt: "
    synthetic_run.rows[1].update(status="operator_error", error="KeyboardInterrupt: ", partial_artifacts=[])
    rows, episodes = load_fixture(synthetic_run, allow_boundary=True)
    assert len(rows) == 2 and len(episodes) == 1
    with pytest.raises(ValueError, match="unresolved"):
        load_fixture(synthetic_run)


@pytest.mark.parametrize("defect", ["pending", "unexpected_error", "invalid_episode", "compact_lap", "false_byte_count", "wrong_source", "wrong_stream", "censor", "lost_intent", "report_reorder", "completion_id", "extra_field", "bool_track", "intent_field", "intent_time"])
def test_boundary_composition_never_hides_invalid_or_interrupted_evidence(synthetic_run, defect):
    saved = synthetic_run.saved(0)
    if defect in ("pending", "false_byte_count"):
        synthetic_run.partial(1, timeout=True)
        if defect == "false_byte_count":
            synthetic_run.rows[1]["partial_artifacts"][0]["bytes"] += 1
    elif defect == "unexpected_error":
        synthetic_run.operator_error = "AssertionError: source guard failed"
    elif defect == "invalid_episode":
        saved.episode["error"] = "invalid world setup"
        synthetic_run.repin_episode(saved)
    elif defect == "compact_lap":
        saved.row["lapTimeMs"] += 1
        synthetic_run.ledger[1]["lapTimeMs"] += 1
    elif defect == "wrong_source":
        data = composite.analysis.read_json(saved.process)
        data["operator_sha256"] = "wrong"
        write_json(saved.process, data)
        saved.row["process_sha256"] = composite.analysis.sha256(saved.process)
        synthetic_run.ledger[1]["process_sha256"] = saved.row["process_sha256"]
    elif defect == "wrong_stream":
        saved.streamed[1]["action"][0] += .01
        from tests.test_koi_steering_generalization_loader import write_rows
        write_rows(saved.stream_path, saved.streamed)
        saved.episode["partial_decisions_sha256"] = composite.analysis.sha256(saved.stream_path)
        synthetic_run.repin_episode(saved)
    elif defect == "censor":
        saved.episode.update(completed=False, lapTimeMs=None, retire_reason="max_steps")
        saved.row.update(completed=False, lapTimeMs=None, retire_reason="max_steps")
        synthetic_run.ledger[1].update(completed=False, lapTimeMs=None, retire_reason="max_steps")
        synthetic_run.repin_episode(saved)
    elif defect == "lost_intent":
        synthetic_run.ledger.pop(0)
    elif defect == "completion_id":
        saved.row["slot_id"] = "wrong"
        synthetic_run.ledger[1]["slot_id"] = "wrong"
    elif defect == "extra_field":
        saved.row["unexpected"] = True
        synthetic_run.ledger[1]["unexpected"] = True
    elif defect == "bool_track":
        saved.row["track_id"] = True
        synthetic_run.ledger[1]["track_id"] = True
    elif defect == "intent_field":
        synthetic_run.ledger[0]["unexpected"] = True
    elif defect == "intent_time":
        synthetic_run.ledger[0]["time"] = True
    else:
        synthetic_run.rows.reverse()
    if not synthetic_run.operator_error:
        synthetic_run.operator_error = "KeyboardInterrupt: "
    with pytest.raises(ValueError):
        load_fixture(synthetic_run, allow_boundary=True)


@pytest.mark.parametrize("boundary", [0, 1, 2, 35, 72, 143, 144])
def test_exact_canonical_union_without_repeated_completed_slots(boundary):
    schedule = composite.original.scheduled_slots()
    rows = {composite.key(r): dict(r, status="completed" if i < boundary else "unrun") for i, r in enumerate(schedule)}
    parent = {composite.key(r): (None, None) for r in schedule[:boundary]}
    child = {composite.key(r): (None, None) for r in schedule[boundary:]}
    child_rows = {composite.key(r): dict(r, status="completed") for r in schedule[boundary:]}
    merged_rows, merged, counts = composite.combine(rows, parent, child_rows, child, schedule[boundary:])
    assert len(merged_rows) == len(merged) == 144
    assert counts["actual_reset_intents"] == 144
    assert counts["repeated_completed_slots"] == 0


@pytest.mark.parametrize("defect", ["repeat", "missing", "reorder", "extra", "incomplete"])
def test_incomplete_or_noncomplementary_continuation_is_not_generalization(defect):
    schedule = composite.original.scheduled_slots()
    split = 41
    rows = {composite.key(r): dict(r, status="completed" if i < split else "unrun") for i, r in enumerate(schedule)}
    parent = {composite.key(r): (None, None) for r in schedule[:split]}
    remaining = deepcopy(schedule[split:])
    child = {composite.key(r): (None, None) for r in remaining}
    child_rows = {composite.key(r): dict(r, status="completed") for r in remaining}
    if defect == "repeat":
        k = next(iter(parent))
        child[k] = parent[k]
    elif defect == "missing":
        child.pop(next(iter(child)))
    elif defect == "reorder":
        remaining.reverse()
    elif defect == "extra":
        child_rows[(4, 3184000099, "crossing_projection")] = dict(status="completed")
    else:
        next(iter(child_rows.values()))["status"] = "operator_error"
    with pytest.raises(ValueError):
        composite.combine(rows, parent, child_rows, child, remaining)
