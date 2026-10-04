"""Producer-compatible passive integration receipts; no environment or Agent."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import analyze_koi_steering_generalization as analyzer
from scripts import evaluate_koi_steering_generalization as operator
from scripts import finalize_koi_steering_generalization as finalizer
from tests.test_analyze_koi_steering_release_ab import fixture


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def six_object_episode(arm, track=1, seed=3184000001):
    episode, raw = fixture(arm, track, seed)
    first = episode["catalog"]["obstacles"][0]
    obstacles = [dict(first, id=i, y=40. + 70 * i, station=40. + 70 * i,
                      anchor_index=12 + 20 * i) for i in range(6)]
    road = [[0., 0., 0., i * 3.5] for i in range(160)]
    for obstacle in obstacles:
        road[obstacle["anchor_index"]][3] = obstacle["station"]
    episode["catalog"] = dict(track=road,
                              obstacles=obstacles)

    def expand(state):
        station = state["station"]
        state.update(front=[station - o["station"] + 2 for o in obstacles],
                     rear=[station - o["station"] - 2 for o in obstacles],
                     clearance=[max(1., abs(station - o["station"]) - 3) for o in obstacles],
                     lateral_separation=[1.] * 6)

    states = [episode["initial_state"], *raw]
    for state in states:
        expand(state)
    by_time = {state["t"]: state for state in states}
    streamed = []
    for decision in episode["decision_trace"]:
        controller = decision["controller"]
        for endpoint in ("pre", "post"):
            controller["evaluation_only"][endpoint] = deepcopy(
                by_time[controller["evaluation_only"][endpoint]["t"]])
        before = dict(steps=decision["step"] - 1, impact_left=0, obstacle_side=0.)
        after = dict(before, steps=decision["step"])
        controller.update(baseline_pre_act_state=before, baseline_post_act_state=after)
        streamed.extend([
            dict(status="act_intent", baseline_pre_act_state=deepcopy(before)),
            dict(status="act_returned", action=[decision[k] for k in ("steer", "gas", "brake")],
                 baseline_pre_act_state=deepcopy(before), baseline_post_act_state=deepcopy(after),
                 steering_terms=deepcopy(controller["steering_terms"]),
                 steering_release_reason=controller.get("steering_release_reason"),
                 steering_generation_ambiguous_motion_ignored=False),
        ])
    episode.update(geometry_sha256=digest(episode["catalog"]),
                   track_geometry_sha256=digest(episode["catalog"]["track"]),
                   peak_rss_bytes=1024, versions={"synthetic": "zero-env"},
                   runtime_module_paths={"agent": "/synthetic/model/agent.py"})
    return episode, raw, streamed


class SyntheticRun:
    def __init__(self, path, protocol):
        self.path, self.protocol = path, protocol
        self.rows = deepcopy(protocol["schedule"])
        self.ledger = []
        self.operator_error = None
        self.review_pin = {}

    def reset_intent(self, index):
        row = self.rows[index]
        key = row["track_id"], row["seed"], row["mode"]
        self.ledger.append(dict(status="reset_intent", track=key[0], seed=key[1], arm=key[2],
                                slot_id=operator.slot_id(*key), time=1.))
        return row, key

    def process(self, key, status="completed", error=None):
        stem = f"{key[0]}-{key[1]}-{key[2]}"
        receipt = dict(status=status, error=error, wall_time_s=1.)
        original = self.path / f"{stem}.process.json"
        write_json(original, receipt)
        bound = self.path / f"{stem}.bound-process.json"
        write_json(bound, dict(receipt, slot_id=operator.slot_id(*key),
                               operator_sha256=self.protocol["operator_sha256"],
                               original_process_file=original.name,
                               original_process_sha256=analyzer.sha256(original)))
        return bound

    def saved(self, index, failure=None):
        row, key = self.reset_intent(index)
        episode, raw, streamed = six_object_episode(key[2], key[0], key[1])
        stem = f"{key[0]}-{key[1]}-{key[2]}"
        if failure == "reset":
            raw, streamed = [], []
            episode.update(error="synthetic reset failure", retire_reason="agent_reset_error",
                           catalog=None, initial_state=None, geometry_sha256=None,
                           track_geometry_sha256=None, decision_trace=[], steps=0, raw_ticks=0,
                           completed=False, lapTimeMs=None, damage_telemetry_valid=False,
                           collision_telemetry_valid=False)
        elif failure == "act_timeout":
            episode.update(error="synthetic act timeout", retire_reason="act_timeout",
                           completed=False, lapTimeMs=None)
            before = deepcopy(episode["decision_trace"][-1]["controller"]["baseline_post_act_state"])
            streamed.extend([dict(status="act_intent", baseline_pre_act_state=before),
                             dict(status="act_returned", action=[.1, 0., 0.],
                                  baseline_pre_act_state=before, baseline_post_act_state=dict(before),
                                  steering_terms=deepcopy(streamed[-1]["steering_terms"]))])
        raw_path = self.path / f"{stem}.raw.jsonl"
        stream_path = self.path / f"{stem}.decisions.jsonl"
        write_rows(raw_path, raw)
        write_rows(stream_path, streamed)
        episode.update(raw_trace_file=raw_path.name, raw_trace_sha256=analyzer.sha256(raw_path),
                       partial_decisions_file=stream_path.name,
                       partial_decisions_sha256=analyzer.sha256(stream_path))
        output = self.path / f"{stem}.json"
        write_json(output, episode)
        process = self.process(key)
        compact = {k: episode[k] for k in ("mode", "track_id", "seed", "completed", "lapTimeMs",
                    "progress", "damage", "collisions", "retire_reason", "error", "invalid_actions",
                    "steps", "raw_ticks", "peak_rss_bytes")}
        compact.update(status="completed", file=output.name, sha256=analyzer.sha256(output),
                       slot_id=operator.slot_id(*key), process_file=process.name,
                       process_sha256=analyzer.sha256(process))
        row.update(compact)
        self.ledger.append(deepcopy(compact))
        if failure:
            self.operator_error = "AssertionError: saved operational failure"
        return SimpleNamespace(episode=episode, raw=raw, streamed=streamed, output=output,
                               raw_path=raw_path, stream_path=stream_path, process=process, row=row)

    def partial(self, index, timeout=False):
        row, key = self.reset_intent(index)
        row.update(status="operator_error", error="synthetic interrupted child")
        self.operator_error = row["error"]
        if timeout:
            self.process(key, status="operator_error", error="TimeoutExpired: synthetic timeout")
            _, raw, streamed = six_object_episode(key[2], key[0], key[1])
            stem = f"{key[0]}-{key[1]}-{key[2]}"
            write_rows(self.path / f"{stem}.raw.jsonl", raw[:3])
            write_rows(self.path / f"{stem}.decisions.jsonl", streamed[:1])
        self.pin_partial(index)

    def pin_partial(self, index):
        row = self.rows[index]
        output = self.path / f"{row['track_id']}-{row['seed']}-{row['mode']}.json"
        row["partial_artifacts"] = [dict(file=path.name, sha256=analyzer.sha256(path),
                                         bytes=path.stat().st_size)
                                    for suffix in (".json", ".raw.jsonl", ".decisions.jsonl",
                                                   ".process.json", ".bound-process.json",
                                                   ".stdout.txt", ".stderr.txt")
                                    if (path := output.with_suffix(suffix)).exists()]

    def persist(self):
        ledger = self.path / "reset-ledger.jsonl"
        write_rows(ledger, self.ledger)
        write_json(self.path / "episode-report.json",
                   dict(protocol_sha256=analyzer.sha256(self.path / "protocol.json"),
                        rows=self.rows, operator_error=self.operator_error,
                        reset_ledger_sha256=analyzer.sha256(ledger),
                        review_receipt=self.review_pin,
                        complete=self.operator_error is None,
                        track_geometry_sha256={}))

    def load(self):
        self.persist()
        return analyzer.load_run(self.path, self.path / "candidate.zip",
                                 self.path / "candidate.manifest.json")

    def guarded_load(self):
        self.persist()
        report = analyzer.read_json(self.path / "episode-report.json")
        finalizer.verify_partial_artifacts(self.path, report)
        return analyzer.load_run(self.path, self.path / "candidate.zip",
                                 self.path / "candidate.manifest.json")

    def repin_episode(self, saved):
        write_json(saved.output, saved.episode)
        saved.row["sha256"] = analyzer.sha256(saved.output)
        for row in self.ledger:
            if row.get("status") == "completed" and row.get("file") == saved.output.name:
                row["sha256"] = saved.row["sha256"]


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    cells = [(1, analyzer.SEEDS[0])]
    monkeypatch.setattr(analyzer, "CELLS", cells)
    monkeypatch.setattr(analyzer, "SLOTS", 2)
    monkeypatch.setattr(operator, "CELLS", cells)
    source = tmp_path / "synthetic-sources"
    source.mkdir()
    environment = {}
    helpers = {}
    copies = []
    for index in range(142):
        path = source / f"environment-{index}.py"
        path.write_text("# Synthetic source inventory, never imported.\n")
        environment[str(path)] = analyzer.sha256(path)
    runtime = source / "haic/algorithms/koi/steering_release.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("# Synthetic model inventory, never imported.\n")
    helpers[str(runtime)] = analyzer.sha256(runtime)
    for index, (path, sha) in enumerate({**environment, **helpers}.items()):
        copied = tmp_path / f"source-copy-{index}.py"
        copied.write_bytes(Path(path).read_bytes())
        copies.append(dict(source=path, file=copied.name, sha256=sha))
    root_agent = source / "agent.py"
    root_agent.write_text("raise AssertionError('must never construct an Agent')\n")
    protocol = dict(study=operator.STUDY, analysis_spec=deepcopy(analyzer.ANALYSIS_SPEC),
                    cells=[dict(partition="TRAIN", track_id=t, geometry_seed=s, obstacles=True)
                           for t, s in cells], episodes=2, schedule=operator.scheduled_slots(),
                    analyzer_sha256=analyzer.sha256(analyzer.__file__),
                    operator_sha256=analyzer.sha256(operator.__file__),
                    frame_skip=4, warmup_ticks=50, raw_fps=50, max_decisions=1200,
                    steering_terms_schema="koi-steering-terms-v1",
                    environment_source_sha256=environment, helper_source_sha256=helpers,
                    source_copies=copies, reference_evidence=dict(source_sha256={}, prior_road_sha256=[]),
                    root_agent_source_sha256={str(root_agent): analyzer.sha256(root_agent)},
                    model_hashes={analyzer.ARMS[0]: analyzer.BASELINE_SHA,
                                  analyzer.ARMS[1]: analyzer.CANDIDATE_SHA},
                    model_source_sha256={arm: {"agent.py": "synthetic-agent",
                                         "haic_agent/steering_release_runtime.py": helpers[str(runtime)]}
                                         for arm in analyzer.ARMS},
                    runtime_versions={"synthetic": "zero-env"},
                    freshness=dict(audit_receipt=dict(path="synthetic-exposure.json", sha256="synthetic")))
    write_json(tmp_path / "protocol.json", protocol)
    run = SyntheticRun(tmp_path, protocol)
    review = tmp_path / "review.json"
    write_json(review, dict(status="passed", environment_resets=0, reviewer="synthetic-independent-review",
                            protocol_sha256=analyzer.sha256(tmp_path / "protocol.json"),
                            operator_sha256=protocol["operator_sha256"],
                            analyzer_sha256=protocol["analyzer_sha256"], audit_sha256="synthetic"))
    run.review_pin = dict(path=str(review), sha256=analyzer.sha256(review))
    write_json(tmp_path / "execution-review.json", run.review_pin)
    # Only source/model inventories and cross-lane authorization are mocked.
    # The loader's file hashes, review, telemetry, process, and ledger checks run.
    monkeypatch.setattr(operator, "validate_frozen", lambda sha: (deepcopy(protocol), {"synthetic": True}))
    monkeypatch.setattr(operator, "helper_paths", lambda: [runtime])
    monkeypatch.setattr(analyzer.prior.physical, "FROZEN_SOURCE_SHA256", {})
    monkeypatch.setattr(analyzer.prior, "verify_models", lambda *args: None)
    monkeypatch.setattr(operator, "check_claims", lambda *args, **kwargs: {})
    monkeypatch.setattr(operator, "worker", lambda *args: pytest.fail("real worker forbidden"))
    return run


def test_production_counts_remain_fixed():
    assert len(analyzer.SEEDS) == 24
    assert len(analyzer.CELLS) == len(operator.CELLS) == 72
    assert analyzer.SLOTS == operator.EPISODES == 144
    assert len(operator.scheduled_slots()) == 144
    assert analyzer.ANALYSIS_SPEC["objects_per_arm"] == 432


def test_complete_load_validates_real_six_object_episode_chain(synthetic_run):
    for index in range(2):
        synthetic_run.saved(index)
    protocol, slots, episodes, errors = synthetic_run.load()
    assert protocol == synthetic_run.protocol
    assert len(episodes) == len(slots) == 2
    assert errors == []
    assert all(len(report["obstacles"]) == 6 for _, report in episodes.values())
    assert analyzer.summarize(slots, episodes)["gates"]["all_slots_valid"]


@pytest.mark.parametrize("timeout", [False, True])
def test_partial_final_intent_preserves_consumption_and_timeout_evidence(synthetic_run, timeout):
    synthetic_run.saved(0)
    synthetic_run.partial(1, timeout=timeout)
    _, slots, episodes, errors = synthetic_run.load()
    assert len(episodes) == 1
    assert "incomplete_reset_ledger" in errors
    assert any(error.startswith("operator_error:") for error in errors)
    failed = next(slot for slot in slots if slot["status"] == "operator_error")
    if timeout:
        assert failed["partial_raw"]["complete_ticks"] == 3
        assert failed["partial_decisions"]["records"] == 1
        assert failed["partial_process"]["receipt"]["status"] == "operator_error"
        assert "TimeoutExpired" in failed["partial_process"]["receipt"]["error"]
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("failure", ["reset", "act_timeout"])
def test_saved_operational_failure_keeps_raw_and_extra_unapplied_stream(synthetic_run, failure):
    failed = synthetic_run.saved(0, failure=failure)
    _, slots, episodes, errors = synthetic_run.load()
    assert not episodes
    assert any(error.startswith("invalid_episode:") for error in errors)
    receipt = next(slot for slot in slots if "invalid_saved_episode" in slot)["invalid_saved_episode"]
    assert receipt["complete_raw_ticks"] == len(failed.raw)
    assert receipt["decision_records"] == len(failed.streamed)
    if failure == "act_timeout":
        assert len(failed.streamed) == failed.episode["steps"] * 2 + 2
    else:
        assert failed.episode["catalog"] is None and failed.raw == []
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("defect,match", [
    ("stream_action", "streamed action"), ("stream_order", "streamed action event"),
    ("stream_state", "streamed controller"), ("stream_hash", "hash"),
    ("raw_arrays", "raw object arrays"), ("raw_endpoint", "decision/world"),
    ("road", "road-only"), ("catalog", "geometry hash"),
    ("process_identity", "process event/source"), ("process_original", "bound process"),
    ("review", None), ("execution_review", "execution review/report"),
    ("ledger_identity", "reset ledger identity"), ("ledger_order", "reset ledger event order"),
    ("ledger_report", "ledger report"), ("episode_identity", "episode identity"),
])
def test_corruption_rejected_with_rehashed_semantic_receipts(synthetic_run, defect, match):
    saved = synthetic_run.saved(0)
    synthetic_run.saved(1)
    if defect.startswith("stream"):
        if defect == "stream_action":
            saved.streamed[1]["action"][0] += .01
        elif defect == "stream_order":
            saved.streamed[:2] = reversed(saved.streamed[:2])
        elif defect == "stream_state":
            saved.streamed[1]["baseline_post_act_state"]["steps"] = -1
        else:
            saved.streamed.pop()
        write_rows(saved.stream_path, saved.streamed)
        if defect != "stream_hash":
            saved.episode["partial_decisions_sha256"] = analyzer.sha256(saved.stream_path)
            synthetic_run.repin_episode(saved)
    elif defect.startswith("raw"):
        if defect == "raw_arrays":
            saved.raw[0]["rear"].pop()
        else:
            saved.raw[3]["lateral"] += 1
        write_rows(saved.raw_path, saved.raw)
        saved.episode["raw_trace_sha256"] = analyzer.sha256(saved.raw_path)
        synthetic_run.repin_episode(saved)
    elif defect in ("road", "catalog", "episode_identity"):
        if defect == "road":
            saved.episode["track_geometry_sha256"] = "wrong"
        elif defect == "catalog":
            saved.episode["catalog"]["obstacles"][0]["station"] += 1
        else:
            saved.episode["seed"] += 1
        synthetic_run.repin_episode(saved)
    elif defect.startswith("process"):
        process = analyzer.read_json(saved.process)
        process["slot_id" if defect == "process_identity" else "wall_time_s"] = "wrong" if defect == "process_identity" else 99.
        write_json(saved.process, process)
        saved.row["process_sha256"] = analyzer.sha256(saved.process)
        synthetic_run.ledger[1]["process_sha256"] = saved.row["process_sha256"]
    elif defect == "review":
        path = Path(synthetic_run.review_pin["path"])
        review = analyzer.read_json(path)
        review["protocol_sha256"] = "wrong"
        write_json(path, review)
        synthetic_run.review_pin["sha256"] = analyzer.sha256(path)
        write_json(synthetic_run.path / "execution-review.json", synthetic_run.review_pin)
    elif defect == "execution_review":
        write_json(synthetic_run.path / "execution-review.json", dict(synthetic_run.review_pin, sha256="wrong"))
    elif defect == "ledger_identity":
        synthetic_run.ledger[0]["slot_id"] = "wrong"
    elif defect == "ledger_order":
        synthetic_run.ledger[:2] = reversed(synthetic_run.ledger[:2])
    else:
        synthetic_run.ledger[1]["steps"] += 1
    with pytest.raises((ValueError, AssertionError), match=match):
        synthetic_run.load()


@pytest.mark.parametrize("defect", ["bound_identity", "original_pin", "bound_status"])
def test_partial_timeout_process_chain_rejects_corruption(synthetic_run, defect):
    synthetic_run.partial(0, timeout=True)
    row = synthetic_run.rows[0]
    bound_path = synthetic_run.path / f"{row['track_id']}-{row['seed']}-{row['mode']}.bound-process.json"
    bound = analyzer.read_json(bound_path)
    if defect == "bound_identity":
        bound["slot_id"] = "wrong"
    elif defect == "original_pin":
        bound["original_process_sha256"] = "wrong"
    else:
        bound["status"] = "completed"
    write_json(bound_path, bound)
    synthetic_run.pin_partial(0)
    with pytest.raises(ValueError):
        synthetic_run.load()


@pytest.mark.parametrize("defect", ["episode_identity", "raw_denominator", "applied_action", "stream_order"])
def test_saved_operational_failure_still_validates_identity_and_applied_prefix(synthetic_run, defect):
    saved = synthetic_run.saved(0, failure="act_timeout")
    if defect == "episode_identity":
        saved.episode["seed"] += 1
    elif defect == "raw_denominator":
        saved.episode["raw_ticks"] += 1
    else:
        if defect == "applied_action":
            saved.streamed[1]["action"][0] += .01
        else:
            saved.streamed[:2] = reversed(saved.streamed[:2])
        write_rows(saved.stream_path, saved.streamed)
        saved.episode["partial_decisions_sha256"] = analyzer.sha256(saved.stream_path)
    synthetic_run.repin_episode(saved)
    with pytest.raises(ValueError):
        synthetic_run.load()


def test_partial_stream_rejects_malformed_event_even_without_saved_episode(synthetic_run):
    synthetic_run.partial(0, timeout=True)
    row = synthetic_run.rows[0]
    path = synthetic_run.path / f"{row['track_id']}-{row['seed']}-{row['mode']}.decisions.jsonl"
    write_rows(path, [dict(status="not_an_action_event")])
    synthetic_run.pin_partial(0)
    with pytest.raises(ValueError):
        synthetic_run.load()


@pytest.mark.parametrize("defect", ["hash", "bytes"])
def test_guarded_workflow_binds_original_parent_failure_evidence(synthetic_run, defect):
    synthetic_run.partial(0, timeout=True)
    row = synthetic_run.rows[0]
    pin = next(pin for pin in row["partial_artifacts"] if pin["file"].endswith(".raw.jsonl"))
    if defect == "hash":
        raw_path = synthetic_run.path / pin["file"]
        raw, _ = analyzer.read_jsonl(raw_path)
        raw[0]["lateral"] += .5  # Still valid telemetry, but not the pinned original.
        write_rows(raw_path, raw)
    else:
        pin["bytes"] += 1
    with pytest.raises(ValueError):
        synthetic_run.guarded_load()


@pytest.mark.parametrize("defect", ["hash", "bytes"])
def test_frozen_raw_loader_residual_risk_accepts_changed_partial_pins_but_cannot_promote(synthetic_run, defect):
    synthetic_run.partial(0, timeout=True)
    row = synthetic_run.rows[0]
    pin = next(pin for pin in row["partial_artifacts"] if pin["file"].endswith(".raw.jsonl"))
    if defect == "hash":
        raw_path = synthetic_run.path / pin["file"]
        raw, _ = analyzer.read_jsonl(raw_path)
        raw[0]["lateral"] += .5
        write_rows(raw_path, raw)
    else:
        pin["bytes"] += 1
    _, slots, episodes, errors = synthetic_run.load()
    assert not episodes and errors
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


def test_guarded_workflow_retains_valid_partial_timeout_and_failclosed_summary(synthetic_run):
    synthetic_run.saved(0)
    synthetic_run.partial(1, timeout=True)
    _, slots, episodes, errors = synthetic_run.guarded_load()
    assert len(episodes) == 1
    failed = next(slot for slot in slots if slot["status"] == "operator_error")
    assert failed["partial_raw"]["complete_ticks"] == 3
    assert failed["partial_decisions"]["records"] == 1
    assert len(failed["partial_artifacts"]) == 4
    assert not analyzer.summarize(slots, episodes, errors)["gate_passed"]


@pytest.mark.parametrize("case", ["complete", "partial", "changed_partial", "unbound_review"])
def test_finalizer_entrypoint_guards_before_load_and_publishes_only_bound_result(synthetic_run, monkeypatch, case):
    synthetic_run.saved(0)
    if case == "complete":
        synthetic_run.saved(1)
    else:
        synthetic_run.partial(1, timeout=True)
    guard_path = synthetic_run.path / "evidence-guard.json"
    result_path = synthetic_run.path / "authoritative-result.json"
    protocol = synthetic_run.protocol
    guard = dict(protocol_sha256=analyzer.sha256(synthetic_run.path / "protocol.json"),
                 operator_sha256=protocol["operator_sha256"],
                 analyzer_sha256=protocol["analyzer_sha256"],
                 finalizer_sha256=analyzer.sha256(finalizer.__file__))
    write_json(guard_path, guard)
    guard_sha = analyzer.sha256(guard_path)
    review_path = Path(synthetic_run.review_pin["path"])
    review = analyzer.read_json(review_path)
    if case != "unbound_review":
        review.update(finalizer_sha256=guard["finalizer_sha256"], evidence_guard_sha256=guard_sha)
    write_json(review_path, review)
    synthetic_run.review_pin["sha256"] = analyzer.sha256(review_path)
    write_json(synthetic_run.path / "execution-review.json", synthetic_run.review_pin)
    synthetic_run.persist()
    monkeypatch.setattr(finalizer, "GUARD_PATH", guard_path)
    monkeypatch.setattr(finalizer, "RESULT_PATH", result_path)
    monkeypatch.setattr(operator, "ROOT", synthetic_run.path)
    monkeypatch.setattr(operator, "RUN_PATH", Path("."))
    if case == "changed_partial":
        row = synthetic_run.rows[1]
        pin = next(pin for pin in row["partial_artifacts"] if pin["file"].endswith(".raw.jsonl"))
        path = synthetic_run.path / pin["file"]
        raw, _ = analyzer.read_jsonl(path)
        raw[0]["lateral"] += .5
        write_rows(path, raw)
    if case in ("changed_partial", "unbound_review"):
        monkeypatch.setattr(analyzer, "load_run", lambda *args: pytest.fail("guard must reject before frozen loader"))
        with pytest.raises(ValueError):
            finalizer.finalize(guard_sha)
        assert not result_path.exists()
    else:
        summary = finalizer.finalize(guard_sha)
        result = analyzer.read_json(result_path)
        assert result["environment_resets"] == result["model_updates"] == 0
        assert result["finalizer_sha256"] == guard["finalizer_sha256"]
        assert result["evidence_guard_sha256"] == guard_sha
        assert result["review_receipt"] == synthetic_run.review_pin
        assert len(result["partial_failure_time_artifacts_verified"]) == (4 if case == "partial" else 0)
        assert summary["episodes_available"] == (1 if case == "partial" else 2)
        assert summary["gate_passed"] is False  # A tiny synthetic cohort cannot establish generalization.


def test_transient_raw_hit_and_raw_lateral_peak_have_chronological_failure_anchors():
    baseline, base_raw, _ = six_object_episode(analyzer.ARMS[0])
    candidate, raw, _ = six_object_episode(analyzer.ARMS[1])
    tick = raw[38]  # Inside step10, absent from its post endpoint at tick40.
    tick.update(contacts=[0], lateral=11.)
    tick["clearance"][0] = 0.
    assert all(not d["controller"]["evaluation_only"]["post"]["contacts"]
               for d in candidate["decision_trace"])
    assert all(abs(d["controller"]["evaluation_only"]["post"]["lateral"]) < 11
               for d in candidate["decision_trace"])
    for episode in (baseline, candidate):
        episode.update(raw_trace_file="synthetic.raw.jsonl", raw_trace_sha256="synthetic")
    report = analyzer.analyze_episode(candidate, raw)
    trace: Any = analyzer.failure_trace(baseline, candidate, analyzer.analyze_episode(baseline, base_raw),
                                   report, "kept", [0])
    excerpts = trace["arms"][analyzer.ARMS[1]]["excerpts"]
    assert report["obstacles"][0]["whole_episode_safety"]["hit"]
    assert excerpts["first_new_contact"]["anchor"]["t"] == tick["t"]
    assert excerpts["first_new_contact"]["anchor"]["step"] == 10
    assert excerpts["peak_lateral"]["anchor"] == dict(step=10, t=tick["t"], raw_lateral=11.)
