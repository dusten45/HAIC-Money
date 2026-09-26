"""Isolated persistence contracts; all data writes use temporary fixtures."""

import json
import os
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from haic_research.config import load_config
from haic_research.models import (
    CheckpointRef, GateResult, GateStatus, IntegrationReport, RunEvent,
    RunManifest, WorkflowState,
)
from haic_research.records import (
    PathSafetyError, RecordError, RecordLockError, append_event, create_run,
    read_events, read_manifest, write_integration_report,
)

ROOT = Path(__file__).resolve().parents[1]
GATES = ("rule_compliance", "mechanism_activation", "competitive_or_product_outcome")
NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def manifest(run_id="run-001", **changes):
    item = RunManifest(
        run_id=run_id, purpose="test", hypothesis_hash="hypothesis", approval_hash="",
        candidate_revision="candidate", control_revision="control", candidate_package_hash="c",
        control_package_hash="p", tool_versions={}, runtime_versions={}, data_ids=(),
        map_ids=(), split_ids=(), resource_limits={}, permission_limits={}, output_paths=(),
        source_hashes={}, plan_hash="plan-001", cycle_id="cycle-001",
    )
    return replace(item, **changes)


def event(event_id="e1", kind="STATE", state=WorkflowState.DISCOVER, **changes):
    return RunEvent(changes.pop("timestamp", NOW), kind, state, event_id=event_id, **changes)


def report(status=GateStatus.PASS):
    return IntegrationReport(tuple(GateResult(name, status) for name in GATES), ())


class RunRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        (self.root / "harness.config.json").write_text(
            (ROOT / "harness.config.json").read_text(encoding="utf-8"), encoding="utf-8")
        self.config = load_config(self.root)
        self.manifest = manifest()
        self.run_dir = create_run(self.config, self.manifest)

    def append(self, item):
        append_event(self.run_dir, item, config=self.config)

    def finish(self, outcome=WorkflowState.REVISE, checkpoint=None):
        decision = event("decision", "CYCLE_DECISION", outcome, checkpoint_ref=checkpoint)
        self.append(decision)
        self.append(event("review", "GATE_REVIEW", WorkflowState[f"GATE_REVIEW_{outcome.value}"]))
        write_integration_report(self.run_dir, report(), config=self.config)
        self.append(event("stop", state=WorkflowState.STOPPED))
        return decision

    def linked(self, **changes):
        return manifest("run-002", plan_hash="plan-002", cycle_id="cycle-002",
                        predecessor_run_id="run-001", predecessor_decision_ref="decision", **changes)

    def redirect(self, link, target, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError:
            if os.name != "nt" or not directory:
                self.skipTest("symlinks unavailable")
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True, text=True)
            if result.returncode:
                self.skipTest("junctions unavailable")

    def test_round_trip_frozen_metadata_and_stable_json(self):
        self.assertEqual(self.run_dir.parent, self.config.run_root)
        self.assertEqual(read_manifest(self.run_dir, config=self.config), self.manifest)
        self.append(event(resource_usage={"cpu": {"seconds": 3}, "samples": [1, 2]}))
        self.append(event("e2", state=WorkflowState.HYPOTHESIZE))
        rows = (self.run_dir / "events.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[0])["resource_usage"]["cpu"]["seconds"], 3)
        self.assertEqual(read_events(self.run_dir, config=self.config)[0].resource_usage["samples"], (1, 2))
        self.assertEqual(read_events(self.run_dir, config=self.config)[0].timestamp, NOW)

    def test_required_config_for_all_existing_run_apis(self):
        for call, args in ((read_manifest, (self.run_dir,)), (read_events, (self.run_dir,)),
                           (append_event, (self.run_dir, event())),
                           (write_integration_report, (self.run_dir, report()))):
            with self.assertRaises(TypeError):
                call(*args)

    def test_reused_and_invalid_run_ids_fail_before_writes(self):
        before = (self.run_dir / "run_manifest.json").read_bytes()
        for run_id in ("run-001", "..", "../escape", "a/b", "a\\b", "CON", "nul.txt",
                       "COM1", "LPT9.log", "bad.", "bad ", "a:b", "", "RUN-001"):
            with self.subTest(run_id=run_id), self.assertRaises((PathSafetyError, RecordError, ValueError)):
                create_run(self.config, manifest(run_id))
        self.assertEqual((self.run_dir / "run_manifest.json").read_bytes(), before)

    def test_config_is_validated_before_creating_root(self):
        for path in (self.root / "artifacts/haic", self.root / "submissions", self.root / "../outside"):
            bad = replace(self.config, run_root=path)
            with self.assertRaises(PathSafetyError):
                create_run(bad, manifest("bad"))
        bad = replace(self.config, primary_metric="other", run_root=self.root / "new-root")
        with self.assertRaises(PathSafetyError):
            create_run(bad, manifest("bad"))
        self.assertFalse((self.root / "new-root").exists())

    def test_read_write_refuse_outside_legacy_nested_and_alias_before_manifest_read(self):
        for path in (self.root / "runs/old", self.root / "artifacts/haic/old",
                     self.root / "submissions/old", self.run_dir / "nested",
                     self.run_dir / ".." / "run-001", self.root / "outside",
                     self.config.run_root / "RUN-001"):
            for call, args in ((read_manifest, (path,)), (read_events, (path,)),
                               (append_event, (path, event())),
                               (write_integration_report, (path, report()))):
                with self.subTest(path=path, call=call), self.assertRaises(PathSafetyError):
                    with patch("haic_research.records._read_json", side_effect=AssertionError("must not read")):
                        call(*args, config=self.config)

    def test_manifest_must_match_directory_and_records_must_be_well_formed(self):
        path = self.run_dir / "run_manifest.json"
        data = json.loads(path.read_text())
        data["run_id"] = "other"
        path.write_text(json.dumps(data))
        with self.assertRaises(PathSafetyError):
            read_manifest(self.run_dir, config=self.config)
        path.write_text("{}")
        with self.assertRaises(RecordError):
            read_manifest(self.run_dir, config=self.config)

    def test_root_and_run_directory_redirects_are_refused(self):
        alias = self.config.run_root / "run-alias"
        self.redirect(alias, self.run_dir, directory=True)
        with self.assertRaises(PathSafetyError):
            read_manifest(alias, config=self.config)
        redirected = self.root / "redirected-root"
        self.redirect(redirected, self.config.run_root, directory=True)
        with self.assertRaises(PathSafetyError):
            create_run(replace(self.config, run_root=redirected), manifest("new"))

    def test_hardlinked_and_redirected_record_files_are_refused(self):
        for filename in ("run_manifest.json", "events.jsonl", "integration_report.json"):
            with self.subTest(filename=filename):
                other = self.root / (filename + ".other")
                other.write_text("{}")
                path = self.run_dir / filename
                original = path.read_bytes() if path.exists() else None
                path.unlink(missing_ok=True)
                os.link(other, path)
                for call in (read_manifest, read_events):
                    with self.assertRaises(PathSafetyError):
                        call(self.run_dir, config=self.config)
                with self.assertRaises(PathSafetyError):
                    self.append(event())
                with self.assertRaises(PathSafetyError):
                    write_integration_report(self.run_dir, report(), config=self.config)
                path.unlink()
                if original is not None:
                    path.write_bytes(original)

    def test_file_symlink_refused_without_changing_target(self):
        target = self.root / "target.json"
        target.write_text("untouched")
        path = self.run_dir / "events.jsonl"
        path.unlink()
        self.redirect(path, target)
        with self.assertRaises(PathSafetyError):
            self.append(event())
        self.assertEqual(target.read_text(), "untouched")

    def test_event_ids_time_and_corrections_are_validated(self):
        self.append(event())
        for item in (event(), event("old", timestamp=NOW - timedelta(seconds=1)),
                     event("bad-ref", correction_ref="missing"),
                     event("self-ref", correction_ref="self-ref"),
                     event("no-report", correction_ref="integration_report.json")):
            with self.assertRaises(RecordError):
                self.append(item)
        self.append(event("fix", correction_ref="e1"))
        write_integration_report(self.run_dir, report(), config=self.config)
        self.append(event("report-fix", correction_ref="integration_report.json"))
        self.assertEqual(len(read_events(self.run_dir, config=self.config)), 3)

    def test_utc_and_nonempty_model_fields_required(self):
        with self.assertRaises(ValueError):
            event(timestamp=datetime(2026, 9, 26))
        with self.assertRaises(ValueError):
            event(event_id="")
        for kwargs in ({"cycle_id": ""}, {"plan_hash": ""}):
            with self.assertRaises(ValueError):
                manifest(**kwargs)
        with self.assertRaises(ValueError):
            CheckpointRef("path", "")

    def test_approval_fields_match_current_plan_and_are_round_tripped(self):
        for fields in ({}, {"approval_stage": "wrong", "approved_plan_hash": "plan-001"},
                       {"approval_stage": "design", "approved_plan_hash": "old"}):
            with self.assertRaises((ValueError, RecordError)):
                self.append(event("approval", "APPROVAL", **fields))
        self.append(event("approval", "APPROVAL", approval_stage="design",
                          approved_plan_hash="plan-001", approval_ref="user:42"))
        approved = read_events(self.run_dir, config=self.config)[0]
        self.assertEqual((approved.event_id, approved.approval_stage, approved.approved_plan_hash),
                         ("approval", "design", "plan-001"))
        with self.assertRaises(ValueError):
            event("mislabel", approval_stage="design", approved_plan_hash="plan-001")

    def test_reports_are_exclusive_and_require_each_gate_exactly_once(self):
        for gates in ((), report().gate_results[:2], report().gate_results + report().gate_results[:1],
                      report().gate_results + (GateResult("unknown", GateStatus.PASS),)):
            with self.assertRaises(RecordError):
                write_integration_report(self.run_dir, IntegrationReport(gates, ()), config=self.config)
        write_integration_report(self.run_dir, report(GateStatus.UNKNOWN), config=self.config)
        original = (self.run_dir / "integration_report.json").read_bytes()
        with self.assertRaises(RecordError):
            write_integration_report(self.run_dir, report(), config=self.config)
        self.assertEqual((self.run_dir / "integration_report.json").read_bytes(), original)

    def test_concurrent_and_stale_locks_fail_closed(self):
        lock = self.run_dir / ".records.lock"
        lock.write_text("stale or concurrent")
        for call, args in ((append_event, (self.run_dir, event())),
                           (write_integration_report, (self.run_dir, report())),
                           (read_manifest, (self.run_dir,)), (read_events, (self.run_dir,))):
            with self.assertRaises(RecordLockError):
                call(*args, config=self.config)
        self.assertEqual(lock.read_text(), "stale or concurrent")
        self.assertEqual((self.run_dir / "events.jsonl").read_text(), "")

    def test_concurrent_writer_is_refused_without_losing_first_event(self):
        import haic_research.records as records
        entered, release = threading.Event(), threading.Event()
        failures = []
        encode = records._encode
        def pause_encode(value):
            entered.set()
            if not release.wait(5):
                raise AssertionError("writer was not released")
            return encode(value)
        def writer():
            try:
                self.append(event())
            except Exception as exc:
                failures.append(exc)
        with patch.object(records, "_encode", side_effect=pause_encode):
            thread = threading.Thread(target=writer)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(RecordLockError):
                    self.append(event("concurrent"))
                with self.assertRaises(RecordLockError):
                    write_integration_report(self.run_dir, report(), config=self.config)
            finally:
                release.set()
                thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual([item.event_id for item in read_events(self.run_dir, config=self.config)], ["e1"])

    def test_checkpoint_hardlink_is_refused_without_reading_content(self):
        self.config.artifact_root.mkdir(parents=True)
        source = self.root / "outside.pt"
        source.write_bytes(b"checkpoint data must remain untouched")
        target = self.config.artifact_root / "linked.pt"
        os.link(source, target)
        with self.assertRaises(PathSafetyError):
            create_run(self.config, manifest("bad", checkpoint_ref=CheckpointRef(
                "artifacts/haic-research-v2/linked.pt", "hash")))
        self.assertEqual(source.read_bytes(), b"checkpoint data must remain untouched")

    def test_nonpassing_complete_reports_allow_continuation_without_release_permission(self):
        for index, status in enumerate(GateStatus):
            prior = manifest(f"parent-{index}")
            prior_dir = create_run(self.config, prior)
            append_event(prior_dir, event("decision", "CYCLE_DECISION", WorkflowState.REVISE), config=self.config)
            append_event(prior_dir, event("review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_REVISE), config=self.config)
            write_integration_report(prior_dir, report(status), config=self.config)
            append_event(prior_dir, event("stop", state=WorkflowState.STOPPED), config=self.config)
            child = replace(self.linked(), run_id=f"child-{index}", predecessor_run_id=prior.run_id)
            self.assertTrue(create_run(self.config, child, previous_run_dir=prior_dir).exists())

    def test_decision_checkpoint_overrides_initial_manifest_checkpoint(self):
        old = CheckpointRef("artifacts/haic-research-v2/old.pt", "old-hash")
        selected = CheckpointRef("artifacts/haic-research-v2/selected.pt", "selected-hash")
        prior_dir = create_run(self.config, manifest("prior", checkpoint_ref=old))
        append_event(prior_dir, event("decision", "CYCLE_DECISION", WorkflowState.PIVOT,
                                     checkpoint_ref=selected), config=self.config)
        append_event(prior_dir, event("review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_PIVOT), config=self.config)
        write_integration_report(prior_dir, report(), config=self.config)
        append_event(prior_dir, event("stop", state=WorkflowState.STOPPED), config=self.config)
        child = replace(self.linked(checkpoint_ref=old), predecessor_run_id="prior")
        with self.assertRaises(RecordError):
            create_run(self.config, child, previous_run_dir=prior_dir)
        self.assertTrue(create_run(self.config, replace(child, checkpoint_ref=selected),
                                   previous_run_dir=prior_dir).exists())

    def test_valid_revise_inherits_no_approvals(self):
        self.append(event("approval", "APPROVAL", approval_stage="design", approved_plan_hash="plan-001"))
        self.finish()
        child = create_run(self.config, self.linked(), previous_run_dir=self.run_dir)
        self.assertEqual(read_events(child, config=self.config), ())

    def test_continuation_requires_explicit_predecessor_and_distinct_identity(self):
        self.finish()
        for item in (self.linked(), replace(self.linked(), predecessor_run_id=None),
                     replace(self.linked(), predecessor_decision_ref=None)):
            with self.assertRaises((RecordError, ValueError)):
                create_run(self.config, item)
        for field, value in (("run_id", "run-001"), ("cycle_id", "cycle-001"),
                             ("plan_hash", "plan-001"), ("predecessor_run_id", "other"),
                             ("predecessor_decision_ref", "missing")):
            with self.subTest(field=field), self.assertRaises(RecordError):
                create_run(self.config, replace(self.linked(), **{field: value}), previous_run_dir=self.run_dir)
        with self.assertRaises(PathSafetyError):
            create_run(self.config, self.linked(), previous_run_dir=self.root / "runs/old")

    def test_continuation_requires_latest_decision_matching_review_report_and_stop(self):
        self.append(event("decision", "CYCLE_DECISION", WorkflowState.REVISE))
        def refused():
            with self.assertRaises(RecordError):
                create_run(self.config, self.linked(), previous_run_dir=self.run_dir)
        refused()
        self.append(event("wrong-review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_PIVOT))
        self.append(event("early-stop", state=WorkflowState.STOPPED))
        refused()
        self.append(event("review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_REVISE))
        refused()
        write_integration_report(self.run_dir, report(GateStatus.FAIL), config=self.config)
        refused()
        self.append(event("stop", state=WorkflowState.STOPPED))
        self.append(event("latest", "CYCLE_DECISION", WorkflowState.REJECT))
        refused()

    def test_gate_review_after_stop_does_not_satisfy_continuation(self):
        self.append(event("decision", "CYCLE_DECISION", WorkflowState.REVISE))
        self.append(event("stop", state=WorkflowState.STOPPED))
        self.append(event("review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_REVISE))
        write_integration_report(self.run_dir, report(), config=self.config)
        with self.assertRaises(RecordError):
            create_run(self.config, self.linked(), previous_run_dir=self.run_dir)

    def test_pivot_uses_selected_decision_checkpoint_without_reading_data(self):
        checkpoint = CheckpointRef("artifacts/haic-research-v2/run-001/checkpoint.pt", "sha256:selected")
        self.finish(WorkflowState.PIVOT, checkpoint)
        for ref in (None, replace(checkpoint, sha256="other"), replace(checkpoint, path="artifacts/haic-research-v2/other.pt")):
            with self.assertRaises(RecordError):
                create_run(self.config, self.linked(checkpoint_ref=ref), previous_run_dir=self.run_dir)
        child = create_run(self.config, self.linked(checkpoint_ref=checkpoint), previous_run_dir=self.run_dir)
        self.assertEqual(read_manifest(child, config=self.config).checkpoint_ref, checkpoint)
        self.assertFalse((self.root / checkpoint.path).exists())

    def test_pivot_falls_back_to_manifest_checkpoint(self):
        checkpoint = CheckpointRef("artifacts/haic-research-v2/run-001/checkpoint.pt", "hash")
        prior = manifest("prior", checkpoint_ref=checkpoint)
        prior_dir = create_run(self.config, prior)
        append_event(prior_dir, event("decision", "CYCLE_DECISION", WorkflowState.PIVOT), config=self.config)
        append_event(prior_dir, event("review", "GATE_REVIEW", WorkflowState.GATE_REVIEW_PIVOT), config=self.config)
        write_integration_report(prior_dir, report(), config=self.config)
        append_event(prior_dir, event("stop", state=WorkflowState.STOPPED), config=self.config)
        child = replace(self.linked(checkpoint_ref=checkpoint), predecessor_run_id="prior")
        self.assertTrue(create_run(self.config, child, previous_run_dir=prior_dir).exists())

    def test_checkpoint_paths_must_stay_in_artifact_root(self):
        for path in ("artifacts/haic/file.pt", "submissions/file.pt", "../outside.pt",
                     "artifacts/haic-research-v2/../haic/file.pt", str(self.root / "outside.pt")):
            with self.subTest(path=path), self.assertRaises(PathSafetyError):
                create_run(self.config, manifest("bad", checkpoint_ref=CheckpointRef(path, "hash")))
        artifact = self.config.artifact_root
        artifact.mkdir(parents=True)
        alias = artifact / "alias"
        self.redirect(alias, self.root, directory=True)
        with self.assertRaises(PathSafetyError):
            self.append(event(checkpoint_ref=CheckpointRef("artifacts/haic-research-v2/alias/file.pt", "hash")))

    def test_tampered_event_history_is_refused_on_read_and_append(self):
        self.append(event())
        path = self.run_dir / "events.jsonl"
        text = path.read_text()
        for bad in (text + text, text + "not json\n", text.rstrip("\n")):
            path.write_text(bad)
            for call, args in ((read_events, (self.run_dir,)), (append_event, (self.run_dir, event("e2")))):
                with self.assertRaises(RecordError):
                    call(*args, config=self.config)
        path.write_text(text)

    def test_malformed_event_fields_and_encoding_fail_as_record_errors(self):
        self.append(event())
        path = self.run_dir / "events.jsonl"
        data = json.loads(path.read_text())
        for key, value in (("correction_ref", []), ("resource_usage", []),
                           ("timestamp", "not a timestamp"), ("event_id", 42)):
            corrupted = dict(data, **{key: value})
            path.write_text(json.dumps(corrupted) + "\n")
            with self.subTest(key=key), self.assertRaises(RecordError):
                read_events(self.run_dir, config=self.config)
        path.write_bytes(b"\xff\n")
        with self.assertRaises(RecordError):
            read_events(self.run_dir, config=self.config)


if __name__ == "__main__":
    unittest.main()
