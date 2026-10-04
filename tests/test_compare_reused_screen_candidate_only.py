"""Guard candidate-only development reuse of the frozen v6 screen."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def subject():
    return importlib.import_module("tools.compare_reused_screen_candidate_only")


@contextmanager
def archived_v6_identity(module):
    """Keep archived V6 identity valid while checking current runtime files stay fixed."""
    _, _, freeze, _ = module.dev._historical_records()
    runtime = {
        key: {name: module.fresh.digest(module.fresh.ROOT / name)
              for name in freeze[key]}
        for key in ("helper_sha256", "environment_sha256")
    }
    check_frozen_inputs = module.fresh.check_frozen_inputs

    def check_current_inputs(identity, paths):
        # V6's frozen helpers/environment predate the current checkout. Keep
        # source, model, protocol, harness, and runtime stability checks real.
        adjusted = {**identity, **runtime}
        check_frozen_inputs(adjusted, paths)

    with patch.object(module.fresh, "build_identity", return_value=freeze), \
         patch.object(module.fresh, "check_frozen_inputs", side_effect=check_current_inputs):
        yield


class ScreenSelectionTests(unittest.TestCase):
    def test_script_help_runs_from_repository_root(self):
        module = subject()
        completed = subprocess.run(
            [sys.executable, str(Path(module.__file__).resolve()), "--help"],
            cwd=module.dev.ROOT, text=True, capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--all-screen", completed.stdout)

    def test_only_completed_v6_screen_cells_are_selectable(self):
        module = subject()
        self.assertEqual(len(module.select_cells([], True)), 32)
        self.assertEqual(module.select_cells(["3:2973604766"], False), [(3, 2973604766)])
        protocol = json.loads(module.dev.V6_PROTOCOL.read_text(encoding="utf-8"))
        sealed = protocol["partitions"]["confirmation"]["seeds"][0]
        for value in ("3:3857792434", f"1:{sealed}", "1:123456789", "5:2973604766"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "screen"):
                module.select_cells([value], False)

    def test_direct_run_refuses_non_screen_cell_before_worker(self):
        module = subject()
        with patch.object(module, "_run_cold_episode") as worker:
            with self.assertRaisesRegex(ValueError, "screen"):
                module.run_cells(Path("unused"), {}, {}, {}, [(3, 3857792434)])
            worker.assert_not_called()


class OriginalReceiptTests(unittest.TestCase):
    def test_original_pair_matches_frozen_screen_summary(self):
        module = subject()
        _, _, freeze, summary = module.dev._historical_records()
        pair = module.load_original_pair(3, 2973604766, freeze, summary)
        self.assertEqual(pair["baseline"]["action_trace_sha256"],
                         "53598f40badf0ea24025de374493333c651b224a8e6e2c2ba8ba04a9accff75a")
        self.assertEqual(pair["v6"]["action_trace_sha256"],
                         "c23a251b98d2b502068ad875f7278d80128278f99c57c625e3c94006454d3455")
        self.assertEqual(pair["baseline"]["progress"], 0.9421965317919075)
        self.assertEqual(pair["v6"]["progress"], 0.43641618497109824)

    def test_recomputed_or_stale_original_receipt_is_rejected(self):
        module = subject()
        _, _, freeze, summary = module.dev._historical_records()
        track, seed = 3, 2973604766
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            for arm in ("control", "candidate"):
                source = module.fresh.cell_path(module.V6_RUN, "screen", arm, track, seed, 0)
                target = module.fresh.cell_path(copied, "screen", arm, track, seed, 0)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            target = module.fresh.cell_path(copied, "screen", "candidate", track, seed, 0)
            envelope = json.loads(target.read_text(encoding="utf-8"))
            envelope["row"]["progress"] = .5
            target.write_text(json.dumps(envelope), encoding="utf-8")
            with patch.object(module, "V6_RUN", copied):
                with self.assertRaisesRegex(ValueError, "digest"):
                    module.load_original_pair(track, seed, freeze, summary)
            envelope["digest"] = module.hashlib.sha256(module.fresh._canonical(
                {"identity": envelope["identity"], "row": envelope["row"]})).hexdigest()
            target.write_text(json.dumps(envelope), encoding="utf-8")
            with patch.object(module, "V6_RUN", copied):
                with self.assertRaisesRegex(ValueError, "summary"):
                    module.load_original_pair(track, seed, freeze, summary)

    def test_original_receipt_identity_must_match_frozen_run(self):
        module = subject()
        _, _, freeze, summary = module.dev._historical_records()
        track, seed = 3, 2973604766
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory)
            for arm in ("control", "candidate"):
                source = module.fresh.cell_path(module.V6_RUN, "screen", arm, track, seed, 0)
                target = module.fresh.cell_path(copied, "screen", arm, track, seed, 0)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            target = module.fresh.cell_path(copied, "screen", "candidate", track, seed, 0)
            envelope = json.loads(target.read_text(encoding="utf-8"))
            envelope["identity"]["model_sha256"] = "0" * 64
            envelope["digest"] = module.hashlib.sha256(module.fresh._canonical(
                {"identity": envelope["identity"], "row": envelope["row"]})).hexdigest()
            target.write_text(json.dumps(envelope), encoding="utf-8")
            with patch.object(module, "V6_RUN", copied):
                with self.assertRaisesRegex(ValueError, "identity"):
                    module.load_original_pair(track, seed, freeze, summary)

    def test_source_bound_manifest_rejects_rehashed_action_trace(self):
        module = subject()
        track, seed = 3, 2973604766
        snapshot = module.ARTIFACTS / "v7-development/snapshots/_TemporalEncounterController.py"
        paths = {"baseline": module.dev.V6_CONTROL, "v6": module.dev.V6_CANDIDATE,
                 "v7": snapshot, "model": module.dev.MODEL}
        _, _, freeze, summary = module.dev._historical_records()
        with archived_v6_identity(module), tempfile.TemporaryDirectory() as directory:
            identity = module.build_identity(snapshot, "_TemporalEncounterController", [(track, seed)])
            self.assertEqual(len(identity["original_receipt_sha256"]), 2)
            copied = Path(directory)
            for arm in ("control", "candidate"):
                source = module.fresh.cell_path(module.V6_RUN, "screen", arm, track, seed, 0)
                target = module.fresh.cell_path(copied, "screen", arm, track, seed, 0)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            target = module.fresh.cell_path(copied, "screen", "candidate", track, seed, 0)
            envelope = json.loads(target.read_text(encoding="utf-8"))
            with patch.object(module, "V6_RUN", copied):
                module.check_inputs(identity, paths, freeze, summary)
            envelope["row"]["action_trace_sha256"] = "0" * 64
            envelope["digest"] = module.hashlib.sha256(module.fresh._canonical(
                {"identity": envelope["identity"], "row": envelope["row"]})).hexdigest()
            target.write_text(json.dumps(envelope), encoding="utf-8")
            with patch.object(module, "V6_RUN", copied):
                with self.assertRaisesRegex(ValueError, "receipt hash"):
                    module.check_inputs(identity, paths, freeze, summary)


class CandidateOnlyComparisonTests(unittest.TestCase):
    def test_report_equals_existing_three_arm_comparator_without_copying_references(self):
        module = subject()
        _, _, freeze, summary = module.dev._historical_records()
        cells = sorted(module.dev.screen_cells())
        identity = {"cells": [[track, seed] for track, seed in cells],
                    "v7_source_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            fast_root, old_root = base / "fast", base / "old"
            for track, seed in cells:
                pair = module.load_original_pair(track, seed, freeze, summary)
                candidate = {**pair["v6"], "arm": "v7"}
                candidate.pop("partition")
                candidate.pop("repeat")
                module.dev.record_cell(fast_root, identity, candidate)
                for arm, row in (("baseline", pair["baseline"]),
                                 ("v6", pair["v6"]), ("v7", candidate)):
                    old_row = {**row, "arm": arm}
                    old_row.pop("partition", None)
                    old_row.pop("repeat", None)
                    module.dev.record_cell(old_root, identity, old_row)
            fast = module.report_cells(fast_root, identity, cells, freeze, summary)
            old = module.dev.report_cells(old_root, identity, cells)
            self.assertEqual(fast["versus_baseline"], old["versus_baseline"])
            self.assertEqual(fast["versus_v6"], old["versus_v6"])
            self.assertEqual(fast["versus_baseline"]["decision"], "REJECT")
            self.assertEqual(fast["versus_v6"]["decision"], "INCONCLUSIVE")
            self.assertFalse((fast_root / "cells" / "baseline").exists())
            self.assertFalse((fast_root / "cells" / "v6").exists())

    def test_run_schedules_candidate_worker_only(self):
        module = subject()
        snapshot = module.ARTIFACTS / "v7-development/snapshots/_TemporalEncounterController.py"
        cells = [(3, 2973604766)]
        paths = {"baseline": module.dev.V6_CONTROL, "v6": module.dev.V6_CANDIDATE,
                 "v7": snapshot, "model": module.dev.MODEL}
        classes = {**module.dev.CLASSES, "v7": "_TemporalEncounterController"}
        measured = {"finished": False, "progress": .5, "lap_time_ms": None,
                    "collision_count": 0, "damage": .0, "retire_reason": "off_track",
                    "initialization_ms": 1, "reset_ms": 1, "action_latency_max_ms": 1,
                    "peak_worker_rss_mib": 100, "error": None}
        module.OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        with archived_v6_identity(module), tempfile.TemporaryDirectory(dir=module.OUTPUT_PARENT) as directory:
            identity = module.build_identity(snapshot, "_TemporalEncounterController", cells)
            with patch.object(module, "_run_cold_episode", return_value=measured) as worker:
                module.run_cells(Path(directory), identity, paths, classes, cells)
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(worker.call_args.args[-3:], ("v7", 3, 2973604766))
                self.assertTrue(module.dev.cell_path(Path(directory), "v7", 3, 2973604766).is_file())
                for arm in ("baseline", "v6"):
                    self.assertFalse(module.dev.cell_path(Path(directory), arm, 3, 2973604766).exists())


if __name__ == "__main__":
    unittest.main()
