"""Safety and receipt tests for the consumed-cell development comparator."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

def subject():
    return importlib.import_module("tools.compare_consumed_candidate")


class ConsumedCellTests(unittest.TestCase):
    def test_allowlist_has_only_screen_and_old_diagnostic_cells(self):
        module = subject()
        allowed = module.allowed_cells()
        self.assertEqual(len(allowed), 42)
        self.assertEqual(len(allowed & module.screen_cells()), 32)
        self.assertEqual(len(allowed & module.old_cells()), 10)
        self.assertIn((3, 2973604766), allowed)
        self.assertIn((2, 4089604952), allowed)
        protocol = json.loads(module.V6_PROTOCOL.read_text(encoding="utf-8"))
        sealed = set(protocol["partitions"]["confirmation"]["seeds"])
        sealed.update(protocol["partitions"]["blind"]["seeds"])
        self.assertFalse({seed for _, seed in allowed} & sealed)

    def test_selection_rejects_sealed_new_and_malformed_cells(self):
        module = subject()
        sealed = json.loads(module.V6_PROTOCOL.read_text(encoding="utf-8"))["partitions"]["blind"]["seeds"][0]
        for value in (f"1:{sealed}", "1:123456789", "5:2973604766", "1:-1", "x:42"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "consumed|cell"):
                module.select_cells([value], False)
        with self.assertRaisesRegex(ValueError, "cell"):
            module.select_cells([], False)

    def test_development_run_rejects_new_cell_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            module = subject()
            with patch.object(module, "_run_cold_episode") as worker:
                with self.assertRaisesRegex(ValueError, "consumed"):
                    module.run_cells(Path(directory), {}, {}, {}, [(1, 123456789)], workers=3)
                worker.assert_not_called()


class ReceiptTests(unittest.TestCase):
    def test_tampered_receipt_is_rejected(self):
        module = subject()
        identity = {"source_sha256": {"candidate": "a" * 64}}
        row = {"arm": "v7", "track_id": 1, "seed": 2973604766, "finished": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module.record_cell(root, identity, row)
            self.assertEqual(module.load_cell(root, identity, "v7", 1, 2973604766), row)
            path = module.cell_path(root, "v7", 1, 2973604766)
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["row"]["finished"] = False
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                module.load_cell(root, identity, "v7", 1, 2973604766)

    def test_three_arm_report_exposes_both_comparisons(self):
        module = subject()
        identity = {"source_sha256": {"candidate": "a" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, finished, progress, contacts, reason, lap in (
                ("baseline", True, 1.0, 0, None, 20000),
                ("v6", False, .5, 1, "off_track", None),
                ("v7", True, 1.0, 0, None, 20500),
            ):
                module.record_cell(root, identity, {
                    "arm": arm, "track_id": 1, "seed": 2973604766,
                    "finished": finished, "progress": progress, "lap_time_ms": lap,
                    "collision_count": contacts, "damage": .2 * contacts,
                    "retire_reason": reason, "initialization_ms": 100,
                    "reset_ms": 1, "action_latency_max_ms": 2,
                    "peak_worker_rss_mib": 300, "error": None,
                })
            report = module.report_cells(root, identity, [(1, 2973604766)])
            self.assertEqual(report["versus_baseline"]["decision"], "INCONCLUSIVE")
            self.assertEqual(report["versus_v6"]["decision"], "RETAIN")
            self.assertEqual(report["cells"], [[1, 2973604766]])


class InputTests(unittest.TestCase):
    def test_worker_limit_is_at_most_three(self):
        module = subject()
        for count in (0, 4, -1, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "workers"):
                module.validate_worker_count(count)
        self.assertEqual(module.validate_worker_count(3), 3)

    def test_candidate_must_be_snapshot_not_live_agent_or_protected_source(self):
        module = subject()
        with self.assertRaisesRegex(ValueError, "snapshot"):
            module.validate_candidate_path(module.ROOT / "agent.py")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            module.validate_candidate_path(module.V6_CONTROL)


if __name__ == "__main__":
    unittest.main()
