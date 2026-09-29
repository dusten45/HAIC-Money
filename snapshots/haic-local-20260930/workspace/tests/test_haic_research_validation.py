import copy
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from haic_research.cli import main
from haic_research.validation import REQUIRED_PATHS, validate_project


ROOT = Path(__file__).resolve().parents[1]


class ProjectValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in REQUIRED_PATHS:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        self.data = json.loads((self.root / "harness.config.json").read_text(encoding="utf-8"))

    def write_config(self, data):
        (self.root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")

    def test_complete_fixture_passes(self):
        self.assertEqual(validate_project(self.root), [])

    def test_missing_required_handoff_template_is_reported(self):
        (self.root / "docs/handoffs/REPORT_TEMPLATE.md").unlink()
        self.assertIn("docs/handoffs/REPORT_TEMPLATE.md", [issue.path for issue in validate_project(self.root)])

    def test_plan_only_and_completion_priority_are_required(self):
        self.data["commands"]["default_mode"] = "execute"
        self.data["metrics"]["primary"] = "lap_time"
        self.write_config(self.data)
        self.assertGreaterEqual(len(validate_project(self.root)), 2)

    def test_exact_required_policy_values_reject_drift_or_missing_fields(self):
        mutations = (
            ("workflow", "states", ["STOPPED"]),
            ("workflow", "separate_approval_stages", ["execution"]),
            ("workflow", "gates", ["rule_compliance"]),
            ("workflow", "default_mode", "unknown"),
            ("workflow", "pivot_after_valid_non_improving_cycles", 2),
            ("search", "max_candidates_per_batch", 9),
            ("search", "threshold_sweep_requires_mechanism_activation", False),
            ("metrics", "tie_breakers", list(reversed(self.data["metrics"]["tie_breakers"]))),
            ("metrics", "official_score_is_separate", False),
            ("splits", "ids", ["train"]),
            ("splits", "protected_ids", ["blind"]),
            ("splits", "blind_tuning_allowed", True),
            ("splits", "consumed_confirmation_reusable_as_fresh", True),
            ("commands", "allow_shell", True),
            ("commands", "profiles", {}),
            ("project", "scope", "unknown"),
        )
        for section, field, value in mutations:
            for remove in (False, True):
                with self.subTest(section=section, field=field, remove=remove):
                    data = copy.deepcopy(self.data)
                    if remove:
                        del data[section][field]
                    else:
                        data[section][field] = value
                    self.write_config(data)
                    self.assertTrue(validate_project(self.root))

    def test_sources_require_precedence_urls_pin_and_mirror_paths(self):
        for index, field, value in ((0, "url", "https://example.org"), (1, "precedence", 1),
                                    (2, "source_commit", "unknown"), (2, "path", "other.md"),
                                    (2, "license_path", "other-license")):
            with self.subTest(index=index, field=field):
                data = copy.deepcopy(self.data)
                data["official_sources"][index][field] = value
                self.write_config(data)
                self.assertTrue(validate_project(self.root))
        self.write_config(self.data)
        for name in ("docs/sources/official-participants/README.md", "docs/sources/official-participants/LICENSE"):
            target = self.root / name
            content = target.read_bytes()
            target.unlink()
            self.assertIn(name, [issue.path for issue in validate_project(self.root)])
            target.write_bytes(content)

    def test_governing_documents_require_sources_and_roles(self):
        for name in ("AGENTS.md", "PROJECT_INFO.md", "RESTRICTIONS.md", "docs/sources/INDEX.md"):
            with self.subTest(name=name):
                target = self.root / name
                original = target.read_bytes()
                target.write_text("# Unknown\n", encoding="utf-8")
                self.assertIn(name, [issue.path for issue in validate_project(self.root)])
                target.write_bytes(original)

    def test_typed_profiles_path_rules_and_secrets_reuse_config_validation(self):
        for change in ("type", "path", "secret"):
            with self.subTest(change=change):
                data = copy.deepcopy(self.data)
                if change == "type":
                    data["commands"]["profiles"]["train_policy"]["arguments"]["output"]["type"] = "str"
                elif change == "path":
                    data["paths"]["artifact_root"] = "../outside"
                else:
                    data["project"]["api_key"] = "do-not-store"
                self.write_config(data)
                self.assertTrue(validate_project(self.root))

    def test_missing_malformed_config_is_a_validation_issue(self):
        for value in ("{broken", "[]", '{"schema_version": 1}'):
            (self.root / "harness.config.json").write_text(value, encoding="utf-8")
            self.assertIn("harness.config.json", [issue.path for issue in validate_project(self.root)])

    def test_no_historical_reads_or_directory_scans(self):
        for name in ("runs/private.json", "artifacts/haic/private.json", "submissions/private.json"):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("historical sentinel", encoding="utf-8")
        original_open = io.open
        reads = []

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)):
                relative = Path(path).relative_to(self.root).as_posix()
                self.assertIn(relative, REQUIRED_PATHS)
                reads.append(relative)
            return original_open(path, *args, **kwargs)

        with patch("io.open", side_effect=guarded_open), \
             patch("builtins.open", side_effect=guarded_open), \
             patch.object(Path, "glob", side_effect=AssertionError("no glob")), \
             patch.object(Path, "rglob", side_effect=AssertionError("no rglob")), \
             patch.object(Path, "iterdir", side_effect=AssertionError("no directory scan")), \
             patch("os.scandir", side_effect=AssertionError("no directory scan")):
            self.assertEqual(validate_project(self.root), [])
        self.assertIn("harness.config.json", reads)

    def test_cli_validate_supports_root_after_subcommand_and_no_runner(self):
        out = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError("no profiles")):
            self.assertEqual(main(["validate", "--root", str(self.root)], stdout=out), 0)
        self.assertTrue(json.loads(out.getvalue())["valid"])
        (self.root / "docs/handoffs/REPORT_TEMPLATE.md").unlink()
        out = io.StringIO()
        self.assertEqual(main(["validate", "--root", str(self.root)], stdout=out), 1)
        self.assertFalse(json.loads(out.getvalue())["valid"])

    @unittest.skipUnless(os.name == "nt", "PowerShell wrapper is Windows-specific")
    def test_wrapper_resolves_root_from_another_directory_and_propagates_exit(self):
        wrapper = self.root / "scripts/harness/validate.ps1"
        # A temporary python shim records the actual root/module arguments and exits 7.
        shim = self.root / "shim"
        shim.mkdir()
        trace = self.root / "arguments.txt"
        (shim / "python.cmd").write_text(f'@echo off\n>"{trace}" echo %*\nexit /b 7\n', encoding="utf-8")
        env = dict(os.environ, PATH=str(shim) + os.pathsep + os.environ["PATH"])
        result = subprocess.run(["pwsh", "-NoProfile", "-File", str(wrapper)], cwd=shim,
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        args = trace.read_text(encoding="utf-8")
        self.assertIn("-m haic_research.cli validate --root", args)
        self.assertIn(str(self.root), args)


if __name__ == "__main__":
    unittest.main()
