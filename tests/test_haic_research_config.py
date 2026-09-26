import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from haic_research.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def messages(issues):
    return [issue.message for issue in issues]


class ConfigTests(unittest.TestCase):
    def test_real_argument_types_path_roles_and_repetition_cannot_drift(self):
        source = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        edits = (
            ("evaluate_closed_loop", "policy-checkpoint", {"type": "int", "path_role": None}),
            ("train_policy", "output", {"path_role": "output_file"}),
            ("train_policy", "total-steps", {"repeated": True}),
            ("train_policy", "defer-tune", {"choices": [False]}),
            ("benchmark_corridor_diagnostic", "site-group", {"choices": ["unsupported"]}),
            ("train_policy", "total-steps", {"minimum": -1}),
            ("train_policy", "total-steps", {"type": {"unexpected": "object"}}),
        )
        for profile, name, changes in edits:
            with self.subTest(profile=profile, name=name):
                data = json.loads(json.dumps(source))
                data["commands"]["profiles"][profile]["arguments"][name].update(changes)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(root)

    def test_command_schema_and_positive_timeout_are_required(self):
        source = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        for field, value in (("timeout_seconds", 0), ("timeout_seconds", True),
                             ("arguments", {}), ("arguments", {"bogus": {"flag": "--bogus", "type": "object", "required": False}})):
            with self.subTest(field=field, value=value):
                data = json.loads(json.dumps(source))
                data["commands"]["profiles"]["train_policy"][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(root)

    def setUp(self):
        self.config = load_config(ROOT)

    def test_current_config_loads_with_resolved_paths(self):
        self.assertEqual(self.config.repo_root, ROOT)
        self.assertEqual(self.config.run_root, ROOT / "runs/haic-research-v2")
        self.assertEqual(self.config.artifact_root, ROOT / "artifacts/haic-research-v2")
        self.assertEqual(validate_config(self.config), [])

    def test_completion_rate_is_required_primary_metric(self):
        config = replace(self.config, primary_metric="lap_time")
        self.assertIn("metrics.primary must be completion_rate", messages(validate_config(config)))

    def test_new_roots_are_project_relative_and_distinct(self):
        config = replace(self.config, run_root=(ROOT / "../../outside").resolve())
        self.assertTrue(validate_config(config))

    def test_legacy_and_submission_roots_are_forbidden(self):
        for path in ("artifacts/haic", "artifacts/haic/child", "submissions", "submissions/child"):
            with self.subTest(path=path):
                config = replace(self.config, artifact_root=ROOT / path)
                self.assertTrue(validate_config(config))

    def test_loader_rejects_forbidden_output_root(self):
        data = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        data["paths"]["artifact_root"] = "artifacts/haic/child"
        for profile in data["commands"]["profiles"].values():
            profile["output_root"] = data["paths"]["artifact_root"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "legacy output root"):
                load_config(root)

    def test_loader_rejects_symlinked_legacy_output_root(self):
        data = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        data["paths"]["artifact_root"] = "artifacts/haic/child"
        for profile in data["commands"]["profiles"].values():
            profile["output_root"] = data["paths"]["artifact_root"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            redirected = root / "redirected"
            redirected.mkdir()
            try:
                (root / "artifacts/haic").symlink_to(redirected, target_is_directory=True)
            except OSError as exc:
                if os.name != "nt":
                    self.skipTest(f"directory symlink unavailable: {exc}")
                try:
                    junction = subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(root / "artifacts/haic"), str(redirected)],
                        capture_output=True, text=True, check=False,
                    )
                except OSError:
                    self.skipTest(f"directory symlink unavailable: {exc}")
                if junction.returncode != 0:
                    self.skipTest(f"directory symlink unavailable: {exc}")
            (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "legacy output root"):
                load_config(root)

    def test_roots_cannot_be_equal_or_nested(self):
        for path in (self.config.run_root, self.config.run_root / "child"):
            with self.subTest(path=path):
                self.assertTrue(validate_config(replace(self.config, artifact_root=path)))

    def test_metric_limits_pivot_and_split_validation(self):
        self.assertTrue(validate_config(replace(self.config, tie_breakers=("unknown",))))
        self.assertTrue(validate_config(replace(self.config, search_limits=(3, 8, 2))))
        self.assertTrue(validate_config(replace(self.config, pivot_after=2)))
        self.assertTrue(validate_config(replace(self.config, split_ids=("train",))))

    def test_malformed_json_missing_file_and_required_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ConfigError):
                load_config(root)
            path = root / "harness.config.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(root)
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(root)

    def test_rejects_invalid_schema_command_and_secret_key(self):
        source = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        for edit in (
            lambda data: data.update(schema_version=2),
            lambda data: data["commands"]["profiles"].update(arbitrary={"module": "os"}),
            lambda data: data["project"].update(api_token="value"),
        ):
            with self.subTest(edit=edit):
                data = json.loads(json.dumps(source))
                edit(data)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(root)

    def test_rejects_secondary_path_escape_and_malformed_metric(self):
        source = json.loads((ROOT / "harness.config.json").read_text(encoding="utf-8"))
        for edit in (
            lambda data: data["paths"].update(pdf_root="../../outside"),
            lambda data: data["official_sources"][2].update(path="../../outside"),
            lambda data: data["metrics"].update(tie_breakers=[{}]),
        ):
            with self.subTest(edit=edit):
                data = json.loads(json.dumps(source))
                edit(data)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "harness.config.json").write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_config(root)


if __name__ == "__main__":
    unittest.main()
