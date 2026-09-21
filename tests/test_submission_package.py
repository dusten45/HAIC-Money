import tempfile
import unittest
import hashlib
import json
import zipfile
from pathlib import Path

import torch

from action_smoothing import canonical_action_smoothing, normalize_action_smoothing
from agent import Baseline1Actor
from export_policy import export_payload
from package_submission import (
    build_submission,
    create_submission_record,
    resolve_python_executable,
    smoke_submission,
    validate_agent_source,
)


ROOT = Path(__file__).resolve().parents[1]


class TestSubmissionPackage(unittest.TestCase):
    def test_resolves_relative_python_path_before_smoke_test_changes_directory(self):
        self.assertEqual(
            resolve_python_executable(".venv/bin/python"),
            str(Path.cwd() / ".venv/bin/python"),
        )
        self.assertEqual(resolve_python_executable("python3.11"), "python3.11")

    def test_builds_root_only_submission_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            archive_path = directory_path / "submission.zip"
            torch.save(
                export_payload(
                    Baseline1Actor(),
                    canonical_action_smoothing("alpha", [0.35, 1.0, 1.0]),
                ),
                model_path,
            )

            build_submission(ROOT / "agent.py", model_path, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        "agent.py",
                        "action_smoothing.py",
                        "action_representation.py",
                        "model.pt",
                    },
                )

    def test_smoke_test_resets_and_runs_two_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            archive_path = directory_path / "submission.zip"
            torch.save(Baseline1Actor().state_dict(), model_path)

            build_submission(ROOT / "agent.py", model_path, archive_path)

            result = smoke_submission(archive_path, str(ROOT / ".venv/bin/python"))

        self.assertEqual(result["shape"], [3])
        self.assertTrue(result["finite"])
        self.assertTrue(result["reset_matches_first"])
        self.assertTrue(result["unreset_matches_first"])

    def test_rejects_banned_submission_import(self):
        with tempfile.TemporaryDirectory() as directory:
            agent_path = Path(directory) / "agent.py"
            agent_path.write_text(
                'import os\nMODEL_FILENAME = "model.pt"\nclass Agent:\n    def act(self, observation):\n        return observation\n'
            )

            with self.assertRaisesRegex(ValueError, "banned import"):
                validate_agent_source(agent_path)

    def test_records_submission_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            source_model_path = directory_path / "source.zip"
            torch.save(Baseline1Actor().state_dict(), model_path)
            source_model_path.write_bytes(b"source-model")

            record_dir, manifest = create_submission_record(
                ROOT / "agent.py",
                model_path,
                source_model_path,
                directory_path / "submissions",
                "baseline 1",
                False,
                "python",
            )

            self.assertTrue((record_dir / "submission.zip").is_file())
            self.assertTrue((record_dir / "manifest.json").is_file())
            self.assertEqual(
                manifest["source_model"]["sha256"],
                hashlib.sha256(b"source-model").hexdigest(),
            )
            self.assertEqual(
                json.loads((record_dir / "manifest.json").read_text())["submission_id"],
                manifest["submission_id"],
            )
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["source_model"]["vecnormalize"], None)
        self.assertEqual(
            [item["path"] for item in manifest["dependencies"]],
            ["action_smoothing.py", "action_representation.py"],
        )

    def test_records_periodic_checkpoint_normalizer_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            checkpoint_dir = directory_path / "checkpoints"
            checkpoint_dir.mkdir()
            source_model_path = checkpoint_dir / "ppo_baseline_10_steps.zip"
            normalizer_path = checkpoint_dir / "ppo_baseline_vecnormalize_10_steps.pkl"
            run_config_path = directory_path / "config.json"
            metrics_path = directory_path / "best_model_metrics.json"
            torch.save(Baseline1Actor().state_dict(), model_path)
            source_model_path.write_bytes(b"source-model")
            normalizer_path.write_bytes(b"normalizer")
            run_config_path.write_text("{}\n")
            metrics_path.write_text("{}\n")

            _record_dir, manifest = create_submission_record(
                ROOT / "agent.py", model_path, source_model_path,
                directory_path / "submissions", "periodic", False, "python",
            )

        self.assertEqual(
            manifest["source_model"]["vecnormalize"]["sha256"],
            hashlib.sha256(b"normalizer").hexdigest(),
        )
        self.assertEqual(
            manifest["source_model"]["run_config"]["path"], str(run_config_path)
        )
        self.assertEqual(
            manifest["source_model"]["evaluation_metrics"]["path"], str(metrics_path),
        )

    def test_cleans_partial_record_when_source_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            torch.save(Baseline1Actor().state_dict(), model_path)
            submissions_dir = directory_path / "submissions"

            with self.assertRaises(FileNotFoundError):
                create_submission_record(
                    ROOT / "agent.py",
                    model_path,
                    directory_path / "missing.zip",
                    submissions_dir,
                    "broken",
                    False,
                    "python",
                )

            self.assertFalse(submissions_dir.exists())

    def test_rejects_model_source_smoothing_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            source_model_path = directory_path / "source.zip"
            torch.save(
                {
                    "state_dict": Baseline1Actor().state_dict(),
                    "action_smoothing": normalize_action_smoothing(),
                },
                model_path,
            )
            source_model_path.write_bytes(b"source-model")
            (directory_path / "config.json").write_text(
                json.dumps({"config": {"action_smoothing": canonical_action_smoothing(
                    "alpha", [0.35, 1.0, 1.0]
                )}})
            )

            with self.assertRaisesRegex(ValueError, "do not match"):
                create_submission_record(
                    ROOT / "agent.py",
                    model_path,
                    source_model_path,
                    directory_path / "submissions",
                    "mismatch",
                    False,
                    "python",
                )

            self.assertFalse((directory_path / "submissions").exists())


if __name__ == "__main__":
    unittest.main()
