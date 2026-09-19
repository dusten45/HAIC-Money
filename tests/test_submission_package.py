import tempfile
import unittest
import hashlib
import json
import zipfile
from pathlib import Path

import torch

from agent import Baseline1Actor
from package_submission import (
    build_submission,
    create_submission_record,
    validate_agent_source,
)


ROOT = Path(__file__).resolve().parents[1]


class TestSubmissionPackage(unittest.TestCase):
    def test_builds_root_only_submission_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / "model.pt"
            archive_path = directory_path / "submission.zip"
            torch.save(Baseline1Actor().state_dict(), model_path)

            build_submission(ROOT / "agent.py", model_path, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), {"agent.py", "model.pt"})

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


if __name__ == "__main__":
    unittest.main()
