import tempfile
import unittest
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

from action_smoothing import canonical_action_smoothing, normalize_action_smoothing
from agent import Baseline1Actor, MODEL_FILENAME
from export_policy import export_payload
from haic.algorithms.rlpd.agent import PixelRLPDAgent
from package_submission import (
    RLPD_ACTOR_FORMAT,
    build_submission,
    create_submission_record,
    resolve_python_executable,
    smoke_submission,
    validate_agent_source,
    validate_submission_archive,
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
            model_path = directory_path / MODEL_FILENAME
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
                        MODEL_FILENAME,
                    },
                )

    def test_smoke_test_resets_and_runs_two_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / MODEL_FILENAME
            archive_path = directory_path / "submission.zip"
            torch.save(Baseline1Actor().state_dict(), model_path)

            build_submission(ROOT / "agent.py", model_path, archive_path)

            result = smoke_submission(archive_path, sys.executable)

        self.assertEqual(result["shape"], [3])
        self.assertTrue(result["finite"])
        self.assertTrue(result["reset_matches_first"])
        self.assertTrue(result["unreset_matches_first"])

    def test_explicit_drq_actor_is_packaged_under_declared_model_filename(self):
        from drq_v2 import DrQv2Agent, DrQv2Config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            actor = DrQv2Agent(DrQv2Config(replay_capacity=8)).export_actor(path / "actor.pt")
            archive = build_submission(ROOT / "agent.py", actor, path / "submission.zip")
            with zipfile.ZipFile(archive) as package:
                self.assertIn("model.pt", package.namelist())
                self.assertNotIn("actor.pt", package.namelist())
                self.assertEqual(package.read("model.pt"), actor.read_bytes())
                self.assertNotIn("drq_v2.py", package.namelist())
                self.assertNotIn("common_adapter.py", package.namelist())
            record, manifest = create_submission_record(
                ROOT / "agent.py", actor, None, path / "records", "drq-explicit-actor",
                smoke_test=False, python_executable=sys.executable,
            )
            self.assertEqual(manifest["model"]["archive_path"], "model.pt")
            self.assertEqual(manifest["model"]["sha256"], hashlib.sha256(actor.read_bytes()).hexdigest())
            with zipfile.ZipFile(record / "submission.zip") as package:
                self.assertEqual(package.read("model.pt"), actor.read_bytes())

    def test_explicit_dreamerv3_actor_is_packaged_under_declared_model_filename(self):
        from dreamer_v3 import DreamerV3Agent, DreamerV3Config

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            agent = DreamerV3Agent(DreamerV3Config(device="cpu", embed_dim=64, hidden_dim=64, num_categoricals=8, num_classes=8))
            actor = agent.export_actor(path / "actor.pt")
            archive = build_submission(ROOT / "agent.py", actor, path / "submission.zip")
            with zipfile.ZipFile(archive) as package:
                self.assertIn("model.pt", package.namelist())
                self.assertNotIn("actor.pt", package.namelist())
                self.assertEqual(package.read("model.pt"), actor.read_bytes())
                self.assertNotIn("dreamer_v3.py", package.namelist())
                self.assertNotIn("common_adapter.py", package.namelist())

    def test_rlpd_actor_package_is_root_only_smokes_tag_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            actor = PixelRLPDAgent(seed=23)
            actor_path = actor.export_actor(
                path / "actor.pt",
                source_sha256="a" * 64,
                protocol_sha256="b" * 64,
                training_seed=23,
                environment_contract={
                    "observation_fingerprint": actor.observation_spec.fingerprint,
                    "action_fingerprint": actor.action_adapter.spec.fingerprint,
                },
            )
            payload = torch.load(actor_path, map_location="cpu", weights_only=True)

            self.assertEqual(payload["format"], RLPD_ACTOR_FORMAT)
            self.assertEqual(
                set(payload),
                {
                    "format", "config", "observation_spec", "action_spec",
                    "actor_state_dict", "training_seed", "environment_steps",
                    "gradient_steps", "source_sha256", "protocol_sha256",
                    "environment_contract",
                },
            )
            self.assertEqual(set(payload["actor_state_dict"]), set(actor.actor.state_dict()))
            self.assertTrue(
                all(value.device.type == "cpu" for value in payload["actor_state_dict"].values())
            )

            record, manifest = create_submission_record(
                ROOT / "agent.py",
                actor_path,
                None,
                path / "records",
                "rlpd actor",
                smoke_test=True,
                python_executable=sys.executable,
            )
            archive_path = record / "submission.zip"
            self.assertEqual(validate_agent_source(ROOT / "agent.py"), MODEL_FILENAME)
            validate_submission_archive(archive_path, MODEL_FILENAME, expected_modules=())
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(archive.namelist(), ["agent.py", "model.pt"])
                self.assertEqual(archive.read("model.pt"), actor_path.read_bytes())

            self.assertEqual(manifest["dependencies"], [])
            self.assertEqual(manifest["model"]["format"], RLPD_ACTOR_FORMAT)
            self.assertEqual(manifest["smoke_test"]["actor_format"], RLPD_ACTOR_FORMAT)

            with patch("package_submission.MAX_ARCHIVE_BYTES", archive_path.stat().st_size - 1):
                with self.assertRaisesRegex(ValueError, "exceeds 500 MiB"):
                    validate_submission_archive(archive_path, MODEL_FILENAME, expected_modules=())

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
            model_path = directory_path / MODEL_FILENAME
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
            model_path = directory_path / MODEL_FILENAME
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
            model_path = directory_path / MODEL_FILENAME
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
            model_path = directory_path / MODEL_FILENAME
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
