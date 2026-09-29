import tempfile
import unittest
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

from action_smoothing import canonical_action_smoothing, normalize_action_smoothing
from agent import Baseline1Actor, MODEL_FILENAME
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

    def test_packaged_bare_model_uses_fast_corner_carry_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_path = directory_path / MODEL_FILENAME
            archive_path = directory_path / "submission.zip"
            extracted = directory_path / "extracted"
            torch.save(Baseline1Actor().state_dict(), model_path)
            build_submission(ROOT / "agent.py", model_path, archive_path)
            with zipfile.ZipFile(archive_path) as package:
                package.extractall(extracted)

            code = """
import numpy as np
from agent import Agent, _FastCornerCarryController
frame = np.full((84, 84), 0.1, dtype=np.float32)
frame[20:63, 31:53] = 0.4
frame[77:83, 10:13] = 0.27 / 18.0
observation = np.tile(frame[None, :, :], (4, 1, 1))
agent = Agent()
assert type(agent._forward_controller) is _FastCornerCarryController
np.testing.assert_array_equal(agent.act(observation), np.array([0.0, 0.18, 0.0], dtype=np.float32))

frame[77:83, 10:13] = (0.27 + 0.085 * 45.0) / 18.0
speed_45 = np.tile(frame[None, :, :], (4, 1, 1))
agent.reset(None)
sustain_action = agent.act(speed_45)
np.testing.assert_allclose(
    sustain_action,
    np.array([0.0, 0.18, 0.0], dtype=np.float32),
    rtol=0.0,
    atol=1e-7,
)

obstacle = frame.copy()
obstacle[32:36, 40:43] = 0.68
obstacle[77:83, 10:13] = (0.27 + 0.085 * 60.0) / 18.0
obstacle_observation = np.tile(obstacle[None, :, :], (4, 1, 1))
candidate = _FastCornerCarryController()
detected = candidate.act(obstacle_observation)
assert detected[1] == 0.0
assert detected[2] > 0.0
for _ in range(candidate.OBSTACLE_MISS_LIMIT):
    latched = candidate.act(speed_45)
    assert latched[1] <= 0.11
    assert latched[1] * latched[2] == 0.0
released_candidate = candidate.act(speed_45)
assert released_candidate[1] > 0.11

compound = np.full((84, 84), 0.1, dtype=np.float32)
for row in range(20, 63):
    center = 42.0 + 0.75 * (54 - row)
    left = int(round(center - 11))
    right = int(round(center + 11))
    compound[row, left:right] = 0.4
compound[50:54, 35:38] = 0.68
compound[77:83, 10:13] = (0.27 + 0.085 * 48.0) / 18.0
agent.reset(None)
compound_action = agent.act(np.tile(compound[None, :, :], (4, 1, 1)))
assert compound_action[1] == 0.0
np.testing.assert_array_equal(
    compound_action[2],
    np.nextafter(np.float32(0.28), np.float32(0.0)),
)
"""
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=extracted,
                check=True,
                capture_output=True,
                text=True,
            )

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
