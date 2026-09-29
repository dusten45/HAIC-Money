"""Synthetic-only tests: no real dataset/checkpoint or environment interaction."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from scripts import diagnose_rlpd_recovery_actions as diagnosis
from haic.algorithms.rlpd.model import PixelEncoder
from haic.algorithms.rlpd.recovery import frame_stack


def tiny_rows(n=65):
    observations = np.zeros((n, 4, 84, 84), dtype=np.uint8)
    observations[:, :, 0, 0] = np.arange(n, dtype=np.uint8)[:, None]
    target = np.tile([.5, .2, -.6], (n, 1)).astype(np.float32)
    official = target.copy()
    official[:, 1:] = (official[:, 1:] + 1) / 2
    return {"observations": observations, "target_native": target, "target_official": official,
            "episode_id": np.zeros(n, dtype=np.int64), "step": np.arange(n),
            "geometry_seed": np.full(n, 4272000003),
            "stratum": np.where(np.arange(n) % 2, "finish-control", "failure"),
            "role": np.where(np.arange(n) % 3, "actor", "oracle")}


class TinyActor:
    def __init__(self):
        self.calls = []

    def sample(self, obs, *, deterministic):
        assert deterministic and not torch.is_grad_enabled()
        assert obs.dtype == torch.uint8
        self.calls.append(len(obs))
        return torch.tensor([-.5, -.2, .6]).repeat(len(obs), 1), None, None


class TinyCritic:
    def __init__(self, shift=0):
        self.observations = []
        self.shift = shift

    def __call__(self, obs, action):
        self.observations.append(obs.clone())
        return action[:, :1] + torch.arange(10)[None, :] + self.shift


class DiagnosisTests(unittest.TestCase):
    def test_batch_exact_images_actions_q_and_splits(self):
        rows = tiny_rows()
        actor, critic = TinyActor(), TinyCritic()
        result = diagnosis.predict(rows, actor, critic)
        self.assertEqual(actor.calls, [64, 1])
        for i in (0, 2):
            torch.testing.assert_close(critic.observations[i], critic.observations[i + 1])
        np.testing.assert_allclose(result["actor_official"][0], [-.5, .4, .8])
        np.testing.assert_allclose(result["q_gap_mean"], 1)
        np.testing.assert_allclose(result["q_gap_min"], 1)
        np.testing.assert_allclose(result["q_actor_variance"], 8.25)
        report = diagnosis.summarize(rows, result)
        group = report["all/all"]
        self.assertEqual(group["rows"], 65)
        self.assertAlmostEqual(group["action_official"]["steering_mae"], 1)
        self.assertAlmostEqual(group["action_official"]["gas_signed_error_actor_minus_target"], -.2)
        self.assertAlmostEqual(group["action_official"]["brake_signed_error_actor_minus_target"], .6)
        self.assertAlmostEqual(group["action_native"]["joint_mse"], (1 + .16 + 1.44) / 3)
        self.assertEqual(group["action_native"]["steering_opposition_fraction"], 1)
        self.assertEqual(group["q"]["target_preferred_fraction_mean"], 1)
        self.assertEqual(report["failure/all"]["rows"] + report["finish-control/all"]["rows"], 65)
        self.assertEqual(report["all/oracle"]["rows"] + report["all/actor"]["rows"], 65)
        shifted = diagnosis.predict(rows, TinyActor(), TinyCritic(1000))
        np.testing.assert_allclose(shifted["q_gap_mean"], result["q_gap_mean"])

    def test_zero_strong_steering_and_empty_groups(self):
        rows = tiny_rows(1)
        rows["target_native"][:, 0] = 0
        rows["target_official"][:, 0] = 0
        summary = diagnosis.summarize(rows, diagnosis.predict(rows, TinyActor(), TinyCritic()))
        self.assertIsNone(summary["all/all"]["action_native"]["steering_opposition_fraction"])
        self.assertEqual(summary["finish-control/all"], {"rows": 0})

    def test_frame_stack_and_encoder_normalization(self):
        frames = np.broadcast_to(np.arange(6, dtype=np.uint8)[:, None, None], (6, 84, 84)).copy()
        np.testing.assert_array_equal(frame_stack(frames, 1)[:, 0, 0], [0, 0, 0, 1])
        np.testing.assert_array_equal(frame_stack(frames, 5)[:, 0, 0], [2, 3, 4, 5])
        normalized = PixelEncoder.normalize(torch.from_numpy(frame_stack(frames, 5)[None]))
        self.assertAlmostEqual(normalized[0, 3, 0, 0].item(), 5 / 255)

    def test_checkpoint_config_fail_closed_without_real_checkpoint(self):
        with patch.object(diagnosis, "file_sha256", return_value="a" * 64), \
                patch.object(torch, "load", return_value={"format": "haic-rlpd-training-checkpoint-v1", "config": {}}):
            with self.assertRaisesRegex(ValueError, "config"):
                diagnosis.load_models(Path("synthetic.pt"), "a" * 64)
        with patch.object(diagnosis, "file_sha256", return_value="b" * 64):
            with self.assertRaisesRegex(ValueError, "changed"):
                diagnosis.load_models(Path("synthetic.pt"), "a" * 64)

    def test_load_rows_exact_mask_roles_and_all_episode_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            frames = np.broadcast_to(np.arange(6, dtype=np.uint8)[:, None, None], (6, 84, 84)).copy()
            native = np.tile([.5, .2, -.6], (5, 1)).astype(np.float32)
            official = native.copy()
            official[:, 1:] = (official[:, 1:] + 1) / 2
            episodes = []
            for index in range(2):
                path = dataset / f"episode-{index}.npz"
                np.savez(path, frames=frames, executed_action=native, applied_action=official,
                         role=np.asarray(["prefix", "oracle", "oracle", "actor", "actor"]),
                         recovery_mask=np.asarray([False, index == 0, False, index == 0, False]))
                episodes.append({"path": path.name, "sha256": diagnosis.file_sha256(path),
                                 "episode_id": index, "geometry_seed": 4272000003, "stratum": "failure"})
            manifest = {"status": "complete", "episodes": episodes, "counts": {"accepted_transitions": 2}}
            (dataset / "manifest.json").write_text(json.dumps(manifest))
            digest = diagnosis.file_sha256(dataset / "manifest.json")
            receipt = {"accepted_transitions": 2, "unique_accepted_transitions": 2}
            with patch.object(diagnosis, "load_recovery_replay", return_value=(None, receipt)) as validator:
                rows, evidence, _ = diagnosis.load_rows(dataset, digest)
                self.assertEqual(validator.call_args.kwargs["required_policy"], diagnosis.FINISH_POLICY)
                self.assertEqual(rows["step"].tolist(), [1, 3])
                self.assertEqual(rows["role"].tolist(), ["oracle", "actor"])
                np.testing.assert_array_equal(rows["observations"][0], frame_stack(frames, 1))
                np.testing.assert_array_equal(rows["observations"][1], frame_stack(frames, 3))
                self.assertEqual(len(evidence["npz_sha256"]), 2)
                # Even an entirely unmasked negative episode must retain its pinned bytes.
                (dataset / episodes[1]["path"]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "NPZ changed"):
                    diagnosis.load_rows(dataset, digest)

    def test_checkpoint_load_only_actor_and_current_critic(self):
        state = {"format": "haic-rlpd-training-checkpoint-v1", "config": asdict(diagnosis.RLPDConfig()),
                 "actor": {"synthetic": "actor"}, "critic": {"synthetic": "critic"},
                 "target_critic": {"synthetic": "must-not-load"}, "environment_steps": 42,
                 "actor_optimizer": {"synthetic": "must-not-load"}}
        actor, critic = MagicMock(), MagicMock()
        with patch.object(diagnosis, "file_sha256", return_value="a" * 64), \
                patch.object(torch, "load", return_value=state) as loader, \
                patch.object(diagnosis, "PixelActor", return_value=actor), \
                patch.object(diagnosis, "PixelCritic", return_value=critic) as constructor:
            _, _, metadata = diagnosis.load_models(Path("synthetic.pt"), "a" * 64)
        constructor.assert_called_once_with(num_qs=10)
        loader.assert_called_once_with(Path("synthetic.pt"), map_location="cpu", weights_only=False)
        actor.load_state_dict.assert_called_once_with(state["actor"], strict=True)
        critic.load_state_dict.assert_called_once_with(state["critic"], strict=True)
        actor.eval.return_value.requires_grad_.assert_called_once_with(False)
        self.assertEqual(metadata["critic"], "current-not-target")

    def test_provenance_before_inference_exclusive_outputs_and_primary_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            digest = diagnosis.file_sha256(dataset / "manifest.json")
            checkpoint = root / "fake.pt"
            checkpoint.write_bytes(b"synthetic provider only")
            rows = tiny_rows()
            evidence = {"counts": {"rows": 65}, "npz_sha256": {}}

            def provider(path, sha):
                receipt = json.loads((root / "runs/diagnosis/provenance.json").read_text())
                self.assertEqual(receipt["runtime"]["device"], "cpu")
                self.assertEqual(receipt["runtime"]["threads"], 1)
                self.assertEqual(receipt["checkpoints"]["source"]["sha256"], sha)
                self.assertEqual(receipt["learner_config_pin"], asdict(diagnosis.RLPDConfig()))
                return TinyActor(), TinyCritic(), {"synthetic": True}

            with patch.object(diagnosis, "load_rows", return_value=(rows, evidence, {})):
                summary = diagnosis.run(dataset, digest, [f"source={checkpoint}"], "runs/diagnosis",
                                        root=root, model_provider=provider)
                with self.assertRaises(FileExistsError):
                    diagnosis.run(dataset, digest, [f"source={checkpoint}"], "runs/diagnosis", root=root)
                with self.assertRaises(ValueError):
                    diagnosis.run(dataset, digest, [f"source={checkpoint}"] * 2, "runs/duplicate", root=root)
            with np.load(root / "runs/diagnosis/predictions.npz", allow_pickle=False) as audit:
                self.assertEqual(audit["source__q_target_heads"].shape, (65, 10))
                np.testing.assert_array_equal(audit["target_native"], rows["target_native"])
                self.assertEqual(audit["observation_sha256"][0], hashlib.sha256(rows["observations"][0].tobytes()).hexdigest())
            self.assertEqual(summary["environment_resets"], 0)
            self.assertEqual(summary["learner_updates"], 0)

    def test_nonfinite_output_rejected(self):
        class BadCritic(TinyCritic):
            def __call__(self, obs, action):
                return torch.full((len(obs), 10), float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            diagnosis.predict(tiny_rows(1), TinyActor(), BadCritic())


if __name__ == "__main__":
    unittest.main()
