import json
import copy
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from action_smoothing import (
    build_action_smoother,
    canonical_action_control,
    canonical_action_smoothing,
)
from agent import (
    Agent,
    Baseline1Actor,
    DRQ_ACTOR_FORMAT,
    DREAMERV3_ACTOR_FORMAT,
    RLPD_ACTOR_FORMAT,
    DreamerV3Encoder,
    DreamerV3RSSM,
    DreamerV3Actor,
    DreamerV3ExportedActor,
)
from export_policy import export_payload, source_action_smoothing
from export_policy import ACTOR_STATE_KEYS, extract_actor_state
from common_adapter import ActionAdapter, ActionSpec, ObservationSpec
from drq_v2 import DrQActor as NativeDrQActor, DrQv2Config
from haic.algorithms.rlpd.model import PixelActor as NativeRLPDActor


class TestSubmissionPolicy(unittest.TestCase):
    def drq_payload(self):
        config = DrQv2Config(feature_dim=16, hidden_dim=16, device="cuda")
        actor = NativeDrQActor(feature_dim=16, hidden_dim=16)
        return actor, {
            "format": DRQ_ACTOR_FORMAT,
            "config": asdict(config),
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
            "state_dict": actor.state_dict(),
        }

    def dreamerv3_payload(self):
        enc = DreamerV3Encoder(4, 64)
        rssm = DreamerV3RSSM(3, 64, 64, 8, 8, 0.01)
        act = DreamerV3Actor(64 + 64, 3)
        model = DreamerV3ExportedActor(enc, rssm, act)
        return model, {
            "format": DREAMERV3_ACTOR_FORMAT,
            "config": {
                "embed_dim": 64, "hidden_dim": 64,
                "num_categoricals": 8, "num_classes": 8, "unimix": 0.01,
            },
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
            "state_dict": model.state_dict(),
        }

    def rlpd_payload(self):
        actor = NativeRLPDActor()
        return actor, {
            "format": RLPD_ACTOR_FORMAT,
            "config": {"latent_dim": 50},
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
            "actor_state_dict": actor.state_dict(),
        }

    def test_predict_action_has_submission_bounds(self):
        actor = Baseline1Actor()
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias.copy_(torch.tensor([2.0, -1.0, 3.0]))

        actions = actor.predict_action(torch.zeros((2, 4, 84, 84)))

        self.assertTrue(torch.equal(actions, torch.tensor([[1.0, 0.0, 1.0]]).repeat(2, 1)))

    def test_extract_actor_state_ignores_critic_weights(self):
        actor = Baseline1Actor()
        source_state = dict(actor.state_dict())
        source_state["value_net.weight"] = torch.zeros((1, 512))
        source_state["value_net.bias"] = torch.zeros(1)

        exported = extract_actor_state(source_state)

        self.assertEqual(tuple(exported), ACTOR_STATE_KEYS)
        self.assertEqual(set(exported), set(actor.state_dict()))

    def test_agent_sequence_uses_exported_smoothing_and_reset(self):
        actor = Baseline1Actor()
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias.copy_(torch.tensor([1.0, 1.0, 1.0]))
        config = canonical_action_smoothing("alpha", [0.5, 1.0, 1.0])
        expected_smoother = build_action_smoother(config)
        expected_smoother.reset(initial_action=config["initial_action"])
        observation = np.zeros((4, 84, 84), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            torch.save(export_payload(actor, config), model_path)
            with patch("agent.MODEL_FILENAME", str(model_path)):
                agent = Agent()
                first = agent.act(observation)
                second = agent.act(observation)
                agent.reset(observation)
                reset_first = agent.act(observation)

        expected_first = np.asarray(expected_smoother.smooth([1.0, 1.0, 1.0]))
        expected_second = np.asarray(expected_smoother.smooth([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(first, expected_first)
        np.testing.assert_allclose(second, expected_second)
        np.testing.assert_allclose(reset_first, expected_first)

    def test_export_reads_action_smoothing_from_evaluation_snapshot(self):
        config = canonical_action_smoothing("alpha", [0.35, 1.0, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidates" / "candidate.zip"
            candidate.parent.mkdir()
            candidate.touch()
            (root / "candidates.json").write_text(json.dumps([
                {
                    "evaluation_archive_path": "candidates/candidate.zip",
                    "action_smoothing": config,
                }
            ]))

            resolved = source_action_smoothing(candidate)

        self.assertEqual(resolved, config)

    def test_agent_rejects_smoothing_fingerprint_mismatch(self):
        actor = Baseline1Actor()
        payload = export_payload(actor, canonical_action_smoothing("alpha", [0.5, 1.0, 1.0]))
        payload["action_smoothing_fingerprint"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            torch.save(payload, model_path)
            with patch("agent.MODEL_FILENAME", str(model_path)):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    Agent()

    def test_five_channel_agent_appends_previous_steering_plane(self):
        actor = Baseline1Actor(input_channels=5)
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias.copy_(torch.tensor([1.0, 1.0, 0.0]))
        smoothing = canonical_action_smoothing("alpha", [0.5, 1.0, 1.0])
        control = canonical_action_control("previous-steering-plane")
        observation = np.zeros((4, 84, 84), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.pt"
            torch.save(export_payload(actor, smoothing, control), model_path)
            with patch("agent.MODEL_FILENAME", str(model_path)):
                agent = Agent()
                first = agent.act(observation)
                second = agent.act(observation)
                agent.reset(observation)
                reset_first = agent.act(observation)

        np.testing.assert_allclose(first, [0.5, 1.0, 0.0])
        np.testing.assert_allclose(second, [0.75, 1.0, 0.0])
        np.testing.assert_allclose(reset_first, first)

    def test_drq_export_matches_native_actor_and_full_reset_sequence(self):
        source, payload = self.drq_payload()
        rng = np.random.default_rng(17)
        observations = [
            np.zeros((4, 84, 84), dtype=np.float32),
            np.ones((4, 84, 84), dtype=np.float32),
            rng.random((4, 84, 84), dtype=np.float32),
        ]
        expected = [ActionAdapter().to_official(source.act(obs)) for obs in observations]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(payload, path)
            before = torch.get_rng_state().clone()
            agent = Agent(path)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            np.testing.assert_array_equal(agent.act(observations[0]), expected[0])
            agent.reset(observations[0])
            first = [agent.act(obs) for obs in observations]
            for obs in reversed(observations):
                agent.act(obs)
            agent.reset(observations[0])
            second = [agent.act(obs) for obs in observations]
            fresh = Agent(path)
            third = [fresh.act(obs) for obs in observations]
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, third)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in agent.model.parameters()))

    def test_drq_rejects_malformed_exports_and_observations(self):
        _, payload = self.drq_payload()
        cases = []
        unknown = copy.deepcopy(payload)
        unknown["format"] = "unknown-actor"
        cases.append(unknown)
        for section, key, value in (
            ("observation_spec", "channel_order", "HWC"),
            ("observation_spec", "high", 255.0),
            ("action_spec", "frame_skip", 8),
            ("action_spec", "order", ("gas", "steer", "brake")),
            ("config", "hidden_dim", 32),
        ):
            malformed = copy.deepcopy(payload)
            malformed[section][key] = value
            cases.append(malformed)
        nonfinite = copy.deepcopy(payload)
        nonfinite["state_dict"]["policy.bias"][0] = float("nan")
        cases.append(nonfinite)
        missing = copy.deepcopy(payload)
        del missing["state_dict"]["policy.bias"]
        cases.append(missing)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            for index, malformed in enumerate(cases):
                with self.subTest(case=index):
                    torch.save(malformed, path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        Agent(path)
            torch.save(payload, path)
            agent = Agent(path)
            for observation in (
                np.zeros((4, 84, 84), dtype=np.uint8),
                np.zeros((84, 84, 4), dtype=np.float32),
                np.full((4, 84, 84), 1.1, dtype=np.float32),
                np.full((4, 84, 84), np.nan, dtype=np.float32),
            ):
                with self.assertRaisesRegex(ValueError, "observation"):
                    agent.act(observation)

    def test_drq_root_only_runtime_does_not_import_training_modules(self):
        _, payload = self.drq_payload()
        root = Path(__file__).resolve().parents[1]
        code = """
import builtins
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'stable_baselines3', 'drq_v2', 'common_adapter', 'train'}:
        raise AssertionError('training import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from agent import Agent
import numpy as np
agent = Agent()
observation = np.zeros((4, 84, 84), dtype=np.float32)
first = agent.act(observation)
agent.reset(observation)
assert np.array_equal(first, agent.act(observation))
assert first.shape == (3,) and np.isfinite(first).all()
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for filename in ("agent.py", "action_smoothing.py", "action_representation.py"):
                shutil.copy2(root / filename, path / filename)
            torch.save(payload, path / "model.pt")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=path,
                env=environment, capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rlpd_export_matches_training_actor_and_official_mapping(self):
        source, payload = self.rlpd_payload()
        rng = np.random.default_rng(29)
        observations = [
            np.zeros((4, 84, 84), dtype=np.float32),
            np.ones((4, 84, 84), dtype=np.float32),
            rng.random((4, 84, 84), dtype=np.float32),
        ]
        expected = []
        for observation in observations:
            tensor = torch.from_numpy(observation).unsqueeze(0)
            native, _, _ = source.sample(tensor, deterministic=True)
            expected.append(
                ActionAdapter().to_official(native.squeeze(0).detach().numpy())
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(payload, path)
            agent = Agent(path)
            actual = [agent.act(observation) for observation in observations]

        self.assertEqual(agent.format, RLPD_ACTOR_FORMAT)
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in agent.model.parameters()))

    def test_rlpd_maps_native_action_to_official_box_once(self):
        source, payload = self.rlpd_payload()
        with torch.no_grad():
            source.mean.weight.zero_()
            source.mean.bias.copy_(torch.tensor([0.2, 0.0, -0.4]))
        payload["actor_state_dict"] = source.state_dict()
        observation = np.zeros((4, 84, 84), dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(payload, path)
            action = Agent(path).act(observation)

        np.testing.assert_allclose(
            action,
            [np.tanh(0.2), 0.5, (np.tanh(-0.4) + 1.0) * 0.5],
            rtol=0.0,
            atol=1e-7,
        )

    def test_rlpd_reset_and_reload_preserve_deterministic_trace(self):
        _source, payload = self.rlpd_payload()
        rng = np.random.default_rng(31)
        observations = [
            rng.random((4, 84, 84), dtype=np.float32) for _ in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(payload, path)
            agent = Agent(path)
            first = [agent.act(observation) for observation in observations]
            agent.reset(observations[0])
            reset_trace = [agent.act(observation) for observation in observations]
            reloaded_agent = Agent(path)
            reloaded_trace = [
                reloaded_agent.act(observation) for observation in observations
            ]

        for expected, reset_action, reloaded_action in zip(
            first, reset_trace, reloaded_trace
        ):
            np.testing.assert_array_equal(expected, reset_action)
            np.testing.assert_array_equal(expected, reloaded_action)

    def test_rlpd_rejects_malformed_exports_and_observations(self):
        _source, payload = self.rlpd_payload()
        cases = []
        wrong_tag = copy.deepcopy(payload)
        wrong_tag["format"] = "haic-rlpd-pixel-actor-v0"
        cases.append(wrong_tag)
        for section, key, value in (
            ("observation_spec", "channel_order", "HWC"),
            ("action_spec", "frame_skip", 8),
            ("config", "latent_dim", 51),
        ):
            malformed = copy.deepcopy(payload)
            malformed[section][key] = value
            cases.append(malformed)
        extra_config = copy.deepcopy(payload)
        extra_config["config"]["hidden_dim"] = 256
        cases.append(extra_config)
        missing = copy.deepcopy(payload)
        del missing["actor_state_dict"]["mean.bias"]
        cases.append(missing)
        extra = copy.deepcopy(payload)
        extra["actor_state_dict"]["unexpected.weight"] = torch.zeros(1)
        cases.append(extra)
        wrong_shape = copy.deepcopy(payload)
        wrong_shape["actor_state_dict"]["encoder.projection.1.weight"] = torch.zeros(
            (51, 4096)
        )
        cases.append(wrong_shape)
        for nonfinite in (float("nan"), float("inf")):
            invalid = copy.deepcopy(payload)
            invalid["actor_state_dict"]["mean.bias"][0] = nonfinite
            cases.append(invalid)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            for index, malformed in enumerate(cases):
                with self.subTest(case=index):
                    torch.save(malformed, path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        Agent(path)

            torch.save(payload, path)
            agent = Agent(path)
            for observation in (
                np.zeros((4, 84, 84), dtype=np.uint8),
                np.zeros((84, 84, 4), dtype=np.float32),
                np.full((4, 84, 84), 1.1, dtype=np.float32),
                np.full((4, 84, 84), np.nan, dtype=np.float32),
            ):
                with self.assertRaisesRegex(ValueError, "RLPD observation"):
                    agent.act(observation)

    def test_rlpd_root_only_runtime_does_not_import_research_modules(self):
        _source, payload = self.rlpd_payload()
        root = Path(__file__).resolve().parents[1]
        code = """
import builtins
import sys
import torch
import numpy as np
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'haic', 'drq_v2', 'dreamer_v3', 'stable_baselines3'}:
        raise AssertionError('research import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from agent import Agent, RLPD_ACTOR_FORMAT
agent = Agent()
assert agent.format == RLPD_ACTOR_FORMAT
observation = np.zeros((4, 84, 84), dtype=np.float32)
first = agent.act(observation)
agent.reset(observation)
assert np.array_equal(first, agent.act(observation))
assert first.shape == (3,) and np.isfinite(first).all()
assert all(parameter.device.type == 'cpu' for parameter in agent.model.parameters())
assert not torch.cuda.is_initialized()
assert not any(name == 'haic' or name.startswith('haic.') for name in sys.modules)
assert 'common_adapter' not in sys.modules
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shutil.copy2(root / "agent.py", path / "agent.py")
            torch.save(payload, path / "model.pt")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=path,
                env=environment, capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dreamerv3_root_only_runtime_does_not_import_training_modules(self):
        _, payload = self.dreamerv3_payload()
        root = Path(__file__).resolve().parents[1]
        code = """
import builtins
original_import = builtins.__import__
def restricted(name, *args, **kwargs):
    if name.split('.')[0] in {'stable_baselines3', 'drq_v2', 'dreamer_v3', 'common_adapter', 'train'}:
        raise AssertionError('training import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = restricted
from agent import Agent
import numpy as np
agent = Agent()
observation = np.zeros((4, 84, 84), dtype=np.float32)
first = agent.act(observation)
agent.reset(observation)
assert np.array_equal(first, agent.act(observation))
assert first.shape == (3,) and np.isfinite(first).all()
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for filename in ("agent.py", "action_smoothing.py", "action_representation.py"):
                shutil.copy2(root / filename, path / filename)
            torch.save(payload, path / "model.pt")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-B", "-c", code], cwd=path,
                env=environment, capture_output=True, text=True, timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
