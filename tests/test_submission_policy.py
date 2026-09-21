import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from action_smoothing import (
    build_action_smoother,
    canonical_action_control,
    canonical_action_smoothing,
)
from agent import Agent, Baseline1Actor
from export_policy import export_payload, source_action_smoothing
from export_policy import ACTOR_STATE_KEYS, extract_actor_state


class TestSubmissionPolicy(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
