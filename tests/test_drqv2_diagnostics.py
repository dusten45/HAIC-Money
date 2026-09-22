import copy
import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import diagnose_drqv2 as diagnostics
from agent import Agent, DrQActor
from common_adapter import ActionSpec, ObservationSpec, Transition
from drq_v2 import DrQv2Config, Uint8Replay, random_shift


def pixel_stack(episode, step):
    pixels = np.arange(84 * 84, dtype=np.uint16).reshape(84, 84)
    return np.stack([((pixels + episode * 31 + max(0, step + channel - 3) * 7) % 256).astype(np.uint8)
                     for channel in range(4)])


class TestDrQDiagnostics(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.actor = self.root / "actor.pt"
        self.config = DrQv2Config(feature_dim=8, hidden_dim=8, replay_capacity=16, batch_size=2,
                                 warmup_steps=4, device="cuda")
        replay = Uint8Replay(capacity=16, n_step=3)
        for episode in range(2):
            for step in range(7):
                replay.add(Transition(
                    observation=pixel_stack(episode, step), action=np.array([.5, 0., -1.], dtype=np.float32),
                    reward=1. if step % 2 else -.4, next_observation=pixel_stack(episode, step + 1),
                    terminated=step == 6, truncated=False, episode_id=episode, step=step,
                ))
        self.payload = {
            "format": "haic-drq-v2-checkpoint-v1", "config": asdict(self.config),
            "observation_spec": asdict(ObservationSpec()), "action_spec": asdict(ActionSpec()),
            "environment_steps": 14, "gradient_steps": 11, "replay": replay.state_dict(),
        }
        torch.save(self.payload, self.checkpoint)
        with torch.random.fork_rng(devices=[]):
            model = DrQActor(8, 8)
        self.export = {"format": "haic-drq-v2-actor-v1", "config": asdict(self.config),
                       "observation_spec": asdict(ObservationSpec()), "action_spec": asdict(ActionSpec()),
                       "state_dict": model.state_dict()}
        torch.save(self.export, self.actor)

    def test_shared_sample_is_seeded_uint8_chw_without_global_rng_changes(self):
        torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
        obs, source, sample = diagnostics.load_sample(self.checkpoint, 8, 9)
        again, _, same = diagnostics.load_sample(self.checkpoint, 8, 9)
        _, _, different = diagnostics.load_sample(self.checkpoint, 8, 10)
        np.testing.assert_array_equal(obs, again)
        self.assertEqual(sample, same)
        self.assertNotEqual(sample["indices"]["sha256"], different["indices"]["sha256"])
        self.assertEqual(obs.dtype, np.uint8)
        self.assertEqual(obs.shape, (8, 4, 84, 84))
        self.assertEqual(source["saved_device"], "cuda")
        self.assertEqual(sample["distribution"]["episode_count"], 2)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        np.testing.assert_equal(numpy_state, np.random.get_state())
        self.assertEqual(python_state, random.getstate())

    def test_sample_reconstructs_order_and_rejects_cross_episode_starts(self):
        self.payload["replay"]["terminal"][6] = False
        self.payload["replay"]["terminated"][6] = False
        torch.save(self.payload, self.checkpoint)
        obs, _, _ = diagnostics.load_sample(self.checkpoint, 11, 1)
        allowed = [pixel_stack(episode, step) for episode in range(2)
                   for step in range(7) if episode == 1 or step < 4]
        self.assertCountEqual([row.tobytes() for row in obs], [row.tobytes() for row in allowed])
        with self.assertRaisesRegex(ValueError, "only 11 valid"):
            diagnostics.load_sample(self.checkpoint, 12, 1)

    def test_malformed_checkpoint_metadata_fails_closed(self):
        changes = [
            lambda p: p.update(format="other"),
            lambda p: p["observation_spec"].update(channel_order="HWC"),
            lambda p: p["action_spec"].update(frame_skip=8),
            lambda p: p["replay"].update(size=17),
            lambda p: p["replay"].update(n_step=2),
            lambda p: p["replay"].update(frames=p["replay"]["frames"].astype(np.float32)),
            lambda p: p["replay"]["rewards"].__setitem__(0, np.nan),
            lambda p: p["replay"]["sequence_ids"].__setitem__(0, 99),
            lambda p: p["replay"]["boundary_observations"].update({6: np.zeros((84, 84, 4), dtype=np.uint8)}),
        ]
        for change in changes:
            payload = copy.deepcopy(self.payload)
            change(payload)
            torch.save(payload, self.checkpoint)
            with self.subTest(change=change), self.assertRaises(ValueError):
                diagnostics.load_sample(self.checkpoint, 2, 0)
        torch.save(self.payload, self.checkpoint)
        for size, seed in ((0, 1), (-1, 1), (1, -1), (True, 1)):
            with self.subTest(size=size, seed=seed), self.assertRaises(ValueError):
                diagnostics.load_sample(self.checkpoint, size, seed)

    def test_shifts_match_native_uint8_augmentation_and_preserve_frame_alignment(self):
        x = torch.from_numpy(np.stack([pixel_stack(0, 5), pixel_stack(1, 4)]))
        original = x.clone()
        for pad in (1, 4):
            generator = torch.Generator(device="cpu").manual_seed(13)
            offsets = np.asarray([[int(torch.randint(0, 2 * pad + 1, (), generator=generator)) - pad
                                   for _ in range(2)] for _ in range(2)])
            actual = diagnostics.shift_uint8(x, offsets, pad)
            self.assertTrue(torch.equal(actual, random_shift(x, pad=pad, seed=13)))
            self.assertTrue(torch.equal(diagnostics.shift_uint8(x, np.zeros((2, 2), dtype=int), pad), x))
            repeated = x[:, -1:].expand(-1, 4, -1, -1)
            shifted = diagnostics.shift_uint8(repeated, offsets, pad)
            self.assertTrue(torch.equal(shifted[:, 0], shifted[:, 3]))
        self.assertTrue(torch.equal(original, x))
        with self.assertRaises(ValueError):
            diagnostics.shift_uint8(x.float(), np.zeros((2, 2), dtype=int), 1)
        with self.assertRaises(ValueError):
            diagnostics.shift_uint8(x, np.ones((2, 2), dtype=int) * 4, 1)

    def test_predictions_match_minimal_agent_and_do_not_mutate_weights_or_pixels(self):
        runtime = Agent(model_path=str(self.actor))
        obs, _, _ = diagnostics.load_sample(self.checkpoint, 4, 3)
        before = obs.copy()
        weights = {key: value.clone() for key, value in runtime.model.state_dict().items()}
        logits, native, official = diagnostics.predict(runtime.model, obs)
        with torch.inference_mode():
            expected = runtime.model(torch.from_numpy(obs).float() / 255).numpy()
        np.testing.assert_array_equal(native, expected)
        for row in range(len(obs)):
            np.testing.assert_allclose(official[row], runtime.act(obs[row].astype(np.float32) / 255), atol=1e-6, rtol=0)
        identical = np.repeat(obs[:, -1:], 4, axis=1)
        reference = diagnostics.predict(runtime.model, identical)
        for variant in ("reverse_past_keep_newest", "duplicate_newest"):
            changed = diagnostics.predict(runtime.model, identical, variant)
            comparison = diagnostics.action_difference(reference, changed)
            self.assertEqual(comparison["official_action_mae"], [0., 0., 0.])
        np.testing.assert_array_equal(before, obs)
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, runtime.model.state_dict()[key]))
        with self.assertRaises(ValueError):
            diagnostics.predict(runtime.model, np.transpose(obs, (0, 2, 3, 1)))

    def test_report_uses_one_replay_for_all_actors_and_finite_aggregates(self):
        second = self.root / "l2.pt"
        export = copy.deepcopy(self.export)
        export["config"]["steering_logit_l2"] = .001
        torch.save(export, second)
        torch_state = torch.get_rng_state()
        with mock.patch.object(diagnostics, "load_sample", wraps=diagnostics.load_sample) as load:
            with mock.patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA initialized")), \
                    mock.patch("drq_v2.DrQv2Agent.__init__", side_effect=AssertionError("learner constructed")):
                report = diagnostics.build_report(self.checkpoint, [self.actor, second], sample_size=4, sample_seed=11)
        self.assertEqual(load.call_count, 1)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        json.dumps(report, allow_nan=False)
        first, other = report["actors"]
        self.assertNotEqual(first["sha256"], other["sha256"])
        for field in ("sample_indices_sha256", "observations_sha256", "baseline", "perturbations", "independent_random_view_pairs"):
            self.assertEqual(first[field], other[field])
        self.assertEqual(report["sample"]["observations"]["dtype"], "|u1")
        self.assertEqual(report["runtime"]["threads"], 1)
        self.assertEqual(report["runtime"]["interop_threads"], 1)
        self.assertEqual(len(report["perturbations"]), 14)
        for name in ("random_pad1_0", "random_pad4_0"):
            self.assertEqual(report["perturbations"][name]["offsets"]["shape"], [4, 2])

    def test_invalid_actor_contract_is_rejected(self):
        self.export["observation_spec"]["channel_order"] = "HWC"
        torch.save(self.export, self.actor)
        with self.assertRaisesRegex(ValueError, "observation spec"):
            diagnostics.build_report(self.checkpoint, [self.actor], sample_size=2)

    def test_cli_never_overwrites_and_requires_cpu21(self):
        output = self.root / "report.json"
        output.write_text("original")
        args = ["--checkpoint", str(self.checkpoint), "--actor", str(self.actor), "--output", str(output), "--sample-size", "4"]
        with mock.patch.object(diagnostics, "build_report", side_effect=AssertionError("loaded before refusal")):
            with self.assertRaises(FileExistsError):
                diagnostics.main(args)
        self.assertEqual(output.read_text(), "original")
        output.unlink()
        with mock.patch.object(torch.version, "cuda", "12.8"), self.assertRaisesRegex(RuntimeError, "Torch 2.1.0"):
            diagnostics.main(args)
        self.assertFalse(output.exists())
        if torch.__version__.split("+")[0] == "2.1.0" and torch.version.cuda is None:
            diagnostics.main(args)
            self.assertEqual(json.loads(output.read_text())["sample"]["size"], 4)
            with self.assertRaises(FileExistsError):
                diagnostics.main(args)


if __name__ == "__main__":
    unittest.main()
