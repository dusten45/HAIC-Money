import hashlib
import unittest
from unittest.mock import patch

import numpy as np

from common_adapter import ActionAdapter
from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from haic.algorithms.dreamer_v3.offline import replay_from_dataset_bytes
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode


SOURCE = "a" * 64


def fixture(*, truncated=False, finished=False, steps=3, source_id="teacher", seed="512345"):
    frames = np.stack([
        np.full((84, 84), value, dtype=np.uint8)
        for value in range(10, 10 + steps + 1)
    ])
    actions = np.tile([0.25, -0.4, 0.15], (steps, 1)).astype(np.float32)
    adapter = ActionAdapter()
    episode = TeacherEpisode(
        frames=frames,
        actions=actions,
        applied_actions=np.stack([adapter.to_official(row, clip=False) for row in actions]),
        rewards=np.arange(steps, dtype=np.float32),
        terminated=np.asarray([False] * (steps - 1) + [not truncated]),
        truncated=np.asarray([False] * (steps - 1) + [truncated]),
        terminal=np.asarray([False] * (steps - 1) + [not truncated or finished]),
        finished=np.asarray([False] * (steps - 1) + [finished]),
        episode_id=0, source_id=source_id, source_actor_sha256=SOURCE,
        geometry_id=seed, track_id=1, complete=True,
    )
    dataset = TeacherDataset([episode])
    digest = dataset.seal()
    data = dataset.to_bytes()
    return data, digest


class TestDreamerOfflineReplay(unittest.TestCase):
    def load(self, data, digest, **overrides):
        kwargs = {
            "archive_sha256": hashlib.sha256(data).hexdigest(),
            "dataset_digest": digest,
            "source_id": "teacher",
            "source_actor_sha256": SOURCE,
            "allowed_cells": [(1, 512345)],
            "excluded_seeds": [930000],
            "capacity": 8,
        }
        kwargs.update(overrides)
        return replay_from_dataset_bytes(data, **kwargs)

    def test_complete_terminal_and_truncated_result_frames(self):
        for truncated in (False, True):
            with self.subTest(truncated=truncated):
                data, digest = fixture(truncated=truncated)
                replay, receipt = self.load(data, digest)
                self.assertEqual(receipt["decisions"], 3)
                self.assertEqual(receipt["distinct_finished_cells"], 0)
                batch = replay.sample_sequence(2, 4, short_episode_fraction=1.0)
                self.assertEqual(batch["actions"].shape, (2, 3, 3))
                self.assertEqual(batch["observations"].shape, (2, 4, 4, 84, 84))
                self.assertTrue(batch["is_last"][:, -1].all().item())
                self.assertEqual(batch["is_terminal"][:, -1].sum().item(), 0 if truncated else 2)
                np.testing.assert_allclose(
                    batch["observations"][:, -1, -1, 0, 0].numpy() * 255,
                    [13, 13], atol=1e-4,
                )
                np.testing.assert_allclose(
                    batch["actions"][:, :, 0].numpy(), 0.25, atol=1e-6,
                )

    def test_real_finish_is_truncated_and_terminal_without_terminated(self):
        data, digest = fixture(truncated=True, finished=True, steps=1)
        replay, receipt = self.load(data, digest)
        self.assertEqual(receipt["distinct_finished_cells"], 1)
        batch = replay.sample_sequence(1, 32, short_episode_fraction=1.0)
        self.assertEqual(batch["effective_seq_len"], 1)
        self.assertTrue(batch["is_last"][0, 0].item())
        self.assertTrue(batch["is_terminal"][0, 0].item())

    def test_rejects_archive_source_cell_exclusion_and_capacity_mismatches(self):
        data, digest = fixture()
        for overrides, expected in (
            ({"archive_sha256": "0" * 64}, "archive hash"),
            ({"dataset_digest": "0" * 64}, "digest"),
            ({"source_id": "random"}, "source actor"),
            ({"source_actor_sha256": "b" * 64}, "source actor"),
            ({"allowed_cells": [(1, 512346)]}, "undeclared TRAIN"),
            ({"excluded_seeds": [512345]}, "reserved"),
            ({"capacity": 2}, "overwrite"),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    self.load(data, digest, **overrides)

    def test_rejects_complete_episode_without_real_done_flag(self):
        steps = 1
        frames = np.stack([
            np.full((84, 84), 10 + value, dtype=np.uint8)
            for value in range(steps + 1)
        ])
        actions = np.zeros((steps, 3), dtype=np.float32)
        episode = TeacherEpisode(
            frames=frames, actions=actions,
            applied_actions=np.stack([ActionAdapter().to_official(row) for row in actions]),
            rewards=np.zeros(steps, dtype=np.float32),
            terminated=[False], truncated=[False], finished=[True], terminal=[True],
            episode_id=0, source_id="teacher", source_actor_sha256=SOURCE,
            geometry_id="512345", track_id=1, complete=True,
        )
        dataset = TeacherDataset([episode])
        digest = dataset.seal()
        with self.assertRaisesRegex(ValueError, "real terminated/truncated"):
            self.load(dataset.to_bytes(), digest)

    def test_rejects_timeout_that_claims_terminal_without_finish_or_retirement(self):
        frames = np.stack([
            np.full((84, 84), 10 + step, dtype=np.uint8) for step in range(2)
        ])
        actions = np.zeros((1, 3), dtype=np.float32)
        episode = TeacherEpisode(
            frames=frames, actions=actions,
            applied_actions=np.stack([ActionAdapter().to_official(row) for row in actions]),
            rewards=[0.0], terminated=[False], truncated=[True],
            finished=[False], terminal=[True], retire_reasons=[None],
            episode_id=0, source_id="teacher", source_actor_sha256=SOURCE,
            geometry_id="512345", track_id=1, complete=True,
        )
        dataset = TeacherDataset([episode])
        digest = dataset.seal()
        with self.assertRaisesRegex(ValueError, "terminal label contradicts"):
            self.load(dataset.to_bytes(), digest)

    def test_capacity_rejects_compressed_corpus_before_full_materialization(self):
        data, digest = fixture(steps=3)
        with patch(
            "haic.algorithms.dreamer_v3.offline.TeacherDataset.from_bytes",
            side_effect=AssertionError("must reject before decompression"),
        ):
            with self.assertRaisesRegex(ValueError, "replay capacity"):
                self.load(data, digest, capacity=2)

        compressed, digest = fixture(steps=20)
        with patch(
            "haic.algorithms.dreamer_v3.offline.np.lib.format.read_magic",
            side_effect=AssertionError("must reject oversized archive before array header"),
        ):
            with self.assertRaisesRegex(ValueError, "memory bound"):
                self.load(compressed, digest, capacity=1)

    def test_offline_replay_supports_synthetic_model_and_actor_updates(self):
        data, digest = fixture(truncated=True)
        replay, _ = self.load(data, digest)
        config = DreamerV3Config(
            device="cpu", embed_dim=64, hidden_dim=32,
            num_categoricals=4, num_classes=4,
            batch_size=2, seq_len=4, burnin_steps=1,
            short_episode_fraction=1.0, imagination_horizon=2,
            warmup_steps=0, replay_capacity=8,
        )
        agent = DreamerV3Agent(config, seed=23)
        agent.replay = replay
        for model_only in (True, False):
            metrics = agent.update(model_only=model_only)
            self.assertTrue(np.isfinite(metrics["loss_wm"]))
            self.assertEqual(metrics["effective_seq_len"], 3.0)
            self.assertEqual(metrics["unique_short_episodes"], 1.0)
            if not model_only:
                self.assertTrue(np.isfinite(metrics["loss_actor"]))
        self.assertEqual(agent.gradient_steps, 2)
        self.assertEqual(agent.environment_steps, 0)
