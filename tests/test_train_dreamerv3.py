import json
import random
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from train_dreamerv3 import (
    evaluate_checkpoint,
    selection_score,
    training_protocol,
    verify_checkpoint,
)


def make_result(finish=0.0, progress=0.5, lap=None, eligible=True):
    return {
        "eligible": eligible,
        "determinism_audited": True,
        "summary": {
            "finish_rate": finish,
            "avg_progress": progress,
            "avg_lap_time_ms": lap,
        },
    }


class TestTrainDreamerV3(unittest.TestCase):
    def test_selection_score_order(self):
        self.assertGreater(
            selection_score(make_result(0.1, 0.1, 20000)),
            selection_score(make_result(0.0, 1.0)),
        )
        self.assertGreater(
            selection_score(make_result(0.1, 0.8, 30000)),
            selection_score(make_result(0.1, 0.1, 20000)),
        )
        self.assertGreater(
            selection_score(make_result(0.1, 0.8, 20000)),
            selection_score(make_result(0.1, 0.8, 30000)),
        )
        with self.assertRaises(RuntimeError):
            selection_score(make_result(1.0, 1.0, 10000, eligible=False))

    def test_protocol_reservations(self):
        protocol = {
            "name": "test",
            "frame_skip": 4,
            "max_steps": 2000,
            "partitions": {
                "screen": {"track_ids": [101], "seeds": [31001], "repeats": 2},
                "confirmation": {"track_ids": [111], "seeds": [31101], "repeats": 2},
            },
            "reserved_training_seeds": [42, 1337],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "protocol.json"
            p.write_text(json.dumps(protocol))
            args = Namespace(protocol_file=p, frame_skip=4, max_steps=2000)
            proto, reserved = training_protocol(args)
            self.assertEqual(reserved, [42, 1337, 31001, 31101])

    def test_verify_checkpoint_preserves_rng_and_verifies_recurrent_parity(self):
        cfg = DreamerV3Config(
            device="cpu",
            embed_dim=64,
            hidden_dim=64,
            num_categoricals=8,
            num_classes=8,
            replay_capacity=20,
        )
        agent = DreamerV3Agent(cfg, seed=123)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            actor_path = Path(tmpdir) / "actor.pt"
            agent.save_checkpoint(ckpt_path)
            agent.export_actor(actor_path)

            cpu_rng = torch.get_rng_state().clone()
            py_rng = random.getstate()
            report = verify_checkpoint(
                agent, ckpt_path, actor_path, np.zeros((4, 84, 84), dtype=np.float32)
            )
            self.assertEqual(report["restore_max_abs_error"], 0.0)
            self.assertEqual(report["cpu_export_max_abs_error"], 0.0)
            self.assertEqual(report["resets"], 2)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            self.assertEqual(random.getstate(), py_rng)

    def test_mock_training_loop(self):
        import sys
        from train_dreamerv3 import main

        class DummyEnv:
            def reset(self, **kwargs):
                return np.zeros((4, 84, 84), dtype=np.float32), {"track_id": 1, "seed": 7}
            def step(self, action):
                return np.zeros((4, 84, 84), dtype=np.float32), 1.0, True, False, {"progress": 0.1, "finished": False}
            def close(self): pass

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            args = [
                "train_dreamerv3.py", "--name", "smoke", "--run-dir", str(run_dir),
                "--total-steps", "15", "--warmup-steps", "5", "--batch-size", "2",
                "--seq-len", "4", "--replay-capacity", "50", "--eval-freq", "10",
                "--device", "cpu", "--eval-python", sys.executable,
            ]
            with patch("sys.argv", args), \
                 patch("train_dreamerv3.build_sampled_env", return_value=DummyEnv()), \
                 patch("train_dreamerv3.evaluate_checkpoint", return_value={
                     "evaluation_dir": "mock_eval",
                     "ranked": [{
                         "eligible": True, "determinism_audited": True,
                         "archive_sha256": "mock",
                         "summary": {"finish_rate": 0.0, "avg_progress": 0.5, "avg_lap_time_ms": None}
                     }]
                 }), \
                 patch("train_dreamerv3.check_evaluation_runtime", return_value={}), \
                 patch("train_dreamerv3.file_sha256", return_value="mock"):
                main()

            self.assertTrue((run_dir / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
