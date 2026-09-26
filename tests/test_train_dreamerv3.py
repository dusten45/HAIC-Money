import hashlib
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
            with self.assertRaisesRegex(ValueError, "non-empty source_sha256"):
                training_protocol(args)
            protocol["source_sha256"] = {
                "train_dreamerv3.py": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "train_dreamerv3.py").read_bytes()
                ).hexdigest(),
            }
            p.write_text(json.dumps(protocol))
            proto, reserved = training_protocol(args)
            self.assertEqual(reserved, [42, 1337, 31001, 31101])

    def test_training_development_geometry_must_be_reserved_from_training(self):
        protocol = {
            "name": "dreamer-dev",
            "frame_skip": 4,
            "max_steps": 2000,
            "partitions": {"screen": {
                "track_ids": [4], "seeds": [51001], "repeats": 2,
            }},
            "training_development": {
                "track_ids": [1], "seeds": [51002],
                "cells": [{"track_id": 1, "seed": 51002}],
            },
            "source_sha256": {
                "train_dreamerv3.py": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "train_dreamerv3.py").read_bytes()
                ).hexdigest(),
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "protocol.json"
            path.write_text(json.dumps(protocol))
            args = Namespace(protocol_file=path, frame_skip=4, max_steps=2000)
            with self.assertRaisesRegex(ValueError, "training_development seeds"):
                training_protocol(args)

            protocol["reserved_training_seeds"] = [51002]
            path.write_text(json.dumps(protocol))
            _, reserved = training_protocol(args)
            self.assertEqual(reserved, [51001, 51002])

    def test_training_protocol_freezes_pretraining_and_burnin_budget(self):
        budget = {
            "total_environment_decisions": 4096,
            "random_prefill_decisions": 1024,
            "policy_start_step": 1025,
            "replay_pretrain_updates": 100,
            "online_updates_per_decision": 1,
            "sequence_length": 32,
            "burnin_steps": 8,
            "reset_start_fraction": 0.0,
            "short_episode_fraction": 0.0,
            "batch_size": 16,
            "replay_capacity": 10000,
            "track_sampler_seed": 917,
            "device": "cpu",
            "terminal_window_fraction": 0.0,
            "observation_loss_scale": 1.0,
            "kl_free_nats": 1.0,
            "overshoot_horizon": 1,
            "overshoot_kl_weight": 0.0,
            "overshoot_free_nats": 0.0,
            "continue_positive_weight": 1.0,
        }
        protocol = {
            "name": "dreamer-budget",
            "learner_seeds": [0, 1],
            "frame_skip": 4,
            "max_steps": 2000,
            "training_budget": budget,
            "partitions": {"screen": {
                "track_ids": [4], "seeds": [52001], "repeats": 2,
            }},
            "source_sha256": {
                "train_dreamerv3.py": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "train_dreamerv3.py").read_bytes()
                ).hexdigest(),
            },
        }
        args = Namespace(
            protocol_file=None,
            frame_skip=4,
            max_steps=2000,
            total_steps=4096,
            warmup_steps=1024,
            policy_start_step=1025,
            replay_pretrain_updates=100,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            reset_start_fraction=0.0,
            short_episode_fraction=0.0,
            batch_size=16,
            replay_capacity=10000,
            track_sampler_seed=917,
            device="cpu",
            terminal_window_fraction=0.0,
            observation_loss_scale=1.0,
            kl_free_nats=1.0,
            overshoot_horizon=1,
            overshoot_kl_weight=0.0,
            overshoot_free_nats=0.0,
            continue_positive_weight=1.0,
            seed=0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "protocol.json"
            path.write_text(json.dumps(protocol))
            args.protocol_file = path
            frozen, _ = training_protocol(args)
            self.assertEqual(frozen["training_budget"], budget)

            args.replay_pretrain_updates = 101
            with self.assertRaisesRegex(ValueError, "training_budget"):
                training_protocol(args)
            args.replay_pretrain_updates = 100
            args.reset_start_fraction = 1.0
            with self.assertRaisesRegex(ValueError, "training_budget"):
                training_protocol(args)
            args.reset_start_fraction = 0.0
            args.short_episode_fraction = 1.0
            with self.assertRaisesRegex(ValueError, "training_budget"):
                training_protocol(args)

    def test_training_protocol_rejects_source_hash_drift(self):
        from train_dreamerv3 import ROOT

        source_path = ROOT / "train_dreamerv3.py"
        protocol = {
            "name": "source-pinned",
            "frame_skip": 4,
            "max_steps": 2000,
            "source_sha256": {
                "train_dreamerv3.py": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            },
            "partitions": {"screen": {
                "track_ids": [1], "seeds": [53001], "repeats": 2,
            }},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "protocol.json"
            path.write_text(json.dumps(protocol))
            args = Namespace(protocol_file=path, frame_skip=4, max_steps=2000)
            training_protocol(args)
            protocol["source_sha256"]["train_dreamerv3.py"] = "0" * 64
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                training_protocol(args)

    def test_frozen_b1_v1_protocol_rejects_changed_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-world-model-local-v1.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=1024,
            warmup_steps=1024,
            replay_pretrain_updates=100,
            policy_start_step=1025,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            batch_size=4,
            replay_capacity=4096,
            track_sampler_seed=917,
            device="cuda",
            seed=0,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_frozen_b1_v2_protocol_rejects_changed_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-world-model-local-v2.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_terminal_balanced_v3_protocol_rejects_changed_observation_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-terminal-balanced-local-v3.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_observation_scale_v4_protocol_rejects_changed_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-observation-scale-local-v4.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=0,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_prior_kl_v5_protocol_rejects_changed_overshoot_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-prior-kl-local-v5.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            kl_free_nats=0.1,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=0,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_overshooting_v6_protocol_rejects_changed_free_floor_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-overshooting-local-v6.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            kl_free_nats=0.1,
            overshoot_horizon=5,
            overshoot_kl_weight=1.0,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_overshooting_floor_v7_protocol_rejects_residual_decoder_change(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-overshooting-floor-local-v7.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            kl_free_nats=0.1,
            overshoot_horizon=5,
            overshoot_kl_weight=1.0,
            overshoot_free_nats=0.0,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_residual_frame_v8_protocol_rejects_terminal_weight_change(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-residual-frame-local-v8.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            kl_free_nats=0.1,
            overshoot_horizon=5,
            overshoot_kl_weight=1.0,
            overshoot_free_nats=0.0,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=0,
        )
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            training_protocol(args)

    def test_terminal_positive_weight_v9_protocol_rejects_changed_source(self):
        from train_dreamerv3 import ROOT

        args = Namespace(
            protocol_file=ROOT / "experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json",
            frame_skip=4,
            max_steps=2000,
            track_ids="1,2,3,4",
            total_steps=12288,
            warmup_steps=12288,
            replay_pretrain_updates=500,
            policy_start_step=12289,
            updates_per_step=1,
            seq_len=32,
            burnin_steps=8,
            terminal_window_fraction=0.5,
            observation_loss_scale=256.0,
            kl_free_nats=0.1,
            overshoot_horizon=5,
            overshoot_kl_weight=1.0,
            overshoot_free_nats=0.0,
            continue_positive_weight=56.0,
            batch_size=4,
            replay_capacity=16384,
            track_sampler_seed=917,
            device="cuda",
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "frozen source hash mismatch: dreamer_v3.py"):
            training_protocol(args)

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

            agent._online_h.fill_(0.25)
            agent._online_z.fill_(0.5)
            agent._online_prev_a.copy_(torch.tensor([[0.1, -0.2, 0.3]]))
            agent._online_first = False
            agent._online_observation_pending = True
            agent._last_policy_stats = {"std": np.asarray([0.3, 0.4, 0.5])}
            online_state = (
                agent._online_h.clone(),
                agent._online_z.clone(),
                agent._online_prev_a.clone(),
                agent._online_first,
                agent._online_observation_pending,
                {key: value.copy() for key, value in agent._last_policy_stats.items()},
            )
            np.random.random(3)

            cpu_rng = torch.get_rng_state().clone()
            py_rng = random.getstate()
            np_rng = np.random.get_state()
            report = verify_checkpoint(
                agent, ckpt_path, actor_path, np.zeros((4, 84, 84), dtype=np.float32)
            )
            self.assertEqual(report["restore_max_abs_error"], 0.0)
            self.assertEqual(report["cpu_export_max_abs_error"], 0.0)
            self.assertEqual(report["resets"], 2)
            self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
            self.assertEqual(random.getstate(), py_rng)
            actual_online_state = (
                agent._online_h,
                agent._online_z,
                agent._online_prev_a,
                agent._online_first,
                agent._online_observation_pending,
                agent._last_policy_stats,
            )
            for actual, expected in zip(actual_online_state[:3], online_state[:3]):
                torch.testing.assert_close(actual, expected)
            self.assertEqual(actual_online_state[3:5], online_state[3:5])
            np.testing.assert_array_equal(
                actual_online_state[5]["std"], online_state[5]["std"]
            )
            restored_np_rng = np.random.get_state()
            self.assertEqual(restored_np_rng[0], np_rng[0])
            np.testing.assert_array_equal(restored_np_rng[1], np_rng[1])
            self.assertEqual(restored_np_rng[2:], np_rng[2:])

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
            episode_rows = [
                json.loads(line)
                for line in (run_dir / "episodes.jsonl").read_text().splitlines()
            ]
            policy_episode = next(
                row for row in episode_rows
                if row.get("event") == "end" and row.get("policy_std_mean") is not None
            )
            self.assertEqual(len(policy_episode["proposed_out_of_bounds_fraction"]), 3)
            self.assertEqual(len(policy_episode["policy_entropy_mean"]), 3)

    def test_pretraining_only_mode_saves_checkpoint_without_screen_evaluation(self):
        import sys
        from train_dreamerv3 import main

        class DummyEnv:
            def reset(self, **kwargs):
                return np.zeros((4, 84, 84), dtype=np.float32), {"track_id": 1, "seed": 7}
            def step(self, action):
                return np.zeros((4, 84, 84), dtype=np.float32), 1.0, True, False, {"progress": 0.1, "finished": False}
            def close(self): pass

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "pretrain"
            args = [
                "train_dreamerv3.py", "--name", "pretrain", "--run-dir", str(run_dir),
                "--total-steps", "5", "--warmup-steps", "5", "--policy-start-step", "6",
                "--skip-screen", "--batch-size", "1", "--seq-len", "2",
                "--replay-capacity", "10", "--eval-freq", "0", "--device", "cpu",
                "--eval-python", sys.executable,
            ]
            with patch("sys.argv", args), \
                 patch("train_dreamerv3.build_sampled_env", return_value=DummyEnv()), \
                 patch("train_dreamerv3.evaluate_checkpoint", side_effect=AssertionError("screen must be skipped")), \
                  patch("train_dreamerv3.check_evaluation_runtime", side_effect=AssertionError("pretraining-only skips evaluator preflight")), \
                 patch("train_dreamerv3.file_sha256", return_value="mock"):
                main()

            result = json.loads((run_dir / "pretraining-result.json").read_text())
            self.assertEqual(result["status"], "pretraining_only")
            self.assertEqual(result["screen_evaluation"], "not_run_by_design")
            self.assertEqual(result["gradient_steps"], 0)
            self.assertTrue((run_dir / "pretraining-checkpoint.pt").is_file())
            self.assertTrue((run_dir / "pretraining-actor.pt").is_file())


if __name__ == "__main__":
    unittest.main()
