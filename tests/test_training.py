import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from train import (
    HoldoutEvaluator,
    find_vecnormalize_path,
    parse_args,
    ppo_training_kwargs,
    train_in_segments,
    validate_training_options,
)


def metrics(finish_rate, progress, lap_time_ms):
    return {
        "finish_rate": finish_rate,
        "avg_progress": progress,
        "avg_lap_time_ms": lap_time_ms,
    }


class TestHoldoutSelection(unittest.TestCase):
    def test_selection_prioritizes_finish_progress_then_lap_time(self):
        finished = HoldoutEvaluator._selection_score(metrics(0.5, 0.2, 20_000))
        no_finish = HoldoutEvaluator._selection_score(metrics(0.0, 1.0, None))
        more_progress = HoldoutEvaluator._selection_score(metrics(0.5, 0.8, 25_000))
        faster_lap = HoldoutEvaluator._selection_score(metrics(0.5, 0.8, 18_000))

        self.assertGreater(finished, no_finish)
        self.assertGreater(more_progress, finished)
        self.assertGreater(faster_lap, more_progress)


class TestBaselineDefaults(unittest.TestCase):
    def test_defaults_match_baseline_1_1(self):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(args.learning_rate, 1e-4)
        self.assertEqual(args.n_epochs, 5)
        self.assertEqual(args.clip_range, 0.2)
        self.assertEqual(args.target_kl, 0.03)
        self.assertEqual(args.save_freq, 65_536)
        self.assertEqual(args.seeds, "42,1337,2024,777")
        self.assertEqual(
            args.holdout_eval_seeds,
            "10001,10002,10003,10004,10005,10006,10007,10008",
        )

    def test_eval_compatibility_flag_is_accepted(self):
        with patch("sys.argv", ["train.py", "--eval"]):
            args = parse_args()

        self.assertTrue(args.eval)

    def test_resume_kwargs_match_recorded_training_hyperparameters(self):
        with patch("sys.argv", ["train.py"]):
            args = parse_args()

        self.assertEqual(
            ppo_training_kwargs(args),
            {
                "learning_rate": 1e-4,
                "n_steps": 2048,
                "batch_size": 256,
                "n_epochs": 5,
                "gamma": 0.99,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "target_kl": 0.03,
            },
        )

    def test_rejects_overlapping_holdout_and_reserved_save_path(self):
        with patch("sys.argv", ["train.py", "--seeds", "42"]):
            overlapping = parse_args()
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            validate_training_options(overlapping, [42], [42])

        with patch("sys.argv", ["train.py", "--save-path", "best_model"]):
            reserved_path = parse_args()
        with self.assertRaisesRegex(ValueError, "reserved"):
            validate_training_options(reserved_path, [42], [10_001])


class TestVecNormalizeResume(unittest.TestCase):
    def test_finds_best_model_vecnormalize_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.zip"
            vecnormalize = Path(directory) / "best_model_vecnormalize.pkl"
            checkpoint.touch()
            vecnormalize.touch()

            self.assertEqual(find_vecnormalize_path(checkpoint, ""), vecnormalize)


class _FakeVecNormalize:
    def save(self, path):
        Path(path).write_bytes(b"vecnormalize")


class _FakeModel:
    num_timesteps = 65_536

    def __init__(self):
        self.vecnormalize = _FakeVecNormalize()

    def save(self, path):
        Path(f"{path}.zip").write_bytes(b"model")

    def get_vec_normalize_env(self):
        return self.vecnormalize


class _SegmentModel:
    def __init__(self):
        self.num_timesteps = 0
        self.learn_calls = []

    def learn(self, total_timesteps, reset_num_timesteps):
        self.learn_calls.append((total_timesteps, reset_num_timesteps))
        self.num_timesteps += total_timesteps
        return self


class _TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 1.0, True, False, {}


def make_tiny_vec_env():
    return DummyVecEnv([_TinyEnv])


class TestHoldoutCheckpoint(unittest.TestCase):
    def test_best_model_includes_matching_vecnormalize_stats(self):
        episode = {
            "track_id": 1,
            "seed": 1,
            "steps": 100,
            "reward": 1.0,
            "progress": 0.9,
            "finished": True,
            "finish_time_s": 1.0,
            "lap_time_ms": 1_000,
        }
        with tempfile.TemporaryDirectory() as directory:
            evaluator = HoldoutEvaluator(
                [1], [42], [10_001], 2_000, 4, 1, Path(directory)
            )
            model = _FakeModel()
            with patch("train.evaluate", side_effect=[[episode], [episode]]):
                with redirect_stdout(io.StringIO()):
                    metrics = evaluator.evaluate(model)

            self.assertTrue(metrics["best_model_updated"])
            self.assertEqual(metrics["best_model_step"], 65_536)
            self.assertTrue((Path(directory) / "best_model.zip").is_file())
            self.assertTrue(
                (Path(directory) / "best_model_vecnormalize.pkl").is_file()
            )
            self.assertTrue((Path(directory) / "best_model_metrics.json").is_file())


class TestPostUpdateSegments(unittest.TestCase):
    def test_checkpoint_and_evaluation_follow_each_completed_segment(self):
        model = _SegmentModel()
        checkpoints = []
        evaluations = []

        train_in_segments(
            model,
            total_timesteps=10,
            save_freq=4,
            eval_freq=4,
            on_checkpoint=checkpoints.append,
            on_evaluate=lambda is_final: evaluations.append(
                (model.num_timesteps, is_final)
            ),
            evaluate_final=True,
        )

        self.assertEqual(model.learn_calls, [(4, False), (4, False), (2, False)])
        self.assertEqual(checkpoints, [4, 8])
        self.assertEqual(evaluations, [(4, False), (8, False), (10, True)])

    def test_throughput_mode_skips_checkpoint_and_evaluation_boundaries(self):
        model = _SegmentModel()
        checkpoints = []
        evaluations = []

        train_in_segments(
            model,
            total_timesteps=10,
            save_freq=0,
            eval_freq=0,
            on_checkpoint=checkpoints.append,
            on_evaluate=lambda is_final: evaluations.append(is_final),
            evaluate_final=False,
        )

        self.assertEqual(model.learn_calls, [(10, False)])
        self.assertEqual(checkpoints, [])
        self.assertEqual(evaluations, [])

    def test_actual_ppo_updates_before_checkpoint_hooks(self):
        env = make_tiny_vec_env()
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            verbose=0,
            device="cpu",
        )
        update_counts = []
        evaluations = []
        try:
            train_in_segments(
                model,
                total_timesteps=8,
                save_freq=4,
                eval_freq=4,
                on_checkpoint=lambda step: update_counts.append(model._n_updates),
                on_evaluate=lambda is_final: evaluations.append(
                    (model.num_timesteps, is_final)
                ),
                evaluate_final=True,
            )
        finally:
            env.close()

        self.assertEqual(update_counts, [1, 2])
        self.assertEqual(evaluations, [(4, False), (8, True)])

    def test_resume_overrides_rebuild_the_rollout_buffer(self):
        env = make_tiny_vec_env()
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=4,
            batch_size=4,
            n_epochs=10,
            verbose=0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model"
            model.save(str(checkpoint))
            resume_env = make_tiny_vec_env()
            try:
                resumed = PPO.load(
                    str(checkpoint),
                    env=resume_env,
                    learning_rate=1e-4,
                    n_steps=8,
                    batch_size=8,
                    n_epochs=5,
                    gamma=0.99,
                    clip_range=0.2,
                    ent_coef=0.0,
                    target_kl=0.03,
                )
            finally:
                resume_env.close()
        env.close()

        self.assertEqual(resumed.n_steps, 8)
        self.assertEqual(resumed.rollout_buffer.buffer_size, 8)
        self.assertEqual(resumed.batch_size, 8)
        self.assertEqual(resumed.n_epochs, 5)
        self.assertEqual(resumed.lr_schedule(1.0), 1e-4)
        self.assertEqual(resumed.clip_range(1.0), 0.2)


if __name__ == "__main__":
    unittest.main()
