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
    ActionSmoothing,
    build_sampled_env,
    CollisionPenalty,
    FinishBonus,
    HoldoutEvaluator,
    HaicTrack,
    find_vecnormalize_path,
    parse_args,
    ppo_training_kwargs,
    sample_track_seed,
    sample_worker_seeds,
    action_smoothing_from_args,
    action_control_from_args,
    train_in_segments,
    validate_action_smoothing_resume,
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

    def test_summary_reports_damage_and_termination_reasons(self):
        from train import summarize

        result = summarize([
            {"finished": True, "progress": 1.0, "reward": 10.0, "steps": 10,
             "lap_time_ms": 1_000, "damage": 0.2, "retire_reason": None},
            {"finished": False, "progress": 0.5, "reward": 5.0, "steps": 20,
             "lap_time_ms": None, "damage": 1.0, "retire_reason": "crash"},
            {"finished": False, "progress": 0.3, "reward": 3.0, "steps": 30,
             "lap_time_ms": None, "damage": 0.0, "retire_reason": None},
        ])

        self.assertEqual(result["termination_reasons"], {
            "crash": 1, "finished": 1, "time_limit": 1,
        })
        self.assertAlmostEqual(result["avg_damage"], 0.4)


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
        self.assertEqual(args.training_obstacles, "official")
        self.assertEqual(args.collision_penalty, 0.0)
        self.assertEqual(args.action_smoothing, "none")
        self.assertEqual(args.action_smoothing_alpha, 0.35)

    def test_eval_compatibility_flag_is_accepted(self):
        with patch("sys.argv", ["train.py", "--eval"]):
            args = parse_args()

        self.assertTrue(args.eval)

    def test_sampled_worker_streams_are_reproducible_and_distinct(self):
        first = sample_worker_seeds(917, 4)
        self.assertEqual(first, sample_worker_seeds(917, 4))
        self.assertEqual(len(set(first)), 4)

    def test_sampled_track_seed_skips_reserved_seed(self):
        rng = np.random.default_rng(7)
        reserved = int(rng.integers(0, 2**32, dtype=np.uint64))
        rng = np.random.default_rng(7)

        sampled = sample_track_seed(rng, frozenset((reserved,)))

        self.assertNotEqual(sampled, reserved)
        self.assertGreaterEqual(sampled, 0)
        self.assertLess(sampled, 2**32)

    def test_no_obstacle_track_reset_omits_track_id_options(self):
        class Environment:
            def __init__(self):
                self.calls = []

            def reset(self, **kwargs):
                self.calls.append(kwargs)
                return np.zeros((4, 84, 84), dtype=np.float32), {}

        track = object.__new__(HaicTrack)
        track.env = Environment()
        track.seed = 123
        track.track_id = 4
        track.obstacles = False

        track.reset()

        self.assertEqual(track.env.calls, [{"seed": 123, "options": None}])

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

    def test_rejects_invalid_collision_penalties(self):
        for value in ("-1", "nan"):
            with self.subTest(value=value):
                with patch("sys.argv", ["train.py", "--collision-penalty", value]):
                    args = parse_args()
                with self.assertRaisesRegex(ValueError, "collision-penalty"):
                    validate_training_options(args, [42], [10_001])

    def test_rejects_invalid_smoothing_alpha(self):
        for value in ("0", "nan", "1.1"):
            with self.subTest(value=value):
                with patch("sys.argv", ["train.py", "--action-smoothing-alpha", value]):
                    args = parse_args()
                with self.assertRaisesRegex(ValueError, "action-smoothing-alpha"):
                    validate_training_options(args, [42], [10_001])

    def test_steering_smoothing_config_leaves_gas_and_brake_unsmoothed(self):
        config = action_smoothing_from_args("steering-ema", 0.5)
        self.assertEqual(config["method"], "alpha")
        self.assertEqual(config["alpha"], [0.5, 1.0, 1.0])

    def test_resume_rejects_transform_change_without_explicit_flag(self):
        source = action_smoothing_from_args("none", 0.35)
        requested = action_smoothing_from_args("steering-ema", 0.35)

        with self.assertRaisesRegex(ValueError, "resumed checkpoint"):
            validate_action_smoothing_resume(source, requested, False)
        self.assertTrue(validate_action_smoothing_resume(source, requested, True))

    def test_sampled_builder_has_agent_step_time_limit(self):
        class Environment(gym.Env):
            observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

        with patch("train.SampledHaicTrack", return_value=Environment()):
            environment = build_sampled_env(
                (1,), 917, 2000, 4, True, (), True,
            )

        self.assertIsInstance(environment.env, gym.wrappers.TimeLimit)


class TestCollisionPenalty(unittest.TestCase):
    def wrapped_env(self, collision, finished=False):
        class Environment(gym.Env):
            observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

            def step(self, action):
                return np.zeros(1, dtype=np.float32), 10.0, False, False, {
                    "collision": collision,
                    "finished": finished,
                }

        return Environment()

    def test_penalty_applies_once_to_aggregated_collision(self):
        wrapper = CollisionPenalty(self.wrapped_env(collision=True), 5.0)

        _observation, reward, _terminated, _truncated, _info = wrapper.step([0.0])

        self.assertEqual(reward, 5.0)

    def test_penalty_preserves_non_collision_and_finish_bonus(self):
        no_collision = CollisionPenalty(self.wrapped_env(collision=False), 5.0)
        _observation, reward, _terminated, _truncated, _info = no_collision.step([0.0])
        self.assertEqual(reward, 10.0)

        finished = FinishBonus(CollisionPenalty(self.wrapped_env(collision=True, finished=True), 5.0))
        _observation, reward, _terminated, _truncated, _info = finished.step([0.0])
        self.assertEqual(reward, 105.0)


class TestActionSmoothingWrapper(unittest.TestCase):
    class Environment(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
        )

        def __init__(self):
            self.actions = []

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32))
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

    def test_wrapper_smooths_once_per_step_and_resets_episode_state(self):
        environment = self.Environment()
        wrapper = ActionSmoothing(
            environment,
            action_smoothing_from_args("steering-ema", 0.5),
        )
        wrapper.reset()
        wrapper.step([1.0, 1.0, 0.0])
        wrapper.step([-1.0, 1.0, 0.0])
        np.testing.assert_allclose(environment.actions[0], [0.5, 1.0, 0.0])
        np.testing.assert_allclose(environment.actions[1], [-0.25, 1.0, 0.0])

        wrapper.reset()
        wrapper.step([1.0, 1.0, 0.0])
        np.testing.assert_allclose(environment.actions[2], [0.5, 1.0, 0.0])

    def test_wrapper_exposes_previous_steering_plane(self):
        class Environment(gym.Env):
            observation_space = gym.spaces.Box(0, 1, shape=(4, 84, 84), dtype=np.float32)
            action_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros((4, 84, 84), dtype=np.float32), {}

            def step(self, action):
                return np.zeros((4, 84, 84), dtype=np.float32), 0.0, False, False, {}

        wrapper = ActionSmoothing(
            Environment(),
            action_smoothing_from_args("steering-ema", 0.35),
            action_control_from_args("previous-steering-plane"),
        )
        observation, _info = wrapper.reset()
        self.assertEqual(observation.shape, (5, 84, 84))
        self.assertTrue(np.all(observation[4] == 0.5))

        observation, _reward, _terminated, _truncated, _info = wrapper.step([1.0, 1.0, 0.0])
        self.assertAlmostEqual(wrapper.last_action[0], 0.35)
        self.assertTrue(np.all(observation[4] == 0.675))


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
            with patch("train.PPO.load", return_value=model):
                with patch("train.evaluate", side_effect=[[episode]] * 4):
                    with redirect_stdout(io.StringIO()):
                        metrics = evaluator.evaluate(model)

            self.assertTrue(metrics["best_model_updated"])
            self.assertEqual(metrics["best_model_step"], 65_536)
            self.assertTrue((Path(directory) / "best_model.zip").is_file())
            self.assertTrue(
                (Path(directory) / "best_model_vecnormalize.pkl").is_file()
            )
            self.assertTrue((Path(directory) / "best_model_metrics.json").is_file())
            self.assertTrue(metrics["cpu_reload_seen_matches_selection"])
            self.assertTrue(metrics["cpu_reload_holdout_matches_selection"])
            self.assertEqual(len(metrics["seen_episodes"]), 1)
            self.assertEqual(len(metrics["holdout_episodes"]), 1)


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
