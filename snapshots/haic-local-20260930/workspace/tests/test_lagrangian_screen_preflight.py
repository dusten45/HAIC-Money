import json
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch


def _write_train_map(path: Path, map_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "map_kind": "custom",
                "map_id": map_id,
                "max_steps": 1200,
                "frame_skip": 4,
                "obstacles": [],
                "geometry": {
                    "centerline": [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
                    "width": 7.0,
                },
            }
        ),
        encoding="utf-8",
    )


class LagrangianScreenPreflightTests(unittest.TestCase):
    def test_preflight_pins_train_only_teacher_warmup_for_actor_warmstart(self):
        from training.preflight_lagrangian_screen import build_preflight_report

        report = build_preflight_report(
            site_map_split="training/maps/site/site_map_split.json",
            run_directory="artifacts/haic/ppo-teacher-warmup-route-rescue-20260925/runs",
            model_seeds=(8104,),
            teacher_warmup_epochs=30,
            teacher_max_decisions=800,
            teacher_warmup_batch_size=32,
            teacher_warmup_learning_rate=1e-3,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["loaded_groups"], ["train"])
        self.assertEqual(report["non_train_map_files_opened"], [])
        self.assertEqual(report["teacher_warmup_epochs"], 30)
        self.assertEqual(report["shared_config"]["teacher_warmup_epochs"], 30)
        self.assertEqual(report["shared_config"]["teacher_max_decisions"], 800)
        self.assertEqual(report["shared_config"]["teacher_warmup_batch_size"], 32)
        self.assertEqual(report["shared_config"]["teacher_warmup_learning_rate"], 1e-3)
        self.assertEqual(
            report["run_directory"],
            str(Path("artifacts/haic/ppo-teacher-warmup-route-rescue-20260925/runs")),
        )
        self.assertTrue(report["teacher_warmup"]["enabled"])
        self.assertFalse(report["teacher_warmup"]["teacher_loss_used_during_ppo"])
        self.assertFalse(report["teacher_warmup"]["privileged_state_in_actor_input"])
        self.assertEqual(report["initialization_mode"], "fresh_seeded_actor")
        self.assertTrue(report["paired_actor_hashes_match"])
        for runner in report["runner_commands"]:
            self.assertIn("--teacher-warmup-epochs 30", runner["command"])
            self.assertIn("--teacher-max-decisions 800", runner["command"])
            self.assertIn("--teacher-warmup-batch-size 32", runner["command"])
            self.assertIn("--teacher-warmup-learning-rate 0.001", runner["command"])
            self.assertIn(
                "--output artifacts\\haic\\ppo-teacher-warmup-route-rescue-20260925\\runs\\seed8104-",
                runner["command"],
            )
            self.assertIn("--train-only-site-map-split", runner["command"])
            self.assertIn("--defer-tune", runner["command"])

    def test_preflight_uses_fresh_actor_with_sota_runtime_architecture(self):
        from training.preflight_lagrangian_screen import build_preflight_report

        report = build_preflight_report(
            site_map_split="training/maps/site/site_map_split.json",
            model_seeds=(8104,),
        )

        self.assertTrue(report["passed"])
        self.assertFalse(report["execution_started"])
        self.assertIsNone(report["initial_checkpoint_sha256"])
        self.assertEqual(report["model_seeds"], [8104])
        self.assertEqual(
            report["actor_settings"],
            {
                "use_hud": False,
                "use_visual_features": True,
                "use_temporal_features": False,
            },
        )
        self.assertFalse(report["strict_actor_load"])
        self.assertFalse(report["source_checkpoint_optimizer_state_used"])
        self.assertEqual(report["optimizer_state_entries"], {"off": 0, "adaptive": 0})
        hashes = report["initial_actor_hashes"]
        self.assertEqual(hashes["8104"]["off"], hashes["8104"]["adaptive"])
        self.assertEqual(report["shared_config"]["throttle_expansion"], 3.5)
        self.assertEqual(report["shared_config"]["brake_expansion"], 1.0)
        self.assertEqual(report["loaded_groups"], ["train"])
        self.assertEqual(report["non_train_map_files_opened"], [])
        self.assertEqual(report["official_map_files_opened"], [])
        with self.assertRaisesRegex(ValueError, "must not initialize from checkpoint"):
            build_preflight_report(
                site_map_split="training/maps/site/site_map_split.json",
                model_seeds=(8104,),
                initialize_from=Path(
                    "artifacts/haic/site-map-official-domain-obstacle-risk-ppo8192-u8-lr2e6-seed8101/policy.pt"
                ),
            )
        for runner in report["runner_commands"]:
            self.assertNotIn("--initialize-from", runner["command"])
            self.assertIn("--disable-hud-branch", runner["command"])
            self.assertIn("--enable-visual-features", runner["command"])
            self.assertIn("--throttle-expansion 3.5", runner["command"])
            self.assertIn("--brake-expansion 1.0", runner["command"])
            self.assertIn("--train-only-site-map-split", runner["command"])
            self.assertIn("--defer-tune", runner["command"])

    def test_runner_cli_binds_matching_architecture_and_fresh_initialization(self):
        from training.preflight_lagrangian_screen import build_preflight_report
        from training.train_policy import main

        manifest = Path("training/maps/site/site_map_split.json").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight_path = root / "preflight.json"
            report = build_preflight_report(
                site_map_split=manifest,
                preflight_report_path=preflight_path,
                model_seeds=(8104,),
                teacher_warmup_epochs=30,
            )
            preflight_path.write_text(json.dumps(report), encoding="utf-8")
            argv = [
                "train_policy.py",
                "--output", str(root / "seed8104-off"),
                "--total-steps", "8192",
                "--updates", "8",
                "--max-decisions", "2000",
                "--learning-rate", "0.000002",
                "--gamma", "0.99",
                "--gae-lambda", "0.95",
                "--seed", "8104",
                "--paired-rollout-seed", "8104",
                "--teacher-warmup-epochs", "30",
                "--teacher-max-decisions", "800",
                "--teacher-warmup-batch-size", "32",
                "--teacher-warmup-learning-rate", "0.001",
                "--throttle-expansion", "3.5",
                "--brake-expansion", "1.0",
                "--speed-target", "70.0",
                "--speed-shortfall-penalty", "0.6",
                "--obstacle-brake-reward", "6.0",
                "--curve-brake-reward", "0.0",
                "--recovery-clearance-reward", "0.0",
                "--disable-recovery-action-rewards",
                "--disable-hud-branch",
                "--enable-visual-features",
                "--train-only-site-map-split", str(manifest),
                "--preflight-report", str(preflight_path),
                "--defer-tune",
                "--lagrangian-initial-lambda", "30",
                "--lagrangian-cost-budget", "0.05",
                "--lagrangian-dual-step-size", "5",
                "--lagrangian-window-size", "8",
                "--lagrangian-min-lambda", "0",
                "--lagrangian-max-lambda", "60",
                "--lagrangian-cost-gamma", "1",
                "--lagrangian-mode", "off",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch("training.train_policy.train", return_value={}) as mock_train,
                redirect_stdout(output),
            ):
                main()

        kwargs = mock_train.call_args.kwargs
        self.assertIsNone(kwargs["initialize_from"])
        self.assertFalse(kwargs["use_hud"])
        self.assertTrue(kwargs["use_visual_features"])
        self.assertFalse(kwargs["use_temporal_features"])
        self.assertEqual(kwargs["teacher_warmup_epochs"], 30)
        self.assertEqual(kwargs["teacher_max_decisions"], 800)
        self.assertEqual(kwargs["teacher_warmup_batch_size"], 32)
        self.assertEqual(kwargs["teacher_warmup_learning_rate"], 1e-3)
        self.assertEqual(
            kwargs["expected_initial_actor_sha256"],
            report["initial_actor_hashes"]["8104"]["off"],
        )
        self.assertEqual(kwargs["throttle_expansion"], 3.5)
        self.assertEqual(kwargs["brake_expansion"], 1.0)

    def test_preflight_actor_hash_matches_trainer_cpu_thread_initialization(self):
        import torch

        from haic_agent.networks import (
            CPU_INFERENCE_THREADS,
            VisualActorCritic,
        )
        from training.preflight_lagrangian_screen import _new_actor
        from training.train_policy import actor_state_sha256, set_reproducible_seed

        original_thread_count = torch.get_num_threads()
        try:
            # Exercise preflight from a fresh process's typical default even if
            # another test previously set the trainer's inference thread count.
            torch.set_num_threads(max(8, CPU_INFERENCE_THREADS + 1))
            preflight_hash = actor_state_sha256(_new_actor(8104))

            torch.set_num_threads(CPU_INFERENCE_THREADS)
            set_reproducible_seed(8104)
            trainer_actor = VisualActorCritic(
                use_hud=False,
                use_visual_features=True,
                use_temporal_features=False,
                throttle_expansion=3.5,
                brake_expansion=1.0,
            )
            self.assertEqual(preflight_hash, actor_state_sha256(trainer_actor))
        finally:
            torch.set_num_threads(original_thread_count)

    def test_trainer_cli_uses_train_only_split_and_defers_tune(self):
        from training.env_factory import split_track_seeds
        from training.train_policy import main

        train_split = split_track_seeds(train=((1, 101),), tune=(), held_out=())
        argv = [
            "train_policy.py",
            "--train-only-site-map-split",
            "maps.json",
            "--defer-tune",
            "--lagrangian-mode",
            "adaptive",
            "--total-steps",
            "8192",
            "--updates",
            "8",
            "--seed",
            "8104",
            "--paired-rollout-seed",
            "8104",
            "--teacher-warmup-epochs",
            "0",
            "--throttle-expansion",
            "3.5",
            "--disable-recovery-action-rewards",
            "--learning-rate",
            "0.000002",
        ]
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            preflight = Path(directory) / "preflight.json"
            preflight.write_text("{}", encoding="utf-8")
            argv.extend(("--preflight-report", str(preflight)))
            with (
                patch.object(sys, "argv", argv),
                patch("training.train_policy.load_train_only_site_map_split", return_value=train_split) as load_train,
                patch(
                    "training.train_policy.validate_lagrangian_screen_preflight",
                    return_value="a" * 64,
                ),
                patch("training.train_policy.train", return_value={}) as mock_train,
                redirect_stdout(output),
            ):
                main()

        load_train.assert_called_once()
        self.assertEqual(mock_train.call_args.kwargs["split"], train_split)
        self.assertFalse(mock_train.call_args.kwargs["tune_selection"])
        self.assertEqual(mock_train.call_args.kwargs["lagrangian_mode"], "adaptive")
        self.assertEqual(mock_train.call_args.kwargs["paired_rollout_seed"], 8104)
        self.assertFalse(mock_train.call_args.kwargs["enable_recovery_action_rewards"])
        self.assertEqual(mock_train.call_args.kwargs["teacher_warmup_epochs"], 0)
        self.assertEqual(mock_train.call_args.kwargs["total_steps"], 8192)
        self.assertEqual(
            mock_train.call_args.kwargs["expected_initial_actor_sha256"], "a" * 64
        )

    def test_frozen_runner_cli_validates_preflight_before_train_is_entered(self):
        from training.preflight_lagrangian_screen import build_preflight_report
        from training.train_policy import main

        manifest = Path("training/maps/site/site_map_split.json").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight_path = root / "preflight.json"
            report = build_preflight_report(
                site_map_split=manifest,
                preflight_report_path=preflight_path,
            )
            preflight_path.write_text(json.dumps(report), encoding="utf-8")
            preflight_hash = hashlib.sha256(preflight_path.read_bytes()).hexdigest()
            argv = [
                "train_policy.py",
                "--output",
                str(root / "seed8104-off"),
                "--total-steps",
                "8192",
                "--updates",
                "8",
                "--max-decisions",
                "2000",
                "--learning-rate",
                "0.000002",
                "--seed",
                "8104",
                "--paired-rollout-seed",
                "8104",
                "--teacher-warmup-epochs",
                "0",
                "--throttle-expansion",
                "3.5",
                "--brake-expansion",
                "1.0",
                "--speed-target",
                "70.0",
                "--speed-shortfall-penalty",
                "0.6",
                "--obstacle-brake-reward",
                "6.0",
                "--curve-brake-reward",
                "0.0",
                "--recovery-clearance-reward",
                "0.0",
                "--disable-recovery-action-rewards",
                "--disable-hud-branch",
                "--enable-visual-features",
                "--train-only-site-map-split",
                str(manifest),
                "--preflight-report",
                str(preflight_path),
                "--defer-tune",
                "--gamma",
                "0.99",
                "--gae-lambda",
                "0.95",
                "--lagrangian-initial-lambda",
                "30",
                "--lagrangian-cost-budget",
                "0.05",
                "--lagrangian-dual-step-size",
                "5",
                "--lagrangian-window-size",
                "8",
                "--lagrangian-min-lambda",
                "0",
                "--lagrangian-max-lambda",
                "60",
                "--lagrangian-cost-gamma",
                "1",
                "--lagrangian-mode",
                "off",
            ]
            output = io.StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch("training.train_policy.train", return_value={}) as mock_train,
                redirect_stdout(output),
            ):
                main()

        self.assertEqual(
            mock_train.call_args.kwargs["expected_initial_actor_sha256"],
            report["initial_actor_hashes"]["8104"]["off"],
        )
        self.assertEqual(
            mock_train.call_args.kwargs["preflight_report_sha256"],
            preflight_hash,
        )

    def test_preflight_loads_only_registered_train_maps_and_matches_actor_hashes(self):
        from training.preflight_lagrangian_screen import build_preflight_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_train_map(root / "train-a.json", "custom-track-haic-obstacles-20260920")
            _write_train_map(root / "train-b.json", "custom-track-haic-train-20260921")
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "train": [
                            {"map": "train-a.json", "seeds": [20260920, 20260924]},
                            {"map": "train-b.json", "seeds": [20260921, 20260925]},
                        ],
                        "tune": [{"map": "must-not-open-tune.json", "seeds": [1]}],
                        "held_out": [
                            {"map": "must-not-open-heldout.json", "seeds": [2]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            preflight_path = root / "preflight.json"

            report = build_preflight_report(
                site_map_split=manifest,
                preflight_report_path=preflight_path,
                model_seeds=(8104, 8105),
                total_steps=8192,
                updates=8,
                throttle_expansion=3.5,
                teacher_warmup_epochs=0,
                initialize_from=None,
                resume=None,
            )
            preflight_path.write_text(json.dumps(report), encoding="utf-8")
            from training.train_policy import (
                load_train_only_site_map_split,
                validate_lagrangian_screen_preflight,
            )

            expected_hash = validate_lagrangian_screen_preflight(
                preflight_path=preflight_path,
                manifest_path=manifest,
                split=load_train_only_site_map_split(manifest),
                seed=8104,
                mode="adaptive",
                shared_config=report["shared_config"],
            )

        self.assertTrue(report["passed"])
        self.assertFalse(report["execution_started"])
        self.assertEqual(report["loaded_groups"], ["train"])
        self.assertEqual(report["non_train_map_files_opened"], [])
        self.assertEqual(report["train_episode_count"], 4)
        self.assertEqual(report["actor_input"], "pixel_observation_only")
        self.assertEqual(report["teacher_warmup_epochs"], 0)
        self.assertIsNone(report["initialize_from"])
        self.assertIsNone(report["resume"])

        for seed in (8104, 8105):
            arms = report["initial_actor_hashes"][str(seed)]
            self.assertEqual(arms["off"], arms["adaptive"])
            self.assertRegex(arms["off"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            report["optimizer_state_entries"], {"off": 0, "adaptive": 0}
        )
        self.assertEqual(
            expected_hash,
            report["initial_actor_hashes"]["8104"]["adaptive"],
        )

    def test_preflight_rejects_unregistered_or_official_train_episode(self):
        from training.preflight_lagrangian_screen import build_preflight_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_train_map(root / "train.json", "custom-track-unregistered")
            manifest = root / "split.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "train": [{"map": "train.json", "seeds": [1]}],
                        "tune": [{"map": "not-opened.json", "seeds": [2]}],
                        "held_out": [{"map": "also-not-opened.json", "seeds": [3]}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "registered TRAIN cells"):
                build_preflight_report(
                    site_map_split=manifest,
                    model_seeds=(8104, 8105),
                    total_steps=8192,
                    updates=8,
                    throttle_expansion=3.5,
                    teacher_warmup_epochs=0,
                    initialize_from=None,
                    resume=None,
                )


if __name__ == "__main__":
    unittest.main()
