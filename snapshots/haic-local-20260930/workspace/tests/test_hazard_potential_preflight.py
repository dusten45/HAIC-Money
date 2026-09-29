import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class HazardPotentialPreflightTests(unittest.TestCase):
    def test_preflight_freezes_two_beta_arms_and_train_only_actor_hashes(self):
        from training.preflight_hazard_potential_screen import build_preflight_report

        report = build_preflight_report(
            site_map_split="training/maps/site/site_map_split.json",
            model_seeds=(8104,),
        )

        self.assertTrue(report["passed"])
        self.assertFalse(report["execution_started"])
        self.assertEqual(report["loaded_groups"], ["train"])
        self.assertEqual(report["non_train_map_files_opened"], [])
        self.assertEqual(report["official_map_files_opened"], [])
        self.assertEqual(report["shared_config"]["teacher_warmup_epochs"], 30)
        self.assertEqual(report["shared_config"]["use_visual_features"], True)
        self.assertEqual(report["arms"], ["beta-0.00", "beta-0.10"])
        hashes = report["initial_actor_hashes"]["8104"]
        self.assertEqual(hashes["beta-0.00"], hashes["beta-0.10"])
        self.assertEqual(report["optimizer_state_entries"], {"beta-0.00": 0, "beta-0.10": 0})
        self.assertEqual(len(report["train_cells"]), 4)
        self.assertTrue(all("--hazard-potential-preflight" in item["command"] for item in report["runner_commands"]))
        self.assertTrue(all("--lagrangian-mode off" in item["command"] for item in report["runner_commands"]))

    def test_runner_validator_rejects_changed_paired_configuration(self):
        from training.preflight_hazard_potential_screen import (
            build_preflight_report,
            validate_preflight_report,
        )
        from training.train_policy import load_train_only_site_map_split

        manifest = Path("training/maps/site/site_map_split.json").resolve()
        report = build_preflight_report(site_map_split=manifest, model_seeds=(8104,))
        split = load_train_only_site_map_split(manifest)
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "preflight.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            expected = validate_preflight_report(
                preflight_path=report_path,
                manifest_path=manifest,
                split=split,
                seed=8104,
                scale=0.10,
                shared_config=report["shared_config"],
            )
            self.assertEqual(expected, report["initial_actor_hashes"]["8104"]["beta-0.10"])

            changed_config = dict(report["shared_config"])
            changed_config["total_steps"] += 1
            with self.assertRaisesRegex(ValueError, "shared run configuration"):
                validate_preflight_report(
                    preflight_path=report_path,
                    manifest_path=manifest,
                    split=split,
                    seed=8104,
                    scale=0.10,
                    shared_config=changed_config,
                )

    def test_trainer_cli_accepts_a_matching_hazard_potential_preflight(self):
        from training.preflight_hazard_potential_screen import build_preflight_report
        from training import train_policy

        manifest = Path("training/maps/site/site_map_split.json").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "preflight.json"
            report = build_preflight_report(
                site_map_split=manifest,
                preflight_report_path=report_path,
                run_directory=root / "runs",
                model_seeds=(8104,),
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            output = root / "seed8104-beta-0.10"
            argv = [
                "train_policy",
                "--output", str(output),
                "--total-steps", "8192",
                "--updates", "8",
                "--max-decisions", "2000",
                "--learning-rate", "0.000002",
                "--gamma", "0.99",
                "--gae-lambda", "0.95",
                "--seed", "8104",
                "--paired-rollout-seed", "8104",
                "--pedal-expansion", "1.0",
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
                "--hazard-potential-scale", "0.10",
                "--train-only-site-map-split", str(manifest),
                "--hazard-potential-preflight", str(report_path),
                "--defer-tune",
                "--lagrangian-mode", "off",
            ]
            captured = {}

            def fake_train(**kwargs):
                captured.update(kwargs)
                return {"passed": True}

            with patch.object(sys, "argv", argv), patch.object(
                train_policy, "train", side_effect=fake_train
            ):
                train_policy.main()

            self.assertEqual(captured["hazard_potential_scale"], 0.10)
            self.assertEqual(
                captured["expected_initial_actor_sha256"],
                report["initial_actor_hashes"]["8104"]["beta-0.10"],
            )
            self.assertEqual(
                captured["preflight_report_sha256"],
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
            )
            self.assertFalse(captured["tune_selection"])
