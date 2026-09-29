import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch


class HazardPotentialTuneEvaluationTests(unittest.TestCase):
    def test_risk_summary_uses_aligned_visible_actor_input_transitions(self):
        from training.evaluate_hazard_potential_tune import risk_trace_summary

        records = [
            {
                "decision_trace": [
                    {
                        "visible_hazard_risk": 0.90,
                        "next_visible_hazard_risk": 0.60,
                        "collision": False,
                    },
                    {
                        "visible_hazard_risk": 0.85,
                        "next_visible_hazard_risk": 0.95,
                        "collision": True,
                    },
                    {
                        "visible_hazard_risk": 0.30,
                        "next_visible_hazard_risk": 0.10,
                        "collision": False,
                    },
                    {
                        "visible_hazard_risk": 0.10,
                        "next_visible_hazard_risk": None,
                        "collision": False,
                    },
                ]
            }
        ]

        summary = risk_trace_summary(records)

        self.assertEqual(summary["risk_transition_count"], 3)
        self.assertEqual(summary["risk_exposed_transition_count"], 3)
        self.assertAlmostEqual(summary["risk_decreased_fraction"], 2.0 / 3.0)
        self.assertEqual(summary["urgent_transition_count"], 2)
        self.assertAlmostEqual(summary["urgent_risk_decreased_fraction"], 0.5)
        self.assertAlmostEqual(summary["mean_urgent_risk_delta"], -0.10)
        self.assertEqual(summary["urgent_collision_count"], 1)

    def test_preflight_requires_all_four_fixed_budget_checkpoints_before_tune(self):
        from training.evaluate_hazard_potential_tune import (
            ARMS,
            _validate_frozen_checkpoints,
        )
        from training.preflight_hazard_potential_screen import build_preflight_report

        manifest = Path("training/maps/site/site_map_split.json").resolve()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "runs"
            report_path = root / "preflight.json"
            report = build_preflight_report(
                site_map_split=manifest,
                preflight_report_path=report_path,
                run_directory=run_directory,
            )
            report_path.write_text(json.dumps(report), encoding="utf-8")
            for arm, seed, scale in ARMS:
                checkpoint = run_directory / arm / "policy.pt"
                checkpoint.parent.mkdir(parents=True)
                torch.save(
                    {
                        "step": 8192,
                        "metadata": {
                            "updates_completed": 8,
                            "hazard_potential_scale": scale,
                            "initial_actor_sha256": report["initial_actor_hashes"][str(seed)][
                                f"beta-{scale:.2f}"
                            ],
                            "preflight_report_sha256": hashlib.sha256(
                                report_path.read_bytes()
                            ).hexdigest(),
                            "tune_selection_enabled": False,
                            "use_visual_features": True,
                        },
                    },
                    checkpoint,
                )

            checkpoints, hashes = _validate_frozen_checkpoints(
                experiment_dir=root,
                preflight_path=report_path,
                manifest_path=manifest,
            )

            self.assertEqual(len(checkpoints), 4)
            self.assertEqual(len(hashes), 4)
