import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dreamer_v3 import DreamerV3Agent, DreamerV3Config
from scripts.diagnose.collect_dreamerv3_development import load_development_cells
from scripts.diagnose.dreamerv3_open_loop import evaluate_world_model
from common_adapter import Transition
from scripts.diagnose.collect_dreamerv3_development import collect, sha256


class TestDreamerV3Development(unittest.TestCase):
    def test_development_cell_manifest_is_exact_and_unique(self):
        protocol = {
            "reserved_training_seeds": [50001, 50002, 51001],
            "partitions": {"screen": {"track_ids": [1], "seeds": [51001]}},
            "training_development": {
                "track_ids": [1, 2],
                "seeds": [50001, 50002],
                "cells": [
                    {"track_id": 1, "seed": 50001},
                    {"track_id": 2, "seed": 50002},
                ],
            },
        }
        self.assertEqual(
            load_development_cells(protocol), [(1, 50001), (2, 50002)]
        )
        protocol["training_development"]["cells"].append(
            {"track_id": 1, "seed": 50002}
        )
        with self.assertRaisesRegex(ValueError, "distinct geometry seed"):
            load_development_cells(protocol)

    def test_prior_only_gate_reads_v2_data_without_using_evaluation_cells(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protocol_path = root / "protocol.json"
            dataset_path = root / "development.npz"
            checkpoint_path = root / "checkpoint.pt"
            output_path = root / "open-loop.json"
            protocol = {
                "frame_skip": 4,
                "max_steps": 32,
                "training_track_ids": [1],
                "learner_seeds": [4],
                "reserved_training_seeds": [50001, 51001],
                "partitions": {"screen": {"track_ids": [1], "seeds": [51001]}},
                "training_development": {
                    "track_ids": [1],
                    "seeds": [50001],
                    "cells": [{"track_id": 1, "seed": 50001}],
                },
                "training_budget": {
                    "total_environment_decisions": 10,
                    "random_prefill_decisions": 0,
                    "replay_pretrain_updates": 0,
                    "policy_start_step": 11,
                    "online_updates_per_decision": 1,
                    "sequence_length": 4,
                    "burnin_steps": 2,
                    "terminal_window_fraction": 0.0,
                    "observation_loss_scale": 1.0,
                    "kl_free_nats": 1.0,
                    "overshoot_horizon": 1,
                    "overshoot_kl_weight": 0.0,
                    "overshoot_free_nats": 0.0,
                    "continue_positive_weight": 1.0,
                    "batch_size": 2,
                    "replay_capacity": 32,
                    "track_sampler_seed": 917,
                    "device": "cpu",
                },
                "world_model_gate": {
                    "context_lengths": [2],
                    "horizons": [1, 2],
                    "latent_draws": 1,
                    "anchor_stride": 2,
                    "terminal_probability_threshold": 0.5,
                    "min_positive_episodes": 1,
                    "min_negative_episodes": 1,
                    "reward_spearman_min": 0.5,
                    "terminal_recall_min": 0.8,
                    "terminal_balanced_accuracy_min": 0.8,
                    "pr_auc_multiple": 1.0,
                    "bce_relative_improvement_min": 0.1,
                    "rng_seed": 0,
                },
            }
            repo_root = Path(__file__).resolve().parents[1]
            protocol["source_sha256"] = {
                relative: sha256(repo_root / relative)
                for relative in (
                    "scripts/diagnose/collect_dreamerv3_development.py",
                    "scripts/diagnose/dreamerv3_open_loop.py",
                )
            }
            protocol_bytes = json.dumps(protocol, sort_keys=True).encode()
            protocol_path.write_bytes(protocol_bytes)
            protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()

            config = DreamerV3Config(
                device="cpu", embed_dim=64, hidden_dim=32,
                num_categoricals=4, num_classes=4, seq_len=4,
                burnin_steps=2, warmup_steps=0, batch_size=2, replay_capacity=32,
            )
            agent = DreamerV3Agent(config, seed=4)
            observations = []
            actions = []
            rewards = []
            terminal = []
            for step in range(10):
                obs = np.full((4, 84, 84), step * 10, dtype=np.uint8)
                next_obs = np.full((4, 84, 84), (step + 1) * 10, dtype=np.uint8)
                observations.append(obs)
                actions.append(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
                rewards.append(float(step % 3))
                terminal.append(step == 9)
                agent.observe(Transition(
                    observation=obs,
                    action=actions[-1],
                    reward=rewards[-1],
                    next_observation=next_obs,
                    terminated=step == 9,
                    truncated=False,
                    terminal=step == 9,
                    episode_id=0,
                    step=step,
                ))
            observations.append(next_obs)
            agent.save_checkpoint(
                checkpoint_path,
                run_metadata={
                    "protocol_sha256": protocol_sha,
                    "run_config": {
                        "seed": 4,
                        "total_steps": 10,
                        "warmup_steps": 0,
                        "replay_pretrain_updates": 0,
                        "policy_start_step": 11,
                        "updates_per_step": 1,
                        "seq_len": 4,
                        "burnin_steps": 2,
                        "terminal_window_fraction": 0.0,
                        "observation_loss_scale": 1.0,
                        "kl_free_nats": 1.0,
                        "overshoot_horizon": 1,
                        "overshoot_kl_weight": 0.0,
                        "overshoot_free_nats": 0.0,
                        "continue_positive_weight": 1.0,
                        "batch_size": 2,
                        "replay_capacity": 32,
                        "track_sampler_seed": 917,
                        "device": "cpu",
                        "track_ids": [1],
                        "excluded_training_seeds": [50001, 51001],
                    },
                },
            )
            metadata = {
                "format": "haic-dreamerv3-development-v1",
                "protocol_sha256": protocol_sha,
                "episodes": [{
                    "episode_id": 0,
                    "track_id": 1,
                    "geometry_seed": 50001,
                    "decisions": 10,
                }],
            }
            np.savez_compressed(
                dataset_path,
                metadata_json=np.asarray(json.dumps(metadata)),
                observation_offsets=np.asarray([0, 11], dtype=np.int64),
                transition_offsets=np.asarray([0, 10], dtype=np.int64),
                observations=np.stack(observations),
                actions=np.stack(actions),
                applied_actions=np.zeros((10, 3), dtype=np.float32),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminated=np.asarray(terminal, dtype=bool),
                truncated=np.zeros(10, dtype=bool),
                terminal=np.asarray(terminal, dtype=bool),
            )

            report = evaluate_world_model(
                checkpoint_path, dataset_path, protocol_path, output_path
            )
            self.assertEqual(report["format"], "haic-dreamerv3-open-loop-v2")
            self.assertIn(report["world_model_gate"], {"pass", "fail", "inconclusive"})
            self.assertEqual(report["development_episode_count"], 1)
            self.assertEqual(len(report["metrics"]), 2)
            row = report["metrics"][0]
            p = row["terminal_prevalence"]
            self.assertNotAlmostEqual(p, report["training_terminal_prevalence"])
            self.assertAlmostEqual(
                row["scored_window_fitted_constant_bce"],
                -p * np.log(p) - (1.0 - p) * np.log(1.0 - p),
            )
            self.assertNotAlmostEqual(
                row["scored_window_fitted_constant_bce"], row["constant_prevalence_bce"]
            )
            self.assertEqual(
                row["gates"]["terminal_bce"],
                row["terminal_bce"]
                <= row["scored_window_fitted_constant_bce"]
                * (1.0 - protocol["world_model_gate"]["bce_relative_improvement_min"]),
            )
            self.assertIn("retrospectively", report["interpretation"])
            self.assertTrue(output_path.is_file())

    def test_development_collector_writes_transition_aligned_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protocol_path = root / "protocol.json"
            output_path = root / "development.npz"
            protocol = {
                "frame_skip": 4,
                "max_steps": 3,
                "reserved_training_seeds": [53001, 53002],
                "partitions": {"screen": {"track_ids": [1], "seeds": [53002]}},
                "training_development": {
                    "track_ids": [1],
                    "seeds": [53001],
                    "cells": [{"track_id": 1, "seed": 53001}],
                    "collector": "uniform_native_random",
                    "collector_seed": 3,
                },
            }
            repo_root = Path(__file__).resolve().parents[1]
            protocol["source_sha256"] = {
                "scripts/diagnose/collect_dreamerv3_development.py": sha256(
                    repo_root / "scripts/diagnose/collect_dreamerv3_development.py"
                )
            }
            protocol_path.write_text(json.dumps(protocol))

            class FakeEnv:
                def __init__(self):
                    self.step_count = 0

                def reset(self, *, seed=None, options=None):
                    return np.zeros((4, 84, 84), dtype=np.float32), {
                        "track_id": 1, "seed": 53001,
                    }

                def step(self, action):
                    self.step_count += 1
                    obs = np.full((4, 84, 84), self.step_count / 4.0, dtype=np.float32)
                    return obs, 1.0, self.step_count == 3, False, {"progress": 0.1}

                def close(self):
                    pass

            with patch(
                "scripts.diagnose.collect_dreamerv3_development.build_env",
                return_value=FakeEnv(),
            ):
                collect(protocol_path, output_path)

            with np.load(output_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                self.assertEqual(archive["observations"].shape, (4, 4, 84, 84))
                self.assertEqual(archive["actions"].shape, (3, 3))
                self.assertEqual(archive["observation_offsets"].tolist(), [0, 4])
                self.assertEqual(archive["transition_offsets"].tolist(), [0, 3])
                self.assertEqual(metadata["episodes"][0]["geometry_seed"], 53001)


if __name__ == "__main__":
    unittest.main()
