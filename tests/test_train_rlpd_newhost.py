"""No-environment-reset and injected-fake checks for the new-host RLPD lane."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig, sample_balanced_batch
from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition
from scripts import train_rlpd_newhost as trainer
from scripts.rlpd_common import sha256_file


def tiny_replay(source: str) -> FrameStackReplay:
    replay = FrameStackReplay(16, seed=17, source=source)
    for index in range(5):
        stack = np.full((4, 84, 84), index, dtype=np.uint8)
        replay.add(PixelTransition(
            observation=stack, proposed_action=np.zeros(3, dtype=np.float32),
            executed_action=np.zeros(3, dtype=np.float32),
            applied_action=np.array([0, .5, .5], dtype=np.float32), reward=float(index) / 10,
            next_observation=np.full_like(stack, index + 1), terminated=index == 4,
            truncated=False, terminal=index == 4, episode_id=0, step=index,
            track_id=1, geometry_seed=4272000001,
        ))
    if source == "offline":
        replay.finalize()
    return replay


class FakeEnv:
    instances: list["FakeEnv"] = []
    fail_second_step = False
    intent_path: Path | None = None

    def __init__(self, track_id, geometry_seed, max_steps, frame_skip, *, reward_shaping, obstacles):
        assert (track_id, max_steps, frame_skip, reward_shaping, obstacles) == (1, 2000, 4, False, True)
        assert 4272000001 <= geometry_seed <= 4272000012
        self.geometry_seed = geometry_seed
        self.steps = 0
        self.closed = False
        self.__class__.instances.append(self)

    def reset(self, *, seed=None, options=None):
        assert self.steps == 0
        assert self.intent_path is not None
        intent = jsonl(self.intent_path)[-1]
        assert intent["event"] == "reset_intent" and intent["cell"]["geometry_seed"] == self.geometry_seed
        return np.zeros((4, 84, 84), np.float32), {"track_id": 1, "seed": self.geometry_seed}

    def step(self, action):
        if self.fail_second_step and self.steps == 1:
            raise RuntimeError("injected step failure")
        assert len(action) == 3 and np.isfinite(action).all()
        assert -1 <= action[0] <= 1 and 0 <= action[1] <= 1 and 0 <= action[2] <= 1
        self.steps += 1
        return (np.full((4, 84, 84), self.steps / 255, np.float32),
                1.0, self.steps == 2, False,
                {"track_id": 1, "finished": self.steps == 2, "progress": .95, "damage": 0.0})

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self, config, *, seed, device):
        assert device == "cuda" and config.batch_size == 64
        self.device = torch.device("cpu")
        self.environment_steps = 0
        self.gradient_steps = 0
        self.mixes: list[tuple[int, int]] = []

    def act_with_details(self, observation, *, deterministic):
        assert not deterministic and observation.shape == (4, 84, 84)
        return np.zeros(3, np.float32), None, None

    def update(self, batch):
        mix = (int((batch["source"] == "offline").sum()),
               int((batch["source"] == "online").sum()))
        assert mix == (32, 32) and len(batch["observation"]) == 64
        self.mixes.append(mix)
        self.gradient_steps += 1
        return {"offline_samples": mix[0], "online_samples": mix[1],
                "gradient_steps": self.gradient_steps, "actor_loss": 0.2, "critic_loss": 0.3}

    def export_actor(self, path, **kwargs):
        assert len(kwargs["protocol_sha256"]) == 64
        torch.save({"actor": self.gradient_steps, "contract": kwargs["environment_contract"]}, path)
        return path

    def checkpoint_state(self, *, offline_replay, online_replay, trainer_state):
        assert offline_replay.immutable and online_replay.source == "online"
        return {"actor": self.gradient_steps, "critic": self.gradient_steps,
                "online_size": len(online_replay), "trainer_state": trainer_state}


def fake_contract():
    return {"seed": 52, "cells": [{"track_id": 1, "geometry_seed": 4272000001 + index,
                                   "partition": "TRAIN", "obstacles": True} for index in range(12)],
            "total_steps": 5, "candidate_steps": [3, 5], "first_update_step": 1,
            "policy_takeover_step": 2, "replay_capacity": 16, "source_hashes": {}}


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class NewHostBoundaryTests(unittest.TestCase):
    def test_original_protocol_and_g0_ledger_are_pinned_without_reset(self):
        v5, g0, cells = trainer.historical_evidence()
        self.assertEqual(v5["name"], "pixel-rlpd-entropy-target-ablation-v5")
        self.assertEqual(len(cells), 12)
        self.assertEqual({cell["track_id"] for cell in cells}, {1})
        self.assertEqual(g0["source_hashes"]["train.py"], sha256_file(trainer.ROOT / "train.py"))
        self.assertEqual(trainer.config_from_v5(v5), RLPDConfig())

    def test_wrong_protocol_hash_fails_before_dataset_or_environment(self):
        with mock.patch.object(trainer, "historical_evidence", side_effect=AssertionError("read data")):
            with self.assertRaises(ValueError):
                trainer.preflight(trainer.V5_PROTOCOL, "f" * 64)

    def test_freeze_preflight_original_prior_roundtrip_rejects_forged_cells_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory) / "newhost.json"
            original_located = trainer.located

            def routed(relative, prefix):
                if relative == "experiments/newhost.json" and prefix == "experiments":
                    return study
                return original_located(relative, prefix)

            with mock.patch.object(trainer, "located", side_effect=routed), \
                 mock.patch.object(trainer, "build_env", side_effect=AssertionError("unexpected env")):
                receipt = trainer.freeze("experiments/newhost.json", seed=52, steps=131072)
                self.assertEqual(receipt["sha256"], sha256_file(study))
                protocol, config, offline = trainer.preflight("experiments/newhost.json", receipt["sha256"])
                self.assertEqual((len(offline), offline.valid_count), (15915, 15915))
                self.assertEqual((config.batch_size, protocol["offline_batch_size"]), (64, 32))
                self.assertEqual(protocol["candidate_steps"], [65536, 131072])
                self.assertEqual(protocol["source_hashes"]["scripts/train_rlpd_newhost.py"],
                                 sha256_file(trainer.ROOT / "scripts/train_rlpd_newhost.py"))
                altered = json.loads(study.read_text())
                altered["cells"][0]["geometry_seed"] = 4000033021
                study.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "contract"):
                    trainer.preflight("experiments/newhost.json", sha256_file(study))
                altered["cells"] = protocol["cells"]
                altered["source_hashes"]["scripts/train_rlpd_newhost.py"] = "0" * 64
                study.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "source drifted"):
                    trainer.preflight("experiments/newhost.json", sha256_file(study))

    def test_relative_paths_reject_traversal_and_symlinks(self):
        for path in ("/tmp/x", "runs/../experiments/x", "runs/./x", "experiments/p"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                trainer.located(path, "runs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "runs" / "link").symlink_to(root)
            with mock.patch.object(trainer, "ROOT", root), self.assertRaises(ValueError):
                trainer.located("runs/link/attempt", "runs")

    def test_g0_wrong_actor_slot_or_protected_road_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("experiments", "runs"):
                (root / folder).mkdir()
            (root / trainer.PRIOR_DIR).mkdir(parents=True)
            v5_path = root / trainer.V5_PROTOCOL
            v5_path.write_text("{}")
            prior_path = root / trainer.PRIOR_DIR / "manifest.json"
            prior_path.write_text(json.dumps({"study_protocol_sha256": sha256_file(v5_path),
                                              "dataset_sha256": trainer.PRIOR_DATASET_SHA,
                                              "study_id": "synthetic-v5"}))
            cells = [{"track_id": 1, "geometry_seed": 4272000001 + i,
                      "partition": "TRAIN", "obstacles": True} for i in range(12)]
            actors = [{"id": "a", "sha256": "a" * 64}, {"id": "b", "sha256": "b" * 64}]
            g0_path = root / trainer.G0_PROTOCOL
            g0_path.write_text(json.dumps({"format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen",
                                           "partition": "TRAIN", "frame_skip": 4, "max_steps": 2000,
                                           "cells": cells, "actors": actors, "source_hashes": {}}))
            ledger = root / trainer.G0_LEDGER
            ledger.parent.mkdir(parents=True)
            rows = [{"track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                     "partition": "TRAIN", "actor_id": actor["id"], "actor_sha256": actor["sha256"],
                     "steps": 2} for cell in cells for actor in actors]
            g0_manifest = root / trainer.G0_MANIFEST
            with mock.patch.object(trainer, "ROOT", root), \
                 mock.patch.object(trainer, "V5_SHA", sha256_file(v5_path)), \
                 mock.patch.object(trainer, "PRIOR_MANIFEST_SHA", sha256_file(prior_path)), \
                 mock.patch.object(trainer, "G0_SHA", sha256_file(g0_path)), \
                 mock.patch.object(trainer, "read_entropy_v5_protocol", return_value={"name": "synthetic-v5"}):
                for kind in ("actor", "protected"):
                    changed = [dict(item) for item in rows]
                    if kind == "actor":
                        changed[1]["actor_id"] = "a"
                    else:
                        changed[1]["geometry_seed"] = 4000033021
                    ledger.write_text("".join(json.dumps(row) + "\n" for row in changed))
                    g0_manifest.write_text(json.dumps({"protocol_sha256": sha256_file(g0_path),
                                                       "cells_sha256": sha256_file(ledger), "cell_count": 24}))
                    with mock.patch.object(trainer, "G0_LEDGER_SHA", sha256_file(ledger)), \
                         mock.patch.object(trainer, "G0_MANIFEST_SHA", sha256_file(g0_manifest)):
                        with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "ledger"):
                            trainer.historical_evidence()

    def test_fake_run_journals_before_every_reset_and_trains_32_32(self):
        FakeEnv.instances = []
        FakeEnv.fail_second_step = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "experiments").mkdir()
            study = root / "experiments" / "fake.json"
            study.write_text("{}")
            digest = sha256_file(study)
            FakeEnv.intent_path = root / "runs" / "fake-online" / "attempts.jsonl"
            contract = fake_contract()
            with mock.patch.object(trainer, "ROOT", root), \
                 mock.patch.object(trainer, "preflight", return_value=(contract, RLPDConfig(), tiny_replay("offline"))), \
                 mock.patch.object(trainer, "build_env", FakeEnv), \
                 mock.patch.object(trainer, "PixelRLPDAgent", FakeAgent):
                result = trainer.run("experiments/fake.json", digest, "runs/fake-online")
                with self.assertRaises(FileExistsError):
                    trainer.run("experiments/fake.json", digest, "runs/fake-online")
            run = root / "runs" / "fake-online"
            attempts = jsonl(run / "attempts.jsonl")
            steps = jsonl(run / "steps.jsonl")
            metrics = jsonl(run / "metrics.jsonl")
            self.assertEqual((result["environment_steps"], result["gradient_steps"]), (5, 4))
            self.assertEqual((result["completed_episodes"], result["open_episode_steps"]), (2, 1))
            self.assertEqual([row["event"] for row in attempts],
                             ["reset_intent", "reset", "end", "reset_intent", "reset",
                              "checkpoint", "end", "reset_intent", "reset", "checkpoint"])
            self.assertEqual(len([row for row in attempts if row["event"] == "reset_intent"]), 3)
            self.assertEqual(len(FakeEnv.instances), 3)
            self.assertTrue(all(env.closed for env in FakeEnv.instances))
            self.assertEqual([row["global_step"] for row in steps], [1, 2, 3, 4, 5])
            self.assertEqual([row["behavior"] for row in steps], ["random", "random", "policy", "policy", "policy"])
            self.assertEqual([row["offline_samples"] for row in metrics], [32, 32])
            self.assertEqual([row["online_samples"] for row in metrics], [32, 32])
            self.assertEqual([row["environment_steps"] for row in result["candidates"]], [3, 5])
            self.assertEqual(result["steps_sha256"], sha256_file(run / "steps.jsonl"))

    def test_fake_step_failure_keeps_attempt_and_partial_receipt(self):
        FakeEnv.instances = []
        FakeEnv.fail_second_step = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "experiments").mkdir()
            study = root / "experiments" / "fake.json"
            study.write_text("{}")
            FakeEnv.intent_path = root / "runs" / "failing-online" / "attempts.jsonl"
            with mock.patch.object(trainer, "ROOT", root), \
                 mock.patch.object(trainer, "preflight", return_value=(fake_contract(), RLPDConfig(), tiny_replay("offline"))), \
                 mock.patch.object(trainer, "build_env", FakeEnv), \
                 mock.patch.object(trainer, "PixelRLPDAgent", FakeAgent):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    trainer.run("experiments/fake.json", sha256_file(study), "runs/failing-online")
            run = root / "runs" / "failing-online"
            self.assertEqual(json.loads((run / "failure.json").read_text())["completed_decisions"], 1)
            self.assertEqual(len(jsonl(run / "steps.jsonl")), 1)
            self.assertEqual([r["event"] for r in jsonl(run / "attempts.jsonl")], ["reset_intent", "reset"])
            self.assertFalse((run / "result.json").exists())
            self.assertTrue(all(env.closed for env in FakeEnv.instances))
        FakeEnv.fail_second_step = False

    def test_checkpoint_failure_preserves_consumed_steps_without_candidate(self):
        FakeEnv.instances = []
        FakeEnv.fail_second_step = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "experiments").mkdir()
            study = root / "experiments" / "fake.json"
            study.write_text("{}")
            FakeEnv.intent_path = root / "runs" / "bad-checkpoint" / "attempts.jsonl"
            with mock.patch.object(trainer, "ROOT", root), \
                 mock.patch.object(trainer, "preflight", return_value=(fake_contract(), RLPDConfig(), tiny_replay("offline"))), \
                 mock.patch.object(trainer, "build_env", FakeEnv), \
                 mock.patch.object(trainer, "PixelRLPDAgent", FakeAgent), \
                 mock.patch.object(trainer, "save_checkpoint", side_effect=OSError("checkpoint full")):
                with self.assertRaisesRegex(OSError, "checkpoint full"):
                    trainer.run("experiments/fake.json", sha256_file(study), "runs/bad-checkpoint")
            run = root / "runs" / "bad-checkpoint"
            self.assertEqual(json.loads((run / "failure.json").read_text())["completed_decisions"], 3)
            self.assertEqual(len(jsonl(run / "steps.jsonl")), 3)
            self.assertEqual(len([row for row in jsonl(run / "attempts.jsonl") if row["event"] == "reset_intent"]), 2)
            self.assertFalse((run / "result.json").exists())
            self.assertTrue(all(env.closed for env in FakeEnv.instances))

    def test_agent_setup_failure_is_recorded_before_any_reset(self):
        FakeEnv.instances = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "experiments").mkdir()
            study = root / "experiments" / "fake.json"
            study.write_text("{}")
            with mock.patch.object(trainer, "ROOT", root), \
                 mock.patch.object(trainer, "preflight", return_value=(fake_contract(), RLPDConfig(), tiny_replay("offline"))), \
                 mock.patch.object(trainer, "PixelRLPDAgent", side_effect=RuntimeError("GPU unavailable")):
                with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                    trainer.run("experiments/fake.json", sha256_file(study), "runs/setup-failed")
            run = root / "runs" / "setup-failed"
            self.assertEqual(json.loads((run / "failure.json").read_text())["status"],
                             "pre-reset-or-unjournaled-failure")
            self.assertTrue((run / "attempt.json").is_file())
            self.assertEqual(FakeEnv.instances, [])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_real_gpu_actor_and_critic_update_from_synthetic_32_32(self):
        # No build_env, collector or environment reset occurs in this test.
        trainer.seed_everything(52)
        agent = PixelRLPDAgent(RLPDConfig(), seed=52, device="cuda")
        actor_before = agent.actor.mean.weight.detach().clone()
        critic_before = agent.critic.q_heads.state_dict()["0.0.weight"].detach().clone()
        batch = sample_balanced_batch(tiny_replay("offline"), tiny_replay("online"),
                                      batch_size=64, offline_count=32)
        metrics = agent.update(batch)
        self.assertEqual((metrics["offline_samples"], metrics["online_samples"], agent.gradient_steps), (32, 32, 1))
        self.assertTrue(all(math.isfinite(float(value)) for value in metrics.values()))
        self.assertGreater(torch.max(torch.abs(actor_before - agent.actor.mean.weight)).item(), 0)
        self.assertGreater(torch.max(torch.abs(critic_before - agent.critic.q_heads.state_dict()["0.0.weight"])).item(), 0)


if __name__ == "__main__":
    unittest.main()
