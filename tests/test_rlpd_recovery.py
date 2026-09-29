"""CPU synthetic recovery loading, fixed mixtures and fresh-state lineage."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from common_adapter import ActionAdapter
from haic.algorithms.rlpd.agent import PixelRLPDAgent, RLPDConfig
from haic.algorithms.rlpd.recovery import (
    DATASET_FORMAT, FINISH_POLICY, file_sha256, frame_stack, import_learning_state,
    load_recovery_replay, sample_recovery_batch,
)
from haic.algorithms.rlpd.replay import FrameStackReplay, PixelTransition
from scripts import train_rlpd_recovery as trainer


CELL = {"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001, "obstacles": True}


def episode_arrays(n=6, *, offset=0):
    frames = np.stack([np.full((84, 84), (index + offset) % 256, np.uint8)
                       for index in range(n + 1)])
    applied = np.tile(np.array([0, .25, .75], np.float32), (n, 1))
    executed = np.stack([ActionAdapter().to_native(value) for value in applied])
    end = np.arange(n) == n - 1
    mask = (np.arange(n) >= 3) & ~end
    return {
        "frames": frames, "initial_stack": frame_stack(frames, 0), "final_stack": frame_stack(frames, n),
        "proposed_action": np.tile(np.array([0, .8, .9], np.float32), (n, 1)),
        "executed_action": executed, "applied_action": applied,
        "reward": np.arange(n, dtype=np.float32) / 10,
        "terminated": end.copy(), "truncated": np.zeros(n, np.bool_), "terminal": end.copy(),
        "episode_end": end.copy(), "step": np.arange(n, dtype=np.int32),
        "role": np.where(mask, "oracle", "actor"), "recovery_mask": mask,
    }


def prepared_arrays(*, offset=0):
    arrays = episode_arrays(85, offset=offset)
    arrays["recovery_mask"][:] = False
    arrays["recovery_mask"][3:78] = True
    arrays["role"][:] = "actor"
    arrays["role"][:3] = "prefix"
    arrays["role"][3:15] = "oracle"
    return arrays


PREPARED_METADATA = {"paired_finish_qualified": True, "training_window_start": 3,
                     "training_window_end": 78, "anchor_step": 3, "horizon": 12, "finished": True}


def write_dataset(root, arrays=None, *, metadata=None, policy=None):
    arrays = episode_arrays() if arrays is None else arrays
    path = root / "episode.npz"
    np.savez_compressed(path, **arrays)
    row = {"episode_id": 0, "path": path.name, "sha256": file_sha256(path),
           "steps": len(arrays["reward"]), "track_id": 1, "geometry_seed": CELL["geometry_seed"],
           "mode": "oracle-12", "stratum": "failure", "censored": False,
           "local_recovery_qualified": True, "accepted_transitions": int(arrays["recovery_mask"].sum())}
    row.update(metadata or {})
    manifest = root / "manifest.json"
    payload = {"format": DATASET_FORMAT, "episodes": [row]}
    if policy is not None:
        payload.update({"eligibility_policy": policy, "source_manifest_sha256": "a" * 64,
                        "source_cost_counts": {"decisions": len(arrays["reward"])},
                        "dataset_source_sha256": {"scripts/prepare_rlpd_recovery_training_data.py": "b" * 64}})
    manifest.write_text(json.dumps(payload))
    return file_sha256(manifest)


def tiny_replay(source):
    arrays = episode_arrays()
    replay = FrameStackReplay(6, seed=4, source=source)
    for step in range(6):
        replay.add(PixelTransition(
            observation=frame_stack(arrays["frames"], step),
            next_observation=frame_stack(arrays["frames"], step + 1),
            proposed_action=arrays["proposed_action"][step],
            executed_action=arrays["executed_action"][step],
            applied_action=arrays["applied_action"][step], reward=float(step),
            terminated=step == 5, truncated=False, terminal=step == 5,
            episode_id=0, step=step, track_id=1, geometry_seed=CELL["geometry_seed"],
        ))
    if source != "online":
        replay.finalize()
    return replay


class RecoveryLoaderTests(unittest.TestCase):
    def load(self, root, digest, **kwargs):
        return load_recovery_replay(root, manifest_sha256=digest, allowed_cells=[CELL], seed=19,
                                    minimum_transitions=1, minimum_geometries=1, **kwargs)

    def test_mask_only_samples_qualified_rows_with_prefix_stack_and_executed_action(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_dataset(root)
            replay, receipt = self.load(root, digest)
            self.assertEqual(len(replay), 6)
            self.assertEqual(replay.valid_count, 2)
            self.assertTrue(replay.immutable)
            self.assertEqual(receipt["manifest_sha256"], digest)
            batch = replay.sample(100)
            self.assertEqual(set(batch["step"]), {3, 4})
            arrays = episode_arrays()
            for index, step in enumerate(batch["step"]):
                np.testing.assert_array_equal(batch["observation"][index], frame_stack(arrays["frames"], step))
                np.testing.assert_array_equal(batch["next_observation"][index], frame_stack(arrays["frames"], step + 1))
                np.testing.assert_array_equal(batch["action"][index], arrays["executed_action"][step])
                self.assertFalse(np.array_equal(batch["action"][index], batch["proposed_action"][index]))
                self.assertAlmostEqual(batch["reward"][index], arrays["reward"][step])
            with self.assertRaises(RuntimeError):
                replay.add(mock.Mock(spec=PixelTransition))

    def test_time_limit_bootstraps_true_next_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = episode_arrays()
            arrays["terminated"][:] = False
            arrays["terminal"][:] = False
            arrays["truncated"][-1] = True
            arrays["recovery_mask"][-1] = True
            arrays["role"][-1] = "oracle"
            replay, _ = self.load(root, write_dataset(root, arrays))
            batch = replay.sample(100)
            rows = batch["step"] == 5
            self.assertTrue(rows.any())
            self.assertTrue(batch["truncated"][rows].all())
            self.assertFalse(batch["terminal"][rows].any())
            np.testing.assert_array_equal(batch["next_observation"][rows][0], arrays["final_stack"])

    def test_hash_cell_qualification_and_path_fail_closed(self):
        changes = [
            {"geometry_seed": 123}, {"track_id": 2}, {"local_recovery_qualified": False},
            {"mode": "actor"}, {"accepted_transitions": 99}, {"path": "../episode.npz"},
            {"sha256": "f" * 64},
        ]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load(root, write_dataset(root, metadata=change))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                self.load(root, "f" * 64)

    def test_malformed_frames_actions_flags_and_masks_rejected(self):
        for mutation in ("frames", "initial", "action", "end", "terminal", "role", "step", "flag_dtype"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                arrays = episode_arrays()
                if mutation == "frames":
                    arrays["frames"] = arrays["frames"][:-1]
                elif mutation == "initial":
                    arrays["initial_stack"][0] = 99
                elif mutation == "action":
                    arrays["executed_action"][3, 1] = .3
                elif mutation == "end":
                    arrays["episode_end"][:] = False
                elif mutation == "terminal":
                    arrays["terminal"][2] = True
                elif mutation == "role":
                    arrays["role"][3] = "prefix"
                elif mutation == "step":
                    arrays["step"][0] = 1
                elif mutation == "flag_dtype":
                    arrays["recovery_mask"] = arrays["recovery_mask"].astype(np.int32)
                with self.assertRaises(ValueError):
                    self.load(root, write_dataset(root, arrays))

    def test_default_sufficiency_gate_rejects_small_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_dataset(root)
            with self.assertRaisesRegex(ValueError, "sufficiency"):
                load_recovery_replay(root, manifest_sha256=digest, allowed_cells=[CELL], seed=0)

    def test_paired_finish_policy_samples_real_oracle_and_actor_handoff_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = prepared_arrays()
            digest = write_dataset(root, arrays, metadata=PREPARED_METADATA, policy=FINISH_POLICY)
            replay, receipt = self.load(root, digest, required_policy=FINISH_POLICY)
            self.assertEqual(replay.valid_count, 75)
            self.assertEqual(receipt["eligibility_policy"], FINISH_POLICY)
            self.assertEqual(receipt["source_manifest_sha256"], "a" * 64)
            batch = replay.sample(1000)
            self.assertEqual(set(batch["step"]), set(range(3, 78)))
            for index, step in enumerate(batch["step"]):
                np.testing.assert_array_equal(batch["action"][index], arrays["executed_action"][step])

    def test_actor_handoff_requires_exact_policy_pair_and_bounded_schedule(self):
        cases = ("legacy", "unknown_policy", "not_paired", "not_finished", "wrong_end",
                 "prefix_mask", "role_schedule", "time_limit_not_finish")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                arrays = prepared_arrays()
                metadata = dict(PREPARED_METADATA)
                policy = FINISH_POLICY
                if case == "legacy":
                    policy = None
                elif case == "unknown_policy":
                    policy = "allow-all-actor-actions"
                elif case == "not_paired":
                    metadata["paired_finish_qualified"] = False
                elif case == "not_finished":
                    metadata["finished"] = False
                elif case == "wrong_end":
                    metadata["training_window_end"] = 79
                elif case == "prefix_mask":
                    arrays["recovery_mask"][0] = True
                elif case == "role_schedule":
                    arrays["role"][16] = "oracle"
                elif case == "time_limit_not_finish":
                    arrays["terminal"][-1] = arrays["terminated"][-1] = False
                    arrays["truncated"][-1] = True
                digest = write_dataset(root, arrays, metadata=metadata, policy=policy)
                with self.assertRaises(ValueError):
                    self.load(root, digest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write_dataset(root)
            with self.assertRaisesRegex(ValueError, "policy"):
                self.load(root, digest, required_policy=FINISH_POLICY)

    def test_censored_evidence_is_verified_but_excluded_without_cap_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            arrays = episode_arrays()
            for name in ("terminated", "truncated", "terminal", "episode_end", "recovery_mask"):
                arrays[name][:] = False
            np.savez_compressed(root / "censored.npz", **arrays)
            manifest["episodes"].append({**manifest["episodes"][0], "episode_id": 1,
                                          "path": "censored.npz", "sha256": file_sha256(root / "censored.npz"),
                                          "censored": True, "local_recovery_qualified": False,
                                          "accepted_transitions": 0})
            manifest_path.write_text(json.dumps(manifest))
            replay, receipt = self.load(root, file_sha256(manifest_path))
            self.assertEqual((len(replay), replay.valid_count), (6, 2))
            self.assertEqual(receipt["stored_transitions"], 12)
            self.assertEqual(receipt["censored_episodes_excluded"], 1)
            self.assertEqual(set(replay.sample(64)["episode_id"]), {0})
            arrays["recovery_mask"][3] = True
            np.savez_compressed(root / "censored.npz", **arrays)
            manifest["episodes"][-1]["sha256"] = file_sha256(root / "censored.npz")
            manifest["episodes"][-1]["accepted_transitions"] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mask"):
                self.load(root, file_sha256(manifest_path))

    def test_duplicate_menu_rows_and_finish_controls_cannot_satisfy_failure_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for index in range(1, 4):
                path = root / f"duplicate{index}.npz"
                np.savez_compressed(path, **episode_arrays())
                manifest["episodes"].append({**manifest["episodes"][0], "episode_id": index,
                                              "path": path.name, "sha256": file_sha256(path)})
            manifest_path.write_text(json.dumps(manifest))
            replay, receipt = self.load(root, file_sha256(manifest_path))
            self.assertEqual(replay.valid_count, 8)
            self.assertEqual(receipt["unique_failure_transitions"], 2)
            with self.assertRaisesRegex(ValueError, "sufficiency"):
                load_recovery_replay(root, manifest_sha256=file_sha256(manifest_path), allowed_cells=[CELL],
                                     seed=0, minimum_transitions=3, minimum_geometries=1)
            for row in manifest["episodes"]:
                row["stratum"] = "finish-control"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sufficiency"):
                self.load(root, file_sha256(manifest_path))

    def test_default_unique_failure_three_geometry_gate_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, episode_arrays(51))
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            cells = [CELL]
            for index in range(1, 3):
                path = root / f"episode{index}.npz"
                np.savez_compressed(path, **episode_arrays(51, offset=100 * index))
                cells.append({**CELL, "geometry_seed": CELL["geometry_seed"] + index})
                manifest["episodes"].append({**manifest["episodes"][0], "episode_id": index,
                                              "path": path.name, "sha256": file_sha256(path),
                                              "geometry_seed": cells[-1]["geometry_seed"]})
            manifest_path.write_text(json.dumps(manifest))
            replay, receipt = load_recovery_replay(root, manifest_sha256=file_sha256(manifest_path),
                                                   allowed_cells=cells, seed=0)
            self.assertEqual(replay.valid_count, 141)
            self.assertEqual(receipt["unique_failure_transitions"], 141)
            self.assertEqual(receipt["failure_geometries"], 3)


class MixtureAndLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_counts_and_no_recovery_in_control(self):
        online, prior, recovery = tiny_replay("online"), tiny_replay("offline"), tiny_replay("recovery")
        for arm, counts in (("control", {"online": 32, "prior": 32}),
                            ("treatment", {"online": 32, "prior": 16, "recovery": 16})):
            batch = sample_recovery_batch(online, prior, recovery, arm=arm)
            self.assertEqual(batch["source_counts"], counts)
            self.assertEqual(batch["observation"].shape, (64, 4, 84, 84))
            self.assertEqual((batch["source"] == "online").sum(), 32)
            self.assertEqual((batch["source"] == "offline").sum(), counts["prior"])
            self.assertEqual((batch["source"] == "recovery").sum(), counts.get("recovery", 0))
        with mock.patch.object(recovery, "sample", side_effect=AssertionError("control sampled recovery")):
            sample_recovery_batch(online, prior, recovery, arm="control")
        with self.assertRaises(ValueError):
            sample_recovery_batch(online, prior, None, arm="treatment")
        with self.assertRaises(ValueError):
            sample_recovery_batch(FrameStackReplay(2, source="online"), prior, recovery, arm="treatment")
        with self.assertRaises(ValueError):
            sample_recovery_batch(online, tiny_replay("online"), recovery, arm="treatment")
        recovery.immutable = False
        with self.assertRaises(ValueError):
            sample_recovery_batch(online, prior, recovery, arm="treatment")

    def test_original_raw_reward_sac_update_and_model_optimizer_import(self):
        torch.manual_seed(5)
        agent = PixelRLPDAgent(seed=5)
        prior, online, recovery = tiny_replay("offline"), tiny_replay("online"), tiny_replay("recovery")
        batch = sample_recovery_batch(online, prior, recovery, arm="treatment")
        metrics = agent.update(batch)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["offline_samples"], 16)
        self.assertEqual(metrics["online_samples"], 32)
        agent.environment_steps = 131072
        agent.gradient_steps = 130072
        state = agent.checkpoint_state(offline_replay=prior, online_replay=online, trainer_state={})
        target = PixelRLPDAgent(seed=99)
        lineage = import_learning_state(target, state, seed=60)
        for name in ("actor", "critic", "target_critic"):
            for key, value in getattr(agent, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(target, name).state_dict()[key]))
        for name in ("actor_optimizer", "critic_optimizer", "temperature_optimizer"):
            original = getattr(agent, name).state_dict()["state"]
            restored = getattr(target, name).state_dict()["state"]
            self.assertTrue(original)
            for key, values in original.items():
                for field, value in values.items():
                    self.assertTrue(torch.equal(value, restored[key][field]))
        self.assertTrue(torch.equal(agent.log_alpha, target.log_alpha))
        self.assertEqual((target.environment_steps, target.gradient_steps), (0, 0))
        self.assertFalse(lineage["exact_continuation"])
        self.assertEqual(lineage["source_gradient_steps"], 130072)
        expected_rng = torch.Generator().manual_seed(60 + 0x524C5044)
        self.assertTrue(torch.equal(target.rng.get_state(), expected_rng.get_state()))
        bad = {**state, "config": {}}
        with self.assertRaises(ValueError):
            import_learning_state(target, bad, seed=60)


class FakeEnv:
    instances = []
    fail = False

    def __init__(self, track, seed, max_steps, skip, **kwargs):
        assert (track, seed, max_steps, skip) == (1, CELL["geometry_seed"], 2000, 4)
        assert kwargs == {"reward_shaping": False, "obstacles": True}
        self.steps = 0
        self.closed = False
        self.instances.append(self)

    def reset(self, **kwargs):
        return np.zeros((4, 84, 84), np.float32), {"track_id": 1}

    def step(self, action):
        if self.fail:
            raise RuntimeError("injected failure")
        self.steps += 1
        return (frame_stack(episode_arrays()["frames"], self.steps).astype(np.float32) / 255,
                1., self.steps == 2, False, {"track_id": 1, "finished": self.steps == 2})

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self, config, **kwargs):
        self.environment_steps = self.gradient_steps = 0

    def act(self, observation, **kwargs):
        return np.zeros(3, np.float32)

    def update(self, batch):
        assert batch["source_counts"] in ({"online": 32, "prior": 32},
                                           {"online": 32, "prior": 16, "recovery": 16})
        self.gradient_steps += 1
        return {"gradient_steps": self.gradient_steps}

    def export_actor(self, path, **kwargs):
        torch.save(kwargs, path)
        return path

    def checkpoint_state(self, *, offline_replay, online_replay, trainer_state):
        return {"trainer_state": trainer_state, "online_replay": online_replay.state_dict()}


class TrainerBoundaryTests(unittest.TestCase):
    def setUp(self):
        FakeEnv.instances = []
        FakeEnv.fail = False

    def test_import_does_not_depend_on_untracked_newhost_runner(self):
        code = """
import sys
from importlib.abc import MetaPathFinder
class BlockNewhost(MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'scripts.train_rlpd_newhost':
            raise ImportError('untracked runner is deliberately unavailable')
sys.meta_path.insert(0, BlockNewhost())
from scripts import train_rlpd_recovery
assert 'scripts.train_rlpd_newhost' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], cwd=trainer.ROOT,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_closure_uses_tracked_dependencies_and_owned_sources_only(self):
        owned = {"haic/algorithms/rlpd/recovery.py", "scripts/train_rlpd_recovery.py"}
        tracked = set(subprocess.check_output(
            ["git", "ls-files"], cwd=trainer.ROOT, text=True).splitlines())
        self.assertFalse((trainer.SOURCE_FILES - owned) - tracked)
        self.assertTrue(owned.issubset(trainer.SOURCE_FILES))
        self.assertNotIn("scripts/train_rlpd_newhost.py", trainer.SOURCE_FILES)
        self.assertNotIn("scripts/diagnose_rlpd_g0.py", trainer.SOURCE_FILES)
        self.assertNotIn("haic/algorithms/rlpd/g0_diagnostic.py", trainer.SOURCE_FILES)

    def test_inlined_historical_evidence_and_config_preserve_consumed_g0_v5_chain(self):
        v5, g0, cells = trainer.historical_evidence()
        self.assertEqual(v5["name"], "pixel-rlpd-entropy-target-ablation-v5")
        self.assertEqual(trainer.config_from_v5(v5), RLPDConfig())
        self.assertEqual(len(cells), 12)
        self.assertEqual({cell["track_id"] for cell in cells}, {1})
        self.assertEqual(g0["source_hashes"]["train.py"], file_sha256(trainer.ROOT / "train.py"))
        with tempfile.TemporaryDirectory() as directory:
            ledger = [json.loads(line) for line in (trainer.ROOT / trainer.G0_LEDGER).read_text().splitlines()]
            ledger[0]["actor_id"] = "wrong-actor-slot"
            path = Path(directory) / "ledger.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in ledger))
            original_pin = trainer.pinned

            def pin(relative, digest, prefix):
                return path if relative == trainer.G0_LEDGER else original_pin(relative, digest, prefix)

            with mock.patch.object(trainer, "pinned", side_effect=pin), \
                    self.assertRaisesRegex(ValueError, "consumed actor/road slots"):
                trainer.historical_evidence()

    def test_inlined_path_and_runtime_guards_fail_closed(self):
        for relative in ("/tmp/x", "runs/../experiments/x", "runs/./x", "experiments/p"):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                trainer.located(relative, "runs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            (root / "runs" / "link").symlink_to(root)
            with mock.patch.object(trainer, "ROOT", root), self.assertRaises(ValueError):
                trainer.located("runs/link/attempt", "runs")
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
                self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            trainer.current_runtime()
        runtime = {"torch": "2.1.0+cu121", "gpu": {
            "name": "NVIDIA GeForce RTX 4060 Ti", "compute_capability": [8, 9]}}
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(trainer, "runtime_metadata", return_value=runtime):
            self.assertEqual(trainer.current_runtime(), runtime)
            runtime["torch"] = "different-build"
            with self.assertRaises(RuntimeError):
                trainer.current_runtime()

    def test_wrong_protocol_sha_fails_before_evidence_or_env(self):
        with mock.patch.object(trainer, "pinned", side_effect=ValueError("changed hash")), \
                mock.patch.object(trainer, "historical_evidence", side_effect=AssertionError("loaded evidence")):
            with self.assertRaises(ValueError):
                trainer.preflight("experiments/test.json", "f" * 64)

    def test_source_candidate_identity_rejects_wrong_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            checkpoint.touch()
            checkpoint.with_name("candidate.json").write_text(json.dumps({"training_seed": 51}))
            with mock.patch.object(trainer, "pinned", return_value=checkpoint):
                with self.assertRaisesRegex(ValueError, "seed50"):
                    trainer.source_checkpoint()

    def test_synthetic_freeze_preflight_rejects_changed_source_and_cell_without_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, prepared_arrays(), metadata=PREPARED_METADATA, policy=FINISH_POLICY)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            cells = [CELL]
            for index in range(1, 3):
                episode = root / f"extra{index}.npz"
                np.savez_compressed(episode, **prepared_arrays(offset=100 * index))
                cells.append({**CELL, "geometry_seed": CELL["geometry_seed"] + index})
                manifest["episodes"].append({**manifest["episodes"][0], "episode_id": index,
                                              "path": episode.name, "sha256": file_sha256(episode),
                                              "geometry_seed": cells[-1]["geometry_seed"]})
            manifest_path.write_text(json.dumps(manifest))
            checkpoint = root / "source-checkpoint.pt"
            checkpoint.touch()
            protocol_path = root / "protocol.json"

            def locate(relative, prefix):
                return root if prefix == "runs" else protocol_path

            def pin(relative, digest, prefix):
                if file_sha256(protocol_path) != digest:
                    raise ValueError("hash mismatch")
                return protocol_path

            def digest(path):
                return trainer.CHECKPOINT_SHA if path == checkpoint else file_sha256(path)

            with mock.patch.object(trainer, "historical_evidence", return_value=({}, {"source_hashes": {}}, cells)), \
                    mock.patch.object(trainer, "config_from_v5", return_value=RLPDConfig()), \
                    mock.patch.object(trainer, "located", side_effect=locate), \
                    mock.patch.object(trainer, "pinned", side_effect=pin), \
                    mock.patch.object(trainer, "source_checkpoint", return_value=checkpoint), \
                    mock.patch.object(trainer, "sha256_file", side_effect=digest), \
                    mock.patch.object(trainer, "current_runtime", return_value={"synthetic": True}), \
                    mock.patch.object(trainer, "load_offline_replay", side_effect=lambda *a, **k: (
                        tiny_replay("offline"), trainer.PRIOR_DATASET_SHA, {})), \
                    mock.patch.object(trainer, "EpisodeCollector", side_effect=AssertionError("constructed collector")):
                receipt = trainer.freeze("experiments/test.json", recovery_dataset="runs/data",
                                         recovery_manifest_sha256=file_sha256(manifest_path))
                protocol, _, prior, recovery = trainer.preflight("experiments/test.json", receipt["sha256"])
                self.assertEqual(recovery.valid_count, 225)
                self.assertEqual(prior.valid_count, 6)
                self.assertEqual(protocol["online_capacity"], 8192)
                altered = json.loads(protocol_path.read_text())
                altered["cells"][0]["geometry_seed"] = 123
                protocol_path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "contract"):
                    trainer.preflight("experiments/test.json", file_sha256(protocol_path))
                altered["cells"] = cells
                altered["source_hashes"]["haic/algorithms/rlpd/recovery.py"] = "0" * 64
                protocol_path.write_text(json.dumps(altered))
                with self.assertRaisesRegex(ValueError, "source changed"):
                    trainer.preflight("experiments/test.json", file_sha256(protocol_path))

    def test_injected_loop_exact_counts_receipts_and_no_cap_terminal(self):
        for arm in ("control", "treatment"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = {"seed": 60, "cells": [CELL], "total_steps": 5,
                            "online_capacity": 5, "first_update_step": 2, "source_hashes": {},
                            "recovery_receipt": {"manifest_sha256": "a" * 64},
                            "arms": {"control": {"online": 32, "prior": 32},
                                     "treatment": {"online": 32, "prior": 16, "recovery": 16}}}
                with mock.patch.object(trainer, "import_learning_state", return_value={"exact_continuation": False}):
                    result = trainer.train_loop(root, protocol, "b" * 64, RLPDConfig(),
                                                tiny_replay("offline"), tiny_replay("recovery"), arm=arm,
                                                env_factory=FakeEnv, agent_factory=FakeAgent, learning_state={})
                self.assertEqual((result["environment_steps"], result["gradient_steps"]), (5, 4))
                self.assertEqual(result["open_episode_steps"], 1)
                self.assertFalse(result["episode_boundary"])
                rows = [json.loads(line) for line in (root / "steps.jsonl").read_text().splitlines()]
                self.assertFalse(rows[-1]["terminated"] or rows[-1]["truncated"] or rows[-1]["terminal"])
                self.assertTrue(all(env.closed for env in FakeEnv.instances))
                for name, digest in result["ledger_hashes"].items():
                    self.assertEqual(file_sha256(root / name), digest)
                metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
                self.assertTrue(all(row["source_counts"] == protocol["arms"][arm] for row in metrics))

    def test_injected_failure_preserves_reset_intent_and_closes_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            FakeEnv.fail = True
            protocol = {"seed": 60, "cells": [CELL], "total_steps": 5, "online_capacity": 5,
                        "first_update_step": 2, "arms": {"control": {"online": 32, "prior": 32}}}
            with mock.patch.object(trainer, "import_learning_state", return_value={}), self.assertRaises(RuntimeError):
                trainer.train_loop(root, protocol, "b" * 64, RLPDConfig(), tiny_replay("offline"),
                                   tiny_replay("recovery"), arm="control", env_factory=FakeEnv,
                                   agent_factory=FakeAgent, learning_state={})
            self.assertTrue(FakeEnv.instances[-1].closed)
            failure = json.loads((root / "failure.json").read_text())
            self.assertEqual(failure["completed_decisions"], 0)
            self.assertIn("reset_intent", (root / "attempts.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
