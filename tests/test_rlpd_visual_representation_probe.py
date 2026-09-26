"""Synthetic archives and fake encoder only; never open G0 traces or fit a real head."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec
from scripts import probe_rlpd_visual_representation as probe


NATIVE = np.array([-0.1, 0.0, 0.5], dtype=np.float32)
OFFICIAL = ActionAdapter().to_official(NATIVE)


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.max_batch = 0
        self.inputs = []

    def forward(self, pixels):
        assert pixels.device.type == "cpu" and pixels.dtype == torch.float32
        assert pixels.ndim == 4 and pixels.shape[1:] == (4, 84, 84)
        assert len(pixels) <= 256
        self.max_batch = max(self.max_batch, len(pixels))
        self.inputs.extend(pixels[:, :, 0, 0].detach().numpy().tolist())
        return pixels[:, :, 0, 0].repeat(1, 13)[:, :50] * self.scale


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()

    def forward(self, pixels):
        feature = self.encoder(pixels)
        return torch.as_tensor(NATIVE).expand(len(pixels), -1) + 0 * feature[:, :3]


class FakeAgent:
    def __init__(self):
        self.model = FakeModel()
        self.actions_checked = 0

    def act(self, observation):
        assert observation.shape == (4, 84, 84) and observation.dtype == np.float32
        self.actions_checked += 1
        return OFFICIAL.copy()


class TestVisualRepresentationProbe(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / "runs/g0"
        (self.run / "traces").mkdir(parents=True)
        (self.root / "experiments").mkdir()
        for name in probe.CODE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic frozen source", encoding="ascii")
        self.source = {
            "protocol_path": "experiments/g0.json",
            "manifest_path": "runs/g0/manifest.json",
            "cells_path": "runs/g0/cells.jsonl",
        }
        self.actor_info = {
            "id": probe.ACTOR["id"], "path": "runs/fake-actor.pt",
            "source_sha256": "1" * 64, "export_protocol_sha256": "2" * 64,
        }
        self.actor_file = self.root / self.actor_info["path"]
        torch.save({
            "format": "haic-rlpd-pixel-actor-v1",
            "source_sha256": self.actor_info["source_sha256"],
            "protocol_sha256": self.actor_info["export_protocol_sha256"],
            "config": {"latent_dim": 50},
            "observation_spec": asdict(ObservationSpec()),
            "action_spec": asdict(ActionSpec()),
        }, self.actor_file)
        self.actor_info["sha256"] = probe.sha256_file(self.actor_file)
        self.cells = [{"geometry_seed": 4272000001 + i, "track_id": 1,
                       "partition": "TRAIN", "obstacles": True} for i in range(12)]
        self.actors = [dict(self.actor_info, action_mode="exported_tanh_mean"),
                       {"id": "other-actor", "sha256": "3" * 64,
                        "action_mode": "exported_tanh_mean"}]
        self.rows = []
        for cell in self.cells:
            for actor in self.actors:
                index = len(self.rows)
                relative = (f"traces/seed-{cell['geometry_seed']}-track-1-{actor['id']}.npz")
                # Two fit roads and one diagnostic road have pre-decision positives.
                speed = [7.0] * 36
                if index in (0, 2, 16):
                    speed[19], speed[21] = 1.0, 2.0
                np.savez_compressed(self.run / relative, **self.trace(index, speed))
                self.rows.append({
                    "partition": "TRAIN", "track_id": 1, "geometry_seed": cell["geometry_seed"],
                    "actor_id": actor["id"], "actor_sha256": actor["sha256"],
                    "road_centerline_sha256": hashlib.sha256(str(cell["geometry_seed"]).encode()).hexdigest(),
                    "trace_path": relative, "steps": 36,
                    "summary": {"outcome": "finished" if index < 6 else "off_track"},
                })
        self.freeze()
        self.agent = FakeAgent()
        self.fit_calls = 0

    @staticmethod
    def trace(index, speed):
        steps = len(speed)
        native = NATIVE if index % 2 == 0 else np.array([0.2, -0.4, 0.25], np.float32)
        official = ActionAdapter().to_official(native)
        initial = np.stack([np.full((84, 84), v, np.uint8)
                            for v in (11, 12, 13, 14)])
        next_frame = np.stack([np.full((84, 84), (index + j + 15) % 256, np.uint8)
                               for j in range(steps)])
        observation = np.concatenate((initial[-1:], next_frame[:-1]))
        terminal = np.zeros(steps, dtype=np.bool_)
        terminal[-1] = True
        return {
            "initial_stack": initial, "observation_frame": observation, "next_frame": next_frame,
            "speed_m_s": np.asarray(speed, dtype=np.float64),
            "finished": terminal if index < 6 else np.zeros(steps, dtype=np.bool_),
            "terminated": terminal, "truncated": np.zeros(steps, dtype=np.bool_),
            "proposed_native_action": np.tile(native, (steps, 1)),
            "executed_native_action": np.tile(native, (steps, 1)),
            "commanded_official_action": np.tile(official, (steps, 1)),
            "raw_official_action": np.tile(official.astype(np.float64), (steps, 1)),
        }

    def freeze(self, output_root="runs/result", freeze_path="experiments/fake-probe.json"):
        self.protocol_path = self.root / self.source["protocol_path"]
        self.protocol_path.write_text(json.dumps({
            "format": "haic-rlpd-g0-diagnostic-v1", "status": "frozen", "partition": "TRAIN",
            "frame_skip": 4, "interventions": False, "learner_updates": 0,
            "geometry_audit_sha256": "c" * 64,
            "cells": self.cells, "actors": self.actors,
        }), encoding="utf-8")
        self.source["protocol_sha256"] = probe.sha256_file(self.protocol_path)
        for row in self.rows:
            row["trace_sha256"] = probe.sha256_file(self.run / row["trace_path"])
        ledger = self.root / self.source["cells_path"]
        ledger.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")
        self.source["cells_sha256"] = probe.sha256_file(ledger)
        manifest = self.root / self.source["manifest_path"]
        manifest.write_text(json.dumps({
            "format": "haic-rlpd-g0-diagnostic-result-v1",
            "role": "TRAIN-only-failure-diagnostic", "ranked": False,
            "protocol_path": self.source["protocol_path"],
            "protocol_sha256": self.source["protocol_sha256"],
            "cells_sha256": self.source["cells_sha256"], "cell_count": 24,
            "geometry_count": 12, "geometry_audit_sha256": "c" * 64,
            "decisions_spent": sum(row["steps"] for row in self.rows),
        }), encoding="utf-8")
        self.source["manifest_sha256"] = probe.sha256_file(manifest)
        self.freeze_path = freeze_path
        rule = self.root / self.freeze_path
        rule.write_text(json.dumps(probe.protocol_template(
            source=self.source, actor=self.actor_info, code_root=self.root,
            freeze_path=self.freeze_path, output_root=output_root)), encoding="utf-8")
        self.freeze_sha = probe.sha256_file(rule)

    def replace_trace(self, index, change):
        path = self.run / self.rows[index]["trace_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        change(arrays)
        np.savez_compressed(path, **arrays)

    def fake_fit(self, features, labels):
        self.fit_calls += 1
        self.assertEqual(features.shape[1], 50)
        self.assertEqual(len(features), sum(row["steps"] - probe.BURN_IN for row in self.rows[:16]))
        linear = torch.nn.Linear(50, 1)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.zero_()
        return linear, min(10.0, max(1.0, (len(labels) - labels.sum()) / labels.sum()))

    def run_probe(self):
        return probe.probe(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                           source=self.source, actor_info=self.actor_info, code_root=self.root,
                           actor_factory=lambda _: self.agent, fit_head=self.fake_fit)

    def test_alignment_grouping_and_finished_controls_without_optimizer(self):
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("no training")):
            result = self.run_probe()
        self.assertEqual(self.fit_calls, 1)
        self.assertEqual(len(result["geometries"]), 12)
        self.assertEqual(result["coverage"]["fit_positive_geometries"], 2)
        self.assertEqual(result["coverage"]["diagnostic_positive_geometries"], 1)
        self.assertEqual(len(result["coverage"]["per_episode"]), 24)
        self.assertEqual(result["fit_decisions"], 256)
        self.assertEqual(result["fit_positive_decisions"], 4)
        self.assertEqual(result["finished_controls"], 6)
        self.assertEqual(result["finished_control_false_positives_at_0_5"], 92)
        self.assertEqual(result["finished_control_baseline_false_positives_at_0_5"], 0)
        self.assertAlmostEqual(result["fit_prevalence_baseline"], 4 / 256)
        self.assertEqual(result["geometries"][0]["partition"], "fit")
        self.assertEqual(result["geometries"][8]["partition"], "retrospective_diagnostic")
        self.assertTrue(all(len(group["episodes"]) == 2 for group in result["geometries"]))
        episode = result["geometries"][0]["episodes"][0]
        self.assertEqual(episode["decision_indices"], list(range(20, 36)))
        self.assertEqual([i for i, label in zip(episode["decision_indices"], episode["labels"]) if label], [20, 22])
        np.testing.assert_allclose(self.agent.model.encoder.inputs[0],
                                   [11 / 255, 12 / 255, 13 / 255, 14 / 255], rtol=0, atol=1e-8)
        np.testing.assert_allclose(self.agent.model.encoder.inputs[2],
                                    [13 / 255, 14 / 255, 15 / 255, 16 / 255], rtol=0, atol=1e-8)
        np.testing.assert_allclose(self.agent.model.encoder.inputs[20],
                                    [31 / 255, 32 / 255, 33 / 255, 34 / 255], rtol=0, atol=1e-8)
        self.assertEqual(episode["labels"][0], 1)
        self.assertEqual(result["integrity"]["excluded_startup_decisions_per_episode"], 20)
        self.assertEqual(self.agent.actions_checked, sum(row["steps"] for row in self.rows[::2]))
        self.assertEqual(result["actor_parameters_sha256_before"], result["actor_parameters_sha256_after"])
        self.assertEqual(result["head_role"], "diagnostic-only-not-a-policy-or-trigger")
        self.assertTrue((self.root / "runs/result/head.pt").is_file())
        self.assertEqual(result, json.loads((self.root / "runs/result/receipt.json").read_text()))
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.run_probe()

    def test_positive_coverage_gate_prevents_all_optimizer_steps_and_output(self):
        for index in (2, 16):
            self.replace_trace(index, lambda arrays: arrays["speed_m_s"].fill(7.0))
        self.freeze()
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("optimizer called")):
            with self.assertRaisesRegex(ValueError, "insufficient positive geometry coverage"):
                self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.assertFalse((self.root / "runs/result").exists())

    def test_no_diagnostic_positive_stops_before_model_load(self):
        self.replace_trace(16, lambda arrays: arrays["speed_m_s"].fill(7.0))
        self.freeze()
        with patch.object(torch, "load", side_effect=AssertionError("must gate before actor load")):
            with self.assertRaisesRegex(ValueError, "insufficient positive geometry coverage"):
                self.run_probe()
        self.assertFalse((self.root / "runs/result").exists())

    def test_preflight_reports_group_coverage_without_actor_load_or_output(self):
        with patch.object(torch, "load", side_effect=AssertionError("no actor load")):
            receipt = probe.preflight(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                                      source=self.source, actor_info=self.actor_info, code_root=self.root)
        self.assertEqual(receipt["status"], "preflight-only")
        self.assertEqual(receipt["eligible_fit_decisions"], 256)
        self.assertEqual(receipt["eligible_diagnostic_decisions"], 128)
        self.assertEqual(receipt["coverage"]["fit_positive_geometries"], 2)
        self.assertEqual(receipt["coverage"]["diagnostic_positive_geometries"], 1)
        self.assertEqual(len(receipt["coverage"]["per_episode"]), 24)
        self.assertEqual(receipt["optimizer_steps"], 0)
        self.assertFalse((self.root / "runs/result").exists())

    def test_rehashed_protocol_change_cannot_override_frozen_hyperparameters_or_split(self):
        path = self.root / self.freeze_path
        rule = json.loads(path.read_text())
        rule["head"]["updates"] = 257
        rule["split"]["fit_geometry_indices"] = list(range(2, 10))
        path.write_text(json.dumps(rule), encoding="utf-8")
        self.freeze_sha = probe.sha256_file(path)
        with self.assertRaisesRegex(ValueError, "exact source-hashed template"):
            self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.assertFalse((self.root / "runs/result").exists())

    def test_recorded_git_revision_is_informational_but_well_formed(self):
        path = self.root / self.freeze_path
        rule = json.loads(path.read_text(encoding="utf-8"))
        rule["execution"]["source_revision"] = "a" * 40
        path.write_text(json.dumps(rule), encoding="utf-8")
        self.freeze_sha = probe.sha256_file(path)
        receipt = probe.preflight(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                                  source=self.source, actor_info=self.actor_info, code_root=self.root)
        self.assertEqual(receipt["status"], "preflight-only")
        self.assertFalse((self.root / "runs/result").exists())
        rule["execution"]["source_revision"] = "not-a-git-hash"
        path.write_text(json.dumps(rule), encoding="utf-8")
        self.freeze_sha = probe.sha256_file(path)
        with self.assertRaisesRegex(ValueError, "source revision provenance"):
            probe.preflight(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                            source=self.source, actor_info=self.actor_info, code_root=self.root)

    def test_frozen_source_actor_protocol_and_trace_bytes_reject_drift(self):
        for changed in ("source", "initializer", "requirements", "actor", "protocol", "trace"):
            with self.subTest(changed=changed):
                if changed == "source":
                    path = self.root / probe.CODE_FILES[0]
                elif changed == "initializer":
                    path = self.root / "scripts/__init__.py"
                elif changed == "requirements":
                    path = self.root / "requirements.txt"
                elif changed == "actor":
                    path = self.actor_file
                elif changed == "protocol":
                    path = self.protocol_path
                else:
                    path = self.run / self.rows[-1]["trace_path"]
                original = path.read_bytes()
                path.write_bytes(original + b"drift")
                try:
                    with self.assertRaisesRegex(ValueError, "hash mismatch|exact source-hashed template"):
                        self.run_probe()
                    self.assertEqual(self.fit_calls, 0)
                    self.assertFalse((self.root / "runs/result").exists())
                finally:
                    path.write_bytes(original)

    def test_rehashed_action_pixel_or_group_drift_rejects_before_fit(self):
        self.replace_trace(23, lambda arrays: arrays["raw_official_action"].__setitem__(0, [0, 0, 0]))
        self.freeze()
        with self.assertRaisesRegex(ValueError, "action.*parity"):
            self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.replace_trace(23, lambda arrays: arrays["raw_official_action"].__setitem__(
            0, ActionAdapter().to_official(np.array([0.2, -0.4, 0.25], np.float32))))
        self.replace_trace(22, lambda arrays: arrays["observation_frame"].__setitem__(1, 255))
        self.freeze()
        with self.assertRaisesRegex(ValueError, "stack history|next frame"):
            self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.assertFalse((self.root / "runs/result").exists())

    def test_seed11_replay_must_match_frozen_action_even_when_ledger_rehashed(self):
        self.replace_trace(0, lambda arrays: (
            arrays["proposed_native_action"].__setitem__(1, [-0.2, 0, 0.5]),
            arrays["executed_native_action"].__setitem__(1, [-0.2, 0, 0.5]),
            arrays["commanded_official_action"].__setitem__(1, [-0.2, 0.5, 0.75]),
            arrays["raw_official_action"].__setitem__(1, [-0.2, 0.5, 0.75]),
        ))
        self.freeze()
        with self.assertRaisesRegex(ValueError, "seed-11 actor action"):
            self.run_probe()
        self.assertFalse((self.root / "runs/result").exists())

    def test_post_extraction_byte_or_parameter_drift_blocks_fit_and_output(self):
        original = probe._extract_episode
        source_file = self.root / probe.CODE_FILES[0]
        with patch.object(probe, "_extract_episode", wraps=original) as extract:
            def drifting(*args, **kwargs):
                result = original(*args, **kwargs)
                if extract.call_count == 1:
                    source_file.write_bytes(source_file.read_bytes() + b"late drift")
                return result
            extract.side_effect = drifting
            with self.assertRaisesRegex(ValueError, "source drifted"):
                self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.assertFalse((self.root / "runs/result").exists())
        source_file.write_bytes(b"synthetic frozen source")
        with patch.object(probe, "_extract_episode", wraps=original) as extract:
            def mutate_model(*args, **kwargs):
                result = original(*args, **kwargs)
                if extract.call_count == 1:
                    with torch.no_grad():
                        self.agent.model.encoder.scale.add_(1.0)
                return result
            extract.side_effect = mutate_model
            with self.assertRaisesRegex(ValueError, "parameters changed"):
                self.run_probe()
        self.assertEqual(self.fit_calls, 0)
        self.assertFalse((self.root / "runs/result").exists())

    def test_post_fit_source_drift_blocks_exclusive_result(self):
        def drifting_fit(features, labels):
            head, weight = self.fake_fit(features, labels)
            source_file = self.root / probe.CODE_FILES[0]
            source_file.write_bytes(source_file.read_bytes() + b"drift after fit")
            return head, weight

        with self.assertRaisesRegex(ValueError, "source drifted"):
            probe.probe(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                        source=self.source, actor_info=self.actor_info, code_root=self.root,
                        actor_factory=lambda _: self.agent, fit_head=drifting_fit)
        self.assertEqual(self.fit_calls, 1)
        self.assertTrue((self.root / "runs/result/attempt.json").is_file())
        failure = json.loads((self.root / "runs/result/failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["phase"], "scoring_geometry_groups")
        self.assertTrue(failure["no_retry_or_relabel"])
        self.assertFalse((self.root / "runs/result/receipt.json").exists())

    def test_real_head_interruption_persists_completed_update_count(self):
        actual_step = torch.optim.Adam.step
        calls = [0]

        def interrupted(optimizer, *args, **kwargs):
            if calls[0] == 1:
                raise RuntimeError("synthetic interrupted head update")
            calls[0] += 1
            return actual_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", interrupted):
            with self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
                probe.probe(self.root, self.freeze_path, self.freeze_sha, "runs/result",
                            source=self.source, actor_info=self.actor_info, code_root=self.root,
                            actor_factory=lambda _: self.agent)
        self.assertEqual(calls[0], 1)
        attempt = json.loads((self.root / "runs/result/attempt.json").read_text(encoding="utf-8"))
        failure = json.loads((self.root / "runs/result/failure.json").read_text(encoding="utf-8"))
        self.assertEqual(attempt["head_updates_planned"], 256)
        self.assertEqual(failure["head_updates_completed"], 1)
        self.assertFalse((self.root / "runs/result/receipt.json").exists())
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.run_probe()

    def test_telemetry_only_changes_labels_not_frozen_pixel_features(self):
        first = self.run_probe()
        self.replace_trace(4, lambda arrays: arrays["speed_m_s"].__setitem__(25, 1.0))
        self.freeze(output_root="runs/second-result", freeze_path="experiments/fake-probe-2.json")
        second = probe.probe(self.root, self.freeze_path, self.freeze_sha, "runs/second-result",
                             source=self.source, actor_info=self.actor_info, code_root=self.root,
                             actor_factory=lambda _: FakeAgent(), fit_head=self.fake_fit)
        self.assertEqual(first["feature_sha256"], second["feature_sha256"])
        self.assertEqual(first["geometries"][2]["episodes"][0]["logits"],
                         second["geometries"][2]["episodes"][0]["logits"])
        self.assertEqual(second["fit_positive_decisions"], first["fit_positive_decisions"] + 1)
        self.assertEqual(second["geometries"][2]["episodes"][0]["decision_indices"][6], 26)
        self.assertEqual(second["geometries"][2]["episodes"][0]["labels"][6], 1)

    def test_bounded_batch_and_only_one_open_npz_at_once(self):
        extended_speed = [7.0] * 300
        extended_speed[19], extended_speed[21] = 1.0, 2.0
        self.replace_trace(0, lambda arrays: arrays.update(self.trace(0, extended_speed)))
        self.rows[0]["steps"] = 300
        self.freeze()
        original = probe.np.load
        open_count = [0, 0]

        class Wrapped:
            def __init__(self, archive):
                self.archive = archive

            def __enter__(self):
                open_count[0] += 1
                open_count[1] = max(open_count)
                return self.archive.__enter__()

            def __exit__(self, *args):
                try:
                    return self.archive.__exit__(*args)
                finally:
                    open_count[0] -= 1

        with patch.object(probe.np, "load", side_effect=lambda *a, **kw: Wrapped(original(*a, **kw))):
            result = self.run_probe()
        self.assertEqual(open_count, [0, 1])
        self.assertEqual(self.agent.model.encoder.max_batch, 256)
        self.assertEqual(result["fit_decisions"], 520)
        self.assertEqual(result["integrity"]["bounded_stack_batch"], 256)

    def test_exactly_twenty_decisions_do_not_create_a_fictitious_eligible_input(self):
        arrays = self.trace(1, [7.0] * 20)
        arrays["speed_m_s"][19] = 1.0
        row = {"steps": 20, "actor_id": "other-actor", "summary": {"outcome": "finished"}}
        digest = hashlib.sha256()
        features = probe._extract_episode(FakeAgent(), row, arrays, digest, probe.ACTOR["id"])
        self.assertEqual(features.shape, (0, 50))
        self.assertEqual(arrays["speed_m_s"][probe.BURN_IN - 1:-1].size, 0)


if __name__ == "__main__":
    unittest.main()
