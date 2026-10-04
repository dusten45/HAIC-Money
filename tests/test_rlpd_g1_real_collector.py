"""Isolated real-entry G1 tests: fake actors and fake environments only."""

from __future__ import annotations

from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from common_adapter import ActionSpec, ObservationSpec
import scripts.diagnose_rlpd_g1_coverage as g1
from scripts.audit_rlpd_g1_coverage_seeds import _walk_sources
from tests import test_rlpd_g1_collector as fixture_module


class FakeActor:
    def __init__(self, path: Path):
        self.path = path

    def reset(self, observation):
        pass

    def act(self, observation):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def model(self, observation):
        return torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float32)


class TestG1RealCollector(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.TestG1CollectorSkeleton(
            "test_real_entry_point_refuses_all_injections_without_creating_env")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.root = fixture.root
        self.created = 0
        self.calls = 0
        self.output = "runs/synthetic"
        fixture.protocol["study_id"] = "g1-synthetic-new-host"
        fixture.protocol["budget"]["max_core_hours"] = None
        for path in fixture.claim_paths:
            claim = json.loads((self.root / path).read_text())
            claim.update(study_id="g1-synthetic-new-host", protocol_id="g1-synthetic-new-host")
            fixture.write_json(path, claim)
        claim_sources = [{"path": path, "sha256": hashlib.sha256((self.root / path).read_bytes()).hexdigest()}
                         for path in fixture.claim_paths]
        fixture.claim_sha = g1._digest(sorted(claim_sources, key=lambda item: item["path"]))
        fixture.protocol["train_claims_sha256"] = fixture.claim_sha
        fixture.protocol_sha = fixture.write_json(fixture.protocol_path, fixture.protocol)
        fixture.audit.update(train_claims_sha256=fixture.claim_sha)
        fixture.audit_sha = fixture.write_json(fixture.audit_path, fixture.audit)

    def context(self):
        def auditor(seed_start, **kwargs):
            self.assertEqual(seed_start, self.fixture.cells[0]["geometry_seed"])
            self.assertEqual(kwargs["self_study_id"], self.fixture.protocol["study_id"])
            self.assertEqual(kwargs["self_protocol_sha256"], self.fixture.protocol_sha)
            self.assertEqual(kwargs["self_audit_sha256"], self.fixture.audit_sha)
            return copy.deepcopy(self.fixture.audit)

        return g1.preflight(self.root, self.fixture.protocol_path, self.fixture.protocol_sha,
                            self.fixture.audit_path, self.fixture.audit_sha, auditor=auditor)

    def payload(self, path):
        info = next(entry for entry in self.fixture.protocol["source_actors"]
                    if self.root / entry["path"] == path)
        return {"format": "haic-rlpd-pixel-actor-v1", "source_sha256": info["source_sha256"],
                "protocol_sha256": info["export_protocol_sha256"],
                "observation_spec": asdict(ObservationSpec()), "action_spec": asdict(ActionSpec())}

    def env(self, **kwargs):
        slot = self.calls - 1
        attempt = self.root / self.output / "attempts" / f"slot-{slot:02d}.json"
        self.assertTrue(attempt.is_file())
        self.assertEqual(json.loads(attempt.read_text())["cell"]["geometry_seed"], kwargs["seed"])
        rows = json.loads((self.root / self.output / "slots.json").read_text())
        self.assertEqual(rows[slot]["status"], "reserved_attempt")
        self.assertEqual(kwargs["max_steps"], 2000)
        self.assertEqual(kwargs["frame_skip"], 4)
        self.assertIs(kwargs["reward_shaping"], False)
        self.assertIs(kwargs["obstacles"], True)
        self.assertEqual(kwargs["collision_penalty"], 0.0)
        self.created += 1
        return object()

    def cell(self, *, row, actor, actor_info, actor_payload, rules,
             centerline_far_threshold_m, env_factory, attempt):
        self.calls += 1
        self.assertIsInstance(actor, FakeActor)
        self.assertEqual(actor_payload, self.fixture.actors[actor_info["id"]])
        self.assertEqual(rules.max_decisions, 2000)
        self.assertEqual(rules.negative_reward_limit, 100)
        self.assertEqual(centerline_far_threshold_m, 10.0)
        env_factory(track_id=row["track_id"], seed=row["geometry_seed"], max_steps=2000,
                    frame_skip=4, reward_shaping=False, obstacles=True, collision_penalty=0.0)
        attempt.update(phase="complete", decisions_completed=3, decision_calls=3,
                       raw_counters={"resets": 1, "reset_initial_raw_frames": 1,
                                     "reset_noop_raw_frames": 50, "driven_raw_frames": 12})
        return ({"partition": "TRAIN", "track_id": row["track_id"],
                 "geometry_seed": row["geometry_seed"], "actor_id": actor_info["id"],
                 "actor_sha256": actor_info["sha256"], "steps": 3,
                 "reset_initial_raw_frames": 1, "reset_noop_raw_frames": 50,
                 "driven_raw_frames": 12,
                 "road_centerline_sha256": hashlib.sha256(str(row["geometry_seed"]).encode()).hexdigest(),
                 "summary": {"outcome": "finished"}},
                {"initial_stack": np.zeros((4, 84, 84), dtype=np.uint8),
                 "finished": np.array([False, False, True]),
                 "terminated": np.array([False, False, False]),
                 "truncated": np.array([False, False, True])})

    def collect(self, *, runner=None, loader=None):
        return g1.collect(self.context(), self.output, env_factory=self.env,
                          run_cell=runner or self.cell, actor_factory=FakeActor,
                          payload_loader=loader or self.payload, clock=lambda: 0.0)

    def test_legacy_four_hour_contract_stays_blocked(self):
        self.fixture.protocol["budget"]["max_core_hours"] = 4.0
        self.fixture.protocol_sha = self.fixture.write_json(self.fixture.protocol_path, self.fixture.protocol)
        result = g1.collect(self.context(), self.output, env_factory=self.env, run_cell=self.cell)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(self.created, 0)
        self.assertFalse((self.root / self.output).exists())

    def test_uncapped_protocol_must_be_distinct_new_host(self):
        self.fixture.protocol["study_id"] = "g1-synthetic"
        self.fixture.protocol_sha = self.fixture.write_json(self.fixture.protocol_path, self.fixture.protocol)
        with self.assertRaisesRegex(ValueError, "new-host"):
            self.context()
        self.assertEqual(self.created, 0)

    def test_actor_metadata_and_source_drift_fail_before_output(self):
        with self.assertRaisesRegex(ValueError, "actor export metadata"):
            self.collect(loader=lambda path: {"format": "wrong"})
        self.assertEqual(self.created, 0)
        self.assertFalse((self.root / self.output).exists())
        self.fixture.write("agent.py", b"changed actor implementation")
        with self.assertRaisesRegex(ValueError, "source/hash drift"):
            self.context()
        self.assertEqual(self.created, 0)

    def test_unrelated_unknown_seed_warning_passes_but_exact_candidate_warning_blocks(self):
        warning = {"path": "experiments/drqv2-geometry-augmentation-v1.json",
                   "field": "geometry_audit.known_excluded_geometry_seed_count",
                   "reason": "unknown seed-bearing field"}
        self.fixture.audit["provenance_warnings"] = [warning]
        self.fixture.audit_sha = self.fixture.write_json(self.fixture.audit_path, self.fixture.audit)
        self.context()
        for field in ("path", "field", "reason"):
            with self.subTest(field=field):
                changed = copy.deepcopy(warning)
                changed[field] += f" seed-{self.fixture.cells[0]['geometry_seed']}"
                self.fixture.audit["provenance_warnings"] = [changed]
                self.fixture.audit_sha = self.fixture.write_json(self.fixture.audit_path, self.fixture.audit)
                with self.assertRaisesRegex(ValueError, "candidate audit"):
                    self.context()
                self.assertEqual(self.created, 0)
        self.assertFalse((self.root / self.output).exists())

    def test_full_real_entry_journals_ordered_traces_and_no_image_verdict(self):
        result = self.collect()
        self.assertEqual(result["status"], "collected_unreviewed")
        self.assertEqual((result["scheduled_slots"], result["complete_slots"],
                          result["censored_slots"], result["unrun_slots"]), (48, 48, 0, 0))
        self.assertEqual((result["decisions_observed"], result["raw_frames_observed"]), (144, 48 * 63))
        self.assertEqual(self.created, 48)
        rows = json.loads((self.root / self.output / "slots.json").read_text())
        self.assertEqual([(row["geometry_seed"], row["actor_id"]) for row in rows],
                         [(cell["geometry_seed"], actor) for cell in self.fixture.cells
                          for actor in g1.ACTORS])
        self.assertTrue(all(row["image_flag"] is None for row in rows))
        self.assertNotIn("pass", result)
        for row in rows:
            trace = self.root / self.output / row["trace_path"]
            self.assertEqual(hashlib.sha256(trace.read_bytes()).hexdigest(), row["trace_sha256"])
        self.assertEqual(hashlib.sha256((self.root / self.output / "slots.json").read_bytes()).hexdigest(),
                         result["slots_sha256"])
        self.assertFalse(any(path.startswith(self.output + "/")
                             for group in _walk_sources(self.root) for path in group))

    def test_keyboard_interrupt_preserves_attempt_censor_partial_and_unrun(self):
        def interrupted(**kwargs):
            self.calls += 1
            kwargs["env_factory"](track_id=1, seed=kwargs["row"]["geometry_seed"],
                                  max_steps=2000, frame_skip=4, reward_shaping=False,
                                  obstacles=True, collision_penalty=0.0)
            kwargs["attempt"].update(phase="driving", decision_calls=3, decisions_completed=2,
                                     raw_counters={"resets": 1, "reset_initial_raw_frames": 1,
                                                   "reset_noop_raw_frames": 50, "driven_raw_frames": 8},
                                     initial_stack=np.zeros((4, 84, 84), dtype=np.uint8),
                                     partial_arrays={"summed_reward": [0.0, -1.0]})
            raise KeyboardInterrupt("simulated interrupt")

        result = self.collect(runner=interrupted)
        self.assertEqual((result["status"], result["complete_slots"],
                          result["censored_slots"], result["unrun_slots"]),
                         ("collection_aborted", 0, 1, 47))
        self.assertEqual(self.created, 1)
        rows = json.loads((self.root / self.output / "slots.json").read_text())
        self.assertEqual((rows[0]["status"], rows[0]["outcome"], rows[0]["decisions"]),
                         ("collection_censored", "unknown", 2))
        self.assertIsNone(rows[0]["image_flag"])
        self.assertTrue(all(row["status"] == "unrun" for row in rows[1:]))
        receipt = json.loads((self.root / self.output / "aborts/slot-00.json").read_text())
        self.assertTrue(receipt["no_retry_or_top_up"])
        self.assertEqual(receipt["phase"], "driving")
        self.assertEqual(receipt["outcome"], "unknown")
        self.assertEqual(hashlib.sha256((self.root / self.output / rows[0]["partial_trace_path"]).read_bytes()).hexdigest(),
                         rows[0]["partial_trace_sha256"])
        with np.load(self.root / self.output / rows[0]["partial_trace_path"]) as partial:
            np.testing.assert_array_equal(partial["summed_reward"], [0.0, -1.0])

    def test_premature_task_timeout_censors_instead_of_relabeling(self):
        def false_timeout(**kwargs):
            result, trace = self.cell(**kwargs)
            result["summary"]["outcome"] = "task_timeout"
            trace["finished"][-1] = False
            return result, trace

        result = self.collect(runner=false_timeout)
        self.assertEqual(result["status"], "collection_aborted")
        self.assertEqual((result["censored_slots"], result["unrun_slots"]), (1, 47))
        rows = json.loads((self.root / self.output / "slots.json").read_text())
        self.assertEqual(rows[0]["outcome"], "unknown")
        self.assertEqual(self.created, 1)

    def test_natural_2000_decision_timeout_after_four_cpu_hours_is_allowed(self):
        def natural_timeout(**kwargs):
            result, trace = self.cell(**kwargs)
            if self.calls == 1:
                result.update(steps=2000, driven_raw_frames=8000)
                result["summary"]["outcome"] = "task_timeout"
                trace["finished"] = np.zeros(2000, dtype=bool)
                trace["terminated"] = np.zeros(2000, dtype=bool)
                trace["truncated"] = np.zeros(2000, dtype=bool)
                trace["truncated"][-1] = True
                kwargs["attempt"]["decisions_completed"] = 2000
                kwargs["attempt"]["raw_counters"]["driven_raw_frames"] = 8000
            return result, trace

        ticks = iter((0.0, 6 * 3600.0))
        result = g1.collect(self.context(), self.output, env_factory=self.env,
                            run_cell=natural_timeout, actor_factory=FakeActor,
                            payload_loader=self.payload, clock=lambda: next(ticks))
        rows = json.loads((self.root / self.output / "slots.json").read_text())
        self.assertEqual((result["status"], result["complete_slots"], result["core_hours_observed"]),
                         ("collected_unreviewed", 48, 6.0))
        self.assertEqual((rows[0]["outcome"], rows[0]["decisions"], rows[1]["outcome"]),
                         ("task_timeout", 2000, "finished"))

    def test_source_drift_after_first_fake_run_stops_without_second_reset(self):
        def drift(**kwargs):
            result = self.cell(**kwargs)
            self.fixture.write("agent.py", b"changed after first fake reset")
            return result

        result = self.collect(runner=drift)
        self.assertEqual((result["status"], result["censored_slots"], result["unrun_slots"]),
                         ("collection_aborted", 1, 47))
        self.assertEqual(self.created, 1)


if __name__ == "__main__":
    unittest.main()
