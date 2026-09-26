"""Synthetic-only tests for the DrQ teacher-replay evaluation orchestrator."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import compare_drq_teacher_replay as compare
from scripts import run_drq_teacher_evaluations as orchestrator_module
from tests.test_drq_teacher_receipts import ReceiptFixture, digest, dump_json


class OrchestratorFixture(ReceiptFixture):
    """ReceiptFixture with trainer-compatible actor paths and checkpoint catalogs."""

    def _build_protocol(self) -> dict:
        protocol = super()._build_protocol()
        for source in protocol["source_actors"]:
            original = self.root / source["run_config_path"]
            value = json.loads(original.read_text(encoding="utf-8"))
            value["config"]["frame_skip"] = protocol["frame_skip"]
            value["config"]["max_steps"] = protocol["max_steps"]
            compatible = original.parent / "config.json"
            dump_json(compatible, value)
            source["run_config_path"] = compatible.relative_to(self.root).as_posix()
            source["run_config_sha256"] = digest(compatible.read_bytes())
        return protocol

    def _actor(self, learner: int, arm: str, step: int | None) -> tuple[Path, str, str | None]:
        if arm == "unchanged-source":
            return super()._actor(learner, arm, step)
        assert step is not None
        source = self.protocol["source_actors"][learner]
        run_dir = self.root / self.protocol["run_root"] / f"learner-{learner}-{arm}"
        checkpoint_dir = run_dir / "checkpoints" / f"step-{step:09d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        actor_path = checkpoint_dir / "actor.pt"
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        manifest_path = checkpoint_dir / "checkpoint.manifest.json"
        actor_path.write_bytes(f"actor-{learner}-{arm}-{step}".encode())
        checkpoint_path.write_bytes(f"checkpoint-{learner}-{arm}-{step}".encode())
        manifest_path.write_bytes(f"manifest-{learner}-{arm}-{step}".encode())
        config_path = run_dir / "config.json"
        if not config_path.exists():
            dump_json(config_path, {
                "name": f"synthetic-{learner}-{arm}",
                "config": {
                    "algorithm": "drq-v2",
                    "seed": 17000 + learner,
                    "total_steps": 131072 + self.protocol["budgets"]["additional_online_decisions_per_arm_source"],
                    "frame_skip": self.protocol["frame_skip"],
                    "max_steps": self.protocol["max_steps"],
                    "study_protocol_sha256": self.protocol_sha,
                    "source_learner_seed": learner,
                    "arm": arm,
                },
            })
        actor_sha = digest(actor_path.read_bytes())
        checkpoint_sha = digest(checkpoint_path.read_bytes())
        record = {
            "source_learner_seed": learner,
            "arm": arm,
            "checkpoint_online_step": step,
            "actor_path": actor_path.relative_to(self.root).as_posix(),
            "actor_sha256": actor_sha,
            "checkpoint_path": checkpoint_path.relative_to(self.root).as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_manifest_path": manifest_path.relative_to(self.root).as_posix(),
            "checkpoint_manifest_sha256": digest(manifest_path.read_bytes()),
            "source_actor_sha256": source["actor_sha256"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "study_protocol_sha256": self.protocol_sha,
        }
        catalog_path = run_dir / "checkpoint-catalog.json"
        if catalog_path.exists():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        else:
            catalog = {
                "schema_version": 1,
                "study_protocol_sha256": self.protocol_sha,
                "source_learner_seed": learner,
                "source_actor_sha256": source["actor_sha256"],
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "arm": arm,
                "teacher_dataset_digest": None,
                "candidates": [],
            }
        catalog["candidates"] = [
            candidate for candidate in catalog["candidates"]
            if candidate["checkpoint_online_step"] != step
        ] + [record]
        catalog["candidates"].sort(key=lambda candidate: candidate["checkpoint_online_step"])
        dump_json(catalog_path, catalog)
        dump_json(run_dir / "result.json", {
            "completed": True,
            "study_protocol_sha256": self.protocol_sha,
            "source_learner_seed": learner,
            "arm": arm,
            "additional_online_steps": self.protocol["budgets"]["additional_online_decisions_per_arm_source"],
            "checkpoint_catalog_path": "checkpoint-catalog.json",
        })
        return actor_path, actor_sha, checkpoint_sha


class SyntheticEvaluator:
    def __init__(self, fixture: OrchestratorFixture, *, mode: str = "pass", fail_after: int | None = None):
        self.fixture = fixture
        self.mode = mode
        self.fail_after = fail_after
        self.calls: list[list[str]] = []
        self.reports: dict[Path, dict] = {}
        self.inventory: dict = {}

    @staticmethod
    def _option(command: list[str], name: str) -> str:
        return command[command.index(name) + 1]

    def run(self, command: list[str], **kwargs) -> SimpleNamespace:
        self.calls.append(command)
        if self.fail_after is not None and len(self.calls) == self.fail_after:
            return SimpleNamespace(returncode=1, stdout="", stderr="synthetic interrupted evaluator")
        self.assert_cpu_command(command)
        stage = self._option(command, "--partition")
        actor_path = Path(self._option(command, "--model")).resolve()
        matches = [actor for actor in self.inventory.values() if actor["actor_path"].resolve() == actor_path]
        if len(matches) != 1:
            raise AssertionError(f"mock actor path was not uniquely cataloged: {actor_path}")
        actor = matches[0]
        identity = actor["identity"]
        previous_pointer = None
        if "--previous-evaluation" in command:
            previous_pointer = Path(self._option(command, "--previous-evaluation")).resolve()
            if previous_pointer not in self.reports:
                raise AssertionError(f"mock previous pointer is unknown: {previous_pointer}")
        previous = self.reports.get(previous_pointer)
        finishes = self.finishes(stage, identity)
        report = self.fixture._write_evaluator(
            stage,
            identity,
            actor_path,
            finishes,
            previous=previous,
            suffix=f"-mock-{len(self.calls)}",
        )
        if stage == "confirmation":
            manifest_path = report["evaluation_dir"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["started_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            dump_json(manifest_path, manifest)
        diagnostic = "--diagnostic-confirmation" in command
        if diagnostic != (stage == "confirmation" and self.mode == "diagnostic"):
            raise AssertionError(f"unexpected diagnostic mode for evaluator call: {command}")
        if diagnostic:
            manifest_path = report["evaluation_dir"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["diagnostic_only"] = True
            dump_json(manifest_path, manifest)
            summary_path = report["evaluation_dir"] / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary[0]["diagnostic_only"] = True
            dump_json(summary_path, summary)
            report["summary"] = summary[0]
            report["pointer_path"].unlink()
            dump_json(report["pointer_path"], {
                "evaluation_dir": str(report["evaluation_dir"].resolve()),
                "protocol_name": f"{self.fixture.protocol['name']}-{stage}",
                "protocol_sha256": self.fixture.protocol_sha,
                "partition": stage,
                "diagnostic_only": True,
                "ranked": summary,
            })

        output_dir = Path(self._option(command, "--evaluations-dir"))
        final_dir = output_dir / f"synthetic-evaluation-{len(self.calls)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(report["evaluation_dir"], final_dir)
        pointer_path = Path(self._option(command, "--output"))
        pointer = {
            "evaluation_dir": str(final_dir.resolve()),
            "protocol_name": f"{self.fixture.protocol['name']}-{stage}",
            "protocol_sha256": self.fixture.protocol_sha,
            "partition": stage,
            "diagnostic_only": diagnostic,
            "ranked": [report["summary"]],
        }
        if self.mode == "malformed-pointer":
            pointer["ranked"][0]["archive_sha256"] = "0" * 64
        dump_json(pointer_path, pointer)
        current = {
            **report,
            "pointer_path": pointer_path.resolve(),
            "evaluation_dir": final_dir.resolve(),
            "summary": report["summary"],
        }
        self.reports[pointer_path.resolve()] = current
        return SimpleNamespace(returncode=0, stdout="synthetic evaluator complete", stderr="")

    def assert_cpu_command(self, command: list[str]) -> None:
        self.assert_has(command, "--workers", "1")
        self.assert_has(command, "--max-steps", "2000")
        self.assert_has(command, "--frame-skip", "4")
        self.assert_has(command, "--protocol-file", str(self.fixture.protocol_path.resolve()))
        if command[:3] != ["python", "-m", "evaluate_policy"]:
            raise AssertionError(f"evaluator did not use the frozen runtime executable: {command[:3]}")

    @staticmethod
    def assert_has(command: list[str], name: str, value: str) -> None:
        if name not in command or command[command.index(name) + 1] != value:
            raise AssertionError(f"missing pinned evaluator argument {name} {value}")

    def finishes(self, stage: str, identity: dict) -> set[tuple[int, int]]:
        if stage == "screen" and self.mode == "screen-fail" and identity["arm"] == "teacher-replay":
            return set()
        if stage == "screen" and self.mode == "diagnostic" and identity["arm"] in {
            "unchanged-source", "online-only",
        }:
            return set()
        if stage == "confirmation" and self.mode == "confirmation-fail" and identity["arm"] == "teacher-replay":
            return set()
        if stage == "blind":
            return self.fixture._finishes(stage, identity["source_learner_seed"], identity["arm"],
                                          identity["checkpoint_online_step"])
        return self.fixture._finishes(stage, identity["source_learner_seed"], identity["arm"],
                                      identity["checkpoint_online_step"])


class DrQTeacherOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="drq-teacher-orchestrator-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = OrchestratorFixture(self.root)
        self.orchestrator = orchestrator_module.DrQTeacherEvaluationOrchestrator(
            self.fixture.protocol_path, self.root, package_backend=self._fake_package_backend
        )
        self.mock_evaluator = SyntheticEvaluator(self.fixture)

    @staticmethod
    def _fake_package_backend(actor: dict, archive_path: Path) -> dict:
        archive_path.write_bytes(f"synthetic-package-{actor['identity']['actor_sha256']}".encode())
        return {
            "expected_model_filename": "model.pt",
            "model_format": "haic-drq-v2-actor-v1",
            "modules": ["agent.py", "drq_v2.py"],
            "smoke": {
                "init_seconds": 0.1,
                "act_seconds": 0.01,
                "shape": [3],
                "finite": True,
                "reset_matches_first": True,
                "unreset_matches_first": True,
            },
        }

    def install_mock(self, evaluator: SyntheticEvaluator | None = None) -> SyntheticEvaluator:
        selected = evaluator or self.mock_evaluator
        selected.inventory = self.orchestrator.inventory
        self.mock_evaluator = selected
        self.run_patch = patch.object(orchestrator_module.subprocess, "run", side_effect=selected.run)
        self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        return selected

    def test_screen_inventory_is_two_sources_and_four_checkpoints_per_source(self) -> None:
        identities = set(self.orchestrator.inventory)
        self.assertEqual(len(identities), 10)
        self.assertEqual(
            {identity for identity in identities if identity[1] == "unchanged-source"},
            {(0, "unchanged-source", None), (1, "unchanged-source", None)},
        )
        self.assertEqual(
            {identity for identity in identities if identity[1] != "unchanged-source"},
            {
                (learner, arm, step)
                for learner in (0, 1)
                for arm in ("online-only", "teacher-replay")
                for step in (16384, 32768)
            },
        )

    def test_screen_tuple_and_exact_checkpoint_ties_keep_earlier_step(self) -> None:
        def candidate(step: int, finishes: int, progress: float, lap: float) -> dict:
            return {
                "identity": {"checkpoint_online_step": step},
                "metrics": {
                    "finish_count": finishes,
                    "canonical_mean_progress": progress,
                    "completed_mean_lap_time_ms": lap,
                },
            }

        lower_finish_progress_wins_nothing = [candidate(16384, 3, 0.99, 1000), candidate(32768, 4, 0.1, 2000)]
        self.assertEqual(orchestrator_module.choose_screen_actor(lower_finish_progress_wins_nothing)["identity"]["checkpoint_online_step"], 32768)
        progress_breaks_finish_tie = [candidate(16384, 4, 0.4, 900), candidate(32768, 4, 0.5, 2000)]
        self.assertEqual(orchestrator_module.choose_screen_actor(progress_breaks_finish_tie)["identity"]["checkpoint_online_step"], 32768)
        lap_breaks_progress_tie = [candidate(16384, 4, 0.5, 900), candidate(32768, 4, 0.5, 1000)]
        self.assertEqual(orchestrator_module.choose_screen_actor(lap_breaks_progress_tie)["identity"]["checkpoint_online_step"], 16384)
        exact_tie_keeps_earlier = [candidate(32768, 4, 0.5, 1000), candidate(16384, 4, 0.5, 1000)]
        self.assertEqual(orchestrator_module.choose_screen_actor(exact_tie_keeps_earlier)["identity"]["checkpoint_online_step"], 16384)

    def test_confirmation_uses_frozen_six_and_per_actor_screen_pointers(self) -> None:
        evaluator = self.install_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            screen = self.orchestrator.run("screen")
            first_screen_call_count = len(evaluator.calls)
            replayed = self.orchestrator.run("screen")
            confirmation = self.orchestrator.run("confirmation")
        self.assertTrue(screen["gate_passed"])
        self.assertEqual(len(screen["package_receipts"]), 10)
        self.assertEqual(len(replayed["wrapper_paths"]), 10)
        self.assertEqual(first_screen_call_count, 10)
        self.assertEqual(len(evaluator.calls), 16)
        self.assertTrue(confirmation["passed"])
        selection = json.loads(screen["selection_path"].read_text(encoding="utf-8"))
        self.assertEqual(len(selection["candidate_receipts"]), 10)
        self.assertEqual(len(selection["selected_identities"]), 6)
        self.assertEqual(selection["blind_finalist"]["source_learner_seed"], 0)
        frozen = {
            (item["source_learner_seed"], item["arm"]): item
            for item in selection["selected_identities"]
        }
        for wrapper_path in confirmation["wrapper_paths"]:
            wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
            identity = compare._identity(wrapper)
            chosen = frozen[(identity["source_learner_seed"], identity["arm"])]
            screen_path = Path(chosen["screen_wrapper_path"]).resolve()
            self.assertEqual(wrapper["screen_receipt_sha256"], digest(screen_path.read_bytes()))
            self.assertEqual(wrapper["screen_candidate_identity"], identity)
            pointer = json.loads(Path(wrapper["evaluator_receipt_path"]).read_text(encoding="utf-8"))
            manifest = json.loads((Path(pointer["evaluation_dir"]) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["previous_evaluation"]["files"]["previous_evaluation.json"]["source_path"],
                json.loads(screen_path.read_text(encoding="utf-8"))["evaluator_receipt_path"],
            )
            self.assertEqual(identity, {
                key: chosen[key] for key in (
                    "source_learner_seed", "arm", "actor_sha256", "checkpoint_sha256", "checkpoint_online_step",
                )
            })

    def test_all_never_opens_blind_after_failed_confirmation(self) -> None:
        evaluator = self.install_mock(SyntheticEvaluator(self.fixture, mode="confirmation-fail"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.orchestrator.run("all")
        self.assertFalse(result["confirmation"]["passed"])
        self.assertIsNone(result["blind"])
        self.assertFalse(result["blind_opened"])
        self.assertEqual(len(evaluator.calls), 16)
        self.assertNotIn("blind", [SyntheticEvaluator._option(call, "--partition") for call in evaluator.calls])

    def test_all_opens_only_the_predeclared_screen_finalist_after_confirmation_pass(self) -> None:
        evaluator = self.install_mock()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.orchestrator.run("all")
            calls_before_idempotent_read = len(evaluator.calls)
            replayed = self.orchestrator.run("all")
        self.assertTrue(result["confirmation"]["passed"])
        self.assertTrue(result["blind_opened"])
        self.assertTrue(result["blind"]["accepted"])
        self.assertEqual(len(evaluator.calls), 17)
        self.assertEqual(len(evaluator.calls), calls_before_idempotent_read)
        self.assertTrue(replayed["blind_opened"])
        self.assertEqual(replayed["blind"]["decision_path"], result["blind"]["decision_path"])
        output_root = (self.root / self.fixture.protocol["run_root"] / "evaluations").resolve()
        for command in evaluator.calls:
            for option in ("--evaluations-dir", "--output"):
                self.assertTrue(Path(SyntheticEvaluator._option(command, option)).resolve().is_relative_to(output_root))
        blind_call = evaluator.calls[-1]
        self.assertEqual(SyntheticEvaluator._option(blind_call, "--partition"), "blind")
        self.assertEqual(SyntheticEvaluator._option(blind_call, "--workers"), "1")
        blind_wrapper = json.loads(result["blind"]["wrapper_paths"][0].read_text(encoding="utf-8"))
        finalist = json.loads(result["screen"]["finalist_path"].read_text(encoding="utf-8"))
        self.assertEqual(compare._identity(blind_wrapper), {
            key: finalist[key] for key in (
                "source_learner_seed", "arm", "actor_sha256", "checkpoint_sha256", "checkpoint_online_step",
            )
        })
        self.assertEqual(
            SyntheticEvaluator._option(blind_call, "--previous-evaluation"),
            json.loads(next(
                path for path in result["confirmation"]["wrapper_paths"]
                if compare._identity(json.loads(path.read_text(encoding="utf-8")))["source_learner_seed"]
                == finalist["source_learner_seed"]
                and json.loads(path.read_text(encoding="utf-8"))["arm"] == "teacher-replay"
            ).read_text(encoding="utf-8"))["evaluator_receipt_path"],
        )

    def test_screen_gate_failure_consumes_screen_but_never_starts_confirmation(self) -> None:
        evaluator = self.install_mock(SyntheticEvaluator(self.fixture, mode="screen-fail"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            orchestrator_module.OrchestrationError, "screen gate failed"
        ):
            self.orchestrator.run("all")
        self.assertEqual(len(evaluator.calls), 10)
        self.assertNotIn("confirmation", [SyntheticEvaluator._option(call, "--partition") for call in evaluator.calls])
        self.assertNotIn("blind", [SyntheticEvaluator._option(call, "--partition") for call in evaluator.calls])

    def test_diagnostic_confirmation_completes_grid_and_never_opens_blind(self) -> None:
        evaluator = self.install_mock(SyntheticEvaluator(self.fixture, mode="diagnostic"))
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.orchestrator.run("all", diagnostic_confirmation=True)
            calls_before_idempotent_read = len(evaluator.calls)
            replayed = self.orchestrator.run("all", diagnostic_confirmation=True)
        self.assertTrue(result["screen"]["diagnostic_confirmation_required"])
        self.assertTrue(result["confirmation"]["diagnostic_only"])
        self.assertFalse(result["blind_opened"])
        self.assertIsNone(result["blind"])
        self.assertEqual(len(evaluator.calls), 16)
        self.assertEqual(len(evaluator.calls), calls_before_idempotent_read)
        self.assertFalse(replayed["blind_opened"])
        confirmation_calls = [call for call in evaluator.calls if SyntheticEvaluator._option(call, "--partition") == "confirmation"]
        self.assertEqual(len(confirmation_calls), 6)
        self.assertTrue(all("--diagnostic-confirmation" in call for call in confirmation_calls))
        self.assertFalse(any(SyntheticEvaluator._option(call, "--partition") == "blind" for call in evaluator.calls))

    def test_partial_run_is_rejected_without_rerunning_consumed_cells(self) -> None:
        evaluator = self.install_mock(SyntheticEvaluator(self.fixture, fail_after=2))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(orchestrator_module.OrchestrationError):
            self.orchestrator.run("screen")
        calls_after_failure = len(evaluator.calls)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            orchestrator_module.OrchestrationError, "partial screen stage artifacts"
        ):
            self.orchestrator.run("screen")
        self.assertEqual(len(evaluator.calls), calls_after_failure)

    def test_malformed_evaluator_identity_is_consumed_and_never_retried(self) -> None:
        evaluator = self.install_mock(SyntheticEvaluator(self.fixture, mode="malformed-pointer"))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            orchestrator_module.OrchestrationError, "actor hash differs from the frozen catalog"
        ):
            self.orchestrator.run("screen")
        calls_after_failure = len(evaluator.calls)
        self.assertEqual(calls_after_failure, 1)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            orchestrator_module.OrchestrationError, "partial screen stage artifacts"
        ):
            self.orchestrator.run("screen")
        self.assertEqual(len(evaluator.calls), calls_after_failure)

    def test_individual_later_stages_require_completed_predecessors(self) -> None:
        evaluator = self.install_mock()
        for stage in ("confirmation", "blind"):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                orchestrator_module.OrchestrationError, "screen stage is not complete"
            ):
                self.orchestrator.run(stage)
        self.assertEqual(evaluator.calls, [])

    def test_identity_catalog_mismatch_fails_before_any_evaluation(self) -> None:
        catalog = next((self.root / self.fixture.protocol["run_root"]).rglob("checkpoint-catalog.json"))
        value = json.loads(catalog.read_text(encoding="utf-8"))
        value["study_protocol_sha256"] = "0" * 64
        dump_json(catalog, value)
        with self.assertRaisesRegex(orchestrator_module.OrchestrationError, "catalog protocol/source identity mismatch"):
            orchestrator_module.DrQTeacherEvaluationOrchestrator(self.fixture.protocol_path, self.root)


if __name__ == "__main__":
    unittest.main()
