"""Synthetic, zero-environment tests for the postrun r2 diagnostic binder."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import diagnose_drq_final_source_reconstruction_r2 as binder


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Synthetic:
    def __init__(self, root: Path):
        self.root = root
        self.postrun: dict = {}
        self.receipt: dict = {}
        self.training: dict = {}
        self.calls: list[str] = []

    def put(self, relative: Path | str, raw: bytes | object) -> str:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = raw if isinstance(raw, bytes) else _bytes(raw)
        path.write_bytes(encoded)
        return _sha(encoded)

    def write_receipt(self) -> None:
        self.put(binder.AUDIT_PATH, self.receipt)


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Synthetic:
    case = Synthetic(tmp_path)
    code = {name: case.put(name, name.encode("ascii")) for name in binder.CODE_PATHS}
    training = {
        "study_id": binder.reconstruction.STUDY_ID,
        "source_replay": {},
    }
    source_rows = {}
    for seed in (0, 1):
        source_path = f"{binder.reconstruction.RUN_ROOT}/collection/seed{seed}/receipt.json"
        source_receipt = {"episode_ledger_sha256": f"{seed + 1}" * 64,
                          "valid_n_step_starts": 99997, "decisions": 100000,
                          "scheduled_episodes_consumed": 218 + seed,
                          "historical_prefix_decisions": 0,
                          "historical_prefix_episodes_consumed": 0}
        source_sha = case.put(source_path, source_receipt)
        pool_sha = f"{seed + 3}" * 64
        training["source_replay"][str(seed)] = {
            "receipt_path": source_path, "receipt_sha256": source_sha, "pool_sha256": pool_sha,
        }
        source_rows[str(seed)] = {
            "pool_sha256": pool_sha, "receipt_sha256": source_sha,
            "episode_ledger_sha256": source_receipt["episode_ledger_sha256"],
            "valid_n_step_starts": 99997, "verified_source_decisions": 100000,
            "source_episodes": 218 + seed, "historical_prefix_decisions": 0,
            "historical_prefix_episodes_consumed": 0,
        }
    case.training = training
    case.put(binder.reconstruction.PROTOCOL_PATH, training)
    monkeypatch.setattr(binder, "TRAINING_PROTOCOL_SHA", _sha(_bytes(training)))
    result_hashes = {}
    runs = []
    warmup_runs = []
    for seed, variant in sorted(binder.ROLES):
        key = (seed, variant)
        name = binder._role_name(key)
        run_dir = f"{binder.reconstruction.RUN_ROOT}/{name}"
        result = {
            "study_id": binder.reconstruction.STUDY_ID,
            "study_protocol_sha256": binder.TRAINING_PROTOCOL_SHA,
            "completed": True, "source_seed": seed, "variant": variant,
            "condition": "final_source", "source_replay_sha256": source_rows[str(seed)]["pool_sha256"],
            "additional_online_steps": 32768, "study_gradient_steps": 22768,
            "source_samples": 22768 * 32, "online_samples": 22768 * 32,
            "candidates": [
                {"checkpoint_sha256": "a" * 64, "sample_trace_sha256": "b" * 64},
                {"checkpoint_sha256": "c" * 64, "sample_trace_sha256": "d" * 64},
            ],
        }
        digest = case.put(f"{run_dir}/result.json", result)
        result_hashes[name] = digest
        runs.append({
            "source_seed": seed, "variant": variant, "run_dir": run_dir,
            "result_sha256": digest, "final_checkpoint_sha256": "c" * 64,
            "final_sample_trace_sha256": "d" * 64, "updates": 22768,
            "batch_size": 64, "source_rows": 22768 * 32, "online_rows": 22768 * 32,
            "unique_source_starts": 99997, "unique_online_starts": 22768,
        })
        warmup_runs.append({
            "source_seed": seed, "variant": variant,
            "first_mismatch": {field: (11 if field == "frames" else None)
                               for field in binder.WARMUP_FIELDS},
            "boundary_equal": True, "boundary_first_mismatch": None,
        })
    case.postrun = {
        "format": "haic-drq-final-source-postrun-protocol-r2", "study_id": binder.STUDY_ID,
        "training_protocol_path": binder.reconstruction.PROTOCOL_PATH,
        "training_protocol_sha256": binder.TRAINING_PROTOCOL_SHA,
        "audit_receipt_path": binder.AUDIT_PATH.as_posix(),
        "diagnostic_output_root": binder.OUTPUT_ROOT.as_posix(),
        "interpretation": binder.INTERPRETATION,
        "original_audit_status": binder.ORIGINAL_FAILURE,
        "original_audit_failure": binder.ORIGINAL_FAILURE_TEXT,
        "diagnostic": binder.DIAGNOSTIC_CONTRACT.copy(),
        "code_sha256": code, "run_result_sha256": result_hashes,
    }
    postrun_sha = case.put(binder.PROTOCOL_PATH, case.postrun)
    case.receipt = {
        "format": "haic-drq-final-source-postrun-sample-audit-r2",
        "passed": True, "original_audit_passed": False, "no_environment_resets": True,
        "postrun_protocol_sha256": postrun_sha,
        "training_protocol_sha256": binder.TRAINING_PROTOCOL_SHA,
        "interpretation": binder.INTERPRETATION, "causal_comparison_eligible": False,
        "historical_warmup_comparison": {
            "matched": False, "warmup_steps": 10000, "runs": warmup_runs,
        },
        "source_pool": source_rows, "runs": runs,
    }
    case.write_receipt()
    monkeypatch.setattr(binder.reconstruction, "_validate_protocol", lambda *_: None)
    monkeypatch.setattr(binder.reconstruction, "_validate_exposure_snapshot", lambda *_: None)
    monkeypatch.setattr(binder.child, "_validate_cpu_runtime", lambda: None)
    return case


def _invoke(case: Synthetic, *, preflight: bool = True) -> int:
    args = ["--repo-root", str(case.root)]
    if preflight:
        args.append("--preflight-only")
    return binder.main(args)


def test_preflight_uses_distinct_receipt_parent_identity_and_no_output(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic
    old_argv = sys.argv

    def fake_pools(*_args: object) -> dict:
        assert module.STUDY_ID == binder.child.PARENT_STUDY_ID
        assert module.RUN_ROOT == Path(binder.child.PARENT_RUN_ROOT)
        case.calls.append("parent pools")
        return {int(seed): {"pool_sha256": row["pool_sha256"],
                            "receipt_sha256": row["receipt_sha256"]}
                for seed, row in case.receipt["source_pool"].items()}

    def fake_prepare(root: Path, protocol_path: Path, loader: object) -> tuple:
        del protocol_path, loader
        pools = module._pools(root, {}, {}, set())
        records = {}
        for row in case.receipt["runs"]:
            key = (row["source_seed"], row["variant"])
            records[key] = {
                "result_sha256": row["result_sha256"],
                "checkpoint_sha256": row["final_checkpoint_sha256"],
                "sample_trace_sha256": row["final_sample_trace_sha256"],
                "source_replay_sha256": pools[key[0]]["pool_sha256"],
            }
        sha = module._sample_audit(root, binder.TRAINING_PROTOCOL_SHA, pools, records)
        return [], {}, {}, {}, {"sample_audit_path": "old-path", "sample_audit_sha256": sha}

    def fake_run(*, root: Path, protocol_path: Path, output_root: Path,
                 preflight_only: bool, environment_factory: object,
                 actor_loader: object) -> dict:
        del environment_factory, actor_loader
        assert preflight_only
        assert protocol_path == root / binder.reconstruction.PROTOCOL_PATH
        assert output_root == root / binder.OUTPUT_ROOT
        module._require(False, binder.OUTPUT_GUARD)
        with pytest.raises(module.DiagnosticError):
            module._require(False, "unrelated frozen safety check")
        lineage = module._prepare(root, protocol_path, lambda *_: None)[4]
        assert lineage["postrun_protocol_sha256"] == case.receipt["postrun_protocol_sha256"]
        assert lineage["sample_audit_path"] == binder.AUDIT_PATH.as_posix()
        assert lineage["sample_audit_sha256"] == _sha(_bytes(case.receipt))
        assert lineage["original_audit_status"] == binder.ORIGINAL_FAILURE
        assert lineage["interpretation"] == binder.INTERPRETATION
        assert lineage["causal_comparison_eligible"] is False
        assert "warmup environment pixel/action/reward parity unproven" in lineage["sample_audit_scope"]
        case.calls.append("parent preflight")
        return {"environment_resets": 0, "output_created": False}

    def fake_main() -> int:
        assert module.STUDY_ID == binder.reconstruction.STUDY_ID
        assert module.RUN_ROOT == Path(binder.reconstruction.RUN_ROOT)
        assert module.SAMPLE_AUDIT == binder.AUDIT_PATH
        assert "--preflight-only" in sys.argv
        module.run_diagnostic(root=case.root, protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                              output_root=case.root / binder.OUTPUT_ROOT, preflight_only=True)
        return 0

    monkeypatch.setattr(module, "_pools", fake_pools)
    monkeypatch.setattr(module, "_prepare", fake_prepare)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    monkeypatch.setattr(module, "main", fake_main)
    before = tuple(getattr(module, key) for key in (
        "STUDY_ID", "RUN_ROOT", "SAMPLE_AUDIT", "_pools", "_sample_audit", "_prepare",
        "_require", "run_diagnostic"))
    assert _invoke(case) == 0
    assert case.calls == ["parent pools", "parent preflight"]
    assert not (case.root / binder.OUTPUT_ROOT).exists()
    assert not (case.root / binder.reconstruction.SAMPLE_AUDIT_PATH).exists()
    assert sys.argv is old_argv
    assert tuple(getattr(module, key) for key in (
        "STUDY_ID", "RUN_ROOT", "SAMPLE_AUDIT", "_pools", "_sample_audit", "_prepare",
        "_require", "run_diagnostic")) == before


def test_frozen_cpu_preflight_dispatches_without_environment_or_output(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    case = synthetic
    module = binder.diagnostic

    def fake_prepare(root: Path, protocol_path: Path, loader: object) -> tuple:
        del protocol_path, loader
        pools = {int(seed): {"pool_sha256": row["pool_sha256"],
                             "receipt_sha256": row["receipt_sha256"]}
                 for seed, row in case.receipt["source_pool"].items()}
        records = {}
        for row in case.receipt["runs"]:
            records[(row["source_seed"], row["variant"])] = {
                "result_sha256": row["result_sha256"],
                "checkpoint_sha256": row["final_checkpoint_sha256"],
                "sample_trace_sha256": row["final_sample_trace_sha256"],
                "source_replay_sha256": pools[row["source_seed"]]["pool_sha256"],
            }
        audited = module._sample_audit(root, binder.TRAINING_PROTOCOL_SHA, pools, records)
        return list(range(16)), {binder._role_name(key): {} for key in binder.ROLES}, (
            {key: {} for key in range(32)}), {key: {} for key in range(96)}, {
                "protocol_sha256": binder.TRAINING_PROTOCOL_SHA,
                "sample_audit_sha256": audited,
            }

    monkeypatch.setattr(module, "_prepare", fake_prepare)
    def forbidden_environment(*_args: object, **_kwargs: object) -> None:
        pytest.fail("preflight constructed an environment")
    monkeypatch.setattr(module.r6, "build_environment", forbidden_environment)
    monkeypatch.setattr(module.r6, "_collect_episode", lambda *_args, **_kwargs: pytest.fail(
        "preflight entered an episode collector"))
    assert _invoke(case) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["expected_episode_count"] == 192
    assert printed["preflight_only"] is True
    assert printed["output_root"] == str(case.root / binder.OUTPUT_ROOT)
    assert not (case.root / binder.OUTPUT_ROOT).exists()
    assert not (case.root / binder.reconstruction.SAMPLE_AUDIT_PATH).exists()


@pytest.mark.parametrize("change", [
    "missing", "old_format", "wrong_postrun_hash", "original_claims_pass",
    "wrong_training_hash", "false_success", "false_no_resets", "causal_claim",
    "absent_role", "duplicate_role", "absent_warmup", "matched_warmup",
    "missing_warmup_field", "missing_frame_mismatch", "bad_boundary", "missing_boundary",
    "wrong_pool", "wrong_pool_count", "wrong_source_episodes", "wrong_result", "wrong_checkpoint",
    "wrong_sample_trace", "wrong_samples", "wrong_batch", "wrong_unique_starts",
    "missing_original_failure_marker",
])
def test_receipt_tamper_or_absence_blocks_before_parent(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    case = synthetic
    monkeypatch.setattr(binder.diagnostic, "main", lambda: pytest.fail("parent diagnostic invoked"))
    receipt = copy.deepcopy(case.receipt)
    if change == "missing":
        (case.root / binder.AUDIT_PATH).unlink()
    elif change == "old_format":
        receipt["format"] = "haic-drq-final-source-sample-audit-v1"
    elif change == "wrong_postrun_hash":
        receipt["postrun_protocol_sha256"] = "0" * 64
    elif change == "original_claims_pass":
        receipt["original_audit_passed"] = True
    elif change == "wrong_training_hash":
        receipt["training_protocol_sha256"] = "0" * 64
    elif change == "false_success":
        receipt["passed"] = False
    elif change == "false_no_resets":
        receipt["no_environment_resets"] = False
    elif change == "causal_claim":
        receipt["causal_comparison_eligible"] = True
    elif change == "absent_role":
        receipt["runs"].pop()
    elif change == "duplicate_role":
        receipt["runs"][1] = receipt["runs"][0]
    elif change == "absent_warmup":
        receipt["historical_warmup_comparison"]["runs"].pop()
    elif change == "matched_warmup":
        receipt["historical_warmup_comparison"]["matched"] = True
    elif change == "missing_warmup_field":
        del receipt["historical_warmup_comparison"]["runs"][0]["first_mismatch"]["actions"]
    elif change == "missing_frame_mismatch":
        receipt["historical_warmup_comparison"]["runs"][0]["first_mismatch"]["frames"] = None
    elif change == "bad_boundary":
        receipt["historical_warmup_comparison"]["runs"][0]["boundary_first_mismatch"] = 9
    elif change == "missing_boundary":
        del receipt["historical_warmup_comparison"]["runs"][0]["boundary_first_mismatch"]
    elif change == "wrong_pool":
        receipt["source_pool"]["0"]["pool_sha256"] = "0" * 64
    elif change == "wrong_pool_count":
        receipt["source_pool"]["0"]["valid_n_step_starts"] = 99996
    elif change == "wrong_source_episodes":
        receipt["source_pool"]["0"]["source_episodes"] = 0
    elif change == "wrong_result":
        receipt["runs"][0]["result_sha256"] = "0" * 64
    elif change == "wrong_checkpoint":
        receipt["runs"][0]["final_checkpoint_sha256"] = "0" * 64
    elif change == "wrong_sample_trace":
        receipt["runs"][0]["final_sample_trace_sha256"] = "0" * 64
    elif change == "wrong_samples":
        receipt["runs"][0]["source_rows"] = 22768 * 32 - 1
    elif change == "wrong_batch":
        receipt["runs"][0]["batch_size"] = 32
    elif change == "wrong_unique_starts":
        receipt["runs"][0]["unique_source_starts"] = 0
    elif change == "missing_original_failure_marker":
        receipt.pop("original_audit_passed")
    if change != "missing":
        case.receipt = receipt
        case.write_receipt()
    with pytest.raises((binder.DiagnosticGateError, binder.reconstruction.exposure_audit.AuditError)):
        _invoke(case)
    assert not (case.root / binder.OUTPUT_ROOT).exists()


@pytest.mark.parametrize("change", [
    "wrong_training_protocol", "tampered_code", "wrong_result_pin", "partial_results",
    "wrong_output", "wrong_original_marker", "wrong_failure_text", "missing_failure_text",
    "occupied_output", "old_receipt", "cpu_runtime",
])
def test_protocol_partial_runtime_and_output_gates(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    case = synthetic
    monkeypatch.setattr(binder.diagnostic, "main", lambda: pytest.fail("parent diagnostic invoked"))
    if change == "wrong_training_protocol":
        case.put(binder.reconstruction.PROTOCOL_PATH, {"source_replay": {}})
    elif change == "tampered_code":
        case.put("scripts/diagnose_drq_final_source_reconstruction_r2.py", b"tampered")
    elif change == "wrong_result_pin":
        case.postrun["run_result_sha256"][binder._role_name((0, "uniform"))] = "0" * 64
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "partial_results":
        case.postrun["run_result_sha256"].pop(binder._role_name((0, "uniform")))
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "wrong_output":
        case.postrun["diagnostic_output_root"] = binder.reconstruction.DIAGNOSTIC_OUTPUT_ROOT
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "wrong_original_marker":
        case.postrun["original_audit_status"] = "passed"
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "wrong_failure_text":
        case.postrun["original_audit_failure"] = "original audit passed"
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "missing_failure_text":
        case.postrun.pop("original_audit_failure")
        case.put(binder.PROTOCOL_PATH, case.postrun)
    elif change == "occupied_output":
        (case.root / binder.OUTPUT_ROOT).mkdir()
    elif change == "old_receipt":
        case.put(binder.reconstruction.SAMPLE_AUDIT_PATH, {"passed": True})
    elif change == "cpu_runtime":
        def wrong_runtime() -> None:
            raise binder.DiagnosticGateError("CPU21 runtime mismatch")
        monkeypatch.setattr(binder.child, "_validate_cpu_runtime", wrong_runtime)
    with pytest.raises(binder.DiagnosticGateError):
        _invoke(case)
    if change != "occupied_output":
        assert not (case.root / binder.OUTPUT_ROOT).exists()


@pytest.mark.parametrize("field", list(binder.DIAGNOSTIC_CONTRACT))
def test_postrun_diagnostic_grid_is_checked_before_delegation(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    case = synthetic
    monkeypatch.setattr(binder.diagnostic, "main", lambda: pytest.fail("parent diagnostic invoked"))
    case.postrun["diagnostic"][field] = False if field == "track_id" else "changed"
    case.put(binder.PROTOCOL_PATH, case.postrun)
    with pytest.raises(binder.DiagnosticGateError, match="diagnostic partition, grid"):
        _invoke(case)
    assert not (case.root / binder.OUTPUT_ROOT).exists()


@pytest.mark.parametrize("relative", [binder.PROTOCOL_PATH, binder.AUDIT_PATH])
def test_parse_and_digest_share_one_safe_read_then_handoff_catches_replacement(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, relative: Path) -> None:
    case = synthetic
    target = case.root / relative
    original = Path.read_bytes
    replaced = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        raw = original(path)
        if path == target and not replaced:
            replaced = True
            target.write_bytes(raw + b"\n")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    monkeypatch.setattr(binder.diagnostic, "main", lambda: pytest.fail("parent diagnostic invoked"))
    with pytest.raises(binder.DiagnosticGateError, match="pinned postrun input changed"):
        _invoke(case)
    assert replaced
    assert not (case.root / binder.OUTPUT_ROOT).exists()


@pytest.mark.parametrize("relative", [
    binder.PROTOCOL_PATH, binder.AUDIT_PATH,
    Path(binder.reconstruction.RUN_ROOT) / "learner-0-uniform-final_source/result.json",
])
def test_symlinked_input_blocks_even_if_target_bytes_match(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, relative: Path) -> None:
    case = synthetic
    target = case.root / relative
    backup = case.root / "backup.json"
    backup.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(backup)
    monkeypatch.setattr(binder.diagnostic, "main", lambda: pytest.fail("parent diagnostic invoked"))
    with pytest.raises(binder.reconstruction.exposure_audit.AuditError, match="symlink"):
        _invoke(case)
    assert not (case.root / binder.OUTPUT_ROOT).exists()


@pytest.mark.parametrize("relative", [
    binder.PROTOCOL_PATH,
    binder.AUDIT_PATH,
    Path(binder.reconstruction.RUN_ROOT) / "learner-0-uniform-final_source/result.json",
])
def test_mutation_before_first_factory_blocks_without_a_synthetic_reset(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, relative: Path) -> None:
    case = synthetic
    module = binder.diagnostic
    previous_factory = module.r6.build_environment
    previous_run = module.run_diagnostic
    previous_pool = module._pools
    old_argv = sys.argv
    called = []

    def fake_factory(*_args: object) -> object:
        called.append("factory")
        return object()

    def fake_run(**_kwargs: object) -> dict:
        target = case.root / relative
        target.write_bytes(target.read_bytes() + b"\n")
        module.r6.build_environment(1234, 2000)
        return {}

    def fake_main() -> int:
        module.run_diagnostic(root=case.root,
                              protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                              output_root=case.root / binder.OUTPUT_ROOT)
        return 0

    monkeypatch.setattr(module.r6, "build_environment", fake_factory)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    monkeypatch.setattr(module, "main", fake_main)
    with pytest.raises(binder.DiagnosticGateError, match="pinned postrun input changed"):
        _invoke(case, preflight=False)
    assert not called
    assert module.r6.build_environment is fake_factory
    assert module.run_diagnostic is fake_run
    assert module._pools is previous_pool
    assert sys.argv is old_argv
    assert not (case.root / binder.OUTPUT_ROOT).exists()
    monkeypatch.setattr(module.r6, "build_environment", previous_factory)
    monkeypatch.setattr(module, "run_diagnostic", previous_run)


def test_each_factory_call_rechecks_pins_and_preserves_downstream_exception(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic
    calls = []

    def fake_factory(*_args: object) -> object:
        calls.append("factory")
        return object()

    def fake_run(**_kwargs: object) -> dict:
        module.r6.build_environment(1234, 2000)
        result = case.root / binder.reconstruction.RUN_ROOT / (
            "learner-1-easy_retention-final_source/result.json")
        result.write_bytes(result.read_bytes() + b"\n")
        module.r6.build_environment(1234, 2000)
        return {}

    monkeypatch.setattr(module.r6, "build_environment", fake_factory)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    with pytest.raises(binder.DiagnosticGateError, match="pinned postrun input changed"):
        with binder._diagnostic_context(case.root, case.postrun,
                                        _sha(_bytes(case.postrun)), case.training,
                                        _sha(_bytes(case.receipt))):
            module.run_diagnostic(root=case.root,
                                  protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                                  output_root=case.root / binder.OUTPUT_ROOT)
    assert calls == ["factory"]
    assert module.r6.build_environment is fake_factory
    assert module.run_diagnostic is fake_run


def test_symlink_replacement_between_handoff_and_factory_restores_globals(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic
    old_factory = module.r6.build_environment
    old_argv = sys.argv
    factory_calls = []

    def fake_factory(*_args: object) -> object:
        factory_calls.append("called")
        return object()

    def fake_run(**_kwargs: object) -> dict:
        target = case.root / binder.AUDIT_PATH
        backup = case.root / "backup-receipt.json"
        backup.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(backup)
        module.r6.build_environment(1234, 2000)
        return {}

    def fake_main() -> int:
        module.run_diagnostic(root=case.root,
                              protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                              output_root=case.root / binder.OUTPUT_ROOT)
        return 0

    monkeypatch.setattr(module.r6, "build_environment", fake_factory)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    monkeypatch.setattr(module, "main", fake_main)
    with pytest.raises(binder.reconstruction.exposure_audit.AuditError, match="symlink"):
        _invoke(case, preflight=False)
    assert not factory_calls
    assert module.r6.build_environment is fake_factory
    assert module.run_diagnostic is fake_run
    assert sys.argv is old_argv
    assert not (case.root / binder.OUTPUT_ROOT).exists()
    monkeypatch.setattr(module.r6, "build_environment", old_factory)


@pytest.mark.parametrize("relative", [
    binder.PROTOCOL_PATH,
    binder.AUDIT_PATH,
    Path(binder.reconstruction.RUN_ROOT) / "learner-1-uniform-final_source/result.json",
])
def test_mutation_after_repeat0_blocks_repeat1_before_underlying_collector(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch, relative: Path) -> None:
    case = synthetic
    module = binder.diagnostic
    previous_collect = module.r6._collect_episode
    previous_factory = module.r6.build_environment
    previous_run = module.run_diagnostic
    previous_pool = module._pools
    old_argv = sys.argv
    collected = []
    created = []

    def fake_factory(*_args: object) -> object:
        created.append("synthetic environment")
        return object()

    def fake_collect(*_args: object, **kwargs: object) -> tuple:
        collected.append(kwargs["repeat"])
        if kwargs["repeat"] == 0:
            target = case.root / relative
            target.write_bytes(target.read_bytes() + b"\n")
        return {}, {}

    def fake_run(**_kwargs: object) -> dict:
        env = module.r6.build_environment(1234, 2000)
        for repeat in (0, 1):
            module.r6._collect_episode(env, {}, {}, repeat=repeat,
                                       episode_id=0, max_steps=2000, root=case.root)
        return {}

    def fake_main() -> int:
        module.run_diagnostic(root=case.root,
                              protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                              output_root=case.root / binder.OUTPUT_ROOT)
        return 0

    monkeypatch.setattr(module.r6, "build_environment", fake_factory)
    monkeypatch.setattr(module.r6, "_collect_episode", fake_collect)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    monkeypatch.setattr(module, "main", fake_main)
    with pytest.raises(binder.DiagnosticGateError, match="pinned postrun input changed"):
        _invoke(case, preflight=False)
    assert created == ["synthetic environment"]
    assert collected == [0]  # Repeat 1's collector/reset was never entered.
    assert module.r6.build_environment is fake_factory
    assert module.r6._collect_episode is fake_collect
    assert module.run_diagnostic is fake_run
    assert module._pools is previous_pool
    assert sys.argv is old_argv
    assert not (case.root / binder.OUTPUT_ROOT).exists()
    monkeypatch.setattr(module.r6, "_collect_episode", previous_collect)
    monkeypatch.setattr(module.r6, "build_environment", previous_factory)
    monkeypatch.setattr(module, "run_diagnostic", previous_run)


def test_all_192_episode_calls_and_96_factories_recheck_without_real_environment(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic
    checked = []
    built = []
    collected = []
    original_check = binder._handoff_check

    def checked_handoff(root: Path, postrun: dict, postrun_sha: str, receipt_sha: str) -> None:
        original_check(root, postrun, postrun_sha, receipt_sha)
        checked.append("hashes")

    def fake_factory(*_args: object) -> object:
        built.append("synthetic factory")
        return object()

    def fake_collect(*_args: object, **kwargs: object) -> tuple:
        collected.append(kwargs["repeat"])
        return {}, {}

    def fake_run(**_kwargs: object) -> dict:
        for _ in range(96):
            env = module.r6.build_environment(1234, 2000)
            for repeat in (0, 1):
                module.r6._collect_episode(env, {}, {}, repeat=repeat,
                                           episode_id=0, max_steps=2000, root=case.root)
        return {}

    monkeypatch.setattr(binder, "_handoff_check", checked_handoff)
    monkeypatch.setattr(module.r6, "build_environment", fake_factory)
    monkeypatch.setattr(module.r6, "_collect_episode", fake_collect)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    with binder._diagnostic_context(case.root, case.postrun,
                                    _sha(_bytes(case.postrun)), case.training,
                                    _sha(_bytes(case.receipt))):
        module.run_diagnostic(root=case.root,
                              protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                              output_root=case.root / binder.OUTPUT_ROOT)
    assert len(built) == 96
    assert collected == [0, 1] * 96
    assert len(checked) == 290  # Dispatch + 96 factories + 192 resets + return.
    assert module.r6.build_environment is fake_factory
    assert module.r6._collect_episode is fake_collect
    assert not (case.root / binder.OUTPUT_ROOT).exists()


def test_underlying_collector_error_survives_scoped_restoration(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic

    def broken_collect(*_args: object, **_kwargs: object) -> None:
        raise ValueError("synthetic collector failed")

    def fake_run(**_kwargs: object) -> dict:
        module.r6._collect_episode(object(), {}, {}, repeat=0,
                                   episode_id=0, max_steps=2000, root=case.root)
        return {}

    monkeypatch.setattr(module.r6, "_collect_episode", broken_collect)
    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    with pytest.raises(ValueError, match="synthetic collector failed"):
        with binder._diagnostic_context(case.root, case.postrun,
                                        _sha(_bytes(case.postrun)), case.training,
                                        _sha(_bytes(case.receipt))):
            module.run_diagnostic(root=case.root,
                                  protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                                  output_root=case.root / binder.OUTPUT_ROOT)
    assert module.r6._collect_episode is broken_collect
    assert module.run_diagnostic is fake_run
    assert not (case.root / binder.OUTPUT_ROOT).exists()


def test_only_r2_output_guard_is_relaxed_and_every_context_restores_on_exception(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    case = synthetic
    module = binder.diagnostic
    original_pool = module._pools
    original_run = module.run_diagnostic
    original_prepare = module._prepare
    original_audit = module._sample_audit
    original_require = module._require
    original_study = module.STUDY_ID
    original_root = module.RUN_ROOT
    original_receipt = module.SAMPLE_AUDIT
    original_collect = module.r6._collect_episode
    original_argv = sys.argv
    postrun_sha = _sha(_bytes(case.postrun))
    receipt_sha = _sha(_bytes(case.receipt))

    def fake_run(**kwargs: object) -> dict:
        del kwargs
        module._require(False, binder.OUTPUT_GUARD)
        with pytest.raises(module.DiagnosticError):
            module._require(False, "frozen unrelated gate")
        raise RuntimeError("simulated downstream failure")

    monkeypatch.setattr(module, "run_diagnostic", fake_run)
    with pytest.raises(RuntimeError, match="downstream failure"):
        with binder._diagnostic_context(case.root, case.postrun, postrun_sha,
                                        case.training, receipt_sha):
            with pytest.raises(binder.DiagnosticGateError, match="only use the r2 output"):
                module.run_diagnostic(root=case.root,
                                      protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                                      output_root=case.root / binder.reconstruction.DIAGNOSTIC_OUTPUT_ROOT)
            with pytest.raises(module.DiagnosticError, match=binder.OUTPUT_GUARD):
                module._require(False, binder.OUTPUT_GUARD)
            with pytest.raises(binder.DiagnosticGateError, match="collector called outside"):
                module.r6._collect_episode(object(), {}, {}, repeat=0,
                                           episode_id=0, max_steps=2000, root=case.root)
            with pytest.raises(binder.DiagnosticGateError, match="may not substitute"):
                module.run_diagnostic(root=case.root,
                                      protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                                      output_root=case.root / binder.OUTPUT_ROOT,
                                      environment_factory=lambda *_: None)
            module.run_diagnostic(root=case.root,
                                  protocol_path=case.root / binder.reconstruction.PROTOCOL_PATH,
                                  output_root=case.root / binder.OUTPUT_ROOT)
    assert module._pools is original_pool
    assert module.run_diagnostic is fake_run
    assert module._prepare is original_prepare
    assert module._sample_audit is original_audit
    assert module._require is original_require
    assert module.STUDY_ID == original_study
    assert module.RUN_ROOT == original_root
    assert module.SAMPLE_AUDIT == original_receipt
    assert module.r6._collect_episode is original_collect
    assert sys.argv is original_argv
    monkeypatch.setattr(module, "run_diagnostic", original_run)


def test_cli_restores_argv_if_frozen_diagnostic_fails(
        synthetic: Synthetic, monkeypatch: pytest.MonkeyPatch) -> None:
    old_argv = sys.argv
    original = binder.diagnostic._sample_audit

    def failure() -> None:
        assert binder.diagnostic.SAMPLE_AUDIT == binder.AUDIT_PATH
        assert "--preflight-only" in sys.argv
        raise RuntimeError("parent diagnostic failed before environment reset")

    monkeypatch.setattr(binder.diagnostic, "main", failure)
    with pytest.raises(RuntimeError, match="before environment reset"):
        _invoke(synthetic)
    assert sys.argv is old_argv
    assert binder.diagnostic._sample_audit is original
    assert not (synthetic.root / binder.OUTPUT_ROOT).exists()
