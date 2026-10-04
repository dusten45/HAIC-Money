"""Synthetic-only tests for the isolated DrQ postrun audit; no real audit/reset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_drq_final_source_reconstruction_r2 as r2


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def contract(tmp_path, monkeypatch):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / r2.reconstruction.RUN_ROOT).mkdir(parents=True)
    training = tmp_path / r2.reconstruction.PROTOCOL_PATH
    roles = [(seed, variant) for seed in (0, 1) for variant in r2.base.VARIANTS]
    training.write_text(json.dumps({"runs": [{"source_seed": seed, "variant": variant}
                                               for seed, variant in roles]}), encoding="utf-8")
    monkeypatch.setattr(r2, "TRAINING_PROTOCOL_SHA", _digest(training))
    monkeypatch.setattr(r2.reconstruction, "_validate_protocol", lambda root, protocol: None)
    monkeypatch.setattr(r2.reconstruction, "_validate_exposure_snapshot", lambda root, protocol: None)
    pins = {}
    for seed, variant in roles:
        role = f"learner-{seed}-{variant}-final_source"
        run = tmp_path / r2.reconstruction.RUN_ROOT / role
        run.mkdir()
        result = run / "result.json"
        result.write_text(json.dumps({"source_seed": seed, "variant": variant}), encoding="utf-8")
        pins[role] = _digest(result)
    sources = {}
    for name in r2.SOURCES:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        sources[name] = _digest(path)
    protocol = {
        "format": "haic-drq-final-source-postrun-protocol-r2",
        "study_id": r2.STUDY_ID,
        "training_protocol_path": r2.reconstruction.PROTOCOL_PATH,
        "training_protocol_sha256": r2.TRAINING_PROTOCOL_SHA,
        "audit_receipt_path": r2.RECEIPT_PATH,
        "diagnostic_output_root": r2.DIAGNOSTIC_OUTPUT_ROOT,
        "code_sha256": sources,
        "run_result_sha256": pins,
        "interpretation": "descriptive_reused_train_only",
        "original_audit_status": "failed_historical_warmup_parity",
        "original_audit_failure": r2.ORIGINAL_FAILURE,
        "diagnostic": dict(r2.DIAGNOSTIC_CONTRACT),
        "limitations": list(r2.LIMITATIONS),
    }
    path = tmp_path / r2.PROTOCOL_PATH
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return tmp_path, path, protocol, pins, roles


def _replay():
    state = {
        "frames": np.broadcast_to(np.zeros((1, 84, 84), dtype=np.uint8),
                                  (r2.base.WARMUP, 84, 84)),
        "actions": np.zeros((r2.base.WARMUP, 3), dtype=np.float32),
        "rewards": np.zeros(r2.base.WARMUP, dtype=np.float32),
        "terminated": np.zeros(r2.base.WARMUP, dtype=np.bool_),
        "truncated": np.zeros(r2.base.WARMUP, dtype=np.bool_),
        "terminal": np.zeros(r2.base.WARMUP, dtype=np.bool_),
        "episode_ids": np.zeros(r2.base.WARMUP, dtype=np.int64),
        "episode_steps": np.arange(r2.base.WARMUP, dtype=np.int64),
        "sequence_ids": np.arange(r2.base.WARMUP, dtype=np.int64),
        "boundary_observations": {},
    }
    return state


def _fake_base_audit(contract, monkeypatch, *, fail_after=None):
    root, _, _, pins, roles = contract
    child = _replay()
    child["frames"] = np.broadcast_to(np.ones((1, 84, 84), dtype=np.uint8),
                                      (r2.base.WARMUP, 84, 84))
    historical = _replay()

    def fake_audit(repo_root, training_path):
        assert repo_root == root
        assert training_path == root / r2.reconstruction.PROTOCOL_PATH
        assert r2.base.ROOT == Path(r2.reconstruction.RUN_ROOT)
        assert r2.base.PROTOCOL == Path(r2.reconstruction.PROTOCOL_PATH)
        for number, _ in enumerate(roles):
            r2.base.check_online_warmup(child, historical)
            if fail_after == number:
                raise ValueError("downstream sample-trace gate failed")
        return {
            "format": "haic-drq-final-source-sample-audit-v1",
            "passed": True, "no_environment_resets": True,
            "protocol_sha256": r2.TRAINING_PROTOCOL_SHA,
            "source_pool": {"0": {"verified_source_decisions": 100000},
                            "1": {"verified_source_decisions": 100000}},
            "runs": [
                {"source_seed": seed, "variant": variant,
                 "run_dir": f"{r2.reconstruction.RUN_ROOT}/learner-{seed}-{variant}-final_source",
                 "result_sha256": pins[f"learner-{seed}-{variant}-final_source"]}
                for seed, variant in roles
            ],
        }

    monkeypatch.setattr(r2.base, "audit", fake_audit)


@pytest.mark.parametrize("field,value", [
    ("format", "wrong"),
    ("study_id", "wrong"),
    ("training_protocol_path", "experiments/elsewhere.json"),
    ("training_protocol_sha256", "0" * 64),
    ("audit_receipt_path", "runs/other.json"),
    ("diagnostic_output_root", "runs/other"),
    ("interpretation", "causal"),
    ("original_audit_status", "passed"),
    ("original_audit_failure", "historical audit passed"),
])
def test_wrong_or_missing_postrun_identity_fails_before_base(contract, monkeypatch, field, value):
    root, path, protocol, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    protocol[field] = value
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="identity"):
        r2.audit(root, path)
    protocol.pop(field)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="identity"):
        r2.audit(root, path)


@pytest.mark.parametrize("field", list(r2.DIAGNOSTIC_CONTRACT))
def test_tampered_or_incomplete_diagnostic_contract_blocks_before_base(contract, monkeypatch, field):
    root, path, protocol, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    expected = protocol["diagnostic"][field]
    protocol["diagnostic"][field] = "BLIND" if field == "partition" else "changed"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="diagnostic contract"):
        r2.audit(root, path)
    protocol["diagnostic"][field] = expected
    protocol["diagnostic"].pop(field)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="diagnostic contract"):
        r2.audit(root, path)


@pytest.mark.parametrize("index", range(len(r2.LIMITATIONS)))
def test_tampered_or_incomplete_limitations_block_before_base(contract, monkeypatch, index):
    root, path, protocol, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    protocol["limitations"][index] = "Original audit passes; causal promotion is authorized."
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="limitations"):
        r2.audit(root, path)
    protocol["limitations"] = r2.LIMITATIONS[:index] + r2.LIMITATIONS[index + 1:]
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="limitations"):
        r2.audit(root, path)


def test_protocol_path_and_frozen_training_sha_are_pinned(contract, monkeypatch):
    root, path, _, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    with pytest.raises(r2.PostrunGateError, match="dedicated protocol path"):
        r2.audit(root, root / r2.reconstruction.PROTOCOL_PATH)
    (root / r2.reconstruction.PROTOCOL_PATH).write_text("changed", encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="training protocol SHA"):
        r2.audit(root, path)


def test_symlinked_postrun_protocol_or_code_source_is_rejected(contract, monkeypatch):
    root, path, protocol, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    replacement = path.with_name("postrun-protocol-copy.json")
    path.rename(replacement)
    path.symlink_to(replacement)
    with pytest.raises(r2.reconstruction.exposure_audit.AuditError, match="symlink"):
        r2.audit(root, path)
    path.unlink()
    replacement.rename(path)
    source = root / next(iter(protocol["code_sha256"]))
    replacement = source.with_name("postrun-code-copy.py")
    source.rename(replacement)
    source.symlink_to(replacement)
    with pytest.raises(r2.reconstruction.exposure_audit.AuditError, match="symlink"):
        r2.audit(root, path)


def test_missing_or_changed_source_pin_fails_before_base(contract, monkeypatch):
    root, path, protocol, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    missing = next(iter(r2.SOURCES))
    protocol["code_sha256"].pop(missing)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="code pin map"):
        r2.audit(root, path)
    protocol["code_sha256"][missing] = "0" * 64
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="source SHA mismatch"):
        r2.audit(root, path)


def test_missing_changed_or_symlinked_result_pin_fails_before_base(contract, monkeypatch):
    root, path, protocol, pins, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    role = next(iter(pins))
    expected = pins[role]
    protocol["run_result_sha256"].pop(role)
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="result SHA pins incomplete"):
        r2.audit(root, path)
    protocol["run_result_sha256"][role] = expected
    path.write_text(json.dumps(protocol), encoding="utf-8")
    result = root / r2.reconstruction.RUN_ROOT / role / "result.json"
    result.write_text("changed", encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="result SHA mismatch"):
        r2.audit(root, path)
    result.unlink()
    other = root / r2.reconstruction.RUN_ROOT / "learner-0-uniform-final_source" / "else.json"
    other.write_text("else", encoding="utf-8")
    result.symlink_to(other)
    with pytest.raises(r2.reconstruction.exposure_audit.AuditError, match="symlink"):
        r2.audit(root, path)


@pytest.mark.parametrize("output", [r2.ORIGINAL_RECEIPT, r2.RECEIPT_PATH,
                                     r2.DIAGNOSTIC_OUTPUT_ROOT])
def test_original_receipt_or_existing_destination_blocks_before_base(contract, monkeypatch, output):
    root, path, _, _, _ = contract
    monkeypatch.setattr(r2.base, "audit", lambda *args: pytest.fail("base audit called"))
    target = root / output
    target.write_text("not an r2 audit", encoding="utf-8")
    with pytest.raises(r2.PostrunGateError, match="already exists"):
        r2.audit(root, path)


def test_first_mismatch_and_boundary_stack_discrepancy_are_recorded():
    historical = _replay()
    child = _replay()
    frame = child["frames"].copy()
    frame[11, 7, 8] = 1
    child["frames"] = frame
    child["actions"][3, 0] = np.float32(.25)
    child["rewards"][12] = np.float32(1.0)
    for replay in (historical, child):
        replay["terminal"][30] = True
    historical["boundary_observations"][30] = np.zeros((4, 84, 84), dtype=np.uint8)
    child["boundary_observations"][30] = np.ones((4, 84, 84), dtype=np.uint8)
    differences = r2._warmup_differences(child, historical)
    assert differences["first_mismatch"] == {
        "frames": 11, "actions": 3, "rewards": 12, "terminated": None,
        "truncated": None, "terminal": None, "episode_ids": None,
        "episode_steps": None, "sequence_ids": None,
    }
    assert differences["boundary_equal"] is False
    assert differences["boundary_first_mismatch"] == 30


@pytest.mark.parametrize("malformed", ["missing", "truncated", "dtype", "boundary", "missing_stack"])
def test_missing_or_malformed_warmup_replay_fails_closed(malformed):
    child, historical = _replay(), _replay()
    if malformed == "missing":
        del child["actions"]
    elif malformed == "truncated":
        child["frames"] = child["frames"][:r2.base.WARMUP - 1]
    elif malformed == "dtype":
        child["rewards"] = child["rewards"].astype(np.float64)
    elif malformed == "boundary":
        child["boundary_observations"] = {2: np.zeros((84, 84), dtype=np.uint8)}
    else:
        child["terminal"][2] = True
    with pytest.raises(r2.PostrunGateError):
        r2._warmup_differences(child, historical)


def test_wrapper_and_child_context_restore_after_success(contract, monkeypatch):
    root, path, _, _, roles = contract
    _fake_base_audit(contract, monkeypatch)
    original = (r2.base.check_online_warmup, r2.base.old.require, r2.base.ROOT,
                r2.base.PROTOCOL, r2.base.RECEIPT)
    receipt = r2.audit(root, path)
    assert original == (r2.base.check_online_warmup, r2.base.old.require,
                        r2.base.ROOT, r2.base.PROTOCOL, r2.base.RECEIPT)
    assert receipt["original_audit_passed"] is False
    assert receipt["passed"] is True
    assert receipt["historical_warmup_comparison"]["matched"] is False
    assert len(receipt["historical_warmup_comparison"]["runs"]) == len(roles) == 6
    assert all(row["first_mismatch"]["frames"] == 0
               for row in receipt["historical_warmup_comparison"]["runs"])
    assert receipt["causal_comparison_eligible"] is False
    assert "NOT independent child environment" in receipt["scope"]
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


def test_wrapper_and_child_context_restore_after_downstream_failure(contract, monkeypatch):
    root, path, _, _, _ = contract
    _fake_base_audit(contract, monkeypatch, fail_after=0)
    original = (r2.base.check_online_warmup, r2.base.old.require, r2.base.ROOT,
                r2.base.PROTOCOL, r2.base.RECEIPT)
    with pytest.raises(ValueError, match="sample-trace gate failed"):
        r2.audit(root, path)
    assert original == (r2.base.check_online_warmup, r2.base.old.require,
                        r2.base.ROOT, r2.base.PROTOCOL, r2.base.RECEIPT)
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


def test_wrapper_restores_when_warmup_replay_is_malformed(contract, monkeypatch):
    root, path, _, _, _ = contract
    original = (r2.base.check_online_warmup, r2.base.old.require, r2.base.ROOT)

    def malformed_audit(*args):
        broken = _replay()
        del broken["frames"]
        r2.base.check_online_warmup(broken, _replay())
        pytest.fail("warmup wrapper accepted malformed replay")

    monkeypatch.setattr(r2.base, "audit", malformed_audit)
    with pytest.raises(r2.PostrunGateError, match="missing frames"):
        r2.audit(root, path)
    assert original == (r2.base.check_online_warmup, r2.base.old.require, r2.base.ROOT)
    assert not (root / r2.RECEIPT_PATH).exists()


@pytest.mark.parametrize("changed", ["protocol", "training", "source"])
def test_postrun_pins_are_rechecked_after_frozen_audit(contract, monkeypatch, changed):
    root, path, protocol, _, _ = contract
    _fake_base_audit(contract, monkeypatch)
    delegated = r2.base.audit

    def changes_during_audit(*args):
        result = delegated(*args)
        target = (path if changed == "protocol" else
                  root / r2.reconstruction.PROTOCOL_PATH if changed == "training" else
                  root / next(iter(protocol["code_sha256"])))
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return result

    monkeypatch.setattr(r2.base, "audit", changes_during_audit)
    with pytest.raises(r2.PostrunGateError, match="changed during sample audit"):
        r2.main(["--repo-root", str(root)])
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


@pytest.mark.parametrize("seed,variant", [(seed, variant) for seed in (0, 1)
                                          for variant in r2.base.VARIANTS])
def test_changed_result_during_delegation_blocks_receipt(contract, monkeypatch, seed, variant):
    root, _, _, pins, _ = contract
    _fake_base_audit(contract, monkeypatch)
    delegated = r2.base.audit
    role = f"learner-{seed}-{variant}-final_source"
    target = root / r2.reconstruction.RUN_ROOT / role / "result.json"

    def changes_during_audit(*args):
        result = delegated(*args)
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return result

    monkeypatch.setattr(r2.base, "audit", changes_during_audit)
    with pytest.raises(r2.PostrunGateError, match=f"result changed during sample audit: {role}"):
        r2.main(["--repo-root", str(root)])
    assert pins[role] != _digest(target)
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


def test_no_receipt_if_base_skips_an_arm_or_changes_result(contract, monkeypatch):
    root, path, _, _, roles = contract
    _fake_base_audit(contract, monkeypatch)
    actual = r2.base.audit

    def wrong_result(*args):
        result = actual(*args)
        result["runs"][0]["result_sha256"] = "0" * 64
        return result

    monkeypatch.setattr(r2.base, "audit", wrong_result)
    with pytest.raises(r2.PostrunGateError, match="pinned arm"):
        r2.audit(root, path)

    def skipped_arm(*args):
        result = actual(*args)
        result["runs"].pop()
        return result

    monkeypatch.setattr(r2.base, "audit", skipped_arm)
    with pytest.raises(r2.PostrunGateError, match="incomplete arm evidence"):
        r2.audit(root, path)
    assert not (root / r2.RECEIPT_PATH).exists()


def test_main_exclusively_writes_unique_r2_receipt_without_original_pass(contract, monkeypatch):
    root, _, _, _, _ = contract
    _fake_base_audit(contract, monkeypatch)
    assert r2.main(["--repo-root", str(root)]) == 0
    receipt = json.loads((root / r2.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["format"] == "haic-drq-final-source-postrun-sample-audit-r2"
    assert receipt["passed"] is True and receipt["original_audit_passed"] is False
    assert receipt["no_environment_resets"] is True
    assert receipt["interpretation"] == "descriptive_reused_train_only"
    assert receipt["causal_comparison_eligible"] is False
    assert "old r7b" in receipt["disclaimer"]
    assert not (root / r2.ORIGINAL_RECEIPT).exists()
    with pytest.raises(r2.PostrunGateError, match="already exists"):
        r2.main(["--repo-root", str(root)])


def test_exclusive_create_refuses_output_created_after_checks(contract, monkeypatch):
    root, _, _, _, _ = contract
    _fake_base_audit(contract, monkeypatch)
    original = r2._absent_outputs
    count = 0

    def race(repo_root):
        nonlocal count
        count += 1
        original(repo_root)
        if count == 3:
            (repo_root / r2.RECEIPT_PATH).write_text("other process", encoding="utf-8")

    monkeypatch.setattr(r2, "_absent_outputs", race)
    with pytest.raises(FileExistsError):
        r2.main(["--repo-root", str(root)])
    assert (root / r2.RECEIPT_PATH).read_text(encoding="utf-8") == "other process"
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


@pytest.mark.parametrize("ancestor", ["runs", "run_root"])
def test_symlink_swap_of_receipt_ancestor_blocks_creation(contract, monkeypatch, ancestor):
    root, _, _, _, _ = contract
    _fake_base_audit(contract, monkeypatch)
    original = r2._absent_outputs
    count = 0
    open_fds_before = len(list(Path("/proc/self/fd").iterdir()))

    def swap_after_checks(repo_root):
        nonlocal count
        count += 1
        original(repo_root)
        if count == 3:
            directory = (repo_root / "runs" if ancestor == "runs"
                         else repo_root / r2.reconstruction.RUN_ROOT)
            detached = directory.with_name(f"detached-{directory.name}")
            directory.rename(detached)
            directory.symlink_to(detached, target_is_directory=True)

    monkeypatch.setattr(r2, "_absent_outputs", swap_after_checks)
    with pytest.raises(r2.PostrunGateError, match="unsafe receipt directory"):
        r2.main(["--repo-root", str(root)])
    assert len(list(Path("/proc/self/fd").iterdir())) == open_fds_before
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (root / r2.ORIGINAL_RECEIPT).exists()


def test_ancestor_swap_during_leaf_open_unlinks_own_receipt(contract, monkeypatch):
    root, _, _, _, _ = contract
    _fake_base_audit(contract, monkeypatch)
    original_open = r2.os.open
    moved = root / "detached-runs"
    open_fds_before = len(list(Path("/proc/self/fd").iterdir()))

    def swap_before_leaf_open(name, flags, mode=0o777, *, dir_fd=None):
        if name == Path(r2.RECEIPT_PATH).name and flags & r2.os.O_EXCL:
            directory = root / "runs"
            directory.rename(moved)
            directory.symlink_to(moved, target_is_directory=True)
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(r2.os, "open", swap_before_leaf_open)
    with pytest.raises(r2.PostrunGateError, match="receipt directory changed"):
        r2.main(["--repo-root", str(root)])
    assert len(list(Path("/proc/self/fd").iterdir())) == open_fds_before
    assert not (root / r2.RECEIPT_PATH).exists()
    assert not (moved / Path(r2.reconstruction.RUN_ROOT).name / Path(r2.RECEIPT_PATH).name).exists()
