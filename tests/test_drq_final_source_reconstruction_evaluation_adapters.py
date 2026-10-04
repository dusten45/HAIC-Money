from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_drq_final_source_reconstruction_samples as sample_adapter
from scripts import diagnose_drq_final_source_reconstruction as diagnostic_adapter


def _child_protocol() -> dict:
    return {
        "format": "haic-drq-final-source-replay-study-v1",
        "study_id": sample_adapter.STUDY_ID,
        "run_root": sample_adapter.RUN_ROOT,
        "r6_protocol_sha256": sample_adapter.sample_audit.R6_SHA,
        "r7_protocol_sha256": sample_adapter.sample_audit.R7_SHA,
        "lambda_preserve": 0.5,
    }


def test_sample_audit_adapter_accepts_only_child_identity():
    protocol = _child_protocol()

    assert sample_adapter._child_protocol_identity(protocol)
    protocol["study_id"] = sample_adapter.PARENT_STUDY_ID
    assert not sample_adapter._child_protocol_identity(protocol)
    protocol["study_id"] = sample_adapter.STUDY_ID
    protocol["r7_protocol_sha256"] = "0" * 64
    assert not sample_adapter._child_protocol_identity(protocol)


def test_sample_audit_adapter_changes_only_the_identity_guard():
    protocol = _child_protocol()
    parent = sample_adapter.sample_audit
    original = (parent.ROOT, parent.PROTOCOL, parent.RECEIPT, parent.old.require)

    with sample_adapter._child_audit_context(protocol):
        assert parent.ROOT == Path(sample_adapter.RUN_ROOT)
        assert parent.PROTOCOL == Path(sample_adapter.PROTOCOL_PATH)
        assert parent.RECEIPT == Path(sample_adapter.RECEIPT_PATH)
        parent.old.require(False, sample_adapter.IDENTITY_GUARD)
        with pytest.raises(ValueError):
            parent.old.require(False, "unrelated invariant")

    assert (parent.ROOT, parent.PROTOCOL, parent.RECEIPT, parent.old.require) == original


def test_diagnostic_adapter_uses_parent_identity_only_for_source_pool_validation(monkeypatch):
    module = diagnostic_adapter.diagnostic
    observed = {}
    child_study = diagnostic_adapter.STUDY_ID
    child_root = Path(diagnostic_adapter.RUN_ROOT)

    def fake_pools(root, protocol, sources, diagnostic_roads):
        observed["study_id"] = module.STUDY_ID
        observed["run_root"] = module.RUN_ROOT
        return {0: {}, 1: {}}

    monkeypatch.setattr(module, "STUDY_ID", child_study)
    monkeypatch.setattr(module, "RUN_ROOT", child_root)
    monkeypatch.setattr(module, "_pools", fake_pools)

    with diagnostic_adapter._parent_collection_identity_context():
        assert module._pools(Path("."), {}, {}, set()) == {0: {}, 1: {}}
        assert observed == {
            "study_id": diagnostic_adapter.PARENT_STUDY_ID,
            "run_root": Path(diagnostic_adapter.PARENT_RUN_ROOT),
        }

    assert module.STUDY_ID == child_study
    assert module.RUN_ROOT == child_root
    assert module._pools is fake_pools
