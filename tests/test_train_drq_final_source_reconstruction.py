from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import train_drq_final_source_reconstruction as launcher


REPO_ROOT = Path(__file__).resolve().parents[1]


def _frozen_document(root: Path) -> dict:
    experiments = root / "experiments"
    scripts = root / "scripts"
    run_root = root / launcher.RUN_ROOT
    (experiments).mkdir(parents=True)
    (scripts).mkdir(parents=True)
    run_root.mkdir(parents=True)
    parent_path = REPO_ROOT / launcher.exposure_audit.PARENT_PROTOCOL
    parent_bytes = parent_path.read_bytes()
    (experiments / Path(launcher.exposure_audit.PARENT_PROTOCOL).name).write_bytes(parent_bytes)
    source = REPO_ROOT / "scripts/train_drq_final_source_replay.py"
    shutil.copyfile(source, scripts / source.name)
    reconstruction_sources = (
        "scripts/audit_drq_final_source_reconstruction.py",
        "scripts/train_drq_final_source_reconstruction.py",
        "scripts/audit_drq_final_source_reconstruction_samples.py",
        "scripts/diagnose_drq_final_source_reconstruction.py",
    )
    for name in reconstruction_sources:
        source = REPO_ROOT / name
        shutil.copyfile(source, root / name)
    receipt_bytes = b"{}\n"
    (run_root / "preflight-exposure-audit.json").write_bytes(receipt_bytes)
    failure_source = REPO_ROOT / launcher.PREFLIGHT_FAILURE_RECEIPT
    failure_target = root / launcher.PREFLIGHT_FAILURE_RECEIPT
    shutil.copyfile(failure_source, failure_target)
    document = json.loads(parent_bytes)
    document["study_id"] = launcher.STUDY_ID
    document["run_root"] = launcher.RUN_ROOT
    document["code_sha256"].update({
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in reconstruction_sources
    })
    for row in document["runs"]:
        seed, variant = row["source_seed"], row["variant"]
        row["run_dir"] = f"{launcher.RUN_ROOT}/learner-{seed}-{variant}-final_source"
    document["reconstruction"] = {
        "method": "fresh_matched_six_arm_from_source_forks",
        "parent_protocol_sha256": launcher.PARENT_PROTOCOL_SHA,
        "protocol_revision": {
            "revision": 2,
            "supersedes_protocol_sha256": launcher.SUPERSEDED_PROTOCOL_SHA,
            "preflight_failure_receipt": {
                "path": launcher.PREFLIGHT_FAILURE_RECEIPT,
                "sha256": launcher.PREFLIGHT_FAILURE_RECEIPT_SHA,
            },
        },
        "partition": "TRAIN",
        "fresh_geometry_seeds": [],
        "all_six_arms_rerun": True,
        "partial_arm_resume": False,
        "postrun_gates": {
            "sample_audit_path": launcher.SAMPLE_AUDIT_PATH,
            "sample_audit_format": "haic-drq-final-source-sample-audit-v1",
            "diagnostic_output_root": launcher.DIAGNOSTIC_OUTPUT_ROOT,
            "diagnostic_partition": "TRAIN-DIAGNOSTIC",
            "diagnostic_episode_count": 192,
            "diagnostic_repeats": 2,
            "fixed_gate": "kept >=9/11 AND gained >=2/21",
        },
        "exposure_audit": {
            "path": launcher.EXPOSURE_RECEIPT,
            "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        },
        "training_runtime": {
            "python": launcher.REQUIRED_PYTHON,
            "packages": launcher.REQUIRED_PACKAGES,
            "torch_cuda_build": launcher.TORCH_CUDA_BUILD,
            "gpu_name": launcher.GPU_NAME,
            "gpu_compute_capability": list(launcher.GPU_CAPABILITY),
            "gpu_driver": launcher.GPU_DRIVER,
            "thread_env": {name: "1" for name in launcher.THREAD_ENV},
        },
        "resource_floors": launcher.RESOURCE_FLOORS,
    }
    return document


def test_reconstruction_protocol_preserves_parent_and_new_six_arm_paths(tmp_path):
    protocol = _frozen_document(tmp_path)

    launcher._validate_protocol(tmp_path, protocol)


def test_reconstruction_protocol_rejects_parent_treatment_change(tmp_path):
    protocol = _frozen_document(tmp_path)
    protocol["lambda_preserve"] = 0.0

    with pytest.raises(launcher.GateError, match="changed inherited field lambda_preserve"):
        launcher._validate_protocol(tmp_path, protocol)


def test_reconstruction_protocol_rejects_fresh_seed_claim(tmp_path):
    protocol = _frozen_document(tmp_path)
    protocol["reconstruction"]["fresh_geometry_seeds"] = [3910800001]

    with pytest.raises(launcher.GateError, match="method/exposure declaration changed"):
        launcher._validate_protocol(tmp_path, protocol)


def test_reconstruction_protocol_binds_the_zero_interaction_preflight_failure(tmp_path):
    protocol = _frozen_document(tmp_path)
    protocol["reconstruction"]["protocol_revision"]["supersedes_protocol_sha256"] = "0" * 64

    with pytest.raises(launcher.GateError, match="protocol revision does not preserve"):
        launcher._validate_protocol(tmp_path, protocol)


def test_pool_receipt_validation_uses_parent_collection_root_temporarily(monkeypatch):
    trainer = launcher.trainer
    child_root = Path(launcher.RUN_ROOT)
    parent_root = Path(launcher.exposure_audit.PARENT_RUN_ROOT)
    seen = []

    def fake_receipt(root, *args, **kwargs):
        seen.append(trainer.RUN_ROOT)
        return {}

    monkeypatch.setattr(trainer, "RUN_ROOT", child_root)
    monkeypatch.setattr(trainer, "_receipt", fake_receipt)

    with launcher._parent_pool_receipt_context():
        trainer._receipt(Path("."), {}, 0, {}, "catalog", "collection", "schedule")
        assert trainer.RUN_ROOT == child_root

    assert seen == [parent_root]
    assert trainer.RUN_ROOT == child_root
    assert trainer._receipt is fake_receipt
