"""Read-only postrun provenance audit; never certify the failed original warmup gate.

The separately pinned postrun protocol permits descriptive reused-TRAIN diagnosis
after recording the historical r7b warmup discrepancy for all six child arms.
Every other check is delegated to the frozen sample auditor without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from scripts import audit_drq_final_replay_samples as base
from scripts import audit_drq_final_source_reconstruction_samples as child
from scripts import train_drq_final_source_reconstruction as reconstruction


PROTOCOL_PATH = "experiments/drqv2-final-source-replay-reconstruction-postrun-r2.json"
TRAINING_PROTOCOL_SHA = "03592f38cc24b9d94e51b3ae0c7688b1881ea03a8d17216d3d1289e88bb1a20a"
STUDY_ID = "drqv2-final-source-replay-reconstruction-postrun-r2"
RECEIPT_PATH = f"{reconstruction.RUN_ROOT}/pre-evaluation-postrun-r2-sample-audit.json"
DIAGNOSTIC_OUTPUT_ROOT = f"{reconstruction.RUN_ROOT}/train-diagnostic-postrun-r2"
ORIGINAL_RECEIPT = reconstruction.SAMPLE_AUDIT_PATH
ORIGINAL_FAILURE = "online warmup differs from paired evolving-source control: frames"
DIAGNOSTIC_CONTRACT = {
    "partition": "TRAIN-DIAGNOSTIC",
    "track_id": 1,
    "already_consumed_roads": 16,
    "treatment_actors": 6,
    "deterministic_repeats": 2,
    "episodes": 192,
    "canonical_pairs": 96,
    "fixed_descriptive_signal_per_mixture": "kept >=9/11 AND gained >=2/21",
}
LIMITATIONS = [
    "The original six-arm sample audit failed and its receipt remains absent; this protocol does not amend or pass it.",
    "Reconstruction and historical r7b collectors ran on different GPUs and their update-free warmup trajectories diverged; replay-only causality versus historical r7b cannot be established.",
    "The postrun receipt attests source/online metadata, n-step, sampling and checkpoint lineage, not independent child environment pixel/action/reward replay parity.",
    "All roads are previously consumed development TRAIN-DIAGNOSTIC roads; repeated runs are not new independent road evidence.",
    "Raw reward and retention signals are internal proxies, not official HAIC rankings. No candidate promotion, confirmation, blind, model confirmation or official submission is authorized.",
]
SOURCES = {
    "scripts/audit_drq_final_source_reconstruction_r2.py",
    "scripts/diagnose_drq_final_source_reconstruction_r2.py",
}
FIELDS = {
    "frames": ((84, 84), np.uint8),
    "actions": ((3,), np.float32),
    "rewards": ((), np.float32),
    "terminated": ((), np.bool_),
    "truncated": ((), np.bool_),
    "terminal": ((), np.bool_),
    "episode_ids": ((), np.int64),
    "episode_steps": ((), np.int64),
    "sequence_ids": ((), np.int64),
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PostrunGateError(ValueError):
    """A postrun-only identity, source, or provenance condition failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostrunGateError(message)


def _file(root: Path, name: str) -> Path:
    # Unlike the parent auditor, also reject symlinks in ancestor directories.
    return reconstruction.exposure_audit._safe_file(root, name)


def _absent_outputs(root: Path) -> None:
    for name in (ORIGINAL_RECEIPT, RECEIPT_PATH, DIAGNOSTIC_OUTPUT_ROOT):
        path = root / name
        _require(not path.exists() and not path.is_symlink(), f"audit/diagnostic output already exists: {name}")
    directory = root / reconstruction.RUN_ROOT
    _require(directory.is_dir() and not directory.is_symlink(), "child run root missing or symlinked")


def _preflight(root: Path, protocol_path: Path) -> tuple[dict[str, Any], str, dict[str, str], dict[str, str]]:
    _require(protocol_path == root / PROTOCOL_PATH, "postrun auditor requires its dedicated protocol path")
    postrun_file = _file(root, PROTOCOL_PATH)
    raw = postrun_file.read_bytes()
    protocol = reconstruction.exposure_audit._json(raw, PROTOCOL_PATH)
    postrun_sha = hashlib.sha256(raw).hexdigest()
    _require(protocol.get("format") == "haic-drq-final-source-postrun-protocol-r2"
             and protocol.get("study_id") == STUDY_ID
             and protocol.get("training_protocol_path") == reconstruction.PROTOCOL_PATH
             and protocol.get("training_protocol_sha256") == TRAINING_PROTOCOL_SHA
             and protocol.get("audit_receipt_path") == RECEIPT_PATH
             and protocol.get("diagnostic_output_root") == DIAGNOSTIC_OUTPUT_ROOT
             and protocol.get("interpretation") == "descriptive_reused_train_only"
             and protocol.get("original_audit_status") == "failed_historical_warmup_parity"
             and protocol.get("original_audit_failure") == ORIGINAL_FAILURE,
             "postrun protocol identity, paths, or descriptive status changed")
    _require(protocol.get("diagnostic") == DIAGNOSTIC_CONTRACT,
             "postrun diagnostic contract changed")
    _require(protocol.get("limitations") == LIMITATIONS,
             "postrun descriptive limitations changed")
    _absent_outputs(root)
    training_path = _file(root, reconstruction.PROTOCOL_PATH)
    _require(base.old.digest(training_path) == TRAINING_PROTOCOL_SHA, "frozen training protocol SHA changed")
    training = reconstruction.exposure_audit._json(training_path.read_bytes(), reconstruction.PROTOCOL_PATH)
    reconstruction._validate_protocol(root, training)
    reconstruction._validate_exposure_snapshot(root, training)
    sources = protocol.get("code_sha256")
    if type(sources) is not dict or set(sources) != SOURCES:
        raise PostrunGateError("postrun code pin map incomplete or changed")
    for name, expected in sources.items():
        _require(type(expected) is str and SHA256.fullmatch(expected) is not None
                 and base.old.digest(_file(root, name)) == expected,
                 f"postrun source SHA mismatch: {name}")
    expected_runs = {
        f"learner-{seed}-{variant}-final_source"
        for seed in (0, 1) for variant in base.VARIANTS
    }
    pins = protocol.get("run_result_sha256")
    if type(pins) is not dict or set(pins) != expected_runs:
        raise PostrunGateError("six result SHA pins incomplete or changed")
    for role, expected in pins.items():
        name = f"{reconstruction.RUN_ROOT}/{role}/result.json"
        _require(type(expected) is str and SHA256.fullmatch(expected) is not None
                 and base.old.digest(_file(root, name)) == expected,
                 f"frozen run result SHA mismatch: {role}")
    return training, postrun_sha, sources, pins


def _warmup_differences(new: dict, control: dict) -> dict[str, Any]:
    first: dict[str, int | None] = {}
    for key, (tail, dtype) in FIELDS.items():
        arrays = []
        for label, replay in (("child", new), ("historical", control)):
            _require(type(replay) is dict and key in replay, f"{label} warmup replay missing {key}")
            value = replay[key]
            _require(isinstance(value, np.ndarray) and value.ndim == 1 + len(tail)
                     and value.shape[0] >= base.WARMUP and value.shape[1:] == tail
                     and value.dtype == dtype,
                     f"{label} warmup replay malformed {key}")
            if key in ("actions", "rewards"):
                _require(bool(np.isfinite(value[:base.WARMUP]).all()),
                         f"{label} warmup replay nonfinite {key}")
            arrays.append(value[:base.WARMUP])
        equal = arrays[0] == arrays[1]
        if equal.ndim > 1:
            equal = equal.all(axis=tuple(range(1, equal.ndim)))
        unequal = np.flatnonzero(~equal)
        first[key] = int(unequal[0]) if len(unequal) else None

    boundaries = []
    for label, replay in (("child", new), ("historical", control)):
        value = replay.get("boundary_observations")
        if type(value) is not dict:
            raise PostrunGateError(f"{label} warmup boundary stacks missing/malformed")
        for index, stack in value.items():
            _require(type(index) is int and 0 <= index < len(replay["terminal"])
                     and isinstance(stack, np.ndarray) and stack.shape == (4, 84, 84)
                     and stack.dtype == np.uint8,
                     f"{label} warmup boundary stack malformed at {index}")
        warmup = {key: stack for key, stack in value.items() if key < base.WARMUP}
        done = np.flatnonzero(replay["terminal"][:base.WARMUP]
                              | replay["terminated"][:base.WARMUP]
                              | replay["truncated"][:base.WARMUP])
        _require(set(warmup) == set(map(int, done)),
                 f"{label} warmup terminal/boundary stacks disagree")
        boundaries.append(warmup)
    discrepancy = [key for key in sorted(boundaries[0].keys() | boundaries[1].keys())
                   if key not in boundaries[0] or key not in boundaries[1]
                   or not np.array_equal(boundaries[0][key], boundaries[1][key])]
    return {"first_mismatch": first, "boundary_equal": not discrepancy,
            "boundary_first_mismatch": discrepancy[0] if discrepancy else None}


def audit(root: Path, protocol_path: Path) -> dict[str, Any]:
    """Call the frozen audit, replacing only its historical warmup comparator."""
    training, postrun_sha, sources, pins = _preflight(root, protocol_path)
    order = [(row["source_seed"], row["variant"])
             for seed in (0, 1) for row in training["runs"] if row["source_seed"] == seed]
    _require(len(order) == 6 and set(order) == {(seed, variant) for seed in (0, 1)
                                                for variant in base.VARIANTS},
             "training protocol has an incomplete six-arm matrix")
    comparisons: list[dict[str, Any]] = []

    def record_warmup(new: dict, control: dict) -> None:
        _require(len(comparisons) < len(order), "frozen auditor compared too many warmup arms")
        seed, variant = order[len(comparisons)]
        detail = _warmup_differences(new, control)
        _require(detail["first_mismatch"]["frames"] is not None,
                 f"historical pixel discrepancy missing in seed{seed}/{variant}")
        comparisons.append({"source_seed": seed, "variant": variant, **detail})

    with child._child_audit_context(training):
        original = base.check_online_warmup
        base.check_online_warmup = record_warmup
        try:
            result = base.audit(root, root / reconstruction.PROTOCOL_PATH)
        finally:
            base.check_online_warmup = original
    _require(len(comparisons) == 6, "frozen sample auditor did not check all six warmup arms")
    _require(type(result) is dict and result.get("format") == "haic-drq-final-source-sample-audit-v1"
             and result.get("passed") is True and result.get("no_environment_resets") is True
             and result.get("protocol_sha256") == TRAINING_PROTOCOL_SHA,
             "frozen auditor did not return complete read-only sample evidence")
    runs = result.get("runs")
    _require(type(result.get("source_pool")) is dict and set(result["source_pool"]) == {"0", "1"},
             "frozen auditor returned incomplete source pool evidence")
    if type(runs) is not list or len(runs) != 6:
        raise PostrunGateError("frozen auditor returned incomplete arm evidence")
    roles = {}
    for row in runs:
        if type(row) is not dict:
            raise PostrunGateError("malformed frozen arm evidence")
        _require(type(row.get("source_seed")) is int
                 and type(row.get("variant")) is str, "malformed frozen arm evidence")
        role = f"learner-{row['source_seed']}-{row['variant']}-final_source"
        _require(role in pins and role not in roles
                 and row.get("run_dir") == f"{reconstruction.RUN_ROOT}/{role}"
                 and row.get("result_sha256") == pins[role],
                 f"frozen audit result no longer binds pinned arm: {role}")
        roles[role] = row
    _require(set(roles) == set(pins), "frozen auditor omitted a pinned result")
    _require(base.old.digest(_file(root, PROTOCOL_PATH)) == postrun_sha,
             "postrun protocol changed during sample audit")
    _require(base.old.digest(_file(root, reconstruction.PROTOCOL_PATH)) == TRAINING_PROTOCOL_SHA,
             "frozen training protocol changed during sample audit")
    for name, expected in sources.items():
        _require(base.old.digest(_file(root, name)) == expected,
                 f"postrun source changed during sample audit: {name}")
    for role, expected in pins.items():
        name = f"{reconstruction.RUN_ROOT}/{role}/result.json"
        _require(base.old.digest(_file(root, name)) == expected,
                 f"frozen run result changed during sample audit: {role}")
    _absent_outputs(root)
    return {
        "format": "haic-drq-final-source-postrun-sample-audit-r2",
        "passed": True,
        "original_audit_passed": False,
        "no_environment_resets": True,
        "postrun_protocol_sha256": postrun_sha,
        "training_protocol_sha256": TRAINING_PROTOCOL_SHA,
        "interpretation": "descriptive_reused_train_only",
        "causal_comparison_eligible": False,
        "historical_warmup_comparison": {
            "matched": False, "warmup_steps": base.WARMUP, "runs": comparisons,
        },
        "source_pool": result["source_pool"],
        "runs": runs,
        "scope": "Frozen source/online metadata, n-step, sample, and checkpoint lineage; "
                 "NOT independent child environment pixel/action/reward replay parity",
        "disclaimer": "Historical r7b warmup parity failed; this is not an original audit pass "
                      "or a causal replay-only comparison to old r7b. Child warmup physical "
                      "pixel/action/reward parity to the environment remains unproven.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=Path(PROTOCOL_PATH))
    args = parser.parse_args(argv)
    _require(args.repo_root.is_dir() and not args.repo_root.is_symlink(), "unsafe repository root")
    root = args.repo_root.resolve(strict=True)
    receipt = audit(root, root / args.protocol)
    _absent_outputs(root)
    raw = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    run_parts = Path(reconstruction.RUN_ROOT).parts
    _require(len(run_parts) == 2 and run_parts[0] == "runs"
             and Path(RECEIPT_PATH).parent.as_posix() == reconstruction.RUN_ROOT,
             "unsafe receipt directory identity")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC

    with ExitStack() as stack:
        def open_directory(name: str | Path, parent_fd: int | None = None) -> int:
            try:
                descriptor = os.open(name, directory_flags, dir_fd=parent_fd)
            except OSError as exc:
                raise PostrunGateError(f"unsafe receipt directory: {name}") from exc
            stack.callback(os.close, descriptor)
            return descriptor

        root_fd = open_directory(root)
        runs_fd = open_directory("runs", root_fd)
        run_fd = open_directory(run_parts[1], runs_fd)

        def check_directories() -> None:
            for descriptor, name, parent_fd in (
                (root_fd, root, None),
                (runs_fd, "runs", root_fd),
                (run_fd, run_parts[1], runs_fd),
            ):
                actual = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(descriptor)
                _require(stat.S_ISDIR(actual.st_mode)
                         and (actual.st_dev, actual.st_ino) == (opened.st_dev, opened.st_ino),
                         f"receipt directory changed: {name}")

        def check_other_outputs() -> None:
            for name in (Path(ORIGINAL_RECEIPT).name, Path(DIAGNOSTIC_OUTPUT_ROOT).name):
                try:
                    os.stat(name, dir_fd=run_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                raise PostrunGateError(f"audit/diagnostic output already exists: {name}")

        check_directories()
        check_other_outputs()
        name = Path(RECEIPT_PATH).name
        created = None
        try:
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                                 0o644, dir_fd=run_fd)
            stack.callback(os.close, descriptor)
            created = os.fstat(descriptor)
            offset = 0
            while offset < len(raw):
                count = os.write(descriptor, raw[offset:])
                _require(count > 0, "postrun receipt write incomplete")
                offset += count
            os.fsync(descriptor)
            check_directories()
            check_other_outputs()
            current = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
            _require((current.st_dev, current.st_ino) == (created.st_dev, created.st_ino),
                     "postrun receipt path changed during creation")
            os.fsync(run_fd)
        except BaseException:
            if created is not None:
                try:
                    current = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                        os.unlink(name, dir_fd=run_fd)
            raise
    print(json.dumps({"passed": True, "original_audit_passed": False,
                      "arms": len(receipt["runs"]), "receipt": RECEIPT_PATH,
                      "sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PostrunGateError, reconstruction.GateError,
            reconstruction.exposure_audit.AuditError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
