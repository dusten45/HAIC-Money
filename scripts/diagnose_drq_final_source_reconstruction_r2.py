"""Bind the frozen CPU diagnostic to the distinct descriptive postrun r2 audit.

The failed historical-warmup sample audit remains failed. This adapter accepts
only the separately identified r2 receipt and writes only to the r2 output root.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, cast

from scripts import diagnose_drq_final_replay as diagnostic
from scripts import diagnose_drq_final_source_reconstruction as child
from scripts import train_drq_final_source_reconstruction as reconstruction


PROTOCOL_PATH = Path("experiments/drqv2-final-source-replay-reconstruction-postrun-r2.json")
STUDY_ID = "drqv2-final-source-replay-reconstruction-postrun-r2"
TRAINING_PROTOCOL_SHA = "03592f38cc24b9d94e51b3ae0c7688b1881ea03a8d17216d3d1289e88bb1a20a"
AUDIT_PATH = Path(reconstruction.RUN_ROOT) / "pre-evaluation-postrun-r2-sample-audit.json"
OUTPUT_ROOT = Path(reconstruction.RUN_ROOT) / "train-diagnostic-postrun-r2"
INTERPRETATION = "descriptive_reused_train_only"
ORIGINAL_FAILURE = "failed_historical_warmup_parity"
ORIGINAL_FAILURE_TEXT = "online warmup differs from paired evolving-source control: frames"
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
CODE_PATHS = {
    "scripts/audit_drq_final_source_reconstruction_r2.py",
    "scripts/diagnose_drq_final_source_reconstruction_r2.py",
}
WARMUP_FIELDS = {
    "frames", "actions", "rewards", "terminated", "truncated", "terminal",
    "episode_ids", "episode_steps", "sequence_ids",
}
OUTPUT_GUARD = "output must be the new final-source run root's train-diagnostic directory"
ROLES = {(seed, variant) for seed in (0, 1) for variant in diagnostic.r6.VARIANTS}


class DiagnosticGateError(RuntimeError):
    """The r2 protocol, receipt, or delegated diagnostic is unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticGateError(message)


def _read_json(root: Path, relative: Path) -> tuple[dict[str, Any], str]:
    path = reconstruction.exposure_audit._safe_file(root, relative.as_posix())
    raw = path.read_bytes()
    return (reconstruction.exposure_audit._json(raw, relative.as_posix()),
            hashlib.sha256(raw).hexdigest())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_sha(root: Path, relative: Path, expected: str) -> None:
    path = reconstruction.exposure_audit._safe_file(root, relative.as_posix())
    _require(_sha(path) == expected, f"pinned postrun input changed: {relative}")


def _role_name(key: tuple[int, str]) -> str:
    return f"learner-{key[0]}-{key[1]}-final_source"


def _protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    _require(path == root / PROTOCOL_PATH, "postrun diagnostic requires the dedicated r2 protocol")
    postrun, postrun_sha = _read_json(root, PROTOCOL_PATH)
    _require(postrun.get("format") == "haic-drq-final-source-postrun-protocol-r2"
             and postrun.get("study_id") == STUDY_ID
             and postrun.get("training_protocol_path") == reconstruction.PROTOCOL_PATH
             and postrun.get("training_protocol_sha256") == TRAINING_PROTOCOL_SHA
             and postrun.get("audit_receipt_path") == AUDIT_PATH.as_posix()
             and postrun.get("diagnostic_output_root") == OUTPUT_ROOT.as_posix()
             and postrun.get("interpretation") == INTERPRETATION
             and postrun.get("original_audit_status") == ORIGINAL_FAILURE
             and postrun.get("original_audit_failure") == ORIGINAL_FAILURE_TEXT,
             "postrun protocol identity, failure disclosure, or output path changed")
    diagnostic_contract = postrun.get("diagnostic")
    _require(type(diagnostic_contract) is dict
             and set(diagnostic_contract) == set(DIAGNOSTIC_CONTRACT)
             and all(type(diagnostic_contract[key]) is type(expected)
                     and diagnostic_contract[key] == expected
                     for key, expected in DIAGNOSTIC_CONTRACT.items()),
             "postrun diagnostic partition, grid, or fixed descriptive threshold changed")
    code = postrun.get("code_sha256")
    _require(type(code) is dict and set(code) == CODE_PATHS,
             "postrun protocol must pin exactly its two new executable sources")
    code = cast(dict[str, str], code)
    for name, expected in code.items():
        _require(type(expected) is str and diagnostic.r6.SHA256_RE.fullmatch(expected) is not None,
                 f"invalid postrun executable source SHA: {name}")
        _verified_sha(root, Path(name), expected)
    run_hashes = postrun.get("run_result_sha256")
    expected_names = {_role_name(key) for key in ROLES}
    _require(type(run_hashes) is dict and set(run_hashes) == expected_names,
             "postrun protocol must pin all and only six canonical TRAIN results")
    run_hashes = cast(dict[str, str], run_hashes)
    for name, expected in run_hashes.items():
        _require(type(expected) is str and diagnostic.r6.SHA256_RE.fullmatch(expected) is not None,
                 f"invalid postrun TRAIN result SHA: {name}")
        _verified_sha(root, Path(reconstruction.RUN_ROOT) / name / "result.json", expected)
    training, training_sha = _read_json(root, Path(reconstruction.PROTOCOL_PATH))
    _require(training_sha == TRAINING_PROTOCOL_SHA,
             "reconstruction training protocol bytes changed")
    reconstruction._validate_protocol(root, training)
    reconstruction._validate_exposure_snapshot(root, training)
    _require(not (root / reconstruction.SAMPLE_AUDIT_PATH).exists()
             and not (root / reconstruction.SAMPLE_AUDIT_PATH).is_symlink(),
             "original failed sample audit must have no passing receipt")
    output = root / OUTPUT_ROOT
    _require(not output.exists() and not output.is_symlink(),
             "postrun TRAIN-DIAGNOSTIC output root is occupied")
    _require(output.parent.is_dir() and output.parent.resolve() == output.parent,
             "postrun output parent is missing or unsafe")
    return postrun, postrun_sha, training


def _receipt(root: Path, postrun: dict[str, Any], postrun_sha: str,
             training: dict[str, Any], *, pools: dict[int, dict[str, str]] | None = None,
             records: dict[tuple[int, str], dict[str, Any]] | None = None) -> str:
    receipt, digest = _read_json(root, AUDIT_PATH)
    _require(receipt.get("format") == "haic-drq-final-source-postrun-sample-audit-r2"
             and receipt.get("passed") is True
             and receipt.get("original_audit_passed") is False
             and receipt.get("no_environment_resets") is True
             and receipt.get("postrun_protocol_sha256") == postrun_sha
             and receipt.get("training_protocol_sha256") == TRAINING_PROTOCOL_SHA
             and receipt.get("interpretation") == INTERPRETATION
             and receipt.get("causal_comparison_eligible") is False,
             "postrun audit is not a passing, noncausal receipt for this protocol")
    warmup = receipt.get("historical_warmup_comparison")
    _require(type(warmup) is dict, "postrun receipt lacks historical warmup comparison")
    warmup = cast(dict[str, Any], warmup)
    warmup_runs = warmup.get("runs")
    _require(warmup.get("matched") is False
             and type(warmup.get("warmup_steps")) is int and warmup["warmup_steps"] == 10000
             and type(warmup_runs) is list and len(warmup_runs) == 6,
             "postrun receipt does not disclose six historical warmup mismatches")
    seen: set[tuple[int, str]] = set()
    for row in warmup["runs"]:
        _require(type(row) is dict, "malformed historical warmup comparison")
        key = (row.get("source_seed"), row.get("variant"))
        first = row.get("first_mismatch")
        boundary = row.get("boundary_first_mismatch")
        _require(type(key[0]) is int and key in ROLES and key not in seen
                 and type(first) is dict and set(first) == WARMUP_FIELDS
                 and all(value is None or (type(value) is int and 0 <= value < 10000)
                         for value in first.values())
                 and type(first["frames"]) is int
                 and type(row.get("boundary_equal")) is bool
                 and "boundary_first_mismatch" in row
                 and (boundary is None or (type(boundary) is int and 0 <= boundary < 10000))
                 and (row["boundary_equal"] is (boundary is None)),
                 "historical warmup comparison is missing, malformed, or claims parity")
        seen.add(key)
    _require(seen == ROLES, "historical warmup comparison omitted an arm")
    source_rows = receipt.get("source_pool")
    declared = training.get("source_replay")
    _require(type(source_rows) is dict and set(source_rows) == {"0", "1"}
             and type(declared) is dict and set(declared) == {"0", "1"},
             "postrun receipt lacks both sealed source pools")
    source_rows = cast(dict[str, Any], source_rows)
    declared = cast(dict[str, Any], declared)
    for seed in (0, 1):
        item, source = source_rows[str(seed)], declared[str(seed)]
        _require(type(item) is dict and type(source) is dict,
                 "malformed final-source pool in postrun receipt")
        item = cast(dict[str, Any], item)
        source = cast(dict[str, Any], source)
        sealed, sealed_sha = _read_json(root, Path(source["receipt_path"]))
        _require(item.get("pool_sha256") == source.get("pool_sha256")
                 and item.get("receipt_sha256") == source.get("receipt_sha256")
                 and sealed_sha == source["receipt_sha256"]
                 and item.get("episode_ledger_sha256") == sealed.get("episode_ledger_sha256")
                 and type(item.get("valid_n_step_starts")) is int
                 and item["valid_n_step_starts"] == sealed.get("valid_n_step_starts")
                 and item["valid_n_step_starts"] >= 32
                 and type(item.get("verified_source_decisions")) is int
                 and item["verified_source_decisions"] == sealed.get("decisions") == 100000
                 and type(item.get("source_episodes")) is int
                 and item["source_episodes"] == sealed.get("scheduled_episodes_consumed")
                 and type(item.get("historical_prefix_decisions")) is int
                 and item["historical_prefix_decisions"] == sealed.get("historical_prefix_decisions")
                 and type(item.get("historical_prefix_episodes_consumed")) is int
                 and item["historical_prefix_episodes_consumed"] == sealed.get(
                     "historical_prefix_episodes_consumed")
                 and (pools is None or (item["pool_sha256"] == pools[seed]["pool_sha256"]
                                       and item["receipt_sha256"] == pools[seed]["receipt_sha256"])),
                 "postrun sample audit source pool or semantic counts changed")
    runs = receipt.get("runs")
    _require(type(runs) is list and len(runs) == 6,
             "postrun sample audit lacks six completed treatment arms")
    runs = cast(list[Any], runs)
    seen.clear()
    for row in runs:
        _require(type(row) is dict, "malformed postrun sample audit treatment")
        key = (row.get("source_seed"), row.get("variant"))
        _require(type(key[0]) is int and key in ROLES and key not in seen,
                 "postrun sample audit has a duplicate or missing treatment")
        name = _role_name(key)
        relative = Path(reconstruction.RUN_ROOT) / name / "result.json"
        result, result_sha = _read_json(root, relative)
        candidates = result.get("candidates")
        _require(type(candidates) is list and len(candidates) == 2
                 and type(candidates[-1]) is dict,
                 "postrun treatment lacks final checkpoint lineage")
        candidates = cast(list[dict[str, Any]], candidates)
        final = candidates[-1]
        _require(row.get("run_dir") == (Path(reconstruction.RUN_ROOT) / name).as_posix()
                 and row.get("result_sha256") == postrun["run_result_sha256"][name]
                 and row["result_sha256"] == result_sha
                 and row.get("final_checkpoint_sha256") == final.get("checkpoint_sha256")
                 and row.get("final_sample_trace_sha256") == final.get("sample_trace_sha256")
                 and result.get("study_id") == reconstruction.STUDY_ID
                 and result.get("study_protocol_sha256") == TRAINING_PROTOCOL_SHA
                 and result.get("completed") is True
                 and result.get("source_seed") == key[0]
                 and result.get("variant") == key[1]
                 and result.get("condition") == "final_source"
                 and result.get("source_replay_sha256") == declared[str(key[0])]["pool_sha256"]
                 and result.get("additional_online_steps") == 32768
                 and result.get("study_gradient_steps") == row.get("updates") == 22768
                 and result.get("source_samples") == row.get("source_rows") == 22768 * 32
                 and result.get("online_samples") == row.get("online_rows") == 22768 * 32
                 and type(row.get("batch_size")) is int and row["batch_size"] == 64
                 and type(row.get("unique_source_starts")) is int
                 and 32 <= row["unique_source_starts"] <= 100000
                 and type(row.get("unique_online_starts")) is int
                 and 32 <= row["unique_online_starts"] <= 32768
                 and (records is None or (
                     row["result_sha256"] == records[key]["result_sha256"]
                     and row["final_checkpoint_sha256"] == records[key]["checkpoint_sha256"]
                     and row["final_sample_trace_sha256"] == records[key]["sample_trace_sha256"]
                     and records[key]["source_replay_sha256"] == declared[str(key[0])]["pool_sha256"])),
                 "postrun audit differs from treatment result/checkpoint/sample counts")
        seen.add(key)
    _require(seen == ROLES, "postrun sample audit omitted a seed/mixture")
    return digest


def _handoff_check(root: Path, postrun: dict[str, Any], postrun_sha: str,
                   receipt_sha: str) -> None:
    _verified_sha(root, PROTOCOL_PATH, postrun_sha)
    _verified_sha(root, AUDIT_PATH, receipt_sha)
    _verified_sha(root, Path(reconstruction.PROTOCOL_PATH), TRAINING_PROTOCOL_SHA)
    _require(not (root / reconstruction.SAMPLE_AUDIT_PATH).exists()
             and not (root / reconstruction.SAMPLE_AUDIT_PATH).is_symlink(),
             "original failed sample audit must have no passing receipt")
    for name, expected in postrun["code_sha256"].items():
        _verified_sha(root, Path(name), expected)
    for name, expected in postrun["run_result_sha256"].items():
        _verified_sha(root, Path(reconstruction.RUN_ROOT) / name / "result.json", expected)


@contextmanager
def _diagnostic_context(root: Path, postrun: dict[str, Any], postrun_sha: str,
                        training: dict[str, Any], receipt_sha: str) -> Iterator[None]:
    originals = (diagnostic.STUDY_ID, diagnostic.RUN_ROOT, diagnostic.SAMPLE_AUDIT,
                 diagnostic._sample_audit, diagnostic._prepare, diagnostic._require,
                 diagnostic.run_diagnostic)
    _, original_prepare, original_require, original_run = originals[3:]
    original_factory = diagnostic.r6.build_environment
    original_collect = diagnostic.r6._collect_episode
    in_run = False

    def audited_sample(root: Path, protocol_sha: str, pools: dict[int, dict[str, str]],
                       records: dict[tuple[int, str], dict[str, Any]]) -> str:
        _require(protocol_sha == TRAINING_PROTOCOL_SHA,
                 "parent diagnostic is not using the pinned training protocol")
        _require(_receipt(root, postrun, postrun_sha, training, pools=pools, records=records)
                 == receipt_sha, "postrun receipt changed between preflight and diagnostic")
        return receipt_sha

    def prepare(*args: Any, **kwargs: Any) -> Any:
        rows, roles, sources, controls, lineage = original_prepare(*args, **kwargs)
        lineage.update({
            "postrun_protocol_path": PROTOCOL_PATH.as_posix(),
            "postrun_protocol_sha256": postrun_sha,
            "postrun_binder_source_sha256": postrun["code_sha256"][
                "scripts/diagnose_drq_final_source_reconstruction_r2.py"],
            "postrun_auditor_source_sha256": postrun["code_sha256"][
                "scripts/audit_drq_final_source_reconstruction_r2.py"],
            "sample_audit_path": AUDIT_PATH.as_posix(),
            "sample_audit_sha256": receipt_sha,
            "original_audit_status": ORIGINAL_FAILURE,
            "original_audit_passed": False,
            "interpretation": INTERPRETATION,
            "causal_comparison_eligible": False,
            "comparison": "Frozen actors on reused TRAIN-DIAGNOSTIC roads; historical warmup bytes diverged; no matched replay-only causal comparison",
            "sample_audit_scope": "Source/online ledger, checkpoint and sampled-update lineage verified; child warmup environment pixel/action/reward parity unproven",
            "performance_scope": "Descriptive reused TRAIN-DIAGNOSTIC proxy only; no physical replay parity, causal, fresh, protected, official or promotion claim",
        })
        return rows, roles, sources, controls, lineage

    def require_scoped(condition: bool, message: str) -> None:
        # Only the parent's literal old-output check changes; every other gate remains intact.
        if not condition and message == OUTPUT_GUARD and in_run:
            return
        original_require(condition, message)

    def factory_scoped(*args: Any, **kwargs: Any) -> Any:
        _require(in_run, "postrun environment factory called outside the diagnostic")
        _handoff_check(root, postrun, postrun_sha, receipt_sha)
        return original_factory(*args, **kwargs)

    def collect_scoped(*args: Any, **kwargs: Any) -> Any:
        _require(in_run, "postrun episode collector called outside the diagnostic")
        _handoff_check(root, postrun, postrun_sha, receipt_sha)
        return original_collect(*args, **kwargs)

    def run_scoped(*, root: Path, protocol_path: Path, output_root: Path,
                   preflight_only: bool = False, environment_factory: Any = None,
                   actor_loader: Any = None) -> dict[str, Any]:
        nonlocal in_run
        _require(root == root_path and protocol_path == root_path / reconstruction.PROTOCOL_PATH
                 and output_root == root_path / OUTPUT_ROOT,
                 "frozen diagnostic may only use the r2 output and pinned TRAIN protocol")
        _require(type(preflight_only) is bool and environment_factory is None and actor_loader is None,
                 "postrun diagnostic may not substitute the environment or actor loader")
        _require(not in_run, "nested postrun diagnostic invocation")
        _handoff_check(root, postrun, postrun_sha, receipt_sha)
        in_run = True
        try:
            result = original_run(root=root, protocol_path=protocol_path, output_root=output_root,
                                  preflight_only=preflight_only, environment_factory=environment_factory,
                                  actor_loader=actor_loader)
            _handoff_check(root, postrun, postrun_sha, receipt_sha)
            return result
        finally:
            in_run = False

    root_path = root
    diagnostic.STUDY_ID = reconstruction.STUDY_ID
    diagnostic.RUN_ROOT = Path(reconstruction.RUN_ROOT)
    diagnostic.SAMPLE_AUDIT = AUDIT_PATH
    diagnostic._sample_audit = audited_sample
    diagnostic._prepare = prepare
    diagnostic._require = require_scoped
    diagnostic.run_diagnostic = run_scoped
    diagnostic.r6.build_environment = factory_scoped
    diagnostic.r6._collect_episode = collect_scoped
    try:
        with child._parent_collection_identity_context():
            yield
    finally:
        (diagnostic.STUDY_ID, diagnostic.RUN_ROOT, diagnostic.SAMPLE_AUDIT,
         diagnostic._sample_audit, diagnostic._prepare, diagnostic._require,
         diagnostic.run_diagnostic) = originals
        diagnostic.r6.build_environment = original_factory
        diagnostic.r6._collect_episode = original_collect


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve(strict=True)
    postrun, postrun_sha, training = _protocol(root, root / args.protocol)
    child._validate_cpu_runtime()
    receipt_sha = _receipt(root, postrun, postrun_sha, training)
    _handoff_check(root, postrun, postrun_sha, receipt_sha)
    diagnostic_argv = ["--repo-root", str(root), "--protocol", str(root / reconstruction.PROTOCOL_PATH),
                       "--output-root", str(root / OUTPUT_ROOT)]
    if args.preflight_only:
        diagnostic_argv.append("--preflight-only")
    with _diagnostic_context(root, postrun, postrun_sha, training, receipt_sha):
        original_argv = sys.argv
        sys.argv = ["diagnose_drq_final_replay.py", *diagnostic_argv]
        try:
            diagnostic.main()
        finally:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DiagnosticGateError, reconstruction.GateError,
            reconstruction.exposure_audit.AuditError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
