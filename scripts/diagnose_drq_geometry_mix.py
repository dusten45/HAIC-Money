"""Post-training TRAIN-DIAGNOSTIC traces for frozen DrQ geometry-mix actors.

Run only with the frozen CPU21 runtime, for example:
``/tmp/kilo/haic-cpu21/bin/python -B -m scripts.diagnose_drq_geometry_mix``.
This is a descriptive TRAIN-DIAGNOSTIC, not screen, confirmation, blind, or
fresh-generalization evaluation. No score-based actor/road selection is allowed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np
import torch

from common_adapter import ActionAdapter, EpisodeCollector, ObservationSpec
from drq_v2 import load_exported_actor


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("experiments/drqv2-geometry-mix-v1-r6.json")
CATALOG_PATH = Path("runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json")
CATALOG_SHA256 = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
CATALOG_PROTOCOL_PATH = Path("experiments/drqv2-geometry-augmentation-v1.json")
CATALOG_PROTOCOL_SHA256 = "be6d1d1c3b1b16f6b56fd4720097ca8fea8b64c5c962d07183961cdfa0d87294"
PROTOCOL_FORMAT = "haic-drq-geometry-mix-study-v1"
CATALOG_FORMAT = "haic-drq-training-geometry-catalog-v1"
CHECKPOINT_CATALOG_FORMAT = "haic-drq-geometry-mix-checkpoint-catalog-v1"
RUN_FORMAT = "haic-drq-geometry-mix-training-diagnostic-v1"
EPISODE_FORMAT = "haic-drq-geometry-mix-training-diagnostic-episode-v1"
VARIANTS = ("uniform", "failure_weighted", "easy_retention")
SOURCE_ACTORS = {
    0: {
        "actor_path": "runs/20260922-drq-augmentation-pad-v1-restart/control-seed0/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37",
        "actor_weights_sha256": "81a8c0785e7f586481af40ae628714aec858c1c0a89db27c79e709311997ba93",
    },
    1: {
        "actor_path": "runs/20260922-drq-augmentation-pad-v1-restart/control-seed1/checkpoints/step-000131072/actor.pt",
        "actor_sha256": "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
        "actor_weights_sha256": "e0984f709d1971754f2e9b64a69bf216922ca5bd0791ad247181b218f830b736",
    },
}
DIAGNOSTIC_SEEDS = (
    3910800153, 3910800045, 3910800148, 3910800124,
    3910800134, 3910800160, 3910800163, 3910800171,
    3910800187, 3910800172, 3910800175, 3910800182,
    3910800169, 3910800162, 3910800174, 3910800164,
)
EXPECTED_FAMILIES = (
    "opening-short-entry-left-turn",
    "opening-delayed-high-turn",
    "easy-curvature-anchor",
    "mid-road-left-right-reversal",
    "mid-road-sustained-or-same-turn",
    "finish-approach-turn",
)
MAX_STEPS = 1200
FRAME_SKIP = 4
REPEATS = 2
SOURCE_STEPS = 131072
FINAL_ADDITIONAL_STEP = 32768
WARMUP_STEPS = 10000
SPARSE_STRIDE = 25
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TRACE_ARRAYS = (
    "native_action", "official_action", "reward", "progress", "damage",
    "terminated", "truncated", "terminal", "retirement", "finished",
    "finish_qualified", "finish_crossing_time_s", "collision",
    "off_track_counter", "episode_id", "repeat_index", "source_actor_sha256",
    "actor_sha256", "checkpoint_sha256", "source_checkpoint_sha256", "sparse_step",
    "sparse_nearest_point_index", "sparse_nearest_point_fraction",
    "sparse_nearest_distance_m", "sparse_speed_m_s",
)


class DiagnosticError(ValueError):
    """A fail-closed input, provenance, determinism, or trace error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DiagnosticError(f"non-finite JSON constant: {value}")


def _parse_json(raw: bytes, label: Path) -> Any:
    try:
        return json.loads(
            raw, object_pairs_hook=_object, parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DiagnosticError(f"cannot read valid JSON: {label}") from error


def _read_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DiagnosticError(f"cannot read JSON: {path}") from error
    return _parse_json(raw, path)


def _pinned_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not isinstance(expected_sha256, str) or SHA256_RE.fullmatch(expected_sha256) is None:
        raise DiagnosticError(f"{label} requires a lowercase SHA-256")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DiagnosticError(f"cannot read pinned {label}: {path}") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DiagnosticError(f"{label} SHA-256 mismatch: {path}")
    value = _parse_json(raw, path)
    if not isinstance(value, dict):
        raise DiagnosticError(f"{label} must be a JSON object")
    return value


def _repo_file(root: Path, value: Any, directory: str, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DiagnosticError(f"{label} must be a repository-relative {directory}/ path")
    relative = Path(value)
    if any(part in (".", "..") for part in relative.parts) or relative.parts[0] != directory:
        raise DiagnosticError(f"{label} must remain under {directory}/")
    path = root / relative
    try:
        path.resolve().relative_to((root / directory).resolve())
    except ValueError as error:
        raise DiagnosticError(f"{label} escapes {directory}/") from error
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise DiagnosticError(f"symlink in {label}: {current}")
    return path


def _user_repo_file(root: Path, value: str | Path, directory: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            value = path.relative_to(root).as_posix()
        except ValueError as error:
            raise DiagnosticError(f"{label} escapes the repository") from error
    return _repo_file(root, str(value), directory, label)


def _int(value: Any, label: str, *, minimum: int = 0, maximum: int = 2**32 - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise DiagnosticError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _state_dict_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise DiagnosticError("actor export has no state_dict")
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise DiagnosticError("actor state_dict has an invalid entry")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _model_weights_sha256(actor: Any) -> str:
    return _state_dict_sha256(actor.state_dict())


def _validate_cpu21(protocol: dict[str, Any]) -> dict[str, Any]:
    diagnostics = protocol.get("diagnostics")
    runtime = diagnostics.get("runtime") if isinstance(diagnostics, dict) else None
    if not isinstance(runtime, dict):
        raise DiagnosticError("frozen diagnostic CPU21 runtime contract is missing")
    expected_python = runtime.get("verified_interpreter")
    if expected_python != "/tmp/kilo/haic-cpu21/bin/python" or Path(sys.executable) != Path(expected_python):
        raise DiagnosticError("diagnostic must run with the frozen CPU21 interpreter")
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != Path(expected_python).parent.parent.resolve():
        raise DiagnosticError("active interpreter is not the verified Python 3.11 CPU21 environment")
    try:
        import cv2
        import gymnasium
    except ImportError as error:
        raise DiagnosticError("CPU21 diagnostic dependencies are unavailable") from error
    versions = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gymnasium": gymnasium.__version__,
        "opencv": cv2.__version__,
    }
    expected = {"torch": "2.1.0+cpu", "numpy": "1.26.0", "gymnasium": "0.29.1", "opencv": "4.8.1"}
    if versions != expected or torch.cuda.is_available():
        raise DiagnosticError(f"CPU21 dependency/device contract mismatch: {versions}")
    return {"python": sys.version.split()[0], "executable": sys.executable, **versions, "device": "cpu"}


def _catalog_rows(root: Path, protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog_spec = protocol.get("catalog")
    if not isinstance(catalog_spec, dict):
        raise DiagnosticError("frozen catalog lineage is missing")
    if (catalog_spec.get("path") != CATALOG_PATH.as_posix()
            or catalog_spec.get("sha256") != CATALOG_SHA256
            or catalog_spec.get("protocol_path") != CATALOG_PROTOCOL_PATH.as_posix()
            or catalog_spec.get("protocol_sha256") != CATALOG_PROTOCOL_SHA256):
        raise DiagnosticError("protocol does not bind the immutable geometry catalog and generator protocol")
    catalog_path = _repo_file(root, catalog_spec["path"], "runs", "geometry catalog")
    catalog = _pinned_json(catalog_path, CATALOG_SHA256, "geometry catalog")
    if catalog.get("format") != CATALOG_FORMAT or catalog.get("protocol_sha256") != CATALOG_PROTOCOL_SHA256:
        raise DiagnosticError("geometry catalog format or generator lineage mismatch")
    catalog_protocol_path = _repo_file(root, catalog_spec["protocol_path"], "experiments", "catalog protocol")
    catalog_protocol = _pinned_json(
        catalog_protocol_path, CATALOG_PROTOCOL_SHA256, "geometry catalog protocol",
    )
    if catalog_protocol.get("format") != "haic-drq-training-geometry-protocol-v1":
        raise DiagnosticError("unexpected geometry catalog protocol format")
    diagnostics = protocol.get("diagnostics")
    diagnostic_seeds = diagnostics.get("geometry_seeds") if isinstance(diagnostics, dict) else None
    if (not isinstance(diagnostics, dict)
            or diagnostics.get("format") != "haic-drq-geometry-mix-diagnostic-v1"
            or diagnostics.get("role") != "training_diagnostic"
            or diagnostics.get("partition") != "TRAIN-DIAGNOSTIC"
            or not isinstance(diagnostic_seeds, list)
            or any(type(seed) is not int for seed in diagnostic_seeds)
            or len(diagnostic_seeds) != len(set(diagnostic_seeds))
            or set(diagnostic_seeds) != set(DIAGNOSTIC_SEEDS)
            or diagnostics.get("track_ids") != [1]
            or diagnostics.get("repeats") != REPEATS
            or diagnostics.get("max_steps") != MAX_STEPS
            or diagnostics.get("frame_skip") != FRAME_SKIP
            or diagnostics.get("raw_reward") is not True
            or diagnostics.get("ranked") is not False
            or diagnostics.get("source_actors_included") is not True
            or diagnostics.get("no_official_performance_claim") is not True):
        raise DiagnosticError("protocol diagnostic pool/horizon/repeat contract differs from the frozen allocation")
    rows = catalog.get("train_diagnostic")
    if not isinstance(rows, list) or len(rows) != len(DIAGNOSTIC_SEEDS):
        raise DiagnosticError("catalog must contain exactly 16 TRAIN-DIAGNOSTIC roads")
    seen_seeds: set[int] = set()
    seen_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise DiagnosticError("invalid TRAIN-DIAGNOSTIC catalog row")
        seed = _int(row.get("geometry_seed"), "geometry_seed")
        road_hash = row.get("road_coordinate_sha256")
        if (seed not in DIAGNOSTIC_SEEDS or seed in seen_seeds or row.get("track_id") != 1
                or type(row.get("track_id")) is not int or row.get("stage") != "diagnostic"
                or row.get("family") not in EXPECTED_FAMILIES
                or not isinstance(road_hash, str) or SHA256_RE.fullmatch(road_hash) is None
                or road_hash in seen_hashes):
            raise DiagnosticError("catalog contains duplicate, reused, blind, or non-diagnostic geometry")
        verification = row.get("verification")
        if (not isinstance(row.get("signature_sha256"), str)
                or SHA256_RE.fullmatch(row["signature_sha256"]) is None
                or not isinstance(verification, dict)
                or verification.get("regenerated_coordinate_hash_match") is not True):
            raise DiagnosticError("TRAIN-DIAGNOSTIC road lacks frozen structural/hash verification")
        seen_seeds.add(seed)
        seen_hashes.add(road_hash)
    if seen_seeds != set(DIAGNOSTIC_SEEDS):
        raise DiagnosticError("TRAIN-DIAGNOSTIC catalog seed set differs from the frozen 16 roads")

    exclusions = catalog_protocol.get("exclusion_seed_ids")
    if not isinstance(exclusions, dict) or set(exclusions) != {"reserved", "heldout", "blind"}:
        raise DiagnosticError("generator protocol exclusion-only seed lists are malformed")
    forbidden: set[int] = set()
    for name, values in exclusions.items():
        if not isinstance(values, list) or any(type(seed) is not int for seed in values):
            raise DiagnosticError(f"invalid geometry exclusion list: {name}")
        if len(values) != len(set(values)):
            raise DiagnosticError(f"reused seed in geometry exclusion list: {name}")
        forbidden.update(values)
    if set(DIAGNOSTIC_SEEDS) & forbidden:
        raise DiagnosticError("TRAIN-DIAGNOSTIC road reuses a reserved/heldout/blind geometry seed")
    audit = catalog.get("seed_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise DiagnosticError("geometry catalog lacks a passing training-only seed audit")
    collisions = audit.get("matched_collisions")
    if not isinstance(collisions, list) or set(DIAGNOSTIC_SEEDS) & set(collisions):
        raise DiagnosticError("TRAIN-DIAGNOSTIC seed audit reports a reused cell")
    if {row["family"] for row in rows} != set(EXPECTED_FAMILIES):
        raise DiagnosticError("TRAIN-DIAGNOSTIC catalog does not cover all six frozen geometry families")
    return sorted(rows, key=lambda row: row["geometry_seed"]), {
        "catalog_path": catalog_spec["path"],
        "catalog_sha256": CATALOG_SHA256,
        "catalog_protocol_path": catalog_spec["protocol_path"],
        "catalog_protocol_sha256": CATALOG_PROTOCOL_SHA256,
        "scan_path": catalog.get("scan_path"),
        "scan_sha256": catalog.get("scan_sha256"),
    }


def _validate_study_contract(protocol: dict[str, Any]) -> None:
    if (protocol.get("catalog_sha256") != CATALOG_SHA256
            or protocol.get("catalog_protocol_path") != CATALOG_PROTOCOL_PATH.as_posix()
            or protocol.get("catalog_protocol_sha256") != CATALOG_PROTOCOL_SHA256):
        raise DiagnosticError("study-level immutable catalog aliases do not match the frozen catalog")
    declared_runs = protocol.get("runs")
    expected_runs = {(seed, variant) for seed in (0, 1) for variant in VARIANTS}
    if not isinstance(declared_runs, list) or len(declared_runs) != 6:
        raise DiagnosticError("study protocol must freeze exactly six online-only run identities")
    seen_runs: set[tuple[int, str]] = set()
    for run in declared_runs:
        if not isinstance(run, dict):
            raise DiagnosticError("invalid top-level frozen run identity")
        seed = _int(run.get("source_seed"), "run source_seed", maximum=1)
        variant = run.get("variant")
        role = (seed, variant)
        expected_path = Path("runs/20260925-drqv2-geometry-mix-v1-r6") / f"learner-{seed}-{variant}"
        if role not in expected_runs or role in seen_runs or run.get("run_dir") != expected_path.as_posix():
            raise DiagnosticError("top-level run identities are duplicate, unexpected, or reused")
        seen_runs.add(role)
    if seen_runs != expected_runs:
        raise DiagnosticError("top-level run identities do not form the exact six-role grid")
    training = protocol.get("environment")
    if not isinstance(training, dict):
        raise DiagnosticError("frozen optimization TRAIN pool is missing")
    training_seeds = training.get("geometry_seeds")
    if (training.get("partition") != "TRAIN"
            or not isinstance(training_seeds, list) or len(training_seeds) != 120
            or any(type(seed) is not int for seed in training_seeds)
            or len(set(training_seeds)) != len(training_seeds)
            or set(training_seeds) & set(DIAGNOSTIC_SEEDS)
            or training.get("track_ids") != [1, 2, 3, 4]
            or training.get("frame_skip") != FRAME_SKIP
            or training.get("reward_shaping") is not False
            or training.get("reward_normalization") is not False):
        raise DiagnosticError("TRAIN optimization rows overlap or violate the frozen TRAIN-only contract")
    learner = protocol.get("learner")
    if (not isinstance(learner, dict) or learner.get("source_environment_steps") != SOURCE_STEPS
            or learner.get("padding") != 4 or learner.get("steering_logit_l2") != 0.0):
        raise DiagnosticError("source checkpoint/model configuration is not the frozen pad-4 control")
    budgets = protocol.get("budgets")
    if (not isinstance(budgets, dict)
            or budgets.get("additional_online_steps") != FINAL_ADDITIONAL_STEP
            or budgets.get("expected_gradient_updates") != FINAL_ADDITIONAL_STEP - WARMUP_STEPS
            or budgets.get("warmup_steps") != WARMUP_STEPS
            or budgets.get("checkpoint_steps") != [16384, FINAL_ADDITIONAL_STEP]):
        raise DiagnosticError("frozen online-only training budget does not identify final step 32768")
    replay = protocol.get("replay")
    if (not isinstance(replay, dict) or replay.get("mode") != "online-only"
            or replay.get("online_rows") != 64 or replay.get("teacher_rows") != 0):
        raise DiagnosticError("final candidate arms are not exactly online-only")
    variants = protocol.get("variants")
    if (not isinstance(variants, dict) or set(variants) != set(VARIANTS)
            or any(not isinstance(value, dict) or value.get("score_selection") is not False
                   for value in variants.values())):
        raise DiagnosticError("frozen distribution variants are incomplete or score-selected")


def _source_records(root: Path, protocol: dict[str, Any],
                    catalog_protocol: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = protocol.get("diagnostics", {}).get("source_actors")
    declared = catalog_protocol.get("source_actors")
    if not isinstance(records, list) or len(records) != 2 or not isinstance(declared, list) or len(declared) != 2:
        raise DiagnosticError("exactly two frozen source actors are required")
    expected_declared = {item.get("learner_seed"): item for item in declared if isinstance(item, dict)}
    if set(expected_declared) != {0, 1}:
        raise DiagnosticError("geometry catalog source identity pair is malformed")
    prepared: dict[int, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise DiagnosticError("invalid frozen source actor entry")
        learner_seed = _int(record.get("learner_seed"), "source learner_seed", maximum=1)
        expected = SOURCE_ACTORS[learner_seed]
        original = expected_declared[learner_seed]
        if (learner_seed in prepared
                or record.get("actor_path") != expected["actor_path"]
                or record.get("actor_sha256") != expected["actor_sha256"]
                or record.get("source_actor_path") != expected["actor_path"]
                or record.get("source_actor_sha256") != expected["actor_sha256"]
                or record.get("source_actor_weights_sha256") != expected["actor_weights_sha256"]
                or record.get("weight_only_fork") is not True
                or record.get("source_checkpoint_path") != record.get("checkpoint_path")
                or record.get("source_checkpoint_sha256") != record.get("checkpoint_sha256")
                or any(original.get(field) != expected[field] for field in expected)):
            raise DiagnosticError("source actor path/export/weight identity differs from the frozen control")
        actor_path = _repo_file(root, record.get("actor_path"), "runs", "source actor")
        checkpoint_path = _repo_file(root, record.get("checkpoint_path"), "runs", "source checkpoint")
        manifest_path = _repo_file(root, record.get("checkpoint_manifest_path"), "runs", "source checkpoint manifest")
        hashes = {
            "actor_sha256": expected["actor_sha256"],
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "checkpoint_manifest_sha256": record.get("checkpoint_manifest_sha256"),
        }
        for name, digest in hashes.items():
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise DiagnosticError(f"source {name} is not a lowercase SHA-256")
        if (sha256_file(actor_path) != hashes["actor_sha256"]
                or sha256_file(checkpoint_path) != hashes["checkpoint_sha256"]):
            raise DiagnosticError("frozen source actor/checkpoint/manifest bytes changed")
        manifest = _pinned_json(manifest_path, hashes["checkpoint_manifest_sha256"], "source checkpoint manifest")
        if (not isinstance(manifest, dict) or manifest.get("algorithm") != "drq-v2"
                or manifest.get("extra", {}).get("environment_steps") != SOURCE_STEPS):
            raise DiagnosticError("source checkpoint manifest does not identify the frozen 131072-step control")
        prepared[learner_seed] = {
            "role": "unchanged-source", "arm": "unchanged-source", "source_learner_seed": learner_seed,
            "actor_path": actor_path, "actor_sha256": hashes["actor_sha256"],
            "actor_weights_sha256": record["source_actor_weights_sha256"],
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": hashes["checkpoint_sha256"],
            "checkpoint_manifest_path": manifest_path,
            "checkpoint_manifest_sha256": hashes["checkpoint_manifest_sha256"],
            "checkpoint_online_step": SOURCE_STEPS,
            "catalog_sha256": None, "catalog_path": None,
        }
    if set(prepared) != {0, 1}:
        raise DiagnosticError("source actor role coverage is not exactly learner seeds 0 and 1")
    return prepared


def _candidate_records(root: Path, protocol: dict[str, Any], protocol_sha256: str,
                       sources: dict[int, dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    diagnostics = protocol.get("diagnostics")
    runs = diagnostics.get("candidate_runs") if isinstance(diagnostics, dict) else None
    if not isinstance(runs, list) or len(runs) != 6:
        raise DiagnosticError("exactly six frozen online-only candidate run directories are required")
    inventory: dict[tuple[int, str], dict[str, Any]] = {}
    expected_roles = {(seed, variant) for seed in (0, 1) for variant in VARIANTS}
    for run in runs:
        if not isinstance(run, dict):
            raise DiagnosticError("invalid frozen candidate run identity")
        learner_seed = _int(run.get("source_seed"), "candidate source_seed", maximum=1)
        variant = run.get("variant")
        role = (learner_seed, variant)
        expected_run_dir = Path("runs/20260925-drqv2-geometry-mix-v1-r6") / f"learner-{learner_seed}-{variant}"
        if role not in expected_roles or role in inventory or run.get("run_dir") != expected_run_dir.as_posix():
            raise DiagnosticError("candidate run roles/paths are duplicate, unexpected, or reused")
        run_dir = _repo_file(root, run.get("run_dir"), "runs", "candidate run directory")
        catalog_path = run_dir / "checkpoint-catalog.json"
        if not catalog_path.is_file() or catalog_path.is_symlink():
            raise DiagnosticError(f"frozen run has no regular checkpoint catalog: {catalog_path}")
        try:
            checkpoint_catalog_bytes = catalog_path.read_bytes()
        except OSError as error:
            raise DiagnosticError(f"cannot read checkpoint catalog: {catalog_path}") from error
        checkpoint_catalog_sha = hashlib.sha256(checkpoint_catalog_bytes).hexdigest()
        checkpoint_catalog = _parse_json(checkpoint_catalog_bytes, catalog_path)
        if not isinstance(checkpoint_catalog, dict):
            raise DiagnosticError(f"checkpoint catalog must be an object: {catalog_path}")
        source = sources[learner_seed]
        if (checkpoint_catalog.get("format") != CHECKPOINT_CATALOG_FORMAT
                or checkpoint_catalog.get("study_protocol_sha256") != protocol_sha256
                or checkpoint_catalog.get("source_seed") != learner_seed
                or checkpoint_catalog.get("variant") != variant
                or checkpoint_catalog.get("source_actor_sha256") != source["actor_sha256"]
                or checkpoint_catalog.get("catalog_sha256") != CATALOG_SHA256):
            raise DiagnosticError(f"checkpoint catalog source/study identity mismatch: {catalog_path}")
        candidates = checkpoint_catalog.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise DiagnosticError(f"checkpoint catalog has no frozen candidates: {catalog_path}")
        steps = [item.get("checkpoint_online_step") for item in candidates if isinstance(item, dict)]
        if (len(steps) != len(candidates) or any(type(step) is not int for step in steps)
                or len(set(steps)) != len(steps)):
            raise DiagnosticError(f"checkpoint catalog has reused/invalid online steps: {catalog_path}")
        final_records = [item for item in candidates if item.get("checkpoint_online_step") == FINAL_ADDITIONAL_STEP]
        if len(final_records) != 1:
            raise DiagnosticError(f"checkpoint catalog must identify one final step-32768 actor: {catalog_path}")
        record = final_records[0]
        required = (
            "actor_path", "actor_sha256", "checkpoint_path", "checkpoint_sha256",
            "checkpoint_manifest_path", "checkpoint_manifest_sha256",
        )
        if any(not isinstance(record.get(key), str) for key in required):
            raise DiagnosticError(f"final candidate lacks actor/checkpoint/manifest path/hash: {catalog_path}")
        for field, expected in (
            ("source_seed", learner_seed), ("variant", variant),
            ("source_actor_sha256", source["actor_sha256"]),
            ("source_checkpoint_sha256", source["checkpoint_sha256"]),
            ("study_protocol_sha256", protocol_sha256),
        ):
            if record.get(field) != expected:
                raise DiagnosticError(f"final candidate lineage mismatch at {field}: {catalog_path}")
        checkpoint_dir = run_dir / "checkpoints" / f"step-{FINAL_ADDITIONAL_STEP:09d}"
        actor_path = _repo_file(root, record["actor_path"], "runs", "candidate actor")
        checkpoint_path = _repo_file(root, record["checkpoint_path"], "runs", "candidate checkpoint")
        manifest_path = _repo_file(root, record["checkpoint_manifest_path"], "runs", "candidate checkpoint manifest")
        if (actor_path != checkpoint_dir / "actor.pt"
                or checkpoint_path != checkpoint_dir / "checkpoint.pt"
                or manifest_path != checkpoint_dir / "checkpoint.manifest.json"):
            raise DiagnosticError("final candidate artifact paths differ from the frozen run layout")
        expected_hashes = {
            actor_path: record["actor_sha256"],
            checkpoint_path: record["checkpoint_sha256"],
            manifest_path: record["checkpoint_manifest_sha256"],
        }
        for path, digest in expected_hashes.items():
            if (not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None
                    or not path.is_file() or path.is_symlink()
                    or (path != manifest_path and sha256_file(path) != digest)):
                raise DiagnosticError(f"candidate actor/checkpoint/manifest hash mismatch: {path}")
        manifest = _pinned_json(manifest_path, record["checkpoint_manifest_sha256"], "candidate checkpoint manifest")
        if not isinstance(manifest, dict):
            raise DiagnosticError(f"final checkpoint manifest must be an object: {manifest_path}")
        extra = manifest.get("extra")
        if (manifest.get("algorithm") != "drq-v2" or not isinstance(extra, dict)
                or extra.get("environment_steps") != SOURCE_STEPS + FINAL_ADDITIONAL_STEP):
            raise DiagnosticError("final checkpoint manifest does not bind the expected DrQ online step")
        expected_updates = protocol.get("budgets", {}).get("expected_gradient_updates")
        for field, expected in (
            ("study_protocol_sha256", protocol_sha256),
            ("source_seed", learner_seed),
            ("variant", variant),
            ("source_actor_sha256", source["actor_sha256"]),
            ("source_checkpoint_sha256", source["checkpoint_sha256"]),
            ("checkpoint_online_step", FINAL_ADDITIONAL_STEP),
            ("catalog_sha256", CATALOG_SHA256),
        ):
            if extra.get(field) != expected:
                raise DiagnosticError(f"checkpoint manifest lineage mismatch at {field}: {manifest_path}")
        if (record.get("study_gradient_steps") != expected_updates
                or record.get("resume_allowed") is not False):
            raise DiagnosticError("final candidate gradient budget or completed-run status is invalid")
        inventory[role] = {
            "role": variant, "arm": variant, "variant": variant,
            "source_learner_seed": learner_seed,
            "source_actor_sha256": source["actor_sha256"],
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "actor_path": actor_path, "actor_sha256": record["actor_sha256"],
            "actor_weights_sha256": record.get("actor_weights_sha256"),
            "checkpoint_path": checkpoint_path, "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_manifest_path": manifest_path,
            "checkpoint_manifest_sha256": record["checkpoint_manifest_sha256"],
            "checkpoint_online_step": FINAL_ADDITIONAL_STEP,
            "catalog_path": catalog_path, "catalog_sha256": checkpoint_catalog_sha,
        }
    if set(inventory) != expected_roles:
        missing, extra = expected_roles - set(inventory), set(inventory) - expected_roles
        raise DiagnosticError(f"checkpoint catalogs do not form the exact six-role grid; missing={missing}, extra={extra}")
    all_roles = list(sources.values()) + list(inventory.values())
    if len({item["actor_sha256"] for item in all_roles}) != 8:
        raise DiagnosticError("actor export bytes are reused across the full eight-role model grid")
    return inventory


def _validate_actor(path: Path, actor_sha256: str, expected_weights_sha256: str | None,
                    *, loader: Callable[..., Any]) -> tuple[Any, Any, Any, str]:
    if sha256_file(path) != actor_sha256:
        raise DiagnosticError(f"actor export changed before CPU loading: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (not isinstance(payload, dict) or payload.get("format") != "haic-drq-v2-actor-v1"
            or not isinstance(payload.get("config"), dict)
            or payload["config"].get("augmentation_pad") != 4
            or payload["config"].get("steering_logit_l2", 0.0) != 0.0):
        raise DiagnosticError(f"unsupported or non-control DrQ actor export: {path}")
    weights_sha256 = _state_dict_sha256(payload.get("state_dict"))
    if expected_weights_sha256 is not None and weights_sha256 != expected_weights_sha256:
        raise DiagnosticError(f"actor weight identity differs from its frozen source record: {path}")
    first, action_adapter, observation_spec = loader(path, device="cpu")
    second, second_adapter, second_spec = loader(path, device="cpu")
    if (sha256_file(path) != actor_sha256
            or action_adapter.spec.fingerprint != ActionAdapter().spec.fingerprint
            or second_adapter.spec.fingerprint != ActionAdapter().spec.fingerprint
            or observation_spec.fingerprint != ObservationSpec().fingerprint
            or second_spec.fingerprint != ObservationSpec().fingerprint
            or _model_weights_sha256(first) != weights_sha256
            or _model_weights_sha256(second) != weights_sha256):
        raise DiagnosticError(f"CPU reload model/action/observation identity mismatch: {path}")
    for value in (0.0, 0.5, 1.0):
        observation = np.full((4, 84, 84), value, dtype=np.float32)
        with torch.inference_mode():
            action_a = first.act(observation, deterministic=True)
            action_b = second.act(observation, deterministic=True)
        if (not np.array_equal(action_a, action_b) or np.asarray(action_a).shape != (3,)
                or not np.isfinite(action_a).all()
                or np.any(np.asarray(action_a) < -1.0) or np.any(np.asarray(action_a) > 1.0)):
            raise DiagnosticError(f"CPU actor reload is nondeterministic: {path}")
    del second, payload
    return first, action_adapter, observation_spec, weights_sha256


def _prepare(root: Path, protocol_path: Path, expected_protocol_sha256: str,
             *, loader: Callable[..., Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve()
    protocol_path = _user_repo_file(root, protocol_path, "experiments", "geometry-mix protocol")
    if protocol_path.relative_to(root).as_posix() != PROTOCOL_PATH.as_posix():
        raise DiagnosticError("only the frozen experiments/drqv2-geometry-mix-v1-r6.json protocol is permitted")
    protocol = _pinned_json(protocol_path, expected_protocol_sha256, "geometry-mix protocol")
    if protocol.get("format") != PROTOCOL_FORMAT:
        raise DiagnosticError("geometry-mix protocol format mismatch")
    runtime = _validate_cpu21(protocol)
    _validate_study_contract(protocol)
    rows, catalog_lineage = _catalog_rows(root, protocol)
    catalog_protocol_path = _repo_file(root, protocol["catalog"]["protocol_path"], "experiments", "catalog protocol")
    catalog_protocol = _pinned_json(catalog_protocol_path, CATALOG_PROTOCOL_SHA256, "geometry catalog protocol")
    sources = _source_records(root, protocol, catalog_protocol)
    candidates = _candidate_records(root, protocol, expected_protocol_sha256, sources)
    roles: dict[str, dict[str, Any]] = {}
    actors = []
    for seed in (0, 1):
        source = sources[seed]
        loaded, action_adapter, observation_spec, weight_hash = _validate_actor(
            source["actor_path"], source["actor_sha256"], source["actor_weights_sha256"], loader=loader,
        )
        source["actor_weights_sha256"] = weight_hash
        roles[f"unchanged-source-seed{seed}"] = {
            **source, "actor": loaded, "action_adapter": action_adapter,
            "observation_spec": observation_spec,
        }
    for seed, variant in sorted(candidates, key=lambda item: (item[0], VARIANTS.index(item[1]))):
        candidate = candidates[(seed, variant)]
        loaded, action_adapter, observation_spec, weight_hash = _validate_actor(
            candidate["actor_path"], candidate["actor_sha256"], candidate["actor_weights_sha256"], loader=loader,
        )
        candidate["actor_weights_sha256"] = weight_hash
        roles[f"{variant}-seed{seed}"] = {
            **candidate, "actor": loaded, "action_adapter": action_adapter,
            "observation_spec": observation_spec,
        }
    if len(roles) != 8:
        raise DiagnosticError("model reload did not produce all eight exact actor roles")
    return protocol, {"sha256": expected_protocol_sha256, "path": protocol_path}, {
        **catalog_lineage, "runtime": runtime,
    }, [{"row": row, "roles": roles} for row in rows]


def build_environment(geometry_seed: int, max_steps: int = MAX_STEPS) -> Any:
    """Create the direct fixed-road HaicTrack stack, preserving raw no-op warmup."""
    from gymnasium.wrappers import TimeLimit
    from train import HaicTrack

    raw = HaicTrack(
        track_id=1, seed=geometry_seed, max_steps=max_steps,
        frame_skip=FRAME_SKIP, obstacles=True,
    )
    return TimeLimit(raw, max_episode_steps=max_steps)


def _find_wrapper(env: Any, required: tuple[str, ...]) -> Any:
    current = env
    for _ in range(16):
        if all(hasattr(current, name) for name in required):
            return current
        current = getattr(current, "env", None)
        if current is None:
            break
    raise DiagnosticError(f"required original environment wrapper is inaccessible: {required}")


def _road_hash(track: Any) -> str:
    points = np.asarray(track, dtype="<f8")
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise DiagnosticError("reset road geometry is invalid")
    return hashlib.sha256(np.ascontiguousarray(points[:, 2:4]).tobytes()).hexdigest()


def _road_sample(env: Any, step: int) -> dict[str, Any]:
    raw = env.unwrapped
    points = np.asarray(raw.track, dtype=np.float64)[:, 2:4]
    position = np.asarray(raw.car.hull.position, dtype=np.float64)
    velocity = np.asarray(raw.car.hull.linearVelocity, dtype=np.float64)
    if position.shape != (2,) or velocity.shape != (2,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise DiagnosticError("nearest-road pose/speed is invalid")
    distances = np.linalg.norm(points - position, axis=1)
    nearest = int(np.argmin(distances))
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    perimeter = float(lengths.sum())
    if not math.isfinite(perimeter) or perimeter <= 0:
        raise DiagnosticError("road perimeter is invalid")
    return {
        "step": step,
        "nearest_point_index": nearest,
        "nearest_point_fraction": float(lengths[:nearest].sum() / perimeter),
        "nearest_distance_m": float(distances[nearest]),
        "speed_m_s": float(np.linalg.norm(velocity)),
    }


def _termination_class(transition: Any, info: dict[str, Any], steps: int) -> str:
    if info.get("finished") is True:
        return "finished"
    retirement = info.get("retire_reason")
    if retirement in ("crash", "off_track", "out_of_bounds"):
        return retirement
    if transition.terminated:
        return "terminated"
    if transition.truncated and steps == MAX_STEPS:
        return "timeout"
    if transition.truncated:
        return "truncated"
    raise DiagnosticError("episode did not end on finish, retirement, or the frozen TimeLimit")


def _safe_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DiagnosticError(f"non-finite {label}")
    return number


def _collect_episode(env: Any, role: dict[str, Any], row: dict[str, Any], repeat: int,
                     *, episode_id: int, max_steps: int,
                     root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seed = row["geometry_seed"]
    collector = EpisodeCollector(
        env, action_adapter=role["action_adapter"], observation_spec=role["observation_spec"],
    )
    wrapper = _find_wrapper(env, ("off_track_counter", "max_off_track_steps", "warmup_steps"))
    if wrapper.max_off_track_steps != 100 or wrapper.warmup_steps != 50:
        raise DiagnosticError("original 50-raw-action no-op warmup/off-track contract changed")
    observation, reset_info = collector.reset(seed=None)
    if reset_info.get("seed") != seed or reset_info.get("track_id") != 1:
        raise DiagnosticError("direct HaicTrack reset did not use the frozen geometry seed and track 1")
    if _road_hash(env.unwrapped.track) != row["road_coordinate_sha256"]:
        raise DiagnosticError("reset road coordinates differ from the immutable TRAIN-DIAGNOSTIC catalog")
    collector_episode_id = collector.episode_id
    actor = role["actor"]
    reset_actor = getattr(actor, "reset_episode", None)
    if reset_actor is not None:
        reset_actor()
    sparse = [_road_sample(env, 0)]
    native, official = [], []
    rewards, progress, damage = [], [], []
    terminated, truncated, terminal, retirement = [], [], [], []
    finished, qualified, finish_times, collisions, off_track = [], [], [], [], []
    final_transition = None
    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            action = actor.act(observation, deterministic=True)
            transition = collector.step(action)
            info = transition.info
            if info.get("seed") != seed or info.get("track_id") != 1:
                raise DiagnosticError("transition escaped the frozen geometry-seed/track cell")
            native_action = np.asarray(transition.action, dtype=np.float32)
            official_action = np.asarray(transition.applied_action, dtype=np.float32)
            if (native_action.shape != (3,) or official_action.shape != (3,)
                    or not np.isfinite(native_action).all() or not np.isfinite(official_action).all()
                    or np.any(native_action < -1.0) or np.any(native_action > 1.0)
                    or not -1.0 <= official_action[0] <= 1.0
                    or np.any(official_action[1:] < 0.0) or np.any(official_action[1:] > 1.0)):
                raise DiagnosticError("native or official action left its frozen three-axis bounds")
            native.append(native_action.copy())
            official.append(official_action.copy())
            rewards.append(_safe_float(transition.reward, "raw reward"))
            progress.append(_safe_float(info["progress"], "progress"))
            damage.append(_safe_float(info["damage"], "damage"))
            terminated.append(bool(transition.terminated))
            truncated.append(bool(transition.truncated))
            terminal.append(bool(transition.terminal))
            retirement.append(str(info.get("retire_reason") or ""))
            finished.append(bool(info.get("finished", False)))
            qualified.append(bool(info.get("finish_qualified", False)))
            crossing_time = info.get("finish_time_s")
            finish_times.append(float(crossing_time) if crossing_time is not None else np.nan)
            collisions.append(bool(info.get("collision", False)))
            off_track.append(int(wrapper.off_track_counter))
            if step == 1 or step % SPARSE_STRIDE == 0 or transition.done or step == max_steps:
                if sparse[-1]["step"] != step:
                    sparse.append(_road_sample(env, step))
            final_transition = transition
            if transition.done:
                break
            observation = transition.next_observation
    if final_transition is None:
        raise DiagnosticError("collector did not produce any decisions")
    steps = len(rewards)
    last_info = final_transition.info
    termination = _termination_class(final_transition, last_info, steps)
    if termination == "timeout" and (last_info.get("finished") or last_info.get("retire_reason")):
        raise DiagnosticError("TimeLimit masked a finish or retirement")
    if any(not 0 <= value <= 1 for value in progress):
        raise DiagnosticError("progress is outside the original visited-tile fraction bounds")
    native_array = np.asarray(native, dtype=np.float32)
    official_array = np.asarray(official, dtype=np.float32)
    finish_time = last_info.get("finish_time_s")
    arrays = {
        "native_action": native_array,
        "official_action": official_array,
        "reward": np.asarray(rewards, dtype=np.float32),
        "progress": np.asarray(progress, dtype=np.float32),
        "damage": np.asarray(damage, dtype=np.float32),
        "terminated": np.asarray(terminated, dtype=np.bool_),
        "truncated": np.asarray(truncated, dtype=np.bool_),
        "terminal": np.asarray(terminal, dtype=np.bool_),
        "retirement": np.asarray(retirement, dtype="U32"),
        "finished": np.asarray(finished, dtype=np.bool_),
        "finish_qualified": np.asarray(qualified, dtype=np.bool_),
        "finish_crossing_time_s": np.asarray(finish_times, dtype=np.float64),
        "collision": np.asarray(collisions, dtype=np.bool_),
        "off_track_counter": np.asarray(off_track, dtype=np.int16),
        "episode_id": np.full(steps, episode_id, dtype=np.int32),
        "repeat_index": np.full(steps, repeat, dtype=np.int8),
        "source_actor_sha256": np.full(steps, role.get("source_actor_sha256", role["actor_sha256"]), dtype="U64"),
        "actor_sha256": np.full(steps, role["actor_sha256"], dtype="U64"),
        "checkpoint_sha256": np.full(steps, role["checkpoint_sha256"], dtype="U64"),
        "source_checkpoint_sha256": np.full(
            steps, role.get("source_checkpoint_sha256", role["checkpoint_sha256"]), dtype="U64",
        ),
        "sparse_step": np.asarray([item["step"] for item in sparse], dtype=np.int32),
        "sparse_nearest_point_index": np.asarray([item["nearest_point_index"] for item in sparse], dtype=np.int32),
        "sparse_nearest_point_fraction": np.asarray([item["nearest_point_fraction"] for item in sparse], dtype=np.float32),
        "sparse_nearest_distance_m": np.asarray([item["nearest_distance_m"] for item in sparse], dtype=np.float32),
        "sparse_speed_m_s": np.asarray([item["speed_m_s"] for item in sparse], dtype=np.float32),
    }
    if set(arrays) != set(TRACE_ARRAYS) or any(
        not np.isfinite(value).all() for name, value in arrays.items()
        if name != "finish_crossing_time_s" and value.dtype.kind in "fiu"
    ):
        raise DiagnosticError("per-step trace is incomplete or contains non-finite values")
    if np.isinf(arrays["finish_crossing_time_s"]).any():
        raise DiagnosticError("finish crossing timestamp is infinite")
    result = {
        "format": EPISODE_FORMAT,
        "role": role["role"],
        "arm": role["arm"],
        "variant": role.get("variant"),
        "source_learner_seed": role["source_learner_seed"],
        "checkpoint_online_step": role["checkpoint_online_step"],
        "actor_path": role["actor_path"].relative_to(root).as_posix(),
        "actor_sha256": role["actor_sha256"],
        "actor_weights_sha256": role["actor_weights_sha256"],
        "checkpoint_path": role["checkpoint_path"].relative_to(root).as_posix(),
        "checkpoint_sha256": role["checkpoint_sha256"],
        "checkpoint_manifest_path": role["checkpoint_manifest_path"].relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": role["checkpoint_manifest_sha256"],
        "source_actor_sha256": role.get("source_actor_sha256", role["actor_sha256"]),
        "source_checkpoint_sha256": role.get("source_checkpoint_sha256", role["checkpoint_sha256"]),
        "checkpoint_catalog_path": (
            role["catalog_path"].relative_to(root).as_posix() if role.get("catalog_path") else None
        ),
        "checkpoint_catalog_sha256": role.get("catalog_sha256"),
        "geometry_seed": seed,
        "track_id": 1,
        "family": row["family"],
        "repeat": repeat,
        "canonical_repeat": repeat == 0,
        "episode_id": episode_id,
        "collector_episode_id": collector_episode_id,
        "steps": steps,
        "raw_reward_sum": float(sum(rewards)),
        "finished": bool(last_info.get("finished", False)),
        "terminal_progress": progress[-1],
        "max_progress": max(progress),
        "terminal_damage": damage[-1],
        "max_damage": max(damage),
        "collision_actions": int(sum(collisions)),
        "off_track": termination == "off_track",
        "off_track_counter_max": max(off_track),
        "terminated": bool(final_transition.terminated),
        "truncated": bool(final_transition.truncated),
        "terminal": bool(final_transition.terminal),
        "retire_reason": last_info.get("retire_reason"),
        "termination_class": termination,
        "finish_qualified": bool(last_info.get("finish_qualified", False)),
        "finish_time_s": None if finish_time is None else _safe_float(finish_time, "finish time"),
        "warmup_raw_noop_actions": wrapper.warmup_steps,
        "warmup_raw_frames": wrapper.warmup_steps * FRAME_SKIP,
        "native_action_mean": native_array.mean(axis=0).tolist(),
        "official_action_mean": official_array.mean(axis=0).tolist(),
        "steering_mean": float(official_array[:, 0].mean()),
        "steering_abs_mean": float(np.abs(official_array[:, 0]).mean()),
        "throttle_mean": float(official_array[:, 1].mean()),
        "sparse_road_samples": sparse,
    }
    return result, arrays


def _trace_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in TRACE_ARRAYS:
        if name == "repeat_index":
            continue
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _axis_summary(rows: list[dict[str, Any]], field: str, index: int) -> dict[str, float]:
    values = np.asarray([row[field][index] for row in rows], dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "max": float(values.max())}


def _run_summary(episodes: list[dict[str, Any]], canonical_actions: dict[str, list[np.ndarray]],
                 protocol_sha256: str, catalog_sha256: str) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if row["repeat"] == 0:
            groups[(row["family"], row["arm"], row["source_learner_seed"])].append(row)
    summaries = []
    for (family, arm, seed), rows in sorted(groups.items()):
        native = [action for row in rows for action in canonical_actions[row["cell_id"]]]
        native_actions = np.concatenate(native, axis=0)
        official_actions = np.concatenate([row["_official_actions"] for row in rows], axis=0)
        termination_counts = Counter(row["termination_class"] for row in rows)
        summaries.append({
            "family": family,
            "arm": arm,
            "source_learner_seed": seed,
            "canonical_repeat": 0,
            "episode_count": len(rows),
            "finish_count": sum(row["finished"] for row in rows),
            "finish_fraction": float(np.mean([row["finished"] for row in rows])),
            "mean_max_progress": float(np.mean([row["max_progress"] for row in rows])),
            "off_track_count": sum(row["off_track"] for row in rows),
            "off_track_fraction": float(np.mean([row["off_track"] for row in rows])),
            "termination_class_counts": dict(sorted(termination_counts.items())),
            "native_action_axes": {
                axis: _axis_summary([{"a": action} for action in native_actions], "a", index)
                for index, axis in enumerate(("steer", "throttle", "brake"))
            },
            "official_action_axes": {
                axis: _axis_summary([{"a": action} for action in official_actions], "a", index)
                for index, axis in enumerate(("steer", "throttle", "brake"))
            },
        })
    return {
        "format": RUN_FORMAT,
        "role": "TRAIN-DIAGNOSTIC",
        "ranked": False,
        "score_selection": False,
        "performance_scope": "internal raw-reward TRAIN-DIAGNOSTIC only; not screen, confirmation, blind, fresh generalization, or official HAIC performance",
        "sample_interpretation": "one fixed training-only geometry run across two source actors and six final distribution candidates; repeat 0 is canonical and repeat 1 checks replay determinism; descriptive only, no significance test",
        "protocol_sha256": protocol_sha256,
        "catalog_sha256": catalog_sha256,
        "geometry_count": len(DIAGNOSTIC_SEEDS),
        "role_count": 8,
        "repeat_count": REPEATS,
        "expected_episode_count": len(DIAGNOSTIC_SEEDS) * 8 * REPEATS,
        "observed_episode_count": len(episodes),
        "canonical_repeat": 0,
        "family_arm_seed": summaries,
        "episodes_jsonl": "episodes.jsonl",
        "trace_directory": "traces/",
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def run_diagnostic(*, root: Path, protocol_path: Path, protocol_sha256: str,
                   output_root: Path, environment_factory: Callable[[int, int], Any] | None = None,
                   actor_loader: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Validate every frozen input before the first reset, then trace the full matrix."""
    root = root.resolve()
    output_root = _user_repo_file(root, output_root, "runs", "diagnostic output")
    if output_root.exists():
        raise FileExistsError(f"diagnostic output is immutable and already exists: {output_root}")
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise DiagnosticError("diagnostic output parent must already exist and must not be a symlink")
    loader = actor_loader or load_exported_actor
    protocol, protocol_pin, catalog, prepared_rows = _prepare(
        root, protocol_path, protocol_sha256, loader=loader,
    )
    roles = prepared_rows[0]["roles"]
    expected_cells = {
        (row["geometry_seed"], role_key, repeat)
        for row in (item["row"] for item in prepared_rows)
        for role_key in roles
        for repeat in range(REPEATS)
    }
    if len(expected_cells) != len(DIAGNOSTIC_SEEDS) * 8 * REPEATS:
        raise DiagnosticError("frozen diagnostic matrix contains reused cells")

    out = output_root
    out.mkdir(parents=False, exist_ok=False)
    traces_dir = out / "traces"
    traces_dir.mkdir()
    episodes_path = out / "episodes.jsonl"
    episode_rows: list[dict[str, Any]] = []
    canonical_actions: dict[str, list[np.ndarray]] = {}
    canonical_cells: dict[tuple[int, str], dict[str, Any]] = {}
    observed: set[tuple[int, str, int]] = set()
    trace_inventory: dict[str, str] = {}
    factory = environment_factory or (lambda seed, steps: build_environment(seed, steps))
    for item in prepared_rows:
        row = item["row"]
        for role_key, role in roles.items():
            cell_episode_id = len(canonical_cells)
            env = factory(row["geometry_seed"], MAX_STEPS)
            try:
                for repeat in range(REPEATS):
                    cell = (row["geometry_seed"], role_key, repeat)
                    if cell in observed:
                        raise DiagnosticError(f"reused diagnostic cell: {cell}")
                    result, arrays = _collect_episode(
                        env, role, row, repeat, episode_id=cell_episode_id,
                        max_steps=MAX_STEPS, root=root,
                    )
                    relative = Path("traces") / f"seed-{row['geometry_seed']}" / role_key / f"repeat-{repeat}.npz"
                    trace_path = out / relative
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with trace_path.open("xb") as destination:
                        np.savez_compressed(destination, **arrays)
                    result["trace_path"] = relative.as_posix()
                    result["trace_sha256"] = sha256_file(trace_path)
                    result["trace_deterministic_sha256"] = _trace_digest(arrays)
                    result["cell_id"] = f"seed-{row['geometry_seed']}:track-1:{role_key}:repeat-{repeat}"
                    result["catalog_sha256"] = CATALOG_SHA256
                    result["protocol_sha256"] = protocol_sha256
                    result["_native_actions"] = arrays["native_action"]
                    result["_official_actions"] = arrays["official_action"]
                    role_cell = (row["geometry_seed"], role_key)
                    if repeat == 0:
                        canonical_actions[result["cell_id"]] = [arrays["native_action"]]
                        canonical_cells[role_cell] = result
                    else:
                        canonical = canonical_cells.get(role_cell)
                        if canonical is None or result["trace_deterministic_sha256"] != canonical["trace_deterministic_sha256"]:
                            raise DiagnosticError(f"repeat trace diverged from canonical repeat 0: {cell}")
                        result["repeat_trace_agrees_with_repeat0"] = True
                        canonical["repeat_trace_agrees_with_repeat0"] = True
                    observed.add(cell)
                    trace_inventory[relative.as_posix()] = result["trace_sha256"]
                    episode_rows.append(result)
            finally:
                env.close()
    if observed != expected_cells:
        raise DiagnosticError(f"episode matrix has missing/extra cells: missing={expected_cells - observed}")
    with episodes_path.open("x", encoding="utf-8") as episodes_stream:
        for row in episode_rows:
            serialized = {key: value for key, value in row.items() if not key.startswith("_")}
            episodes_stream.write(json.dumps(serialized, sort_keys=True, allow_nan=False) + "\n")
    summary = _run_summary(episode_rows, canonical_actions, protocol_sha256, CATALOG_SHA256)
    for row in episode_rows:
        row.pop("_official_actions", None)
        row.pop("_native_actions", None)
    _write_json(out / "summary.json", summary)
    inventory = {
        "episodes.jsonl": sha256_file(episodes_path),
        "summary.json": sha256_file(out / "summary.json"),
        **trace_inventory,
    }
    manifest = {
        "format": RUN_FORMAT,
        "role": "TRAIN-DIAGNOSTIC",
        "ranked": False,
        "score_selection": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": protocol_pin["path"].relative_to(root).as_posix(),
        "protocol_sha256": protocol_sha256,
        "catalog": catalog,
        "episode_count": len(episode_rows),
        "role_identities": [
            {key: value for key, value in role.items()
             if key in ("role", "arm", "variant", "source_learner_seed", "actor_sha256",
                        "actor_weights_sha256", "checkpoint_sha256", "checkpoint_online_step",
                        "checkpoint_manifest_sha256", "catalog_sha256")}
            for role in roles.values()
        ],
        "repeat_replay_determinism": "every repeat 1 trace must match repeat 0 across actions, raw rewards, progress, damage, terminal/retirement/finish/collision state, episode identity and sparse road features",
        "cell_inventory": "exactly 16 TRAIN-DIAGNOSTIC seeds x 8 roles x 2 repeats; no outcome-based filtering",
        "files_sha256": dict(sorted(inventory.items())),
    }
    _write_json(out / "manifest.json", manifest)
    with (out / "manifest.sha256").open("x", encoding="ascii") as sidecar:
        sidecar.write(f"{sha256_file(out / 'manifest.json')}  manifest.json\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new runs/ directory; existing outputs are never overwritten")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        summary = run_diagnostic(
            root=args.repo_root, protocol_path=args.protocol,
            protocol_sha256=args.protocol_sha256, output_root=args.output_root,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"geometry-mix TRAIN-DIAGNOSTIC stopped: {error}\n")
    print(json.dumps({
        "output_root": str(args.output_root),
        "episode_count": summary["observed_episode_count"],
        "role_count": summary["role_count"],
        "geometry_count": summary["geometry_count"],
        "protocol_sha256": summary["protocol_sha256"],
        "catalog_sha256": summary["catalog_sha256"],
        "ranked": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
