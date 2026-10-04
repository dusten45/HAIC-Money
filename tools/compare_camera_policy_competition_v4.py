"""Source-bound clear-road camera evaluation on fresh geometry.

Binding is a one-time operation after the v4 runner and comparator are committed.
All episodes use the existing cold worker; this module never imports an older
camera-policy decision or seal implementation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import competition_camera_gate_v4 as gate
from tools import evaluate_bare_generalization as fresh


NAME = "camera-policy-competition-v4"
TEMPLATE_PATH = ROOT / "experiments" / f"{NAME}.template.json"
PROTOCOL_PATH = ROOT / "experiments" / f"{NAME}.json"
OUTPUT_ROOT = ROOT / ".haic-artifacts" / NAME / "run"
SOURCE_PATHS = {arm: f".haic-artifacts/{NAME}/snapshots/{arm}_agent.py"
                for arm in fresh.ARMS}
CONTROL_CLASS = "_CompoundClearingBrakeCarryController"
CANDIDATE_CLASS = "_ClearRoadRow42DropoutController"
CONTROL_IMPLEMENTATION_COMMIT = "f12c5b62c5f02bff4053e6f4775d202d79917938"
CONTROL_BLOB_SHA256 = "291d93081a64d49f18507ec7baaba411e06510bb533221ce07ae585bc4e77b61"
# The committed implementation remains on the live control route until promotion.
CANDIDATE_BASE_COMMIT = "7df85ae822fa3234b9ccd30844776420f0a56012"
CANDIDATE_BASE_BLOB_SHA256 = "b639b9595da2e117410f1556eb13ab8576b873c464e689f5dd1fd8f38f2157c0"
GATE_SHA256 = "7bd905e6e83d4035528c0cec375c964c0e094c1dfef3f63dd916c70320b607d5"
REQUIREMENTS_BLOB_SHA256 = "7fe177422d03b66b720a30c14b2dff05da95db95e238d3841afd76cb09a73ea2"
PARTICIPANTS_COMMIT = "dfb7a2de2178825ca5c5ce20bab01ba67052ba31"
RUNTIME_PROVENANCE = {"python": "3.11", "torch": "2.1.0", "numpy": "1.26.0",
                      "gymnasium": "0.29.1", "opencv_python": "4.8.1.78",
                      "box2d": "2.3.10"}
TRACKS = [1, 2, 3, 4]
SEED_COUNTS = {"screen": 8, "confirmation": 16, "blind": 8}
SPOT_INDEX = {"screen": ((1, 0), (4, 7)),
              "confirmation": ((1, 0), (2, 1), (3, 2), (4, 3)),
              "blind": ((1, 0), (4, 7))}
THRESHOLDS = gate.THRESHOLDS
compare_pairs = gate.compare_pairs
combined_decision = gate.combined_decision
PRIOR_REJECTION_RESULTS = (
    ("camera-policy-generalization-v1-result.json",
     "c56af797774b07d7874d5d4947847c3d84f9957309fab65358f048d6b0593ebc"),
    ("camera-policy-generalization-v2-result.json",
     "ae4618a2bcf34541a33536402bfda835963f0f4cff2ede88159302da8538e724"),
    ("camera-policy-competition-v3-result.json",
     "b10a5cd7a4b74fff9d01b09162ad60a269d33da5353780e0eb6539bac9424cc5"),
)
EVALUATION_POLICY = {
    "profile": "practical",
    "consecutive_prior_rejections": 3,
    "prior_rejection_results": dict(PRIOR_REJECTION_RESULTS),
    "selected_before_geometry_binding": True,
    "maximum_candidates": 2,
    "maximum_fresh_studies": 2,
    "fallback": {
        "trigger": "The practical fresh study rejects before promotion.",
        "minimum_net_finish_gain": {"screen": 1, "confirmation": 2, "blind": 1},
        "minimum_combined_net_finish_gain": 8,
        "remaining_thresholds": "Unchanged practical safety, loss, pace, seed and track floors.",
        "requires_distinct_protocol_and_unused_geometry": True,
        "maximum_additional_studies": 1,
    },
    "historical_decisions_and_sealed_holdouts_unchanged": True,
    "no_rescoring_bound_study": True,
}


def prior_rejection_evidence(experiments_dir: Path) -> list[dict]:
    """Validate the recorded failures that selected this profile before seeds."""
    evidence = []
    for name, expected in PRIOR_REJECTION_RESULTS:
        path = experiments_dir / name
        if not path.is_file() or fresh.digest(path) != expected:
            raise ValueError(f"historical prior rejection result changed: {name}")
        result = fresh._read_json(path)
        if not isinstance(result, dict) or result.get("decision") != "REJECT":
            raise ValueError(f"historical prior rejection decision changed: {name}")
        evidence.append({"path": f"experiments/{name}", "sha256": expected,
                         "decision": "REJECT"})
    return evidence


def validate_template(template: dict) -> None:
    expected = {"schema_version": 1, "name": NAME, "status": "UNBOUND",
                "control_implementation_commit": CONTROL_IMPLEMENTATION_COMMIT,
                "control_blob_sha256": CONTROL_BLOB_SHA256,
                "candidate_base_commit": CANDIDATE_BASE_COMMIT,
                "candidate_base_blob_sha256": CANDIDATE_BASE_BLOB_SHA256,
                "decision_engine_sha256": GATE_SHA256,
                "requirements_blob_sha256": REQUIREMENTS_BLOB_SHA256,
                "control_class": CONTROL_CLASS, "candidate_class": CANDIDATE_CLASS,
                "track_ids": TRACKS, "seed_counts": SEED_COUNTS,
                "spot_indices": {phase: [list(cell) for cell in SPOT_INDEX[phase]]
                                 for phase in fresh.PHASES},
                "evaluation_policy": EVALUATION_POLICY}
    if not isinstance(template, dict) or any(template.get(key) != value
                                             for key, value in expected.items()):
        raise ValueError("unbound v4 template contract changed")
    if any(key in template for key in ("partitions", "seed_salt_hex", "seeds")):
        raise ValueError("unbound v4 template must contain no seeds")


def derived_seeds(salt: str, phase: str) -> list[int]:
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        raise ValueError("seed salt must be 128-bit lowercase hex")
    if phase not in SEED_COUNTS:
        raise ValueError("unknown phase")
    return [int.from_bytes(hashlib.sha256(
        f"{NAME}:{salt}:{phase}:{index}".encode("ascii")).digest()[:4], "big")
        for index in range(SEED_COUNTS[phase])]


def derived_seed_partitions(salt: str) -> dict[str, list[int]]:
    return {phase: derived_seeds(salt, phase) for phase in fresh.PHASES}


def require_fresh_seeds(seeds: dict[str, list[int]], historical: set[int]) -> None:
    if set(seeds) != set(fresh.PHASES) or any(
            not isinstance(seeds[phase], list) or len(seeds[phase]) != SEED_COUNTS[phase]
            or any(type(seed) is not int or not 0 <= seed <= fresh.MAX_SEED
                   for seed in seeds[phase]) for phase in fresh.PHASES):
        raise ValueError("invalid fresh seed partitions")
    flat = [seed for phase in fresh.PHASES for seed in seeds[phase]]
    if len(set(flat)) != len(flat):
        raise ValueError("geometry seed collision: all v4 seeds must be distinct")
    if set(flat) & historical:
        raise ValueError("historical geometry seed reuse is not fresh")


def historical_geometry_seeds_v4(experiments_dir: Path, protocol_path: Path) -> set[int]:
    """Protect all documented seeds, discounting only this protocol's own result."""
    def documented(value: object) -> set[int]:
        protected: set[int] = set()

        def inspect(child: object) -> None:
            if isinstance(child, dict):
                for key, item in child.items():
                    if key in ("seed", "geometry_seed") and type(item) is int and 0 <= item <= fresh.MAX_SEED:
                        protected.add(item)
                    if key in ("seeds", "geometry_seeds") or ("reserved" in key and "seed" in key):
                        protected.update(fresh._seed_list(item))
                    if key.endswith("_cells") and isinstance(item, list):
                        for cell in item:
                            if (isinstance(cell, (list, tuple)) and len(cell) >= 2
                                    and type(cell[1]) is int and 0 <= cell[1] <= fresh.MAX_SEED):
                                protected.add(cell[1])
                    inspect(item)
            elif isinstance(child, list):
                for item in child:
                    inspect(item)

        inspect(value)
        return protected

    result_path = experiments_dir / f"{NAME}-result.json"
    own_seeds = documented(fresh._read_json(protocol_path)) if protocol_path.exists() else set()
    protected: set[int] = set()
    for path in sorted(experiments_dir.glob("*.json")):
        if path.resolve() == protocol_path.resolve():
            continue
        record = fresh._read_json(path)
        if path.resolve() == result_path.resolve():
            if (not protocol_path.exists()
                    or not isinstance(record, dict)
                    or record.get("protocol_sha256") != fresh.digest(protocol_path)):
                raise ValueError("v4 result protocol SHA256 mismatch")
            protected.update(documented(record) - own_seeds)
        else:
            protected.update(documented(record))
    return protected


def _decision_engine_sha256() -> str:
    measured = fresh.digest(Path(gate.__file__))
    if measured != GATE_SHA256:
        raise ValueError("fixed v4 decision engine SHA256 changed")
    return measured


def _environment_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "local_simulator").rglob("*.py"))
    paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): fresh.digest(path)
            for path in paths}


def _committed_bytes(path: str, commit: str = "HEAD") -> bytes:
    try:
        return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{path} must exist in committed Git source") from error


def _requirements_blob_sha256() -> str:
    measured = hashlib.sha256(_committed_bytes("requirements.txt")).hexdigest()
    if measured != REQUIREMENTS_BLOB_SHA256:
        raise ValueError("committed requirements.txt provenance changed")
    return measured


def _require_committed_runtime() -> None:
    for name, path in (("v4 runner", Path(__file__)),
                       ("v4 decision engine", Path(gate.__file__)),
                       ("v4 template", TEMPLATE_PATH)):
        relative = path.resolve().relative_to(ROOT).as_posix()
        if _committed_bytes(relative) != path.read_bytes():
            raise ValueError(f"{name} must be committed before binding")


def _committed_agent_sources() -> tuple[bytes, bytes]:
    if not re.fullmatch(r"[0-9a-f]{40}", CANDIDATE_BASE_COMMIT) or not re.fullmatch(
            r"[0-9a-f]{64}", CANDIDATE_BASE_BLOB_SHA256):
        raise ValueError("v4 candidate implementation commit and blob are not pinned")
    control = _committed_bytes("agent.py", CONTROL_IMPLEMENTATION_COMMIT)
    candidate_base = _committed_bytes("agent.py", CANDIDATE_BASE_COMMIT)
    selector = f"{CONTROL_CLASS}()".encode("ascii")
    if (hashlib.sha256(control).hexdigest() != CONTROL_BLOB_SHA256
            or control.count(selector) != 1
            or re.search(rb"(?m)^class " + CANDIDATE_CLASS.encode("ascii") + rb"\b", control)):
        raise ValueError("pinned v4 control blob or selector changed")
    subclass = (rb"(?m)^class " + CANDIDATE_CLASS.encode("ascii") +
                rb"\(\s*_BoundedSideHoldController\s*\)\s*:")
    if (hashlib.sha256(candidate_base).hexdigest() != CANDIDATE_BASE_BLOB_SHA256
            or candidate_base == control or candidate_base.count(selector) != 1
            or re.search(subclass, candidate_base) is None):
        raise ValueError("pinned v4 candidate base blob, selector or class changed")
    return control, candidate_base


def selected_source(source: bytes) -> bytes:
    old = f"{CONTROL_CLASS}()".encode("ascii")
    new = f"{CANDIDATE_CLASS}()".encode("ascii")
    if source.count(old) != 1 or re.search(
            rb"(?m)^class " + CANDIDATE_CLASS.encode("ascii") + rb"\b", source) is None:
        raise ValueError("source must contain the frozen candidate class and one control selector")
    return source.replace(old, new, 1)


def require_root_model(model: Path) -> None:
    if model.resolve() != (ROOT / "model.pt").resolve():
        raise ValueError("v4 binding and evaluation require the root model.pt path")


def _runtime_versions() -> tuple[str, str, str, str, str, str]:
    from importlib import metadata
    import Box2D
    import numpy as np
    import torch

    torch_distribution = metadata.version("torch")
    numpy_distribution = metadata.version("numpy")
    if (torch.__version__.split("+", 1)[0] != torch_distribution
            or np.__version__ != numpy_distribution):
        raise ValueError("v4 imported Torch or NumPy differs from installed distribution")
    return (f"{sys.version_info.major}.{sys.version_info.minor}",
            torch_distribution, numpy_distribution, metadata.version("gymnasium"),
            metadata.version("opencv-python"), str(Box2D.__version__))


def require_runtime_versions() -> None:
    python, torch, numpy, gymnasium, opencv, box2d = _runtime_versions()
    if python != RUNTIME_PROVENANCE["python"]:
        raise ValueError(f"v4 requires Python 3.11, found {python}")
    if torch != RUNTIME_PROVENANCE["torch"]:
        raise ValueError(f"v4 requires Torch 2.1.0, found {torch}")
    if numpy != RUNTIME_PROVENANCE["numpy"]:
        raise ValueError(f"v4 requires NumPy 1.26.0, found {numpy}")
    if gymnasium != RUNTIME_PROVENANCE["gymnasium"]:
        raise ValueError(f"v4 requires Gymnasium 0.29.1, found {gymnasium}")
    if opencv != RUNTIME_PROVENANCE["opencv_python"]:
        raise ValueError(f"v4 requires OpenCV 4.8.1.78, found {opencv}")
    if box2d != RUNTIME_PROVENANCE["box2d"]:
        raise ValueError(f"v4 requires Box2D 2.3.10, found {box2d}")


def validate_protocol(protocol: dict, historical_seeds: set[int]) -> None:
    # Reject an unbound template before deriving any geometry or opening a worker.
    if not isinstance(protocol, dict) or protocol.get("name") != NAME or protocol.get("status") != "PREREGISTERED":
        raise ValueError("unbound v4 protocol: committed source and one-time salt required")
    salt = protocol.get("seed_salt_hex")
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        raise ValueError("bound v4 protocol needs its one-time seed salt")
    fresh.validate_protocol(protocol, historical_seeds)
    if protocol.get("source_snapshots") != SOURCE_PATHS:
        raise ValueError("v4 source snapshot paths changed")
    if protocol.get("controller_classes") != {"control": CONTROL_CLASS, "candidate": CANDIDATE_CLASS}:
        raise ValueError("v4 route classes changed")
    construction = protocol.get("source_construction", {})
    if (construction.get("control_implementation_commit") != CONTROL_IMPLEMENTATION_COMMIT
            or construction.get("control_blob_sha256") != CONTROL_BLOB_SHA256
            or construction.get("control_blob_sha256") != protocol.get("control_agent_sha256")
            or construction.get("candidate_base_commit") != CANDIDATE_BASE_COMMIT
            or construction.get("candidate_base_blob_sha256") != CANDIDATE_BASE_BLOB_SHA256
            or construction.get("selector_from") != f"{CONTROL_CLASS}()"
            or construction.get("selector_to") != f"{CANDIDATE_CLASS}()"):
        raise ValueError("pinned v4 two-blob source construction changed")
    control_base, candidate_base = _committed_agent_sources()
    if (hashlib.sha256(control_base).hexdigest() != protocol.get("control_agent_sha256")
            or hashlib.sha256(selected_source(candidate_base)).hexdigest() !=
            protocol.get("candidate_agent_sha256")):
        raise ValueError("bound v4 source hashes differ from pinned committed blobs")
    if protocol.get("training") is not False:
        raise ValueError("v4 evaluation requires fixed bare Agent sources")
    if (protocol.get("official_participants_commit") != PARTICIPANTS_COMMIT
            or protocol.get("runtime_provenance") != RUNTIME_PROVENANCE):
        raise ValueError("v4 official source or runtime provenance changed")
    if protocol.get("requirements_blob_sha256") != _requirements_blob_sha256():
        raise ValueError("v4 committed requirements provenance changed")
    if protocol.get("decision_thresholds") != THRESHOLDS:
        raise ValueError("v4 decision thresholds changed")
    if (protocol.get("evaluation_policy") != EVALUATION_POLICY
            or protocol.get("prior_rejection_evidence") !=
            prior_rejection_evidence(ROOT / "experiments")):
        raise ValueError("v4 preregistered practical evaluation policy changed")
    if protocol.get("template_sha256") != fresh.digest(TEMPLATE_PATH):
        raise ValueError("v4 unbound template hash changed")
    if (protocol.get("decision_engine_sha256") != _decision_engine_sha256()
            or protocol.get("runner_sha256") != fresh.digest(Path(__file__))
            or protocol.get("evaluation_harness_sha256") != fresh.digest(Path(fresh.__file__))):
        raise ValueError("bound v4 decision engine, runner, or harness hash changed")
    if protocol.get("environment_sha256") != _environment_hashes():
        raise ValueError("bound v4 environment hashes changed")
    seeds = derived_seed_partitions(salt)
    require_fresh_seeds(seeds, historical_seeds)
    for phase in fresh.PHASES:
        partition = protocol["partitions"][phase]
        if partition.get("track_ids") != TRACKS or partition.get("seeds") != seeds[phase]:
            raise ValueError(f"{phase} registered v4 track or seed grid changed")
        spots = [[track, seeds[phase][index]] for track, index in SPOT_INDEX[phase]]
        if partition.get("spot_check_cells") != spots:
            raise ValueError(f"{phase} v4 deterministic repeat cells changed")


def write_new_protocol(path: Path, protocol: dict) -> None:
    """Persist the one-time binding with exclusive creation and durable bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(fresh._canonical(protocol) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def bind_source(template: dict, model: Path, protocol_path: Path,
                control: Path, candidate: Path) -> dict:
    """Bind once to two pinned committed sources; write protocol last."""
    validate_template(template)
    require_runtime_versions()
    if any(path.exists() for path in (protocol_path, control, candidate)):
        raise ValueError("bound v4 protocol or snapshot already exists")
    _require_committed_runtime()
    source, candidate_base = _committed_agent_sources()
    selected = selected_source(candidate_base)
    require_root_model(model)
    model_hash = fresh.digest(model)
    helpers = {name: fresh.digest(ROOT / name) for name in
               ("action_smoothing.py", "action_representation.py")}
    environment = _environment_hashes()
    harness_hash = fresh.digest(Path(fresh.__file__))
    runner_hash = fresh.digest(Path(__file__))
    engine_hash = _decision_engine_sha256()
    requirements_hash = _requirements_blob_sha256()
    prior_rejections = prior_rejection_evidence(ROOT / "experiments")
    historical = historical_geometry_seeds_v4(ROOT / "experiments", protocol_path)
    salt = secrets.token_hex(16)
    seeds = derived_seed_partitions(salt)
    require_fresh_seeds(seeds, historical)
    partitions = {phase: {"track_ids": TRACKS, "seeds": seeds[phase],
                          "spot_check_cells": [[track, seeds[phase][index]]
                                               for track, index in SPOT_INDEX[phase]]}
                  for phase in fresh.PHASES}
    protocol = {
        "schema_version": 1, "name": NAME, "status": "PREREGISTERED",
        "hypothesis": "The frozen clear-road camera route improves fresh official-generator completion under the fixed phase, seed-cluster, combined finish, and track gates.",
        "evidence_scope": "Fresh source-bound paired screen, confirmation and blind; prior geometry is development only.",
        "control_agent_sha256": hashlib.sha256(source).hexdigest(),
        "candidate_agent_sha256": hashlib.sha256(selected).hexdigest(),
        "model_sha256": model_hash, "runtime_helper_sha256": helpers,
        "controller_classes": {"control": CONTROL_CLASS, "candidate": CANDIDATE_CLASS},
        "source_snapshots": SOURCE_PATHS,
        "source_construction": {"control_implementation_commit": CONTROL_IMPLEMENTATION_COMMIT,
                                "control_blob_sha256": hashlib.sha256(source).hexdigest(),
                                "candidate_base_commit": CANDIDATE_BASE_COMMIT,
                                "candidate_base_blob_sha256": hashlib.sha256(candidate_base).hexdigest(),
                                "selector_from": f"{CONTROL_CLASS}()",
                                "selector_to": f"{CANDIDATE_CLASS}()",
                                "rule": "Control is the f12 blob; candidate is the distinct clear-road implementation blob with one bare Agent route selector swap."},
        "seed_salt_hex": salt,
        "seed_derivation": "First four bytes of SHA256(name:salt:phase:index), unsigned big-endian uint32.",
        "seed_freshness_audit": f"All 32 v4 seeds are distinct and absent from {len(historical)} documented historical seeds at binding.",
        "template_sha256": fresh.digest(TEMPLATE_PATH),
        "official_participants_commit": PARTICIPANTS_COMMIT,
        "runtime_provenance": RUNTIME_PROVENANCE,
        "requirements_blob_sha256": requirements_hash,
        "decision_engine_sha256": engine_hash, "runner_sha256": runner_hash,
        "evaluation_harness_sha256": harness_hash, "environment_sha256": environment,
        "max_steps": 2000, "frame_skip": 4, "training": False,
        "partitions": partitions, "decision_thresholds": THRESHOLDS,
        "evaluation_policy": EVALUATION_POLICY,
        "prior_rejection_evidence": prior_rejections,
        "decision_rule": "Preregistered practical V4 phase, seed-cluster, and combined track gates apply; all phases RETAIN and at least 12 combined net finishes are needed for activation. A fallback requires a distinct later protocol with new unused seeds; bound V4 is never rescored.",
        "evaluation_budget": "32/64/32 canonical pairs plus 2/4/2 exact repeat pairs; at most 272 cold episodes.",
        "promotion_gate": "Screen and confirmation phase and seed-cluster RETAIN seals unlock later phases; blind RETAIN and the combined finish and per-track gates decide activation.",
    }
    validate_protocol(protocol, historical)
    created = []
    try:
        for path, content in ((control, source), (candidate, selected)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
            created.append(path)
        write_new_protocol(protocol_path, protocol)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return protocol


def validate_source_pair(protocol: dict, control: Path, candidate: Path) -> None:
    """Prove each snapshot's exact relationship to its own pinned Git blob."""
    control_base, candidate_base = _committed_agent_sources()
    selected = selected_source(candidate_base)
    if (control.read_bytes() != control_base
            or candidate.read_bytes() != selected):
        raise ValueError("v4 source snapshots differ from pinned blobs or selector swap")
    if (fresh.digest(control) != protocol["control_agent_sha256"]
            or fresh.digest(candidate) != protocol["candidate_agent_sha256"]
            or hashlib.sha256(candidate_base).hexdigest() !=
            protocol["source_construction"]["candidate_base_blob_sha256"]):
        raise ValueError("v4 source snapshot SHA256 mismatch")


def restore_snapshots(protocol: dict, control: Path, candidate: Path) -> None:
    """Reconstruct ignored snapshots from both immutable pinned Git blobs."""
    source, candidate_base = _committed_agent_sources()
    selected = selected_source(candidate_base)
    if (hashlib.sha256(source).hexdigest() != protocol["control_agent_sha256"]
            or hashlib.sha256(selected).hexdigest() != protocol["candidate_agent_sha256"]):
        raise ValueError("registered source SHA256 differs from pinned committed source")
    for path, expected in ((control, source), (candidate, selected)):
        if path.exists() and path.read_bytes() != expected:
            raise ValueError(f"existing v4 snapshot differs; refusing overwrite: {path}")
    created = []
    try:
        for path, expected in ((control, source), (candidate, selected)):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(expected)
                created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    validate_source_pair(protocol, control, candidate)


def verify_committed_protocol(path: Path) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    if _committed_bytes(relative) != path.read_bytes():
        raise ValueError("v4 protocol must be committed before any phase")


def build_identity(protocol_path: Path, protocol: dict, paths: dict[str, Path]) -> dict:
    identity = fresh.build_identity(protocol_path, protocol, paths["control"],
                                    paths["candidate"], paths["model"])
    return {**identity, "runner_sha256": fresh.digest(Path(__file__)),
            "decision_engine_sha256": _decision_engine_sha256(),
            "protocol_name": NAME, "candidate_route_class": CANDIDATE_CLASS,
            "template_sha256": fresh.digest(TEMPLATE_PATH)}


def check_frozen_inputs(identity: dict, protocol: dict, paths: dict[str, Path]) -> None:
    fresh.check_frozen_inputs(identity, paths)
    if (identity["runner_sha256"] != fresh.digest(Path(__file__))
            or identity["decision_engine_sha256"] != _decision_engine_sha256()
            or identity["template_sha256"] != fresh.digest(TEMPLATE_PATH)
            or identity["protocol_name"] != NAME
            or identity["candidate_route_class"] != CANDIDATE_CLASS):
        raise ValueError("v4 runner, decision engine, template, or route changed")
    if (identity["runner_sha256"] != protocol["runner_sha256"]
            or identity["decision_engine_sha256"] != protocol["decision_engine_sha256"]
            or identity["template_sha256"] != protocol["template_sha256"]
            or identity["harness_sha256"] != protocol["evaluation_harness_sha256"]
            or identity["environment_sha256"] != protocol["environment_sha256"]):
        raise ValueError("bound v4 runtime hashes changed")
    if protocol["requirements_blob_sha256"] != _requirements_blob_sha256():
        raise ValueError("bound v4 requirements provenance changed")
    validate_source_pair(protocol, paths["control"], paths["candidate"])


def _unexpected_receipts(root: Path, protocol: dict, phase: str) -> list[str]:
    expected = {fresh.cell_path(root, phase, arm, track, seed, repeat)
                for track, seed, repeat in fresh.expected_cells(protocol, phase)
                for arm in fresh.ARMS}
    phase_root = root / "cells" / phase
    for directory in (root, root / "cells", phase_root):
        if directory.is_symlink():
            return [f"symlink v4 receipt directory: {directory}"]
    if not phase_root.exists():
        return []
    problems = []
    for path in sorted(phase_root.rglob("*")):
        if path.is_symlink():
            problems.append(f"symlink v4 receipt path: {path}")
        elif path.is_file() and path not in expected:
            problems.append(f"unexpected v4 receipt: {path}")
    return problems


def report_phase(root: Path, identity: dict, protocol: dict, phase: str) -> dict:
    if phase not in fresh.PHASES:
        raise ValueError("unknown v4 phase")
    cells = fresh.expected_cells(protocol, phase)
    rows = []
    hashed_paths = []
    for track, seed, repeat in cells:
        for arm in fresh.ARMS:
            path = fresh.cell_path(root, phase, arm, track, seed, repeat)
            before = fresh.digest(path) if path.is_file() else None
            row = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
            after = fresh.digest(path) if path.is_file() else None
            if before != after:
                raise ValueError(f"v4 receipt changed while reporting: {path}")
            if row is not None:
                rows.append(row)
                if after is None:
                    raise ValueError(f"v4 receipt vanished while reporting: {path}")
                hashed_paths.append((path.relative_to(root).as_posix(), after))
    summary = compare_pairs(rows, cells, phase)
    tree_input = "".join(f"{name}\0{digest}\n"
                         for name, digest in sorted(hashed_paths))
    summary["receipt_tree_sha256"] = hashlib.sha256(tree_input.encode("utf-8")).hexdigest()
    extras = _unexpected_receipts(root, protocol, phase)
    if extras:
        summary["reasons"].extend(extras)
        summary["decision"] = "REJECT"
    summary.update({"partition": phase, "protocol_sha256": identity["protocol_sha256"],
                    "control_agent_sha256": identity["source_sha256"]["control"],
                    "candidate_agent_sha256": identity["source_sha256"]["candidate"],
                    "model_sha256": identity["model_sha256"],
                    "harness_sha256": identity["harness_sha256"],
                    "helper_sha256": identity.get("helper_sha256"),
                    "environment_sha256": identity.get("environment_sha256"),
                    "runner_sha256": identity["runner_sha256"],
                    "decision_engine_sha256": identity["decision_engine_sha256"]})
    return summary


def _require_seal(root: Path, identity: dict, protocol: dict,
                  name: str, predecessor: str) -> None:
    seal = fresh._read_seal(root, identity, name)
    if seal is None:
        raise ValueError(f"v4 {name} seal is missing")
    summary = report_phase(root, identity, protocol, predecessor)
    if hashlib.sha256(fresh._canonical(summary)).hexdigest() != seal["summary_sha256"]:
        raise ValueError(f"v4 {name} seal no longer matches {predecessor} receipts")


def require_phase(root: Path, identity: dict, protocol: dict, phase: str) -> None:
    if phase == "screen":
        return
    if phase == "confirmation":
        _require_seal(root, identity, protocol, "finalist-v4", "screen")
        return
    if phase == "blind":
        require_phase(root, identity, protocol, "confirmation")
        _require_seal(root, identity, protocol, "confirmation-accepted-v4", "confirmation")
        return
    raise ValueError("unknown v4 phase")


def freeze_finalist(root: Path, identity: dict, protocol: dict,
                    paths: dict[str, Path] | None = None) -> None:
    summary = report_phase(root, identity, protocol, "screen")
    if summary["decision"] != "RETAIN":
        raise ValueError("v4 screen must RETAIN before finalist freeze")
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    fresh._write_seal(root, identity, "finalist-v4", summary)
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)


def seal_confirmation(root: Path, identity: dict, protocol: dict,
                      paths: dict[str, Path] | None = None) -> None:
    require_phase(root, identity, protocol, "confirmation")
    summary = report_phase(root, identity, protocol, "confirmation")
    if summary["decision"] != "RETAIN":
        raise ValueError("v4 confirmation must RETAIN before blind")
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    fresh._write_seal(root, identity, "confirmation-accepted-v4", summary)
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)


def report_combined(root: Path, identity: dict, protocol: dict,
                    paths: dict[str, Path] | None = None) -> dict:
    """Recompute all receipts and predecessor seals before the combined gate."""
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    require_phase(root, identity, protocol, "blind")
    summaries = {phase: report_phase(root, identity, protocol, phase)
                 for phase in fresh.PHASES}
    expected = {"protocol_sha256": identity["protocol_sha256"],
                "control_agent_sha256": identity["source_sha256"]["control"],
                "candidate_agent_sha256": identity["source_sha256"]["candidate"],
                "model_sha256": identity["model_sha256"],
                "harness_sha256": identity["harness_sha256"],
                "runner_sha256": identity["runner_sha256"],
                "decision_engine_sha256": identity["decision_engine_sha256"]}
    for phase, summary in summaries.items():
        if summary.get("partition") != phase or any(summary.get(key) != value
                                                      for key, value in expected.items()):
            raise ValueError(f"v4 {phase} summary identity mismatch")
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    return {**combined_decision(summaries), "phase_summaries": summaries}


def validate_worker_count(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from 1 to 3")
    return workers


def _run_cold_episode(identity: dict, protocol: dict, paths: dict[str, Path],
                      arm: str, track: int, seed: int) -> dict:
    check_frozen_inputs(identity, protocol, paths)
    payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
               "controller_class": protocol["controller_classes"][arm],
               "track_id": track, "seed": seed}
    command = [sys.executable, str(Path(fresh.__file__).resolve()),
               "--worker", json.dumps(payload, separators=(",", ":"))]
    try:
        try:
            completed = subprocess.run(command, cwd=ROOT, text=True,
                                       capture_output=True, timeout=300, check=True)
            measured = json.loads(completed.stdout)
            if not isinstance(measured, dict):
                raise ValueError("cold worker returned a non-object result")
            return measured
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, ValueError) as error:
            stderr = getattr(error, "stderr", "") or ""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            return {"error": str(error), "stderr": stderr[-4000:]}
    finally:
        check_frozen_inputs(identity, protocol, paths)


def run_partition(root: Path, identity: dict, protocol: dict,
                  paths: dict[str, Path], phase: str, workers: int = 1) -> dict:
    validate_worker_count(workers)
    require_phase(root, identity, protocol, phase)
    check_frozen_inputs(identity, protocol, paths)
    with fresh._run_lock(root):
        jobs = []
        for track, seed, repeat in fresh.expected_cells(protocol, phase):
            for arm in fresh.ARMS:
                prior = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
                if prior is not None:
                    if prior.get("error"):
                        raise RuntimeError(f"recorded operational failure: {phase}/{arm}/{track}/{seed}/{repeat}")
                    continue
                jobs.append((arm, track, seed, repeat))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    arm, track, seed, _ = jobs[next_job]
                    pending[next_job] = pool.submit(_run_cold_episode, identity, protocol,
                                                    paths, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed, repeat) in enumerate(jobs):
                measured = pending.pop(index).result()
                check_frozen_inputs(identity, protocol, paths)
                row = {"partition": phase, "arm": arm, "track_id": track,
                       "seed": seed, "repeat": repeat, **measured}
                fresh.record_cell(root, identity, row)
                if row.get("error"):
                    failure_summary = report_phase(root, identity, protocol, phase)
                    check_frozen_inputs(identity, protocol, paths)
                    fresh._atomic_json(root / f"{phase}-summary.json", failure_summary)
                    check_frozen_inputs(identity, protocol, paths)
                    raise RuntimeError(f"v4 worker failed at {phase}/{arm}/{track}/{seed}/{repeat}")
                print(json.dumps({"cell": [phase, arm, track, seed, repeat],
                                  "finished": row["finished"], "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    check_frozen_inputs(identity, protocol, paths)
    summary = report_phase(root, identity, protocol, phase)
    check_frozen_inputs(identity, protocol, paths)
    target = root / f"{phase}-summary.json"
    if target.exists() and fresh._read_json(target) != summary:
        raise ValueError(f"existing v4 {phase} summary differs from frozen receipts")
    if not target.exists():
        fresh._atomic_json(target, summary)
    check_frozen_inputs(identity, protocol, paths)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", type=Path, default=ROOT / "model.pt")
    parser.add_argument("--control-agent", type=Path, default=ROOT / SOURCE_PATHS["control"])
    parser.add_argument("--candidate-agent", type=Path, default=ROOT / SOURCE_PATHS["candidate"])
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("bind")
    modes.add_parser("restore-snapshots")
    modes.add_parser("preflight")
    run = modes.add_parser("run-phase")
    run.add_argument("phase", choices=fresh.PHASES)
    run.add_argument("--workers", type=int, default=1)
    report = modes.add_parser("report-phase")
    report.add_argument("phase", choices=fresh.PHASES)
    modes.add_parser("freeze-finalist")
    modes.add_parser("seal-confirmation")
    modes.add_parser("final-decision")
    args = parser.parse_args()
    if args.mode != "restore-snapshots":
        require_runtime_versions()
    require_root_model(args.model)
    if args.protocol.resolve() != PROTOCOL_PATH.resolve() or args.output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError("registered v4 protocol and output root are required")
    for arm in fresh.ARMS:
        if getattr(args, f"{arm}_agent").resolve() != (ROOT / SOURCE_PATHS[arm]).resolve():
            raise ValueError(f"registered v4 {arm} source path is required")
    paths = {"control": args.control_agent, "candidate": args.candidate_agent,
             "model": args.model, "protocol": args.protocol}
    if args.mode == "bind":
        protocol = bind_source(fresh._read_json(TEMPLATE_PATH), args.model, args.protocol,
                               args.control_agent, args.candidate_agent)
        print(json.dumps({"binding": "CREATED", "protocol": str(args.protocol),
                          "control_commit": protocol["source_construction"]["control_implementation_commit"],
                          "candidate_base_commit": protocol["source_construction"]["candidate_base_commit"]},
                         indent=2))
        return 0
    if not args.protocol.is_file():
        raise ValueError("unbound v4 protocol: run bind after committing runner and gate")
    verify_committed_protocol(args.protocol)
    protocol = fresh._read_json(args.protocol)
    historical = historical_geometry_seeds_v4(ROOT / "experiments", args.protocol)
    validate_protocol(protocol, historical)
    if args.mode == "restore-snapshots":
        restore_snapshots(protocol, paths["control"], paths["candidate"])
        print(json.dumps({"restore_snapshots": "PASS",
                          "control_agent_sha256": protocol["control_agent_sha256"],
                          "candidate_agent_sha256": protocol["candidate_agent_sha256"]}, indent=2))
        return 0
    validate_source_pair(protocol, paths["control"], paths["candidate"])
    identity = build_identity(args.protocol, protocol, paths)
    check_frozen_inputs(identity, protocol, paths)
    if args.mode == "preflight":
        for arm in fresh.ARMS:
            fresh._load_agent(paths[arm], paths["model"], protocol["controller_classes"][arm])
        check_frozen_inputs(identity, protocol, paths)
        print(json.dumps({"preflight": "PASS", "identity": identity}, indent=2))
        return 0
    fresh.prepare_run(args.output_root, identity)
    if args.mode == "run-phase":
        summary = run_partition(args.output_root, identity, protocol, paths,
                                args.phase, workers=args.workers)
    elif args.mode == "report-phase":
        summary = report_phase(args.output_root, identity, protocol, args.phase)
    elif args.mode == "freeze-finalist":
        freeze_finalist(args.output_root, identity, protocol, paths)
        summary = report_phase(args.output_root, identity, protocol, "screen")
    elif args.mode == "seal-confirmation":
        seal_confirmation(args.output_root, identity, protocol, paths)
        summary = report_phase(args.output_root, identity, protocol, "confirmation")
    else:
        summary = report_combined(args.output_root, identity, protocol, paths)
    check_frozen_inputs(identity, protocol, paths)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
