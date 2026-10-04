"""Prospective, source-bound speed evaluation against the promoted V4 Agent.

Commit this runner, gate and seedless template before binding. Bind once per
profile, then commit the protocol before opening any cold worker. Confirmation
requires a freshly recomputed screen seal. Historical studies are never edited.
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import competition_camera_speed_gate_v5 as gate
from tools import evaluate_bare_generalization as fresh

NAME = "camera-speed-v5"
PHASES = gate.PHASES
TEMPLATE_PATH = ROOT / "experiments" / f"{NAME}.template.json"
CONTROL_COMMIT = "1d45611bef833a52e3042d847c984b744b9513d9"
CONTROL_BLOB_SHA256 = "d77b27e4489564f98df985b138d6c8ba2d74ff9ed275aeb62cd61477f2b46c18"
CONTROL_CLASS = "_ClearRoadRow42DropoutController"
CANDIDATE_CLASS = "_SpeedOptimizedController"
PARTICIPANTS_COMMIT = "dfb7a2de2178825ca5c5ce20bab01ba67052ba31"
REQUIREMENTS_BLOB_SHA256 = "7fe177422d03b66b720a30c14b2dff05da95db95e238d3841afd76cb09a73ea2"
RUNTIME_PROVENANCE = {"python": "3.11", "torch": "2.1.0", "numpy": "1.26.0",
                      "gymnasium": "0.29.1", "opencv_python": "4.8.1.78", "box2d": "2.3.10"}
TRACKS = [1, 2, 3, 4]
SEED_COUNTS = {"screen": 4, "confirmation": 4}
SPOT_INDEX = {"screen": ((1, 0), (4, 3)), "confirmation": ((2, 0), (3, 3))}
HELPERS = ("action_smoothing.py", "action_representation.py")
EVALUATION_POLICY = {
    "selected_before_geometry_binding": True,
    "maximum_candidates": 2,
    "candidate_budget_scope": "retained production revisions; exploratory prototypes are development",
    "maximum_fresh_studies": 2,
    "maximum_cold_episodes_per_study": 72,
    "maximum_workers": 3,
    "worker_timeout_seconds": 300,
    "maximum_wall_minutes_per_study": 30,
    "profiles": {profile: gate.thresholds_for(profile) for profile in gate.PROFILES},
    "fallback_requires_distinct_committed_performance_failures": 2,
    "failure_evidence_scopes": ["consumed-development", "fresh-strict"],
    "failure_evidence_requires_pinned_source_and_recomputed_canonical_pairs": True,
    "development_failure_includes_designated_lap_target": True,
    "fallback_requires_new_source_protocol_and_unused_geometry": True,
    "historical_decisions_and_sealed_holdouts_unchanged": True,
    "no_rescoring_bound_study": True,
    "development_maps": [[1, 516237], [2, 644062], [3, 1007]],
    "development_lap_target_seconds": [10, 13],
    "development_target_is_not_fresh_evidence": True,
    "progress_metric": "1.0 for an official finished outcome; raw progress for DNF",
}


def study_paths(profile: str) -> dict[str, Path]:
    gate.thresholds_for(profile)
    base = ROOT / ".haic-artifacts" / NAME / profile
    return {"protocol": ROOT / "experiments" / f"{NAME}-{profile}.json",
            "root": base / "run", "control": base / "snapshots/control_agent.py",
            "candidate": base / "snapshots/candidate_agent.py", "model": ROOT / "model.pt"}


def committed_bytes(path: str, commit: str = "HEAD") -> bytes:
    try:
        return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(f"{path} must exist in committed Git source") from error


def validate_template(template: dict) -> None:
    expected = {"schema_version": 1, "name": NAME, "status": "UNBOUND",
                "control_implementation_commit": CONTROL_COMMIT,
                "control_blob_sha256": CONTROL_BLOB_SHA256,
                "candidate_base_commit": "REQUIRED_AT_BIND",
                "candidate_base_blob_sha256": "REQUIRED_AT_BIND",
                "requirements_blob_sha256": REQUIREMENTS_BLOB_SHA256,
                "control_class": CONTROL_CLASS, "candidate_class": CANDIDATE_CLASS,
                "track_ids": TRACKS, "seed_counts": SEED_COUNTS,
                "spot_indices": {phase: [list(cell) for cell in SPOT_INDEX[phase]] for phase in PHASES},
                "evaluation_policy": EVALUATION_POLICY}
    if not isinstance(template, dict) or any(template.get(key) != value for key, value in expected.items()):
        raise ValueError("unbound V5 template contract changed")
    if any(key in template for key in ("partitions", "seed_salt_hex", "seeds")):
        raise ValueError("unbound V5 template must contain no geometry")


def derived_seed_partitions(profile: str, salt: str) -> dict[str, list[int]]:
    gate.thresholds_for(profile)
    if not isinstance(salt, str) or not re.fullmatch(r"[0-9a-f]{32}", salt):
        raise ValueError("seed salt must be 128-bit lowercase hex")
    return {phase: [int.from_bytes(hashlib.sha256(
        f"{NAME}:{profile}:{salt}:{phase}:{index}".encode("ascii")).digest()[:4], "big")
        for index in range(SEED_COUNTS[phase])] for phase in PHASES}


def _documented_seeds(value: object) -> set[int]:
    """Read seed metadata only; do not interpret unopened outcomes."""
    protected = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("seed", "geometry_seed") and type(item) is int and 0 <= item <= fresh.MAX_SEED:
                protected.add(item)
            if key in ("seeds", "geometry_seeds") or ("reserved" in key and "seed" in key):
                protected.update(fresh._seed_list(item))
            if key.endswith("_cells") or key == "development_maps":
                if isinstance(item, list):
                    protected.update(cell[1] for cell in item if isinstance(cell, (list, tuple))
                                     and len(cell) >= 2 and type(cell[1]) is int
                                     and 0 <= cell[1] <= fresh.MAX_SEED)
            protected.update(_documented_seeds(item))
    elif isinstance(value, list):
        for item in value:
            protected.update(_documented_seeds(item))
    return protected


def historical_geometry_seeds(experiments_dir: Path, protocol_path: Path) -> set[int]:
    result_path = protocol_path.with_name(protocol_path.stem + "-result.json")
    own = _documented_seeds(fresh._read_json(protocol_path)) if protocol_path.exists() else set()
    protected = set()
    for path in sorted(experiments_dir.glob("*.json")):
        if path.resolve() == protocol_path.resolve():
            continue
        record = fresh._read_json(path)
        documented = _documented_seeds(record)
        if path.resolve() == result_path.resolve():
            if not protocol_path.exists() or not isinstance(record, dict) or record.get("protocol_sha256") != fresh.digest(protocol_path):
                raise ValueError("V5 result protocol SHA256 mismatch")
            documented -= own
        protected.update(documented)
    return protected


def require_fresh_seeds(seeds: dict, historical: set[int]) -> None:
    if not isinstance(seeds, dict) or set(seeds) != set(PHASES) or any(
            not isinstance(seeds[phase], list) or len(seeds[phase]) != SEED_COUNTS[phase]
            or any(type(seed) is not int or not 0 <= seed <= fresh.MAX_SEED for seed in seeds[phase]) for phase in PHASES):
        raise ValueError("invalid V5 geometry partitions")
    flat = [seed for phase in PHASES for seed in seeds[phase]]
    if len(flat) != len(set(flat)) or set(flat) & historical:
        raise ValueError("geometry collision or historical geometry reuse")


def performance_failure_reasons(pairs: list[dict]) -> list[str]:
    """Validate a completed development comparison and recompute strict floors.

    Development comparisons need not use the fresh sixteen-cell phase grid.
    These outcomes justify a prospective fallback only; they cannot promote an
    Agent. Promotion always requires all fresh cold receipts and exact repeats.
    """
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("performance proof needs canonical paired outcomes")
    seen = set()
    totals = {arm: {"finishes": 0, "crashes": 0, "contacts": 0, "damage": 0.0, "progress": 0.0} for arm in fresh.ARMS}
    tracks = {track: {arm: 0 for arm in fresh.ARMS} for track in TRACKS}
    losses, shared = 0, {arm: 0.0 for arm in fresh.ARMS}
    common = 0
    for pair in pairs:
        if not isinstance(pair, dict) or pair.get("status") != "COMPLETE":
            raise ValueError("performance proof has incomplete canonical outcomes")
        track, seed = pair.get("track_id"), pair.get("seed")
        if type(track) is not int or track not in TRACKS or type(seed) is not int or not 0 <= seed <= fresh.MAX_SEED or (track, seed) in seen:
            raise ValueError("performance proof has invalid or duplicate coordinates")
        seen.add((track, seed))
        for arm in fresh.ARMS:
            outcome = pair.get(arm)
            if not isinstance(outcome, dict) or any(field not in outcome for field in gate._OUTCOME_FIELDS):
                raise ValueError("performance proof lacks canonical outcome metrics")
            synthetic = {"partition": "screen", "error": None, "initialization_ms": 0.0, "reset_ms": 0.0,
                         "action_latency_max_ms": 0.0, "peak_worker_rss_mib": 0.0, "steps": 0,
                         "offtrack_samples": 0, "partial_offtrack_samples": 0, "action_trace_sha256": "0" * 64,
                         **{field: outcome[field] for field in gate._OUTCOME_FIELDS}}
            if gate._metric_errors(synthetic, "performance proof", "screen"):
                raise ValueError("performance proof has invalid official outcome metrics")
            totals[arm]["finishes"] += int(outcome["finished"])
            tracks[track][arm] += int(outcome["finished"])
            totals[arm]["crashes"] += int(outcome["retire_reason"] == "crash")
            totals[arm]["contacts"] += outcome["collision_count"]
            totals[arm]["damage"] += outcome["damage"]
            totals[arm]["progress"] += 1.0 if outcome["finished"] else outcome["progress"]
        control, candidate = pair["control"], pair["candidate"]
        losses += int(control["finished"] and not candidate["finished"])
        if control["finished"] and candidate["finished"]:
            common += 1
            for arm in fresh.ARMS: shared[arm] += pair[arm]["lap_time_ms"]
    reasons = []
    if losses: reasons.append("strict lost control finish budget exceeded")
    if totals["candidate"]["finishes"] < totals["control"]["finishes"]: reasons.append("completion regression")
    for track, counts in tracks.items():
        if counts["candidate"] < counts["control"]: reasons.append(f"track {track} completion regression")
    for metric in ("crashes", "contacts", "damage"):
        if totals["candidate"][metric] > totals["control"][metric]: reasons.append(f"aggregate {metric} increased")
    if totals["candidate"]["progress"] < totals["control"]["progress"]: reasons.append("completion-aware progress regression")
    if not common: reasons.append("no shared finishes to verify speed")
    elif shared["candidate"] > .90 * shared["control"]: reasons.append("strict speed improvement below ten percent")
    designated = {tuple(cell) for cell in EVALUATION_POLICY["development_maps"]}
    if designated <= seen and any(pair["candidate"]["finished"] and pair["candidate"]["lap_time_ms"] >
                                 1000 * EVALUATION_POLICY["development_lap_target_seconds"][1]
                                 for pair in pairs if (pair["track_id"], pair["seed"]) in designated):
        reasons.append("designated development lap target above thirteen seconds")
    return reasons


def _committed_evidence_file(name: str, suffix: str) -> tuple[str, bytes]:
    if not isinstance(name, str):
        raise ValueError("performance proof needs committed source/report references")
    requested = ROOT / name
    if requested.is_symlink():
        raise ValueError("symlink performance source/report is forbidden")
    path = requested.resolve()
    if (not path.is_relative_to(ROOT / "experiments") or not path.is_file()
            or path.is_symlink() or path.suffix != suffix):
        raise ValueError("performance source/report must be a committed experiments file")
    relative = path.relative_to(ROOT).as_posix()
    data = path.read_bytes()
    if committed_bytes(relative) != data:
        raise ValueError("performance source/report must match exact committed bytes")
    return relative, data


def failure_evidence(profile: str, records: list[str]) -> list[dict]:
    gate.thresholds_for(profile)
    if profile == "strict":
        if records:
            raise ValueError("strict profile must not provide fallback performance failures")
        return []
    if len(records) != 2 or len(set(records)) != 2:
        raise ValueError("fallback needs two distinct committed performance failure records")
    proofs, candidate_hashes = [], set()
    for name in records:
        requested = ROOT / name
        if requested.is_symlink():
            raise ValueError("symlink performance failure proof is forbidden")
        path = requested.resolve()
        if not path.is_relative_to(ROOT / "experiments") or not path.is_file() or path.is_symlink():
            raise ValueError("performance failure proof must be a committed experiments JSON file")
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        if committed_bytes(relative) != data:
            raise ValueError("performance failure proof must match committed Git bytes")
        record = json.loads(data)
        if not isinstance(record, dict):
            raise ValueError("performance failure proof must be a JSON object")
        candidate_hash = record.get("candidate_agent_sha256")
        if (record.get("decision") != "REJECT" or record.get("failure_kind") != "performance"
                or record.get("study_family") != NAME or record.get("evaluation_profile") != "strict"
                or record.get("completion_status") != "COMPLETE"
                or record.get("evaluation_scope") not in EVALUATION_POLICY["failure_evidence_scopes"]
                or not isinstance(record.get("reasons"), list) or not record["reasons"]
                or not isinstance(candidate_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash)):
            raise ValueError("fallback requires complete strict V5 performance rejection evidence")
        source_name, source = _committed_evidence_file(record.get("candidate_source_path"), ".py")
        report_name, report_bytes = _committed_evidence_file(record.get("comparison_report_path"), ".json")
        report_sha = hashlib.sha256(report_bytes).hexdigest()
        report = json.loads(report_bytes)
        if (hashlib.sha256(source).hexdigest() != candidate_hash
                or record.get("control_agent_sha256") != CONTROL_BLOB_SHA256
                or record.get("comparison_report_sha256") != report_sha
                or not isinstance(report, dict) or report.get("control_agent_sha256") != CONTROL_BLOB_SHA256
                or report.get("candidate_agent_sha256") != candidate_hash):
            raise ValueError("performance proof source/report identity differs from exact committed evidence")
        pairs = report.get("paired_cells")
        if type(report.get("canonical_cells")) is not int or not isinstance(pairs, list) or report["canonical_cells"] != len(pairs):
            raise ValueError("performance proof canonical inventory count mismatch")
        if record["evaluation_scope"] == "consumed-development":
            if not {tuple(cell) for cell in EVALUATION_POLICY["development_maps"]} <= {
                    (pair.get("track_id"), pair.get("seed")) for pair in pairs if isinstance(pair, dict)}:
                raise ValueError("development proof must include all three designated consumed maps")
        recomputed = performance_failure_reasons(pairs)
        if not recomputed:
            raise ValueError("performance proof has no recomputable strict performance failure")
        if record["evaluation_scope"] == "fresh-strict":
            strict_path = study_paths("strict")["protocol"]
            if not strict_path.is_file():
                raise ValueError("fresh performance proof needs its committed strict protocol")
            verify_committed_protocol(strict_path)
            strict = fresh._read_json(strict_path)
            if (record.get("protocol_sha256") != fresh.digest(strict_path) or strict.get("candidate_agent_sha256") != candidate_hash):
                raise ValueError("fresh performance proof protocol/source mismatch")
        candidate_hashes.add(candidate_hash)
        proofs.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest(),
                       "candidate_agent_sha256": candidate_hash, "decision": "REJECT",
                       "evaluation_scope": record["evaluation_scope"], "candidate_source_path": source_name,
                       "comparison_report_path": report_name, "comparison_report_sha256": report_sha,
                       "recomputed_performance_failures": recomputed})
    if len(candidate_hashes) != 2:
        raise ValueError("two performance failures must concern distinct candidate sources")
    return proofs


def selected_source(source: bytes) -> bytes:
    selector = f"{CONTROL_CLASS}()".encode("ascii")
    if source.count(selector) != 1 or re.search(rb"(?m)^class " + CANDIDATE_CLASS.encode("ascii") + rb"\b", source) is None:
        raise ValueError("candidate must declare its route and contain exactly one control selector")
    return source.replace(selector, f"{CANDIDATE_CLASS}()".encode("ascii"), 1)


def source_pair(candidate_commit: str, candidate_blob: str) -> tuple[bytes, bytes, bytes]:
    if not isinstance(candidate_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", candidate_commit) or not isinstance(candidate_blob, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_blob):
        raise ValueError("candidate commit and agent.py SHA256 pins are required")
    control = committed_bytes("agent.py", CONTROL_COMMIT)
    base = committed_bytes("agent.py", candidate_commit)
    if hashlib.sha256(control).hexdigest() != CONTROL_BLOB_SHA256 or control.count(f"{CONTROL_CLASS}()".encode()) != 1:
        raise ValueError("promoted V4 control source changed")
    if hashlib.sha256(base).hexdigest() != candidate_blob or base == control:
        raise ValueError("candidate implementation blob differs from its pin or equals control")
    return control, base, selected_source(base)


def _environment_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "local_simulator").rglob("*.py"))
    paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    if any(path.is_symlink() for path in paths):
        raise ValueError("symlink environment source is forbidden")
    return {path.relative_to(ROOT).as_posix(): fresh.digest(path) for path in paths}


def _requirements_blob_sha256() -> str:
    measured = hashlib.sha256(committed_bytes("requirements.txt")).hexdigest()
    if measured != REQUIREMENTS_BLOB_SHA256:
        raise ValueError("committed official requirements provenance changed")
    return measured


def require_runtime_versions() -> None:
    from importlib import metadata
    import Box2D
    import numpy
    import torch
    actual = {"python": f"{sys.version_info.major}.{sys.version_info.minor}",
              "torch": metadata.version("torch"), "numpy": metadata.version("numpy"),
              "gymnasium": metadata.version("gymnasium"), "opencv_python": metadata.version("opencv-python"),
              "box2d": str(Box2D.__version__)}
    if (actual != RUNTIME_PROVENANCE or torch.__version__.split("+", 1)[0] != actual["torch"]
            or numpy.__version__ != actual["numpy"]):
        raise ValueError(f"official runtime provenance mismatch: {actual}")


def _require_committed_runtime() -> None:
    for path in (Path(__file__), Path(gate.__file__), TEMPLATE_PATH):
        if committed_bytes(path.relative_to(ROOT).as_posix()) != path.read_bytes():
            raise ValueError("V5 runner, gate and template must be committed with exact bytes before binding")


def _source_paths(profile: str) -> dict[str, str]:
    paths = study_paths(profile)
    return {arm: paths[arm].relative_to(ROOT).as_posix() for arm in fresh.ARMS}


def validate_protocol(protocol: dict, historical: set[int]) -> None:
    if (not isinstance(protocol, dict) or protocol.get("status") != "PREREGISTERED"
            or type(protocol.get("schema_version")) is not int or protocol["schema_version"] != 1):
        raise ValueError("unbound V5 protocol: one-time binding and source pins required")
    profile = protocol.get("evaluation_profile")
    paths = study_paths(profile)
    if protocol.get("name") != f"{NAME}-{profile}":
        raise ValueError("V5 study name/profile mismatch")
    construction = protocol.get("source_construction", {})
    if not isinstance(construction, dict):
        raise ValueError("V5 source construction must be an object")
    control, base, candidate = source_pair(construction.get("candidate_base_commit"), construction.get("candidate_base_blob_sha256"))
    expected_construction = {"control_implementation_commit": CONTROL_COMMIT, "control_blob_sha256": CONTROL_BLOB_SHA256,
                             "candidate_base_commit": construction["candidate_base_commit"],
                             "candidate_base_blob_sha256": hashlib.sha256(base).hexdigest(),
                             "selector_from": f"{CONTROL_CLASS}()", "selector_to": f"{CANDIDATE_CLASS}()"}
    expected = {"source_construction": expected_construction, "source_snapshots": _source_paths(profile),
                "control_agent_sha256": hashlib.sha256(control).hexdigest(), "candidate_agent_sha256": hashlib.sha256(candidate).hexdigest(),
                "controller_classes": {"control": CONTROL_CLASS, "candidate": CANDIDATE_CLASS},
                "max_steps": 2000, "frame_skip": 4, "training": False,
                "official_participants_commit": PARTICIPANTS_COMMIT, "runtime_provenance": RUNTIME_PROVENANCE,
                "requirements_blob_sha256": _requirements_blob_sha256(),
                "decision_thresholds": gate.thresholds_for(profile), "evaluation_policy": EVALUATION_POLICY,
                "template_sha256": fresh.digest(TEMPLATE_PATH), "runner_sha256": fresh.digest(Path(__file__)),
                "decision_engine_sha256": fresh.digest(Path(gate.__file__)),
                "evaluation_harness_sha256": fresh.digest(Path(fresh.__file__)),
                "environment_sha256": _environment_hashes(),
                "runtime_helper_sha256": {name: fresh.digest(ROOT / name) for name in HELPERS}}
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("bound V5 source, runtime, or prospective policy contract changed")
    if (protocol.get("training") is not False or type(protocol.get("max_steps")) is not int
            or type(protocol.get("frame_skip")) is not int):
        raise ValueError("V5 exact official runtime field types changed")
    fresh._sha256_string(protocol.get("model_sha256"), "model_sha256")
    proofs = protocol.get("prior_performance_failure_evidence")
    if (not isinstance(proofs, list) or any(not isinstance(proof, dict) or not isinstance(proof.get("path"), str) for proof in proofs)
            or proofs != failure_evidence(profile, [proof["path"] for proof in proofs])):
        raise ValueError("V5 performance failure proof changed")
    if profile == "fallback" and protocol["candidate_agent_sha256"] in {proof["candidate_agent_sha256"] for proof in proofs}:
        raise ValueError("fallback needs a new candidate source")
    seeds = derived_seed_partitions(profile, protocol.get("seed_salt_hex"))
    require_fresh_seeds(seeds, historical)
    expected_partitions = {phase: {"track_ids": TRACKS, "seeds": seeds[phase],
                          "spot_check_cells": [[track, seeds[phase][index]] for track, index in SPOT_INDEX[phase]]} for phase in PHASES}
    if protocol.get("partitions") != expected_partitions:
        raise ValueError("V5 phase, geometry, or exact repeat grid changed")
    for partition in protocol["partitions"].values():
        if (any(type(value) is not int for key in ("track_ids", "seeds") for value in partition[key])
                or any(type(value) is not int for cell in partition["spot_check_cells"] for value in cell)):
            raise ValueError("V5 geometry coordinates must be exact integers")
    if paths["model"].resolve() != (ROOT / "model.pt").resolve():
        raise ValueError("V5 requires the root model.pt")


def _exclusive_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def bind_source(profile: str, candidate_commit: str, candidate_blob: str, failure_records: list[str]) -> dict:
    # Missing source pins reject before runtime loading or generating a random salt.
    control, base, candidate = source_pair(candidate_commit, candidate_blob)
    paths = study_paths(profile)
    validate_template(fresh._read_json(TEMPLATE_PATH))
    proofs = failure_evidence(profile, failure_records)
    if profile == "fallback" and hashlib.sha256(candidate).hexdigest() in {proof["candidate_agent_sha256"] for proof in proofs}:
        raise ValueError("fallback needs a new candidate source")
    if any(paths[name].exists() for name in ("protocol", "control", "candidate", "root")):
        raise ValueError("V5 one-time binding already exists; resampling or overwrite is forbidden")
    _require_committed_runtime()
    require_runtime_versions()
    historical = historical_geometry_seeds(ROOT / "experiments", paths["protocol"])
    salt = secrets.token_hex(16)
    seeds = derived_seed_partitions(profile, salt)
    require_fresh_seeds(seeds, historical)
    protocol = {"schema_version": 1, "name": f"{NAME}-{profile}", "status": "PREREGISTERED", "evaluation_profile": profile,
                "hypothesis": "The committed speed route maintains completion and safety without overall slowdown; strict promotion needs at least 10% faster shared finish time.",
                "evidence_scope": "Prospective paired screen and independent confirmation; consumed historical grids are development only.",
                "control_agent_sha256": hashlib.sha256(control).hexdigest(), "candidate_agent_sha256": hashlib.sha256(candidate).hexdigest(),
                "model_sha256": fresh.digest(paths["model"]), "source_snapshots": _source_paths(profile),
                "controller_classes": {"control": CONTROL_CLASS, "candidate": CANDIDATE_CLASS},
                "source_construction": {"control_implementation_commit": CONTROL_COMMIT, "control_blob_sha256": CONTROL_BLOB_SHA256,
                                        "candidate_base_commit": candidate_commit, "candidate_base_blob_sha256": hashlib.sha256(base).hexdigest(),
                                        "selector_from": f"{CONTROL_CLASS}()", "selector_to": f"{CANDIDATE_CLASS}()"},
                "seed_salt_hex": salt, "seed_derivation": "First four SHA256 bytes of name:profile:salt:phase:index, unsigned big-endian uint32.",
                "partitions": {phase: {"track_ids": TRACKS, "seeds": seeds[phase],
                               "spot_check_cells": [[track, seeds[phase][index]] for track, index in SPOT_INDEX[phase]]} for phase in PHASES},
                "template_sha256": fresh.digest(TEMPLATE_PATH), "runner_sha256": fresh.digest(Path(__file__)),
                "decision_engine_sha256": fresh.digest(Path(gate.__file__)), "evaluation_harness_sha256": fresh.digest(Path(fresh.__file__)),
                "environment_sha256": _environment_hashes(), "runtime_helper_sha256": {name: fresh.digest(ROOT / name) for name in HELPERS},
                "requirements_blob_sha256": _requirements_blob_sha256(), "official_participants_commit": PARTICIPANTS_COMMIT,
                "runtime_provenance": RUNTIME_PROVENANCE, "max_steps": 2000, "frame_skip": 4, "training": False,
                "decision_thresholds": gate.thresholds_for(profile), "evaluation_policy": EVALUATION_POLICY,
                "prior_performance_failure_evidence": proofs,
                "evaluation_budget": {"canonical_pairs": 32, "repeat_pairs": 4, "cold_episodes": 72},
                "promotion_gate": "Freshly recomputed RETAIN screen seal opens confirmation; both phases and combined prospective gate must RETAIN."}
    validate_protocol(protocol, historical)
    created = []
    try:
        for path, content in ((paths["control"], control), (paths["candidate"], candidate),
                              (paths["protocol"], fresh._canonical(protocol) + b"\n")):
            _exclusive_bytes(path, content)
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return protocol


def validate_source_pair(protocol: dict, paths: dict[str, Path]) -> None:
    construction = protocol["source_construction"]
    control, _, candidate = source_pair(construction["candidate_base_commit"], construction["candidate_base_blob_sha256"])
    if paths["control"].read_bytes() != control or paths["candidate"].read_bytes() != candidate:
        raise ValueError("V5 source snapshots differ from pinned Git blobs and selector swap")


def restore_snapshots(protocol: dict, paths: dict[str, Path]) -> None:
    construction = protocol["source_construction"]
    control, _, candidate = source_pair(construction["candidate_base_commit"], construction["candidate_base_blob_sha256"])
    for arm, expected in (("control", control), ("candidate", candidate)):
        if paths[arm].exists() and paths[arm].read_bytes() != expected:
            raise ValueError("existing snapshot differs; refusing overwrite")
    created = []
    try:
        for arm, expected in (("control", control), ("candidate", candidate)):
            if not paths[arm].exists():
                _exclusive_bytes(paths[arm], expected)
                created.append(paths[arm])
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    validate_source_pair(protocol, paths)


def verify_committed_protocol(path: Path) -> None:
    if path.is_symlink() or committed_bytes(path.relative_to(ROOT).as_posix()) != path.read_bytes():
        raise ValueError("V5 protocol must match committed exact Git bytes before evaluation")


def load_study(profile: str, restore: bool = False) -> tuple[dict, dict, dict[str, Path]]:
    paths = study_paths(profile)
    if not paths["protocol"].is_file():
        raise ValueError("unbound V5 protocol: commit tooling, pin candidate, then bind")
    verify_committed_protocol(paths["protocol"])
    validate_template(fresh._read_json(TEMPLATE_PATH))
    protocol = fresh._read_json(paths["protocol"])
    validate_protocol(protocol, historical_geometry_seeds(ROOT / "experiments", paths["protocol"]))
    if restore:
        restore_snapshots(protocol, paths)
    validate_source_pair(protocol, paths)
    identity = {**fresh.build_identity(paths["protocol"], protocol, paths["control"], paths["candidate"], paths["model"]),
                "runner_sha256": fresh.digest(Path(__file__)), "decision_engine_sha256": fresh.digest(Path(gate.__file__)),
                "template_sha256": fresh.digest(TEMPLATE_PATH), "protocol_name": protocol["name"], "evaluation_profile": profile,
                "candidate_route_class": CANDIDATE_CLASS}
    check_frozen_inputs(identity, protocol, paths)
    return protocol, identity, paths


def check_frozen_inputs(identity: dict, protocol: dict, paths: dict[str, Path]) -> None:
    if any(paths[key].is_symlink() for key in ("control", "candidate", "model", "protocol")):
        raise ValueError("symlink V5 frozen input is forbidden")
    fresh.check_frozen_inputs(identity, paths)
    if (identity["runner_sha256"] != protocol["runner_sha256"] or identity["runner_sha256"] != fresh.digest(Path(__file__))
            or identity["decision_engine_sha256"] != protocol["decision_engine_sha256"] or identity["decision_engine_sha256"] != fresh.digest(Path(gate.__file__))
            or identity["template_sha256"] != protocol["template_sha256"] or identity["template_sha256"] != fresh.digest(TEMPLATE_PATH)
            or identity["harness_sha256"] != protocol["evaluation_harness_sha256"]
            or identity["helper_sha256"] != protocol["runtime_helper_sha256"]
            or identity["environment_sha256"] != protocol["environment_sha256"] or _environment_hashes() != protocol["environment_sha256"]
            or identity["protocol_name"] != protocol["name"] or identity["evaluation_profile"] != protocol["evaluation_profile"]
            or identity["candidate_route_class"] != CANDIDATE_CLASS):
        raise ValueError("V5 bound runtime, source inventory, profile, or route changed")
    if _requirements_blob_sha256() != protocol["requirements_blob_sha256"]:
        raise ValueError("V5 requirements changed")
    validate_source_pair(protocol, paths)
    proofs = protocol["prior_performance_failure_evidence"]
    if proofs != failure_evidence(protocol["evaluation_profile"], [proof["path"] for proof in proofs]):
        raise ValueError("V5 prospective failure evidence changed")

def _unexpected_receipts(root: Path, protocol: dict, phase: str) -> list[str]:
    expected = {fresh.cell_path(root, phase, arm, track, seed, repeat)
                for track, seed, repeat in fresh.expected_cells(protocol, phase)
                for arm in fresh.ARMS}
    phase_root = root / "cells" / phase
    for directory in (root, root / "cells", phase_root):
        if directory.is_symlink():
            return [f"symlink V5 receipt directory: {directory}"]
    if not phase_root.exists():
        return []
    problems = []
    for path in sorted(phase_root.rglob("*")):
        if path.is_symlink():
            problems.append(f"symlink V5 receipt path: {path}")
        elif path.is_file() and path not in expected:
            problems.append(f"unexpected V5 receipt: {path}")
    return problems


def report_phase(root: Path, identity: dict, protocol: dict, phase: str) -> dict:
    if phase not in PHASES:
        raise ValueError("unknown V5 phase")
    require_phase(root, identity, protocol, phase)
    extras = _unexpected_receipts(root, protocol, phase)
    if any("symlink" in problem for problem in extras):
        raise ValueError("symlink V5 receipt path or directory")
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
                raise ValueError(f"V5 receipt changed while reporting: {path}")
            if row is not None:
                rows.append(row)
                if after is None:
                    raise ValueError(f"V5 receipt vanished while reporting: {path}")
                hashed_paths.append((path.relative_to(root).as_posix(), after))
    summary = gate.compare_pairs(rows, cells, phase, protocol["evaluation_profile"])
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
                    "decision_engine_sha256": identity["decision_engine_sha256"],
                    "evaluation_profile": identity["evaluation_profile"]})
    return summary


def _require_seal(root: Path, identity: dict, protocol: dict,
                  name: str, predecessor: str) -> None:
    seal = fresh._read_seal(root, identity, name)
    if seal is None:
        raise ValueError(f"V5 {name} seal is missing")
    summary = report_phase(root, identity, protocol, predecessor)
    if hashlib.sha256(fresh._canonical(summary)).hexdigest() != seal["summary_sha256"]:
        raise ValueError(f"V5 {name} seal no longer matches {predecessor} receipts")


def require_phase(root: Path, identity: dict, protocol: dict, phase: str) -> None:
    if phase == "screen":
        return
    if phase == "confirmation":
        _require_seal(root, identity, protocol, "speed-finalist-v5", "screen")
        return
    raise ValueError("unknown V5 phase")


def freeze_finalist(root: Path, identity: dict, protocol: dict,
                    paths: dict[str, Path] | None = None) -> None:
    summary = report_phase(root, identity, protocol, "screen")
    if summary["decision"] != "RETAIN":
        raise ValueError("V5 screen must RETAIN before finalist freeze")
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    fresh._write_seal(root, identity, "speed-finalist-v5", summary)
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)


def report_combined(root: Path, identity: dict, protocol: dict,
                    paths: dict[str, Path] | None = None) -> dict:
    """Recompute all receipts and predecessor seals before the combined gate."""
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    require_phase(root, identity, protocol, "confirmation")
    summaries = {phase: report_phase(root, identity, protocol, phase)
                 for phase in PHASES}
    expected = {"protocol_sha256": identity["protocol_sha256"],
                "control_agent_sha256": identity["source_sha256"]["control"],
                "candidate_agent_sha256": identity["source_sha256"]["candidate"],
                "model_sha256": identity["model_sha256"],
                "harness_sha256": identity["harness_sha256"],
                "runner_sha256": identity["runner_sha256"],
                "decision_engine_sha256": identity["decision_engine_sha256"],
                "evaluation_profile": identity["evaluation_profile"]}
    for phase, summary in summaries.items():
        if summary.get("partition") != phase or any(summary.get(key) != value
                                                      for key, value in expected.items()):
            raise ValueError(f"V5 {phase} summary identity mismatch")
    if paths is not None:
        check_frozen_inputs(identity, protocol, paths)
    return {**gate.combined_decision(summaries, protocol["evaluation_profile"]),
            **expected, "study_family": NAME, "phase_summaries": summaries}


def validate_worker_count(workers: int) -> int:
    if type(workers) is not int or not 1 <= workers <= 3:
        raise ValueError("workers must be an integer from 1 to 3")
    return workers


def cold_episode(identity: dict, protocol: dict, paths: dict[str, Path],
                 arm: str, track: int, seed: int) -> dict:
    check_frozen_inputs(identity, protocol, paths)
    payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
               "controller_class": protocol["controller_classes"][arm],
               "track_id": track, "seed": seed}
    command = [sys.executable, str(Path(fresh.__file__).resolve()),
               "--worker", json.dumps(payload, separators=(",", ":"))]
    try:
        try:
            timeout = min(EVALUATION_POLICY["worker_timeout_seconds"], _budget_clock(paths["root"], identity))
            completed = subprocess.run(command, cwd=ROOT, text=True,
                                       capture_output=True, timeout=timeout, check=True)
            measured = json.loads(completed.stdout)
            if not isinstance(measured, dict):
                raise ValueError("cold worker returned a non-object result")
            return measured
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, ValueError, RuntimeError) as error:
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
        _budget_clock(root, identity)
        jobs = []
        for track, seed, repeat in fresh.expected_cells(protocol, phase):
            for arm in fresh.ARMS:
                prior = fresh.load_cell(root, identity, phase, arm, track, seed, repeat)
                if prior is not None:
                    if prior.get("error") is not None:
                        raise RuntimeError(f"recorded operational failure: {phase}/{arm}/{track}/{seed}/{repeat}")
                    continue
                jobs.append((arm, track, seed, repeat))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            next_job = 0

            def fill_window() -> None:
                nonlocal next_job
                while len(pending) < workers and next_job < len(jobs):
                    _budget_clock(root, identity)
                    arm, track, seed, _ = jobs[next_job]
                    pending[next_job] = pool.submit(cold_episode, identity, protocol,
                                                    paths, arm, track, seed)
                    next_job += 1

            fill_window()
            for index, (arm, track, seed, repeat) in enumerate(jobs):
                measured = pending.pop(index).result()
                check_frozen_inputs(identity, protocol, paths)
                row = {**measured, "partition": phase, "arm": arm, "track_id": track,
                       "seed": seed, "repeat": repeat}
                metric_errors = gate._metric_errors(row, f"{track}/{seed}/{repeat}/{arm}", phase)
                if metric_errors and row.get("error") is None:
                    row = {"partition": phase, "arm": arm, "track_id": track, "seed": seed,
                           "repeat": repeat, "error": "invalid cold worker receipt: " + "; ".join(metric_errors)}
                fresh.record_cell(root, identity, row)
                if row.get("error") is not None:
                    failure_summary = report_phase(root, identity, protocol, phase)
                    check_frozen_inputs(identity, protocol, paths)
                    fresh._atomic_json(root / f"{phase}-summary.json", failure_summary)
                    check_frozen_inputs(identity, protocol, paths)
                    raise RuntimeError(f"V5 worker failed at {phase}/{arm}/{track}/{seed}/{repeat}")
                print(json.dumps({"cell": [phase, arm, track, seed, repeat],
                                  "finished": row["finished"], "progress": row["progress"],
                                  "contacts": row["collision_count"]}), flush=True)
                fill_window()
    check_frozen_inputs(identity, protocol, paths)
    summary = report_phase(root, identity, protocol, phase)
    check_frozen_inputs(identity, protocol, paths)
    target = root / f"{phase}-summary.json"
    if target.exists() and fresh._read_json(target) != summary:
        raise ValueError(f"existing V5 {phase} summary differs from frozen receipts")
    if not target.exists():
        fresh._atomic_json(target, summary)
    check_frozen_inputs(identity, protocol, paths)
    return summary


def _budget_clock(root: Path, identity: dict) -> float:
    """Persist one wall deadline; resuming cannot reset the prospective budget."""
    path = root / "budget.json"
    if path.is_symlink():
        raise ValueError("symlink evaluation wall budget")
    if not path.exists():
        fresh._atomic_json(path, {"identity": identity, "started_unix_seconds": time.time(),
                                 "maximum_wall_minutes": EVALUATION_POLICY["maximum_wall_minutes_per_study"]})
    record = fresh._read_json(path)
    if (not isinstance(record, dict) or record.get("identity") != identity
            or record.get("maximum_wall_minutes") != EVALUATION_POLICY["maximum_wall_minutes_per_study"]
            or type(record.get("started_unix_seconds")) not in (int, float)
            or not 0 < record["started_unix_seconds"] <= time.time()):
        raise ValueError("invalid evaluation wall budget receipt")
    remaining = 60 * record["maximum_wall_minutes"] - (time.time() - record["started_unix_seconds"])
    if remaining <= 0:
        raise RuntimeError("prospective V5 wall budget exhausted; no further episodes are permitted")
    return remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=gate.PROFILES, default="strict")
    modes = parser.add_subparsers(dest="mode", required=True)
    bind = modes.add_parser("bind")
    bind.add_argument("--candidate-commit", required=True)
    bind.add_argument("--candidate-blob-sha256", required=True)
    bind.add_argument("--failure-record", action="append", default=[])
    modes.add_parser("restore-snapshots")
    modes.add_parser("preflight")
    run = modes.add_parser("run-phase")
    run.add_argument("phase", choices=PHASES)
    run.add_argument("--workers", type=int, default=3)
    report = modes.add_parser("report-phase")
    report.add_argument("phase", choices=PHASES)
    modes.add_parser("freeze-finalist")
    modes.add_parser("final-decision")
    args = parser.parse_args()
    if args.mode == "bind":
        protocol = bind_source(args.profile, args.candidate_commit, args.candidate_blob_sha256, args.failure_record)
        print(json.dumps({"binding": "CREATED", "protocol": str(study_paths(args.profile)["protocol"]),
                          "candidate_base_commit": protocol["source_construction"]["candidate_base_commit"]}, indent=2))
        return 0
    if args.mode != "restore-snapshots":
        require_runtime_versions()
    protocol, identity, paths = load_study(args.profile, restore=args.mode == "restore-snapshots")
    if args.mode == "restore-snapshots":
        print(json.dumps({"restore_snapshots": "PASS", "identity": identity}, indent=2))
        return 0
    if args.mode == "preflight":
        for arm in fresh.ARMS:
            fresh._load_agent(paths[arm], paths["model"], protocol["controller_classes"][arm])
        check_frozen_inputs(identity, protocol, paths)
        print(json.dumps({"preflight": "PASS", "identity": identity}, indent=2))
        return 0
    fresh.prepare_run(paths["root"], identity)
    if args.mode == "run-phase":
        summary = run_partition(paths["root"], identity, protocol, paths, args.phase, args.workers)
    elif args.mode == "report-phase":
        summary = report_phase(paths["root"], identity, protocol, args.phase)
    elif args.mode == "freeze-finalist":
        freeze_finalist(paths["root"], identity, protocol, paths)
        summary = report_phase(paths["root"], identity, protocol, "screen")
    else:
        summary = report_combined(paths["root"], identity, protocol, paths)
    check_frozen_inputs(identity, protocol, paths)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
