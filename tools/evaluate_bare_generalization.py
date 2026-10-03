"""Preregistered, paired evaluation of actual bare Agent routes on fresh official maps.

The protocol owns the cell grid and source hashes. Run screen, freeze its finalist,
run confirmation, seal its decision, then run blind. Each official episode is a
cold worker process. Existing records and seals are immutable for safe resume.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PHASES = ("screen", "confirmation", "blind")
ARMS = ("control", "candidate")
MAX_SEED = 2**32 - 1


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _seed_list(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {seed for seed in value if type(seed) is int and 0 <= seed <= MAX_SEED}


def historical_geometry_seeds(experiments_dir: Path, exclude: Path | None = None) -> set[int]:
    """Conservatively protect all documented geometry seeds across every track ID."""
    protected: set[int] = set()
    excluded = exclude.resolve() if exclude is not None else None

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                # A nested run/map seed is geometry unless proved otherwise.
                # Excluding an occasional RNG seed is conservative.
                if key == "seed" and type(child) is int and 0 <= child <= MAX_SEED:
                    protected.add(child)
                if key == "seeds" or ("reserved" in key and "seed" in key):
                    protected.update(_seed_list(child))
                if key.endswith("_cells") and isinstance(child, list):
                    for cell in child:
                        if isinstance(cell, (list, tuple)) and len(cell) >= 2 and type(cell[1]) is int:
                            protected.add(cell[1])
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for path in sorted(experiments_dir.glob("*.json")):
        if excluded is not None and path.resolve() == excluded:
            continue
        record = _read_json(path)
        inspect(record)
    return protected


def _exact_int(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an exact integer in [{low}, {high}]")
    return value


def _sha256_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def validate_protocol(protocol: dict, historical_seeds: set[int]) -> None:
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version must be 1")
    if protocol.get("status") != "PREREGISTERED":
        raise ValueError("protocol status must be PREREGISTERED")
    if not isinstance(protocol.get("name"), str) or not protocol["name"].strip():
        raise ValueError("protocol needs a nonempty name")
    for field in ("control_agent_sha256", "candidate_agent_sha256", "model_sha256"):
        _sha256_string(protocol.get(field), field)
    helpers = protocol.get("runtime_helper_sha256")
    if not isinstance(helpers, dict) or set(helpers) != {"action_smoothing.py", "action_representation.py"}:
        raise ValueError("runtime_helper_sha256 must pin both bare Agent helpers")
    for name, value in helpers.items():
        _sha256_string(value, name)
    classes = protocol.get("controller_classes")
    if not isinstance(classes, dict) or any(not isinstance(classes.get(arm), str) or not classes[arm] for arm in ARMS):
        raise ValueError("controller_classes must name both actual Agent routes")
    if protocol.get("max_steps") != 2000 or protocol.get("frame_skip") != 4:
        raise ValueError("official MapSpec requires max_steps 2000 and frame_skip 4")
    partitions = protocol.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(PHASES):
        raise ValueError("protocol needs exact screen, confirmation, and blind partitions")
    all_seeds: set[int] = set()
    for phase in PHASES:
        partition = partitions[phase]
        if not isinstance(partition, dict):
            raise ValueError(f"{phase} partition must be an object")
        tracks = partition.get("track_ids")
        seeds = partition.get("seeds")
        if not isinstance(tracks, list) or not tracks or not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{phase} needs nonempty track_ids and seeds")
        tracks = [_exact_int(track, f"{phase} track_id", 1, 2**63 - 1) for track in tracks]
        seeds = [_exact_int(seed, f"{phase} seed", 0, MAX_SEED) for seed in seeds]
        if len(set(tracks)) != len(tracks) or len(set(seeds)) != len(seeds):
            raise ValueError(f"{phase} has duplicate track IDs or seeds")
        if set(seeds) & all_seeds:
            raise ValueError("geometry seed overlap between partitions")
        all_seeds.update(seeds)
        spots = partition.get("spot_check_cells", [])
        if not isinstance(spots, list):
            raise ValueError(f"{phase} spot_check_cells must be an array")
        normalized_spots = []
        for cell in spots:
            if not isinstance(cell, list) or len(cell) != 2 or (cell[0], cell[1]) not in {(t, s) for t in tracks for s in seeds}:
                raise ValueError(f"{phase} spot check must be a cell in its partition")
            normalized_spots.append(tuple(cell))
        if len(set(normalized_spots)) != len(normalized_spots):
            raise ValueError(f"{phase} has duplicate spot checks")
    overlap = all_seeds & historical_seeds
    if overlap:
        raise ValueError(f"historical geometry seed reuse: {sorted(overlap)}")


def expected_cells(protocol: dict, phase: str) -> list[tuple[int, int, int]]:
    partition = protocol["partitions"][phase]
    cells = [(track, seed, 0) for track in partition["track_ids"] for seed in partition["seeds"]]
    cells.extend((track, seed, 1) for track, seed in partition.get("spot_check_cells", []))
    return cells


def validate_action(action: Any):
    import numpy as np

    try:
        array = np.asarray(action, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Agent.act returned a nonnumeric action") from error
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError("Agent.act must return exactly three finite scalars")
    if np.any(array < np.array([-1.0, 0.0, 0.0], dtype=np.float32)) or np.any(array > 1.0):
        raise ValueError("Agent.act returned an action outside official bounds")
    return array


def cell_path(root: Path, phase: str, arm: str, track: int, seed: int, repeat: int) -> Path:
    if phase not in PHASES or arm not in ARMS:
        raise ValueError("invalid phase or arm")
    return root / "cells" / phase / arm / f"track-{track}-seed-{seed}-repeat-{repeat}.json"


def prepare_run(root: Path, identity: dict) -> None:
    path = root / "freeze.json"
    if path.exists():
        if _read_json(path) != identity:
            raise ValueError("run identity mismatch; use a distinct output root")
        return
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, identity)


def load_cell(root: Path, identity: dict, phase: str, arm: str, track: int, seed: int, repeat: int) -> dict | None:
    path = cell_path(root, phase, arm, track, seed, repeat)
    if not path.exists():
        return None
    envelope = _read_json(path)
    if not isinstance(envelope, dict) or envelope.get("identity") != identity:
        raise ValueError(f"cell identity mismatch: {path}")
    row = envelope.get("row")
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in (
        ("partition", phase), ("arm", arm), ("track_id", track), ("seed", seed), ("repeat", repeat)
    )):
        raise ValueError(f"cell coordinates mismatch: {path}")
    payload = {"identity": identity, "row": row}
    if envelope.get("digest") != hashlib.sha256(_canonical(payload)).hexdigest():
        raise ValueError(f"cell digest mismatch: {path}")
    return row


def record_cell(root: Path, identity: dict, row: dict) -> bool:
    path = cell_path(root, row["partition"], row["arm"], row["track_id"], row["seed"], row["repeat"])
    if path.exists():
        if load_cell(root, identity, row["partition"], row["arm"], row["track_id"], row["seed"], row["repeat"]) != row:
            raise ValueError(f"existing cell differs: {path}")
        return False
    payload = {"identity": identity, "row": row}
    _atomic_json(path, {**payload, "digest": hashlib.sha256(_canonical(payload)).hexdigest()})
    return True


def _sealed_path(root: Path, name: str) -> Path:
    return root / f"{name}.json"


def _read_seal(root: Path, identity: dict, name: str) -> dict | None:
    path = _sealed_path(root, name)
    if not path.exists():
        return None
    seal = _read_json(path)
    if seal.get("identity") != identity:
        raise ValueError(f"{name} identity mismatch")
    if seal.get("summary_sha256") != hashlib.sha256(_canonical(seal.get("summary"))).hexdigest():
        raise ValueError(f"{name} digest mismatch")
    if seal["summary"].get("decision") != "RETAIN":
        raise ValueError(f"{name} must have a RETAIN decision")
    return seal


def _write_seal(root: Path, identity: dict, name: str, summary: dict) -> None:
    value = {"identity": identity, "summary": summary, "summary_sha256": hashlib.sha256(_canonical(summary)).hexdigest()}
    path = _sealed_path(root, name)
    if path.exists():
        if _read_json(path) != value:
            raise ValueError(f"{name} seal already exists with different content")
        return
    _atomic_json(path, value)


def freeze_finalist(root: Path, identity: dict, protocol: dict) -> None:
    summary = report_phase(root, identity, protocol, "screen")
    if summary.get("decision") != "RETAIN":
        raise ValueError("screen must RETAIN candidate before finalist freeze")
    _write_seal(root, identity, "finalist", summary)


def seal_confirmation(root: Path, identity: dict, protocol: dict) -> None:
    require_phase(root, identity, "confirmation", protocol)
    summary = report_phase(root, identity, protocol, "confirmation")
    if summary.get("decision") != "RETAIN":
        raise ValueError("confirmation must RETAIN candidate to unlock blind")
    _write_seal(root, identity, "confirmation-accepted", summary)


def _require_current_seal(root: Path, identity: dict, protocol: dict, name: str, predecessor: str) -> None:
    seal = _read_seal(root, identity, name)
    if seal is None:
        raise ValueError(f"{name} seal is missing")
    current = report_phase(root, identity, protocol, predecessor)
    if hashlib.sha256(_canonical(current)).hexdigest() != seal["summary_sha256"]:
        raise ValueError(f"{name} seal no longer matches {predecessor} cells")


def require_phase(root: Path, identity: dict, phase: str, protocol: dict) -> None:
    if phase == "screen":
        return
    if phase == "confirmation":
        _require_current_seal(root, identity, protocol, "finalist", "screen")
        return
    if phase == "blind":
        require_phase(root, identity, "confirmation", protocol)
        _require_current_seal(root, identity, protocol, "confirmation-accepted", "confirmation")
        return
    raise ValueError("unknown phase")


def compare_pairs(rows: list[dict], cells: list[tuple[int, int, int]]) -> dict:
    """Official priorities plus conservative preregistered safety/time vetoes.

    Only canonical repeat-0 cells count as evidence. Repeat-1 is a deterministic
    reload check and never creates an extra statistical observation.
    """
    pair_lookup = {(row["track_id"], row["seed"], row["repeat"], row["arm"]): row for row in rows}
    reasons: list[str] = []
    missing: list[list[int]] = []
    finishes = {arm: 0 for arm in ARMS}
    progress = {arm: 0.0 for arm in ARMS}
    contacts = {arm: 0 for arm in ARMS}
    damage = {arm: 0.0 for arm in ARMS}
    offtrack = {arm: 0 for arm in ARMS}
    partial_offtrack = {arm: 0 for arm in ARMS}
    paired_lap_delta = 0
    paired_laps = 0
    for track, seed, repeat in cells:
        if repeat != 0:
            continue
        pair = [pair_lookup.get((track, seed, repeat, arm)) for arm in ARMS]
        if any(row is None for row in pair):
            missing.append([track, seed])
            continue
        control, candidate = pair
        if control.get("error") or candidate.get("error"):
            reasons.append(f"{track}/{seed}: operational failure")
            continue
        for arm, row in zip(ARMS, pair):
            finishes[arm] += int(row["finished"])
            progress[arm] += float(row["progress"])
            contacts[arm] += int(row["collision_count"])
            damage[arm] += float(row["damage"])
            offtrack[arm] += int(row.get("offtrack_samples", 0))
            partial_offtrack[arm] += int(row.get("partial_offtrack_samples", 0))
        if control["finished"] and not candidate["finished"]:
            reasons.append(f"{track}/{seed}: control finish lost")
        if candidate["collision_count"] > control["collision_count"]:
            reasons.append(f"{track}/{seed}: collisions increased")
        if candidate["damage"] > control["damage"] + 1e-9:
            reasons.append(f"{track}/{seed}: damage increased")
        if not control["finished"] and not candidate["finished"] and candidate["progress"] + 1e-9 < control["progress"]:
            reasons.append(f"{track}/{seed}: DNF progress decreased")
        if control["finished"] and candidate["finished"]:
            paired_laps += 1
            paired_lap_delta += candidate["lap_time_ms"] - control["lap_time_ms"]
            if candidate["lap_time_ms"] > 1.05 * control["lap_time_ms"]:
                reasons.append(f"{track}/{seed}: shared finish more than 5% slower")
    denominator = len([cell for cell in cells if cell[2] == 0])
    mean_progress_delta = (progress["candidate"] - progress["control"]) / denominator
    decision = "REJECT" if reasons else "INCOMPLETE" if missing else (
        "RETAIN" if finishes["candidate"] > finishes["control"] or
        (finishes["candidate"] == finishes["control"] and (
            mean_progress_delta > 1e-9 or (abs(mean_progress_delta) <= 1e-9 and paired_lap_delta < 0)
        )) else "INCONCLUSIVE"
    )
    return {"decision": decision, "reasons": reasons, "missing_cells": missing,
            "canonical_cells": denominator, "control_finishes": finishes["control"],
            "candidate_finishes": finishes["candidate"], "control_mean_progress": progress["control"] / denominator,
            "candidate_mean_progress": progress["candidate"] / denominator,
            "control_contacts": contacts["control"], "candidate_contacts": contacts["candidate"],
            "control_damage": damage["control"], "candidate_damage": damage["candidate"],
            "control_offtrack_samples": offtrack["control"], "candidate_offtrack_samples": offtrack["candidate"],
            "control_partial_offtrack_samples": partial_offtrack["control"],
            "candidate_partial_offtrack_samples": partial_offtrack["candidate"],
            "shared_completed_cells": paired_laps, "paired_completed_time_delta_ms": paired_lap_delta}


def _verify_spot_checks(rows: list[dict], protocol: dict, phase: str) -> list[str]:
    lookup = {(row["track_id"], row["seed"], row["repeat"], row["arm"]): row for row in rows}
    failures = []
    for track, seed in protocol["partitions"][phase].get("spot_check_cells", []):
        for arm in ARMS:
            original = lookup.get((track, seed, 0, arm))
            repeat = lookup.get((track, seed, 1, arm))
            if original is None or repeat is None:
                failures.append(f"{track}/{seed}/{arm}: spot check missing")
            elif any(original.get(key) != repeat.get(key) for key in (
                "finished", "progress", "lap_time_ms", "collision_count", "damage", "retire_reason",
                "offtrack_samples", "partial_offtrack_samples", "action_trace_sha256"
            )):
                failures.append(f"{track}/{seed}/{arm}: nondeterministic cold reload")
    return failures


def report_phase(root: Path, identity: dict, protocol: dict, phase: str) -> dict:
    cells = expected_cells(protocol, phase)
    rows = []
    for track, seed, repeat in cells:
        for arm in ARMS:
            row = load_cell(root, identity, phase, arm, track, seed, repeat)
            if row is not None:
                rows.append(row)
    summary = compare_pairs(rows, cells)
    spot_failures = _verify_spot_checks(rows, protocol, phase)
    if spot_failures:
        if summary["decision"] != "REJECT":
            summary["decision"] = "INCOMPLETE" if summary["missing_cells"] or any("missing" in item for item in spot_failures) else "REJECT"
        summary["reasons"].extend(spot_failures)
    summary["partition"] = phase
    summary["protocol_sha256"] = identity["protocol_sha256"]
    summary["candidate_agent_sha256"] = identity["source_sha256"]["candidate"]
    return summary


def build_identity(protocol_path: Path, protocol: dict, control: Path, candidate: Path, model: Path) -> dict:
    sources = {"control": digest(control), "candidate": digest(candidate)}
    for arm in ARMS:
        if sources[arm] != protocol[f"{arm}_agent_sha256"]:
            raise ValueError(f"{arm} agent source hash mismatch")
    model_hash = digest(model)
    if model_hash != protocol["model_sha256"]:
        raise ValueError("model hash mismatch")
    helper_hashes = {name: digest(ROOT / name) for name in protocol["runtime_helper_sha256"]}
    if helper_hashes != protocol["runtime_helper_sha256"]:
        raise ValueError("runtime helper hash mismatch")
    environment_paths = sorted((ROOT / "core").rglob("*.py")) + sorted((ROOT / "local_simulator").rglob("*.py"))
    environment_paths += [ROOT / "env_wrapper.py", ROOT / "damage.py"]
    return {"protocol_sha256": digest(protocol_path), "source_sha256": sources,
            "model_sha256": model_hash, "harness_sha256": digest(Path(__file__)),
            "helper_sha256": helper_hashes,
            "environment_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in environment_paths},
            "platform": platform.platform(), "python": platform.python_version()}


def check_frozen_inputs(identity: dict, paths: dict[str, Path]) -> None:
    for arm in ARMS:
        if digest(paths[arm]) != identity["source_sha256"][arm]:
            raise ValueError(f"{arm} source changed during the run")
    if digest(paths["model"]) != identity["model_sha256"]:
        raise ValueError("model changed during the run")
    if digest(paths["protocol"]) != identity["protocol_sha256"]:
        raise ValueError("protocol changed during the run")
    if digest(Path(__file__)) != identity["harness_sha256"]:
        raise ValueError("harness changed during the run")
    for name, expected in identity["helper_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"runtime helper changed during the run: {name}")
    for name, expected in identity["environment_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"environment changed during the run: {name}")


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    path = root / "run.lock"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"another evaluation owns {path}") from error
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _load_agent(source_path: Path, model_path: Path, expected_class: str):
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    spec = importlib.util.spec_from_file_location("agent", source_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)
    agent = module.Agent(model_path=str(model_path))
    actual = type(getattr(agent, "_forward_controller", None)).__name__
    if actual != expected_class:
        raise ValueError(f"actual Agent route {actual!r}, expected {expected_class!r}")
    return agent


def _peak_worker_rss_mib() -> float:
    """Return peak RSS of this process, including the local simulator."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        size = ctypes.c_size_t

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", size), ("WorkingSetSize", size),
                        ("QuotaPeakPagedPoolUsage", size), ("QuotaPagedPoolUsage", size),
                        ("QuotaPeakNonPagedPoolUsage", size), ("QuotaNonPagedPoolUsage", size),
                        ("PagefileUsage", size), ("PeakPagefileUsage", size), ("PrivateUsage", size)]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024 * 1024)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _worker(payload: dict) -> dict:
    import numpy as np
    from local_simulator.schema import MapSpec
    from local_simulator.session import SimulationSession

    random.seed(0)
    np.random.seed(0)
    started = time.perf_counter()
    agent = _load_agent(Path(payload["source"]), Path(payload["model"]), payload["controller_class"])
    init_ms = (time.perf_counter() - started) * 1000
    if init_ms > 10000:
        raise RuntimeError(f"Agent initialization exceeded 10 seconds: {init_ms:.1f}ms")

    class ExplicitPolicy:
        name = "actual-Agent-route"
        reset_ms = 0.0

        def reset(self, observation):
            began = time.perf_counter()
            agent.reset(observation)
            self.reset_ms = (time.perf_counter() - began) * 1000
            if self.reset_ms > 5000:
                raise RuntimeError(f"Agent.reset exceeded 5 seconds: {self.reset_ms:.1f}ms")

        def act(self, observation):
            raise AssertionError("worker must explicitly validate Agent.act")

    policy = ExplicitPolicy()
    spec = MapSpec(payload["track_id"], payload["seed"], "official", (), 2000, 4)
    session = SimulationSession.start(spec, policy)
    action_hash = hashlib.sha256()
    max_act_ms = 0.0
    action_count = 0
    offtrack_samples = 0
    partial_offtrack_samples = 0
    try:
        while not session.done and len(session.steps) < 2000:
            began = time.perf_counter()
            action = validate_action(agent.act(session.observation))
            act_ms = (time.perf_counter() - began) * 1000
            if act_ms > 5000:
                raise RuntimeError(f"Agent.act exceeded 5 seconds: {act_ms:.1f}ms")
            max_act_ms = max(max_act_ms, act_ms)
            action_hash.update(action.tobytes())
            session.step(action)
            action_count += 1
            wheel_contacts = [bool(wheel.tiles) for wheel in session.raw_environment.car.wheels]
            offtrack_samples += int(not any(wheel_contacts))
            partial_offtrack_samples += int(not all(wheel_contacts))
        summary = session.finish().summary
    finally:
        session.close()
    peak_rss_mib = _peak_worker_rss_mib()
    if peak_rss_mib > 1024:
        raise RuntimeError(f"worker peak RSS exceeded 1024 MiB: {peak_rss_mib:.1f} MiB")
    return {"finished": summary["finished"], "progress": summary["progress"],
            "lap_time_ms": summary["lap_time_ms"], "collision_count": summary["collision_count"],
            "damage": summary["damage"], "retire_reason": summary["retire_reason"],
            "steps": action_count, "initialization_ms": init_ms, "reset_ms": policy.reset_ms,
            "action_latency_max_ms": max_act_ms, "peak_worker_rss_mib": peak_rss_mib,
            "offtrack_samples": offtrack_samples, "partial_offtrack_samples": partial_offtrack_samples,
            "action_trace_sha256": action_hash.hexdigest(), "error": None}


def run_partition(root: Path, identity: dict, protocol: dict, paths: dict[str, Path], phase: str) -> dict:
    require_phase(root, identity, phase, protocol)
    with _run_lock(root):
        for track, seed, repeat in expected_cells(protocol, phase):
            for arm in ARMS:
                prior = load_cell(root, identity, phase, arm, track, seed, repeat)
                if prior is not None:
                    if prior.get("error"):
                        raise RuntimeError(f"recorded operational failure: {phase}/{arm}/{track}/{seed}/{repeat}; use a new protocol run")
                    continue
                check_frozen_inputs(identity, paths)
                payload = {"source": str(paths[arm].resolve()), "model": str(paths["model"].resolve()),
                           "controller_class": protocol["controller_classes"][arm],
                           "track_id": track, "seed": seed}
                row = {"partition": phase, "arm": arm, "track_id": track, "seed": seed, "repeat": repeat}
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(payload, separators=(",", ":"))]
                try:
                    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=300, check=True)
                    measured = json.loads(completed.stdout)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    row.update({"error": str(error), "stderr": stderr[-4000:]})
                    record_cell(root, identity, row)
                    failure_summary = report_phase(root, identity, protocol, phase)
                    _atomic_json(root / f"{phase}-summary.json", failure_summary)
                    print(json.dumps(failure_summary, indent=2, allow_nan=False), flush=True)
                    raise RuntimeError(f"worker failed at {phase}/{arm}/{track}/{seed}/{repeat}") from error
                check_frozen_inputs(identity, paths)
                row.update(measured)
                record_cell(root, identity, row)
                print(json.dumps({"cell": [phase, arm, track, seed, repeat], "finished": row["finished"],
                                  "progress": row["progress"], "contacts": row["collision_count"]}), flush=True)
    summary = report_phase(root, identity, protocol, phase)
    summary_path = root / f"{phase}-summary.json"
    if summary_path.exists() and _read_json(summary_path) != summary:
        raise ValueError(f"existing {phase} summary differs")
    if not summary_path.exists():
        _atomic_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--control-agent", type=Path)
    parser.add_argument("--candidate-agent", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--partition", choices=PHASES)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--freeze-finalist", action="store_true")
    parser.add_argument("--seal-confirmation", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        print(json.dumps(_worker(json.loads(args.worker)), allow_nan=False))
        return 0
    if any(value is None for value in (args.protocol, args.control_agent, args.candidate_agent, args.model, args.output_root)):
        parser.error("protocol, both agent source snapshots, model, and output root are required")
    selected = sum((args.partition is not None, args.preflight_only, args.freeze_finalist, args.seal_confirmation))
    if selected != 1:
        parser.error("choose exactly one of --partition, --preflight-only, --freeze-finalist, --seal-confirmation")
    protocol = _read_json(args.protocol)
    validate_protocol(protocol, historical_geometry_seeds(ROOT / "experiments", exclude=args.protocol))
    identity = build_identity(args.protocol, protocol, args.control_agent, args.candidate_agent, args.model)
    paths = {"control": args.control_agent, "candidate": args.candidate_agent,
             "model": args.model, "protocol": args.protocol}
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "identity": identity, "cells": {
            phase: len(expected_cells(protocol, phase)) for phase in PHASES}}, indent=2))
        return 0
    prepare_run(args.output_root, identity)
    if args.freeze_finalist:
        freeze_finalist(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "screen")
    elif args.seal_confirmation:
        seal_confirmation(args.output_root, identity, protocol)
        summary = report_phase(args.output_root, identity, protocol, "confirmation")
    else:
        summary = run_partition(args.output_root, identity, protocol, paths, args.partition)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
