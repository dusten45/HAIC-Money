"""Source-bound G1 TRAIN coverage collector; no intervention or image verdict.

The historic four-core-hour contract still cannot stop G0 in a running cell and
remains blocked. A distinct frozen new-host contract may explicitly omit that local
CPU cap; each episode still has the original 2,000-decision task deadline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec
from haic.algorithms.rlpd.g1_coverage import ACTOR_HASHES, CoverageRules
from haic.algorithms.rlpd.g0_diagnostic import G0EventRules
from haic.train_seed_reservations import validate_train_claim
from scripts.audit_rlpd_g1_coverage_seeds import _EXCLUSION_KEYS, _Inventory
from scripts.diagnose_rlpd_g0 import SOURCE_FILES as G0_SOURCE_FILES


SHA = re.compile(r"[0-9a-f]{64}\Z")
ACTORS = ("entropy-v5-author-seed50", "long-horizon-seed11")
ACTOR_PATHS = (
    "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/rlpd-author-target-seed50/checkpoints/step-000131072/actor.pt",
    "runs/20260924-pixel-rlpd-long-horizon-followup-v1/rlpd-seed11/checkpoints/step-000131072/actor.pt",
)
SOURCE_FILES = G0_SOURCE_FILES | frozenset({
    "scripts/diagnose_rlpd_g1_coverage.py", "scripts/audit_rlpd_g1_coverage_seeds.py",
    "haic/train_seed_reservations.py", "haic/algorithms/rlpd/g1_coverage.py",
})
BLOCKER = "G0 run_cell has no enforceable mid-episode 4 core-hour process stop; real G1 collection disabled"


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError(f"{name}: lowercase SHA-256 required")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("ascii")).hexdigest()


def _file(root: Path, relative: str, prefix: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"unsafe {prefix} path")
    parts = Path(relative).parts
    if parts[0] != prefix or any(part in (".", "..") for part in parts):
        raise ValueError(f"unsafe {prefix} path")
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlink in {relative}")
    if not current.is_file():
        raise ValueError(f"missing {relative}")
    return current


def _pinned(root: Path, relative: str, digest: str, prefix: str) -> bytes:
    raw = _file(root, relative, prefix).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(digest, relative):
        raise ValueError(f"source/hash drift: {relative}")
    return raw


def _json(raw: bytes, name: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name}: duplicate JSON field {key}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"{name}: {x}")))
    if type(value) is not dict:
        raise ValueError(f"{name}: JSON object required")
    return value


def _audit_clear(report: dict[str, Any], cells: list[dict[str, Any]], claim_sha: str) -> None:
    warnings = report.get("provenance_warnings")
    candidate_ids = re.compile(r"(?<![A-Za-z0-9_.])(?:" + "|".join(
        str(cell["geometry_seed"]) for cell in cells) + r")(?![A-Za-z0-9_.])")
    if (report.get("format") != "haic-rlpd-g1-coverage-seed-inventory-v2"
            or report.get("status") != "no_known_recorded_overlap"
            or report.get("protocol_frozen") is not True
            or report.get("self_claims_verified") is not True
            or report.get("seed_start") != cells[0]["geometry_seed"]
            or report.get("candidate_seeds") != [row["geometry_seed"] for row in cells]
            or report.get("cells") != cells
            or report.get("train_claims_sha256") != claim_sha
            or report.get("collisions") != [] or report.get("blockers") != []
             or type(warnings) is not list
             or any(type(warning) is not dict or set(warning) != {"path", "field", "reason"}
                    or any(type(warning[key]) is not str for key in ("path", "field", "reason"))
                    or (all(type(warning[key]) is str for key in ("path", "field", "reason"))
                        and ("candidate" in f"{warning['field']} {warning['reason']}".lower()
                             or candidate_ids.search(" ".join(str(warning[key]) for key in
                                                              ("path", "field", "reason"))) is not None))
                    for warning in warnings)):
        raise ValueError("BLOCKED or unresolved candidate audit/warning/claim evidence")


def preflight(
    root: Path, protocol_path: str, protocol_sha256: str, audit_path: str,
    audit_sha256: str, *, auditor: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Verify a caller-pinned, candidate-specific self-audit before any env creation.

    The auditor is injected for synthetic tests; its implementation source must be
    in the protocol's SHA-pinned source map. This does not reserve any seed.
    """
    if not callable(auditor):
        raise ValueError("candidate-specific auditor required")
    root = Path(root).resolve()
    protocol = _json(_pinned(root, protocol_path, protocol_sha256, "experiments"), "protocol")
    if (set(protocol) != {"format", "status", "partition", "study_id", "cells",
                          "source_hashes", "source_actors", "budget", "runtime",
                          "event_rules", "train_claims_sha256", "image_rubric_sha256",
                          "exclusions"}
            or protocol["format"] != "haic-rlpd-g1-coverage-protocol-v1"
            or protocol["status"] != "frozen" or protocol["partition"] != "TRAIN"
            or not isinstance(protocol["study_id"], str) or not protocol["study_id"].strip()):
        raise ValueError("G1 requires a complete frozen TRAIN protocol")
    cells = protocol["cells"]
    if (type(cells) is not list or len(cells) != 24
            or any(type(row) is not dict
                   or set(row) != {"partition", "track_id", "geometry_seed", "obstacles"}
                   or row["partition"] != "TRAIN" or row["obstacles"] is not True
                   for row in cells)):
        raise ValueError("G1 needs 24 exact TRAIN obstacle cells")
    budget = protocol["budget"]
    if (type(budget) is not dict or set(budget) != {"max_decisions_per_episode",
            "max_total_decisions", "max_total_raw_frames", "max_core_hours"}
            or any(type(budget[key]) is not int or budget[key] != expected for key, expected in (
                ("max_decisions_per_episode", 2000), ("max_total_decisions", 96000),
                ("max_total_raw_frames", 386448)))
            or (type(budget["max_core_hours"]) is not float
                or budget["max_core_hours"] != 4.0) and budget["max_core_hours"] is not None
            or budget["max_core_hours"] is None and "new-host" not in protocol["study_id"]):
        raise ValueError("G1 requires exact decision/raw limits and an explicit new-host uncapped protocol")
    rules = CoverageRules(
        cells=tuple((row["track_id"], row["geometry_seed"]) for row in cells),
        primary_actor_id=ACTORS[0], comparator_actor_id=ACTORS[1],
        image_rubric_sha256=protocol["image_rubric_sha256"],
        max_core_hours=budget["max_core_hours"],
    )
    exclusions = protocol["exclusions"]
    if type(exclusions) is not dict or set(exclusions) - _EXCLUSION_KEYS:
        raise ValueError("G1 requires known, explicit TRAIN exclusions")
    excluded = _Inventory(root)
    excluded.fields(exclusions, protocol_path, "exclusions")
    if set(row["geometry_seed"] for row in cells) & excluded.ids.keys():
        raise ValueError("G1 TRAIN cells overlap frozen exclusions")
    if (protocol["runtime"] != {"frame_skip": 4, "reward_shaping": False,
                                       "collision_penalty": 0, "interventions": False,
                                       "learner_updates": 0, "reset_initial_raw_frames": 1,
                                       "reset_noop_raw_frames": 50}):
        raise ValueError("G1 fixed decision/raw/CPU and unchanged runtime caps required")
    events = protocol["event_rules"]
    if (type(events) is not dict or set(events) != {"max_decisions", "negative_reward_limit",
                                                 "stall_window", "tile_window",
                                                 "directed_delta_epsilon", "centerline_far_threshold_m"}
            or type(events["max_decisions"]) is not int or events["max_decisions"] != 2000
            or type(events["negative_reward_limit"]) is not int
            or events["negative_reward_limit"] != 100
            or any(type(events[key]) is not int or events[key] < 1
                   for key in ("stall_window", "tile_window"))
            or any(type(events[key]) not in (int, float) or not math.isfinite(events[key])
                   or events[key] <= 0 for key in ("directed_delta_epsilon", "centerline_far_threshold_m"))):
        raise ValueError("G1 original G0 classification rules must be explicit")
    sources = protocol["source_hashes"]
    if type(sources) is not dict or set(sources) != SOURCE_FILES:
        raise ValueError("incomplete G1 executable/auditor source map")
    for relative, digest in sources.items():
        _pinned(root, relative, digest, relative.split("/", 1)[0])

    actors = protocol["source_actors"]
    if type(actors) is not list or len(actors) != 2:
        raise ValueError("G1 needs two ordered exported actors")
    payloads = {}
    for entry, actor_id, actor_path in zip(actors, ACTORS, ACTOR_PATHS):
        if (type(entry) is not dict or set(entry) != {"id", "path", "sha256",
                                                    "source_sha256", "export_protocol_sha256",
                                                    "action_mode"}
                or entry["id"] != actor_id or entry["path"] != actor_path
                or entry["sha256"] != ACTOR_HASHES[actor_id]
                or entry["action_mode"] != "exported_tanh_mean"):
            raise ValueError("G1 V5-primary/seed11 actor order or identity drift")
        _sha(entry["source_sha256"], "actor source")
        _sha(entry["export_protocol_sha256"], "actor export protocol")
        payloads[actor_id] = _pinned(root, actor_path, entry["sha256"], "runs")

    claim_sources = []
    for cell in cells:
        seed = cell["geometry_seed"]
        path = f"experiments/train-seed-claims/seed-{seed}.json"
        raw = _file(root, path, "experiments").read_bytes()
        claim = _json(raw, path)
        validate_train_claim(claim, seed)
        if (claim["status"] != "reserved" or claim["study_id"] != protocol["study_id"]
                or claim["protocol_id"] != protocol["study_id"]
                or claim["protocol_path"] not in (None, protocol_path)
                or claim["protocol_sha256"] not in (None, protocol_sha256)
                or any(claim[key] != cell[key] for key in cell)
                or not claim["audit_source"].startswith(
                    "haic-rlpd-g1-coverage-seed-inventory-v2:")):
            raise ValueError("G1 atomic claim does not bind exact frozen cell/study")
        claim_sources.append({"path": path, "sha256": hashlib.sha256(raw).hexdigest()})
    claim_sha = _digest(sorted(claim_sources, key=lambda item: item["path"]))
    if claim_sha != _sha(protocol["train_claims_sha256"], "24 claim digest"):
        raise ValueError("G1 frozen protocol does not bind all 24 atomic claims")
    audit = _json(_pinned(root, audit_path, audit_sha256, "experiments"), "audit receipt")
    _audit_clear(audit, cells, claim_sha)
    fresh = auditor(cells[0]["geometry_seed"], repo_root=root,
                    self_study_id=protocol["study_id"], self_protocol_path=protocol_path,
                    self_protocol_sha256=protocol_sha256,
                    self_audit_path=audit_path, self_audit_sha256=audit_sha256)
    if type(fresh) is not dict:
        raise ValueError("candidate auditor did not return a report")
    _audit_clear(fresh, cells, claim_sha)
    for key in ("format", "status", "seed_start", "candidate_seeds", "cells",
                 "train_claims_sha256", "self_claims_verified", "protocol_frozen",
                 "collisions", "blockers"):
        if fresh[key] != audit[key]:
            raise ValueError("candidate auditor differs from pinned audit receipt")
    return {"root": root, "protocol_path": protocol_path, "protocol_sha256": protocol_sha256,
            "audit_path": audit_path, "audit_sha256": audit_sha256, "auditor": auditor,
            "protocol": protocol, "rules": rules, "actor_payloads": payloads}


def collect(
    context: dict[str, Any], output_root: str | None = None, *, env_factory=None,
    run_cell=None, actor_factory=None, payload_loader=None,
    clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Collect an uncapped, separately frozen 24x2 TRAIN cohort exactly once.

    Injections are for isolated tests. With no overrides this constructs root Agent
    exports, uses train.build_env and drives the unmodified G0 run_cell. A reserved
    slot is never retried after any reset/step/storage failure or interruption.
    """
    if context["protocol"]["budget"]["max_core_hours"] is not None:
        return {"status": "BLOCKED", "reason": BLOCKER, "scheduled_slots": 48,
                "environment_creations": 0}
    if not isinstance(output_root, str) or not output_root.startswith("runs/"):
        raise ValueError("new-host G1 needs a new repository-relative runs/ output_root")
    root = Path(context["root"]).resolve()
    relative = Path(output_root)
    if (relative.is_absolute() or relative.as_posix() != output_root or len(relative.parts) < 2
            or relative.parts[0] != "runs" or any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError("unsafe G1 run path")
    output = root
    for part in relative.parts:
        output /= part
        if output.is_symlink():
            raise ValueError("symlinked G1 run path")
    if not output.parent.is_dir() or output.exists():
        raise ValueError("G1 requires an unallocated run directory with an existing parent")
    if not callable(clock):
        raise ValueError("process clock required")

    check = lambda: preflight(root, context["protocol_path"], context["protocol_sha256"],
                              context["audit_path"], context["audit_sha256"],
                              auditor=context["auditor"])
    context = check()
    if env_factory is None:
        from train import build_env
        env_factory = build_env
    if run_cell is None:
        from scripts.diagnose_rlpd_g0 import run_cell as original_run_cell
        run_cell = original_run_cell
    if actor_factory is None:
        from agent import Agent
        actor_factory = lambda path: Agent(model_path=str(path))
    if payload_loader is None:
        payload_loader = lambda path: torch.load(path, map_location="cpu", weights_only=True)
    if any(not callable(item) for item in (env_factory, run_cell, actor_factory, payload_loader)):
        raise ValueError("G1 requires real callable environment, runner and actor loaders")
    protocol = context["protocol"]
    actors = {}
    fixture = np.zeros((4, 84, 84), dtype=np.float32)
    adapter = ActionAdapter()
    for entry in protocol["source_actors"]:
        path = _file(root, entry["path"], "runs")
        payload = payload_loader(path)
        if (type(payload) is not dict or payload.get("format") != "haic-rlpd-pixel-actor-v1"
                or payload.get("source_sha256") != entry["source_sha256"]
                or payload.get("protocol_sha256") != entry["export_protocol_sha256"]
                or payload.get("observation_spec") != asdict(ObservationSpec())
                or payload.get("action_spec") != asdict(ActionSpec())):
            raise ValueError("G1 actor export metadata differs from pinned identity")
        first, second = actor_factory(path), actor_factory(path)
        first.reset(fixture)
        second.reset(fixture)
        first_action = np.asarray(first.act(fixture))
        second_action = np.asarray(second.act(fixture))
        with torch.inference_mode():
            native = first.model(torch.from_numpy(fixture).unsqueeze(0)).squeeze(0).detach().numpy()
        if (first_action.shape != (3,) or first_action.dtype != np.float32
                or not np.array_equal(first_action, second_action)
                or not np.allclose(first_action, adapter.to_official(native, clip=False), atol=1e-6, rtol=0)):
            raise ValueError("G1 CPU actor reload/native action mismatch")
        actors[entry["id"]] = first
    check()
    output.mkdir()
    for directory in ("attempts", "aborts", "traces", "cells"):
        (output / directory).mkdir()
    rows = [{"partition": "TRAIN", "track_id": cell["track_id"],
             "geometry_seed": cell["geometry_seed"], "obstacles": True,
             "actor_id": actor["id"], "actor_sha256": actor["sha256"],
             "image_rubric_sha256": protocol["image_rubric_sha256"],
             "image_flag": None, "status": "unrun", "outcome": None,
             "decisions": 0, "driven_raw_frames": 0, "reset_initial_raw_frames": 0,
             "reset_noop_raw_frames": 0}
            for cell in protocol["cells"] for actor in protocol["source_actors"]]
    _slots(output / "slots.json", rows)
    start = clock()
    status = "collected_unreviewed"
    for index, row in enumerate(rows):
        name = f"slot-{index:02d}"
        cell, actor_info = protocol["cells"][index // 2], protocol["source_actors"][index % 2]
        spent = sum(item["decisions"] for item in rows if item["status"] != "unrun")
        raw_spent = sum(item["driven_raw_frames"] + item["reset_initial_raw_frames"]
                        + item["reset_noop_raw_frames"] for item in rows if item["status"] != "unrun")
        try:
            check()
            if (spent + 2000 > 96000 or raw_spent + 8051 > 386448):
                raise ValueError("G1 fixed decision/raw reservation exhausted")
        except BaseException as exc:
            _exclusive(output / "halt.json", {"slot": index, "phase": "before_attempt",
                                                    "error": f"{type(exc).__name__}: {exc}"})
            status = "pre_reset_halt"
            break
        _exclusive(output / "attempts" / f"{name}.json", {
            "format": "haic-rlpd-g1-attempt-v1", "status": "reserved_attempt",
            "slot": index, "cell": cell, "actor_id": actor_info["id"],
            "actor_sha256": actor_info["sha256"],
            "protocol_sha256": context["protocol_sha256"],
            "audit_sha256": context["audit_sha256"],
            "decision_reservation": 2000, "completed_decisions_before": spent,
        })
        row["status"] = "reserved_attempt"
        _slots(output / "slots.json", rows)
        attempt: dict[str, Any] = {}
        try:
            result, trace = run_cell(
                row=cell, actor=actors[actor_info["id"]], actor_info=actor_info,
                actor_payload=context["actor_payloads"][actor_info["id"]],
                rules=G0EventRules(**{key: value for key, value in protocol["event_rules"].items()
                                      if key != "centerline_far_threshold_m"}),
                centerline_far_threshold_m=protocol["event_rules"]["centerline_far_threshold_m"],
                env_factory=env_factory, attempt=attempt,
            )
            check()
            steps, driven = result["steps"], result["driven_raw_frames"]
            initial, noop = result["reset_initial_raw_frames"], result["reset_noop_raw_frames"]
            outcome = result["summary"]["outcome"]
            if (any(type(value) is not int for value in (steps, driven, initial, noop))
                    or not 1 <= steps <= 2000 or not 4 * (steps - 1) + 1 <= driven <= 4 * steps
                    or initial != 1 or noop != 50 or spent + steps > 96000
                    or raw_spent + driven + initial + noop > 386448
                    or any(result.get(key) != cell[key] for key in ("partition", "track_id", "geometry_seed"))
                    or result.get("actor_id") != actor_info["id"]
                    or result.get("actor_sha256") != actor_info["sha256"]
                    or not isinstance(result.get("road_centerline_sha256"), str)
                    or SHA.fullmatch(result["road_centerline_sha256"]) is None
                    or outcome not in ("finished", "off_track", "crash", "out_of_bounds", "task_timeout", "unknown")
                    or index % 2 == 1 and rows[index - 1]["status"] == "complete"
                    and rows[index - 1]["road_centerline_sha256"] != result["road_centerline_sha256"]):
                raise ValueError("G1 result identity/outcome or decision/raw budget mismatch")
            if (type(trace) is not dict or any(key not in trace for key in ("initial_stack", "finished", "terminated", "truncated"))
                    or any(len(np.asarray(trace[key])) != steps for key in ("finished", "terminated", "truncated"))):
                raise ValueError("G1 result lacks observed terminal trace")
            finished, terminated, truncated = (bool(np.asarray(trace[key])[-1])
                                               for key in ("finished", "terminated", "truncated"))
            if (not terminated and not truncated or outcome == "finished" and not finished
                    or outcome == "task_timeout" and (steps != 2000 or not truncated or finished)
                    or outcome in ("off_track", "crash", "out_of_bounds") and (not terminated or finished)
                    or finished and outcome != "finished"):
                raise ValueError("G1 task outcome cannot replace a censored or incomplete episode")
            trace_path = output / "traces" / f"{name}.npz"
            with trace_path.open("xb") as stream:
                np.savez_compressed(stream, **trace)
                stream.flush()
                os.fsync(stream.fileno())
            result.update(trace_path=f"traces/{name}.npz", trace_sha256=_hash_file(trace_path),
                          protocol_sha256=context["protocol_sha256"])
            _exclusive(output / "cells" / f"{name}.json", result)
            row.update(status="complete", outcome=outcome, decisions=steps,
                       driven_raw_frames=driven, reset_initial_raw_frames=initial,
                       reset_noop_raw_frames=noop, finished=finished, terminated=terminated,
                       truncated=truncated, road_centerline_sha256=result["road_centerline_sha256"],
                       trace_path=result["trace_path"], trace_sha256=result["trace_sha256"],
                       cell_sha256=_hash_file(output / "cells" / f"{name}.json"))
            _slots(output / "slots.json", rows)
        except BaseException as exc:
            counters = attempt.get("raw_counters", {})
            row.update(status="collection_censored", outcome="unknown",
                       decisions=attempt.get("decisions_completed", 0),
                       decision_calls_at_least=attempt.get("decision_calls", 0),
                       driven_raw_frames=counters.get("driven_raw_frames", 0),
                       reset_initial_raw_frames=counters.get("reset_initial_raw_frames", 0),
                       reset_noop_raw_frames=counters.get("reset_noop_raw_frames", 0),
                       partial_trace_path=None, partial_trace_sha256=None)
            receipt = {"format": "haic-rlpd-g1-censor-v1", "slot": index,
                       "status": "collection_censored", "outcome": "unknown",
                       "protocol_sha256": context["protocol_sha256"],
                       "phase": attempt.get("phase", "before_environment_creation"),
                       "decision_calls_at_least": row["decision_calls_at_least"],
                       "decisions_completed": row["decisions"], "raw_counters": counters,
                       "error": f"{type(exc).__name__}: {exc}", "no_retry_or_top_up": True}
            # The small terminal receipt precedes a potentially disk-filling NPZ.
            _exclusive(output / "aborts" / f"{name}.json", receipt)
            partial = attempt.get("partial_arrays")
            if partial is not None:
                try:
                    path = output / "aborts" / f"{name}.npz"
                    with path.open("xb") as stream:
                        np.savez_compressed(stream, initial_stack=attempt["initial_stack"],
                                            **{key: np.asarray(value) for key, value in partial.items()})
                        stream.flush()
                        os.fsync(stream.fileno())
                    row.update(partial_trace_path=f"aborts/{name}.npz",
                               partial_trace_sha256=_hash_file(path))
                    _exclusive(output / "aborts" / f"{name}-partial.json", {
                        "slot": index, "protocol_sha256": context["protocol_sha256"],
                        "partial_trace_path": row["partial_trace_path"],
                        "partial_trace_sha256": row["partial_trace_sha256"],
                    })
                except BaseException:
                    # The original censor receipt and attempt remain authoritative.
                    row.update(partial_trace_path=None, partial_trace_sha256=None)
            _slots(output / "slots.json", rows)
            status = "collection_aborted"
            break
    summary = {"format": "haic-rlpd-g1-collection-v1", "status": status,
               "role": "TRAIN-only-coverage-unreviewed-not-official", "scheduled_slots": 48,
               "complete_slots": sum(row["status"] == "complete" for row in rows),
               "censored_slots": sum(row["status"] == "collection_censored" for row in rows),
               "unrun_slots": sum(row["status"] == "unrun" for row in rows),
               "decisions_observed": sum(row["decisions"] for row in rows),
               "raw_frames_observed": sum(row["driven_raw_frames"] + row["reset_initial_raw_frames"]
                                          + row["reset_noop_raw_frames"] for row in rows),
               "core_hours_observed": (clock() - start) / 3600,
               "protocol_path": context["protocol_path"],
               "protocol_sha256": context["protocol_sha256"],
               "audit_sha256": context["audit_sha256"],
               "slots_sha256": _hash_file(output / "slots.json"),
               "image_review": "not_sealed; no coverage pass or winner selection"}
    _exclusive(output / "manifest.json", summary)
    return summary


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _slots(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    _exclusive(temporary, rows)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def simulate_schedule(
    context: dict[str, Any], output: Path, *, env_factory, run_cell,
    clock: Callable[[], float] = time.process_time,
    store_trace: Callable[[Path, Any], None] = _exclusive,
) -> dict[str, Any]:
    """Synthetic-only attempt journal. Never pass a real environment or G0 run_cell.

    Only the separate collect() entry point may use the real G0 runner. This fake
    callback has the same signature and may call an injected fake environment.
    """
    if not callable(run_cell) or not callable(env_factory) or not callable(clock):
        raise ValueError("synthetic injected runner/environment/clock required")
    if run_cell.__module__ == "scripts.diagnose_rlpd_g0":
        raise ValueError(BLOCKER)
    check = lambda: preflight(context["root"], context["protocol_path"],
                              context["protocol_sha256"], context["audit_path"],
                              context["audit_sha256"], auditor=context["auditor"])
    check()
    output = Path(output)
    if output.is_symlink() or output.exists() or not output.parent.is_dir():
        raise ValueError("synthetic output must be a new directory")
    output.mkdir()
    for folder in ("attempts", "aborts", "traces"):
        (output / folder).mkdir()
    protocol = context["protocol"]
    rows = [{"partition": "TRAIN", "track_id": cell["track_id"],
             "geometry_seed": cell["geometry_seed"], "obstacles": True,
             "actor_id": actor["id"], "actor_sha256": actor["sha256"],
             "status": "unrun", "outcome": None, "image_flag": None}
            for cell in protocol["cells"] for actor in protocol["source_actors"]]
    _slots(output / "slots.json", rows)
    start = clock()
    spent = raw_spent = 0
    status = "synthetic_only_complete"
    for index, row in enumerate(rows):
        attempt = {}
        name = f"slot-{index:02d}"
        try:
            check()
            elapsed = (clock() - start) / 3600
            if not math.isfinite(elapsed) or elapsed < 0 or elapsed >= 4:
                raise ValueError("synthetic process CPU budget exhausted before reset")
            if spent + 2000 > 96000 or raw_spent + 8051 > 386448:
                raise ValueError("G1 decision/raw reservation exhausted")
        except Exception as exc:
            _exclusive(output / "halt.json", {"slot": index, "phase": "before_attempt",
                                              "error": f"{type(exc).__name__}: {exc}"})
            status = "synthetic_only_halted"
            break
        # This exclusive receipt must exist before the fake environment is called.
        _exclusive(output / "attempts" / f"{name}.json", {
            "slot": index, "cell": protocol["cells"][index // 2],
            "actor_id": row["actor_id"], "protocol_sha256": context["protocol_sha256"],
            "decision_reservation": 2000, "completed_decisions_before": spent,
            "status": "reserved_attempt", "synthetic_only": True,
        })
        try:
            actor = protocol["source_actors"][index % 2]
            result, trace = run_cell(
                row=protocol["cells"][index // 2], actor=None, actor_info=actor,
                actor_payload=context["actor_payloads"][actor["id"]],
                rules=protocol["event_rules"],
                centerline_far_threshold_m=protocol["event_rules"]["centerline_far_threshold_m"],
                env_factory=env_factory, attempt=attempt,
            )
            check()
            elapsed = (clock() - start) / 3600
            steps = result["steps"]
            driven = result["driven_raw_frames"]
            initial = result["reset_initial_raw_frames"]
            noop = result["reset_noop_raw_frames"]
            if (type(steps) is not int or not 1 <= steps <= 2000
                    or any(type(value) is not int for value in (driven, initial, noop))
                    or not 4 * (steps - 1) + 1 <= driven <= 4 * steps
                    or initial != 1 or noop != 50
                    or spent + steps > 96000 or raw_spent + driven + initial + noop > 386448
                    or not math.isfinite(elapsed) or not 0 <= elapsed < 4
                    or result.get("partition") != "TRAIN"
                    or result.get("track_id") != row["track_id"]
                    or result.get("geometry_seed") != row["geometry_seed"]
                    or result.get("actor_id") != row["actor_id"]
                    or result.get("actor_sha256") != row["actor_sha256"]
                    or not isinstance(result.get("road_centerline_sha256"), str)
                    or SHA.fullmatch(result["road_centerline_sha256"]) is None
                    or result.get("summary", {}).get("outcome") not in
                    ("finished", "off_track", "crash", "out_of_bounds", "task_timeout", "unknown")
                    or index % 2 == 1 and rows[index - 1]["status"] == "complete"
                    and rows[index - 1]["road_centerline_sha256"] != result["road_centerline_sha256"]):
                raise ValueError("synthetic cell identity/outcome or fixed budget mismatch")
            trace_path = output / "traces" / f"{name}.json"
            store_trace(trace_path, trace)
            row.update(status="complete", outcome=result["summary"]["outcome"], decisions=steps,
                       driven_raw_frames=driven, reset_initial_raw_frames=initial,
                       reset_noop_raw_frames=noop,
                       road_centerline_sha256=result["road_centerline_sha256"],
                       trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
            spent += steps
            raw_spent += driven + initial + noop
            _slots(output / "slots.json", rows)
        except Exception as exc:
            # Reset/step/storage failures are unknown, never task failures. Preserve
            # observations from the fake runner even if it did not return a result.
            try:
                check()
                recheck_error = None
            except Exception as drift:
                recheck_error = f"{type(drift).__name__}: {drift}"
            counters = attempt.get("raw_counters", {})
            row.update(status="collection_censored", outcome="unknown",
                       decisions=attempt.get("decisions_completed", 0),
                       decision_calls_at_least=attempt.get("decision_calls", 0),
                       driven_raw_frames=counters.get("driven_raw_frames", 0),
                       reset_initial_raw_frames=counters.get("reset_initial_raw_frames", 0),
                       reset_noop_raw_frames=counters.get("reset_noop_raw_frames", 0))
            _slots(output / "slots.json", rows)
            _exclusive(output / "aborts" / f"{name}.json", {
                "slot": index, "status": "collection_censored", "outcome": "unknown",
                "phase": attempt.get("phase", "before_environment_creation"),
                "error": f"{type(exc).__name__}: {exc}", "recheck_error": recheck_error,
                "partial": attempt.get("partial"), "raw_counters": counters,
                "decision_calls_at_least": row["decision_calls_at_least"],
                "decisions_completed": row["decisions"], "no_retry_or_top_up": True,
            })
            status = "synthetic_only_aborted"
            break
    return {"status": status, "scheduled_slots": len(rows),
            "complete_slots": sum(row["status"] == "complete" for row in rows),
            "censored_slots": sum(row["status"] == "collection_censored" for row in rows),
            "unrun_slots": sum(row["status"] == "unrun" for row in rows),
            "decisions_completed": spent, "raw_frames_completed": raw_spent,
            "cpu_core_hours_observed": (clock() - start) / 3600,
            "image_review": "not_sealed; no coverage pass or winner selection",
            "real_collection": "synthetic schedule only; use collect with a separate uncapped protocol"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="frozen experiments/ protocol path")
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--audit", required=True, help="frozen experiments/ self-audit receipt")
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-root", help="new runs/ directory; omit for zero-reset preflight")
    args = parser.parse_args()
    from scripts.audit_rlpd_g1_coverage_seeds import audit_g1_coverage_seeds
    context = preflight(Path(__file__).resolve().parents[1], args.protocol, args.protocol_sha256,
                        args.audit, args.audit_sha256, auditor=audit_g1_coverage_seeds)
    if args.output_root:
        result = collect(context, args.output_root)
    else:
        result = {"status": "preflight_only", "scheduled_slots": 48,
                  "core_hour_cap": context["protocol"]["budget"]["max_core_hours"]}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["status"] not in ("BLOCKED", "collection_aborted", "pre_reset_halt") else 2


if __name__ == "__main__":
    raise SystemExit(main())
