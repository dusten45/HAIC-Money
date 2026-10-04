"""Frozen, in-sample TRAIN screen of the final new-host RLPD actor against V5.

Both actors drive the same twelve already-consumed G0 roads on this host. This is
not fresh coverage, a matched training intervention, or an official evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, ObservationSpec
from haic.algorithms.rlpd.g0_diagnostic import G0EventRules, summarize_g0_pairs
from scripts.diagnose_rlpd_g0 import SOURCE_FILES as G0_SOURCE_FILES, run_cell
from train import build_env


ROOT = Path(__file__).resolve().parents[1]
G0_PROTOCOL = "experiments/rlpd-g0-completion-v1.json"
G0_SHA = "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b"
G0_MANIFEST = "runs/20260926-rlpd-g0-completion-v1/manifest.json"
G0_MANIFEST_SHA = "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358"
G0_LEDGER = "runs/20260926-rlpd-g0-completion-v1/cells.jsonl"
G0_LEDGER_SHA = "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559"
LEARNER_PROTOCOL = "experiments/rlpd-newhost-reused-train-20260929-v1.json"
LEARNER_PROTOCOL_SHA = "7f54719796af058deac99e208df84412e1021497bee844011f7515c21126b598"
LEARNER_SNAPSHOT = "runs/20260929-rlpd-newhost-reused-train-v1/study_protocol.json"
LEARNER_RESULT = "runs/20260929-rlpd-newhost-reused-train-v1/result.json"
LEARNER_RESULT_SHA = "dcce1072609d40d9fd210a192841137b4d22cb0101bf815d27ecea0630a28c03"
FINAL_CANDIDATE = "runs/20260929-rlpd-newhost-reused-train-v1/checkpoints/step-000131072/candidate.json"
FINAL_CANDIDATE_SHA = "b5942025705a90d7aaabffb94ac47db293584f89aeee70a27c4bbcdf99daacb3"
FINAL_ACTOR = "runs/20260929-rlpd-newhost-reused-train-v1/checkpoints/step-000131072/actor.pt"
FINAL_ACTOR_SHA = "28d82208ac8f3a99d06f8f2dd35b41f85babfa0e2d362c80ab0dcc01359190f6"
FINAL_CHECKPOINT = "runs/20260929-rlpd-newhost-reused-train-v1/checkpoints/step-000131072/checkpoint.pt"
FINAL_CHECKPOINT_SHA = "7b6f0dd958385559afcd992a550a70ded0f7a6930f41e4dc3497af0e3d781de1"
V5_ACTOR_SHA = "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98"
SOURCE_FILES = G0_SOURCE_FILES | {"scripts/diagnose_rlpd_newhost_train_screen.py"}
FORMAT = "haic-rlpd-newhost-in-sample-train-screen-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def located(root: Path, relative: str, prefix: str | None = None) -> Path:
    if not isinstance(relative, str):
        raise ValueError("path must be a repository-relative string")
    value = Path(relative)
    if (value.is_absolute() or not value.parts or value.as_posix() != relative
            or (prefix is not None and (len(value.parts) < 2 or value.parts[0] != prefix))
            or any(part in (".", "..") for part in value.parts)):
        raise ValueError(f"unsafe repository path: {relative}")
    current = root
    for part in value.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked repository path: {relative}")
    return current


def pinned(root: Path, relative: str, expected: str, prefix: str) -> Path:
    if not isinstance(expected, str) or SHA.fullmatch(expected) is None:
        raise ValueError(f"invalid SHA-256 for {relative}")
    path = located(root, relative, prefix)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"missing or changed frozen evidence: {relative}")
    return path


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def cpu_runtime() -> dict[str, Any]:
    import cv2
    import gymnasium
    import pygame

    torch.set_num_threads(1)
    return {
        "device": "cpu", "torch_num_threads": torch.get_num_threads(),
        "python": sys.version, "machine": platform.machine(), "torch": torch.__version__,
        "numpy": np.__version__, "opencv": getattr(cv2, "__version__"),
        "gymnasium": gymnasium.__version__, "pygame": pygame.version.ver,
        "box2d_py": importlib.metadata.version("box2d-py"),
    }


def evidence(root: Path, *, checkpoint: bool = True) -> dict[str, Any]:
    g0 = json.loads(pinned(root, G0_PROTOCOL, G0_SHA, "experiments").read_text())
    manifest = json.loads(pinned(root, G0_MANIFEST, G0_MANIFEST_SHA, "runs").read_text())
    ledger = [json.loads(line) for line in pinned(root, G0_LEDGER, G0_LEDGER_SHA, "runs").read_text().splitlines()]
    learner = json.loads(pinned(root, LEARNER_PROTOCOL, LEARNER_PROTOCOL_SHA, "experiments").read_text())
    pinned(root, LEARNER_SNAPSHOT, LEARNER_PROTOCOL_SHA, "runs")
    result = json.loads(pinned(root, LEARNER_RESULT, LEARNER_RESULT_SHA, "runs").read_text())
    candidate = json.loads(pinned(root, FINAL_CANDIDATE, FINAL_CANDIDATE_SHA, "runs").read_text())
    if (g0.get("format") != "haic-rlpd-g0-diagnostic-v1" or g0.get("status") != "frozen"
            or g0.get("partition") != "TRAIN" or g0.get("frame_skip") != 4
            or g0.get("max_steps") != 2000 or g0.get("decision_cap") != 48000
            or g0.get("reward_shaping") is not False or g0.get("collision_penalty") != 0
            or g0.get("interventions") is not False or g0.get("learner_updates") != 0
            or manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("protocol_sha256") != G0_SHA or manifest.get("cells_sha256") != G0_LEDGER_SHA
            or manifest.get("geometry_count") != 12 or manifest.get("cell_count") != 24
            or len(g0.get("cells", [])) != 12 or len(ledger) != 24):
        raise ValueError("historical G0 protocol/manifest/ledger is not the complete TRAIN cohort")
    original = next((actor for actor in g0.get("actors", [])
                     if actor.get("id") == "entropy-v5-author-seed50"), None)
    if (len(g0.get("actors", [])) != 2 or original is None or original["sha256"] != V5_ACTOR_SHA
            or original["environment_steps"] != 131072 or original["training_seed"] != 50
            or original["action_mode"] != "exported_tanh_mean"):
        raise ValueError("original G0 V5 actor identity changed")
    if (learner.get("format") != "haic-rlpd-newhost-reused-train-v1"
            or learner.get("status") != "frozen" or learner.get("cells") != g0["cells"]
            or learner.get("g0_protocol_sha256") != G0_SHA
            or learner.get("g0_manifest_sha256") != G0_MANIFEST_SHA
            or learner.get("g0_ledger_sha256") != G0_LEDGER_SHA
            or learner.get("seed") != 52 or learner.get("total_steps") != 131072
            or learner.get("frame_skip") != 4 or learner.get("max_steps") != 2000
            or learner.get("obstacles") is not True or learner.get("reward_shaping") is not False
            or not isinstance(learner.get("v5_prior_dataset_sha256"), str)
            or SHA.fullmatch(learner["v5_prior_dataset_sha256"]) is None
            or learner.get("official_performance_claim") is not False
            or result.get("format") != "haic-rlpd-newhost-result-v1"
            or result.get("status") != "complete" or result.get("reused_train_only") is not True
            or result.get("official_performance_claim") is not False
            or result.get("protocol_sha256") != LEARNER_PROTOCOL_SHA
            or result.get("g0_ledger_sha256") != G0_LEDGER_SHA
            or result.get("environment_steps") != 131072 or result.get("gradient_steps") != 130072
            or learner.get("candidate_steps") != [65536, 131072]
            or not isinstance(result.get("candidates"), list) or len(result["candidates"]) != 2
            or result["candidates"][1] != candidate
            or [row.get("environment_steps") for row in result["candidates"]] != [65536, 131072]
            or candidate.get("actor_path") != FINAL_ACTOR.split("20260929-rlpd-newhost-reused-train-v1/", 1)[1]
            or candidate.get("checkpoint_path") != FINAL_CHECKPOINT.split("20260929-rlpd-newhost-reused-train-v1/", 1)[1]
            or candidate.get("actor_sha256") != FINAL_ACTOR_SHA
            or candidate.get("checkpoint_sha256") != FINAL_CHECKPOINT_SHA
            or candidate.get("protocol_sha256") != LEARNER_PROTOCOL_SHA
            or candidate.get("environment_steps") != 131072 or candidate.get("gradient_steps") != 130072):
        raise ValueError("new-host final candidate does not bind the complete reused-TRAIN learner")
    if (not isinstance(g0.get("source_hashes"), dict) or set(g0["source_hashes"]) != G0_SOURCE_FILES
            or not isinstance(learner.get("source_hashes"), dict)
            or any(learner["source_hashes"].get(name) != digest
                   for name, digest in g0["source_hashes"].items())):
        raise ValueError("G0 and new-host learner source closures disagree")
    roads = {}
    for index, cell in enumerate(g0["cells"]):
        if (cell != {"partition": "TRAIN", "track_id": 1, "geometry_seed": 4272000001 + index,
                     "obstacles": True}):
            raise ValueError("screen may use only the twelve consumed track-1 obstacle TRAIN roads")
        first, second = ledger[2 * index:2 * index + 2]
        for row, actor in ((first, g0["actors"][0]), (second, original)):
            if (row.get("partition") != "TRAIN" or row.get("track_id") != 1
                    or row.get("geometry_seed") != cell["geometry_seed"]
                    or row.get("actor_id") != actor["id"] or row.get("actor_sha256") != actor["sha256"]
                    or type(row.get("steps")) is not int or not 0 < row["steps"] <= 2000
                    or row.get("summary", {}).get("outcome") in (None, "collection_censored", "unknown")):
                raise ValueError("G0 ledger does not attest both complete, consumed actor/road slots")
        road = first.get("road_centerline_sha256")
        if not isinstance(road, str) or SHA.fullmatch(road) is None or road != second.get("road_centerline_sha256"):
            raise ValueError("original G0 road-centerline pairing is invalid")
        roads[cell["geometry_seed"]] = road
    pinned(root, original["path"], V5_ACTOR_SHA, "runs")
    pinned(root, FINAL_ACTOR, FINAL_ACTOR_SHA, "runs")
    if checkpoint:
        pinned(root, FINAL_CHECKPOINT, FINAL_CHECKPOINT_SHA, "runs")
    return {"g0": g0, "learner": learner, "original": original, "roads": roads}


def actors_from_evidence(data: dict[str, Any]) -> list[dict[str, Any]]:
    original = data["original"]
    return [
        {key: original[key] for key in ("id", "path", "sha256", "source_sha256",
                                           "export_protocol_sha256", "action_mode")},
        {"id": "newhost-seed52-final", "path": FINAL_ACTOR, "sha256": FINAL_ACTOR_SHA,
         "source_sha256": canonical_sha(data["learner"]["source_hashes"]),
         "export_protocol_sha256": LEARNER_PROTOCOL_SHA, "action_mode": "exported_tanh_mean"},
    ]


def source_hashes(root: Path, data: dict[str, Any]) -> dict[str, str]:
    hashes = {relative: sha256(located(root, relative)) for relative in sorted(SOURCE_FILES)}
    if any(hashes[relative] != digest for relative, digest in data["g0"]["source_hashes"].items()):
        raise ValueError("original G0 executable source changed; cannot rebind historical semantics")
    return hashes


def write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_exclusive(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def output_path(root: Path, run_dir: str) -> Path:
    if not isinstance(run_dir, str):
        raise ValueError("output must be a new runs/<slug> directory")
    relative = Path(run_dir)
    if (len(relative.parts) != 2 or relative.parts[0] != "runs"
            or re.fullmatch(r"[a-z0-9][a-z0-9-]*", relative.parts[1]) is None):
        raise ValueError("output must be a new runs/<slug> directory")
    return located(root, run_dir, "runs")


def freeze(root: Path, protocol_path: str, run_dir: str, *, runtime_fn=None) -> dict[str, str]:
    output = located(root, protocol_path, "experiments")
    if output.exists():
        raise FileExistsError(output)
    target = output_path(root, run_dir)
    if target.exists():
        raise FileExistsError(target)
    data = evidence(root)
    rules = G0EventRules(**data["g0"]["event_rules"])
    if rules.max_decisions != 2000 or rules.negative_reward_limit != 100:
        raise ValueError("original G0 terminal rules differ from the episode deadline")
    protocol = {
        "format": FORMAT, "status": "frozen", "partition": "TRAIN",
        "role": "in-sample-consumed-TRAIN-descriptive-not-generalization-or-official-evaluation",
        "output_root": run_dir,
        "source_artifacts": {"g0_protocol": [G0_PROTOCOL, G0_SHA],
                             "g0_manifest": [G0_MANIFEST, G0_MANIFEST_SHA],
                             "g0_ledger": [G0_LEDGER, G0_LEDGER_SHA],
                             "learner_protocol": [LEARNER_PROTOCOL, LEARNER_PROTOCOL_SHA],
                             "learner_snapshot": [LEARNER_SNAPSHOT, LEARNER_PROTOCOL_SHA],
                             "learner_result": [LEARNER_RESULT, LEARNER_RESULT_SHA],
                             "final_candidate": [FINAL_CANDIDATE, FINAL_CANDIDATE_SHA],
                             "final_checkpoint": [FINAL_CHECKPOINT, FINAL_CHECKPOINT_SHA]},
        "actors": actors_from_evidence(data), "cells": data["g0"]["cells"],
        "event_rules": data["g0"]["event_rules"],
        "centerline_far_threshold_m": data["g0"]["centerline_far_threshold_m"],
        "source_hashes": source_hashes(root, data),
        "runtime": (runtime_fn or cpu_runtime)(),
        "execution": {"actor_order": ["entropy-v5-author-seed50", "newhost-seed52-final"],
                      "one_attempt_per_actor_road": True, "scheduled_slots": 24,
                      "paired_roads": 12, "max_decisions_per_episode": 2000,
                      "frame_skip": 4, "reward_shaping": False, "collision_penalty": 0.0,
                      "obstacles": True, "policy_updates": 0, "core_hour_cap": None,
                      "cpu_nice_delta": 10},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output, protocol)
    return {"protocol": protocol_path, "sha256": sha256(output)}


def _check_contract(root: Path, protocol: dict[str, Any], data: dict[str, Any], runtime_fn) -> None:
    refs = {"g0_protocol": [G0_PROTOCOL, G0_SHA], "g0_manifest": [G0_MANIFEST, G0_MANIFEST_SHA],
            "g0_ledger": [G0_LEDGER, G0_LEDGER_SHA],
            "learner_protocol": [LEARNER_PROTOCOL, LEARNER_PROTOCOL_SHA],
            "learner_snapshot": [LEARNER_SNAPSHOT, LEARNER_PROTOCOL_SHA],
            "learner_result": [LEARNER_RESULT, LEARNER_RESULT_SHA],
            "final_candidate": [FINAL_CANDIDATE, FINAL_CANDIDATE_SHA],
            "final_checkpoint": [FINAL_CHECKPOINT, FINAL_CHECKPOINT_SHA]}
    if (set(protocol) != {"format", "status", "partition", "role", "output_root", "source_artifacts",
                         "actors", "cells", "event_rules", "centerline_far_threshold_m",
                         "source_hashes", "runtime", "execution", "created_at_utc"}
            or protocol["format"] != FORMAT or protocol["status"] != "frozen"
            or protocol["partition"] != "TRAIN"
            or protocol["role"] != "in-sample-consumed-TRAIN-descriptive-not-generalization-or-official-evaluation"
            or not isinstance(protocol["output_root"], str)
            or protocol["source_artifacts"] != refs
            or protocol["actors"] != actors_from_evidence(data)
            or protocol["cells"] != data["g0"]["cells"]
            or protocol["event_rules"] != data["g0"]["event_rules"]
            or protocol["centerline_far_threshold_m"] != data["g0"]["centerline_far_threshold_m"]
            or protocol["execution"] != {"actor_order": ["entropy-v5-author-seed50", "newhost-seed52-final"],
                                         "one_attempt_per_actor_road": True, "scheduled_slots": 24,
                                         "paired_roads": 12, "max_decisions_per_episode": 2000,
                                         "frame_skip": 4, "reward_shaping": False, "collision_penalty": 0.0,
                                         "obstacles": True, "policy_updates": 0, "core_hour_cap": None,
                                         "cpu_nice_delta": 10}
            or protocol["runtime"] != runtime_fn()):
        raise ValueError("screen differs from frozen in-sample TRAIN / CPU contract")
    output_path(root, protocol["output_root"])
    if (not isinstance(protocol["source_hashes"], dict)
            or set(protocol["source_hashes"]) != SOURCE_FILES
            or protocol["source_hashes"] != source_hashes(root, data)):
        raise ValueError("current executable source closure differs from frozen screen")


def preflight(root: Path, protocol_path: str, expected_sha: str, *,
              actor_factory=None, payload_loader=None, runtime_fn=None) -> dict[str, Any]:
    path = pinned(root, protocol_path, expected_sha, "experiments")
    protocol = json.loads(path.read_text())
    data = evidence(root)
    runtime_fn = runtime_fn or cpu_runtime
    _check_contract(root, protocol, data, runtime_fn)
    rules = G0EventRules(**protocol["event_rules"])
    if rules.max_decisions != 2000 or rules.negative_reward_limit != 100:
        raise ValueError("screen must use original 2,000-decision G0 terminal semantics")
    if payload_loader is None:
        payload_loader = lambda actor_path: torch.load(actor_path, map_location="cpu", weights_only=True)
    if actor_factory is None:
        from agent import Agent
        actor_factory = lambda actor_path: Agent(model_path=str(actor_path))
    fixture_rng = np.random.default_rng(52731)
    fixtures = (np.zeros((4, 84, 84), dtype=np.float32),
                fixture_rng.integers(0, 256, size=(4, 84, 84), dtype=np.uint8).astype(np.float32) / 255)
    spec = ActionSpec()
    adapter = ActionAdapter(spec)
    low, high = np.asarray(spec.official_low), np.asarray(spec.official_high)
    actors = {}
    for entry in protocol["actors"]:
        actor_path = pinned(root, entry["path"], entry["sha256"], "runs")
        payload = payload_loader(actor_path)
        seed = 50 if entry["id"] == "entropy-v5-author-seed50" else 52
        if (not isinstance(payload, dict) or payload.get("format") != "haic-rlpd-pixel-actor-v1"
                or payload.get("source_sha256") != entry["source_sha256"]
                or payload.get("protocol_sha256") != entry["export_protocol_sha256"]
                or payload.get("training_seed") != seed or payload.get("environment_steps") != 131072
                or payload.get("observation_spec") != asdict(ObservationSpec())
                or payload.get("action_spec") != asdict(ActionSpec())):
            raise ValueError("actor export metadata does not match frozen identity")
        if entry["id"] == "newhost-seed52-final" and (
            payload.get("gradient_steps") != 130072
            or payload.get("environment_contract") != {
                "partition": "TRAIN", "reused_g0_cells": protocol["cells"],
                "v5_prior_dataset_sha256": data["learner"]["v5_prior_dataset_sha256"],
                "obstacles": True, "reward_shaping": False, "frame_skip": 4, "max_steps": 2000,
            }
        ):
            raise ValueError("final actor does not declare the completed consumed-TRAIN export")
        first, second = actor_factory(actor_path), actor_factory(actor_path)
        for observation in fixtures:
            first.reset(observation)
            second.reset(observation)
            a, b = np.asarray(first.act(observation)), np.asarray(second.act(observation))
            if (a.shape != (3,) or a.dtype != np.float32 or not np.isfinite(a).all()
                    or np.any(a < low) or np.any(a > high) or not np.array_equal(a, b)):
                raise ValueError("CPU actor reload differs on synthetic pixel fixtures")
            with torch.inference_mode():
                native = first.model(torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0))
            mapped = adapter.to_official(native.squeeze(0).detach().numpy(), clip=False)
            if not np.allclose(a, mapped, rtol=0, atol=1e-6):
                raise ValueError("CPU actor/adapter action parity failed")
        actors[entry["id"]] = (first, actor_path)
    return {"protocol": protocol, "path": path, "sha256": expected_sha, "root": root,
            "data": data, "actors": actors, "rules": rules, "runtime_fn": runtime_fn}


def _recheck(context: dict[str, Any], *, checkpoint: bool = False) -> None:
    root = context["root"]
    pinned(root, context["path"].relative_to(root).as_posix(), context["sha256"], "experiments")
    data = evidence(root, checkpoint=checkpoint)
    _check_contract(root, context["protocol"], data, context["runtime_fn"])


def collect(context: dict[str, Any], run_dir: str, *, env_factory=None, runner=None) -> dict[str, Any]:
    root = context["root"]
    if run_dir != context["protocol"]["output_root"]:
        raise ValueError("run path differs from the single frozen no-retry cohort")
    output = output_path(root, run_dir)
    if output.exists():
        raise FileExistsError(output)
    _recheck(context, checkpoint=True)
    output.mkdir(exist_ok=False)
    for name in ("attempts", "cells", "traces", "failures", "partials"):
        (output / name).mkdir()
    write_exclusive(output / "study_protocol.json", context["path"].read_bytes())
    if sha256(output / "study_protocol.json") != context["sha256"]:
        raise ValueError("exclusive protocol snapshot differs from frozen bytes")
    env_factory = env_factory or build_env
    runner = runner or run_cell
    rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for cell in context["protocol"]["cells"]:
        for actor_info in context["protocol"]["actors"]:
            index = len(rows)
            name = f"slot-{index:02d}"
            try:
                _recheck(context)
            except BaseException as exc:
                write_json(output / "halt.json", {"status": "pre-reset-halt", "next_slot": index,
                                                 "completed_slots": len(rows),
                                                 "error": f"{type(exc).__name__}: {exc}", "no_retry": True})
                raise
            write_json(output / "attempts" / f"{name}.json", {
                "status": "attempted-once", "slot": index, "cell": cell,
                "actor_id": actor_info["id"], "actor_sha256": actor_info["sha256"],
                "protocol_sha256": context["sha256"], "decision_reservation": 2000,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "reused_in_sample_train": True,
            })
            state: dict[str, Any] = {}
            actor, actor_path = context["actors"][actor_info["id"]]
            try:
                result, trace = runner(
                    row=cell, actor=actor, actor_info=actor_info, actor_payload=actor_path.read_bytes(),
                    rules=context["rules"],
                    centerline_far_threshold_m=context["protocol"]["centerline_far_threshold_m"],
                    env_factory=env_factory, attempt=state,
                )
                _recheck(context)
                road = context["data"]["roads"][cell["geometry_seed"]]
                if (result.get("partition") != "TRAIN" or result.get("track_id") != cell["track_id"]
                        or result.get("geometry_seed") != cell["geometry_seed"]
                        or result.get("actor_id") != actor_info["id"]
                        or result.get("actor_sha256") != actor_info["sha256"]
                        or result.get("road_centerline_sha256") != road
                        or (index % 2 and result.get("road_centerline_sha256")
                            != rows[-1]["road_centerline_sha256"])
                        or type(result.get("steps")) is not int or not 0 < result["steps"] <= 2000
                        or result.get("summary", {}).get("outcome") in
                        (None, "collection_censored", "unknown")):
                    raise ValueError("result is not a complete matched original G0 TRAIN road/actor slot")
                trace_path = output / "traces" / f"{name}.npz"
                with trace_path.open("xb") as stream:
                    np.savez_compressed(stream, **trace)
                    stream.flush()
                    os.fsync(stream.fileno())
                result = {**result, "format": "haic-rlpd-in-sample-train-cell-v1",
                          "status": "complete", "protocol_sha256": context["sha256"], "slot": index,
                          "trace_path": f"traces/{name}.npz", "trace_sha256": sha256(trace_path)}
                cell_path = output / "cells" / f"{name}.json"
                write_json(cell_path, result)
                rows.append(result)
                artifacts.append({"slot": index, "cell_sha256": sha256(cell_path),
                                  "trace_sha256": result["trace_sha256"]})
            except BaseException as exc:
                partial = state.get("partial_arrays")
                has_partial = isinstance(partial, dict) and bool(partial.get("summed_reward"))
                failure = {"status": "censored-invalid-attempt", "slot": index, "cell": cell,
                           "actor_id": actor_info["id"], "protocol_sha256": context["sha256"],
                           "phase": state.get("phase", "before_environment_creation"),
                           "decision_calls_at_least": state.get("decision_calls", 0),
                           "decisions_completed": state.get("decisions_completed", 0),
                           "raw_counters": state.get("raw_counters", {}),
                           "partial_receipt_path": None,
                           "error": f"{type(exc).__name__}: {exc}", "no_retry_or_topup": True}
                failure_written = False
                try:
                    write_json(output / "failures" / f"{name}.json", failure)
                    failure_written = True
                except OSError as receipt_error:
                    print(f"failure receipt unavailable: {receipt_error}; preserve attempt receipt", file=sys.stderr)
                if has_partial and isinstance(partial, dict):
                    try:
                        partial_path = output / "partials" / f"{name}.npz"
                        arrays = {key: np.asarray(values) for key, values in partial.items()}
                        if "initial_stack" in state:
                            arrays["initial_stack"] = state["initial_stack"]
                        with partial_path.open("xb") as stream:
                            np.savez_compressed(stream, **arrays)
                            stream.flush()
                            os.fsync(stream.fileno())
                        receipt = {
                            "status": "partial-only-not-an-episode", "slot": index,
                            "trace_path": f"partials/{name}.npz", "trace_sha256": sha256(partial_path),
                            "decisions_recorded": len(partial["summed_reward"]),
                        }
                        if failure_written:
                            receipt["failure_path"] = f"failures/{name}.json"
                        write_json(output / "partials" / f"{name}.json", receipt)
                    except Exception as partial_error:
                        print(f"optional partial trace unavailable: {partial_error}", file=sys.stderr)
                raise
    try:
        _recheck(context, checkpoint=True)
        for item in artifacts:
            name = f"slot-{item['slot']:02d}"
            if (sha256(output / "cells" / f"{name}.json") != item["cell_sha256"]
                    or sha256(output / "traces" / f"{name}.npz") != item["trace_sha256"]):
                raise ValueError(f"completed slot {name} artifact changed before summary")
        summary = summarize_g0_pairs(rows,
                                     cells=[(cell["track_id"], cell["geometry_seed"])
                                            for cell in context["protocol"]["cells"]],
                                     actor_ids=tuple(a["id"] for a in context["protocol"]["actors"]))
    except BaseException as exc:
        try:
            write_json(output / "halt.json", {"status": "post-collection-invalid", "completed_slots": len(rows),
                                             "error": f"{type(exc).__name__}: {exc}", "no_retry": True})
        except OSError as receipt_error:
            print(f"post-collection halt receipt unavailable: {receipt_error}; preserve attempts/cells", file=sys.stderr)
        raise
    old_id, new_id = context["protocol"]["execution"]["actor_order"]
    pairs = summary["paired_cells"]
    new_only = sum(pair["outcomes"][new_id] == "finished" and pair["outcomes"][old_id] != "finished"
                   for pair in pairs)
    old_only = sum(pair["outcomes"][old_id] == "finished" and pair["outcomes"][new_id] != "finished"
                   for pair in pairs)
    both = sum(pair["outcomes"][old_id] == pair["outcomes"][new_id] == "finished" for pair in pairs)
    manifest = {"format": FORMAT + "-result", "status": "complete-in-sample-train-only",
                "role": "descriptive-reused-TRAIN-not-generalization-or-official-evaluation",
                "protocol_sha256": context["sha256"], "protocol_snapshot_sha256": sha256(output / "study_protocol.json"),
                "scheduled_slots": 24, "complete_slots": len(rows), "censored_slots": 0,
                "paired_roads": len(pairs), "paired_roads_denominator": 12,
                "finishes": {old_id: {"count": both + old_only, "denominator": 12},
                             new_id: {"count": both + new_only, "denominator": 12}},
                "paired_finish_table": {"both": both, "new_only": new_only, "old_only": old_only,
                                        "neither": 12 - both - new_only - old_only, "denominator": 12},
                "g0_summary": summary, "artifacts": artifacts,
                "limitations": "Actors saw these same TRAIN roads in learner data; roads match, trajectories need not. No causal learning, held-out, protected, or official claim."}
    try:
        pending = output / "manifest.json.pending"
        write_json(pending, manifest)
        final = output / "manifest.json"
        if final.exists():
            raise FileExistsError(final)
        pending.rename(final)
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException as exc:
        try:
            write_json(output / "halt.json", {"status": "post-collection-write-failure",
                                             "completed_slots": len(rows),
                                             "error": f"{type(exc).__name__}: {exc}", "no_retry": True})
        except OSError as receipt_error:
            print(f"post-collection halt receipt unavailable: {receipt_error}; preserve attempts/cells", file=sys.stderr)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("freeze", "preflight", "run"):
        command = sub.add_parser(operation)
        command.add_argument("--protocol", required=True)
        if operation != "freeze":
            command.add_argument("--protocol-sha256", required=True)
        if operation in ("freeze", "run"):
            command.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        print(json.dumps(freeze(ROOT, args.protocol, args.run_dir), sort_keys=True), flush=True)
    else:
        context = preflight(ROOT, args.protocol, args.protocol_sha256)
        if args.operation == "preflight":
            print(json.dumps({"status": "preflight-only", "in_sample_train": True,
                              "environment_resets": 0, "actors": list(context["actors"]),
                              "scheduled_slots": 24, "paired_roads": 12}, sort_keys=True), flush=True)
        else:
            os.nice(10)
            result = collect(context, args.run_dir)
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
