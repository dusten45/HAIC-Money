"""Zero-interaction, fixed-cohort TRAIN allocation audit and immutable claims.

No simulator/model imports, outcome-based selection, replacement, or top-up.
Hashes bind individual metadata sources, not repository-wide freshness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from haic.train_seed_reservations import (
    AUDIT_FORMAT, ReservationError, reserve_train_seeds, validate_train_claim,
)
from scripts.audit_rlpd_g0_seeds import REQUIRED_COLLECTIONS, REQUIRED_LEDGERS, REQUIRED_PROTOCOLS
from scripts.audit_rlpd_g1_coverage_seeds import InventoryError, _Inventory, _digest, _json, _sha
from scripts.audit_tdmpc2_train_diag_seeds import _source_relevant

STUDY_ID = "rlpd-gate-unseen-train-v1"
FORMAT = "haic-rlpd-gate-unseen-train-audit-v1"
AUDIT_PATH = f"experiments/{STUDY_ID}-audit.json"
PROTOCOL_PATH = f"experiments/{STUDY_ID}.json"
RUN_PATH = f"runs/{STUDY_ID}"
ATTEMPT_PATH = f"experiments/{STUDY_ID}-audit-attempt-1.json"
PREVIOUS_RESOLUTION_PATH = f"experiments/{STUDY_ID}-audit-attempt-2-result.json"
PREVIOUS_RESOLUTION_SHA = "1d657e07f106fb8b6b4b1577dfb4650d2f277df3b332a527d4badb3289af9dc7"
RESOLUTION_PATH = f"experiments/{STUDY_ID}-audit-attempt-3-result.json"
ATTEMPT_SHA = "a2972a85b034fb2f37d118410e4e04b3d10cd5908898d146d7ae4ef7eace77cf"
SCOPE = "candidate-unseen TRAIN; project-global unknown legacy PPO sampling"
FRESHNESS_DEFINITION = "Unexposed to V5/gate learning, correction and selection lineage; all known explicit interactions and cross-lane protected/retired/reserved/claimed road IDs remain excluded. Not project-global never-used."
EXCEPTION_SOURCE = {"authority": "actual user message", "received_at_utc": "2026-09-30T09:36:15Z",
                    "answer": "후보 미노출 기준으로 진행", "original_question_answer_timestamp": None,
                    "timestamp_note": "Original question-answer time was not supplied; received_at_utc identifies this actual user scope instruction, not a peer approval or an invented answer time.",
                    "meaning": "candidate-unexposed criterion; unrelated historical PPO sampling uncertainty must remain disclosed"}
V5_ROOT = "runs/20260925-pixel-rlpd-entropy-target-ablation-v5"
V5_RUN = V5_ROOT + "/rlpd-author-target-seed50"
TEACHER_ROOT = "runs/20260922-drq-augmentation-pad-v1-restart"
GATE_ROOT = "runs/rlpd-local-recovery-gate-v1"
SOURCE_ACTOR_SHA = "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98"
GATE_SHA = "cf2f61101e5b2b90b44e26dca292cc9284432f89b2cf0db87e92f7cdb1501120"
REGISTRY = "experiments/train-seed-claims"
SEED_START = 2823000001
COHORT_SIZE = 64
FORBIDDEN_SEEDS = list(range(4272000001, 4272000013))
IMPLEMENTATION_SOURCES = ("scripts/audit_rlpd_gate_unseen_train.py", "haic/train_seed_reservations.py",
                          "scripts/audit_rlpd_g0_seeds.py", "scripts/audit_rlpd_g1_coverage_seeds.py",
                          "scripts/audit_tdmpc2_train_diag_seeds.py", "train.py",
                          "docs/evaluation/generalization-policy.md")
_PROTECTED = re.compile(r"blind|confirm|private|held[-_]?out|holdout|screen", re.I)
_OUTCOME = re.compile(r"(?:^|[-_])(?:result|results|summary|score|analysis|trajectory-review)(?:[-_.]|$)", re.I)
_METADATA = {"protocol.json", "protocol_spec.json", "study_protocol.json", "config.json",
             "run-config.json", "run_config.json", "catalog.json", "cells.jsonl",
             "training.jsonl", "episodes.jsonl", "collection.jsonl"}
_METADATA_MARKER = re.compile(r"reset[-_]?intent|exposure|claim|reserv|precheckpoint|superseded|partial|abort", re.I)
_COUNT_METADATA = {"candidate_seed_count", "reset_geometry_seed_count", "excluded_seed_count",
                   "known_excluded_seed_count", "known_excluded_geometry_seed_count", "proposed_seed_count",
                   "recorded_sampled_geometry_seed_count", "train_diagnostic_seed_count",
                   "train_geometry_seed_count", "reserved_training_seed_count",
                   "drq_catalog_seed_intersection_count"}


def replay_sampler_ids(track_ids: list[int], initializations: list[int], reset_bound: int,
                       excluded_seeds: list[int], candidates: set[int]) -> dict[str, Any]:
    """Replay ONLY independent NumPy road-ID draws, never a world or policy.

    This calculation cannot authenticate a producer, initialization set or count
    bound. Callers must establish those from original source/reset metadata first.
    It is never used to waive the actual legacy dirty-source HOLD.
    """
    import numpy as np
    if (not track_ids or not initializations or type(reset_bound) is not int or reset_bound < 0
            or any(type(v) is not int or not 0 <= v < 2**32
                   for v in track_ids + initializations + excluded_seeds)):
        raise InventoryError("invalid metadata-only sampler inputs")
    excluded = set(excluded_seeds)
    digest = hashlib.sha256()
    hits = []
    for initialization in sorted(set(initializations)):
        rng = np.random.default_rng(initialization)
        for index in range(reset_bound):
            track = int(rng.choice(track_ids))
            while True:
                road = int(rng.integers(0, 2**32, dtype=np.uint64))
                if road not in excluded:
                    break
            digest.update(f"{initialization}:{index}:{track}:{road}\n".encode("ascii"))
            if road in candidates:
                hits.append({"initialization": initialization, "reset_index": index,
                             "track_id": track, "geometry_seed": road})
    return {"numpy_version": np.__version__, "draw_sequence_sha256": digest.hexdigest(),
            "initializations": sorted(set(initializations)), "reset_bound_per_stream": reset_bound,
            "candidate_hits": hits, "world_resets": 0, "model_inferences": 0,
            "scope": "conditional metadata calculation, not source/count authentication or clearance"}


def _git_source(inv: _Inventory, revision: str, name: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{7,40}", revision) is None:
        raise InventoryError("legacy recorded Git source revision is unknown")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=inv.root,
            stderr=subprocess.PIPE, timeout=10).decode("ascii").strip()
        raw = subprocess.check_output(["git", "show", commit + ":" + name], cwd=inv.root,
                                      stderr=subprocess.PIPE, timeout=10)
    except (subprocess.SubprocessError, OSError, UnicodeError) as exc:
        raise InventoryError("recorded Git source is unavailable") from exc
    path = f"git:{commit}:{name}"
    inv.raw[path] = raw
    return {"path": path, "sha256": _sha(raw), "bytes": len(raw)}


def _ppo_resolution(inv: _Inventory, path: str, obj: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Classify mode FIRST. A range describing sampler capability is not allocation."""
    cfg = obj["config"]
    mode = cfg.get("training_track_mode")
    if mode not in ("fixed", "sampled"):
        raise InventoryError("legacy PPO training mode is unknown")
    tracks, roads = cfg.get("track_ids"), cfg.get("seeds")
    if (not isinstance(tracks, list) or not tracks or not isinstance(roads, list) or not roads
            or any(type(v) is not int or not 0 <= v < 2**32 for v in tracks + roads)
            or type(cfg.get("n_envs")) is not int or cfg["n_envs"] <= 0):
        raise InventoryError("legacy PPO fixed-track/seed/worker metadata is malformed")
    normalized = dict(obj)
    normalized["config"] = dict(cfg)
    normalized["config"].pop("sampled_seed_range", None)
    disposition: dict[str, Any] = {"path": path, "config_sha256": _sha(inv.read(path)),
        "training_track_mode": mode, "status": "HOLD", "reason": "source identity unresolved",
        "evidence": [], "world_resets": 0, "model_inferences": 0}
    directory = str(Path(path).parent)
    logger_path = directory + "/train.log"
    try:
        raw = inv.read(logger_path)
        text = raw.decode("utf-8")
        disposition["evidence"].append({"path": logger_path, "sha256": _sha(raw), "bytes": len(raw)})
    except (InventoryError, UnicodeError) as exc:
        disposition["reason"] = f"actual producer logger missing/unreadable: {exc}"
        return normalized, disposition
    # Parse identity and training-counter fields only, never evaluation scores.
    match = re.findall(r"^training track mode: (fixed|sampled) sampler_seed: (\d+)(?: obstacles: (\w+))?\s*$", text, re.M)
    logged_tracks = re.findall(r"^track_ids: (\[[0-9, ]+\])\s*$", text, re.M)
    logged_roads = re.findall(r"^seeds: (\[[0-9, ]+\])\s*$", text, re.M)
    worker_line = re.findall(r"^n_envs: (\d+)  total_timesteps: (\d+)\s*$", text, re.M)
    if (len(match) != 1 or match[0][0] != mode or int(match[0][1]) != cfg.get("track_sampler_seed")
            or len(logged_tracks) != 1 or json.loads(logged_tracks[0]) != tracks
            or len(logged_roads) != 1 or json.loads(logged_roads[0]) != roads
            or len(worker_line) != 1 or tuple(map(int, worker_line[0])) !=
            (cfg["n_envs"], cfg.get("total_timesteps"))):
        disposition["reason"] = "actual producer logger disagrees with config or has multiple initializations"
        return normalized, disposition
    args = shlex.split(obj.get("command_line", ""))
    for flag, expected in (("--training-track-mode", mode), ("--track-sampler-seed", str(cfg["track_sampler_seed"])),
                           ("--n-envs", str(cfg["n_envs"])),
                           ("--track-ids", ",".join(map(str, tracks))),
                           ("--seeds", ",".join(map(str, roads)))):
        if flag in args and (args.count(flag) != 1 or args.index(flag) + 1 >= len(args)
                             or args[args.index(flag) + 1] != expected):
            disposition["reason"] = f"command-line/config identity mismatch: {flag}"
            return normalized, disposition
    if mode == "fixed":
        disposition.update(status="resolved-fixed", reason="capability range inactive; explicit worker cells retained",
            declared_worker_cells=[{"track_id": tracks[i % len(tracks)], "geometry_seed": roads[i % len(roads)]}
                                   for i in range(cfg["n_envs"])])
        normalized["declared_fixed_worker_cells"] = disposition["declared_worker_cells"]
        return normalized, disposition
    numbers = [int(v) for v in re.findall(r"\|\s*total_timesteps\s*\|\s*(\d+)\s*\|", text)]
    disposition.update(track_sampler_seed=cfg["track_sampler_seed"], agent_seed=cfg.get("seed"),
        n_envs=cfg["n_envs"], requested_total_timesteps=cfg.get("total_timesteps"),
        observed_logger_max_num_timesteps=max(numbers) if numbers else None,
        requested_rollout_steps=cfg.get("n_steps"), resume_from=cfg.get("resume_from"),
        reset_count_upper_bound=None, initialization_flows_authenticated=False,
        snapshot_present=False, monitor_files=[], conditional_rng_replay_performed=False)
    root = inv.root / directory
    for entry in sorted(root.rglob("*.csv")):
        # Only metadata identity/length ledgers, never metrics/evaluation CSVs.
        if "monitor" in entry.name and not any(_PROTECTED.search(part) for part in entry.relative_to(root).parts):
            ref = entry.relative_to(inv.root).as_posix()
            monitor = inv.read(ref)
            disposition["monitor_files"].append({"path": ref, "sha256": _sha(monitor), "bytes": len(monitor)})
    snapshot = directory + "/source/train.py"
    if (inv.root / snapshot).exists():
        snapshot_raw = inv.read(snapshot)
        disposition["snapshot_present"] = True
        disposition["evidence"].append({"path": snapshot, "sha256": _sha(snapshot_raw), "bytes": len(snapshot_raw)})
        # Never execute a snapshot or accept a plausible substring as authentication.
        disposition["reason"] = "sampled source snapshot requires verified RNG initialization and reset-bound profile"
    else:
        disposition["reason"] = "sampled dirty producer has no run-local source snapshot or exact reset/monitor ledger"
    git = obj.get("git", {})
    disposition["recorded_git"] = git
    if isinstance(git, dict) and isinstance(git.get("commit"), str):
        try:
            evidence = _git_source(inv, git["commit"], "train.py")
            disposition["evidence"].append(evidence)
            source = inv.raw[evidence["path"]].decode("utf-8")
            supported = "--training-track-mode" in source and "sample_track_seed" in source
            disposition["recorded_git_supports_sampled_command"] = supported
            if not supported:
                disposition["reason"] += "; recorded Git train.py does not implement the logged sampled-mode command"
        except (InventoryError, UnicodeError) as exc:
            disposition["reason"] += f"; {exc}"
    # Requested/logged timesteps do not authenticate the missing RNG producer.
    # Do not invent a finite set of all dirty-source variants or a reset bound.
    return normalized, disposition


def _candidate_lineage(inv: _Inventory, candidates: set[int]) -> dict[str, Any]:
    """Authenticate the frozen candidate's actual producer/data/selection graph.

    Weight/data archives are SHA-hashed as bytes, never deserialized or inferred.
    Formal input/selection declarations establish unrelatedness, not architecture
    names, chronology, historical score tables, or a repository-wide hash.
    """
    evidence: dict[str, dict[str, Any]] = {}
    descriptors: list[Any] = []
    def pin(path: str, expected: str | None = None, *, archive: bool = False) -> bytes:
        if archive:
            target = inv.root
            for part in path.split("/"):
                if part in ("", ".", "..") or "\\" in part:
                    raise InventoryError("unsafe lineage archive path")
                target /= part
                if target.is_symlink():
                    raise InventoryError("symlinked lineage archive")
            with target.open("rb") as stream:
                before = os.fstat(stream.fileno())
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            current = target.stat()
            stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
            if stamp(before) != stamp(after) or stamp(before) != stamp(current):
                raise InventoryError("lineage archive changed during hash")
            size, raw = before.st_size, b""
        else:
            try:
                raw = inv.read(path)
            except InventoryError as exc:
                if str(exc).endswith(": empty source") and expected == _sha(b""):
                    inv.raw[path] = raw = b""
                else:
                    raise
            sha, size = _sha(raw), len(raw)
        if expected is not None and sha != expected:
            raise InventoryError(f"{path}: candidate lineage SHA differs")
        evidence[path] = {"path": path, "sha256": sha, "bytes": size}
        return raw
    def metadata(path: str, expected: str | None = None) -> dict[str, Any]:
        raw = pin(path, expected)
        obj = _json(raw, path)
        if any(str(seed) in raw.decode("utf-8") for seed in candidates):
            raise InventoryError(f"{path}: candidate road token occurs in lineage metadata")
        return obj
    def closure(protocol: dict[str, Any], directory: str, key: str = "source_hashes") -> None:
        mapping = protocol.get(key)
        if not isinstance(mapping, dict) or not mapping:
            raise InventoryError("candidate producer source closure is absent")
        for name, sha in sorted(mapping.items()):
            pin(directory + "/" + name, sha)

    # These failed attempts contain proposed cells, not interaction allocations.
    prior_attempt = _json(pin(ATTEMPT_PATH, ATTEMPT_SHA), ATTEMPT_PATH)
    pin(PREVIOUS_RESOLUTION_PATH, PREVIOUS_RESOLUTION_SHA)
    legacy = {row["path"]: row["sha256"] for row in prior_attempt["initial_ambiguous_sources"]}
    v5 = metadata("experiments/pixel-rlpd-entropy-target-ablation-v5.json")
    v5_sha = evidence["experiments/pixel-rlpd-entropy-target-ablation-v5.json"]["sha256"]
    cfg = metadata(V5_RUN + "/config.json")
    if (cfg.get("training_seed") != 50 or cfg.get("algorithm") != "rlpd"
            or cfg.get("protocol_sha256") != v5_sha or cfg.get("environment_steps") != 131072):
        raise InventoryError("original actor is not the frozen V5 seed50 lineage")
    pin(V5_RUN + "/study_protocol.json", v5_sha)
    closure(v5, V5_RUN + "/source")
    teacher = v5["teacher"]
    teacher_cfg = metadata(teacher["source_config_path"], teacher["source_config_sha256"])
    teacher_proto = metadata(teacher["source_protocol_path"], teacher["source_protocol_sha256"])
    tc = teacher_cfg["config"]
    if (tc.get("algorithm") != "drq-v2" or tc.get("resume_from") is not None
            or tc.get("resume_sha256") is not None or "--resume" in shlex.split(teacher_cfg["command_line"])
            or tc.get("seed") != 1 or tc.get("total_steps") != 131072
            or "start from scratch" not in teacher_proto["matched_training"]["initialization"]):
        raise InventoryError("teacher has unknown/inherited initialization lineage")
    closure(tc, TEACHER_ROOT + "/source", "training_source_sha256")
    producer = pin(TEACHER_ROOT + "/source/train_drqv2.py", tc["training_source_sha256"]["train_drqv2.py"])
    if b"agent = DrQv2Agent(config, seed=args.seed)" not in producer:
        raise InventoryError("teacher source does not match reviewed from-scratch producer")
    pin(teacher["actor_path"], teacher["actor_sha256"], archive=True)
    pin(teacher["checkpoint_path"], teacher["checkpoint_sha256"], archive=True)
    teacher_ledger = TEACHER_ROOT + "/control-seed1/episodes.jsonl"
    ledger_pin = next(row for row in v5["geometry_audit"]["teacher_source_ledgers"]
                      if row["path"] == teacher_ledger and row["source_training_seed"] == 1)
    teacher_raw = pin(teacher_ledger, ledger_pin["sha256"])
    try:
        inv.ledger(teacher_ledger)
    except InventoryError as exc:
        # The frozen teacher checkpoint ends during its last indexed road. Every
        # reset ID remains consumed; an unfinished known road is not an unknown
        # ID. This is NOT permission to ignore an unlogged reset or live producer.
        if not str(exc).endswith("unclosed final reset; no typed cap/partial receipt verified (live/incomplete use cannot be cleared)"):
            raise
        rows = [_json(line, teacher_ledger) for line in teacher_raw.splitlines()]
        if (teacher["source_checkpoint_environment_decisions"] != tc["total_steps"]
                or rows[-1].get("event") != "reset" or rows[-2].get("event") != "end"
                or type(rows[-2].get("global_step")) is not int
                or not 0 < rows[-2]["global_step"] < tc["total_steps"]
                or {r.get("seed") for r in rows if r.get("event") == "reset"}
                != set(ledger_pin["geometry_seeds"])):
            raise InventoryError("teacher final partial lacks frozen SHA-bound reset identity/cap") from exc
    metadata("experiments/drqv2-steering-logit-v1.json")  # Explicit parent exclusions/selection rules, no outcomes.
    manifest = metadata(V5_ROOT + "/prior-data/manifest.json", cfg["offline_collection_manifest_sha256"])
    if (manifest.get("dataset_sha256") != cfg["offline_dataset_sha256"]
            or manifest.get("study_protocol_sha256") != v5_sha
            or manifest.get("teacher_actor_sha256") != teacher["actor_sha256"]
            or manifest.get("teacher_checkpoint_sha256") != teacher["checkpoint_sha256"]):
        raise InventoryError("V5 prior has an unknown teacher or data parent")
    declared = {(c["track_id"], c["geometry_seed"]) for c in v5["teacher_data_cells"]}
    if not manifest.get("episodes") or any((row["track_id"], row["geometry_seed"]) not in declared
            or row["teacher_actor_sha256"] != teacher["actor_sha256"] for row in manifest["episodes"]):
        raise InventoryError("prior episode ancestry differs from frozen teacher pool")
    for row in manifest["episodes"]:
        pin(V5_ROOT + "/prior-data/" + row["path"], row["sha256"], archive=True)
    inv.ledger(V5_ROOT + "/prior-data/collection.jsonl", collection=True)
    inv.ledger(V5_RUN + "/episodes.jsonl")
    frozen = metadata(V5_RUN + "/frozen_candidates.json")
    last = frozen["candidates"][-1]
    if last.get("actor_sha256") != SOURCE_ACTOR_SHA or last.get("training_seed") != 50:
        raise InventoryError("original actor identity is not V5 seed50 final checkpoint")
    pin(V5_RUN + "/" + last["actor_path"], SOURCE_ACTOR_SHA, archive=True)
    pin(V5_RUN + "/" + last["checkpoint_path"], last["checkpoint_sha256"], archive=True)
    gate = metadata(GATE_ROOT + "/protocol.json")
    if gate != metadata("experiments/rlpd-local-recovery-gate-v1.json"):
        raise InventoryError("gate protocol copies differ")
    if (gate["source_actor"]["sha256"] != SOURCE_ACTOR_SHA
            or gate["source_actor"]["path"] != V5_RUN + "/" + last["actor_path"]):
        raise InventoryError("gate original actor ancestry differs")
    closure(gate, GATE_ROOT + "/source")
    pin(GATE_ROOT + "/gate.pt", GATE_SHA, archive=True)
    repair = metadata("experiments/rlpd-recovery-actor-only-v1.json")
    if (repair["source_checkpoint"] != V5_RUN + "/" + last["checkpoint_path"]
            or repair["source_checkpoint_sha256"] != last["checkpoint_sha256"]
            or repair["inputs"] != gate["inputs"]):
        raise InventoryError("correction/source/prior/prototype parent differs")
    closure(repair, "runs/rlpd-recovery-actor-only-v1/source")
    pin(gate["corrected_actor"]["path"], gate["corrected_actor"]["sha256"], archive=True)
    metadata(gate["repair_receipt"]["path"], gate["repair_receipt"]["sha256"])
    recovery = metadata("runs/rlpd-recovery-training-data-v1/manifest.json", gate["inputs"]["dataset_manifest_sha256"])
    metadata("runs/rlpd-coupled-recovery-r2/manifest.json", recovery["source_manifest_sha256"])
    if not recovery.get("episodes") or any(row.get("track_id") != 1
            or row.get("geometry_seed") not in FORBIDDEN_SEEDS for row in recovery["episodes"]):
        raise InventoryError("correction fit includes an unknown/non-G0 geometry")
    for row in recovery["episodes"]:
        pin("runs/rlpd-recovery-training-data-v1/" + row["path"], row["sha256"], archive=True)
    g0 = metadata("experiments/rlpd-g0-completion-v1.json")
    if {c["geometry_seed"] for c in g0["cells"]} != set(FORBIDDEN_SEEDS):
        raise InventoryError("G0 calibration/selection allocation changed")
    for name in ("rlpd-recovery-learning-v1", "rlpd-recovery-guided-learning-v1", "rlpd-coupled-recovery-r2"):
        descriptors.append(metadata(f"experiments/{name}.json"))
    protected = gate["inputs"]["protected"]
    if {c["geometry_seed"] for c in protected["sources"]} != set(FORBIDDEN_SEEDS):
        raise InventoryError("protected correction input is not the exact consumed G0 cohort")
    preservation = metadata("runs/rlpd-recovery-evaluation-r2/manifest.json", protected["manifest_sha256"])
    for row in protected["sources"]:
        name = f"v5-seed-{row['geometry_seed']}"
        files = preservation["files_sha256"]
        if files.get(name + "-receipt.json") != row["receipt_sha256"] or files.get(name + ".npz") != row["trace_sha256"]:
            raise InventoryError("protected calibration source is not in the frozen TRAIN manifest")
        metadata("runs/rlpd-recovery-evaluation-r2/" + name + "-receipt.json", row["receipt_sha256"])
        pin("runs/rlpd-recovery-evaluation-r2/" + name + ".npz", row["trace_sha256"], archive=True)
    for run in ("rlpd-recovery-evaluation-r2", "rlpd-recovery-guided-evaluation-v1",
                "rlpd-recovery-actor-only-evaluation-v1", "rlpd-local-recovery-gate-evaluation-v1",
                "rlpd-local-recovery-gate-evaluation-repeat-v1"):
        descriptors.append(metadata(f"runs/{run}/protocol.json"))
    # The G0 selection comparator has the same independent DrQ teacher, not PPO.
    comparator = metadata("experiments/pixel-rlpd-long-horizon-followup-v1.json")
    if comparator["teacher"]["actor_sha256"] != teacher["actor_sha256"]:
        raise InventoryError("selection comparator has an unknown teacher ancestor")
    comparator_actor = next(actor for actor in g0["actors"] if actor["id"] == "long-horizon-seed11")
    comparator_dir = str(Path(comparator_actor["candidate_path"]).parents[2])
    comparator_cfg = metadata(comparator_dir + "/config.json")
    comparator_sha = evidence["experiments/pixel-rlpd-long-horizon-followup-v1.json"]["sha256"]
    if (comparator_cfg.get("protocol_sha256") != comparator_sha or comparator_cfg.get("training_seed") != 11
            or "no model weights or transitions resumed" not in comparator["source"]["port"]):
        raise InventoryError("G0 correction/selection comparator initialization is unknown")
    closure(comparator, comparator_dir + "/source")
    comparator_prior = str(Path(comparator_dir).parent) + "/prior-data"
    comparator_manifest = metadata(comparator_prior + "/manifest.json", comparator_cfg["offline_collection_manifest_sha256"])
    if (comparator_manifest["teacher_actor_sha256"] != teacher["actor_sha256"]
            or comparator_manifest["dataset_sha256"] != comparator_cfg["offline_dataset_sha256"]):
        raise InventoryError("comparator prior parent is unknown")
    comparator_cells = {(c["track_id"], c["geometry_seed"]) for c in comparator["teacher_data_cells"]}
    if any((r["track_id"], r["geometry_seed"]) not in comparator_cells for r in comparator_manifest["episodes"]):
        raise InventoryError("comparator prior contains undeclared roads")
    for row in comparator_manifest["episodes"]:
        pin(comparator_prior + "/" + row["path"], row["sha256"], archive=True)
    inv.ledger(comparator_dir + "/episodes.jsonl")
    inv.ledger(comparator_prior + "/collection.jsonl", collection=True)
    pin(comparator_actor["candidate_path"], comparator_actor["candidate_sha256"])
    pin(comparator_actor["path"], comparator_actor["sha256"], archive=True)
    descriptors += [v5["teacher"], v5["student_training"], cfg, teacher_proto["matched_training"],
                    {k: tc[k] for k in ("algorithm", "resume_from", "resume_sha256", "drq_config")},
                    manifest, gate["inputs"], repair["inputs"], recovery, g0["actors"], comparator["teacher"],
                    comparator_manifest, comparator_cfg]
    serialized = json.dumps(descriptors, sort_keys=True)
    related = [path for path in legacy if str(Path(path).parent) in serialized]
    if related:
        raise InventoryError("legacy PPO artifact/outcome is in candidate lineage: " + ",".join(related))
    if set(inv.ids) & candidates:
        raise InventoryError("candidate road was exposed to teacher/student TRAIN ledger")
    for path in (TEACHER_ROOT + "/control-seed1/episodes.jsonl", V5_ROOT + "/prior-data/collection.jsonl",
                 V5_RUN + "/episodes.jsonl", comparator_dir + "/episodes.jsonl", comparator_prior + "/collection.jsonl"):
        pin(path)
    why = "Frozen DrQ teacher starts from scratch with null resume state; fresh V5 teacher-only prior and V5 seed50 learner feed source/correction/prototypes; formal G0 and DrQ/V5 selection inputs do not reference any of these 15 PPO artifacts, RGB buffers, stats, weights or outcomes."
    return {"status": "verified", "candidate": {"source_actor_sha256": SOURCE_ACTOR_SHA, "gate_sha256": GATE_SHA},
            "sources": sorted(evidence.values(), key=lambda row: row["path"]),
            "lineage_source_pins_sha256": _digest(sorted(evidence.values(), key=lambda row: row["path"])),
            "known_teacher_student_road_ids": sorted(inv.ids), "related_legacy_paths": [],
            "unrelated_legacy_config_pins": legacy, "why_unrelated": why,
            "limitation": "Formal artifact/data/selection lineage, not a certificate of all human research influences or project-global historical non-use."}


def _scoped_source(receipt: dict[str, Any]) -> str:
    keys = ("sources", "freshness_definition", "scope", "exception_source", "legacy_uncertainties",
            "candidate_lineage_evidence", "all_known_exclusions", "warnings", "freshness_scope",
            "project_global_unseen_certified")
    try:
        return "candidate-unseen-TRAIN/source-bound-scoped-evidence-sha256:" + _digest({key: receipt[key] for key in keys})
    except (KeyError, TypeError, ValueError) as exc:
        raise InventoryError("missing or malformed scoped claim evidence") from exc


def _identity_metadata(value: Any) -> Any:
    """Do not interpret reviewed cardinality statistics as unbounded seed ranges."""
    if isinstance(value, list):
        return [_identity_metadata(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _identity_metadata(item) for key, item in value.items()}
    for key in _COUNT_METADATA:
        item = result.get(key)
        if type(item) is int and item >= 0:
            result.pop(key)
    if (type(result.get("geometry_seed_count")) is int and isinstance(result.get("geometry_seeds"), list)
            and result["geometry_seed_count"] == len(result["geometry_seeds"])):
        result.pop("geometry_seed_count")
    candidates = result.get("candidate_seeds")
    start = result.get("seed_start")
    if (isinstance(candidates, list) and candidates and type(start) is int
            and candidates == list(range(start, start + len(candidates)))):
        result.pop("seed_start")
    return result


def _self_metadata(inv: _Inventory, path: str, raw: bytes, receipt: dict[str, Any],
                   protocol_sha: str | None) -> None:
    """Authenticate only producer protocol/reset identities, never episode outcomes."""
    protocol_raw = inv.read(PROTOCOL_PATH)
    protocol = _json(protocol_raw, PROTOCOL_PATH)
    audit_raw = inv.read(AUDIT_PATH)
    if (protocol_sha is None or _sha(protocol_raw) != protocol_sha
            or protocol.get("format") != "haic-rlpd-gate-unseen-train-v1"
            or protocol.get("study_id") != STUDY_ID or protocol.get("cells") != proposed_cells()
            or _json(audit_raw, AUDIT_PATH) != receipt
            or protocol.get("freshness", {}).get("audit_receipt") !=
            {"path": AUDIT_PATH, "sha256": _sha(audit_raw)}):
        raise InventoryError("self protocol/audit receipt identity chain differs")
    freshness = protocol["freshness"]
    definition = freshness.get("definition", {})
    if (not isinstance(definition, dict) or definition.get("scope") != "candidate-lineage-unseen"
            or definition.get("project_global_unseen_certified") is not False
            or definition.get("candidate_lineage") != ["V5-source-learning", "gate-learning-and-calibration",
                                                       "correction-learning", "candidate-selection"]
            or freshness.get("legacy_warnings") != receipt["warnings"]
            or freshness.get("legacy_warning_pin") != {
                "audit_receipt_sha256": _sha(audit_raw), "warnings_sha256": _digest(receipt["warnings"]),
                "legacy_ppo_dispositions_sha256": _digest(receipt["legacy_ppo_dispositions"])}):
        raise InventoryError("self protocol scope or legacy-warning evidence differs")
    if path in (PROTOCOL_PATH, RUN_PATH + "/protocol.json"):
        if _sha(raw) != protocol_sha:
            raise InventoryError("copied self protocol differs")
        return
    ledger_path = RUN_PATH + "/reset-intents.jsonl"
    ledger_raw = inv.read(ledger_path)
    if not ledger_raw.endswith(b"\n"):
        raise InventoryError("partial self reset-intent row")
    rows = [_json(line, f"{ledger_path}:{index}") for index, line in enumerate(ledger_raw.splitlines(), 1)]
    keys = {"slot", "track_id", "geometry_seed", "actor_id", "freshness", "study_id",
            "protocol_sha256", "protocol_path", "audit_receipt_sha256", "case", "actor",
            "no_retry", "partition", "obstacles"}
    if len(rows) > 128:
        raise InventoryError("self study exceeds fixed 128 reset intents")
    for slot, row in enumerate(rows):
        cell = proposed_cells()[slot // 2]
        actor_id = "original" if slot % 2 == 0 else "local-gate"
        case = row.get("case")
        if (set(row) != keys or type(row.get("slot")) is not int or row["slot"] != slot
                or any(row.get(key) != value for key, value in cell.items())
                or row.get("actor_id") != actor_id or row.get("study_id") != STUDY_ID
                or row.get("protocol_sha256") != protocol_sha or row.get("protocol_path") != PROTOCOL_PATH
                or row.get("audit_receipt_sha256") != _sha(audit_raw)
                or row.get("freshness") != ("first-unseen" if slot % 2 == 0 else "paired-self-study")
                or row.get("actor") != protocol.get("actors", {}).get(actor_id)
                or row.get("no_retry") is not True or not isinstance(case, dict)
                or set(case) not in ({"track_id", "geometry_seed", "max_steps"},
                                     {"track_id", "geometry_seed", "max_steps", "source_road_centerline_sha256"})
                or case.get("track_id") != 1 or case.get("geometry_seed") != cell["geometry_seed"]
                or type(case.get("max_steps")) is not int or case["max_steps"] != 2000):
            raise InventoryError(f"self reset-intent slot {slot}: producer identity/condition mismatch")
        if "source_road_centerline_sha256" in case and (
                actor_id != "local-gate" or not isinstance(case["source_road_centerline_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", case["source_road_centerline_sha256"])):
            raise InventoryError("invalid paired road hash metadata")
    if path != ledger_path:
        row = _json(raw, path)
        slot = row.get("slot")
        if type(slot) is not int or slot < 0 or slot >= len(rows) or row != rows[slot]:
            raise InventoryError("self reset-intent sidecar not authenticated by ledger")
        expected = f"{RUN_PATH}/slot-{slot:03d}-{row['actor_id']}-seed-{row['geometry_seed']}-reset-intent.json"
        if path != expected:
            raise InventoryError("self reset-intent sidecar filename differs")


def _relevant(raw: bytes, path: str, seeds: set[int], token: re.Pattern[str]) -> bool:
    if path.endswith(".jsonl"):
        return any(_relevant(line, path + f":{index}", seeds, token)
                   for index, line in enumerate(raw.splitlines(), 1))
    try:
        data = _identity_metadata(_json(raw, path))
        normalized = json.dumps(data, sort_keys=True).encode("ascii")
    except InventoryError:
        normalized = raw
    return _source_relevant(normalized, path, seeds, token)


def _catalog_counts(value: Any, inv: _Inventory) -> Any:
    """A pinned catalog cardinality is not a road interval; audit its actual IDs."""
    if isinstance(value, list):
        return [_catalog_counts(item, inv) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _catalog_counts(item, inv) for key, item in value.items()}
    if "geometry_seed_count" in result and "catalog_path" in result and "catalog_sha256" in result:
        path = result["catalog_path"]
        if not isinstance(path, str):
            raise InventoryError("catalog metadata path is not text")
        raw = inv.read(path)
        catalog = _json(raw, path)
        if _sha(raw) != result["catalog_sha256"]:
            raise InventoryError("catalog cardinality source SHA differs")
        rows = catalog.get("train")
        if (not isinstance(rows, list) or type(result["geometry_seed_count"]) is not int
                or len(rows) != result["geometry_seed_count"]
                or any(not isinstance(row, dict) or type(row.get("geometry_seed")) is not int for row in rows)):
            raise InventoryError("catalog cardinality/source IDs differ")
        result.pop("geometry_seed_count")
    return result


def proposed_cells() -> list[dict[str, Any]]:
    return [{"partition": "TRAIN", "track_id": 1, "geometry_seed": seed, "obstacles": True}
            for seed in range(SEED_START, SEED_START + COHORT_SIZE)]


def _discover(root: Path) -> list[str]:
    """Enumerate metadata only; protected directories expose protocols, never outcomes."""
    paths: set[str] = set()
    experiments = root / "experiments"
    if experiments.is_symlink() or not experiments.is_dir():
        raise InventoryError("experiments: missing or unsafe directory")
    for path in experiments.iterdir():
        if path.suffix in (".json", ".invalid-frozen") and path.as_posix() != (root / AUDIT_PATH).as_posix():
            if not _OUTCOME.search(path.name):
                paths.add(path.relative_to(root).as_posix())
    for base in ("runs", "evaluations", "transferred/runs"):
        directory = root / base
        if not directory.exists() and base != "runs":
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise InventoryError(f"{base}: missing or unsafe directory")
        def onerror(exc: OSError) -> None:
            raise InventoryError(f"{base}: discovery failed: {exc}")
        for current, dirs, files in os.walk(directory, followlinks=False, onerror=onerror):
            location = Path(current)
            relative_dir = location.relative_to(root).as_posix()
            protected = base == "evaluations" or bool(_PROTECTED.search(relative_dir))
            for name in list(dirs):
                child = location / name
                if child.is_symlink():
                    raise InventoryError(f"{child.relative_to(root)}: symlinked evidence directory")
                if name in ("source", "traces", "episodes", "candidates", "checkpoints", "runtime", "__pycache__"):
                    dirs.remove(name)
            for name in files:
                if not name.endswith((".json", ".jsonl", ".invalid-frozen")):
                    continue
                if protected:
                    eligible = name in ("protocol.json", "protocol_spec.json", "study_protocol.json")
                else:
                    eligible = name in _METADATA or bool(_METADATA_MARKER.search(name))
                if relative_dir == RUN_PATH:
                    eligible = name == "protocol.json" or "reset-intent" in name
                if eligible:
                    paths.add((location / name).relative_to(root).as_posix())
    registry = root / REGISTRY
    if registry.is_symlink() or not registry.is_dir():
        raise InventoryError("shared TRAIN registry missing or unsafe")
    for path in registry.iterdir():
        if path.name != ".gitkeep":
            paths.add(path.relative_to(root).as_posix())
    return sorted(paths)


def verify_claims(receipt: dict[str, Any], *, root: Path = Path(".")) -> list[dict[str, Any]]:
    """Authenticate own reservations against the original under-lock audit contract."""
    if (receipt.get("format") != FORMAT or receipt.get("study_id") != STUDY_ID
            or receipt.get("cells") != proposed_cells() or receipt.get("status") != "clear"
            or receipt.get("collisions") != [] or receipt.get("blockers") != []
            or receipt.get("zero_interaction") is not True
            or receipt.get("scope") != SCOPE or receipt.get("freshness_definition") != FRESHNESS_DEFINITION
            or receipt.get("exception_source") != EXCEPTION_SOURCE
            or receipt.get("project_global_never_used_claim") is not False
            or receipt.get("project_global_unseen_certified") is not False
            or receipt.get("freshness_scope") != "candidate-lineage-unseen"
            or any(row not in receipt.get("warnings", []) for row in receipt.get("legacy_uncertainties", []))):
        raise InventoryError("invalid or blocked cohort receipt")
    contract = receipt.get("claim_audit")
    if (not isinstance(contract, dict) or contract.get("format") != AUDIT_FORMAT
            or contract.get("cells") != proposed_cells() or contract.get("status") != "clear"
            or any(contract.get(key) != [] for key in
                   ("collisions", "blockers", "consumed_seeds", "reserved_seeds"))
            or contract.get("source") != _scoped_source(receipt)):
        raise InventoryError("claim audit evidence chain differs")
    inventory = _Inventory(Path(root).absolute())
    expected = receipt.get("claims")
    if not isinstance(expected, list) or len(expected) != COHORT_SIZE:
        raise InventoryError("require exactly 64 authenticated claims")
    records = []
    for cell, pin in zip(proposed_cells(), expected):
        seed = cell["geometry_seed"]
        path = f"{REGISTRY}/seed-{seed}.json"
        raw = inventory.read(path)
        claim = _json(raw, path)
        validate_train_claim(claim, seed)
        if (pin != {"path": path, "sha256": _sha(raw)} or claim["study_id"] != STUDY_ID
                or claim["audit_digest_sha256"] != _digest(contract)
                or claim["audit_source"] != contract["source"]
                or any(claim[key] != value for key, value in cell.items())
                or claim["status"] != "reserved"
                or claim["protocol_path"] is not None or claim["protocol_sha256"] is not None):
            raise InventoryError(f"{path}: self-claim chain differs")
        records.append(claim)
    return records


def load_audit(path: Path | str, expected_sha256: str, *, root: Path = Path(".")) -> dict[str, Any]:
    """A caller must pin receipt bytes in its frozen protocol, not trust a study name."""
    inventory = _Inventory(Path(root).absolute())
    relative = Path(path).relative_to(Path(root).absolute()).as_posix() if Path(path).is_absolute() else str(path)
    raw = inventory.read(relative)
    if _sha(raw) != expected_sha256:
        raise InventoryError("audit receipt SHA differs")
    receipt = _json(raw, relative)
    verify_claims(receipt, root=root)
    return receipt


def audit(root: Path = Path("."), *, authenticated_receipt: dict[str, Any] | None = None,
          self_protocol_sha256: str | None = None,
          required_sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Candidate-only recheck. Own records are exempt only with a claim/protocol chain."""
    root = Path(root).absolute()
    inv = _Inventory(root)
    cells = proposed_cells()
    seeds = {cell["geometry_seed"] for cell in cells}
    token = re.compile(r"(?<!\d)(?:" + "|".join(map(str, sorted(seeds))) + r")(?!\d)")
    collisions: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, str]] = []
    ppo_dispositions: list[dict[str, Any]] = []
    legacy_uncertainties: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {"status": "not-required-in-synthetic-empty-lineage", "sources": []}
    self_claim_paths: set[str] = set()
    actual_sources = required_sources is None
    if authenticated_receipt is not None:
        verify_claims(authenticated_receipt, root=root)
        self_claim_paths = {pin["path"] for pin in authenticated_receipt["claims"]}
    if required_sources is None:
        required_sources = tuple(REQUIRED_PROTOCOLS) + tuple(REQUIRED_LEDGERS) + tuple(REQUIRED_COLLECTIONS)
    elif root == Path(__file__).resolve().parents[1]:
        raise InventoryError("source-list override is synthetic-only")
    try:
        discovered = _discover(root)
    except (InventoryError, OSError) as exc:
        discovered = []
        blockers.append({"path": ".", "reason": str(exc)})
    for path in sorted(set(discovered) | set(required_sources)):
        try:
            try:
                raw = inv.read(path)
            except InventoryError as exc:
                # An empty optional ledger contains no identity, but is not evidence
                # of zero interaction: its other allocation/reset metadata still applies.
                if (str(exc).endswith(": empty source") and path.endswith(".jsonl")
                        and path not in required_sources):
                    raw = b""
                    inv.raw[path] = raw
                    warnings.append({"path": path, "reason": "empty optional ledger; no zero-use inference"})
                else:
                    raise
            if path in self_claim_paths:
                continue
            if path == ATTEMPT_PATH:
                if _sha(raw) != ATTEMPT_SHA:
                    raise InventoryError("preserved initial zero-interaction HOLD receipt differs")
                continue  # Proposed-only failed attempt, never an allocation/interaction waiver.
            if path == PROTOCOL_PATH and authenticated_receipt is not None:
                _self_metadata(inv, path, raw, authenticated_receipt, self_protocol_sha256)
                continue
            if path.startswith(RUN_PATH + "/") and authenticated_receipt is not None:
                _self_metadata(inv, path, raw, authenticated_receipt, self_protocol_sha256)
                continue
            if path.startswith(REGISTRY + "/"):
                match = re.fullmatch(r"seed-(\d+)\.json", Path(path).name)
                if match is None:
                    raise InventoryError("unknown registry entry")
                validate_train_claim(_json(raw, path), int(match[1]))
            normalized = raw
            if not path.endswith(".jsonl"):
                try:
                    metadata = _catalog_counts(_json(raw, path), inv)
                    cfg = metadata.get("config")
                    if (path.startswith("runs/") and Path(path).name == "config.json"
                            and isinstance(cfg, dict) and cfg.get("algorithm") == "PPO"
                            and "sampled_seed_range" in cfg):
                        metadata, disposition = _ppo_resolution(inv, path, metadata)
                        ppo_dispositions.append(disposition)
                    normalized = json.dumps(metadata).encode("ascii")
                except InventoryError as exc:
                    if "catalog" in str(exc) or "legacy PPO" in str(exc):
                        raise
            if _relevant(normalized, path, seeds, token):
                issue = {"path": path, "sha256": _sha(raw)}
                if token.search(normalized.decode("utf-8", errors="replace")):
                    collisions.append({**issue, "reason": "candidate-relevant recorded road identity"})
                else:
                    blockers.append({**issue, "reason": "candidate-possible road interval or unknown identity; disjointness unproven"})
        except (InventoryError, ReservationError, OSError) as exc:
            blockers.append({"path": path, "reason": str(exc)})
    if actual_sources or any(row["status"] == "HOLD" for row in ppo_dispositions):
        try:
            lineage = _candidate_lineage(inv, seeds)
        except (InventoryError, OSError, KeyError, TypeError) as exc:
            lineage = {"status": "HOLD", "sources": [], "reason": str(exc)}
            blockers.append({"path": "candidate-lineage", "reason": str(exc)})
    for disposition in ppo_dispositions:
        if disposition["status"] != "HOLD":
            continue
        path = disposition["path"]
        if (lineage.get("status") == "verified" and path not in lineage.get("related_legacy_paths", [])
                and lineage.get("unrelated_legacy_config_pins", {}).get(path) == disposition["config_sha256"]):
            legacy_uncertainties.append({**disposition, "status": "WARN-unrelated-legacy-uncertainty",
                "why_unrelated": lineage["why_unrelated"],
                "lineage_source_pins_sha256": lineage["lineage_source_pins_sha256"],
                "project_global_non_use_proven": False})
        else:
            blockers.append({"path": path, "sha256": disposition["config_sha256"],
                             "reason": "candidate-related or unclassified legacy uncertainty: " + disposition["reason"]})
    warnings.extend(legacy_uncertainties)
    if actual_sources:
        for path in IMPLEMENTATION_SOURCES:
            try:
                inv.read(path)
            except InventoryError as exc:
                blockers.append({"path": path, "reason": str(exc)})
    sources = [{"path": path, "sha256": _sha(raw), "bytes": len(raw)}
               for path, raw in sorted(inv.raw.items()) if path not in self_claim_paths
               and path != PROTOCOL_PATH and not path.startswith(RUN_PATH + "/")]
    combined_sources = {row["path"]: row for row in sources}
    combined_sources.update({row["path"]: row for row in lineage["sources"]})
    sources = sorted(combined_sources.values(), key=lambda row: row["path"])
    all_known_exclusions = {"policy": "All explicit interaction, allocation, retired/protected/reserved and claimed road IDs across every lane remain hard exclusions regardless of candidate ancestry.",
                           "cross_track_seed_exclusions": True, "checked_source_count": len(sources),
                           "source_pins_sha256": _digest(sources), "candidate_matches": collisions,
                           "blind_observation_reads": 0}
    return {"format": FORMAT, "study_id": STUDY_ID, "partition": "TRAIN", "cells": cells,
            "status": "clear" if not collisions and not blockers else "HOLD",
            "collisions": collisions, "blockers": blockers, "warnings": warnings, "sources": sources,
            "legacy_ppo_dispositions": ppo_dispositions,
            "legacy_uncertainties": legacy_uncertainties, "candidate_lineage_evidence": lineage,
            "freshness_definition": FRESHNESS_DEFINITION, "exception_source": EXCEPTION_SOURCE,
            "project_global_never_used_claim": False, "all_known_exclusions": all_known_exclusions,
            "freshness_scope": "candidate-lineage-unseen", "project_global_unseen_certified": False,
            "previous_attempt": {"path": ATTEMPT_PATH, "sha256": ATTEMPT_SHA} if ATTEMPT_PATH in inv.raw else None,
            "source_count": len(sources), "blind_observation_reads": 0, "zero_interaction": True,
            "auditor_environment_resets": 0, "auditor_optimizer_updates": 0,
            "forbidden_consumed_g0_seeds": FORBIDDEN_SEEDS,
            "cohort_rule": "fixed 64; no replacement/top-up/tuning/early-stop; finish all unless critical infrastructure/provenance failure",
            "scope": SCOPE,
            "limitation": "Recorded candidate-specific metadata clearance, not proof of unrecorded global non-use; >2^31 alone is not freshness",
            "claims": [], "claim_audit": None}


def audit_cell(receipt: dict[str, Any], cell: dict[str, Any], *, root: Path = Path("."),
               self_protocol_sha256: str | None = None) -> dict[str, Any]:
    if cell not in proposed_cells() or set(cell) != set(proposed_cells()[0]):
        raise InventoryError("cell is outside fixed TRAIN cohort/conditions")
    report = audit(root, authenticated_receipt=receipt, self_protocol_sha256=self_protocol_sha256)
    if report["status"] != "clear":
        raise InventoryError("pre-reset candidate recheck HOLD: " + json.dumps(
            {key: report[key] for key in ("collisions", "blockers")}, sort_keys=True))
    return report


def claim(root: Path = Path("."), *, required_sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Re-audit under the shared registry lock, then reserve all 64 without rollback."""
    report: dict[str, Any] = {}
    def recheck(immutable_cells: Any) -> dict[str, Any]:
        nonlocal report
        report = audit(root, required_sources=required_sources)
        if list(map(dict, immutable_cells)) != report["cells"] or report["status"] != "clear":
            raise InventoryError("under-lock audit HOLD: " + json.dumps(report["collisions"] + report["blockers"]))
        contract = {"format": AUDIT_FORMAT, "partition": "TRAIN", "status": "clear",
                    "cells": report["cells"], "collisions": [], "blockers": [],
                    "consumed_seeds": [], "reserved_seeds": [],
                    "source": _scoped_source(report)}
        report["claim_audit"] = contract
        return contract
    reserve_train_seeds(Path(root) / REGISTRY, proposed_cells(), recheck, study_id=STUDY_ID)
    inv = _Inventory(Path(root).absolute())
    report["claims"] = [{"path": path, "sha256": _sha(inv.read(path))} for path in
                        (f"{REGISTRY}/seed-{cell['geometry_seed']}.json" for cell in proposed_cells())]
    verify_claims(report, root=root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--claim", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attempt-output", action="store_true",
                        help="preserve the metadata-only resolution report in a distinct attempt receipt")
    args = parser.parse_args()
    if args.attempt_output:
        if args.claim or args.output is not None:
            parser.error("attempt output is read-only and cannot be combined with claims or --output")
        args.output = Path(RESOLUTION_PATH)
    if args.claim and args.output is None:
        args.output = Path(AUDIT_PATH)
    if args.output is not None and args.output.as_posix() not in (AUDIT_PATH, RESOLUTION_PATH):
        parser.error("only the owned audit receipt path may be written")
    if args.output is not None:
        target = args.root / args.output
        if target.exists() or target.is_symlink() or target.parent.is_symlink():
            parser.error("audit output exists or is unsafe; do not claim an unrecordable batch")
    report = claim(args.root) if args.claim else audit(args.root)
    raw = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    if args.output is not None:
        path = args.root / args.output
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "cells": len(report["cells"]),
                      "source_count": report["source_count"], "claims": len(report["claims"]),
                      "collisions": report["collisions"], "blockers": report["blockers"],
                      "receipt_sha256": _sha(raw)}, sort_keys=True))
    return 0 if report["status"] == "clear" else 2


if __name__ == "__main__":
    raise SystemExit(main())
