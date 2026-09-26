"""Collect new, TRAIN-only Dreamer P1 whole episodes from a pinned DrQ source.

This sidecar neither audits seed freshness nor freezes a study. A caller must first
provide a separately audited, SHA-pinned seed-ID receipt and a new source-pinned
protocol. No evaluation data, old teacher datasets, or generated road catalogs are
read. The explicit episode schedule is never extended to replace a failed road.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common_adapter import ActionAdapter, ActionSpec, EpisodeCollector, ObservationSpec, Transition
from drq_v2 import load_exported_actor
from haic.algorithms.drq_v2.teacher_replay import TeacherDataset, TeacherEpisode, audit_source_actor_pair
from scripts import audit_dreamerv3_p1_seeds as seed_auditor
from train import build_env


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-dreamerv3-train-support-v1"
AUDIT_FORMAT = "haic-dreamerv3-p1-cross-lane-seed-audit-v1"
AUDIT_SCOPE = "SHA-pinned ID-only protocols, prior audits, TRAIN episode ledgers and TRAIN prior collection ledgers"
FRESHNESS_LIMITATION = "no known recorded overlap; historical pilot schedules are incomplete"
REQUIRED_SOURCES = frozenset({
    "scripts/collect_dreamerv3_support.py",
    "common_adapter.py", "drq_v2.py", "train.py", "env_wrapper.py", "damage.py",
    "tracking.py", "action_smoothing.py", "action_representation.py",
    "haic/algorithms/drq_v2/teacher_replay.py", "haic/algorithms/drq_v2/teacher_study.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py", "core/finish_line.py",
    "core/track_variables.py", "core/obstacle_contacts.py",
})
AUDIT_KINDS = frozenset({"protocol", "prior_seed_audit", "training_ledger", "prior_collection_ledger"})
KNOWN_AUDIT_INPUTS = {
    "protocol": frozenset({
        "experiments/drqv2-geometry-mix-v1-r6.json",
        "experiments/drqv2-teacher-replay-v1-r3.json",
        "experiments/pixel-rlpd-entropy-target-ablation-v5.json",
        "experiments/dreamerv3-b1-terminal-positive-weight-local-v9.json",
    }),
    "prior_collection_ledger": frozenset({
        "runs/20260925-pixel-rlpd-entropy-target-ablation-v5/prior-data/collection.jsonl",
    }),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _file(root: Path, name: str, *, kind: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"{kind}: unsafe path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{kind}: path must be normalized and repository-relative")
    if kind == "source":
        allowed = name in REQUIRED_SOURCES
    elif kind == "auditor":
        allowed = len(parts) == 2 and parts[0] == "scripts" and parts[-1].startswith("audit_dreamerv3_") and name.endswith(".py")
    elif kind in ("protocol", "prior_seed_audit"):
        allowed = len(parts) == 2 and parts[0] == "experiments" and name.endswith(".json")
        allowed &= not any(token in parts[-1].lower() for token in ("result", "diagnostic", "outcome", "road"))
        allowed &= (parts[-1].endswith("-geometry-audit.json") == (kind == "prior_seed_audit"))
    elif kind == "training_ledger":
        allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "episodes.jsonl"
    elif kind == "prior_collection_ledger":
        allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "collection.jsonl"
    elif kind == "seed_audit":
        allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1] == "seed-audit.json"
    elif kind == "actor":
        allowed = len(parts) >= 3 and parts[0] == "runs" and parts[-1].endswith(".pt")
    else:
        raise ValueError(f"unsupported file kind: {kind}")
    if kind in ("training_ledger", "prior_collection_ledger", "seed_audit", "actor"):
        allowed &= not any(
            token in part.lower() for part in parts[1:-1]
            for token in ("blind", "confirm", "screen", "eval", "held-out", "holdout", "submission")
        )
    if not allowed:
        raise ValueError(f"{kind}: path is outside TRAIN-only allowlist: {name}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{kind}: symlink paths are forbidden: {name}")
    if not path.is_file():
        raise ValueError(f"{kind}: missing pinned file: {name}")
    return path


def _pinned(root: Path, name: str, expected: str, *, kind: str) -> Path:
    path = _file(root, name, kind=kind)
    if sha256(path) != _digest(expected, f"{kind} hash"):
        raise ValueError(f"{kind}: SHA-256 mismatch: {name}")
    return path


def _relative(root: Path, supplied: Path) -> str:
    path = Path(supplied)
    if not path.is_absolute():
        path = root / path
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("file must be inside the repository") from exc


def _json(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError("pinned JSON must be an object")
    return result


def _seeds(value: Any, name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{name}: expected a list of geometry seed IDs")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in value):
        raise ValueError(f"{name}: seeds must be uint32 integers")
    if len(set(value)) != len(value):
        raise ValueError(f"{name}: duplicate geometry seed")
    return value


def preflight(
    protocol_path: Path, expected_protocol_sha256: str,
    seed_audit_path: Path, expected_seed_audit_sha256: str, *, repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Verify explicit ID-only audit, sources, source actor AND checkpoint before any reset.

    The pinned auditor must independently reproduce the receipt before any reset.
    No paths are discovered or followed from audit inputs. This cannot prove
    absence of unrecorded historical roads.
    """
    root = Path(repo_root).resolve()
    protocol_name = _relative(root, protocol_path)
    if not protocol_name.startswith("experiments/dreamerv3-p1-"):
        raise ValueError("support protocol must be a new experiments/*.json file")
    protocol_file = _pinned(root, protocol_name, expected_protocol_sha256, kind="protocol")
    protocol = _json(protocol_file)
    if set(protocol) != {
        "format", "study_id", "purpose", "source_sha256", "seed_audit", "source_actor",
        "training_pool", "training_development", "partitions", "reserved_training_seeds",
        "frame_skip", "max_steps", "budgets", "spec_fingerprints",
    } or protocol["format"] != FORMAT or protocol["purpose"] != "TRAIN-only":
        raise ValueError("unsupported Dreamer TRAIN-only support protocol")
    if (not isinstance(protocol["study_id"], str) or not protocol["study_id"].startswith("dreamerv3-p1-")
            or protocol["frame_skip"] != 4 or type(protocol["frame_skip"]) is not int
            or type(protocol["max_steps"]) is not int or not 0 < protocol["max_steps"] <= 2000):
        raise ValueError("support study identity or unshaped frame/episode environment contract")
    fingerprints = protocol["spec_fingerprints"]
    if not isinstance(fingerprints, dict) or fingerprints != {
        "action": ActionSpec().fingerprint, "observation": ObservationSpec().fingerprint,
    }:
        raise ValueError("action/observation fingerprint mismatch")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != REQUIRED_SOURCES:
        raise ValueError("protocol must pin the exact collector/actor/environment executable sources")
    for name, expected in sources.items():
        _pinned(root, name, expected, kind="source")

    pool = protocol["training_pool"]
    if not isinstance(pool, dict) or set(pool) != {"cells", "episode_schedule"}:
        raise ValueError("training_pool requires explicit cells and a fixed episode schedule")
    cells = pool["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError("training_pool.cells must be nonempty")
    seen: set[int] = set()
    for cell in cells:
        if (not isinstance(cell, dict) or set(cell) != {"track_id", "geometry_seed"}
                or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)
                or type(cell["geometry_seed"]) is not int or not 0 <= cell["geometry_seed"] < 2**32
                or cell["geometry_seed"] in seen):
            raise ValueError("training cells need distinct uint32 geometry seeds on training tracks 1..4")
        seen.add(cell["geometry_seed"])
    schedule = pool["episode_schedule"]
    if (not isinstance(schedule, list) or not schedule
            or any(type(index) is not int or index not in range(len(cells)) for index in schedule)
            or set(schedule) != set(range(len(cells)))):
        raise ValueError("episode schedule must use only declared cells and attempt every cell")
    budgets = protocol["budgets"]
    if (not isinstance(budgets, dict) or set(budgets) != {"decision_cap", "minimum_distinct_finished_geometries"}
            or type(budgets["decision_cap"]) is not int or budgets["decision_cap"] < 1
            or type(budgets["minimum_distinct_finished_geometries"]) is not int
            or not 1 <= budgets["minimum_distinct_finished_geometries"] <= len(cells)
            or len(schedule) > budgets["decision_cap"]):
        raise ValueError("invalid frozen decision cap, coverage gate, or schedule")
    reserved = set(_seeds(protocol["reserved_training_seeds"], "reserved_training_seeds"))
    development = protocol["training_development"]
    if not isinstance(development, dict) or set(development) != {"seeds"}:
        raise ValueError("training_development.seeds must be explicitly excluded")
    partitions = protocol["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError("screen/confirmation/blind exclusion IDs must all be declared")
    excluded = _seeds(development["seeds"], "training_development.seeds")[:]
    for name, partition in partitions.items():
        if not isinstance(partition, dict) or set(partition) != {"seeds"}:
            raise ValueError(f"{name}.seeds must be explicitly declared")
        excluded.extend(_seeds(partition["seeds"], f"{name}.seeds"))
    if len(excluded) != len(set(excluded)) or not set(excluded) <= reserved or seen & reserved:
        raise ValueError("TRAIN pool, development, and evaluation geometry IDs must be disjoint and excluded")

    audit_spec = protocol["seed_audit"]
    if not isinstance(audit_spec, dict) or set(audit_spec) != {"path", "sha256", "sources", "auditor"}:
        raise ValueError("protocol requires a pinned seed-audit receipt and explicit source inventory")
    if (_relative(root, seed_audit_path) != audit_spec["path"]
            or _digest(expected_seed_audit_sha256, "caller audit SHA") != audit_spec["sha256"]):
        raise ValueError("caller seed-audit receipt does not match the protocol")
    audit_file = _pinned(root, audit_spec["path"], expected_seed_audit_sha256, kind="seed_audit")
    auditor = audit_spec["auditor"]
    if not isinstance(auditor, dict) or set(auditor) != {"path", "sha256"}:
        raise ValueError("seed audit must pin an independent Dreamer ID-only auditor")
    _pinned(root, auditor["path"], auditor["sha256"], kind="auditor")
    if (auditor["path"] != "scripts/audit_dreamerv3_p1_seeds.py"
            or sha256(Path(seed_auditor.__file__)) != auditor["sha256"]):
        raise ValueError("pinned seed auditor differs from the executing auditor")
    audit_sources = audit_spec["sources"]
    if not isinstance(audit_sources, dict) or set(audit_sources) != AUDIT_KINDS:
        raise ValueError("seed audit must list all four ID-only source categories")
    inventory = {}
    for kind in ("protocol", "prior_seed_audit", "training_ledger", "prior_collection_ledger"):
        declarations = audit_sources[kind]
        if not isinstance(declarations, dict) or not declarations:
            raise ValueError(f"seed audit has no {kind} source inventory")
        if not KNOWN_AUDIT_INPUTS.get(kind, frozenset()) <= set(declarations):
            raise ValueError(f"seed audit omits known cross-lane {kind} source inputs")
        for name, expected in declarations.items():
            if name in inventory:
                raise ValueError("seed audit declares the same input twice")
            _pinned(root, name, expected, kind=kind)
            inventory[name] = {"kind": kind, "path": name, "sha256": expected}
    receipt = _json(audit_file)
    seed_list = sorted(seen)
    seed_digest = hashlib.sha256(json.dumps(seed_list, separators=(",", ":")).encode("ascii")).hexdigest()
    if (receipt.get("format") != AUDIT_FORMAT or receipt.get("passed") is not True
            or receipt.get("study_id") != protocol["study_id"]
            or receipt.get("purpose") != "Dreamer P1 TRAIN-only geometry seed IDs"
            or receipt.get("inventory_complete") is not True
            or receipt.get("schema_validation") != "pass"
            or receipt.get("auditor_source_sha256") != auditor["sha256"]
            or receipt.get("proposed_seeds") != seed_list
            or type(receipt.get("proposed_seed_count")) is not int
            or receipt.get("proposed_seed_count") != len(seed_list)
            or receipt.get("proposed_seeds_sha256") != seed_digest
            or receipt.get("matched_collisions") != []
            or receipt.get("parse_errors") != []
            or receipt.get("retired_pool_collisions") != []
            or receipt.get("freshness_claim") != FRESHNESS_LIMITATION
            or receipt.get("read_scope") != AUDIT_SCOPE
            or receipt.get("blind_data_access") != "none; partition seed IDs are exclusion-only"
            or receipt.get("structural_blind_geometry_comparison") != "not performed"):
        raise ValueError("seed-audit pass receipt does not certify this exact TRAIN-only pool")
    evidence = receipt.get("source_evidence")
    paths = receipt.get("read_paths")
    if (not isinstance(evidence, list) or len(evidence) != len(inventory)
            or any(not isinstance(row, dict) or set(row) != {"kind", "path", "sha256", "schema_checked"}
                   or row.get("schema_checked") is not True
                   or not isinstance(row.get("path"), str)
                   or {key: row.get(key) for key in ("kind", "path", "sha256")} != inventory.get(row.get("path"))
                   for row in evidence)
            or len({row["path"] for row in evidence}) != len(inventory)
            or not isinstance(paths, list) or len(paths) != len(inventory)
            or any(not isinstance(path, str) for path in paths)
            or set(paths) != set(inventory)):
        raise ValueError("seed-audit evidence is incomplete or differs from pinned ID-only sources")
    try:
        reproduced = seed_auditor.audit_dreamerv3_p1_seeds(
            protocol["study_id"], cells, repo_root=root,
            protocol_sources=audit_sources["protocol"],
            prior_audit_sources=audit_sources["prior_seed_audit"],
            training_ledgers=audit_sources["training_ledger"],
            prior_collection_ledgers=audit_sources["prior_collection_ledger"],
        )
    except seed_auditor.SeedAuditError as exc:
        raise ValueError("pinned seed auditor rejected the declared ID inventory") from exc
    if reproduced != receipt or reproduced.get("passed") is not True:
        raise ValueError("seed-audit receipt was not reproduced as a passing auditor result")

    source = protocol["source_actor"]
    if not isinstance(source, dict) or set(source) != {
        "source_id", "learner_seed", "source_revision", "checkpoint_path", "checkpoint_sha256",
        "actor_path", "actor_sha256", "actor_weights_sha256",
        "training_ledger_path", "training_ledger_sha256",
    } or type(source["learner_seed"]) is not int or source["learner_seed"] not in (0, 1):
        raise ValueError("source actor/checkpoint identity is incomplete")
    if source["source_id"] != f"drq-source-{source['learner_seed']}":
        raise ValueError("source_id must identify the frozen DrQ source actor")
    if audit_sources["training_ledger"].get(source["training_ledger_path"]) != source["training_ledger_sha256"]:
        raise ValueError("source actor training ledger must be in the SHA-pinned cross-lane seed audit")
    if not isinstance(source["source_revision"], str) or not source["source_revision"].strip():
        raise ValueError("source revision must be pinned")
    checkpoint = _pinned(root, source["checkpoint_path"], source["checkpoint_sha256"], kind="actor")
    actor_file = _pinned(root, source["actor_path"], source["actor_sha256"], kind="actor")
    pair = audit_source_actor_pair(
        checkpoint, actor_file, learner_seed=source["learner_seed"],
        source_revision=source["source_revision"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )
    for field, expected in (
        ("checkpoint_sha256", source["checkpoint_sha256"]),
        ("actor_sha256", source["actor_sha256"]),
        ("actor_weights_sha256", _digest(source["actor_weights_sha256"], "actor weights hash")),
        ("action_fingerprint", fingerprints["action"]),
        ("observation_fingerprint", fingerprints["observation"]),
        ("source_revision", source["source_revision"]),
        ("learner_seed", source["learner_seed"]),
    ):
        if pair.get(field) != expected:
            raise ValueError(f"verified actor/checkpoint {field} differs from the protocol")
    if pair.get("cpu_smoke_observations") != 3:
        raise ValueError("actor/checkpoint CPU identity audit is incomplete")
    return {"protocol": protocol, "protocol_sha256": expected_protocol_sha256,
            "seed_audit_sha256": expected_seed_audit_sha256, "source_pair_audit": pair}


def collect(
    protocol_path: Path, expected_protocol_sha256: str,
    seed_audit_path: Path, expected_seed_audit_sha256: str, output_dir: Path, *,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Collect a fixed TRAIN schedule with no synthesized ends or adaptive retries."""
    root = Path(repo_root).resolve()
    output_name = _relative(root, output_dir)
    output = root / output_name
    parts = output_name.split("/")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (len(parts) < 3 or parts[0] != "runs" or any(part in ("", ".", "..") for part in parts)
            or any(token in part.lower() for part in parts[1:]
                   for token in ("blind", "confirm", "screen", "eval", "held-out", "holdout", "submission"))
            or any(root.joinpath(*parts[:n]).is_symlink() for n in range(1, len(parts)))
            or not output.parent.is_dir()):
        raise ValueError("output must be a new directory beneath an existing TRAIN run root")
    checked = preflight(
        protocol_path, expected_protocol_sha256, seed_audit_path,
        expected_seed_audit_sha256, repo_root=root,
    )
    protocol = checked["protocol"]
    source = protocol["source_actor"]
    actor, adapter, observation_spec = load_exported_actor(root / source["actor_path"], device="cpu")
    if (adapter.spec.fingerprint != protocol["spec_fingerprints"]["action"]
            or observation_spec.fingerprint != protocol["spec_fingerprints"]["observation"]):
        raise ValueError("loaded source actor contract differs from the verified checkpoint")

    output.mkdir(exist_ok=False)
    dataset = TeacherDataset()
    rows: list[dict[str, Any]] = []
    finished_geometries: set[int] = set()
    decisions = 0
    cap = protocol["budgets"]["decision_cap"]
    for attempt, index in enumerate(protocol["training_pool"]["episode_schedule"]):
        if decisions == cap:
            break
        cell = protocol["training_pool"]["cells"][index]
        env = build_env(
            cell["track_id"], cell["geometry_seed"], protocol["max_steps"],
            protocol["frame_skip"], reward_shaping=False, obstacles=True, collision_penalty=0.0,
        )
        transitions: list[Transition] = []
        proposed_actions: list[list[float]] = []
        try:
            collector = EpisodeCollector(env, action_adapter=adapter, observation_spec=observation_spec, gamma=1.0)
            observation, info = collector.reset()
            if (type(info.get("track_id")) is not int or info["track_id"] != cell["track_id"]
                    or type(info.get("seed")) is not int or info["seed"] != cell["geometry_seed"]):
                raise ValueError("environment reset did not honor its frozen TRAIN cell")
            for step in range(min(cap - decisions, protocol["max_steps"])):
                with torch.inference_mode():
                    proposed = actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0).cpu().numpy()
                proposed = ActionAdapter._validate(proposed).copy()
                transition = collector.step(proposed)
                if not isinstance(transition, Transition) or transition.episode_id != collector.episode_id or transition.step != step:
                    raise ValueError("collector did not return a contiguous genuine Transition")
                if (transition.applied_action is None
                        or not np.allclose(transition.applied_action, adapter.to_official(proposed), atol=1e-6, rtol=0)
                        or not np.allclose(transition.action, adapter.to_native(transition.applied_action, clip=False), atol=1e-6, rtol=0)):
                    raise ValueError("proposed/executed/applied action parity failed")
                if not np.array_equal(observation_spec.to_uint8(transition.observation), observation_spec.to_uint8(observation)):
                    raise ValueError("collector observation differs from preceding result frame")
                observation_spec.to_uint8(transition.next_observation)
                if (type(transition.info.get("track_id")) is not int or transition.info["track_id"] != cell["track_id"]
                        or type(transition.info.get("seed")) is not int or transition.info["seed"] != cell["geometry_seed"]):
                    raise ValueError("TRAIN geometry changed inside an episode")
                finished = transition.info.get("finished")
                if type(finished) is not bool or (finished and (not transition.done or not transition.is_terminal)):
                    raise ValueError("finished flag is absent or inconsistent with the real boundary")
                if transition.is_terminal != collector._is_terminal(
                    transition.terminated, transition.truncated, transition.info,
                ) or (transition.is_terminal and not transition.done):
                    raise ValueError("terminal flag disagrees with actual environment termination")
                for name in ("progress", "damage"):
                    value = transition.info.get(name)
                    if type(value) not in (int, float) or not np.isfinite(value):
                        raise ValueError(f"transition has no finite {name} label")
                if not np.isfinite(transition.reward) or abs(transition.reward) > np.finfo(np.float32).max:
                    raise ValueError("transition has no representable raw reward")
                transitions.append(transition)
                proposed_actions.append(proposed.tolist())
                decisions += 1
                observation = transition.next_observation
                if transition.done:
                    # Each cell gets a new collector whose local episode ID starts at zero.
                    # Rekey only that ID; all transition fields still come from real steps.
                    episode = TeacherEpisode.from_transitions(
                        [replace(row, episode_id=attempt) for row in transitions],
                        source_id=source["source_id"],
                        source_actor_sha256=source["actor_sha256"],
                        geometry_id=str(cell["geometry_seed"]), track_id=cell["track_id"],
                        metadata={"study_id": protocol["study_id"], "attempt": attempt,
                                  "collector_episode_id": transition.episode_id,
                                  "proposed_native_actions": proposed_actions},
                    )
                    dataset.add_episode(episode)
                    if finished:
                        finished_geometries.add(cell["geometry_seed"])
                    break
                if step + 1 == protocol["max_steps"]:
                    raise ValueError("environment reached max_steps without a real end or truncation")
            last = transitions[-1]
            rows.append({
                "attempt": attempt, "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                "episode_id": attempt, "collector_episode_id": last.episode_id,
                "decisions": len(transitions), "complete": last.done,
                "finished": bool(last.info["finished"]) if last.done else False,
                "terminated": last.terminated, "truncated": last.truncated,
                "terminal": last.is_terminal, "retire_reason": last.info.get("retire_reason"),
                "progress": float(last.info["progress"]), "damage": float(last.info["damage"]),
                "raw_reward_sum": float(sum(row.reward for row in transitions)),
            })
        finally:
            env.close()

    if dataset.episodes:
        digest = dataset.seal()
        with (output / "support-dataset.npz").open("xb") as stream:
            stream.write(dataset.to_bytes())
        dataset_sha = sha256(output / "support-dataset.npz")
    else:
        digest = dataset_sha = None
    minimum = protocol["budgets"]["minimum_distinct_finished_geometries"]
    result = {
        "format": "haic-dreamerv3-train-support-result-v1", "study_id": protocol["study_id"],
        "protocol_sha256": checked["protocol_sha256"],
        "seed_audit_sha256": checked["seed_audit_sha256"],
        "source_id": source["source_id"],
        "source_actor_sha256": source["actor_sha256"], "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_pair_audit": checked["source_pair_audit"],
        "decision_cap": cap, "decisions_spent": decisions,
        "stored_decisions": dataset.transition_count, "discarded_decisions": decisions - dataset.transition_count,
        "complete_episode_count": len(dataset.episodes), "episode_rows": rows,
        "schedule_attempts": len(rows), "schedule_exhausted": len(rows) == len(protocol["training_pool"]["episode_schedule"]),
        "distinct_finished_geometries": sorted(finished_geometries),
        "minimum_distinct_finished_geometries": minimum,
        "coverage_pass": len(finished_geometries) >= minimum,
        "dataset_path": "support-dataset.npz" if dataset_sha else None,
        "dataset_digest": digest, "archive_sha256": dataset_sha,
        "allowed_cells": protocol["training_pool"]["cells"],
        "excluded_seeds": protocol["reserved_training_seeds"],
    }
    with (output / "collection-result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed-audit", type=Path, required=True)
    parser.add_argument("--seed-audit-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = collect(
        args.protocol, args.protocol_sha256, args.seed_audit, args.seed_audit_sha256,
        args.output, repo_root=args.repo_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["coverage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
