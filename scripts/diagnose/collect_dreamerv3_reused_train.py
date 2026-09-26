"""Collect a source-pinned, explicitly REUSED r6 TRAIN engineering diagnostic.

This is not the fresh Dreamer P1 collector or an evaluation. Nothing runs on import;
collection requires a new caller-supplied, SHA-pinned protocol and output directory.
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
from train import build_env


ROOT = Path(__file__).resolve().parents[2]
FORMAT = "haic-dreamerv3-reused-train-diagnostic-v1"
PURPOSE = "reused-TRAIN-engineering-diagnostic"
LIMITATION = "reused r6 TRAIN cells; not fresh P1/P1b, evaluation, or promotion"
R6_PATH = "experiments/drqv2-geometry-mix-v1-r6.json"
R6_SHA256 = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
COLLECTOR_PATH = "scripts/diagnose/collect_dreamerv3_reused_train.py"
REQUIRED_SOURCES = frozenset({
    COLLECTOR_PATH, "common_adapter.py", "drq_v2.py", "train.py", "env_wrapper.py",
    "damage.py", "tracking.py", "action_smoothing.py", "action_representation.py",
    "haic/algorithms/drq_v2/teacher_replay.py", "haic/algorithms/drq_v2/teacher_study.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py", "core/finish_line.py",
    "core/track_variables.py", "core/obstacle_contacts.py",
})
FORBIDDEN_OUTPUT_TOKENS = ("blind", "confirm", "screen", "eval", "held-out", "holdout", "submission")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _relative(root: Path, supplied: Path | str) -> str:
    path = Path(supplied)
    if not path.is_absolute():
        path = root / path
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("path must be repository-relative") from exc


def _path(root: Path, name: str, *, kind: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"{kind}: unsafe path")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{kind}: path is not normalized")
    if kind == "protocol":
        allowed = (len(parts) == 2 and parts[0] == "experiments"
                   and parts[1].startswith("dreamerv3-reused-train-") and parts[1].endswith(".json"))
    elif kind == "r6":
        allowed = name == R6_PATH
    elif kind == "source":
        allowed = name in REQUIRED_SOURCES
    elif kind == "actor":
        allowed = (len(parts) >= 3 and parts[0] == "runs" and parts[-1].endswith(".pt")
                   and not any(any(token in part.lower() for token in FORBIDDEN_OUTPUT_TOKENS)
                               for part in parts[1:]))
    else:
        raise ValueError(f"unsupported file kind: {kind}")
    if not allowed:
        raise ValueError(f"{kind}: outside permitted TRAIN-only paths: {name}")
    result = root
    for part in parts:
        result = result / part
        if result.is_symlink():
            raise ValueError(f"{kind}: symlinks are forbidden: {name}")
    if not result.is_file():
        raise ValueError(f"{kind}: missing pinned file: {name}")
    return result


def _pinned(root: Path, name: str, expected: str, *, kind: str) -> Path:
    path = _path(root, name, kind=kind)
    if sha256(path) != _hash(expected, f"{kind} hash"):
        raise ValueError(f"{kind}: SHA-256 mismatch: {name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("pinned JSON exceeds 8 MiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid pinned JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError("pinned JSON must be an object")
    return result


def _seeds(value: Any, name: str) -> list[int]:
    if (not isinstance(value, list) or not value
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name}: require distinct uint32 geometry seeds")
    return value


def random_source_hash(collector_sha256: str, rng_seed: int) -> str:
    """Identity for the random policy, not a claim that it has actor weights."""
    _hash(collector_sha256, "collector SHA")
    if type(rng_seed) is not int or not 0 <= rng_seed < 2**32:
        raise ValueError("random RNG seed must be uint32")
    identity = {"policy": "uniform-native-random-v1", "collector_sha256": collector_sha256,
                "rng_seed": rng_seed, "native_bounds": [-1.0, 1.0], "action_dim": 3}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


def preflight(protocol_path: Path, expected_protocol_sha256: str, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Validate reuse, fixed schedules, executable hashes and CPU actor identity before reset."""
    root = Path(repo_root).resolve()
    protocol_name = _relative(root, protocol_path)
    protocol = _json(_pinned(root, protocol_name, expected_protocol_sha256, kind="protocol"))
    if (set(protocol) != {"format", "study_id", "purpose", "freshness_claim", "r6_protocol",
                         "source_sha256", "cells", "episode_schedule", "frame_skip", "max_steps",
                         "budgets", "random", "source_actor", "spec_fingerprints"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["freshness_claim"] != LIMITATION
            or not isinstance(protocol["study_id"], str)
            or not protocol["study_id"].startswith("dreamerv3-reused-train-")):
        raise ValueError("unsupported reused-TRAIN engineering diagnostic protocol")
    if (type(protocol["frame_skip"]) is not int or protocol["frame_skip"] != 4
            or type(protocol["max_steps"]) is not int or not 1 <= protocol["max_steps"] <= 2000):
        raise ValueError("unshaped frame_skip or max_steps contract")
    if protocol["spec_fingerprints"] != {"action": ActionSpec().fingerprint,
                                          "observation": ObservationSpec().fingerprint}:
        raise ValueError("action/observation fingerprint mismatch")

    r6_spec = protocol["r6_protocol"]
    if (not isinstance(r6_spec, dict) or set(r6_spec) != {"path", "sha256"}
            or r6_spec["path"] != R6_PATH or r6_spec["sha256"] != R6_SHA256):
        raise ValueError("must pin the exact frozen r6 protocol path and SHA")
    r6 = _json(_pinned(root, R6_PATH, r6_spec["sha256"], kind="r6"))
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != REQUIRED_SOURCES:
        raise ValueError("protocol must pin the exact executable source inventory")
    for name, digest in sources.items():
        _pinned(root, name, digest, kind="source")
    if sha256(Path(__file__)) != sources[COLLECTOR_PATH]:
        raise ValueError("executing collector differs from pinned source")

    pool = r6.get("training_pool")
    diagnostic = r6.get("diagnostic_pool")
    environment = r6.get("environment")
    if (r6.get("study_id") != "drqv2-geometry-mix-v1-r6" or r6.get("format") != "haic-drq-geometry-mix-study-v1"
            or not isinstance(pool, dict) or pool.get("partition") != "TRAIN"
            or not isinstance(diagnostic, dict) or diagnostic.get("partition") != "TRAIN-DIAGNOSTIC"
            or not isinstance(environment, dict) or environment.get("partition") != "TRAIN"
            or environment.get("frame_skip") != 4 or type(environment.get("max_steps")) is not int
            or protocol["max_steps"] > environment["max_steps"]
            or environment.get("reward_shaping") is not False or environment.get("obstacles") is not True):
        raise ValueError("r6 does not establish an unshaped TRAIN-only environment")
    train_seeds = set(_seeds(pool.get("geometry_seeds"), "r6 TRAIN pool"))
    diagnostic_seeds = set(_seeds(diagnostic.get("geometry_seeds"), "r6 TRAIN-DIAGNOSTIC pool"))
    if (train_seeds & diagnostic_seeds or pool.get("track_ids") != [1, 2, 3, 4]
            or environment.get("track_ids") != [1, 2, 3, 4]):
        raise ValueError("r6 TRAIN pool overlaps diagnostic or has unrecognized tracks")
    cells = protocol["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError("fixed TRAIN cells are required")
    used: set[int] = set()
    for cell in cells:
        if (not isinstance(cell, dict) or set(cell) != {"track_id", "geometry_seed"}
                or type(cell["track_id"]) is not int or cell["track_id"] not in (1, 2, 3, 4)
                or type(cell["geometry_seed"]) is not int or cell["geometry_seed"] not in train_seeds
                or cell["geometry_seed"] in used):
            raise ValueError("all unique cells must belong to r6 training_pool.geometry_seeds, not diagnostic/heldout")
        used.add(cell["geometry_seed"])
    schedule = protocol["episode_schedule"]
    if (not isinstance(schedule, list) or len(schedule) != len(cells)
            or any(type(index) is not int for index in schedule)
            or set(schedule) != set(range(len(cells)))):
        raise ValueError("fixed episode_schedule must be a permutation of every unique TRAIN cell")
    budgets = protocol["budgets"]
    if (not isinstance(budgets, dict) or set(budgets) != {"random_decision_cap", "teacher_decision_cap"}
            or any(type(cap) is not int or not len(cells) <= cap <= 32768 for cap in budgets.values())):
        raise ValueError("both arms require fixed bounded decision caps")

    random = protocol["random"]
    if not isinstance(random, dict) or set(random) != {"rng_seed", "source_id", "source_actor_sha256"}:
        raise ValueError("random policy source identity is incomplete")
    rng_seed = random["rng_seed"]
    expected_random_hash = random_source_hash(sources[COLLECTOR_PATH], rng_seed)
    if (random["source_id"] != f"uniform-native-random-seed-{rng_seed}"
            or random["source_actor_sha256"] != expected_random_hash):
        raise ValueError("random policy identity must bind collector code and RNG seed")

    source = protocol["source_actor"]
    if (not isinstance(source, dict) or set(source) != {
            "source_id", "learner_seed", "source_revision", "checkpoint_path", "checkpoint_sha256",
            "actor_path", "actor_sha256", "actor_weights_sha256",
    } or source["source_id"] != "drq-source-0" or type(source["learner_seed"]) is not int
            or source["learner_seed"] != 0 or not isinstance(source["source_revision"], str)
            or not source["source_revision"].strip()):
        raise ValueError("teacher must be the frozen pad-4 DrQ source0")
    if (r6.get("learner", {}).get("padding") != 4
            or r6["learner"].get("source_environment_steps") != 131072
            or not isinstance(r6.get("source_actors"), list) or len(r6["source_actors"]) != 2):
        raise ValueError("r6 does not identify two frozen pad-4 source actors")
    r6_actor = r6["source_actors"][0]
    if (not isinstance(r6_actor, dict) or r6_actor.get("source_seed") != 0
            or r6_actor.get("weight_only_fork") is not True
            or any(source[field] != r6_actor.get(r6_field) for field, r6_field in (
                ("source_revision", "source_revision"), ("checkpoint_path", "source_checkpoint_path"),
                ("checkpoint_sha256", "source_checkpoint_sha256"), ("actor_path", "source_actor_path"),
                ("actor_sha256", "source_actor_sha256"),
                ("actor_weights_sha256", "source_actor_weights_sha256"),
            )) or random["source_actor_sha256"] == source["actor_sha256"]):
        raise ValueError("teacher source0 identity differs from pinned r6 source")
    checkpoint = _pinned(root, source["checkpoint_path"], source["checkpoint_sha256"], kind="actor")
    actor_path = _pinned(root, source["actor_path"], source["actor_sha256"], kind="actor")
    pair = audit_source_actor_pair(
        checkpoint, actor_path, learner_seed=0, source_revision=source["source_revision"],
        expected_checkpoint_sha256=source["checkpoint_sha256"], expected_actor_sha256=source["actor_sha256"],
    )
    for field, expected in (
        ("learner_seed", 0), ("source_revision", source["source_revision"]),
        ("checkpoint_sha256", source["checkpoint_sha256"]), ("actor_sha256", source["actor_sha256"]),
        ("actor_weights_sha256", _hash(source["actor_weights_sha256"], "actor weights")),
        ("action_fingerprint", protocol["spec_fingerprints"]["action"]),
        ("observation_fingerprint", protocol["spec_fingerprints"]["observation"]),
        ("cpu_smoke_observations", 3),
    ):
        if pair.get(field) != expected:
            raise ValueError(f"teacher actor/checkpoint CPU source identity mismatch: {field}")
    return {"protocol": protocol, "protocol_sha256": expected_protocol_sha256,
            "r6_protocol_sha256": r6_spec["sha256"], "excluded_seeds": sorted(diagnostic_seeds),
            "source_pair_audit": pair, "protocol_path": protocol_name}


def _check_pins(root: Path, checked: dict[str, Any]) -> None:
    protocol = checked["protocol"]
    _pinned(root, checked["protocol_path"], checked["protocol_sha256"], kind="protocol")
    _pinned(root, R6_PATH, checked["r6_protocol_sha256"], kind="r6")
    for name, digest in protocol["source_sha256"].items():
        _pinned(root, name, digest, kind="source")
    source = protocol["source_actor"]
    for which in ("checkpoint", "actor"):
        _pinned(root, source[f"{which}_path"], source[f"{which}_sha256"], kind="actor")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _collect_arm(root: Path, checked: dict[str, Any], output: Path, arm: str, actor: Any,
                 adapter: ActionAdapter, observation_spec: ObservationSpec) -> dict[str, Any]:
    protocol = checked["protocol"]
    source = protocol["random"] if arm == "random" else protocol["source_actor"]
    actor_hash = source["source_actor_sha256"] if arm == "random" else source["actor_sha256"]
    cap = protocol["budgets"][f"{arm}_decision_cap"]
    rng = np.random.default_rng(protocol["random"]["rng_seed"]) if arm == "random" else None
    dataset = TeacherDataset()
    rows: list[dict[str, Any]] = []
    spent = calls = 0
    failure: BaseException | None = None
    for attempt, index in enumerate(protocol["episode_schedule"]):
        if spent == cap:
            break
        cell = protocol["cells"][index]
        row: dict[str, Any] = {"attempt": attempt, **cell, "decisions": 0, "decision_calls": 0,
                               "complete": False, "finished": False, "status": "aborted",
                               "raw_reward_sum": 0.0}
        rows.append(row)
        transitions: list[Transition] = []
        proposed_actions: list[list[float]] = []
        try:
            _check_pins(root, checked)
            env = build_env(cell["track_id"], cell["geometry_seed"], protocol["max_steps"],
                            protocol["frame_skip"], reward_shaping=False, obstacles=True,
                            collision_penalty=0.0)
            try:
                collector = EpisodeCollector(env, action_adapter=adapter,
                                             observation_spec=observation_spec, gamma=1.0)
                observation, info = collector.reset()
                if (type(info.get("track_id")) is not int or info["track_id"] != cell["track_id"]
                        or type(info.get("seed")) is not int or info["seed"] != cell["geometry_seed"]):
                    raise ValueError("environment reset did not honor its frozen TRAIN cell")
                for step in range(min(cap - spent, protocol["max_steps"])):
                    if arm == "random":
                        proposed = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    else:
                        with torch.inference_mode():
                            proposed = actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0).cpu().numpy()
                    proposed = ActionAdapter._validate(proposed).copy()
                    calls += 1
                    row["decision_calls"] += 1
                    transition = collector.step(proposed)
                    spent += 1
                    row["decisions"] += 1
                    if (not isinstance(transition, Transition) or transition.episode_id != collector.episode_id
                            or transition.step != step):
                        raise ValueError("collector did not return a contiguous genuine Transition")
                    if (transition.applied_action is None
                            or not np.allclose(transition.applied_action, adapter.to_official(proposed), atol=1e-6, rtol=0)
                            or not np.allclose(transition.action, adapter.to_native(transition.applied_action, clip=False), atol=1e-6, rtol=0)):
                        raise ValueError("proposed/executed/applied action parity failed")
                    if not np.array_equal(observation_spec.to_uint8(transition.observation),
                                          observation_spec.to_uint8(observation)):
                        raise ValueError("collector observation differs from preceding result frame")
                    observation_spec.to_uint8(transition.next_observation)
                    if (type(transition.info.get("track_id")) is not int
                            or transition.info["track_id"] != cell["track_id"]
                            or type(transition.info.get("seed")) is not int
                            or transition.info["seed"] != cell["geometry_seed"]):
                        raise ValueError("TRAIN geometry changed within an episode")
                    finished = transition.info.get("finished")
                    if type(finished) is not bool or (finished and (not transition.done or not transition.is_terminal)):
                        raise ValueError("finished flag disagrees with actual episode boundary")
                    if (transition.is_terminal != collector._is_terminal(
                            transition.terminated, transition.truncated, transition.info)
                            or transition.is_terminal and not transition.done):
                        raise ValueError("terminal flag contradicts actual environment outcome")
                    for field in ("progress", "damage"):
                        value = transition.info.get(field)
                        if type(value) not in (int, float) or not np.isfinite(value):
                            raise ValueError(f"transition has no finite {field} label")
                    if not np.isfinite(transition.reward) or abs(transition.reward) > np.finfo(np.float32).max:
                        raise ValueError("transition has no representable raw reward")
                    transitions.append(transition)
                    proposed_actions.append(proposed.tolist())
                    row["raw_reward_sum"] += float(transition.reward)
                    if not np.isfinite(row["raw_reward_sum"]):
                        raise ValueError("episode raw reward sum is not finite")
                    observation = transition.next_observation
                    if transition.done:
                        episode = TeacherEpisode.from_transitions(
                            [replace(value, episode_id=attempt) for value in transitions],
                            source_id=source["source_id"], source_actor_sha256=actor_hash,
                            geometry_id=str(cell["geometry_seed"]), track_id=cell["track_id"],
                            metadata={"study_id": protocol["study_id"], "arm": arm, "attempt": attempt,
                                      "collector_episode_id": transition.episode_id,
                                      "proposed_native_actions": proposed_actions},
                        )
                        dataset.add_episode(episode)
                        row.update(status="complete", complete=True, finished=finished,
                                   episode_id=attempt, collector_episode_id=transition.episode_id,
                                   terminated=transition.terminated, truncated=transition.truncated,
                                   terminal=transition.is_terminal,
                                   retire_reason=transition.info.get("retire_reason"),
                                   progress=float(transition.info["progress"]),
                                   damage=float(transition.info["damage"]))
                        break
                    if step + 1 == protocol["max_steps"]:
                        raise ValueError("environment reached max_steps without a real end or truncation")
                if transitions and not transitions[-1].done:
                    row.update(status="cap_partial", complete=False, finished=False,
                               terminated=False, truncated=False, terminal=False,
                               retire_reason=transitions[-1].info.get("retire_reason"),
                               progress=float(transitions[-1].info["progress"]),
                               damage=float(transitions[-1].info["damage"]))
            finally:
                env.close()
        except BaseException as exc:
            failure = exc
            row["status"] = "aborted"
            row["error"] = f"{type(exc).__name__}: {exc}"
            break

    arm_dir = output / arm
    arm_dir.mkdir(exist_ok=False)
    archive_sha = digest = None
    if dataset.episodes:
        try:
            digest = dataset.seal()
            with (arm_dir / "support-dataset.npz").open("xb") as stream:
                stream.write(dataset.to_bytes())
            archive_sha = sha256(arm_dir / "support-dataset.npz")
        except BaseException as exc:
            if failure is None:
                failure = exc
    result = {
        "format": "haic-dreamerv3-reused-train-collection-result-v1",
        "study_id": protocol["study_id"], "purpose": PURPOSE, "freshness_claim": LIMITATION,
        "protocol_sha256": checked["protocol_sha256"], "r6_protocol_sha256": checked["r6_protocol_sha256"],
        "arm": arm, "status": "aborted" if failure else "completed",
        "source_id": source["source_id"], "source_actor_sha256": actor_hash,
        "source_checkpoint_sha256": source["checkpoint_sha256"] if arm == "teacher" else None,
        "source_pair_audit": checked["source_pair_audit"] if arm == "teacher" else None,
        "random_rng_seed": source["rng_seed"] if arm == "random" else None,
        "decision_cap": cap, "decisions_spent": spent, "decision_calls": calls,
        "unresolved_decision_calls": calls - spent,
        "stored_decisions": dataset.transition_count, "partial_decisions": spent - dataset.transition_count,
        "complete_episode_count": len(dataset.episodes), "episode_rows": rows,
        "schedule_attempts": len(rows), "schedule_exhausted": len(rows) == len(protocol["episode_schedule"]),
        "distinct_finished_geometries": sorted({row["geometry_seed"] for row in rows if row["finished"]}),
        "dataset_path": "support-dataset.npz" if archive_sha else None,
        "dataset_digest": digest, "archive_sha256": archive_sha,
        "allowed_cells": protocol["cells"], "excluded_seeds": checked["excluded_seeds"],
    }
    if failure:
        result["abort_reason"] = f"{type(failure).__name__}: {failure}"
    _write_json(arm_dir / "collection-result.json", result)
    if failure:
        raise failure
    return result


def collect(protocol_path: Path, expected_protocol_sha256: str, output_dir: Path, *,
            repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output_name = _relative(root, output_dir)
    parts = output_name.split("/")
    output = root / output_name
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (len(parts) < 3 or parts[0] != "runs" or any(part in ("", ".", "..") for part in parts)
            or any(any(token in part.lower() for token in FORBIDDEN_OUTPUT_TOKENS) for part in parts[1:])
            or any(root.joinpath(*parts[:index]).is_symlink() for index in range(1, len(parts)))
            or not output.parent.is_dir()):
        raise ValueError("output must be a new directory beneath an existing TRAIN run root")
    checked = preflight(protocol_path, expected_protocol_sha256, repo_root=root)
    source = checked["protocol"]["source_actor"]
    actor, adapter, observation_spec = load_exported_actor(root / source["actor_path"], device="cpu")
    if (adapter.spec.fingerprint != checked["protocol"]["spec_fingerprints"]["action"]
            or observation_spec.fingerprint != checked["protocol"]["spec_fingerprints"]["observation"]):
        raise ValueError("loaded CPU actor contract differs from verified source")
    _check_pins(root, checked)
    output.mkdir(exist_ok=False)
    results: dict[str, dict[str, Any]] = {}
    try:
        for arm in ("random", "teacher"):
            results[arm] = _collect_arm(root, checked, output, arm, actor, adapter, observation_spec)
    except BaseException as exc:
        _write_json(output / "abort-receipt.json", {
            "format": "haic-dreamerv3-reused-train-abort-v1", "study_id": checked["protocol"]["study_id"],
            "purpose": PURPOSE, "freshness_claim": LIMITATION,
            "protocol_sha256": checked["protocol_sha256"],
            "completed_arms": {name: {"decisions_spent": row["decisions_spent"],
                                       "archive_sha256": row["archive_sha256"]} for name, row in results.items()},
            "failed_arm": "random" if "random" not in results else "teacher",
            "failed_arm_receipt_path": ("random" if "random" not in results else "teacher") + "/collection-result.json",
            "reason": f"{type(exc).__name__}: {exc}",
        })
        raise
    result = {"format": "haic-dreamerv3-reused-train-result-v1",
              "study_id": checked["protocol"]["study_id"], "purpose": PURPOSE,
              "freshness_claim": LIMITATION, "protocol_sha256": checked["protocol_sha256"],
              "r6_protocol_sha256": checked["r6_protocol_sha256"],
              "allowed_cells": checked["protocol"]["cells"], "excluded_seeds": checked["excluded_seeds"],
              "arms": results}
    _write_json(output / "collection-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(collect(args.protocol, args.protocol_sha256, args.output, repo_root=args.repo_root),
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
