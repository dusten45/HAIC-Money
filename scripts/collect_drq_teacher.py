"""Collect a bounded, compact, immutable dataset from one frozen DrQ source actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from common_adapter import ActionSpec, EpisodeCollector, ObservationSpec
from drq_v2 import load_exported_actor
from haic.algorithms.drq_v2.teacher_replay import (
    GeometryPoolRNG,
    TeacherDataset,
    TeacherEpisode,
    audit_source_actor_pair,
    find_sampled_track_env,
)
from train import build_sampled_env
from scripts.validate_drq_teacher_protocol import ProtocolError, validate_protocol_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_file(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("protocol source paths must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"invalid source path in protocol: {relative_path}")
    return resolved


def _frame_stack(frames: list[np.ndarray], index: int) -> np.ndarray:
    return np.stack([frames[max(0, index - 3 + channel)] for channel in range(4)], axis=0)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("xb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _source_actor(protocol: dict[str, Any], root: Path, learner_seed: int) -> dict[str, Any]:
    for source in protocol["source_actors"]:
        if source["learner_seed"] == learner_seed:
            return source
    raise ValueError(f"source actor {learner_seed} is absent from the protocol")


def collect_source(
    protocol: dict[str, Any],
    protocol_sha256: str,
    root: Path,
    learner_seed: int,
    run_dir: Path,
    preflight_receipt_path: Path,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = _source_actor(protocol, root, learner_seed)
    preflight_receipt_path = (
        preflight_receipt_path
        if preflight_receipt_path.is_absolute() else root / preflight_receipt_path
    ).resolve()
    if not preflight_receipt_path.is_relative_to(root.resolve()) or not preflight_receipt_path.is_file():
        raise ValueError("A0 collection gate must be an existing repository-relative receipt")
    preflight = json.loads(preflight_receipt_path.read_text(encoding="utf-8"))
    if (
        preflight.get("format") != "haic-drq-teacher-replay-a0-collection-gate-v1"
        or preflight.get("status") != "pass"
        or preflight.get("study_id") != protocol["study_id"]
        or preflight.get("study_protocol_sha256") != protocol_sha256
        or preflight.get("geometry_audit_path") != protocol["geometry_audit"]["report_path"]
        or preflight.get("geometry_audit_sha256") != protocol["geometry_audit"]["report_sha256"]
    ):
        raise ValueError("A0 collection gate belongs to another or incomplete study protocol")
    live_audit = preflight.get("live_geometry_audit", {})
    expected_candidates = sorted({
        seed for pool in protocol["training_pools"].values() for seed in pool["seeds"]
    } | {
        seed for partition in protocol["partitions"].values() for seed in partition["seeds"]
    })
    if (
        live_audit.get("passed") is not True
        or live_audit.get("candidate_seeds") != expected_candidates
        or live_audit.get("parse_errors")
        or any(live_audit.get("candidate_hits", {}).values())
        or live_audit.get("known_excluded_geometry_seeds")
        != protocol["known_excluded_geometry_seeds"]
    ):
        raise ValueError("A0 exact-token freshness scan did not pass for all frozen pools")
    checkpoint_path = _repo_file(root, source["checkpoint_path"])
    actor_path = _repo_file(root, source["actor_path"])
    if _sha256(actor_path) != source["actor_sha256"]:
        raise ValueError("source actor changed after protocol freeze")
    source_pair_audit = audit_source_actor_pair(
        checkpoint_path,
        actor_path,
        learner_seed=learner_seed,
        source_revision=source["source_revision"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )

    pool = protocol["training_pools"]["teacher_training"]
    if set(pool["seeds"]) & set(protocol["reserved_training_seeds"]):
        raise ValueError("teacher collection pool overlaps frozen exclusions")
    actor, action_adapter, observation_spec = load_exported_actor(actor_path, device="cpu")
    if action_adapter.spec.fingerprint != ActionSpec().fingerprint:
        raise ValueError("source actor action contract differs from the study")
    if observation_spec.fingerprint != ObservationSpec().fingerprint:
        raise ValueError("source actor observation contract differs from the study")
    env = build_sampled_env(
        track_ids=pool["track_ids"],
        sampler_seed=pool["sampler_seed"],
        max_steps=protocol["max_steps"],
        frame_skip=protocol["frame_skip"],
        reward_shaping=False,
        excluded_seeds=protocol["reserved_training_seeds"],
        obstacles=True,
        collision_penalty=0.0,
    )
    try:
        sampler = find_sampled_track_env(env)
    except RuntimeError:
        env.close()
        raise
    sampler._rng = GeometryPoolRNG(pool["track_ids"], pool["seeds"], seed=pool["sampler_seed"])
    collector = EpisodeCollector(env, action_adapter=action_adapter, observation_spec=observation_spec)
    dataset = TeacherDataset()
    episode_summaries: list[dict[str, Any]] = []
    finished_seeds: set[int] = set()
    finished_by_seed: dict[int, int] = {}
    incomplete_episode: dict[str, Any] | None = None
    episode_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    rewards: list[float] = []
    progress: list[float] = []
    damage: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    finished: list[bool] = []
    terminal: list[bool] = []
    retire_reasons: list[str | None] = []
    episode_info: dict[str, Any] | None = None
    steps = 0
    resets = 0
    started = time.monotonic()
    observation_encoder = ObservationSpec()

    def finish_episode(complete: bool) -> None:
        nonlocal episode_frames, actions, applied_actions, rewards, progress, damage
        nonlocal terminated, truncated, finished, terminal, retire_reasons, episode_info
        nonlocal incomplete_episode
        if not actions:
            return
        assert episode_info is not None
        summary = {
            "episode_id": resets - 1,
            "track_id": int(episode_info["track_id"]),
            "geometry_seed": int(episode_info["seed"]),
            "steps": len(actions),
            "complete": bool(complete),
            "finished": bool(finished[-1]),
            "progress": float(progress[-1]),
            "damage": float(damage[-1]),
            "retire_reason": retire_reasons[-1],
        }
        episode_summaries.append(summary)
        if complete:
            episode = TeacherEpisode(
                frames=np.stack(episode_frames),
                actions=np.stack(actions),
                applied_actions=np.stack(applied_actions),
                rewards=np.asarray(rewards, dtype=np.float32),
                progress=np.asarray(progress, dtype=np.float32),
                damage=np.asarray(damage, dtype=np.float32),
                terminated=np.asarray(terminated, dtype=np.bool_),
                truncated=np.asarray(truncated, dtype=np.bool_),
                finished=np.asarray(finished, dtype=np.bool_),
                terminal=np.asarray(terminal, dtype=np.bool_),
                retire_reasons=retire_reasons,
                episode_id=resets - 1,
                source_id=f"source-learner-{learner_seed}",
                source_actor_sha256=source["actor_sha256"],
                geometry_id=str(episode_info["seed"]),
                track_id=int(episode_info["track_id"]),
                complete=True,
                metadata=summary,
            )
            dataset.add_episode(episode)
            if summary["finished"]:
                finished_seeds.add(summary["geometry_seed"])
                finished_by_seed[summary["geometry_seed"]] = (
                    finished_by_seed.get(summary["geometry_seed"], 0) + 1
                )
        else:
            incomplete_episode = summary
        episode_frames, actions, applied_actions = [], [], []
        rewards, progress, damage = [], [], []
        terminated, truncated, finished, terminal, retire_reasons = [], [], [], [], []
        episode_info = None

    observation, info = collector.reset(seed=None)
    resets += 1
    episode_info = {"track_id": info.get("track_id"), "seed": info.get("seed")}
    if episode_info["track_id"] not in pool["track_ids"] or episode_info["seed"] not in pool["seeds"]:
        raise RuntimeError("reset sampler produced a cell outside the teacher-training pool")
    episode_frames = [observation_encoder.to_uint8(observation)[-1].copy()]
    if not np.array_equal(observation_encoder.to_uint8(observation), _frame_stack(episode_frames, 0)):
        raise ValueError("reset observation does not use frozen repeated-frame stack padding")

    try:
        while steps < protocol["budgets"]["teacher_decisions_per_source_cap"]:
            with torch.inference_mode():
                native_action = actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0).cpu().numpy()
            transition = collector.step(native_action)
            current = observation_encoder.to_uint8(transition.observation)
            next_value = observation_encoder.to_uint8(transition.next_observation)
            if not np.array_equal(current, _frame_stack(episode_frames, len(actions))):
                raise ValueError("collector current observation does not align to staged latest frames")
            next_index = len(actions) + 1
            candidate_frames = episode_frames + [next_value[-1].copy()]
            if not np.array_equal(next_value, _frame_stack(candidate_frames, next_index)):
                raise ValueError("collector next observation does not align to staged latest frames")
            observed_track = transition.info.get("track_id")
            observed_seed = transition.info.get("seed")
            if observed_track != episode_info["track_id"] or observed_seed != episode_info["seed"]:
                raise ValueError("environment geometry changed inside a teacher episode")
            if int(observed_seed) in set(protocol["reserved_training_seeds"]):
                raise ValueError("teacher sampler crossed into a reserved geometry seed")
            episode_frames.append(next_value[-1].copy())
            actions.append(transition.action.copy())
            applied_actions.append(transition.applied_action.copy())
            rewards.append(transition.reward)
            progress.append(float(transition.info.get("progress", 0.0)))
            damage.append(float(transition.info.get("damage", 0.0)))
            terminated.append(transition.terminated)
            truncated.append(transition.truncated)
            finished.append(bool(transition.info.get("finished", False)))
            terminal.append(bool(transition.is_terminal))
            retire_reasons.append(transition.info.get("retire_reason"))
            steps += 1
            observation = transition.next_observation
            if transition.terminated or transition.truncated:
                finish_episode(complete=True)
                if steps < protocol["budgets"]["teacher_decisions_per_source_cap"]:
                    observation, info = collector.reset(seed=None)
                    resets += 1
                    episode_info = {"track_id": info.get("track_id"), "seed": info.get("seed")}
                    if episode_info["track_id"] not in pool["track_ids"] or episode_info["seed"] not in pool["seeds"]:
                        raise RuntimeError("reset sampler produced a cell outside the teacher-training pool")
                    episode_frames = [observation_encoder.to_uint8(observation)[-1].copy()]
                    if not np.array_equal(observation_encoder.to_uint8(observation), _frame_stack(episode_frames, 0)):
                        raise ValueError("reset observation does not use frozen repeated-frame stack padding")
            elif steps % 4096 == 0:
                elapsed = time.monotonic() - started
                print(f"learner_seed={learner_seed} decisions={steps} complete_episodes={len(dataset.episodes)} elapsed={elapsed:.1f}s", flush=True)
        if actions:
            finish_episode(complete=False)
    finally:
        env.close()

    if dataset.episodes:
        dataset_digest = dataset.seal()
        dataset_bytes = dataset.to_bytes()
        dataset_path = run_dir / "teacher-dataset.npz"
        _atomic_write(dataset_path, dataset_bytes)
        dataset_sha256 = _sha256(dataset_path)
    else:
        dataset_digest = None
        dataset_sha256 = None
        dataset_path = None

    minimum_finished = 4
    coverage_pass = len(finished_seeds) >= minimum_finished
    result = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "study_protocol_sha256": protocol_sha256,
        "a0_collection_gate_path": preflight_receipt_path.relative_to(root).as_posix(),
        "a0_collection_gate_sha256": _sha256(preflight_receipt_path),
        "phase": "teacher-data-collection",
        "learner_seed": learner_seed,
        "source_actor_sha256": source["actor_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_pair_audit": source_pair_audit,
        "source_revision": source["source_revision"],
        "training_pool": pool,
        "decisions": steps,
        "decision_cap": protocol["budgets"]["teacher_decisions_per_source_cap"],
        "complete_episode_count": len(dataset.episodes),
        "complete_transition_count": dataset.transition_count,
        "dataset_path": dataset_path.relative_to(root).as_posix() if dataset_path else None,
        "dataset_digest": dataset_digest,
        "dataset_file_sha256": dataset_sha256,
        "dataset_memory_bytes": dataset.memory_bytes,
        "finished_geometry_seed_count": len(finished_seeds),
        "finished_geometry_seeds": sorted(finished_seeds),
        "finishes_by_geometry_seed": finished_by_seed,
        "required_distinct_finished_geometry_seeds": minimum_finished,
        "coverage_gate": "pass" if coverage_pass else "inconclusive-fail",
        "coverage_pass": coverage_pass,
        "incomplete_episode": incomplete_episode,
        "episode_summaries": episode_summaries,
        "environment_steps_are_official_local_decisions": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = run_dir / "collection-result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_sha256"] = _sha256(result_path)
    if not coverage_pass:
        raise RuntimeError("teacher collection did not meet the frozen 4-geometry finish gate")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--learner-seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--preflight-receipt", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    protocol_path = args.protocol_file.resolve()
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    try:
        validate_protocol_file(protocol_path, root)
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if args.learner_seed != _source_actor(protocol, root, args.learner_seed)["learner_seed"]:
            raise ValueError("source learner identity mismatch")
        if run_dir.exists():
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        result = collect_source(
            protocol, _sha256(protocol_path), root, args.learner_seed, run_dir,
            args.preflight_receipt,
        )
    except (OSError, ValueError, ProtocolError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
