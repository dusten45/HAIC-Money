"""Train one frozen online-only DrQ-v2 geometry-mix arm.

This study starts from one of two SHA-pinned pad-4 source checkpoints. It never
loads a teacher replay dataset; the catalog sampler exposes only TRAIN rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from common_adapter import EpisodeCollector
from drq_v2 import DrQv2Agent, DrQv2Config
from haic.algorithms.drq_v2.geometry_sampler import CatalogGeometrySampler, build_catalog_env
from haic.algorithms.drq_v2.teacher_study import (
    RNGStreams,
    audit_source_actor_pair,
    fork_from_source,
    learner_rng_state,
    restore_learner_rng_state,
    update_from_mixture,
)
from haic.algorithms.drq_v2.teacher_replay import TwoSourceReplay


STUDY_ID = "drqv2-geometry-mix-v1-r6"
PROTOCOL_FORMAT = "haic-drq-geometry-mix-study-v1"
TRAINER_FORMAT = "haic-drq-geometry-mix-trainer-v1"
VARIANTS = ("uniform", "failure_weighted", "easy_retention")
FAMILY_ANCHOR = "easy-curvature-anchor"
FAMILY_OPENING_SHORT = "opening-short-entry-left-turn"
FAMILY_OPENING_DELAYED = "opening-delayed-high-turn"
FAMILY_MID_REVERSAL = "mid-road-left-right-reversal"
FAMILY_SUSTAINED = "mid-road-sustained-or-same-turn"
FAMILY_FINISH = "finish-approach-turn"
FAMILIES = (
    FAMILY_OPENING_SHORT,
    FAMILY_OPENING_DELAYED,
    FAMILY_ANCHOR,
    FAMILY_MID_REVERSAL,
    FAMILY_SUSTAINED,
    FAMILY_FINISH,
)
EXPECTED_VARIANT_DISTRIBUTIONS = {
    "uniform": (
        {family: "boundary" for family in FAMILIES},
        {"easy": 0.0, "boundary": 1.0, "difficult": 0.0},
    ),
    "failure_weighted": (
        {
            FAMILY_ANCHOR: "easy",
            FAMILY_OPENING_SHORT: "difficult",
            FAMILY_OPENING_DELAYED: "difficult",
            FAMILY_MID_REVERSAL: "difficult",
            FAMILY_SUSTAINED: "boundary",
            FAMILY_FINISH: "boundary",
        },
        {"easy": 0.20, "boundary": 0.25, "difficult": 0.55},
    ),
    "easy_retention": (
        {
            FAMILY_ANCHOR: "easy",
            FAMILY_OPENING_SHORT: "boundary",
            FAMILY_OPENING_DELAYED: "boundary",
            FAMILY_MID_REVERSAL: "boundary",
            FAMILY_SUSTAINED: "boundary",
            FAMILY_FINISH: "difficult",
        },
        {"easy": 0.45, "boundary": 0.40, "difficult": 0.15},
    ),
}
EXPECTED_BUDGETS = {
    "additional_online_steps": 32_768,
    "warmup_steps": 10_000,
    "replay_capacity": 100_000,
    "batch_size": 64,
    "expected_gradient_updates": 22_768,
    "checkpoint_steps": [16_384, 32_768],
    "critic_updates_per_learning_decision": 1,
    "n_step": 3,
    "gamma": 0.99,
}
EXPECTED_LEARNER = {
    "feature_dim": 256,
    "hidden_dim": 256,
    "n_step": 3,
    "gamma": 0.99,
    "actor_lr": 1e-4,
    "critic_lr": 1e-4,
    "tau": 0.01,
    "actor_update_frequency": 2,
    "target_update_frequency": 2,
    "padding": 4,
    "target_noise_std": 0.2,
    "target_noise_clip": 0.5,
    "steering_logit_l2": 0.0,
}
EXPECTED_ENVIRONMENT = {
    "frame_skip": 4,
    "track_ids": [1, 2, 3, 4],
}
_RNG_SEED_NAMES = (
    "track_seed",
    "geometry_seed",
    "actor_rng_seed",
    "replay_rng_seed",
    "target_noise_seed",
    "update_rng_seed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as destination:
        destination.write(data)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_bytes(path, raw)


def _repo_file(root: Path, relative_path: str, label: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"{label} path must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"{label} path is missing or outside the repository: {relative_path}")
    return resolved


def _repo_directory(root: Path, relative_path: str, label: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"{label} path must be repository-relative")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} path must remain under the repository root")
    return resolved


def _validate_catalog(protocol: Mapping[str, Any], root: Path) -> tuple[dict[str, int], list[str]]:
    catalog_ref = protocol["catalog"]
    catalog_path = _repo_file(root, catalog_ref["path"], "catalog")
    catalog_protocol_path = _repo_file(root, catalog_ref["protocol_path"], "catalog protocol")
    catalog_raw = catalog_path.read_bytes()
    catalog_protocol_raw = catalog_protocol_path.read_bytes()
    if _sha256(catalog_path) != catalog_ref["sha256"]:
        raise ValueError("geometry catalog SHA-256 mismatch")
    if _sha256(catalog_protocol_path) != catalog_ref["protocol_sha256"]:
        raise ValueError("catalog-generation protocol SHA-256 mismatch")

    catalog = json.loads(catalog_raw)
    generation_protocol = json.loads(catalog_protocol_raw)
    if (
        catalog.get("format") != "haic-drq-training-geometry-catalog-v1"
        or catalog.get("protocol_sha256") != catalog_ref["protocol_sha256"]
        or generation_protocol.get("format") != "haic-drq-training-geometry-protocol-v1"
        or generation_protocol.get("train_target") != 120
        or generation_protocol.get("diagnostic_target") != 16
    ):
        raise ValueError("catalog or catalog-generation protocol identity/quota mismatch")
    rules = generation_protocol.get("family_rules")
    if not isinstance(rules, list) or len(rules) != 6:
        raise ValueError("catalog-generation protocol must define exactly six families")
    families = [rule.get("name") for rule in rules if isinstance(rule, dict)]
    if len(families) != 6 or set(families) != set(FAMILIES) or len(set(families)) != 6:
        raise ValueError("frozen catalog family names differ from the geometry-mix study")

    train = catalog.get("train")
    diagnostic = catalog.get("train_diagnostic")
    if not isinstance(train, list) or len(train) != 120:
        raise ValueError("geometry catalog must contain exactly 120 TRAIN rows")
    if not isinstance(diagnostic, list) or len(diagnostic) != 16:
        raise ValueError("geometry catalog must retain exactly 16 TRAIN-DIAGNOSTIC rows")
    allowed: dict[str, int] = {}
    all_ids: set[int] = set()
    for row in train:
        if not isinstance(row, dict):
            raise ValueError("catalog TRAIN contains a non-object row")
        seed, family = row.get("geometry_seed"), row.get("family")
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or row.get("stage") not in {"representative", "variant"}
            or row.get("track_id") != 1
            or family not in FAMILIES
            or seed in all_ids
        ):
            raise ValueError("catalog TRAIN rows contain an invalid or duplicate geometry")
        allowed[family + ":" + str(seed)] = seed
        all_ids.add(seed)
    train_seeds = {row["geometry_seed"] for row in train}
    diagnostic_seeds = set()
    for row in diagnostic:
        seed = row.get("geometry_seed") if isinstance(row, dict) else None
        if type(seed) is not int or not 0 <= seed < 2**32 or seed in all_ids:
            raise ValueError("catalog TRAIN-DIAGNOSTIC rows overlap or contain invalid geometry")
        diagnostic_seeds.add(seed)
        all_ids.add(seed)
    if len(train_seeds) != 120 or len(diagnostic_seeds) != 16:
        raise ValueError("catalog road identities must be unique across TRAIN partitions")
    counts = {family: sum(row["family"] == family for row in train) for family in FAMILIES}
    if any(count != 20 for count in counts.values()):
        raise ValueError("catalog TRAIN must have 20 rows in each fixed family")
    pool = protocol.get("training_pool")
    if pool is not None:
        pool_seeds = pool.get("geometry_seeds")
        if (
            not isinstance(pool_seeds, list)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in pool_seeds)
            or set(pool_seeds) != train_seeds
        ):
            raise ValueError("frozen training_pool geometry IDs do not equal catalog TRAIN IDs")
    return {str(seed): seed for seed in train_seeds}, families


def _catalog_train_index(root: Path, protocol: Mapping[str, Any]
                         ) -> tuple[dict[str, int], list[str]]:
    """Expose only the frozen catalog TRAIN seed index to checkpoint metadata."""
    return _validate_catalog(protocol, root)


def _same_number(actual: Any, expected: float) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and float(actual) == float(expected)
    )


def _budget(protocol: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "additional_online_steps": ("additional_online_steps", "additional_online_steps_per_run"),
        "warmup_steps": ("warmup_steps", "startup_decisions_without_updates"),
        "replay_capacity": ("replay_capacity", "online_replay_capacity"),
        "checkpoint_steps": ("checkpoint_steps", "checkpoint_online_steps"),
        "n_step": ("n_step",),
        "gamma": ("gamma",),
        "frame_skip": ("frame_skip",),
        "max_steps": ("max_steps",),
        "expected_gradient_updates": ("expected_gradient_updates",),
        "critic_updates_per_learning_decision": ("critic_updates_per_learning_decision",),
        "batch_size": ("batch_size",),
    }
    budgets = protocol["budgets"]
    for key in aliases[name]:
        if key in budgets:
            return budgets[key]
    raise ValueError(f"frozen protocol is missing budget field {name}")


def _environment_value(protocol: Mapping[str, Any], name: str, default: Any = None) -> Any:
    environment = protocol.get("environment", {})
    if name in environment:
        return environment[name]
    if name in protocol:
        return protocol[name]
    if name == "track_ids" and isinstance(protocol.get("training_pool"), Mapping):
        return protocol["training_pool"].get(name, default)
    if name == "frame_skip":
        return _budget(protocol, "frame_skip")
    if name == "max_steps":
        return _budget(protocol, "max_steps")
    if name in {"reward_shaping", "reward_normalization"}:
        return protocol.get("learner", {}).get(name, default)
    return default


def _learner_value(protocol: Mapping[str, Any], name: str) -> Any:
    learner = protocol["learner"]
    aliases = {
        "actor_lr": ("actor_lr", "actor_learning_rate"),
        "critic_lr": ("critic_lr", "critic_learning_rate"),
        "padding": ("padding", "augmentation_pad"),
    }
    for key in aliases.get(name, (name,)):
        if key in learner:
            return learner[key]
    if name in {"n_step", "gamma"}:
        return _budget(protocol, name)
    raise ValueError(f"frozen protocol is missing learner field {name}")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("format") != PROTOCOL_FORMAT or protocol.get("study_id") != STUDY_ID:
        raise ValueError("protocol is not the frozen DrQ-v2 geometry-mix v1 study")
    for name, expected in EXPECTED_ENVIRONMENT.items():
        if _environment_value(protocol, name) != expected:
            raise ValueError(f"frozen environment contract mismatch: {name}")
    if type(_environment_value(protocol, "max_steps")) is not int or _environment_value(protocol, "max_steps") != 2_000:
        raise ValueError("geometry-mix max_steps must remain 2000 frame-skip-4 decisions")
    raw_reward = _environment_value(protocol, "raw_reward")
    if raw_reward is not None and raw_reward is not True:
        raise ValueError("the frozen catalog environment must use raw, unshaped reward")
    obstacles = _environment_value(protocol, "obstacles")
    if obstacles is not None and obstacles is not True:
        raise ValueError("official obstacles must remain enabled")

    if any(_budget(protocol, name) != expected for name, expected in EXPECTED_BUDGETS.items()):
        raise ValueError("frozen geometry-mix budget differs from the preregistered contract")
    if any(not _same_number(_learner_value(protocol, name), expected)
           for name, expected in EXPECTED_LEARNER.items()):
        raise ValueError("frozen learner config differs from the pad-4 DrQ-v2 contract")
    learner = protocol["learner"]
    if learner.get("algorithm") != "DrQ-v2":
        raise ValueError("study must use the current DrQ-v2 learner")
    if any(
        learner.get(name, _environment_value(protocol, name, False)) is not False
        for name in ("reward_shaping", "reward_normalization")
    ):
        raise ValueError("reward shaping and normalization must remain disabled")
    collision_penalty = _environment_value(protocol, "collision_penalty")
    if collision_penalty is not None and not _same_number(collision_penalty, 0.0):
        raise ValueError("collision reward penalty must remain disabled")
    training_pool = protocol.get("training_pool")
    if training_pool is not None and (
        training_pool.get("partition") != "TRAIN"
        or len(training_pool.get("geometry_seeds", [])) != 120
        or len(set(training_pool.get("geometry_seeds", []))) != 120
        or training_pool.get("track_ids") != [1, 2, 3, 4]
    ):
        raise ValueError("training pool must identify exactly the catalog TRAIN rows and track IDs 1-4")

    variants = protocol["variants"]
    if set(variants) != set(VARIANTS):
        raise ValueError("protocol must declare exactly the three frozen family-mix variants")
    for name in VARIANTS:
        tiers, weights = EXPECTED_VARIANT_DISTRIBUTIONS[name]
        actual = variants[name]
        if actual.get("family_tiers") != tiers:
            raise ValueError(f"{name} family-to-tier map changed from the predeclared study")
        actual_weights = actual.get("tier_weights")
        if not isinstance(actual_weights, Mapping) or set(actual_weights) != set(weights):
            raise ValueError(f"{name} tier weights are incomplete")
        if any(not _same_number(actual_weights[tier], value) for tier, value in weights.items()):
            raise ValueError(f"{name} tier weights changed from the predeclared study")
        if actual.get("curriculum", False) is not False or actual.get("auto_update", False) is not False:
            raise ValueError("family weights must remain static; curriculum updates are prohibited")

    actors = protocol["source_actors"]
    if not isinstance(actors, list) or len(actors) != 2:
        raise ValueError("study must bind exactly two original source actors")
    if {row.get("learner_seed") for row in actors} != {0, 1}:
        raise ValueError("source actors must be the original DrQ learner seeds 0 and 1")
    if len({row.get("actor_sha256") for row in actors}) != 2:
        raise ValueError("the two source actor exports must be distinct")

    runs = protocol["runs"]
    expected_pairs = {(seed, variant) for seed in (0, 1) for variant in VARIANTS}
    if not isinstance(runs, list) or len(runs) != 6 or any(not isinstance(row, dict) for row in runs):
        raise ValueError("frozen training matrix must contain six run objects")
    actual_pairs = {
        (row.get("source_seed"), row.get("variant"))
        for row in runs
    }
    if actual_pairs != expected_pairs:
        raise ValueError("frozen training matrix must contain exactly six source-seed/variant runs")
    if len({row.get("run_dir") for row in runs}) != 6:
        raise ValueError("each frozen source-seed/variant cell must have a unique run_dir")
    run_root = protocol["run_root"].rstrip("/")
    for row in runs:
        expected_run_dir = f"{run_root}/learner-{row['source_seed']}-{row['variant']}"
        if row.get("run_dir") != expected_run_dir:
            raise ValueError("run_dir must follow the frozen learner-seed/variant matrix layout")
        if any(
            type(row.get(key)) is not int or not 0 <= row[key] < 2**32
            for key in _RNG_SEED_NAMES
        ):
            raise ValueError("each arm must freeze uint32 sampler and learner RNG seeds")
        if len({row[key] for key in _RNG_SEED_NAMES}) != len(_RNG_SEED_NAMES):
            raise ValueError("sampler, actor, replay, target-noise and update RNGs must be independent")


def _agent_config(protocol: Mapping[str, Any]) -> DrQv2Config:
    budgets = protocol["budgets"]
    runtime = protocol["runtime"]
    return DrQv2Config(
        observation_shape=tuple(_environment_value(protocol, "observation_shape", (4, 84, 84))),
        action_dim=3,
        feature_dim=_learner_value(protocol, "feature_dim"),
        hidden_dim=_learner_value(protocol, "hidden_dim"),
        replay_capacity=_budget(protocol, "replay_capacity"),
        batch_size=budgets["batch_size"],
        warmup_steps=_budget(protocol, "warmup_steps"),
        n_step=_learner_value(protocol, "n_step"),
        gamma=_learner_value(protocol, "gamma"),
        actor_learning_rate=_learner_value(protocol, "actor_lr"),
        critic_learning_rate=_learner_value(protocol, "critic_lr"),
        steering_logit_l2=_learner_value(protocol, "steering_logit_l2"),
        tau=_learner_value(protocol, "tau"),
        actor_update_frequency=_learner_value(protocol, "actor_update_frequency"),
        target_update_frequency=_learner_value(protocol, "target_update_frequency"),
        augmentation_pad=_learner_value(protocol, "padding"),
        exploration_initial_std=0.2,
        exploration_final_std=0.05,
        exploration_duration=100_000,
        target_policy_noise=_learner_value(protocol, "target_noise_std"),
        target_policy_noise_clip=_learner_value(protocol, "target_noise_clip"),
        device=runtime["training_device"],
    )


def _arm(protocol: Mapping[str, Any], source_seed: int, variant: str) -> dict[str, Any]:
    return next(
        row for row in protocol["runs"]
        if row["source_seed"] == source_seed and row["variant"] == variant
    )


def _make_update_rngs(agent: DrQv2Agent, arm: Mapping[str, Any]) -> RNGStreams:
    agent.rng = np.random.default_rng(arm["actor_rng_seed"])
    agent.replay.rng = np.random.default_rng(arm["replay_rng_seed"])
    streams = RNGStreams(arm["update_rng_seed"], agent.device)
    target_noise = torch.Generator(device=agent.device)
    target_noise.manual_seed(arm["target_noise_seed"])
    streams.target_noise = target_noise
    return streams


def _assert_online_only_fork(agent: DrQv2Agent) -> None:
    if agent.environment_steps != 131_072 or agent.gradient_steps != 0 or agent.replay.size != 0:
        raise ValueError("source fork must inherit only 131072 policy steps, not source replay or gradients")
    if abs(agent.exploration_std() - 0.05) > 1e-12:
        raise ValueError("inherited DrQ exploration schedule must start at the frozen 0.05 noise floor")


def _updates_due(additional_step: int, startup_steps: int) -> bool:
    return additional_step > startup_steps


def _expected_gradient_steps(additional_steps: int, startup_steps: int) -> int:
    return max(0, additional_steps - startup_steps)


def _online_batch(replay_sampler: TwoSourceReplay, batch_size: int) -> dict[str, Any]:
    batch = replay_sampler.sample(batch_size)
    if (len(batch.get("source_tags", ())) != batch_size
            or len(batch.get("source_indices", ())) != batch_size
            or not np.array_equal(batch.get("source"), np.zeros(batch_size, dtype=np.uint8))):
        raise ValueError("online replay did not return complete row provenance")
    batch = dict(batch)
    batch["sequence"] = np.asarray(batch["source_indices"], dtype=np.int64)
    batch["episode_id"] = np.asarray(
        [int(tag["episode_id"]) for tag in batch["source_tags"]], dtype=np.int64
    )
    return batch


def _find_geometry_sampler(env: Any) -> Any:
    sampler = getattr(env, "env", None)
    if not isinstance(sampler, CatalogGeometrySampler):
        raise RuntimeError("training environment must be TimeLimit(CatalogGeometrySampler(...))")
    return sampler


def _build_training_env(
    protocol: Mapping[str, Any], root: Path, variant: str, arm: Mapping[str, Any]
) -> Any:
    catalog = protocol["catalog"]
    return build_catalog_env(
        track_ids=_environment_value(protocol, "track_ids"),
        track_seed=arm["track_seed"],
        geometry_seed=arm["geometry_seed"],
        family_tiers=protocol["variants"][variant]["family_tiers"],
        tier_weights=protocol["variants"][variant]["tier_weights"],
        max_steps=_environment_value(protocol, "max_steps"),
        repo_root=root,
        catalog_path=catalog["path"],
        protocol_path=catalog["protocol_path"],
        expected_catalog_sha256=catalog["sha256"],
        expected_protocol_sha256=catalog["protocol_sha256"],
    )


def _empty_sample_trace(total_updates: int) -> dict[str, np.ndarray]:
    return {
        "sequence_ids": np.full((total_updates, 64), -1, dtype=np.int64),
        "episode_ids": np.full((total_updates, 64), -1, dtype=np.int64),
        "geometry_seeds": np.zeros((total_updates, 64), dtype=np.uint32),
        "track_ids": np.zeros((total_updates, 64), dtype=np.uint8),
        "families": np.full((total_updates, 64), "", dtype="U64"),
        "source_codes": np.zeros((total_updates, 64), dtype=np.uint8),
    }


def _record_sample_trace(
    trace: dict[str, np.ndarray], offset: int, batch: Mapping[str, Any],
    episode_metadata: Mapping[int, Mapping[str, Any]],
) -> None:
    if not np.array_equal(batch["source"], np.zeros(64, dtype=np.uint8)):
        raise ValueError("geometry-mix study replay must be online-only")
    trace["sequence_ids"][offset] = batch["sequence"]
    trace["episode_ids"][offset] = batch["episode_id"]
    for row, episode_id in enumerate(batch["episode_id"]):
        metadata = episode_metadata.get(int(episode_id))
        if metadata is None:
            raise ValueError(f"sampled online replay row has no reset ledger: episode {episode_id}")
        trace["geometry_seeds"][offset, row] = metadata["seed"]
        trace["track_ids"][offset, row] = metadata["track_id"]
        trace["families"][offset, row] = metadata["geometry_family"]


def _save_sample_trace(
    run_dir: Path, trace: dict[str, np.ndarray], rows: int,
    protocol_sha256: str, source_seed: int, variant: str, additional_step: int,
) -> tuple[Path, str]:
    path = run_dir / f"replay-sample-trace-step-{additional_step:09d}.npz"
    stream = io.BytesIO()
    arrays = {name: value[:rows] for name, value in trace.items()}
    arrays["study_protocol_sha256"] = np.frombuffer(protocol_sha256.encode("ascii"), dtype=np.uint8)
    arrays["source_seed"] = np.asarray([source_seed], dtype=np.int64)
    arrays["variant"] = np.frombuffer(variant.encode("ascii"), dtype=np.uint8)
    np.savez_compressed(stream, **arrays)
    _atomic_bytes(path, stream.getvalue())
    return path, _sha256(path)


def _load_sample_trace(
    path: Path, total_updates: int, protocol_sha256: str,
    source_seed: int, variant: str,
) -> tuple[dict[str, np.ndarray], int]:
    trace = _empty_sample_trace(total_updates)
    with np.load(path, allow_pickle=False) as archive:
        expected = set(trace) | {"study_protocol_sha256", "source_seed", "variant"}
        if set(archive.files) != expected:
            raise ValueError("sample trace has missing or unsupported fields")
        if archive["study_protocol_sha256"].tobytes().decode("ascii") != protocol_sha256:
            raise ValueError("sample trace belongs to another frozen study protocol")
        if int(archive["source_seed"][0]) != source_seed:
            raise ValueError("sample trace source seed differs from this run")
        if archive["variant"].tobytes().decode("ascii") != variant:
            raise ValueError("sample trace variant differs from this run")
        rows = int(archive["sequence_ids"].shape[0])
        if rows > total_updates:
            raise ValueError("sample trace exceeds the fixed gradient-update budget")
        for name, target in trace.items():
            values = np.asarray(archive[name])
            if values.shape != (rows, 64) or values.dtype != target.dtype:
                raise ValueError(f"sample trace field {name} has an invalid shape or dtype")
            target[:rows] = values
    return trace, rows


def _episode_event(transition: Any, actions: list[np.ndarray], reward: float,
                   source_seed: int, variant: str, additional_step: int) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float32)
    return {
        "event": "end",
        "episode_id": int(transition.episode_id),
        "additional_online_step": additional_step,
        "source_seed": source_seed,
        "variant": variant,
        "track_id": int(transition.info["track_id"]),
        "seed": int(transition.info["seed"]),
        "geometry_family": transition.info["geometry_family"],
        "steps": int(transition.step + 1),
        "reward": float(reward),
        "terminated": bool(transition.terminated),
        "truncated": bool(transition.truncated),
        "terminal": bool(transition.terminal),
        **{key: transition.info.get(key) for key in ("finished", "progress", "damage", "retire_reason")},
        "native_action_mean": values.mean(axis=0).tolist(),
        "native_saturation_fraction": (np.abs(values) >= 0.99).mean(axis=0).tolist(),
        "steering_abs_ge_0_46_fraction": float((np.abs(values[:, 0]) >= 0.46).mean()),
    }


def _checkpoint_record(
    agent: DrQv2Agent,
    *, protocol: Mapping[str, Any], protocol_sha256: str, source: Mapping[str, Any],
    source_seed: int, variant: str, arm: Mapping[str, Any], run_dir: Path,
    root: Path, sampler: Any, replay_sampler: TwoSourceReplay,
    rng_streams: RNGStreams, trace_sha256: str,
    trace_path: Path, trace_rows: int, additional_step: int, transition_done: bool,
    next_episode_id: int, episode_metadata: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints" / f"step-{additional_step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    total_steps = _budget(protocol, "additional_online_steps")
    boundary = bool(transition_done)
    resume_allowed = boundary and additional_step < total_steps
    trainer_state = {
        "format": TRAINER_FORMAT,
        "study_protocol_sha256": protocol_sha256,
        "source_seed": source_seed,
        "variant": variant,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "catalog_sha256": protocol["catalog"]["sha256"],
        "additional_online_steps": additional_step,
        "study_gradient_steps": agent.gradient_steps,
        "catalog_sampler_state": sampler.state_dict(),
        "online_replay_sampler_state": replay_sampler.state_dict(),
        "rng_streams": rng_streams.state_dict(),
        "global_learner_rng": learner_rng_state(agent),
        "episode_metadata": copy.deepcopy(dict(episode_metadata)),
        "sample_trace_path": trace_path.relative_to(run_dir).as_posix(),
        "sample_trace_sha256": trace_sha256,
        "sample_trace_rows": trace_rows,
        "resume_allowed": resume_allowed,
        "checkpoint_at_episode_boundary": boundary,
        "next_episode_id": next_episode_id if resume_allowed else None,
        "environment_snapshot": None,
        "resume_contract": "restart the wrapper at the next catalog-sampler episode boundary only",
    }
    source_paths = _checkpoint_source_paths(root, protocol)
    run_metadata = {
        "study_id": STUDY_ID,
        "study_protocol_sha256": protocol_sha256,
        "source_seed": source_seed,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "variant": variant,
        "checkpoint_online_step": additional_step,
        "catalog_sha256": protocol["catalog"]["sha256"],
        "family_tiers": protocol["variants"][variant]["family_tiers"],
        "tier_weights": protocol["variants"][variant]["tier_weights"],
        "rng_seeds": dict(arm),
        "reward_contract": {"raw_reward": True, "shaping": False, "normalization": False},
        "frame_skip": 4,
        "max_steps": _environment_value(protocol, "max_steps"),
        "weight_only_fork": True,
        "source_environment_steps_inherited_for_exploration_schedule": 131_072,
        "seeds": sorted(int(value) for value in _catalog_train_index(root, protocol)[0].values()),
    }
    checkpoint = agent.save_checkpoint(
        checkpoint_dir / "checkpoint.pt",
        source_paths=source_paths,
        run_metadata=run_metadata,
        trainer_state=trainer_state,
    )
    actor_path = agent.export_actor(checkpoint_dir / "actor.pt")
    manifest_path = checkpoint.with_suffix(".manifest.json")
    return {
        "source_seed": source_seed,
        "variant": variant,
        "checkpoint_online_step": additional_step,
        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_manifest_path": manifest_path.relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": _sha256(manifest_path),
        "actor_path": actor_path.relative_to(root).as_posix(),
        "actor_sha256": _sha256(actor_path),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "study_protocol_sha256": protocol_sha256,
        "study_gradient_steps": agent.gradient_steps,
        "resume_allowed": resume_allowed,
        "checkpoint_at_episode_boundary": boundary,
        "sample_trace_path": trace_path.relative_to(root).as_posix(),
        "sample_trace_sha256": trace_sha256,
        "sample_trace_rows": trace_rows,
    }


def _checkpoint_source_paths(root: Path, protocol: Mapping[str, Any]) -> list[Path]:
    declared = protocol.get(
        "source_snapshots", protocol.get("source_sha256", protocol.get("source_hashes"))
    )
    if isinstance(declared, Mapping) and declared:
        paths = []
        for relative, expected_sha in declared.items():
            path = _repo_file(root, relative, "source snapshot")
            if _sha256(path) != expected_sha:
                raise ValueError(f"frozen trainer source SHA-256 mismatch: {relative}")
            paths.append(path)
        return paths
    relative_paths = [
        "scripts/train_drq_geometry_mix.py",
        "common_adapter.py",
        "drq_v2.py",
        "train.py",
        "env_wrapper.py",
        "damage.py",
        "haic/algorithms/drq_v2/geometry_sampler.py",
        "haic/algorithms/drq_v2/teacher_study.py",
    ]
    paths = [_repo_file(root, relative, "training source") for relative in relative_paths]
    paths.extend(sorted((root / "core").rglob("*.py")))
    return paths


def _require_episode_boundary_resume(state: Mapping[str, Any]) -> None:
    if (
        state.get("resume_allowed") is not True
        or state.get("checkpoint_at_episode_boundary") is not True
        or state.get("next_episode_id") is None
    ):
        raise ValueError("resume is permitted only from a checkpoint at an episode boundary")


def _load_resume_state(
    checkpoint_path: Path,
    agent: DrQv2Agent,
    *, protocol_sha256: str, source_seed: int, variant: str,
    source: Mapping[str, Any], protocol: Mapping[str, Any], run_dir: Path,
    sampler: Any, replay_sampler: TwoSourceReplay, rng_streams: RNGStreams,
) -> tuple[dict[str, Any], dict[str, np.ndarray], int, int]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = copy.deepcopy(payload.get("trainer_state"))
    del payload
    if not isinstance(state, dict) or state.get("format") != TRAINER_FORMAT:
        raise ValueError("resume checkpoint lacks full geometry-mix trainer state")
    _require_episode_boundary_resume(state)
    identity = {
        "study_protocol_sha256": protocol_sha256,
        "source_seed": source_seed,
        "variant": variant,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "catalog_sha256": protocol["catalog"]["sha256"],
    }
    if any(state.get(key) != expected for key, expected in identity.items()):
        raise ValueError("resume checkpoint belongs to another frozen source, arm or catalog")
    additional_steps = state.get("additional_online_steps")
    total_steps = _budget(protocol, "additional_online_steps")
    warmup_steps = _budget(protocol, "warmup_steps")
    if type(additional_steps) is not int or not 0 < additional_steps < total_steps:
        raise ValueError("resume checkpoint is outside the unfinished online decision budget")
    trainer_state = agent.load_checkpoint(checkpoint_path)
    if (
        not isinstance(trainer_state, dict)
        or trainer_state.get("format") != state.get("format")
        or trainer_state.get("study_protocol_sha256") != state.get("study_protocol_sha256")
        or trainer_state.get("additional_online_steps") != state.get("additional_online_steps")
    ):
        raise ValueError("geometry-mix trainer state changed while loading checkpoint")
    if agent.environment_steps != 131_072 + additional_steps:
        raise ValueError("resume checkpoint learner steps do not match the online offset")
    expected_gradients = _expected_gradient_steps(
        additional_steps, warmup_steps
    )
    if agent.gradient_steps != expected_gradients or state.get("study_gradient_steps") != expected_gradients:
        raise ValueError("resume checkpoint gradient count differs from the frozen cadence")
    sampler.load_state_dict(state["catalog_sampler_state"])
    replay_sampler.load_state_dict(state["online_replay_sampler_state"])
    rng_streams.load_state_dict(state["rng_streams"])
    restore_learner_rng_state(agent, state["global_learner_rng"])
    trace_path = run_dir / state["sample_trace_path"]
    if _sha256(trace_path) != state["sample_trace_sha256"]:
        raise ValueError("resume sample-trace SHA-256 differs from checkpoint provenance")
    trace, trace_rows = _load_sample_trace(
        trace_path,
        total_steps - warmup_steps,
        protocol_sha256,
        source_seed,
        variant,
    )
    if trace_rows != _expected_gradient_steps(
        additional_steps, warmup_steps
    ) or trace_rows != state["sample_trace_rows"]:
        raise ValueError("resume sample trace is incomplete through the episode-boundary checkpoint")
    return state, trace, trace_rows, additional_steps


def _load_run_config(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError("resume run is missing its frozen run-config.json")
    actual = json.loads(path.read_text(encoding="utf-8"))
    normalized_expected = json.loads(json.dumps(expected, sort_keys=True, allow_nan=False))
    if actual != normalized_expected:
        raise ValueError("resume run-config differs from current protocol/source/arm identity")


def train_arm(
    protocol: dict[str, Any], protocol_path: Path, root: Path, source_seed: int,
    variant: str, run_dir: Path, resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    _validate_protocol(protocol)
    if source_seed not in (0, 1) or variant not in VARIANTS:
        raise ValueError("source seed or geometry-mix variant is outside the frozen matrix")
    protocol_sha256 = _sha256(protocol_path)
    allowed_geometry, _ = _validate_catalog(protocol, root)
    run_root = _repo_directory(root, protocol["run_root"], "study run root")
    run_dir = run_dir if run_dir.is_absolute() else root / run_dir
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(run_root.resolve()):
        raise ValueError("--run-dir must remain under the protocol's frozen run_root")
    arm = _arm(protocol, source_seed, variant)
    expected_run_dir = _repo_directory(root, arm["run_dir"], "arm run directory")
    if run_dir != expected_run_dir:
        raise ValueError("--run-dir must equal this source-seed/variant cell's frozen run_dir")
    source = next(row for row in protocol["source_actors"] if row["learner_seed"] == source_seed)
    source_checkpoint = _repo_file(root, source["checkpoint_path"], "source checkpoint")
    source_actor = _repo_file(root, source["actor_path"], "source actor")
    if source.get("checkpoint_manifest_path"):
        manifest_path = _repo_file(root, source["checkpoint_manifest_path"], "source checkpoint manifest")
        if _sha256(manifest_path) != source.get("checkpoint_manifest_sha256"):
            raise ValueError("source checkpoint manifest SHA-256 mismatch")

    config = _agent_config(protocol)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("frozen GPU training runtime is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    source_audit = audit_source_actor_pair(
        source_checkpoint,
        source_actor,
        learner_seed=source_seed,
        source_revision=source["source_revision"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )
    if source_audit["source_revision"] != source["source_revision"]:
        raise ValueError("original source revision changed after protocol freeze")
    study_seed = int(arm["actor_rng_seed"])
    agent, fork_identity = fork_from_source(
        source_checkpoint,
        source_actor,
        config,
        learner_seed=source_seed,
        study_seed=study_seed,
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_actor_sha256=source["actor_sha256"],
    )
    _assert_online_only_fork(agent)
    rng_streams = _make_update_rngs(agent, arm)
    replay_sampler = TwoSourceReplay(
        agent.replay,
        None,
        mode="online-only",
        seed=arm["replay_rng_seed"],
        online_source_id=f"online-{variant}-seed{source_seed}",
    )

    run_config = {
        "format": "haic-drq-geometry-mix-run-config-v1",
        "study_id": STUDY_ID,
        "study_protocol_path": protocol_path.relative_to(root).as_posix(),
        "study_protocol_sha256": protocol_sha256,
        "catalog": protocol["catalog"],
        "source_seed": source_seed,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "source_pair_audit": source_audit,
        "weight_only_fork": fork_identity,
        "variant": variant,
        "family_tiers": protocol["variants"][variant]["family_tiers"],
        "tier_weights": protocol["variants"][variant]["tier_weights"],
        "rng_seeds": dict(arm),
        "agent_config": config.__dict__,
        "environment": protocol.get("environment", {}),
        "budgets": protocol["budgets"],
        "online_replay_only": True,
        "teacher_data_loaded": False,
        "static_distribution_no_curriculum": True,
        "reward_contract": {"raw_reward": True, "shaping": False, "normalization": False},
        "obstacles": True,
        "runtime": protocol["runtime"],
        "training_source_sha256": {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in _checkpoint_source_paths(root, protocol)
        },
    }

    if resume_checkpoint is None:
        if run_dir.exists():
            raise FileExistsError(f"new geometry-mix run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        _atomic_bytes(run_dir / "study_protocol.json", protocol_path.read_bytes())
        _atomic_json(run_dir / "run-config.json", run_config)
    else:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        _load_run_config(run_dir / "run-config.json", run_config)
        frozen_copy = run_dir / "study_protocol.json"
        if not frozen_copy.is_file() or _sha256(frozen_copy) != protocol_sha256:
            raise ValueError("resume run does not retain the exact frozen study protocol bytes")

    env = _build_training_env(protocol, root, variant, arm)
    try:
        sampler = _find_geometry_sampler(env)
    except Exception:
        env.close()
        raise
    if sampler.catalog_sha256 != protocol["catalog"]["sha256"]:
        env.close()
        raise ValueError("training sampler catalog identity differs from the frozen protocol")
    collector = EpisodeCollector(env, action_adapter=agent.action_adapter, gamma=config.gamma)
    total_steps = _budget(protocol, "additional_online_steps")
    warmup = _budget(protocol, "warmup_steps")
    expected_updates = total_steps - warmup
    checkpoints = set(_budget(protocol, "checkpoint_steps"))
    trace, trace_rows = _empty_sample_trace(expected_updates), 0
    episode_metadata: dict[int, dict[str, Any]] = {}
    episode_reward = 0.0
    episode_actions: list[np.ndarray] = []
    latest_metrics: dict[str, float] = {}
    additional_steps = 0

    if resume_checkpoint is None:
        observation, reset_info = collector.reset()
    else:
        resume_checkpoint = resume_checkpoint.resolve()
        # Restore the full DrQ learner before sampler/RNG state and the next reset.
        state, trace, trace_rows, additional_steps = _load_resume_state(
            resume_checkpoint,
            agent,
            protocol_sha256=protocol_sha256,
            source_seed=source_seed,
            variant=variant,
            source=source,
            protocol=protocol,
            run_dir=run_dir,
            sampler=sampler,
            replay_sampler=replay_sampler,
            rng_streams=rng_streams,
        )
        episode_metadata = {
            int(key): dict(value) for key, value in state["episode_metadata"].items()
        }
        collector.episode_id = int(state["next_episode_id"]) - 1
        observation, reset_info = collector.reset()
        if int(reset_info["seed"]) not in allowed_geometry:
            env.close()
            raise ValueError("resumed catalog sampler selected a non-TRAIN geometry")

    def validate_reset(info: Mapping[str, Any]) -> None:
        seed = int(info["geometry_seed"])
        if str(seed) not in allowed_geometry:
            raise ValueError("catalog sampler selected a geometry outside the 120 TRAIN rows")
        if info.get("geometry_family") not in protocol["variants"][variant]["family_tiers"]:
            raise ValueError("catalog sampler selected an unknown TRAIN family")
        if info.get("catalog_sha256") != protocol["catalog"]["sha256"]:
            raise ValueError("catalog sampler reset metadata has a foreign catalog identity")

    validate_reset(reset_info)
    episode_metadata[int(collector.episode_id)] = {
        "seed": int(reset_info["geometry_seed"]),
        "track_id": int(reset_info["track_id"]),
        "geometry_family": str(reset_info["geometry_family"]),
    }
    started = time.monotonic()
    log_mode = "a" if resume_checkpoint is not None else "x"
    episode_path = run_dir / "episodes.jsonl"
    metrics_path = run_dir / "step-metrics.jsonl"
    candidates: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        for step in _budget(protocol, "checkpoint_steps"):
            candidate_dir = run_dir / "checkpoints" / f"step-{step:09d}"
            if candidate_dir.exists():
                checkpoint_path = candidate_dir / "checkpoint.pt"
                actor_path = candidate_dir / "actor.pt"
                manifest_path = checkpoint_path.with_suffix(".manifest.json")
                candidates.append({
                    "source_seed": source_seed,
                    "variant": variant,
                    "checkpoint_online_step": step,
                    "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "checkpoint_manifest_path": manifest_path.relative_to(root).as_posix(),
                    "checkpoint_manifest_sha256": _sha256(manifest_path),
                    "actor_path": actor_path.relative_to(root).as_posix(),
                    "actor_sha256": _sha256(actor_path),
                    "source_checkpoint_sha256": source["checkpoint_sha256"],
                    "source_actor_sha256": source["actor_sha256"],
                    "study_protocol_sha256": protocol_sha256,
                })

    try:
        with episode_path.open(log_mode, encoding="utf-8") as episodes, metrics_path.open(
            log_mode, encoding="utf-8"
        ) as step_metrics:
            reset_event = {
                "event": "resume-reset" if resume_checkpoint else "reset",
                "episode_id": collector.episode_id,
                "additional_online_step": additional_steps,
                "source_seed": source_seed,
                "variant": variant,
                **reset_info,
            }
            episodes.write(json.dumps(reset_event, sort_keys=True) + "\n")
            episodes.flush()
            transition = None
            for additional_step in range(additional_steps + 1, total_steps + 1):
                action = agent.act(observation, deterministic=False)
                transition = collector.step(action)
                agent.observe(transition)
                episode_reward += transition.reward
                episode_actions.append(transition.action.copy())

                batch = None
                update_metrics: dict[str, float] = {}
                if _updates_due(additional_step, warmup):
                    if agent.gradient_steps != additional_step - warmup - 1:
                        raise RuntimeError("learner gradient count drifted from the frozen update cadence")
                    batch = _online_batch(replay_sampler, config.batch_size)
                    _record_sample_trace(trace, trace_rows, batch, episode_metadata)
                    update_metrics = update_from_mixture(agent, batch, rng_streams)
                    if agent.gradient_steps != additional_step - warmup:
                        raise RuntimeError("one scheduled decision did not advance exactly one update")
                    trace_rows += 1
                    latest_metrics = update_metrics
                elif agent.gradient_steps != 0:
                    raise RuntimeError("startup decisions unexpectedly updated the learner")

                if transition.done:
                    if collector.current_observation is not None:
                        raise RuntimeError("terminal transition did not close the collector episode")
                    end_row = _episode_event(
                        transition, episode_actions, episode_reward, source_seed, variant, additional_step
                    )
                    episodes.write(json.dumps(end_row, sort_keys=True, allow_nan=False) + "\n")
                    episodes.flush()
                    episode_reward, episode_actions = 0.0, []

                step_record = {
                    "additional_online_step": additional_step,
                    "source_seed": source_seed,
                    "variant": variant,
                    "environment_steps": agent.environment_steps,
                    "gradient_steps": agent.gradient_steps,
                    "scheduled_gradient_steps": _expected_gradient_steps(additional_step, warmup),
                    "exploration_std": agent.exploration_std(),
                    "online_replay_size": agent.replay.size,
                    "episode_id": transition.episode_id,
                    "episode_step": transition.step,
                    "geometry_seed": transition.info["geometry_seed"],
                    "track_id": transition.info["track_id"],
                    "geometry_family": transition.info["geometry_family"],
                    "update_performed": batch is not None,
                    "update_metrics": update_metrics,
                }
                step_metrics.write(json.dumps(step_record, sort_keys=True, allow_nan=False) + "\n")
                if additional_step % 1000 == 0 or additional_step in checkpoints:
                    step_metrics.flush()
                if additional_step in checkpoints:
                    trace_path, trace_sha = _save_sample_trace(
                        run_dir, trace, trace_rows, protocol_sha256, source_seed, variant, additional_step
                    )
                    candidate = _checkpoint_record(
                        agent,
                        protocol=protocol,
                        protocol_sha256=protocol_sha256,
                        source=source,
                        source_seed=source_seed,
                        variant=variant,
                        arm=arm,
                        run_dir=run_dir,
                        root=root,
                        sampler=sampler,
                        replay_sampler=replay_sampler,
                        rng_streams=rng_streams,
                        trace_sha256=trace_sha,
                        trace_path=trace_path,
                        trace_rows=trace_rows,
                        additional_step=additional_step,
                        transition_done=bool(transition.done),
                        next_episode_id=int(transition.episode_id) + 1,
                        episode_metadata=episode_metadata,
                    )
                    candidates = [
                        row for row in candidates if row["checkpoint_online_step"] != additional_step
                    ] + [candidate]

                if transition.done and additional_step < total_steps:
                    observation, reset_info = collector.reset()
                    validate_reset(reset_info)
                    episode_metadata[int(collector.episode_id)] = {
                        "seed": int(reset_info["geometry_seed"]),
                        "track_id": int(reset_info["track_id"]),
                        "geometry_family": str(reset_info["geometry_family"]),
                    }
                    episodes.write(json.dumps({
                        "event": "reset",
                        "episode_id": collector.episode_id,
                        "additional_online_step": additional_step,
                        "source_seed": source_seed,
                        "variant": variant,
                        **reset_info,
                    }, sort_keys=True) + "\n")
                    episodes.flush()
                elif not transition.done:
                    observation = transition.next_observation

                if additional_step % 1000 == 0 or additional_step == total_steps:
                    print(json.dumps({
                        "source_seed": source_seed,
                        "variant": variant,
                        "additional_online_step": additional_step,
                        "environment_steps": agent.environment_steps,
                        "gradient_steps": agent.gradient_steps,
                        "expected_gradient_steps": _expected_gradient_steps(additional_step, warmup),
                        "exploration_std": agent.exploration_std(),
                        "online_replay_size": agent.replay.size,
                        "last_update": latest_metrics,
                        "elapsed_seconds": time.monotonic() - started,
                    }, sort_keys=True), flush=True)

            if transition is None:
                raise RuntimeError("training loop did not execute an online decision")
            incomplete_episode = None
            if not transition.done:
                actions = np.asarray(episode_actions, dtype=np.float32)
                incomplete_episode = {
                    "episode_id": transition.episode_id,
                    "track_id": int(transition.info["track_id"]),
                    "geometry_seed": int(transition.info["geometry_seed"]),
                    "geometry_family": transition.info["geometry_family"],
                    "steps": transition.step + 1,
                    "reward": episode_reward,
                    "budget_interrupted": True,
                    "native_action_mean": actions.mean(axis=0).tolist(),
                    "native_saturation_fraction": (np.abs(actions) >= 0.99).mean(axis=0).tolist(),
                    "steering_abs_ge_0_46_fraction": float((np.abs(actions[:, 0]) >= 0.46).mean()),
                }
                episodes.write(json.dumps({
                    "event": "budget-stop",
                    "additional_online_step": total_steps,
                    "source_seed": source_seed,
                    "variant": variant,
                    **incomplete_episode,
                }, sort_keys=True, allow_nan=False) + "\n")
            step_metrics.flush()
            episodes.flush()
    finally:
        env.close()

    if agent.gradient_steps != expected_updates or trace_rows != expected_updates:
        raise RuntimeError(
            f"completed {agent.gradient_steps} updates and {trace_rows} sample rows; expected {expected_updates}"
        )
    trace_path = run_dir / f"replay-sample-trace-step-{total_steps:09d}.npz"
    trace_sha256 = _sha256(trace_path)
    candidates.sort(key=lambda row: row["checkpoint_online_step"])
    _atomic_json(run_dir / "checkpoint-catalog.json", {
        "format": "haic-drq-geometry-mix-checkpoint-catalog-v1",
        "study_protocol_sha256": protocol_sha256,
        "source_seed": source_seed,
        "source_actor_sha256": source["actor_sha256"],
        "variant": variant,
        "catalog_sha256": protocol["catalog"]["sha256"],
        "candidates": candidates,
    })
    result = {
        "format": "haic-drq-geometry-mix-run-result-v1",
        "study_id": STUDY_ID,
        "study_protocol_sha256": protocol_sha256,
        "source_seed": source_seed,
        "source_revision": source["source_revision"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_actor_sha256": source["actor_sha256"],
        "variant": variant,
        "family_tiers": protocol["variants"][variant]["family_tiers"],
        "tier_weights": protocol["variants"][variant]["tier_weights"],
        "catalog_sha256": protocol["catalog"]["sha256"],
        "additional_online_steps": total_steps,
        "source_environment_steps_inherited": 131_072,
        "final_environment_steps": agent.environment_steps,
        "study_gradient_steps": agent.gradient_steps,
        "expected_gradient_steps": expected_updates,
        "online_replay_only": True,
        "teacher_data_loaded": False,
        "sample_trace_path": trace_path.relative_to(root).as_posix(),
        "sample_trace_sha256": trace_sha256,
        "sample_trace_rows": trace_rows,
        "checkpoint_catalog_path": "checkpoint-catalog.json",
        "candidates": candidates,
        "incomplete_budget_episode": incomplete_episode,
        "completed": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-file", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-seed", required=True, choices=(0, 1), type=int)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    protocol_path = args.protocol_file if args.protocol_file.is_absolute() else root / args.protocol_file
    try:
        protocol_path = protocol_path.resolve(strict=True)
        if not protocol_path.is_relative_to(root):
            raise ValueError("protocol path must be under the repository root")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        result = train_arm(
            protocol,
            protocol_path,
            root,
            args.source_seed,
            args.variant,
            args.run_dir,
            args.resume_checkpoint,
        )
    except (OSError, ValueError, KeyError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
