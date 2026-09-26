"""Verify deterministic prefixes and compare one residual option with KEEP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

from common_adapter import ActionAdapter
from drq_v2 import load_exported_actor
from haic.algorithms.drq_v2 import (
    ResidualOption,
    ResidualOptionQ,
    apply_native_residual,
    select_greedy_option,
)
from scripts.drqv2_residual_options import infer_base_and_features
from train import HaicTrack


ROOT = Path(__file__).resolve().parents[2]
INFO_KEYS = (
    "collision",
    "damage",
    "progress",
    "finish_qualified",
    "finish_qualified_time_s",
    "finish_time_s",
    "finished",
    "retire_reason",
)
SOURCE_STEPS_PATH = (
    "runs/20260924-drqv2-residual-options-prefix-branch-v1/source-trajectories.jsonl"
)
BRANCH_STEPS_PATH = (
    "runs/20260924-drqv2-residual-options-prefix-branch-v1/branch-trajectories.jsonl"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _seed_values(value, key: str = "") -> set[int]:
    found = set()
    if isinstance(value, dict):
        for child_key, child in value.items():
            if "seed" in child_key.lower():
                found.update(_integer_values(child))
            else:
                found.update(_seed_values(child, child_key))
    elif isinstance(value, list):
        for child in value:
            found.update(_seed_values(child, key))
    return found


def _integer_values(value) -> set[int]:
    if isinstance(value, bool):
        return set()
    if isinstance(value, int):
        return {value} if 0 <= value < 2**32 else set()
    if isinstance(value, list):
        return set().union(*(_integer_values(item) for item in value)) if value else set()
    if isinstance(value, dict):
        return set().union(*(_integer_values(item) for item in value.values())) if value else set()
    return set()


def audit_training_seeds(protocol: dict, protocol_path: Path) -> tuple[set[int], dict[str, str]]:
    reserved = set()
    sources = {}
    pattern = protocol["seed_audit"]["enumerate_existing_protocols"]
    for path in sorted((ROOT / "experiments").glob(Path(pattern).name)):
        if path.resolve() == protocol_path.resolve():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        reserved.update(_seed_values(document))
        sources[str(path.relative_to(ROOT))] = sha256_file(path)
    requested = set(protocol["environment"]["geometry_seeds"])
    overlap = requested & reserved
    if overlap:
        raise ValueError(f"training geometry seeds overlap recorded DrQ protocol seeds: {sorted(overlap)}")
    return reserved, sources


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _field(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return str(value)


def _obs_hash(observation: np.ndarray) -> str:
    observation = np.ascontiguousarray(observation)
    return _sha256_bytes(observation.dtype.str.encode("ascii") + str(observation.shape).encode("ascii") + observation.tobytes())


def _info_signature(info: dict) -> dict:
    return {key: _field(info.get(key)) for key in INFO_KEYS}


def _state_snapshot(env, info: dict) -> dict:
    raw_env = env.unwrapped
    car_env = env.env.env
    car = raw_env.car
    hull = car.hull
    effects = car_env.damage.effects
    return {
        "position": [float(hull.position.x), float(hull.position.y)],
        "linear_velocity": [float(hull.linearVelocity.x), float(hull.linearVelocity.y)],
        "angle": float(hull.angle),
        "angular_velocity": float(hull.angularVelocity),
        "damage": float(car_env.damage.damage),
        "damage_effects": [
            float(effects.grip_multiplier),
            float(effects.engine_multiplier),
            float(effects.steering_multiplier),
        ],
        "off_track_counter": int(car_env.off_track_counter),
        "progress": _field(info.get("progress")),
        "tile_visited_count": int(raw_env.tile_visited_count),
        "sim_time": float(raw_env.t),
    }


def _step_signature(env, observation, reward, terminated, truncated, info, action) -> dict:
    action = np.asarray(action, dtype=np.float32)
    return {
        "observation_sha256": _obs_hash(observation),
        "official_action": action.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": _info_signature(info),
        "state": _state_snapshot(env, info),
    }


def _reset_signature(env, observation, info) -> dict:
    return {
        "observation_sha256": _obs_hash(observation),
        "reset_identity": {
            "track_id": _field(info.get("track_id")),
            "seed": _field(info.get("seed")),
        },
        "state": _state_snapshot(env, info),
    }


def compare_prefix_signature(expected: dict, observed: dict) -> str | None:
    for key in ("observation_sha256", "official_action", "reward", "terminated", "truncated", "info", "state", "reset_identity"):
        if key not in expected and key not in observed:
            continue
        if expected.get(key) != observed.get(key):
            return key
    return None


def _make_env(track_id: int, seed: int, env_config: dict):
    base = HaicTrack(
        track_id,
        seed,
        env_config["max_decisions_per_episode"],
        env_config["frame_skip"],
        env_config["obstacles"],
    )
    return TimeLimit(base, max_episode_steps=env_config["max_decisions_per_episode"])


def _outcome(total_reward, decisions, terminated, truncated, info) -> dict:
    finish_time = info.get("finish_time_s")
    return {
        "decisions": int(decisions),
        "raw_reward": float(total_reward),
        "finished": bool(info.get("finished", False)),
        "finish_time_s": _field(finish_time),
        "progress": float(info.get("progress", 0.0)),
        "damage": float(info.get("damage", 0.0)),
        "retire_reason": info.get("retire_reason"),
        "collision": bool(info.get("collision", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _base_step(actor, adapter, observation, device):
    _, native = infer_base_and_features(actor, observation, device)
    return adapter.to_official(native)


def collect_base_episode(actor, adapter, cell: dict, protocol: dict, device) -> dict:
    env = _make_env(cell["track_id"], cell["geometry_seed"], protocol["environment"])
    observations_at_branches = {}
    actions = []
    signatures = []
    total_reward = 0.0
    terminated = truncated = False
    observation, reset_info = env.reset()
    reset_signature = _reset_signature(env, observation, reset_info)
    start_time = env.unwrapped.t
    last_info = reset_info
    points = protocol["procedure"]["predeclared_branch_points_after_decisions"]
    try:
        for decision_index in range(protocol["environment"]["max_decisions_per_episode"]):
            if decision_index in points:
                observations_at_branches[decision_index] = observation.copy()
            action = _base_step(actor, adapter, observation, device)
            next_observation, reward, terminated, truncated, info = env.step(action)
            signature = _step_signature(env, next_observation, reward, terminated, truncated, info, action)
            actions.append(action.copy())
            signatures.append(signature)
            total_reward += float(reward)
            observation = next_observation
            last_info = info
            if terminated or truncated:
                break
    finally:
        env.close()
    outcome = _outcome(total_reward, len(actions), terminated, truncated, last_info)
    return {
        **cell,
        "reset_signature": reset_signature,
        "official_actions": actions,
        "step_signatures": signatures,
        "branch_observation_indices": sorted(observations_at_branches),
        "branch_observations": observations_at_branches,
        "start_time_s": start_time,
        "outcome": outcome,
        "trace_sha256": _sha256_bytes(json.dumps({
            "reset": reset_signature,
            "actions": [action.tolist() for action in actions],
            "signatures": signatures,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }


def verify_prefix(actor_unused, cell: dict, source: dict, decisions: int, protocol: dict) -> dict:
    env = _make_env(cell["track_id"], cell["geometry_seed"], protocol["environment"])
    compared = 0
    try:
        observation, reset_info = env.reset()
        observed_reset = _reset_signature(env, observation, reset_info)
        mismatch = compare_prefix_signature(source["reset_signature"], observed_reset)
        if mismatch:
            return {"passed": False, "compared_decisions": 0, "mismatch_at_decision": 0, "mismatch_field": mismatch}
        for index in range(decisions):
            action = source["official_actions"][index]
            observation, reward, terminated, truncated, info = env.step(action)
            observed = _step_signature(env, observation, reward, terminated, truncated, info, action)
            mismatch = compare_prefix_signature(source["step_signatures"][index], observed)
            compared += 1
            if mismatch:
                return {"passed": False, "compared_decisions": compared, "mismatch_at_decision": index + 1, "mismatch_field": mismatch}
            if terminated or truncated:
                if index + 1 < decisions:
                    return {"passed": False, "compared_decisions": compared, "mismatch_at_decision": index + 1, "mismatch_field": "episode_ended_before_branch"}
                break
        return {"passed": True, "compared_decisions": compared, "mismatch_at_decision": None, "mismatch_field": None}
    finally:
        env.close()


@torch.inference_mode()
def _q_values(head, actor, observation, device):
    features, base = infer_base_and_features(actor, observation, device)
    q = head(
        torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0),
        torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0),
    )[0].detach().cpu().numpy()
    return features, base, q


def _load_models(protocol: dict):
    base_config = protocol["base_driver"]
    actor_path = ROOT / base_config["actor_path"]
    actor_hash = sha256_file(actor_path)
    if actor_hash != base_config["actor_sha256"]:
        raise ValueError("frozen DrQ actor hash does not match protocol")
    actor, _, _ = load_exported_actor(actor_path, device="cpu")
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    models = {}
    for label, config in protocol["option_heads"].items():
        checkpoint_path = ROOT / config["checkpoint_path"]
        if sha256_file(checkpoint_path) != config["checkpoint_sha256"]:
            raise ValueError(f"{label} option head hash does not match protocol")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("base_actor_sha256") != actor_hash:
            raise ValueError(f"{label} option head references another base actor")
        if checkpoint.get("format") != "haic-drq-v2-residual-option-q-v1":
            raise ValueError(f"unsupported {label} Q-head format")
        saved_spec = checkpoint.get("option_spec", {})
        expected_spec = {
            "duration": config["duration_decisions"],
            "steering_delta_native": config["steering_delta_native"],
            "brake_floor_official": config["brake_floor_official"],
            "intervention_margin_q_minus_keep": config["margin_q_minus_keep"],
        }
        if any(saved_spec.get(key, 0.0) != value for key, value in expected_spec.items()):
            raise ValueError(f"{label} Q-head option specification differs from protocol")
        q_head = ResidualOptionQ(checkpoint["feature_dim"], checkpoint["hidden_dim"])
        q_head.load_state_dict(checkpoint["option_q"])
        q_head.eval()
        models[label] = {"head": q_head, "config": config, "checkpoint": checkpoint}
    return actor, models, actor_hash


def _serialize_source(source: dict) -> dict:
    return {
        key: value
        for key, value in source.items()
        if key not in {"branch_observations"}
    } | {
        "official_actions": [action.tolist() for action in source["official_actions"]],
    }


def _branch_episode(
    actor,
    head,
    adapter,
    model_config,
    source,
    branch_index,
    expected_option,
    protocol,
    device,
):
    cell = {"track_id": source["track_id"], "geometry_seed": source["geometry_seed"]}
    env = _make_env(cell["track_id"], cell["geometry_seed"], protocol["environment"])
    action_changes = []
    branch_trace = []
    branch_actions = []
    option_rewards = []
    prefix_reward = float(sum(signature["reward"] for signature in source["step_signatures"][:branch_index]))
    episode_reward = prefix_reward
    last_info = {}
    terminated = truncated = False
    total_decisions = 0
    try:
        observation, reset_info = env.reset()
        mismatch = compare_prefix_signature(
            source["reset_signature"], _reset_signature(env, observation, reset_info),
        )
        if mismatch:
            return {"passed": False, "mismatch_at_decision": 0, "mismatch_field": mismatch}
        for index in range(branch_index):
            action = source["official_actions"][index]
            observation, reward, terminated, truncated, info = env.step(action)
            observed = _step_signature(env, observation, reward, terminated, truncated, info, action)
            mismatch = compare_prefix_signature(source["step_signatures"][index], observed)
            total_decisions += 1
            if mismatch:
                return {"passed": False, "mismatch_at_decision": index + 1, "mismatch_field": mismatch}
            if terminated or truncated:
                return {"passed": False, "mismatch_at_decision": index + 1, "mismatch_field": "episode_ended_before_branch"}

        _, base_at_branch, q_values = _q_values(head, actor, observation, device)
        option = select_greedy_option(
            q_values,
            intervention_margin=model_config["margin_q_minus_keep"],
        )
        if option != expected_option:
            return {
                "passed": False,
                "mismatch_at_decision": branch_index,
                "mismatch_field": "selected_option_changed_after_prefix_replay",
            }
        frozen_option = option
        duration = model_config["duration_decisions"]
        for option_step in range(duration):
            _, base_action, current_q = _q_values(head, actor, observation, device)
            if option_step == 0:
                selected = select_greedy_option(
                    current_q,
                    intervention_margin=model_config["margin_q_minus_keep"],
                )
                if selected != frozen_option:
                    raise RuntimeError("option head selected inconsistently at a parity-matched branch state")
            else:
                base_action = infer_base_and_features(actor, observation, device)[1]
            native_action = apply_native_residual(
                base_action,
                frozen_option,
                steering_delta=model_config["steering_delta_native"],
                brake_floor_official=model_config["brake_floor_official"],
            )
            changed = not np.array_equal(native_action, base_action)
            official_action = adapter.to_official(native_action)
            observation, reward, terminated, truncated, info = env.step(official_action)
            signature = _step_signature(env, observation, reward, terminated, truncated, info, official_action)
            branch_actions.append(official_action.tolist())
            action_changes.append(changed)
            option_rewards.append(float(reward))
            branch_trace.append(signature)
            episode_reward += float(reward)
            total_decisions += 1
            last_info = info
            if terminated or truncated:
                break

        while not (terminated or truncated) and total_decisions < protocol["environment"]["max_decisions_per_episode"]:
            official_action = _base_step(actor, adapter, observation, device)
            observation, reward, terminated, truncated, info = env.step(official_action)
            signature = _step_signature(env, observation, reward, terminated, truncated, info, official_action)
            branch_actions.append(official_action.tolist())
            branch_trace.append(signature)
            episode_reward += float(reward)
            total_decisions += 1
            last_info = info

        outcome = _outcome(episode_reward, total_decisions, terminated, truncated, last_info)
        branch_after_prefix = _outcome(
            episode_reward - prefix_reward,
            total_decisions - branch_index,
            terminated,
            truncated,
            last_info,
        )
        return {
            "passed": True,
            "mismatch_at_decision": None,
            "mismatch_field": None,
            "selected_option": frozen_option.name,
            "q_values": [float(value) for value in q_values],
            "q_gap_non_keep_minus_keep": float(np.max(q_values[1:]) - q_values[0]),
            "margin_q_minus_keep": model_config["margin_q_minus_keep"],
            "duration_decisions_requested": duration,
            "duration_decisions_executed": len(option_rewards),
            "option_raw_rewards": option_rewards,
            "actual_action_change_by_option_step": action_changes,
            "effective_action_change": bool(any(action_changes)),
            "prefix_reward": prefix_reward,
            "branch_and_continuation_reward": episode_reward - prefix_reward,
            "environment_decisions": total_decisions,
            "total_outcome": outcome,
            "branch_outcome": branch_after_prefix,
            "prefix_action_sha256": _sha256_bytes(json.dumps(
                [action.tolist() for action in source["official_actions"][:branch_index]],
                separators=(",", ":"),
            ).encode("utf-8")),
            "branch_actions": branch_actions,
            "branch_step_signatures": branch_trace,
        }
    finally:
        env.close()


def run(protocol_path: Path, run_dir: Path, output_path: Path) -> Path:
    protocol_path = protocol_path.resolve()
    run_dir = run_dir.resolve()
    output_path = output_path.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic run: {run_dir}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic result: {output_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("name") != "drqv2-residual-options-prefix-branch-v1":
        raise ValueError("unsupported prefix-branch protocol")
    reserved, seed_source_hashes = audit_training_seeds(protocol, protocol_path)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    actor, models, actor_hash = _load_models(protocol)
    adapter = ActionAdapter()
    expected_adapter = protocol["environment"]["action_adapter_fingerprint"]
    if adapter.spec.fingerprint != expected_adapter:
        raise ValueError("ActionAdapter fingerprint differs from frozen prefix protocol")

    run_dir.mkdir(parents=True)
    protocol_copy = run_dir / "protocol.json"
    shutil.copy2(protocol_path, protocol_copy)
    source_hashes = {
        relative: sha256_file(ROOT / relative)
        for relative in [
            "scripts/diagnose/drqv2_residual_option_prefix_branches.py",
            "scripts/drqv2_residual_options.py",
            "haic/algorithms/drq_v2/residual_options.py",
            *protocol["environment"]["official_mirror_files"],
            "common_adapter.py",
        ]
    }
    manifest = {
        "schema_version": 1,
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_copy),
        "base_actor_sha256": actor_hash,
        "option_head_sha256": {
            label: sha256_file(ROOT / config["checkpoint_path"])
            for label, config in protocol["option_heads"].items()
        },
        "source_sha256": source_hashes,
        "seed_source_protocol_sha256": seed_source_hashes,
        "reserved_seed_count_from_existing_drq_json": len(reserved),
        "action_adapter_fingerprint": adapter.spec.fingerprint,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "environment_interaction_role": "training-only deterministic prefix mechanism diagnostic",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_json(run_dir / "manifest.json", manifest)

    cells = [
        {"track_id": int(track_id), "geometry_seed": int(seed)}
        for seed in protocol["environment"]["geometry_seeds"]
        for track_id in protocol["environment"]["track_ids"]
    ]
    if len(cells) > protocol["hard_budgets"]["baseline_source_episodes_max"]:
        raise ValueError("protocol cell count exceeds the source-episode budget")
    base_started = time.perf_counter()
    sources = []
    for cell in cells:
        sources.append(collect_base_episode(actor, adapter, cell, protocol, device))
    write_jsonl(run_dir / "source-trajectories.jsonl", [_serialize_source(source) for source in sources])

    branch_points = protocol["procedure"]["predeclared_branch_points_after_decisions"]
    contexts = []
    gate_rows = []
    for source in sources:
        available = [point for point in branch_points if point in source["branch_observation_indices"]]
        if not available:
            continue
        max_prefix = max(available)
        parity = verify_prefix(actor, source, source, max_prefix, protocol)
        row = {
            "track_id": source["track_id"],
            "geometry_seed": source["geometry_seed"],
            "max_prefix_decisions": max_prefix,
            **parity,
        }
        gate_rows.append(row)
        for point in available:
            contexts.append((source, point))
    if len(contexts) > protocol["hard_budgets"]["max_branch_decision_contexts"]:
        raise ValueError("available branch contexts exceed the frozen protocol budget")
    gate_passed = bool(gate_rows) and all(row["passed"] for row in gate_rows)
    branch_rows = []
    branch_replay_gate_failed = False
    branch_cache = {}
    if gate_passed:
        for source, point in contexts:
            cell_key = f"id{source['track_id']}-seed{source['geometry_seed']}-after{point}"
            observation = source["branch_observations"][point]
            for label, model in models.items():
                features, base_action, q_values = _q_values(model["head"], actor, observation, device)
                selected = select_greedy_option(
                    q_values,
                    intervention_margin=model["config"]["margin_q_minus_keep"],
                )
                residual = apply_native_residual(
                    base_action,
                    selected,
                    steering_delta=model["config"]["steering_delta_native"],
                    brake_floor_official=model["config"]["brake_floor_official"],
                )
                effective_at_start = not np.array_equal(residual, base_action)
                prefix_recheck = verify_prefix(actor, source, source, point, protocol)
                branch = {
                    "cell_key": cell_key,
                    "track_id": source["track_id"],
                    "geometry_seed": source["geometry_seed"],
                    "branch_after_decisions": point,
                    "head": label,
                    "selected_option": selected.name,
                    "q_values": [float(value) for value in q_values],
                    "q_gap_non_keep_minus_keep": float(np.max(q_values[1:]) - q_values[0]),
                    "margin_q_minus_keep": model["config"]["margin_q_minus_keep"],
                    "effective_action_at_first_option_step": effective_at_start,
                    "prefix_replay_parity": prefix_recheck,
                }
                if not prefix_recheck["passed"]:
                    branch_replay_gate_failed = True
                    branch.update({"status": "aborted_prefix_replay_mismatch", "outcome": None})
                    branch_rows.append(branch)
                    break
                if selected is ResidualOption.KEEP:
                    branch.update({
                        "status": "selected_KEEP; no branch needed",
                        "outcome": source["outcome"],
                        "paired_finish_delta": 0,
                    })
                    branch_rows.append(branch)
                    continue
                cache_key = (
                    source["track_id"], source["geometry_seed"], point, selected.name,
                    model["config"]["duration_decisions"],
                    model["config"]["steering_delta_native"],
                    model["config"]["brake_floor_official"],
                )
                if cache_key in branch_cache:
                    outcome = branch_cache[cache_key]
                    branch.update({
                        "status": "paired_branch_reused_identical_option",
                        "shared_branch_key": outcome["shared_branch_key"],
                        "environment_decisions": 0,
                        "reused_total_outcome": outcome["total_outcome"],
                        "branch_outcome": outcome["branch_outcome"],
                        "branch_action_trace_sha256": outcome["branch_action_trace_sha256"],
                        "branch_step_trace_sha256": outcome["branch_step_trace_sha256"],
                    })
                else:
                    outcome = _branch_episode(
                        actor,
                        model["head"],
                        adapter,
                        model["config"],
                        source,
                        point,
                        selected,
                        protocol,
                        device,
                    )
                    if not outcome.get("passed"):
                        branch_replay_gate_failed = True
                        branch.update({"status": "aborted_branch_prefix_mismatch", "outcome": None, **outcome})
                        branch_rows.append(branch)
                        break
                    outcome["shared_branch_key"] = f"branch-{len(branch_cache) + 1:03d}"
                    outcome["branch_action_trace_sha256"] = _sha256_bytes(json.dumps(
                        outcome.get("branch_actions", []), separators=(",", ":"),
                    ).encode("utf-8"))
                    outcome["branch_step_trace_sha256"] = _sha256_bytes(json.dumps(
                        outcome.get("branch_step_signatures", []), sort_keys=True, separators=(",", ":"),
                    ).encode("utf-8"))
                    branch_cache[cache_key] = outcome
                    branch.update({"status": "branch_executed", **outcome})
                branch_outcome = branch.get("branch_outcome") or branch.get("reused_total_outcome")
                if branch_outcome is not None:
                    branch["paired_finish_delta"] = int(branch_outcome["finished"]) - int(source["outcome"]["finished"])
                    branch["baseline_outcome"] = source["outcome"]
                branch_rows.append(branch)
            if branch_replay_gate_failed:
                break

    all_replay_checks_passed = gate_passed and not branch_replay_gate_failed
    baseline_decisions = sum(len(source["official_actions"]) for source in sources)
    prefix_gate_decisions = sum(row["compared_decisions"] for row in gate_rows)
    branch_environment_decisions = sum(
        int(outcome.get("environment_decisions", 0)) for outcome in branch_cache.values()
    )
    total_environment_decisions = baseline_decisions + prefix_gate_decisions + branch_environment_decisions
    total_environment_episodes = len(sources) + len(gate_rows) + len(branch_cache)
    if total_environment_decisions > protocol["hard_budgets"]["total_environment_decisions_max"]:
        raise RuntimeError("prefix-branch diagnostic exceeded its predeclared decision budget")
    if total_environment_episodes > protocol["hard_budgets"]["total_environment_episodes_max"]:
        raise RuntimeError("prefix-branch diagnostic exceeded its predeclared episode budget")
    result = {
        "schema_version": 1,
        "name": protocol["name"],
        "status": (
            "prefix_gate_failed_no_counterfactual_branches"
            if not gate_passed else
            "branch_prefix_replay_failed_partial_results_invalidated_for_failed_branches"
            if branch_replay_gate_failed else
            "completed_training_only_single_intervention_diagnostic"
        ),
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(protocol_copy),
        "manifest": "manifest.json",
        "run_artifacts": str(run_dir.relative_to(ROOT)),
        "base_actor_sha256": actor_hash,
        "option_head_sha256": manifest["option_head_sha256"],
        "seed_audit": {
            "explicit_training_geometry_seeds": protocol["environment"]["geometry_seeds"],
            "reserved_or_recorded_seed_values_from_existing_drq_protocol_json": len(reserved),
            "protocol_sources_sha256": seed_source_hashes,
            "overlap_count": 0,
            "historical_global_non_use_proven": False,
        },
        "environment_usage": {
            "base_source_episodes": len(sources),
            "prefix_gate_episodes": len(gate_rows),
            "prefix_gate_passed": gate_passed,
            "prefix_gate_rows": gate_rows,
            "branch_contexts_available": len(contexts),
            "branch_rows": len(branch_rows),
            "branch_replay_gate_failed": branch_replay_gate_failed,
            "all_performed_prefix_checks_passed": all_replay_checks_passed,
            "geometry_clusters": len(protocol["environment"]["geometry_seeds"]),
            "track_geometry_cells": len(cells),
            "episode_cap": protocol["hard_budgets"]["total_environment_episodes_max"],
            "decision_cap": protocol["hard_budgets"]["total_environment_decisions_max"],
            "episodes_used": total_environment_episodes,
            "environment_decisions_used": total_environment_decisions,
            "baseline_decisions": baseline_decisions,
            "prefix_gate_decisions": prefix_gate_decisions,
            "branch_decisions": branch_environment_decisions,
        },
        "base_outcomes": [
            {key: source[key] for key in ("track_id", "geometry_seed", "trace_sha256", "outcome")}
            for source in sources
        ],
        "branches": branch_rows,
        "summary": _summarize_branches(branch_rows, sources),
        "interpretation_limits": [
            "Four geometry seeds are only four road-shape clusters; track IDs and branch points are nested observations, not independent roads.",
            "The branch compares one chosen two-decision intervention followed by the base policy against one deterministic base-policy rollout. It does not evaluate the full learned option controller as a deployed episode policy.",
            "Exact observation/action/reward/flag and exposed physical-state parity is a necessary local replay gate, not a formal proof that every hidden simulator state is identical.",
            "These seeds were selected for training-only mechanism diagnosis and are not screen, confirmation, blind, official, or generalization evidence.",
            "A selected option's paired outcome is conditional on this particular base-driver state and does not prove a general causal effect outside the branch contexts.",
        ],
        "environment_files_sha256": source_hashes,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.perf_counter() - base_started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_jsonl(run_dir / "branch-trajectories.jsonl", [
        {
            "cell_key": row["cell_key"],
            "branch_after_decisions": row["branch_after_decisions"],
            "head": row["head"],
            "selected_option": row["selected_option"],
            "branch_actions": row.get("branch_actions"),
            "branch_step_signatures": row.get("branch_step_signatures"),
            "prefix_action_sha256": row.get("prefix_action_sha256"),
            "shared_branch_key": row.get("shared_branch_key"),
            "status": row["status"],
        }
        for row in branch_rows
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    manifest["status"] = result["status"]
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_dir / "manifest.completed.json", manifest)
    return output_path


def _summarize_branches(branches: list[dict], sources: list[dict]) -> dict:
    summary = {}
    base_by_cell = {
        f"id{source['track_id']}-seed{source['geometry_seed']}": source["outcome"]
        for source in sources
    }
    for label in ("v1", "v2"):
        rows = [row for row in branches if row.get("head") == label]
        executed = [row for row in rows if row.get("status") in {
            "branch_executed", "paired_branch_reused_identical_option",
        }]
        changes = [row for row in rows if row.get("paired_finish_delta") is not None]
        summary[label] = {
            "branch_context_rows": len(rows),
            "effective_intervention_branches": len(executed),
            "selected_keep_count": sum(
                row.get("selected_option") == ResidualOption.KEEP.name
                for row in rows
            ),
            "first_option_action_ineffective_count": sum(
                not row.get("effective_action_at_first_option_step", True)
                for row in rows if row.get("selected_option") != ResidualOption.KEEP.name
            ),
            "prefix_replay_failures": sum(not row.get("prefix_replay_parity", {}).get("passed", False) for row in rows),
            "finish_delta_positive_count": sum(row.get("paired_finish_delta", 0) > 0 for row in changes),
            "finish_delta_zero_count": sum(row.get("paired_finish_delta", 0) == 0 for row in changes),
            "finish_delta_negative_count": sum(row.get("paired_finish_delta", 0) < 0 for row in changes),
            "mean_progress_delta_for_executed_branches": (
                float(np.mean([
                    row["branch_outcome"]["progress"] - base_by_cell[row["cell_key"].rsplit("-after", 1)[0]]["progress"]
                    for row in executed
                ])) if executed else None
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/drqv2-residual-options-prefix-branch-v1.json"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/20260924-drqv2-residual-options-prefix-branch-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluations/drqv2-residual-options-prefix-branch-v1.json"),
    )
    args = parser.parse_args()
    output = run(args.protocol, args.run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
