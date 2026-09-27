"""Frozen r6 encoder/head attribution on 11 reused TRAIN-DIAGNOSTIC source wins.

Recover only the first 20 frames by replaying archived SOURCE official actions.
No hybrid action is ever sent to an environment; no optimizer is constructed.
The v2 replay saved observation SHA-256 values, not the observation tensors.
Run from the repository root with the pinned CPU21 interpreter and -B.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common_adapter import EpisodeCollector
from scripts.diagnose_drq_geometry_mix import _find_wrapper, _road_hash, _road_sample, build_environment
from scripts.diagnose_drq_geometry_regression import CATALOG_SHA, MANIFEST_SHA, PROTOCOL_SHA, ROOT, RUN, VARIANTS, digest
from scripts.diagnose_drq_source_states import _model, _preflight, _state_sha


N = 20
THRESHOLD = 0.10
OUTPUT_BASE = ROOT / "runs/20260926-drqv2-retention-r7/hybrid"
REPLAY = RUN / "source-state-replay-v2"
REPLAY_PREFLIGHT_SHA = "b2fe774cc756a39fc2a6c9d586f7d11bab0085f9b98a012cb2a3e00cc14be431"
REPLAY_RESULT_SHA = "2c1e1a3ce21137c00bc2042d069a9a410b77f92eb76bac3334c2c548888c07cc"
REPLAY_SCRIPT_SHA = "c1d842c9d6896ce78fffc2b2608b7d15771b49660a2d0abac03ebad91960fdd5"
ACTION_NAMES = ("source", "r6", "source_encoder_r6_head", "r6_encoder_source_head")


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise ValueError(explanation)


def pinned_json(root: Path, relative: Path, expected: str) -> dict:
    path = root / relative
    require(path.is_file() and not path.is_symlink() and digest(path) == expected, f"SHA mismatch: {relative}")
    return json.loads(path.read_text(encoding="utf-8"))


def reference_index(result: dict, selected: list[dict]) -> dict[tuple[int, int, int, str], dict]:
    required = {(row["source_learner_seed"], row["geometry_seed"]) for row in selected}
    require(len(required) == 11 and sum(seed == 0 for seed, _ in required) == 6, "wrong source-success inventory")
    require(result["driven_decisions"] == 5998 and result["new_policy_rollouts"] == 0, "wrong v2 replay scope")
    require(result["same_source_observation_records"] == 1998 and len(result["states"]) == 1998,
            "incomplete v2 observation hashes")
    index = {}
    for row in result["states"]:
        key = (row["source_seed"], row["geometry_seed"], row["decision"], row["variant"])
        require((key[0], key[1]) in required and key[3] in VARIANTS and key not in index,
                "unexpected or duplicate v2 observation")
        require(1 <= key[2] <= next(s["steps"] for s in selected
                                      if (s["source_learner_seed"], s["geometry_seed"]) == key[:2]),
                "v2 observation outside the original episode")
        index[key] = row
    sampled = {(seed, road, decision) for seed, road, decision, _ in index}
    require(len(index) == 3 * len(sampled) and len(sampled) == 666 and all(
        all((seed, road, decision, variant) in index for variant in VARIANTS)
        for seed, road, decision in sampled), "v2 sampled-state coverage is incomplete")
    for source in selected:
        seed, road = source["source_learner_seed"], source["geometry_seed"]
        for decision in range(1, N + 1):
            for variant in VARIANTS:
                require((seed, road, decision, variant) in index, "missing early v2 hash")
    return index


def check_step(trace: dict, index: int, reward: float, terminated: bool, truncated: bool,
               info: dict, off_track: int) -> None:
    """Match archived post-action telemetry exactly, never just within tolerance."""
    terminal = EpisodeCollector._is_terminal(bool(terminated), bool(truncated), info)
    expected = {
        "reward": np.float32(reward), "progress": np.float32(info["progress"]),
        "damage": np.float32(info["damage"]), "terminated": bool(terminated),
        "truncated": bool(truncated), "terminal": bool(terminal),
        "finished": bool(info.get("finished")), "collision": bool(info.get("collision")),
        "finish_qualified": bool(info.get("finish_qualified")),
        "retirement": str(info.get("retire_reason") or ""), "off_track_counter": int(off_track),
    }
    for key, actual in expected.items():
        require(actual == trace[key][index], f"source parity mismatch at decision {index + 1}: {key}")
    crossing = info.get("finish_time_s")
    expected_crossing = trace["finish_crossing_time_s"][index]
    require((crossing is None and np.isnan(expected_crossing)) or
            (crossing is not None and float(crossing) == float(expected_crossing)),
            f"source finish time mismatch at decision {index + 1}")


def recover(root: Path, source: dict, references: dict, road: dict, source_model: dict) -> np.ndarray:
    """Recorded official-action parity for the *whole* source episode; keep 20 inputs."""
    trace_path = root / RUN / "train-diagnostic" / source["trace_path"]
    require(digest(trace_path) == source["trace_sha256"], "archived source trace changed")
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    seed, geometry = source["source_learner_seed"], source["geometry_seed"]
    require(len(trace["reward"]) == source["steps"] and source["steps"] >= N, "wrong trace length")
    require(np.all(trace["source_actor_sha256"] == source["actor_sha256"]) and
            np.all(trace["actor_sha256"] == source["actor_sha256"]) and
            np.all(trace["checkpoint_sha256"] == source_model["checkpoint_sha256"]), "wrong trace actor")
    env = build_environment(geometry, 1200)
    frames = []
    try:
        observation, info = env.reset(seed=None)
        spec = source_model["observation_spec"]
        observation = spec.validate(observation)
        wrapper = _find_wrapper(env, ("off_track_counter", "max_off_track_steps", "warmup_steps"))
        require(wrapper.max_off_track_steps == 100 and wrapper.warmup_steps == 50, "wrong wrapper")
        require((info.get("seed"), info.get("track_id")) == (geometry, 1), "wrong road or track")
        require(_road_hash(env.unwrapped.track) == road["road_coordinate_sha256"], "road coordinates changed")
        sparse = {int(decision): i for i, decision in enumerate(trace["sparse_step"])}
        require(0 in sparse and source["steps"] in sparse, "missing sparse road parity")
        for decision in range(1, source["steps"] + 1):
            i = decision - 1
            # Compare all v2 sampled hashes, including samples outside the cache.
            record = references.get((seed, geometry, decision, VARIANTS[0]))
            if record is not None:
                frame_sha = _state_sha(observation)
                with torch.inference_mode():
                    action = source_model["actor"](torch.as_tensor(observation).unsqueeze(0)).squeeze(0).numpy()
                require(np.max(np.abs(action - trace["native_action"][i])) <= 1e-5,
                        f"source policy mismatch at {seed}/{geometry}/{decision}")
                for variant in VARIANTS:
                    prior = references[(seed, geometry, decision, variant)]
                    require(prior["observation_sha256"] == frame_sha and
                            prior["source_trace_sha256"] == source["trace_sha256"] and
                            np.max(np.abs(action - prior["source_native_action"])) <= 1e-5,
                            f"v2 pixel/action mismatch at {seed}/{geometry}/{decision}/{variant}")
            if decision <= N:
                # v2 hashes the original float32 [0, 1] input, not a uint8
                # quantization used separately by the training replay buffer.
                frames.append(observation.copy())
            native, official = trace["native_action"][i], trace["official_action"][i]
            require(np.max(np.abs(source_model["adapter"].to_official(native) - official)) <= 1e-6,
                    f"official adapter mismatch at {seed}/{geometry}/{decision}")
            next_observation, reward, terminated, truncated, step_info = env.step(official)
            step_info = {**info, **dict(step_info or {})}
            require((step_info.get("seed"), step_info.get("track_id")) == (geometry, 1), "road changed")
            check_step(trace, i, reward, terminated, truncated, step_info, wrapper.off_track_counter)
            if decision in sparse:
                measured = _road_sample(env, decision)
                j = sparse[decision]
                for field, name in (("nearest_point_index", "sparse_nearest_point_index"),
                                    ("nearest_point_fraction", "sparse_nearest_point_fraction"),
                                    ("nearest_distance_m", "sparse_nearest_distance_m"),
                                    ("speed_m_s", "sparse_speed_m_s")):
                    value = int(measured[field]) if field == "nearest_point_index" else np.float32(measured[field])
                    require(value == trace[name][j], f"sparse road mismatch at {seed}/{geometry}/{decision}: {field}")
            require(bool(terminated or truncated) == (decision == source["steps"]), "wrong terminal decision")
            if decision < source["steps"]:
                observation = spec.validate(next_observation)
        require(bool(step_info.get("finished")) and len(frames) == N, "source did not finish with 20 inputs")
    finally:
        env.close()
    require(digest(trace_path) == source["trace_sha256"], "source trace changed during replay")
    return np.stack(frames)


def actor_actions(source: torch.nn.Module, variant: torch.nn.Module, observation: torch.Tensor) -> torch.Tensor:
    """Swap exactly encoder vs (trunk + policy), retaining the tanh native contract."""
    source_feature = source.encoder(observation)
    variant_feature = variant.encoder(observation)
    return torch.stack((
        torch.tanh(source.policy(source.trunk(source_feature))),
        torch.tanh(variant.policy(variant.trunk(variant_feature))),
        torch.tanh(variant.policy(variant.trunk(source_feature))),
        torch.tanh(source.policy(source.trunk(variant_feature))),
    ), dim=1)


def critic_values(source: list[torch.nn.Module], variant: list[torch.nn.Module],
                  observation: torch.Tensor, actions: torch.Tensor) -> dict[str, np.ndarray]:
    """Swap corresponding Q1/Q2 encoders and Q heads, never cross Q1 with Q2."""
    require(len(source) == len(variant) == 2 and actions.shape == (len(observation), 4, 3),
            "twin critic or action shape differs")
    features = {"source": [critic.encoder(observation) for critic in source],
                "r6": [critic.encoder(observation) for critic in variant]}
    values = {}
    for name, encoder_role, head_role in (
        ("source", "source", "source"), ("r6", "r6", "r6"),
        ("source_encoder_r6_head", "source", "r6"),
        ("r6_encoder_source_head", "r6", "source"),
    ):
        heads = source if head_role == "source" else variant
        values[name] = np.stack([np.stack([
            critic.q(torch.cat((features[encoder_role][j], actions[:, k]), dim=-1)).squeeze(-1).numpy()
            for k in range(len(ACTION_NAMES))], axis=1)
            for j, critic in enumerate(heads)], axis=1)
        require(values[name].shape == (len(observation), 2, len(ACTION_NAMES)) and
                np.isfinite(values[name]).all(), "invalid frozen twin-critic Q")
    return values


def axis_stats(differences: np.ndarray) -> dict:
    require(differences.shape == (N, 3) and np.isfinite(differences).all() and
            np.all(differences >= 0), "invalid 20 x 3 axis differences")
    return {axis: {"mean": float(np.mean(differences[:, i], dtype=np.float64)),
                   "median": float(np.median(differences[:, i])),
                   "ge_0_10_count": int(np.count_nonzero(differences[:, i] >= THRESHOLD)),
                   "n": N} for i, axis in enumerate(("steer", "gas", "brake"))}


def offline_cell(frames: np.ndarray, source: dict, variant: str, models: dict,
                 references: dict) -> dict:
    seed, road = source["source_learner_seed"], source["geometry_seed"]
    original, updated = models["unchanged-source"], models[variant]
    require(frames.shape == (N, 4, 84, 84) and frames.dtype == np.float32, "unverified cache shape/dtype")
    tensor = torch.as_tensor(frames)
    with torch.inference_mode():
        action_tensor = actor_actions(original["actor"], updated["actor"], tensor)
        q = critic_values(original["critics"], updated["critics"], tensor, action_tensor)
        actions = action_tensor.numpy()
    official = np.asarray([[original["adapter"].to_official(action) for action in row] for row in actions])
    require(actions.shape == (N, 4, 3) and official.shape == actions.shape and
            np.isfinite(actions).all() and np.isfinite(official).all(), "non-finite actions")
    observations = []
    for i, frame in enumerate(frames):
        decision = i + 1
        prior = references[(seed, road, decision, variant)]
        require(_state_sha(frame) == prior["observation_sha256"], "cache/v2 pixel hash mismatch")
        require(np.max(np.abs(actions[i, 0] - prior["source_native_action"])) <= 1e-5 and
                np.max(np.abs(actions[i, 1] - prior["variant_native_action"])) <= 1e-5,
                "source or r6 actor control differs from v2")
        for role, label in (("source", "source_critic"), ("r6", "variant_critic")):
            for j, head in enumerate(("q1", "q2")):
                for k, action_label in enumerate(("source_action", "variant_action")):
                    require(abs(float(q[role][i, j, k]) - prior[label][head][action_label]) <= 1e-3,
                            "frozen Q baseline differs from v2")
        observations.append({"decision": decision, "observation_sha256": prior["observation_sha256"],
                             "native_actions": {name: actions[i, j].tolist() for j, name in enumerate(ACTION_NAMES)},
                             "official_actions": {name: official[i, j].tolist() for j, name in enumerate(ACTION_NAMES)},
                             "q_by_critic": {role: {head: {name: float(q[role][i, j, k])
                                                          for k, name in enumerate(ACTION_NAMES)}
                                                      for j, head in enumerate(("q1", "q2"))}
                                             for role in ACTION_NAMES}})
    differences = {}
    for j, name in enumerate(ACTION_NAMES[1:], start=1):
        differences[name] = {
            baseline: {"native": axis_stats(np.abs(actions[:, j] - actions[:, k])),
                       "official": axis_stats(np.abs(official[:, j] - official[:, k]))}
            for baseline, k in (("source", 0), ("r6", 1)) if j != k
        }
    return {"source_seed": seed, "geometry_seed": road, "track_id": 1, "family": source["family"],
            "variant": variant, "n": N, "source_trace_sha256": source["trace_sha256"],
            "source_checkpoint_sha256": original["checkpoint_sha256"],
            "r6_checkpoint_sha256": updated["checkpoint_sha256"],
            "abs_axis_differences": differences,
            "q_ranking_vs_source_action_count": {role: {
                head: {name: int(np.count_nonzero(values[:, j, k] > values[:, j, 0]))
                       for k, name in enumerate(ACTION_NAMES[1:], start=1)}
                for j, head in enumerate(("q1", "q2"))} | {
                    "twin_min": {name: int(np.count_nonzero(
                        np.min(values[:, :, k], axis=1) > np.min(values[:, :, 0], axis=1)))
                        for k, name in enumerate(ACTION_NAMES[1:], start=1)}}
                for role, values in q.items()},
            "observations": observations}


def run(root: Path, out: Path) -> dict:
    require(out.parent.resolve().is_relative_to(OUTPUT_BASE.resolve()) and
            out.resolve().is_relative_to(OUTPUT_BASE.resolve()) and not out.exists() and
            not OUTPUT_BASE.is_symlink() and not out.parent.is_symlink(),
            "output must be a NEW directory under the owned hybrid path")
    require(digest(root / "scripts/diagnose_drq_source_states.py") == REPLAY_SCRIPT_SHA,
            "original parity replay implementation changed")
    old_preflight = pinned_json(root, REPLAY / "preflight.json", REPLAY_PREFLIGHT_SHA)
    old_result = pinned_json(root, REPLAY / "result.json", REPLAY_RESULT_SHA)
    protocol, inputs, selected = _preflight(root)
    require(old_preflight["protocol_sha256"] == old_result["protocol_sha256"] == PROTOCOL_SHA and
            old_result["catalog_sha256"] == old_preflight["catalog_sha256"] == CATALOG_SHA and
            old_result["diagnostic_manifest_sha256"] == old_preflight["diagnostic_manifest_sha256"] == MANIFEST_SHA and
            old_result["preflight_sha256"] == REPLAY_PREFLIGHT_SHA and
            old_result["diagnostic_script_sha256"] == REPLAY_SCRIPT_SHA and
            old_result["selected"] == old_preflight["selected"] == [
                {"source_seed": r["source_learner_seed"], "geometry_seed": r["geometry_seed"],
                 "recorded_steps": r["steps"], "source_trace_sha256": r["trace_sha256"]} for r in selected],
            "v2 provenance or selected cells changed")
    require(sum(row["steps"] for row in selected) == 5998 and all(row["steps"] >= N for row in selected),
            "recorded parity budget differs")
    references = reference_index(old_result, selected)
    expected_checkpoints = {(r["source_seed"], r["variant"]): r for r in old_result["parameter_shifts"]}
    require(len(expected_checkpoints) == 6, "missing v2 checkpoint attribution")
    for r in selected:
        for variant in VARIANTS:
            cp = expected_checkpoints[(r["source_learner_seed"], variant)]
            require(cp["source_checkpoint_sha256"] == next(
                item["checkpoint_sha256"] for item in protocol["source_actors"]
                if item["source_seed"] == r["source_learner_seed"]), "source checkpoint identity changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    provenance = {"format": "drq-r7-frozen-hybrid-first20-v1", "n": N,
                  "scope": "reused TRAIN-DIAGNOSTIC source successes only; offline frozen hybrid scoring",
                  "source_replay_preflight_sha256": REPLAY_PREFLIGHT_SHA,
                  "source_replay_result_sha256": REPLAY_RESULT_SHA,
                  "source_replay_script_sha256": REPLAY_SCRIPT_SHA,
                  "protocol_sha256": PROTOCOL_SHA, "catalog_sha256": CATALOG_SHA,
                  "diagnostic_manifest_sha256": MANIFEST_SHA,
                  "runtime_source_sha256": protocol["source_sha256"],
                  "script_sha256": digest(Path(__file__)), "selected": old_result["selected"]}
    (out / "preflight.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    all_frames, cells = [], []
    try:
        for seed in (0, 1):
            source_record = next(x for x in protocol["source_actors"] if x["source_seed"] == seed)
            models = {"unchanged-source": _model(root, source_record, source_record)}
            for variant in VARIANTS:
                result = json.loads((root / RUN / f"learner-{seed}-{variant}" / "result.json").read_text())
                final = result["candidates"][-1]
                require(final["checkpoint_online_step"] == 32768 and
                        final["checkpoint_sha256"] == expected_checkpoints[(seed, variant)]["variant_checkpoint_sha256"] and
                        final["source_checkpoint_sha256"] == source_record["checkpoint_sha256"],
                        "r6 final checkpoint differs from original v2")
                models[variant] = _model(root, final, final)
            for source in (r for r in selected if r["source_learner_seed"] == seed):
                frames = recover(root, source, references, inputs["road"][source["geometry_seed"]],
                                 models["unchanged-source"])
                all_frames.append(frames)
                for variant in VARIANTS:
                    cells.append(offline_cell(frames, source, variant, models, references))
            del models
            gc.collect()
        cache = np.stack(all_frames)
        require(cache.shape == (11, N, 4, 84, 84) and cache.dtype == np.float32 and len(cells) == 33,
                "cache or cell inventory incomplete")
        # Persist only after every episode has passed full parity and every actor/Q
        # baseline has matched the original v2 analysis.
        np.savez_compressed(out / "first20-observations.npz", observations=cache,
                            source_seed=np.asarray([s["source_learner_seed"] for s in selected], dtype=np.int8),
                            geometry_seed=np.asarray([s["geometry_seed"] for s in selected], dtype=np.int64))
        cache_sha = digest(out / "first20-observations.npz")
        result = {**provenance, "preflight_sha256": digest(out / "preflight.json"),
                  "cache_path": "first20-observations.npz", "cache_sha256": cache_sha,
                  "cache_frame_sha256": [[_state_sha(frame) for frame in frames] for frames in cache],
                  "source_episodes_replayed": 11, "recorded_official_decisions_replayed": 5998,
                  "parity_sampled_observation_instances": 666, "cached_observation_instances": 220,
                  "hybrid_environment_decisions": 0, "learner_updates": 0,
                  "q_note": "Four frozen corresponding-Q encoder/head combinations evaluated on four actions per same source state; Q levels across critics or hybrid feature/head bases are not calibrated, and within-critic action rankings are not counterfactual returns",
                  "cells": cells}
        with (out / "result.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return {"result": str(out / "result.json"), "result_sha256": digest(out / "result.json"),
                "cache_sha256": cache_sha, "source_episodes": 11, "cached_frames": 220}
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"error": repr(error),
            "completed_source_episodes": len(all_frames), "preflight_sha256": digest(out / "preflight.json")}, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_BASE / "first20-v1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(ROOT, args.output_root.absolute()), sort_keys=True))


if __name__ == "__main__":
    main()
