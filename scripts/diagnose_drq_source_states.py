"""Minimal parity-checked source-action replay for offline actor/critic attribution.

Only the eleven previously completed, canonical source-success TRAIN-DIAGNOSTIC
episodes are replayed. Alternative actors are never applied to the environment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from common_adapter import EpisodeCollector
from drq_v2 import DrQCritic, load_exported_actor
from scripts.diagnose_drq_geometry_mix import _find_wrapper, _road_hash, _road_sample, build_environment
from scripts.diagnose_drq_geometry_regression import (
    CATALOG_SHA, MANIFEST_SHA, PROTOCOL, PROTOCOL_SHA, ROOT, RUN, VARIANTS, digest,
)


STATE_STRIDE = 25
START_STATES = 20
END_STATES = 20
LOCAL_STEERING_DELTA = 0.1
MIN_HEADROOM_BYTES = 12 * 1024**3


def sample_decisions(length: int) -> set[int]:
    assert 1 <= length <= 1200
    return set(range(1, min(length, START_STATES) + 1)) | set(range(STATE_STRIDE, length + 1, STATE_STRIDE)) | set(
        range(max(1, length - END_STATES + 1), length + 1))


def _state_sha(observation: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()


def _relative_parameter_shift(reference: torch.nn.Module, updated: torch.nn.Module) -> dict:
    original, candidate = reference.state_dict(), updated.state_dict()
    assert original.keys() == candidate.keys()
    delta, norm, other, dot = 0.0, 0.0, 0.0, 0.0
    for key in original:
        a = original[key].detach().cpu().double().flatten()
        b = candidate[key].detach().cpu().double().flatten()
        delta += float((a - b).square().sum())
        norm += float(a.square().sum())
        other += float(b.square().sum())
        dot += float((a * b).sum())
    assert norm > 0 and other > 0
    return {"relative_l2": math.sqrt(delta / norm), "cosine": dot / math.sqrt(norm * other)}


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    norm = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(torch.dot(a, b) / norm) if norm else 0.0


def _critic_values(critic: DrQCritic, encoded: torch.Tensor,
                   source_action: torch.Tensor, variant_action: torch.Tensor) -> dict:
    def q(action: torch.Tensor) -> float:
        return float(critic.q(torch.cat((encoded, action), dim=-1)).squeeze().item())
    return {"source_action": q(source_action), "variant_action": q(variant_action)}


def _road_state(env) -> dict:
    raw = env.unwrapped
    points = np.asarray(raw.track, dtype=np.float64)[:, 2:4]
    position = np.asarray(raw.car.hull.position, dtype=np.float64)
    velocity = np.asarray(raw.car.hull.linearVelocity, dtype=np.float64)
    distances = np.linalg.norm(points - position, axis=1)
    index = int(np.argmin(distances))
    tangent = points[(index + 1) % len(points)] - points[index]
    heading = float(raw.car.hull.angle) + np.pi / 2
    road_heading = float(np.arctan2(tangent[1], tangent[0]))
    return {"x_m": float(position[0]), "y_m": float(position[1]),
            "centerline_distance_m": float(distances[index]), "nearest_point_index": index,
            "speed_m_s": float(np.linalg.norm(velocity)),
            "road_relative_heading_rad": float(np.arctan2(np.sin(heading - road_heading),
                                                           np.cos(heading - road_heading)))}


def _model(root: Path, actor_record: dict, checkpoint_record: dict) -> dict:
    actor_path = root / actor_record["actor_path"]
    cp_path = root / checkpoint_record["checkpoint_path"]
    assert digest(actor_path) == actor_record["actor_sha256"]
    assert digest(cp_path) == checkpoint_record["checkpoint_sha256"]
    assert digest(root / checkpoint_record["checkpoint_manifest_path"]) == checkpoint_record["checkpoint_manifest_sha256"]
    actor, adapter, observation_spec = load_exported_actor(actor_path, device="cpu")
    actor.eval()
    # Trusted, pinned checkpoint; tensor storages are mapped, not copied into a
    # new training agent. Optimizer, replay and saved RNG are NEVER restored.
    payload = torch.load(str(cp_path), map_location="cpu", mmap=True, weights_only=False)
    assert payload["format"] == "haic-drq-v2-checkpoint-v1"
    config = payload["config"]
    assert config["augmentation_pad"] == 4 and config.get("steering_logit_l2", 0.0) == 0.0
    assert all(torch.equal(actor.state_dict()[k], value) for k, value in payload["actor"].items())
    critics = []
    for key in ("critic_one", "critic_two"):
        critic = DrQCritic(input_channels=4, action_dim=3,
                           feature_dim=config["feature_dim"], hidden_dim=config["hidden_dim"])
        critic.load_state_dict(payload[key])
        critic.eval()
        critics.append(critic)
    del payload
    gc.collect()
    return {"actor": actor, "critics": critics, "adapter": adapter,
            "observation_spec": observation_spec,
            "actor_sha256": actor_record["actor_sha256"],
            "checkpoint_sha256": checkpoint_record["checkpoint_sha256"]}


def _preflight(root: Path) -> tuple[dict, dict, list[dict]]:
    assert sys.executable == "/tmp/kilo/haic-cpu21/bin/python"
    assert torch.__version__ == "2.1.0+cpu" and not torch.cuda.is_available()
    memory_max = int(Path("/sys/fs/cgroup/memory.max").read_text())
    memory_current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    assert memory_max - memory_current >= MIN_HEADROOM_BYTES, "insufficient memory headroom"
    assert digest(root / PROTOCOL) == PROTOCOL_SHA
    protocol = json.loads((root / PROTOCOL).read_text())
    assert digest(root / protocol["catalog_path"]) == CATALOG_SHA
    for name, sha in protocol["source_sha256"].items():
        assert digest(root / name) == sha, f"r6 pinned runtime drift: {name}"
    diagnostic = root / RUN / "train-diagnostic"
    assert digest(diagnostic / "manifest.json") == MANIFEST_SHA
    manifest = json.loads((diagnostic / "manifest.json").read_text())
    assert manifest["role"] == "TRAIN-DIAGNOSTIC" and manifest["protocol_sha256"] == PROTOCOL_SHA
    episodes_path = diagnostic / "episodes.jsonl"
    assert digest(episodes_path) == manifest["files_sha256"]["episodes.jsonl"]
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    canonical = {(r["source_learner_seed"], r["geometry_seed"], r["arm"]): r
                 for r in rows if r["repeat"] == 0}
    assert len(rows) == 256 and len(canonical) == 128
    selected = [r for r in canonical.values() if r["arm"] == "unchanged-source" and r["finished"]]
    assert len(selected) == 11 and sorted([r["source_learner_seed"] for r in selected]).count(0) == 6
    catalog = json.loads((root / protocol["catalog_path"]).read_text())
    diagnostic_roads = {r["geometry_seed"]: r for r in catalog["train_diagnostic"]}
    assert {r["geometry_seed"] for r in selected} <= set(diagnostic_roads)
    assert not {r["geometry_seed"] for r in selected} & set(protocol["training_pool"]["geometry_seeds"])
    for r in selected:
        path = diagnostic / r["trace_path"]
        assert path.is_file() and digest(path) == r["trace_sha256"] == manifest["files_sha256"][r["trace_path"]]
        assert 1 <= r["steps"] <= 1200 and r["track_id"] == 1
    for run in protocol["runs"]:
        result = json.loads((root / run["run_dir"] / "result.json").read_text())
        assert result["completed"] is True and result["study_gradient_steps"] == 22768
        assert result["study_protocol_sha256"] == PROTOCOL_SHA and result["catalog_sha256"] == CATALOG_SHA
        assert result["source_seed"] == run["source_seed"] and result["variant"] == run["variant"]
        role = result["candidates"][-1]
        assert role["checkpoint_online_step"] == 32768
        assert digest(root / role["checkpoint_path"]) == role["checkpoint_sha256"]
        assert digest(root / role["actor_path"]) == role["actor_sha256"]
        assert digest(root / role["checkpoint_manifest_path"]) == role["checkpoint_manifest_sha256"]
    return protocol, {"manifest": manifest, "canonical": canonical, "road": diagnostic_roads}, sorted(
        selected, key=lambda r: (r["source_learner_seed"], r["geometry_seed"]))


def _analyze_state(observation: np.ndarray, models: dict, source: dict,
                   variants: tuple[str, ...], decision: int, road_state: dict,
                   source_action_recorded: np.ndarray, sample_local: bool) -> list[dict]:
    tensor = torch.as_tensor(observation).unsqueeze(0)
    original = models["unchanged-source"]
    original_action = original["actor"](tensor)
    assert np.max(np.abs(original_action.squeeze(0).numpy() - source_action_recorded)) <= 1e-5
    original_features = original["actor"].encoder(tensor)
    original_critic_features = [critic.encoder(tensor) for critic in original["critics"]]
    result = []
    for variant in variants:
        updated = models[variant]
        candidate_action = updated["actor"](tensor)
        candidate_features = updated["actor"].encoder(tensor)
        candidate_critic_features = [critic.encoder(tensor) for critic in updated["critics"]]
        original_q = [_critic_values(critic, features, original_action, candidate_action)
                      for critic, features in zip(original["critics"], original_critic_features)]
        updated_q = [_critic_values(critic, features, original_action, candidate_action)
                     for critic, features in zip(updated["critics"], candidate_critic_features)]
        detail = {
            "source_seed": source["source_learner_seed"], "geometry_seed": source["geometry_seed"],
            "family": source["family"], "variant": variant, "decision": decision,
            "source_trace_sha256": source["trace_sha256"],
            "observation_sha256": _state_sha(observation),
            "road_pre_action": road_state,
            "source_native_action": original_action.squeeze(0).tolist(),
            "variant_native_action": candidate_action.squeeze(0).tolist(),
            "action_linf_difference": float(torch.max(torch.abs(original_action - candidate_action))),
            "source_actor_encoder_cosine": _cosine(original_features, candidate_features),
            "source_q1_encoder_cosine": _cosine(original_critic_features[0], candidate_critic_features[0]),
            "source_q2_encoder_cosine": _cosine(original_critic_features[1], candidate_critic_features[1]),
            "source_critic": {"q1": original_q[0], "q2": original_q[1]},
            "variant_critic": {"q1": updated_q[0], "q2": updated_q[1]},
            "source_q1_ranking_variant_minus_source": original_q[0]["variant_action"] - original_q[0]["source_action"],
            "variant_q1_ranking_variant_minus_source": updated_q[0]["variant_action"] - updated_q[0]["source_action"],
            "source_q_min_ranking_variant_minus_source": min(q["variant_action"] for q in original_q) - min(q["source_action"] for q in original_q),
            "variant_q_min_ranking_variant_minus_source": min(q["variant_action"] for q in updated_q) - min(q["source_action"] for q in updated_q),
            "source_twin_gap_at_source_action": abs(original_q[0]["source_action"] - original_q[1]["source_action"]),
            "variant_twin_gap_at_variant_action": abs(updated_q[0]["variant_action"] - updated_q[1]["variant_action"]),
        }
        if sample_local:
            landscape = {}
            for action_name, center in (("source_action", original_action), ("variant_action", candidate_action)):
                for delta in (-LOCAL_STEERING_DELTA, 0.0, LOCAL_STEERING_DELTA):
                    probe = center.clone()
                    probe[:, 0] = (probe[:, 0] + delta).clamp(-1.0, 1.0)
                    label = f"{action_name}_steer_{delta:+.1f}"
                    landscape[label] = {label_critic: [float(critic.q(torch.cat((feat, probe), dim=-1)).squeeze().item())
                                                        for critic, feat in zip(bundle["critics"], features)]
                                        for label_critic, bundle, features in (("source", original, original_critic_features),
                                                                              ("variant", updated, candidate_critic_features))}
            detail["local_q_landscape"] = landscape
        result.append(detail)
    return result


def replay_source(root: Path, source: dict, models: dict, canonical: dict,
                  road_row: dict, manifest: dict) -> list[dict]:
    diagnostic = root / RUN / "train-diagnostic"
    with np.load(diagnostic / source["trace_path"], allow_pickle=False) as arrays:
        trace = {key: arrays[key].copy() for key in arrays.files}
    assert len(trace["reward"]) == source["steps"]
    for variant in VARIANTS:
        candidate = canonical[(source["source_learner_seed"], source["geometry_seed"], variant)]
        assert candidate["source_actor_sha256"] == source["actor_sha256"]
        assert digest(diagnostic / candidate["trace_path"]) == candidate["trace_sha256"] == manifest["files_sha256"][candidate["trace_path"]]
    env = build_environment(source["geometry_seed"], 1200)
    details = []
    try:
        observation, info = env.reset(seed=None)
        spec = models["unchanged-source"]["observation_spec"]
        observation = spec.validate(observation)
        wrapper = _find_wrapper(env, ("off_track_counter", "max_off_track_steps", "warmup_steps"))
        assert wrapper.max_off_track_steps == 100 and wrapper.warmup_steps == 50
        assert (info.get("seed"), info.get("track_id")) == (source["geometry_seed"], 1)
        assert _road_hash(env.unwrapped.track) == road_row["road_coordinate_sha256"]
        measured = sample_decisions(source["steps"])
        sparse = {int(step): index for index, step in enumerate(trace["sparse_step"])}
        for step in range(1, source["steps"] + 1):
            action = trace["native_action"][step - 1]
            if step in measured:
                with torch.inference_mode():
                    states = _analyze_state(
                        observation, models, source, VARIANTS, step, _road_state(env), action,
                        step in (1, 25, max(1, source["steps"] - END_STATES + 1)),
                    )
                if step == 1:
                    for entry in states:
                        candidate = canonical[(source["source_learner_seed"], source["geometry_seed"], entry["variant"])]
                        with np.load(diagnostic / candidate["trace_path"], allow_pickle=False) as candidate_trace:
                            assert np.max(np.abs(np.asarray(entry["variant_native_action"]) - candidate_trace["native_action"][0])) <= 1e-5
                details.extend(states)
            # Applying the *recorded official action* avoids another native ->
            # official -> native float32 conversion and preserves simulator input.
            official = trace["official_action"][step - 1]
            expected_official = models["unchanged-source"]["adapter"].to_official(action)
            assert np.max(np.abs(expected_official - official)) <= 1e-6
            next_observation, reward, terminated, truncated, step_info = env.step(official)
            next_observation = spec.validate(next_observation)
            step_info = {**info, **dict(step_info or {})}
            terminal = EpisodeCollector._is_terminal(bool(terminated), bool(truncated), step_info)
            idx = step - 1
            assert np.float32(reward) == trace["reward"][idx]
            assert np.float32(step_info["progress"]) == trace["progress"][idx]
            assert np.float32(step_info["damage"]) == trace["damage"][idx]
            assert bool(terminated) == bool(trace["terminated"][idx])
            assert bool(truncated) == bool(trace["truncated"][idx])
            assert bool(terminal) == bool(trace["terminal"][idx])
            assert bool(step_info.get("finished")) == bool(trace["finished"][idx])
            assert str(step_info.get("retire_reason") or "") == str(trace["retirement"][idx])
            assert int(wrapper.off_track_counter) == int(trace["off_track_counter"][idx])
            if step in sparse:
                sample = _road_sample(env, step)
                j = sparse[step]
                assert sample["nearest_point_index"] == trace["sparse_nearest_point_index"][j]
                for key, array in (("nearest_point_fraction", "sparse_nearest_point_fraction"),
                                   ("nearest_distance_m", "sparse_nearest_distance_m"),
                                   ("speed_m_s", "sparse_speed_m_s")):
                    assert np.float32(sample[key]) == trace[array][j], (source["geometry_seed"], step, key)
            if step == source["steps"]:
                assert (terminated or truncated) and bool(step_info["finished"])
            else:
                assert not (terminated or truncated)
                observation = next_observation
    finally:
        env.close()
    return details


def analyze(root: Path, out: Path) -> dict:
    protocol, inputs, selected = _preflight(root)
    out.mkdir(parents=False, exist_ok=False)
    provenance = {
        "format": "drq-r6-source-state-attribution-v1",
        "scope": "exact action-replay on 11 already-consumed TRAIN-DIAGNOSTIC source-success episodes; no alternate-policy rollout, training, or new evaluation",
        "protocol_sha256": PROTOCOL_SHA, "catalog_sha256": CATALOG_SHA,
        "diagnostic_manifest_sha256": MANIFEST_SHA,
        "selected": [{"source_seed": r["source_learner_seed"], "geometry_seed": r["geometry_seed"],
                      "recorded_steps": r["steps"], "source_trace_sha256": r["trace_sha256"]} for r in selected],
        "total_max_decisions": sum(r["steps"] for r in selected),
        "observation_sampling": "decisions 1..20, every 25th, and last 20; on the same source-action replay",
        "local_q_probe": "native steering +/-0.1 clipped, decisions 1, 25, final-20",
        "diagnostic_script_sha256": digest(Path(__file__)),
        "checkpoint_loading": "CPU21 torch.load(mmap=True) for pinned model states only; no replay, optimizer or RNG restore",
        "runtime_source_sha256": protocol["source_sha256"],
    }
    (out / "preflight.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    records = []
    shifts = []
    driven = 0
    try:
        for seed in (0, 1):
            source_info = next(x for x in protocol["source_actors"] if x["source_seed"] == seed)
            bundles = {"unchanged-source": _model(root, source_info, source_info)}
            for variant in VARIANTS:
                result = json.loads((root / RUN / f"learner-{seed}-{variant}" / "result.json").read_text())
                bundles[variant] = _model(root, result["candidates"][-1], result["candidates"][-1])
                a, b = bundles["unchanged-source"], bundles[variant]
                shifts.append({"source_seed": seed, "variant": variant,
                               "source_checkpoint_sha256": a["checkpoint_sha256"],
                               "variant_checkpoint_sha256": b["checkpoint_sha256"],
                               "actor_encoder": _relative_parameter_shift(a["actor"].encoder, b["actor"].encoder),
                               "actor_head": _relative_parameter_shift(a["actor"].policy, b["actor"].policy),
                               "actor_trunk": _relative_parameter_shift(a["actor"].trunk, b["actor"].trunk),
                               "critic_q1_encoder": _relative_parameter_shift(a["critics"][0].encoder, b["critics"][0].encoder),
                               "critic_q2_encoder": _relative_parameter_shift(a["critics"][1].encoder, b["critics"][1].encoder),
                               "critic_q1_head": _relative_parameter_shift(a["critics"][0].q, b["critics"][0].q),
                               "critic_q2_head": _relative_parameter_shift(a["critics"][1].q, b["critics"][1].q)})
            for source in (r for r in selected if r["source_learner_seed"] == seed):
                records.extend(replay_source(root, source, bundles, inputs["canonical"],
                                             inputs["road"][source["geometry_seed"]], inputs["manifest"]))
                driven += source["steps"]
            del bundles
            gc.collect()
        assert driven == provenance["total_max_decisions"]
        result = {**provenance, "driven_decisions": driven, "new_policy_rollouts": 0,
                  "same_source_observation_records": len(records), "parameter_shifts": shifts,
                  "states": records, "preflight_sha256": digest(out / "preflight.json")}
        with (out / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return {"output": str(out), "sha256": digest(out / "result.json"),
                "driven_decisions": driven, "same_state_records": len(records)}
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"error": repr(error), "completed_source_decisions": driven,
                                                        "preflight_sha256": digest(out / "preflight.json")}, indent=2) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.relative_to((ROOT / RUN).resolve())
    assert not out.exists() and out.parent.is_dir() and not out.parent.is_symlink()
    torch.set_num_threads(1)
    print(json.dumps(analyze(ROOT, out), sort_keys=True))


if __name__ == "__main__":
    main()
