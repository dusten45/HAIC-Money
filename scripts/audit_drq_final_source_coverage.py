"""Parity-gated, offline source-success coverage audit on consumed r6 roads.

Run with CPU21 from the repository root. Only archived source official actions
control the simulator; no diagnostic observation is inserted into learner replay.
All 99,994 valid source-replay starts per seed are streamed through the frozen
source encoder; feature proximity is not proof of same-road source-state coverage.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from drq_v2 import load_exported_actor
from scripts.audit_drq_retention_r7 import valid_starts, verified
from scripts.diagnose_drq_geometry_mix import _find_wrapper, _road_hash, _road_sample, build_environment
from scripts.diagnose_drq_geometry_regression import CATALOG_SHA, MANIFEST_SHA, PROTOCOL_SHA, ROOT, RUN, VARIANTS, digest
from scripts.diagnose_drq_r7_hybrids import check_step, reference_index, require
from scripts.diagnose_drq_source_states import MIN_HEADROOM_BYTES, _model, _preflight, _road_state, _state_sha


BASE = Path("runs/20260927-drqv2-final-source-replay-v1")
R7 = Path("runs/20260926-drqv2-retention-r7")
R7_PROTOCOL_SHA = "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9"
R7_RESULT_SHA = "998a816a92ca293b2459cb7b857fef1872ee1674d1a2f5efce91b06312c10651"
R7_TRACE_AUDIT_SHA = "826e6fad1a58ea5880c25b7f3354ab6cf06122099d803f465f1d7b02c95c0db9"
V2_PREFLIGHT_SHA = "b2fe774cc756a39fc2a6c9d586f7d11bab0085f9b98a012cb2a3e00cc14be431"
V2_RESULT_SHA = "2c1e1a3ce21137c00bc2042d069a9a410b77f92eb76bac3334c2c548888c07cc"
V2_SCRIPT_SHA = "c1d842c9d6896ce78fffc2b2608b7d15771b49660a2d0abac03ebad91960fdd5"
HYBRID_PREFLIGHT_SHA = "67fde9d2923c2043f3ecc2163d3e96ef325ae9826935921a472957cb4691381f"
HYBRID_RESULT_SHA = "5614b8c2a1c7b261857c3fde4f0a2896b525ba217a4a50acf28c1931efd84369"
CACHE_SHA = "2679b63c5569b9c2b6a3ecbc9f3461563560a5e44b0581103e2366fa46dc7f28"
PRIMARY_RECEIPT_SHA = "e1746c05a1e11af31444a99505067e400e53734c535bfda8a8b778963e34b042"
PRIMARY_SCRIPT_SHA = "f66655521c5226f718ea292cebc5878744c0e53385521ebfa9c493eca6ac335f"
VALID_STARTS = 99994
PHASES = ("early", "middle", "major_corner", "recovery_proxy", "late", "finish_approach")
PHASE_DEFINITIONS = {
    "early": "pre-action decisions 1..20",
    "middle": "40%-60% decision quantile, inclusive",
    "major_corner": "visited-track local tangent angle change over +/-3 segments >= max(.10 rad, road's positive 90th percentile); three strongest sampled",
    "recovery_proxy": "post-action off_track_counter rises/falls versus previous or pre-action |road-relative heading|>=.35 rad; observational proxy, NOT verified recovery",
    "late": "75%-90% decision quantile, inclusive",
    "finish_approach": "last 20 pre-action decisions OR prior-action visited-tile progress>=.90; NOT a guaranteed finish maneuver",
    "representatives": "earliest, median, latest per phase; strongest three major corners; first three recovery-proxy decisions; missing phases explicitly zero",
}
PAIRED_DRIFT_CONTRACT = {
    "input": "same 187 parity-reconstructed source-success float32 CHW observations, same seed/road/decision as SHA-pinned immutable primary coverage receipt",
    "features": "source and each frozen r7b actor.encoder on the identical ordered batched float32 tensor; 256 raw post-tanh feature coordinates, inference mode, CPU21",
    "metrics": "mse=mean((r7b-source)^2) over 256 coordinates; cosine=dot/(source_norm*r7b_norm); source_normalized_l2=||r7b-source||_2/||source||_2",
    "actions": "source and r7b actor native actions forwarded singly per observation, native L-infinity; never used to drive the environment",
    "interpretation": "paired observational representation/action drift on reused diagnostic source states; arbitrary evolving latent coordinates and overlapping serial phases; no causal relationship, coverage proof, or threshold tuning",
}


def pinned_json(root: Path, name: str, sha: str) -> dict:
    return json.loads(verified(root, name, sha).read_text(encoding="utf-8"))


def fixed_positions(indices: list[int]) -> list[int]:
    """At most three deterministic representatives without outcome-based picking."""
    if not indices:
        return []
    return sorted({indices[i] for i in (0, (len(indices) - 1) // 2, len(indices) - 1)})


def action_distribution(reference: np.ndarray, comparison: np.ndarray) -> dict:
    require(reference.shape == comparison.shape and reference.ndim == 2 and reference.shape[1] == 3 and
            reference.shape[0] > 0 and np.isfinite(reference).all() and np.isfinite(comparison).all(),
            "invalid native action distribution")
    diff = np.max(np.abs(reference - comparison), axis=1)
    return {"n": len(diff), "native_linf_mean": float(diff.mean()),
            "native_linf_median": float(np.median(diff)), "native_linf_p90": float(np.quantile(diff, .90)),
            "near_0_10_n": int(np.count_nonzero(diff <= .10)),
            "stored_native_axis_mean": comparison.mean(axis=0).tolist(),
            "final_source_native_axis_mean": reference.mean(axis=0).tolist()}


def neighbor_counts(cosine: np.ndarray, stored: np.ndarray, action: np.ndarray,
                    sampled_weights: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Count full-pool neighbors and actual sampled slots, not just the nearest."""
    require(cosine.ndim == 1 and cosine.size > 0 and stored.shape == (cosine.size, 3) and
            action.shape == (3,) and np.isfinite(cosine).all() and
            np.isfinite(stored).all() and np.isfinite(action).all(), "invalid feature/action neighbors")
    similar = cosine >= .95
    action_near = np.max(np.abs(stored - action), axis=1) <= .10
    both = similar & action_near
    full = {"valid_starts_n": int(cosine.size),
            "feature_neighbors_cosine_ge_0_95_n": int(np.count_nonzero(similar)),
            "feature_and_stored_action_near_0_10_n": int(np.count_nonzero(both))}
    per_arm = {}
    for variant, weights in sampled_weights.items():
        require(weights.shape == cosine.shape and weights.dtype.kind in "iu" and np.all(weights >= 0),
                "invalid r7b sampled source weights")
        sampled = weights > 0
        per_arm[variant] = {
            "sampled_source_unique_starts_n": int(np.count_nonzero(sampled)),
            "sampled_source_slots_n": int(weights.sum(dtype=np.int64)),
            "feature_neighbors_cosine_ge_0_95_unique_n": int(np.count_nonzero(similar & sampled)),
            "feature_neighbors_cosine_ge_0_95_slots_n": int(weights[similar].sum(dtype=np.int64)),
            "feature_and_stored_action_near_0_10_unique_n": int(np.count_nonzero(both & sampled)),
            "feature_and_stored_action_near_0_10_slots_n": int(weights[both].sum(dtype=np.int64)),
        }
    return full, per_arm


def representative_actions(actor: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    """Mirror the original v2/diagnostic singleton policy-forward contract."""
    require(tensor.ndim == 4 and tensor.shape[1:] == (4, 84, 84) and len(tensor) > 0,
            "invalid representative source observation batch")
    with torch.inference_mode():
        actions = np.stack([actor(tensor[i:i + 1]).squeeze(0).numpy() for i in range(len(tensor))])
    require(actions.shape == (len(tensor), 3) and np.isfinite(actions).all(),
            "invalid singleton representative native actions")
    return actions


def actor_encoder_drift(source: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    """Raw actor-encoder coordinates on identical observations, not aligned latents."""
    require(source.ndim == 2 and source.shape == candidate.shape and source.shape[1] == 256 and
            source.shape[0] > 0 and np.isfinite(source).all() and np.isfinite(candidate).all(),
            "invalid paired actor encoder features")
    original = source.astype(np.float64)
    changed = candidate.astype(np.float64)
    source_norm = np.linalg.norm(original, axis=1)
    candidate_norm = np.linalg.norm(changed, axis=1)
    require(np.all(source_norm > 1e-12) and np.all(candidate_norm > 1e-12),
            "undefined cosine/source-normalized encoder drift at zero feature norm")
    difference = changed - original
    return {"mse": np.mean(difference ** 2, axis=1),
            "cosine": np.clip(np.sum(original * changed, axis=1) / (source_norm * candidate_norm), -1.0, 1.0),
            "source_normalized_l2": np.linalg.norm(difference, axis=1) / source_norm}


def encoder_action_drift_summary(rows: list[dict], variant: str) -> dict:
    """Descriptive paired summaries; overlapping phases are never independent."""
    summary = {"paired_representative_n": len(rows)}
    for field in ("action_native_linf", "encoder_mse", "encoder_cosine", "encoder_source_normalized_l2"):
        values = [row["r7b_vs_source_native_linf"][variant] if field == "action_native_linf"
                  else row["r7b_actor_encoder_drift"][variant][field.removeprefix("encoder_")]
                  for row in rows]
        require(np.isfinite(values).all(), f"invalid paired {field} drift")
        summary[field] = {"mean": float(np.mean(values)) if values else None,
                          "median": float(np.median(values)) if values else None}
    return summary


def phase_plan(length: int, progress: np.ndarray, off_track: np.ndarray,
               corner: np.ndarray, heading: np.ndarray) -> tuple[dict, dict]:
    """All indices are 1-based PRE-action states; telemetry uses previous action."""
    require(length >= 20 and all(len(x) == length for x in (progress, off_track, corner, heading)),
            "invalid longitudinal source episode")
    decisions = np.arange(1, length + 1)
    prior_progress = np.concatenate(([0.0], progress[:-1]))
    changes = np.diff(np.concatenate(([0], off_track.astype(np.int64))))
    require(np.isfinite(corner).all() and np.isfinite(heading).all(), "non-finite road metadata")
    # High local tangent turn on this actual road, not a catalog-family label.
    positive = corner[corner > 0]
    corner_floor = max(0.10, float(np.quantile(positive, .90))) if positive.size else math.inf
    groups = {
        "early": decisions[decisions <= 20],
        "middle": decisions[(decisions >= math.ceil(.40 * length)) & (decisions <= math.floor(.60 * length))],
        "major_corner": decisions[corner >= corner_floor],
        # Observational hazard/improvement proxy only; NOT a true recovery label.
        "recovery_proxy": decisions[(changes != 0) | (np.abs(heading) >= .35)],
        "late": decisions[(decisions >= math.ceil(.75 * length)) & (decisions <= math.floor(.90 * length))],
        "finish_approach": decisions[(decisions > length - 20) | (prior_progress >= .90)],
    }
    labels = {name: [int(i) for i in values] for name, values in groups.items()}
    reps = {name: fixed_positions(indices) for name, indices in labels.items()}
    # Prefer strongest three actual visited turns; keep deterministic tie order.
    reps["major_corner"] = sorted(sorted(labels["major_corner"], key=lambda i: (-corner[i - 1], i))[:3])
    reps["recovery_proxy"] = labels["recovery_proxy"][:3]
    return labels, {"representatives": reps, "corner_threshold_rad": None if not positive.size else corner_floor,
                    "counter_rising": int(np.count_nonzero(changes > 0)),
                    "counter_falling": int(np.count_nonzero(changes < 0)),
                    "heading_proxy_count": int(np.count_nonzero(np.abs(heading) >= .35))}


def sparse_parity(env, trace: dict, decision: int, index: int) -> None:
    measured = _road_sample(env, decision)
    for field, name in (("nearest_point_index", "sparse_nearest_point_index"),
                        ("nearest_point_fraction", "sparse_nearest_point_fraction"),
                        ("nearest_distance_m", "sparse_nearest_distance_m"),
                        ("speed_m_s", "sparse_speed_m_s")):
        value = int(measured[field]) if field == "nearest_point_index" else np.float32(measured[field])
        require(value == trace[name][index], f"sparse road mismatch at {decision}: {field}")


def preflight(root: Path) -> dict:
    """File/model identity checks only: no environment construction or output."""
    require(digest(root / "scripts/diagnose_drq_source_states.py") == V2_SCRIPT_SHA,
            "v2 source replay implementation changed")
    require(digest(root / "scripts/diagnose_drq_r7_hybrids.py") ==
            "9f7783d67342932260ee7eca3662461949f27abe7ab46dca265ce71cdb28b99b" and
            digest(root / "scripts/audit_drq_retention_r7.py") ==
            "b283580c92445702b34152c436f515cb1481ebcd1d6a6572cb138ec31239d726",
            "pinned reference index, parity check or n-step validator changed")
    protocol, inputs, selected = _preflight(root)
    require(sum(s["steps"] for s in selected) == 5998, "source-action decision budget differs")
    v2_preflight = pinned_json(root, str(RUN / "source-state-replay-v2/preflight.json"), V2_PREFLIGHT_SHA)
    v2 = pinned_json(root, str(RUN / "source-state-replay-v2/result.json"), V2_RESULT_SHA)
    hybrid_preflight = pinned_json(root, str(R7 / "hybrid/first20-v2/preflight.json"), HYBRID_PREFLIGHT_SHA)
    hybrid = pinned_json(root, str(R7 / "hybrid/first20-v2/result.json"), HYBRID_RESULT_SHA)
    r7 = pinned_json(root, "experiments/drqv2-retention-r7.json", R7_PROTOCOL_SHA)
    frozen_result = pinned_json(root, "experiments/drqv2-retention-r7-result.json", R7_RESULT_SHA)
    audit = pinned_json(root, str(R7 / "pre-evaluation-trace-audit-v1.json"), R7_TRACE_AUDIT_SHA)
    primary = pinned_json(root, str(BASE / "offline-coverage/result.json"), PRIMARY_RECEIPT_SHA)
    require(primary["parity_passed"] is True and primary["script_sha256"] == PRIMARY_SCRIPT_SHA and
            primary["archived_source_actions_replayed"] == 5998 and
            primary["source_success_cells"] == 11 and
            len(primary["representative_frames"]) == 187 and
            all(primary["source_replay"][str(seed)]["scanned_n"] == VALID_STARTS for seed in (0, 1)),
            "original full source coverage receipt changed")
    selection = [{"source_seed": s["source_learner_seed"], "geometry_seed": s["geometry_seed"],
                  "recorded_steps": s["steps"], "source_trace_sha256": s["trace_sha256"]} for s in selected]
    require(v2_preflight["selected"] == v2["selected"] == hybrid_preflight["selected"] ==
            hybrid["selected"] == selection and v2["preflight_sha256"] == V2_PREFLIGHT_SHA and
            hybrid["preflight_sha256"] == HYBRID_PREFLIGHT_SHA and
            hybrid["source_replay_result_sha256"] == V2_RESULT_SHA and
            hybrid["source_replay_preflight_sha256"] == V2_PREFLIGHT_SHA and
            all(row["protocol_sha256"] == PROTOCOL_SHA and row["catalog_sha256"] == CATALOG_SHA and
                row["diagnostic_manifest_sha256"] == MANIFEST_SHA
                for row in (v2, v2_preflight, hybrid, hybrid_preflight)), "prior parity lineage changed")
    require(v2["diagnostic_script_sha256"] == V2_SCRIPT_SHA and
            v2["driven_decisions"] == hybrid["recorded_official_decisions_replayed"] == 5998 and
            hybrid["cached_observation_instances"] == 220 and
            hybrid["parity_sampled_observation_instances"] == 666 and
            hybrid["source_episodes_replayed"] == 11 and
            hybrid["cache_sha256"] == r7["diagnostic_cache"]["sha256"] == CACHE_SHA and
            r7["diagnostic_cache"]["hybrid_result_sha256"] == HYBRID_RESULT_SHA and
            r7["r6_protocol_sha256"] == PROTOCOL_SHA and
            r7["catalog_sha256"] == CATALOG_SHA and r7["diagnostic_manifest_sha256"] == MANIFEST_SHA and
            frozen_result["development_diagnostic"]["source_success_cells"] == 11 and
            audit["passed"] is True and audit["study_protocol_sha256"] == R7_PROTOCOL_SHA,
            "r6/r7 parity or replay scope changed")
    refs = reference_index(v2, selected)
    cache_path = verified(root, r7["diagnostic_cache"]["path"], CACHE_SHA)
    with np.load(cache_path, allow_pickle=False) as data:
        require(set(data.files) == {"observations", "source_seed", "geometry_seed"} and
                data["observations"].shape == (11, 20, 4, 84, 84) and
                data["observations"].dtype == np.float32, "invalid first20 cache")
        require(len(hybrid["cache_frame_sha256"]) == 11, "incomplete cache hashes")
        for j, row in enumerate(selected):
            seed, road = row["source_learner_seed"], row["geometry_seed"]
            require((int(data["source_seed"][j]), int(data["geometry_seed"][j])) == (seed, road)
                    and len(hybrid["cache_frame_sha256"][j]) == 20, "cache cell order differs")
            for i, frame in enumerate(data["observations"][j]):
                require(_state_sha(frame) == hybrid["cache_frame_sha256"][j][i] ==
                        refs[(seed, road, i + 1, VARIANTS[0])]["observation_sha256"],
                        f"first20 float32 cache hash differs at {seed}/{road}/{i + 1}")
    require(r7["source_replay"].keys() == {"0", "1"} and
            len(audit["arms"]) == 12 and all(row["source_rows"] == row["online_rows"] == 728576
                                            for row in audit["arms"]), "source sampled trace audit incomplete")
    catalog = json.loads((root / protocol["catalog_path"]).read_text(encoding="utf-8"))
    train = {int(row["geometry_seed"]) for row in catalog["train"]}
    diagnostic = {int(row["geometry_seed"]) for row in catalog["train_diagnostic"]}
    require(len(train) == 120 and len(diagnostic) == 16 and not train & diagnostic and
            {r["geometry_seed"] for r in selected} <= diagnostic and
            all(a["source_seed"] in (0, 1) for a in audit["arms"]), "TRAIN/diagnostic partition changed")
    arms = {}
    for seed in (0, 1):
        source = next(row for row in protocol["source_actors"] if row["source_seed"] == seed)
        replay = r7["source_replay"][str(seed)]
        require(source["checkpoint_sha256"] == replay["checkpoint_sha256"] and
                verified(root, replay["episode_ledger_path"], replay["episode_ledger_sha256"]) and
                any(a["source_seed"] == seed for a in audit["arms"]), "original replay identity changed")
        for variant in VARIANTS:
            run = next(a for a in audit["arms"] if (a["source_seed"], a["variant"], a["condition"]) ==
                       (seed, variant, "r7b"))
            result = pinned_json(root, str(Path(run["run_dir"]) / "result.json"), run["result_sha256"])
            candidate = result["candidates"][-1]
            require(result["completed"] and result["source_seed"] == seed and
                    result["variant"] == variant and result["condition"] == "r7b" and
                    result["source_checkpoint_sha256"] == replay["checkpoint_sha256"] and
                    result["source_replay_sha256"] == replay["checkpoint_sha256"] and
                    candidate["checkpoint_sha256"] == run["final_checkpoint_sha256"] and
                    candidate["sample_trace_sha256"] == run["final_trace_sha256"] and
                    result["study_protocol_sha256"] == R7_PROTOCOL_SHA and
                    candidate["checkpoint_online_step"] == 32768 and
                    result["study_gradient_steps"] == 22768 and
                    result["source_samples"] == 728576, "r7b final actor or sample lineage changed")
            verified(root, candidate["actor_path"], candidate["actor_sha256"])
            verified(root, candidate["sample_trace_path"], candidate["sample_trace_sha256"])
            arms[(seed, variant)] = candidate
        require(audit["source_replays"][str(seed)]["checkpoint_sha256"] == replay["checkpoint_sha256"] and
                audit["source_replays"][str(seed)]["episode_ledger_sha256"] == replay["episode_ledger_sha256"] and
                audit["source_replays"][str(seed)]["valid_original_n_step_starts"] == 99994,
                "original source replay audit does not bind this checkpoint")
    return {"protocol": protocol, "inputs": inputs, "selected": selected, "refs": refs,
            "hybrid": hybrid, "r7": r7, "audit": audit, "arms": arms, "primary": primary}


def reconstruct(root: Path, source: dict, row: dict, model: dict, refs: dict,
                cache: np.ndarray) -> tuple[dict, dict[int, np.ndarray]]:
    """No writes, no changed actor controls; all 5,998 original steps must pass."""
    path = verified(root, str(RUN / "train-diagnostic" / source["trace_path"]), source["trace_sha256"])
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    seed, road, length = source["source_learner_seed"], source["geometry_seed"], source["steps"]
    require(len(trace["reward"]) == length and len(trace["native_action"]) == length and
            len(trace["official_action"]) == length and np.all(trace["actor_sha256"] == model["actor_sha256"]) and
            np.all(trace["source_actor_sha256"] == model["actor_sha256"]) and
            np.all(trace["checkpoint_sha256"] == model["checkpoint_sha256"]) and
            np.all(trace["source_checkpoint_sha256"] == model["checkpoint_sha256"]) and
            np.array_equal(trace["sparse_step"], np.unique(trace["sparse_step"])) and
            int(trace["sparse_step"][0]) == 0 and int(trace["sparse_step"][-1]) == length and
            set(map(int, trace["sparse_step"])) == (set(range(0, length + 1, 25)) | {1, length}),
            "archived source action/road schema changed")
    sparse = {int(step): i for i, step in enumerate(trace["sparse_step"])}
    corner = np.empty(length, dtype=np.float64)
    heading = np.empty(length, dtype=np.float64)
    # Static windows may be selected in advance; dynamic phase candidates retain
    # only three frames each (not the whole trajectory or a training dataset).
    static_labels, _ = phase_plan(length, trace["progress"], trace["off_track_counter"],
                                  np.zeros(length), np.zeros(length))
    static_reps = {i for name in ("early", "middle", "late", "finish_approach")
                   for i in fixed_positions(static_labels[name])}
    frames: dict[int, np.ndarray] = {}
    corner_best: list[tuple[float, int]] = []
    recovery_first: list[int] = []
    env = build_environment(road, 1200)
    try:
        observation, info = env.reset(seed=None)
        spec = model["observation_spec"]
        observation = spec.validate(observation)
        wrapper = _find_wrapper(env, ("off_track_counter", "max_off_track_steps", "warmup_steps"))
        require((wrapper.max_off_track_steps, wrapper.warmup_steps) == (100, 50) and
                (info.get("seed"), info.get("track_id")) == (road, 1) and
                _road_hash(env.unwrapped.track) == row["road_coordinate_sha256"], "reset/road parity mismatch")
        sparse_parity(env, trace, 0, sparse[0])
        points = np.asarray(env.unwrapped.track, dtype=np.float64)[:, 2:4]
        tangent = np.diff(np.concatenate((points, points[:1]), axis=0), axis=0)
        angles = np.arctan2(tangent[:, 1], tangent[:, 0])
        local = np.abs(np.arctan2(np.sin(np.roll(angles, -3) - np.roll(angles, 3)),
                                  np.cos(np.roll(angles, -3) - np.roll(angles, 3))))
        for decision in range(1, length + 1):
            i = decision - 1
            state = _road_state(env)
            heading[i] = state["road_relative_heading_rad"]
            corner[i] = local[state["nearest_point_index"]]
            old = refs.get((seed, road, decision, VARIANTS[0]))
            if old is not None or decision in static_reps:
                frame_hash = _state_sha(observation)
                if old is not None:
                    with torch.inference_mode():
                        action = model["actor"](torch.as_tensor(observation).unsqueeze(0)).squeeze(0).numpy()
                    require(np.max(np.abs(action - trace["native_action"][i])) <= 1e-5,
                            f"source policy differs at {seed}/{road}/{decision}")
                    for variant in VARIANTS:
                        prior = refs[(seed, road, decision, variant)]
                        require(prior["observation_sha256"] == frame_hash and
                                prior["source_trace_sha256"] == source["trace_sha256"] and
                                np.max(np.abs(action - prior["source_native_action"])) <= 1e-5,
                                f"v2 float32 frame/action parity differs at {seed}/{road}/{decision}")
                    if decision <= 20:
                        require(np.array_equal(observation, cache[i]),
                                f"first20 cached float32 pixels differ at {seed}/{road}/{decision}")
                    require(state["nearest_point_index"] == old["road_pre_action"]["nearest_point_index"] and
                            abs(state["road_relative_heading_rad"] -
                                old["road_pre_action"]["road_relative_heading_rad"]) <= 1e-8,
                            "v2 road pre-action parity differs")
                if decision in static_reps:
                    frames[decision] = observation.copy()
            # The counter is archived POST-action; its rise/fall is only a proxy.
            previous = int(trace["off_track_counter"][i - 1]) if i else 0
            if ((int(trace["off_track_counter"][i]) != previous or abs(heading[i]) >= .35)
                    and len(recovery_first) < 3):
                recovery_first.append(decision)
                frames[decision] = observation.copy()
            corner_best.append((float(corner[i]), decision))
            corner_best.sort(key=lambda item: (-item[0], item[1]))
            if len(corner_best) > 3:
                _, evicted = corner_best.pop()
                if evicted not in static_reps and evicted not in recovery_first:
                    frames.pop(evicted, None)
            if decision in {d for _, d in corner_best}:
                frames[decision] = observation.copy()
            native, official = trace["native_action"][i], trace["official_action"][i]
            require(np.max(np.abs(model["adapter"].to_official(native) - official)) <= 1e-6,
                    f"archived native/official action mismatch at {seed}/{road}/{decision}")
            next_obs, reward, terminated, truncated, step_info = env.step(official)
            step_info = {**info, **dict(step_info or {})}
            require((step_info.get("seed"), step_info.get("track_id")) == (road, 1), "road changed during replay")
            check_step(trace, i, reward, terminated, truncated, step_info, wrapper.off_track_counter)
            if decision in sparse:
                sparse_parity(env, trace, decision, sparse[decision])
            require(bool(terminated or truncated) == (decision == length), "source terminal step differs")
            if decision < length:
                observation = spec.validate(next_obs)
        require(bool(step_info.get("finished")) and bool(trace["finished"][-1]), "source did not finish")
    finally:
        env.close()
    labels, details = phase_plan(length, trace["progress"], trace["off_track_counter"], corner, heading)
    reps = details["representatives"]
    # Dynamic corner candidates may include subthreshold turns; only evaluated
    # selected representatives survive in the output, never buffered for training.
    require(all(decision in frames for group in reps.values() for decision in group),
            "missing phase representative float32 frame")
    keep = {decision: frames[decision] for group in reps.values() for decision in group}
    require(digest(path) == source["trace_sha256"], "source trace changed during replay")
    overlap = {a: {b: len(set(labels[a]) & set(labels[b])) for b in PHASES} for a in PHASES}
    return ({"source_seed": seed, "geometry_seed": road, "track_id": 1, "family": source["family"],
             "steps": length, "source_trace_sha256": source["trace_sha256"],
             "phase_counts": {name: len(labels[name]) for name in PHASES},
             "phase_overlap_decisions": overlap, "phase_representatives": reps,
             "proxy_support": {key: value for key, value in details.items() if key != "representatives"}}, keep)


def sampled_indices(root: Path, candidate: dict, audit_row: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    path = verified(root, candidate["sample_trace_path"], audit_row["final_trace_sha256"])
    with np.load(path, allow_pickle=False) as trace:
        require(set(trace.files) == {"source", "source_indices", "episode_id", "protocol_sha256"} and
                bytes(trace["protocol_sha256"]).decode("ascii") == R7_PROTOCOL_SHA and
                trace["source"].shape == trace["source_indices"].shape == (22768, 64) and
                trace["source"].dtype == np.uint8 and trace["source_indices"].dtype == np.int64 and
                np.all((trace["source"] == 0) | (trace["source"] == 1)) and
                np.all(np.sum(trace["source"] == 1, axis=1) == 32), "r7b sampled trace schema/quota differs")
        ids, counts = np.unique(trace["source_indices"][trace["source"] == 1], return_counts=True)
    require(int(counts.sum()) == 728576 and np.all((ids >= 31072) & (ids <= 131071)) and
            audit_row["source_seed"] == seed and audit_row["unique_source_starts"] == len(ids),
            "r7b sampled source indices differ from independent audit")
    return ids, counts


def replay_valid_starts(state: dict) -> np.ndarray:
    require(state["capacity"] == 100000 and state["n_step"] == 3 and state["gamma"] == .99 and
            state["action_dim"] == 3 and state["size"] == 100000 and state["next_sequence"] == 131072,
            "original rolling source replay shape differs")
    valid, _ = valid_starts(state, first=31072, last=131072)
    sequences = np.flatnonzero(valid).astype(np.int64) + 31072
    require(sequences.size == VALID_STARTS and sequences[0] >= 31072 and sequences[-1] <= 131071,
            "original 99994 terminal-safe n-step starts differ")
    return sequences


def score_seed(root: Path, seed: int, cells: list[dict], frames: list[dict], context: dict) -> dict:
    headroom = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text())
    require(headroom >= MIN_HEADROOM_BYTES, "insufficient headroom for complete source replay encoder scan")
    source_info = next(row for row in context["protocol"]["source_actors"] if row["source_seed"] == seed)
    source = _model(root, source_info, source_info)
    actors, samples = {}, {}
    for variant in VARIANTS:
        candidate = context["arms"][(seed, variant)]
        actor, _, _ = load_exported_actor(root / candidate["actor_path"], device="cpu")
        actor.eval()
        actors[variant] = actor
        audit_row = next(a for a in context["audit"]["arms"] if
                         (a["source_seed"], a["variant"], a["condition"]) == (seed, variant, "r7b"))
        samples[variant] = sampled_indices(root, candidate, audit_row, seed)
    replay_meta = context["r7"]["source_replay"][str(seed)]
    path = verified(root, replay_meta["checkpoint_path"], replay_meta["checkpoint_sha256"])
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    require(payload["format"] == "haic-drq-v2-checkpoint-v1" and payload["environment_steps"] == 131072,
            "not original source-training replay")
    state = payload["replay"]
    sequences = replay_valid_starts(state)
    from drq_v2 import Uint8Replay
    pool = Uint8Replay(capacity=100000, n_step=3, gamma=.99)
    pool.load_state_dict(state)
    del payload, state
    gc.collect()
    scan_headroom = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text())
    require(scan_headroom >= 8 * 1024**3, "insufficient headroom after original replay load")
    # Batched CPU inference; the same immutable source encoder is used on BOTH
    # quantized replay stacks and original reconstructed float32 query stacks.
    features, stored, evolved, hashes = [], [], [], []
    with torch.inference_mode():
        for index, chunk in enumerate(np.array_split(sequences, math.ceil(VALID_STARTS / 64))):
            if index % 128 == 0:
                available = int(Path("/sys/fs/cgroup/memory.max").read_text()) - int(Path("/sys/fs/cgroup/memory.current").read_text())
                require(available >= 4 * 1024**3, "memory headroom fell during complete source replay scan")
            stacks = np.stack([pool._stack(int(seq)) for seq in chunk])
            require(stacks.dtype == np.uint8 and stacks.shape == (len(chunk), 4, 84, 84),
                    "invalid terminal-safe source replay stack")
            tensor = torch.as_tensor(stacks)
            encoded = source["actor"].encoder(tensor)
            features.append(encoded.numpy())
            evolved.append(torch.tanh(source["actor"].policy(source["actor"].trunk(encoded))).numpy())
            stored.append(pool.actions[chunk % 100000].copy())
            hashes.extend(_state_sha(stack) for stack in stacks)
    feature = np.concatenate(features)
    evolved_action, stored_action = np.concatenate(evolved), np.concatenate(stored)
    require(np.isfinite(feature).all() and np.isfinite(stored_action).all() and
            np.isfinite(evolved_action).all(), "non-finite source replay proxy")
    quantized_hashes = set(hashes)
    normal = feature / np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
    sampled_weights = {}
    for variant in VARIANTS:
        ids, counts = samples[variant]
        positions = np.searchsorted(sequences, ids)
        require(np.all(positions < len(sequences)) and np.array_equal(sequences[positions], ids),
                "actual r7b source sample is not a valid original 3-step start")
        weights = np.zeros(len(sequences), dtype=np.int32)
        weights[positions] = counts
        require(np.count_nonzero(weights) == len(ids) and
                weights.sum(dtype=np.int64) == 728576, "incomplete actual r7b sample join")
        sampled_weights[variant] = weights
    query = np.stack([row["observation"] for row in frames])
    with torch.inference_mode():
        tensor = torch.as_tensor(query)
        query_feature = source["actor"].encoder(tensor).numpy()
        require(query_feature.shape == (len(frames), 256), "source actor encoder feature shape changed")
        source_action = representative_actions(source["actor"], tensor)
        for i, row in enumerate(frames):
            native_gap = float(np.max(np.abs(source_action[i] - row["archived_native_action"])))
            official_gap = float(np.max(np.abs(source["adapter"].to_official(source_action[i]) -
                                                row["archived_official_action"])))
            require(native_gap <= 1e-5 and official_gap <= 1e-6,
                    f"representative source actor/archived action mismatch at seed={seed} "
                    f"road={row['geometry_seed']} decision={row['decision']} "
                    f"native_linf={native_gap:.9g} official_linf={official_gap:.9g}")
        for variant, actor in actors.items():
            candidate_feature = actor.encoder(tensor).numpy()
            drift = actor_encoder_drift(query_feature, candidate_feature)
            for i, row in enumerate(frames):
                row.setdefault("r7b_actor_encoder_drift", {})[variant] = {
                    "mse": float(drift["mse"][i]),
                    "cosine": float(drift["cosine"][i]),
                    "source_normalized_l2": float(drift["source_normalized_l2"][i]),
                }
            for row, action in zip(frames, representative_actions(actor, tensor)):
                row["r7b_native_actions"][variant] = action.tolist()
    distances = (query_feature / np.maximum(np.linalg.norm(query_feature, axis=1, keepdims=True), 1e-12)) @ normal.T
    require(np.isfinite(distances).all(), "invalid source-encoder feature cosine")
    for i, row in enumerate(frames):
        nearest = int(np.argmax(distances[i]))
        query_sha = _state_sha(query[i])
        quantized_sha = _state_sha(source["observation_spec"].to_uint8(query[i]))
        near_seq = int(sequences[nearest])
        full_counts, sampled_counts = neighbor_counts(distances[i], stored_action, source_action[i], sampled_weights)
        row.update({"observation_sha256_float32": query_sha,
                    "source_native_action": source_action[i].tolist(),
                    "source_replay_valid_starts_scanned": VALID_STARTS,
                    "nearest_replay_sequence": near_seq,
                    "nearest_replay_feature_cosine": float(distances[i, nearest]),
                    "nearest_replay_feature_distance_one_minus_cosine": float(1.0 - distances[i, nearest]),
                    "full_source_replay_neighbor_counts": full_counts,
                    "r7b_sampled_source_neighbor_counts": sampled_counts,
                    "nearest_replay_uint8_stack_sha256": hashes[nearest],
                    "quantized_observation_equal_nearest": quantized_sha == hashes[nearest],
                    "quantized_observation_exact_any_replay_stack": quantized_sha in quantized_hashes,
                    "nearest_stored_native_action": stored_action[nearest].tolist(),
                    "nearest_final_source_native_action": evolved_action[nearest].tolist(),
                    "source_vs_nearest_stored_native_linf": float(np.max(np.abs(source_action[i] - stored_action[nearest]))),
                    "source_vs_nearest_final_native_linf": float(np.max(np.abs(source_action[i] - evolved_action[nearest]))),
                    "r7b_vs_source_native_linf": {variant: float(np.max(np.abs(np.asarray(action) - source_action[i])))
                                                    for variant, action in row["r7b_native_actions"].items()}})
        row.pop("observation")
        for variant in VARIANTS:
            ids, counts = samples[variant]
            sampled_mask = sampled_weights[variant] > 0
            eligible = np.flatnonzero(sampled_mask)
            nearest_sampled = int(eligible[np.argmax(distances[i, eligible])])
            seq = int(sequences[nearest_sampled])
            ix = int(np.searchsorted(ids, seq))
            require(ix < len(ids) and ids[ix] == seq, "sampled source sequence join differs")
            row.setdefault("r7b_sampled_source_neighbors", {})[variant] = {
                "sequence": seq, "sample_slots": int(counts[ix]),
                "feature_cosine": float(distances[i, nearest_sampled]),
                "stored_action_linf_vs_source": float(np.max(np.abs(stored_action[nearest_sampled] - source_action[i]))),
                "stored_action_near_0_10": bool(np.max(np.abs(stored_action[nearest_sampled] - source_action[i])) <= .10),
                "nearest_replay_start_was_sampled": bool(sampled_mask[nearest]),
            }
    for cell in cells:
        for phase in PHASES:
            matched = [row for row in frames if row["geometry_seed"] == cell["geometry_seed"] and
                       row["decision"] in cell["phase_representatives"][phase]]
            cell.setdefault("phase_proxy", {})[phase] = {
                "representative_n": len(matched),
                "status": "observed_proxy" if matched else "unobserved",
                "full_replay_valid_start_comparisons_n": len(matched) * VALID_STARTS,
                "nearest_feature_distance_mean": (float(np.mean([r["nearest_replay_feature_distance_one_minus_cosine"]
                                                            for r in matched])) if matched else None),
                "feature_neighbor_count_ge_0_95_sum": sum(r["full_source_replay_neighbor_counts"]
                                                        ["feature_neighbors_cosine_ge_0_95_n"] for r in matched),
                "feature_and_stored_action_near_count_sum": sum(r["full_source_replay_neighbor_counts"]
                                                               ["feature_and_stored_action_near_0_10_n"] for r in matched),
                "feature_cosine_ge_0_95_n": sum(r["nearest_replay_feature_cosine"] >= .95 for r in matched),
                "stored_action_near_0_10_n": sum(r["source_vs_nearest_stored_native_linf"] <= .10 for r in matched),
                "quantized_exact_replay_stack_n": sum(r["quantized_observation_exact_any_replay_stack"] for r in matched),
                "r7b": {v: {"same_source_action_near_0_10_n": sum(r["r7b_vs_source_native_linf"][v] <= .10
                                                                   for r in matched),
                             "paired_encoder_action_drift": encoder_action_drift_summary(matched, v),
                             "sampled_start_comparisons_n": sum(r["r7b_sampled_source_neighbor_counts"][v]
                                                                ["sampled_source_unique_starts_n"] for r in matched),
                             "sampled_slot_comparisons_n": sum(r["r7b_sampled_source_neighbor_counts"][v]
                                                               ["sampled_source_slots_n"] for r in matched),
                             "sampled_feature_neighbor_unique_count_sum": sum(r["r7b_sampled_source_neighbor_counts"][v]
                                                                              ["feature_neighbors_cosine_ge_0_95_unique_n"] for r in matched),
                             "sampled_feature_and_action_near_unique_count_sum": sum(r["r7b_sampled_source_neighbor_counts"][v]
                                                                                  ["feature_and_stored_action_near_0_10_unique_n"] for r in matched),
                             "sampled_stored_action_near_0_10_n": sum(r["r7b_sampled_source_neighbors"][v]
                                                                   ["stored_action_near_0_10"] for r in matched)}
                          for v in VARIANTS}}
    require(digest(path) == replay_meta["checkpoint_sha256"], "source checkpoint changed during analysis")
    return {"valid_n_step_starts": VALID_STARTS, "scanned_n": len(sequences),
            "scan_headroom_bytes_at_start": headroom,
            "scan_headroom_bytes_after_pool_load": scan_headroom,
            "scan_headroom_floor_bytes": MIN_HEADROOM_BYTES,
            "valid_sequence_first_last": [int(sequences[0]), int(sequences[-1])],
            "replay_stored_vs_final_source_actor": action_distribution(evolved_action, stored_action),
            "r7b_sampled_source_starts_per_variant": {v: len(samples[v][0]) for v in VARIANTS}}


def run(root: Path, output: Path, *, preflight_only: bool = False) -> dict:
    root, output = root.resolve(), output.resolve()
    require(output.is_relative_to((root / BASE).resolve()) and output != (root / BASE).resolve()
            and not (root / BASE).is_symlink() and not output.exists() and not output.parent.is_symlink(),
            "output must be a NEW directory under the final-source run root")
    context = preflight(root)
    if preflight_only:
        return {"preflight": "passed", "source_cells": 11, "archived_official_actions": 5998,
                "environment_resets": 0, "output_written": False,
                "script_sha256": digest(Path(__file__)), "phase_definitions": PHASE_DEFINITIONS,
                "full_valid_source_starts_to_scan_per_seed": VALID_STARTS,
                "paired_primary_representative_frames": 187,
                "primary_coverage_receipt_sha256": PRIMARY_RECEIPT_SHA,
                "scan_headroom_floor_bytes": MIN_HEADROOM_BYTES}
    cache_path = verified(root, context["r7"]["diagnostic_cache"]["path"], CACHE_SHA)
    with np.load(cache_path, allow_pickle=False) as archive:
        cache = archive["observations"]
    cells, frames, replay = [], [], {}
    with torch.inference_mode():
        for seed in (0, 1):
            info = next(row for row in context["protocol"]["source_actors"] if row["source_seed"] == seed)
            model = _model(root, info, info)
            for j, source in enumerate(context["selected"]):
                if source["source_learner_seed"] != seed:
                    continue
                cell, picked = reconstruct(root, source, context["inputs"]["road"][source["geometry_seed"]],
                                           model, context["refs"], cache[j])
                cells.append(cell)
                trace_path = root / RUN / "train-diagnostic" / source["trace_path"]
                with np.load(trace_path, allow_pickle=False) as archive:
                    native = archive["native_action"]
                    official = archive["official_action"]
                    archived = [(decision, observation, native[decision - 1].tolist(),
                                 official[decision - 1].tolist()) for decision, observation in sorted(picked.items())]
                for decision, observation, native_action, official_action in archived:
                    frames.append({"source_seed": seed, "geometry_seed": source["geometry_seed"],
                                   "decision": decision, "phases": [p for p in PHASES if
                                       decision in cell["phase_representatives"][p]],
                                   "observation": observation, "archived_native_action": native_action,
                                   "archived_official_action": official_action, "r7b_native_actions": {}})
            del model
            gc.collect()
    require(len(cells) == 11 and sum(c["steps"] for c in cells) == 5998 and
            len({(c["source_seed"], c["geometry_seed"]) for c in cells}) == 11 and
            len({c["geometry_seed"] for c in cells}) == 9 and
            len(frames) == 187 and all("observation" in row for row in frames),
            "incomplete full-episode source-action parity or changed representative count")
    prior = {(r["source_seed"], r["geometry_seed"], r["decision"]): r
             for r in context["primary"]["representative_frames"]}
    require(len(prior) == 187, "duplicate original representative seed/road/decision")
    for row in frames:
        key = (row["source_seed"], row["geometry_seed"], row["decision"])
        original = prior.get(key)
        require(original is not None and row["phases"] == original["phases"] and
                _state_sha(row["observation"]) == original["observation_sha256_float32"] and
                row["archived_native_action"] == original["archived_native_action"] and
                row["archived_official_action"] == original["archived_official_action"],
                f"parity reconstruction differs from immutable primary receipt: {key}")
    # Only after ALL eleven complete traces have passed parity may either seed's
    # old training replay or final r7b actor be forward-scored on these states.
    with torch.inference_mode():
        for seed in (0, 1):
            replay[str(seed)] = score_seed(root, seed, [c for c in cells if c["source_seed"] == seed],
                                           [f for f in frames if f["source_seed"] == seed], context)
    require(all("observation" not in row for row in frames) and
            all(replay[str(seed)]["scanned_n"] == VALID_STARTS for seed in (0, 1)),
            "incomplete two-seed full source replay scan")
    counts = {str(seed): {phase: {"eligible_decisions": sum(c["phase_counts"][phase] for c in cells
                                                          if c["source_seed"] == seed),
                                   "eligible_cells": sum(c["phase_counts"][phase] > 0 for c in cells
                                                         if c["source_seed"] == seed),
                                   "representative_instances": sum(len(c["phase_representatives"][phase]) for c in cells
                                                                   if c["source_seed"] == seed),
                                   "full_replay_valid_start_comparisons_n": sum(c["phase_proxy"][phase]
                                                                              ["full_replay_valid_start_comparisons_n"]
                                                                              for c in cells if c["source_seed"] == seed),
                                   "feature_neighbor_count_ge_0_95_sum": sum(c["phase_proxy"][phase]
                                                                            ["feature_neighbor_count_ge_0_95_sum"]
                                                                            for c in cells if c["source_seed"] == seed),
                                    "sampled_slot_comparisons_by_variant": {v: sum(c["phase_proxy"][phase]["r7b"][v]
                                                                                ["sampled_slot_comparisons_n"]
                                                                                for c in cells if c["source_seed"] == seed)
                                                                            for v in VARIANTS},
                                    "paired_encoder_action_drift_by_variant": {
                                        v: encoder_action_drift_summary(
                                            [r for r in frames if r["source_seed"] == seed and phase in r["phases"]], v)
                                        for v in VARIANTS}}
                           for phase in PHASES} for seed in (0, 1)}
    result = {"format": "haic-drq-final-source-offline-coverage-drift-v1", "parity_passed": True,
              "scope": "reused TRAIN-DIAGNOSTIC; source official-action replay only; development proxy, not fresh labels or official scores",
              "archived_source_actions_replayed": 5998, "source_success_cells": 11, "distinct_roads": 9,
              "parity_v2_float32_observation_instances": 666, "parity_first20_cache_frames": 220,
              "original_pixel_byte_equality_all_5998": "UNKNOWN: original per-step pixels were never archived",
              "changed_actor_environment_actions": 0, "learner_updates": 0,
              "exact_geometric_train_vs_diagnostic_cell_overlap": 0,
              "exact_pixel_overlap_scope": "all 99994/99994 valid uint8 replay stacks per seed checked against quantized reconstructed representative frames; original per-step float32 pixels unarchived",
              "causal_source_retention_explanation": "UNKNOWN",
               "feature_proxy": {"encoder": "original final source actor, shared between recovered float32 input and source replay uint8 stacks",
                                 "metric": "nearest cosine; not geometric identity, causal coverage or a learned policy return",
                                 "source_replay_scan": "ALL sorted 99994 terminal-safe 3-step starts per seed, batches <=64; no subset or outcome selection",
                                 "coverage_interpretation": "per-frame cosine >=.95 and combined stored-action <=.10 counts use all 99994 valid starts; per-arm sampled counts distinguish unique source starts from repeated slots, never independent road evidence",
                                 "phase_denominators": "phase instances overlap and are serially correlated; summed neighbor counts have representative-instance x valid-start or sampled-slot denominators, not independent roads",
                                 "near_action_rule": "native L-infinity <=0.10 against historically stored evolving-policy action; nonmonotone nearest-neighbor proxy, NOT a coverage lower bound"},
              "paired_encoder_action_drift": PAIRED_DRIFT_CONTRACT,
              "phases": PHASE_DEFINITIONS,
              "per_seed_phase_denominators": counts, "source_replay": replay,
              "input_sha256": {"r6_protocol": PROTOCOL_SHA, "r6_catalog": CATALOG_SHA, "r6_manifest": MANIFEST_SHA,
                               "v2_preflight": V2_PREFLIGHT_SHA, "v2_result": V2_RESULT_SHA,
                               "hybrid_preflight": HYBRID_PREFLIGHT_SHA, "hybrid_result": HYBRID_RESULT_SHA,
                               "first20_cache": CACHE_SHA, "r7_protocol": R7_PROTOCOL_SHA,
                               "r7_result": R7_RESULT_SHA, "r7_trace_audit": R7_TRACE_AUDIT_SHA,
                               "immutable_primary_coverage_receipt": PRIMARY_RECEIPT_SHA,
                               "source_replay_checkpoints": {s: context["r7"]["source_replay"][s]["checkpoint_sha256"]
                                                             for s in ("0", "1")},
                               "source_episode_ledgers": {s: context["r7"]["source_replay"][s]["episode_ledger_sha256"]
                                                          for s in ("0", "1")},
                               "r7b_final_traces": {f"{seed}/{v}": context["arms"][(seed, v)]["sample_trace_sha256"]
                                                     for seed in (0, 1) for v in VARIANTS},
                               "r7b_final_actors": {f"{seed}/{v}": context["arms"][(seed, v)]["actor_sha256"]
                                                     for seed in (0, 1) for v in VARIANTS},
                               "source_code": {"scripts/diagnose_drq_source_states.py": V2_SCRIPT_SHA,
                                               "scripts/diagnose_drq_r7_hybrids.py":
                                               "9f7783d67342932260ee7eca3662461949f27abe7ab46dca265ce71cdb28b99b",
                                               "scripts/audit_drq_retention_r7.py":
                                               "b283580c92445702b34152c436f515cb1481ebcd1d6a6572cb138ec31239d726"}},
              "runtime_source_sha256": context["protocol"]["source_sha256"],
              "script_sha256": digest(Path(__file__)), "cells": cells, "representative_frames": frames}
    # Absolutely no file is created until every complete action replay and all
    # pinned trace/pixel/model joins have passed. Never persist diagnostic pixels.
    require(not output.exists(), "output path appeared during audit; refusing overwrite")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "result.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"result": str(path), "sha256": digest(path), "source_cells": 11,
            "archived_official_actions": 5998, "representative_instances": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / BASE / "offline-coverage-drift")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(run(ROOT, args.output_root, preflight_only=args.preflight_only), sort_keys=True))


if __name__ == "__main__":
    main()
