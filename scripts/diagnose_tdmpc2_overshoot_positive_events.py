"""Source-bound, zero-reset RAW replay positive-event reward mechanism screen.

Preflight proposes a schema; --score needs its separately frozen exact SHA.
Neither mode creates an environment or updates a model. Run from repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import diagnose_tdmpc2_h5_logged as raw
from scripts import score_tdmpc2_overshoot_archived_branches as source


ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/diagnose_tdmpc2_overshoot_positive_events.py"
TEST = "tests/test_diagnose_tdmpc2_overshoot_positive_events.py"
PROTOCOL = "experiments/tdmpc2-overshoot-positive-events-v1.json"
OUTPUT = "runs/tdmpc2-overshoot-positive-events-20260929-v1.json"
SOURCE_SCORE_SHA = "44edd04cc7a579c3867ed35f5763645d0c92b2971fd979d5d141ea87f5802aa8"
COUNT = 2048
SEED = 20260929
AUGMENTATION_SEED = 834
BATCH = 32
GATE = {"minimum_positive_per_road": 100, "minimum_nonpositive_per_road": 100,
        "minimum_qualifying_roads": 3, "new_true_positive_signed_bias_at_most": -0.25,
        "new_true_positive_mae_increase_at_least": 0.05}


def _ref(path: str, sha: str) -> dict:
    return {"path": path, "sha256": sha}


def _sha_body(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def select(episodes: list[dict], *, count: int = COUNT, seed: int = SEED,
           roads: tuple[int, ...] = raw.ROADS) -> list[tuple[int, int]]:
    """Choose starts using only episode identity, road and length, never labels."""
    if (type(seed) is not int or type(count) is not int or count < 1
            or any(ep.get("episode") != i or ep.get("track_id") != 1
                   or ep.get("geometry_seed") not in roads or type(ep.get("length")) is not int
                   or not 1 <= ep["length"] <= 2000 for i, ep in enumerate(episodes))):
        raise ValueError("invalid complete four-road episode metadata")
    rng = np.random.RandomState(seed)
    selected = []
    for road in roads:
        candidates = [(i, step) for i, ep in enumerate(episodes)
                      if ep["geometry_seed"] == road for step in range(max(0, ep["length"] - 4))]
        if not candidates:
            raise ValueError("no H5-extendable same-episode starts on road")
        selected.extend(candidates[int(index)] for index in rng.choice(
            len(candidates), size=count, replace=len(candidates) < count))
    return sorted(selected)


def _episodes(root: Path) -> list[dict]:
    """Read only reward-free episode metadata, after fixed ledger hash checks."""
    ledger = source.branch.pinned(root, f"{raw.RUN}/training.jsonl", raw.TRAIN_SHA)
    episodes = []
    with ledger.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                raise ValueError("incomplete episode ledger")
            row = raw._json(line)
            if row.get("event") == "episode":
                episodes.append({key: row[key] for key in ("episode", "track_id", "geometry_seed", "length")})
    if len(episodes) != raw.FINAL_EPISODES:
        raise ValueError("incomplete RAW100k episodes")
    return episodes


def _proposal(root: Path, selected: list[tuple[int, int]]) -> dict:
    score = source.branch._json(source.branch.pinned(root, source.PROTOCOL, SOURCE_SCORE_SHA).read_bytes())
    hashes = {name: raw._digest(raw._file(root, name)) for name in (SELF, TEST)}
    return {"format": "haic-tdmpc2-overshoot-positive-events-protocol-v1",
            "scope": "same_original_RAW100k_consumed_track1_TRAIN_replay_only",
            "source_score_protocol": _ref(source.PROTOCOL, SOURCE_SCORE_SHA),
            "raw_source": {"protocol": _ref("experiments/tdmpc2-long-reused-train-v2.json", raw.SOURCE_PROTOCOL_SHA),
                           "result": _ref(f"{raw.RUN}/result.json", raw.RESULT_SHA),
                           "checkpoint": _ref(raw.CHECKPOINT, raw.CHECKPOINT_SHA),
                           "training_ledger": _ref(f"{raw.RUN}/training.jsonl", raw.TRAIN_SHA),
                           "step_ledger": _ref(f"{raw.RUN}/steps.jsonl", raw.STEPS_SHA)},
            "overshoot_source": score["training_source"],
            "source_sha256": {**score["source_sha256"], **hashes},
            "selection": {"method": "numpy.RandomState.choice_per_road_episode_contained_H5_reward_blind",
                          "seed": SEED, "starts_per_road": COUNT, "horizon": 5,
                          "manifest_sha256": _sha_body({"starts": selected}),
                          "augmentation_seed": AUGMENTATION_SEED, "batch_size": BATCH},
            "gate": GATE, "output": OUTPUT, "environment_resets": 0, "optimizer_updates": 0}


def preflight(root: Path, protocol_sha256: str | None = None) -> dict:
    root = Path(root).resolve(strict=True)
    # Select before calling source preflight, which reads individual step rewards.
    selected = select(_episodes(root))
    spec = _proposal(root, selected)
    if protocol_sha256 is not None:
        path = source.branch.pinned(root, PROTOCOL, source.branch._sha(protocol_sha256))
        if raw._json(path.read_bytes()) != spec:
            raise ValueError("frozen positive-event protocol differs from source-bound proposal")
    bundle = source.preflight(root, SOURCE_SCORE_SHA)
    # The two learners have different episode boundaries; selected pixels and
    # rewards must be checked against the ORIGINAL RAW replay, not the new run.
    original = raw._prepared(root)[2]
    original_episodes = original["episodes"]
    if len(original_episodes) != raw.FINAL_EPISODES or any(
            ep[key] != metadata[key] for ep, metadata in zip(original_episodes, _episodes(root))
            for key in ("episode", "track_id", "geometry_seed", "length")):
        raise ValueError("reward-blind selection metadata differs from complete source")
    # RAW cursor checks the entire original producer source map and final whole-episode boundary.
    if original["pin"]["step_prefix_sha256"] != raw.STEPS_SHA:
        raise ValueError("RAW final step cursor differs")
    return {"schema": spec, "selected": selected, "bundle": bundle,
            "original_episodes": original_episodes,
            "status": "preflight_only", "environment_resets": 0, "torch_load_calls": 0}


def bind_windows(replay: dict, selected: list[tuple[int, int]]) -> tuple[list[dict], str]:
    """Bind unaugmented action/observation/reward bytes and prohibit boundary leakage."""
    windows = []
    digest = hashlib.sha256()
    for eid, start in selected:
        ep = replay["episodes"][eid]
        actions, rewards, observations = (ep[key] for key in ("actions", "rewards", "observations"))
        term, trunc, terminal = (ep[key] for key in ("terminated", "truncated", "terminal"))
        if (start < 0 or start + 5 > len(actions) or len(observations) != len(actions) + 1
                or any(np.any(flags[start:start + 4]) for flags in (term, trunc, terminal))
                or np.any(term[start:start + 5] & ~terminal[start:start + 5])
                or np.any(terminal[start:start + 5] & ~(
                    term[start:start + 5] | trunc[start:start + 5]))):
            raise ValueError("selected window crosses termination or timeout")
        a, o, r = (np.ascontiguousarray(actions[start:start + 5]),
                   np.ascontiguousarray(observations[start:start + 5]),
                   np.ascontiguousarray(rewards[start:start + 5]))
        if a.shape != (5, 3) or a.dtype != np.float32 or o.shape != (5, 4, 64, 64) or o.dtype != np.uint8 \
                or r.shape != (5,) or r.dtype != np.float32 or not np.isfinite(r).all():
            raise ValueError("selected raw window dtype/shape/nonfinite reward")
        digest.update(np.asarray((eid, start), dtype="<i8").tobytes())
        for value in (o, a, r, term[start:start + 5], trunc[start:start + 5], terminal[start:start + 5]):
            digest.update(np.ascontiguousarray(value).tobytes())
        windows.append({"episode_id": eid, "start_step": start, "obs": o,
                        "actions": a, "rewards": r, "flags": (term[start:start + 5],
                                                                 trunc[start:start + 5], terminal[start:start + 5]),
                        "target_mask": (True, True, True, True, True)})
    return windows, digest.hexdigest()


def predict(model, windows: list[dict], *, batch_size: int = BATCH,
            seed: int = AUGMENTATION_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Encode true observation at action time; roll from the SAME encoded z0."""
    from haic.algorithms.tdmpc2.model import two_hot_inv

    true = np.empty((len(windows), 5), dtype=np.float64)
    imagined = np.empty_like(true)
    model.eval()
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        for first in range(0, len(windows), batch_size):
            group = windows[first:first + batch_size]
            # The same fixed shift stream is replayed for old/new checkpoints.
            torch.manual_seed(seed + first)
            obs = torch.from_numpy(np.stack([w["obs"] for w in group]).reshape(-1, 4, 64, 64))
            latent = model.encode(obs, None).reshape(len(group), 5, -1)
            actions = torch.from_numpy(np.stack([w["actions"] for w in group]))
            z = latent[:, 0]
            for depth in range(5):
                a = actions[:, depth]
                for output, state in ((true, latent[:, depth]), (imagined, z)):
                    decoded = two_hot_inv(model.reward(state, a, None), model.cfg)
                    if decoded.shape != (len(group), 1) or not torch.isfinite(decoded).all():
                        raise ValueError("nonfinite categorical reward prediction")
                    output[first:first + len(group), depth] = decoded[:, 0].cpu().numpy()
                if depth < 4:
                    z = model.next(z, a, None)
                    if z.shape != latent[:, 0].shape or not torch.isfinite(z).all():
                        raise ValueError("nonfinite imagined latent")
    return true, imagined


def summarize(windows: list[dict], predictions: dict[str, tuple[np.ndarray, np.ndarray]],
              roads: tuple[int, ...] = raw.ROADS) -> tuple[dict, dict]:
    target = np.stack([w["rewards"] for w in windows]).astype(np.float64)
    if target.shape != (len(roads) * COUNT, 5) or not np.isfinite(target).all():
        raise ValueError("incomplete/nonfinite five-step targets")
    summaries = {}
    counts = {}
    for road in roads:
        indices = [i for i, w in enumerate(windows) if w["road"] == road]
        if len(indices) != COUNT:
            raise ValueError("not exactly 2048 starts per road")
        distinct = {(windows[i]["episode_id"], windows[i]["start_step"] + d)
                    for i in indices for d in range(5)}
        positive = int((target[indices] > 0).sum())
        counts[str(road)] = {"windows": len(indices), "step_targets": 5 * len(indices),
                             "positive": positive, "nonpositive": 5 * len(indices) - positive,
                             "unique_transitions": len(distinct),
                             "unique_positive": len({(windows[i]["episode_id"], windows[i]["start_step"] + d)
                                                     for i in indices for d in range(5) if target[i, d] > 0}),
                             "unique_nonpositive": len({(windows[i]["episode_id"], windows[i]["start_step"] + d)
                                                        for i in indices for d in range(5) if target[i, d] <= 0}),
                             "by_depth": [{"depth": d + 1, "valid_reward_targets": len(indices),
                                           "terminated": sum(bool(windows[i]["flags"][0][d]) for i in indices),
                                           "finished_truncation": sum(bool(windows[i]["flags"][1][d]
                                                                       and windows[i]["flags"][2][d]) for i in indices),
                                           "ordinary_timeout": sum(bool(windows[i]["flags"][1][d]
                                                                   and not windows[i]["flags"][2][d]) for i in indices)}
                                          for d in range(5)]}
        for model, pair in predictions.items():
            for path, values in zip(("true_latent", "imagined"), pair):
                if values.shape != target.shape or not np.isfinite(values).all():
                    raise ValueError("incomplete/nonfinite model predictions")
                for depth in range(5):
                    for label, mask in (("positive", target[indices, depth] > 0),
                                        ("nonpositive", target[indices, depth] <= 0)):
                        error = values[indices, depth][mask] - target[indices, depth][mask]
                        summaries[f"{road}/{model}/{path}/{depth + 1}/{label}"] = {
                            "count": int(mask.sum()), "mean_signed_bias": float(error.mean()) if len(error) else None,
                            "mae": float(np.abs(error).mean()) if len(error) else None}
                for label, mask in (("positive", target[indices] > 0), ("nonpositive", target[indices] <= 0)):
                    error = (values[indices] - target[indices])[mask]
                    summaries[f"{road}/{model}/{path}/all/{label}"] = {
                        "count": int(mask.sum()), "mean_signed_bias": float(error.mean()) if len(error) else None,
                        "mae": float(np.abs(error).mean()) if len(error) else None}
    return summaries, counts


def mechanism_gate(summary: dict, counts: dict, roads: tuple[int, ...] = raw.ROADS) -> dict:
    checks = {}
    for road in roads:
        old = summary[f"{road}/old/true_latent/all/positive"]
        new = summary[f"{road}/new/true_latent/all/positive"]
        enough = counts[str(road)]["positive"] >= 100 and counts[str(road)]["nonpositive"] >= 100
        checks[str(road)] = {"adequate_counts": enough, "old_positive_true_mae": old["mae"],
                             "new_positive_true_mae": new["mae"],
                             "new_positive_true_bias": new["mean_signed_bias"],
                             "qualifies": bool(enough and new["mean_signed_bias"] is not None
                                               and old["mae"] is not None
                                               and new["mean_signed_bias"] <= -0.25
                                               and new["mae"] >= old["mae"] + 0.05)}
    adequate = all(row["adequate_counts"] for row in checks.values())
    passed = adequate and sum(row["qualifies"] for row in checks.values()) >= 3
    return {"status": "PASS" if passed else "FAIL", "adequate_counts": adequate,
            "qualifying_roads": sum(row["qualifies"] for row in checks.values()), "by_road": checks,
            "interpretation": ("reward_head_only_adaptation_hypothesis_only" if passed else
                               "head_only_gate_failed_consider_dynamics_or_coverage_if_errors_only_imagined"),
            "archived_four_way_score_still_failed": True, "full_policy_unlocked": False}


def score(root: Path, protocol_sha256: str) -> dict:
    root = Path(root).resolve(strict=True)
    prepared = preflight(root, protocol_sha256)
    if (root / OUTPUT).exists() or (root / OUTPUT).is_symlink():
        raise ValueError("exclusive output already exists")
    # Rehash all pins immediately before both checkpoint deserializations.
    preflight(root, protocol_sha256)
    old, new = source._models(root, prepared["bundle"])
    preflight(root, protocol_sha256)  # The RAW replay is loaded again solely for selected pixels.
    with source._file(root, prepared["schema"]["raw_source"]["checkpoint"]).open("rb") as stream:
        if raw._digest_stream(stream) != raw.CHECKPOINT_SHA:
            raise ValueError("RAW checkpoint changed after model load")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    raw._replay(state, prepared["original_episodes"], raw._windows(
        prepared["original_episodes"], prepared["selected"],
        raw._file(root, f"{raw.RUN}/steps.jsonl")))
    windows, bytes_sha = bind_windows(state["replay"], prepared["selected"])
    for w in windows:
        w["road"] = prepared["original_episodes"][w["episode_id"]]["geometry_seed"]
    predictions = {"old": predict(old, windows), "new": predict(new, windows)}
    summary, counts = summarize(windows, predictions)
    gate = mechanism_gate(summary, counts)
    preflight(root, protocol_sha256)  # Recheck every input SHA after inference.
    body = {"format": "haic-tdmpc2-overshoot-positive-events-result-v1",
            "protocol": _ref(PROTOCOL, protocol_sha256), "source": prepared["schema"],
            "selected_bytes_sha256": bytes_sha, "road_counts": counts,
            "metrics_by_road_model_path_depth_label": summary, "gate": gate,
            "all_five_targets_valid_in_each_episode_contained_window": True,
            "no_action_or_observation_after_terminal_or_timeout": True,
            "dependent_overlapping_windows": True, "environment_resets": 0, "optimizer_updates": 0,
            "scope": "consumed_TRAIN_replay_mechanism_only_not_policy_or_official_score"}
    write_receipt(root, body)
    return body


def write_receipt(root: Path, body: dict) -> None:
    output = root / OUTPUT
    if output.parent.is_symlink() or not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise ValueError("exclusive receipt requires absent file and existing real runs directory")
    payload = json.dumps({**body, "body_sha256": _sha_body(body)}, sort_keys=True,
                         indent=2, allow_nan=False) + "\n"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(output, flags, 0o644), "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score and not args.protocol_sha256:
        parser.error("--score requires separately frozen --protocol-sha256")
    result = score(ROOT, args.protocol_sha256) if args.score else preflight(ROOT, args.protocol_sha256)
    if not args.score:
        result = {key: result[key] for key in ("schema", "status", "environment_resets", "torch_load_calls")}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
