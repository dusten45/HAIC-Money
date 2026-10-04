"""No-reset, in-replay H3/H5 logged-return fit of the frozen RAW 100k model.

Default preflight proposes a protocol for separate review/freeze. Scoring requires
that protocol's externally supplied SHA and writes an exclusive runs/ receipt.
torch.load is permitted only on the fixed, SHA-checked trusted local checkpoint.
This is neither an H5 planner test nor a same-state counterfactual ranking.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = "runs/tdmpc2-long-20260928-v2"
PROTOCOL = "experiments/tdmpc2-h5-logged-v1.json"  # Frozen by the main operator, not this CLI.
HELPER = "scripts/diagnose_tdmpc2_checkpoint_losses.py"
HELPER_SHA = "ef6552479192e2e113951edb3bfbf7a4c87b3a508dd884c45ce6a270a0aebe6f"
SOURCE_PROTOCOL_SHA = "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"
RESULT_SHA = "287470932dbba9a7eeed778a5be7762b736e6679d99fd29078c5103890a75a4b"
CHECKPOINT = f"{RUN}/checkpoint-at-least-100000-step-100354.pt"
CHECKPOINT_SHA = "aa268b95a7398b18474fbbaea153fb5e20cc363345cf3c0e6176aaa9dc72c295"
TRAIN_SHA = "84f06ee4dcc8568bd5d384d1811252321ad5dc6a98ce19d18badb448d09123d7"
STEPS_SHA = "a94ce156afcb6e4f706474d6cf2d2ccb4754a868eb75a3eb4b53df5d2be37945"
FINAL_TARGET, FINAL_DECISIONS, FINAL_UPDATES, FINAL_EPISODES = 100000, 100354, 100354, 307
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
SELECTION_SEED = 20260929
AUGMENTATION_SEED = 834
COUNT = 256
END_ALIGNED = 32
GAMMA = 0.995
QUALITY_GATE = {
    "scope": "necessary_for_separately_frozen_real_H5_branch_design_only_not_planner_launch",
    "finite_predictions_required": True,
    "h5_mae_per_discounted_step_strictly_below_zero_constant": True,
    "h5_semantic_terminal_tpr_strictly_greater_than_fpr": True,
    "terminal_positive_probability": ">0.5",
}


def _file(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("noncanonical artifact path")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in artifact path: {name}")
    if not path.is_file():
        raise ValueError(f"missing artifact: {name}")
    return path


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return _digest_stream(stream)


def _digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")
    return value


def _json(data: bytes) -> dict:
    if not data or len(data) > 1024 * 1024:
        raise ValueError("invalid JSON record size")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    result = json.loads(data, object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def _select(episodes: list[dict], *, seed: int = SELECTION_SEED,
            count: int = COUNT, end_aligned: int = END_ALIGNED) -> list[tuple[int, int]]:
    """Select without reading rewards/finish flags; include episode-end coverage."""
    if any(type(ep.get("length")) is not int or ep["length"] < 5 or ep.get("episode") != i
           for i, ep in enumerate(episodes)) or len(episodes) < end_aligned:
        raise ValueError("not enough complete H5 replay episodes")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(episodes), size=end_aligned, replace=False)
    chosen = {(int(i), episodes[int(i)]["length"] - 5) for i in ids}
    eligible = [(i, start) for i, ep in enumerate(episodes) for start in range(ep["length"] - 4)
                if (i, start) not in chosen]
    if len(eligible) < count - end_aligned:
        raise ValueError("not enough distinct complete H5 windows")
    chosen.update(eligible[int(i)] for i in rng.choice(len(eligible), count - end_aligned, replace=False))
    return sorted(chosen)


def _windows(episodes: list[dict], selections: list[tuple[int, int]], steps: Path) -> list[dict]:
    """Bind selected starts/actions/targets to the already SHA-verified full ledger."""
    wanted = set(selections)
    found = []
    with steps.open("rb") as stream:
        for eid, ep in enumerate(episodes):
            recent: deque = deque(maxlen=5)
            for offset in range(ep["length"]):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete step ledger")
                row = _json(line)
                if (row.get("episode") != eid or row.get("decision") != ep["decisions"] - ep["length"] + offset + 1
                        or row.get("track_id") != ep["track_id"]
                        or row.get("geometry_seed") != ep["geometry_seed"]):
                    raise ValueError("step/episode identity mismatch")
                if not isinstance(row.get("action_f32_hex"), str) or len(row["action_f32_hex"]) != 24:
                    raise ValueError("invalid logged 3D action bytes")
                try:
                    action = bytes.fromhex(row["action_f32_hex"])
                except ValueError as exc:
                    raise ValueError("invalid logged action hex") from exc
                values = np.frombuffer(action, dtype="<f4")
                if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
                    raise ValueError("nonfinite or out-of-bounds logged action")
                recent.append((action, row["reward"], row["terminated"], row["truncated"], row["terminal"]))
                start = offset - 4
                if (eid, start) in wanted:
                    actions, rewards, terminated, truncated, terminal = map(tuple, zip(*recent))
                    found.append({"episode_id": eid, "start_step": start, "track_id": ep["track_id"],
                                  "geometry_seed": ep["geometry_seed"],
                                  "actions_sha256": hashlib.sha256(b"".join(actions)).hexdigest(),
                                  "actions_bytes": b"".join(actions), "raw_rewards": rewards,
                                  "terminated": terminated, "truncated": truncated, "terminal": terminal})
        if stream.read(1):
            raise ValueError("step ledger extends past completed source")
    if len(found) != len(selections) or sorted((w["episode_id"], w["start_step"]) for w in found) != selections:
        raise ValueError("selected H5 windows missing from complete episode ledger")
    return sorted(found, key=lambda w: (w["episode_id"], w["start_step"]))


def _targets(windows: list[dict]) -> dict:
    rewards = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float64)
    if rewards.shape != (COUNT, 5) or not np.isfinite(rewards).all():
        raise ValueError("missing/nonfinite H5 reward targets")
    result = {}
    for horizon in (3, 5):
        values = rewards[:, :horizon] @ (GAMMA ** np.arange(horizon))
        if np.ptp(values) <= 1e-6 or len(set(np.rint(values * 1e6))) < 2:
            raise ValueError(f"degenerate H{horizon} return targets")
        positive = sum(sum(w["terminal"][:horizon]) for w in windows)
        result[f"h{horizon}"] = {"target_min": float(values.min()), "target_max": float(values.max()),
                                 "terminal_positive_transitions": positive,
                                 "terminal_negative_transitions": COUNT * horizon - positive,
                                  "terminated_positive_transitions": sum(sum(w["terminated"][:horizon]) for w in windows),
                                  "truncated_positive_transitions": sum(sum(w["truncated"][:horizon]) for w in windows),
                                  "finished_truncation_transitions": sum(sum(t and d and not r for t, d, r in zip(
                                      w["truncated"][:horizon], w["terminal"][:horizon],
                                      w["terminated"][:horizon])) for w in windows),
                                  "ordinary_timeout_transitions": sum(sum(t and not d for t, d in zip(
                                      w["truncated"][:horizon], w["terminal"][:horizon])) for w in windows),
                                  "transition_count": COUNT * horizon}
    if not 0 < result["h5"]["terminal_positive_transitions"] < COUNT * 5:
        raise ValueError("degenerate H5 semantic-terminal labels")
    if result["h3"]["terminal_positive_transitions"] != 0:
        raise ValueError("H3 prefix crosses a reset in a complete H5 window")
    return result


def _prepared(root: Path) -> tuple[dict, list[dict], dict]:
    """All source/result/full-cursor checks precede any torch.load."""
    if _digest(_file(root, HELPER)) != HELPER_SHA:
        raise ValueError("source hash mismatch: checkpoint cursor helper")
    from scripts import diagnose_tdmpc2_checkpoint_losses as bound

    original = bound._protocol(root)  # Checks the entire frozen original source map.
    result_path = _file(root, f"{RUN}/result.json")
    if _digest(result_path) != RESULT_SHA:
        raise ValueError("RAW source result SHA mismatch")
    result = _json(result_path.read_bytes())
    if (result.get("status") != "completed_boundary_at_least_100k"
            or result.get("protocol_sha256") != SOURCE_PROTOCOL_SHA
            or result.get("source_sha256") != original["source_sha256"]
            or result.get("reused_train_only") is not True or result.get("evaluation") is not None
            or result.get("resume_supported") is not False
            or any(result.get(k) != v for k, v in {"decisions": FINAL_DECISIONS, "updates": FINAL_UPDATES,
                                                  "episodes": FINAL_EPISODES, "action_dim": 3}.items())
            or not isinstance(result.get("checkpoints"), list)
            or [r.get("target") for r in result["checkpoints"]] != list(bound.TARGETS)
            or result.get("training_ledger_sha256") != TRAIN_SHA
            or result.get("step_ledger_sha256") != STEPS_SHA):
        raise ValueError("not the complete RAW 100k source result")
    for name, expected in ((f"{RUN}/training.jsonl", TRAIN_SHA), (f"{RUN}/steps.jsonl", STEPS_SHA),
                           (CHECKPOINT, CHECKPOINT_SHA)):
        if _digest(_file(root, name)) != expected:
            raise ValueError(f"full artifact SHA mismatch: {name}")
    pin = bound._cursor(root, original, FINAL_TARGET)
    if (pin["row"] != {"event": "checkpoint", **result["checkpoints"][-1]}
            or pin["checkpoint_sha256"] != CHECKPOINT_SHA or pin["step_prefix_sha256"] != STEPS_SHA
            or pin["ledger_prefix_sha256"] != TRAIN_SHA):
        raise ValueError("final whole-episode checkpoint/full-ledger cursor differs from result")
    episodes = []
    with _file(root, f"{RUN}/training.jsonl").open("rb") as stream:
        for line in stream:
            row = _json(line)
            if row.get("event") == "episode":
                episodes.append(row)
    if (len(episodes) != FINAL_EPISODES or any(ep.get("episode") != i or ep.get("track_id") != 1
                                    or ep.get("geometry_seed") != ROADS[i % 4] for i, ep in enumerate(episodes))):
        raise ValueError("completed episodes differ from four consumed TRAIN roads")
    windows = _windows(episodes, _select(episodes, seed=SELECTION_SEED, count=COUNT,
                                         end_aligned=END_ALIGNED), _file(root, f"{RUN}/steps.jsonl"))
    targets = _targets(windows)
    manifest = [{key: w[key] for key in ("episode_id", "start_step", "actions_sha256")} for w in windows]
    manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    proposal = {
        "format": "haic-tdmpc2-h5-logged-v1",
        "purpose": "consumed-TRAIN-in-replay-observational-reward-fit",
        "source_protocol": {"path": bound.PROTOCOL, "sha256": SOURCE_PROTOCOL_SHA},
        "source_result": {"path": f"{RUN}/result.json", "sha256": RESULT_SHA},
        "checkpoint": {"path": CHECKPOINT, "sha256": CHECKPOINT_SHA},
        "training_ledger": {"path": f"{RUN}/training.jsonl", "sha256": TRAIN_SHA},
        "step_ledger": {"path": f"{RUN}/steps.jsonl", "sha256": STEPS_SHA},
        "source_sha256": {"scripts/diagnose_tdmpc2_h5_logged.py": _digest(_file(root, "scripts/diagnose_tdmpc2_h5_logged.py")),
                          HELPER: HELPER_SHA},
        "selection": {"method": "32-uniform-episode-end-plus-224-uniform-other-H5-windows-no-replacement",
                      "seed": SELECTION_SEED, "window_count": COUNT, "end_aligned_count": END_ALIGNED,
                      "manifest_sha256": manifest_sha, "augmentation_seed": AUGMENTATION_SEED,
                      "horizons": [3, 5], "discount": GAMMA, "constant_reward_per_step": 0.0,
                      "quality_gate": QUALITY_GATE},
        "environment_resets": 0,
    }
    return proposal, windows, {"targets": targets, "episodes": episodes, "pin": pin,
                               "manifest": manifest, "original_sources": original["source_sha256"]}


def preflight(*, root: Path = ROOT, protocol_sha256: str | None = None) -> dict:
    root = root.resolve(strict=True)
    proposal, windows, context = _prepared(root)
    if protocol_sha256 is not None:
        _frozen_protocol(root, proposal, protocol_sha256)
    return {"status": "preflight_only", "environment_resets": 0, "torch_load_calls": 0,
            "proposed_protocol_path": PROTOCOL, "frozen_protocol_sha256": protocol_sha256,
            "protocol_schema": proposal, "window_manifest": context["manifest"],
            "window_count": len(windows), "distinct_episode_count": len(set(w["episode_id"] for w in windows)),
            "road_window_counts": {str(road): sum(w["geometry_seed"] == road for w in windows) for road in ROADS},
            "target_coverage": context["targets"]}


def _frozen_protocol(root: Path, proposal: dict, expected_sha: str) -> None:
    path = _file(root, PROTOCOL)
    if _digest(path) != _sha(expected_sha):
        raise ValueError("own frozen H5 protocol SHA mismatch")
    if _json(path.read_bytes()) != proposal:
        raise ValueError("own frozen H5 protocol differs from source-bound schema")


def _replay(state: dict, episodes: list[dict], windows: list[dict]) -> tuple[dict, list[str]]:
    replay = state.get("replay")
    if (not isinstance(replay, dict) or replay.get("format") != "haic-tdmpc2-episode-replay-v1"
            or any(replay.get(k) != v for k, v in {"capacity": 120000, "horizon": 3,
                                                   "action_dim": 3, "augmentation_pad": 3,
                                                   "include_partial": False, "bootstrap_on_truncation": True,
                                                   "observation_shape": (4, 64, 64),
                                                   "next_episode_id": len(episodes),
                                                   "size": sum(ep["length"] for ep in episodes),
                                                   "active": None}.items())
            or not isinstance(replay.get("episodes"), list) or len(replay["episodes"]) != len(episodes)):
        raise ValueError("checkpoint replay is not the complete RAW H3 episode boundary")
    for i, (ep, row) in enumerate(zip(replay["episodes"], episodes)):
        length = row["length"]
        if (not isinstance(ep, dict) or ep.get("episode_id") != i or ep.get("start_step") != 0
                or any(not isinstance(ep.get(k), np.ndarray) or ep[k].shape != shape or ep[k].dtype != dtype
                       for k, shape, dtype in (("observations", (length + 1, 4, 64, 64), np.uint8),
                                               ("actions", (length, 3), np.float32),
                                               ("rewards", (length,), np.float32),
                                               *((key, (length,), np.bool_) for key in ("terminated", "truncated", "terminal"))))):
            raise ValueError("replay episode shape/dtype/identity differs from completed ledger")
        if (not np.isfinite(ep["actions"]).all() or np.any(np.abs(ep["actions"]) > 1)
                or not np.isfinite(ep["rewards"]).all()
                or any(ep[k][:-1].any() for k in ("terminated", "truncated", "terminal"))
                or any(bool(ep[k][-1]) is not row[k] for k in ("terminated", "truncated", "terminal"))
                or hashlib.sha256(ep["actions"].tobytes()).hexdigest() != row["action_trace_sha256"]):
            raise ValueError("replay crosses a reset or action/terminal ledger differs")
    anchors = []
    for w in windows:
        ep = replay["episodes"][w["episode_id"]]
        start = w["start_step"]
        if (start < 0 or start + 5 > len(ep["actions"])
                or ep["actions"][start:start + 5].tobytes() != w["actions_bytes"]
                or not np.array_equal(ep["rewards"][start:start + 5], np.asarray(w["raw_rewards"], dtype=np.float32))
                or any(not np.array_equal(ep[key][start:start + 5], np.asarray(w[key], dtype=np.bool_))
                       for key in ("terminated", "truncated", "terminal"))):
            raise ValueError("selected replay H5 window differs from logged actions/rewards/flags")
        anchors.append(hashlib.sha256(ep["observations"][start].tobytes()).hexdigest())
    return replay, anchors


def _rollout(model, replay: dict, windows: list[dict], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode each original anchor once; free-roll next/reward/termination five times."""
    from haic.algorithms.tdmpc2.model import two_hot_inv

    obs = torch.from_numpy(np.stack([replay["episodes"][w["episode_id"]]["observations"][w["start_step"]]
                                     for w in windows]))
    actions = torch.from_numpy(np.stack([np.frombuffer(w["actions_bytes"], dtype="<f4").reshape(5, 3)
                                         for w in windows]).copy()).transpose(0, 1)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.default_generator.manual_seed(seed)  # CPU shifts only; leave CUDA RNG untouched.
        model.eval()
        z = model.encode(obs, None)
        predicted, terminal_probability = [], []
        for t in range(5):
            predicted.append(two_hot_inv(model.reward(z, actions[t], None), model.cfg).squeeze(-1).cpu().numpy())
            z = model.next(z, actions[t], None)
            terminal_probability.append(model.termination(z, None).squeeze(-1).cpu().numpy())
    predicted, terminal_probability = np.stack(predicted), np.stack(terminal_probability)
    if predicted.shape != (5, len(windows)) or terminal_probability.shape != predicted.shape:
        raise ValueError("invalid world-model five-step prediction shape")
    return predicted, terminal_probability


def _metrics(windows: list[dict], predicted: np.ndarray, probability: np.ndarray) -> tuple[dict, list[dict]]:
    actual = np.asarray([w["raw_rewards"] for w in windows], dtype=np.float64).T
    labels = np.asarray([w["terminal"] for w in windows], dtype=bool).T
    raw_ends = np.asarray([w["terminated"] for w in windows], dtype=bool).T
    truncated = np.asarray([w["truncated"] for w in windows], dtype=bool).T
    valid_probability = np.isfinite(probability) & (probability >= 0) & (probability <= 1)
    predicted_positive = valid_probability & (probability > 0.5)  # Planner uses strict >, not >=.
    depths = []
    for t in range(5):
        positive = labels[t]
        guessed = predicted_positive[t]
        valid = valid_probability[t]
        depths.append({"depth": t + 1, "transitions": len(windows),
                       "reward_target_min": float(actual[t].min()),
                       "reward_target_max": float(actual[t].max()),
                       "reward_mae": float(np.abs(actual[t] - predicted[t]).mean())
                       if np.isfinite(predicted[t]).all() else None,
                       "zero_constant_reward_mae": float(np.abs(actual[t]).mean()),
                       "raw_terminated_positive": int(raw_ends[t].sum()),
                       "finished_truncation_positive": int((truncated[t] & positive & ~raw_ends[t]).sum()),
                       "ordinary_time_limit_positive": int((truncated[t] & ~positive).sum()),
                       "raw_terminated_true_positive": int((guessed & raw_ends[t]).sum()),
                       "finished_truncation_true_positive": int((guessed & truncated[t] & positive & ~raw_ends[t]).sum()),
                       "ordinary_time_limit_false_positive": int((guessed & truncated[t] & ~positive).sum()),
                       "semantic_terminal_positive": int(positive.sum()),
                       "semantic_terminal_negative": int((~positive).sum()),
                       "predicted_positive_gt_0_5": int(guessed.sum()),
                       "predicted_invalid_probability": int((~valid).sum()),
                       "true_positive": int((guessed & positive).sum()),
                       "false_positive": int((guessed & ~positive).sum()),
                       "false_negative": int((valid & ~guessed & positive).sum()),
                       "true_negative": int((valid & ~guessed & ~positive).sum())})
    def finite_or_null(value):
        return float(value) if math.isfinite(value) else None

    scored = []
    for i, w in enumerate(windows):
        scored.append({"episode_id": w["episode_id"], "start_step": w["start_step"],
                       "track_id": w["track_id"], "geometry_seed": w["geometry_seed"],
                       "actions_sha256": w["actions_sha256"],
                       "actual_rewards": list(w["raw_rewards"]),
                       "predicted_rewards": [finite_or_null(x) for x in predicted[:, i]],
                       "terminated": list(w["terminated"]), "truncated": list(w["truncated"]),
                       "terminal": list(w["terminal"]),
                       "predicted_terminal_probability": [finite_or_null(x) for x in probability[:, i]]})
    horizons = {}
    for h in (3, 5):
        weights = GAMMA ** np.arange(h)
        real = weights @ actual[:h]
        with np.errstate(invalid="ignore", over="ignore"):
            estimate = weights @ predicted[:h]
        finite_rewards = bool(np.isfinite(predicted[:h]).all() and np.isfinite(estimate).all())
        finite_probs = bool(valid_probability[:h].all())
        horizon_depths = depths[:h]
        mae = float(np.abs(real - estimate).mean()) if finite_rewards else None
        constant_mae = float(np.abs(real).mean())
        horizons[f"h{h}"] = {
            "windows": len(windows), "reward_transitions": len(windows) * h,
            "target_return_min": float(real.min()), "target_return_max": float(real.max()),
            "predicted_return_min": float(estimate.min()) if finite_rewards else None,
            "predicted_return_max": float(estimate.max()) if finite_rewards else None,
            "reward_mae": float(np.mean(np.abs(actual[:h] - predicted[:h]))) if finite_rewards else None,
            "discounted_return_mae": mae,
            "constant_reward_per_step": 0.0,
            "constant_reward_discounted_return_mae": constant_mae,
            "effective_discounted_steps": float(weights.sum()),
            "mae_per_discounted_step": mae / float(weights.sum()) if mae is not None else None,
            "constant_mae_per_discounted_step": constant_mae / float(weights.sum()),
            "finite_predictions": finite_rewards and finite_probs,
            "semantic_terminal_positive_transitions": int(labels[:h].sum()),
            "semantic_terminal_negative_transitions": int(labels[:h].size - labels[:h].sum()),
            "raw_terminated_positive_transitions": sum(d["raw_terminated_positive"] for d in horizon_depths),
            "finished_truncation_positive_transitions": sum(d["finished_truncation_positive"] for d in horizon_depths),
            "ordinary_time_limit_transitions": sum(d["ordinary_time_limit_positive"] for d in horizon_depths),
            "raw_terminated_true_positive": sum(d["raw_terminated_true_positive"] for d in horizon_depths),
            "finished_truncation_true_positive": sum(d["finished_truncation_true_positive"] for d in horizon_depths),
            "ordinary_time_limit_false_positive": sum(d["ordinary_time_limit_false_positive"] for d in horizon_depths),
            "predicted_terminal_positive_gt_0_5": sum(d["predicted_positive_gt_0_5"] for d in horizon_depths),
            "terminal_true_positive": sum(d["true_positive"] for d in horizon_depths),
            "terminal_false_positive": sum(d["false_positive"] for d in horizon_depths),
            "terminal_false_negative": sum(d["false_negative"] for d in horizon_depths),
            "terminal_invalid_probability": sum(d["predicted_invalid_probability"] for d in horizon_depths),
            "terminal_by_depth": horizon_depths,
        }
        for i, row in enumerate(scored):
            row[f"h{h}"] = {"actual_discounted_reward": float(real[i]),
                             "predicted_discounted_reward": finite_or_null(estimate[i])}
    return horizons, scored


def _quality_gate(h5: dict, gate: dict) -> dict:
    if gate != QUALITY_GATE:
        raise ValueError("H5 branch-design gate differs from frozen proposal")
    positives, negatives = (h5[key] for key in ("semantic_terminal_positive_transitions",
                                               "semantic_terminal_negative_transitions"))
    valid = h5["finite_predictions"] and positives > 0 and negatives > 0
    tpr = h5["terminal_true_positive"] / positives if valid else None
    fpr = h5["terminal_false_positive"] / negatives if valid else None
    return_fit = valid and h5["mae_per_discounted_step"] < h5["constant_mae_per_discounted_step"]
    terminal_fit = bool(valid and tpr is not None and fpr is not None and tpr > fpr)
    reasons = []
    if not valid:
        reasons.append("nonfinite_or_invalid_predictions_or_missing_terminal_class")
    if not return_fit:
        reasons.append("h5_reward_return_mae_not_strictly_below_zero_constant")
    if not terminal_fit:
        reasons.append("h5_semantic_terminal_tpr_not_strictly_above_fpr")
    return {"quality_gate_passed": not reasons, "failure_reasons": reasons,
            "finite_predictions": h5["finite_predictions"], "h5_return_below_constant": bool(return_fit),
            "h5_terminal_tpr_above_fpr": bool(terminal_fit), "h5_terminal_tpr": tpr, "h5_terminal_fpr": fpr,
            "h5_terminal_positive_denominator": positives, "h5_terminal_negative_denominator": negatives}


def score(*, root: Path = ROOT, protocol_sha256: str) -> dict:
    root = root.resolve(strict=True)
    proposal, windows, context = _prepared(root)
    _frozen_protocol(root, proposal, protocol_sha256)
    pin = context["pin"]
    # Recheck all input bytes after selecting windows and before pickle deserialization.
    _frozen_protocol(root, proposal, protocol_sha256)
    for name, digest in ((HELPER, HELPER_SHA), ("scripts/diagnose_tdmpc2_h5_logged.py",
                                             proposal["source_sha256"]["scripts/diagnose_tdmpc2_h5_logged.py"])):
        if _digest(_file(root, name)) != digest:
            raise ValueError(f"source/checkpoint changed before torch.load: {name}")
    for name, digest in ((f"{RUN}/result.json", RESULT_SHA), (f"{RUN}/training.jsonl", TRAIN_SHA),
                         (f"{RUN}/steps.jsonl", STEPS_SHA), *context["original_sources"].items()):
        if _digest(_file(root, name)) != digest:
            raise ValueError(f"source/result/ledger changed before torch.load: {name}")
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel

    with torch.random.fork_rng(devices=[]):
        # Hash and deserialize the SAME open file, so replacement after hashing cannot swap the pickle.
        with _file(root, CHECKPOINT).open("rb") as stream:
            if _digest_stream(stream) != CHECKPOINT_SHA:
                raise ValueError("checkpoint changed before torch.load")
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=False)
        if (not isinstance(state, dict) or state.get("format") != "haic-tdmpc2-long-train-v1"
                or state.get("protocol_sha256") != SOURCE_PROTOCOL_SHA
                or state.get("source_sha256") != context["original_sources"]
                or any(state.get(k) != v for k, v in {"target": FINAL_TARGET, "decisions": FINAL_DECISIONS,
                                                      "episodes": FINAL_EPISODES, "updates": FINAL_UPDATES,
                                                      "action_dim": 3,
                                                      "resume_supported": False}.items())):
            raise ValueError("checkpoint metadata differs from RAW full source")
        replay, anchor_hashes = _replay(state, context["episodes"], windows)
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
        weights = state.get("learner")
        if not isinstance(weights, dict) or "q_scale" not in weights:
            raise ValueError("checkpoint has no frozen final learner weights")
        model.load_state_dict({name.removeprefix("model."): value for name, value in weights.items()
                               if name.startswith("model.")}, strict=True)
        predicted, probability = _rollout(model, replay, windows, AUGMENTATION_SEED)
    horizons, rows = _metrics(windows, predicted, probability)
    gate = _quality_gate(horizons["h5"], proposal["selection"]["quality_gate"])
    limitations = ["H3 semantic-terminal recall unavailable: no positive labels by H5-window construction",
                   f"H5 ordinary time-limit behavior unmeasured: {horizons['h5']['ordinary_time_limit_transitions']} plain timeouts",
                   f"H5 finish-specific recall not validated: only {horizons['h5']['finished_truncation_positive_transitions']} "
                   "finished truncations among semantic terminal labels"]
    for row, anchor in zip(rows, anchor_hashes):
        row["anchor_observation_sha256"] = anchor
    return {"format": "haic-tdmpc2-h5-logged-result-v1", "scope": "in_replay_observational_fit_only",
            "same_state_counterfactual_ranking": False, "planner_or_policy_benefit": False,
            "generalization_or_official_score": False, "environment_resets": 0, "optimizer_steps": 0,
            "protocol_path": PROTOCOL, "protocol_sha256": protocol_sha256,
            "source_protocol_sha256": SOURCE_PROTOCOL_SHA, "source_result_sha256": RESULT_SHA,
            "source_checkpoint_sha256": CHECKPOINT_SHA, "training_ledger_sha256": TRAIN_SHA,
            "step_ledger_sha256": STEPS_SHA, "source_sha256": proposal["source_sha256"],
            "window_manifest_sha256": proposal["selection"]["manifest_sha256"],
            "window_count": len(windows), "distinct_episode_count": len(set(w["episode_id"] for w in windows)),
            "episode_ids": sorted(set(w["episode_id"] for w in windows)),
            "road_window_counts": {str(road): sum(w["geometry_seed"] == road for w in windows) for road in ROADS},
            "h3_positive_labels_unavailable_by_H5_window_construction": True,
            "timeout_is_not_semantic_terminal": True, "horizons": horizons, "windows": rows,
            "quality_gate": proposal["selection"]["quality_gate"],
            "quality_gate_passed": gate["quality_gate_passed"], "quality_gate_checks": gate,
            "terminal_label_limitations": limitations,
            "h5_real_branch_and_planner_unverified": True,
            "torch_version": str(torch.__version__)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: verify source and propose protocol, no load")
    mode.add_argument("--score", action="store_true", help="score only after a separate protocol freeze")
    parser.add_argument("--protocol-sha256", help="SHA-256 of main-owned frozen H5 experiment protocol")
    parser.add_argument("--output", type=Path, help="new exclusive direct runs/ receipt; --score only")
    args = parser.parse_args()
    if args.score:
        if args.protocol_sha256 is None or args.output is None:
            parser.error("--score requires --protocol-sha256 and --output")
        output = args.output.absolute()
        if (output.parent != ROOT / "runs" or output.parent.is_symlink()
                or not output.name.startswith("tdmpc2-long-v2-h5-logged-") or not output.name.endswith(".json")
                or output.is_symlink() or output.exists()):
            parser.error("--output must be a unique unused runs/tdmpc2-long-v2-h5-logged-*.json receipt")
        report = score(root=ROOT, protocol_sha256=args.protocol_sha256)
        payload = json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                          "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                          "environment_resets": 0}, sort_keys=True))
    else:
        if args.output is not None:
            parser.error("--output requires --score")
        print(json.dumps(preflight(root=ROOT, protocol_sha256=args.protocol_sha256), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
