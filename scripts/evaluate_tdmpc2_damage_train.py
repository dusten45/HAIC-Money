"""CPU-only, raw-outcome evaluation of a COMPLETE damage-target TD-MPC2 model.

Run from the repository root with ``python -m scripts.evaluate_tdmpc2_damage_train``.
Default --preflight reads files only. --execute requires a separately frozen
evaluation protocol and its externally supplied SHA; it never resumes an output.
These four repeatedly used TRAIN roads are not a generalization or official test.
Never point this operator at untrusted checkpoints: torch.load uses pickle, and
is reached only after the protocol, source, entire ledgers and checkpoint SHA pass.
Replay/probe checks attest stored data targets, not the optimizer history of weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_tdmpc2_full_train as raw


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-damage-full-consumed-train-v1"
TRAIN_FORMAT = "haic-tdmpc2-damage-train-v1"
TARGETS = (20000, 40000, 70000, 100000)
CELLS = raw.CELLS
MODES = raw.MODES
ENVIRONMENT = raw.ENVIRONMENT
REWARD_TARGET = {"kind": "cumulative_damage_increment", "coefficient": 5.0,
                 "initial_damage": 0.0, "maximum_damage": 1.0,
                 "negative_delta_tolerance": 1e-6, "replay_reward": "training_reward",
                 "environment_reward": "raw", "checkpoint_reward_prediction": "training_reward"}
BASELINE = {
    "training_protocol": {"path": "experiments/tdmpc2-long-reused-train-v2.json",
                          "sha256": "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec"},
    "training_result": {"path": "experiments/tdmpc2-long-reused-train-v2-100k-result.json",
                        "sha256": "bacf4c7e7a53794ecec7797746388695bac01096287f7aff09b5feafc4e3b515"},
    "full_eval_protocol": {"path": "experiments/tdmpc2-full-consumed-train-v1.json",
                           "sha256": "874d01d1396697efc9e4b119b0cd9ce8189fe36fa8d44773dcc13d30ab9ddbcd"},
    "full_eval_result": {"path": "experiments/tdmpc2-full-consumed-train-v1-result.json",
                         "sha256": "3bc41dd2a35bfd488a6cd13a2778d9344f6929873edd77e701b067e31fd1366b"},
}
RAW_EVALUATOR_SHA = "d2f3df9522257f2dfbffe5167571c5a491b7929acbbbcb3b84ebb0c2f8b91089"
TRAIN_SOURCE_PATHS = raw.SOURCE_PATHS | {
    "scripts/train_tdmpc2_damage.py", "haic/algorithms/tdmpc2/reward.py"}


def _exact(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(value, dict)
        return value.keys() == expected.keys() and all(_exact(value[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        assert isinstance(value, list)
        return len(value) == len(expected) and all(_exact(a, b) for a, b in zip(value, expected))
    return value == expected


def _baseline(root: Path, refs: dict) -> dict:
    if refs != BASELINE or raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_full_train.py")) != RAW_EVALUATOR_SHA:
        raise ValueError("frozen raw baseline/evaluator source differs")
    training, _ = raw._reference(root, refs["training_protocol"], "experiments/")
    result, _ = raw._reference(root, refs["training_result"], "experiments/")
    eval_protocol, _ = raw._reference(root, refs["full_eval_protocol"], "experiments/")
    evaluated, _ = raw._reference(root, refs["full_eval_result"], "experiments/")
    if (training.get("format") != "haic-tdmpc2-long-train-v1"
            or training.get("cells") != CELLS or training.get("environment") != ENVIRONMENT
            or eval_protocol.get("source", {}).get("protocol") != refs["training_protocol"]
            or eval_protocol.get("cells") != CELLS or eval_protocol.get("environment") != ENVIRONMENT
            or eval_protocol.get("modes") != list(MODES) or eval_protocol.get("repeats") != 2
            or eval_protocol.get("seed") != 20260928 or eval_protocol.get("max_steps") != 2000
            or eval_protocol.get("evaluation_source_sha256") != RAW_EVALUATOR_SHA
            or result.get("training_protocol_sha256") != refs["training_protocol"]["sha256"]
            or result.get("run_status") != "completed_boundary_at_least_100k"
            or result.get("checkpoint_target") != 100000
            or evaluated.get("status") != "complete_valid_full_episode_finish_comparison"
            or evaluated.get("evaluation_protocol_sha256") != refs["full_eval_protocol"]["sha256"]
            or evaluated.get("evaluation_operator_sha256") != RAW_EVALUATOR_SHA
            or evaluated.get("source_model_sha256") != result.get("checkpoint_sha256")
            or evaluated.get("source_training_result_sha256") != result.get("run_result_sha256")
            or evaluated.get("denominators") != {
                "distinct_training_roads": 4, "repeats_per_road_per_mode": 2,
                "prior_episodes": 8, "mppi_episodes": 8, "total_episodes": 16,
                "max_decisions_per_episode": 2000}
            or evaluated.get("validity", {}).get("all_scheduled_episodes_complete") is not True
            or evaluated["validity"].get("full_episode_finish_comparison_valid") is not True
            or evaluated["validity"].get("censored_episodes") != 0
            or evaluated["validity"].get("generalization_claim") is not False
            or any(evaluated.get(mode, {}).get("finishes") != 0
                   or evaluated[mode].get("episodes") != 8 for mode in MODES)):
        raise ValueError("baseline signed 0/8 full-episode evidence differs")
    run_result, _ = raw._reference(root, {"path": result["run_result"],
                                           "sha256": result["run_result_sha256"]}, "runs/")
    primary, _ = raw._reference(root, {"path": evaluated["primary_result"],
                                        "sha256": evaluated["primary_result_sha256"]}, "runs/")
    ledger = raw._path(root, evaluated["episode_ledger"])
    if (raw._digest(ledger) != raw._sha(evaluated["episode_ledger_sha256"])
            or run_result.get("status") != "completed_boundary_at_least_100k"
            or run_result.get("checkpoints", [{}])[-1].get("sha256") != result["checkpoint_sha256"]
            or primary.get("status") != "complete" or primary.get("episodes_sha256") != evaluated["episode_ledger_sha256"]
            or primary.get("protocol_sha256") != refs["full_eval_protocol"]["sha256"]
            or primary.get("source_result_sha256") != result["run_result_sha256"]
            or primary.get("full_episode_finish_comparison_valid") is not True
            or primary.get("denominators", {}).get("completed_episodes") != 16
            or any(primary.get("per_checkpoint", {}).get("100000", {}).get("per_mode", {}).get(mode, {}).get("finishes") != 0
                   or primary["per_checkpoint"]["100000"]["per_mode"][mode].get("episodes") != 8
                   for mode in MODES)):
        raise ValueError("baseline raw evaluation receipt differs")
    for name, expected in ((result["checkpoint"], result["checkpoint_sha256"]),
                           (f"{result['run']}/training.jsonl", result["training_ledger_sha256"]),
                           (f"{result['run']}/steps.jsonl", result["step_ledger_sha256"])):
        if raw._digest(raw._path(root, name)) != raw._sha(expected):
            raise ValueError(f"baseline source artifact hash mismatch: {name}")
    return training


def _source(root: Path, p: dict, baseline: dict) -> tuple[dict, dict]:
    source = p["source"]
    if not isinstance(source, dict) or set(source) != {"protocol", "result", "checkpoints"}:
        raise ValueError("source must pin separate damage protocol, complete result and final checkpoint")
    protocol, _ = raw._reference(root, source["protocol"], "experiments/tdmpc2-damage-")
    result, path = raw._reference(root, source["result"], "runs/tdmpc2-damage-")
    run = protocol.get("run_dir")
    if (protocol.get("format") != TRAIN_FORMAT or protocol.get("purpose") != "consumed-TRAIN-development"
            or protocol.get("selection") != {"arm": "independent_3d", "action_dim": 3}
            or protocol.get("cells") != CELLS or protocol.get("episode_schedule") != [0, 1, 2, 3]
            or protocol.get("environment") != ENVIRONMENT or protocol.get("checkpoint_targets") != list(TARGETS)
            or protocol.get("upstream_revision") != baseline.get("upstream_revision")
            or protocol.get("seed_schedule") != baseline.get("seed_schedule")
            or not _exact(protocol.get("training"), baseline.get("training"))
            or not _exact(protocol.get("resources"), baseline.get("resources"))
            or not _exact(protocol.get("reward_target"), REWARD_TARGET)
            or any(protocol.get(key) != p["baseline"][name] for key, name in (
                ("baseline_training_protocol", "training_protocol"),
                ("baseline_training_result", "training_result"),
                ("baseline_full_eval_protocol", "full_eval_protocol"),
                ("baseline_full_eval_result", "full_eval_result")))
            or not isinstance(run, str) or not run.startswith("runs/tdmpc2-damage-")
            or path != raw._path(root, f"{run}/result.json")):
        raise ValueError("source is not the separately pinned H3 damage-only complete TRAIN run")
    sources = protocol.get("source_sha256")
    baseline_sources = baseline.get("source_sha256")
    if (not isinstance(sources, dict) or not isinstance(baseline_sources, dict)
            or set(sources) != TRAIN_SOURCE_PATHS or set(baseline_sources) != raw.SOURCE_PATHS
            or any(sources[name] != sha for name, sha in baseline_sources.items())):
        raise ValueError("damage TRAIN source map cannot be proved against raw baseline")
    for name, expected in sources.items():
        if raw._digest(raw._path(root, name)) != raw._sha(expected):
            raise ValueError(f"damage TRAIN source hash mismatch: {name}")
    checkpoints = result.get("checkpoints")
    if (result.get("status") != "completed_boundary_at_least_100k"
            or result.get("protocol_sha256") != source["protocol"]["sha256"]
            or result.get("source_sha256") != sources or not _exact(result.get("reward_target"), REWARD_TARGET)
            or result.get("seed_schedule") != protocol["seed_schedule"]
            or result.get("pretrain_updates") != protocol["training"]["pretrain_updates"]
            or result.get("action_dim") != 3 or result.get("reused_train_only") is not True
            or result.get("resume_supported") is not False or result.get("evaluation") is not None
            or type(result.get("decisions")) is not int
            or not TARGETS[-1] <= result["decisions"] <= protocol["training"]["decision_cap"]
            or result.get("updates") != result["decisions"]
            or type(result.get("episodes")) is not int or result["episodes"] < 1
            or not isinstance(checkpoints, list) or len(checkpoints) != len(TARGETS)
            or any(not isinstance(checkpoint, dict) for checkpoint in checkpoints)
            or [c.get("target") for c in checkpoints] != list(TARGETS)
            or not isinstance(result.get("runtime"), dict) or result["runtime"].get("device") != "cuda"):
        raise ValueError("missing complete shaped-target result and four boundary checkpoints")
    return protocol, result


def _finite(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _producer_repr(text: object, value: object) -> bool:
    if not isinstance(text, str) or len(text) > 128 or not isinstance(value, (int, float)) or not _finite(value):
        return False
    try:
        number = float(value)
        return (math.isfinite(float(text)) and float(text) == number
                and (text == repr(number) or (number.is_integer() and text == repr(int(number)))))
    except ValueError:
        return False


def _ledger(root: Path, protocol: dict, result: dict, source: dict) -> dict:
    """Bind every raw/shaped step, complete episode, checkpoint and SHA cursor."""
    run = protocol["run_dir"]
    training = raw._path(root, f"{run}/training.jsonl")
    steps = raw._path(root, f"{run}/steps.jsonl")
    if (raw._digest(training) != raw._sha(result.get("training_ledger_sha256"))
            or raw._digest(steps) != raw._sha(result.get("step_ledger_sha256"))):
        raise ValueError("complete damage training/step ledger SHA mismatch")
    selected = source["checkpoints"]
    if (not isinstance(selected, list) or len(selected) != 1 or not isinstance(selected[0], dict)
            or set(selected[0]) != {"target", "path", "sha256", "training_cursor_sha256", "step_cursor_sha256"}
            or selected[0]["target"] != TARGETS[-1]):
        raise ValueError("select only the first completed >=100k damage checkpoint")
    step_digest, train_digest = hashlib.sha256(), hashlib.sha256()
    decisions = updates = checkpoint_index = 0
    episodes: list[dict] = []
    phase = "start"
    pin = None
    with training.open("rb") as journal, steps.open("rb") as step_stream:
        for line_number, line in enumerate(journal, 1):
            if not line.endswith(b"\n"):
                raise ValueError("torn damage training ledger")
            train_digest.update(line)
            row = raw._json(line)
            event = row.get("event")
            if phase == "start":
                if (event != "start" or row.get("protocol_sha256") != source["protocol"]["sha256"]
                        or row.get("source_sha256") != protocol["source_sha256"]
                        or not _exact(row.get("reward_target"), REWARD_TARGET)
                        or row.get("runtime") != result.get("runtime")
                        or row.get("seed_schedule") != protocol["seed_schedule"]
                        or row.get("resume_supported") is not False):
                    raise ValueError("damage training start lacks bound source and shaped target")
                phase = "intent"
            elif phase == "checkpoint":
                if (event != "checkpoint" or checkpoint_index >= len(TARGETS)
                        or row != {"event": "checkpoint", **result["checkpoints"][checkpoint_index]}
                        or row.get("target") != TARGETS[checkpoint_index]
                        or any(row.get(k) != v for k, v in (
                            ("decisions", decisions), ("updates", updates), ("episodes", len(episodes))))
                        or row.get("reward_prediction_target") != "training_reward"
                        or not isinstance(row.get("frozen_train_probe"), dict)
                        or "training_reward_mae" not in row["frozen_train_probe"]
                        or any("raw_return" in k or k == "reward_mae" for k in row["frozen_train_probe"])
                        or row.get("step_ledger_sha256") != step_digest.hexdigest()
                        or row.get("train_episodes_since_previous") != [
                            {k: ep[k] for k in ("episode", "track_id", "geometry_seed", "decisions",
                                                  "length", "return", "training_return", "progress", "damage", "finished")}
                            for ep in episodes[0 if checkpoint_index == 0 else
                                               result["checkpoints"][checkpoint_index - 1]["episodes"]:]]):
                    raise ValueError("damage checkpoint lacks a consistent whole-episode cursor")
                name = f"checkpoint-at-least-{row['target']:06d}-step-{decisions:06d}.pt"
                if row.get("path") != name or raw._sha(row.get("sha256")) != row["sha256"]:
                    raise ValueError("damage checkpoint name/SHA differs from boundary")
                if checkpoint_index == len(TARGETS) - 1:
                    path = raw._path(root, f"{run}/{name}")
                    expected = {"target": row["target"], "path": f"{run}/{name}", "sha256": row["sha256"],
                                "training_cursor_sha256": train_digest.hexdigest(),
                                "step_cursor_sha256": step_digest.hexdigest()}
                    if selected[0] != expected or raw._digest(path) != row["sha256"]:
                        raise ValueError("final damage checkpoint SHA or complete ledger cursor mismatch")
                    pin = {**expected, "line": line_number, "decisions": decisions,
                           "updates": updates, "episodes": len(episodes)}
                checkpoint_index += 1
                phase = "intent"
            elif event in ("reset_intent", "reset"):
                if checkpoint_index == len(TARGETS):
                    raise ValueError("TRAIN interaction after final damage checkpoint")
                if phase not in ("intent", "reset"):
                    raise ValueError("duplicate or skipped damage TRAIN reset")
                expected = "reset_intent" if phase == "intent" else "reset"
                cell = CELLS[len(episodes) % len(CELLS)]
                if row != {"event": expected, "episode": len(episodes), "decisions": decisions, **cell}:
                    raise ValueError("damage TRAIN reset cursor differs from frozen road order")
                phase = "reset" if phase == "intent" else "episode"
            elif event == "episode" and phase == "episode":
                length = row.get("length")
                if (type(length) is not int or not 1 <= length <= 2000
                        or row.get("episode") != len(episodes)
                        or any(row.get(k) != v for k, v in CELLS[len(episodes) % len(CELLS)].items())
                        or row.get("decisions") != decisions + length
                        or row.get("updates") != (0 if decisions + length < protocol["training"]["seed_steps"]
                            else protocol["training"]["pretrain_updates"] + decisions + length
                            - protocol["training"]["seed_steps"])
                        or any(type(row.get(k)) is not bool for k in ("finished", "terminated", "truncated", "terminal"))
                        or not (row["terminated"] or row["truncated"])
                        or row["terminal"] != (row["terminated"] or row["finished"])
                        or (row["finished"] and not row["truncated"])):
                    raise ValueError("damage TRAIN episode/decision/terminal cursor differs")
                actions, native = hashlib.sha256(), hashlib.sha256()
                raw_total = training_total = previous_damage = 0.0
                step = None
                for offset in range(length):
                    step_line = step_stream.readline()
                    if not step_line.endswith(b"\n"):
                        raise ValueError("torn or missing raw/shaped step cursor")
                    step_digest.update(step_line)
                    step = raw._json(step_line)
                    if (step.get("decision") != decisions + offset + 1 or step.get("episode") != row["episode"]
                            or any(step.get(k) != row[k] for k in ("track_id", "geometry_seed"))
                            or any(type(step.get(k)) is not bool for k in ("finished", "terminated", "truncated", "terminal"))
                            or (offset < length - 1 and (step["finished"] or step["terminated"] or step["truncated"]))
                            or step["terminal"] != (step["terminated"] or step["finished"])
                            or any(not _finite(step.get(k)) for k in (
                                "reward", "training_reward", "damage_delta", "damage", "progress"))
                            or not _producer_repr(step.get("raw_reward_repr"), step.get("reward"))
                            or not _producer_repr(step.get("damage_repr"), step.get("damage"))
                            or not 0 <= step["damage"] <= 1
                            or step["damage"] < previous_damage - REWARD_TARGET["negative_delta_tolerance"]):
                        raise ValueError("raw/shaped step, damage or episode cursor differs")
                    delta = max(0.0, step["damage"] - previous_damage)
                    if (not math.isclose(step["damage_delta"], delta, rel_tol=0, abs_tol=1e-6)
                            or not math.isclose(step["training_reward"], step["reward"] - 5.0 * delta,
                                                rel_tol=0, abs_tol=1e-6)):
                        raise ValueError("training reward is not the pinned raw-minus-damage target")
                    previous_damage = max(previous_damage, step["damage"])
                    for key, digest in (("action_f32_hex", actions), ("native_action_f32_hex", native)):
                        try:
                            action = bytes.fromhex(step[key])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError("invalid damage TRAIN step action") from exc
                        if len(action) != 12:
                            raise ValueError("damage TRAIN step action is not 3D float32")
                        digest.update(action)
                    raw_total += step["reward"]
                    training_total += step["training_reward"]
                if (step is None or any(step[k] != row[k] for k in ("finished", "terminated", "truncated", "terminal"))
                        or not _finite(row.get("return")) or not _finite(row.get("training_return"))
                        or not _finite(row.get("progress")) or not _finite(row.get("damage"))
                        or not math.isclose(raw_total, row["return"], rel_tol=0, abs_tol=1e-4)
                        or not math.isclose(training_total, row["training_return"], rel_tol=0, abs_tol=1e-4)
                        or not math.isclose(row["training_return"], row["return"] - 5.0 * row["damage"],
                                            rel_tol=0, abs_tol=1e-4)
                        or row["damage"] != step["damage"] or row["progress"] != step["progress"]
                        or actions.hexdigest() != row.get("action_trace_sha256")
                        or native.hexdigest() != row.get("native_action_trace_sha256")):
                    raise ValueError("damage episode raw/shaped returns differ from complete steps")
                decisions += length
                updates = row["updates"]
                episodes.append(row)
                phase = "checkpoint" if checkpoint_index < len(TARGETS) and decisions >= TARGETS[checkpoint_index] else "intent"
            else:
                raise ValueError(f"partial or unexpected damage training event: {event}")
        if (phase != "intent" or checkpoint_index != len(TARGETS) or pin is None
                or step_stream.read(1) or decisions != result["decisions"] or updates != result["updates"]
                or pin["decisions"] != decisions or pin["episodes"] != len(episodes)
                or len(episodes) != result["episodes"]
                or train_digest.hexdigest() != result["training_ledger_sha256"]
                or step_digest.hexdigest() != result["step_ledger_sha256"]):
            raise ValueError("damage run is partial or lacks the final complete cursor")
    return pin


def _check(protocol_path: Path, sha: str, *, root: Path, output_reserved: bool = False) -> tuple[dict, dict]:
    path = raw._evaluation_path(root, protocol_path)
    if path.parent != root / "experiments" or raw._digest(path) != raw._sha(sha):
        raise ValueError("separate frozen damage evaluation protocol SHA mismatch")
    p = raw._json(path.read_bytes())
    if (set(p) != {"format", "purpose", "evaluation_source_sha256", "baseline", "source",
                   "cells", "environment", "modes", "repeats", "seed", "max_steps", "output_dir"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["cells"] != CELLS or p["environment"] != ENVIRONMENT
            or p["modes"] != list(MODES) or type(p["repeats"]) is not int or p["repeats"] != 2
            or type(p["seed"]) is not int or p["seed"] != 20260928
            or type(p["max_steps"]) is not int or p["max_steps"] != 2000):
        raise ValueError("only sixteen raw full-episode reused-TRAIN comparisons are allowed")
    if raw._digest(raw._path(root, "scripts/evaluate_tdmpc2_damage_train.py")) != raw._sha(p["evaluation_source_sha256"]):
        raise ValueError("damage evaluation executable SHA mismatch")
    out = raw._path(root, p["output_dir"], existing=False)
    if (out.parent != root / "runs" or not out.name.startswith("tdmpc2-damage-full-train-")
            or not out.parent.is_dir() or (out.exists() if not output_reserved else not out.is_dir())):
        raise ValueError("output requires a unique runs/tdmpc2-damage-full-train-* receipt")
    baseline = _baseline(root, p["baseline"])
    protocol, result = _source(root, p, baseline)
    pin = _ledger(root, protocol, result, p["source"])
    if torch.version.cuda is not None or not str(torch.__version__).startswith("2.1.0+cpu"):
        raise ValueError("damage evaluation requires CPU-only Torch 2.1.0")
    return p, {"status": "preflight_only", "environment_resets": 0, "reused_train_only": True,
               "generalization_claim": False, "official_score": False, "evaluation_reward": "raw_environment_only",
               "source_reward_target": REWARD_TARGET, "protocol_sha256": sha,
               "source_protocol_sha256": p["source"]["protocol"]["sha256"],
               "source_result_sha256": p["source"]["result"]["sha256"],
               "baseline_full_eval_result_sha256": BASELINE["full_eval_result"]["sha256"],
               "source_sha256": protocol["source_sha256"], "output_dir": p["output_dir"],
               "cells": CELLS, "checkpoints": [pin], "modes": list(MODES), "repeats": 2,
               "max_steps": 2000, "planned_episodes": 16}


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """Read-only checks before torch.load, environment import, construction or reset."""
    return _check(protocol_path, protocol_sha256, root=root.resolve(strict=True))[1]


def _verify_replay(state: dict, pin: dict, protocol: dict, result: dict, root: Path) -> None:
    """Bind all untrimmed replay transitions and the seed-frozen probe to shaped TRAIN steps."""
    cfg = protocol["training"]
    replay, probe = state["replay"], state["probe"]
    if (set(replay) != {"format", "capacity", "horizon", "action_dim", "include_partial",
                        "bootstrap_on_truncation", "augmentation_pad", "observation_shape",
                        "next_episode_id", "size", "episodes", "active", "rng_state"}
            or replay["format"] != "haic-tdmpc2-episode-replay-v1"
            or any(type(replay.get(k)) is not int or replay[k] != cfg[v] for k, v in (
                ("capacity", "replay_capacity"), ("horizon", "horizon"),
                ("action_dim", "action_dim"), ("augmentation_pad", "augmentation_pad")))
            or replay["include_partial"] is not False or replay["bootstrap_on_truncation"] is not True
            or replay["observation_shape"] != tuple(cfg["observation_shape"])
            or cfg["replay_capacity"] < pin["decisions"] or replay["size"] != pin["decisions"]
            or replay["next_episode_id"] != pin["episodes"] or replay["active"] is not None
            or not isinstance(replay["episodes"], list) or len(replay["episodes"]) != pin["episodes"]
            or not isinstance(replay["rng_state"], dict)):
        raise ValueError("checkpoint replay cannot prove complete untrimmed shaped TRAIN history")
    run = protocol["run_dir"]
    training = raw._path(root, f"{run}/training.jsonl")
    steps = raw._path(root, f"{run}/steps.jsonl")
    if (raw._digest(training) != result["training_ledger_sha256"]
            or raw._digest(steps) != result["step_ledger_sha256"]):
        raise ValueError("TRAIN ledger changed before checkpoint replay validation")
    with training.open("rb") as stream:
        episode_rows = [row for line in stream if (row := raw._json(line)).get("event") == "episode"]
    if len(episode_rows) != pin["episodes"]:
        raise ValueError("checkpoint replay episode count differs from TRAIN ledger")
    saved: dict[int, tuple[dict, dict]] = {}
    decisions = 0
    shaped_steps = 0
    with steps.open("rb") as stream:
        for index, (episode, row) in enumerate(zip(replay["episodes"], episode_rows)):
            length = row["length"]
            if (not isinstance(episode, dict)
                    or set(episode) != {"episode_id", "start_step", "observations", "actions", "rewards",
                                        "terminated", "truncated", "terminal"}
                    or type(episode["episode_id"]) is not int or episode["episode_id"] != index
                    or type(episode["start_step"]) is not int or episode["start_step"] != 0
                    or row["episode"] != index or row["decisions"] != decisions + length):
                raise ValueError("checkpoint replay episode is missing or trimmed")
            shapes = {"observations": (length + 1, *cfg["observation_shape"]),
                      "actions": (length, 3), "rewards": (length,),
                      "terminated": (length,), "truncated": (length,), "terminal": (length,)}
            types = {"observations": np.uint8, "actions": np.float32, "rewards": np.float32,
                     "terminated": np.bool_, "truncated": np.bool_, "terminal": np.bool_}
            if any(not isinstance(episode[k], np.ndarray) or episode[k].dtype != types[k]
                   or episode[k].shape != shape for k, shape in shapes.items()):
                raise ValueError("checkpoint replay arrays have wrong shape or dtype")
            if (not np.isfinite(episode["actions"]).all() or np.any(np.abs(episode["actions"]) > 1)
                    or not np.isfinite(episode["rewards"]).all()):
                raise ValueError("checkpoint replay contains invalid actions or shaped rewards")
            for offset in range(length):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("checkpoint replay outlasts TRAIN step cursor")
                step = raw._json(line)
                action = np.frombuffer(bytes.fromhex(step["action_f32_hex"]), dtype="<f4")
                if (step["episode"] != index or step["decision"] != decisions + offset + 1
                        or not np.array_equal(episode["actions"][offset], action)
                        or episode["rewards"][offset] != np.float32(step["training_reward"])
                        or any(bool(episode[k][offset]) != step[k] for k in ("terminated", "truncated", "terminal"))):
                    raise ValueError("checkpoint replay reward/action/terminal differs from shaped TRAIN step")
                shaped_steps += step["reward"] - step["training_reward"] > 1e-4
            if (any(bool(episode[k][-1]) != row[k] for k in ("terminated", "truncated", "terminal"))
                    or not (episode["terminated"][-1] or episode["truncated"][-1])):
                raise ValueError("checkpoint replay episode lacks the TRAIN semantic boundary")
            saved[index] = (episode, row)
            decisions += length
        if stream.read(1) or decisions != pin["decisions"] or not shaped_steps:
            raise ValueError("checkpoint replay lacks complete, distinguishable shaped TRAIN steps")
    if (raw._digest(training) != result["training_ledger_sha256"]
            or raw._digest(steps) != result["step_ledger_sha256"]):
        raise ValueError("TRAIN ledger changed during checkpoint replay validation")

    horizon, batch = cfg["horizon"], cfg["batch_size"]
    shapes = {"obs": (horizon + 1, batch, *cfg["observation_shape"]),
              "action": (horizon, batch, 3), "reward": (horizon, batch, 1),
              **{k: (horizon, batch, 1) for k in ("terminated", "truncated", "terminal", "bootstrap_mask")},
              "episode_id": (batch,), "start_step": (batch,)}
    if (set(probe) != set(shapes) or any(not isinstance(probe[k], torch.Tensor)
            or probe[k].device.type != "cpu" or probe[k].shape != shape
            or probe[k].dtype != (torch.uint8 if k == "obs" else torch.int64 if k in ("episode_id", "start_step")
                                  else torch.float32) for k, shape in shapes.items())):
        raise ValueError("checkpoint frozen TRAIN probe has wrong shape, dtype or device")
    for column in range(batch):
        episode_id, start = int(probe["episode_id"][column]), int(probe["start_step"][column])
        if episode_id not in saved:
            raise ValueError("frozen TRAIN probe references an unknown replay episode")
        episode, row = saved[episode_id]
        if (row["decisions"] >= cfg["seed_steps"] or start < 0 or start + horizon > row["length"]
                or not torch.equal(probe["obs"][:, column], torch.as_tensor(
                    episode["observations"][start:start + horizon + 1]))
                or not torch.equal(probe["action"][:, column], torch.as_tensor(
                    episode["actions"][start:start + horizon]))
                or not torch.equal(probe["reward"][:, column, 0], torch.as_tensor(
                    episode["rewards"][start:start + horizon]))):
            raise ValueError("frozen TRAIN probe is not an eligible shaped replay window")
        for key in ("terminated", "truncated", "terminal"):
            expected = torch.as_tensor(episode[key][start:start + horizon].astype(np.float32))
            if not torch.equal(probe[key][:, column, 0], expected):
                raise ValueError("frozen TRAIN probe has mismatched semantic labels")
        if not torch.equal(probe["bootstrap_mask"][:, column, 0], 1.0 - probe["terminal"][:, column, 0]):
            raise ValueError("frozen TRAIN probe has mismatched bootstrap labels")
    summary = result["checkpoints"][-1]["frozen_train_probe"]
    if (summary.get("sample_count") != batch
            or summary.get("episode_ids") != probe["episode_id"].tolist()
            or summary.get("start_steps") != probe["start_step"].tolist()
            or summary.get("distinct_episode_count") != len(set(probe["episode_id"].tolist()))
            or summary.get("terminal_positive_transitions") != int(probe["terminal"].sum().item())
            or not math.isclose(summary.get("training_reward_min", math.inf), probe["reward"].min().item(), abs_tol=1e-6)
            or not math.isclose(summary.get("training_reward_max", math.inf), probe["reward"].max().item(), abs_tol=1e-6)):
        raise ValueError("frozen TRAIN probe summary does not describe its shaped sample")


def _model(pin: dict, checked: dict, protocol: dict, result: dict, root: Path, *, model_factory=None):
    if raw._digest(Path(pin["path"])) != pin["sha256"]:
        raise ValueError("damage checkpoint changed after complete preflight")
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    if (not isinstance(state, dict) or set(state) != {
            "format", "protocol_sha256", "source_sha256", "reward_target", "target", "decisions",
            "updates", "episodes", "action_dim", "learner", "optim", "pi_optim", "replay", "probe",
            "rng", "resume_supported"}
            or state["format"] != TRAIN_FORMAT or state["protocol_sha256"] != checked["source_protocol_sha256"]
            or state["source_sha256"] != checked["source_sha256"]
            or not _exact(state["reward_target"], REWARD_TARGET) or state["resume_supported"] is not False
            or state["action_dim"] != 3 or any(state[k] != pin[k] for k in ("target", "decisions", "updates", "episodes"))
            or any(not isinstance(state[k], dict) for k in ("learner", "optim", "pi_optim", "replay", "probe", "rng"))
            or "q_scale" not in state["learner"]
            or not all(isinstance(k, str) and (k == "q_scale" or k.startswith("model."))
                       for k in state["learner"])):
        raise ValueError("checkpoint is not the bound damage-target learner")
    _verify_replay(state, pin, protocol, result, root)
    if model_factory is None:
        from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    else:
        model = model_factory()
    model.load_state_dict({k[6:]: v for k, v in state["learner"].items() if k.startswith("model.")}, strict=True)
    return model.to("cpu").eval()


def _summary(rows: list[dict], pin: dict) -> dict:
    schedule = [(repeat, cell, mode) for repeat in range(2) for cell in CELLS for mode in MODES]
    if (len(rows) != len(schedule) or any(
            row.get("target") != pin["target"] or row.get("mode") != mode or row.get("repeat") != repeat
            or any(row.get(k) != value for k, value in cell.items())
            or row.get("event") != "episode" or "training_return" in row or "training_reward" in row
            for row, (repeat, cell, mode) in zip(rows, schedule))):
        raise ValueError("incomplete, duplicate or non-raw damage evaluation denominator")
    return raw._summary(rows, [pin], 2)


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT,
            env_factory=None, model_factory=None, planner_factory=None) -> dict:
    """One exclusive raw-metric receipt; recheck immutable sources before every reset."""
    root = root.resolve(strict=True)
    p, checked = _check(protocol_path, protocol_sha256, root=root)
    if _check(protocol_path, protocol_sha256, root=root)[1] != checked:
        raise ValueError("damage source changed before evaluation began")
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
    if env_factory is None:
        from haic.algorithms.tdmpc2.haic_env import make_training_env
        env_factory = make_training_env
    if planner_factory is None:
        planner_factory = TDMPC2Planner
    output = raw._path(root, p["output_dir"], existing=False)
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    rows: list[dict] = []
    env = None
    reset_intents = 0
    try:
        pin = checked["checkpoints"][0]
        source_protocol, _ = raw._reference(root, p["source"]["protocol"], "experiments/tdmpc2-damage-")
        source_result, _ = raw._reference(root, p["source"]["result"], "runs/tdmpc2-damage-")
        model = _model({**pin, "path": str(raw._path(root, pin["path"]))}, checked,
                       source_protocol, source_result, root, model_factory=model_factory)
        planner = planner_factory(model, PlannerConfig(action_dim=3, discount=.995, episodic=True))
        for repeat in range(2):
            for index, cell in enumerate(CELLS):
                seed = 20260928 + repeat * len(CELLS) + index
                for mode in MODES:
                    # Source and full raw/shaped TRAIN cursors are checked immediately before each reset.
                    if _check(protocol_path, protocol_sha256, root=root, output_reserved=True)[1] != checked:
                        raise ValueError("damage source or evidence changed before reset")
                    reset_intents += 1
                    raw._journal(ledger, {"event": "reset_intent", "target": pin["target"], "mode": mode,
                                          "repeat": repeat, "episode_seed": seed, **cell})
                    if env is None:
                        env = env_factory(2000)
                    row = raw._episode(env, model, planner, cell, mode=mode, repeat=repeat, seed=seed,
                                       checkpoint_target=pin["target"], max_steps=2000)
                    raw._journal(ledger, row)
                    rows.append(row)
        by_checkpoint = _summary(rows, pin)
        mppi = by_checkpoint[str(pin["target"])]["per_mode"]["mppi"]
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "reused_train_only": True, "generalization_claim": False, "official_score": False,
                  "checkpoint_replay_probe_verified": True,
                  "optimizer_history_proven": False,
                  "full_episode_finish_comparison_valid": True,
                  "pairing": "same consumed TRAIN road/reset RNG seed; separate resets and different trajectories",
                  "evaluation_reward": "raw_environment_only", "source_reward_target": REWARD_TARGET,
                  "training_reward_is_evaluation_metric": False, "protocol_sha256": protocol_sha256,
                  "source_protocol_sha256": checked["source_protocol_sha256"],
                  "source_result_sha256": checked["source_result_sha256"],
                  "baseline_full_eval_result_sha256": checked["baseline_full_eval_result_sha256"],
                  "checkpoints": checked["checkpoints"], "cpu_only_torch": str(torch.__version__),
                  "denominators": {"checkpoints": 1, "distinct_roads": 4,
                                   "repeats_per_road_mode_checkpoint": 2, "modes": list(MODES),
                                   "planned_episodes": 16, "completed_episodes": len(rows)},
                  "episodes_sha256": raw._digest(ledger), "per_checkpoint": by_checkpoint,
                  "primary_descriptive_gate": {
                      "mode": "mppi", "baseline_finishes": 0, "baseline_episodes": 8,
                      "threshold_finishes": 4, "episodes": mppi["episodes"], "finishes": mppi["finishes"],
                      "censored": mppi["censored"], "met_on_reused_train": mppi["finishes"] >= 4,
                      "fresh_generalization_claim": False}}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return report
    except BaseException as exc:
        try:
            raw._journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                                  "complete_episodes": len(rows), "reset_intents": reset_intents,
                                  "resume_supported": False})
        except BaseException:
            try:
                receipt = {"event": "failure", "reason": type(exc).__name__,
                           "protocol_sha256": protocol_sha256, "source_result_sha256": checked["source_result_sha256"],
                           "episode_ledger_sha256": raw._digest(ledger) if ledger.is_file() else None,
                           "complete_episodes": len(rows), "reset_intents": reset_intents,
                           "environment_resets": None if reset_intents else 0, "resume_supported": False}
                with (output / "failure.json").open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except BaseException as preserve_error:
                raise RuntimeError("cannot preserve evaluation receipt; exposure is unknown") from preserve_error
        raise
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path, help="separately frozen experiments/*.json")
    parser.add_argument("--protocol-sha256", required=True, help="external SHA-256 of the evaluation protocol")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: file-only checks, zero resets")
    mode.add_argument("--execute", action="store_true", help="CPU full-episode reused-TRAIN interaction")
    args = parser.parse_args()
    report = (execute(args.protocol, args.protocol_sha256) if args.execute else
              preflight(args.protocol, args.protocol_sha256))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
