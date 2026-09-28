"""Full-episode CPU development evaluation on four *consumed* TD-MPC2 TRAIN roads.

Run from the repository root with ``python -m scripts.evaluate_tdmpc2_full_train``.
The default --preflight reads files only. --execute requires a separately frozen
experiments/*.json protocol and an externally supplied SHA-256; it writes a new,
exclusive runs/ receipt. No fresh TRAIN, TRAIN-DIAGNOSTIC, confirmation, blind,
official, or generalization evaluation is supported. Never load untrusted .pt
files: torch.load uses pickle, albeit only AFTER all file/cursor SHA checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-full-consumed-train-v1"
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
CELLS = [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
ENVIRONMENT = {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}
MODES = ("prior", "mppi")
TARGETS = (20000, 40000, 70000, 100000)
R6_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
PRIOR_SHA = "2e4a837795abad44623653f789d966cdeed3bb93b154bb2b17cbff18d1c35386"
SOURCE_PATHS = frozenset({
    "scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/__init__.py",
    "haic/algorithms/tdmpc2/action_2d.py", "haic/algorithms/tdmpc2/haic_env.py",
    "haic/algorithms/tdmpc2/model.py", "haic/algorithms/tdmpc2/replay.py",
    "haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/planner.py",
    "env_wrapper.py", "damage.py", "core/__init__.py", "core/vendor/__init__.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py",
    "core/track_variables.py", "core/obstacle_contacts.py", "core/finish_line.py",
})
TRAIN_SETTINGS = {
    "action_dim": 3, "observation_shape": [4, 64, 64], "model_size": 5,
    "num_bins": 101, "episodic": True, "horizon": 3, "discount": .995,
    "rho": .5, "augmentation_pad": 3, "batch_size": 256, "max_steps": 2000,
    "seed_steps": 10000, "pretrain_updates": 10000,
    "updates_per_post_seed_decision": 1,
}


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256")
    return value


def _path(root: Path, name: str, *, existing: bool = True) -> Path:
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or Path(name).as_posix() != name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"noncanonical relative path: {name}")
    path = root
    for part in name.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in artifact path: {name}")
    if existing and not path.is_file():
        raise ValueError(f"missing artifact: {name}")
    return path


def _evaluation_path(root: Path, value: Path) -> Path:
    if value.is_absolute():
        try:
            value = value.relative_to(root)
        except ValueError as exc:
            raise ValueError("evaluation protocol must be inside the repository") from exc
    return _path(root, value.as_posix())


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(data: bytes) -> dict:
    if not 0 < len(data) <= 1024 * 1024:
        raise ValueError("invalid JSON record length")
    value = json.loads(data, object_pairs_hook=_unique,
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    if not isinstance(value, dict):
        raise ValueError("JSON record must be an object")
    return value


def _reference(root: Path, ref: dict, prefix: str) -> tuple[dict, Path]:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
            or not isinstance(ref["path"], str) or not ref["path"].startswith(prefix)):
        raise ValueError(f"invalid {prefix} reference")
    path = _path(root, ref["path"])
    if _digest(path) != _sha(ref["sha256"]):
        raise ValueError(f"SHA mismatch: {ref['path']}")
    return _json(path.read_bytes()), path


def _source(root: Path, evaluation: dict) -> tuple[dict, dict, Path]:
    source = evaluation["source"]
    if not isinstance(source, dict) or set(source) != {"protocol", "result", "checkpoints"}:
        raise ValueError("source must pin protocol, completed result and checkpoints")
    protocol, _ = _reference(root, source["protocol"], "experiments/tdmpc2-long-")
    result, result_path = _reference(root, source["result"], "runs/tdmpc2-long-")
    run = protocol.get("run_dir")
    cfg = protocol.get("training")
    if not isinstance(cfg, dict):
        raise ValueError("missing frozen 3D long-run training settings")
    if (protocol.get("format") not in ("haic-tdmpc2-long-train-v1", "haic-tdmpc2-long-train-v2")
            or protocol.get("purpose") != "consumed-TRAIN-development"
            or protocol.get("selection") != {"arm": "independent_3d", "action_dim": 3}
            or protocol.get("cells") != CELLS or protocol.get("episode_schedule") != [0, 1, 2, 3]
            or protocol.get("environment") != ENVIRONMENT
            or not isinstance(run, str) or not run.startswith("runs/tdmpc2-long-")
            or result_path != _path(root, f"{run}/result.json")
            or any(cfg.get(k) != v for k, v in TRAIN_SETTINGS.items())):
        raise ValueError("source is not the frozen 3D H3 consumed-TRAIN long run")
    targets = protocol.get("checkpoint_targets")
    if not isinstance(targets, list) or targets != list(TARGETS):
        raise ValueError("invalid source milestone schedule")
    if type(cfg.get("decision_cap")) is not int or targets[-1] > cfg["decision_cap"]:
        raise ValueError("invalid source milestone schedule")
    if (protocol.get("r6_protocol") != {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": R6_SHA}
            or protocol.get("prior_consumption") != {
                "path": "experiments/dreamerv3-reused-train-diagnostic-v1.json", "sha256": PRIOR_SHA}):
        raise ValueError("source lacks pinned consumed-TRAIN lineage")
    r6, _ = _reference(root, protocol["r6_protocol"], "experiments/")
    prior, _ = _reference(root, protocol["prior_consumption"], "experiments/")
    if (r6.get("training_pool", {}).get("partition") != "TRAIN"
            or not set(ROADS) <= set(r6["training_pool"].get("geometry_seeds", []))
            or 1 not in r6["training_pool"].get("track_ids", [])
            or set(ROADS) & set(r6.get("diagnostic_pool", {}).get("geometry_seeds", []))
            or r6.get("environment", {}).get("partition") != "TRAIN"
            or r6["environment"].get("obstacles") is not True
            or r6["environment"].get("max_steps") != 2000
            or r6["environment"].get("frame_skip") != 4
            or prior.get("cells") != CELLS or prior.get("max_steps") != 2000
            or prior.get("frame_skip") != 4
            or prior.get("purpose") != "reused-TRAIN-engineering-diagnostic"):
        raise ValueError("four roads are not proven previously consumed TRAIN cells")
    sources = protocol.get("source_sha256")
    required = SOURCE_PATHS - {"scripts/train_tdmpc2_long.py"}
    if not isinstance(sources, dict):
        raise ValueError("incomplete training/runtime/environment source map")
    runners = set(sources) - required
    if (len(runners) != 1 or set(sources) != required | runners
            or not all(name.startswith("scripts/train_tdmpc2_long") and name.endswith(".py")
                       for name in runners)):
        raise ValueError("incomplete training/runtime/environment source map")
    for name, expected in sources.items():
        if _digest(_path(root, name)) != _sha(expected):
            raise ValueError(f"source hash mismatch: {name}")
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(targets):
        raise ValueError("source run has no complete checkpoint list")
    if (result.get("status") != "completed_boundary_at_least_100k"
            or result.get("protocol_sha256") != source["protocol"]["sha256"]
            or result.get("source_sha256") != sources or result.get("action_dim") != 3
            or result.get("reused_train_only") is not True or result.get("resume_supported") is not False
            or result.get("evaluation") is not None or result.get("seed_schedule") != protocol.get("seed_schedule")
            or result.get("pretrain_updates") != cfg["pretrain_updates"]
            or type(result.get("decisions")) is not int or result["decisions"] < targets[-1]
            or type(result.get("episodes")) is not int or result["episodes"] < 1
            or type(result.get("updates")) is not int or result["updates"] < cfg["pretrain_updates"]):
        raise ValueError("source run has no complete bound training result")
    return protocol, result, _path(root, run, existing=False)


def _ledger(root: Path, protocol: dict, result: dict, run: Path, source: dict) -> list[dict]:
    """Verify EVERY training/step row and the checkpoint-time step prefix cursors."""
    training = _path(root, f"{protocol['run_dir']}/training.jsonl")
    steps = _path(root, f"{protocol['run_dir']}/steps.jsonl")
    if (_digest(training) != _sha(result.get("training_ledger_sha256"))
            or _digest(steps) != _sha(result.get("step_ledger_sha256"))):
        raise ValueError("complete training/step ledger SHA mismatch")
    step_digest = hashlib.sha256()
    train_digest = hashlib.sha256()
    decisions = updates = checkpoint_index = 0
    episodes: list[dict] = []
    selected = source["checkpoints"]
    targets = protocol["checkpoint_targets"]
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(pin, dict) or set(pin) != {
                "target", "path", "sha256", "training_cursor_sha256", "step_cursor_sha256"}
                for pin in selected)
            or any(type(pin["target"]) is not int for pin in selected)
            or [pin["target"] for pin in selected] != sorted(set(pin["target"] for pin in selected))
            or any(pin["target"] not in targets for pin in selected)):
        raise ValueError("select distinct ordered frozen checkpoint targets")
    pins = []
    phase = "start"
    with training.open("rb") as journal, steps.open("rb") as step_stream:
        for line_number, line in enumerate(journal, 1):
            if not line.endswith(b"\n"):
                raise ValueError("incomplete training ledger row")
            train_digest.update(line)
            row = _json(line)
            event = row.get("event")
            if phase == "start":
                if (event != "start" or row.get("protocol_sha256") != source["protocol"]["sha256"]
                        or row.get("source_sha256") != protocol["source_sha256"]
                        or row.get("resume_supported") is not False
                        or row.get("seed_schedule") != protocol["seed_schedule"]):
                    raise ValueError("training start does not bind protocol/source")
                phase = "intent"
            elif phase == "checkpoint":
                if (event != "checkpoint" or checkpoint_index >= len(targets)
                        or row.get("target") != targets[checkpoint_index]
                        or any(row.get(k) != v for k, v in {
                            "decisions": decisions, "updates": updates, "episodes": len(episodes)}.items())
                        or row != {"event": "checkpoint", **result["checkpoints"][checkpoint_index]}
                        or row.get("step_ledger_sha256") != step_digest.hexdigest()
                        or row.get("train_episodes_since_previous") != [
                            {key: ep[key] for key in ("episode", "track_id", "geometry_seed",
                                                      "decisions", "length", "return", "progress", "damage", "finished")}
                            for ep in episodes[0 if checkpoint_index == 0 else result["checkpoints"][checkpoint_index - 1]["episodes"]:]]):
                    raise ValueError("missing or inconsistent whole-episode checkpoint cursor")
                name = f"checkpoint-at-least-{row['target']:06d}-step-{decisions:06d}.pt"
                if row.get("path") != name or _sha(row.get("sha256")) != row["sha256"]:
                    raise ValueError("checkpoint path/SHA differs from frozen boundary")
                for pin in selected:
                    if pin["target"] == row["target"]:
                        path = _path(root, f"{protocol['run_dir']}/{name}")
                        if (pin != {"target": row["target"], "path": f"{protocol['run_dir']}/{name}",
                                    "sha256": row["sha256"],
                                    "training_cursor_sha256": train_digest.hexdigest(),
                                    "step_cursor_sha256": step_digest.hexdigest()}
                                or _digest(path) != row["sha256"]):
                            raise ValueError("checkpoint SHA or complete ledger cursor mismatch")
                        pins.append({**pin, "line": line_number,
                                     "decisions": decisions, "updates": updates, "episodes": len(episodes)})
                checkpoint_index += 1
                phase = "intent"
            elif event in ("reset_intent", "reset"):
                expected = "reset_intent" if phase == "intent" else "reset"
                cell = CELLS[len(episodes) % len(CELLS)]
                if event != expected or row != {"event": event, "episode": len(episodes),
                                                 "decisions": decisions, **cell}:
                    raise ValueError("TRAIN reset cursor/cell differs from frozen schedule")
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
                        or row["finished"] and not row["truncated"]):
                    raise ValueError("source TRAIN episode/decision/terminal cursor differs")
                reward_sum = 0.0
                actions, native = hashlib.sha256(), hashlib.sha256()
                step = None
                for offset in range(length):
                    step_line = step_stream.readline()
                    if not step_line.endswith(b"\n"):
                        raise ValueError("incomplete training step cursor")
                    step_digest.update(step_line)
                    step = _json(step_line)
                    if (step.get("decision") != decisions + offset + 1
                            or step.get("episode") != row["episode"]
                            or any(step.get(k) != row[k] for k in ("track_id", "geometry_seed"))
                            or any(type(step.get(k)) is not bool for k in ("terminated", "truncated", "terminal"))
                            or step["terminal"] != (step["terminated"] or (offset == length - 1 and row["finished"]))
                            or (offset < length - 1 and (step["terminated"] or step["truncated"]))
                            or type(step.get("reward")) not in (float, int) or not math.isfinite(step["reward"])):
                        raise ValueError("step/episode cursor or flags disagree")
                    for key, digest in (("action_f32_hex", actions), ("native_action_f32_hex", native)):
                        try:
                            raw = bytes.fromhex(step[key])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError("invalid 3D training step action") from exc
                        if len(raw) != 12:
                            raise ValueError("training step action is not 3D float32")
                        digest.update(raw)
                    reward_sum += step["reward"]
                if (step is None or any(step[k] != row[k] for k in ("terminated", "truncated", "terminal"))
                        or actions.hexdigest() != row.get("action_trace_sha256")
                        or native.hexdigest() != row.get("native_action_trace_sha256")
                        or not math.isfinite(row.get("return", math.inf))
                        or not math.isclose(reward_sum, row["return"], rel_tol=0, abs_tol=1e-4)):
                    raise ValueError("episode differs from complete step ledger")
                decisions += length
                updates = row["updates"]
                episodes.append(row)
                phase = "checkpoint" if checkpoint_index < len(targets) and decisions >= targets[checkpoint_index] else "intent"
            else:
                raise ValueError(f"partial or unexpected training event: {event}")
        if (phase != "intent" or checkpoint_index != len(targets)
                or len(pins) != len(selected) or step_stream.read(1)
                or decisions != result["decisions"] or updates != result["updates"]
                or len(episodes) != result["episodes"]
                or train_digest.hexdigest() != result["training_ledger_sha256"]
                or step_digest.hexdigest() != result["step_ledger_sha256"]):
            raise ValueError("source result lacks a complete final ledger cursor")
    return pins


def preflight(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT) -> dict:
    """No torch.load, output file, env import, construction, or reset."""
    root = root.resolve(strict=True)
    path = _evaluation_path(root, protocol_path)
    if path.parent != root / "experiments" or _digest(path) != _sha(protocol_sha256):
        raise ValueError("separate frozen evaluation protocol SHA mismatch")
    p = _json(path.read_bytes())
    if (set(p) != {"format", "purpose", "evaluation_source_sha256", "source", "cells",
                   "environment", "modes", "repeats", "seed", "max_steps", "output_dir"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["cells"] != CELLS or p["environment"] != ENVIRONMENT
            or p["modes"] != list(MODES) or type(p["repeats"]) is not int or not 1 <= p["repeats"] <= 4
            or type(p["seed"]) is not int or not 0 <= p["seed"] <= 2**32 - 4 * p["repeats"]
            or type(p["max_steps"]) is not int or p["max_steps"] != 2000):
        raise ValueError("only full 2000-decision, paired consumed-TRAIN episodes are allowed")
    if _digest(_path(root, "scripts/evaluate_tdmpc2_full_train.py")) != _sha(p["evaluation_source_sha256"]):
        raise ValueError("evaluation executable SHA mismatch")
    out = _path(root, p["output_dir"], existing=False)
    if (out.parent != root / "runs" or not out.name.startswith("tdmpc2-full-train-")
            or not out.name or out.exists() or not out.parent.is_dir()):
        raise ValueError("output receipt must use a new, unique runs/tdmpc2-full-train-* directory")
    source, result, run = _source(root, p)
    pins = _ledger(root, source, result, run, p["source"])
    return {"status": "preflight_only", "environment_resets": 0,
            "reused_train_only": True, "generalization_claim": False,
            "protocol_sha256": protocol_sha256, "source_protocol_sha256": p["source"]["protocol"]["sha256"],
            "source_result_sha256": p["source"]["result"]["sha256"],
            "source_sha256": source["source_sha256"], "output_dir": p["output_dir"],
            "cells": CELLS, "checkpoints": pins, "modes": list(MODES), "repeats": p["repeats"],
            "max_steps": 2000, "planned_episodes": len(pins) * len(CELLS) * p["repeats"] * len(MODES)}


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _journal(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _model(pin: dict, source: dict, *, model_factory=None):
    if _digest(Path(pin["path"])) != pin["sha256"]:
        raise ValueError("checkpoint changed after preflight")
    state = torch.load(pin["path"], map_location="cpu", weights_only=False)
    if (not isinstance(state, dict) or state.get("format") != source["format"]
            or state.get("protocol_sha256") != source["protocol_sha256"]
            or state.get("source_sha256") != source["source_sha256"]
            or state.get("resume_supported") is not False or state.get("action_dim") != 3
            or any(state.get(k) != pin[k] for k in ("target", "decisions", "updates", "episodes"))
            or not isinstance(state.get("learner"), dict)
            or "q_scale" not in state["learner"]
            or not all(isinstance(k, str) and (k == "q_scale" or k.startswith("model."))
                       for k in state["learner"])):
        raise ValueError("checkpoint contents differ from source/ledger cursor")
    if model_factory is None:
        from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    else:
        model = model_factory()
    model.load_state_dict({key[6:]: value for key, value in state["learner"].items()
                           if key.startswith("model.")}, strict=True)
    model.to("cpu").eval()
    return model


def _episode(env, model, planner, cell: dict, *, mode: str, repeat: int, seed: int,
             checkpoint_target: int, max_steps: int = 2000, clock=time.perf_counter) -> dict:
    """Exactly one reset; a finish at decision 2000 is NOT time-limit censoring."""
    from haic.algorithms.tdmpc2.haic_env import environment_action, episode_boundary, model_observation

    _seed(seed)
    planner.reset()
    obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
    if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
            or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
        raise ValueError("environment reset did not honor consumed TRAIN road")
    pixels = model_observation(obs)
    total = 0.0
    first_latency = 0.0
    latency_sum = latency_max = 0.0
    action_trace = hashlib.sha256()
    native_trace = hashlib.sha256()
    for decision in range(1, max_steps + 1):
        started = clock()
        with torch.inference_mode():
            tensor = torch.as_tensor(pixels[None], device="cpu")
            if mode == "prior":
                _, info = model.pi(model.encode(tensor, None), None)
                action = info["mean"][0]
            elif mode == "mppi":
                action = planner.plan(tensor, t0=(decision == 1), eval_mode=True)
            else:
                raise ValueError("unsupported evaluator mode")
            action = action.detach().cpu().numpy().astype(np.float32)
        latency = clock() - started
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid CPU action latency")
        native = environment_action(action)
        latency_sum += latency
        latency_max = max(latency_max, latency)
        action_trace.update(action.tobytes())
        native_trace.update(native.tobytes())
        obs, reward, terminated, truncated, info = env.step(native)
        if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
            raise ValueError("environment road changed during episode")
        if (type(terminated) is not bool or type(truncated) is not bool or not isinstance(info, dict)
                or type(info.get("finished")) is not bool or type(reward) not in (int, float)
                or not math.isfinite(reward)
                or any(type(info.get(k)) not in (int, float) or not math.isfinite(info[k])
                       for k in ("progress", "damage"))):
            raise ValueError("invalid environment flags or raw metrics")
        total += reward
        if not math.isfinite(total):
            raise ValueError("nonfinite raw episode return")
        done, terminal = episode_boundary(terminated, truncated, info)
        if decision == 1:
            first_latency = latency
        if done:
            finished = info["finished"]
            if (finished and not truncated or truncated and not finished and not terminated and decision != max_steps):
                raise ValueError("finish/truncation conflicts with full-episode semantics")
            if (info.get("finish_time_s") is not None
                    and (type(info["finish_time_s"]) not in (int, float)
                         or not math.isfinite(info["finish_time_s"]))):
                raise ValueError("nonfinite finish time")
            return {"event": "episode", "target": checkpoint_target, "mode": mode, "repeat": repeat,
                    "episode_seed": seed, **cell, "decisions": decision, "max_steps": max_steps,
                    "raw_return": total, "progress": float(info["progress"]), "damage": float(info["damage"]),
                    "finished": finished, "terminated": terminated, "truncated": truncated,
                    "terminal": terminal, "censored": bool(truncated and not finished and not terminated),
                    "finish_time_s": info.get("finish_time_s"),
                    "action_trace_sha256": action_trace.hexdigest(),
                    "native_action_trace_sha256": native_trace.hexdigest(),
                    "action_latency_first_s": first_latency,
                    "action_latency_mean_s": latency_sum / decision,
                    "action_latency_max_s": latency_max, "action_latency_total_s": latency_sum}
        pixels = model_observation(obs)
    raise ValueError("environment did not end its full 2000-decision episode")


def _summary(rows: list[dict], pins: list[dict], repeats: int) -> dict:
    per_checkpoint = {}
    for pin in pins:
        by_mode = {}
        for mode in MODES:
            by_road = []
            for cell in CELLS:
                cohort = [row for row in rows if row["target"] == pin["target"] and row["mode"] == mode
                          and row["track_id"] == cell["track_id"] and row["geometry_seed"] == cell["geometry_seed"]]
                by_road.append({**cell, "episodes": len(cohort), "finishes": sum(row["finished"] for row in cohort),
                                "uncensored": sum(not row["censored"] for row in cohort),
                                "terminal": sum(row["terminal"] for row in cohort),
                                "censored": sum(row["censored"] for row in cohort),
                                "decisions": sum(row["decisions"] for row in cohort),
                                "mean_progress": sum(row["progress"] for row in cohort) / repeats,
                                "mean_raw_return": sum(row["raw_return"] for row in cohort) / repeats,
                                "mean_damage": sum(row["damage"] for row in cohort) / repeats,
                                "mean_action_latency_s": sum(row["action_latency_total_s"] for row in cohort)
                                / sum(row["decisions"] for row in cohort),
                                "max_action_latency_s": max(row["action_latency_max_s"] for row in cohort)})
            cohort = [row for row in rows if row["target"] == pin["target"] and row["mode"] == mode]
            n = len(CELLS) * repeats
            if len(cohort) != n or any(road["episodes"] != repeats for road in by_road):
                raise ValueError("incomplete road/mode/repeat denominator")
            by_mode[mode] = {"episodes": n, "finishes": sum(r["finishes"] for r in by_road),
                             "uncensored": sum(r["uncensored"] for r in by_road),
                             "terminal": sum(r["terminal"] for r in by_road),
                             "censored": sum(r["censored"] for r in by_road),
                             "decisions": sum(r["decisions"] for r in by_road),
                             "mean_progress": sum(row["progress"] for row in cohort) / n,
                             "mean_raw_return": sum(row["raw_return"] for row in cohort) / n,
                             "mean_damage": sum(row["damage"] for row in cohort) / n,
                             "mean_action_latency_s": sum(row["action_latency_total_s"] for row in cohort)
                             / sum(row["decisions"] for row in cohort),
                             "max_action_latency_s": max(row["action_latency_max_s"] for row in cohort),
                             "roads": by_road}
        per_checkpoint[str(pin["target"])] = {"checkpoint_sha256": pin["sha256"], "per_mode": by_mode}
    return per_checkpoint


def execute(protocol_path: Path, protocol_sha256: str, *, root: Path = ROOT,
            env_factory=None, model_factory=None, planner_factory=None) -> dict:
    """One exclusive receipt; never retry a partial run or choose replacement roads."""
    if torch.version.cuda is not None or not str(torch.__version__).startswith("2.1.0"):
        raise ValueError("execute requires a CPU-only Torch 2.1.0 interpreter")
    root = root.resolve(strict=True)
    checked = preflight(protocol_path, protocol_sha256, root=root)
    p = _json(_evaluation_path(root, protocol_path).read_bytes())
    if (p["output_dir"] != checked["output_dir"]
            or preflight_recheck(root, p, protocol_path, protocol_sha256) != checked["checkpoints"]):
        raise ValueError("frozen evaluation protocol or source changed after preflight")
    source, _, _ = _source(root, p)
    from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
    if env_factory is None:
        from haic.algorithms.tdmpc2.haic_env import make_training_env
        env_factory = make_training_env
    if planner_factory is None:
        planner_factory = TDMPC2Planner
    output = _path(root, p["output_dir"], existing=False)
    output.mkdir(mode=0o700, exist_ok=False)
    ledger = output / "episodes.jsonl"
    rows = []
    env = None
    try:
        for pin in checked["checkpoints"]:
            bound = {**pin, "path": str(_path(root, pin["path"]))}
            model = _model(bound, {**source, "protocol_sha256": checked["source_protocol_sha256"]},
                           model_factory=model_factory)
            planner = planner_factory(model, PlannerConfig(action_dim=3, discount=.995, episodic=True))
            for repeat in range(p["repeats"]):
                for index, cell in enumerate(CELLS):
                    seed = p["seed"] + repeat * len(CELLS) + index
                    for mode in MODES:
                        # Recheck all source bytes, full ledgers and checkpoint hashes immediately before reset.
                        if (preflight_recheck(root, p, protocol_path, protocol_sha256)
                                != checked["checkpoints"]):
                            raise ValueError("source checkpoint cursor changed before reset")
                        _journal(ledger, {"event": "reset_intent", "target": pin["target"],
                                          "mode": mode, "repeat": repeat, "episode_seed": seed, **cell})
                        if env is None:
                            env = env_factory(2000)
                        row = _episode(env, model, planner, cell, mode=mode, repeat=repeat, seed=seed,
                                       checkpoint_target=pin["target"])
                        _journal(ledger, row)
                        rows.append(row)
        report = {"status": "complete", "scope": "reused_consumed_TRAIN_development_only",
                  "reused_train_only": True, "generalization_claim": False, "official_score": False,
                  "full_episode_finish_comparison_valid": True,
                  "pairing": "same consumed TRAIN road/reset RNG seed; separate resets and different trajectories",
                  "protocol_sha256": protocol_sha256, "source_protocol_sha256": checked["source_protocol_sha256"],
                  "source_result_sha256": checked["source_result_sha256"],
                  "checkpoints": checked["checkpoints"], "cpu_only_torch": str(torch.__version__),
                  "denominators": {"checkpoints": len(checked["checkpoints"]), "distinct_roads": len(CELLS),
                                   "repeats_per_road_mode_checkpoint": p["repeats"],
                                   "modes": list(MODES), "planned_episodes": checked["planned_episodes"],
                                   "completed_episodes": len(rows)},
                  "episodes_sha256": _digest(ledger),
                  "per_checkpoint": _summary(rows, checked["checkpoints"], p["repeats"])}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return report
    except BaseException as exc:
        _journal(ledger, {"event": "partial", "reason": type(exc).__name__,
                          "complete_episodes": len(rows), "resume_supported": False})
        raise
    finally:
        if env is not None:
            env.close()


def preflight_recheck(root: Path, p: dict, protocol_path: Path, sha: str) -> list[dict]:
    """Same full preflight with the already reserved output directory allowed."""
    if _digest(_evaluation_path(root, protocol_path)) != sha:
        raise ValueError("evaluation protocol changed before reset")
    if _digest(_path(root, "scripts/evaluate_tdmpc2_full_train.py")) != p["evaluation_source_sha256"]:
        raise ValueError("evaluation executable changed before reset")
    source, result, run = _source(root, p)
    return _ledger(root, source, result, run, p["source"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="separately frozen experiments/*.json")
    parser.add_argument("--protocol-sha256", required=True, help="external SHA-256 of the evaluation protocol")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="default: file-only checks, zero resets")
    mode.add_argument("--execute", action="store_true", help="CPU full-episode TRAIN interaction")
    args = parser.parse_args()
    report = (execute(args.protocol, args.protocol_sha256) if args.execute else
              preflight(args.protocol, args.protocol_sha256))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
