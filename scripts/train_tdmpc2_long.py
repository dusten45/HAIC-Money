"""Independent, source-pinned ~100k TD-MPC2 run on four consumed TRAIN roads.

Run from the repository root with ``python -m scripts.train_tdmpc2_long``.
Only a new, externally SHA-pinned JSON protocol can enable --run. There is no
resume mode: interrupted episodes and checkpoint gaps remain consumed evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from haic.algorithms.tdmpc2.action_2d import environment_action as action_2d
from haic.algorithms.tdmpc2.haic_env import (
    environment_action as action_3d, episode_boundary, make_training_env, model_observation,
)
from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from haic.algorithms.tdmpc2.replay import EpisodeReplay


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-long-train-v1"
UPSTREAM = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
R6_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
PRIOR_SHA = "2e4a837795abad44623653f789d966cdeed3bb93b154bb2b17cbff18d1c35386"
EXPLORATION_SHA = "2013024e45e80dd3dd75c532ccb378b9df9a10d6627344487ecb20fbbae4b48f"
SOURCE_PATHS = frozenset({
    "scripts/train_tdmpc2_long.py", "haic/algorithms/tdmpc2/__init__.py",
    "haic/algorithms/tdmpc2/action_2d.py", "haic/algorithms/tdmpc2/haic_env.py",
    "haic/algorithms/tdmpc2/model.py", "haic/algorithms/tdmpc2/replay.py",
    "haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/planner.py",
    "env_wrapper.py", "damage.py", "core/__init__.py", "core/vendor/__init__.py",
    "core/vendor/car_racing.py", "core/vendor/car_dynamics.py",
    "core/track_variables.py", "core/obstacle_contacts.py", "core/finish_line.py",
})
CELLS = [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
ENVIRONMENT = {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}
TARGETS = [20000, 40000, 70000, 100000]
SEED_SCHEDULE = ("exactly 10000 uniform actions; pretrain after decision 10000 before "
                 "first planned action; v2 used 10001 uniform actions")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError("expected lowercase SHA-256 digest")
    return value


def _inside(root: Path, name: str | Path, *, required: bool = True) -> Path:
    path = Path(name)
    path = path if path.is_absolute() else root / path
    root = root.resolve(strict=True)
    path = path.absolute()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"path outside repository: {path}")
    relative = path.relative_to(root)
    if any((root / Path(*relative.parts[:n])).is_symlink() for n in range(1, len(relative.parts) + 1)):
        raise ValueError(f"symlink path is not pinnable: {path}")
    if required and not path.is_file():
        raise ValueError(f"required file missing: {path}")
    return path


def _json(path: Path) -> dict:
    if not 0 < path.stat().st_size < 1024 * 1024:
        raise ValueError(f"invalid JSON file size: {path}")

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON field: {key}")
            obj[key] = value
        return obj

    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(result, dict):
        raise ValueError("JSON root must be an object")
    return result


def _reference(root: Path, ref: dict, name: str) -> dict:
    if not isinstance(ref, dict) or ref != {"path": name, "sha256": _sha(ref.get("sha256"))}:
        raise ValueError(f"invalid frozen reference: {name}")
    path = _inside(root, name)
    if digest(path) != ref["sha256"]:
        raise ValueError(f"frozen reference hash mismatch: {name}")
    return _json(path)


def _check_sources(root: Path, sources: dict) -> None:
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("source map must pin exactly executable and environment files")
    for name, pinned in sources.items():
        if digest(_inside(root, name)) != _sha(pinned):
            raise ValueError(f"source hash mismatch: {name}")


def _selected_arm(result: dict) -> str:
    arms = result["arms"]
    if set(arms) != {"independent_3d", "exclusive_2d"}:
        raise ValueError("exploration arms missing")
    a, b = arms["independent_3d"], arms["exclusive_2d"]
    if any(arm.get("episodes") != 8 or type(arm.get("decisions")) is not int or arm["decisions"] <= 0
           or any(not isinstance(arm.get(key), (int, float)) or not math.isfinite(arm[key])
                  for key in ("mean_progress", "mean_damage", "mean_return", "mean_unique_spatial_bins"))
           for arm in (a, b)):
        raise ValueError("incomplete exploration denominator or metric")
    for first, second, label in ((a, b, "independent_3d"), (b, a, "exclusive_2d")):
        if first["mean_progress"] - second["mean_progress"] >= .01 and first["mean_damage"] <= second["mean_damage"] + .1:
            return label
    if abs(a["mean_return"] - b["mean_return"]) >= 1:
        return "independent_3d" if a["mean_return"] > b["mean_return"] else "exclusive_2d"
    if a["mean_unique_spatial_bins"] != b["mean_unique_spatial_bins"]:
        return "independent_3d" if a["mean_unique_spatial_bins"] > b["mean_unique_spatial_bins"] else "exclusive_2d"
    if a["mean_damage"] != b["mean_damage"]:
        return "independent_3d" if a["mean_damage"] < b["mean_damage"] else "exclusive_2d"
    return "exclusive_2d"


def preflight(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT) -> dict:
    """Read-only protocol, consumed-cell, selection, source, and raw resource gate."""
    root = root.resolve(strict=True)
    path = _inside(root, protocol_path)
    if path.parent != root / "experiments" or path.suffix != ".json" or digest(path) != _sha(protocol_sha256):
        raise ValueError("frozen new experiments/*.json protocol SHA mismatch")
    p = _json(path)
    if (set(p) != {"format", "purpose", "upstream_revision", "source_sha256", "r6_protocol",
                   "prior_consumption", "exploration_protocol", "exploration_result", "selection",
                   "cells", "episode_schedule", "environment", "run_dir", "training",
                   "checkpoint_targets", "resources", "seed_schedule"}
            or p["format"] != FORMAT or p["purpose"] != "consumed-TRAIN-development"
            or p["upstream_revision"] != UPSTREAM or p["seed_schedule"] != SEED_SCHEDULE):
        raise ValueError("unrecognized long TRAIN protocol")
    output = _inside(root, run_dir, required=False)
    if (output.parent != root / "runs" or not output.name.startswith("tdmpc2-long-")
            or not (root / "runs").is_dir() or (output.exists() and not output.is_dir())
            or p["run_dir"] != output.relative_to(root).as_posix()):
        raise ValueError("run directory differs from frozen dedicated output")
    if p["cells"] != CELLS or p["episode_schedule"] != [0, 1, 2, 3] or p["environment"] != ENVIRONMENT:
        raise ValueError("only the exact four consumed obstacle-enabled TRAIN cells are allowed")
    r6 = _reference(root, p["r6_protocol"], "experiments/drqv2-geometry-mix-v1-r6.json")
    prior = _reference(root, p["prior_consumption"], "experiments/dreamerv3-reused-train-diagnostic-v1.json")
    if p["r6_protocol"]["sha256"] != R6_SHA or p["prior_consumption"]["sha256"] != PRIOR_SHA:
        raise ValueError("consumed TRAIN lineage differs from pinned references")
    r6_env = r6.get("environment", {})
    if (r6.get("training_pool", {}).get("partition") != "TRAIN"
            or not set(ROADS) <= set(r6["training_pool"].get("geometry_seeds", []))
            or 1 not in r6["training_pool"].get("track_ids", [])
            or set(ROADS) & set(r6.get("diagnostic_pool", {}).get("geometry_seeds", []))
            or r6_env.get("partition") != "TRAIN" or r6_env.get("obstacles") is not True
            or r6_env.get("frame_skip") != 4 or r6_env.get("max_steps") != 2000
            or r6_env.get("reward_shaping") is not False or r6_env.get("reward_normalization") is not False
            or prior.get("cells") != CELLS or prior.get("frame_skip") != 4 or prior.get("max_steps") != 2000
            or prior.get("purpose") != "reused-TRAIN-engineering-diagnostic"):
        raise ValueError("prior allocation does not prove four consumed TRAIN roads")
    exploration = _reference(root, p["exploration_protocol"], "experiments/tdmpc2-exploration-v2.json")
    if (p["exploration_protocol"]["sha256"] != EXPLORATION_SHA or exploration.get("cells") != CELLS
            or exploration.get("environment") != ENVIRONMENT or exploration.get("repeats") != 2
            or exploration.get("arms") != ["independent_3d", "exclusive_2d"]
            or exploration.get("selection", "").startswith("Compare 8 paired road/repeat episodes.") is False):
        raise ValueError("exploration selection contract changed")
    selection = _reference(root, p["exploration_result"], "experiments/tdmpc2-exploration-v2-result.json")
    receipt = _inside(root, "runs/tdmpc2-exploration-20260928-v2/result.json")
    episodes = _inside(root, "runs/tdmpc2-exploration-20260928-v2/episodes.jsonl")
    steps = _inside(root, "runs/tdmpc2-exploration-20260928-v2/steps.jsonl")
    result = _json(receipt)
    if (selection.get("format") != "haic-tdmpc2-exploration-result-v2"
            or selection.get("status") != "completed"
            or selection.get("protocol") != p["exploration_protocol"]["path"]
            or selection.get("protocol_sha256") != EXPLORATION_SHA
            or selection.get("run_result") != "runs/tdmpc2-exploration-20260928-v2/result.json"
            or selection.get("run_result_sha256") != digest(receipt)
            or selection.get("episode_ledger_sha256") != digest(episodes)
            or selection.get("step_ledger_sha256") != digest(steps)
            or result.get("status") != "complete" or result.get("partition") != "consumed-TRAIN-development"
            or result.get("protocol_sha256") != EXPLORATION_SHA or result.get("episodes") != 16
            or result.get("episodes_sha256") != selection["episode_ledger_sha256"]
            or result.get("steps_sha256") != selection["step_ledger_sha256"]
            or selection.get("selection") != _selected_arm(result)):
        raise ValueError("exploration selection evidence or ledger hashes differ")
    denominator = selection.get("denominators", {})
    if (not isinstance(denominator, dict) or denominator.get("distinct_roads") != 4
            or denominator.get("repeats_per_road") != 2
            or denominator.get("paired_episodes") != 8
            or denominator.get("maximum_steps_per_episode") != 2000):
        raise ValueError("exploration selection denominators differ")
    for arm in ("independent_3d", "exclusive_2d"):
        arm_result = result["arms"][arm]
        arm_selection = selection.get(arm, {})
        if (denominator.get(f"{arm}_decisions") != arm_result["decisions"]
                or not isinstance(arm_selection, dict)
                or arm_selection.get("finishes") != arm_result.get("finishes")
                or any(not math.isclose(arm_selection.get(selected, math.inf), arm_result[measured],
                                        rel_tol=0, abs_tol=1e-9)
                       for selected, measured in (("mean_progress", "mean_progress"),
                                                  ("mean_episode_return", "mean_return"),
                                                  ("mean_damage", "mean_damage"),
                                                  ("mean_steps", "mean_steps"),
                                                  ("mean_unique_spatial_bins", "mean_unique_spatial_bins")))):
            raise ValueError("exploration selection summary differs from frozen run receipt")
    arm = selection["selection"]
    if p["selection"] != {"arm": arm, "action_dim": 2 if arm == "exclusive_2d" else 3}:
        raise ValueError("model action dimension differs from selected exploration arm")
    _check_sources(root, p["source_sha256"])

    train = p["training"]
    if (not isinstance(train, dict) or set(train) != {
            "seed", "decision_cap", "seed_steps", "pretrain_updates", "update_cap",
            "updates_per_post_seed_decision", "max_steps", "replay_capacity", "batch_size",
            "device", "horizon", "discount", "rho", "model_size", "num_bins",
            "augmentation_pad", "episodic", "observation_shape", "action_dim"}
            or type(train["seed"]) is not int or not 0 <= train["seed"] <= 2**32 - 1
            or train["decision_cap"] != 102000 or train["update_cap"] != 102000
            or train["seed_steps"] != 10000 or train["pretrain_updates"] != 10000
            or train["updates_per_post_seed_decision"] != 1 or train["max_steps"] != 2000
            or type(train["replay_capacity"]) is not int or train["replay_capacity"] < 100000
            or train["batch_size"] != 256 or train["horizon"] != 3 or train["discount"] != .995
            or train["rho"] != .5 or train["model_size"] != 5 or train["num_bins"] != 101
            or train["augmentation_pad"] != 3 or train["episodic"] is not True
            or train["observation_shape"] != [4, 64, 64]
            or type(train["action_dim"]) is not int or train["action_dim"] != p["selection"]["action_dim"]
            or train["device"] not in ("cpu", "cuda") or p["checkpoint_targets"] != TARGETS):
        raise ValueError("long-run seed/pretraining/model/action/budget settings changed")
    if train["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("pinned CUDA device unavailable")
    resources = p["resources"]
    if (not isinstance(resources, dict) or set(resources) != {
            "min_cgroup_available_bytes", "min_disk_available_bytes", "max_wall_seconds"}
            or type(resources["min_cgroup_available_bytes"]) is not int
            or resources["min_cgroup_available_bytes"] < 16 * 1024**3
            or type(resources["min_disk_available_bytes"]) is not int
            or resources["min_disk_available_bytes"] < 16 * 1024**3
            or type(resources["max_wall_seconds"]) is not int
            or not 7200 <= resources["max_wall_seconds"] <= 86400):
        raise ValueError("raw resource floors or wall budget invalid")
    _resources(output, p)
    return p


def _resources(run_dir: Path, protocol: dict) -> None:
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError("raw cgroup headroom unavailable") from exc
    if limit - current < protocol["resources"]["min_cgroup_available_bytes"]:
        raise ValueError("insufficient raw cgroup headroom")
    disk = os.statvfs(run_dir.parent)
    if disk.f_bavail * disk.f_frsize < protocol["resources"]["min_disk_available_bytes"]:
        raise ValueError("insufficient raw run disk headroom")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _journal(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _cost(clock, started: float, started_cpu: float) -> dict:
    return {"elapsed_seconds": float(clock() - started),
            "cpu_seconds": float(time.process_time() - started_cpu),
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available() else None}


def _components(protocol: dict):
    cfg = protocol["training"]
    model = WorldModel(TDMPC2ModelConfig(action_dim=cfg["action_dim"],
                                         obs_shape={"rgb": (4, 64, 64)}, episodic=True)).to(cfg["device"])
    learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=cfg["max_steps"],
                                                      horizon=cfg["horizon"], batch_size=cfg["batch_size"]))
    replay = EpisodeReplay(capacity=cfg["replay_capacity"], horizon=cfg["horizon"],
                           action_dim=cfg["action_dim"], seed=cfg["seed"], include_partial=False,
                           bootstrap_on_truncation=True, augmentation_pad=3)
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=cfg["action_dim"],
                                                 discount=learner.discount, episodic=True))
    if not math.isclose(learner.discount, cfg["discount"], abs_tol=1e-12):
        raise ValueError("learner discount differs from frozen protocol")
    return learner, replay, planner


def _planned_action(model, planner, pixels: np.ndarray, device: str, *, t0: bool,
                    capture_prior: bool = True) -> tuple[np.ndarray, dict | None]:
    """Capture the FIRST planner prior mean without another encode/pi or RNG draw."""
    captured = []

    def hook(_module, _inputs, output):
        if not captured:
            captured.append(output[0, :model.cfg.action_dim].tanh().detach().cpu().numpy().copy())

    handle = model._pi.register_forward_hook(hook) if capture_prior else None
    try:
        with torch.inference_mode():
            action = planner.plan(torch.as_tensor(pixels[None], device=device), t0=t0)
            action = action.detach().cpu().numpy().astype(np.float32)
            mppi_mean = planner.prev_mean[0].detach().cpu().numpy().astype(np.float32).copy()
    finally:
        if handle is not None:
            handle.remove()
    if not capture_prior:
        return action, None
    if (len(captured) != 1 or captured[0].shape != action.shape or mppi_mean.shape != action.shape
            or not np.isfinite(captured[0]).all() or not np.isfinite(mppi_mean).all()):
        raise ValueError("planner did not expose same-observation first policy-prior mean")
    prior = captured[0]
    return action, {"kind": "first_planner_pi_tanh_mean_vs_mppi_weighted_elite_mean",
                    "prior_mean": prior.tolist(),
                    "mppi_weighted_elite_mean": mppi_mean.tolist(),
                    "delta": (mppi_mean - prior).tolist(),
                    "delta_l2": float(np.linalg.norm(mppi_mean - prior)),
                    "applied_exploration_noised_action": action.tolist(),
                    "applied_delta_l2": float(np.linalg.norm(action - prior))}


def _freeze_probe(replay, batch_size: int) -> dict:
    state = deepcopy(replay.rng.bit_generator.state)
    try:
        return {key: value.cpu().clone() for key, value in replay.sample(batch_size).items()}
    finally:
        replay.rng.bit_generator.state = state


def _diagnostic(learner, probe: dict, seed: int) -> dict:
    """Fixed TRAIN windows; seeded stochastic encoder/Q and no optimizer step."""
    saved = _rng_state()
    try:
        _seed(seed)
        learner.model.eval()
        device = learner.q_scale.device
        with torch.inference_mode():
            obs = probe["obs"][:2].to(device)
            action = probe["action"][0].to(device)
            reward = probe["reward"][0].to(device)
            terminal = probe["terminal"][0].to(device)
            z, next_z = learner.model.encode(obs[0], None), learner.model.encode(obs[1], None)
            predicted = learner.model.next(z, action, None)
            reward_pred = two_hot_inv(learner.model.reward(z, action, None), learner.model.cfg)
            target = learner._td_target(next_z[None], reward[None], terminal[None])[0]
            q = learner.model.Q(z, action, None, return_type="avg")
            all_rewards = probe["reward"].float().squeeze(-1)
            h3_raw_return = all_rewards.sum(dim=0)
            h3_discounted_return = torch.zeros_like(h3_raw_return)
            for t in range(all_rewards.shape[0]):
                h3_discounted_return += learner.discount ** t * all_rewards[t]
            metrics = {"sample_count": int(action.shape[0]),
                       "episode_ids": probe["episode_id"].tolist(),
                       "start_steps": probe["start_step"].tolist(),
                       "distinct_episode_count": int(probe["episode_id"].unique().numel()),
                       "terminal_positive_transitions": int(probe["terminal"].sum().item()),
                       "reward_min": all_rewards.min().item(), "reward_max": all_rewards.max().item(),
                       "h3_raw_return_min": h3_raw_return.min().item(),
                       "h3_raw_return_max": h3_raw_return.max().item(),
                       "h3_discounted_return_min": h3_discounted_return.min().item(),
                       "h3_discounted_return_max": h3_discounted_return.max().item(),
                       "h3_unique_raw_returns_at_1e_minus_6": int(torch.unique(
                           (h3_raw_return.double() * 1e6).round()).numel()),
                       "latent_mse": F.mse_loss(predicted, next_z).item(),
                       "latent_frozen_anchor_mse": F.mse_loss(z, next_z).item(),
                       "reward_mae": (reward_pred - reward).abs().mean().item(),
                       "constant_reward_mae": (reward.mean() - reward).abs().mean().item(),
                       "q_td_residual_abs_mean": (q - target).abs().mean().item(),
                       "termination_bce": F.binary_cross_entropy(
                           learner.model.termination(predicted, None), terminal).item(),
                       "q_scale": learner.q_scale.item()}
        if any(not math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError("nonfinite frozen TRAIN probe")
        return metrics
    finally:
        _restore_rng(saved)


def _checkpoint(path: Path, learner, replay, probe, decisions: int, updates: int, episodes: int,
                protocol_sha256: str, sources: dict, target: int) -> str:
    if replay.active_length or path.exists():
        raise ValueError("checkpoint requires a new whole-episode boundary file")
    state = {"format": FORMAT, "protocol_sha256": protocol_sha256, "source_sha256": sources,
             "target": target, "decisions": decisions, "updates": updates, "episodes": episodes,
             "action_dim": replay.action_dim, "learner": learner.state_dict(),
             "optim": learner.optim.state_dict(), "pi_optim": learner.pi_optim.state_dict(),
             "replay": replay.state_dict(), "probe": probe, "rng": _rng_state(),
             "resume_supported": False}
    # Exclusive creation retains even an interrupted, incomplete checkpoint for audit.
    with path.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return digest(path)


def run(protocol: dict, protocol_sha256: str, run_dir: Path, *, env_factory=make_training_env,
        clock=time.monotonic) -> dict:
    """Start from scratch only; no restart or exact-resume claim on partial data."""
    if protocol["cells"] != CELLS or protocol["environment"] != ENVIRONMENT:
        raise ValueError("only four consumed TRAIN roads are permitted")
    cfg = protocol["training"]
    run_dir.mkdir(exist_ok=False)
    ledger, steps = run_dir / "training.jsonl", run_dir / "steps.jsonl"
    started, started_cpu = clock(), time.process_time()
    runtime = {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "gymnasium": gym.__version__,
               "torch_cuda": torch.version.cuda, "device": cfg["device"],
               "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
               if cfg["device"] == "cuda" else None}
    _journal(ledger, {"event": "start", "protocol_sha256": protocol_sha256,
                       "source_sha256": protocol["source_sha256"], "seed_schedule": SEED_SCHEDULE,
                       "runtime": runtime, "resume_supported": False})
    decisions = updates = episodes = 0
    probe = None
    latest_delta = None
    delta_count = 0
    delta_sum = np.zeros(cfg["action_dim"], dtype=np.float64)
    delta_abs_sum = np.zeros(cfg["action_dim"], dtype=np.float64)
    native_delta_abs_sum = np.zeros(3, dtype=np.float64)
    delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
    rolling_sum: dict[str, float] = {}
    rolling_count = 0
    interval_episodes: list[dict] = []
    checkpoints = []
    env = None
    try:
        _resources(run_dir, protocol)
        _check_sources(ROOT, protocol["source_sha256"])
        _seed(cfg["seed"])
        learner, replay, planner = _components(protocol)
        while decisions < TARGETS[-1]:
            if cfg["decision_cap"] - decisions < cfg["max_steps"]:
                raise RuntimeError("decision cap prevents required 100k whole-episode target")
            if clock() - started >= protocol["resources"]["max_wall_seconds"]:
                raise TimeoutError("wall budget expired before final TRAIN milestone")
            _resources(run_dir, protocol)
            _check_sources(ROOT, protocol["source_sha256"])
            cell = protocol["cells"][protocol["episode_schedule"][episodes % 4]]
            _journal(ledger, {"event": "reset_intent", "episode": episodes, "decisions": decisions, **cell})
            if env is None:
                env = env_factory(cfg["max_steps"])
            obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
            if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                    or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                raise ValueError("reset did not honor the pinned consumed TRAIN cell")
            pixels = model_observation(obs)
            replay.start_episode(pixels)
            planner.reset()
            _journal(ledger, {"event": "reset", "episode": episodes, "decisions": decisions, **cell})
            length = 0
            total = 0.0
            done = False
            progress = damage = 0.0
            info: dict = {}
            terminated = truncated = terminal = False
            action_trace, native_trace = hashlib.sha256(), hashlib.sha256()
            while length < cfg["max_steps"]:
                if clock() - started >= protocol["resources"]["max_wall_seconds"]:
                    raise TimeoutError("partial TRAIN episode: wall budget exhausted")
                index = decisions
                if index < cfg["seed_steps"]:
                    action = np.random.uniform(-1, 1, size=cfg["action_dim"]).astype(np.float32)
                else:
                    if updates < cfg["pretrain_updates"]:
                        raise RuntimeError("seed pretraining incomplete before first planned action")
                    action, latest_delta = _planned_action(
                        learner.model, planner, pixels, cfg["device"], t0=(length == 0))
                native = action_2d(action) if cfg["action_dim"] == 2 else action_3d(action)
                if latest_delta is not None and index >= cfg["seed_steps"]:
                    delta = np.asarray(latest_delta["delta"], dtype=np.float64)
                    prior = np.asarray(latest_delta["prior_mean"], dtype=np.float32)
                    prior_native = action_2d(prior) if cfg["action_dim"] == 2 else action_3d(prior)
                    mppi_mean = np.asarray(latest_delta["mppi_weighted_elite_mean"], dtype=np.float32)
                    mppi_native = action_2d(mppi_mean) if cfg["action_dim"] == 2 else action_3d(mppi_mean)
                    delta_count += 1
                    delta_sum += delta
                    delta_abs_sum += np.abs(delta)
                    native_delta_abs_sum += np.abs(mppi_native.astype(np.float64) - prior_native)
                    delta_l2_sum += latest_delta["delta_l2"]
                    delta_l2_max = max(delta_l2_max, latest_delta["delta_l2"])
                    applied_delta_l2_sum += latest_delta["applied_delta_l2"]
                action_trace.update(action.tobytes())
                native_trace.update(native.tobytes())
                obs, reward, terminated, truncated, info = env.step(native)
                if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                        or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                    raise ValueError("TRAIN cell changed during episode")
                following = model_observation(obs)
                done, terminal = episode_boundary(terminated, truncated, info)
                progress, damage = float(info["progress"]), float(info["damage"])
                if not all(math.isfinite(value) for value in (float(reward), progress, damage)):
                    raise ValueError("nonfinite TRAIN reward/progress/damage")
                if not done:
                    replay.add_step(following, action, reward, terminal=False)
                decisions += 1
                length += 1
                total += float(reward)
                _journal(steps, {"episode": episodes, "decision": decisions, **cell,
                                 "action_f32_hex": action.tobytes().hex(),
                                 "native_action_f32_hex": native.tobytes().hex(),
                                 "reward": float(reward), "terminated": bool(terminated),
                                 "truncated": bool(truncated), "terminal": terminal})
                if index == cfg["seed_steps"] - 1 and not replay.eligible_windows():
                    raise RuntimeError("no completed H+1 replay window at 10k seed boundary")
                count = cfg["pretrain_updates"] if index == cfg["seed_steps"] - 1 else int(index >= cfg["seed_steps"])
                if count and probe is None:
                    probe = _freeze_probe(replay, cfg["batch_size"])
                for _ in range(count):
                    if updates >= cfg["update_cap"]:
                        raise RuntimeError("update cap reached before 100k boundary")
                    if clock() - started >= protocol["resources"]["max_wall_seconds"]:
                        raise TimeoutError("partial learner updates: wall budget exhausted")
                    if updates % 256 == 0:
                        _resources(run_dir, protocol)
                    metrics = learner.update(replay)
                    if any(not math.isfinite(float(value)) for value in metrics.values()):
                        raise FloatingPointError("nonfinite learner update")
                    updates += 1
                    rolling_count += 1
                    for key, value in metrics.items():
                        rolling_sum[key] = rolling_sum.get(key, 0.0) + float(value)
                # The just-ended episode only enters replay AFTER this decision's updates.
                if done:
                    replay.add_step(following, action, reward, terminated=bool(terminated),
                                    truncated=bool(truncated), terminal=terminal)
                pixels = following
                if done:
                    break
            if replay.active_length or not done:
                raise RuntimeError("environment failed to close its full TRAIN episode")
            row = {"event": "episode", "episode": episodes, **cell,
                   "decisions": decisions, "updates": updates, "length": length,
                   "return": total, "progress": progress, "damage": damage,
                   "finished": bool(info.get("finished", False)),
                   "terminated": bool(terminated), "truncated": bool(truncated),
                   "terminal": terminal, "finish_time_s": info.get("finish_time_s"),
                   "action_trace_sha256": action_trace.hexdigest(),
                   "native_action_trace_sha256": native_trace.hexdigest()}
            _journal(ledger, row)
            interval_episodes.append({key: row[key] for key in (
                "episode", "track_id", "geometry_seed", "decisions", "length", "return", "progress",
                "damage", "finished")})
            episodes += 1
            if len(checkpoints) < len(TARGETS) and decisions >= TARGETS[len(checkpoints)]:
                target = TARGETS[len(checkpoints)]
                if probe is None or rolling_count == 0 or latest_delta is None or delta_count == 0:
                    raise RuntimeError("checkpoint lacks frozen TRAIN probe, updates or planned action")
                _resources(run_dir, protocol)
                _check_sources(ROOT, protocol["source_sha256"])
                diagnostic = _diagnostic(learner, probe, cfg["seed"] + 101)
                report = {"target": target, "decisions": decisions, "updates": updates,
                          "episodes": episodes, "rolling_update_count": rolling_count,
                          "rolling_update_means": {key: value / rolling_count for key, value in rolling_sum.items()},
                          "train_episodes_since_previous": interval_episodes,
                           "q_scale": float(learner.q_scale.item()),
                           "policy_entropy_rolling_mean": rolling_sum["pi_entropy"] / rolling_count,
                           "resource_usage": _cost(clock, started, started_cpu),
                           "same_observation_prior_vs_mppi_interval": {
                               "planned_actions": delta_count,
                               "mean_signed_model_delta": (delta_sum / delta_count).tolist(),
                               "mean_absolute_model_delta": (delta_abs_sum / delta_count).tolist(),
                               "mean_absolute_native_delta": (native_delta_abs_sum / delta_count).tolist(),
                               "mean_model_delta_l2": delta_l2_sum / delta_count,
                               "max_model_delta_l2": delta_l2_max,
                               "mean_applied_exploration_noised_delta_l2": applied_delta_l2_sum / delta_count},
                           "same_observation_prior_vs_mppi": latest_delta,
                          "frozen_train_probe": diagnostic, "step_ledger_sha256": digest(steps)}
                name = f"checkpoint-at-least-{target:06d}-step-{decisions:06d}.pt"
                checksum = _checkpoint(run_dir / name, learner, replay, probe, decisions, updates,
                                       episodes, protocol_sha256, protocol["source_sha256"], target)
                report.update(path=name, sha256=checksum)
                _journal(ledger, {"event": "checkpoint", **report})
                checkpoints.append(report)
                rolling_sum, rolling_count = {}, 0
                interval_episodes = []
                delta_count = 0
                delta_sum.fill(0)
                delta_abs_sum.fill(0)
                native_delta_abs_sum.fill(0)
                delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
        result = {"status": "completed_boundary_at_least_100k", "protocol_sha256": protocol_sha256,
                  "source_sha256": protocol["source_sha256"], "decisions": decisions,
                  "updates": updates, "pretrain_updates": cfg["pretrain_updates"],
                  "seed_schedule": SEED_SCHEDULE,
                   "episodes": episodes, "action_dim": cfg["action_dim"], "checkpoints": checkpoints,
                   "runtime": runtime, "resource_usage": _cost(clock, started, started_cpu),
                   "training_ledger_sha256": digest(ledger), "step_ledger_sha256": digest(steps),
                  "reused_train_only": True, "evaluation": None, "resume_supported": False}
        with (run_dir / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return result
    except BaseException as exc:
        _journal(ledger, {"event": "partial", "reason": type(exc).__name__, "episodes": episodes,
                           "decisions": decisions, "updates": updates,
                           "resource_usage": _cost(clock, started, started_cpu),
                           "active_length": locals()["replay"].active_length if "replay" in locals() else None,
                          "resume_supported": False})
        raise
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="file/resource validation only; zero env resets")
    mode.add_argument("--run", action="store_true", help="start distinct TRAIN run; never resume")
    args = parser.parse_args()
    protocol = preflight(args.protocol, args.protocol_sha256, args.run_dir)
    if args.preflight:
        print(json.dumps({"status": "preflight_only", "protocol_sha256": args.protocol_sha256,
                          "cells": CELLS, "action_dim": protocol["training"]["action_dim"],
                          "environment_resets": 0}, sort_keys=True))
    else:
        print(json.dumps(run(protocol, args.protocol_sha256, _inside(ROOT, args.run_dir, required=False)),
                         sort_keys=True))


if __name__ == "__main__":
    main()
