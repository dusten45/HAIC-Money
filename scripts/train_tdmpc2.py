"""Source-pinned, reused-TRAIN-only TD-MPC2 engineering pilot operator.

Run from the repository root with ``python -m scripts.train_tdmpc2``. Preflight
never constructs or resets an environment. A separately frozen JSON protocol and
its externally supplied SHA-256 are required for either interaction mode.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from haic.algorithms.tdmpc2.haic_env import (
    environment_action, episode_boundary, make_training_env, model_observation,
)
from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.model import two_hot_inv
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from haic.algorithms.tdmpc2.replay import EpisodeReplay


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-reused-train-pilot-v1"
UPSTREAM = "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
R6_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
PRIOR_SHA = "2e4a837795abad44623653f789d966cdeed3bb93b154bb2b17cbff18d1c35386"
SOURCE_PATHS = frozenset({
    "haic/algorithms/tdmpc2/__init__.py",
    "scripts/train_tdmpc2.py", "haic/algorithms/tdmpc2/haic_env.py",
    "haic/algorithms/tdmpc2/model.py", "haic/algorithms/tdmpc2/replay.py",
    "haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/planner.py",
    "env_wrapper.py", "damage.py", "core/__init__.py", "core/vendor/__init__.py",
    "core/vendor/car_racing.py",
    "core/vendor/car_dynamics.py", "core/track_variables.py",
    "core/obstacle_contacts.py", "core/finish_line.py",
})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inside(root: Path, path: Path, *, may_not_exist: bool = False) -> Path:
    root = root.resolve(strict=True)
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"path must be inside repository: {path}")
    relative = path.relative_to(root)
    if any(part.is_symlink() for part in (root / Path(*relative.parts[:n]) for n in range(1, len(relative.parts) + 1))):
        raise ValueError(f"symlink path is not source-pinnable: {path}")
    if not may_not_exist and not path.is_file():
        raise ValueError(f"required file missing: {path}")
    return path


def _sha(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA-256 hex digest")
    return value


def _positive(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _check_sources(root: Path, sources: dict) -> None:
    for name, pinned in sources.items():
        if digest(_inside(root, root / name)) != _sha(pinned):
            raise ValueError(f"executable/environment source hash mismatch: {name}")


def _train_cells(protocol: dict) -> None:
    if (protocol.get("purpose") != "reused-TRAIN-engineering-pilot"
            or protocol.get("cells") != [{"track_id": 1, "geometry_seed": seed} for seed in ROADS]
            or protocol.get("episode_schedule") != [0, 1, 2, 3]):
        raise ValueError("operator may only reset the fixed four consumed TRAIN roads")


def preflight(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT,
              require_training_device: bool = True) -> dict:
    """Validate frozen identity, reused TRAIN cells, exact source and resource bounds."""
    root = root.resolve(strict=True)
    protocol_path = _inside(root, protocol_path)
    if protocol_path.parent != root / "experiments" or protocol_path.suffix != ".json":
        raise ValueError("protocol must be an experiments/*.json file")
    if digest(protocol_path) != _sha(protocol_sha256):
        raise ValueError("protocol SHA-256 mismatch")
    if not 0 < protocol_path.stat().st_size < 1024 * 1024:
        raise ValueError("protocol size exceeds limit")

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate protocol field: {key}")
            obj[key] = value
        return obj

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(protocol, dict) or set(protocol) != {
        "format", "purpose", "upstream_revision", "source_sha256", "r6_protocol",
        "prior_consumption", "cells", "episode_schedule", "environment", "run_dir",
        "training", "evaluation", "resources",
    } or protocol["format"] != FORMAT or protocol["purpose"] != "reused-TRAIN-engineering-pilot" or protocol["upstream_revision"] != UPSTREAM:
        raise ValueError("not the pinned TD-MPC2 reused-TRAIN protocol")
    run_dir = _inside(root, run_dir, may_not_exist=True)
    if run_dir.parent != root / "runs" or not run_dir.name.startswith("tdmpc2-reused-train-"):
        raise ValueError("run directory must be a dedicated runs/tdmpc2-reused-train-* child")
    if protocol["run_dir"] != run_dir.relative_to(root).as_posix():
        raise ValueError("output run directory differs from frozen protocol")
    if not run_dir.parent.is_dir():
        raise ValueError("runs/ parent missing")
    if run_dir.exists() and not run_dir.is_dir():
        raise ValueError("run output is not a directory")
    if protocol["r6_protocol"] != {"path": "experiments/drqv2-geometry-mix-v1-r6.json", "sha256": R6_SHA} or protocol["prior_consumption"] != {"path": "experiments/dreamerv3-reused-train-diagnostic-v1.json", "sha256": PRIOR_SHA}:
        raise ValueError("consumed TRAIN references changed")
    r6_path = _inside(root, root / protocol["r6_protocol"]["path"])
    prior_path = _inside(root, root / protocol["prior_consumption"]["path"])
    if digest(r6_path) != R6_SHA or digest(prior_path) != PRIOR_SHA:
        raise ValueError("consumed TRAIN source reference hash mismatch")
    r6 = json.loads(r6_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    _train_cells(protocol)
    if protocol["environment"] != {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}:
        raise ValueError("TRAIN environment conditions differ from fixed obstacles/raw-reward wrapper")
    r6_env = r6.get("environment", {})
    if (r6.get("training_pool", {}).get("partition") != "TRAIN"
            or not set(ROADS) <= set(r6["training_pool"].get("geometry_seeds", []))
            or 1 not in r6["training_pool"].get("track_ids", [])
            or set(ROADS) & set(r6.get("diagnostic_pool", {}).get("geometry_seeds", []))
            or r6_env.get("partition") != "TRAIN" or r6_env.get("obstacles") is not True
            or r6_env.get("frame_skip") != 4 or r6_env.get("max_steps") != 2000
            or r6_env.get("reward_shaping") is not False or r6_env.get("reward_normalization") is not False
            or prior.get("cells") != protocol["cells"]
            or prior.get("frame_skip") != 4 or prior.get("max_steps") != 2000
            or prior.get("purpose") != "reused-TRAIN-engineering-diagnostic"):
        raise ValueError("schedule must reuse exactly four previously consumed TRAIN cells")
    sources = protocol["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != SOURCE_PATHS:
        raise ValueError("source map must pin exactly executing and environment files")
    _check_sources(root, sources)

    train = protocol["training"]
    if not isinstance(train, dict) or set(train) != {
        "seed", "decision_cap", "seed_steps", "pretrain_updates", "update_cap",
        "min_post_seed_updates", "updates_per_post_seed_decision", "max_steps",
        "replay_capacity", "batch_size", "device", "horizon", "discount", "rho",
        "model_size", "num_bins", "augmentation_pad", "episodic", "observation_shape",
    }:
        raise ValueError("training settings are not fully frozen")
    for key in ("seed", "decision_cap", "seed_steps", "pretrain_updates", "update_cap",
                "min_post_seed_updates", "updates_per_post_seed_decision", "max_steps",
                "replay_capacity", "batch_size", "horizon", "model_size", "num_bins", "augmentation_pad"):
        _positive(train[key], key, minimum=0 if key in ("seed", "min_post_seed_updates") else 1)
    if (train["max_steps"] != 2000 or train["seed_steps"] != max(1000, 5 * train["max_steps"])
            or train["pretrain_updates"] != train["seed_steps"]
            or train["updates_per_post_seed_decision"] != 1 or train["horizon"] != 3
            or train["discount"] != .995 or train["rho"] != .5 or train["model_size"] != 5
            or train["num_bins"] != 101 or train["augmentation_pad"] != 3
            or train["episodic"] is not True or train["observation_shape"] != [4, 64, 64]
            or train["device"] not in ("cpu", "cuda")
            or train["decision_cap"] != 14000 or train["replay_capacity"] != 14000
            or train["batch_size"] != 256 or train["min_post_seed_updates"] != 1000
            or train["update_cap"] != 14000):
        raise ValueError("pilot changes upstream seed/pretraining/architecture or fixed caps")
    evaluation = protocol["evaluation"]
    if (not isinstance(evaluation, dict) or set(evaluation) != {"seed", "repeats", "max_episodes", "max_steps"}
            or type(evaluation["seed"]) is not int or not 0 <= evaluation["seed"] <= 2**32 - 1
            or evaluation["repeats"] != 2 or evaluation["max_episodes"] != 16
            or _positive(evaluation["max_steps"], "evaluation max_steps") > 500):
        raise ValueError("evaluation must pair both policies on same roads with a capped episode horizon")
    resources = protocol["resources"]
    if not isinstance(resources, dict) or set(resources) != {
        "min_cgroup_available_bytes", "min_disk_available_bytes", "max_wall_seconds",
    }:
        raise ValueError("resource limits must be frozen")
    if (_positive(resources["min_cgroup_available_bytes"], "min_cgroup_available_bytes") < 16 * 1024**3
            or _positive(resources["min_disk_available_bytes"], "min_disk_available_bytes") < 5 * 1024**3
            or _positive(resources["max_wall_seconds"], "max_wall_seconds") > 7200):
        raise ValueError("resource floor/cap outside pilot bounds")
    if require_training_device and train["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("pinned CUDA device unavailable")
    return protocol


def _resources(run_dir: Path, protocol: dict) -> None:
    resources = protocol["resources"]
    max_path, current_path = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
    try:
        available = int(max_path.read_text().strip()) - int(current_path.read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError("measured cgroup memory unavailable") from exc
    if available < resources["min_cgroup_available_bytes"]:
        raise ValueError("insufficient measured raw cgroup headroom")
    disk = os.statvfs(run_dir.parent)
    if disk.f_bavail * disk.f_frsize < resources["min_disk_available_bytes"]:
        raise ValueError("insufficient run disk space")


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG restoration")
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _journal(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _cost(start: float, cpu_start: float) -> dict:
    # Host ru_maxrss can include inherited accounting in this environment;
    # /proc/self/statm is the actual process RSS at this receipt's timestamp.
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return {"elapsed_seconds": time.perf_counter() - start,
            "cpu_seconds": time.process_time() - cpu_start,
            "process_rss_bytes": resident_pages * os.sysconf("SC_PAGE_SIZE")}


def _components(protocol: dict):
    cfg = protocol["training"]
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True)).to(cfg["device"])
    learner = TDMPC2Learner(model, TDMPC2LearnerConfig(episode_length=cfg["max_steps"],
                                                      horizon=cfg["horizon"], batch_size=cfg["batch_size"]))
    replay = EpisodeReplay(capacity=cfg["replay_capacity"], horizon=cfg["horizon"],
                           seed=cfg["seed"], include_partial=False, bootstrap_on_truncation=True)
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=learner.discount, episodic=True))
    if not math.isclose(learner.discount, cfg["discount"], abs_tol=1e-12):
        raise ValueError("learner discount differs from source-pinned protocol")
    return learner, replay, planner


def _checkpoint(path: Path, learner, replay, decisions: int, updates: int,
                pretrain_updates: int, episodes: int, protocol_sha256: str, sources: dict,
                probe: dict | None, diagnostics: dict) -> str:
    if replay.active_length:
        raise ValueError("cannot checkpoint an in-flight episode")
    state = {"format": FORMAT, "protocol_sha256": protocol_sha256, "source_sha256": sources,
             "decisions": decisions, "updates": updates, "pretrain_updates": pretrain_updates,
             "episodes": episodes, "learner": learner.state_dict(), "optim": learner.optim.state_dict(),
             "pi_optim": learner.pi_optim.state_dict(), "replay": replay.state_dict(), "rng": _rng_state(),
             "probe": probe, "diagnostics": diagnostics}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return digest(path)


def _boundary_state(path: Path, journal: Path, protocol_sha256: str, sources: dict) -> dict:
    if not path.is_file() or not journal.is_file():
        raise ValueError("resume requires both boundary checkpoint and full ledger")
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[-1].get("event") != "checkpoint" or rows[-1].get("sha256") != digest(path):
        raise ValueError("non-boundary or partial ledger: exact resume prohibited")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("format") != FORMAT or state.get("protocol_sha256") != protocol_sha256
            or state.get("source_sha256") != sources
            or any(state.get(k) != rows[-1].get(k) for k in ("decisions", "updates", "pretrain_updates", "episodes"))):
        raise ValueError("checkpoint source/protocol/cursor mismatch")
    if (state["replay"].get("active") is not None
            or state["replay"]["next_episode_id"] != state["episodes"]
            or not 0 <= state["pretrain_updates"] <= state["updates"]
            or state["decisions"] < state["episodes"]):
        raise ValueError("checkpoint replay and training cursors disagree")
    return state


def _resume(path: Path, journal: Path, learner, replay, protocol_sha256: str, sources: dict) -> tuple:
    state = _boundary_state(path, journal, protocol_sha256, sources)
    learner.load_state_dict(state["learner"])
    learner.optim.load_state_dict(state["optim"])
    learner.pi_optim.load_state_dict(state["pi_optim"])
    replay.load_state_dict(state["replay"])
    if replay.active_length:
        raise ValueError("checkpoint contains in-flight episode")
    _restore_rng(state["rng"])
    return (state["decisions"], state["updates"], state["pretrain_updates"], state["episodes"],
            state["probe"], state["diagnostics"])


def _freeze_probe(replay, batch_size: int) -> dict:
    """Select once from completed TRAIN replay without changing sampler lineage."""
    previous = deepcopy(replay.rng.bit_generator.state)
    try:
        batch = replay.sample(batch_size)
        return {name: value.cpu().clone() for name, value in batch.items()}
    finally:
        replay.rng.bit_generator.state = previous


def _diagnostic(learner, probe: dict, seed: int) -> dict:
    """Read-only one-step within-TRAIN fit against frozen pixel-sequence anchors."""
    from torch.nn import functional as F

    saved = _rng_state()
    device = next(learner.model.parameters()).device
    try:
        _seed(seed)
        learner.model.eval()
        with torch.inference_mode():
            obs = probe["obs"][:2].to(device)
            action = probe["action"][0].to(device)
            reward = probe["reward"][0].to(device)
            terminal = probe["terminal"][0].to(device)
            z, next_z = learner.model.encode(obs[0], None), learner.model.encode(obs[1], None)
            predicted = learner.model.next(z, action, None)
            decoded_reward = two_hot_inv(learner.model.reward(z, action, None), learner.model.cfg)
            target = learner._td_target(next_z[None], reward[None], terminal[None])[0]
            q = learner.model.Q(z, action, None, return_type="avg")
            metrics = {
                "sample_count": int(action.shape[0]),
                "episode_ids": probe["episode_id"].tolist(),
                "start_steps": probe["start_step"].tolist(),
                "latent_mse": F.mse_loss(predicted, next_z).item(),
                "latent_frozen_anchor_mse": F.mse_loss(z, next_z).item(),
                "reward_mae": (decoded_reward - reward).abs().mean().item(),
                "constant_reward_mae": (reward.mean() - reward).abs().mean().item(),
                "q_td_residual_abs_mean": (q - target).abs().mean().item(),
                "q_scale": learner.q_scale.item(),
            }
            if learner.model.cfg.episodic:
                metrics["termination_bce"] = F.binary_cross_entropy(
                    learner.model.termination(predicted, None), terminal).item()
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise FloatingPointError("nonfinite frozen TRAIN probe")
        return metrics
    finally:
        _restore_rng(saved)


def _action(model, planner, pixels: np.ndarray, device: str, mode: str, t0: bool) -> np.ndarray:
    obs = torch.as_tensor(pixels[None], device=device)
    with torch.inference_mode():
        if mode == "mppi":
            action = planner.plan(obs, t0=t0, eval_mode=True)
        else:
            _, info = model.pi(model.encode(obs, None), None)
            action = info["mean"][0]
    return action.detach().cpu().numpy().astype(np.float32)


def evaluate(protocol: dict, model, path: Path, *, env_factory=make_training_env) -> dict:
    """Reset-paired, capped prior/MPPI development on SAME consumed TRAIN cells.

    The policies produce different trajectories. A 500-decision cap censors
    nonterminal outcomes; finish counts here are NOT a full-episode comparison.
    """
    cfg, ev = protocol["training"], protocol["evaluation"]
    _train_cells(protocol)
    if path.exists():
        raise ValueError("evaluation ledger already exists; never rerun consumed episodes")
    saved = _rng_state()
    device = next(model.parameters()).device
    records = []
    start, cpu_start = time.perf_counter(), time.process_time()
    wall_start = time.monotonic()
    try:
        model.to("cpu").eval()
        planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=cfg["discount"], episodic=True))
        env = env_factory(ev["max_steps"])
        try:
            for repeat in range(ev["repeats"]):
                for index, cell in enumerate(protocol["cells"]):
                    episode_seed = ev["seed"] + repeat * len(ROADS) + index
                    for mode in ("prior", "mppi"):
                        _resources(path.parent, protocol)
                        if time.monotonic() - wall_start >= protocol["resources"]["max_wall_seconds"]:
                            raise TimeoutError("paired evaluation wall budget exhausted")
                        _seed(episode_seed)
                        planner.reset()
                        _journal(path, {"event": "reset_intent", "mode": mode, "repeat": repeat,
                                        "episode_seed": episode_seed, **cell})
                        obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
                        pixels = model_observation(obs)
                        count, total, latencies = 0, 0.0, []
                        action_trace, native_trace = hashlib.sha256(), hashlib.sha256()
                        episode_start, episode_cpu = time.perf_counter(), time.process_time()
                        while count < ev["max_steps"]:
                            if time.monotonic() - wall_start >= protocol["resources"]["max_wall_seconds"]:
                                raise TimeoutError("partial paired evaluation: wall budget exhausted")
                            action_start = time.perf_counter()
                            action = _action(model, planner, pixels, "cpu", mode, count == 0)
                            latencies.append(time.perf_counter() - action_start)
                            native_action = environment_action(action)
                            action_trace.update(action.tobytes())
                            native_trace.update(native_action.tobytes())
                            obs, reward, terminated, truncated, info = env.step(native_action)
                            pixels = model_observation(obs)
                            total += float(reward)
                            count += 1
                            done, _ = episode_boundary(terminated, truncated, info)
                            if done:
                                break
                        else:
                            raise RuntimeError("evaluation wrapper did not terminate at its capped max_steps")
                        row = {"event": "episode", "mode": mode, "repeat": repeat, "episode_seed": episode_seed,
                               "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                               "decisions": count, "reward": total, "progress": float(info["progress"]),
                               "finished": bool(info.get("finished", False)), "terminated": bool(terminated),
                               "truncated": bool(truncated), "finish_time_s": info.get("finish_time_s"),
                               "censored": bool(count == ev["max_steps"] and not terminated
                                                and not info.get("finished", False)),
                               "evaluation_max_steps": ev["max_steps"],
                               "action_trace_sha256": action_trace.hexdigest(),
                               "native_action_trace_sha256": native_trace.hexdigest(),
                               "action_latency_first_s": latencies[0], "action_latency_mean_s": sum(latencies) / count,
                               "action_latency_max_s": max(latencies), **_cost(episode_start, episode_cpu)}
                        _journal(path, row)
                        records.append(row)
        finally:
            env.close()
    except BaseException as exc:
        _journal(path, {"event": "partial", "reason": type(exc).__name__,
                        "episodes": len(records), "decisions_in_current_episode": locals().get("count", 0),
                        **_cost(start, cpu_start)})
        raise
    finally:
        model.to(device)
        _restore_rng(saved)
    return {"episodes": len(records), "pairing": "same TRAIN road and reset RNG seed; trajectories differ",
            "capped_finish_comparison_valid": False, "evaluation_max_steps": ev["max_steps"],
            "per_mode": {mode: {
        "episodes": sum(row["mode"] == mode for row in records),
        "finishes": sum(row["finished"] for row in records if row["mode"] == mode),
        "censored": sum(row["censored"] for row in records if row["mode"] == mode),
        "mean_reward": float(np.mean([row["reward"] for row in records if row["mode"] == mode])),
        "mean_progress": float(np.mean([row["progress"] for row in records if row["mode"] == mode])),
    } for mode in ("prior", "mppi")}, **_cost(start, cpu_start)}


def evaluate_only(protocol: dict, protocol_sha256: str, run_dir: Path,
                  *, env_factory=make_training_env) -> dict:
    """Reload a verified model-only export in a separate CPU-only interpreter."""
    if torch.version.cuda is not None or not torch.__version__.startswith("2.1.0"):
        raise ValueError("CPU-only PyTorch 2.1.0 interpreter required for standalone evaluation")
    _resources(run_dir, protocol)
    _check_sources(ROOT, protocol["source_sha256"])
    checkpoint = run_dir / "boundary.pt"
    ledger = run_dir / "training.jsonl"
    if not checkpoint.is_file() or not ledger.is_file():
        raise ValueError("evaluation requires a complete training boundary receipt")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[-1].get("event") != "checkpoint" or rows[-1].get("sha256") != digest(checkpoint):
        raise ValueError("evaluation requires the final sealed episode boundary")
    export_path = run_dir / "cpu-model.pt"
    receipt_path = run_dir / "cpu-model-export.json"
    if not export_path.is_file() or not receipt_path.is_file():
        raise ValueError("CPU-only evaluation requires separately sealed model and sidecar")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("format") != "haic-tdmpc2-cpu-export-v1"
            or receipt.get("protocol_sha256") != protocol_sha256
            or receipt.get("source_sha256") != protocol["source_sha256"]
            or receipt.get("checkpoint_sha256") != digest(checkpoint)
            or receipt.get("sha256") != digest(export_path)
            or any(receipt.get(key) != rows[-1].get(key)
                   for key in ("decisions", "updates", "pretrain_updates", "episodes"))):
        raise ValueError("CPU export source, protocol or training checkpoint mismatch")
    _ready_for_export(protocol, {**receipt, "probe": receipt.get("probe_frozen"),
                                 "diagnostics": {"final": receipt.get("final_diagnostic")}
                                 if receipt.get("final_diagnostic") is not None else {}})
    export = torch.load(export_path, map_location="cpu", weights_only=False)
    if (not isinstance(export, dict) or export.get("format") != "haic-tdmpc2-cpu-model-v1"
            or export.get("protocol_sha256") != protocol_sha256
            or export.get("source_sha256") != protocol["source_sha256"]
            or export.get("checkpoint_sha256") != receipt["checkpoint_sha256"]
            or any(export.get(key) != receipt[key]
                   for key in ("decisions", "updates", "pretrain_updates", "episodes"))
            or not isinstance(export.get("model_state"), dict)):
        raise ValueError("CPU model export internal lineage differs from sealed sidecar")
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    model.load_state_dict(export["model_state"], strict=True)
    report = evaluate(protocol, model, run_dir / "train-evaluation.jsonl",
                      env_factory=env_factory)
    result = {"protocol_sha256": protocol_sha256, "checkpoint_sha256": digest(run_dir / "boundary.pt"),
              "cpu_export_sha256": digest(export_path), "source_sha256": protocol["source_sha256"],
              "reused_train_only": True, "cpu_only_torch": torch.__version__, **report}
    _atomic_json(run_dir / "evaluation-result.json", result)
    return result


def _ready_for_export(protocol: dict, state: dict) -> None:
    cfg = protocol["training"]
    if (state["decisions"] < cfg["decision_cap"] and state["updates"] < cfg["update_cap"]
            and cfg["decision_cap"] - state["decisions"] >= cfg["max_steps"]):
        raise ValueError("training is not at its frozen decision/update boundary")
    if (state["updates"] - state["pretrain_updates"] < cfg["min_post_seed_updates"]
            or not state["probe"] or "final" not in state["diagnostics"]):
        raise ValueError("TRAIN learning and frozen-probe gates have not completed")


def export_cpu(protocol: dict, protocol_sha256: str, run_dir: Path) -> dict:
    """Produce CPU model weights without rebuilding CUDA optimizer/replay/RNG."""
    _resources(run_dir, protocol)
    _check_sources(ROOT, protocol["source_sha256"])
    path = run_dir / "cpu-model.pt"
    if path.exists():
        raise ValueError("CPU model export already exists; never overwrite frozen weights")
    checkpoint = run_dir / "boundary.pt"
    state = _boundary_state(checkpoint, run_dir / "training.jsonl", protocol_sha256,
                            protocol["source_sha256"])
    _ready_for_export(protocol, state)
    model_state = {key.removeprefix("model."): value.cpu() for key, value in state["learner"].items()
                   if key.startswith("model.")}
    model = WorldModel(TDMPC2ModelConfig(obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    model.load_state_dict(model_state, strict=True)
    payload = {"format": "haic-tdmpc2-cpu-model-v1", "protocol_sha256": protocol_sha256,
               "source_sha256": protocol["source_sha256"], "checkpoint_sha256": digest(checkpoint),
               "decisions": state["decisions"], "updates": state["updates"],
               "pretrain_updates": state["pretrain_updates"], "episodes": state["episodes"],
               "model_state": model_state}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    receipt = {"format": "haic-tdmpc2-cpu-export-v1", "status": "model_only_cpu_export",
               "path": path.name, "sha256": digest(path), "checkpoint_sha256": digest(checkpoint),
               "protocol_sha256": protocol_sha256, "source_sha256": protocol["source_sha256"],
               "decisions": state["decisions"], "updates": state["updates"],
               "pretrain_updates": state["pretrain_updates"], "episodes": state["episodes"],
               "probe_frozen": True, "final_diagnostic": state["diagnostics"]["final"],
               "environment_resets": 0}
    _atomic_json(run_dir / "cpu-model-export.json", receipt)
    return receipt


def run(protocol: dict, protocol_sha256: str, run_dir: Path, *, resume: bool = False,
        env_factory=make_training_env, clock=time.monotonic) -> dict:
    """Consume only frozen TRAIN roads; journal every reset intent and decision."""
    _train_cells(protocol)
    cfg = protocol["training"]
    if resume:
        if not run_dir.is_dir():
            raise ValueError("resume run directory missing")
    else:
        run_dir.mkdir(exist_ok=False)
    ledger, checkpoint = run_dir / "training.jsonl", run_dir / "boundary.pt"
    _resources(run_dir, protocol)
    if not resume:
        _seed(cfg["seed"])
    learner, replay, planner = _components(protocol)
    if resume:
        decisions, updates, pretrain, episodes, probe, diagnostics = _resume(
            checkpoint, ledger, learner, replay, protocol_sha256, protocol["source_sha256"])
        if (decisions >= cfg["decision_cap"] or updates >= cfg["update_cap"]
                or cfg["decision_cap"] - decisions < cfg["max_steps"]):
            raise ValueError("budget already exhausted; resume prohibited")
        if decisions > cfg["seed_steps"] and pretrain != cfg["pretrain_updates"]:
            raise ValueError("cannot resume a trained-action phase before seed pretraining completes")
    else:
        if checkpoint.exists() or ledger.exists():
            raise ValueError("new run has existing ledger or checkpoint")
        decisions = updates = pretrain = episodes = 0
        probe, diagnostics = None, {}
        _journal(ledger, {"event": "start", "protocol_sha256": protocol_sha256,
                          "source_sha256": protocol["source_sha256"]})
    env = None
    start, cpu_start, wall_start = time.perf_counter(), time.process_time(), clock()
    status = "stopped"
    try:
        while decisions < cfg["decision_cap"] and updates < cfg["update_cap"]:
            if cfg["decision_cap"] - decisions < cfg["max_steps"]:
                status = "boundary_decision_budget"
                break
            if clock() - wall_start >= protocol["resources"]["max_wall_seconds"]:
                status = "boundary_wall_budget"
                break
            _resources(run_dir, protocol)
            _check_sources(ROOT, protocol["source_sha256"])
            cell = protocol["cells"][protocol["episode_schedule"][episodes % 4]]
            length, total, latencies, last_info = 0, 0.0, [], {}
            action_trace, native_trace = hashlib.sha256(), hashlib.sha256()
            episode_start, episode_cpu = time.perf_counter(), time.process_time()
            # An intent BEFORE reset makes a crash during reset fail closed on resume.
            _journal(ledger, {"event": "reset_intent", "episode": episodes, "decisions": decisions,
                              "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"]})
            if env is None:
                env = env_factory(cfg["max_steps"])
            obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
            pixels = model_observation(obs)
            replay.start_episode(pixels)
            planner.reset()
            _journal(ledger, {"event": "reset", "episode": episodes, "decisions": decisions,
                              "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"]})
            while length < cfg["max_steps"]:
                if clock() - wall_start >= protocol["resources"]["max_wall_seconds"]:
                    status = "partial_wall_budget"
                    break
                action_start = time.perf_counter()
                if decisions <= cfg["seed_steps"]:
                    action = np.random.uniform(-1, 1, size=3).astype(np.float32)
                else:
                    if pretrain != cfg["pretrain_updates"]:
                        raise RuntimeError("seed-boundary pretraining incomplete before first planned action")
                    obs_tensor = torch.as_tensor(pixels[None], device=cfg["device"])
                    with torch.inference_mode():
                        action = planner.plan(obs_tensor, t0=(length == 0)).detach().cpu().numpy().astype(np.float32)
                latencies.append(time.perf_counter() - action_start)
                native_action = environment_action(action)
                action_trace.update(action.tobytes())
                native_trace.update(native_action.tobytes())
                obs, reward, terminated, truncated, info = env.step(native_action)
                last_info = info
                following = model_observation(obs)
                done, terminal = episode_boundary(terminated, truncated, info)
                if not done:
                    replay.add_step(following, action, reward, terminated=False,
                                    truncated=False, terminal=False)
                total += float(reward)
                length += 1
                decision_index = decisions
                decisions += 1
                _journal(ledger, {"event": "step", "episode": episodes, "decisions": decisions,
                                  "reward": float(reward), "terminated": bool(terminated),
                                  "truncated": bool(truncated), "terminal": terminal})
                if decision_index == cfg["seed_steps"] and not replay.eligible_windows():
                    raise RuntimeError("no completed H+1 replay window at official seed/pretrain boundary")
                if replay.eligible_windows():
                    count = 0
                    if decision_index >= cfg["seed_steps"]:
                        if probe is None:
                            probe = _freeze_probe(replay, cfg["batch_size"])
                            diagnostics["initial"] = _diagnostic(learner, probe, cfg["seed"] + 101)
                            _journal(ledger, {"event": "diagnostic", "stage": "initial",
                                              "decisions": decisions, "updates": updates,
                                              "metrics": diagnostics["initial"]})
                        if pretrain < cfg["pretrain_updates"]:
                            count = min(cfg["pretrain_updates"] - pretrain, cfg["update_cap"] - updates)
                        elif decision_index > cfg["seed_steps"]:
                            count = min(cfg["updates_per_post_seed_decision"], cfg["update_cap"] - updates)
                    for _ in range(count):
                        if clock() - wall_start >= protocol["resources"]["max_wall_seconds"]:
                            status = "partial_wall_budget"
                            break
                        if updates % 256 == 0:
                            _resources(run_dir, protocol)
                        metrics = learner.update(replay)
                        if not all(math.isfinite(value) for value in metrics.values()):
                            raise FloatingPointError("nonfinite learner metrics")
                        updates += 1
                        if pretrain < cfg["pretrain_updates"]:
                            pretrain += 1
                        if updates <= 8 or pretrain >= cfg["pretrain_updates"] - 8 or updates % 1000 == 0:
                            _journal(ledger, {"event": "metric", "stage": "pretrain" if pretrain <= cfg["pretrain_updates"] and updates == pretrain else "post_seed",
                                              "decisions": decisions, "updates": updates,
                                              "metrics": metrics})
                    if pretrain == cfg["pretrain_updates"] and "post_pretrain" not in diagnostics and status != "partial_wall_budget":
                        diagnostics["post_pretrain"] = _diagnostic(learner, probe, cfg["seed"] + 101)
                        _journal(ledger, {"event": "diagnostic", "stage": "post_pretrain",
                                          "decisions": decisions, "updates": updates,
                                          "metrics": diagnostics["post_pretrain"]})
                    if count and status != "partial_wall_budget":
                        _journal(ledger, {"event": "updates", "episode": episodes, "decisions": decisions,
                                          "updates": updates, "pretrain_updates": pretrain, "metrics": metrics})
                # Upstream OnlineTrainer inserts a completed episode only at
                # the next reset, after this decision's learner updates.
                if done and status != "partial_wall_budget":
                    replay.add_step(following, action, reward, terminated=bool(terminated),
                                    truncated=bool(truncated), terminal=terminal)
                pixels = following
                if done or status == "partial_wall_budget":
                    break
            if replay.active_length or status == "partial_wall_budget":
                if status == "stopped":
                    status = "partial_missing_environment_boundary"
                _journal(ledger, {"event": "partial", "episode": episodes, "decisions": decisions,
                                  "length": length, "updates": updates, "pretrain_updates": pretrain,
                                  "reason": status,
                                  "reward": total, "progress": last_info.get("progress"),
                                  "finished": bool(last_info.get("finished", False)),
                                  "finish_time_s": last_info.get("finish_time_s"),
                                  "action_trace_sha256": action_trace.hexdigest(),
                                  "native_action_trace_sha256": native_trace.hexdigest(),
                                  "action_latency_first_s": latencies[0] if latencies else None,
                                  "action_latency_max_s": max(latencies) if latencies else None,
                                  **_cost(episode_start, episode_cpu)})
                break
            row = {"event": "episode", "episode": episodes, "track_id": cell["track_id"],
                   "geometry_seed": cell["geometry_seed"], "decisions": decisions,
                   "length": length, "reward": total, "progress": float(info["progress"]),
                   "finished": bool(info.get("finished", False)), "terminated": bool(terminated),
                   "truncated": bool(truncated), "terminal": terminal,
                   "finish_time_s": info.get("finish_time_s"), "updates": updates,
                   "pretrain_updates": pretrain, "action_latency_first_s": latencies[0],
                   "action_latency_mean_s": sum(latencies) / length, "action_latency_max_s": max(latencies),
                   "action_trace_sha256": action_trace.hexdigest(),
                   "native_action_trace_sha256": native_trace.hexdigest(),
                   **_cost(episode_start, episode_cpu)}
            _journal(ledger, row)
            episodes += 1
            if (probe is not None and "final" not in diagnostics
                    and (decisions >= cfg["decision_cap"] or updates >= cfg["update_cap"]
                         or cfg["decision_cap"] - decisions < cfg["max_steps"])):
                diagnostics["final"] = _diagnostic(learner, probe, cfg["seed"] + 101)
                _journal(ledger, {"event": "diagnostic", "stage": "final",
                                  "decisions": decisions, "updates": updates,
                                  "metrics": diagnostics["final"]})
            source = protocol["source_sha256"]
            _check_sources(ROOT, source)
            checksum = _checkpoint(checkpoint, learner, replay, decisions, updates, pretrain, episodes,
                                   protocol_sha256, source, probe, diagnostics)
            _journal(ledger, {"event": "checkpoint", "sha256": checksum,
                              "decisions": decisions, "updates": updates,
                              "pretrain_updates": pretrain, "episodes": episodes})
        else:
            status = "decision_or_update_cap"
    except BaseException as exc:
        _journal(ledger, {"event": "partial", "episode": episodes, "decisions": decisions,
                          "active_length": replay.active_length, "reason": type(exc).__name__,
                          "reward": locals().get("total", 0.0),
                          "progress": locals().get("last_info", {}).get("progress"),
                          "finished": bool(locals().get("last_info", {}).get("finished", False)),
                          "finish_time_s": locals().get("last_info", {}).get("finish_time_s"),
                          **_cost(locals().get("episode_start", start),
                                  locals().get("episode_cpu", cpu_start))})
        raise
    finally:
        if env is not None:
            env.close()
    result = {"status": status, "protocol_sha256": protocol_sha256, "decisions": decisions,
              "updates": updates, "pretrain_updates": pretrain, "episodes": episodes,
              "min_post_seed_updates_met": updates - pretrain >= cfg["min_post_seed_updates"],
              "diagnostics": diagnostics, "evaluation": None,
              "evaluation_status": "separate_cpu_export_and_evaluation_required",
              **_cost(start, cpu_start)}
    _atomic_json(run_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="validate only, never construct/reset environment")
    mode.add_argument("--run", action="store_true", help="start an explicitly frozen TRAIN-only pilot")
    mode.add_argument("--resume", action="store_true", help="resume ONLY the last fully journaled episode boundary")
    mode.add_argument("--export-cpu", action="store_true", help="export only CPU model weights from a finished checkpoint; no reset")
    mode.add_argument("--evaluate", action="store_true", help="paired reused-TRAIN eval of a completed boundary checkpoint")
    args = parser.parse_args()
    protocol = preflight(args.protocol, args.protocol_sha256, args.run_dir,
                         require_training_device=not (args.export_cpu or args.evaluate))
    run_dir = _inside(ROOT, args.run_dir, may_not_exist=True)
    _resources(run_dir, protocol)
    if args.preflight:
        print(json.dumps({"status": "preflight_only", "protocol_sha256": args.protocol_sha256,
                          "cells": protocol["cells"], "environment_resets": 0}, sort_keys=True))
    elif args.export_cpu:
        print(json.dumps(export_cpu(protocol, args.protocol_sha256, run_dir), sort_keys=True))
    elif args.evaluate:
        print(json.dumps(evaluate_only(protocol, args.protocol_sha256, run_dir), sort_keys=True))
    else:
        print(json.dumps(run(protocol, args.protocol_sha256, run_dir, resume=args.resume), sort_keys=True))


if __name__ == "__main__":
    main()
