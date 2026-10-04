"""Source-bound, from-scratch H3 reward-overshoot TRAIN runner on four consumed roads.

Run from the repository root with ``python -m scripts.train_tdmpc2_reward_overshoot``.
The original RAW100k runner and its algorithm modules are never modified or
injected. This runner has no resume mode and never evaluates a frozen policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import cast

import gymnasium as gym
import numpy as np
import torch

from haic.algorithms.tdmpc2.haic_env import (
    environment_action, episode_boundary, make_training_env, model_observation,
)
from haic.algorithms.tdmpc2.learner import TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.planner import PlannerConfig, TDMPC2Planner
from haic.algorithms.tdmpc2.reward_overshoot import OvershootLearner, OvershootReplay
from scripts import train_tdmpc2_long as base


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-reward-overshoot-train-v1"
RUN_DIR = "runs/tdmpc2-overshoot-20260929-v1"
BASE_PROTOCOL = ("experiments/tdmpc2-long-reused-train-v2.json",
                 "d4e25fe336998357fec0194f6423e2e4b63808896b2267a0a0f43b04ad5cc5ec")
BASE_RESULT = ("experiments/tdmpc2-long-reused-train-v2-100k-result.json",
               "bacf4c7e7a53794ecec7797746388695bac01096287f7aff09b5feafc4e3b515")
AUDIT = ("experiments/tdmpc2-reward-overshoot-target-audit-v1-result.json",
         "33db1781de0534c8870ba12a2ded095e9c45c96b844b1be9eaf4cdcdcb7579ff")
BENCHMARK = "runs/tdmpc2-reward-overshoot-throughput-20260929-v1.json"
VARIANT_SHA = "2a0c21abb909348d444e0418f84666119ff1032f56f5f1924a7ad7723d5d7b73"
SOURCE_PATHS = base.SOURCE_PATHS | {
    "haic/algorithms/tdmpc2/reward_overshoot.py", "scripts/train_tdmpc2_reward_overshoot.py",
}
OVERSHOOT = {"kind": "masked_imagined_reward_only", "base_horizon": 3,
             "reward_steps": [4, 5], "same_episode_suffix": True,
             "reward_target": "raw", "rho": 0.5, "reward_coef": 0.1,
             "normalization_horizon": 3, "aux_enabled": True}
STARTUP_GATE = {"min_cgroup_available_bytes": 24 * 1024**3,
                "min_disk_available_bytes": 24 * 1024**3,
                "min_cuda_reserve_bytes": 2 * 1024**3}
FIELDS = {"format", "purpose", "upstream_revision", "source_sha256", "baseline_training_protocol",
          "baseline_training_result", "replay_audit", "throughput_benchmark", "selection", "cells",
          "episode_schedule", "environment", "run_dir", "training", "checkpoint_targets",
          "resources", "startup_gate", "runtime", "seed_schedule", "reward_overshoot"}


def _exact(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        actual = cast(dict, actual)
        return actual.keys() == expected.keys() and all(
            _exact(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        actual = cast(list, actual)
        return len(actual) == len(expected) and all(
            _exact(a, b) for a, b in zip(actual, expected))
    return actual == expected


def runtime() -> dict:
    """Report imports and device without constructing or resetting the HAIC env."""
    cuda = torch.cuda.is_available()
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "gymnasium": gym.__version__, "torch_cuda": torch.version.cuda,
            "device": "cuda" if cuda else "cpu",
            "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if cuda else None,
            "cuda_capability": list(torch.cuda.get_device_capability()) if cuda else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads()}


def _sources(root: Path, sources: dict, baseline: dict) -> None:
    if (not isinstance(sources, dict) or set(sources) != SOURCE_PATHS
            or {name: sources[name] for name in baseline} != baseline
            or sources["haic/algorithms/tdmpc2/reward_overshoot.py"] != VARIANT_SHA):
        raise ValueError("overshoot source map must preserve every frozen RAW100k source byte")
    for name, pinned in sources.items():
        if base.digest(base._inside(root, name)) != base._sha(pinned):
            raise ValueError(f"source hash mismatch: {name}")


def _evidence(root: Path, protocol: dict) -> dict:
    prior = base._reference(root, protocol["baseline_training_protocol"], BASE_PROTOCOL[0])
    result = base._reference(root, protocol["baseline_training_result"], BASE_RESULT[0])
    audit = base._reference(root, protocol["replay_audit"], AUDIT[0])
    if (protocol["baseline_training_protocol"]["sha256"] != BASE_PROTOCOL[1]
            or protocol["baseline_training_result"]["sha256"] != BASE_RESULT[1]
            or protocol["replay_audit"]["sha256"] != AUDIT[1]):
        raise ValueError("original RAW100k or replay audit SHA differs")
    # Original preflight checks consumed allocation, exploration selection and all
    # original executable/environment SHA pins without constructing an environment.
    base.preflight(root / BASE_PROTOCOL[0], BASE_PROTOCOL[1], root / prior["run_dir"], root=root)
    if (result.get("format") != "haic-tdmpc2-long-reused-train-v2-final-model-result-v1"
            or result.get("training_protocol_sha256") != BASE_PROTOCOL[1]
            or result.get("run") != prior["run_dir"]
            or result.get("run_status") != "completed_boundary_at_least_100k"
            or (result.get("decisions"), result.get("updates"), result.get("episodes")) != (100354, 100354, 307)
            or result.get("checkpoint_target") != 100000
            or (result.get("seed_decisions"), result.get("pretrain_updates")) != (10000, 10000)):
        raise ValueError("original RAW100k result is not a complete selected source")
    for name, sha in ((result["run_result"], result["run_result_sha256"]),
                      (str(Path(result["run"]) / "training.jsonl"), result["training_ledger_sha256"]),
                      (str(Path(result["run"]) / "steps.jsonl"), result["step_ledger_sha256"]),
                      (result["checkpoint"], result["checkpoint_sha256"])):
        if base.digest(base._inside(root, name)) != base._sha(sha):
            raise ValueError(f"RAW100k artifact hash mismatch: {name}")
    receipt = base._json(base._inside(root, result["run_result"]))
    if (receipt.get("protocol_sha256") != BASE_PROTOCOL[1]
            or receipt.get("status") != result["run_status"]
            or (receipt.get("decisions"), receipt.get("updates"), receipt.get("episodes")) != (100354, 100354, 307)
            or receipt.get("training_ledger_sha256") != result["training_ledger_sha256"]
            or receipt.get("step_ledger_sha256") != result["step_ledger_sha256"]
            or receipt.get("checkpoints", [])[-1].get("sha256") != result["checkpoint_sha256"]):
        raise ValueError("original RAW100k receipt disagrees with frozen result")
    if (audit.get("format") != "haic-tdmpc2-reward-overshoot-target-audit-summary-v1"
            or audit.get("status") != "complete_replay_sufficiency_gate_passed_not_training_clearance"
            or audit.get("source_raw100k_protocol_sha256") != BASE_PROTOCOL[1]
            or audit.get("source_raw100k_result_sha256") != result["run_result_sha256"]
            or audit.get("source_raw100k_checkpoint_sha256") != result["checkpoint_sha256"]
            or audit.get("safety_evidence", {}).get("environment_resets") != 0
            or audit.get("predeclared_gate", {}).get("passed") is not True
            or audit.get("predeclared_gate", {}).get("minimum_extension_fraction") != 0.95
            or audit.get("predeclared_gate", {}).get("minimum_informative_roads") != 3
            or audit.get("predeclared_gate", {}).get("minimum_suffix_range_exclusive") != 0.1
            or audit.get("overall", {}).get("h3_eligible_starts") != 99740
            or audit.get("overall", {}).get("h5_extended_starts") != 99126
            or {row["geometry_seed"] for row in audit.get("per_consumed_train_road", [])} != set(base.ROADS)
            or any(row["two_step_discounted_reward_range"] <= 0.1
                   for row in audit["per_consumed_train_road"])):
        raise ValueError("pinned no-reset replay sufficiency audit failed")
    for name, sha in (("scripts/audit_tdmpc2_reward_overshoot_targets.py", audit["operator_sha256"]),
                      ("tests/test_audit_tdmpc2_reward_overshoot_targets.py", audit["operator_test_sha256"])):
        if base.digest(base._inside(root, name)) != base._sha(sha):
            raise ValueError(f"replay audit producer hash mismatch: {name}")
    return prior


def _startup_resources(output: Path, gate: dict, *, cuda_peak: int = 0) -> None:
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
    except (OSError, ValueError) as exc:
        raise ValueError("raw cgroup headroom unavailable") from exc
    if limit - current < gate["min_cgroup_available_bytes"]:
        raise ValueError("insufficient startup raw cgroup headroom")
    stat = os.statvfs(output.parent)
    if stat.f_bavail * stat.f_frsize < gate["min_disk_available_bytes"]:
        raise ValueError("insufficient startup raw disk headroom")
    if cuda_peak:
        available, _ = torch.cuda.mem_get_info()
        if available < cuda_peak + gate["min_cuda_reserve_bytes"]:
            raise ValueError("insufficient available CUDA memory for benchmark peak plus 2GiB reserve")


def _benchmark(root: Path, protocol: dict, device_runtime: dict) -> dict:
    """Strictly bind the separately measured synthetic CUDA update receipt."""
    ref = protocol["throughput_benchmark"]
    report = base._reference(root, ref, BENCHMARK)
    if ref["path"] != BENCHMARK:
        raise ValueError("throughput benchmark must use dedicated exclusive receipt")
    source_before = report.get("source_sha256_before")
    source_after = report.get("source_sha256_after")
    if (not isinstance(source_before, dict) or source_before != source_after
            or source_before.get("haic/algorithms/tdmpc2/reward_overshoot.py") != VARIANT_SHA
            or any(source_before.get(name) != protocol["source_sha256"][name]
                   for name in ("haic/algorithms/tdmpc2/learner.py", "haic/algorithms/tdmpc2/model.py",
                                "haic/algorithms/tdmpc2/replay.py"))):
        raise ValueError("CUDA benchmark did not pin the tested base and variant learner sources")
    for name, sha in source_before.items():
        if base.digest(base._inside(root, name)) != base._sha(sha):
            raise ValueError(f"CUDA benchmark source drift: {name}")
    inputs = report.get("input_sha256_before")
    if (not isinstance(inputs, dict) or set(inputs) != {"base", "variant"}
            or inputs != report.get("input_sha256_after")
            or any(base._sha(value) != value for value in inputs.values())
            or inputs["base"] != inputs["variant"]):
        raise ValueError("CUDA benchmark input lineage changed during measurement")
    rng_before, rng_after = report.get("sample_rng_sha256_before"), report.get("sample_rng_sha256_after")
    if (not isinstance(rng_before, dict) or not isinstance(rng_after, dict)
            or set(rng_before) != {"base", "variant"} or set(rng_after) != {"base", "variant"}
            or any(base._sha(value) != value for value in (*rng_before.values(), *rng_after.values()))
            or rng_before["base"] != rng_before["variant"]
            or rng_after["base"] != rng_after["variant"]):
        raise ValueError("CUDA benchmark replay sampler RNG parity failed")
    fixture = report.get("fixture")
    if (not isinstance(fixture, dict) or fixture.get("batch_size") != 256
            or fixture.get("horizon") != 3 or fixture.get("action_dim") != 3
            or fixture.get("obs_shape") != [4, 64, 64] or fixture.get("model_size") != 5
            or fixture.get("num_bins") != 101
            or fixture.get("synthetic_only") is not True
            or type(fixture.get("valid_step4")) is not int
            or type(fixture.get("valid_step5")) is not int
            or not 256 >= fixture["valid_step4"] >= fixture["valid_step5"] >= 244
            or type(fixture.get("trainable_parameters")) is not int
            or not 4_000_000 <= fixture["trainable_parameters"] <= 8_000_000
            or any(base._sha(fixture.get(key)) != fixture.get(key)
                   for key in ("proof_h3_sample_sha256", "proof_extended_sample_sha256"))):
        raise ValueError("CUDA benchmark fixture differs from full model/active suffix")
    timing = report.get("timing")
    if (not isinstance(timing, dict) or type(timing.get("warmup_updates_each")) is not int
            or timing["warmup_updates_each"] < 32
            or type(timing.get("timed_updates_each")) is not int
            or timing["timed_updates_each"] < 128
            or timing.get("scope") != "full_update_with_replay_sample"
            or timing.get("block_order") != ["base", "variant", "variant", "base"] * 2
            or timing.get("synchronization") != "torch.cuda.synchronize per update"):
        raise ValueError("CUDA benchmark update counts or synchronization failed")
    for arm in ("base", "variant"):
        values = timing.get(arm, {})
        seconds = values.get("update_wall_seconds")
        if (not isinstance(seconds, list) or len(seconds) != timing["timed_updates_each"]
                or any(type(v) is not float or not math.isfinite(v) or v <= 0 for v in seconds)
                or not math.isclose(values.get("mean_seconds", math.inf), math.fsum(seconds) / len(seconds),
                                    rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(values.get("median_seconds", math.inf), statistics.median(seconds),
                                    rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("CUDA benchmark timed updates malformed or inconsistent")
    resources = report.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("CUDA benchmark raw resource fields missing")
    peak = resources.get("max_cuda_allocated_bytes")
    if type(peak) is not int or peak <= 0:
        raise ValueError("CUDA benchmark peak allocation must be a positive byte count")
    if (any(
            type(resources.get(key)) is not int or resources[key] <= 0
            for key in ("cgroup_available_bytes_before", "cgroup_available_bytes_after",
                        "disk_available_bytes_before", "disk_available_bytes_after",
                        "cuda_free_bytes_before", "cuda_free_bytes_after", "cuda_total_bytes",
                        "max_cuda_allocated_bytes"))
            or (resources.get("min_cgroup_available_bytes"), resources.get("min_disk_available_bytes"),
                resources.get("gpu_reserve_bytes")) != (24 * 1024**3, 24 * 1024**3, 2 * 1024**3)
            or min(resources["cgroup_available_bytes_before"], resources["cgroup_available_bytes_after"]) < 24 * 1024**3
            or min(resources["disk_available_bytes_before"], resources["disk_available_bytes_after"]) < 24 * 1024**3
            or min(resources["cuda_free_bytes_before"], resources["cuda_free_bytes_after"]) < peak + 2 * 1024**3):
        raise ValueError("CUDA benchmark raw cgroup/disk/VRAM reserve failed")
    forecast = report.get("forecast")
    baseline = 17459.373508695047
    seconds = baseline + 100354 * max(0., timing["variant"]["mean_seconds"] - timing["base"]["mean_seconds"])
    if (not isinstance(forecast, dict)
            or forecast.get("baseline_elapsed_seconds") != baseline
            or forecast.get("baseline_updates") != 100354
            or (forecast.get("threshold_seconds_exclusive"), forecast.get("wall_cap_seconds")) != (20600, 21600)
            or forecast.get("passed") is not True or not seconds < 20600
            or not math.isclose(forecast.get("forecast_seconds", math.inf), seconds, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError("CUDA benchmark forecasts more than permitted 100k wall budget")
    if not math.isclose(forecast.get("positive_update_delta_seconds", math.inf),
                        max(0., timing["variant"]["mean_seconds"] - timing["base"]["mean_seconds"]),
                        rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("CUDA benchmark positive update delta differs")
    observed = report.get("runtime")
    if (set(report) != {"format", "status", "environment_resets", "real_replay_reads",
                        "expected_script_sha256", "output_path", "source_sha256_before", "source_sha256_after",
                        "input_sha256_before", "input_sha256_after", "sample_rng_sha256_before",
                        "sample_rng_sha256_after", "runtime", "fixture", "timing", "resources", "forecast",
                        "body_sha256"} or report.get("output_path") != BENCHMARK
            or report.get("expected_script_sha256") != source_before.get(
                "scripts/benchmark_tdmpc2_reward_overshoot.py")
            or report.get("format") != "haic-tdmpc2-reward-overshoot-throughput-v1"
            or report.get("status") != "PASS" or report.get("environment_resets") != 0
            or report.get("real_replay_reads") != 0 or not isinstance(observed, dict)
            or observed.get("torch") != device_runtime["torch"]
            or observed.get("torch_cuda") != device_runtime["torch_cuda"]
            or observed.get("device") != "cuda:0"
            or observed.get("cuda_name") != device_runtime["gpu_name"]
            or observed.get("cuda_visible_devices") != device_runtime["cuda_visible_devices"]
            or observed.get("torch_num_threads") != device_runtime["torch_num_threads"]
            or observed.get("torch_num_interop_threads") != device_runtime["torch_num_interop_threads"]):
        raise ValueError("CUDA benchmark status or device/runtime differs")
    body = {key: value for key, value in report.items() if key != "body_sha256"}
    actual_hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                           allow_nan=False).encode("utf-8")).hexdigest()
    if report.get("body_sha256") != actual_hash:
        raise ValueError("CUDA benchmark canonical body SHA differs")
    return report


def preflight(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT) -> dict:
    """Read-only SHA/schema/source/audit/benchmark/device/resource checks; zero resets."""
    root = root.resolve(strict=True)
    path = base._inside(root, protocol_path)
    if (path.parent != root / "experiments" or path.suffix != ".json"
            or base.digest(path) != base._sha(protocol_sha256)):
        raise ValueError("frozen overshoot experiments/*.json protocol SHA mismatch")
    p = base._json(path)
    output = base._inside(root, run_dir, required=False)
    if (set(p) != FIELDS or p["format"] != FORMAT
            or p["purpose"] != "consumed-TRAIN-development" or p["upstream_revision"] != base.UPSTREAM
            or p["run_dir"] != RUN_DIR or output != root / RUN_DIR or output.exists()
            or p["seed_schedule"] != base.SEED_SCHEDULE
            or not _exact(p["selection"], {"arm": "independent_3d", "action_dim": 3})
            or not _exact(p["cells"], base.CELLS)
            or not _exact(p["episode_schedule"], [0, 1, 2, 3])
            or not _exact(p["environment"], base.ENVIRONMENT)
            or not _exact(p["checkpoint_targets"], base.TARGETS)
            or not _exact(p["startup_gate"], STARTUP_GATE)
            or not _exact(p["reward_overshoot"], OVERSHOOT)):
        raise ValueError("invalid isolated overshoot protocol, cells, objective or exclusive output path")
    prior = _evidence(root, p)
    if (not _exact(p["training"], prior["training"])
            or not _exact(p["resources"], prior["resources"])
            or p["resources"]["max_wall_seconds"] != 21600
            or p["training"]["device"] != "cuda"
            or not _exact(p["selection"], prior["selection"])):
        raise ValueError("original RAW100k 3D/H3 seed, model, update, resource or wall caps changed")
    _sources(root, p["source_sha256"], prior["source_sha256"])
    live = runtime()
    if not _exact(p["runtime"], live) or live["device"] != "cuda":
        raise ValueError("pinned CUDA/Python/Torch runtime differs or CUDA unavailable")
    report = _benchmark(root, p, live)
    _startup_resources(output, p["startup_gate"], cuda_peak=report["resources"]["max_cuda_allocated_bytes"])
    base._resources(output, p)
    return p


def _recheck(root: Path, path: Path, sha: str, protocol: dict, baseline: dict) -> None:
    if base.digest(base._inside(root, path)) != sha:
        raise ValueError("overshoot protocol changed during run")
    for ref in ("baseline_training_protocol", "baseline_training_result", "replay_audit", "throughput_benchmark"):
        item = protocol[ref]
        if base.digest(base._inside(root, item["path"])) != item["sha256"]:
            raise ValueError(f"pinned {ref} changed during run")
    _sources(root, protocol["source_sha256"], baseline["source_sha256"])
    if not _exact(protocol["runtime"], runtime()):
        raise ValueError("runtime changed during run")


def _components(p: dict):
    cfg = p["training"]
    model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)},
                                         episodic=True)).to(cfg["device"])
    learner = OvershootLearner(model, TDMPC2LearnerConfig(episode_length=cfg["max_steps"],
                                                         horizon=3, batch_size=256), aux_enabled=True)
    replay = OvershootReplay(capacity=cfg["replay_capacity"], horizon=3, action_dim=3,
                             seed=cfg["seed"], include_partial=False,
                             bootstrap_on_truncation=True, augmentation_pad=3)
    planner = TDMPC2Planner(model, PlannerConfig(action_dim=3, discount=learner.discount,
                                                 episodic=True))
    if not math.isclose(learner.discount, cfg["discount"], abs_tol=1e-12):
        raise ValueError("overshoot learner discount differs from original RAW100k")
    return learner, replay, planner


def _checkpoint(path: Path, learner, replay, probe: dict, decisions: int, updates: int,
                episodes: int, sha: str, p: dict, target: int, step_ledger_sha256: str,
                training_ledger_sha256_before_checkpoint: str) -> str:
    if replay.active_length or path.exists():
        raise ValueError("overshoot checkpoint requires a new whole-episode boundary")
    state = {"format": FORMAT, "protocol_sha256": sha, "source_sha256": p["source_sha256"],
             "baseline_training_protocol": p["baseline_training_protocol"],
             "baseline_training_result": p["baseline_training_result"],
             "replay_audit": p["replay_audit"], "throughput_benchmark": p["throughput_benchmark"],
             "reward_overshoot": p["reward_overshoot"], "target": target, "decisions": decisions,
             "updates": updates, "episodes": episodes, "action_dim": replay.action_dim,
             "step_ledger_sha256": step_ledger_sha256,
             "training_ledger_sha256_before_checkpoint": training_ledger_sha256_before_checkpoint,
             "learner": learner.state_dict(), "optim": learner.optim.state_dict(),
             "pi_optim": learner.pi_optim.state_dict(), "replay": replay.state_dict(),
             "probe": probe, "rng": base._rng_state(), "resume_supported": False}
    with path.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return base.digest(path)


def run(protocol_path: Path, protocol_sha256: str, run_dir: Path, *, root: Path = ROOT,
        env_factory=make_training_env, clock=time.monotonic) -> dict:
    """No environment construction until complete preflight; never resume a partial run."""
    p = preflight(protocol_path, protocol_sha256, run_dir, root=root)
    root = root.resolve(strict=True)
    output = base._inside(root, run_dir, required=False)
    output.mkdir(exist_ok=False)
    ledger, steps = output / "training.jsonl", output / "steps.jsonl"
    decisions = updates = episodes = environment_steps_applied = reset_intents = 0
    started = started_cpu = None
    env = replay = None
    pending_step = None
    try:
        cfg = p["training"]
        started, started_cpu = clock(), time.process_time()
        base._journal(ledger, {"event": "start", "protocol_sha256": protocol_sha256,
                               "source_sha256": p["source_sha256"], "runtime": p["runtime"],
                               "reward_overshoot": OVERSHOOT, "seed_schedule": base.SEED_SCHEDULE,
                               "resume_supported": False})
        prior = base._json(base._inside(root, BASE_PROTOCOL[0]))
        benchmark_peak = _benchmark(root, p, p["runtime"])["resources"]["max_cuda_allocated_bytes"]
        _startup_resources(output, p["startup_gate"], cuda_peak=benchmark_peak)
        _recheck(root, protocol_path, protocol_sha256, p, prior)
        base._seed(cfg["seed"])
        learner, replay, planner = _components(p)
        probe = latest_delta = None
        rolling_sum: dict[str, float] = {}
        rolling_count = delta_count = 0
        delta_sum = np.zeros(3, dtype=np.float64)
        delta_abs_sum = np.zeros(3, dtype=np.float64)
        native_delta_abs_sum = np.zeros(3, dtype=np.float64)
        delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
        interval_episodes: list[dict] = []
        checkpoints: list[dict] = []
        while decisions < base.TARGETS[-1]:
            if cfg["decision_cap"] - decisions < cfg["max_steps"]:
                raise RuntimeError("decision cap prevents required 100k whole-episode target")
            if clock() - started >= p["resources"]["max_wall_seconds"]:
                raise TimeoutError("wall cap expired before final TRAIN milestone")
            _startup_resources(output, p["startup_gate"], cuda_peak=benchmark_peak)
            base._resources(output, p)
            _recheck(root, protocol_path, protocol_sha256, p, prior)
            cell = p["cells"][p["episode_schedule"][episodes % 4]]
            reset_intents += 1
            base._journal(ledger, {"event": "reset_intent", "episode": episodes,
                                   "decisions": decisions, **cell})
            if env is None:
                env = env_factory(cfg["max_steps"])
            obs, _ = env.reset(seed=cell["geometry_seed"], options={"track_id": cell["track_id"]})
            if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                    or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                raise ValueError("reset did not honor pinned consumed TRAIN road")
            pixels = model_observation(obs)
            replay.start_episode(pixels)
            planner.reset()
            base._journal(ledger, {"event": "reset", "episode": episodes, "decisions": decisions, **cell})
            length = 0
            total = progress = damage = 0.0
            done = terminated = truncated = terminal = False
            info: dict = {}
            action_trace, native_trace = hashlib.sha256(), hashlib.sha256()
            while length < cfg["max_steps"]:
                if clock() - started >= p["resources"]["max_wall_seconds"]:
                    raise TimeoutError("partial TRAIN episode: wall cap exhausted")
                index = decisions
                if index < cfg["seed_steps"]:
                    action = np.random.uniform(-1, 1, size=3).astype(np.float32)
                else:
                    if updates < cfg["pretrain_updates"]:
                        raise RuntimeError("seed pretraining incomplete before first planned action")
                    action, latest_delta = base._planned_action(
                        learner.model, planner, pixels, cfg["device"], t0=(length == 0))
                native = environment_action(action)
                if latest_delta is not None and index >= cfg["seed_steps"]:
                    delta = np.asarray(latest_delta["delta"], dtype=np.float64)
                    prior_native = environment_action(np.asarray(latest_delta["prior_mean"], dtype=np.float32))
                    mean_native = environment_action(np.asarray(
                        latest_delta["mppi_weighted_elite_mean"], dtype=np.float32))
                    delta_count += 1
                    delta_sum += delta
                    delta_abs_sum += np.abs(delta)
                    native_delta_abs_sum += np.abs(mean_native.astype(np.float64) - prior_native)
                    delta_l2_sum += latest_delta["delta_l2"]
                    delta_l2_max = max(delta_l2_max, latest_delta["delta_l2"])
                    applied_delta_l2_sum += latest_delta["applied_delta_l2"]
                action_trace.update(action.tobytes())
                native_trace.update(native.tobytes())
                pending_step = {"episode": episodes, "decision": decisions + 1, **cell,
                                "action_f32_hex": action.tobytes().hex(),
                                "native_action_f32_hex": native.tobytes().hex()}
                obs, reward, terminated, truncated, info = env.step(native)
                environment_steps_applied += 1
                pending_step.update(raw_reward_repr=repr(reward),
                                    terminated=bool(terminated), truncated=bool(truncated),
                                    finished=bool(info.get("finished", False)))
                if (getattr(env.unwrapped, "track_id", None) != cell["track_id"]
                        or getattr(env.unwrapped, "track_seed", None) != cell["geometry_seed"]):
                    raise ValueError("TRAIN road changed during episode")
                following = model_observation(obs)
                done, terminal = episode_boundary(terminated, truncated, info)
                if (truncated and not info.get("finished", False) and not terminated
                        and length + 1 != cfg["max_steps"]):
                    raise ValueError("ordinary TRAIN timeout occurred before pinned max_steps")
                finish_time = info.get("finish_time_s")
                if (bool(info.get("finished", False)) != (finish_time is not None)
                        or (finish_time is not None and (
                            not isinstance(finish_time, (float, int)) or not math.isfinite(finish_time)))):
                    raise ValueError("TRAIN finished flag and finish time disagree")
                progress, damage = float(info["progress"]), float(info["damage"])
                if not all(math.isfinite(value) for value in (float(reward), progress, damage)):
                    raise ValueError("nonfinite RAW TRAIN reward/progress/damage")
                if not done:
                    replay.add_step(following, action, reward, terminal=False)
                decisions += 1
                length += 1
                total += float(reward)
                base._journal(steps, {**pending_step, "reward": float(reward),
                                      "terminated": bool(terminated), "truncated": bool(truncated),
                                      "terminal": terminal})
                pending_step = None
                if index == cfg["seed_steps"] - 1 and not replay.eligible_windows():
                    raise RuntimeError("no completed H3 replay window at seed boundary")
                count = cfg["pretrain_updates"] if index == cfg["seed_steps"] - 1 else int(index >= cfg["seed_steps"])
                if count and probe is None:
                    probe = base._freeze_probe(replay, cfg["batch_size"])
                for _ in range(count):
                    if updates >= cfg["update_cap"]:
                        raise RuntimeError("update cap reached before 100k boundary")
                    if clock() - started >= p["resources"]["max_wall_seconds"]:
                        raise TimeoutError("partial learner updates: wall cap exhausted")
                    if updates % 256 == 0:
                        base._resources(output, p)
                    metrics = learner.update(replay)
                    if ("overshoot_reward_loss" not in metrics
                            or any(not math.isfinite(float(value)) for value in metrics.values())):
                        raise FloatingPointError("nonfinite or missing overshoot learner metric")
                    updates += 1
                    rolling_count += 1
                    for key, value in metrics.items():
                        rolling_sum[key] = rolling_sum.get(key, 0.0) + float(value)
                # Original H3 ordering: do not admit the closing step until its updates finish.
                if done:
                    replay.add_step(following, action, reward, terminated=bool(terminated),
                                    truncated=bool(truncated), terminal=terminal)
                pixels = following
                if done:
                    break
            if replay.active_length or not done:
                raise RuntimeError("environment failed to close its full TRAIN episode")
            row = {"event": "episode", "episode": episodes, **cell, "decisions": decisions,
                   "updates": updates, "length": length, "return": total, "progress": progress,
                   "damage": damage, "finished": bool(info.get("finished", False)),
                   "terminated": bool(terminated), "truncated": bool(truncated), "terminal": terminal,
                   "finish_time_s": info.get("finish_time_s"),
                   "action_trace_sha256": action_trace.hexdigest(),
                   "native_action_trace_sha256": native_trace.hexdigest()}
            base._journal(ledger, row)
            interval_episodes.append({key: row[key] for key in (
                "episode", "track_id", "geometry_seed", "decisions", "length", "return",
                "progress", "damage", "finished")})
            episodes += 1
            if len(checkpoints) < len(base.TARGETS) and decisions >= base.TARGETS[len(checkpoints)]:
                target = base.TARGETS[len(checkpoints)]
                if probe is None or rolling_count == 0 or latest_delta is None or delta_count == 0:
                    raise RuntimeError("checkpoint lacks frozen probe, updates or planned action")
                base._resources(output, p)
                _recheck(root, protocol_path, protocol_sha256, p, prior)
                diagnostic = base._diagnostic(learner, probe, cfg["seed"] + 101)
                step_hash = base.digest(steps)
                training_hash = base.digest(ledger)
                report = {"target": target, "decisions": decisions, "updates": updates,
                          "episodes": episodes, "rolling_update_count": rolling_count,
                          "rolling_update_means": {key: value / rolling_count for key, value in rolling_sum.items()},
                          "train_episodes_since_previous": interval_episodes,
                          "q_scale": float(learner.q_scale.item()),
                          "policy_entropy_rolling_mean": rolling_sum["pi_entropy"] / rolling_count,
                          "resource_usage": base._cost(clock, started, started_cpu),
                          "same_observation_prior_vs_mppi_interval": {
                              "planned_actions": delta_count,
                              "mean_signed_model_delta": (delta_sum / delta_count).tolist(),
                              "mean_absolute_model_delta": (delta_abs_sum / delta_count).tolist(),
                              "mean_absolute_native_delta": (native_delta_abs_sum / delta_count).tolist(),
                              "mean_model_delta_l2": delta_l2_sum / delta_count,
                              "max_model_delta_l2": delta_l2_max,
                              "mean_applied_exploration_noised_delta_l2": applied_delta_l2_sum / delta_count},
                          "same_observation_prior_vs_mppi": latest_delta,
                          "frozen_train_probe": diagnostic, "step_ledger_sha256": step_hash,
                          "training_ledger_sha256_before_checkpoint": training_hash}
                name = f"checkpoint-at-least-{target:06d}-step-{decisions:06d}.pt"
                checksum = _checkpoint(output / name, learner, replay, probe, decisions, updates,
                                       episodes, protocol_sha256, p, target, step_hash, training_hash)
                report.update(path=name, sha256=checksum)
                base._journal(ledger, {"event": "checkpoint", **report})
                checkpoints.append(report)
                rolling_sum, rolling_count = {}, 0
                interval_episodes = []
                delta_count = 0
                delta_sum.fill(0)
                delta_abs_sum.fill(0)
                native_delta_abs_sum.fill(0)
                delta_l2_sum = delta_l2_max = applied_delta_l2_sum = 0.0
        _recheck(root, protocol_path, protocol_sha256, p, prior)
        _evidence(root, p)
        result = {"status": "completed_boundary_at_least_100k", "format": FORMAT,
                  "protocol_sha256": protocol_sha256, "source_sha256": p["source_sha256"],
                  "baseline_training_result": p["baseline_training_result"], "replay_audit": p["replay_audit"],
                  "throughput_benchmark": p["throughput_benchmark"], "reward_overshoot": OVERSHOOT,
                  "decisions": decisions, "updates": updates, "pretrain_updates": cfg["pretrain_updates"],
                  "episodes": episodes, "action_dim": 3, "seed_schedule": base.SEED_SCHEDULE,
                  "checkpoints": checkpoints, "runtime": p["runtime"],
                  "resource_usage": base._cost(clock, started, started_cpu),
                  "training_ledger_sha256": base.digest(ledger), "step_ledger_sha256": base.digest(steps),
                  "reused_train_only": True, "evaluation": None, "resume_supported": False}
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
    except BaseException as exc:
        cost = None
        if started is not None and started_cpu is not None:
            try:
                cost = base._cost(clock, started, started_cpu)
            except BaseException:
                pass
        partial = {"event": "partial", "reason": type(exc).__name__,
                   "protocol_sha256": protocol_sha256, "source_sha256": p["source_sha256"],
                   "reward_overshoot": OVERSHOOT, "episodes": episodes, "decisions": decisions,
                   "updates": updates, "reset_intents": reset_intents,
                   "environment_steps_applied": environment_steps_applied, "pending_step": pending_step,
                   "resource_usage": cost, "active_length": replay.active_length if replay is not None else None,
                   "resume_supported": False}
        try:
            base._journal(ledger, partial)
        except BaseException:
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(partial, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        raise
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-runtime", action="store_true", help="show device/runtime only; no protocol or env")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--run-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="read-only strict gate; zero environment resets")
    mode.add_argument("--run", action="store_true", help="new from-scratch TRAIN run; never resume")
    args = parser.parse_args()
    if args.print_runtime:
        if args.preflight or args.run or args.protocol or args.protocol_sha256 or args.run_dir:
            parser.error("--print-runtime takes no other arguments")
        print(json.dumps(runtime(), sort_keys=True))
        return
    if not (args.protocol and args.protocol_sha256 and args.run_dir) or not (args.preflight or args.run):
        parser.error("require --protocol, --protocol-sha256, --run-dir and exactly one mode")
    if args.preflight:
        p = preflight(args.protocol, args.protocol_sha256, args.run_dir)
        print(json.dumps({"status": "preflight_only", "protocol_sha256": args.protocol_sha256,
                          "cells": p["cells"], "reward_overshoot": OVERSHOOT,
                          "throughput_benchmark": p["throughput_benchmark"],
                          "environment_resets": 0}, sort_keys=True))
    else:
        print(json.dumps(run(args.protocol, args.protocol_sha256, args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
