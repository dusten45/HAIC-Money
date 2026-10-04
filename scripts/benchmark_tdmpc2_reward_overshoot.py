"""No-environment, source-bound CUDA optimizer throughput screen for H3 reward overshooting.

Run from the repository root with ``python -m scripts.benchmark_tdmpc2_reward_overshoot
--expected-script-sha256 <sha256>`` only when the GPU is available and idle. This
uses generated pixels/actions/rewards, never a HAIC environment or archived replay.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from haic.algorithms.tdmpc2.learner import TDMPC2Learner, TDMPC2LearnerConfig
from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
from haic.algorithms.tdmpc2.replay import EpisodeReplay
from haic.algorithms.tdmpc2.reward_overshoot import OvershootLearner, OvershootReplay


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-tdmpc2-reward-overshoot-throughput-v1"
SOURCE_SHA256 = {
    "haic/algorithms/tdmpc2/learner.py": "e0f1c814a5162b6cf34383a675ec10e768f3a046ac82c6500400cdfed23d2c9a",
    "haic/algorithms/tdmpc2/model.py": "a121d8688e5ab954abe2e6468ca84760e035f15b49004b695fc465696eb90c65",
    "haic/algorithms/tdmpc2/replay.py": "bef63630556a1adbfb26278190c2549f4415ab1c90b5775d920187cfc4576254",
    "haic/algorithms/tdmpc2/reward_overshoot.py": "2a0c21abb909348d444e0418f84666119ff1032f56f5f1924a7ad7723d5d7b73",
}
SELF = "scripts/benchmark_tdmpc2_reward_overshoot.py"
SEED = 20260929
BATCH_SIZE = 256
HORIZON = 3
WARMUP = 32
TIMED = 128
GIB = 1024**3
MIN_CGROUP = 24 * GIB
MIN_DISK = 24 * GIB
GPU_RESERVE = 2 * GIB
MIB = 1024**2
BASELINE_GPU_LIMIT_MIB = 512
BASELINE_DRIFT_LIMIT_MIB = 128
BASELINE_SECONDS = Decimal("17459.373508695047")
BASELINE_UPDATES = 100354
FORECAST_LIMIT = 20600
WALL_CAP = 21600


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root: Path) -> dict[str, str]:
    return {name: _sha(root / name) for name in sorted({*SOURCE_SHA256, SELF})}


def _check_sources(before: dict[str, str], expected_script_sha256: str, root: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_script_sha256):
        raise ValueError("expected script SHA must be lowercase SHA-256")
    if before != {**SOURCE_SHA256, SELF: expected_script_sha256}:
        raise ValueError("pinned TD/source or benchmark script SHA mismatch")
    for module_name, path in (
        (TDMPC2Learner.__module__, "haic/algorithms/tdmpc2/learner.py"),
        (WorldModel.__module__, "haic/algorithms/tdmpc2/model.py"),
        (EpisodeReplay.__module__, "haic/algorithms/tdmpc2/replay.py"),
        (OvershootLearner.__module__, "haic/algorithms/tdmpc2/reward_overshoot.py"),
    ):
        module_file = sys.modules[module_name].__file__
        if module_file is None or Path(module_file).resolve() != (root / path).resolve():
            raise ValueError("imported TD module differs from pinned source path")


def _raw_resources(device: torch.device, root: Path) -> dict[str, int]:
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        if not 0 <= used <= limit:
            raise ValueError("invalid raw cgroup memory readings")
        disk = os.statvfs(root / "runs")
        disk_available = disk.f_bavail * disk.f_frsize
        free, total = torch.cuda.mem_get_info(device)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("raw cgroup/disk/CUDA telemetry unavailable") from exc
    if not (0 < disk_available and 0 < free <= total):
        raise ValueError("invalid disk or CUDA free-memory telemetry")
    return {"cgroup_available_bytes": limit - used, "disk_available_bytes": disk_available,
            "cuda_free_bytes": free, "cuda_total_bytes": total}


def _nvidia_smi(query: str) -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("nvidia-smi contention telemetry unavailable") from exc
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gpu_snapshot(*, require_idle: bool = True) -> dict:
    """NVIDIA reports host PIDs, which need not equal the container Python PID."""
    lines = _nvidia_smi("compute-apps=pid,used_gpu_memory")
    if lines == ["No running processes found"]:
        lines = []
    processes = {}
    try:
        for line in lines:
            fields = next(csv.reader([line], skipinitialspace=True))
            if len(fields) != 2 or any(not re.fullmatch(r"[0-9]+", item) for item in fields):
                raise ValueError("invalid NVIDIA process or memory reading")
            pid, memory_mib = (int(item) for item in fields)
            if pid <= 0 or memory_mib <= 0 or pid in processes:
                raise ValueError("duplicate or nonresident NVIDIA process")
            processes[pid] = memory_mib
    except ValueError as exc:
        raise ValueError("unrecognized GPU compute process telemetry") from exc
    try:
        utilization = [int(line) for line in _nvidia_smi("gpu=utilization.gpu")]
    except ValueError as exc:
        raise ValueError("GPU utilization telemetry unavailable") from exc
    if len(utilization) != 1 or not 0 <= utilization[0] <= 100 or (require_idle and utilization[0] > 10):
        raise ValueError("GPU is not idle or device utilization cannot be attributed")
    return {"process_memory_mib": {str(pid): memory for pid, memory in processes.items()},
            "gpu_utilization_percent": utilization[0]}


def _check_gpu_context(baseline: dict, observed: dict, owner_host_pid: str | None = None) -> str:
    baseline_memory = baseline["process_memory_mib"]
    current_memory = observed["process_memory_mib"]
    if sum(baseline_memory.values()) > BASELINE_GPU_LIMIT_MIB:
        raise ValueError("baseline GPU contexts exceed 512 MiB")
    if not set(baseline_memory).issubset(current_memory):
        raise ValueError("baseline GPU context disappeared")
    new = set(current_memory) - set(baseline_memory)
    if len(new) != 1 or (owner_host_pid is not None and new != {owner_host_pid}):
        raise ValueError("benchmark CUDA context cannot be distinguished from peer process")
    if sum(abs(current_memory[pid] - memory) for pid, memory in baseline_memory.items()) > BASELINE_DRIFT_LIMIT_MIB:
        raise ValueError("baseline GPU context memory changed by more than 128 MiB")
    return new.pop()


def _allocate_cuda_context(device: torch.device) -> torch.Tensor:
    return torch.empty((1,), device=device)


def _check_resources(before: dict[str, int], after: dict[str, int] | None = None,
                     peak_allocated: int | None = None) -> None:
    for snapshot in (before, after):
        if snapshot is None:
            continue
        if (snapshot["cgroup_available_bytes"] < MIN_CGROUP
                or snapshot["disk_available_bytes"] < MIN_DISK):
            raise ValueError("raw cgroup or disk headroom below 24 GiB")
        if snapshot["cuda_free_bytes"] < GPU_RESERVE:
            raise ValueError("CUDA free memory below 2 GiB reserve")
    if after is not None and (before["cuda_total_bytes"] != after["cuda_total_bytes"]
                              or peak_allocated is None or peak_allocated <= 0
                              or peak_allocated + GPU_RESERVE > min(before["cuda_free_bytes"],
                                                                    after["cuda_free_bytes"])):
        raise ValueError("peak CUDA allocation cannot fit raw free memory plus 2 GiB reserve")


def _batch_sha(batch: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(batch.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([key, str(array.dtype), array.shape], separators=(",", ":")).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _replay_sha(replay: EpisodeReplay) -> str:
    state = replay.state_dict()
    state.pop("rng_state")  # Sampler state advances equally in both arms; episode data must not.
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        if key in ("episodes", "active"):
            episodes = value if key == "episodes" else ([] if value is None else [value])
            digest.update(json.dumps([key, len(episodes)], separators=(",", ":")).encode())
            for episode in episodes:
                for field, item in sorted(episode.items()):
                    if isinstance(item, np.ndarray):
                        digest.update(json.dumps([field, str(item.dtype), item.shape],
                                                 separators=(",", ":")).encode())
                        digest.update(item.tobytes())
                    else:
                        digest.update(json.dumps([field, item], separators=(",", ":")).encode())
        else:
            digest.update(json.dumps([key, value], separators=(",", ":")).encode())
    return digest.hexdigest()


def _rng_sha(replay: EpisodeReplay) -> str:
    value = json.dumps(replay.rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _make_replays() -> tuple[EpisodeReplay, OvershootReplay, dict]:
    # Only generated episodes; base replay samples the exact same RNG/window population.
    rng = np.random.default_rng(SEED)
    replay = OvershootReplay(120000, seed=SEED, include_partial=False)
    replay.start_episode(rng.integers(0, 256, (4, 64, 64), dtype=np.uint8))
    for index in range(128):
        replay.add_step(
            rng.integers(0, 256, (4, 64, 64), dtype=np.uint8),
            rng.uniform(-1, 1, 3).astype(np.float32), float(rng.normal()),
            truncated=index == 127,
        )
    # Match the completed RAW100k run's 307 replay-episode metadata entries
    # without generating roads or allocating its full 100k pixel transitions.
    for _ in range(306):
        replay.start_episode(rng.integers(0, 256, (4, 64, 64), dtype=np.uint8))
        replay.add_step(rng.integers(0, 256, (4, 64, 64), dtype=np.uint8),
                        rng.uniform(-1, 1, 3).astype(np.float32), float(rng.normal()),
                        truncated=True)
    base_replay = EpisodeReplay(120000, horizon=HORIZON, seed=SEED)
    initial_state = replay.state_dict()
    base_replay.load_state_dict(initial_state)
    base = base_replay.sample(BATCH_SIZE)
    extended = replay.sample(BATCH_SIZE)
    if any(not torch.equal(base[name], extended[name]) for name in base):
        raise ValueError("synthetic H3 base and overshoot replay samples diverged")
    valid4 = int(extended["overshoot_mask"][0].sum().item())
    valid5 = int(extended["overshoot_mask"][1].sum().item())
    if (base["obs"].shape != (4, BATCH_SIZE, 4, 64, 64)
            or base["action"].shape != (3, BATCH_SIZE, 3)
            or extended["overshoot_action"].shape != (2, BATCH_SIZE, 3)
            or extended["overshoot_reward"].shape != (2, BATCH_SIZE, 1)
            or extended["overshoot_mask"].shape != (2, BATCH_SIZE, 1)
            or valid5 < BATCH_SIZE * .95 or valid4 < valid5):
        raise ValueError("synthetic B256 H3 fixture lacks valid masked step-4/5 targets")
    base_replay.load_state_dict(initial_state)
    replay.load_state_dict(initial_state)
    if _rng_sha(base_replay) != _rng_sha(replay):
        raise ValueError("matched synthetic replay sampler RNG states diverged")
    return base_replay, replay, {
        "batch_size": BATCH_SIZE, "horizon": HORIZON, "action_dim": 3,
        "obs_shape": [4, 64, 64], "num_bins": 101, "model_size": 5,
        "valid_step4": valid4, "valid_step5": valid5, "synthetic_only": True,
        "synthetic_replay_episodes": replay.num_episodes,
        "synthetic_replay_transitions": len(replay), "replay_capacity": replay.capacity,
        "eligible_h3_windows": replay.eligible_windows(),
        "proof_h3_sample_sha256": _batch_sha(base),
        "proof_extended_sample_sha256": _batch_sha(extended),
    }


def _make_learners(device: torch.device) -> tuple[TDMPC2Learner, OvershootLearner, int]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = WorldModel(TDMPC2ModelConfig(
        action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True,
    ))
    if not 4_000_000 <= model.total_params <= 8_000_000:
        raise ValueError("expected original 5M-size-class TD-MPC2 model")
    twin = deepcopy(model)
    cfg = TDMPC2LearnerConfig(episode_length=2000, horizon=HORIZON, batch_size=BATCH_SIZE)
    base = TDMPC2Learner(model.to(device), cfg)
    variant = OvershootLearner(twin.to(device), cfg)
    return base, variant, model.total_params


def _timed_updates(learner, replay: object, count: int, device: torch.device,
                   *, clock=None, synchronize=None) -> list[float]:
    clock = time.perf_counter if clock is None else clock
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    elapsed = []
    for _ in range(count):
        synchronize(device)
        start = clock()
        metrics = learner.update(replay)
        synchronize(device)
        duration = clock() - start
        if (not math.isfinite(duration) or duration <= 0
                or not metrics or any(not math.isfinite(float(value)) for value in metrics.values())):
            raise ValueError("nonfinite/zero update timing or nonfinite synthetic optimizer metric")
        elapsed.append(duration)
    return elapsed


def _summary(samples: list[float]) -> dict:
    if len(samples) != TIMED or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("incomplete or nonfinite measured updates")
    ordered = sorted(samples)
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    first = statistics.median(samples[:TIMED // 2])
    second = statistics.median(samples[TIMED // 2:])
    p95 = ordered[math.ceil(.95 * TIMED) - 1]
    if (not math.isfinite(mean) or p95 > 1.5 * median
            or abs(mean - median) > .1 * median or abs(first - second) > .1 * median):
        raise ValueError("optimizer timings too noisy or GPU contended")
    return {"mean_seconds": mean, "median_seconds": median, "p95_seconds": p95,
            "first_half_median_seconds": first, "second_half_median_seconds": second,
            "update_wall_seconds": samples}


def _forecast(base_mean, variant_mean) -> dict:
    if any(not math.isfinite(float(value)) or value <= 0 for value in (base_mean, variant_mean)):
        raise ValueError("nonfinite/nonpositive optimizer update mean")
    base = base_mean if isinstance(base_mean, Fraction) else Fraction(Decimal(str(base_mean)))
    variant = variant_mean if isinstance(variant_mean, Fraction) else Fraction(Decimal(str(variant_mean)))
    positive_delta = max(Fraction(0), variant - base)
    forecast = Fraction(BASELINE_SECONDS) + BASELINE_UPDATES * positive_delta
    passed = forecast < FORECAST_LIMIT
    return {"baseline_elapsed_seconds": float(BASELINE_SECONDS), "baseline_updates": BASELINE_UPDATES,
            "positive_update_delta_seconds": float(positive_delta), "forecast_seconds": float(forecast),
            "threshold_seconds_exclusive": FORECAST_LIMIT, "wall_cap_seconds": WALL_CAP,
            "passed": passed}


def _seal(body: dict) -> dict:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**body, "body_sha256": hashlib.sha256(canonical).hexdigest()}


def benchmark(expected_script_sha256: str, *, root: Path = ROOT, output_path: str | None = None) -> dict:
    body = {"format": FORMAT, "status": "FAIL", "environment_resets": 0,
            "real_replay_reads": 0, "expected_script_sha256": expected_script_sha256,
            "output_path": output_path}
    problems = []
    before = replays = resource_before = device = gpu_context = sentinel = None
    baseline_gpu = owner_host_pid = None
    try:
        before = _sources(root)
        body["source_sha256_before"] = before
        _check_sources(before, expected_script_sha256, root)
        if torch.cuda.is_initialized():
            raise ValueError("CUDA initialized before ownership baseline snapshot")
        baseline_gpu = _gpu_snapshot()
        gpu_context = {
            "method": "NVIDIA host PID set delta around a deliberate CUDA allocation",
            "baseline_resident_limit_bytes": BASELINE_GPU_LIMIT_MIB * MIB,
            "baseline_drift_limit_bytes": BASELINE_DRIFT_LIMIT_MIB * MIB,
            "owner_host_pid": None,
            "snapshots": {"before_cuda_allocation": baseline_gpu},
        }
        if sum(baseline_gpu["process_memory_mib"].values()) > BASELINE_GPU_LIMIT_MIB:
            raise ValueError("baseline GPU contexts exceed 512 MiB")
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; CPU benchmark cannot pass")
        if torch.cuda.is_initialized():
            raise ValueError("CUDA initialized before deliberate benchmark allocation")
        device = torch.device("cuda:0")
        sentinel = _allocate_cuda_context(device)
        allocated_gpu = _gpu_snapshot(require_idle=False)
        gpu_context["snapshots"]["after_cuda_allocation"] = allocated_gpu
        owner_host_pid = _check_gpu_context(baseline_gpu, allocated_gpu)
        gpu_context["owner_host_pid"] = owner_host_pid
        resource_before = _raw_resources(device, root)
        _check_resources(resource_before)
        body["runtime"] = {
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "device": str(device), "cuda_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        }
        torch.cuda.reset_peak_memory_stats(device)
        base_replay, variant_replay, fixture = _make_replays()
        replays = {"base": base_replay, "variant": variant_replay}
        body["fixture"] = fixture
        body["input_sha256_before"] = {name: _replay_sha(replay) for name, replay in replays.items()}
        body["sample_rng_sha256_before"] = {name: _rng_sha(replay) for name, replay in replays.items()}
        base, variant, params = _make_learners(device)
        body["fixture"]["trainable_parameters"] = params
        initial_cuda_rng = torch.cuda.get_rng_state(device)
        _timed_updates(base, base_replay, WARMUP, device)
        base_cuda_rng = torch.cuda.get_rng_state(device)
        torch.cuda.set_rng_state(initial_cuda_rng, device)
        _timed_updates(variant, variant_replay, WARMUP, device)
        variant_cuda_rng = torch.cuda.get_rng_state(device)
        cuda_rng = {"base": base_cuda_rng, "variant": variant_cuda_rng}
        measured = {"base": [], "variant": []}
        for block, name in enumerate(("base", "variant", "variant", "base") * 2):
            if block == 4:
                midpoint = _gpu_snapshot(require_idle=False)
                gpu_context["snapshots"]["mid"] = midpoint
                _check_gpu_context(baseline_gpu, midpoint, owner_host_pid)
            torch.cuda.set_rng_state(cuda_rng[name], device)
            measured[name].extend(_timed_updates(
                base if name == "base" else variant,
                replays[name], TIMED // 4, device,
            ))
            cuda_rng[name] = torch.cuda.get_rng_state(device)
        body["timing"] = {
            "scope": "full_update_with_replay_sample",
            "synchronization": "torch.cuda.synchronize per update",
            "warmup_updates_each": WARMUP, "timed_updates_each": TIMED,
            "block_order": ["base", "variant", "variant", "base"] * 2,
            "base": _summary(measured["base"]), "variant": _summary(measured["variant"]),
        }
        body["forecast"] = _forecast(body["timing"]["base"]["mean_seconds"],
                                      body["timing"]["variant"]["mean_seconds"])
        if not body["forecast"]["passed"]:
            raise ValueError("strict forecast below 20600 seconds failed")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        if sentinel is not None and baseline_gpu is not None:
            try:
                final_gpu = _gpu_snapshot(require_idle=False)
                if gpu_context is not None:
                    gpu_context["snapshots"]["final"] = final_gpu
                _check_gpu_context(baseline_gpu, final_gpu, owner_host_pid)
            except Exception as exc:
                problems.append(f"GPU context recheck: {type(exc).__name__}: {exc}")
        if device is not None and resource_before is not None:
            try:
                resource_after = _raw_resources(device, root)
                peak = torch.cuda.max_memory_allocated(device)
                body["resources"] = {
                    **{f"{key}_before": value for key, value in resource_before.items()},
                    **{f"{key}_after": value for key, value in resource_after.items()},
                    "cuda_total_bytes": resource_before["cuda_total_bytes"],
                    "max_cuda_allocated_bytes": peak,
                    "min_cgroup_available_bytes": MIN_CGROUP, "min_disk_available_bytes": MIN_DISK,
                    "gpu_reserve_bytes": GPU_RESERVE,
                    "gpu_context_telemetry": gpu_context,
                }
                _check_resources(resource_before, resource_after, peak)
            except Exception as exc:
                problems.append(f"resource recheck: {type(exc).__name__}: {exc}")
        if "resources" not in body and gpu_context is not None:
            body["resources"] = {"gpu_context_telemetry": gpu_context}
        if replays is not None:
            try:
                body["input_sha256_after"] = {name: _replay_sha(replay) for name, replay in replays.items()}
                if body["input_sha256_after"] != body["input_sha256_before"]:
                    raise ValueError("synthetic replay data mutated during updates")
                body["sample_rng_sha256_after"] = {name: _rng_sha(replay) for name, replay in replays.items()}
                if len(set(body["sample_rng_sha256_after"].values())) != 1:
                    raise ValueError("base and variant replay RNG diverged after updates")
            except Exception as exc:
                problems.append(f"input rehash: {type(exc).__name__}: {exc}")
        if before is not None:
            try:
                body["source_sha256_after"] = _sources(root)
                if body["source_sha256_after"] != before:
                    raise ValueError("source changed during benchmark")
            except Exception as exc:
                problems.append(f"source rehash: {type(exc).__name__}: {exc}")
    if problems:
        body["failure_reasons"] = problems
    else:
        body["status"] = "PASS"
    return _seal(body)


def _output_file(argument: str | None, root: Path) -> Path | None:
    if argument is None:
        return None
    path = Path(argument)
    if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "runs"
            or not re.fullmatch(r"tdmpc2-reward-overshoot-throughput-[a-zA-Z0-9_-]+\.json", path.name)
            or (root / "runs").resolve(strict=True) != (root / path.parent).resolve(strict=True)):
        raise ValueError("output must be a new runs/tdmpc2-reward-overshoot-throughput-*.json")
    result = root / path
    if result.exists() or result.is_symlink():
        raise FileExistsError("benchmark receipt already exists; never overwrite")
    return result


def _write_receipt(path: Path, report: dict) -> None:
    data = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-script-sha256", required=True)
    parser.add_argument("--output", help="optional exclusive fsynced runs/ benchmark receipt")
    args = parser.parse_args(argv)
    try:
        output = _output_file(args.output, ROOT)
    except Exception as exc:
        result = _seal({"format": FORMAT, "status": "FAIL", "environment_resets": 0,
                        "real_replay_reads": 0, "failure_reasons": [f"{type(exc).__name__}: {exc}"]})
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 1
    result = benchmark(args.expected_script_sha256, output_path=args.output)
    if output is not None:
        try:
            _write_receipt(output, result)
        except Exception as exc:
            result = _seal({**{key: value for key, value in result.items() if key != "body_sha256"},
                            "status": "FAIL", "failure_reasons": result.get("failure_reasons", []) +
                            [f"receipt write: {type(exc).__name__}: {exc}"]})
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
