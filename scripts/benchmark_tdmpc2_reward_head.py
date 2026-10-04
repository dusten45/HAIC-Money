"""Synthetic-only CUDA feasibility screen; never load a checkpoint or HAIC environment.

Run after source freeze: python -m scripts.benchmark_tdmpc2_reward_head
--expected-script-sha256 <sha256>. Writes one exclusive PASS/FAIL receipt in runs/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv
from haic.algorithms.tdmpc2.reward_head_adapt import RewardHeadAdapter


ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/benchmark_tdmpc2_reward_head.py"
OUTPUT = "runs/tdmpc2-head-only-benchmark-20260929-v2.json"
FORMAT = "haic-tdmpc2-head-only-benchmark-v2"
SOURCE_SHA256 = {
    "haic/algorithms/tdmpc2/model.py": "a121d8688e5ab954abe2e6468ca84760e035f15b49004b695fc465696eb90c65",
    "haic/algorithms/tdmpc2/reward_head_adapt.py": "c2a1626b3ab321100ae22579d1f665a2b3fdc69661d3cdcce796f4df715fc615",
}
ROADS = (3910800001, 3910800004, 3910800034, 3910800085)
WARMUP = 32
TIMED = 128
HOLDOUT_TIMED = 32
HOLDOUT_FORECAST_BATCHES = 86
UPDATES = 512
IO_RESERVE_SECONDS = 600
WALL_CAP_SECONDS = 1800
GIB = 1024 ** 3
MIB = 1024 ** 2
MIN_HEADROOM = 24 * GIB
GPU_RESERVE = 2 * GIB


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(root: Path) -> dict[str, str]:
    return {name: _sha(root / name) for name in (*SOURCE_SHA256, SELF)}


def _check_sources(source: dict[str, str], expected_script_sha256: str, root: Path) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_script_sha256):
        raise ValueError("expected script SHA must be lowercase SHA-256")
    if source != {**SOURCE_SHA256, SELF: expected_script_sha256}:
        raise ValueError("pinned benchmark/adapter/model source SHA mismatch")
    for module, name in ((WorldModel.__module__, "haic/algorithms/tdmpc2/model.py"),
                         (RewardHeadAdapter.__module__, "haic/algorithms/tdmpc2/reward_head_adapt.py")):
        module_file = sys.modules[module].__file__
        if module_file is None or Path(module_file).resolve() != (root / name).resolve():
            raise ValueError("imported TD source path differs from pinned path")


def _nvidia_smi(query: str) -> list[str]:
    try:
        result = subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("GPU contention telemetry unavailable") from exc
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gpu_snapshot(*, require_idle: bool = True) -> dict:
    rows = _nvidia_smi("compute-apps=pid,used_gpu_memory")
    if rows == ["No running processes found"]:
        rows = []
    processes = {}
    for row in rows:
        fields = next(csv.reader([row], skipinitialspace=True))
        if len(fields) != 2 or any(not re.fullmatch(r"[0-9]+", item) for item in fields):
            raise ValueError("ambiguous GPU process telemetry")
        pid, memory = (int(item) for item in fields)
        if pid <= 0 or memory <= 0 or pid in processes:
            raise ValueError("ambiguous GPU process telemetry")
        processes[pid] = memory
    util = _nvidia_smi("gpu=utilization.gpu")
    if len(util) != 1 or not re.fullmatch(r"[0-9]+", util[0]) or not 0 <= int(util[0]) <= 100:
        raise ValueError("ambiguous GPU utilization telemetry")
    if require_idle and int(util[0]) > 10:
        raise ValueError("GPU not idle before benchmark CUDA allocation")
    return {"process_memory_mib": {str(pid): mem for pid, mem in processes.items()},
            "gpu_utilization_percent": int(util[0])}


def _check_gpu_context(baseline: dict, observed: dict, owner: str | None = None) -> str:
    old, new = baseline["process_memory_mib"], observed["process_memory_mib"]
    if sum(old.values()) > 512 or not old.keys() <= new.keys():
        raise ValueError("baseline GPU process missing or too large")
    introduced = new.keys() - old.keys()
    if len(introduced) != 1 or owner is not None and introduced != {owner}:
        raise ValueError("benchmark context cannot be distinguished from external GPU contention")
    if sum(abs(new[pid] - mem) for pid, mem in old.items()) > 128:
        raise ValueError("external baseline GPU context changed by >128 MiB")
    return introduced.pop()


def _raw_resources(device: torch.device, root: Path) -> dict[str, int]:
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        host = next(int(row.split()[1]) * 1024 for row in Path("/proc/meminfo").read_text().splitlines()
                    if row.startswith("MemAvailable:"))
        stat = os.statvfs(root / "runs")
        disk = stat.f_bavail * stat.f_frsize
        free, total = torch.cuda.mem_get_info(device)
    except (OSError, ValueError, TypeError, StopIteration) as exc:
        raise ValueError("raw host/cgroup/disk/CUDA resources unavailable") from exc
    if not 0 <= used <= limit or min(host, disk) <= 0 or not 0 < free <= total:
        raise ValueError("invalid raw resource readings")
    return {"host_available_bytes": host, "cgroup_available_bytes": limit - used,
            "disk_available_bytes": disk, "cuda_free_bytes": free, "cuda_total_bytes": total}


def _check_resources(before: dict, after: dict | None = None, *, peak_rss: int = 0,
                     output_bytes: int = 0, peak_reserved: int = 0) -> None:
    for snapshot in (before, after):
        if snapshot is None:
            continue
        if min(snapshot["host_available_bytes"], snapshot["cgroup_available_bytes"]) < MIN_HEADROOM + peak_rss:
            raise ValueError("raw host/cgroup headroom below 24 GiB plus measured peak RSS")
        if snapshot["disk_available_bytes"] < MIN_HEADROOM + output_bytes:
            raise ValueError("raw disk headroom below 24 GiB plus forecast output")
        if snapshot["cuda_free_bytes"] < GPU_RESERVE:
            raise ValueError("CUDA free memory below 2 GiB reserve")
    if after is not None and (before["cuda_total_bytes"] != after["cuda_total_bytes"]
                              or peak_reserved <= 0
                              or peak_reserved + GPU_RESERVE > min(before["cuda_free_bytes"],
                                                                   after["cuda_free_bytes"])):
        raise ValueError("measured peak CUDA reserved plus 2 GiB exceeds raw free")


def _fixture() -> tuple[dict[str, torch.Tensor], dict]:
    rng = np.random.default_rng(20260929)
    obs = rng.integers(0, 256, (256, 4, 64, 64), dtype=np.uint8)
    actions = rng.uniform(-1, 1, (256, 3)).astype(np.float32)
    rewards = np.empty((256, 1), dtype=np.float32)
    labels = []
    for road in ROADS:
        labels.extend([road] * 64)
    for index in range(4):
        start = index * 64
        rewards[start:start + 16, 0] = rng.uniform(.01, 3, 16).astype(np.float32)
        rewards[start + 16:start + 64, 0] = rng.uniform(-2, 0, 48).astype(np.float32)
    indices = rng.permutation(256)
    batch = {"obs": torch.from_numpy(obs[indices].copy()),
             "action": torch.from_numpy(actions[indices].copy()),
             "reward": torch.from_numpy(rewards[indices].copy())}
    labels = np.asarray(labels, dtype=np.int64)[indices]
    digest = hashlib.sha256()
    for field, tensor in sorted(batch.items()):
        digest.update(field.encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return batch, {"synthetic_only": True, "full_adapter_step": True, "batch_size": 256,
                   "model_size": 5, "roads": 4,
                   "obs_shape": [4, 64, 64], "action_dim": 3, "num_bins": 101,
                   "road_labels": list(ROADS), "positive_per_road": 16,
                   "nonpositive_per_road": 48,
                   "counts": {str(road): {"positive": int(((labels == road) & (batch["reward"][:, 0].numpy() > 0)).sum()),
                                         "nonpositive": int(((labels == road) & (batch["reward"][:, 0].numpy() <= 0)).sum())}
                              for road in ROADS}, "input_sha256": digest.hexdigest()}


def _timed(operation, count: int, device: torch.device, *, clock=None, synchronize=None) -> list[float]:
    clock = time.perf_counter if clock is None else clock
    synchronize = torch.cuda.synchronize if synchronize is None else synchronize
    times = []
    for _ in range(count):
        synchronize(device)
        start = clock()
        result = operation()
        synchronize(device)
        elapsed = clock() - start
        if not math.isfinite(elapsed) or elapsed <= 0 or result is None:
            raise ValueError("invalid CUDA timing or synthetic output")
        times.append(elapsed)
    return times


def _summary(samples: list[float], expected: int) -> dict:
    if len(samples) != expected or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("incomplete/nonfinite timed calls")
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    first, second = (statistics.median(samples[:expected // 2]),
                     statistics.median(samples[expected // 2:]))
    p95 = sorted(samples)[math.ceil(.95 * expected) - 1]
    drift = abs(first - second) / median
    if p95 > 1.5 * median or abs(mean - median) > .1 * median or drift > .1:
        raise ValueError("unstable CUDA timing or GPU contention")
    return {"mean_seconds": mean, "median_seconds": median, "p95_seconds": p95,
            "first_half_median_seconds": first, "second_half_median_seconds": second,
            "half_drift_fraction": drift, "seconds": samples}


def _forecast(step: dict, inference: dict, peak_rss: int, peak_cuda: int, output_bytes: int) -> dict:
    forecast = UPDATES * step["mean_seconds"] + HOLDOUT_FORECAST_BATCHES * inference["mean_seconds"] + IO_RESERVE_SECONDS
    return {"measured_update_seconds": step["mean_seconds"],
            "measured_holdout_seconds": HOLDOUT_FORECAST_BATCHES * inference["mean_seconds"],
            "measured_peak_rss_bytes": peak_rss, "measured_peak_cuda_bytes": peak_cuda,
            "measured_output_bytes": output_bytes, "planned_updates": UPDATES,
            "planned_holdout_batches": HOLDOUT_FORECAST_BATCHES,
            "source_load_seconds": IO_RESERVE_SECONDS // 2,
            "checkpoint_seconds": IO_RESERVE_SECONDS // 2,
            "io_reserve_seconds": IO_RESERVE_SECONDS,
            "source_checkpoint_ledger_receipt_io_reserve_seconds": IO_RESERVE_SECONDS,
            "forecast_seconds": forecast, "threshold_seconds_exclusive": WALL_CAP_SECONDS,
            "passed": math.isfinite(forecast) and forecast < WALL_CAP_SECONDS}


def _seal(body: dict) -> dict:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**body, "body_sha256": hashlib.sha256(canonical).hexdigest()}


def benchmark(expected_script_sha256: str, *, root: Path = ROOT) -> dict:
    body = {"format": FORMAT, "status": "FAIL", "environment_resets": 0,
            "real_replay_reads": 0, "synthetic_only_optimizer_updates": 0,
            "output_path": OUTPUT, "expected_script_sha256": expected_script_sha256}
    errors = []
    source = baseline = owner = device = before = sentinel = None
    context = {"method": "new NVIDIA host PID from before/after deliberate CUDA allocation",
               "snapshots": {}, "owner_host_pid": None}
    try:
        source = _sources(root)
        body["source_sha256_before"] = source
        _check_sources(source, expected_script_sha256, root)
        if str(torch.__version__) != "2.1.0+cu121" or torch.version.cuda != "12.1":
            raise ValueError("pinned Torch/CUDA runtime mismatch")
        if torch.cuda.is_initialized():
            raise ValueError("CUDA initialized before baseline GPU process snapshot")
        baseline = _gpu_snapshot()
        context["snapshots"]["before_cuda_allocation"] = baseline
        if sum(baseline["process_memory_mib"].values()) > 512:
            raise ValueError("baseline GPU contexts exceed 512 MiB")
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable; CPU timing cannot pass")
        device = torch.device("cuda:0")
        sentinel = torch.empty((1,), device=device)
        allocated = _gpu_snapshot(require_idle=False)
        context["snapshots"]["after_cuda_allocation"] = allocated
        owner = _check_gpu_context(baseline, allocated)
        context["owner_host_pid"] = owner
        before = _raw_resources(device, root)
        _check_resources(before)
        body["runtime"] = {"python": platform.python_version(), "torch": str(torch.__version__),
                           "torch_cuda": torch.version.cuda, "device": str(device),
                           "cuda_name": torch.cuda.get_device_name(device),
                           "cuda_capability": list(torch.cuda.get_device_capability(device)),
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                           "torch_num_threads": torch.get_num_threads()}
        batch, fixture = _fixture()
        body["fixture"] = fixture
        torch.manual_seed(20260929)
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
        if not 4_000_000 <= model.total_params <= 8_000_000:
            raise ValueError("unexpected TD 5M model size")
        fixture["model_parameters"] = model.total_params
        adapter = RewardHeadAdapter(model.to(device))
        if adapter.optim.param_groups[0]["lr"] != 3e-5:
            raise ValueError("reward-head Adam learning rate mismatch")
        torch.cuda.reset_peak_memory_stats(device)

        def step():
            metrics = adapter.step(batch)
            if any(not math.isfinite(value) for value in metrics.values()):
                raise ValueError("nonfinite synthetic reward-head metrics")
            body["synthetic_only_optimizer_updates"] = adapter.updates
            return metrics

        _timed(step, WARMUP, device)
        measured_step = _timed(step, TIMED, device)
        context["snapshots"]["mid"] = _gpu_snapshot(require_idle=False)
        _check_gpu_context(baseline, context["snapshots"]["mid"], owner)

        def infer():
            with torch.no_grad():
                z = model.encode(batch["obs"].to(device))
                decoded = two_hot_inv(model.reward(z, batch["action"].to(device)), model.cfg)
                if decoded.shape != (256, 1) or not bool(torch.isfinite(decoded).all()):
                    raise ValueError("invalid synthetic holdout inference")
            return decoded

        _timed(infer, WARMUP, device)
        measured_infer = _timed(infer, HOLDOUT_TIMED, device)
        if adapter.updates != WARMUP + TIMED or not adapter.verify_only_reward_changed():
            raise ValueError("expected synthetic updates or reward-head-only state change missing")
        body["timing"] = {"scope": "full_adapter_step_including_encode_head_optimizer_state_digest",
                          "synchronization": "torch.cuda.synchronize before and after each call",
                          "warmup_updates": WARMUP, "timed_updates": TIMED,
                          "holdout_warmup_batches": WARMUP, "holdout_timed_batches": HOLDOUT_TIMED,
                          "holdout_batches": HOLDOUT_FORECAST_BATCHES,
                          "step": _summary(measured_step, TIMED),
                          "holdout_inference": _summary(measured_infer, HOLDOUT_TIMED)}
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        output_bytes = max(64 * MIB, 3 * sum(t.numel() * t.element_size() for t in model.state_dict().values()) + 64 * MIB)
        body["forecast"] = _forecast(body["timing"]["step"], body["timing"]["holdout_inference"],
                                     peak_rss, torch.cuda.max_memory_reserved(device), output_bytes)
        if not body["forecast"]["passed"]:
            raise ValueError("forecast exceeds strict 1800-second hard cap")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if sentinel is not None and baseline is not None:
            try:
                final = _gpu_snapshot(require_idle=False)
                context["snapshots"]["final"] = final
                _check_gpu_context(baseline, final, owner)
            except Exception as exc:
                errors.append(f"final GPU contention: {type(exc).__name__}: {exc}")
        if before is not None and device is not None:
            try:
                after = _raw_resources(device, root)
                peak_allocated = torch.cuda.max_memory_allocated(device)
                peak_reserved = torch.cuda.max_memory_reserved(device)
                peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                output_bytes = body.get("forecast", {}).get("measured_output_bytes", 64 * MIB)
                body["resources"] = {"before": before, "after": after,
                                     "peak_process_rss_bytes": peak_rss,
                                     "max_cuda_allocated_bytes": peak_allocated,
                                     "max_cuda_reserved_bytes": peak_reserved,
                                     "forecast_output_bytes": output_bytes,
                                     "min_raw_headroom_bytes": MIN_HEADROOM,
                                     "gpu_reserve_bytes": GPU_RESERVE,
                                     "gpu_context_telemetry": context}
                _check_resources(before, after, peak_rss=peak_rss, output_bytes=output_bytes,
                                 peak_reserved=peak_reserved)
                if "forecast" in body and (peak_rss > body["forecast"]["measured_peak_rss_bytes"]
                                           or peak_reserved > body["forecast"]["measured_peak_cuda_bytes"]):
                    raise ValueError("post-measurement memory grew beyond forecast")
            except Exception as exc:
                errors.append(f"final resources: {type(exc).__name__}: {exc}")
        else:
            body["resources"] = {"gpu_context_telemetry": context}
        if source is not None:
            try:
                body["source_sha256_after"] = _sources(root)
                if body["source_sha256_after"] != source:
                    raise ValueError("pinned sources changed during benchmark")
            except Exception as exc:
                errors.append(f"source rehash: {type(exc).__name__}: {exc}")
    if errors:
        body["failure_reasons"] = errors
    else:
        body["status"] = "PASS"
    return _seal(body)


def _write_receipt(path: Path, report: dict) -> None:
    data = (json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-script-sha256", required=True)
    args = parser.parse_args(argv)
    path = ROOT / OUTPUT
    if path.exists() or path.is_symlink() or (ROOT / "runs").is_symlink():
        result = _seal({"format": FORMAT, "status": "FAIL", "environment_resets": 0,
                        "real_replay_reads": 0, "synthetic_only_optimizer_updates": 0,
                        "failure_reasons": ["exclusive benchmark receipt already exists or runs is symlink; never overwrite"]})
    else:
        result = benchmark(args.expected_script_sha256)
        try:
            _write_receipt(path, result)
        except Exception as exc:
            result = _seal({**{key: value for key, value in result.items() if key != "body_sha256"},
                            "status": "FAIL", "failure_reasons": result.get("failure_reasons", []) +
                            [f"receipt write: {type(exc).__name__}: {exc}"]})
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
