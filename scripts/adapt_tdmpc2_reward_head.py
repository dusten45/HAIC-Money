"""Isolated, zero-reset reward-head adaptation on completed consumed TRAIN replay.

``python -m scripts.adapt_tdmpc2_reward_head --preflight`` proposes an exact
protocol without deserializing a checkpoint. Execution requires an independently
frozen protocol SHA; an interrupted output is preserved and cannot be resumed.
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

import numpy as np
import torch

from scripts import diagnose_tdmpc2_h5_logged as raw
from scripts import score_tdmpc2_overshoot_archived_branches as overshoot


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "experiments/tdmpc2-head-only-adaptation-v2.json"
OUTPUT = "runs/tdmpc2-head-only-20260929-v2"
BENCHMARK = "runs/tdmpc2-head-only-benchmark-20260929-v2.json"
BENCHMARK_SCRIPT = "scripts/benchmark_tdmpc2_reward_head.py"
SELF = "scripts/adapt_tdmpc2_reward_head.py"
TEST = "tests/test_adapt_tdmpc2_reward_head.py"
ADAPTER = "haic/algorithms/tdmpc2/reward_head_adapt.py"
RAW_SUMMARY = ("experiments/tdmpc2-long-reused-train-v2-100k-result.json",
               "bacf4c7e7a53794ecec7797746388695bac01096287f7aff09b5feafc4e3b515")
NEW_SUMMARY = ("experiments/tdmpc2-reward-overshoot-train-v1-result.json",
               "15b5e795d6e6c3e7127a7d804e41c428e9ad2904fdcec527ce27c7983da3a63e")
NEW_RESULT = (f"{overshoot.RUN}/result.json", "8ab5c0af8dea9edc785616143788e7a51313504fa13d5a30e12624f7bd0d40bb")
NEW_CHECKPOINT = (f"{overshoot.RUN}/checkpoint-at-least-100000-step-100159.pt",
                  "51b1da8c4e2c2584524a3d51ccbd82daa1121479c123237d2e809cbde1bcdaa0")
NEW_TRAIN = (f"{overshoot.RUN}/training.jsonl", "f7915c447d5dd9636648ad555d91043dbe30dddfdb303df9d7c41d14f7e11a97")
NEW_STEPS = (f"{overshoot.RUN}/steps.jsonl", "dac7e2ddcaa3168b5773c814b4dd36ea7898574ba7d4ba2854821ff8d45f1b6d")
SPLIT = {"previously_exposed": [0, 7], "reserved_future_branch": [8, 11],
         "adaptation_excluded_logged_train": [12, 43], "adaptation": [44, 306]}
EXPECTED_SPLIT_COUNTS = {
    "3910800001": {"holdout": {"positive": 144, "nonpositive": 2137},
                   "adaptation": {"positive": 7492, "nonpositive": 13704}},
    "3910800004": {"holdout": {"positive": 199, "nonpositive": 2826},
                   "adaptation": {"positive": 6443, "nonpositive": 15187}},
    "3910800034": {"holdout": {"positive": 137, "nonpositive": 2073},
                   "adaptation": {"positive": 7130, "nonpositive": 13360}},
    "3910800085": {"holdout": {"positive": 151, "nonpositive": 2262},
                   "adaptation": {"positive": 7453, "nonpositive": 15812}},
}
TRAINING = {"seed": 20260929, "updates": 512, "batch_size": 256,
            "positive_per_road": 16, "nonpositive_per_road": 48,
            "optimizer": "Adam", "lr": 3e-5, "weight_decay": 0.0,
            "target": "observed_RAW_step_reward_true_encoded_obs_action_soft_two_hot_CE",
            "max_wall_seconds": 1800, "journal_max_update_interval": 64}
GATE = {"minimum_holdout_positive_per_road": 100,
        "minimum_holdout_nonpositive_per_road": 100,
        "minimum_qualifying_roads": 3, "positive_mae_improvement_at_least": 0.10,
        "positive_absolute_signed_bias_improvement_at_least": 0.10,
        "nonpositive_mae_worsening_at_most": 0.05}


def ref(path: str, sha: str) -> dict:
    return {"path": path, "sha256": sha}


def _sha_body(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _pinned(root: Path, pair: tuple[str, str]) -> Path:
    return overshoot.branch.pinned(root, *pair)


def _runtime() -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_num_threads": torch.get_num_threads()}


def _source(root: Path) -> tuple[dict, list[dict], dict]:
    """Validate BOTH full producers and ledgers without reading pickle bytes as objects."""
    original = raw._prepared(root)[2]
    raw_summary = raw._json(_pinned(root, RAW_SUMMARY).read_bytes())
    if (raw_summary.get("checkpoint_sha256") != raw.CHECKPOINT_SHA
            or raw_summary.get("step_ledger_sha256") != raw.STEPS_SHA
            or raw_summary.get("training_ledger_sha256") != raw.TRAIN_SHA
            or raw_summary.get("run_result_sha256") != raw.RESULT_SHA
            or raw_summary.get("training_protocol_sha256") != raw.SOURCE_PROTOCOL_SHA
            or raw_summary.get("run_status") != "completed_boundary_at_least_100k"
            or (raw_summary.get("decisions"), raw_summary.get("updates"), raw_summary.get("episodes"))
            != (100354, 100354, 307)):
        raise ValueError("RAW summary/checkpoint/ledger binding differs")
    summary = raw._json(_pinned(root, NEW_SUMMARY).read_bytes())
    training = raw._json(_pinned(root, (overshoot.TRAIN_PROTOCOL, overshoot.TRAIN_PROTOCOL_SHA)).read_bytes())
    refs = {"protocol": ref(overshoot.TRAIN_PROTOCOL, overshoot.TRAIN_PROTOCOL_SHA),
            "result": ref(*NEW_RESULT), "checkpoint": ref(*NEW_CHECKPOINT),
            "training_ledger": ref(*NEW_TRAIN), "step_ledger": ref(*NEW_STEPS)}
    if (summary.get("status") != "complete_100k_reused_train_model_internal_only"
            or summary.get("protocol_sha256") != overshoot.TRAIN_PROTOCOL_SHA
            or summary.get("primary_result") != NEW_RESULT[0]
            or summary.get("primary_result_sha256") != NEW_RESULT[1]
            or summary.get("selected_checkpoint") != NEW_CHECKPOINT[0]
            or summary.get("selected_checkpoint_sha256") != NEW_CHECKPOINT[1]
            or summary.get("training_ledger") != NEW_TRAIN[0]
            or summary.get("training_ledger_sha256") != NEW_TRAIN[1]
            or summary.get("step_ledger") != NEW_STEPS[0]
            or summary.get("step_ledger_sha256") != NEW_STEPS[1]
            or summary.get("operator_sha256") != training.get("source_sha256", {}).get(
                "scripts/train_tdmpc2_reward_overshoot.py")
            or summary.get("auxiliary_module_sha256") != training.get("source_sha256", {}).get(
                "haic/algorithms/tdmpc2/reward_overshoot.py")
            or training.get("format") != "haic-tdmpc2-reward-overshoot-train-v1"
            or training.get("cells") != [{"track_id": 1, "geometry_seed": road} for road in raw.ROADS]
            or training.get("episode_schedule") != [0, 1, 2, 3]):
        raise ValueError("overshoot summary/producer/schedule binding differs")
    peak_rss = summary.get("peak_process_rss_mib")
    if not isinstance(peak_rss, (int, float)) or isinstance(peak_rss, bool):
        raise ValueError("completed source peak RSS measurement unavailable")
    if not math.isfinite(peak_rss) or peak_rss < 16000:
        raise ValueError("completed source peak RSS measurement unavailable")
    for name, sha in training["source_sha256"].items():
        overshoot.branch.pinned(root, name, sha)
    source_spec = {"training_source": refs}
    result, last = overshoot._finished_result(root, source_spec, training)
    if (result["decisions"], result["updates"], result["episodes"]) != (100159, 100159, 309):
        raise ValueError("unexpected overshoot terminal source")
    overshoot._ledgers(root, source_spec, training, result)
    return original, original["episodes"], {"training": training, "result": result,
                                           "last": last, "refs": refs,
                                           "source_peak_rss_bytes": math.ceil(peak_rss * 1024**2),
                                           "original_sources": original["original_sources"]}


def partition(eid: int) -> str:
    if type(eid) is not int or not 0 <= eid <= 306:
        raise ValueError("not an original completed RAW episode ID")
    if eid >= SPLIT["adaptation"][0]:
        return "adaptation"
    if eid >= SPLIT["adaptation_excluded_logged_train"][0]:
        return "holdout"
    return "excluded"


def _indices(episodes: list[dict], steps: Path) -> tuple[dict[int, dict[str, list[tuple[int, int]]]], dict]:
    """Full step cursor, strict ending semantics and unique signed pools; no pixels."""
    if len(episodes) != 307:
        raise ValueError("RAW complete 307 episode ledger required")
    pools = {road: {"positive": [], "nonpositive": []} for road in raw.ROADS}
    holdout = {road: {"positive": [], "nonpositive": []} for road in raw.ROADS}
    decision = 0
    with steps.open("rb") as stream:
        for eid, ep in enumerate(episodes):
            road, length = ep.get("geometry_seed"), ep.get("length")
            if (ep.get("episode") != eid or ep.get("track_id") != 1
                    or type(road) is not int or road != raw.ROADS[eid % 4] or type(length) is not int
                    or not 1 <= length <= 2000 or ep.get("decisions") != decision + length):
                raise ValueError("RAW episode schedule/length differs")
            actions, native, total = hashlib.sha256(), hashlib.sha256(), 0.0
            row = None
            for index in range(length):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("RAW step ledger truncated")
                row = raw._json(line)
                try:
                    action = bytes.fromhex(row["action_f32_hex"])
                    applied = bytes.fromhex(row["native_action_f32_hex"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("RAW action bytes invalid") from exc
                from haic.algorithms.tdmpc2.haic_env import environment_action
                if (row.get("episode") != eid or row.get("decision") != decision + 1
                        or row.get("track_id") != 1 or row.get("geometry_seed") != road
                        or len(action) != 12 or len(applied) != 12
                        or not np.isfinite(np.frombuffer(action, dtype="<f4")).all()
                        or np.any(np.abs(np.frombuffer(action, dtype="<f4")) > 1)
                        or environment_action(np.frombuffer(action, dtype="<f4")).tobytes() != applied
                        or type(row.get("reward")) not in (float, int) or not math.isfinite(row["reward"])
                        or any(type(row.get(k)) is not bool for k in ("terminated", "truncated", "terminal"))
                        or (index < length - 1 and any(row[k] for k in ("terminated", "truncated", "terminal")))
                        or row["terminated"] and not row["terminal"]
                        or row["terminal"] and not (row["terminated"] or row["truncated"])):
                    raise ValueError("RAW step identity/action/reward/ending differs")
                actions.update(action)
                native.update(applied)
                total += row["reward"]
                split = partition(eid)
                if split != "excluded":
                    target = holdout if split == "holdout" else pools
                    target[road]["positive" if np.float32(row["reward"]) > 0 else "nonpositive"].append((eid, index))
                decision += 1
            if (row is None or not (row["terminated"] or row["truncated"])
                    or any(ep.get(k) is not row[k] for k in ("terminated", "truncated", "terminal"))
                    or ep.get("finished") is not (row["truncated"] and row["terminal"] and not row["terminated"])
                    or actions.hexdigest() != ep.get("action_trace_sha256")
                    or native.hexdigest() != ep.get("native_action_trace_sha256")
                    or not math.isclose(total, ep.get("return", math.inf), rel_tol=0, abs_tol=1e-4)):
                raise ValueError("RAW finished/timeout/episode action trace differs")
        if stream.read(1) or decision != 100354:
            raise ValueError("RAW step ledger incomplete or has extra steps")
    for road in raw.ROADS:
        if (len(pools[road]["positive"]) < 512 or len(pools[road]["nonpositive"]) < 1536
                or len(holdout[road]["positive"]) < 100 or len(holdout[road]["nonpositive"]) < 100
                or {e for sign in holdout[road].values() for e, _ in sign} !=
                {eid for eid in range(12, 44) if raw.ROADS[eid % 4] == road}):
            raise ValueError(f"inadequate distinct adaptation/holdout RAW step targets: {road}")
    if {item for road in raw.ROADS for sign in pools[road] for item in pools[road][sign]} & {
            item for road in raw.ROADS for sign in holdout[road] for item in holdout[road][sign]}:
        raise ValueError("adaptation and holdout targets overlap")
    return pools, holdout


def _benchmark(root: Path, spec: dict) -> dict:
    """Verify the independent synthetic full-step receipt, never trusting CLI numbers."""
    reference = spec["throughput_benchmark"]
    receipt = raw._json(_pinned(root, (BENCHMARK, reference["sha256"])).read_bytes())
    if receipt.get("body_sha256") != _sha_body({key: value for key, value in receipt.items()
                                                  if key != "body_sha256"}):
        raise ValueError("synthetic benchmark canonical body SHA differs")
    sources = receipt.get("source_sha256_before")
    required = {"haic/algorithms/tdmpc2/model.py", ADAPTER, BENCHMARK_SCRIPT}
    if (receipt.get("format") != "haic-tdmpc2-head-only-benchmark-v2"
            or receipt.get("status") != "PASS" or "failure_reasons" in receipt
            or receipt.get("output_path") != BENCHMARK
            or receipt.get("environment_resets") != 0 or receipt.get("real_replay_reads") != 0
            or not isinstance(sources, dict) or set(sources) != required
            or receipt.get("source_sha256_after") != sources
            or receipt.get("expected_script_sha256") != sources.get(BENCHMARK_SCRIPT)
            or any(sources.get(key) != spec["source_sha256"].get(key) for key in required)):
        raise ValueError("synthetic benchmark source/status/provenance mismatch")
    observed = receipt.get("runtime")
    expected = spec["runtime"]
    if (not isinstance(observed, dict) or spec["device"] != "cuda"
            or not expected["cuda_available"] or observed.get("device") != "cuda:0"
            or observed.get("python") != expected["python"]
            or observed.get("torch") != expected["torch"]
            or observed.get("torch_cuda") != expected["torch_cuda"]
            or observed.get("cuda_name") != expected["gpu_name"]
            or observed.get("cuda_capability") != expected["cuda_capability"]
            or observed.get("cuda_visible_devices") != expected["cuda_visible_devices"]
            or observed.get("torch_num_threads") != expected["torch_num_threads"]):
        raise ValueError("synthetic CUDA runtime/device differs from frozen operator")
    fixture = receipt.get("fixture")
    if (not isinstance(fixture, dict) or fixture.get("synthetic_only") is not True
            or fixture.get("batch_size") != 256 or fixture.get("model_size") != 5
            or fixture.get("obs_shape") != [4, 64, 64] or fixture.get("action_dim") != 3
            or fixture.get("num_bins") != 101 or fixture.get("road_labels") != list(raw.ROADS)
            or fixture.get("positive_per_road") != 16 or fixture.get("nonpositive_per_road") != 48
            or type(fixture.get("model_parameters")) is not int
            or not 4_000_000 <= fixture["model_parameters"] <= 8_000_000
            or not isinstance(fixture.get("input_sha256"), str)
            or raw._sha(fixture["input_sha256"]) != fixture["input_sha256"]
            or fixture.get("counts") != {str(road): {"positive": 16, "nonpositive": 48}
                                         for road in raw.ROADS}):
        raise ValueError("synthetic batch is not the four-road observed-target head objective")
    timing = receipt.get("timing")
    if (not isinstance(timing, dict)
            or timing.get("scope") != "full_adapter_step_including_encode_head_optimizer_state_digest"
            or timing.get("synchronization") != "torch.cuda.synchronize before and after each call"
            or type(timing.get("warmup_updates")) is not int or timing["warmup_updates"] < 32
            or type(timing.get("timed_updates")) is not int or timing["timed_updates"] < 128
            or type(timing.get("holdout_warmup_batches")) is not int
            or timing["holdout_warmup_batches"] < 32
            or type(timing.get("holdout_timed_batches")) is not int
            or timing["holdout_timed_batches"] < 32
            or type(timing.get("holdout_batches")) is not int or timing["holdout_batches"] < 86
            or receipt.get("synthetic_only_optimizer_updates") !=
            timing["warmup_updates"] + timing["timed_updates"]):
        raise ValueError("benchmark did not time full synchronized head-only updates")
    for name, count in (("step", timing["timed_updates"]),
                        ("holdout_inference", timing["holdout_timed_batches"])):
        summary = timing.get(name)
        if (not isinstance(summary, dict) or not isinstance(summary.get("seconds"), list)
                or len(summary["seconds"]) != count
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in summary["seconds"])
                or not math.isclose(summary.get("mean_seconds", math.inf),
                                    statistics.fmean(summary["seconds"]), rel_tol=1e-9, abs_tol=1e-9)
                or not math.isclose(summary.get("median_seconds", math.inf),
                                    statistics.median(summary["seconds"]), rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError(f"missing or inconsistent synthetic {name} timings")
        median = summary["median_seconds"]
        ordered = sorted(summary["seconds"])
        if (summary.get("p95_seconds") != ordered[math.ceil(.95 * count) - 1]
                or not median > 0 or summary["p95_seconds"] > 1.5 * median
                or abs(summary["mean_seconds"] - median) > .1 * median):
            raise ValueError("synthetic throughput unstable or noisy")
    forecast = receipt.get("forecast")
    if not isinstance(forecast, dict):
        raise ValueError("benchmark measured forecast absent")
    for key in ("measured_update_seconds", "measured_holdout_seconds", "measured_peak_rss_bytes",
                "measured_peak_cuda_bytes", "measured_output_bytes", "forecast_seconds",
                "source_checkpoint_ledger_receipt_io_reserve_seconds"):
        value = forecast.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"invalid benchmark forecast field: {key}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid benchmark forecast field: {key}")
    expected_seconds = (512 * timing["step"]["mean_seconds"] +
                        forecast["planned_holdout_batches"] * timing["holdout_inference"]["mean_seconds"] +
                        forecast["source_checkpoint_ledger_receipt_io_reserve_seconds"])
    if (forecast.get("planned_updates") != 512
            or type(forecast.get("planned_holdout_batches")) is not int
            or forecast["planned_holdout_batches"] < 86
            or forecast["planned_holdout_batches"] != timing["holdout_batches"]
            or forecast["source_checkpoint_ledger_receipt_io_reserve_seconds"] < 600
            or forecast.get("source_load_seconds") != 300
            or forecast.get("checkpoint_seconds") != 300
            or forecast["source_load_seconds"] + forecast["checkpoint_seconds"] !=
            forecast["source_checkpoint_ledger_receipt_io_reserve_seconds"]
            or not math.isclose(forecast["measured_update_seconds"], timing["step"]["mean_seconds"],
                                rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(forecast["measured_holdout_seconds"],
                                forecast["planned_holdout_batches"] * timing["holdout_inference"]["mean_seconds"],
                                rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(forecast["forecast_seconds"], expected_seconds,
                                rel_tol=1e-9, abs_tol=1e-6)
            or forecast.get("threshold_seconds_exclusive") != 1800
            or forecast.get("passed") is not True or not expected_seconds < 1800):
        raise ValueError("source-load/holdout/update/checkpoint 1800-second forecast failed")
    resources = receipt.get("resources")
    if (not isinstance(resources, dict) or resources.get("min_raw_headroom_bytes") != 24 * 1024**3
            or resources.get("gpu_reserve_bytes") != 2 * 1024**3
            or resources.get("peak_process_rss_bytes") != forecast["measured_peak_rss_bytes"]
            or resources.get("max_cuda_reserved_bytes") != forecast["measured_peak_cuda_bytes"]
            or resources.get("forecast_output_bytes") != forecast["measured_output_bytes"]):
        raise ValueError("benchmark peak RAM/CUDA/output measurements differ from forecast")
    for name in ("before", "after"):
        values = resources.get(name)
        if (not isinstance(values, dict) or any(type(values.get(key)) is not int or values[key] <= 0
                for key in ("host_available_bytes", "cgroup_available_bytes", "disk_available_bytes",
                            "cuda_free_bytes", "cuda_total_bytes"))
                or min(values["host_available_bytes"], values["cgroup_available_bytes"]) <
                24 * 1024**3 + forecast["measured_peak_rss_bytes"]
                or values["disk_available_bytes"] < 24 * 1024**3 + forecast["measured_output_bytes"]
                or values["cuda_free_bytes"] < 2 * 1024**3 + forecast["measured_peak_cuda_bytes"]):
            raise ValueError("synthetic benchmark raw resource reserve failed")
    telemetry = resources.get("gpu_context_telemetry")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("snapshots"), dict):
        raise ValueError("synthetic benchmark GPU isolation telemetry missing")
    snapshots = telemetry["snapshots"]
    owner = telemetry.get("owner_host_pid")
    if not isinstance(owner, str) or not owner.isdecimal():
        raise ValueError("synthetic GPU benchmark host PID missing")
    before = snapshots.get("before_cuda_allocation")
    if not isinstance(before, dict) or not isinstance(before.get("process_memory_mib"), dict):
        raise ValueError("synthetic GPU benchmark baseline processes missing")
    baseline = before["process_memory_mib"]
    if (type(before.get("gpu_utilization_percent")) is not int
            or before["gpu_utilization_percent"] > 10
            or any(not isinstance(pid, str) or not pid.isdecimal() or type(mem) is not int or mem <= 0
                   for pid, mem in baseline.items()) or sum(baseline.values()) > 512):
        raise ValueError("synthetic GPU benchmark baseline contention")
    for stage in ("after_cuda_allocation", "mid", "final"):
        reading = snapshots.get(stage)
        processes = reading.get("process_memory_mib") if isinstance(reading, dict) else None
        if (not isinstance(processes, dict) or processes.keys() - baseline.keys() != {owner}
                or any(pid not in processes or type(processes[pid]) is not int
                       or abs(processes[pid] - mem) > 128 for pid, mem in baseline.items())
                or type(processes.get(owner)) is not int or processes[owner] <= 0):
            raise ValueError("synthetic GPU benchmark context contention")
    return receipt


def _resources(root: Path, forecast: dict, runtime: dict, *, source_peak_rss_bytes: int) -> dict:
    required = {"measured_update_seconds", "measured_holdout_seconds", "measured_peak_rss_bytes",
                "measured_peak_cuda_bytes", "measured_output_bytes", "forecast_seconds"}
    if (not required <= set(forecast)
             or any(type(forecast[k]) not in (float, int) or not math.isfinite(forecast[k])
                    or forecast[k] <= 0 for k in required)
            or forecast["forecast_seconds"] >= 1800):
        raise ValueError("missing positive measured 512-update/holdout forecast below hard wall")
    try:
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text().strip())
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        with Path("/proc/meminfo").open() as stream:
            available = next(int(line.split()[1]) * 1024 for line in stream if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError) as exc:
        raise ValueError("raw host/cgroup headroom unavailable") from exc
    stat = os.statvfs(root / "runs")
    disk = stat.f_bavail * stat.f_frsize
    cgroup = limit - current
    gpu = torch.cuda.mem_get_info(0)[0] if runtime["cuda_available"] else None
    peak = max(source_peak_rss_bytes, forecast["measured_peak_rss_bytes"])
    if (min(available, cgroup) < 24 * 1024**3 + peak
            or disk < 24 * 1024**3 + forecast["measured_output_bytes"]
            or gpu is not None and gpu < forecast["measured_peak_cuda_bytes"] + 2 * 1024**3):
        raise ValueError("measured forecast plus raw 24GiB cgroup/disk or 2GiB CUDA reserve failed")
    return {"host_available_bytes": available, "raw_cgroup_available_bytes": cgroup,
            "disk_available_bytes": disk, "cuda_free_bytes": gpu,
            "source_peak_rss_bytes": source_peak_rss_bytes, "required_peak_rss_bytes": peak,
            "forecast_seconds": forecast["forecast_seconds"]}


def proposal(root: Path, context: dict, counts: dict, benchmark_sha256: str, device: str) -> dict:
    sources = {**context["original_sources"], **context["training"]["source_sha256"]}
    for path in (SELF, TEST, ADAPTER, BENCHMARK_SCRIPT, "scripts/diagnose_tdmpc2_h5_logged.py",
                 "scripts/score_tdmpc2_overshoot_archived_branches.py",
                 "scripts/diagnose_tdmpc2_h5_branches.py",
                 "scripts/diagnose_tdmpc2_checkpoint_losses.py"):
        sources[path] = raw._digest(raw._file(root, path))
    return {"format": "haic-tdmpc2-head-only-adaptation-v2",
            "scope": "original_RAW_replay_consumed_TRAIN_no_environment_or_official_score",
            "raw_source": {"protocol": ref("experiments/tdmpc2-long-reused-train-v2.json", raw.SOURCE_PROTOCOL_SHA),
                           "summary": ref(*RAW_SUMMARY), "result": ref(f"{raw.RUN}/result.json", raw.RESULT_SHA),
                           "checkpoint": ref(raw.CHECKPOINT, raw.CHECKPOINT_SHA),
                           "training_ledger": ref(f"{raw.RUN}/training.jsonl", raw.TRAIN_SHA),
                           "step_ledger": ref(f"{raw.RUN}/steps.jsonl", raw.STEPS_SHA)},
            "overshoot_source": {"summary": ref(*NEW_SUMMARY), **context["refs"]},
            "source_sha256": sources, "runtime": _runtime(), "device": device,
            "split": SPLIT, "unique_step_counts": counts, "training": TRAINING,
            "holdout_gate": GATE, "throughput_benchmark": ref(BENCHMARK, benchmark_sha256),
            "output_dir": OUTPUT,
            "environment_resets": 0, "resume_supported": False}


def preflight(root: Path = ROOT, protocol_sha256: str | None = None,
              *, benchmark_sha256: str | None = None, device: str = "cpu",
              allow_output: bool = False) -> dict:
    root = Path(root).resolve(strict=True)
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("explicit supported device required")
    output = root / OUTPUT
    if ((output.exists() or output.is_symlink()) and not allow_output
            or allow_output and (not output.is_dir() or output.is_symlink())
            or (root / "runs").is_symlink()):
        raise ValueError("exclusive output directory already exists or runs is symlink")
    frozen = None
    if protocol_sha256 is not None:
        frozen = raw._json(overshoot.branch.pinned(root, PROTOCOL, raw._sha(protocol_sha256)).read_bytes())
        bound_benchmark = frozen.get("throughput_benchmark")
        if (not isinstance(bound_benchmark, dict) or set(bound_benchmark) != {"path", "sha256"}
                or bound_benchmark["path"] != BENCHMARK or benchmark_sha256 is not None
                and benchmark_sha256 != bound_benchmark["sha256"]):
            raise ValueError("frozen benchmark reference differs")
        benchmark_sha256 = bound_benchmark["sha256"]
    if benchmark_sha256 is None:
        raise ValueError("independent synthetic benchmark receipt SHA required")
    benchmark_sha256 = raw._sha(benchmark_sha256)
    original, episodes, context = _source(root)
    pools, holdout = _indices(episodes, raw._file(root, f"{raw.RUN}/steps.jsonl"))
    counts = {str(road): {"adaptation": {s: len(pools[road][s]) for s in pools[road]},
                          "holdout": {s: len(holdout[road][s]) for s in holdout[road]}}
              for road in raw.ROADS}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError("exact predeclared v2 RAW signed target counts differ")
    spec = proposal(root, context, counts, benchmark_sha256, device)
    if frozen is not None:
        if frozen != spec:
            raise ValueError("frozen protocol differs from exact source/split/runtime/forecast proposal")
    receipt = _benchmark(root, spec)
    resources = _resources(root, receipt["forecast"], spec["runtime"],
                           source_peak_rss_bytes=context["source_peak_rss_bytes"])
    return {"status": "preflight_only", "schema": spec, "resources": resources,
            "environment_resets": 0, "torch_load_calls": 0,
            "_pools": pools, "_holdout": holdout, "_episodes": episodes, "_context": context,
            "_original": original}


def sample_batch(replay: dict, pools: dict, rng: np.random.Generator) -> tuple[dict, list[tuple[int, int]]]:
    indices = [(eid, step) for road in raw.ROADS for sign, n in (("positive", 16), ("nonpositive", 48))
               for eid, step in (pools[road][sign][int(i)] for i in
                                 rng.choice(len(pools[road][sign]), size=n, replace=False))]
    if len(indices) != 256 or len(set(indices)) != 256 or any(partition(eid) != "adaptation" for eid, _ in indices):
        raise ValueError("batch includes excluded or reserved episode")
    rng.shuffle(indices)
    episodes = replay["episodes"]
    batch = {"obs": torch.from_numpy(np.stack([episodes[e]["observations"][s] for e, s in indices])),
             "action": torch.from_numpy(np.stack([episodes[e]["actions"][s] for e, s in indices])),
             "reward": torch.from_numpy(np.asarray([episodes[e]["rewards"][s] for e, s in indices],
                                                    np.float32)[:, None])}
    return batch, indices


def _predict(model, replay: dict, holdout: dict, device: str) -> dict:
    from haic.algorithms.tdmpc2.model import two_hot_inv

    model.eval()
    metrics = {}
    # torch.manual_seed touches CPU and every CUDA generator, even for CPU inference.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []), torch.inference_mode():
        for road in raw.ROADS:
            for sign in ("positive", "nonpositive"):
                errors = []
                indices = holdout[road][sign]
                for start in range(0, len(indices), 256):
                    group = indices[start:start + 256]
                    torch.manual_seed(834 + road + start)
                    obs = torch.from_numpy(np.stack([replay["episodes"][e]["observations"][s]
                                                      for e, s in group])).to(device)
                    action = torch.from_numpy(np.stack([replay["episodes"][e]["actions"][s]
                                                         for e, s in group])).to(device)
                    target = np.asarray([replay["episodes"][e]["rewards"][s] for e, s in group], np.float64)
                    output = two_hot_inv(model.reward(model.encode(obs, None), action, None), model.cfg)
                    if output.shape != (len(group), 1) or not torch.isfinite(output).all():
                        raise ValueError("nonfinite true-latent reward prediction")
                    errors.extend((output[:, 0].cpu().numpy().astype(np.float64) - target).tolist())
                values = np.asarray(errors, np.float64)
                if len(values) != len(indices) or not np.isfinite(values).all():
                    raise ValueError("holdout prediction count mismatch")
                metrics[f"{road}/{sign}"] = {"count": len(values), "signed_bias": float(values.mean()),
                                               "mae": float(np.abs(values).mean())}
    return metrics


def holdout_gate(before: dict, after: dict) -> dict:
    checks = {}
    for road in raw.ROADS:
        old, new = before[f"{road}/positive"], after[f"{road}/positive"]
        old_neg, new_neg = before[f"{road}/nonpositive"], after[f"{road}/nonpositive"]
        count_ok = old["count"] == new["count"] and old_neg["count"] == new_neg["count"] and old["count"] >= 100 and old_neg["count"] >= 100
        checks[str(road)] = {"adequate_counts": count_ok,
                             "positive_mae_improvement": old["mae"] - new["mae"],
                             "positive_absolute_signed_bias_improvement": abs(old["signed_bias"]) - abs(new["signed_bias"]),
                             "nonpositive_mae_worsening": new_neg["mae"] - old_neg["mae"]}
        checks[str(road)]["qualifies"] = (count_ok and old["signed_bias"] < 0
                                            and new["signed_bias"] - old["signed_bias"] >= 0.10
                                            and checks[str(road)]["positive_mae_improvement"] >= 0.10
                                            and checks[str(road)]["positive_absolute_signed_bias_improvement"] >= 0.10)
    passed = (all(row["adequate_counts"] and row["nonpositive_mae_worsening"] <= 0.05 for row in checks.values())
              and sum(row["qualifies"] for row in checks.values()) >= 3)
    return {"status": "PASS" if passed else "FAIL", "by_road": checks,
            "qualifying_roads": sum(row["qualifies"] for row in checks.values())}


def _journal(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load(root: Path, pre: dict, protocol_sha256: str):
    from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel
    from haic.algorithms.tdmpc2.reward_head_adapt import RewardHeadAdapter

    # Rehash full sources, protocols, ledgers and both checkpoints before either pickle.
    preflight(root, protocol_sha256, device=pre["schema"]["device"], allow_output=True)
    states = []
    for pair in ((raw.CHECKPOINT, raw.CHECKPOINT_SHA), NEW_CHECKPOINT):
        with _pinned(root, pair).open("rb") as stream:
            if raw._digest_stream(stream) != pair[1]:
                raise ValueError("trusted source checkpoint changed before deserialization")
            stream.seek(0)
            states.append(torch.load(stream, map_location="cpu", weights_only=False))
    old, new = states
    from scripts import diagnose_tdmpc2_h5_branches as binder
    binder.bind_replay(old, raw._json(_pinned(root, ("experiments/tdmpc2-long-reused-train-v2.json",
                                             raw.SOURCE_PROTOCOL_SHA)).read_bytes()),
                       pre["_episodes"], _pinned(root, (f"{raw.RUN}/steps.jsonl", raw.STEPS_SHA)))
    overshoot._optimizer(old.get("optim"), 100354)
    overshoot._optimizer(old.get("pi_optim"), 100354)
    context = pre["_context"]
    if (new.get("format") != context["training"]["format"]
            or new.get("protocol_sha256") != overshoot.TRAIN_PROTOCOL_SHA
            or new.get("source_sha256") != context["training"]["source_sha256"]
            or any(new.get(k) != context["last"][k] for k in ("target", "decisions", "updates", "episodes"))
            or new.get("step_ledger_sha256") != context["last"]["step_ledger_sha256"]
            or new.get("training_ledger_sha256_before_checkpoint") != context["last"]["training_ledger_sha256_before_checkpoint"]
            or new.get("resume_supported") is not False):
        raise ValueError("overshoot checkpoint metadata differs from complete source")
    overshoot._bind_checkpoint_replay(root, new, {"spec": {"training_source": context["refs"]},
                                                 "episodes": overshoot._ledgers(root, {"training_source": context["refs"]},
                                                                                  context["training"], context["result"])})
    overshoot._optimizer(new.get("optim"), 100159)
    overshoot._optimizer(new.get("pi_optim"), 100159)
    overshoot._overshoot_probe(new, context["training"], context["last"])
    model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    learner = new.get("learner")
    if not isinstance(learner, dict) or not isinstance(learner.get("q_scale"), torch.Tensor):
        raise ValueError("overshoot model missing learner weights")
    model.load_state_dict({k.removeprefix("model."): v for k, v in learner.items()
                           if k.startswith("model.")}, strict=True)
    model.to(pre["schema"]["device"])
    adapter = RewardHeadAdapter(model)
    if adapter.optim.param_groups[0]["lr"] != 3e-5 or adapter.optim.param_groups[0]["weight_decay"] != 0:
        raise ValueError("reward adapter optimizer differs from frozen objective")
    return adapter, old["replay"]


def execute(root: Path, protocol_sha256: str, *, device: str = "cpu", clock=time.monotonic) -> dict:
    root = Path(root).resolve(strict=True)
    pre = preflight(root, protocol_sha256, device=device)
    output = root / OUTPUT
    if output.exists() or output.is_symlink():
        raise ValueError("exclusive output already exists; partial runs cannot resume")
    output.mkdir(exist_ok=False)
    journal = output / "journal.jsonl"
    completed = 0
    started = clock()
    try:
        _journal(journal, {"event": "start", "protocol_sha256": protocol_sha256,
                           "source": pre["schema"]["raw_source"], "overshoot_source": pre["schema"]["overshoot_source"],
                           "throughput_benchmark": pre["schema"]["throughput_benchmark"],
                           "source_sha256": pre["schema"]["source_sha256"], "updates": 0,
                           "intent": "no_resume_512_head_only_updates", "environment_resets": 0})
        adapter, replay = _load(root, pre, protocol_sha256)
        if adapter.updates != 0:
            raise ValueError("reward-head adapter must start with zero optimizer updates")
        initial_state = adapter.snapshot()
        rng = np.random.default_rng(20260929)
        before = _predict(adapter.model, replay, pre["_holdout"], device)
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired")
        metrics = []
        for update in range(1, 513):
            if clock() - started >= 1800:
                raise TimeoutError("hard 1800-second adaptation wall expired")
            batch, indices = sample_batch(replay, pre["_pools"], rng)
            _journal(journal, {"event": "update_intent", "update": update,
                               "sample_ids_sha256": _sha_body({"indices": indices}),
                               "protocol_sha256": protocol_sha256})
            report = adapter.step(batch)
            if adapter.updates != update:
                raise ValueError("reward-head optimizer update count differs from hard schedule")
            if not isinstance(report, dict) or not report or any(
                    not isinstance(value, (float, int)) or not math.isfinite(value) for value in report.values()):
                raise ValueError("nonfinite reward-head step metrics")
            completed = update
            metrics.append(report)
            if update % 64 == 0:
                _journal(journal, {"event": "update_chunk", "updates": completed,
                                   "source_sha256": pre["schema"]["source_sha256"],
                                   "protocol_sha256": protocol_sha256,
                                   "last_sample_ids_sha256": _sha_body({"indices": indices}),
                                   "last_metrics": report})
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired before holdout")
        after = _predict(adapter.model, replay, pre["_holdout"], device)
        parity = adapter.verify_only_reward_changed()
        final_state = adapter.snapshot()
        nonhead = {key: value for key, value in initial_state.items() if not key.startswith("_reward.")}
        changed_head = [key for key, value in initial_state.items()
                        if key.startswith("_reward.") and final_state[key] != value]
        if (parity is not True or not changed_head or initial_state.keys() != final_state.keys()
                or any(final_state[key] != value for key, value in nonhead.items())):
            raise ValueError("non-reward weights/buffers changed or reward head did not change")
        checked = preflight(root, protocol_sha256, device=device, allow_output=True)
        if checked["schema"] != pre["schema"] or completed != 512 or clock() - started >= 1800:
            raise ValueError("source changed or hard update/wall cap exceeded")
        checkpoint = output / "adapted-model.pt"
        with checkpoint.open("xb") as stream:
            torch.save({"format": "haic-tdmpc2-head-only-model-v2",
                        "protocol_sha256": protocol_sha256, "source": pre["schema"]["overshoot_source"],
                        "throughput_benchmark": pre["schema"]["throughput_benchmark"],
                        "source_sha256": pre["schema"]["source_sha256"],
                        "updates": completed, "seed": 20260929,
                        "model": {key: value.detach().cpu() for key, value in adapter.model.state_dict().items()},
                        "optimizer": adapter.optim.state_dict(),
                        "sampler_rng": rng.bit_generator.state,
                        "nonhead_bitwise_parity": True}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired during checkpoint write")
        result = {"format": "haic-tdmpc2-head-only-result-v2", "status": "complete",
                  "protocol": ref(PROTOCOL, protocol_sha256), "source": pre["schema"],
                  "throughput_benchmark": pre["schema"]["throughput_benchmark"],
                  "checkpoint": ref(f"{OUTPUT}/adapted-model.pt", raw._digest(checkpoint)),
                  "optimizer_updates": completed, "labels_per_update": 256,
                  "training_metrics": metrics,
                  "nonhead_bitwise_parity": {"passed": True, "sha256_before": nonhead,
                                             "sha256_after": {key: final_state[key] for key in nonhead},
                                             "changed_reward_keys": changed_head},
                  "before": before, "after": after,
                  "gate": holdout_gate(before, after), "elapsed_seconds": clock() - started,
                  "environment_resets": 0, "policy_or_official_claim": False,
                  "scope": "adaptation_excluded_logged_TRAIN_same_pixels_actions_true_encoded_latents"}
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired before result")
        _journal(journal, {"event": "complete", "updates": 512,
                           "checkpoint_sha256": result["checkpoint"]["sha256"],
                           "protocol_sha256": protocol_sha256})
        result["journal_sha256"] = raw._digest(journal)
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired before result write")
        with (output / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return result
    except BaseException as exc:
        row = {"event": "partial", "reason": type(exc).__name__, "updates": completed,
               "protocol_sha256": protocol_sha256, "source_sha256": pre["schema"]["source_sha256"],
               "environment_resets": 0, "resume_supported": False}
        try:
            _journal(journal, row)
        except BaseException:
            with (output / "failure.json").open("x", encoding="utf-8") as stream:
                json.dump(row, stream, sort_keys=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--benchmark-sha256", help="independent synthetic full-step benchmark receipt")
    args = parser.parse_args()
    if args.execute and (not args.protocol_sha256 or args.benchmark_sha256):
        parser.error("--execute requires frozen --protocol-sha256; benchmark must be pinned by protocol")
    if args.preflight:
        output = preflight(ROOT, args.protocol_sha256, benchmark_sha256=args.benchmark_sha256,
                           device=args.device)
        print(json.dumps({key: output[key] for key in ("status", "schema", "resources",
                                                    "environment_resets", "torch_load_calls")}, sort_keys=True))
    else:
        result = execute(ROOT, args.protocol_sha256, device=args.device)
        print(json.dumps({"status": result["status"], "gate": result["gate"],
                          "checkpoint": result["checkpoint"], "optimizer_updates": 512}, sort_keys=True))


if __name__ == "__main__":
    main()
