"""CPU-only contract tests for the synthetic benchmark; no GPU or environment."""

import builtins
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_tdmpc2_reward_head as bench


def test_generated_balanced_fixture_never_loads_replay_environment_or_checkpoint(monkeypatch):
    original = builtins.__import__

    def forbid(name, *args, **kwargs):
        if "haic_env" in name or "replay" in name or "local_runner" in name:
            pytest.fail("forbidden environment/replay import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid)
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("checkpoint load"))
    batch, fixture = bench._fixture()
    second, repeated = bench._fixture()
    assert fixture == repeated and fixture["synthetic_only"] is True
    assert fixture["counts"] == {str(road): {"positive": 16, "nonpositive": 48} for road in bench.ROADS}
    assert fixture["obs_shape"] == [4, 64, 64] and fixture["batch_size"] == 256
    assert fixture["num_bins"] == 101 and fixture["action_dim"] == 3
    assert batch["obs"].dtype == torch.uint8 and batch["obs"].shape == (256, 4, 64, 64)
    assert batch["action"].dtype == batch["reward"].dtype == torch.float32
    assert batch["action"].shape == (256, 3) and batch["reward"].shape == (256, 1)
    assert all(torch.equal(batch[key], second[key]) for key in batch)
    assert bool((batch["action"].abs() <= 1).all())


def test_clock_synchronizes_full_adapter_step_and_holdout_calls():
    events = []

    def sync(device):
        assert str(device) == "cuda:0"
        events.append("sync")

    def clock():
        events.append("clock")
        return len(events) / 4

    def operation():
        events.append("step")
        return {"loss": .1}

    assert bench._timed(operation, bench.WARMUP, torch.device("cuda:0"),
                        clock=clock, synchronize=sync) == [.75] * 32
    assert events == ["sync", "clock", "step", "sync", "clock"] * 32
    events.clear()
    assert bench._timed(operation, bench.TIMED, torch.device("cuda:0"),
                        clock=clock, synchronize=sync) == [.75] * 128
    assert events == ["sync", "clock", "step", "sync", "clock"] * 128


@pytest.mark.parametrize("values", [[.1] * 127, [.1] * 127 + [math.nan],
                                           [.1] * 120 + [.4] * 8,
                                           [.1] * 64 + [.12] * 64])
def test_timing_noise_and_incomplete_stream_fail_closed(values):
    with pytest.raises(ValueError):
        bench._summary(values, 128)
    report = bench._summary([.1] * 128, 128)
    assert report["p95_seconds"] == .1 and report["half_drift_fraction"] == 0
    assert len(report["seconds"]) == 128


def test_forecast_exact_hard_boundary_and_reserves():
    step = {"mean_seconds": (bench.WALL_CAP_SECONDS - bench.IO_RESERVE_SECONDS) / bench.UPDATES}
    infer = {"mean_seconds": 0.01}
    at_limit = bench._forecast(step, infer, 123, 456, 789)
    assert not at_limit["passed"]
    assert at_limit["forecast_seconds"] > 1800
    assert at_limit["measured_holdout_seconds"] == .86
    assert at_limit["measured_peak_rss_bytes"] == 123
    assert not bench._forecast({"mean_seconds": 0.01}, {"mean_seconds": math.nan}, 123, 456, 789)["passed"]
    assert bench._forecast({"mean_seconds": .1}, {"mean_seconds": .1}, 123, 456, 789)["forecast_seconds"] == pytest.approx(659.8)


def test_gpu_host_pid_delta_not_container_pid_or_own_mid_util(monkeypatch):
    rows = ["3639616, 344"]
    util = ["7"]
    monkeypatch.setattr(bench, "_nvidia_smi", lambda query: rows if query.startswith("compute-apps") else util)
    baseline = bench._gpu_snapshot()
    host_pid = str(bench.os.getpid() + 1_000_000_000)
    rows.append(f"{host_pid}, 300")
    assert bench._check_gpu_context(baseline, bench._gpu_snapshot()) == host_pid
    util[0] = "85"
    assert bench._check_gpu_context(baseline, bench._gpu_snapshot(require_idle=False), host_pid) == host_pid
    with pytest.raises(ValueError, match="not idle"):
        bench._gpu_snapshot()
    rows.append("9999999, 100")
    with pytest.raises(ValueError, match="contention"):
        bench._check_gpu_context(baseline, bench._gpu_snapshot(require_idle=False), host_pid)
    rows.pop()
    rows[0] = "3639616, 500"
    with pytest.raises(ValueError, match="baseline GPU context changed"):
        bench._check_gpu_context(baseline, bench._gpu_snapshot(require_idle=False), host_pid)


@pytest.mark.parametrize("rows", [["N/A, 100"], ["2, N/A"], ["2, 10", "2, 10"]])
def test_ambiguous_gpu_telemetry_fails(monkeypatch, rows):
    monkeypatch.setattr(bench, "_nvidia_smi", lambda query: rows if query.startswith("compute-apps") else ["0"])
    with pytest.raises(ValueError, match="ambiguous"):
        bench._gpu_snapshot()


def test_raw_resource_measurement_and_additive_reserves(monkeypatch, tmp_path):
    original = Path.read_text
    values = {"memory.max": str(64 * bench.GIB), "memory.current": str(30 * bench.GIB),
              "meminfo": "MemTotal: 999999999 kB\nMemAvailable: 40000000 kB\n"}

    def read(path, *args, **kwargs):
        if path.name in values and str(path).startswith(("/sys/fs/cgroup/", "/proc/")):
            return values[path.name]
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(bench.os, "statvfs", lambda _: SimpleNamespace(f_bavail=32 * bench.GIB, f_frsize=1))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (6 * bench.GIB, 8 * bench.GIB))
    snapshot = bench._raw_resources(torch.device("cuda:0"), tmp_path)
    bench._check_resources(snapshot, snapshot, peak_rss=2 * bench.GIB, output_bytes=1 * bench.GIB,
                           peak_reserved=2 * bench.GIB)
    with pytest.raises(ValueError, match="headroom"):
        bench._check_resources({**snapshot, "cgroup_available_bytes": 24 * bench.GIB},
                               peak_rss=1)
    with pytest.raises(ValueError, match="disk"):
        bench._check_resources({**snapshot, "disk_available_bytes": 24 * bench.GIB},
                               output_bytes=1)
    with pytest.raises(ValueError, match="measured peak CUDA"):
        bench._check_resources(snapshot, {**snapshot, "cuda_free_bytes": 3 * bench.GIB},
                               peak_reserved=2 * bench.GIB)
    values["memory.max"] = "max"
    with pytest.raises(ValueError, match="unavailable"):
        bench._raw_resources(torch.device("cuda:0"), tmp_path)


def test_source_pin_and_cuda_absence_fail_before_fixture(monkeypatch):
    script_sha = bench._sha(bench.ROOT / bench.SELF)
    sources = bench._sources(bench.ROOT)
    bench._check_sources(sources, script_sha, bench.ROOT)
    with pytest.raises(ValueError, match="SHA mismatch"):
        bench._check_sources(sources, "0" * 64, bench.ROOT)
    monkeypatch.setattr(bench, "_gpu_snapshot", lambda: {"process_memory_mib": {}, "gpu_utilization_percent": 0})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench, "_fixture", lambda: pytest.fail("fixture before CUDA gate"))
    result = bench.benchmark(script_sha)
    assert result["status"] == "FAIL"
    assert result["environment_resets"] == result["real_replay_reads"] == result["synthetic_only_optimizer_updates"] == 0
    assert "CUDA unavailable" in str(result["failure_reasons"])
    assert result["source_sha256_before"] == result["source_sha256_after"] == sources
    assert result["body_sha256"] == bench._seal({k: v for k, v in result.items() if k != "body_sha256"})["body_sha256"]


def test_receipt_exclusive_fsynced_status_and_no_overwrite(monkeypatch, tmp_path, capsys):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "benchmark", lambda expected: bench._seal({"format": bench.FORMAT,
        "status": "FAIL", "environment_resets": 0, "real_replay_reads": 0,
        "synthetic_only_optimizer_updates": 0, "failure_reasons": ["fake CUDA unavailable"]}))
    path = tmp_path / bench.OUTPUT
    assert bench.main(["--expected-script-sha256", "0" * 64]) == 1
    original = path.read_bytes()
    assert original == capsys.readouterr().out.encode()
    assert json.loads(original)["body_sha256"] == bench._seal({k: v for k, v in json.loads(original).items()
                                                                if k != "body_sha256"})["body_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
    assert bench.main(["--expected-script-sha256", "0" * 64]) == 1
    assert "never overwrite" in capsys.readouterr().out
    assert path.read_bytes() == original
    assert path.stat().st_mode & 0o777 == 0o600


def test_symlink_receipt_fails_without_writing(monkeypatch, tmp_path, capsys):
    (tmp_path / "runs").mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    (tmp_path / bench.OUTPUT).symlink_to(target)
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "benchmark", lambda expected: pytest.fail("executed benchmark"))
    assert bench.main(["--expected-script-sha256", "0" * 64]) == 1
    assert target.read_bytes() == b"unchanged"
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"


def test_mocked_full_cuda_pass_and_external_context_fail_closed(monkeypatch):
    baseline = {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 3}
    active = {"process_memory_mib": {"3639616": 344, "9999999": 256},
              "gpu_utilization_percent": 84}
    snapshots = [baseline, active, active, active]
    monkeypatch.setattr(bench, "_gpu_snapshot", lambda **kwargs: snapshots.pop(0))
    monkeypatch.setattr(bench, "_check_sources", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "stub GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (8, 9))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: bench.GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: bench.GIB)
    monkeypatch.setattr(bench.torch, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(bench.torch, "manual_seed", torch.default_generator.manual_seed)
    original_to = torch.Tensor.to

    def fake_to(tensor, *args, **kwargs):
        if args and str(args[0]) == "cuda:0":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    monkeypatch.setattr(bench, "_raw_resources", lambda *args: {
        "host_available_bytes": 40 * bench.GIB,
        "cgroup_available_bytes": 40 * bench.GIB,
        "disk_available_bytes": 40 * bench.GIB,
        "cuda_free_bytes": 6 * bench.GIB,
        "cuda_total_bytes": 8 * bench.GIB,
    })
    monkeypatch.setattr(bench.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=512 * 1024))

    class FakeModel:
        total_params = 5_000_000
        cfg = SimpleNamespace()

        def __init__(self, cfg):
            assert cfg.action_dim == 3 and cfg.obs_shape["rgb"] == (4, 64, 64)

        def to(self, device):
            return self

        def state_dict(self):
            return {"_reward.weight": torch.ones(2)}

        def encode(self, obs):
            assert obs.shape == (256, 4, 64, 64)
            return torch.ones(256, 512)

        def reward(self, z, action):
            return torch.zeros(256, 101)

    class FakeAdapter:
        def __init__(self, model):
            self.updates = 0
            self.optim = SimpleNamespace(param_groups=[{"lr": 3e-5}])

        def step(self, batch):
            self.updates += 1
            return {"loss": 0.5, "grad_norm": 1.0}

        def verify_only_reward_changed(self):
            return self.updates == bench.WARMUP + bench.TIMED

    monkeypatch.setattr(bench, "WorldModel", FakeModel)
    monkeypatch.setattr(bench, "RewardHeadAdapter", FakeAdapter)
    monkeypatch.setattr(bench, "two_hot_inv", lambda logits, cfg: torch.ones(256, 1))
    def timed(operation, count, device):
        for _ in range(count):
            operation()
        return [0.1] * count

    monkeypatch.setattr(bench, "_timed", timed)
    result = bench.benchmark("0" * 64)
    assert result["status"] == "PASS", result.get("failure_reasons")
    assert result["synthetic_only_optimizer_updates"] == 160
    assert result["timing"]["timed_updates"] == 128
    assert result["forecast"]["planned_holdout_batches"] == 86
    assert result["timing"]["holdout_batches"] == 86
    assert result["forecast"]["source_checkpoint_ledger_receipt_io_reserve_seconds"] == 600
    assert result["resources"]["gpu_context_telemetry"]["owner_host_pid"] == "9999999"
    assert result["source_sha256_before"] == result["source_sha256_after"]

    snapshots[:] = [baseline, active, active, {"process_memory_mib": {
        "3639616": 344, "9999999": 256, "7777777": 128}, "gpu_utilization_percent": 10}]
    failed = bench.benchmark("0" * 64)
    assert failed["status"] == "FAIL"
    assert "external GPU contention" in str(failed["failure_reasons"])
    assert failed["body_sha256"] == bench._seal({k: v for k, v in failed.items()
                                                 if k != "body_sha256"})["body_sha256"]
