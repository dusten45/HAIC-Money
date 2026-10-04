"""CPU-only synthetic/stub checks; never create an environment or run a CUDA update."""

import builtins
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import benchmark_tdmpc2_reward_overshoot as bench

GPU_SNAPSHOT = bench._gpu_snapshot


def test_generated_replay_fixture_matches_seeded_h3_and_has_valid_suffix(monkeypatch):
    original = builtins.__import__

    def forbid_environment(name, *args, **kwargs):
        if "haic_env" in name or "env_wrapper" in name or "local_runner" in name:
            pytest.fail("benchmark attempted to import a HAIC environment")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_environment)
    base_replay, variant_replay, fixture = bench._make_replays()
    base = base_replay.sample(256)
    extended = variant_replay.sample(256)
    assert fixture["synthetic_only"] is True
    assert fixture["batch_size"] == 256 and fixture["horizon"] == 3
    assert fixture["synthetic_replay_episodes"] == 307
    assert fixture["synthetic_replay_transitions"] == 434
    assert fixture["eligible_h3_windows"] == 126
    assert fixture["replay_capacity"] == 120000
    assert fixture["valid_step5"] >= .95 * 256
    assert fixture["valid_step4"] >= fixture["valid_step5"]
    assert all(torch.equal(value, extended[name]) for name, value in base.items())
    assert fixture["proof_h3_sample_sha256"] == bench._batch_sha(base)
    assert fixture["proof_extended_sample_sha256"] == bench._batch_sha(extended)
    assert bench._rng_sha(base_replay) == bench._rng_sha(variant_replay)
    assert bench._batch_sha(base) != bench._batch_sha(extended)
    before = bench._replay_sha(variant_replay)
    variant_replay._episodes[0].observations[0][0, 0, 0] ^= 1
    assert bench._replay_sha(variant_replay) != before


def test_stub_clock_synchronizes_every_warmup_and_timed_optimizer_update():
    events = []

    class Learner:
        def update(self, batch):
            events.append("update")
            return {"loss": 0.75}

    def synchronize(device):
        assert str(device) == "cuda:0"
        events.append("sync")

    def clock():
        events.append("clock")
        return len(events) / 4

    for count in (bench.WARMUP, bench.TIMED):
        events.clear()
        durations = bench._timed_updates(
            Learner(), {}, count, torch.device("cuda:0"), clock=clock,
            synchronize=synchronize,
        )
        assert durations == [0.75] * count
        assert events == ["sync", "clock", "update", "sync", "clock"] * count

    with pytest.raises(ValueError, match="nonfinite"):
        bench._timed_updates(SimpleNamespace(update=lambda _: {"loss": math.nan}), {}, 1,
                             torch.device("cpu"), clock=clock, synchronize=lambda _: None)


@pytest.mark.parametrize("samples", [
    [0.1] * 127,
    [0.1] * 127 + [math.nan],
    [0.1] * 121 + [1.0] * 7,
    [0.1] * 64 + [0.125] * 64,
])
def test_noisy_nonfinite_or_incomplete_update_streams_fail_closed(samples):
    with pytest.raises(ValueError):
        bench._summary(samples)
    result = bench._summary([0.1] * 128)
    assert result["mean_seconds"] == pytest.approx(.1)
    assert result["median_seconds"] == pytest.approx(.1)
    assert len(result["update_wall_seconds"]) == 128


def test_forecast_strict_exact_rational_border_and_positive_delta_only():
    threshold_delta = (Fraction(bench.FORECAST_LIMIT) - Fraction(bench.BASELINE_SECONDS)) / bench.BASELINE_UPDATES
    base = Fraction(1, 100)
    assert bench._forecast(base, base + threshold_delta)["forecast_seconds"] == 20600.0
    assert bench._forecast(base, base + threshold_delta)["passed"] is False
    assert bench._forecast(base, base + threshold_delta - Fraction(1, 10**20))["passed"] is True
    assert bench._forecast(base, base + threshold_delta + Fraction(1, 10**20))["passed"] is False
    faster = bench._forecast(Fraction(2, 100), Fraction(1, 100))
    assert faster["positive_update_delta_seconds"] == 0
    assert faster["forecast_seconds"] == float(bench.BASELINE_SECONDS)
    with pytest.raises(ValueError, match="nonfinite"):
        bench._forecast(float("nan"), 0.01)


def test_raw_cgroup_disk_and_gpu_reserves_require_measured_headroom(monkeypatch, tmp_path):
    original_read = Path.read_text
    values: dict[str, int | str] = {"memory.max": 48 * bench.GIB, "memory.current": 24 * bench.GIB}

    def read_text(path, *args, **kwargs):
        if path.name in values and str(path).startswith("/sys/fs/cgroup/"):
            return str(values[path.name])
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    def disk(path):
        assert path == tmp_path / "runs"
        return SimpleNamespace(f_bavail=24 * bench.GIB, f_frsize=1)

    monkeypatch.setattr(bench.os, "statvfs", disk)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (4 * bench.GIB, 8 * bench.GIB))
    before = bench._raw_resources(torch.device("cuda:0"), tmp_path)
    assert before["cgroup_available_bytes"] == before["disk_available_bytes"] == 24 * bench.GIB
    bench._check_resources(before, before, 2 * bench.GIB)
    with pytest.raises(ValueError, match="peak CUDA"):
        bench._check_resources(before, before, 2 * bench.GIB + 1)
    with pytest.raises(ValueError, match="peak CUDA"):
        bench._check_resources(before, {**before, "cuda_free_bytes": 2 * bench.GIB},
                               2 * bench.GIB)
    with pytest.raises(ValueError, match="headroom"):
        bench._check_resources({**before, "cgroup_available_bytes": 24 * bench.GIB - 1})
    with pytest.raises(ValueError, match="headroom"):
        bench._check_resources({**before, "disk_available_bytes": 24 * bench.GIB - 1})
    with pytest.raises(ValueError, match="reserve"):
        bench._check_resources({**before, "cuda_free_bytes": 2 * bench.GIB - 1})
    with pytest.raises(ValueError, match="headroom"):
        bench._check_resources(before, {**before, "disk_available_bytes": 24 * bench.GIB - 1},
                               2 * bench.GIB)
    values["memory.max"] = "max"
    with pytest.raises(ValueError, match="telemetry unavailable"):
        bench._raw_resources(torch.device("cuda:0"), tmp_path)


def test_host_pid_namespace_snapshot_accepts_small_stable_baseline(monkeypatch):
    host_pid = bench.os.getpid() + 1_000_000_000
    rows = ["3639616, 344"]
    monkeypatch.setattr(bench, "_nvidia_smi", lambda query: rows if query.startswith("compute-apps") else ["7"])
    baseline = bench._gpu_snapshot()
    assert baseline == {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 7}
    rows = ["3639616, 344", f"{host_pid}, 300"]
    allocated = bench._gpu_snapshot()
    assert str(host_pid) != str(bench.os.getpid())
    assert bench._check_gpu_context(baseline, allocated) == str(host_pid)
    rows = ["3639616, 350", f"{host_pid}, 450"]
    assert bench._check_gpu_context(baseline, bench._gpu_snapshot(), str(host_pid)) == str(host_pid)


@pytest.mark.parametrize("rows,util", [
    (["N/A, 344"], ["0"]), (["3639616, N/A"], ["0"]),
    (["3639616, 344", "3639616, 344"], ["0"]),
    (["3639616, 344"], ["11"]), (["3639616, 344"], ["0", "0"]),
])
def test_incomplete_ambiguous_or_high_utilization_gpu_telemetry_fails(monkeypatch, rows, util):
    monkeypatch.setattr(bench, "_nvidia_smi", lambda query: rows if query.startswith("compute-apps") else util)
    with pytest.raises(ValueError, match="telemetry|not idle|attributed"):
        bench._gpu_snapshot()


def test_context_must_be_single_stable_new_host_pid_with_stable_baseline():
    baseline = {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 7}

    def check(processes, owner=None):
        return bench._check_gpu_context(baseline, {"process_memory_mib": processes,
                                                   "gpu_utilization_percent": 0}, owner)

    assert check({"3639616": 344, "3725938": 300}) == "3725938"
    for processes in (
        {"3639616": 344},
        {"3639616": 344, "3725938": 300, "9999999": 300},
        {"3725938": 300},
        {"3639616": 473, "3725938": 300},
    ):
        with pytest.raises(ValueError):
            check(processes, "3725938")
    with pytest.raises(ValueError, match="distinguished"):
        check({"3639616": 344, "9999999": 300}, "3725938")
    with pytest.raises(ValueError, match="512 MiB"):
        bench._check_gpu_context({"process_memory_mib": {"3639616": 513}},
                                 {"process_memory_mib": {"3639616": 513, "3725938": 300}})


def test_source_pin_and_cuda_absence_return_sealed_fail_without_env(monkeypatch):
    sha = bench._sha(bench.ROOT / bench.SELF)
    before = bench._sources(bench.ROOT)
    assert before[bench.SELF] == sha
    bench._check_sources(before, sha, bench.ROOT)
    with pytest.raises(ValueError, match="SHA mismatch"):
        bench._check_sources(before, "0" * 64, bench.ROOT)
    monkeypatch.setattr(bench, "_gpu_snapshot", lambda: {"process_memory_mib": {},
                                                        "gpu_utilization_percent": 0})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench, "_make_replays", lambda: pytest.fail("CUDA absent but fixture constructed"))
    result = bench.benchmark(sha)
    assert result["status"] == "FAIL" and result["environment_resets"] == 0
    assert "CUDA unavailable" in str(result["failure_reasons"])
    assert result["source_sha256_before"] == result["source_sha256_after"] == before
    assert result["body_sha256"] == bench._seal({k: v for k, v in result.items() if k != "body_sha256"})["body_sha256"]


def test_cli_cuda_absence_prints_only_sealed_failure_json(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench, "_gpu_snapshot", lambda: {"process_memory_mib": {},
                                                        "gpu_utilization_percent": 0})
    sha = bench._sha(bench.ROOT / bench.SELF)
    assert bench.main(["--expected-script-sha256", sha]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    body = json.loads(captured.out)
    assert body["status"] == "FAIL" and body["environment_resets"] == 0
    assert body["body_sha256"] == bench._seal({
        k: v for k, v in body.items() if k != "body_sha256"
    })["body_sha256"]


def _stubbed_run(monkeypatch, tmp_path, *, tamper=None, variant_seconds=0.11,
                 gpu_snapshots=None, receipt=False):
    for name in [*bench.SOURCE_SHA256, bench.SELF]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((bench.ROOT / name).read_bytes())
    script_sha = bench._sha(tmp_path / bench.SELF)
    monkeypatch.setattr(bench, "_check_sources", lambda before, expected, root: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "stub-GPU")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 2 * bench.GIB)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _: torch.tensor([1], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *args: None)
    baseline = {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 7}
    stable = {"process_memory_mib": {"3639616": 344, "3725938": 300},
              "gpu_utilization_percent": 7}
    snapshots = list(gpu_snapshots or (baseline, stable, stable, stable))
    stages = []
    read_snapshot = GPU_SNAPSHOT

    def snapshot(*, require_idle=True):
        stages.append("snapshot")
        expected = snapshots.pop(0)
        rows = [f"{pid}, {memory}" for pid, memory in expected["process_memory_mib"].items()]
        monkeypatch.setattr(bench, "_nvidia_smi", lambda query: (
            rows if query.startswith("compute-apps") else
            [str(expected["gpu_utilization_percent"])]
        ))
        return read_snapshot(require_idle=require_idle)

    def allocate(device):
        stages.append("allocate")
        return object()

    monkeypatch.setattr(bench, "_gpu_snapshot", snapshot)
    monkeypatch.setattr(bench, "_allocate_cuda_context", allocate)
    monkeypatch.setattr(bench, "_raw_resources", lambda device, root: {
        "cgroup_available_bytes": 25 * bench.GIB, "disk_available_bytes": 25 * bench.GIB,
        "cuda_free_bytes": 6 * bench.GIB, "cuda_total_bytes": 8 * bench.GIB,
    })
    replays = []
    make_replays = bench._make_replays

    def make():
        base, variant, fixture = make_replays()
        replays.extend((base, variant))
        return base, variant, fixture

    monkeypatch.setattr(bench, "_make_replays", make)
    original = object()
    variant = object()
    monkeypatch.setattr(bench, "_make_learners", lambda _: (original, variant, 5_000_000))
    invoked = []

    def fake_updates(learner, replay, count, device):
        invoked.append((learner, count))
        if tamper is not None and len(invoked) == 3:
            tamper(tmp_path, replays)
        replay.sample(1)  # No environment or CUDA, but exercise independent sampler state.
        return [0.1 if learner is original else variant_seconds] * count

    monkeypatch.setattr(bench, "_timed_updates", fake_updates)
    if receipt:
        (tmp_path / "runs").mkdir(exist_ok=True)
        monkeypatch.setattr(bench, "ROOT", tmp_path)
        output = "runs/tdmpc2-reward-overshoot-throughput-context-fail.json"
        code = bench.main(["--expected-script-sha256", script_sha, "--output", output])
        result = json.loads((tmp_path / output).read_bytes())
        assert code == (0 if result["status"] == "PASS" else 1)
    else:
        result = bench.benchmark(script_sha, root=tmp_path)
    assert stages == ["snapshot", "allocate", "snapshot", "snapshot", "snapshot"]
    assert 6 <= len(invoked) <= 10  # Midpoint contention may stop the second four-block wave.
    assert result["environment_resets"] == 0 and result["real_replay_reads"] == 0
    return result


def test_full_stubbed_pass_has_pinned_inputs_telemetry_and_mean_forecast(monkeypatch, tmp_path):
    result = _stubbed_run(monkeypatch, tmp_path)
    assert result["status"] == "PASS"
    assert result["fixture"]["trainable_parameters"] == 5_000_000
    assert result["timing"]["warmup_updates_each"] == 32
    assert result["timing"]["timed_updates_each"] == 128
    assert result["timing"]["scope"] == "full_update_with_replay_sample"
    assert result["timing"]["synchronization"] == "torch.cuda.synchronize per update"
    assert result["timing"]["base"]["mean_seconds"] == pytest.approx(.1)
    assert result["timing"]["variant"]["mean_seconds"] == pytest.approx(.11)
    assert result["forecast"]["forecast_seconds"] == pytest.approx(17459.373508695047 + 100354 * .01)
    assert result["source_sha256_before"] == result["source_sha256_after"]
    assert result["input_sha256_before"] == result["input_sha256_after"]
    assert result["input_sha256_before"]["base"] == result["input_sha256_before"]["variant"]
    assert result["sample_rng_sha256_before"]["base"] == result["sample_rng_sha256_before"]["variant"]
    assert result["sample_rng_sha256_after"]["base"] == result["sample_rng_sha256_after"]["variant"]
    assert result["resources"]["max_cuda_allocated_bytes"] == 2 * bench.GIB
    assert set(result) == {"format", "status", "environment_resets", "real_replay_reads",
                           "expected_script_sha256", "output_path", "source_sha256_before",
                           "source_sha256_after", "input_sha256_before", "input_sha256_after",
                           "sample_rng_sha256_before", "sample_rng_sha256_after", "runtime",
                           "fixture", "timing", "resources", "forecast", "body_sha256"}
    telemetry = result["resources"]["gpu_context_telemetry"]
    assert telemetry["owner_host_pid"] == "3725938"
    assert set(telemetry["snapshots"]) == {"before_cuda_allocation", "after_cuda_allocation", "mid", "final"}
    assert telemetry["snapshots"]["before_cuda_allocation"]["process_memory_mib"] == {"3639616": 344}


@pytest.mark.parametrize("index,snapshot,reason", [
    (2, {"process_memory_mib": {"3639616": 344, "3725938": 300, "9999999": 55},
         "gpu_utilization_percent": 0}, "distinguished"),
    (3, {"process_memory_mib": {"3639616": 480, "3725938": 300},
         "gpu_utilization_percent": 0}, "128 MiB"),
])
def test_mid_or_final_gpu_contention_writes_durable_fail_receipt(monkeypatch, tmp_path, capsys,
                                                                  index, snapshot, reason):
    baseline = {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 7}
    stable = {"process_memory_mib": {"3639616": 344, "3725938": 300},
              "gpu_utilization_percent": 7}
    stages = [baseline, stable, stable.copy(), stable.copy()]
    stages[index] = snapshot
    failure = _stubbed_run(monkeypatch, tmp_path, gpu_snapshots=stages, receipt=True)
    receipt = (tmp_path / "runs/tdmpc2-reward-overshoot-throughput-context-fail.json").read_bytes()
    assert failure["status"] == "FAIL"
    assert reason in str(failure["failure_reasons"])
    if snapshot["gpu_utilization_percent"] <= 10:
        assert "mid" in failure["resources"]["gpu_context_telemetry"]["snapshots"]
    assert receipt == capsys.readouterr().out.encode()
    assert failure["body_sha256"] == bench._seal({k: v for k, v in failure.items()
                                                  if k != "body_sha256"})["body_sha256"]


def test_own_mid_and_final_gpu_work_does_not_fail_idle_baseline(monkeypatch, tmp_path):
    baseline = {"process_memory_mib": {"3639616": 344}, "gpu_utilization_percent": 7}
    stable = {"process_memory_mib": {"3639616": 344, "3725938": 300},
              "gpu_utilization_percent": 7}
    active = {**stable, "gpu_utilization_percent": 82}
    result = _stubbed_run(monkeypatch, tmp_path, gpu_snapshots=[baseline, stable, active, active])
    assert result["status"] == "PASS"
    assert result["resources"]["gpu_context_telemetry"]["snapshots"]["mid"]["gpu_utilization_percent"] == 82


def test_source_tamper_after_updates_cannot_produce_pass(monkeypatch, tmp_path):
    def tamper(root, inputs):
        with (root / "haic/algorithms/tdmpc2/reward_overshoot.py").open("ab") as stream:
            stream.write(b"tampered")

    result = _stubbed_run(monkeypatch, tmp_path, tamper=tamper)
    assert result["status"] == "FAIL" and "source changed" in str(result["failure_reasons"])
    assert result["source_sha256_before"] != result["source_sha256_after"]


def test_input_tamper_after_updates_cannot_produce_pass(monkeypatch, tmp_path):
    def tamper(root, replays):
        replays[0]._episodes[0].observations[0][0, 0, 0] ^= 1

    result = _stubbed_run(monkeypatch, tmp_path, tamper=tamper)
    assert result["status"] == "FAIL" and "synthetic replay data mutated" in str(result["failure_reasons"])


def test_rng_divergence_or_excessive_measured_forecast_cannot_pass(monkeypatch, tmp_path):
    def extra_sample(root, replays):
        replays[1].sample(1)

    diverged = _stubbed_run(monkeypatch, tmp_path, tamper=extra_sample)
    assert diverged["status"] == "FAIL" and "RNG diverged" in str(diverged["failure_reasons"])
    excessive = _stubbed_run(monkeypatch, tmp_path, variant_seconds=.14)
    assert excessive["status"] == "FAIL" and excessive["forecast"]["passed"] is False
    assert "strict forecast" in str(excessive["failure_reasons"])


def test_optional_output_is_exclusive_fsynced_json_and_never_overwritten(monkeypatch, tmp_path, capsys):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "benchmark", lambda expected, output_path=None: bench._seal({
        "format": bench.FORMAT, "status": "PASS", "environment_resets": 0,
        "output_path": output_path,
    }))
    name = "runs/tdmpc2-reward-overshoot-throughput-20260929-v1.json"
    args = ["--expected-script-sha256", "0" * 64, "--output", name]
    assert bench.main(args) == 0
    output = tmp_path / name
    first = output.read_bytes()
    assert first == capsys.readouterr().out.encode()
    assert hashlib.sha256(first).hexdigest() == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(first)["body_sha256"] == bench._seal({
        k: v for k, v in json.loads(first).items() if k != "body_sha256"
    })["body_sha256"]
    assert bench.main(args) == 1
    fail = json.loads(capsys.readouterr().out)
    assert fail["status"] == "FAIL" and output.read_bytes() == first
    with output.open("ab") as stream:
        stream.write(b"external mutation")
    assert bench.main(args) == 1
    assert "never overwrite" in capsys.readouterr().out
    assert output.read_bytes().endswith(b"external mutation")


def test_failed_gate_can_emit_durable_failure_receipt_without_gpu(monkeypatch, tmp_path, capsys):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "benchmark", lambda expected, output_path=None: bench._seal({
        "format": bench.FORMAT, "status": "FAIL", "environment_resets": 0,
        "real_replay_reads": 0, "output_path": output_path,
        "failure_reasons": ["stubbed GPU contention"],
    }))
    name = "runs/tdmpc2-reward-overshoot-throughput-failed-v1.json"
    assert bench.main(["--expected-script-sha256", "0" * 64, "--output", name]) == 1
    receipt = (tmp_path / name).read_bytes()
    assert receipt == capsys.readouterr().out.encode()
    assert json.loads(receipt)["status"] == "FAIL"
