"""Synthetic MSE operator contracts; no real model loads, GPU timing or env."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts import adapt_tdmpc2_reward_head_mse as script
from scripts import benchmark_tdmpc2_reward_head as telemetry


def test_v2_split_unique_balanced_sampler_seed_and_protected_paths():
    assert script.PROTOCOL == "experiments/tdmpc2-head-only-raw-mse-v1.json"
    assert script.OUTPUT != script.parent.OUTPUT
    assert script.BENCHMARK != script.parent.BENCHMARK
    assert script.TRAINING == {**script.parent.TRAINING, "target": script.OBJECTIVE}
    assert [script.partition(i) for i in (0, 7, 8, 11, 12, 43, 44, 306)] == [
        "excluded", "excluded", "excluded", "excluded", "holdout", "holdout", "adaptation", "adaptation"]
    invalid_ids: list[Any] = [-1, 307, True, 44.0]
    for invalid in invalid_ids:
        with pytest.raises(ValueError):
            script.partition(invalid)
    pools = {r: {"positive": [(44 + i, s) for s in range(40)],
                 "nonpositive": [(44 + i, s) for s in range(40, 120)]} for i, r in enumerate(script.raw.ROADS)}
    replay = {"episodes": {44 + i: {"observations": np.zeros((121, 4, 64, 64), np.uint8),
                                   "actions": np.zeros((120, 3), np.float32),
                                   "rewards": np.r_[np.ones(40, np.float32), -np.ones(80, np.float32)]} for i in range(4)}}
    a, ids = script.sample_batch(replay, pools, np.random.default_rng(20260929))
    b, repeated = script.sample_batch(replay, pools, np.random.default_rng(20260929))
    assert ids == repeated and len(set(ids)) == 256
    assert all(torch.equal(a[k], b[k]) for k in a)
    for r in pools:
        assert sum(item in pools[r]["positive"] for item in ids) == 16
        assert sum(item in pools[r]["nonpositive"] for item in ids) == 48
    pools[script.raw.ROADS[0]]["positive"] = [(43, s) for s in range(40)]
    with pytest.raises(ValueError, match="excluded or reserved"):
        script.sample_batch(replay, pools, np.random.default_rng(1))


def test_lineage_source_drift_is_zero_load_and_old_loop_never_called(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(script, "_sources", lambda root: (_ for _ in ()).throw(ValueError("source drift")))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("model load"))
    monkeypatch.setattr(script.parent, "execute", lambda *a, **k: pytest.fail("oldpilot"))
    monkeypatch.setattr(script.parent, "_load", lambda *a, **k: pytest.fail("old loader"))
    with pytest.raises(ValueError, match="source drift"):
        script.preflight(tmp_path, benchmark_sha256="a" * 64)
    assert not (tmp_path / script.OUTPUT).exists()


def test_load_rehashes_same_stream_before_any_deserialization(tmp_path, monkeypatch):
    payload = tmp_path / "synthetic-untrusted-bytes"
    payload.write_bytes(b"not-a-model")
    monkeypatch.setattr(script, "preflight", lambda *a, **k: {})
    monkeypatch.setattr(script.parent, "_pinned", lambda *a: payload)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("loaded before trusted SHA"))
    with pytest.raises(ValueError, match="before torch.load"):
        script._load(tmp_path, {"schema": {"device": "cpu"}}, "a" * 64)


def fake_receipt(tmp_path):
    (tmp_path / "runs").mkdir(exist_ok=True)
    sources = {k: "a" * 64 for k in set(script.PINNED_SOURCES) | {script.SELF, script.ADAPTER, *script.TESTS}}
    runtime = {"python": "3.11", "torch": "2.1.0+cu121", "torch_cuda": "12.1",
        "cuda_available": True, "gpu_name": "synthetic", "cuda_capability": [8, 9],
        "cuda_visible_devices": None, "torch_num_threads": 2}
    step, infer = telemetry._summary([.1] * 128, 128), telemetry._summary([.01] * 32, 32)
    snapshot = {"host_available_bytes": 60 * telemetry.GIB, "cgroup_available_bytes": 60 * telemetry.GIB,
        "disk_available_bytes": 60 * telemetry.GIB, "cuda_free_bytes": 10 * telemetry.GIB, "cuda_total_bytes": 16 * telemetry.GIB}
    receipt = {"format": script.BENCHMARK_FORMAT, "status": "PASS", "objective": script.OBJECTIVE,
        "output_path": script.BENCHMARK, "expected_script_sha256": sources[script.SELF],
        "source_sha256_before": sources.copy(), "source_sha256_after": sources.copy(),
        "environment_resets": 0, "real_replay_reads": 0, "torch_load_calls": 0, "synthetic_only_optimizer_updates": 160,
        "runtime": {"python": runtime["python"], "torch": runtime["torch"], "torch_cuda": "12.1", "device": "cuda:0",
                    "cuda_name": "synthetic", "cuda_capability": [8, 9], "cuda_visible_devices": None, "torch_num_threads": 2},
        "fixture": {"synthetic_only": True, "full_adapter_step": True, "batch_size": 256, "model_size": 5,
                    "obs_shape": [4, 64, 64], "action_dim": 3, "num_bins": 101, "model_parameters": 5_000_000,
                    "road_labels": list(script.raw.ROADS), "positive_per_road": 16, "nonpositive_per_road": 48,
                    "counts": {str(r): {"positive": 16, "nonpositive": 48} for r in script.raw.ROADS},
                    "input_sha256": "b" * 64, "generated_head_sha256": "c" * 64,
                    "generated_head_initialization": "seed20260929_normal_std0.02_weight_only"},
        "timing": {"scope": "full_adapter_step_including_encode_head_optimizer_state_digest",
                   "synchronization": "torch.cuda.synchronize before and after each call", "warmup_updates": 32,
                   "timed_updates": 128, "holdout_warmup_batches": 32, "holdout_timed_batches": 32, "holdout_batches": 86,
                   "step": step, "holdout_inference": infer},
        "forecast": script._forecast(step, infer, 100, 100, 100),
        "resources": {"before": snapshot.copy(), "after": snapshot.copy(), "peak_process_rss_bytes": 100,
                      "max_cuda_reserved_bytes": 100, "forecast_output_bytes": 100,
                      "min_raw_headroom_bytes": 24 * telemetry.GIB, "gpu_reserve_bytes": 2 * telemetry.GIB,
                      "gpu_context_telemetry": {"owner_host_pid": "12345", "snapshots": {
                          "before_cuda_allocation": {"process_memory_mib": {}, "gpu_utilization_percent": 0},
                          **{stage: {"process_memory_mib": {"12345": 100}, "gpu_utilization_percent": 95}
                             for stage in ("after_cuda_allocation", "mid", "final")}}}}}

    def seal():
        receipt["body_sha256"] = script._sha_body({k: v for k, v in receipt.items() if k != "body_sha256"})
        path = tmp_path / script.BENCHMARK
        path.write_text(json.dumps(receipt))
        return {"source_sha256": sources, "runtime": runtime, "device": "cuda",
                "throughput_benchmark": script.ref(script.BENCHMARK, hashlib.sha256(path.read_bytes()).hexdigest())}
    return receipt, seal


@pytest.mark.parametrize("corrupt", ["status", "source", "objective", "warmup", "inference", "forecast", "context", "noise", "cgroup", "runtime", "body"])
def test_fake_receipt_source_runtime_timing_resource_context_gate(tmp_path, corrupt):
    receipt, seal = fake_receipt(tmp_path)
    assert script._benchmark(tmp_path, seal())["status"] == "PASS"
    if corrupt == "status":
        receipt["status"] = "FAIL"
    elif corrupt == "source":
        receipt["source_sha256_after"][script.ADAPTER] = "d" * 64
    elif corrupt == "objective":
        receipt["objective"] = "soft_ce"
    elif corrupt == "warmup":
        receipt["timing"]["warmup_updates"] = 31
    elif corrupt == "inference":
        receipt["timing"]["holdout_batches"] = 85
    elif corrupt == "forecast":
        receipt["forecast"]["forecast_seconds"] = 1800
    elif corrupt == "context":
        receipt["resources"]["gpu_context_telemetry"]["snapshots"]["mid"]["process_memory_mib"]["6789"] = 1
    elif corrupt == "noise":
        receipt["timing"]["step"]["seconds"][-1] = 10
    elif corrupt == "cgroup":
        receipt["resources"]["before"]["cgroup_available_bytes"] = 24 * telemetry.GIB
    elif corrupt == "runtime":
        receipt["runtime"]["torch"] = "other"
    spec = seal()
    if corrupt == "body":
        receipt["body_sha256"] = "0" * 64
        path = tmp_path / script.BENCHMARK
        path.write_text(json.dumps(receipt))
        spec["throughput_benchmark"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        script._benchmark(tmp_path, spec)


def calibration(improvement=.125, worsening=.03125):
    before, after = {}, {}
    for r in script.raw.ROADS:
        before[f"{r}/positive"] = {"count": 100, "mae": 1., "signed_bias": -.5}
        after[f"{r}/positive"] = {"count": 100, "mae": 1. - improvement, "signed_bias": -.5 + improvement}
        before[f"{r}/nonpositive"] = {"count": 100, "mae": .125, "signed_bias": 0.}
        after[f"{r}/nonpositive"] = {"count": 100, "mae": .125 + worsening, "signed_bias": 0.}
    return before, after


def test_calibration_gate_borderlines_both_fit_and_excluded_no_waivers():
    # Exactly representable subtraction reaches the literal declared boundary.
    exact_before, exact_after = calibration()
    for r in script.raw.ROADS:
        exact_before[f"{r}/positive"].update(mae=.2, signed_bias=-.2)
        exact_after[f"{r}/positive"].update(mae=.1, signed_bias=-.1)
        exact_before[f"{r}/nonpositive"]["mae"] = 0.
        exact_after[f"{r}/nonpositive"]["mae"] = .05
    assert script.calibration_gate(exact_before, exact_after)["status"] == "PASS"
    before, after = calibration()
    assert script.calibration_gate(before, after)["status"] == "PASS"
    for r in script.raw.ROADS[:2]:
        after[f"{r}/positive"]["mae"] = .901
    assert script.calibration_gate(before, after)["status"] == "FAIL"
    before, after = calibration(worsening=.050001)
    assert script.calibration_gate(before, after)["status"] == "FAIL"
    before, after = calibration(improvement=.100001, worsening=.049999)
    assert script.calibration_gate(before, after)["status"] == "PASS"
    after[f"{script.raw.ROADS[0]}/positive"]["mae"] = float("nan")
    with pytest.raises(ValueError):
        script.calibration_gate(before, after)


class FakeAdapter:
    def __init__(self, fail_at=0):
        self.updates = 0
        self.fail_at = fail_at
        self.model = SimpleNamespace(state_dict=lambda: {"_reward.weight": torch.ones(1), "_pi.weight": torch.ones(1)})
        self.optim = SimpleNamespace(state_dict=lambda: {})

    def snapshot(self):
        return {"_reward.weight": str(self.updates), "_pi.weight": "unchanged"}

    def step(self, batch):
        if self.updates + 1 == self.fail_at:
            raise RuntimeError("synthetic interrupted optimizer")
        self.updates += 1
        return {"loss": .25, "grad_norm": .1}


def stub_execute(tmp_path, monkeypatch, adapter, *, fail_fit=False):
    (tmp_path / "runs").mkdir(exist_ok=True)
    spec = {"source_sha256": {}, "device": "cpu"}
    pre = {"schema": spec, "_pools": "fit", "_holdout": "excluded"}
    monkeypatch.setattr(script, "preflight", lambda *a, **k: pre)
    monkeypatch.setattr(script, "_load", lambda *a: (adapter, {}))
    monkeypatch.setattr(script, "_state_parity", lambda *a: None)
    monkeypatch.setattr(script, "_recheck", lambda *a: spec)
    monkeypatch.setattr(script, "sample_batch", lambda *a: ({}, [(44, 0)]))

    def predict(model, replay, pool, device):
        before, after = calibration()
        return before if adapter.updates == 0 or fail_fit and pool == "fit" else after
    monkeypatch.setattr(script, "_predict", predict)


@pytest.mark.parametrize("fail_fit", [False, True])
def test_exact512_cap_exclusive_complete_and_fit_failure_blocks_release(tmp_path, monkeypatch, fail_fit):
    adapter = FakeAdapter()
    stub_execute(tmp_path, monkeypatch, adapter, fail_fit=fail_fit)
    report = script.execute(tmp_path, "b" * 64, clock=lambda: 0.)
    assert report["optimizer_updates"] == adapter.updates == 512
    assert report["environment_resets"] == 0 and report["resume_supported"] is False
    assert report["gate"]["status"] == ("FAIL" if fail_fit else "PASS")
    assert report["further_scoring_released"] is not fail_fit
    journal = [json.loads(row) for row in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "update_intent" for row in journal) == 512
    assert [row["updates"] for row in journal if row["event"] == "update_chunk"] == list(range(64, 513, 64))
    assert journal[-1]["event"] == "complete"
    with pytest.raises(ValueError, match="exclusive"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.)


@pytest.mark.parametrize("failure", ["step", "wall", "source"])
def test_partial_no_resume_and_cap_or_source_failures(tmp_path, monkeypatch, failure):
    adapter = FakeAdapter(fail_at=65 if failure == "step" else 0)
    stub_execute(tmp_path, monkeypatch, adapter)
    if failure == "source":
        monkeypatch.setattr(script, "_recheck", lambda *a: (_ for _ in ()).throw(ValueError("source drift")))
    ticks = iter([0., 1800.])
    clock = (lambda: next(ticks, 1800.)) if failure == "wall" else lambda: 0.
    with pytest.raises((RuntimeError, TimeoutError, ValueError)):
        script.execute(tmp_path, "b" * 64, clock=clock)
    journal = [json.loads(row) for row in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert journal[-1]["event"] == "partial" and journal[-1]["resume_supported"] is False
    assert journal[-1]["updates"] == {"step": 64, "wall": 0, "source": 512}[failure]
    assert not (tmp_path / script.OUTPUT / "result.json").exists()
    with pytest.raises(ValueError, match="exclusive"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.)


@pytest.mark.parametrize("timeout_stage", ["write", "promotion"])
def test_timeout_during_result_write_never_releases_complete_artifact(tmp_path, monkeypatch, timeout_stage):
    adapter = FakeAdapter()
    stub_execute(tmp_path, monkeypatch, adapter)
    elapsed = [0.0]
    write = telemetry._write_receipt

    def slow_write(path, result):
        write(path, result)
        if path.name == "result.pending.json" and timeout_stage == "write":
            elapsed[0] = 1800.0

    monkeypatch.setattr(telemetry, "_write_receipt", slow_write)
    link = script.os.link

    def slow_link(source, destination):
        link(source, destination)
        if timeout_stage == "promotion":
            elapsed[0] = 1800.0

    monkeypatch.setattr(script.os, "link", slow_link)
    with pytest.raises(TimeoutError):
        script.execute(tmp_path, "b" * 64, clock=lambda: elapsed[0])
    output = tmp_path / script.OUTPUT
    assert not (output / "result.json").exists()
    assert (output / "result.pending.json").exists()
    assert json.loads((output / "journal.jsonl").read_text().splitlines()[-1])["event"] == "partial"


@pytest.mark.parametrize("foreign", [False, True])
def test_interrupt_after_link_cleans_only_owned_completion(tmp_path, monkeypatch, foreign):
    adapter = FakeAdapter()
    stub_execute(tmp_path, monkeypatch, adapter)
    link = script.os.link

    def interrupted_link(source, destination):
        if foreign:
            destination.write_text("foreign exclusive artifact")
        else:
            link(source, destination)
        raise KeyboardInterrupt()

    monkeypatch.setattr(script.os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    final = tmp_path / script.OUTPUT / "result.json"
    if foreign:
        assert final.read_text() == "foreign exclusive artifact"
    else:
        assert not final.exists()
    assert json.loads((tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()[-1])["event"] == "partial"


def test_benchmark_cli_exclusive_symlink_no_run(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    path = tmp_path / script.BENCHMARK
    target = tmp_path / "target"
    target.write_text("unchanged")
    path.symlink_to(target)
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "benchmark", lambda *a: pytest.fail("real benchmark"))
    with pytest.raises(SystemExit):
        script.main(["--benchmark", "--expected-script-sha256", "a" * 64])
    assert target.read_text() == "unchanged"


def test_mocked_benchmark_runs_only_full_mse_updates_and_inference(monkeypatch):
    sources = {k: "a" * 64 for k in set(script.PINNED_SOURCES) | {script.SELF, script.ADAPTER, *script.TESTS}}
    monkeypatch.setattr(script, "_sources", lambda root: sources)
    baseline = {"process_memory_mib": {}, "gpu_utilization_percent": 0}
    active = {"process_memory_mib": {"12345": 100}, "gpu_utilization_percent": 95}
    snapshots = [baseline, active, active, active]
    monkeypatch.setattr(telemetry, "_gpu_snapshot", lambda **k: snapshots.pop(0))
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 100)
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **k: object() if "device" in k else original_empty(*a, **k))
    monkeypatch.setattr(torch, "manual_seed", torch.default_generator.manual_seed)
    monkeypatch.setattr(script.parent, "_runtime", lambda: {"python": "3.11", "torch": "2.1.0+cu121",
        "torch_cuda": "12.1", "gpu_name": "synthetic", "cuda_capability": [8, 9],
        "cuda_visible_devices": None, "torch_num_threads": 2})
    monkeypatch.setattr(telemetry, "_raw_resources", lambda *a: {
        "host_available_bytes": 60 * telemetry.GIB, "cgroup_available_bytes": 60 * telemetry.GIB,
        "disk_available_bytes": 60 * telemetry.GIB, "cuda_free_bytes": 10 * telemetry.GIB, "cuda_total_bytes": 16 * telemetry.GIB})
    monkeypatch.setattr(script.resource, "getrusage", lambda *a: SimpleNamespace(ru_maxrss=100))
    counts = {"step": 0, "infer": 0}

    class Model:
        total_params = 5_000_000
        cfg = object()

        def __init__(self, cfg):
            self._reward = [SimpleNamespace(weight=torch.zeros(1))]

        def to(self, device):
            return self

        def state_dict(self):
            return {"_reward.weight": torch.ones(1)}

        def encode(self, obs):
            counts["infer"] += 1
            return torch.ones(256, 1)

        def reward(self, z, action):
            return torch.ones(256, 101)

    class MSEAdapter:
        def __init__(self, model):
            self.updates = 0

        def step(self, batch):
            counts["step"] += 1
            self.updates += 1
            return {"loss": .1, "grad_norm": 1.}

        def verify_only_reward_changed(self):
            return True

    monkeypatch.setattr(script, "WorldModel", Model)
    monkeypatch.setattr(script, "RawRewardMSEAdapter", MSEAdapter)
    monkeypatch.setattr(script, "_state_parity", lambda adapter, updates: None)
    monkeypatch.setattr(script, "two_hot_inv", lambda *a: torch.ones(256, 1))
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)

    def timed(op, count, device):
        for _ in range(count):
            op()
        return [.1] * count
    monkeypatch.setattr(telemetry, "_timed", timed)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("model load"))
    monkeypatch.setattr(script.parent, "execute", lambda *a, **k: pytest.fail("oldpilot"))
    result = script.benchmark("a" * 64)
    assert result["status"] == "PASS", result.get("failure_reasons")
    assert counts == {"step": 160, "infer": 64}
    assert result["objective"] == script.OBJECTIVE and result["torch_load_calls"] == 0
    assert result["forecast"]["planned_holdout_batches"] == 86
    assert result["forecast"]["planned_fit_batches"] == 686
    assert result["resources"]["gpu_context_telemetry"]["owner_host_pid"] == "12345"
