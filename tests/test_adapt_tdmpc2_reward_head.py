"""Synthetic-only safety checks; never deserialize a real model or create HAIC env."""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import adapt_tdmpc2_reward_head as script


def test_split_is_exact_and_road_balanced_without_label_leakage():
    assert script.SPLIT == {"previously_exposed": [0, 7], "reserved_future_branch": [8, 11],
                            "adaptation_excluded_logged_train": [12, 43], "adaptation": [44, 306]}
    assert script.PROTOCOL == "experiments/tdmpc2-head-only-adaptation-v2.json"
    assert script.OUTPUT == "runs/tdmpc2-head-only-20260929-v2"
    assert [script.EXPECTED_SPLIT_COUNTS[str(road)]["holdout"]["positive"]
            for road in script.raw.ROADS] == [144, 199, 137, 151]
    assert [script.EXPECTED_SPLIT_COUNTS[str(road)]["adaptation"]["positive"]
            for road in script.raw.ROADS] == [7492, 6443, 7130, 7453]
    assert [script.partition(i) for i in (0, 7, 8, 11, 12, 43, 44, 306)] == [
        "excluded", "excluded", "excluded", "excluded", "holdout", "holdout",
        "adaptation", "adaptation"]
    with pytest.raises(ValueError):
        script.partition(307)
    pools = {road: {"positive": [(44 + 4 * j + i, k) for j in range(10) for k in range(60)],
                    "nonpositive": [(44 + 4 * j + i, k + 60) for j in range(10) for k in range(60)]}
             for i, road in enumerate(script.raw.ROADS)}
    replay = {"episodes": {eid: {"observations": np.zeros((121, 4, 64, 64), np.uint8),
                                 "actions": np.zeros((120, 3), np.float32),
                                 "rewards": np.concatenate((np.ones(60, np.float32),
                                                              -np.ones(60, np.float32)))}
                            for road in pools for sign in pools[road] for eid, _ in pools[road][sign]}}
    batch, indices = script.sample_batch(replay, pools, np.random.default_rng(20260929))
    assert len(indices) == len(set(indices)) == 256
    assert {e for e, _ in indices}.isdisjoint(range(44))
    assert batch["obs"].shape == (256, 4, 64, 64)
    assert batch["reward"].shape == (256, 1)
    assert sum(float(r) > 0 for r in batch["reward"][:, 0]) == 64
    assert sum(float(r) <= 0 for r in batch["reward"][:, 0]) == 192
    for road in script.raw.ROADS:
        assert sum(item in pools[road]["positive"] for item in indices) == 16
        assert sum(item in pools[road]["nonpositive"] for item in indices) == 48
    pools[script.raw.ROADS[0]]["positive"][0] = (43, 0)
    pools[script.raw.ROADS[0]]["positive"] = [(43, index) for index in range(600)]
    with pytest.raises(ValueError, match="excluded or reserved"):
        script.sample_batch(replay, pools, np.random.default_rng(1))


def test_preflight_rejects_source_drift_before_any_torch_load(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(script, "_source", lambda root: (_ for _ in ()).throw(ValueError("source drift")))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint deserialized before checks"))
    with pytest.raises(ValueError, match="source drift"):
        script.preflight(tmp_path, benchmark_sha256="a" * 64)
    assert not (tmp_path / script.OUTPUT).exists()


def test_resource_forecast_is_hard_and_has_no_cap_increase(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    forecast = {"measured_update_seconds": 4.0, "measured_holdout_seconds": 1.0,
                "measured_peak_rss_bytes": 1, "measured_peak_cuda_bytes": 1,
                "measured_output_bytes": 1}
    with pytest.raises(ValueError, match="hard wall"):
        script._resources(tmp_path, forecast, {"cuda_available": False}, source_peak_rss_bytes=1)
    assert script.TRAINING["updates"] == 512
    assert script.TRAINING["max_wall_seconds"] == 1800


def test_resource_forecast_accepts_valid_benchmark_boolean_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace
    forecast = {"measured_update_seconds": .25, "measured_holdout_seconds": 1.0,
                "measured_peak_rss_bytes": 1, "measured_peak_cuda_bytes": 1,
                "measured_output_bytes": 1, "forecast_seconds": 729.0, "passed": True}
    monkeypatch.setattr(script.Path, "read_text", lambda p: str(100 * 1024**3)
                        if p.name == "memory.max" else "0")
    import io
    monkeypatch.setattr(script.Path, "open", lambda *a, **k: io.StringIO("MemAvailable: 104857600 kB\n"))
    monkeypatch.setattr(script.os, "statvfs", lambda p: SimpleNamespace(f_bavail=100 * 1024**3, f_frsize=1))
    assert script._resources(tmp_path, forecast, {"cuda_available": False},
                             source_peak_rss_bytes=1)["forecast_seconds"] == 729.0


def _synthetic_benchmark(tmp_path, *, holdout_batches=86):
    (tmp_path / "runs").mkdir(exist_ok=True)
    sources = {"haic/algorithms/tdmpc2/model.py": "a" * 64,
               script.ADAPTER: "b" * 64, script.BENCHMARK_SCRIPT: "c" * 64}
    runtime = {"python": "3.11", "torch": "2.1.0+cu121", "torch_cuda": "12.1",
               "cuda_available": True, "gpu_name": "synthetic", "cuda_capability": [8, 9],
               "cuda_visible_devices": None, "torch_num_threads": 2}
    def summary(value, n):
        return {"seconds": [value] * n, "mean_seconds": value, "median_seconds": value,
                "p95_seconds": value}
    gi = 1024**3
    snapshot = {"host_available_bytes": 50 * gi, "cgroup_available_bytes": 50 * gi,
                "disk_available_bytes": 50 * gi, "cuda_free_bytes": 8 * gi,
                "cuda_total_bytes": 16 * gi}
    fixture = {"synthetic_only": True, "batch_size": 256, "model_size": 5,
               "obs_shape": [4, 64, 64], "action_dim": 3, "num_bins": 101,
               "road_labels": list(script.raw.ROADS), "positive_per_road": 16,
               "nonpositive_per_road": 48, "model_parameters": 5_000_000,
               "input_sha256": "d" * 64,
               "counts": {str(road): {"positive": 16, "nonpositive": 48} for road in script.raw.ROADS}}
    telemetry = {"owner_host_pid": "1234", "snapshots": {
        "before_cuda_allocation": {"process_memory_mib": {}, "gpu_utilization_percent": 0},
        **{stage: {"process_memory_mib": {"1234": 100}, "gpu_utilization_percent": 1}
           for stage in ("after_cuda_allocation", "mid", "final")}}}
    receipt = {"format": "haic-tdmpc2-head-only-benchmark-v2", "status": "PASS",
               "output_path": script.BENCHMARK, "expected_script_sha256": sources[script.BENCHMARK_SCRIPT],
               "source_sha256_before": sources.copy(), "source_sha256_after": sources.copy(),
               "runtime": {"python": runtime["python"], "torch": runtime["torch"],
                           "torch_cuda": runtime["torch_cuda"], "device": "cuda:0",
                           "cuda_name": runtime["gpu_name"],
                           "cuda_capability": runtime["cuda_capability"],
                           "cuda_visible_devices": None, "torch_num_threads": 2},
               "fixture": fixture,
               "timing": {"scope": "full_adapter_step_including_encode_head_optimizer_state_digest",
                          "synchronization": "torch.cuda.synchronize before and after each call",
                          "warmup_updates": 32, "timed_updates": 128,
                          "holdout_warmup_batches": 32, "holdout_timed_batches": 32,
                          "holdout_batches": holdout_batches,
                          "step": summary(0.05, 128), "holdout_inference": summary(0.01, 32)},
               "forecast": {"measured_update_seconds": 0.05,
                            "measured_holdout_seconds": holdout_batches * 0.01,
                            "measured_peak_rss_bytes": 100,
                            "measured_peak_cuda_bytes": 100,
                            "measured_output_bytes": 100, "planned_updates": 512,
                            "planned_holdout_batches": holdout_batches,
                            "source_checkpoint_ledger_receipt_io_reserve_seconds": 600,
                            "source_load_seconds": 300, "checkpoint_seconds": 300,
                            "forecast_seconds": 512 * 0.05 + holdout_batches * 0.01 + 600,
                            "threshold_seconds_exclusive": 1800, "passed": True},
               "resources": {"before": snapshot, "after": snapshot,
                             "peak_process_rss_bytes": 100, "max_cuda_reserved_bytes": 100,
                             "forecast_output_bytes": 100,
                             "min_raw_headroom_bytes": 24 * gi, "gpu_reserve_bytes": 2 * gi,
                             "gpu_context_telemetry": telemetry},
               "environment_resets": 0, "real_replay_reads": 0,
               "synthetic_only_optimizer_updates": 160}
    path = tmp_path / script.BENCHMARK
    def seal():
        receipt["body_sha256"] = script._sha_body({key: value for key, value in receipt.items()
                                                    if key != "body_sha256"})
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"source_sha256": sources, "runtime": runtime, "device": "cuda",
                "throughput_benchmark": script.ref(script.BENCHMARK, digest)}
    return receipt, seal


def test_benchmark_receipt_blocks_insufficient_holdout_and_tampered_source(tmp_path):
    receipt, seal = _synthetic_benchmark(tmp_path)
    spec = seal()
    assert script._benchmark(tmp_path, spec)["forecast"]["planned_holdout_batches"] == 86
    receipt["forecast"]["planned_holdout_batches"] = 80
    receipt["forecast"]["measured_holdout_seconds"] = .8
    receipt["forecast"]["forecast_seconds"] = 512 * .05 + .8 + 600
    receipt["timing"]["holdout_batches"] = 80
    spec = seal()
    with pytest.raises(ValueError, match="full synchronized head-only updates"):
        script._benchmark(tmp_path, spec)
    receipt["forecast"]["planned_holdout_batches"] = 86
    receipt["forecast"]["measured_holdout_seconds"] = .86
    receipt["forecast"]["forecast_seconds"] = 512 * .05 + .86 + 600
    receipt["timing"]["holdout_batches"] = 86
    receipt["source_sha256_after"][script.ADAPTER] = "e" * 64
    spec = seal()
    with pytest.raises(ValueError, match="source/status"):
        script._benchmark(tmp_path, spec)
    receipt["source_sha256_after"][script.ADAPTER] = "b" * 64
    spec = seal()
    (tmp_path / script.BENCHMARK).write_text("{}\n")
    with pytest.raises(ValueError):
        script._benchmark(tmp_path, spec)


def test_preflight_refuses_failed_benchmark_before_checkpoint_deserialization(tmp_path, monkeypatch):
    receipt, seal = _synthetic_benchmark(tmp_path)
    receipt["status"] = "FAIL"
    spec = seal()
    monkeypatch.setattr(script, "_source", lambda root: ({}, [], {"source_peak_rss_bytes": 1}))
    pools = {road: {sign: range(script.EXPECTED_SPLIT_COUNTS[str(road)]["adaptation"][sign])
                    for sign in ("positive", "nonpositive")} for road in script.raw.ROADS}
    holdout = {road: {sign: range(script.EXPECTED_SPLIT_COUNTS[str(road)]["holdout"][sign])
                      for sign in ("positive", "nonpositive")} for road in script.raw.ROADS}
    monkeypatch.setattr(script, "_indices", lambda *args: (pools, holdout))
    monkeypatch.setattr(script, "proposal", lambda *args: spec)
    monkeypatch.setattr(script.raw, "_file", lambda root, name: tmp_path / script.BENCHMARK)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("checkpoint loaded on FAIL receipt"))
    with pytest.raises(ValueError, match="source/status"):
        script.preflight(tmp_path, benchmark_sha256=spec["throughput_benchmark"]["sha256"],
                         device="cuda")
    assert not (tmp_path / script.OUTPUT).exists()


def test_holdout_gate_requires_every_count_and_nonpositive_nonregression():
    before, after = {}, {}
    for road in script.raw.ROADS:
        before[f"{road}/positive"] = {"count": 100, "signed_bias": -0.5, "mae": 0.8}
        after[f"{road}/positive"] = {"count": 100, "signed_bias": -0.3, "mae": 0.6}
        before[f"{road}/nonpositive"] = {"count": 100, "signed_bias": 0.0, "mae": 0.2}
        after[f"{road}/nonpositive"] = {"count": 100, "signed_bias": 0.0, "mae": 0.23}
    assert script.holdout_gate(before, after)["status"] == "PASS"
    after[f"{script.raw.ROADS[0]}/nonpositive"]["mae"] = 0.26
    assert script.holdout_gate(before, after)["status"] == "FAIL"
    after[f"{script.raw.ROADS[0]}/nonpositive"]["mae"] = 0.23
    after[f"{script.raw.ROADS[0]}/positive"]["count"] = 99
    assert script.holdout_gate(before, after)["status"] == "FAIL"
    after[f"{script.raw.ROADS[0]}/positive"]["count"] = 100
    for road in script.raw.ROADS[:2]:
        before[f"{road}/positive"]["signed_bias"] = 0.5
        after[f"{road}/positive"]["signed_bias"] = 0.3
    assert script.holdout_gate(before, after)["status"] == "FAIL"


def test_holdout_before_after_use_identical_rng_without_outer_cpu_leak(monkeypatch):
    from haic.algorithms.tdmpc2 import model as td_model

    monkeypatch.setattr(td_model, "two_hot_inv", lambda logits, cfg: logits)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class FakeModel:
        cfg = object()

        def eval(self):
            pass

        def encode(self, obs, task=None):
            return torch.rand(len(obs), 1)

        def reward(self, z, actions, task=None):
            return z

    holdout = {road: {"positive": [(i * 2, 0)], "nonpositive": [(i * 2 + 1, 0)]}
               for i, road in enumerate(script.raw.ROADS)}
    replay = {"episodes": [{"observations": np.zeros((1, 4, 64, 64), np.uint8),
                             "actions": np.zeros((1, 3), np.float32),
                             "rewards": np.zeros(1, np.float32)} for _ in range(8)]}
    outer = torch.get_rng_state().clone()
    first = script._predict(FakeModel(), replay, holdout, "cpu")
    second = script._predict(FakeModel(), replay, holdout, "cpu")
    assert first == second and torch.equal(torch.get_rng_state(), outer)


class FakeAdapter:
    def __init__(self, *, fail_at=0):
        self.model = SimpleNamespace(state_dict=lambda: {"_reward.weight": torch.ones(1),
                                                       "_pi.weight": torch.ones(1)})
        self.optim = SimpleNamespace(state_dict=lambda: {"state": {}, "param_groups": []})
        self.updates = 0
        self.fail_at = fail_at
        self._value = 0
        self.bad_parity = False

    def snapshot(self):
        return {"_reward.weight": str(self._value), "_pi.weight": "unchanged"}

    def step(self, batch):
        self.updates += 1
        if self.updates == self.fail_at:
            raise RuntimeError("synthetic interrupted update")
        self._value += 1
        return {"loss": 0.5, "grad_norm": 0.1}

    def verify_only_reward_changed(self):
        return not self.bad_parity


def _stub_execute(tmp_path: Path, monkeypatch, adapter):
    (tmp_path / "runs").mkdir(exist_ok=True)
    spec = {"raw_source": {"checkpoint": {"sha256": script.raw.CHECKPOINT_SHA}},
            "overshoot_source": {"checkpoint": {"sha256": script.NEW_CHECKPOINT[1]}},
            "throughput_benchmark": {"path": script.BENCHMARK, "sha256": "c" * 64},
            "source_sha256": {"fake": "a" * 64}, "device": "cpu"}
    monkeypatch.setattr(script, "preflight", lambda *a, **k: {
        "schema": spec, "_holdout": {}, "_pools": {}, "environment_resets": 0, "torch_load_calls": 0})
    monkeypatch.setattr(script, "_load", lambda *a: (adapter, {}))
    monkeypatch.setattr(script, "_predict", lambda *a: {
        f"{road}/{sign}": {"count": 100, "mae": 0.3, "signed_bias": -0.2}
        for road in script.raw.ROADS for sign in ("positive", "nonpositive")})
    monkeypatch.setattr(script, "sample_batch", lambda *a: ({"obs": torch.empty(0)}, [(44, 0)] * 256))
    monkeypatch.setattr(script, "_sha_body", lambda body: "a" * 64)


def test_exact_update_cap_exclusive_result_and_complete_checkpoint(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _stub_execute(tmp_path, monkeypatch, adapter)
    report = script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    assert report["optimizer_updates"] == adapter.updates == 512
    assert report["checkpoint"]["sha256"] == script.raw._digest(tmp_path / script.OUTPUT / "adapted-model.pt")
    assert report["source"]["raw_source"]["checkpoint"]["sha256"] == script.raw.CHECKPOINT_SHA
    assert (tmp_path / script.OUTPUT / "result.json").is_file()
    rows = [json.loads(line) for line in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert sum(row["event"] == "update_intent" for row in rows) == 512
    assert [row["updates"] for row in rows if row["event"] == "update_chunk"] == list(range(64, 513, 64))
    with pytest.raises(ValueError, match="exclusive"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)


def test_partial_is_preserved_no_resume_on_failed_update(tmp_path, monkeypatch):
    adapter = FakeAdapter(fail_at=65)
    _stub_execute(tmp_path, monkeypatch, adapter)
    with pytest.raises(RuntimeError, match="interrupted"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    rows = [json.loads(line) for line in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "partial"
    assert rows[-1]["updates"] == 64 and rows[-1]["resume_supported"] is False
    assert not (tmp_path / script.OUTPUT / "result.json").exists()
    with pytest.raises(ValueError, match="exclusive"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)


def test_final_source_drift_aborts_before_checkpoint_and_preserves_partial(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _stub_execute(tmp_path, monkeypatch, adapter)
    original_preflight = script.preflight
    calls = 0

    def drift(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("source changed")
        return original_preflight(*args, **kwargs)

    monkeypatch.setattr(script, "preflight", drift)
    with pytest.raises(ValueError, match="source changed"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    assert adapter.updates == 512
    assert not (tmp_path / script.OUTPUT / "adapted-model.pt").exists()
    rows = [json.loads(line) for line in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "partial" and rows[-1]["updates"] == 512


def test_nonhead_parity_failure_blocks_completed_output(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _stub_execute(tmp_path, monkeypatch, adapter)
    adapter.bad_parity = True
    with pytest.raises(ValueError, match="non-reward"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    assert not (tmp_path / script.OUTPUT / "result.json").exists()


def test_fake_report_cannot_hide_missing_optimizer_update(tmp_path, monkeypatch):
    adapter = FakeAdapter()
    _stub_execute(tmp_path, monkeypatch, adapter)
    adapter.step = lambda batch: {"loss": 0.5, "grad_norm": 0.1}
    with pytest.raises(ValueError, match="update count"):
        script.execute(tmp_path, "b" * 64, clock=lambda: 0.0)
    rows = [json.loads(line) for line in (tmp_path / script.OUTPUT / "journal.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "partial" and rows[-1]["updates"] == 0
