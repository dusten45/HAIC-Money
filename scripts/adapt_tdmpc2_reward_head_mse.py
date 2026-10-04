"""Isolated raw-MSE objective control; no environments and no resume.

Benchmark generated pixels only with --benchmark --expected-script-sha256 SHA.
Then --preflight --benchmark-sha256 SHA --device cuda emits the exact schema.
The operator must independently freeze that schema at PROTOCOL before --execute
--protocol-sha256 SHA --device cuda. No gate can be waived by CLI flags.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import resource
import sys
import time
import json

import numpy as np
import torch

from haic.algorithms.tdmpc2.model import TDMPC2ModelConfig, WorldModel, two_hot_inv
from haic.algorithms.tdmpc2.reward_head_mse import RawRewardMSEAdapter
from scripts import adapt_tdmpc2_reward_head as parent
from scripts import benchmark_tdmpc2_reward_head as telemetry


ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/adapt_tdmpc2_reward_head_mse.py"
ADAPTER = "haic/algorithms/tdmpc2/reward_head_mse.py"
TESTS = ("tests/test_tdmpc2_reward_head_mse.py", "tests/test_adapt_tdmpc2_reward_head_mse.py")
PROTOCOL = "experiments/tdmpc2-head-only-raw-mse-v1.json"
OUTPUT = "runs/tdmpc2-head-raw-mse-20260929-v1"
BENCHMARK = "runs/tdmpc2-head-raw-mse-benchmark-20260929-v3.json"
FORMAT = "haic-tdmpc2-head-only-raw-mse-v1"
BENCHMARK_FORMAT = "haic-tdmpc2-head-raw-mse-benchmark-v1"
LINEAGE = (parent.PROTOCOL, "d2e7c833671c1e16e529172ed7b38ffc2be9a107a2c57242334277fac67a206b")
PINNED_SOURCES = {
    "haic/algorithms/tdmpc2/model.py": "a121d8688e5ab954abe2e6468ca84760e035f15b49004b695fc465696eb90c65",
    parent.ADAPTER: "c2a1626b3ab321100ae22579d1f665a2b3fdc69661d3cdcce796f4df715fc615",
    parent.SELF: "a195ec205f49bd1b6abd8eb814b5259ea1d99cce81c248c5ba4d8c8018ef1d65",
    parent.BENCHMARK_SCRIPT: "788b3b96f9a9eebf408f8bea3aad861f90a0fa9aa9f08cb76557fb77a8f5d584",
}
OBJECTIVE = "MSE(two_hot_inv(reward_logits,cfg),observed_RAW_step_reward)"
TRAINING = {**parent.TRAINING, "target": OBJECTIVE}
SPLIT = parent.SPLIT
EXPECTED_SPLIT_COUNTS = parent.EXPECTED_SPLIT_COUNTS
partition = parent.partition
sample_batch = parent.sample_batch
_indices = parent._indices
_predict = parent._predict
raw = parent.raw
overshoot = parent.overshoot
ref = parent.ref
_sha_body = parent._sha_body
_journal = parent._journal


def _sources(root: Path) -> dict:
    for name, sha in PINNED_SOURCES.items():
        parent._pinned(root, (name, sha))
    result = {**PINNED_SOURCES, **{name: raw._digest(raw._file(root, name))
                                  for name in (SELF, ADAPTER, *TESTS)}}
    for module, name in ((WorldModel.__module__, "haic/algorithms/tdmpc2/model.py"),
                         (RawRewardMSEAdapter.__module__, ADAPTER),
                         (parent.__name__, parent.SELF), (telemetry.__name__, parent.BENCHMARK_SCRIPT)):
        location = sys.modules[module].__file__
        if location is None or Path(location).resolve() != (root / name).resolve():
            raise ValueError("imported source path differs from pinned file")
    return result


def _source(root: Path) -> tuple[dict, list[dict], dict]:
    _sources(root)
    lineage = raw._json(parent._pinned(root, LINEAGE).read_bytes())
    if (lineage.get("format") != "haic-tdmpc2-head-only-adaptation-v2"
            or lineage.get("split") != SPLIT or lineage.get("training") != parent.TRAINING
            or lineage.get("unique_step_counts") != EXPECTED_SPLIT_COUNTS
            or lineage.get("overshoot_source", {}).get("checkpoint") != ref(*parent.NEW_CHECKPOINT)):
        raise ValueError("completed v2 lineage differs from original parent/split/settings")
    for name, sha in lineage["source_sha256"].items():
        parent._pinned(root, (name, sha))
    return parent._source(root)


def proposal(root: Path, context: dict, counts: dict, benchmark_sha256: str, device: str) -> dict:
    # The parent proposal is pure metadata construction, not the old pilot loop.
    spec = parent.proposal(root, context, counts, benchmark_sha256, device)
    spec.update(format=FORMAT, output_dir=OUTPUT, training=TRAINING,
                source_sha256={**spec["source_sha256"], **_sources(root)},
                throughput_benchmark=ref(BENCHMARK, benchmark_sha256),
                lineage_protocol=ref(*LINEAGE), fit_gate=parent.GATE,
                gate_policy="FIT_natural_AND_adaptation_excluded_PASS_no_waiver",
                inference_forecast_caveat="86_excluded_plus_686_FIT_batches_and_600s_IO_reserve_under_hard_1800s_cap")
    return spec


def _forecast(step: dict, inference: dict, peak_rss: int, peak_cuda: int, output_bytes: int) -> dict:
    result = telemetry._forecast(step, inference, peak_rss, peak_cuda, output_bytes)
    result["planned_fit_batches"] = 686
    result["measured_fit_seconds"] = 686 * inference["mean_seconds"]
    result["forecast_seconds"] += result["measured_fit_seconds"]
    result["passed"] = result["forecast_seconds"] < 1800
    return result


def _benchmark(root: Path, spec: dict) -> dict:
    reference = spec["throughput_benchmark"]
    if reference.get("path") != BENCHMARK:
        raise ValueError("benchmark receipt path differs")
    receipt = raw._json(parent._pinned(root, (BENCHMARK, raw._sha(reference["sha256"]))).read_bytes())
    sources = receipt.get("source_sha256_before")
    required = set(PINNED_SOURCES) | {SELF, ADAPTER, *TESTS}
    if (receipt.get("body_sha256") != _sha_body({k: v for k, v in receipt.items() if k != "body_sha256"})
            or receipt.get("format") != BENCHMARK_FORMAT or receipt.get("status") != "PASS"
            or "failure_reasons" in receipt or receipt.get("objective") != OBJECTIVE
            or receipt.get("output_path") != BENCHMARK or receipt.get("environment_resets") != 0
            or receipt.get("real_replay_reads") != 0 or receipt.get("torch_load_calls") != 0
            or not isinstance(sources, dict) or set(sources) != required
            or receipt.get("source_sha256_after") != sources
            or receipt.get("expected_script_sha256") != sources.get(SELF)
            or any(sources[name] != spec["source_sha256"].get(name) for name in required)):
        raise ValueError("MSE benchmark source/status/objective mismatch")
    expected, observed = spec["runtime"], receipt.get("runtime", {})
    if (spec["device"] != "cuda" or expected["cuda_available"] is not True
            or observed.get("device") != "cuda:0"
            or any(observed.get(a) != expected.get(b) for a, b in (
                ("python", "python"), ("torch", "torch"), ("torch_cuda", "torch_cuda"),
                ("cuda_name", "gpu_name"), ("cuda_capability", "cuda_capability"),
                ("cuda_visible_devices", "cuda_visible_devices"), ("torch_num_threads", "torch_num_threads")))):
        raise ValueError("benchmark runtime differs")
    fixture = receipt.get("fixture", {})
    fields = {"synthetic_only": True, "full_adapter_step": True, "batch_size": 256,
              "model_size": 5, "obs_shape": [4, 64, 64], "action_dim": 3, "num_bins": 101,
              "road_labels": list(raw.ROADS), "positive_per_road": 16, "nonpositive_per_road": 48,
              "counts": {str(r): {"positive": 16, "nonpositive": 48} for r in raw.ROADS}}
    if (any(fixture.get(k) != v for k, v in fields.items())
            or type(fixture.get("model_parameters")) is not int
            or not 4_000_000 <= fixture["model_parameters"] <= 8_000_000):
        raise ValueError("benchmark fixture differs from original size5/B256")
    raw._sha(fixture.get("input_sha256"))
    if fixture.get("generated_head_initialization") != "seed20260929_normal_std0.02_weight_only":
        raise ValueError("generated nonzero head initialization absent")
    raw._sha(fixture.get("generated_head_sha256"))
    timing = receipt.get("timing", {})
    if (timing.get("scope") != "full_adapter_step_including_encode_head_optimizer_state_digest"
            or timing.get("synchronization") != "torch.cuda.synchronize before and after each call"
            or any(type(timing.get(k)) is not int or timing[k] != n for k, n in (
                ("warmup_updates", 32), ("timed_updates", 128), ("holdout_warmup_batches", 32),
                ("holdout_timed_batches", 32), ("holdout_batches", 86)))
            or receipt.get("synthetic_only_optimizer_updates") != 160):
        raise ValueError("benchmark full-step counts differ")
    for name, n in (("step", 128), ("holdout_inference", 32)):
        summary = timing.get(name, {})
        values = summary.get("seconds")
        if (not isinstance(values, list) or len(values) != n
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or telemetry._summary(values, n) != summary):
            raise ValueError("benchmark timing/noise gate failed")
    forecast = receipt.get("forecast", {})
    numeric = ("measured_peak_rss_bytes", "measured_peak_cuda_bytes", "measured_output_bytes")
    if any(type(forecast.get(k)) is not int or forecast[k] <= 0 for k in numeric):
        raise ValueError("benchmark resource measurements invalid")
    calculated = _forecast(timing["step"], timing["holdout_inference"],
                                     *(forecast[k] for k in numeric))
    if forecast != calculated or calculated["passed"] is not True:
        raise ValueError("benchmark strict 1800-second forecast failed")
    resources = receipt.get("resources", {})
    if (resources.get("min_raw_headroom_bytes") != telemetry.MIN_HEADROOM
            or resources.get("gpu_reserve_bytes") != telemetry.GPU_RESERVE
            or resources.get("peak_process_rss_bytes") != forecast["measured_peak_rss_bytes"]
            or resources.get("max_cuda_reserved_bytes") != forecast["measured_peak_cuda_bytes"]
            or resources.get("forecast_output_bytes") != forecast["measured_output_bytes"]):
        raise ValueError("benchmark measured peak bindings differ")
    for stage in ("before", "after"):
        values = resources.get(stage)
        if not isinstance(values, dict) or any(type(values.get(k)) is not int or values[k] <= 0
                                               for k in ("host_available_bytes", "cgroup_available_bytes",
                                                         "disk_available_bytes", "cuda_free_bytes", "cuda_total_bytes")):
            raise ValueError("benchmark raw resource readings invalid")
    telemetry._check_resources(resources["before"], resources["after"],
                               peak_rss=forecast["measured_peak_rss_bytes"],
                               output_bytes=forecast["measured_output_bytes"],
                               peak_reserved=forecast["measured_peak_cuda_bytes"])
    context = resources.get("gpu_context_telemetry", {})
    snapshots, owner = context.get("snapshots", {}), context.get("owner_host_pid")
    if not isinstance(owner, str) or not owner.isdecimal():
        raise ValueError("benchmark host PID absent")
    baseline = snapshots.get("before_cuda_allocation", {})
    if (type(baseline.get("gpu_utilization_percent")) is not int
            or not 0 <= baseline["gpu_utilization_percent"] <= 10):
        raise ValueError("benchmark preallocation contention")
    for stage in ("before_cuda_allocation", "after_cuda_allocation", "mid", "final"):
        reading = snapshots.get(stage, {})
        processes = reading.get("process_memory_mib")
        if (not isinstance(processes, dict) or any(not isinstance(pid, str) or not pid.isdecimal()
                or type(mem) is not int or mem <= 0 for pid, mem in processes.items())
                or type(reading.get("gpu_utilization_percent")) is not int
                or not 0 <= reading["gpu_utilization_percent"] <= 100):
            raise ValueError("benchmark GPU telemetry invalid")
        if stage != "before_cuda_allocation":
            telemetry._check_gpu_context(baseline, reading, owner)
    return receipt


def preflight(root: Path = ROOT, protocol_sha256: str | None = None, *,
              benchmark_sha256: str | None = None, device: str = "cpu", allow_output: bool = False) -> dict:
    root = Path(root).resolve(strict=True)
    output = root / OUTPUT
    if ((output.exists() or output.is_symlink()) and not allow_output
            or allow_output and (not output.is_dir() or output.is_symlink())
            or (root / "runs").is_symlink()):
        raise ValueError("exclusive output exists; partial runs cannot resume")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("explicit available device required")
    frozen = None
    if protocol_sha256 is not None:
        frozen = raw._json(parent._pinned(root, (PROTOCOL, raw._sha(protocol_sha256))).read_bytes())
        bound = frozen.get("throughput_benchmark", {})
        if bound.get("path") != BENCHMARK or benchmark_sha256 is not None and benchmark_sha256 != bound.get("sha256"):
            raise ValueError("frozen benchmark reference differs")
        benchmark_sha256 = bound.get("sha256")
    if benchmark_sha256 is None:
        raise ValueError("independent synthetic MSE benchmark SHA required")
    benchmark_sha256 = raw._sha(benchmark_sha256)
    original, episodes, context = _source(root)
    pools, holdout = _indices(episodes, raw._file(root, f"{raw.RUN}/steps.jsonl"))
    counts = {str(r): {"adaptation": {s: len(pools[r][s]) for s in pools[r]},
                       "holdout": {s: len(holdout[r][s]) for s in holdout[r]}} for r in raw.ROADS}
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError("exact v2 unique signed target counts differ")
    spec = proposal(root, context, counts, benchmark_sha256, device)
    if frozen is not None and frozen != spec:
        raise ValueError("frozen protocol differs from exact MSE schema")
    receipt = _benchmark(root, spec)
    resources = parent._resources(root, receipt["forecast"], spec["runtime"],
                                  source_peak_rss_bytes=context["source_peak_rss_bytes"])
    return {"status": "preflight_only", "schema": spec, "resources": resources,
            "environment_resets": 0, "torch_load_calls": 0, "optimizer_updates": 0,
            "_pools": pools, "_holdout": holdout, "_episodes": episodes,
            "_context": context, "_original": original}


def _load(root: Path, pre: dict, protocol_sha256: str):
    # Never call the old pilot loader: construct the MSE adapter from original parent.
    preflight(root, protocol_sha256, device=pre["schema"]["device"], allow_output=True)
    states = []
    for pair in ((raw.CHECKPOINT, raw.CHECKPOINT_SHA), parent.NEW_CHECKPOINT):
        with parent._pinned(root, pair).open("rb") as stream:
            if raw._digest_stream(stream) != pair[1]:
                raise ValueError("trusted checkpoint changed before torch.load")
            stream.seek(0)
            states.append(torch.load(stream, map_location="cpu", weights_only=False))
    old, new = states
    from scripts import diagnose_tdmpc2_h5_branches as binder
    binder.bind_replay(old, raw._json(parent._pinned(root, (
        "experiments/tdmpc2-long-reused-train-v2.json", raw.SOURCE_PROTOCOL_SHA)).read_bytes()),
        pre["_episodes"], parent._pinned(root, (f"{raw.RUN}/steps.jsonl", raw.STEPS_SHA)))
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
        raise ValueError("original overshoot checkpoint metadata differs")
    overshoot._bind_checkpoint_replay(root, new, {"spec": {"training_source": context["refs"]},
        "episodes": overshoot._ledgers(root, {"training_source": context["refs"]}, context["training"], context["result"])})
    overshoot._optimizer(new.get("optim"), 100159)
    overshoot._optimizer(new.get("pi_optim"), 100159)
    overshoot._overshoot_probe(new, context["training"], context["last"])
    learner = new.get("learner")
    if not isinstance(learner, dict) or not isinstance(learner.get("q_scale"), torch.Tensor):
        raise ValueError("original parent learner weights absent")
    model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
    weights = {k.removeprefix("model."): v for k, v in learner.items() if k.startswith("model.")}
    expected = model.state_dict()
    if (weights.keys() != expected.keys() or any(not isinstance(v, torch.Tensor)
            or v.shape != expected[k].shape or v.dtype != expected[k].dtype
            or not bool(torch.isfinite(v).all()) for k, v in weights.items())):
        raise ValueError("parent model tensor keys/shape/dtype/finite parity differs")
    model.load_state_dict(weights, strict=True)
    if any(not torch.equal(v, model.state_dict()[k]) for k, v in weights.items()):
        raise ValueError("strict loaded parent tensors differ")
    model.to(pre["schema"]["device"])
    adapter = RawRewardMSEAdapter(model)
    _state_parity(adapter, 0)
    return adapter, old["replay"]


def _state_parity(adapter: RawRewardMSEAdapter, updates: int) -> None:
    """Finite exact model state plus reward-only Adam membership/step parity."""
    adapter.verify_only_reward_changed()
    if adapter.updates != updates or any(not bool(torch.isfinite(t).all()) for t in adapter.model.state_dict().values()):
        raise ValueError("model finite tensor/update parity failed")
    params = tuple(adapter.model._reward.parameters())
    groups = adapter.optim.param_groups
    if (not isinstance(adapter.optim, torch.optim.Adam) or len(groups) != 1
            or tuple(groups[0]["params"]) != params or groups[0]["lr"] != 3e-5
            or groups[0]["weight_decay"] != 0 or groups[0]["betas"] != (0.9, 0.999)
            or groups[0]["eps"] != 1e-8 or groups[0]["amsgrad"] is not False
            or set(adapter.optim.state) != (set(params) if updates else set())):
        raise ValueError("reward-only Adam parameter/settings parity failed")
    for parameter, state in adapter.optim.state.items():
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or any(not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all()) for t in state.values())
                or state["step"].numel() != 1 or state["step"].item() != updates
                or any(state[k].shape != parameter.shape or state[k].dtype != parameter.dtype
                       for k in ("exp_avg", "exp_avg_sq"))):
            raise ValueError("Adam finite state/step parity failed")


def calibration_gate(before: dict, after: dict) -> dict:
    # The criterion is absolute bias improvement, with no sign-dependent waiver.
    checks = {}
    for road in raw.ROADS:
        positive, negative = f"{road}/positive", f"{road}/nonpositive"
        for key in (positive, negative):
            a, b = before[key], after[key]
            if (type(a.get("count")) is not int or a["count"] < 100 or a["count"] != b.get("count")
                    or any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k])
                           or k == "mae" and row[k] < 0 for row in (a, b) for k in ("mae", "signed_bias"))):
                raise ValueError("calibration count/nonfinite metrics gate failed")
        mae = before[positive]["mae"] - after[positive]["mae"]
        bias = abs(before[positive]["signed_bias"]) - abs(after[positive]["signed_bias"])
        worsening = after[negative]["mae"] - before[negative]["mae"]
        checks[str(road)] = {"positive_mae_improvement": mae,
                             "positive_absolute_signed_bias_improvement": bias,
                             "nonpositive_mae_worsening": worsening,
                             "qualifies": mae >= .10 and bias >= .10}
    qualifies = sum(row["qualifies"] for row in checks.values())
    passed = qualifies >= 3 and all(row["nonpositive_mae_worsening"] <= .05 for row in checks.values())
    return {"status": "PASS" if passed else "FAIL", "qualifying_roads": qualifies, "by_road": checks}


def execute(root: Path, protocol_sha256: str, *, device: str = "cpu", clock=time.monotonic) -> dict:
    root = Path(root).resolve(strict=True)
    baseline = owner = None
    live_context: dict = {"snapshots": {}}
    if device == "cuda":
        if torch.cuda.is_initialized():
            raise ValueError("execute requires a fresh process for live GPU isolation")
        baseline = telemetry._gpu_snapshot()
        live_context["snapshots"]["before_cuda_allocation"] = baseline
    pre = preflight(root, protocol_sha256, device=device)
    if baseline is not None:
        observed = telemetry._gpu_snapshot(require_idle=False)
        owner = telemetry._check_gpu_context(baseline, observed)
        live_context["owner_host_pid"] = owner
        live_context["snapshots"]["after_preflight"] = observed
    output = root / OUTPUT
    if output.exists() or output.is_symlink():
        raise ValueError("exclusive output exists; no resume")
    output.mkdir(exist_ok=False)
    journal = output / "journal.jsonl"
    completed, started = 0, clock()
    result_identity = None

    def cap():
        if clock() - started >= 1800:
            raise TimeoutError("hard 1800-second adaptation wall expired")

    def live(stage):
        if baseline is not None:
            observed = telemetry._gpu_snapshot(require_idle=False)
            telemetry._check_gpu_context(baseline, observed, owner)
            live_context["snapshots"][stage] = observed

    try:
        _journal(journal, {"event": "start", "intent": "no_resume_512_raw_MSE_head_only_updates",
                           "protocol_sha256": protocol_sha256, "source": pre["schema"],
                           "updates": 0, "environment_resets": 0})
        adapter, replay = _load(root, pre, protocol_sha256)
        live("after_load")
        _state_parity(adapter, 0)
        initial = adapter.snapshot()
        rng = np.random.default_rng(20260929)
        before = {}
        for name, pool in (("fit_natural", pre["_pools"]), ("excluded", pre["_holdout"])):
            cap()
            before[name] = _predict(adapter.model, replay, pool, device)
            live("before_" + name)
        metrics = []
        for update in range(1, 513):
            cap()
            batch, indices = sample_batch(replay, pre["_pools"], rng)
            _journal(journal, {"event": "update_intent", "update": update,
                               "sample_ids_sha256": _sha_body({"indices": indices}), "protocol_sha256": protocol_sha256})
            report = adapter.step(batch)
            if adapter.updates != update or not report or any(type(v) not in (float, int) or not math.isfinite(v) for v in report.values()):
                raise ValueError("finite optimizer update count/metrics differ")
            completed = update
            metrics.append(report)
            if update % 64 == 0:
                live(f"update_{update}")
                _state_parity(adapter, completed)
                _journal(journal, {"event": "update_chunk", "updates": completed, "last_metrics": report,
                                   "last_sample_ids_sha256": _sha_body({"indices": indices}),
                                   "protocol_sha256": protocol_sha256})
        after = {}
        for name, pool in (("fit_natural", pre["_pools"]), ("excluded", pre["_holdout"])):
            cap()
            after[name] = _predict(adapter.model, replay, pool, device)
            live("after_" + name)
        _state_parity(adapter, 512)
        final = adapter.snapshot()
        nonhead = {k: v for k, v in initial.items() if not k.startswith("_reward.")}
        changed = [k for k, v in initial.items() if k.startswith("_reward.") and final[k] != v]
        if not changed or initial.keys() != final.keys() or any(final[k] != v for k, v in nonhead.items()):
            raise ValueError("non-head parity failed or head did not change")
        gates = {name: calibration_gate(before[name], after[name]) for name in before}
        gate = {"status": "PASS" if all(g["status"] == "PASS" for g in gates.values()) else "FAIL", **gates}
        # End-of-run source checks do not reapply pre-allocation free-memory floors.
        checked = _recheck(root, pre, protocol_sha256)
        if checked != pre["schema"] or completed != 512:
            raise ValueError("source/schema changed at completion")
        cap()
        checkpoint = output / "adapted-model.pt"
        with checkpoint.open("xb") as stream:
            torch.save({"format": FORMAT + "-model", "protocol_sha256": protocol_sha256,
                        "source": pre["schema"], "updates": completed, "seed": 20260929,
                        "model": {k: v.detach().cpu() for k, v in adapter.model.state_dict().items()},
                        "optimizer": adapter.optim.state_dict(), "sampler_rng": rng.bit_generator.state,
                        "nonhead_bitwise_parity": True, "gate": gate, "resume_supported": False}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        cap()
        result = {"format": FORMAT + "-result", "status": "complete", "protocol": ref(PROTOCOL, protocol_sha256),
                  "source": pre["schema"], "checkpoint": ref(f"{OUTPUT}/adapted-model.pt", raw._digest(checkpoint)),
                  "optimizer_updates": completed, "training_metrics": metrics, "before": before, "after": after,
                  "gate": gate, "further_scoring_released": gate["status"] == "PASS",
                  "nonhead_bitwise_parity": {"passed": True, "sha256_before": nonhead,
                      "sha256_after": {k: final[k] for k in nonhead}, "changed_reward_keys": changed},
                  "environment_resets": 0, "policy_or_official_claim": False, "resume_supported": False,
                  "elapsed_seconds": clock() - started}
        live("final")
        result["live_gpu_context_telemetry"] = live_context
        _journal(journal, {"event": "complete", "updates": 512, "gate": gate["status"],
                           "checkpoint_sha256": result["checkpoint"]["sha256"], "protocol_sha256": protocol_sha256})
        result["journal_sha256"] = raw._digest(journal)
        cap()
        pending_result = output / "result.pending.json"
        telemetry._write_receipt(pending_result, result)
        cap()
        identity = pending_result.stat()
        result_identity = (identity.st_dev, identity.st_ino)
        os.link(pending_result, output / "result.json")
        cap()
        pending_result.unlink()
        return result
    except BaseException as exc:
        result_path = output / "result.json"
        if result_identity is not None and result_path.exists():
            identity = result_path.lstat()
            if (identity.st_dev, identity.st_ino) == result_identity:
                result_path.unlink()
        row = {"event": "partial", "reason": type(exc).__name__, "updates": completed,
               "protocol_sha256": protocol_sha256, "environment_resets": 0, "resume_supported": False}
        try:
            _journal(journal, row)
        except BaseException:
            telemetry._write_receipt(output / "failure.json", row)
        raise


def _recheck(root: Path, pre: dict, protocol_sha256: str) -> dict:
    _source(root)
    frozen = raw._json(parent._pinned(root, (PROTOCOL, protocol_sha256)).read_bytes())
    context = pre["_context"]
    spec = proposal(root, context, pre["schema"]["unique_step_counts"],
                    pre["schema"]["throughput_benchmark"]["sha256"], pre["schema"]["device"])
    if spec != frozen:
        raise ValueError("source-bound completion schema changed")
    _benchmark(root, spec)
    return spec


def benchmark(expected_script_sha256: str, *, root: Path = ROOT) -> dict:
    """Generated original-size model only; full MSE step, no pickle or replay."""
    body = {"format": BENCHMARK_FORMAT, "status": "FAIL", "objective": OBJECTIVE,
            "environment_resets": 0, "real_replay_reads": 0, "torch_load_calls": 0,
            "synthetic_only_optimizer_updates": 0, "output_path": BENCHMARK,
            "expected_script_sha256": expected_script_sha256}
    errors = []
    sources = baseline = owner = device = before = sentinel = None
    context = {"method": "new NVIDIA host PID from deliberate CUDA allocation", "snapshots": {}, "owner_host_pid": None}
    try:
        sources = _sources(root)
        body["source_sha256_before"] = sources
        if sources[SELF] != raw._sha(expected_script_sha256):
            raise ValueError("expected MSE runner SHA differs")
        if str(torch.__version__) != "2.1.0+cu121" or torch.version.cuda != "12.1":
            raise ValueError("pinned Torch/CUDA runtime differs")
        if torch.cuda.is_initialized():
            raise ValueError("CUDA initialized before baseline context snapshot")
        baseline = telemetry._gpu_snapshot()
        context["snapshots"]["before_cuda_allocation"] = baseline
        if sum(baseline["process_memory_mib"].values()) > 512 or not torch.cuda.is_available():
            raise ValueError("CUDA unavailable or baseline contention")
        device = torch.device("cuda:0")
        sentinel = torch.empty(1, device=device)
        allocated = telemetry._gpu_snapshot(require_idle=False)
        context["snapshots"]["after_cuda_allocation"] = allocated
        owner = telemetry._check_gpu_context(baseline, allocated)
        context["owner_host_pid"] = owner
        before = telemetry._raw_resources(device, root)
        telemetry._check_resources(before)
        runtime = parent._runtime()
        body["runtime"] = {"python": runtime["python"], "torch": runtime["torch"],
            "torch_cuda": runtime["torch_cuda"], "device": str(device), "cuda_name": runtime["gpu_name"],
            "cuda_capability": runtime["cuda_capability"], "cuda_visible_devices": runtime["cuda_visible_devices"],
            "torch_num_threads": runtime["torch_num_threads"]}
        batch, fixture = telemetry._fixture()
        body["fixture"] = fixture
        torch.manual_seed(20260929)
        model = WorldModel(TDMPC2ModelConfig(action_dim=3, obs_shape={"rgb": (4, 64, 64)}, episodic=True))
        fixture["model_parameters"] = model.total_params
        if not 4_000_000 <= model.total_params <= 8_000_000:
            raise ValueError("model is not original size5")
        # A loaded parent has nonzero logits. Generated weights avoid the frozen
        # decoder's zero-at-origin derivative without changing its implementation.
        with torch.no_grad():
            model._reward[-1].weight.normal_(0, .02)
        fixture["generated_head_initialization"] = "seed20260929_normal_std0.02_weight_only"
        fixture["generated_head_sha256"] = hashlib.sha256(
            model._reward[-1].weight.detach().contiguous().numpy().tobytes()).hexdigest()
        adapter = RawRewardMSEAdapter(model.to(device))
        _state_parity(adapter, 0)
        torch.cuda.reset_peak_memory_stats(device)

        def step():
            report = adapter.step(batch)
            if any(not math.isfinite(v) for v in report.values()):
                raise ValueError("nonfinite synthetic MSE metrics")
            body["synthetic_only_optimizer_updates"] = adapter.updates
            return report

        telemetry._timed(step, 32, device)
        timed_step = telemetry._timed(step, 128, device)
        context["snapshots"]["mid"] = telemetry._gpu_snapshot(require_idle=False)
        telemetry._check_gpu_context(baseline, context["snapshots"]["mid"], owner)

        def infer():
            with torch.no_grad():
                decoded = two_hot_inv(model.reward(model.encode(batch["obs"].to(device)), batch["action"].to(device)), model.cfg)
                if decoded.shape != (256, 1) or not bool(torch.isfinite(decoded).all()):
                    raise ValueError("nonfinite synthetic inference")
                return decoded

        telemetry._timed(infer, 32, device)
        timed_infer = telemetry._timed(infer, 32, device)
        _state_parity(adapter, 160)
        if not adapter.verify_only_reward_changed():
            raise ValueError("synthetic head did not change")
        body["timing"] = {"scope": "full_adapter_step_including_encode_head_optimizer_state_digest",
            "synchronization": "torch.cuda.synchronize before and after each call",
            "warmup_updates": 32, "timed_updates": 128, "holdout_warmup_batches": 32,
            "holdout_timed_batches": 32, "holdout_batches": 86,
            "step": telemetry._summary(timed_step, 128), "holdout_inference": telemetry._summary(timed_infer, 32)}
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        output_bytes = max(64 * telemetry.MIB, 3 * sum(t.numel() * t.element_size() for t in model.state_dict().values()) + 64 * telemetry.MIB)
        body["forecast"] = _forecast(body["timing"]["step"], body["timing"]["holdout_inference"],
            peak_rss, torch.cuda.max_memory_reserved(device), output_bytes)
        if not body["forecast"]["passed"]:
            raise ValueError("strict 1800-second MSE forecast failed")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if sentinel is not None and baseline is not None:
            try:
                final = telemetry._gpu_snapshot(require_idle=False)
                context["snapshots"]["final"] = final
                telemetry._check_gpu_context(baseline, final, owner)
            except Exception as exc:
                errors.append(f"final GPU context: {exc}")
        if before is not None and device is not None:
            try:
                after = telemetry._raw_resources(device, root)
                peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                peak_cuda = torch.cuda.max_memory_reserved(device)
                output_bytes = body.get("forecast", {}).get("measured_output_bytes", 64 * telemetry.MIB)
                body["resources"] = {"before": before, "after": after, "peak_process_rss_bytes": peak_rss,
                    "max_cuda_reserved_bytes": peak_cuda, "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "forecast_output_bytes": output_bytes, "min_raw_headroom_bytes": telemetry.MIN_HEADROOM,
                    "gpu_reserve_bytes": telemetry.GPU_RESERVE, "gpu_context_telemetry": context}
                telemetry._check_resources(before, after, peak_rss=peak_rss, output_bytes=output_bytes, peak_reserved=peak_cuda)
                if "forecast" in body and (peak_rss != body["forecast"]["measured_peak_rss_bytes"]
                                           or peak_cuda != body["forecast"]["measured_peak_cuda_bytes"]):
                    raise ValueError("post-measurement memory differs from forecast")
            except Exception as exc:
                errors.append(f"final resources: {exc}")
        if sources is not None:
            try:
                body["source_sha256_after"] = _sources(root)
                if body["source_sha256_after"] != sources:
                    raise ValueError("source changed during synthetic benchmark")
            except Exception as exc:
                errors.append(f"source rehash: {exc}")
    if errors:
        body["failure_reasons"] = errors
    else:
        body["status"] = "PASS"
    return telemetry._seal(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-script-sha256")
    parser.add_argument("--benchmark-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.benchmark:
        if not args.expected_script_sha256 or args.protocol_sha256 or args.benchmark_sha256 or args.device != "cuda":
            parser.error("--benchmark requires --expected-script-sha256 and CUDA only; no protocol/replay")
        path = ROOT / BENCHMARK
        if path.exists() or path.is_symlink() or (ROOT / "runs").is_symlink():
            parser.error("exclusive benchmark receipt exists; never overwrite")
        result = benchmark(args.expected_script_sha256)
        telemetry._write_receipt(path, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] == "PASS" else 1
    if args.expected_script_sha256:
        parser.error("expected script SHA is benchmark-only")
    if args.execute:
        if not args.protocol_sha256 or args.benchmark_sha256:
            parser.error("execute requires frozen protocol SHA; benchmark pinned by protocol")
        result = execute(ROOT, args.protocol_sha256, device=args.device)
        print(json.dumps({k: result[k] for k in ("status", "gate", "checkpoint", "optimizer_updates")}, sort_keys=True))
        return 0
    result = preflight(ROOT, args.protocol_sha256, benchmark_sha256=args.benchmark_sha256, device=args.device)
    print(json.dumps({k: result[k] for k in ("status", "schema", "resources", "environment_resets", "torch_load_calls", "optimizer_updates")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
