"""Isolated model-only prior-image treatment on the consumed mixed-source TRAIN replay.

Protocol schema (all keys required, no extras): format, purpose, study_id,
mixed_protocol, each reference in mixed.REFS, cells, datasets, source_sha256,
baseline_results (two original pure-256 receipts/lineages),
expected_lineage_sha256, learner, resources, output_root. Learner contains
seeds, config, base_updates, aux_updates, aux_every_base_updates,
prior_horizon, prior_weight, anchor_mode, aux_rng_mode, aux_rng_seed_base,
aux_rng_seed_stride. Resources are the mixed protocol's exact limits except
min_cgroup_available_bytes=12 GiB. Prepare the empty output_root directory
before invoking, freeze the protocol bytes and pass its SHA. Run one seed per
process, sequentially; this module never creates or extends source archives.

The 320 world-model optimizer steps are NOT a compute-matched or causal
loss-shape comparison with the original pure-256 models. No actor or policy
evaluation, fresh data, environment interaction or promotion occurs here.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import resource
import threading
from typing import Any, Iterator

import numpy as np
import torch

from dreamer_v3 import DreamerV3Agent, NoValidSequenceError
from scripts import train_dreamerv3_prior_image as prior
from scripts import train_dreamerv3_reused_multisource as mixed


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-dreamerv3-reused-train-multisource-prior-v1"
PURPOSE = "reused-TRAIN-prior-image-interaction-diagnostic"
STUDY_ID = "dreamerv3-reused-train-multisource-prior-v1"
PROTOCOL_PATH = "experiments/dreamerv3-reused-train-multisource-prior-v1.json"
MIXED_PROTOCOL_PATH = mixed.PROTOCOL_PATH
MIXED_PROTOCOL_SHA256 = "1a7ddee161bf1958eb5c59152326b6f2fed8899eed41294e55e86edc5ccce42d"
RUN_ROOT = "runs/20260926-dreamerv3-reused-train-multisource-v1/prior-interaction-v1"
RUNNER_SOURCE = "scripts/train_dreamerv3_reused_multisource_prior.py"
PRIOR_RUNNER_SHA256 = "59dbefa7c8d51b1b7808862ccdb7253933f46936f22eeb810a24a7f8c5216f34"
MIXED_RUNNER_SHA256 = "522201e8e608c967b59b5b8e80fff3129e3593aed1f4e6b2922f18c5b0b7aaa7"
LINEAGE_SHA256 = "431990749c7c9069a5fb33269cfd726d38d1ff53a21f50f28f59380a8aade52a"
BASELINE_RESULTS = [
    {"seed": seed,
     "result_path": f"{mixed.OUTPUT_ROOT}/learner-{seed}/training-result.json",
     "result_sha256": digest,
     "lineage_path": f"{mixed.OUTPUT_ROOT}/learner-{seed}/lineage.json",
     "lineage_sha256": LINEAGE_SHA256}
    for seed, digest in (
        (0, "ecb070a98f20a679f3743ed81a1423b4b899696871607a2990c7b7c5e19c6896"),
        (1, "516487dba3ca2b55c018bd2cbf874a820ebb5646ab58dbb06444d99db0aa7dc0"),
    )
]
BASE_UPDATES = 256
AUX_UPDATES = 64
AUX_INTERVAL = 4
HORIZON = 8
WEIGHT = 0.25
MIN_FREE_BYTES = 12 * 1024**3
EMBED_DIM = 128
HIDDEN_DIM = 128
SOURCE_PATHS = mixed.SOURCE_PATHS | {RUNNER_SOURCE, prior.RUNNER_SOURCE, prior.HELPER_SOURCE}
_OVERRIDE_LOCK = threading.Lock()


@contextmanager
def _scoped_output_root() -> Iterator[None]:
    """Only alter the frozen runner's path check while its preflight runs."""
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ValueError("mixed output-root override requires a single-threaded main process")
    if not _OVERRIDE_LOCK.acquire(blocking=False):
        raise ValueError("mixed output-root override is already in use")
    old_root = mixed.OUTPUT_ROOT
    try:
        mixed.OUTPUT_ROOT = RUN_ROOT
        yield
    finally:
        mixed.OUTPUT_ROOT = old_root
        _OVERRIDE_LOCK.release()


def _oom_kills() -> int:
    try:
        lines = Path("/sys/fs/cgroup/memory.events").read_text(encoding="ascii").splitlines()
        values = [int(line.split()[1]) for line in lines if line.split()[:1] == ["oom_kill"]]
    except (OSError, ValueError, IndexError) as exc:
        raise ValueError("cgroup-v2 oom_kill counter is required") from exc
    if len(values) != 1 or values[0] < 0:
        raise ValueError("invalid cgroup-v2 oom_kill counter")
    return values[0]


def _resources_ok(checked: dict[str, Any], *, loading: bool = False) -> dict[str, int]:
    resources = checked["prior_protocol"]["resources"]
    memory = mixed.single._cgroup_memory()
    required = checked["required_available_bytes"] if loading else resources["min_cgroup_available_bytes"]
    if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or memory["available_bytes"] < required
            or mixed._disk_available(checked["output"].parent if loading else checked["output"])
            < (checked["required_disk_bytes"] if loading else resources["min_disk_available_bytes"])):
        raise ValueError("cgroup/disk below pinned 12 GiB training or archive-load floor")
    if _oom_kills() != checked["oom_kills_before"]:
        raise ValueError("cgroup oom_kill counter increased during this run")
    return memory


def _recheck(checked: dict[str, Any], protocol_sha256: str) -> None:
    root, protocol = checked["root"], checked["prior_protocol"]
    mixed._reference(root, PROTOCOL_PATH, protocol_sha256)
    mixed._reference(root, MIXED_PROTOCOL_PATH, MIXED_PROTOCOL_SHA256)
    for key, (name, digest) in mixed.REFS.items():
        mixed._reference(root, name, digest)
    for name, digest in protocol["source_sha256"].items():
        mixed.single._pinned(root, name, digest, "source")
    for row in protocol["datasets"].values():
        mixed.single._pinned(root, row["receipt_path"], row["receipt_sha256"], "receipt")
        if mixed.single._sha256(root / row["archive_path"]) != row["archive_sha256"]:
            raise ValueError("sealed source archive changed during training")
    for row in protocol["baseline_results"]:
        mixed._reference(root, row["result_path"], row["result_sha256"])
        mixed._reference(root, row["lineage_path"], row["lineage_sha256"])


def preflight(protocol_path: Path, protocol_sha256: str, seed: int, output_dir: Path,
              *, repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if mixed.single._relative(root, protocol_path) != PROTOCOL_PATH:
        raise ValueError("separate prior-interaction protocol path is fixed")
    protocol = mixed._reference(root, PROTOCOL_PATH, protocol_sha256)
    if (set(protocol) != {"format", "purpose", "study_id", "mixed_protocol", *mixed.REFS,
                          "source_sha256", "cells", "datasets", "learner", "resources",
                          "output_root", "expected_lineage_sha256", "baseline_results"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != STUDY_ID or protocol["output_root"] != RUN_ROOT
            or protocol["mixed_protocol"] != {"path": MIXED_PROTOCOL_PATH,
                                               "sha256": MIXED_PROTOCOL_SHA256}
            or protocol["expected_lineage_sha256"] != LINEAGE_SHA256):
        raise ValueError("not the separately SHA-frozen mixed-source prior protocol")
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("learner seed must be 0 or 1")
    if mixed.single._relative(root, output_dir) != f"{RUN_ROOT}/learner-{seed}":
        raise ValueError("output must be the isolated prior-interaction learner directory")
    base = mixed._reference(root, MIXED_PROTOCOL_PATH, MIXED_PROTOCOL_SHA256)
    if (base["format"] != mixed.FORMAT or base["purpose"] != mixed.PURPOSE
            or base["study_id"] != mixed.STUDY_ID
            or protocol["cells"] != base["cells"]
            or protocol["datasets"] != base["datasets"]
            or any(protocol[key] != base[key] for key in mixed.REFS)):
        raise ValueError("mixed TRAIN cells, archive actors, receipts or reference identity drift")
    if protocol["baseline_results"] != BASELINE_RESULTS:
        raise ValueError("both original pure-256 model-only receipts and lineages must be pinned")
    for row in BASELINE_RESULTS:
        result = mixed._reference(root, row["result_path"], row["result_sha256"])
        lineage = mixed._reference(root, row["lineage_path"], row["lineage_sha256"])
        if (result.get("format") != mixed.FORMAT + "-result"
                or result.get("status") != "complete" or result.get("seed") != row["seed"]
                or result.get("protocol_sha256") != MIXED_PROTOCOL_SHA256
                or result.get("sources") != base["datasets"]
                or result.get("lineage_sha256") != LINEAGE_SHA256
                or result.get("lineage_path") != row["lineage_path"]
                or result.get("model_only_updates") != BASE_UPDATES
                or result.get("environment_steps") != 0
                or result.get("actor_trained") is not False
                or result.get("actor_critic_target_optimizers_unchanged") is not True
                or result.get("fresh_claim") is not False
                or result.get("promotion_eligible") is not False
                or result.get("decisions") != sum(item["stored_decisions"] for item in base["datasets"].values())
                or result.get("episodes") != 2 * len(mixed.ROADS)
                or lineage.get("format") != mixed.FORMAT + "-lineage"
                or lineage.get("range_convention") != "inclusive"
                or lineage.get("transition_count") != result["decisions"]
                or not isinstance(lineage.get("episodes"), list)
                or len(lineage["episodes"]) != result["episodes"]):
            raise ValueError("original pure-256 result or source-tagged lineage drift")
    resources = protocol["resources"]
    if (not isinstance(resources, dict)
            or resources != {**base["resources"], "min_cgroup_available_bytes": MIN_FREE_BYTES}
            or any(type(value) is not int for value in resources.values())
            or MIN_FREE_BYTES < base["resources"]["min_cgroup_available_bytes"]):
        raise ValueError("resource pins must preserve archive/cgroup/disk caps and the 12 GiB floor")
    learner = protocol["learner"]
    expected_learner = {
            "seeds": [0, 1], "config": base["learner"]["config"],
            "base_updates": BASE_UPDATES, "aux_updates": AUX_UPDATES,
            "aux_every_base_updates": AUX_INTERVAL, "prior_horizon": HORIZON,
            "prior_weight": WEIGHT, "anchor_mode": prior.ANCHOR_MODE,
            "aux_rng_mode": prior.AUX_RNG_MODE, "aux_rng_seed_base": prior.AUX_RNG_SEED_BASE,
            "aux_rng_seed_stride": prior.AUX_RNG_SEED_STRIDE,
    }
    if (not isinstance(learner, dict) or learner != expected_learner
            or any(type(learner[key]) is not type(value) for key, value in expected_learner.items())
            or any(type(seed_value) is not int for seed_value in learner["seeds"])
            or base["learner"]["seeds"] != [0, 1] or base["learner"]["updates"] != BASE_UPDATES
            or BASE_UPDATES // AUX_INTERVAL != AUX_UPDATES
            or learner["config"]["seq_len"] < HORIZON):
        raise ValueError("exact small CPU config and 256 plus 64 H8 model steps required")
    sources = protocol["source_sha256"]
    if (not isinstance(sources, dict) or set(sources) != SOURCE_PATHS
            or any(sources[name] != digest for name, digest in base["source_sha256"].items())
            or sources["scripts/train_dreamerv3_reused_multisource.py"] != MIXED_RUNNER_SHA256
            or sources[prior.RUNNER_SOURCE] != PRIOR_RUNNER_SHA256
            or sources[prior.HELPER_SOURCE] != prior.HELPER_SHA256):
        raise ValueError("new and unchanged old executable source hashes are required")
    for name, digest in sources.items():
        mixed.single._pinned(root, name, digest, "source")
    if (mixed.single._sha256(Path(__file__)) != sources[RUNNER_SOURCE]
            or mixed.single._sha256(Path(prior.__file__)) != PRIOR_RUNNER_SHA256):
        raise ValueError("executing prior runner differs from source-pinned bytes")
    aux_update = prior._load_aux_update(root, prior.HELPER_SHA256)
    with _scoped_output_root():
        checked = mixed.preflight(MIXED_PROTOCOL_PATH, MIXED_PROTOCOL_SHA256,
                                  seed, output_dir, repo_root=root)
    if (checked["protocol"] != base or checked["config"].device != "cpu"
            or checked["config"].replay_capacity != mixed.REPLAY_CAPACITY
            or checked["config"].embed_dim != EMBED_DIM or checked["config"].hidden_dim != HIDDEN_DIM
            or checked["output"] != root / RUN_ROOT / f"learner-{seed}"):
        raise ValueError("mixed-source preflight did not retain the frozen CPU learner")
    checked.update(prior_protocol=protocol, output=checked["output"], aux_update=aux_update,
                   required_available_bytes=max(checked["required_available_bytes"], MIN_FREE_BYTES),
                   oom_kills_before=_oom_kills())
    _resources_ok(checked, loading=True)
    return checked


def _model_finite(agent: DreamerV3Agent) -> bool:
    for module in (agent.encoder, agent.rssm, agent.decoder, agent.reward_head, agent.continue_head):
        if any(not torch.isfinite(param).all().item() for param in module.parameters()):
            return False
    return all(torch.isfinite(value).all().item() for state in agent.wm_optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor))


def train(protocol_path: Path, protocol_sha256: str, seed: int, output_dir: Path,
          *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, seed, output_dir, repo_root=repo_root)
    output = checked["output"]
    output.mkdir(exist_ok=False)
    base_done = aux_done = optimizer_steps = 0
    phase = "source and archive verification"
    try:
        _resources_ok(checked, loading=True)
        _recheck(checked, protocol_sha256)
        replay, lineage, audit = mixed.materialize(checked)
        if (replay.size != sum(row["stored_decisions"] for row in checked["prior_protocol"]["datasets"].values())
                or replay.total_steps != replay.size or replay.capacity != checked["config"].replay_capacity
                or len(lineage) != 2 * len(mixed.ROADS)):
            raise ValueError("mixed replay capacity, episode or source lineage changed")
        phase = "lineage sidecar"
        lineage_path = output / "lineage.json"
        mixed.single._write_json(lineage_path, {
            "format": mixed.FORMAT + "-lineage", "range_convention": "inclusive",
            "episodes": lineage, "transition_count": replay.size,
        })
        lineage_sha = mixed.single._sha256(lineage_path)
        if lineage_sha != LINEAGE_SHA256:
            raise ValueError("combined episode/sequence lineage differs from both pure-256 runs")
        lineage_name = mixed.single._relative(checked["root"], lineage_path)
        phase = "base and auxiliary model-only updates"
        _resources_ok(checked, loading=True)
        torch.set_num_threads(1)
        agent = DreamerV3Agent(checked["config"], seed=seed)
        agent.replay = replay
        snapshot = mixed._snapshot(agent)

        def counted_step(_optimizer: Any, _args: Any, _kwargs: Any) -> None:
            nonlocal optimizer_steps
            optimizer_steps += 1

        handle = agent.wm_optimizer.register_step_post_hook(counted_step)
        try:
            with ((output / "base-update-metrics.jsonl").open("x", encoding="utf-8") as base_stream,
                  (output / "aux-update-metrics.jsonl").open("x", encoding="utf-8") as aux_stream):
                for index in range(BASE_UPDATES):
                    _resources_ok(checked)
                    previous = agent.gradient_steps
                    try:
                        metrics = agent.update(model_only=True)
                    except NoValidSequenceError as exc:
                        raise ValueError("no valid same-source model-only sequence") from exc
                    if (not prior._metrics(metrics, base=True)
                            or any(key.startswith("loss_aux") for key in metrics)
                            or agent.gradient_steps != previous + 1
                            or optimizer_steps != base_done + aux_done + 1
                            or agent.environment_steps != 0 or not mixed._unchanged(agent, snapshot)
                            or not _model_finite(agent)):
                        raise ValueError("base model-only metrics, optimizer or actor/critic state invalid")
                    _resources_ok(checked)
                    base_stream.write(json.dumps({"base_update": index + 1, "metrics": metrics},
                                                 sort_keys=True, allow_nan=False) + "\n")
                    base_stream.flush()
                    base_done += 1
                    if base_done % AUX_INTERVAL:
                        continue
                    _resources_ok(checked)
                    previous = agent.gradient_steps
                    aux_seed = prior.AUX_RNG_SEED_BASE + seed * prior.AUX_RNG_SEED_STRIDE + aux_done + 1
                    numpy_state, python_state = np.random.get_state(), random.getstate()
                    try:
                        with torch.random.fork_rng(devices=[], enabled=True):
                            torch.random.default_generator.manual_seed(aux_seed)
                            np.random.seed(aux_seed)
                            random.seed(aux_seed)
                            aux_metrics = checked["aux_update"](agent, horizon=HORIZON, weight=WEIGHT)
                    finally:
                        np.random.set_state(numpy_state)
                        random.setstate(python_state)
                    if (not prior._metrics(aux_metrics, base=False)
                            or not prior._aux_targets(aux_metrics, checked["config"].batch_size)
                            or agent.gradient_steps != previous
                            or optimizer_steps != base_done + aux_done + 1
                            or agent.environment_steps != 0 or not mixed._unchanged(agent, snapshot)
                            or not _model_finite(agent)):
                        raise ValueError("aux update metrics, H8 targets, optimizer or actor/critic state invalid")
                    _resources_ok(checked)
                    aux_stream.write(json.dumps({"aux_update": aux_done + 1,
                                                 "after_base_update": base_done, "metrics": aux_metrics},
                                                sort_keys=True, allow_nan=False) + "\n")
                    aux_stream.flush()
                    aux_done += 1
        finally:
            handle.remove()
        if (base_done != BASE_UPDATES or aux_done != AUX_UPDATES
                or optimizer_steps != BASE_UPDATES + AUX_UPDATES
                or agent.gradient_steps != BASE_UPDATES or agent.environment_steps != 0
                or not mixed._unchanged(agent, snapshot) or not _model_finite(agent)):
            raise ValueError("final optimizer budget or model-only invariant failed")
        phase = "source recheck and checkpoint"
        _recheck(checked, protocol_sha256)
        _resources_ok(checked)
        if mixed.single._sha256(lineage_path) != lineage_sha:
            raise ValueError("lineage sidecar changed before checkpoint")
        base_path, aux_path = output / "base-update-metrics.jsonl", output / "aux-update-metrics.jsonl"
        metadata = {
            "purpose": PURPOSE, "study_id": STUDY_ID, "seed": seed,
            "protocol_sha256": protocol_sha256,
            "mixed_protocol": checked["prior_protocol"]["mixed_protocol"],
            "pure_256_baselines": checked["prior_protocol"]["baseline_results"],
            "source_sha256": checked["prior_protocol"]["source_sha256"],
            "sources": checked["prior_protocol"]["datasets"],
            "lineage_path": lineage_name, "lineage_sha256": lineage_sha,
            "base_metrics_sha256": mixed.single._sha256(base_path),
            "aux_metrics_sha256": mixed.single._sha256(aux_path),
            "base_model_only_updates": base_done, "aux_world_model_optimizer_steps": aux_done,
            "world_model_optimizer_steps": optimizer_steps, "aux_after_every_base_updates": AUX_INTERVAL,
            "prior_horizon": HORIZON, "prior_weight": WEIGHT, "anchor_mode": prior.ANCHOR_MODE,
            "aux_rng_mode": prior.AUX_RNG_MODE, "aux_rng_seed_base": prior.AUX_RNG_SEED_BASE,
            "aux_rng_seed_stride": prior.AUX_RNG_SEED_STRIDE,
            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
            "promotion_eligible": False, "matched_pure_256_step_model": False,
        }
        checkpoint_path = output / "world-model-checkpoint.pt"
        agent.save_checkpoint(checkpoint_path, run_metadata=metadata)
        _recheck(checked, protocol_sha256)
        post_cgroup = _resources_ok(checked)
        if mixed.single._sha256(lineage_path) != lineage_sha:
            raise ValueError("lineage sidecar changed after checkpoint")
        result = {
            "format": FORMAT + "-result", "status": "complete", **metadata,
            "source_audits": audit, "distinct_training_roads": len(mixed.ROADS),
            "episodes": len(lineage), "decisions": replay.size,
            "union_finished_roads": sorted(checked["finished"]["source0"] | checked["finished"]["source1"]),
            "environment_steps": agent.environment_steps, "actor_critic_target_optimizers_unchanged": True,
            "resource_snapshot": {
                "preflight_cgroup": checked["cgroup"], "post_cgroup": post_cgroup,
                "required_available_bytes": checked["required_available_bytes"],
                "replay_memory_bytes": replay.memory_bytes, "oom_kills_before_and_after": checked["oom_kills_before"],
                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "checkpoint_path": checkpoint_path.name, "checkpoint_sha256": mixed.single._sha256(checkpoint_path),
            "base_metrics_path": base_path.name, "aux_metrics_path": aux_path.name,
        }
        mixed.single._write_json(output / "training-result.json", result)
        return result
    except BaseException as exc:
        mixed.single._write_json(output / "abort.json", {
            "format": FORMAT + "-abort", "status": "aborted", "purpose": PURPOSE,
            "study_id": STUDY_ID, "seed": seed, "protocol_sha256": protocol_sha256,
            "phase": phase, "completed_base_model_only_updates": base_done,
            "completed_aux_world_model_optimizer_steps": aux_done,
            "observed_world_model_optimizer_steps": optimizer_steps,
            "error_type": type(exc).__name__, "error": str(exc),
            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
            "promotion_eligible": False,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(train(args.protocol, args.protocol_sha256, args.seed, args.output,
                           repo_root=args.repo_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
