"""Non-promoting prior-image auxiliary diagnostic on sealed, reused TRAIN replay.

Run one arm and one learner seed per invocation. This treatment uses 256 standard
model-only updates PLUS 64 separate world-model optimizer steps, not a matched
256-step comparison with the frozen original model.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import resource
from typing import Any

import numpy as np
import torch

from dreamer_v3 import DreamerV3Agent, NoValidSequenceError
from haic.algorithms.dreamer_v3.offline import replay_from_dataset_bytes
from scripts import train_dreamerv3_reused_train as original


ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-dreamerv3-reused-prior-image-offline-v1"
PURPOSE = "reused-TRAIN-prior-image-auxiliary-diagnostic"
RESULT_FORMAT = "haic-dreamerv3-reused-prior-image-result-v1"
BASE_PROTOCOL_PATH = "experiments/dreamerv3-reused-train-offline-v2.json"
BASE_PROTOCOL_SHA256 = "ba837ebc3f64282bbd77b4444c6f337c23a3ebce12602d328e9626c1c406b290"
RUN_ROOT = "runs/20260926-dreamerv3-reused-train-diagnostic-v1/offline-prior-v1"
RUNNER_SOURCE = "scripts/train_dreamerv3_prior_image.py"
HELPER_SOURCE = "haic/algorithms/dreamer_v3/prior_image.py"
HELPER_SHA256 = "bb878da6f8903cee17383061e759baf2c87abd4101e78c9dc27cf3d7bfed1166"
SOURCE_PATHS = original.SOURCE_PATHS | {RUNNER_SOURCE, HELPER_SOURCE}
BASE_UPDATES = 256
AUX_UPDATES = 64
AUX_INTERVAL = 4
HORIZON = 8
WEIGHT = 0.25
ANCHOR_MODE = "deterministic-first-last-full-horizon"
MIN_FREE_BYTES = 8 * 1024**3
AUX_RNG_MODE = "fork-torch-numpy-python"
AUX_RNG_SEED_BASE = 260926000
AUX_RNG_SEED_STRIDE = 1000


def _same_state(before: Any, after: Any) -> bool:
    if type(before) is not type(after):
        return False
    if isinstance(before, torch.Tensor):
        return torch.equal(before, after)
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            _same_state(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, (list, tuple)):
        return len(before) == len(after) and all(
            _same_state(left, right) for left, right in zip(before, after)
        )
    return before == after


def _metrics(value: Any, *, base: bool) -> bool:
    return (isinstance(value, dict) and bool(value)
            and (("loss_wm" in value) if base else {
                "aux_loss", "frame_mse", "anchor_count", "target_count", "effective_horizon",
            } <= value.keys())
            and all(isinstance(key, str) and key
                    and not key.startswith(("loss_actor", "loss_critic"))
                    and type(number) in (int, float) and math.isfinite(number)
                    for key, number in value.items()))


def _aux_targets(metrics: dict[str, float], batch_size: int) -> bool:
    """Normal T32 samples need two *full* H8 rollouts; short T may have fewer."""
    horizon = metrics["effective_horizon"]
    anchors = metrics["anchor_count"]
    if (horizon not in range(1, HORIZON + 1)
            or anchors not in (batch_size, 2 * batch_size)):
        return False
    target_count = (anchors * HORIZON if horizon == HORIZON
                    else batch_size * (horizon if anchors == batch_size else horizon + 1))
    return (metrics["target_count"] == target_count
            and ("clipped_pixel_fraction" not in metrics
                 or 0 <= metrics["clipped_pixel_fraction"] <= 1))


def _load_aux_update(root: Path, expected_sha256: str):
    module = importlib.import_module("haic.algorithms.dreamer_v3.prior_image")
    if (not getattr(module, "__file__", None)
            or Path(module.__file__).resolve() != (root / HELPER_SOURCE).resolve()
            or original._sha256(Path(module.__file__)) != expected_sha256):
        raise ValueError("executing prior-image helper differs from pinned executable source")
    return module.prior_image_aux_update


def _resources_ok(protocol: dict[str, Any], required: int) -> dict[str, int]:
    memory = original._cgroup_memory()
    resources = protocol["resources"]
    if (memory["limit_bytes"] > resources["max_cgroup_memory_bytes"]
            or memory["available_bytes"] < required):
        raise ValueError("cgroup memory lacks frozen 8 GiB resource floor")
    return memory


def preflight(protocol_path: Path, protocol_sha256: str, arm: str, seed: int,
              output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Reapply the original dual-arm sealed preflight without opening archive bytes."""
    root = Path(repo_root).resolve()
    name = original._relative(root, protocol_path)
    if not name.startswith("experiments/dreamerv3-reused-train-prior-image-"):
        raise ValueError("a separate reused-TRAIN prior-image protocol is required")
    protocol = original._json(original._pinned(root, name, protocol_sha256, "protocol"))
    if (set(protocol) != {"format", "purpose", "study_id", "base_offline_protocol",
                          "source_sha256", "learner", "resources", "output_root"}
            or protocol["format"] != FORMAT or protocol["purpose"] != PURPOSE
            or protocol["study_id"] != "dreamerv3-reused-train-diagnostic-v1"
            or protocol["base_offline_protocol"] != {
                "path": BASE_PROTOCOL_PATH, "sha256": BASE_PROTOCOL_SHA256,
            } or protocol["output_root"] != RUN_ROOT):
        raise ValueError("wrong prior-image protocol or frozen base protocol")
    if arm not in ("random", "teacher") or type(seed) is not int or seed not in (0, 1):
        raise ValueError("one frozen data arm and learner seed 0 or 1 required")
    output_name = original._relative(root, output_dir)
    if output_name != f"{RUN_ROOT}/{arm}-seed-{seed}":
        raise ValueError("output must use the isolated frozen prior-image arm/seed path")
    output = original._path(root, output_name, "output", existing=False)
    if not output.parent.is_dir():
        if (output.parent.exists() or output.parent.is_symlink()
                or not output.parent.parent.is_dir()):
            raise ValueError("isolated output parent must be a new directory under the existing run root")
        original_preflight_output = output.parent
    else:
        original_preflight_output = output

    # The frozen original checks both collection arms, source/receipt/archive paths,
    # r6 TRAIN membership and exclusions, replay capacity, CPU config, ZIP size and cgroup.
    checked = original.preflight(BASE_PROTOCOL_PATH, BASE_PROTOCOL_SHA256, arm, seed,
                                 original_preflight_output, repo_root=root)
    base = checked["protocol"]
    if (base["study_id"] != protocol["study_id"] or base["learner"]["seeds"] != [0, 1]
            or base["learner"]["updates"] != BASE_UPDATES
            or checked["config"].device != "cpu"
            or checked["config"].seq_len < HORIZON):
        raise ValueError("base learner budget, seeds, horizon, CPU or study changed")
    learner = protocol["learner"]
    if (not isinstance(learner, dict) or learner != {
        "base_updates": BASE_UPDATES, "aux_updates": AUX_UPDATES,
        "aux_every_base_updates": AUX_INTERVAL, "prior_horizon": HORIZON,
        "prior_weight": WEIGHT, "anchor_mode": ANCHOR_MODE,
        "aux_rng_mode": AUX_RNG_MODE, "aux_rng_seed_base": AUX_RNG_SEED_BASE,
        "aux_rng_seed_stride": AUX_RNG_SEED_STRIDE,
    }):
        raise ValueError("prior learner must freeze 256 base plus 64 auxiliary optimizer steps")
    if (protocol["resources"] != base["resources"]
            or base["resources"]["min_cgroup_available_bytes"] < MIN_FREE_BYTES):
        raise ValueError("resource limits cannot weaken the frozen 8 GiB floor")
    sources = protocol["source_sha256"]
    if (not isinstance(sources, dict) or set(sources) != SOURCE_PATHS
            or sources[HELPER_SOURCE] != HELPER_SHA256
            or any(sources[name] != base["source_sha256"][name]
                   for name in original.SOURCE_PATHS)):
        raise ValueError("prior-image protocol must pin the corrected helper and unchanged original sources")
    for source_name, digest in sources.items():
        original._pinned(root, source_name, digest, "source")
    if sources[RUNNER_SOURCE] != original._sha256(Path(__file__)):
        raise ValueError("executing prior-image runner differs from pinned executable source")
    checked.update(prior_protocol=protocol, prior_protocol_sha256=protocol_sha256,
                   prior_output=output, required_available_bytes=max(
                       checked["required_available_bytes"], MIN_FREE_BYTES,
                   ))
    _resources_ok(protocol, checked["required_available_bytes"])
    return checked


def _recheck_sources(checked: dict[str, Any], protocol_path: Path, protocol_sha256: str) -> None:
    root = checked["root"]
    protocol = checked["prior_protocol"]
    original._pinned(root, original._relative(root, protocol_path), protocol_sha256, "protocol")
    original._pinned(root, BASE_PROTOCOL_PATH, BASE_PROTOCOL_SHA256, "protocol")
    ref = checked["protocol"]["collection_protocol"]
    original._pinned(root, ref["path"], ref["sha256"], "protocol")
    for source_name, digest in checked["protocol"]["source_sha256"].items():
        original._pinned(root, source_name, digest, "source")
    for source_name, digest in protocol["source_sha256"].items():
        original._pinned(root, source_name, digest, "source")
    collection = original._json(root / ref["path"])
    for source_name, digest in collection["source_sha256"].items():
        original._pinned(root, source_name, digest, "source")
    for label, item in checked["datasets"].items():
        row = item["row"]
        original._pinned(root, row["receipt_path"], row["receipt_sha256"], "receipt")
        if original._sha256(item["archive"]) != row["archive_sha256"]:
            raise ValueError(f"{label}: frozen archive changed during prior-image training")


def train_arm(protocol_path: Path, protocol_sha256: str, arm: str, seed: int,
              output_dir: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    checked = preflight(protocol_path, protocol_sha256, arm, seed, output_dir, repo_root=repo_root)
    protocol = checked["prior_protocol"]
    base = checked["protocol"]
    row = checked["datasets"][arm]["row"]
    output = checked["prior_output"]
    if not output.parent.is_dir():
        output.parent.mkdir(exist_ok=False)
    output.mkdir(exist_ok=False)
    base_done = aux_done = optimizer_steps = 0
    phase = "source and archive verification"
    try:
        aux_update = _load_aux_update(checked["root"], protocol["source_sha256"][HELPER_SOURCE])
        _resources_ok(protocol, checked["required_available_bytes"])
        _recheck_sources(checked, protocol_path, protocol_sha256)
        archive = checked["datasets"][arm]["archive"]
        data = original._archive_bytes(
            archive, expected_size=checked["datasets"][arm]["size"],
            cap=protocol["resources"]["max_archive_bytes"],
        )
        if hashlib.sha256(data).hexdigest() != row["archive_sha256"]:
            raise ValueError("dataset archive SHA-256 differs from frozen protocol")
        phase = "sealed replay materialization"
        replay, audit = replay_from_dataset_bytes(
            data, archive_sha256=row["archive_sha256"], dataset_digest=row["dataset_digest"],
            source_id=row["source_id"], source_actor_sha256=row["source_actor_sha256"],
            allowed_cells=[(cell["track_id"], cell["geometry_seed"]) for cell in row["allowed_cells"]],
            excluded_seeds=row["excluded_seeds"], capacity=checked["config"].replay_capacity,
        )
        with np.load(BytesIO(data), allow_pickle=False) as sealed:
            episodes = json.loads(sealed["manifest"].tobytes())["episodes"]
            offsets = sealed["transition_offsets"]
            finished = sealed["finished"]
            finished_cells = sorted({
                (episode["track_id"], int(episode["geometry_id"]))
                for episode, end in zip(episodes, offsets[1:]) if bool(finished[end - 1])
            })
        del data
        if (audit["decisions"] != row["stored_decisions"] or replay.size != row["stored_decisions"]
                or len(finished_cells) != audit["distinct_finished_cells"]):
            raise ValueError("actual stored TRAIN decisions differ from frozen collection")
        terminal_events = int(replay.is_terminal[:replay.size].sum())
        phase = "base and auxiliary model-only updates"
        torch.set_num_threads(1)
        agent = DreamerV3Agent(checked["config"], seed=seed)
        agent.replay = replay
        snapshots = {
            name: {key: value.detach().clone() for key, value in getattr(agent, name).state_dict().items()}
            for name in ("actor", "critic", "critic_target")
        }
        optimizers = {name: copy.deepcopy(getattr(agent, name).state_dict())
                      for name in ("actor_optimizer", "critic_optimizer")}

        def counted_step(_optimizer, _args, _kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1

        handle = agent.wm_optimizer.register_step_post_hook(counted_step)
        try:
            with ((output / "base-update-metrics.jsonl").open("x", encoding="utf-8") as base_stream,
                  (output / "aux-update-metrics.jsonl").open("x", encoding="utf-8") as aux_stream):
                for index in range(BASE_UPDATES):
                    previous = agent.gradient_steps
                    try:
                        base_metrics = agent.update(model_only=True)
                    except NoValidSequenceError as exc:
                        raise original.UnavailableSequenceError(
                            f"unavailable replay sequence at base update {index + 1}"
                        ) from exc
                    if not base_metrics:
                        raise original.UnavailableSequenceError(
                            f"unavailable replay sequence at base update {index + 1}"
                        )
                    if (not _metrics(base_metrics, base=True) or agent.gradient_steps != previous + 1
                            or optimizer_steps != base_done + aux_done + 1
                            or agent.environment_steps != 0 or not original._unchanged(agent, snapshots)
                            or any(not _same_state(snapshot, getattr(agent, name).state_dict())
                                   for name, snapshot in optimizers.items())):
                        raise ValueError("base update changed actor/critic/target, optimizer, counter or metrics")
                    _resources_ok(protocol, protocol["resources"]["min_cgroup_available_bytes"])
                    base_stream.write(json.dumps({"base_update": index + 1, "metrics": base_metrics},
                                                 sort_keys=True, allow_nan=False) + "\n")
                    base_stream.flush()
                    base_done += 1
                    if base_done % AUX_INTERVAL == 0:
                        previous = agent.gradient_steps
                        aux_seed = AUX_RNG_SEED_BASE + seed * AUX_RNG_SEED_STRIDE + aux_done + 1
                        numpy_state = np.random.get_state()
                        python_state = random.getstate()
                        try:
                            with torch.random.fork_rng(devices=[], enabled=True):
                                torch.random.default_generator.manual_seed(aux_seed)
                                np.random.seed(aux_seed)
                                random.seed(aux_seed)
                                aux_metrics = aux_update(agent, horizon=HORIZON, weight=WEIGHT)
                        finally:
                            np.random.set_state(numpy_state)
                            random.setstate(python_state)
                        if (not _metrics(aux_metrics, base=False)
                                or not _aux_targets(aux_metrics, checked["config"].batch_size)
                                or agent.gradient_steps != previous
                                or optimizer_steps != base_done + aux_done + 1
                                or agent.environment_steps != 0 or not original._unchanged(agent, snapshots)
                                or any(not _same_state(snapshot, getattr(agent, name).state_dict())
                                       for name, snapshot in optimizers.items())):
                            raise ValueError("aux update changed actor/critic/target, optimizer, counter or metrics")
                        _resources_ok(protocol, protocol["resources"]["min_cgroup_available_bytes"])
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
                or not original._unchanged(agent, snapshots)
                or any(not _same_state(snapshot, getattr(agent, name).state_dict())
                       for name, snapshot in optimizers.items())):
            raise ValueError("base/aux optimizer budget or actor/critic/target invariants failed")
        _recheck_sources(checked, protocol_path, protocol_sha256)
        _resources_ok(protocol, protocol["resources"]["min_cgroup_available_bytes"])
        phase = "checkpoint and receipt"
        checkpoint_path = output / "world-model-checkpoint.pt"
        base_metrics_path = output / "base-update-metrics.jsonl"
        aux_metrics_path = output / "aux-update-metrics.jsonl"
        base_metrics_sha256 = original._sha256(base_metrics_path)
        aux_metrics_sha256 = original._sha256(aux_metrics_path)
        agent.save_checkpoint(checkpoint_path, run_metadata={
            "purpose": PURPOSE, "study_id": protocol["study_id"], "arm": arm, "seed": seed,
            "prior_protocol_sha256": protocol_sha256, "base_offline_protocol_sha256": BASE_PROTOCOL_SHA256,
            "source_sha256": protocol["source_sha256"],
            "collection_receipt_sha256": row["receipt_sha256"],
            "archive_sha256": row["archive_sha256"], "dataset_digest": row["dataset_digest"],
            "base_metrics_sha256": base_metrics_sha256, "aux_metrics_sha256": aux_metrics_sha256,
            "base_model_only_updates": base_done,
            "aux_world_model_optimizer_steps": aux_done, "world_model_optimizer_steps": optimizer_steps,
            "prior_horizon": HORIZON, "prior_weight": WEIGHT, "anchor_mode": ANCHOR_MODE,
            "aux_rng_mode": AUX_RNG_MODE, "aux_rng_seed_base": AUX_RNG_SEED_BASE,
            "aux_rng_seed_stride": AUX_RNG_SEED_STRIDE,
            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
            "promotion_eligible": False,
        })
        _resources_ok(protocol, protocol["resources"]["min_cgroup_available_bytes"])
        result = {
            "format": RESULT_FORMAT, "purpose": PURPOSE, "study_id": protocol["study_id"],
            "status": "complete", "arm": arm, "seed": seed,
            "prior_protocol_sha256": protocol_sha256,
            "base_offline_protocol_path": BASE_PROTOCOL_PATH,
            "base_offline_protocol_sha256": BASE_PROTOCOL_SHA256,
            "collection_protocol_sha256": base["collection_protocol"]["sha256"],
            "source_sha256": protocol["source_sha256"],
            "collection_receipt_path": row["receipt_path"], "collection_receipt_sha256": row["receipt_sha256"],
            "archive_path": row["archive_path"], "archive_sha256": row["archive_sha256"],
            "dataset_digest": row["dataset_digest"], "source_id": row["source_id"],
            "source_actor_sha256": row["source_actor_sha256"],
            "dataset_evidence": {
                **audit, "terminal_events": terminal_events,
                "finished_cells": [{"track_id": track, "geometry_seed": road}
                                   for track, road in finished_cells],
            },
            "collection_decisions_spent": checked["datasets"][arm]["receipt"]["decisions_spent"],
            "environment_steps": agent.environment_steps, "base_model_only_updates": base_done,
            "aux_world_model_optimizer_steps": aux_done,
            "world_model_optimizer_steps": optimizer_steps,
            "aux_after_every_base_updates": AUX_INTERVAL, "prior_horizon": HORIZON,
            "prior_weight": WEIGHT, "anchor_mode": ANCHOR_MODE,
            "aux_rng_mode": AUX_RNG_MODE, "aux_rng_seed_base": AUX_RNG_SEED_BASE,
            "aux_rng_seed_stride": AUX_RNG_SEED_STRIDE,
            "actor_critic_target_unchanged": True, "actor_critic_optimizers_unchanged": True,
            "actor_trained": False, "fresh_claim": False, "p1b_claim": False,
            "promotion_eligible": False, "matched_pure_256_step_model": False,
            "resource_snapshot": {
                "preflight_cgroup": checked["cgroup"],
                "required_available_bytes": checked["required_available_bytes"],
                "archive_size_bytes": checked["datasets"][arm]["size"],
                "replay_memory_bytes": replay.memory_bytes, "post_cgroup": original._cgroup_memory(),
                "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": original._sha256(checkpoint_path),
            "base_metrics_path": base_metrics_path.name,
            "base_metrics_sha256": base_metrics_sha256,
            "aux_metrics_path": aux_metrics_path.name,
            "aux_metrics_sha256": aux_metrics_sha256,
        }
        original._write_json(output / "training-result.json", result)
        return result
    except BaseException as exc:
        original._write_json(output / "abort.json", {
            "format": "haic-dreamerv3-reused-prior-image-abort-v1", "purpose": PURPOSE,
            "study_id": protocol["study_id"], "arm": arm, "seed": seed,
            "prior_protocol_sha256": protocol_sha256,
            "base_offline_protocol_sha256": BASE_PROTOCOL_SHA256,
            "phase": phase, "completed_base_model_only_updates": base_done,
            "completed_aux_world_model_optimizer_steps": aux_done,
            "observed_world_model_optimizer_steps": optimizer_steps,
            "error_type": type(exc).__name__, "error": str(exc),
            "actor_trained": False, "fresh_claim": False,
            "p1b_claim": False, "promotion_eligible": False,
        })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--arm", required=True, choices=("random", "teacher"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = train_arm(args.protocol, args.protocol_sha256, args.arm, args.seed,
                       args.output, repo_root=args.repo_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
