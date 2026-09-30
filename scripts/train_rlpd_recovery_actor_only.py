"""Zero-reset constrained offline actor repair, not SAC or generic imitation.

Only proven failure Oracle actions guide the native three-action mean. Ordinary
prior and consumed positive-preservation snapshots retain the original V5 mean
on exactly the same images. Freeze/run are explicit; preflight can precede freeze.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from haic.algorithms.rlpd.agent import PixelRLPDAgent
from haic.algorithms.rlpd.recovery import FINISH_POLICY, import_learning_state, load_recovery_replay
from scripts import train_rlpd_recovery as parent
from scripts.rlpd_common import canonical_sha256, load_offline_replay, sha256_file


FORMAT = "haic-rlpd-recovery-actor-only-v1"
DATASET = "runs/rlpd-recovery-training-data-v1"
DATASET_SHA = "40ee39ac4ba6cab2b46ada76c028806d1d476903e5a912bb0108e5ba317c350d"
PROTECTED = "runs/rlpd-recovery-evaluation-r2"
PROTECTED_SHA = "13ab192252078e78a3cec7897505cb0ea4331fa136de828cbc5404bb35d55612"
ACTOR_SHA = "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98"
FINISHED = frozenset(4272000000 + i for i in (1, 6, 8, 9, 12))
SEEDS = tuple(range(4272000001, 4272000013))
SOURCE_FILES = parent.SOURCE_FILES | frozenset({
    "scripts/train_rlpd_recovery_actor_only.py", "tests/test_rlpd_recovery_actor_only.py"})
BUDGET = {"seed": 60, "actor_only_updates": 2048, "sac_updates": 0,
          "environment_resets": 0, "environment_steps": 0,
          "batch": {"guide": 16, "prior": 32, "protected": 16},
          "guide_weight": 1.0, "retention_weight": 1.0, "gradient_clip": 10.0,
          "feature_batch_size": 128, "torch_threads": 1}


def state_sha(value):
    """Hash tensor bytes plus structure, including optimizer moments/steps."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def deduplicate(images, targets=None):
    """Fixed uint8 image pool; multiplicities preserve the original row measure."""
    unique, weights, labels, seen = [], [], [], {}
    for i, image in enumerate(images):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != (4, 84, 84):
            raise ValueError("expected uint8 four-frame images")
        key = np.ascontiguousarray(image).tobytes()
        if key in seen:
            index = seen[key]
            if targets is not None and not np.array_equal(labels[index], targets[i]):
                raise ValueError("same guide image has conflicting executed targets")
            weights[index] += 1
        else:
            seen[key] = len(unique)
            unique.append(image.copy())
            weights.append(1)
            if targets is not None:
                labels.append(np.asarray(targets[i], dtype=np.float32))
    if not unique:
        raise ValueError("empty fixed image pool")
    pool = {"images": np.stack(unique), "weights": np.asarray(weights, np.int64)}
    if targets is not None:
        pool["targets"] = np.stack(labels)
    return pool


def pool_receipt(pool):
    return {"rows": int(pool["weights"].sum()), "unique_images": len(pool["images"]),
            "images_sha256": hashlib.sha256(pool["images"].tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(pool["weights"].tobytes()).hexdigest(),
            **({"targets_sha256": hashlib.sha256(pool["targets"].tobytes()).hexdigest()}
               if "targets" in pool else {})}


def validate_snapshots(data, receipt, seed):
    n = receipt["steps"]
    ended = data["terminated"] | data["truncated"]
    if (receipt.get("actor_id") != "v5" or receipt.get("actor_sha256") != ACTOR_SHA
            or receipt.get("geometry_seed") != seed or receipt.get("track_id") != 1
            or receipt.get("censored") is not False or receipt.get("mode") != "policy"
            or receipt.get("source_road_hash_matches") is not True
            or type(n) is not int or not 0 < n <= 2000
            or ended.shape != (n,) or ended[:-1].any() or not ended[-1]
            or not (receipt.get("terminated") or receipt.get("truncated"))
            or bool(data["terminated"][-1]) != receipt["terminated"]
            or bool(data["truncated"][-1]) != receipt["truncated"]
            or not np.array_equal(data["step"], np.arange(n))
            or receipt.get("finished") != (seed in FINISHED)
            or bool(data["finished"][-1]) != receipt["finished"]
            or receipt["reason"] != ("finished" if seed in FINISHED else str(data["retire_reason"][-1]))
            or seed in FINISHED and str(data["retire_reason"][-1]) != ""):
        raise ValueError("protected trace lacks original identity/full real ending")
    steps, images = data["snapshot_steps"], data["observation_snapshots"]
    if (not np.array_equal(steps, np.arange(0, n, 25))
            or images.shape != (len(steps), 4, 84, 84) or images.dtype != np.uint8):
        raise ValueError("protected saved snapshots are incomplete or malformed")
    for step, image in zip(steps, images):
        normalized = image.astype(np.float32) / 255.0
        digest = hashlib.sha256(normalized.tobytes()).hexdigest()
        if digest != str(data["observation_sha256"][step]):
            raise ValueError("protected snapshot image/observation hash mismatch")
    if str(data["observation_sha256"][0]) != receipt["initial_observation_sha256"]:
        raise ValueError("protected initial image differs from receipt")
    # The initial-state stratum includes all twelve roads, independently of the
    # positive-road stratum. Deduplication preserves these five double weights.
    return np.concatenate((images[:1], images)) if seed in FINISHED else images[:1]


def load_protected():
    manifest_path = parent.pinned(f"{PROTECTED}/manifest.json", PROTECTED_SHA, "runs")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("episodes_complete") != 36:
        raise ValueError("protected evaluation must be the complete r2")
    files = manifest["files_sha256"]
    protocol_path = parent.pinned(f"{PROTECTED}/protocol.json", files["protocol.json"], "runs")
    protocol = json.loads(protocol_path.read_text())
    contract = protocol["contract"]
    if (protocol["actors"]["v5"]["sha256"] != ACTOR_SHA
            or contract["partition"] != "consumed TRAIN" or contract["track_id"] != 1
            or contract["geometry_seeds"] != list(SEEDS) or contract["frame_skip"] != 4
            or contract["obstacles"] is not True or contract["reward_shaping"] is not False):
        raise ValueError("protected evaluation exposure/actor contract changed")
    images, sources = [], []
    for seed in SEEDS:
        name = f"v5-seed-{seed}"
        receipt_name, trace_name = f"{name}-receipt.json", f"{name}.npz"
        receipt_path = parent.pinned(f"{PROTECTED}/{receipt_name}", files[receipt_name], "runs")
        trace_path = parent.pinned(f"{PROTECTED}/{trace_name}", files[trace_name], "runs")
        receipt = json.loads(receipt_path.read_text())
        if receipt["trace_path"] != trace_name or receipt["trace_sha256"] != files[trace_name]:
            raise ValueError("protected receipt/manifest trace mismatch")
        with np.load(trace_path, allow_pickle=False) as data:
            selected = validate_snapshots(data, receipt, seed)
            images.extend(selected)
            sources.append({"geometry_seed": seed, "receipt_sha256": files[receipt_name],
                            "trace_sha256": files[trace_name], "snapshots_selected": len(selected),
                            "selection": "initial-plus-all-saved" if seed in FINISHED else "initial-only"})
    pool = deduplicate(images)
    return pool, {"manifest_sha256": PROTECTED_SHA, "sources": sources,
                  "exposure": "consumed-positive-preservation-not-fresh", **pool_receipt(pool)}


def load_inputs():
    v5, _, cells = parent.historical_evidence()
    config = parent.config_from_v5(v5)
    parent.source_checkpoint()
    prior, digest, _ = load_offline_replay(parent.ROOT / parent.PRIOR_DIR, v5, seed=60)
    if digest != parent.PRIOR_DATASET_SHA or prior.valid_count != len(prior) or len(prior) != 15915:
        raise ValueError("ordinary prior differs from verified 15915-row source")
    prior_pool = deduplicate(prior._observation_stack(i) for i in range(len(prior)))
    del prior
    recovery, receipt = load_recovery_replay(parent.located(DATASET, "runs"),
        manifest_sha256=DATASET_SHA, allowed_cells=cells, seed=60, required_policy=FINISH_POLICY)
    if receipt["accepted_transitions"] != 665 or receipt["unique_failure_transitions"] != 339:
        raise ValueError("failure support differs from declared qualified data")
    manifest = json.loads(parent.pinned(f"{DATASET}/manifest.json", DATASET_SHA, "runs").read_text())
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    images, targets = [], []
    for index in recovery._valid_rows[:recovery.valid_count]:
        row = episodes[int(recovery.episode_ids[index])]
        step = int(recovery.steps[index])
        if row["stratum"] == "failure" and row["anchor_step"] <= step < row["anchor_step"] + row["horizon"]:
            images.append(recovery._observation_stack(int(index)))
            targets.append(recovery.executed_actions[index].copy())
    if len(images) != 87:
        raise ValueError("expected exactly 87 executed failure Oracle guide rows")
    guide_pool = deduplicate(images, targets)
    del recovery
    protected_pool, protected_receipt = load_protected()
    pools = {"guide": guide_pool, "prior": prior_pool, "protected": protected_pool}
    return config, pools, {"dataset_manifest_sha256": DATASET_SHA, "recovery": receipt,
        "prior_dataset_sha256": parent.PRIOR_DATASET_SHA,
        "protected": protected_receipt, "pools": {k: pool_receipt(v) for k, v in pools.items()}}


class ActorOnlyAgent(PixelRLPDAgent):
    def __init__(self, config, state, *, device="cpu"):
        super().__init__(config, seed=60, device=device)
        import_learning_state(self, state, seed=60)
        self.actor.encoder.requires_grad_(False)
        self.actor.log_std.requires_grad_(False)
        self.critic.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.log_alpha.requires_grad_(False)
        self.actor.eval()
        self.reference_actor = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.actor_only_updates = 0

    def update(self, batch):
        raise RuntimeError("SAC forbidden in actor-only repair")

    def frozen_hashes(self):
        optimizer = self.actor_optimizer
        frozen = list(self.actor.encoder.parameters()) + list(self.actor.log_std.parameters())
        return {"encoder": state_sha(self.actor.encoder.state_dict()),
                "critic": state_sha(self.critic.state_dict()),
                "target_critic": state_sha(self.target_critic.state_dict()),
                "temperature": state_sha(self.log_alpha),
                "log_std_head": state_sha(self.actor.log_std.state_dict()),
                "reference": state_sha(self.reference_actor.state_dict()),
                "frozen_actor_adam": state_sha([optimizer.state.get(p, {}) for p in frozen]),
                "critic_adam": state_sha(self.critic_optimizer.state_dict()),
                "temperature_adam": state_sha(self.temperature_optimizer.state_dict())}

    @torch.no_grad()
    def prepare_features(self, pools, batch_size=128):
        result = {}
        for name, pool in pools.items():
            features, source_targets = [], []
            for start in range(0, len(pool["images"]), batch_size):
                observation = torch.as_tensor(pool["images"][start:start + batch_size], device=self.device)
                latent = self.reference_actor.encoder(observation)
                source_mean = self.reference_actor.mean(self.reference_actor.trunk(latent)).tanh()
                features.append(latent.detach())
                source_targets.append(source_mean.detach())
            result[name] = {"features": torch.cat(features), "source_targets": torch.cat(source_targets),
                            "weights": pool["weights"].copy()}
            result[name]["targets"] = (torch.as_tensor(pool["targets"], device=self.device)
                if name == "guide" else result[name]["source_targets"])
        return result


@torch.no_grad()
def prediction_errors(agent, pools):
    result = {}
    for name, pool in pools.items():
        sums = torch.zeros(3, device=agent.device)
        drift = torch.zeros_like(sums)
        for start in range(0, len(pool["features"]), 1024):
            sl = slice(start, start + 1024)
            predicted = agent.actor.mean(agent.actor.trunk(pool["features"][sl])).tanh()
            weight = torch.as_tensor(pool["weights"][sl], device=agent.device).unsqueeze(1)
            sums += ((predicted - pool["targets"][sl]).square() * weight).sum(0)
            drift += ((predicted - pool["source_targets"][sl]).square() * weight).sum(0)
        components = (sums / pool["weights"].sum()).cpu().tolist()
        result[name] = {"mse": float(np.mean(components)), "axis_mse": components,
                        "source_mean_axis_mse": (drift / pool["weights"].sum()).cpu().tolist(),
                        "rows": int(pool["weights"].sum())}
    return result


def fit(agent, pools, *, updates=2048, journal=None):
    if type(updates) is not int or updates <= 0 or agent.actor_only_updates:
        raise ValueError("repair is a single non-resumable positive update budget")
    frozen = agent.frozen_hashes()
    initial = prediction_errors(agent, pools)
    rng = np.random.default_rng(60)
    parameters = list(agent.actor.trunk.parameters()) + list(agent.actor.mean.parameters())
    for step in range(1, updates + 1):
        errors = {}
        for name, count in BUDGET["batch"].items():
            pool = pools[name]
            indices = rng.choice(len(pool["weights"]), size=count, replace=True,
                                 p=pool["weights"] / pool["weights"].sum())
            indices = torch.as_tensor(indices, device=agent.device)
            mean = agent.actor.mean(agent.actor.trunk(pool["features"][indices])).tanh()
            errors[name] = (mean - pool["targets"][indices]).square().mean(0)
        guide = errors["guide"].mean()
        retention_axis = (32 * errors["prior"] + 16 * errors["protected"]) / 48
        retention = retention_axis.mean()
        loss = guide + retention
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite actor repair loss")
        agent.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        agent.actor_optimizer.step()
        agent.actor_only_updates = step
        if journal:
            journal({"actor_only_updates": step, "sac_gradient_steps": 0,
                     "environment_resets": 0, "loss": float(loss.detach()),
                     "guide_mse": float(guide.detach()), "retention_mse": float(retention.detach()),
                     "guide_axis_mse": errors["guide"].detach().cpu().tolist(),
                     "retention_axis_mse": retention_axis.detach().cpu().tolist(),
                     "prior_axis_mse": errors["prior"].detach().cpu().tolist(),
                     "protected_axis_mse": errors["protected"].detach().cpu().tolist(),
                     "gradient_norm_before_clip": float(norm), "batch": BUDGET["batch"]})
    after = agent.frozen_hashes()
    if frozen != after or agent.gradient_steps != 0 or agent.environment_steps != 0:
        raise RuntimeError("frozen weights/Adam state or zero-SAC/reset contract violated")
    return {"initial_prediction_errors": initial, "final_prediction_errors": prediction_errors(agent, pools),
            "frozen_hashes_before": frozen, "frozen_hashes_after": after,
            "actor_only_updates": agent.actor_only_updates, "sac_gradient_steps": 0,
            "environment_steps": 0, "environment_resets": 0}


def contract(config, inputs):
    return {"format": FORMAT, "status": "frozen", "partition": "consumed TRAIN",
            "algorithm": "constrained-offline-actor-repair-on-proven-closed-loop-data",
            "source_checkpoint": parent.CHECKPOINT, "source_checkpoint_sha256": parent.CHECKPOINT_SHA,
            "learner_config": asdict(config), "budget": copy.deepcopy(BUDGET), "inputs": inputs,
            "source_hashes": {name: sha256_file(parent.source_path(name)) for name in sorted(SOURCE_FILES)},
            "objective": {"guide": "joint-native-three-action-executed-failure-Oracle-MSE",
                          "retention": "original-V5-mean-on-same-image-MSE",
                          "retention_reduction": "mean-over-48-rows", "reward_used": False,
                          "reward_modified": False, "critic_loss": False,
                          "trainable": ["actor.trunk", "actor.mean"], "augmentation": False},
            "resume_supported": False, "official_performance_claim": False}


def preflight(protocol_path=None, expected_sha=None):
    torch.set_num_threads(1)
    config, pools, inputs = load_inputs()
    expected = contract(config, inputs)
    if (protocol_path is None) != (expected_sha is None):
        raise ValueError("protocol and SHA must be supplied together")
    if protocol_path is not None:
        assert expected_sha is not None
        frozen = json.loads(parent.pinned(protocol_path, expected_sha, "experiments").read_text())
        if ({k: v for k, v in frozen.items() if k != "created_at_utc"} != expected
                or not isinstance(frozen.get("created_at_utc"), str)):
            raise ValueError("actor-only frozen protocol/data/source changed")
    # Map the trusted source archive: historical replay/RNG payloads are unused.
    state = torch.load(str(parent.source_checkpoint()), map_location="cpu", weights_only=False, mmap=True)
    original_export = parent.pinned(str(Path(parent.CHECKPOINT).with_name("actor.pt")), ACTOR_SHA, "runs")
    exported = torch.load(original_export, map_location="cpu", weights_only=False)
    if state_sha(state["actor"]) != state_sha(exported["actor_state_dict"]):
        raise ValueError("protected original export differs from original checkpoint actor")
    agent = ActorOnlyAgent(config, state)
    return expected, pools, agent


def freeze(output):
    path = parent.located(output, "experiments")
    if path.exists():
        raise FileExistsError(path)
    protocol, _, _ = preflight()
    protocol["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    parent.write_receipt(path, protocol)
    return {"status": "frozen-zero-reset", "protocol": output, "sha256": sha256_file(path)}


def run(protocol_path, expected_sha, run_dir="runs/rlpd-recovery-actor-only-v1"):
    protocol, pools, cpu_agent = preflight(protocol_path, expected_sha)
    parent.current_runtime()
    parent.seed_everything(60)
    del cpu_agent
    output = parent.located(run_dir, "runs")
    output.mkdir(exist_ok=False)
    parent.write_receipt(output / "attempt.json", {"format": FORMAT, "status": "started",
        "protocol_sha256": expected_sha, "resume_supported": False, "budget": BUDGET})
    agent = None
    started = time.monotonic()
    try:
        for name, digest in protocol["source_hashes"].items():
            destination = output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent.source_path(name), destination)
            if sha256_file(destination) != digest:
                raise ValueError("source changed during snapshot")
        shutil.copyfile(parent.pinned(protocol_path, expected_sha, "experiments"), output / "study_protocol.json")
        if sha256_file(output / "study_protocol.json") != expected_sha:
            raise ValueError("protocol changed during snapshot")
        parent.write_receipt(output / "protected-manifest.json", protocol["inputs"]["protected"])
        state = torch.load(str(parent.source_checkpoint()), map_location="cpu", weights_only=False, mmap=True)
        agent = ActorOnlyAgent(parent.RLPDConfig(**protocol["learner_config"]), state, device="cuda")
        del state
        features = agent.prepare_features(pools)
        del pools
        parent.write_receipt(output / "feature-receipt.json", {
            name: {"features_sha256": state_sha(pool["features"]),
                   "targets_sha256": state_sha(pool["targets"]),
                   "source_targets_sha256": state_sha(pool["source_targets"])}
            for name, pool in features.items()})
        parent.write_receipt(output / "initial-predictions.json", {
            "prediction_errors": prediction_errors(agent, features),
            "frozen_hashes": agent.frozen_hashes(), "actor_only_updates": 0})
        with (output / "step-metrics.jsonl").open("x") as handle:
            result = fit(agent, features, journal=lambda row: parent.journal(handle, row))
        checkpoint = {"format": FORMAT + "-checkpoint", "resume_supported": False,
            "config": asdict(agent.config), "protocol_sha256": expected_sha,
            "source_checkpoint_sha256": parent.CHECKPOINT_SHA,
            **{name: getattr(agent, name).state_dict() for name in ("actor", "critic", "target_critic")},
            **{name: getattr(agent, name).state_dict() for name in
               ("actor_optimizer", "critic_optimizer", "temperature_optimizer")},
            "log_alpha": agent.log_alpha.detach(), "environment_steps": 0,
            "gradient_steps": 0, "actor_only_updates": agent.actor_only_updates,
            "rng_seed": 60, "training_result": result}
        torch.save(checkpoint, output / "checkpoint.pt")
        agent.export_actor(output / "actor.pt", source_sha256=canonical_sha256(protocol["source_hashes"]),
            protocol_sha256=expected_sha, training_seed=60,
            environment_contract={"partition": "consumed TRAIN", "frame_skip": 4,
                "max_steps": 2000, "obstacles": True, "reward_shaping": False,
                "actor_only_updates": agent.actor_only_updates, "sac_updates": 0,
                "algorithm": protocol["algorithm"], "resume_supported": False})
        exported = torch.load(output / "actor.pt", map_location="cpu", weights_only=False)
        if state_sha(exported["actor_state_dict"]) != state_sha(agent.actor.state_dict()):
            raise RuntimeError("export weights differ from repaired actor")
        result.update({"format": FORMAT, "status": "complete", "protocol_sha256": expected_sha,
            "actor_sha256": sha256_file(output / "actor.pt"),
            "checkpoint_sha256": sha256_file(output / "checkpoint.pt"),
            "protected_manifest_sha256": sha256_file(output / "protected-manifest.json"),
            "wall_seconds": time.monotonic() - started, "resume_supported": False,
            "official_performance_claim": False})
        parent.write_receipt(output / "finalstats.json", result)
        return result
    except BaseException as exc:
        parent.write_receipt(output / "failure.json", {"status": "partial-not-resumable",
            "error": str(exc), "error_type": type(exc).__name__, "environment_resets": 0,
            "sac_updates": 0, "actor_only_updates": agent.actor_only_updates if agent else 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--output", required=True)
    for operation in ("preflight", "run"):
        command = sub.add_parser(operation)
        command.add_argument("--protocol", required=operation == "run")
        command.add_argument("--protocol-sha256", required=operation == "run")
        if operation == "run":
            command.add_argument("--run-dir", default="runs/rlpd-recovery-actor-only-v1")
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(args.output)
    elif args.operation == "preflight":
        protocol, _, _ = preflight(args.protocol, args.protocol_sha256)
        result = {"status": "passed-zero-reset", "budget": protocol["budget"],
                  "inputs": protocol["inputs"], "source_hashes": protocol["source_hashes"]}
    else:
        result = run(args.protocol, args.protocol_sha256, args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
