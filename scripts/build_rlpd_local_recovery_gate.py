"""Source-bound, zero-environment local-support gate calibration and export."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch

from haic.algorithms.rlpd.local_recovery_gate import (
    FORMAT, POLICY, LocalRecoveryGate, calibrate_support, tensor_state_sha)
from haic.algorithms.rlpd.model import PixelActor
from scripts import train_rlpd_recovery_actor_only as actor_only
from scripts.rlpd_common import sha256_file


parent = actor_only.parent
PROTOCOL = "experiments/rlpd-local-recovery-gate-v1.json"
CORRECTED = "runs/rlpd-recovery-actor-only-v1/actor.pt"
CORRECTED_SHA = "9198802b7cd573bf359fa1217d1edcaec3496a68e7e99d712fd1ef4135e2553f"
PRIMARY_RESULT = "experiments/rlpd-recovery-actor-only-evaluation-v1-result.json"
PRIMARY_SHA = "d6a30d214579764196e63a356884dc4b19419e7c171d314cefdb5734ac623e4f"
REPAIR_RECEIPT = "runs/rlpd-recovery-actor-only-v1/finalstats.json"
REPAIR_SHA = "d1ef7810834b961877dcc925e2ce9d7625cc0ee143135855551a70665c6eace7"
NEW_FILES = frozenset({"haic/algorithms/rlpd/local_recovery_gate.py",
                      "scripts/build_rlpd_local_recovery_gate.py",
                      "tests/test_rlpd_local_recovery_gate.py"})
SOURCE_FILES = actor_only.SOURCE_FILES | NEW_FILES


@torch.inference_mode()
def encode_images(actor, images, *, chunk_size=512):
    # CPU is canonical for runtime and calibration; one thread, no image augmentation.
    features = []
    for start in range(0, len(images), chunk_size):
        observations = images[start:start + chunk_size].astype(np.float32) / 255.0
        features.append(actor.encoder(torch.from_numpy(observations)).cpu())
    return torch.cat(features)


def preflight(protocol_path=None, protocol_sha=None, *, calibrate=False):
    if (protocol_path is None) != (protocol_sha is None):
        raise ValueError("protocol and SHA must be supplied together")
    torch.set_num_threads(1)
    started = time.monotonic()
    original_protocol, pools, original_agent = actor_only.preflight()
    source = original_agent.reference_actor
    source_path = str(Path(parent.CHECKPOINT).with_name("actor.pt"))
    source_export = torch.load(parent.pinned(source_path, actor_only.ACTOR_SHA, "runs"),
                               map_location="cpu", weights_only=True)
    corrected_export = torch.load(parent.pinned(CORRECTED, CORRECTED_SHA, "runs"),
                                  map_location="cpu", weights_only=True)
    if (source_export["format"] != "haic-rlpd-pixel-actor-v1"
            or corrected_export["format"] != source_export["format"]
            or corrected_export["config"] != source_export["config"]):
        raise ValueError("actors must share the native PixelActor export schema")
    corrected = PixelActor(source.encoder.latent_dim).eval().requires_grad_(False)
    corrected.load_state_dict(corrected_export["actor_state_dict"], strict=True)
    source_hash = tensor_state_sha(source.state_dict())
    if source_hash != tensor_state_sha(source_export["actor_state_dict"]):
        raise ValueError("source checkpoint/export mismatch")
    encoder_hash = tensor_state_sha(source.encoder.state_dict())
    if encoder_hash != tensor_state_sha(corrected.encoder.state_dict()):
        raise ValueError("corrected encoder differs bitwise from frozen source")
    repair = json.loads(parent.pinned(REPAIR_RECEIPT, REPAIR_SHA, "runs").read_text())
    parent.pinned(PRIMARY_RESULT, PRIMARY_SHA, "experiments")
    if (repair["actor_sha256"] != CORRECTED_SHA or repair["actor_only_updates"] != 2048
            or repair["environment_resets"] != 0 or repair["sac_gradient_steps"] != 0
            or repair["frozen_hashes_before"] != repair["frozen_hashes_after"]):
        raise ValueError("corrected actor repair provenance changed")
    protocol = {"format": FORMAT, "status": "frozen", "policy": POLICY,
        "partition": "consumed TRAIN", "environment_resets": 0, "critic_updates": 0,
        "source_actor": {"path": source_path, "sha256": actor_only.ACTOR_SHA,
                         "state_sha256": source_hash},
        "corrected_actor": {"path": CORRECTED, "sha256": CORRECTED_SHA,
                            "state_sha256": tensor_state_sha(corrected.state_dict())},
        "encoder_sha256": encoder_hash, "inputs": original_protocol["inputs"],
        "primary_result": {"path": PRIMARY_RESULT, "sha256": PRIMARY_SHA},
        "repair_receipt": {"path": REPAIR_RECEIPT, "sha256": REPAIR_SHA},
        "source_hashes": {name: sha256_file(parent.source_path(name)) for name in sorted(SOURCE_FILES)},
        "calibration": {"chunk_size": 512, "device": "cpu", "torch_threads": 1,
                        "radius_floor": None, "radius_cap": None,
                        "feature_normalization": "none", "protected": "all-prior-plus-protected-snapshots"},
        "limitations": ["Local-support preservation hypothesis, not guaranteed generalization.",
                        "Consumed training prototype calibration is a dataset memorization upper bound.",
                        "Protected point exclusion is not protection of entire closed-loop trajectories."],
        "official_performance_claim": False}
    if protocol_path is not None:
        assert protocol_sha is not None
        frozen = json.loads(parent.pinned(protocol_path, protocol_sha, "experiments").read_text())
        if {key: val for key, val in frozen.items() if key != "created_at_utc"} != protocol:
            raise ValueError("local gate protocol/source/data changed")
    # Drop unused critics, optimizer and replay state before encoding the support pool.
    del original_agent, source_export, corrected_export
    gate, calibration = None, None
    if calibrate:
        prototypes = encode_images(source, pools["guide"]["images"])
        prior_features = encode_images(source, pools["prior"]["images"])
        protected_features = encode_images(source, pools["protected"]["images"])
        protected = torch.cat((prior_features, protected_features))
        radii, calibration = calibrate_support(prototypes, protected)
        gate = LocalRecoveryGate(source, corrected, prototypes, radii)
        calibration.update({"feature_sha256": tensor_state_sha({"prototypes": prototypes,
                            "protected": protected}),
                            "prior_rows": int(pools["prior"]["weights"].sum()),
                            "snapshot_rows": int(pools["protected"]["weights"].sum()),
                            "guide_rows": int(pools["guide"]["weights"].sum()),
                            "training_prototype_coverage_rows": int(pools["guide"]["weights"][
                                (radii > 0).numpy()].sum())})
        # Verify exact native/official source path on every protected image, with
        # reset per point: the guarantee excludes active holds entered elsewhere.
        from common_adapter import ActionAdapter
        adapter = ActionAdapter()
        checked = 0
        for pool in (pools["prior"], pools["protected"]):
            for image in pool["images"]:
                obs = image.astype(np.float32) / 255.0
                gate.reset()
                actual = gate.act(obs)
                native = source.sample(torch.from_numpy(obs).unsqueeze(0), deterministic=True)[0][0]
                if gate.active or not np.array_equal(actual, adapter.to_official(native.numpy())):
                    raise RuntimeError("protected image triggered or source action drifted")
                checked += 1
        gate.reset()
        if source_hash != tensor_state_sha(source.state_dict()):
            raise RuntimeError("source actor mutated during calibration")
        calibration["protected_images_exact_source_action_checked"] = checked
        calibration["protected_pool_no_trigger"] = True
        covered = 0
        prototype_query_distances = []
        for image in pools["guide"]["images"]:
            gate.reset()
            gate.act(image.astype(np.float32) / 255.0)
            covered += int(gate.active)
            prototype_query_distances.append(gate.distance)
        calibration["runtime_training_prototype_coverage"] = covered
        calibration["runtime_training_prototype_distance_quantiles"] = np.quantile(
            prototype_query_distances, calibration["quantiles"]).tolist()
        gate.reset()
    resources = {"device": "cpu", "torch_threads": 1, "feature_dim": source.encoder.latent_dim,
                 "wall_seconds": time.monotonic() - started, "environment_resets": 0,
                 "critic_updates": 0, "calibrated": calibrate,
                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "torch_version": str(torch.__version__), "query_chunk_size": 512}
    return protocol, gate, calibration, resources


def freeze(output=PROTOCOL):
    path = parent.located(output, "experiments")
    if path.exists():
        raise FileExistsError(path)
    protocol, _, _, resources = preflight()
    protocol["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    parent.write_receipt(path, protocol)
    return {"status": "frozen-zero-reset", "protocol": output,
            "sha256": sha256_file(path), "resources": resources}


def build(protocol_path, protocol_sha, run_dir="runs/rlpd-local-recovery-gate-v1"):
    output = parent.located(run_dir, "runs")
    if output.exists():
        raise FileExistsError(output)
    protocol, gate, calibration, resources = preflight(protocol_path, protocol_sha, calibrate=True)
    assert gate is not None
    output.mkdir(exist_ok=False)
    for name, digest in protocol["source_hashes"].items():
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent.source_path(name), destination)
        if sha256_file(destination) != digest:
            raise ValueError("source changed while snapshotting")
    shutil.copyfile(parent.pinned(protocol_path, protocol_sha, "experiments"), output / "protocol.json")
    provenance = {"protocol_sha256": protocol_sha, "source_actor": protocol["source_actor"],
        "corrected_actor": protocol["corrected_actor"], "inputs": protocol["inputs"],
        "source_hashes": protocol["source_hashes"], "primary_result": protocol["primary_result"],
        "calibration": calibration, "limitations": protocol["limitations"]}
    artifact = output / "gate.pt"
    torch.save(gate.artifact(provenance=provenance), artifact)
    loaded = LocalRecoveryGate.load(artifact, expected_sha256=sha256_file(artifact))
    if tensor_state_sha(loaded.source_actor.state_dict()) != protocol["source_actor"]["state_sha256"]:
        raise RuntimeError("exported source actor changed")
    receipt = {"format": FORMAT, "status": "complete-zero-reset", "protocol_sha256": protocol_sha,
        "artifact_path": str(artifact.relative_to(parent.ROOT)), "artifact_sha256": sha256_file(artifact),
        "calibration": calibration, "resources": resources, "environment_resets": 0,
        "critic_updates": 0, "official_performance_claim": False}
    parent.write_receipt(output / "result.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--output", default=PROTOCOL)
    for operation in ("preflight", "build"):
        command = sub.add_parser(operation)
        command.add_argument("--protocol", default=PROTOCOL if operation == "build" else None)
        command.add_argument("--protocol-sha", "--protocol-sha256", dest="protocol_sha",
                             required=operation == "build")
        if operation == "build":
            command.add_argument("--run-dir", default="runs/rlpd-local-recovery-gate-v1")
        else:
            command.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(args.output)
    elif args.operation == "build":
        result = build(args.protocol, args.protocol_sha, args.run_dir)
    else:
        protocol, _, calibration, resources = preflight(args.protocol, args.protocol_sha,
                                                        calibrate=args.calibrate)
        result = {"status": "passed-zero-reset", "protocol": protocol,
                  "calibration": calibration, "resources": resources}
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
