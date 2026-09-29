"""Separate source-bound joint-guidance child of the frozen recovery learner.

Freeze and preflight do not construct an environment. Run is explicit and uses
the original V5 model and all three optimizer states, never the failed recovery.
Failure Oracle rows receive joint native guidance only. Ordinary prior, all
recovery actor-handoff rows, and finish-control Oracle rows retain the frozen
original V5 mean on the same images; the two auxiliary masks are disjoint.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import shutil

import torch

from haic.algorithms.rlpd.recovery_guidance import (
    GuidanceConfig, RecoveryGuidedRLPDAgent, load_role_lookup,
)
from scripts import train_rlpd_recovery as parent
from scripts.rlpd_common import canonical_sha256, sha256_file


FORMAT = "haic-rlpd-recovery-guided-learning-v1"
PARENT_PROTOCOL = "experiments/rlpd-recovery-learning-v1.json"
PARENT_SHA = "19efa2c357da708778b6d6cec3366aab94c56ad050a3751ffdd4975201b52264"
DATASET = "runs/rlpd-recovery-training-data-v1"
DATASET_SHA = "40ee39ac4ba6cab2b46ada76c028806d1d476903e5a912bb0108e5ba317c350d"
CHILD_FILES = frozenset({"haic/algorithms/rlpd/recovery_guidance.py",
                         "scripts/train_rlpd_recovery_guided.py",
                         "tests/test_rlpd_recovery_guidance.py"})
SOURCE_FILES = parent.SOURCE_FILES | CHILD_FILES
ARMS = {"control": {"online": 32, "prior": 32},
        "treatment": {"online": 32, "prior": 16, "recovery": 16}}


def child_contract(protocol, role_receipt, hashes):
    if (protocol["seed"] != 60 or protocol["total_steps"] != 8192
            or protocol["first_update_step"] != 2 or protocol["arms"] != ARMS
            or protocol["recovery_dataset"] != DATASET
            or protocol["recovery_receipt"]["manifest_sha256"] != DATASET_SHA):
        raise ValueError("parent differs from the predeclared seed60/8192/data contract")
    if (role_receipt["accepted_rows"] != 665
            or role_receipt["counts"]["failure/oracle"] != 87
            or sum(role_receipt["counts"][key] for key in ("failure/oracle", "failure/actor")) != 339
            or sum(role_receipt["counts"][key] for key in ("finish-control/oracle", "finish-control/actor")) != 326):
        raise ValueError("guidance role support differs from the predeclared dataset")
    return {"format": FORMAT, "status": "frozen", "partition": "TRAIN",
            "parent_protocol": PARENT_PROTOCOL, "parent_protocol_sha256": PARENT_SHA,
            "source_checkpoint": parent.CHECKPOINT, "source_checkpoint_sha256": parent.CHECKPOINT_SHA,
            "dataset": DATASET, "dataset_manifest_sha256": DATASET_SHA,
            "guidance_config": asdict(GuidanceConfig()), "role_receipt": role_receipt,
            "objective": {"guide": "joint-native-tanh-mean-MSE",
                          "guide_rows": "recovery/failure/oracle",
                          "retention_rows": ["ordinary-prior", "recovery/actor-handoff",
                                             "recovery/finish-control/oracle"],
                          "retention_target": "original-V5-source-actor-mean-on-same-images",
                          "guidance_retention_masks_disjoint": True,
                          "reference": "original-V5-source-actor-mean",
                          "target": "executed_action", "q_filter": False,
                          "encoder_detached": True, "actor_optimizer_only": True,
                          "raw_sac_unchanged": True, "reward_shaping": False,
                          "action_relabeling": False, "environment_modified": False},
            "seed": 60, "total_steps": 8192, "sac_updates": 8191,
            "extra_actor_updates": 8191, "matched_compute_to_v1": False,
            "arms": copy.deepcopy(ARMS), "run_arm": "treatment",
            "source_hashes": hashes, "runtime": copy.deepcopy(protocol["runtime"]),
            "evaluation": {"cells": copy.deepcopy(protocol["cells"]),
                           "comparators": ["original-v5", "control-v1", "recovery-v1"],
                           "fresh_cells": False, "launch_automatically": False},
            "official_performance_claim": False}


def source_hashes():
    return {name: sha256_file(parent.source_path(name)) for name in sorted(SOURCE_FILES)}


def freeze(output):
    target = parent.located(output, "experiments")
    if target.exists():
        raise FileExistsError(target)
    protocol, _, prior, recovery = parent.preflight(PARENT_PROTOCOL, PARENT_SHA)
    _, receipt = load_role_lookup(parent.located(DATASET, "runs"), DATASET_SHA)
    child = child_contract(protocol, receipt, source_hashes())
    child["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    del prior, recovery
    parent.write_receipt(target, child)
    return {"protocol": output, "sha256": sha256_file(target), "status": "frozen-zero-reset"}


def preflight(protocol_path, expected_sha):
    child = json.loads(parent.pinned(protocol_path, expected_sha, "experiments").read_text())
    protocol, config, prior, recovery = parent.preflight(PARENT_PROTOCOL, PARENT_SHA)
    lookup, receipt = load_role_lookup(parent.located(DATASET, "runs"), DATASET_SHA)
    expected = child_contract(protocol, receipt, source_hashes())
    if ({key: value for key, value in child.items() if key != "created_at_utc"} != expected
            or not isinstance(child.get("created_at_utc"), str)):
        raise ValueError("guided child contract/source changed after freeze")
    # Never mutate the parent protocol returned by its frozen validator.
    runtime_protocol = copy.deepcopy(protocol)
    runtime_protocol["source_hashes"] = copy.deepcopy(child["source_hashes"])
    runtime_protocol["behavior_cloning"] = True
    runtime_protocol["guidance"] = {"child_protocol_sha256": expected_sha,
                                    "child_source_sha256": canonical_sha256(child["source_hashes"]),
                                    "role_receipt": copy.deepcopy(receipt),
                                    "config": copy.deepcopy(child["guidance_config"]),
                                    "objective": copy.deepcopy(child["objective"])}
    return child, runtime_protocol, config, prior, recovery, lookup


def run(protocol_path, expected_sha, run_dir):
    child, protocol, config, prior, recovery, lookup = preflight(protocol_path, expected_sha)
    output = parent.located(run_dir, "runs")
    output.mkdir(exist_ok=False)
    parent.write_receipt(output / "attempt.json", {"format": FORMAT, "status": "started",
                                                   "protocol_sha256": expected_sha,
                                                   "guidance": protocol["guidance"]})
    agents = []
    try:
        for name, digest in child["source_hashes"].items():
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent.source_path(name), target)
            if sha256_file(target) != digest:
                raise ValueError("guided source snapshot changed")
        shutil.copyfile(parent.pinned(protocol_path, expected_sha, "experiments"),
                        output / "study_protocol.json")
        if sha256_file(output / "study_protocol.json") != expected_sha:
            raise ValueError("guided protocol snapshot changed")
        shutil.copyfile(parent.pinned(PARENT_PROTOCOL, PARENT_SHA, "experiments"),
                        output / "parent_protocol.json")
        state = torch.load(parent.source_checkpoint(), map_location="cpu", weights_only=False)

        def factory(config, *, seed, device):
            agent = RecoveryGuidedRLPDAgent(
                config, seed=seed, device=device, reference_state=state["actor"],
                role_lookup=lookup, guidance_receipt={**protocol["guidance"],
                    "source_checkpoint_sha256": parent.CHECKPOINT_SHA},
                guidance_config=GuidanceConfig(**child["guidance_config"]))
            agents.append(agent)
            return agent

        result = parent.train_loop(output, protocol, expected_sha, config, prior, recovery,
                                   arm="treatment", agent_factory=factory, learning_state=state)
        agent = agents[0]
        if agent.extra_actor_steps != child["extra_actor_updates"]:
            raise RuntimeError("guided optimizer budget mismatch")
        finalstats = {**result, "format": FORMAT, "guidance": agent.guidance_specification(),
                      "parent_protocol_sha256": PARENT_SHA,
                      "sac_gradient_steps": agent.gradient_steps,
                      "extra_actor_optimizer_steps": agent.extra_actor_steps,
                      "source_hashes": child["source_hashes"],
                      "matched_compute_to_v1": False}
        parent.write_receipt(output / "finalstats.json", finalstats)
        return finalstats
    except BaseException as exc:
        parent.write_receipt(output / "guided-failure.json", {
            "status": "partial-not-exactly-resumable", "protocol_sha256": expected_sha,
            "error_type": type(exc).__name__, "error": str(exc),
            "sac_gradient_steps": agents[0].gradient_steps if agents else 0,
            "extra_actor_optimizer_steps": agents[0].extra_actor_steps if agents else 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    command = sub.add_parser("freeze")
    command.add_argument("--output", required=True)
    for operation in ("preflight", "run"):
        command = sub.add_parser(operation)
        command.add_argument("--protocol", required=True)
        command.add_argument("--protocol-sha256", required=True)
        if operation == "run":
            command.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.operation == "freeze":
        result = freeze(args.output)
    elif args.operation == "preflight":
        child, _, _, prior, recovery, _ = preflight(args.protocol, args.protocol_sha256)
        result = {"status": "passed-zero-reset", "prior_rows": prior.valid_count,
                  "recovery_rows": recovery.valid_count, "guidance": child["guidance_config"],
                  "objective": child["objective"],
                  "sac_updates": child["sac_updates"], "extra_actor_updates": child["extra_actor_updates"]}
    else:
        result = run(args.protocol, args.protocol_sha256, args.run_dir)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
