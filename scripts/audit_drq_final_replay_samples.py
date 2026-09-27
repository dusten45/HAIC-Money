"""Read-only six-arm provenance audit before final-source development diagnostics.

This independently joins each sampled sequence to the sealed source collection
or to the online insertion ledger. A passing receipt is written only after all
six complete runs and both sealed source pools have passed.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from scripts import audit_drq_retention_r7 as old


ROOT = Path("runs/20260927-drqv2-final-source-replay-v1")
PROTOCOL = Path("experiments/drqv2-final-source-replay-v1.json")
RECEIPT = ROOT / "pre-evaluation-sample-audit.json"
R7_SHA = "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9"
R6_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
OLD_SAMPLE_AUDIT_SHA = "826e6fad1a58ea5880c25b7f3354ab6cf06122099d803f465f1d7b02c95c0db9"
VARIANTS = ("uniform", "failure_weighted", "easy_retention")
WARMUP = 10000


def check_pool(state: dict, receipt: dict, reset_rows: list[dict], excluded: set[int]) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Verify source n-step starts and their per-episode TRAIN origin."""
    old.require(state["next_sequence"] == state["size"] == state["capacity"] == 100000,
                "source pool must be exactly 100000 unoverwritten transitions")
    valid, _ = old.valid_starts(state, first=0, last=100000)
    old.require(int(valid.sum()) == receipt["valid_n_step_starts"] and int(valid.sum()) >= 32,
                "source n-step validity differs from sealed receipt")
    ids = state["episode_ids"][:100000]
    steps = state["episode_steps"][:100000]
    boundaries = ids[1:] != ids[:-1]
    old.require(steps[0] == 0 and np.all(steps[1:][boundaries] == 0)
                and np.all(steps[1:][~boundaries] == steps[:-1][~boundaries] + 1)
                and np.all((state["terminal"][:99999] |
                            state["terminated"][:99999] |
                            state["truncated"][:99999])[boundaries]),
                "source episode sequence, step or terminal boundary is inconsistent")
    road_by_episode: dict[int, int] = {}
    ended: set[int] = set()
    capped: set[int] = set()
    prefix_start = receipt["retained_window_schedule_episodes"]
    old.require(type(prefix_start) is int and 0 < prefix_start,
                "source schedule phase boundary missing")
    for row in reset_rows:
        old.require(row["event"] in ("reset", "end", "capped_partial") and type(row["episode_id"]) is int,
                    "invalid sealed source episode event")
        if row["event"] == "reset":
            episode, road = row["episode_id"], row["geometry_seed"]
            phase = "retained_window" if episode < prefix_start else "historical_prefix_once"
            old.require(episode not in road_by_episode and row.get("seed", road) == road
                        and row["schedule_index"] == episode and row["schedule_phase"] == phase
                        and row["track_id"] in (1, 2, 3, 4) and road not in excluded,
                        "source reset duplicates episode or uses excluded road")
            road_by_episode[episode] = road
        else:
            old.require(row["episode_id"] in road_by_episode and row["episode_id"] not in ended and
                        row["schedule_index"] == row["episode_id"] and
                        row["schedule_phase"] == ("retained_window" if row["episode_id"] < prefix_start
                                                  else "historical_prefix_once") and
                        row["geometry_seed"] == road_by_episode[row["episode_id"]],
                        "source end without corresponding reset")
            if row["event"] == "capped_partial":
                old.require(row["episode_id"] == int(ids[-1]) and
                            not (state["terminal"][-1] or state["terminated"][-1] or state["truncated"][-1]),
                            "capped partial is not final unterminated episode")
                capped.add(row["episode_id"])
            ended.add(row["episode_id"])
    old.require(len(road_by_episode) > 0 and set(map(int, np.unique(ids))).issubset(road_by_episode),
                "source sequence is not joined to its TRAIN reset ledger")
    old.require(not capped or (len(capped) == 1 and reset_rows[-1]["event"] == "capped_partial"),
                "source collection has nonfinal capped episodes")
    if "geometry_seeds" in receipt:
        old.require(set(road_by_episode.values()) == set(receipt["geometry_seeds"]),
                    "source episode roads differ from sealed source receipt")
    old.require(state["frames"].shape == (100000, 84, 84)
                and state["frames"].dtype == np.uint8
                and state["actions"].shape == (100000, 3)
                and np.isfinite(state["actions"]).all(), "invalid source pixels or actions")
    old.require(receipt["scheduled_episodes_consumed"] == len(road_by_episode) and
                receipt["historical_prefix_episodes_consumed"] ==
                sum(episode >= prefix_start for episode in road_by_episode),
                "source collection episode phase counts differ")
    return valid, ids, road_by_episode


def check_source_steps(path: Path, state: dict, receipt: dict, roads: dict[int, int]) -> int:
    """Join every persisted source transition to its replay slot and actor/noise log."""
    count = 0
    prefix_steps = 0
    prefix_start = receipt["retained_window_schedule_episodes"]
    rng = np.random.default_rng(receipt["collection_rng_seeds"]["action_noise_seed"])
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            old.require(count <= receipt["decisions"], "source step ledger exceeds fixed collection budget")
            row = json.loads(line)
            index = count - 1
            episode = int(state["episode_ids"][index])
            unnoised = np.asarray(row["unnoised_native_action"], dtype=np.float32)
            noise = np.asarray(row["noise"], dtype=np.float64)
            applied = np.asarray(row["native_action"], dtype=np.float32)
            expected = np.clip(unnoised.astype(np.float64) + noise,
                               -1.0, 1.0).astype(np.float32)
            old.require(row["decision"] == count and row["sequence_id"] == index
                        and row["source_seed"] == receipt["source_seed"]
                        and row["partition"] == "TRAIN" and row["episode_id"] == episode
                        and row["schedule_index"] == episode and
                        row["schedule_phase"] == ("retained_window" if episode < prefix_start
                                                  else "historical_prefix_once")
                        and row["episode_step"] == int(state["episode_steps"][index])
                        and row["geometry_seed"] == roads[episode]
                        and row["track_id"] in (1, 2, 3, 4)
                        and unnoised.shape == noise.shape == applied.shape == (3,)
                        and np.isfinite(unnoised).all() and np.isfinite(noise).all()
                        and np.array_equal(noise, rng.normal(0.0, 0.05, size=3))
                        and np.max(np.abs(expected - applied)) <= 1e-6
                        and np.max(np.abs(state["actions"][index] - applied)) <= 1e-6
                        and np.float32(row["reward"]) == state["rewards"][index]
                        and bool(row["terminal"]) == bool(state["terminal"][index])
                        and bool(row["terminated"]) == bool(state["terminated"][index])
                        and bool(row["truncated"]) == bool(state["truncated"][index]),
                        f"source replay/actor-noise ledger mismatch at decision {count}")
            prefix_steps += int(episode >= prefix_start)
    old.require(count == receipt["decisions"] == 100000, "source step ledger is incomplete")
    old.require(prefix_steps == receipt["historical_prefix_decisions"],
                "historical prefix decisions disagree with sealed receipt")
    return count


def check_online_warmup(new: dict, control: dict) -> None:
    """Replay provenance cannot affect the update-free online warmup prefix."""
    keys = ("frames", "actions", "rewards", "terminated", "truncated", "terminal",
            "episode_ids", "episode_steps", "sequence_ids")
    for key in keys:
        old.require(np.array_equal(new[key][:WARMUP], control[key][:WARMUP]),
                    f"online warmup differs from paired evolving-source control: {key}")
    new_boundaries = {int(k): v for k, v in new["boundary_observations"].items() if int(k) < WARMUP}
    old_boundaries = {int(k): v for k, v in control["boundary_observations"].items() if int(k) < WARMUP}
    old.require(new_boundaries.keys() == old_boundaries.keys() and
                all(np.array_equal(frame, old_boundaries[key]) for key, frame in new_boundaries.items()),
                "online warmup terminal stacks differ from evolving-source control")


def audit(root: Path, protocol_path: Path) -> dict:
    old.require(protocol_path.is_file() and not protocol_path.is_symlink()
                and protocol_path.resolve().is_relative_to(root.resolve()),
                "protocol must be an in-repository regular file")
    protocol = old.read_json(protocol_path)
    protocol_sha = old.digest(protocol_path)
    old.require(protocol.get("format") == "haic-drq-final-source-replay-study-v1"
                and protocol.get("study_id") == "drqv2-final-source-replay-v1"
                and protocol.get("run_root") == ROOT.as_posix()
                and protocol.get("r6_protocol_sha256") == R6_SHA
                and protocol.get("r7_protocol_sha256") == R7_SHA
                and protocol.get("lambda_preserve") == 0.5,
                "wrong frozen six-arm protocol")
    for key in ("r6_protocol_path", "r7_protocol_path"):
        old.verified(root, protocol[key], protocol[key.replace("path", "sha256")])
    for name, expected in protocol["code_sha256"].items():
        old.verified(root, name, expected)
    catalog = old.read_json(old.verified(root,
        "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json",
        protocol["catalog_sha256"]))
    training = {int(row["geometry_seed"]) for row in catalog["train"]}
    diagnostic = {int(row["geometry_seed"]) for row in catalog["train_diagnostic"]}
    old.require(len(training) == 120 and len(diagnostic) == 16 and not training & diagnostic,
                "r6 TRAIN/TRAIN-DIAGNOSTIC partition changed")
    contract = protocol["replay_contract"]
    old.require((contract["batch_size"], contract["source_rows"], contract["online_rows"])
                == (64, 32, 32) and protocol["source_replay"].keys() == {"0", "1"},
                "replay contract differs from original r7b")
    arms = protocol["runs"]
    expected_arms = {(seed, variant) for seed in (0, 1) for variant in VARIANTS}
    old.require(len(arms) == 6 and {(r["source_seed"], r["variant"]) for r in arms} == expected_arms
                and all(r["condition"] == "final_source" for r in arms), "six-arm matrix changed")
    old_protocol = old.read_json(old.verified(root, protocol["r7_protocol_path"], R7_SHA))
    old.require(protocol["catalog_sha256"] == old_protocol["catalog_sha256"] and
                protocol["diagnostic_cache"] == old_protocol["diagnostic_cache"],
                "catalog or forward-only diagnostic cache changed")
    old_audit = old.read_json(old.verified(root,
        "runs/20260926-drqv2-retention-r7/pre-evaluation-trace-audit-v1.json",
        OLD_SAMPLE_AUDIT_SHA))
    old_controls = {(r["source_seed"], r["variant"]): r for r in old_audit["arms"]
                    if r["condition"] == "r7b"}
    old.require(len(old_controls) == 6 and old_audit["passed"] is True,
                "original evolving-source r7b controls are not fully audited")
    source_pool_evidence, evidence = {}, []
    for seed in (0, 1):
        source = protocol["source_replay"][str(seed)]
        receipt_path = old.verified(root, source["receipt_path"], source["receipt_sha256"])
        receipt = old.read_json(receipt_path)
        pool_path = old.verified(root, source["pool_path"], source["pool_sha256"])
        old.require(receipt["completed"] is True and receipt["source_seed"] == seed
                    and receipt["pool_sha256"] == source["pool_sha256"]
                    and receipt["pool_path"] == source["pool_path"]
                    and receipt["partition"] == "TRAIN"
                    and receipt["excluded_diagnostic_roads"] is True
                    and receipt["decisions"] == receipt["capacity"] == 100000
                    and receipt["collection_noise_std"] == 0.05
                    and receipt["original_ledger_sha256"] == old_protocol["source_replay"][str(seed)]["episode_ledger_sha256"]
                    and receipt["source_checkpoint_sha256"] == old_protocol["source_replay"][str(seed)]["checkpoint_sha256"],
                    "pool identity/partition not bound by sealed receipt")
        ledger_path = old.verified(root, receipt["episode_ledger_path"], receipt["episode_ledger_sha256"])
        reset_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
        step_path = old.verified(root, receipt["step_ledger_path"], receipt["step_ledger_sha256"])
        payload = torch.load(str(pool_path), map_location="cpu", mmap=True, weights_only=False)
        old.require(isinstance(payload, dict) and "replay" in payload, "sealed source replay is missing")
        state = payload["replay"]
        valid, ids, roads = check_pool(state, receipt, reset_rows, diagnostic)
        checked_steps = check_source_steps(step_path, state, receipt, roads)
        source_pool_evidence[str(seed)] = {"pool_sha256": source["pool_sha256"],
             "receipt_sha256": source["receipt_sha256"], "episode_ledger_sha256": receipt["episode_ledger_sha256"],
             "valid_n_step_starts": int(valid.sum()), "source_episodes": len(roads),
             "verified_source_decisions": checked_steps,
             "historical_prefix_decisions": receipt["historical_prefix_decisions"],
             "historical_prefix_episodes_consumed": receipt["historical_prefix_episodes_consumed"]}
        for arm in (a for a in arms if a["source_seed"] == seed):
            variant = arm["variant"]
            old_control_receipt = old_controls[(seed, variant)]
            original_run = old.read_json(old.verified(root,
                f"{old_control_receipt['run_dir']}/result.json",
                old_control_receipt["result_sha256"]))
            original_arm = next(r for r in old_protocol["runs"]
                                if (r["source_seed"], r["variant"], r["condition"])
                                == (seed, variant, "r7b"))
            old.require(arm["rng_seeds"] == original_arm["rng_seeds"] and
                        original_run["source_actor_sha256"] == receipt["source_actor_sha256"],
                        "source actor or learner RNG differs from original matched r7b")
            run = ROOT / f"learner-{seed}-{variant}-final_source"
            old.require(arm["run_dir"] == run.as_posix(), "unfrozen arm output path")
            old.verified(root, f"{run}/study_protocol.json", protocol_sha)
            result_path = root / run / "result.json"
            result = old.read_json(result_path)
            old.require(result.get("format") == "haic-drq-retention-run-result-v1"
                        and result.get("completed") is True
                        and (result["study_id"], result["source_seed"], result["variant"], result["condition"])
                        == (protocol["study_id"], seed, variant, "final_source")
                        and result["study_protocol_sha256"] == protocol_sha
                        and result["source_replay_sha256"] == source["pool_sha256"]
                        and result["source_checkpoint_sha256"] == receipt["source_checkpoint_sha256"]
                        and result["source_actor_sha256"] == receipt["source_actor_sha256"]
                        and result["initial_checkpoint_sha256"] == original_run["initial_checkpoint_sha256"]
                        and (result["additional_online_steps"], result["study_gradient_steps"])
                        == (32768, 22768)
                        and (result["source_samples"], result["online_samples"]) == (728576, 728576),
                        "invalid treatment run receipt")
            config = old.read_json(root / run / "run-config.json")
            old.require(config["protocol_sha256"] == protocol_sha
                        and config["source_replay_sha256"] == source["pool_sha256"]
                        and config["source_rows_per_batch"] == config["online_rows_per_batch"] == 32
                        and config["lambda_preserve"] == 0.5
                        and config["diagnostic_cache_sha256"] == protocol["diagnostic_cache"]["sha256"]
                        and config["optimizer"] == "Adam" and config["actor_lr"] == config["critic_lr"] == 1e-4
                        and config["encoder_update"] is True
                        and config["rng_seeds"] == arm["rng_seeds"], "treatment run settings changed")
            candidates = result["candidates"]
            old.require(len(candidates) == 2
                        and [c["checkpoint_online_step"] for c in candidates] == [16384, 32768]
                        and candidates[-1]["study_gradient_steps"] == 22768,
                        "wrong checkpoint opportunities")
            final = candidates[-1]
            old.verified(root, f"{run}/initial-weights.pt", result["initial_checkpoint_sha256"])
            old.verified(root, f"{run}/drift.jsonl", result["diagnostic_trace_sha256"])
            cp_path = old.verified(root, final["checkpoint_path"], final["checkpoint_sha256"])
            trace_path = old.verified(root, final["sample_trace_path"], final["sample_trace_sha256"])
            online_resets = old.episode_resets(root / run / "episodes.jsonl", diagnostic=diagnostic, train=training)
            checkpoint = torch.load(str(cp_path), map_location="cpu", mmap=True, weights_only=False)
            old.require(checkpoint["environment_steps"] == 131072 + 32768
                        and checkpoint["gradient_steps"] == 22768
                        and checkpoint["trainer_state"]["source_replay_sha256"] == source["pool_sha256"]
                        and checkpoint["trainer_state"]["sample_trace_sha256"] == final["sample_trace_sha256"],
                        "checkpoint trainer/replay/trace lineage differs")
            online = checkpoint["replay"]
            old_final = original_run["candidates"][-1]
            old_checkpoint_path = old.verified(root, old_final["checkpoint_path"],
                                                old_final["checkpoint_sha256"])
            old_checkpoint = torch.load(str(old_checkpoint_path), map_location="cpu",
                                        mmap=True, weights_only=False)
            check_online_warmup(online, old_checkpoint["replay"])
            del old_checkpoint
            online_valid, latest = old.valid_starts(online, first=0, last=32768)
            online_ids = online["episode_ids"][:32768]
            old.check_online_ledger(old.verified(root, f"{run}/step-metrics.jsonl",
                result["online_replay_manifest_sha256"]), online, online_resets, training)
            with np.load(trace_path, allow_pickle=False) as trace:
                old.require(set(trace.files) == old.TRACE_KEYS and
                            bytes(trace["protocol_sha256"]).decode("ascii") == protocol_sha,
                            "invalid final sample-trace schema")
                counts = old.check_trace(trace["source"], trace["source_indices"], trace["episode_id"],
                    source_first=0, source_valid=valid, source_ids=ids, online_valid=online_valid,
                    online_ids=online_ids, online_latest=latest, source_roads=roads,
                    online_roads=online_resets, diagnostic=diagnostic)
            evidence.append({"source_seed": seed, "variant": variant, "run_dir": run.as_posix(),
                             "result_sha256": old.digest(result_path),
                             "final_checkpoint_sha256": final["checkpoint_sha256"],
                             "final_sample_trace_sha256": final["sample_trace_sha256"], **counts})
            del checkpoint, online, online_valid, latest, online_ids
            gc.collect()
        del payload, state, valid, ids
        gc.collect()
        old.require(old.digest(pool_path) == source["pool_sha256"] and
                    old.digest(ledger_path) == receipt["episode_ledger_sha256"],
                    "sealed source bytes changed while auditing")
    old.require(len(evidence) == 6 and {(r["source_seed"], r["variant"]) for r in evidence} == expected_arms,
                "incomplete sample audit")
    return {"format": "haic-drq-final-source-sample-audit-v1", "passed": True,
            "protocol_sha256": protocol_sha, "no_environment_resets": True,
            "source_pool": source_pool_evidence, "runs": evidence,
            "scope": "TRAIN-only episode/3-step/sample provenance, not pixel coverage or performance"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    old.require(not (root / RECEIPT).exists(), "sample audit output exists; refusing overwrite")
    protocol_path = root / args.protocol
    result = audit(root, protocol_path)
    (root / RECEIPT).parent.mkdir(parents=True, exist_ok=True)
    with (root / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "arms": len(result["runs"]),
                      "receipt": str(RECEIPT), "sha256": old.digest(root / RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    main()
