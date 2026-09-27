"""Read-only semantic pre-evaluation audit of the twelve frozen DrQ r7 traces.

The only possible write is a new passing receipt after every arm has passed.
Run from the repository root with ``python -m scripts.audit_drq_retention_r7``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PROTOCOL = Path("experiments/drqv2-retention-r7.json")
PROTOCOL_SHA = "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9"
ROOT = Path("runs/20260926-drqv2-retention-r7")
RECEIPT = ROOT / "pre-evaluation-trace-audit-v1.json"
UPDATES = 22768
SLOTS = UPDATES * 32
WARMUP = 10000
ONLINE_STEPS = 32768
TRACE_KEYS = {"source", "source_indices", "episode_id", "protocol_sha256"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified(root: Path, name: str, expected: str) -> Path:
    require(isinstance(name, str) and isinstance(expected, str) and len(expected) == 64,
            "missing pinned artifact path or SHA")
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe path: {name}")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and digest(path) == expected,
            f"missing or SHA-mismatched artifact: {name}")
    return path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def episode_resets(path: Path, *, diagnostic: set[int], train: set[int] | None = None) -> dict[int, int]:
    resets: dict[int, int] = {}
    ends: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        episode = row["episode_id"]
        require(type(episode) is int, f"invalid episode ID: {path}")
        if row["event"] == "reset":
            road = row["seed"]
            require(type(road) is int and row.get("geometry_seed", road) == road
                    and road not in diagnostic and (train is None or road in train)
                    and episode not in resets, f"duplicate or forbidden reset road: {path}")
            resets[episode] = road
        elif row["event"] == "end":
            require(episode in resets and episode not in ends and row["seed"] == resets[episode],
                    f"end without matching reset: {path}")
            ends.add(episode)
        else:
            raise ValueError(f"unknown episode event: {path}")
    require(bool(resets), f"empty episode ledger: {path}")
    return resets


def valid_starts(state: dict, *, first: int, last: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate Uint8Replay._record/_stack/_build_n_step without materializing pixels.

    The returned latest sequence is the last transition needed for this n-step
    start (including its next-observation stack), for insertion-time validation.
    """
    cap = state["capacity"]
    require(type(cap) is int and cap == 100000 and state["action_dim"] == 3
            and state["n_step"] == 3 and state["gamma"] == .99
            and state["next_sequence"] == last and state["size"] == last - first,
            "replay config or rolling window mismatch")
    names = ("sequence_ids", "episode_ids", "episode_steps", "terminal", "terminated", "truncated")
    for name in names:
        require(isinstance(state[name], np.ndarray) and state[name].shape == (cap,),
                f"invalid replay field: {name}")
    seq_ids, episodes, steps = (state[name] for name in names[:3])
    done = state["terminal"] | state["terminated"] | state["truncated"]
    seq = np.arange(first, last, dtype=np.int64)
    require(np.array_equal(seq_ids[seq % cap], seq), "incomplete replay sequence ring")
    ids = episodes[seq % cap]
    first_steps = steps[seq % cap]
    good = (ids >= 0) & (first_steps >= 0)

    def record(target: np.ndarray, expected_step: np.ndarray) -> np.ndarray:
        slot = np.clip(target, first, last - 1) % cap
        return ((target >= first) & (target < last) & (seq_ids[slot] == target)
                & (episodes[slot] == ids) & (steps[slot] == expected_step))

    def stack(target: np.ndarray, current_step: np.ndarray) -> np.ndarray:
        result = np.ones(len(seq), dtype=bool)
        for channel in range(-3, 1):
            offset = np.maximum(channel, -current_step)
            result &= record(target + offset, current_step + offset)
        return result

    good &= stack(seq, first_steps)
    horizon = np.zeros(len(seq), dtype=np.int64)
    active = np.ones(len(seq), dtype=bool)
    for offset in range(3):
        target = seq + offset
        good &= ~active | record(target, first_steps + offset)
        horizon += active
        current_done = done[np.clip(target, first, last - 1) % cap]
        active &= ~current_done
    endpoint = seq + horizon - 1
    terminal_at_endpoint = done[np.clip(endpoint, first, last - 1) % cap]
    boundary = state["boundary_observations"]
    require(isinstance(boundary, dict), "missing replay boundary table")
    # Terminal next-observation stacks live in the side table; nonterminals need
    # the transition at start+horizon, which might not yet exist at sampling time.
    has_boundary = np.isin(endpoint, np.fromiter(boundary, dtype=np.int64, count=len(boundary)))
    good &= np.where(terminal_at_endpoint, has_boundary,
                     record(seq + horizon, first_steps + horizon)
                     & stack(seq + horizon, first_steps + horizon))
    latest = np.where(terminal_at_endpoint, endpoint, seq + horizon)
    return good, latest


def check_trace(source: np.ndarray, indices: np.ndarray, episode_id: np.ndarray,
                *, source_first: int, source_valid: np.ndarray, source_ids: np.ndarray,
                online_valid: np.ndarray, online_ids: np.ndarray, online_latest: np.ndarray,
                source_roads: dict[int, int], online_roads: dict[int, int],
                diagnostic: set[int]) -> dict:
    require(source.shape == indices.shape == episode_id.shape == (UPDATES, 64),
            "final trace is not (22768,64) in every sampled array")
    require(source.dtype == np.uint8 and indices.dtype == np.int64 and episode_id.dtype == np.int64,
            "trace array dtypes differ from trainer")
    require(np.all((source == 0) | (source == 1)), "trace has invalid source tags")
    mask = source == 1
    require(np.all(mask.sum(axis=1) == 32), "update does not sample exactly 32 source/32 online")
    src = np.where(mask, indices, -1)
    on = np.where(~mask, indices, -1)
    require(np.all(np.diff(np.sort(src, axis=1)[:, 32:], axis=1) != 0),
            "duplicate within-source update")
    require(np.all(np.diff(np.sort(on, axis=1)[:, 32:], axis=1) != 0),
            "duplicate within-online update")
    require(np.all((indices[mask] >= source_first) & (indices[mask] < source_first + len(source_valid)))
            and np.all((indices[~mask] >= 0) & (indices[~mask] < len(online_valid))),
            "sample index outside replay window")
    source_seq = indices[mask] - source_first
    online_seq = indices[~mask]
    require(np.all(source_valid[source_seq]) and np.all(online_valid[online_seq]),
            "sample is not a valid original source/final online n-step start")
    require(np.array_equal(episode_id[mask], source_ids[source_seq])
            and np.array_equal(episode_id[~mask], online_ids[online_seq]),
            "sample episode_id does not match replay")
    require(set(map(int, np.unique(episode_id[mask]))).issubset(source_roads)
            and set(map(int, np.unique(episode_id[~mask]))).issubset(online_roads),
            "sampled episode missing from source or online reset ledger")
    require(not ({source_roads[int(ep)] for ep in np.unique(episode_id[mask])}
                 | {online_roads[int(ep)] for ep in np.unique(episode_id[~mask])}) & diagnostic,
            "TRAIN-DIAGNOSTIC road sampled")
    # Update 1 follows insertion of online step 10001, whose replay index is 10000.
    insertion_step = WARMUP + np.broadcast_to(np.arange(1, UPDATES + 1)[:, None], mask.shape)[~mask]
    require(np.all(online_seq + 1 <= insertion_step)
            and np.all(online_latest[online_seq] + 1 <= insertion_step),
            "online sample or required n-step successor inserted after update")
    return {"updates": UPDATES, "batch_size": 64, "source_rows": int(mask.sum()),
            "online_rows": int((~mask).sum()), "unique_source_starts": int(np.unique(source_seq).size),
            "unique_online_starts": int(np.unique(online_seq).size)}


def check_online_ledger(path: Path, state: dict, roads: dict[int, int], train: set[int]) -> None:
    count = 0
    for count, line in enumerate(path.open("r", encoding="utf-8"), 1):
        require(count <= ONLINE_STEPS, "online step ledger exceeds budget")
        row = json.loads(line)
        ep = int(state["episode_ids"][(count - 1) % state["capacity"]])
        require(row["additional_online_step"] == count and row["episode_id"] == ep
                and ep in roads and row["geometry_seed"] == roads[ep]
                and roads[ep] in train and row["track_id"] in (1, 2, 3, 4)
                and row["gradient_steps"] == max(0, count - WARMUP)
                and row["source_samples"] == max(0, count - WARMUP) * 32
                and row["online_samples"] == max(0, count - WARMUP) * 32,
                f"online insertion/road/slot ledger mismatch at step {count}")
    require(count == ONLINE_STEPS, "incomplete online insertion ledger")


def audit(root: Path) -> dict:
    protocol = read_json(verified(root, PROTOCOL.as_posix(), PROTOCOL_SHA))
    require(protocol.get("format") == "haic-drq-retention-study-v1"
            and protocol.get("study_id") == "drqv2-retention-r7" and protocol.get("run_root") == ROOT.as_posix(),
            "wrong r7 frozen study")
    catalog = read_json(verified(root, "runs/20260925-drqv2-geometry-augmentation-v1/catalog/catalog.json",
                                 protocol["catalog_sha256"]))
    train = {int(row["geometry_seed"]) for row in catalog["train"]}
    diagnostic = {int(row["geometry_seed"]) for row in catalog["train_diagnostic"]}
    require(len(train) == 120 and len(diagnostic) == 16 and not train & diagnostic, "TRAIN split mismatch")
    require(protocol["replay_contract"]["batch_size"] == 64
            and protocol["replay_contract"]["source_rows"] == 32
            and protocol["replay_contract"]["online_rows"] == 32
            and protocol["replay_contract"]["sample_without_replacement_per_pool"] is True,
            "frozen replay contract mismatch")
    expected = {(seed, variant, condition) for seed in (0, 1)
                for variant in ("uniform", "failure_weighted", "easy_retention")
                for condition in ("r7a", "r7b")}
    arms = protocol["runs"]
    require(len(arms) == 12 and {(a["source_seed"], a["variant"], a["condition"]) for a in arms} == expected,
            "incomplete or duplicate 12-arm frozen grid")
    evidence = []
    sources = {}
    for seed in (0, 1):
        source = protocol["source_replay"][str(seed)]
        source_path = verified(root, source["checkpoint_path"], source["checkpoint_sha256"])
        ledger_path = verified(root, source["episode_ledger_path"], source["episode_ledger_sha256"])
        resets = episode_resets(ledger_path, diagnostic=diagnostic)
        source_payload = torch.load(source_path, map_location="cpu", mmap=True, weights_only=False)
        require(source_payload["format"] == "haic-drq-v2-checkpoint-v1"
                and source_payload["environment_steps"] == 131072,
                "source checkpoint is not original 131072-step replay")
        source_state = source_payload["replay"]
        valid, _ = valid_starts(source_state, first=31072, last=131072)
        ids = source_state["episode_ids"][np.arange(31072, 131072) % 100000]
        require(set(map(int, np.unique(ids))).issubset(resets), "source replay episode absent from original ledger")
        sources[str(seed)] = {"checkpoint_sha256": source["checkpoint_sha256"],
                              "episode_ledger_sha256": source["episode_ledger_sha256"],
                              "valid_original_n_step_starts": int(valid.sum())}
        for arm in [a for a in arms if a["source_seed"] == seed]:
            run = arm["run_dir"]
            require(run == (ROOT / f"learner-{seed}-{arm['variant']}-{arm['condition']}").as_posix(),
                    "non-frozen run path")
            verified(root, f"{run}/study_protocol.json", PROTOCOL_SHA)
            result_path = root / run / "result.json"
            result = read_json(result_path)
            require(result.get("format") == "haic-drq-retention-run-result-v1"
                    and result.get("completed") is True and result.get("study_protocol_sha256") == PROTOCOL_SHA
                    and (result.get("source_seed"), result.get("variant"), result.get("condition"))
                    == (seed, arm["variant"], arm["condition"])
                    and result.get("source_checkpoint_sha256") == source["checkpoint_sha256"]
                    and result.get("source_replay_sha256") == source["checkpoint_sha256"]
                    and result.get("additional_online_steps") == ONLINE_STEPS
                    and result.get("study_gradient_steps") == UPDATES
                    and result.get("source_samples") == SLOTS and result.get("online_samples") == SLOTS,
                    f"invalid result receipt: {run}")
            config = read_json(root / run / "run-config.json")
            require(config.get("protocol_sha256") == PROTOCOL_SHA
                    and config.get("source_replay_sha256") == source["checkpoint_sha256"]
                    and config.get("source_episode_ledger_sha256") == source["episode_ledger_sha256"]
                    and config.get("source_rows_per_batch") == 32 and config.get("online_rows_per_batch") == 32
                    and config.get("rng_seeds") == arm["rng_seeds"]
                    and config.get("lambda_preserve") == (protocol["lambda_preserve"] if arm["condition"] == "r7b" else 0.0),
                    f"invalid run configuration: {run}")
            catalog_receipt = read_json(root / run / "checkpoint-catalog.json")
            require(catalog_receipt.get("study_protocol_sha256") == PROTOCOL_SHA
                    and catalog_receipt.get("candidates") == result.get("candidates")
                    and len(result["candidates"]) == 2
                    and [c["checkpoint_online_step"] for c in result["candidates"]] == [16384, ONLINE_STEPS],
                    f"checkpoint catalog differs from run result: {run}")
            final = result["candidates"][-1]
            checkpoint_name = f"{run}/checkpoints/step-000032768/checkpoint.pt"
            trace_name = f"{run}/replay-sample-trace-step-000032768.npz"
            require(final["checkpoint_path"] == checkpoint_name and final["sample_trace_path"] == trace_name
                    and final["study_gradient_steps"] == UPDATES,
                    f"wrong final artifact identity: {run}")
            checkpoint_path = verified(root, checkpoint_name, final["checkpoint_sha256"])
            trace_path = verified(root, trace_name, final["sample_trace_sha256"])
            online_resets = episode_resets(root / run / "episodes.jsonl", diagnostic=diagnostic, train=train)
            payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
            trainer = payload["trainer_state"]
            require(payload["format"] == "haic-drq-v2-checkpoint-v1"
                    and payload["environment_steps"] == 131072 + ONLINE_STEPS
                    and payload["gradient_steps"] == UPDATES
                    and trainer["format"] == "haic-drq-retention-trainer-v1"
                    and (trainer["source_seed"], trainer["variant"], trainer["condition"])
                    == (seed, arm["variant"], arm["condition"])
                    and trainer["study_protocol_sha256"] == PROTOCOL_SHA
                    and trainer["source_replay_sha256"] == source["checkpoint_sha256"]
                    and trainer["sample_trace_sha256"] == final["sample_trace_sha256"]
                    and trainer["sample_trace_rows"] == UPDATES
                    and trainer["source_samples"] == SLOTS and trainer["online_samples"] == SLOTS
                    and trainer["additional_online_steps"] == ONLINE_STEPS
                    and trainer["study_gradient_steps"] == UPDATES,
                    f"final checkpoint trainer state mismatch: {run}")
            online = payload["replay"]
            online_valid, latest = valid_starts(online, first=0, last=ONLINE_STEPS)
            online_ids = online["episode_ids"][:ONLINE_STEPS]
            require(set(map(int, np.unique(online_ids))).issubset(online_resets),
                    f"online replay episode absent from TRAIN reset ledger: {run}")
            step_path = verified(root, f"{run}/step-metrics.jsonl", result["online_replay_manifest_sha256"])
            check_online_ledger(step_path, online, online_resets, train)
            with np.load(trace_path, allow_pickle=False) as trace:
                require(set(trace.files) == TRACE_KEYS
                        and bytes(trace["protocol_sha256"]).decode("ascii") == PROTOCOL_SHA,
                        f"wrong final trace schema or protocol: {run}")
                counts = check_trace(trace["source"], trace["source_indices"], trace["episode_id"],
                                     source_first=31072, source_valid=valid, source_ids=ids,
                                     online_valid=online_valid, online_ids=online_ids,
                                     online_latest=latest, source_roads=resets,
                                     online_roads=online_resets, diagnostic=diagnostic)
            evidence.append({"run_dir": run, "source_seed": seed, "variant": arm["variant"],
                             "condition": arm["condition"], "result_sha256": digest(result_path),
                             "final_checkpoint_sha256": final["checkpoint_sha256"],
                             "final_trace_sha256": final["sample_trace_sha256"],
                             "online_step_ledger_sha256": result["online_replay_manifest_sha256"], **counts})
            del payload, online, online_valid, latest, online_ids
            gc.collect()
        del source_payload, source_state, valid, ids
        gc.collect()
        require(digest(source_path) == source["checkpoint_sha256"]
                and digest(ledger_path) == source["episode_ledger_sha256"],
                "original source replay or ledger changed during audit")
    require(len(evidence) == 12 and { (row["source_seed"], row["variant"], row["condition"])
                                      for row in evidence } == expected, "not all frozen arms audited")
    return {"format": "haic-drq-retention-pre-evaluation-trace-audit-v1", "passed": True,
            "study_protocol_sha256": PROTOCOL_SHA, "evaluations_or_environment_resets": 0,
            "scope": "offline sampled replay provenance and n-step validity, not source-state coverage or performance",
            "source_replays": sources, "arms": evidence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    require(not (root / RECEIPT).exists(), "audit receipt already exists; refusing overwrite")
    result = audit(root)
    with (root / RECEIPT).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "arms": len(result["arms"]),
                      "receipt": RECEIPT.as_posix(), "sha256": digest(root / RECEIPT)}, sort_keys=True))


if __name__ == "__main__":
    main()
