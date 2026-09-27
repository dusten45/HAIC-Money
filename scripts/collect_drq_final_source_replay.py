"""Collect one sealed, frozen-final-actor TRAIN replay; never update a learner.

The original checkpoint's retained window supplies the road order, not pixels or
actions. Preflight requires a separately frozen, candidate-specific cross-lane
road audit; a source ledger by itself cannot certify protected allocations.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

from common_adapter import EpisodeCollector
from drq_v2 import Uint8Replay, load_exported_actor
from haic.algorithms.drq_v2.teacher_study import audit_source_actor_pair
from scripts import train_drq_geometry_mix as r6
from scripts import train_drq_retention_r7 as r7
from train import build_env


STUDY_ID = "drqv2-final-source-replay-v1"
RUN_ROOT = Path("runs/20260927-drqv2-final-source-replay-v1")
PROTOCOL_PATH = Path("experiments/drqv2-final-source-replay-collection-v1.json")
FORMAT = "haic-drq-final-source-collection-v1"
SCHEDULE_ALGORITHM = "original-retained-then-earlier-once-v1"
DECISIONS = CAPACITY = 100_000
FIRST_SOURCE_SEQUENCE = 31_072
LAST_SOURCE_SEQUENCE = 131_071


def require(ok: bool, explanation: str) -> None:
    if not ok:
        raise ValueError(explanation)


def schedule_sha256(pairs: list[dict]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _original_ledger(path: Path) -> list[dict]:
    resets: list[dict] = []
    previous_end = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        row = json.loads(line)
        if row.get("event") == "reset":
            episode = row.get("episode_id")
            require(type(episode) is int and episode == len(resets), "source episode ID discontinuity")
            require(type(row.get("track_id")) is int and row["track_id"] in (1, 2, 3, 4)
                    and type(row.get("seed")) is int and 0 <= row["seed"] < 2**32,
                    "source reset is not an original obstacle TRAIN road")
            require(not resets or "end_step" in resets[-1], "source ledger reset without end")
            resets.append({"episode_id": episode, "track_id": row["track_id"],
                           "geometry_seed": row["seed"], "start_sequence": previous_end})
        elif row.get("event") == "end":
            require(bool(resets) and row.get("episode_id") == resets[-1]["episode_id"]
                    and row.get("seed") == resets[-1]["geometry_seed"]
                    and row.get("track_id") == resets[-1]["track_id"]
                    and type(row.get("global_step")) is int
                    and type(row.get("steps")) is int and 0 < row["steps"] <= 2000
                    and row["global_step"] == previous_end + row["steps"],
                    "source episode ledger steps/road do not join")
            previous_end = row["global_step"]
            resets[-1]["end_step"] = previous_end
        else:
            raise ValueError("unsupported original source ledger event")
    require(bool(resets) and previous_end <= 131072, "source ledger has no usable resets")
    return resets


def derive_schedule(state: dict, ledger: list[dict], excluded: set[tuple[int, int]]) -> list[dict]:
    require(state.get("capacity") == CAPACITY and state.get("size") == CAPACITY
            and state.get("next_sequence") == LAST_SOURCE_SEQUENCE + 1
            and state.get("action_dim") == 3 and state.get("n_step") == 3
            and state.get("gamma") == .99, "original checkpoint rolling replay configuration mismatch")
    positions = np.arange(FIRST_SOURCE_SEQUENCE, LAST_SOURCE_SEQUENCE + 1, dtype=np.int64)
    ids = np.asarray(state["sequence_ids"])[positions % CAPACITY]
    episodes = np.asarray(state["episode_ids"])[positions % CAPACITY]
    steps = np.asarray(state["episode_steps"])[positions % CAPACITY]
    require(np.array_equal(ids, positions), "original checkpoint retained sequence ring is incomplete")
    boundary_episode = int(episodes[0])
    require(0 <= boundary_episode < len(ledger)
            and ledger[boundary_episode]["start_sequence"] < FIRST_SOURCE_SEQUENCE
            and ledger[boundary_episode]["end_step"] > FIRST_SOURCE_SEQUENCE
            and int(steps[0]) == FIRST_SOURCE_SEQUENCE - ledger[boundary_episode]["start_sequence"],
            "first retained source episode is not the expected boundary-straddling partial")
    require(len(ledger) > boundary_episode + 1, "no original reset after replay boundary")
    for episode in np.unique(episodes):
        ep = int(episode)
        require(boundary_episode <= ep < len(ledger), "replay references an unknown original episode")
        selected = episodes == ep
        first = int(positions[selected][0])
        start = ledger[ep]["start_sequence"]
        require(first == max(FIRST_SOURCE_SEQUENCE, start)
                and np.array_equal(steps[selected], positions[selected] - start)
                and int(positions[selected][-1]) < ledger[ep].get("end_step", LAST_SOURCE_SEQUENCE + 1),
                "original replay episode steps do not join source ledger")
    scheduled = []
    # Restart the straddling episode from its original reset. Earlier historical
    # source TRAIN resets are a fixed, finite contingency if the final actor's
    # shorter episodes exhaust the retained-window order before 100k decisions.
    for row in ledger[boundary_episode:] + ledger[:boundary_episode]:
        start = row["start_sequence"]
        if row["episode_id"] == boundary_episode:
            require(start < FIRST_SOURCE_SEQUENCE < row["end_step"],
                    "first scheduled road is not the historical boundary episode")
        elif row["episode_id"] > boundary_episode:
            require(FIRST_SOURCE_SEQUENCE <= start <= LAST_SOURCE_SEQUENCE
                    and int(episodes[start - FIRST_SOURCE_SEQUENCE]) == row["episode_id"]
                    and int(steps[start - FIRST_SOURCE_SEQUENCE]) == 0,
                    "scheduled retained reset is absent from original replay")
        else:
            require(0 <= start < row["end_step"] <= FIRST_SOURCE_SEQUENCE,
                    "historical prefix road was not completed before the retained window")
        pair = (row["track_id"], row["geometry_seed"])
        require(pair not in excluded, "original source schedule contains diagnostic/protected/reserved road")
        scheduled.append({"original_episode_id": row["episode_id"],
                          "track_id": pair[0], "geometry_seed": pair[1]})
    require(len(scheduled) == len(ledger) and
            len({(row["track_id"], row["geometry_seed"]) for row in scheduled}) == len(scheduled),
            "finite source TRAIN schedule duplicates or omits an original reset")
    return scheduled


def _excluded_pairs(reference: dict, root: Path) -> set[tuple[int, int]]:
    catalog_path = r7.pinned(root, reference["catalog"]["path"], reference["catalog"]["sha256"])
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    require(len(catalog["train_diagnostic"]) == 16, "original diagnostic exclusion catalog changed")
    return {(int(row["track_id"]), int(row["geometry_seed"])) for row in catalog["train_diagnostic"]}


def load_protocol(root: Path, path: Path, *, source_seed: int | None = None,
                  check_outputs: bool = True) -> tuple[dict, dict, dict[int, list[dict]], str]:
    require(path.resolve(strict=True).is_relative_to(root), "collection protocol must be repository-local")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    require(protocol.get("format") == FORMAT and protocol.get("study_id") == STUDY_ID
            and protocol.get("run_root") == RUN_ROOT.as_posix()
            and protocol.get("schedule_algorithm") == SCHEDULE_ALGORITHM,
            "not the separately frozen original-ledger final-source collection protocol")
    require(protocol.get("r6_protocol_sha256") == r7.R6_SHA
            and protocol.get("r7_protocol_sha256") == "774b4a7119b32bde359d5a6e5b789b69badd6c899bf15eebf5e5825b6e6a59c9",
            "original r6/r7 references changed")
    r7_protocol = json.loads(r7.pinned(root, protocol["r7_protocol_path"],
                                       protocol["r7_protocol_sha256"]).read_text(encoding="utf-8"))
    require(r7_protocol["r6_protocol_path"] == protocol["r6_protocol_path"]
            and r7_protocol["r6_protocol_sha256"] == protocol["r6_protocol_sha256"],
            "collection original source does not match frozen r7 reference")
    reference = json.loads(r7.pinned(root, protocol["r6_protocol_path"], r7.R6_SHA).read_text(encoding="utf-8"))
    r6._validate_protocol(reference)
    r6._validate_catalog(reference, root)
    r6._checkpoint_source_paths(root, reference)
    require(protocol.get("catalog_sha256") == reference["catalog"]["sha256"],
            "r6 online/diagnostic catalog SHA differs")
    require(protocol.get("decisions") == DECISIONS and protocol.get("capacity") == CAPACITY
            and protocol.get("n_step") == 3 and protocol.get("gamma") == .99
            and protocol.get("collection_noise_std") == .05
            and protocol.get("environment") == {"frame_skip": 4, "max_steps": 2000,
                                                  "obstacles": True, "raw_reward": True,
                                                  "track_ids": [1, 2, 3, 4]},
            "collection budget/environment/action noise changed")
    hashes = protocol.get("code_sha256")
    require(isinstance(hashes, dict) and {"scripts/collect_drq_final_source_replay.py", "train.py",
                                              "common_adapter.py", "drq_v2.py"}.issubset(hashes),
            "collector/runtime executable SHA map must be frozen")
    for name, digest in hashes.items():
        r7.pinned(root, name, digest)
    require(protocol.get("runtime") == reference["runtime"]["training_dependencies"],
            "frozen collection dependency runtime differs from source stack")
    for package, version in protocol["runtime"].items():
        actual = sys.version.split()[0] if package == "python" else metadata.version(package)
        require(actual == version, f"collection runtime version mismatch: {package}")
    require(set(protocol.get("sources", {})) == {"0", "1"}, "collection must bind both original seeds")
    excluded = _excluded_pairs(reference, root)
    schedules = {}
    for seed in (0, 1):
        source = next(item for item in reference["source_actors"] if item["learner_seed"] == seed)
        frozen = protocol["sources"][str(seed)]
        old = r7_protocol["source_replay"][str(seed)]
        require(frozen.get("source_checkpoint_sha256") == source["checkpoint_sha256"] == old["checkpoint_sha256"]
                and frozen.get("source_actor_sha256") == source["actor_sha256"]
                and frozen.get("original_ledger_sha256") == old["episode_ledger_sha256"]
                and frozen.get("schedule_sha256") is not None,
                "source checkpoint/export/ledger/schedule must be frozen")
        ledger = _original_ledger(r7.pinned(root, old["episode_ledger_path"], old["episode_ledger_sha256"]))
        checkpoint = r7.pinned(root, source["checkpoint_path"], source["checkpoint_sha256"])
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        require(payload.get("format") == "haic-drq-v2-checkpoint-v1"
                and payload.get("environment_steps") == 131072,
                "source replay is not original 131072-decision checkpoint")
        schedules[seed] = derive_schedule(payload["replay"], ledger, excluded)
        del payload
        require(schedule_sha256(schedules[seed]) == frozen["schedule_sha256"],
                "frozen original TRAIN reset schedule SHA mismatch")
        noise = frozen.get("collection_rng_seeds")
        require(isinstance(noise, dict) and set(noise) == {"action_noise_seed"}
                and type(noise["action_noise_seed"]) is int
                and 0 <= noise["action_noise_seed"] < 2**32,
                "independent collection action-noise seed must be frozen")
    require(protocol["sources"]["0"]["collection_rng_seeds"] != protocol["sources"]["1"]["collection_rng_seeds"],
            "collection streams must be independent across seeds")
    audit = protocol.get("cross_lane_audit")
    require(isinstance(audit, dict) and isinstance(audit.get("path"), str)
            and isinstance(audit.get("sha256"), str),
            "prospective cross-lane audit unavailable: FAIL CLOSED before reset")
    evidence = json.loads(r7.pinned(root, audit["path"], audit["sha256"]).read_text(encoding="utf-8"))
    require(evidence.get("format") == "haic-drq-final-source-cross-lane-audit-v1"
            and evidence.get("passed") is True and evidence.get("partition") == "TRAIN"
            and evidence.get("schedule_sha256") == {str(seed): schedule_sha256(schedules[seed]) for seed in (0, 1)}
            and evidence.get("original_ledger_sha256") == {
                str(seed): protocol["sources"][str(seed)]["original_ledger_sha256"] for seed in (0, 1)}
            and evidence.get("excluded_diagnostic_roads") is True
            and evidence.get("protected_or_reserved_overlap") == []
            and evidence.get("ambiguous_records") == [],
            "candidate-specific cross-lane audit did not clear every scheduled TRAIN pair")
    if check_outputs:
        for seed in (0, 1) if source_seed is None else (source_seed,):
            require(not (root / RUN_ROOT / "collection" / f"seed{seed}").exists(),
                    "collection output exists; refusing overwrite before reset")
    return protocol, reference, schedules, r7.sha(path)


def _write_jsonl(stream, row: dict) -> None:
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def collect(root: Path, protocol_path: Path, seed: int) -> dict:
    require(seed in (0, 1), "only original source seeds 0/1 can be collected")
    protocol, reference, schedules, protocol_sha = load_protocol(root, protocol_path, source_seed=seed)
    source = next(item for item in reference["source_actors"] if item["learner_seed"] == seed)
    frozen = protocol["sources"][str(seed)]
    old = json.loads((root / protocol["r7_protocol_path"]).read_text(encoding="utf-8"))[
        "source_replay"][str(seed)]
    original_ledger = r7.pinned(root, old["episode_ledger_path"], frozen["original_ledger_sha256"])
    checkpoint = r7.pinned(root, source["checkpoint_path"], source["checkpoint_sha256"])
    actor_path = r7.pinned(root, source["actor_path"], source["actor_sha256"])
    audit_source_actor_pair(checkpoint, actor_path, learner_seed=seed,
                            source_revision=source["source_revision"],
                            expected_checkpoint_sha256=source["checkpoint_sha256"],
                            expected_actor_sha256=source["actor_sha256"])
    actor, adapter, spec = load_exported_actor(actor_path, device="cpu")
    actor.eval().requires_grad_(False)
    require(not any(p.requires_grad for p in actor.parameters()), "collection actor must be frozen")
    noise = np.random.default_rng(frozen["collection_rng_seeds"]["action_noise_seed"])
    replay = Uint8Replay(capacity=CAPACITY, n_step=3, gamma=.99)
    output = root / RUN_ROOT / "collection" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "collection_protocol.json").write_bytes(protocol_path.read_bytes())
    episodes_path, steps_path = output / "episodes.jsonl", output / "steps.jsonl"
    decisions, consumed = 0, 0
    prefix_decisions, prefix_episodes = 0, 0
    prefix_start = sum(row["original_episode_id"] >= schedules[seed][0]["original_episode_id"]
                       for row in schedules[seed])
    visited_roads: set[int] = set()
    try:
        with episodes_path.open("x", encoding="utf-8") as episodes, steps_path.open("x", encoding="utf-8") as steps:
            for index, road in enumerate(schedules[seed]):
                if decisions == DECISIONS:
                    break
                phase = "retained_window" if index < prefix_start else "historical_prefix_once"
                env = build_env(track_id=road["track_id"], seed=road["geometry_seed"],
                                max_steps=2000, frame_skip=4, reward_shaping=False,
                                obstacles=True, collision_penalty=0.0)
                collector = EpisodeCollector(env, observation_spec=spec, action_adapter=adapter, gamma=.99)
                collector.episode_id = index - 1
                try:
                    observation, info = collector.reset()
                    require(info.get("track_id") == road["track_id"]
                            and info.get("seed") == road["geometry_seed"],
                            "actual reset differs from original source TRAIN pair")
                    consumed += 1
                    prefix_episodes += int(index >= prefix_start)
                    visited_roads.add(road["geometry_seed"])
                    _write_jsonl(episodes, {"event": "reset", "source_seed": seed,
                                            "schedule_phase": phase,
                                            "schedule_index": index, "original_episode_id": road["original_episode_id"],
                                            "episode_id": index, "collection_step": decisions,
                                            "track_id": road["track_id"], "seed": road["geometry_seed"],
                                            "geometry_seed": road["geometry_seed"], "partition": "TRAIN"})
                    episode_reward = 0.0
                    while decisions < DECISIONS:
                        with torch.inference_mode():
                            deterministic = actor(torch.as_tensor(observation).unsqueeze(0)).squeeze(0).cpu().numpy()
                        perturbation = noise.normal(0.0, .05, size=3)
                        noised = deterministic + perturbation
                        native = np.clip(noised, -1., 1.).astype(np.float32)
                        transition = collector.step(native)
                        require((transition.info.get("track_id"), transition.info.get("seed")) ==
                                (road["track_id"], road["geometry_seed"]), "step crossed original TRAIN road")
                        require(np.max(np.abs(transition.action - native)) <= 1e-6,
                                "action adapter altered native action beyond float32 roundoff")
                        replay.add(transition)
                        decisions += 1
                        prefix_decisions += int(index >= prefix_start)
                        episode_reward += transition.reward
                        _write_jsonl(steps, {"decision": decisions, "sequence_id": decisions - 1,
                                             "schedule_phase": phase,
                                             "episode_id": index, "episode_step": transition.step,
                                             "original_episode_id": road["original_episode_id"],
                                             "schedule_index": index, "source_seed": seed,
                                             "track_id": road["track_id"], "geometry_seed": road["geometry_seed"],
                                             "partition": "TRAIN", "unnoised_native_action": deterministic.tolist(),
                                             "noise": perturbation.tolist(), "noised_native_action": noised.tolist(),
                                             "pre_adapter_native_action": native.tolist(),
                                             "native_action": transition.action.tolist(),
                                             "official_action": transition.applied_action.tolist(),
                                             "reward": transition.reward,
                                             "terminated": transition.terminated, "truncated": transition.truncated,
                                             "terminal": transition.terminal,
                                             "finished": transition.info.get("finished"),
                                             "progress": transition.info.get("progress"),
                                             "damage": transition.info.get("damage"),
                                             "retire_reason": transition.info.get("retire_reason")})
                        if transition.done or decisions == DECISIONS:
                            _write_jsonl(episodes, {"event": "end" if transition.done else "capped_partial",
                                                    "schedule_phase": phase,
                                                    "episode_id": index, "original_episode_id": road["original_episode_id"],
                                                    "schedule_index": index, "source_seed": seed,
                                                    "track_id": road["track_id"], "seed": road["geometry_seed"],
                                                    "geometry_seed": road["geometry_seed"],
                                                    "collection_step": decisions, "steps": transition.step + 1,
                                                    "reward": episode_reward, "terminated": transition.terminated,
                                                    "truncated": transition.truncated, "terminal": transition.terminal,
                                                    "finished": transition.info.get("finished")})
                            break
                        observation = transition.next_observation
                finally:
                    env.close()
            require(decisions == DECISIONS, "finite original TRAIN reset schedule exhausted before 100000 decisions")
        require(replay.size == CAPACITY and replay.oldest_sequence == 0 and replay.newest_sequence == DECISIONS - 1,
                "new source pool must contain precisely sequence IDs 0..99999")
        valid = len(replay.valid_indices())
        require(valid >= 32, "sealed final source replay has fewer than 32 terminal-safe starts")
        pool_path = output / "pool.pt"
        torch.save({"format": "haic-drq-final-source-replay-v1", "source_seed": seed,
                    "collection_protocol_sha256": protocol_sha, "source_actor_sha256": source["actor_sha256"],
                    "source_checkpoint_sha256": source["checkpoint_sha256"], "replay": replay.state_dict()}, pool_path)
        result = {"format": "haic-drq-final-source-pool-v1", "completed": True,
                  "source_seed": seed, "source_actor_sha256": source["actor_sha256"],
                  "source_checkpoint_sha256": source["checkpoint_sha256"],
                  "pool_path": pool_path.relative_to(root).as_posix(), "pool_sha256": r7.sha(pool_path),
                  "partition": "TRAIN", "excluded_diagnostic_roads": True,
                  "decisions": DECISIONS, "capacity": CAPACITY, "valid_n_step_starts": valid,
                  "episode_ledger_path": episodes_path.relative_to(root).as_posix(),
                  "episode_ledger_sha256": r7.sha(episodes_path),
                  "step_ledger_path": steps_path.relative_to(root).as_posix(),
                  "step_ledger_sha256": r7.sha(steps_path),
                  "collection_rng_seeds": frozen["collection_rng_seeds"],
                  "collection_noise_std": .05,
                  "original_ledger_sha256": frozen["original_ledger_sha256"],
                  "schedule_sha256": frozen["schedule_sha256"],
                   "catalog_sha256": reference["catalog"]["sha256"],
                   "collection_protocol_sha256": protocol_sha, "scheduled_episodes_consumed": consumed,
                   "retained_window_schedule_episodes": prefix_start,
                   "historical_prefix_episodes_consumed": prefix_episodes,
                   "historical_prefix_decisions": prefix_decisions}
        result["geometry_seeds"] = sorted(visited_roads)
        require(r7.sha(checkpoint) == source["checkpoint_sha256"]
                and r7.sha(actor_path) == source["actor_sha256"]
                and r7.sha(root / protocol["r7_protocol_path"]) == protocol["r7_protocol_sha256"]
                and r7.sha(root / protocol["r6_protocol_path"]) == protocol["r6_protocol_sha256"]
                and r7.sha(original_ledger) == frozen["original_ledger_sha256"]
                and r7.sha(protocol_path) == protocol_sha,
                "original source, ledger or collection protocol changed during collection")
        r6._atomic_json(output / "receipt.json", result)
        return result
    except Exception as error:
        failure = {"format": "haic-drq-final-source-pool-failure-v1", "completed": False,
                   "source_seed": seed, "decisions": decisions, "scheduled_episodes_consumed": consumed,
                   "historical_prefix_episodes_consumed": prefix_episodes,
                   "historical_prefix_decisions": prefix_decisions,
                   "schedule_sha256": frozen["schedule_sha256"], "error": str(error),
                   "episode_ledger_sha256": r7.sha(episodes_path) if episodes_path.exists() else None,
                   "step_ledger_sha256": r7.sha(steps_path) if steps_path.exists() else None}
        r6._atomic_json(output / "failure.json", failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-seed", type=int, choices=(0, 1))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    if args.preflight_only:
        _, _, schedules, digest = load_protocol(root, path, source_seed=args.source_seed)
        print(json.dumps({"collection_protocol_sha256": digest,
                          "scheduled_reset_pairs": {str(seed): len(schedules[seed]) for seed in (0, 1)},
                          "environment_decisions": 0, "learner_updates": 0}, sort_keys=True))
        return 0
    if args.source_seed is None:
        parser.error("--source-seed is required for collection")
    print(json.dumps(collect(root, path, args.source_seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
