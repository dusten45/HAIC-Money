"""Read-only, source-pinned DrQ failure analysis on already consumed road cells.

Run from the repository root with ``python -m scripts.analyze_drq_geometry_failures``.
This tool never evaluates an actor, trains a model, or discovers input directories.
Raw road resets are opt-in and restricted to validated, consumed seed lists.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from haic.algorithms.drq_v2.geometry_features import describe_track, validate_track_structure


ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "pad": ("drqv2-augmentation-pad-v1", "20260922-drq-augmentation-pad-v1-restart"),
    "l2": ("drqv2-steering-logit-v1", "20260922-drq-steering-l2-v1-fast"),
}
R3 = "drqv2-teacher-replay-v1-r3"
R3_RUN = "runs/20260924-drqv2-teacher-replay-v1-r3"
PREFIX = "drqv2-residual-options-prefix-branch-v1"
PREFIX_RUN = "runs/20260924-drqv2-residual-options-prefix-branch-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
EVAL_STAMP = re.compile(r"[0-9]{8}T[0-9]{12}Z_([a-z0-9-]+)-(screen|confirmation)\Z")
FROZEN_PROTOCOL_SHA256 = {
    R3: "7b73cab4fb46ca2fb8e34b99a8667022f3027640e1582343b0b91aa4df4e0e14",
    STUDIES["pad"][0]: "4a383ec6b83c501a814b0a18c22a39b851d23b22093d56244aa6540b977f160b",
    STUDIES["l2"][0]: "348d1534a429a313a2927200f027f8e14b957f109aa7e2451fecb0ae1c76b235",
    PREFIX: "e39287ef8e38408a3a0184c8b9f7a0cf4212711965e52ab91b2ddbefd34f1cac",
}
FROZEN_SOURCE_SHA256 = {
    0: "433115ce01f6261373571de1c7bd8a6dc0b74f8e2714cb8e16f820dae4c4ad37",
    1: "c0ceab5e640f35d73694a76e75302b855179556e0bbc193f9e4a64e3d7992954",
}


def _integer(value: object, name: str, *, low: int = 0) -> int:
    if type(value) is not int or not low <= value < 2**32:
        raise ValueError(f"{name} must be an integer in [{low}, 2**32)")
    return value


def _number(value: object, name: str, *, low: float | None = None, high: float | None = None
            ) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} is outside its expected range")
    return float(value)


def _path(root: Path, relative: str, *, expected: str | None = None) -> Path:
    """Do not follow artifact pointers outside their exact, source-owned location."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("input path must be a nonempty repository-relative POSIX path")
    parts = Path(relative).parts
    if (Path(relative).is_absolute() or any(part in (".", "..") for part in parts)
            or any("blind" in part.lower() for part in parts)):
        raise ValueError(f"blind or untrusted input path: {relative}")
    if expected is not None and relative != expected:
        raise ValueError(f"input path is outside the exact source allowlist: {relative}")
    path = root / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"input path escapes repository root: {relative}")
    if any("blind" in part.lower() for part in resolved.relative_to(root.resolve()).parts):
        raise ValueError(f"blind symlink target is forbidden: {relative}")
    if not path.is_file():
        raise ValueError(f"expected sealed input file does not exist: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Inputs:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.hashes: dict[str, str] = {}

    def file(self, relative: str, *, expected: str | None = None,
             sha256: str | None = None) -> Path:
        path = _path(self.root, relative, expected=expected)
        actual = _sha256(path)
        if sha256 is not None and (not SHA.fullmatch(sha256) or actual != sha256):
            raise ValueError(f"sealed input hash mismatch: {relative}")
        self.hashes[relative] = actual
        return path

    def json(self, relative: str, *, expected: str | None = None,
             sha256: str | None = None):
        path = self.file(relative, expected=expected, sha256=sha256)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object: {relative}")
        return value


def _pool(matrix: dict, *, name: str) -> tuple[list[int], list[int], int]:
    if not isinstance(matrix, dict):
        raise ValueError(f"{name} partition is missing")
    ids, seeds = matrix.get("track_ids"), matrix.get("seeds")
    if not isinstance(ids, list) or not ids or not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{name} partition must contain explicit IDs and seeds")
    for value in ids:
        _integer(value, f"{name} track_id", low=1)
    for value in seeds:
        _integer(value, f"{name} seed")
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} partition contains duplicates")
    repeats = _integer(matrix.get("repeats", 1), f"{name} repeats", low=1)
    return ids, seeds, repeats


def _protocol(inputs: Inputs, name: str) -> tuple[dict, str]:
    relative = f"experiments/{name}.json"
    record = inputs.json(relative, expected=relative)
    if inputs.hashes[relative] != FROZEN_PROTOCOL_SHA256[name]:
        raise ValueError(f"consumed protocol SHA-256 differs from frozen source: {relative}")
    if record.get("name") != name and record.get("study_id") != name:
        raise ValueError(f"wrong frozen protocol: {relative}")
    if record.get("frame_skip") != 4 or record.get("max_steps") != 2000:
        raise ValueError("source protocol must use the frozen decision/frame horizon")
    partitions = record.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"screen", "confirmation", "blind"}:
        raise ValueError("expected the frozen three-partition protocol")
    # Blind seed *identifiers* are used only to exclude them from every read/reset.
    pools = [_pool(partitions[key], name=key) for key in ("screen", "confirmation", "blind")]
    for left in range(3):
        for right in range(left + 1, 3):
            if set(pools[left][1]) & set(pools[right][1]):
                raise ValueError("partition seed leakage")
    return record, inputs.hashes[relative]


def _actions(actions: object, steps: int | None, digest: str | None = None) -> tuple[dict | None, str | None]:
    if actions is None:
        if digest is not None:
            raise ValueError("action digest exists without per-step actions")
        return None, None
    array = np.asarray(actions, dtype=np.float32)
    if (array.ndim != 2 or array.shape[1] != 3 or not len(array)
            or (steps is not None and len(array) != steps) or not np.isfinite(array).all()
            or np.any(array < [-1, 0, 0]) or np.any(array > [1, 1, 1])):
        raise ValueError("invalid executed official action sequence")
    actual = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    if digest is not None and actual != digest:
        raise ValueError("executed action trace SHA-256 mismatch")
    steer, gas, brake = array.T
    return {
        "decisions": len(array),
        "steering_mean": float(steer.mean()),
        "steering_abs_mean": float(np.abs(steer).mean()),
        "steering_saturated_fraction_abs_ge_0_95": float(np.mean(np.abs(steer) >= .95)),
        "steering_delta_abs_mean": float(np.mean(np.abs(np.diff(steer)))) if len(array) > 1 else 0.0,
        "gas_mean": float(gas.mean()),
        "brake_mean": float(brake.mean()),
        "simultaneous_gas_brake_fraction_gt_0_1": float(np.mean((gas > .1) & (brake > .1))),
    }, actual


def _outcome(row: dict, *, steps_key: str = "steps", actions_key: str = "actions") -> dict:
    if type(row.get("finished")) is not bool:
        raise ValueError("finished must be an explicit boolean")
    steps = _integer(row.get(steps_key), "episode decisions", low=1)
    progress = _number(row.get("progress"), "terminal visited-tile progress", low=0, high=1)
    damage = _number(row["damage"], "terminal damage", low=0) if row.get("damage") is not None else None
    max_progress = (_number(row["max_progress"], "max visited-tile progress", low=0, high=1)
                    if row.get("max_progress") is not None else None)
    if max_progress is not None and max_progress + 1e-6 < progress:
        raise ValueError("maximum progress is smaller than terminal progress")
    reason = row.get("retire_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("invalid retirement reason")
    # The wrapper may report both a finish and a crash on the same final step;
    # the evaluator gives finished priority without erasing the retirement label.
    qualified = row.get("finish_qualified")
    if qualified is not None and type(qualified) is not bool:
        raise ValueError("finish qualification must be boolean when recorded")
    if qualified is False and row["finished"]:
        raise ValueError("finish is inconsistent with finish qualification")
    finish_time = row.get("finish_time_s")
    if finish_time is not None:
        finish_time = _number(finish_time, "finish time", low=0)
    terminal_class = row.get("termination_class") or ("finished" if row["finished"] else reason)
    if terminal_class is not None and not isinstance(terminal_class, str):
        raise ValueError("terminal class must be a string when recorded")
    stats, trace = _actions(row.get(actions_key), steps, row.get("action_trace_sha256"))
    reached_high = (not row["finished"] and (progress >= .9 or max_progress is not None and max_progress >= .9))
    return {
        "finished": row["finished"], "progress": progress, "max_progress": max_progress,
        "steps": steps, "damage": damage, "retire_reason": reason,
        "high_progress_nonfinish": not row["finished"] and progress >= .9,
        "ever_reached_ge_0_9_nonfinish_when_observable": (
            reached_high if progress >= .9 or max_progress is not None else None),
        "finish_qualified": qualified,
        "finish_time_s": finish_time,
        "terminal_class": terminal_class,
        "action_stats": stats, "action_trace_sha256": trace,
    }


def _summary(rows: list[dict]) -> dict:
    def mean(name: str):
        values = [row[name] for row in rows if row.get(name) is not None]
        return sum(values) / len(values) if values else None

    reasons = Counter(row["terminal_class"] or "unknown" for row in rows)
    bins = Counter(f"{min(9, int(row['progress'] * 10)) / 10:.1f}-{(min(9, int(row['progress'] * 10)) + 1) / 10:.1f}"
                   for row in rows if not row["finished"])
    return {
        "episodes": len(rows), "finishes": sum(row["finished"] for row in rows),
        "high_progress_nonfinishes_ge_0_9": sum(row["high_progress_nonfinish"] for row in rows),
        "ever_reached_ge_0_9_nonfinishes_when_observable": sum(
            row["ever_reached_ge_0_9_nonfinish_when_observable"] is True for row in rows),
        "ever_reached_ge_0_9_observable_episodes": sum(
            row["ever_reached_ge_0_9_nonfinish_when_observable"] is not None for row in rows),
        "terminal_class_counts": dict(sorted(reasons.items())),
        "failure_terminal_progress_bins": dict(sorted(bins.items())),
        "mean_terminal_progress": mean("progress"), "mean_max_progress_when_available": mean("max_progress"),
        "max_progress_available_episodes": sum(row["max_progress"] is not None for row in rows),
        "mean_decisions": mean("steps"), "mean_damage_when_available": mean("damage"),
        "damage_available_episodes": sum(row["damage"] is not None for row in rows),
        "action_stats_mean_when_available": {
            key: sum(row["action_stats"][key] for row in rows if row["action_stats"] is not None)
                 / sum(row["action_stats"] is not None for row in rows)
            for key in ("steering_mean", "steering_abs_mean", "steering_saturated_fraction_abs_ge_0_95",
                        "steering_delta_abs_mean", "gas_mean", "brake_mean",
                        "simultaneous_gas_brake_fraction_gt_0_1")
        } if any(row["action_stats"] is not None for row in rows) else None,
    }


def aggregate(rows: list[dict]) -> dict:
    """Count episodes, distinct roads, obstacle variants and identical source traces separately."""
    groups: dict[int, list[dict]] = defaultdict(list)
    duplicates: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["geometry_seed"]].append(row)
        if row["action_trace_sha256"] is not None:
            duplicates[(row["actor_sha256"], row["track_id"], row["geometry_seed"],
                        row["action_trace_sha256"])].append(row)
    return {
        "summary": _summary(rows),
        "canonical_episode_rows": sorted(rows, key=lambda r: (
            r["actor_sha256"], r["study"], r["partition"], r["geometry_seed"],
            r["track_id"], r.get("episode_id", -1))),
        "unique_road_seeds": len(groups),
        "unique_track_seed_obstacle_cells": len({(r["track_id"], r["geometry_seed"]) for r in rows}),
        "repeated_identical_source_trajectories": [
            {"actor_sha256": key[0], "track_id": key[1], "geometry_seed": key[2],
             "action_trace_sha256": key[3], "count": len(repeats),
             "observed_in": [{"study": study, "partition": partition}
                             for study, partition in sorted({(r["study"], r["partition"]) for r in repeats})]}
            for key, repeats in sorted(duplicates.items()) if len(repeats) > 1
        ],
        "by_road_seed": [
            {"geometry_seed": seed, "track_id_obstacle_variants": sorted({r["track_id"] for r in group}),
             "outcome": "mixed" if any(r["finished"] for r in group) and not all(r["finished"] for r in group)
             else "any_finish" if any(r["finished"] for r in group) else "no_finish",
             **_summary(group)}
            for seed, group in sorted(groups.items())
        ],
    }


def _evaluation(inputs: Inputs, protocol: dict, protocol_hash: str, name: str,
                run: str, learner: int, source_sha: str, partition: str,
                pointer: dict) -> list[dict]:
    base = f"runs/{run}/control-seed{learner}"
    directory = pointer.get("evaluation_dir")
    if not isinstance(directory, str):
        raise ValueError("selected source lacks a sealed evaluation directory")
    relative = Path(directory)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(inputs.root)
        except ValueError as exc:
            raise ValueError("evaluation directory is outside the source run") from exc
    relative = relative.as_posix()
    if (not relative.startswith(base + "/evaluations/")
            or not EVAL_STAMP.fullmatch(Path(relative).name)
            or EVAL_STAMP.fullmatch(Path(relative).name).groups() != (name, partition)
            or Path(relative).parent.as_posix() != base + "/evaluations"):
        raise ValueError("evaluation directory is not the selected source partition")
    matrix = protocol["partitions"][partition]
    ids, seeds, repeats = _pool(matrix, name=partition)
    manifest = inputs.json(f"{relative}/manifest.json", expected=f"{relative}/manifest.json")
    if (manifest.get("partition") != partition or manifest.get("protocol_sha256") != protocol_hash
            or manifest.get("protocol") != f"{name}-{partition}"
            or manifest.get("cell_matrix") != matrix
            or manifest.get("frame_skip") != 4 or manifest.get("max_steps") != 2000
            or manifest.get("diagnostic_only", False) != pointer.get("diagnostic_only", False)):
        raise ValueError("selected evaluation manifest/partition does not match frozen protocol")
    audits_path = f"{relative}/determinism.json"
    audits_path_obj = inputs.file(audits_path, expected=audits_path)
    audits = json.loads(audits_path_obj.read_text(encoding="utf-8"))
    candidate = pointer["candidate_id"]
    expected_cells = {(track, seed) for track in ids for seed in seeds}
    if (not isinstance(audits, list) or len(audits) != len(expected_cells)
            or {(r.get("track_id"), r.get("seed")) for r in audits} != expected_cells
            or any(r.get("candidate_id") != candidate or r.get("repeats") != repeats
                   or r.get("audited") is not True or r.get("matches_canonical") is not True
                   for r in audits)):
        raise ValueError("selected evaluation determinism is incomplete or mismatched")
    episodes_path = f"{relative}/episodes.jsonl"
    raw_path = inputs.file(episodes_path, expected=episodes_path)
    seen = set()
    canonical = []
    baseline = {}
    reloaded = {}
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cell = (row.get("track_id"), row.get("seed"))
            repeat = row.get("repeat")
            if (cell not in expected_cells or type(repeat) is not int or repeat not in range(repeats)
                    or (cell, repeat) in seen or row.get("candidate_id") != candidate
                    or row.get("status") != "ok" or row.get("loaded_archive_sha256") != source_sha):
                raise ValueError("unknown actor, seed, partition, repeat or failed selected episode")
            seen.add((cell, repeat))
            outcome = _outcome(row)
            signature = tuple(outcome.get(key) for key in (
                "finished", "progress", "max_progress", "steps", "damage",
                "retire_reason", "action_trace_sha256", "terminal_class"))
            if repeat == 0:
                baseline[cell] = signature
            else:
                reloaded[(cell, repeat)] = signature
            if repeat == 0:
                if outcome["terminal_class"] != ("finished" if outcome["finished"] else
                                                   row.get("termination_class")):
                    raise ValueError("terminal class conflicts with finish flag")
                canonical.append({"actor_sha256": source_sha, "study": name, "partition": partition,
                                  "source_learner_seed": learner, "track_id": cell[0],
                                  "geometry_seed": cell[1], "repeat": 0, **outcome})
    if seen != {(cell, repeat) for cell in expected_cells for repeat in range(repeats)}:
        raise ValueError("selected evaluator episodes are incomplete")
    if any(signature != baseline[cell] for (cell, _), signature in reloaded.items()):
        raise ValueError("reloaded source trajectory does not match its canonical outcome")
    return sorted(canonical, key=lambda r: (r["track_id"], r["geometry_seed"]))


def _teacher(inputs: Inputs, protocol: dict, protocol_hash: str, learner: int,
             source: dict) -> tuple[list[dict], dict]:
    from haic.algorithms.drq_v2.teacher_replay import TeacherDataset

    base = f"{R3_RUN}/teacher-data/learner-{learner}"
    result = inputs.json(f"{base}/collection-result.json", expected=f"{base}/collection-result.json")
    pool = protocol["training_pools"]["teacher_training"]
    ids, seeds, _ = _pool(pool, name="teacher training")
    reserved = set(protocol["reserved_training_seeds"])
    if set(seeds) & reserved or any(seed in protocol["partitions"][part]["seeds"]
                                    for part in ("screen", "confirmation", "blind") for seed in seeds):
        raise ValueError("training pool crosses protected partition")
    if (result.get("phase") != "teacher-data-collection" or result.get("study_id") != R3
            or result.get("study_protocol_sha256") != protocol_hash
            or result.get("learner_seed") != learner
            or result.get("source_actor_sha256") != source["actor_sha256"]
            or result.get("source_checkpoint_sha256") != source["checkpoint_sha256"]
            or result.get("training_pool") != pool):
        raise ValueError("teacher collection is not sealed to the selected source/pool")
    dataset_relative = f"{base}/teacher-dataset.npz"
    dataset_path = inputs.file(result.get("dataset_path"), expected=dataset_relative,
                               sha256=result.get("dataset_file_sha256"))
    dataset = TeacherDataset.from_bytes(dataset_path.read_bytes(), expected_digest=result["dataset_digest"])
    complete = result.get("episode_summaries")
    if not isinstance(complete, list) or len(complete) != len(dataset.episodes) + 1:
        raise ValueError("teacher episode ledger does not match sealed dataset")
    rows = []
    for summary, episode in zip(complete, dataset.episodes):
        seed = summary.get("geometry_seed")
        if (type(seed) is not int or seed not in seeds or summary.get("track_id") not in ids
                or summary.get("complete") is not True or summary != dict(episode.metadata)
                or episode.source_id != f"source-learner-{learner}"
                or episode.source_actor_sha256 != source["actor_sha256"]
                or episode.geometry_id != str(seed) or episode.track_id != summary["track_id"]
                or episode.steps != summary["steps"] or episode.episode_id != str(summary["episode_id"])
                or bool(episode.finished[-1]) != summary["finished"]
                or not np.isclose(float(episode.progress[-1]), summary["progress"], atol=1e-6)):
            raise ValueError("sealed teacher trajectory differs from collection ledger")
        outcome = _outcome({**summary, "max_progress": float(episode.progress.max()),
                            "actions": episode.applied_actions.tolist()})
        rows.append({"actor_sha256": source["actor_sha256"], "study": R3,
                     "partition": "teacher_training", "source_learner_seed": learner,
                     "track_id": summary["track_id"], "geometry_seed": seed,
                     "episode_id": summary["episode_id"], **outcome})
    partial = complete[-1]
    if (partial != result.get("incomplete_episode") or partial.get("complete") is not False
            or partial.get("geometry_seed") not in seeds or partial.get("track_id") not in ids
            or result.get("complete_episode_count") != len(rows)
            or result.get("complete_transition_count") != dataset.transition_count
            or result.get("decisions") != dataset.transition_count + partial.get("steps")
            or result.get("decision_cap") != result.get("decisions")):
        raise ValueError("teacher budget and incomplete episode receipt do not reconcile")
    finished = sorted({r["geometry_seed"] for r in rows if r["finished"]})
    if finished != result.get("finished_geometry_seeds"):
        raise ValueError("teacher finished-road ledger does not match sealed transitions")
    return rows, {"incomplete_budget_interrupted_episode": partial,
                  "complete_episode_count": len(rows), "distinct_finished_road_seeds": finished,
                  "coverage_pass": result.get("coverage_pass")}


def _geometry(consumed_cells: dict[int, int]) -> dict[str, dict]:
    from core.vendor.car_racing import CarRacing

    env = CarRacing(continuous=True, render_mode=None)
    descriptions = {}
    try:
        for seed, track_id in sorted(consumed_cells.items()):
            _integer(seed, "validated consumed road seed")
            _integer(track_id, "validated consumed obstacle variant", low=1)
            env.reset(seed=seed, options={"track_id": track_id})
            track = np.asarray(env.track, dtype="<f8")
            if track.ndim != 2 or track.shape[1] != 4:
                raise ValueError("raw official track has unsupported coordinate layout")
            road_sha = hashlib.sha256(np.ascontiguousarray(track[:, 2:4]).tobytes()).hexdigest()
            descriptor = describe_track(env.track)
            descriptions[str(seed)] = {
                "descriptor_source_track_id": track_id,
                "road_coordinate_sha256": road_sha,
                "signature": descriptor["signature"],
                "signature_sha256": descriptor["signature_sha256"],
                "cyclic_signature_sha256": descriptor.get("cyclic_signature_sha256"),
                "descriptor": descriptor,
                "structure": validate_track_structure(env.track),
            }
    finally:
        env.close()
    return descriptions


def _structural(rows: list[dict], geometry: dict[str, dict]) -> list[dict]:
    # Success/failure labels are grouped at unique road-seed level; obstacle IDs
    # and duplicate source trajectories are correlated observations.
    comparisons = []
    for learner in (0, 1):
        for study, partition in (("pad", "screen"), ("pad", "confirmation"),
                                 ("l2", "confirmation"), (R3, "teacher_training")):
            subset = [r for r in rows if r["source_learner_seed"] == learner
                      and r["study"] == (STUDIES[study][0] if study in STUDIES else study)
                      and r["partition"] == partition]
            outcomes = defaultdict(list)
            for row in subset:
                outcomes[row["geometry_seed"]].append(row["finished"])
            if not outcomes:
                continue
            entry = {"source_learner_seed": learner, "study": study, "partition": partition,
                     "status": "descriptive_hypothesis_not_a_causal_test",
                     "hypothesis": "Earlier/stronger bends, short approach straights, or rapid reversals might coincide with failures; compare unique-road any-finish versus no-finish rows below. Obstacle variants, selection and small samples prevent causal attribution."}
            for label, selected in (("any_finish", [s for s, flags in outcomes.items() if any(flags)]),
                                    ("no_finish", [s for s, flags in outcomes.items() if not any(flags)])):
                entry[label] = {"road_seeds": sorted(selected), "count": len(selected)}
                for metric in ("abs_curvature_p90_per_m", "sharp_entry_count", "rapid_reversals_65m",
                               "longest_straight_before_sharp_m", "perimeter_m", "early_bend_count",
                               "mid_bend_count", "mid_reversals_65m"):
                    values = [geometry[str(seed)]["descriptor"].get(metric) for seed in selected]
                    values = [value for value in values if value is not None]
                    entry[label][f"mean_{metric}"] = sum(values) / len(values) if values else None
                early = [geometry[str(seed)]["descriptor"].get("early_strongest_bend") for seed in selected]
                early = [event for event in early if event is not None]
                entry[label]["early_strongest_bend_direction_counts"] = dict(sorted(
                    Counter(event["direction"] for event in early).items()))
                for metric in ("start_m", "turn_angle_rad", "entry_straight_since_spawn_m"):
                    values = [abs(event[metric]) if metric == "turn_angle_rad" else event[metric]
                              for event in early]
                    entry[label][f"early_strongest_bend_mean_{metric}"] = (
                        sum(values) / len(values) if values else None)
            comparisons.append(entry)
    return comparisons


def _prefix(inputs: Inputs, source_sha: str, forbidden: set[int]) -> tuple[list[dict], dict]:
    protocol_path = f"experiments/{PREFIX}.json"
    protocol = inputs.json(protocol_path, expected=protocol_path)
    if inputs.hashes[protocol_path] != FROZEN_PROTOCOL_SHA256[PREFIX]:
        raise ValueError("consumed prefix protocol SHA-256 differs from frozen source")
    if (protocol.get("name") != PREFIX or protocol.get("base_driver", {}).get("actor_sha256") != source_sha
            or protocol.get("environment", {}).get("partition", "").startswith("training-only") is False):
        raise ValueError("prefix-branch protocol is not consumed source-only training")
    env = protocol["environment"]
    ids, seeds = env.get("track_ids"), env.get("geometry_seeds")
    if (not isinstance(ids, list) or not ids or not isinstance(seeds, list) or not seeds
            or any(type(track) is not int or track <= 0 for track in ids)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds)
            or set(seeds) & forbidden):
        raise ValueError("prefix source cells cross reserved/unknown road seeds")
    manifest_path = f"{PREFIX_RUN}/manifest.completed.json"
    manifest = inputs.json(manifest_path, expected=manifest_path)
    if (manifest.get("protocol_sha256") != inputs.hashes[protocol_path]
            or manifest.get("base_actor_sha256") != source_sha
            or manifest.get("status") != "completed_training_only_single_intervention_diagnostic"):
        raise ValueError("prefix source manifest is not sealed to the source actor")
    permitted = {(track, seed) for track in ids for seed in seeds}
    source_path = f"{PREFIX_RUN}/source-trajectories.jsonl"
    rows = []
    seen = set()
    with inputs.file(source_path, expected=source_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cell = (row.get("track_id"), row.get("geometry_seed"))
            if cell not in permitted or cell in seen:
                raise ValueError("unknown or repeated training-only prefix source cell")
            seen.add(cell)
            signatures = row.get("step_signatures")
            outcome = row.get("outcome")
            if not isinstance(outcome, dict) or not isinstance(signatures, list):
                raise ValueError("prefix source has no complete decision/outcome trace")
            reported = _outcome({**outcome, "steps": outcome.get("decisions"),
                                 "actions": row.get("official_actions"),
                                 "max_progress": max((signature["info"]["progress"] for signature in signatures),
                                                     default=None)})
            if len(signatures) != reported["steps"]:
                raise ValueError("prefix source action/state/outcome lengths differ")
            last_state = signatures[-1].get("state", {})
            location = last_state.get("position")
            if (not isinstance(location, list) or len(location) != 2
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in location)):
                location = None
            rows.append({"actor_sha256": source_sha, "study": PREFIX, "partition": "training_only_prefix",
                         "source_learner_seed": 1, "track_id": cell[0], "geometry_seed": cell[1],
                         "terminal_position_xy_when_available": location,
                         "off_track_counter_when_available": last_state.get("off_track_counter"), **reported})
    if seen != permitted:
        raise ValueError("prefix source training-only cell matrix is incomplete")
    branches_path = f"{PREFIX_RUN}/branch-trajectories.jsonl"
    branch_counts = Counter()
    contexts = set()
    with inputs.file(branches_path, expected=branches_path).open(encoding="utf-8") as handle:
        for line in handle:
            branch = json.loads(line)
            match = re.fullmatch(r"id([0-9]+)-seed([0-9]+)-after([0-9]+)", branch.get("cell_key", ""))
            if (match is None or (int(match[1]), int(match[2])) not in permitted
                    or branch.get("head") not in ("v1", "v2")
                    or branch.get("branch_after_decisions") != int(match[3])
                    or int(match[3]) not in protocol["procedure"]["predeclared_branch_points_after_decisions"]):
                raise ValueError("branch trace uses an unconsumed prefix cell/context")
            key = (branch["cell_key"], branch["head"])
            if key in contexts:
                raise ValueError("duplicate intervention context")
            contexts.add(key)
            branch_counts[(branch["head"], branch["status"])] += 1
    return rows, {
        "baseline_only": aggregate(rows),
        "intervention_contexts_not_independent_roads": len(contexts),
        "branch_status_by_head": [{"head": head, "status": status, "count": count}
                                  for (head, status), count in sorted(branch_counts.items())],
        "limitation": "Branches change the policy after a fixed prefix, so their outcomes are not baseline or generalization episodes; only source baseline rows enter geometry comparisons.",
    }


def analyze(root: Path, *, describe_geometry: bool = False, include_prefix: bool = False) -> dict:
    """Validate selected frozen inputs before reading episode data; never traverse directories."""
    inputs = Inputs(root)
    r3, r3_hash = _protocol(inputs, R3)
    sources = r3.get("source_actors")
    if not isinstance(sources, list) or len(sources) != 2 or {s.get("learner_seed") for s in sources} != {0, 1}:
        raise ValueError("r3 must pin exactly the two selected pad-4 source actors")
    sources = sorted(sources, key=lambda s: s["learner_seed"])
    for learner, source in enumerate(sources):
        if (not SHA.fullmatch(source.get("actor_sha256", ""))
                or source["actor_sha256"] != FROZEN_SOURCE_SHA256[learner]
                or source.get("actor_path") != f"runs/{STUDIES['pad'][1]}/control-seed{learner}/checkpoints/step-000131072/actor.pt"):
            raise ValueError("r3 source actor is not the selected pad-4 control")
    protocols = {study: _protocol(inputs, name) for study, (name, _) in STUDIES.items()}
    rows = []
    cohorts = {}
    for learner, source in enumerate(sources):
        actor = source["actor_sha256"]
        for study, (name, run) in STUDIES.items():
            protocol, protocol_hash = protocols[study]
            base = f"runs/{run}/control-seed{learner}"
            selection = inputs.json(f"{base}/selection.json", expected=f"{base}/selection.json")
            confirmation = inputs.json(f"{base}/confirmation.json", expected=f"{base}/confirmation.json")
            selected = selection.get("cpu_result")
            ranked = confirmation.get("ranked")
            if (selection.get("selected") is not True or selection.get("protocol_sha256") != protocol_hash
                    or selection.get("actor_sha256") != actor or not isinstance(selected, dict)
                    or selected.get("archive_sha256") != actor or selected.get("algorithm") != "drq-v2"
                    or selected.get("export_metadata", {}).get("config", {}).get("augmentation_pad") != 4
                    or selected.get("eligible") is not True or selected.get("cpu_reload_matches") is not True
                    or not isinstance(ranked, list) or len(ranked) != 1
                    or ranked[0].get("archive_sha256") != actor or ranked[0].get("candidate_id") != selected.get("candidate_id")
                    or confirmation.get("partition") != "confirmation"
                    or confirmation.get("protocol_sha256") != protocol_hash):
                raise ValueError("selected screen/confirmation actor identity is not frozen pad-4 source")
            for partition, pointer in (("screen", {**selected, "evaluation_dir": selection.get("evaluation_dir"),
                                                  "diagnostic_only": False}),
                                       ("confirmation", {**ranked[0], "evaluation_dir": confirmation.get("evaluation_dir"),
                                                         "diagnostic_only": confirmation.get("diagnostic_only", False)})):
                group = _evaluation(inputs, protocol, protocol_hash, name, run, learner, actor, partition, pointer)
                rows.extend(group)
                cohorts[f"{study}-source{learner}-{partition}"] = aggregate(group)
        teacher_rows, provenance = _teacher(inputs, r3, r3_hash, learner, source)
        rows.extend(teacher_rows)
        cohorts[f"r3-source{learner}-teacher_training"] = {**aggregate(teacher_rows), **provenance}
    safe_seeds = {r["geometry_seed"] for r in rows}
    forbidden = set()
    for protocol, _ in [(r3, r3_hash), *protocols.values()]:
        forbidden.update(protocol["partitions"]["blind"]["seeds"])
    if safe_seeds & forbidden:
        raise ValueError("blind geometry seed found in consumed source episodes")
    prefix = None
    if include_prefix:
        prefix_rows, prefix = _prefix(inputs, sources[1]["actor_sha256"], forbidden)
        rows.extend(prefix_rows)
        safe_seeds.update(row["geometry_seed"] for row in prefix_rows)
    consumed_cells = {}
    for row in rows:
        seed = row["geometry_seed"]
        consumed_cells[seed] = min(consumed_cells.get(seed, row["track_id"]), row["track_id"])
    geometry = _geometry(consumed_cells) if describe_geometry else {}
    return {
        "format": "haic-drq-consumed-geometry-failures-v1", "status": "complete",
        "inputs_sha256_exact_allowlist": dict(sorted(inputs.hashes.items())),
        "source_actors": [{"learner_seed": i, "actor_sha256": s["actor_sha256"]} for i, s in enumerate(sources)],
        "cohorts": cohorts, "overall": aggregate(rows),
        "optional_training_only_prefix": prefix,
        "road_geometry": geometry if describe_geometry else None,
        "structural_hypotheses_vs_success": _structural(rows, geometry) if describe_geometry else None,
        "limitations": [
            "Only repeat 0 counts as an outcome; repeat 1 is a determinism check, not another independent road.",
            "Road seed identifies geometry; track ID changes obstacles, not independent centerlines. Repeated identical source trajectories and pad/L2 screen reuse are correlated.",
            "Evaluation episodes record terminal visited-tile progress only: max progress and per-step pose are null unless the particular artifact provides them. Progress is not directed finish.",
            "Teacher maximum visited-tile progress is available only for sealed complete r3 collection episodes; budget-interrupted partial episodes are excluded.",
            "off_track means more than 100 consecutive wrapper decisions whose aggregate raw reward is negative; crash retirement takes priority. It does not locate a physical road edge.",
            "Progress >=0.9 without finished=true is NOT a completed lap: finish requires qualification >=0.95 and a valid directed finish-line crossing.",
            "Failure-location bins are terminal visited-tile coverage, not longitudinal position or a mapped failure point. Static descriptors cannot prove any driving mechanism.",
            "r3 is training-only, pad/L2 screen reuse is development, and confirmation is already consumed; comparisons across unmatched cohorts are descriptive, not matched performance claims.",
            "Raw road shape reset is opt-in and restricted to observed (seed, track ID) source cells; no blind geometry/outcome is opened, reset, or analyzed.",
            "Source actor SHA identities are bound to frozen protocol and selected receipts; actor binaries are not opened or independently rehashed.",
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="Root containing the fixed consumed artifacts")
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence file; never overwrite an artifact")
    parser.add_argument("--describe-geometry", action="store_true",
                        help="Reset raw official roads ONLY for validated consumed source seeds")
    parser.add_argument("--include-prefix", action="store_true",
                        help="Read only consumed training-only source/branch prefix traces")
    args = parser.parse_args(argv)
    if any("blind" in part.lower() for part in args.output.parts):
        parser.error("output must not be a blind path")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("output must be a new file in an existing directory")
    report = analyze(args.repo_root, describe_geometry=args.describe_geometry,
                     include_prefix=args.include_prefix)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
