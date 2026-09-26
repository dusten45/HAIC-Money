"""Read-only paired r6 TRAIN-DIAGNOSTIC outcome and action-trace analysis.

This analyzes existing canonical repeat-0 traces. It never resets an environment
or loads model weights, and its divergence is closed-loop, not a same-state test.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("experiments/drqv2-geometry-mix-v1-r6.json")
RUN = Path("runs/20260925-drqv2-geometry-mix-v1-r6")
PROTOCOL_SHA = "d257d242349f01ebf3ae8b1b741de7edc16d20c4523444a125928ab6aec2d5ab"
CATALOG_SHA = "a8cbe5d2445563f9065a944254f6410486135b20eb8574ec1cf9eb611ebbf00b"
MANIFEST_SHA = "fb14fe9eab14f61cdecc45453b69e44d34fdb967368c44c255e827292804b380"
EPISODES_SHA = "aee2ae9b85172feb0e9a09697bdd92a34b29631cfc094586364f7c0b9f283441"
VARIANTS = ("uniform", "failure_weighted", "easy_retention")
KINDS = ("lost", "gained", "both_finished", "both_failed")
# Fixed prior to inspecting paired outcomes. This is an action-scale descriptor,
# not the first causal branch or a training/evaluation selection criterion.
ACTION_THRESHOLD = 0.10
PERSISTENCE = 3
HISTORY = 10


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def first_divergence(source: np.ndarray, candidate: np.ndarray) -> dict:
    n = min(len(source), len(candidate))
    difference = np.max(np.abs(source[:n] - candidate[:n]), axis=1)
    strict = np.flatnonzero(difference > 1e-6)
    sustained = next(
        (i for i in range(n - PERSISTENCE + 1)
         if np.all(difference[i:i + PERSISTENCE] >= ACTION_THRESHOLD)),
        None,
    )
    return {
        "first_nontrivial_action_decision": int(strict[0]) + 1 if len(strict) else None,
        "first_sustained_action_decision": sustained + 1 if sustained is not None else None,
        "initial_native_linf_difference": float(difference[0]) if n else None,
        "overlap_decisions": n,
    }


def classification(source_finished: bool, candidate_finished: bool) -> str:
    if source_finished:
        return "both_finished" if candidate_finished else "lost"
    return "gained" if candidate_finished else "both_failed"


def _snapshot(trace: dict, index: int) -> dict:
    if index < 0 or index >= len(trace["reward"]):
        return {"available": False}
    road_indices = np.flatnonzero(trace["sparse_step"] <= index + 1)
    road_index = int(road_indices[-1]) if len(road_indices) else 0
    return {
        "available": True, "decision": index + 1,
        "native_action": trace["native_action"][index].tolist(),
        "official_action": trace["official_action"][index].tolist(),
        "reward": float(trace["reward"][index]),
        "progress": float(trace["progress"][index]),
        "damage": float(trace["damage"][index]),
        "off_track_counter": int(trace["off_track_counter"][index]),
        "collision": bool(trace["collision"][index]),
        "terminal": bool(trace["terminal"][index]),
        "finished": bool(trace["finished"][index]),
        "road_last_sample": {
            "decision": int(trace["sparse_step"][road_index]),
            "nearest_point_fraction": float(trace["sparse_nearest_point_fraction"][road_index]),
            "nearest_distance_m": float(trace["sparse_nearest_distance_m"][road_index]),
            "speed_m_s": float(trace["sparse_speed_m_s"][road_index]),
        },
    }


def analyze_pair(source: dict, candidate: dict, source_trace: dict, candidate_trace: dict) -> dict:
    divergence = first_divergence(source_trace["native_action"], candidate_trace["native_action"])
    t = divergence["first_sustained_action_decision"]
    anchor = (t - 1) if t is not None else None
    history_start = max(0, anchor - HISTORY) if anchor is not None else None
    samples = {}
    if anchor is not None:
        for offset in (-10, -1, 0, 1, 10):
            samples[str(offset)] = {
                "source": _snapshot(source_trace, anchor + offset),
                "candidate": _snapshot(candidate_trace, anchor + offset),
            }
    result = {
        "source_seed": source["source_learner_seed"],
        "geometry_seed": source["geometry_seed"],
        "track_id": 1,
        "family": source["family"],
        "variant": candidate["arm"],
        "transition": classification(source["finished"], candidate["finished"]),
        "source_finished": source["finished"],
        "candidate_finished": candidate["finished"],
        "source_trace_sha256": source["trace_sha256"],
        "candidate_trace_sha256": candidate["trace_sha256"],
        "source_steps": source["steps"],
        "candidate_steps": candidate["steps"],
        "terminal_decision_difference_candidate_minus_source": candidate["steps"] - source["steps"],
        "source_max_progress": source["max_progress"],
        "candidate_max_progress": candidate["max_progress"],
        "max_progress_difference_candidate_minus_source": candidate["max_progress"] - source["max_progress"],
        "source_raw_reward": source["raw_reward_sum"],
        "candidate_raw_reward": candidate["raw_reward_sum"],
        "source_terminal_damage": source["terminal_damage"],
        "candidate_terminal_damage": candidate["terminal_damage"],
        "source_retirement": source["termination_class"],
        "candidate_retirement": candidate["termination_class"],
        "divergence": divergence,
        "aligned_trace_samples": samples,
        "history_before_divergence": {
            "decision_start": history_start + 1 if history_start is not None else None,
            "source_native_action": source_trace["native_action"][history_start:anchor].tolist() if anchor is not None else [],
            "candidate_native_action": candidate_trace["native_action"][history_start:anchor].tolist() if anchor is not None else [],
        },
    }
    if anchor is not None:
        result["source_progress_at_divergence"] = float(source_trace["progress"][anchor])
        result["candidate_progress_at_divergence"] = float(candidate_trace["progress"][anchor])
    return result


def analyze(root: Path) -> dict:
    protocol_path = root / PROTOCOL
    assert digest(protocol_path) == PROTOCOL_SHA, "protocol hash changed"
    protocol = json.loads(protocol_path.read_text())
    assert digest(root / protocol["catalog_path"]) == CATALOG_SHA == protocol["catalog_sha256"]
    diagnostic = root / RUN / "train-diagnostic"
    manifest_path = diagnostic / "manifest.json"
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest_sha == MANIFEST_SHA, "published diagnostic manifest changed"
    assert manifest["protocol_sha256"] == PROTOCOL_SHA
    assert manifest["catalog"]["catalog_sha256"] == CATALOG_SHA
    assert manifest["role"] == "TRAIN-DIAGNOSTIC" and manifest["ranked"] is False
    assert (diagnostic / "manifest.sha256").read_text().split()[0] == manifest_sha
    episodes_path = diagnostic / "episodes.jsonl"
    assert digest(episodes_path) == EPISODES_SHA == manifest["files_sha256"]["episodes.jsonl"]
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    assert len(rows) == 256
    canonical = {(row["source_learner_seed"], row["geometry_seed"], row["arm"]): row
                 for row in rows if row["repeat"] == 0}
    assert len(canonical) == 128
    assert set(protocol["diagnostics"]["geometry_seeds"]) == {key[1] for key in canonical}
    trace_cache = {}

    def trace(row: dict) -> dict:
        relative = row["trace_path"]
        path = (diagnostic / relative).resolve(strict=True)
        assert relative in manifest["files_sha256"]
        assert path.relative_to((diagnostic / "traces").resolve()).parts
        assert not path.is_symlink() and path.is_file()
        assert digest(path) == row["trace_sha256"] == manifest["files_sha256"][relative], path
        if relative not in trace_cache:
            with np.load(path, allow_pickle=False) as arrays:
                trace_cache[relative] = {key: arrays[key].copy() for key in arrays.files}
        return trace_cache[relative]

    pairs = []
    by_family = defaultdict(Counter)
    by_variant = defaultdict(Counter)
    for seed in (0, 1):
        for road in sorted(protocol["diagnostics"]["geometry_seeds"]):
            source = canonical[(seed, road, "unchanged-source")]
            for variant in VARIANTS:
                candidate = canonical[(seed, road, variant)]
                assert candidate["family"] == source["family"] and source["track_id"] == candidate["track_id"] == 1
                assert candidate["source_actor_sha256"] == source["actor_sha256"]
                pair = analyze_pair(source, candidate, trace(source), trace(candidate))
                pairs.append(pair)
                by_family[(variant, source["family"])][pair["transition"]] += 1
                by_variant[variant][pair["transition"]] += 1
    assert len(pairs) == 96
    return {
        "scope": "reused TRAIN-DIAGNOSTIC canonical repeat 0; descriptive, unranked, not fresh or official",
        "protocol_sha256": PROTOCOL_SHA, "catalog_sha256": CATALOG_SHA,
        "diagnostic_manifest_sha256": manifest_sha,
        "episodes_sha256": manifest["files_sha256"]["episodes.jsonl"],
        "definition": {"action": "native action L-infinity difference >= 0.10 for three consecutive overlapping decisions",
                       "threshold": ACTION_THRESHOLD, "persistence": PERSISTENCE, "history_decisions": HISTORY,
                       "caution": "After different actions, same-index closed-loop trajectories do not represent identical observations."},
        "variant_transitions": {variant: {kind: by_variant[variant][kind] for kind in KINDS} for variant in VARIANTS},
        "family_transitions": [
            {"variant": variant, "family": family, "n": sum(by_family[(variant, family)].values()),
             **{kind: by_family[(variant, family)][kind] for kind in KINDS}}
            for variant in VARIANTS for family in sorted({pair["family"] for pair in pairs})
        ],
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(ROOT)
    output = args.output.resolve()
    output.relative_to((ROOT / RUN).resolve())
    assert not output.exists(), "do not overwrite an existing analysis artifact"
    assert output.parent.is_dir() and not output.parent.is_symlink()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"variant_transitions": result["variant_transitions"],
                      "output": str(output), "sha256": digest(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
