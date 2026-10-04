"""Read-only, no-reset H3-to-H5 RAW100k reward-target sufficiency audit.

Run from the repository root with ``python -m scripts.audit_tdmpc2_reward_overshoot_targets``.
The SHA-bound cursor helper verifies the complete frozen source and checkpoint
bytes; this module never deserializes the checkpoint or opens an environment.
Counts are for the ORIGINAL completed replay, not a future online sample mix.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path

from scripts import diagnose_tdmpc2_h5_logged as bound
from scripts import diagnose_tdmpc2_checkpoint_losses as cursor


ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/audit_tdmpc2_reward_overshoot_targets.py"
BOUND_SOURCE = "scripts/diagnose_tdmpc2_h5_logged.py"
BOUND_SOURCE_SHA256 = "ab266fd101dc6ad3584ebba835c6449b6a64ab82bd3822a93b782b5806c9df13"
MIN_EXTENSION_FRACTION = 0.95
MIN_ROADS_WITH_RANGE = 3
MIN_SUFFIX_RANGE_EXCLUSIVE = 0.1


def _input_hashes(root: Path) -> dict[str, str]:
    paths = (set(cursor.SOURCE_PATHS) |
             {SELF, BOUND_SOURCE, bound.HELPER, cursor.PROTOCOL,
              f"{bound.RUN}/result.json", f"{bound.RUN}/training.jsonl",
              f"{bound.RUN}/steps.jsonl", bound.CHECKPOINT})
    return {name: bound._digest(bound._file(root, name)) for name in sorted(paths)}


def _audit_steps(episodes: list[dict], path: Path) -> dict:
    """Check every transition and count contained starts using O(1) window memory."""
    roads = {str(seed): {"episodes": 0, "h3_eligible_starts": 0,
                         "h5_extended_starts": 0, "suffix_min": None,
                         "suffix_max": None} for seed in bound.ROADS}
    decisions = 0
    boundaries = {"terminated": 0, "finished_truncation": 0, "ordinary_timeout": 0}
    with path.open("rb") as stream:
        for eid, ep in enumerate(episodes):
            length = ep.get("length")
            seed = bound.ROADS[eid % len(bound.ROADS)]
            if (type(length) is not int or not 1 <= length <= 2000
                    or ep.get("episode") != eid or ep.get("track_id") != 1
                    or ep.get("geometry_seed") != seed
                    or ep.get("decisions") != decisions + length
                    or type(ep.get("finished")) is not bool):
                raise ValueError("invalid completed TRAIN episode boundary")
            road = roads[str(seed)]
            road["episodes"] += 1
            h3, extended = max(0, length - 2), max(0, length - 4)
            road["h3_eligible_starts"] += h3
            road["h5_extended_starts"] += extended
            recent: deque[float] = deque(maxlen=5)
            for offset in range(length):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete step ledger")
                row = bound._json(line)
                decisions += 1
                reward = row.get("reward")
                if (row.get("decision") != decisions or row.get("episode") != eid
                        or row.get("track_id") != 1 or row.get("geometry_seed") != seed
                        or any(type(row.get(flag)) is not bool
                               for flag in ("terminated", "truncated", "terminal"))):
                    raise ValueError("step identity or flags invalid")
                if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                    raise ValueError("raw reward must be numeric")
                raw_reward = float(reward)
                if not math.isfinite(raw_reward):
                    raise ValueError("nonfinite raw reward")
                terminated, truncated, terminal = (row[k] for k in ("terminated", "truncated", "terminal"))
                if (terminated and not terminal or terminal and not (terminated or truncated)
                        or offset < length - 1 and (terminated or truncated or terminal)):
                    raise ValueError("step crosses an episode boundary")
                if offset == length - 1:
                    if (not (terminated or truncated)
                            or any(row[k] is not ep.get(k) for k in ("terminated", "truncated", "terminal"))
                            or ep["finished"] != (truncated and terminal and not terminated)):
                        raise ValueError("final step disagrees with episode end/finish")
                    category = ("terminated" if terminated else
                                "finished_truncation" if terminal else "ordinary_timeout")
                    boundaries[category] += 1
                recent.append(raw_reward)
                if offset >= 4:
                    suffix = recent[-2] + bound.GAMMA * recent[-1]
                    if not math.isfinite(suffix):
                        raise ValueError("nonfinite discounted suffix")
                    road["suffix_min"] = suffix if road["suffix_min"] is None else min(road["suffix_min"], suffix)
                    road["suffix_max"] = suffix if road["suffix_max"] is None else max(road["suffix_max"], suffix)
            if len(recent) != min(length, 5):
                raise ValueError("incomplete episode window")
        if stream.read(1):
            raise ValueError("step ledger extends beyond final episode")
    if decisions != sum(ep["length"] for ep in episodes):
        raise ValueError("step and episode cursor mismatch")
    for road in roads.values():
        road["extension_fraction"] = (road["h5_extended_starts"] / road["h3_eligible_starts"]
                                      if road["h3_eligible_starts"] else None)
        road["suffix_range"] = (road["suffix_max"] - road["suffix_min"]
                                if road["suffix_min"] is not None else None)
    return {"step_rows": decisions, "episode_boundaries": len(episodes),
            "boundary_types": boundaries, "roads": roads}


def _gate(roads: dict) -> dict:
    eligible = sum(r["h3_eligible_starts"] for r in roads.values())
    extended = sum(r["h5_extended_starts"] for r in roads.values())
    informative = sum(r["suffix_range"] is not None and r["suffix_range"] > MIN_SUFFIX_RANGE_EXCLUSIVE
                      for r in roads.values())
    coverage = bool(eligible and extended * 100 >= eligible * 95)
    return {"h3_eligible_starts": eligible, "h5_extended_starts": extended,
            "extension_fraction": extended / eligible if eligible else None,
            "roads_with_suffix_range_gt_0_1": informative,
            "gate": {"minimum_extension_fraction": MIN_EXTENSION_FRACTION,
                     "minimum_informative_roads": MIN_ROADS_WITH_RANGE,
                     "suffix_range_strictly_above": MIN_SUFFIX_RANGE_EXCLUSIVE,
                     "extension_coverage_passed": coverage,
                     "road_range_passed": informative >= MIN_ROADS_WITH_RANGE,
                     "passed": coverage and informative >= MIN_ROADS_WITH_RANGE}}


def audit(*, root: Path = ROOT) -> dict:
    root = root.resolve(strict=True)
    before = _input_hashes(root)
    if (before[BOUND_SOURCE] != BOUND_SOURCE_SHA256
            or before[cursor.PROTOCOL] != bound.SOURCE_PROTOCOL_SHA
            or before[f"{bound.RUN}/result.json"] != bound.RESULT_SHA
            or before[f"{bound.RUN}/training.jsonl"] != bound.TRAIN_SHA
            or before[f"{bound.RUN}/steps.jsonl"] != bound.STEPS_SHA
            or before[bound.CHECKPOINT] != bound.CHECKPOINT_SHA):
        raise ValueError("pinned source/protocol/result/checkpoint/ledger SHA mismatch before audit")
    _, _, context = bound._prepared(root)  # Full source map, result, cursor, step and checkpoint checks.
    if any(before[name] != digest for name, digest in context["original_sources"].items()):
        raise ValueError("original producer source changed before audit")
    steps = _audit_steps(context["episodes"], bound._file(root, f"{bound.RUN}/steps.jsonl"))
    if (steps["step_rows"] != bound.FINAL_DECISIONS
            or steps["episode_boundaries"] != bound.FINAL_EPISODES
            or steps["boundary_types"]["terminated"] + steps["boundary_types"]["finished_truncation"]
               + steps["boundary_types"]["ordinary_timeout"] != bound.FINAL_EPISODES
            or context["pin"]["row"]["decisions"] != steps["step_rows"]
            or context["pin"]["row"]["episodes"] != steps["episode_boundaries"]
            or context["pin"]["step_prefix_sha256"] != before[f"{bound.RUN}/steps.jsonl"]
            or context["pin"]["ledger_prefix_sha256"] != before[f"{bound.RUN}/training.jsonl"]):
        raise ValueError("full RAW100k cursor or episode boundary mismatch")
    after = _input_hashes(root)
    if after != before:
        raise ValueError("source, checkpoint or ledger changed during audit")
    body = {"format": "haic-tdmpc2-reward-overshoot-target-audit-v1",
            "scope": "original_RAW100k_completed_replay_ONLY_consumed_TRAIN_not_performance_or_training_clearance",
            "environment_resets": 0, "torch_load_calls": 0, "optimizer_steps": 0,
            "source_protocol_sha256": bound.SOURCE_PROTOCOL_SHA, "source_result_sha256": bound.RESULT_SHA,
            "source_checkpoint_sha256": bound.CHECKPOINT_SHA,
            "source_sha256": {"producer": context["original_sources"], "cursor_helper": before[bound.HELPER],
                              "h5_diagnostic": before[BOUND_SOURCE], "audit": before[SELF]},
            "input_sha256_before": before, "input_sha256_after": after,
            "step_rows": steps["step_rows"], "episode_boundaries": steps["episode_boundaries"],
            "boundary_types": steps["boundary_types"], "track_ids": [1], "roads": steps["roads"],
            **_gate(steps["roads"])}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {**body, "body_sha256": hashlib.sha256(canonical).hexdigest()}


def main() -> None:
    print(json.dumps(audit(root=ROOT), sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
