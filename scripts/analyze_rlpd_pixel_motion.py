"""Offline pixel-only score on ALL frozen G0 TRAIN decisions; never drives or trains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from haic.algorithms.rlpd.visual_motion import visual_motion_score


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "protocol_path": "experiments/rlpd-g0-completion-v1.json",
    "protocol_sha256": "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b",
    "manifest_path": "runs/20260926-rlpd-g0-completion-v1/manifest.json",
    "manifest_sha256": "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358",
    "cells_path": "runs/20260926-rlpd-g0-completion-v1/cells.jsonl",
    "cells_sha256": "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559",
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ACTOR_ID = re.compile(r"[a-z0-9-]{1,48}\Z")
ACTION_KEYS = (
    "proposed_native_action", "executed_native_action",
    "commanded_official_action", "raw_official_action",
)
TRACE_KEYS = (
    "initial_stack", "observation_frame", "next_frame", "speed_m_s",
    "new_tiles", "contact", "finished", "terminated", "truncated",
    "raw_frames", "summed_reward", "progress", *ACTION_KEYS,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rule_template(source: dict[str, str] = SOURCE) -> dict:
    """Print for external review; never creates or freezes the rule file."""
    return {
        "format": "haic-rlpd-pixel-motion-rule-v1",
        "status": "frozen",
        "mode": "TRAIN-only-retrospective-diagnostic",
        "source": dict(source),
        "source_hashes": {
            "haic/algorithms/rlpd/visual_motion.py": sha256_file(ROOT / "haic/algorithms/rlpd/visual_motion.py"),
            "scripts/analyze_rlpd_pixel_motion.py": sha256_file(Path(__file__).resolve()),
        },
        "score": {
            "input": "four float32 [0,1] CHW 84x84 frames, or the exact uint8 stack divided by 255",
            "definition": "sum(abs(frame[k+1]-frame[k]) for k=0..2, all 84x84 pixels)/(3*84*84)",
            "learned_threshold": None,
        },
        "diagnostic_label": {
            "role": "retrospective-telemetry-only-not-a-score-or-policy-input",
            "definition": "speed_m_s <= 2 AND at least 5 consecutive new_tiles == 0 decisions including current AND contact in prior 20 decisions of same episode (current excluded)",
            "speed_m_s_max": 2,
            "no_new_tiles_min_consecutive": 5,
            "prior_contact_decisions": 20,
        },
    }


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys,
                       parse_constant=_reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file(root: Path, relative: str, prefix: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("unsafe repository-relative input path")
    name = Path(relative)
    if (name.is_absolute() or len(name.parts) < 2 or name.parts[0] != prefix
            or name.as_posix() != relative or any(part in (".", "..") for part in name.parts)):
        raise ValueError(f"input must be a repository-relative {prefix}/ file")
    path = root
    for part in name.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"symlink in frozen input: {relative}")
    if not path.is_file():
        raise ValueError(f"missing frozen input: {relative}")
    return path


def _pinned(root: Path, relative: str, sha: str, prefix: str) -> Path:
    if not isinstance(sha, str) or SHA256.fullmatch(sha) is None:
        raise ValueError("expected 64-digit lowercase SHA-256")
    path = _file(root, relative, prefix)
    if sha256_file(path) != sha:
        raise ValueError(f"frozen hash mismatch: {relative}")
    return path


def contact_stall_labels(speed: np.ndarray, new_tiles: np.ndarray, contact: np.ndarray) -> list[bool]:
    """Retrospective label only; a contact on the current decision does not count."""
    if (speed.ndim != new_tiles.ndim or speed.ndim != contact.ndim or speed.ndim != 1
            or not (len(speed) == len(new_tiles) == len(contact))):
        raise ValueError("diagnostic label arrays must share a one-dimensional decision axis")
    streak = 0
    labels = []
    for step in range(len(speed)):
        streak = streak + 1 if new_tiles[step] == 0 else 0
        labels.append(bool(speed[step] <= 2 and streak >= 5
                           and np.any(contact[max(0, step - 20):step])))
    return labels


def _trace(root: Path, relative: str, sha: str, row: dict) -> tuple[list[float], list[bool]]:
    path = _pinned(root, relative, sha, "runs")
    with np.load(path, allow_pickle=False) as archive:
        if not set(TRACE_KEYS).issubset(archive.files) or len(set(archive.files)) != len(archive.files):
            raise ValueError(f"G0 trace has missing or duplicate columns: {relative}")
        arrays = {key: archive[key] for key in TRACE_KEYS}
    steps = row["steps"]
    initial = arrays["initial_stack"]
    if initial.dtype != np.uint8 or initial.shape != (4, 84, 84):
        raise ValueError("malformed initial_stack")
    for key in ("observation_frame", "next_frame"):
        if arrays[key].dtype != np.uint8 or arrays[key].shape != (steps, 84, 84):
            raise ValueError(f"malformed pixel column: {key}")
    for key in ACTION_KEYS:
        value = arrays[key]
        if value.shape != (steps, 3) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"missing or malformed G0 action: {key}")
    if (not np.allclose(arrays["proposed_native_action"], arrays["executed_native_action"], rtol=0, atol=1e-6)
            or not np.allclose(arrays["commanded_official_action"], arrays["raw_official_action"], rtol=0, atol=1e-6)):
        raise ValueError("G0 action streams disagree")
    for key in ("contact", "finished", "terminated", "truncated"):
        if arrays[key].shape != (steps,) or arrays[key].dtype.kind != "b":
            raise ValueError(f"malformed boolean column: {key}")
    for key in ("new_tiles", "raw_frames"):
        if arrays[key].shape != (steps,) or arrays[key].dtype.kind not in "iu":
            raise ValueError(f"malformed integer column: {key}")
    for key in ("speed_m_s", "summed_reward", "progress"):
        if (arrays[key].shape != (steps,) or arrays[key].dtype.kind != "f"
                or not np.isfinite(arrays[key]).all()):
            raise ValueError(f"malformed finite telemetry: {key}")
    if (np.any(arrays["new_tiles"] < 0) or np.any(arrays["speed_m_s"] < 0)
            or np.any((arrays["raw_frames"] < 1) | (arrays["raw_frames"] > 4))
            or np.any((arrays["progress"] < 0) | (arrays["progress"] > 1))
            or np.any(np.diff(arrays["progress"]) < -1e-6)
            or int(arrays["raw_frames"].sum()) != row["driven_raw_frames"]
            or not np.isclose(arrays["summed_reward"].sum(), row["raw_reward_sum"], atol=1e-6, rtol=0)
            or not np.isclose(arrays["progress"].max(), row["max_progress"], atol=1e-8, rtol=0)):
        raise ValueError("G0 trace/ledger bounds or counters disagree")
    if (np.any(arrays["finished"][:-1] | arrays["terminated"][:-1] | arrays["truncated"][:-1])
            or not (arrays["terminated"][-1] or arrays["truncated"][-1])
            or bool(arrays["finished"][-1]) != (row["summary"]["outcome"] == "finished")):
        raise ValueError("G0 terminal decision/outcome disagree")

    stack = initial.copy()
    scores = []
    for step in range(steps):
        if not np.array_equal(stack[-1], arrays["observation_frame"][step]):
            raise ValueError("G0 observation_frame differs from reconstructed stack")
        normalized = stack.astype(np.float32) / 255.0
        if not np.array_equal(np.rint(normalized * 255).astype(np.uint8), stack):
            raise ValueError("G0 policy stack does not roundtrip through float32 [0,1]")
        scores.append(visual_motion_score(normalized))
        stack[:-1] = stack[1:].copy()
        stack[-1] = arrays["next_frame"][step]
        if step + 1 < steps and not np.array_equal(stack[-1], arrays["observation_frame"][step + 1]):
            raise ValueError("G0 next_frame/observation_frame history disagrees")
    labels = contact_stall_labels(arrays["speed_m_s"], arrays["new_tiles"], arrays["contact"])
    return scores, labels


def analyze(root: Path, rule_path: str, rule_sha256: str, output_path: str, *,
            source: dict[str, str] = SOURCE) -> dict:
    """Validate pinned provenance and ALL 24 NPZs before writing one exclusive JSON."""
    root = root.resolve()
    output = Path(output_path) if isinstance(output_path, str) else Path()
    if (not isinstance(output_path, str) or "\\" in output_path or output.is_absolute()
            or len(output.parts) != 2 or output.parts[0] != "runs" or output.suffix != ".json"
            or output.as_posix() != output_path or output.parts[1] in (".", "..")
            or (root / "runs").is_symlink() or not (root / "runs").is_dir()
            or (root / output).exists() or (root / output).is_symlink()):
        raise ValueError("output must be a new exclusive runs/<name>.json file")
    prefix = Path(rule_path).parts[0] if isinstance(rule_path, str) and rule_path else ""
    if prefix not in ("experiments", "runs"):
        raise ValueError("rule must be a repository-relative experiments/ or runs/ JSON")
    rule = _json(_pinned(root, rule_path, rule_sha256, prefix))
    # JSON serialization differentiates true and 1; the entire rule is immutable.
    if json.dumps(rule, sort_keys=True) != json.dumps(rule_template(source), sort_keys=True):
        raise ValueError("pixel-motion rule differs from exact diagnostic-only freeze schema")
    manifest = _json(_pinned(root, source["manifest_path"], source["manifest_sha256"], "runs"))
    protocol = _json(_pinned(root, source["protocol_path"], source["protocol_sha256"], "experiments"))
    ledger = _pinned(root, source["cells_path"], source["cells_sha256"], "runs")
    if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("role") != "TRAIN-only-failure-diagnostic" or manifest.get("ranked") is not False
            or manifest.get("protocol_path") != source["protocol_path"]
            or manifest.get("protocol_sha256") != source["protocol_sha256"]
            or manifest.get("cells_sha256") != source["cells_sha256"]
            or manifest.get("cell_count") != 24 or manifest.get("geometry_count") != 12
            or manifest.get("geometry_audit_sha256") != protocol.get("geometry_audit_sha256")
            or protocol.get("format") != "haic-rlpd-g0-diagnostic-v1"
            or protocol.get("status") != "frozen" or protocol.get("partition") != "TRAIN"
            or protocol.get("interventions") is not False or protocol.get("learner_updates") != 0
            or protocol.get("frame_skip") != 4 or protocol.get("max_steps") != 2000):
        raise ValueError("G0 original manifest/protocol provenance differs")
    cells, actors = protocol.get("cells"), protocol.get("actors")
    if (not isinstance(cells, list) or len(cells) != 12 or not isinstance(actors, list) or len(actors) != 2
            or any(not isinstance(cell, dict) or cell.get("partition") != "TRAIN"
                   or cell.get("obstacles") is not True or type(cell.get("track_id")) is not int
                   or type(cell.get("geometry_seed")) is not int for cell in cells)
            or len({(cell["track_id"], cell["geometry_seed"]) for cell in cells}) != 12
            or any(not isinstance(actor, dict) or not isinstance(actor.get("id"), str)
                   or ACTOR_ID.fullmatch(actor["id"]) is None or actor.get("action_mode") != "exported_tanh_mean"
                   or not isinstance(actor.get("sha256"), str)
                   or SHA256.fullmatch(actor["sha256"]) is None for actor in actors)
            or actors[0]["id"] == actors[1]["id"]):
        raise ValueError("G0 requires 12 paired TRAIN roads and two frozen exported actors")
    rows = [json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
            for line in ledger.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("G0 must contain all 24 ledger rows")
    trace_paths = []
    for index, row in enumerate(rows):
        cell, actor = cells[index // 2], actors[index % 2]
        relative = (Path(source["manifest_path"]).parent / "traces" /
                    f"seed-{cell['geometry_seed']}-track-{cell['track_id']}-{actor['id']}.npz").as_posix()
        if (row.get("partition") != "TRAIN" or row.get("track_id") != cell["track_id"]
                or row.get("geometry_seed") != cell["geometry_seed"]
                or row.get("actor_id") != actor["id"] or row.get("actor_sha256") != actor["sha256"]
                or row.get("trace_path") != Path(relative).relative_to(Path(source["manifest_path"]).parent).as_posix()
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= 2000
                or not isinstance(row.get("summary"), dict)
                or row["summary"].get("outcome") not in {"finished", "off_track", "crash", "out_of_bounds", "task_timeout", "unknown"}
                or type(row.get("driven_raw_frames")) is not int
                or type(row.get("reset_initial_raw_frames")) is not int
                or type(row.get("reset_noop_raw_frames")) is not int
                or not isinstance(row.get("road_centerline_sha256"), str)
                or SHA256.fullmatch(row["road_centerline_sha256"]) is None):
            raise ValueError("G0 ledger differs from frozen paired-cell schedule")
        if index % 2 and rows[index - 1]["road_centerline_sha256"] != row["road_centerline_sha256"]:
            raise ValueError("G0 paired road-centerline hashes disagree")
        trace_paths.append(relative)
    if (len(set(trace_paths)) != 24 or sum(row["steps"] for row in rows) != manifest.get("decisions_spent")
            or sum(row["driven_raw_frames"] for row in rows) != manifest.get("driven_raw_frames")
            or sum(row["reset_initial_raw_frames"] for row in rows) != manifest.get("reset_initial_raw_frames")
            or sum(row["reset_noop_raw_frames"] for row in rows) != manifest.get("reset_noop_raw_frames")
            or any(sum(row["summary"]["outcome"] == "finished" for row in rows[side::2]) != 3
                   for side in range(2))):
        raise ValueError("G0 decision/finish/control counts disagree")
    for row, relative in zip(rows, trace_paths):
        _pinned(root, relative, row["trace_sha256"], "runs")

    geometries = []
    labelled, unlabelled, controls = [], [], []
    for index in range(0, 24, 2):
        cell = cells[index // 2]
        episodes = []
        for row, relative in zip(rows[index:index + 2], trace_paths[index:index + 2]):
            scores, labels = _trace(root, relative, row["trace_sha256"], row)
            episodes.append({
                "actor_id": row["actor_id"], "outcome": row["summary"]["outcome"],
                "source_trace_path": relative, "source_trace_sha256": row["trace_sha256"],
                "scores_by_decision": scores, "contact_stall_labels_by_decision": labels,
            })
            for score, label in zip(scores, labels):
                (labelled if label else unlabelled).append(score)
            if row["summary"]["outcome"] == "finished":
                controls.extend(scores)
        geometries.append({"track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                           "episodes": episodes})
    for key, prefix in (("protocol", "experiments"), ("manifest", "runs"), ("cells", "runs")):
        _pinned(root, source[f"{key}_path"], source[f"{key}_sha256"], prefix)
    for row, relative in zip(rows, trace_paths):
        _pinned(root, relative, row["trace_sha256"], "runs")
    for relative, sha in rule["source_hashes"].items():
        if sha256_file(ROOT / relative) != sha:
            raise ValueError(f"pixel-motion analysis source drifted: {relative}")
    _pinned(root, rule_path, rule_sha256, Path(rule_path).parts[0])

    def distribution(scores: list[float]) -> dict:
        return {"count": len(scores), "scores_sorted": sorted(scores)}

    result = {
        "format": "haic-rlpd-pixel-motion-result-v1", "role": "TRAIN-only-retrospective-diagnostic",
        "source": dict(source), "rule_path": rule_path, "rule_sha256": rule_sha256,
        "source_hashes": rule["source_hashes"], "score": rule["score"],
        "diagnostic_label": rule["diagnostic_label"], "selected_threshold": None,
        "causal_claim": None, "updated_policy": False,
        "geometry_clusters": 12, "episode_count": 24, "finished_controls": 6,
        "decision_count": len(labelled) + len(unlabelled),
        "distributions": {
            "contact_stall": distribution(labelled), "other_decisions": distribution(unlabelled),
            "all_successful_controls": distribution(controls),
        },
        "geometries": geometries,
    }
    with (root / output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-rule-template", action="store_true")
    parser.add_argument("--rule")
    parser.add_argument("--rule-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.print_rule_template:
        if any((args.rule, args.rule_sha256, args.output)):
            parser.error("--print-rule-template cannot be combined with analysis arguments")
        print(json.dumps(rule_template(), sort_keys=True, indent=2))
        return
    if not all((args.rule, args.rule_sha256, args.output)):
        parser.error("--rule, --rule-sha256 and --output are required")
    result = analyze(ROOT, args.rule, args.rule_sha256, args.output)
    print(json.dumps({key: result[key] for key in ("format", "episode_count", "decision_count", "finished_controls")},
                     sort_keys=True))


if __name__ == "__main__":
    main()
