"""Full-episode exported RLPD diagnostics on twelve consumed G0 TRAIN roads.

Oracle outputs are telemetry only, never applied to the exported policy. The
archived-screen review is a separate zero-reset command.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np

from scripts import diagnose_oracle_policy_failures as diagnosis

ROOT = Path(__file__).resolve().parents[1]
SCREEN = "runs/20260929-rlpd-newhost-train-screen-v1"
SCREEN_SHA = "bb6d151bdab8e14cf191bd48ccf77d1a1f643368c3ebb75349c67f3243fcc009"
G0 = "runs/20260926-rlpd-g0-completion-v1/cells.jsonl"
G0_SHA = "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559"
V5_SHA = "ba56dca388a204a902d0f198d25a909a64866511002854972c4d0dd9627e8b98"
SEEDS = tuple(range(4272000001, 4272000013))
CONTRACT = {"partition": "consumed TRAIN", "track_id": 1,
            "geometry_seeds": list(SEEDS), "frame_skip": 4,
            "warmup_raw_noops": 50, "max_decisions": 2000,
            "obstacles": True, "collision_penalty": 0.0,
            "reward_shaping": False, "oracle_diagnostic_only": True}
THRESHOLDS = {"speed_m_s": 14.0, "steering_abs": 0.6, "brake": 0.1,
              "lateral_m": 6.0, "heading_abs_rad": 0.55,
              "heading_sustained": 3, "lateral_sustained": 2,
              "curvature_abs": 0.025, "opposition_delta": 0.6,
              "terminal_curve_window_decisions": 63}


def sha(path: Path) -> str:
    return diagnosis._sha_file(path)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def checked(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != expected:
        raise ValueError(f"frozen evidence changed: {relative}")
    return path


def first(active: np.ndarray, sustained: int = 1) -> int | None:
    for index in range(len(active) - sustained + 1):
        if np.all(active[index:index + sustained]):
            return index
    return None


def events(trace: dict[str, np.ndarray], *, archived: bool = False) -> dict[str, Any]:
    """Keep all event types, including successful controls and missing telemetry."""
    action = trace["commanded_official_action" if archived else "policy_or_oracle_action"]
    speed = trace["speed_m_s" if archived else "pre_speed"]
    lateral = trace["centerline_distance_m" if archived else "center_error"]
    heading = trace["heading_error_rad" if archived else "heading_error"]
    count = len(speed)
    if action.shape != (count, 3) or lateral.shape != (count,) or heading.shape != (count,):
        raise ValueError("misaligned diagnostic fields")
    if any(not np.isfinite(x).all() for x in (action, speed, lateral, heading)):
        raise ValueError("nonfinite diagnostic fields")
    predicates = {
        "speed-threshold": (speed >= 14, 1),
        "large-steering": (np.abs(action[:, 0]) >= .6, 1),
        "brake-request": (action[:, 2] >= .1, 1),
        "lateral-excursion": (np.abs(lateral) >= 6, 2),
        "heading-error": (np.abs(heading) >= .55, 3),
    }
    missing = []
    if archived:
        missing = ["curve-entry-overspeed", "steering-opposition"]
    else:
        teacher = trace["oracle_action_at_state"]
        curvature = trace["curvature"]
        if (teacher.shape != action.shape or curvature.shape != (count,)
                or not np.isfinite(teacher).all() or not np.isfinite(curvature).all()):
            raise ValueError("misaligned Oracle telemetry")
        predicates["curve-entry-overspeed"] = (
            (np.abs(curvature) >= .025) & (speed >= 14) & (action[:, 1] >= .35)
            & (action[:, 2] < .05) & (teacher[:, 2] >= .1), 1)
        predicates["steering-opposition"] = (
            (speed >= 2.5) & (np.abs(teacher[:, 0]) >= .25)
            & (action[:, 0] * teacher[:, 0] < 0)
            & (np.abs(action[:, 0] - teacher[:, 0]) >= .6), 1)
    found = {}
    for name, (active, sustained) in predicates.items():
        index = first(active, sustained)
        found[name] = None if index is None else {
            "decision_index_zero_based": index, "sustained_decisions": sustained,
            "speed_m_s": float(speed[index]), "steering": float(action[index, 0]),
            "throttle": float(action[index, 1]), "brake": float(action[index, 2]),
            "lateral_m": float(lateral[index]), "heading_error_rad": float(heading[index]),
        }
    risks = [name for name in found if name in (
        "curve-entry-overspeed", "steering-opposition", "heading-error", "lateral-excursion")]
    observed = [(found[k]["decision_index_zero_based"], k) for k in risks if found[k] is not None]
    earliest = min(observed) if observed else None
    return {"first_events": found, "first_risk_type": earliest[1] if earliest else "unclassified",
            "unassessed": missing, "causal": False,
            "state_timing": "post-decision" if archived else "pre-decision",
            "lateral_signed": not archived}


def census(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    subsets = {
        "successful_controls": [r for r in rows if r["finished"]],
        "nonfinishes": [r for r in rows if not r["finished"] and not r.get("censored", False)],
        "censored_unknown": [r for r in rows if r.get("censored", False)],
    }
    for label, subset in subsets.items():
        result[label] = {
            "denominator": len(subset),
            "first_risk_types": dict(Counter(r["events"]["first_risk_type"] for r in subset)),
            "event_episode_counts": dict(Counter(k for r in subset
                                                 for k, v in r["events"]["first_events"].items()
                                                 if v is not None)),
        }
    return result


def terminal_curve_association(trace: dict[str, np.ndarray], summary: dict[str, Any]) -> dict[str, Any]:
    """A prospective temporal failure endpoint, not a claim about causation."""
    n = len(trace["pre_speed"])
    action, teacher = trace["policy_or_oracle_action"], trace["oracle_action_at_state"]
    active = ((np.abs(trace["curvature"]) >= .025) & (trace["pre_speed"] >= 14)
              & (action[:, 1] >= .35) & (action[:, 2] < .05) & (teacher[:, 2] >= .1))
    indices = np.flatnonzero(active[max(0, n - 63):]) + max(0, n - 63)
    failure = (not summary["finished"] and summary["reason"] in (
        "crash", "off_track", "out_of_bounds") and (summary["terminated"] or summary["truncated"]))
    witnesses = []
    if failure:
        for index in indices:
            lane_loss = first(np.abs(trace["center_error"][index:]) >= 6, 2) is not None
            damage_gain = float(summary["damage"]) > float(trace["damage"][index]) + 1e-6
            if lane_loss or damage_gain:
                witnesses.append(int(index))
    return {"associated_terminal_failure": bool(witnesses),
            "curve_entries_in_terminal_window": len(indices),
            "witness_entry_indices": witnesses, "window_decisions": 63,
            "maximum_entry_to_terminal_seconds": 5.04, "causal": False,
            "definition": "Genuine crash/off-track/out-of-bounds within 63 decisions of overspeed, with sustained lane loss or added damage."}


def archived_screen(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_path = checked(root, f"{SCREEN}/manifest.json", SCREEN_SHA)
    manifest = json.loads(manifest_path.read_text())
    hashes = {f"{SCREEN}/manifest.json": SCREEN_SHA}
    rows = []
    for item in manifest["artifacts"]:
        slot = item["slot"]
        relative = f"{SCREEN}/cells/slot-{slot:02d}.json"
        cell = json.loads(checked(root, relative, item["cell_sha256"]).read_text())
        trace_relative = f"{SCREEN}/{cell['trace_path']}"
        path = checked(root, trace_relative, item["trace_sha256"])
        if cell["trace_sha256"] != item["trace_sha256"]:
            raise ValueError("cell/manifest trace hash mismatch")
        hashes.update({relative: item["cell_sha256"], trace_relative: item["trace_sha256"]})
        with np.load(path, allow_pickle=False) as data:
            analyzed = events(dict(data), archived=True)
        rows.append({"actor_id": cell["actor_id"], "actor_sha256": cell["actor_sha256"],
                     "track_id": cell["track_id"], "geometry_seed": cell["geometry_seed"],
                     "finished": cell["summary"]["outcome"] == "finished", "steps": cell["steps"],
                     "trace_path": trace_relative, "trace_sha256": item["trace_sha256"],
                     "road_centerline_sha256": cell["road_centerline_sha256"], "events": analyzed})
    expected = {(actor, seed) for actor in ("entropy-v5-author-seed50", "newhost-seed52-final") for seed in SEEDS}
    if len(rows) != 24 or {(r["actor_id"], r["geometry_seed"]) for r in rows} != expected or any(r["track_id"] != 1 for r in rows):
        raise ValueError("archived screen is not the fixed 24-episode cohort")
    if any(r["actor_sha256"] != V5_SHA for r in rows if r["actor_id"] == "entropy-v5-author-seed50"):
        raise ValueError("immutable V5 identity changed")
    return rows, hashes


def review(root: Path = ROOT) -> dict[str, Any]:
    rows, hashes = archived_screen(root)
    return {"format": "haic-rlpd-recovery-precursor-review-v1", "environment_resets": 0,
            "episodes": 24, "source_artifacts_sha256": hashes, "thresholds": THRESHOLDS,
            "per_actor": {label: census([r for r in rows if r["actor_id"] == label])
                          for label in sorted({r["actor_id"] for r in rows})}, "episodes_reviewed": rows,
            "limitations": ["No curvature or same-state Oracle actions exist in these archives; curve-entry overspeed and steering opposition are unassessed.",
                             "Post-step speed/heading/unsigned nearest-centerline distance are descriptive proxies, not causal precursors or Oracle geometry.",
                             "First-event thresholds also occur in successful controls; no fresh generalization or official performance claim."]}


def parse_actors(values: list[str], root: Path = ROOT) -> dict[str, Path]:
    actors = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", label) is None or label in actors:
            raise ValueError("actors require unique safe label=path entries")
        candidate = (root / path).resolve()
        if not candidate.is_file():
            raise ValueError(f"actor does not exist: {path}")
        actors[label] = candidate
    if not actors:
        raise ValueError("at least one actor export is required")
    return actors


def reload_preflight(path: Path, factory=None) -> dict[str, Any]:
    import torch
    from agent import Agent

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "haic-rlpd-pixel-actor-v1":
        raise ValueError("only exported RLPD actors are allowed")
    factory = factory or (lambda p: Agent(model_path=str(p)))
    fixtures = [np.zeros((4, 84, 84), np.float32),
                np.linspace(0, 1, 4 * 84 * 84, dtype=np.float32).reshape(4, 84, 84)]
    actors = [factory(path), factory(path)]
    actions = []
    for obs in fixtures:
        values = []
        for actor in actors:
            actor.reset(obs)
            action = np.asarray(actor.act(obs))
            if (action.shape != (3,) or action.dtype != np.float32 or not np.isfinite(action).all()
                    or np.any(action < [-1, 0, 0]) or np.any(action > [1, 1, 1])):
                raise ValueError("invalid synthetic official action")
            values.append(action)
        if not np.array_equal(*values):
            raise ValueError("double reload action mismatch")
        actions.append(values[0].tolist())
    return {"fixtures": 2, "double_reload_bit_exact": True, "actions": actions,
            "fixture_sha256": [hashlib.sha256(o.tobytes()).hexdigest() for o in fixtures],
            "environment_resets": 0}


def runtime() -> dict[str, Any]:
    import torch
    torch.set_num_threads(1)
    return {"python": sys.version, "platform": platform.platform(), "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "packages": {p: importlib.metadata.version(p) for p in
                         ("torch", "numpy", "gymnasium", "opencv-python", "pygame", "box2d-py")}}


def source_hashes(root: Path) -> dict[str, str]:
    from scripts.diagnose_rlpd_g0 import SOURCE_FILES
    files = set(SOURCE_FILES) | {"scripts/evaluate_rlpd_recovery.py",
                               "scripts/diagnose_oracle_policy_failures.py"}
    files.update(p.relative_to(root).as_posix() for p in (root / "haic/oracle_v1").rglob("*.py"))
    files.update(p.relative_to(root).as_posix() for p in (root / "core/vendor").rglob("*.py"))
    return {name: sha(root / name) for name in sorted(files)}


def transition_table(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    a = {(r["track_id"], r["geometry_seed"]): r for r in left}
    b = {(r["track_id"], r["geometry_seed"]): r for r in right}
    if len(left) != 12 or len(right) != 12 or len(a) != 12 or len(b) != 12 or a.keys() != b.keys():
        raise ValueError("finish comparison requires all twelve paired cells")
    counts = Counter()
    for key in a:
        if a[key]["road_centerline_sha256"] != b[key]["road_centerline_sha256"]:
            raise ValueError("paired road hash mismatch")
        if not a[key].get("censored", False) and not b[key].get("censored", False):
            counts[(a[key]["finished"], b[key]["finished"])]+=1
    censored = sum(a[k].get("censored", False) or b[k].get("censored", False) for k in a)
    return {"denominator": 12, "kept": counts[True, True], "lost": counts[True, False],
            "gained": counts[False, True], "neither": counts[False, False],
            "censored_pairs": censored, "full_finish_comparison_valid": censored == 0,
            "matched_cells_not_same_state": True, "causal_training_effect": False}


def evaluate(actors: dict[str, Path], output: str, *, preflight_only: bool = False,
             root: Path = ROOT) -> dict[str, Any]:
    if diagnosis.FRAME_SKIP != 4 or diagnosis.WARMUP != 50 or diagnosis.TARGET_SPEED != 12.0:
        raise ValueError("shared collector environment/Oracle contract changed")
    if re.fullmatch(r"runs/[a-z0-9][a-z0-9-]*", output) is None:
        raise ValueError("output must be runs/<newslug>")
    target = root / output
    if target.exists() or target.parent.is_symlink() or not target.parent.is_dir():
        raise ValueError("output must be new with an existing nonsymlink runs parent")
    archive, archive_hashes = archived_screen(root)
    g0 = checked(root, G0, G0_SHA)
    historical = [json.loads(line) for line in g0.read_text().splitlines() if line.strip()]
    roads = {r["geometry_seed"]: r["road_centerline_sha256"] for r in historical}
    if set(roads) != set(SEEDS) or any(r["track_id"] != 1 for r in historical):
        raise ValueError("consumed G0 cell evidence changed")
    for row in archive:
        if row["road_centerline_sha256"] != roads[row["geometry_seed"]]:
            raise ValueError("screen/G0 road hash mismatch")
    frozen = {"format": "haic-rlpd-recovery-evaluation-v1", "contract": CONTRACT,
              "thresholds": THRESHOLDS, "sources_sha256": source_hashes(root),
              "oracle_contract": {"target_speed_m_s": 12.0, "avoid_obstacles": True},
              "actors": {k: {"path": str(p), "sha256": sha(p)} for k, p in actors.items()},
              "runtime": runtime(), "evidence_sha256": {**archive_hashes, G0: G0_SHA}}
    frozen["preflight"] = {k: reload_preflight(p) for k, p in actors.items()}
    frozen_hash = canonical_sha(frozen)
    target.mkdir()
    write_json(target / "protocol.json", frozen)
    write_json(target / "preflight.json", {"status": "passed", "environment_resets": 0,
                                          "protocol_content_sha256": frozen_hash})

    def recheck() -> None:
        if canonical_sha(json.loads((target / "protocol.json").read_text())) != frozen_hash:
            raise ValueError("protocol changed")
        for name, expected in frozen["sources_sha256"].items():
            checked(root, name, expected)
        for name, expected in frozen["evidence_sha256"].items():
            checked(root, name, expected)
        if runtime() != frozen["runtime"] or any(sha(actors[k]) != v["sha256"] for k, v in frozen["actors"].items()):
            raise ValueError("runtime/actor changed")

    recheck()
    if preflight_only:
        write_json(target / "manifest.json", {"files_sha256": {
            p.name: sha(p) for p in target.iterdir() if p.is_file()}, "environment_resets": 0})
        return {"status": "preflight_only", "environment_resets": 0, "output": output}
    from agent import Agent
    rows = []
    try:
        for label, path in actors.items():
            actor = Agent(model_path=str(path))
            for seed in SEEDS:
                recheck()
                name = f"{label}-seed-{seed}"
                case = {"track_id": 1, "geometry_seed": seed, "max_steps": 2000,
                        "source_road_centerline_sha256": roads[seed]}
                write_json(target / f"{name}-reset-intent.json", {
                    "case": case, "actor": frozen["actors"][label],
                    "protocol_content_sha256": frozen_hash, "no_retry": True})
                summary, trace = diagnosis._collect_episode(case, "policy", actor)
                recheck()
                if not 0 < summary["steps"] <= 2000 or (not summary["terminated"] and not summary["truncated"] and summary["steps"] != 2000):
                    raise ValueError("incomplete full episode")
                trace_path = target / f"{name}.npz"
                trace_hash = diagnosis._write_trace(trace_path, trace)
                row = {**summary, "actor_id": label, "actor_sha256": frozen["actors"][label]["sha256"],
                        "events": events(trace), "trace_path": trace_path.name, "trace_sha256": trace_hash,
                        "terminal_curve_association": terminal_curve_association(trace, summary),
                       "censored": not summary["terminated"] and not summary["truncated"] and not summary["finished"]}
                write_json(target / f"{name}-receipt.json", row)
                rows.append(row)
                print(f"{label} seed={seed} steps={row['steps']} "
                      f"finished={row['finished']} censored={row['censored']} "
                      f"precursor={row['events']['first_risk_type']}", flush=True)
        baseline = [r for r in archive if r["actor_sha256"] == V5_SHA]
        for i, a in enumerate(actors):
            for b in list(actors)[i+1:]:
                left = {r["geometry_seed"]: r for r in rows if r["actor_id"] == a}
                right = {r["geometry_seed"]: r for r in rows if r["actor_id"] == b}
                if any(left[s]["initial_observation_sha256"] != right[s]["initial_observation_sha256"] for s in SEEDS):
                    raise ValueError("contemporaneous reset observation mismatch")
        result = {"format": "haic-rlpd-recovery-evaluation-result-v1", "status": "complete",
                  "episodes": rows, "official_score": False, "fresh_generalization": False,
                  "per_actor": {k: {"finish_count": sum(r["finished"] for r in rows if r["actor_id"] == k),
                                     "denominator": 12, "censored": sum(r["censored"] for r in rows if r["actor_id"] == k),
                                     "curve_entry_associated_terminal_failures": sum(
                                         r["terminal_curve_association"]["associated_terminal_failure"]
                                         for r in rows if r["actor_id"] == k),
                                    "precursors": census([r for r in rows if r["actor_id"] == k])}
                                for k in actors},
                  "versus_archived_immutable_v5": {k: transition_table(baseline, [r for r in rows if r["actor_id"] == k]) for k in actors},
                  "contemporaneous_pairs": {f"{a}->{b}": transition_table([r for r in rows if r["actor_id"] == a], [r for r in rows if r["actor_id"] == b])
                                            for i, a in enumerate(actors) for b in list(actors)[i+1:]},
                  "limitations": ["Consumed TRAIN only; no causal training or fresh-road claim.",
                                   "Archived V5 is immutable historical control, not a contemporaneous same-state control.",
                                   "Oracle controller telemetry on policy states is diagnostic only; thresholds also occur on successes."]}
        write_json(target / "result.json", result)
    except BaseException as error:
        write_json(target / "failure.json", {"completed_episodes": len(rows), "no_retry": True,
                                            "error": f"{type(error).__name__}: {error}",
                                            "limitation": "Shared collector does not expose a partial trace after an exception; preserve reset intent."})
        write_json(target / "manifest.json", {"files_sha256": {
            p.name: sha(p) for p in target.iterdir() if p.is_file()}, "status": "failed",
            "episodes_complete": len(rows)})
        raise
    hashes = {p.name: sha(p) for p in target.iterdir() if p.is_file()}
    write_json(target / "manifest.json", {"files_sha256": hashes, "protocol_content_sha256": frozen_hash,
                                          "episodes_complete": len(rows)})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", nargs="+", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--review-existing", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.review_existing:
            if args.actors:
                raise ValueError("review-existing cannot evaluate actors")
            path = (ROOT / args.output).resolve()
            if path != ROOT / "experiments/rlpd-recovery-precursor-review-v1-result.json":
                raise ValueError("review output must be the requested experiment artifact")
            value = review()
            write_json(path, value)
        else:
            value = evaluate(parse_actors(args.actors), args.output, preflight_only=args.preflight_only)
        print(json.dumps({k: value[k] for k in ("status", "environment_resets") if k in value}))
        return 0
    except Exception as error:
        print(f"recovery evaluation failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
