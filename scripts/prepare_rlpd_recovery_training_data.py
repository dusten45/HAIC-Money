"""Derive immutable paired-finish recovery trajectories without environment use."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "haic-rlpd-recovery-dataset-v1"
POLICY = "paired-finish-recovery-plus-handoff-v1"
FOLLOWUP = 63


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def source_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("source episode path must be a string")
    name = Path(relative)
    if name.is_absolute() or not name.parts or any(p in ("..", ".") for p in name.parts):
        raise ValueError("unsafe source episode path")
    if any(root.joinpath(*name.parts[:i]).is_symlink() for i in range(1, len(name.parts) + 1)):
        raise ValueError("symlinked source episode path")
    return root / name


def stack(frames: np.ndarray, step: int) -> np.ndarray:
    return frames[np.maximum(np.arange(step - 3, step + 1), 0)]


def qualification(arrays: dict, anchor: int, horizon: int, censored: bool) -> bool:
    end = anchor + horizon + FOLLOWUP
    n = len(arrays["reward"])
    if censored or anchor >= n or end > n:
        return False
    damage = arrays["damage"][anchor:end + 1]
    early_failure = np.any(arrays["episode_end"][anchor:end] & ~arrays["finished"][anchor:end])
    return bool(not early_failure and abs(arrays["center_error"][end]) <= 6
                and arrays["progress"][end] - arrays["progress"][anchor] >= .02
                and np.max(damage) - damage[0] <= 0)


def validate_episode(row: dict, arrays: dict) -> None:
    n, anchor, horizon = row["steps"], row["anchor_step"], row["horizon"]
    for key in ("finished", "censored", "local_recovery_qualified"):
        if type(row.get(key)) is not bool:
            raise ValueError(f"source metadata requires boolean {key}")
    frames = arrays["frames"]
    if frames.shape != (n + 1, 84, 84) or frames.dtype != np.uint8:
        raise ValueError("source must preserve N+1 uint8 frames")
    for key, at in (("initial_stack", 0), ("final_stack", n)):
        if arrays[key].dtype != np.uint8 or not np.array_equal(arrays[key], stack(frames, at)):
            raise ValueError("source stack reconstruction mismatch")
    for key in ("reward", "step", "role", "terminated", "truncated", "terminal", "episode_end", "finished", "recovery_mask"):
        if arrays[key].shape != (n,):
            raise ValueError(f"source field shape mismatch: {key}")
    for key in ("terminated", "truncated", "terminal", "episode_end", "finished", "recovery_mask"):
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"source flags must be boolean: {key}")
    if not np.issubdtype(arrays["step"].dtype, np.integer) or not np.array_equal(arrays["step"], np.arange(n)):
        raise ValueError("source steps must be contiguous from reset")
    ended = arrays["terminated"] | arrays["truncated"]
    if (not np.array_equal(ended, arrays["episode_end"]) or ended[:-1].any()
            or bool(ended[-1]) == row["censored"]
            or (arrays["terminated"] & ~arrays["terminal"]).any()
            or (arrays["terminal"] & ~ended).any()
            or (arrays["finished"] & ~arrays["terminal"]).any()
            or arrays["finished"][:-1].any()
            or bool(arrays["finished"][-1]) != row["finished"]
            or row["censored"] and (n != 2000 or row["finished"])):
        raise ValueError("source final done/finished/censor flags disagree")
    expected_role = np.where(np.arange(n) < anchor, "prefix",
                             np.where(np.arange(n) < anchor + horizon, "oracle", "actor"))
    if not np.array_equal(arrays["role"], expected_role):
        raise ValueError("source intervention/handoff role schedule mismatch")
    for key in ("speed", "curvature", "center_error", "progress", "damage"):
        if arrays[key].shape != (n + 1,) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"source state telemetry invalid: {key}")
    if arrays["oracle_action"].shape != (n + 1, 3) or not np.isfinite(arrays["oracle_action"]).all():
        raise ValueError("source Oracle telemetry invalid")
    for key in ("proposed_action", "executed_action", "applied_action"):
        action = arrays[key]
        low = [-1, 0, 0] if key == "applied_action" else [-1, -1, -1]
        if (action.shape != (n, 3) or action.dtype != np.float32 or not np.isfinite(action).all()
                or np.any(action < low) or np.any(action > 1)):
            raise ValueError(f"source executed/action data invalid: {key}")
    native = arrays["applied_action"].copy()
    native[:, 1:] = native[:, 1:] * 2 - 1
    if not np.array_equal(native, arrays["executed_action"]):
        raise ValueError("source executed native/action mapping mismatch")
    oracle_rows = arrays["role"] == "oracle"
    actor_rows = arrays["role"] == "actor"
    if (not np.array_equal(arrays["applied_action"][oracle_rows], arrays["oracle_action"][:-1][oracle_rows])
            or not np.array_equal(arrays["executed_action"][actor_rows], arrays["proposed_action"][actor_rows])):
        raise ValueError("source executed actions disagree with intervention/actor roles")
    if not np.isfinite(arrays["reward"]).all():
        raise ValueError("source raw rewards must be finite")
    local = qualification(arrays, anchor, horizon, row["censored"])
    if local != row["local_recovery_qualified"]:
        raise ValueError("source local qualification does not match arrays")
    expected_mask = (arrays["role"] == "oracle") & local
    if (not np.array_equal(arrays["recovery_mask"], expected_mask)
            or type(row.get("accepted_transitions")) is not int
            or row["accepted_transitions"] != int(expected_mask.sum())):
        raise ValueError("source Oracle-only qualification mask/count mismatch")
    if horizon == 0:
        overspeed = bool(anchor < n and abs(arrays["curvature"][anchor]) >= .025
                         and arrays["speed"][anchor] >= 14
                         and arrays["applied_action"][anchor, 1] >= .35
                         and arrays["applied_action"][anchor, 2] < .05
                         and arrays["oracle_action"][anchor, 2] >= .1)
        if type(row.get("current_anchor_curve_entry_overspeed")) is not bool or row["current_anchor_curve_entry_overspeed"] != overspeed:
            raise ValueError("source baseline overspeed flag does not match actual arrays")


def prepare(source: Path, source_manifest_sha256: str, output: str, *, root: Path = ROOT) -> dict:
    source = Path(source)
    if source.is_symlink():
        raise ValueError("source dataset directory cannot be symlinked")
    source = source.resolve()
    manifest_path = source / "manifest.json"
    if (re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None
            or manifest_path.is_symlink() or sha(manifest_path) != source_manifest_sha256):
        raise ValueError("source manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("episodes", [])
    if (manifest.get("format") != FORMAT or manifest.get("status") != "complete"
            or manifest.get("eligibility_policy") is not None or len(rows) != 42):
        raise ValueError("only complete original 42-episode collector datasets are supported")
    if (re.fullmatch(r"runs/[a-z0-9][a-z0-9-]*", output) is None
            or (root / "runs").is_symlink() or not (root / "runs").is_dir()
            or (root / output).exists()):
        raise ValueError("output must be a new runs/<slug> with existing nonsymlink parent")
    groups, arrays, paths = defaultdict(dict), {}, {}
    ids = set()
    for row in rows:
        if (type(row.get("episode_id")) is not int or row["episode_id"] < 0 or row["episode_id"] in ids
                or type(row.get("steps")) is not int or not 0 < row["steps"] <= 2000
                or type(row.get("anchor_step")) is not int or not 0 <= row["anchor_step"] < 2000
                or type(row.get("horizon")) is not int or row["horizon"] not in (0, 12, 25)
                or row.get("track_id") != 1 or type(row.get("geometry_seed")) is not int
                or not 4272000001 <= row["geometry_seed"] <= 4272000012
                or row.get("policy_id") not in ("rlpd-seed11", "rlpd-seed50")
                or row.get("stratum") not in ("failure", "finish-control")
                or row.get("mode") != ("actor" if row["horizon"] == 0 else f"oracle-{row['horizon']}")):
            raise ValueError("invalid source episode identity or consumed cell")
        ids.add(row["episode_id"])
        key = (row["policy_id"], row["geometry_seed"])
        if row["horizon"] in groups[key]:
            raise ValueError("duplicate baseline/branch source slot")
        groups[key][row["horizon"]] = row
        path = source_path(source, row["path"])
        if path in paths.values() or sha(path) != row["sha256"]:
            raise ValueError("duplicate source path or source NPZ hash mismatch")
        paths[row["episode_id"]] = path
        with np.load(path, allow_pickle=False) as data:
            arrays[row["episode_id"]] = {k: data[k] for k in data.files}
        validate_episode(row, arrays[row["episode_id"]])
    if len(groups) != 14 or any(set(g) != {0, 12, 25} for g in groups.values()):
        raise ValueError("source must contain fourteen complete h0/12/25 groups")
    if sum(g[0]["stratum"] == "failure" for g in groups.values()) != 10:
        raise ValueError("source cohort must retain ten failure and four finish-control anchors")
    for group in groups.values():
        base = group[0]
        for row in group.values():
            if any(row.get(k) != base.get(k) for k in ("anchor_step", "stratum", "actor_sha256", "road_centerline_sha256")):
                raise ValueError("source branch pairing metadata differs")
        if any(not isinstance(base.get(k), str) or re.fullmatch(r"[0-9a-f]{64}", base[k]) is None
               for k in ("actor_sha256", "road_centerline_sha256")):
            raise ValueError("source pairing requires actual actor/road hashes")
    target = root / output
    new_rows = []
    unique = {s: set() for s in ("failure", "finish-control")}
    geometries = {s: set() for s in unique}
    masks = {}
    for row in rows:
        base = groups[row["policy_id"], row["geometry_seed"]][0]
        paired = bool(row["horizon"] and not base["censored"] and not row["censored"]
                      and row["finished"] and row["local_recovery_qualified"]
                      and (not base["finished"] and base["current_anchor_curve_entry_overspeed"]
                           if row["stratum"] == "failure" else base["finished"]))
        start, end = row["anchor_step"], row["anchor_step"] + row["horizon"] + FOLLOWUP
        mask = np.zeros(row["steps"], dtype=np.bool_)
        if paired:
            if end > row["steps"]:
                raise ValueError("qualified trajectory has unobserved handoff rows")
            mask[start:end] = True
        masks[row["episode_id"]] = mask
        a = arrays[row["episode_id"]]
        for step in np.flatnonzero(mask):
            identity = hashlib.sha256(stack(a["frames"], step).tobytes() + a["executed_action"][step].tobytes()).digest()
            unique[row["stratum"]].add(identity)
        if mask.any():
            geometries[row["stratum"]].add(row["geometry_seed"])
        new_rows.append({**row, "source_path": row["path"], "source_sha256": row["sha256"],
                         "path": f"episode-{row['episode_id']:03d}.npz",
                         "paired_finish_qualified": paired,
                         "training_window_start": start, "training_window_end": end,
                         "accepted_transitions": int(mask.sum())})
    # Recheck all pinned inputs before any derived dataset is created.
    if sha(manifest_path) != source_manifest_sha256 or any(sha(paths[r["episode_id"]]) != r["sha256"] for r in rows):
        raise ValueError("source changed during preparation")
    target.mkdir()
    script = Path(__file__).resolve()
    write_json(target / "preparation.json", {"eligibility_policy": POLICY, "source_manifest_sha256": source_manifest_sha256,
                                           "dataset_source_sha256": {"scripts/prepare_rlpd_recovery_training_data.py": sha(script)},
                                           "environment_resets": 0, "learner_updates": 0})
    for row in new_rows:
        a = arrays[row["episode_id"]]
        with (target / row["path"]).open("xb") as stream:
            np.savez_compressed(stream, **{**a, "recovery_mask": masks[row["episode_id"]]})
            stream.flush()
            os.fsync(stream.fileno())
        row["sha256"] = sha(target / row["path"])
    support = {s: {"accepted_transitions": sum(r["accepted_transitions"] for r in new_rows if r["stratum"] == s),
                   "unique_accepted_transitions": len(unique[s]), "accepted_geometries": len(geometries[s]),
                   "geometry_seeds": sorted(geometries[s])} for s in unique}
    result = {**manifest, "format": FORMAT, "status": "complete", "eligibility_policy": POLICY,
              "source_dataset": str(source), "source_manifest_sha256": source_manifest_sha256,
              "source_cost_counts": manifest.get("counts", {}),
              "dataset_source_sha256": {"scripts/prepare_rlpd_recovery_training_data.py": sha(script)},
              "episodes": new_rows, "counts": {"episodes": 42, "anchors": 14,
                  "stored_transitions": sum(r["steps"] for r in rows),
                  "accepted_transitions": sum(r["accepted_transitions"] for r in new_rows),
                  "unique_accepted_transitions": len(unique["failure"] | unique["finish-control"]),
                  "accepted_geometries": len(geometries["failure"] | geometries["finish-control"]),
                  "support_by_stratum": support,
                  "paired_finish_qualified_branches": sum(r["paired_finish_qualified"] for r in new_rows),
                  "censored_episodes_retained": sum(r["censored"] for r in rows)},
              "accepted_geometry_seeds": sorted(geometries["failure"] | geometries["finish-control"]),
              "sufficiency_gate": {"minimum_unique_failure_transitions": 128, "minimum_failure_geometries": 3,
                                   "passed": len(unique["failure"]) >= 128 and len(geometries["failure"]) >= 3},
              "environment_resets": 0, "learner_updates": 0,
              "limitation": "Privileged full-finish-selected consumed-TRAIN trajectory data, not independent generalization or deployable selection."}
    if sha(manifest_path) != source_manifest_sha256 or any(sha(paths[r["episode_id"]]) != r["sha256"] for r in rows):
        raise ValueError("source changed before derived manifest sealing; output remains unsealed")
    write_json(target / "manifest.json", result)
    write_json(target / "manifest.sha256.json", {"manifest_sha256": sha(target / "manifest.json"),
                                               "preparation_sha256": sha(target / "preparation.json")})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.source, args.source_manifest_sha256, args.output)
        print(json.dumps({"counts": result["counts"], "sufficiency_gate": result["sufficiency_gate"], "environment_resets": 0}, sort_keys=True))
        return 0
    except Exception as error:
        print(f"training data preparation failed closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
