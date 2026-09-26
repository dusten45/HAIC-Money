"""Read-only, fixed-window extraction of the frozen TRAIN-only RLPD G0 traces.

No environment or actor is imported. Privileged telemetry is diagnostic output,
never an inference input. The contact sheet is a compressed uint8 tile array,
not a selected set of visually interesting episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "manifest_path": "runs/20260926-rlpd-g0-completion-v1/manifest.json",
    "manifest_sha256": "52abe52381009f392b849c475afea7fa15aec93aaf8001fbc5eb14611ecbf358",
    "protocol_path": "experiments/rlpd-g0-completion-v1.json",
    "protocol_sha256": "6d4c50aaa03a68bad27bafc8eccf99f2df4377908a21fd31ec7729736f29946b",
    "cells_path": "runs/20260926-rlpd-g0-completion-v1/cells.jsonl",
    "cells_sha256": "924d3ea84b6a393ab962e3b2821d49b11e2c5b23c21f898caf4b795fd58d9559",
}
ANCHORS = (
    "first_observed", "first_contact", "first_centerline_far",
    "after_last_new_tile", "first_qualified", "terminal",
)
RUBRIC = (
    "road_near_reverse", "post_contact_stall", "centerline_far_departure_proxy",
    "clean_continuation", "unknown",
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ACTOR_ID = re.compile(r"[a-z0-9-]{1,48}\Z")
SCALARS = {
    "summed_reward", "new_tiles", "directed_delta", "centerline_distance_m",
    "centerline_fraction", "speed_m_s", "heading_error_rad", "x_m", "y_m",
    "progress", "damage", "contact", "off_track_counter", "finish_qualified",
    "finished", "terminated", "truncated", "finish_phase", "raw_frames",
}
ACTIONS = {
    "proposed_native_action", "executed_native_action",
    "commanded_official_action", "raw_official_action",
}
PIXELS = {"observation_frame", "next_frame"}
TRACE_KEYS = SCALARS | ACTIONS | PIXELS | {"initial_stack", "raw_finish_phase_bits"}
BOOLEAN = {"contact", "finish_qualified", "finished", "terminated", "truncated"}
INTEGER = {"new_tiles", "off_track_counter", "raw_frames"}


def freeze_template(source: dict[str, str] = SOURCE) -> dict:
    """Exact freeze JSON object; its *file bytes* require a separate SHA argument."""
    return {
        "format": "haic-rlpd-g0-fixed-windows-freeze-v1",
        "status": "frozen",
        "source": dict(source),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "window_radius_decisions": 20,
        "contact_sheet_stride_decisions": 5,
        "anchors": list(ANCHORS),
        "anchor_definitions": {
            "first_observed": "frozen ledger summary.first_observed_step",
            "first_contact": "first decision with contact=true",
            "first_centerline_far": "first decision with centerline_distance_m > frozen protocol threshold",
            "after_last_new_tile": "decision immediately after final new_tiles > 0, or missing if no such decision",
            "first_qualified": "first decision with finish_qualified=true",
            "terminal": "last terminated or truncated driven decision",
        },
        "contact_sheet_sampling": "every fifth decision from clipped window start, plus anchor and inclusive end",
        "manual_rubric": {
            "status": "unreviewed",
            "categories": list(RUBRIC),
            "allow_overlaps": True,
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(root: Path, relative: str, prefix: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or not relative:
        raise ValueError("unsafe input path")
    path = Path(relative)
    if path.is_absolute() or len(path.parts) < 2 or path.parts[0] != prefix or (
        any(part in (".", "..") for part in path.parts)
    ) or path.as_posix() != relative:
        raise ValueError(f"input must be a repository-relative {prefix}/ file")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in input path: {relative}")
    if not current.is_file():
        raise ValueError(f"missing input: {relative}")
    return current


def _pinned(root: Path, relative: str, sha: str, prefix: str) -> Path:
    if not isinstance(sha, str) or SHA256.fullmatch(sha) is None:
        raise ValueError("expected SHA-256 must be 64 lowercase hex digits")
    path = _file(root, relative, prefix)
    if sha256_file(path) != sha:
        raise ValueError(f"frozen hash mismatch: {relative}")
    return path


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys, parse_constant=_bad_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _trace(root: Path, relative: str, sha: str, steps: int) -> dict[str, np.ndarray]:
    path = _pinned(root, relative, sha, "runs")
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(TRACE_KEYS) or set(archive.files) != TRACE_KEYS:
            raise ValueError(f"G0 trace keys differ: {relative}")
        arrays = {key: archive[key] for key in TRACE_KEYS}
    if arrays["initial_stack"].shape != (4, 84, 84) or arrays["initial_stack"].dtype != np.uint8:
        raise ValueError("G0 initial pixel stack is malformed")
    for key in PIXELS:
        if arrays[key].shape != (steps, 84, 84) or arrays[key].dtype != np.uint8:
            raise ValueError(f"G0 pixel frames are malformed: {key}")
    for key in ACTIONS:
        array = arrays[key]
        if array.shape != (steps, 3) or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError(f"G0 actions are malformed: {key}")
    if not np.allclose(arrays["proposed_native_action"], arrays["executed_native_action"], atol=1e-6, rtol=0) or (
        not np.allclose(arrays["commanded_official_action"], arrays["raw_official_action"], atol=1e-6, rtol=0)
    ):
        raise ValueError("G0 recorded actions disagree")
    for key in SCALARS:
        array = arrays[key]
        if array.shape != (steps,):
            raise ValueError(f"G0 scalar shape differs: {key}")
        if key in BOOLEAN and array.dtype.kind != "b":
            raise ValueError(f"G0 boolean is malformed: {key}")
        if key in INTEGER and array.dtype.kind not in "iu":
            raise ValueError(f"G0 integer is malformed: {key}")
        if key == "finish_phase" and array.dtype.kind not in "US":
            raise ValueError("G0 finish phase is malformed")
        if key not in BOOLEAN | INTEGER | {"finish_phase"} and (
            array.dtype.kind != "f" or (key != "directed_delta" and not np.isfinite(array).all())
            or (key == "directed_delta" and np.isinf(array).any())
        ):
            raise ValueError(f"G0 telemetry is malformed: {key}")
    if arrays["raw_finish_phase_bits"].shape != (steps, 4) or arrays["raw_finish_phase_bits"].dtype != np.uint8:
        raise ValueError("G0 raw phase flags are malformed")
    valid_raw = np.arange(4)[None, :] < arrays["raw_frames"][:, None]
    bits = arrays["raw_finish_phase_bits"]
    if np.any(bits[~valid_raw] != 255) or np.any((bits[valid_raw] > 31) & (bits[valid_raw] != 255)):
        raise ValueError("G0 raw phase padding or flags are malformed")
    if not np.array_equal(arrays["initial_stack"][-1], arrays["observation_frame"][0]) or (
        not np.array_equal(arrays["observation_frame"][1:], arrays["next_frame"][:-1])
    ):
        raise ValueError("G0 pixel observation/next indexing is inconsistent")
    if (np.any(arrays["new_tiles"] < 0) or np.any(arrays["raw_frames"] < 1)
            or np.any(arrays["raw_frames"] > 4) or np.any(np.diff(arrays["progress"]) < -1e-6)
            or np.any(arrays["progress"] < 0) or np.any(arrays["progress"] > 1)
            or np.any(arrays["damage"] < 0) or np.any(arrays["damage"] > 1)
            or np.any(np.diff(arrays["damage"]) < -1e-6)):
        raise ValueError("G0 telemetry bounds or ordering are inconsistent")
    if np.any(arrays["finished"][:-1] | arrays["terminated"][:-1] | arrays["truncated"][:-1]):
        raise ValueError("G0 trace continues after a terminal decision")
    if not (arrays["terminated"][-1] or arrays["truncated"][-1]):
        raise ValueError("G0 complete trace has no terminal decision")
    return arrays


def _check_summary(row: dict, arrays: dict[str, np.ndarray], threshold: float, rules: dict) -> None:
    """Compare the frozen first-event chronology and missingness to actual trace columns."""
    summary = row["summary"]
    first_step = None
    first_events: list[str] = []
    missing = []
    motion_streak = tile_streak = negative_streak = 0
    for index in range(row["steps"]):
        distance = arrays["centerline_distance_m"][index]
        delta = arrays["directed_delta"][index]
        far = bool(distance > threshold)
        if (np.isnan(delta) and not far) or arrays["finish_phase"][index] == "unknown":
            missing.append(index)
        motion_streak = motion_streak + 1 if np.isfinite(delta) and delta <= rules["directed_delta_epsilon"] else 0
        tile_streak = tile_streak + 1 if arrays["new_tiles"][index] == 0 else 0
        negative_streak = negative_streak + 1 if arrays["summed_reward"][index] < 0 else 0
        events = []
        if far:
            events.append("centerline_distance_exceeds_proxy")
        if arrays["contact"][index]:
            events.append("observed_contact")
        if motion_streak == rules["stall_window"]:
            events.append("low_directed_motion")
        if tile_streak == rules["tile_window"]:
            events.append("no_new_tiles")
        if events and first_step is None:
            first_step, first_events = index, sorted(events)
    status = (
        "unknown" if missing and (first_step is None or missing[0] <= first_step) else
        "mixed" if len(first_events) > 1 else "observed" if first_events else "none"
    )
    outcome = summary.get("outcome")
    if outcome not in {"finished", "off_track", "crash", "out_of_bounds", "task_timeout", "unknown"}:
        raise ValueError("G0 ledger outcome is not a completed episode")
    qualified_nonfinish = bool(arrays["finish_qualified"][-1] and outcome != "finished")
    if (summary.get("first_observed_step") != first_step
            or summary.get("first_observed_events") != first_events
            or summary.get("first_event_status") != status
            or summary.get("missing_steps") != missing
            or summary.get("final_negative_streak") != negative_streak
            or summary.get("qualified_nonfinish") is not qualified_nonfinish
            or summary.get("unassessed_precursors") != (["finish_phase"] if qualified_nonfinish else [])
            or bool(arrays["finished"][-1]) != (outcome == "finished")):
        raise ValueError("G0 ledger summary differs from the hashed trace")


def _anchors(row: dict, arrays: dict[str, np.ndarray], threshold: float) -> dict[str, tuple[int | None, str | None]]:
    n = row["steps"]

    def first(mask: np.ndarray, missing: str) -> tuple[int | None, str | None]:
        indices = np.flatnonzero(mask)
        return (int(indices[0]), None) if len(indices) else (None, missing)

    first_observed = row["summary"]["first_observed_step"]
    tile_indices = np.flatnonzero(arrays["new_tiles"] > 0)
    after_tile = int(tile_indices[-1]) + 1 if len(tile_indices) else None
    return {
        "first_observed": (first_observed, None if first_observed is not None else "no_observed_first_event"),
        "first_contact": first(arrays["contact"], "no_observed_contact"),
        "first_centerline_far": first(arrays["centerline_distance_m"] > threshold, "no_centerline_far_proxy"),
        "after_last_new_tile": (
            after_tile if after_tile is not None and after_tile < n else None,
            "no_new_tile_visit" if after_tile is None else "no_decision_after_last_new_tile" if after_tile == n else None,
        ),
        "first_qualified": first(arrays["finish_qualified"], "not_qualified"),
        "terminal": (n - 1, None),
    }


def _stack_at(arrays: dict[str, np.ndarray], index: int) -> np.ndarray:
    if index < 4:
        return np.concatenate((arrays["initial_stack"][index:], arrays["next_frame"][:index]))
    return arrays["next_frame"][index - 4:index]


def analyze(
    root: Path, freeze_path: str, freeze_sha256: str, output_root: str, *,
    source: dict[str, str] = SOURCE,
) -> dict:
    """Validate every frozen byte and trace before creating the exclusive derived run."""
    root = root.resolve()
    output = Path(output_root) if isinstance(output_root, str) else Path()
    if (not isinstance(output_root, str) or "\\" in output_root or len(output.parts) != 2
            or output.parts[0] != "runs" or output.parts[1] in (".", "..")
            or output.is_absolute() or output.as_posix() != output_root
            or output.parts[1] == Path(source["manifest_path"]).parts[1]
            or (root / "runs").is_symlink() or not (root / "runs").is_dir()
            or (root / output).exists() or (root / output).is_symlink()):
        raise ValueError("output must be a new exclusive runs/<name> directory outside G0")
    freeze_prefix = Path(freeze_path).parts[0] if isinstance(freeze_path, str) and freeze_path else ""
    if freeze_prefix not in ("experiments", "runs"):
        raise ValueError("freeze must be a repository-relative experiments/ or runs/ JSON")
    freeze = _json(_pinned(root, freeze_path, freeze_sha256, freeze_prefix))
    # JSON serialization distinguishes true from 1, unlike Python dict equality.
    if json.dumps(freeze, sort_keys=True) != json.dumps(freeze_template(source), sort_keys=True):
        raise ValueError("analysis freeze differs from the exact fixed-window schema")

    manifest = _json(_pinned(root, source["manifest_path"], source["manifest_sha256"], "runs"))
    protocol = _json(_pinned(root, source["protocol_path"], source["protocol_sha256"], "experiments"))
    ledger = _pinned(root, source["cells_path"], source["cells_sha256"], "runs")
    if (manifest.get("format") != "haic-rlpd-g0-diagnostic-result-v1"
            or manifest.get("role") != "TRAIN-only-failure-diagnostic" or manifest.get("ranked") is not False
            or manifest.get("protocol_path") != source["protocol_path"]
            or manifest.get("protocol_sha256") != source["protocol_sha256"]
            or manifest.get("cells_sha256") != source["cells_sha256"]
            or manifest.get("cell_count") != 24 or manifest.get("geometry_count") != 12
            or protocol.get("format") != "haic-rlpd-g0-diagnostic-v1" or protocol.get("status") != "frozen"
            or protocol.get("partition") != "TRAIN" or protocol.get("interventions") is not False
            or protocol.get("learner_updates") != 0
            or protocol.get("geometry_audit_sha256") != manifest.get("geometry_audit_sha256")):
        raise ValueError("G0 manifest/protocol identity or TRAIN-only status differs")
    cells = protocol.get("cells")
    actors = protocol.get("actors")
    if (not isinstance(cells, list) or len(cells) != 12 or not isinstance(actors, list) or len(actors) != 2
            or len({(cell.get("track_id"), cell.get("geometry_seed")) for cell in cells}) != 12
            or any(cell.get("partition") != "TRAIN" or cell.get("obstacles") is not True for cell in cells)
            or any(not isinstance(actor.get("id"), str) or ACTOR_ID.fullmatch(actor["id"]) is None
                   or actor.get("action_mode") != "exported_tanh_mean" for actor in actors)
            or actors[0]["id"] == actors[1]["id"]):
        raise ValueError("G0 must retain all 12 paired TRAIN geometries and two exported actors")
    rows = [json.loads(line, object_pairs_hook=_unique_keys, parse_constant=_bad_constant)
            for line in ledger.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 24 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("G0 must retain all 24 ledger rows")
    run_relative = Path(source["manifest_path"]).parent
    trace_paths = []
    for index, row in enumerate(rows):
        cell, actor = cells[index // 2], actors[index % 2]
        trace_name = f"traces/seed-{cell['geometry_seed']}-track-{cell['track_id']}-{actor['id']}.npz"
        if (row.get("partition") != "TRAIN" or row.get("track_id") != cell["track_id"]
                or row.get("geometry_seed") != cell["geometry_seed"] or row.get("actor_id") != actor["id"]
                or row.get("actor_sha256") != actor.get("sha256")
                or row.get("trace_path") != trace_name
                or type(row.get("steps")) is not int or not 1 <= row["steps"] <= 2000
                or not isinstance(row.get("summary"), dict)
                or not isinstance(row.get("road_centerline_sha256"), str)
                or SHA256.fullmatch(row["road_centerline_sha256"]) is None):
            raise ValueError("G0 ledger differs from the fixed paired-cell schedule")
        trace_paths.append((run_relative / trace_name).as_posix())
        if index % 2 and rows[index - 1]["road_centerline_sha256"] != row["road_centerline_sha256"]:
            raise ValueError("G0 paired road-centerline hashes disagree")
    if (len(set(trace_paths)) != 24 or sum(row["steps"] for row in rows) != manifest.get("decisions_spent")
            or sum(row["driven_raw_frames"] for row in rows) != manifest.get("driven_raw_frames")
            or sum(row["reset_initial_raw_frames"] for row in rows) != manifest.get("reset_initial_raw_frames")
            or sum(row["reset_noop_raw_frames"] for row in rows) != manifest.get("reset_noop_raw_frames")
            or any(sum(row["summary"].get("outcome") == "finished" for row in rows[side::2]) != 3
                   for side in range(2))):
        raise ValueError("G0 counts or six finished controls disagree with the frozen run")

    # Pin ALL source trace bytes before reading any arrays or creating output files.
    for row, trace_path in zip(rows, trace_paths):
        _pinned(root, trace_path, row["trace_sha256"], "runs")
    threshold = protocol["centerline_far_threshold_m"]
    rules = protocol["event_rules"]
    plans = []
    for row, trace_path in zip(rows, trace_paths):
        arrays = _trace(root, trace_path, row["trace_sha256"], row["steps"])
        if int(arrays["raw_frames"].sum()) != row["driven_raw_frames"]:
            raise ValueError("G0 trace raw-frame count differs from ledger")
        if (not np.isclose(arrays["summed_reward"].sum(), row.get("raw_reward_sum"), atol=1e-6, rtol=0)
                or not np.isclose(arrays["progress"].max(), row.get("max_progress"), atol=1e-8, rtol=0)):
            raise ValueError("G0 trace reward/progress differs from ledger")
        _check_summary(row, arrays, threshold, rules)
        plans.append(_anchors(row, arrays, threshold))

    output_path = root / output
    output_path.mkdir(exist_ok=False)
    (output_path / "tiles").mkdir()
    written = []
    for index, (row, trace_path, plan) in enumerate(zip(rows, trace_paths, plans)):
        # Recheck bytes on the second pass: do not silently export a replaced trace.
        arrays = _trace(root, trace_path, row["trace_sha256"], row["steps"])
        for anchor in ANCHORS:
            step, reason = plan[anchor]
            item = {
                "geometry_seed": row["geometry_seed"], "track_id": row["track_id"],
                "actor_id": row["actor_id"], "outcome": row["summary"]["outcome"],
                "source_trace_path": trace_path, "source_trace_sha256": row["trace_sha256"],
                "anchor": anchor, "anchor_step": step, "missing_anchor_reason": reason,
                "first_observed_events": row["summary"]["first_observed_events"],
                "first_event_status": row["summary"]["first_event_status"],
                "window_start": None, "window_end_inclusive": None,
                "left_clipped_decisions": None, "right_clipped_decisions": None,
                "missing_diagnostic_steps": [], "missing_directed_delta_steps": [],
                "missing_finish_phase_steps": [], "missing_raw_phase_steps": [],
                "observations": None, "contact_sheet_steps": [], "tiles_path": None, "tiles_sha256": None,
            }
            if step is not None:
                start, end = max(0, step - 20), min(row["steps"] - 1, step + 20)
                indices = np.arange(start, end + 1, dtype=np.int32)
                sheet_steps = sorted(set(range(start, end + 1, 5)) | {step, end})
                tile_name = f"tiles/row-{index:02d}-{anchor}.npz"
                tile_path = output_path / tile_name
                sliced = {key: arrays[key][start:end + 1] for key in TRACE_KEYS - {"initial_stack"}}
                sheet = np.stack((arrays["observation_frame"][sheet_steps], arrays["next_frame"][sheet_steps]), axis=1)
                with tile_path.open("xb") as stream:
                    np.savez_compressed(
                        stream, decision_steps=indices, observation_stack_at_start=_stack_at(arrays, start),
                        contact_sheet_steps=np.asarray(sheet_steps, dtype=np.int32), contact_sheet_tiles=sheet,
                        **sliced,
                    )
                item.update({
                    "window_start": start, "window_end_inclusive": end,
                    "left_clipped_decisions": max(0, 20 - step),
                    "right_clipped_decisions": max(0, step + 20 - (row["steps"] - 1)),
                    "missing_diagnostic_steps": [i for i in row["summary"]["missing_steps"] if start <= i <= end],
                    "missing_directed_delta_steps": (np.flatnonzero(np.isnan(arrays["directed_delta"][start:end + 1])) + start).tolist(),
                    "missing_finish_phase_steps": (np.flatnonzero(arrays["finish_phase"][start:end + 1] == "unknown") + start).tolist(),
                    "missing_raw_phase_steps": (np.flatnonzero(np.any(
                        (arrays["raw_finish_phase_bits"][start:end + 1] == 255)
                        & (np.arange(4)[None, :] < arrays["raw_frames"][start:end + 1, None]), axis=1,
                    )) + start).tolist(),
                    "observations": {
                        "contact_decisions": int(arrays["contact"][start:end + 1].sum()),
                        "centerline_far_proxy_decisions": int((arrays["centerline_distance_m"][start:end + 1] > threshold).sum()),
                        "new_tiles": int(arrays["new_tiles"][start:end + 1].sum()),
                        "summed_raw_reward": float(arrays["summed_reward"][start:end + 1].sum()),
                        "qualified_decisions": int(arrays["finish_qualified"][start:end + 1].sum()),
                    },
                    "contact_sheet_steps": sheet_steps, "tiles_path": tile_name,
                    "tiles_sha256": sha256_file(tile_path),
                })
            written.append(item)
    windows_path = output_path / "windows.jsonl"
    with windows_path.open("x", encoding="utf-8") as stream:
        for item in written:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    for key, prefix in (("manifest", "runs"), ("protocol", "experiments"), ("cells", "runs")):
        _pinned(root, source[f"{key}_path"], source[f"{key}_sha256"], prefix)
    receipt = {
        "format": "haic-rlpd-g0-fixed-windows-result-v1", "role": "TRAIN-only-observational-diagnostic",
        "source": source, "freeze_path": freeze_path, "freeze_sha256": freeze_sha256,
        "analysis_source_sha256": freeze["analysis_source_sha256"],
        "windows_path": "windows.jsonl", "windows_sha256": sha256_file(windows_path),
        "row_count": 24, "geometry_clusters": 12, "finished_controls": 6,
        "window_count": len(written), "missing_anchor_count": sum(item["anchor_step"] is None for item in written),
        "window_radius_decisions": 20, "contact_sheet_stride_decisions": 5,
        "anchor_definitions": freeze["anchor_definitions"],
        "contact_sheet_sampling": freeze["contact_sheet_sampling"],
        "pixel_contract": "decision_steps are zero-based; each tile pair is [observation_frame,next_frame]; observation_stack_at_start is the full 4-frame policy input at window_start; every original per-decision trace column is sliced without reindexing",
        "input_boundary": "pixel stack was the only policy input; action/telemetry columns are diagnostic output only",
        "manual_rubric": freeze["manual_rubric"], "labels_assigned": False,
        "causal_claim": None,
    }
    with (output_path / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-freeze-template", action="store_true")
    parser.add_argument("--freeze")
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    if args.print_freeze_template:
        if any((args.freeze, args.freeze_sha256, args.output_root)):
            parser.error("--print-freeze-template cannot be combined with analysis arguments")
        print(json.dumps(freeze_template(), sort_keys=True, indent=2))
        return
    if not all((args.freeze, args.freeze_sha256, args.output_root)):
        parser.error("--freeze, --freeze-sha256 and --output-root are required")
    print(json.dumps(analyze(ROOT, args.freeze, args.freeze_sha256, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
