"""Offline r6 replay exposure audit from frozen TRAIN provenance only.

No environment resets, training, blind/held-out data or checkpoint writes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.diagnose_drq_geometry_regression import CATALOG_SHA, PROTOCOL, PROTOCOL_SHA, ROOT, RUN, digest


SOURCE_SUMMARY = Path("experiments/drqv2-geometry-augmentation-v1-training-summary.json")
WINDOWS = ((1, 8192), (8193, 16384), (16385, 24576), (24577, 32768))


def progress_bin(progress: float | None) -> str:
    if progress is None:
        return "budget_stop_unknown"
    if progress < 0.2:
        return "below_0.2"
    if progress < 0.9:
        return "0.2_to_below_0.9"
    return "at_least_0.9"


def thirds(step: int, length: int) -> str:
    if step * 3 < length:
        return "early"
    if step * 3 < length * 2:
        return "middle"
    return "late"


def _counter(records: list[dict], field: str) -> dict:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def audit_run(root: Path, run: dict, seed_families: dict, source_success: set[int]) -> dict:
    run_dir = root / run["run_dir"]
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text())
    assert result["completed"] is True and result["study_gradient_steps"] == 22768
    assert result["additional_online_steps"] == 32768
    assert result["study_protocol_sha256"] == PROTOCOL_SHA and result["catalog_sha256"] == CATALOG_SHA
    assert result["source_seed"] == run["source_seed"] and result["variant"] == run["variant"]
    assert result["teacher_data_loaded"] is False and result["online_replay_only"] is True
    episode_path = run_dir / "episodes.jsonl"
    metrics_path = run_dir / "step-metrics.jsonl"
    resets, endings = {}, {}
    for line in episode_path.read_text().splitlines():
        row = json.loads(line)
        eid = int(row["episode_id"])
        if row["event"] == "reset":
            assert eid not in resets and row["seed"] in seed_families
            assert row["geometry_family"] == seed_families[row["seed"]]
            resets[eid] = row
        elif row["event"] in ("end", "budget-stop"):
            assert eid not in endings
            endings[eid] = row
        else:
            raise ValueError(f"unexpected episode ledger event: {row['event']}")
    assert set(resets) == set(endings) == set(range(len(resets)))
    rows = []
    episode_steps = Counter()
    samples_by_episode = Counter()
    for line in metrics_path.read_text().splitlines():
        row = json.loads(line)
        step = len(rows) + 1
        assert row["additional_online_step"] == step and row["episode_id"] in resets
        reset = resets[row["episode_id"]]
        assert (row["geometry_seed"], row["track_id"], row["geometry_family"]) == (
            reset["seed"], reset["track_id"], reset["geometry_family"])
        assert row["episode_step"] == episode_steps[row["episode_id"]]
        assert row["source_seed"] == run["source_seed"] and row["variant"] == run["variant"]
        assert row["update_performed"] == (step > 10000)
        assert row["gradient_steps"] == max(0, step - 10000)
        assert row["online_replay_size"] == step
        episode_steps[row["episode_id"]] += 1
        rows.append({"step": step, "episode_id": row["episode_id"], "episode_step": row["episode_step"],
                     "seed": row["geometry_seed"], "track": row["track_id"], "family": row["geometry_family"]})
    assert len(rows) == 32768
    for eid, terminal in endings.items():
        assert episode_steps[eid] == terminal["steps"] and terminal["additional_online_step"] == (
            resets[eid]["additional_online_step"] + terminal["steps"])
        assert terminal["event"] != "budget-stop" or eid == len(resets) - 1
    sampled_path = root / result["sample_trace_path"]
    assert digest(sampled_path) == result["sample_trace_sha256"] == result["candidates"][-1]["sample_trace_sha256"]
    assert result["sample_trace_rows"] == 22768
    sampled = defaultdict(Counter)
    sampled_episodes_by_window = defaultdict(set)
    all_sampled_ids = set()
    with np.load(sampled_path, allow_pickle=False) as arrays:
        sequence = arrays["sequence_ids"]
        episode = arrays["episode_ids"]
        geometry = arrays["geometry_seeds"]
        track = arrays["track_ids"]
        source = arrays["source_codes"]
        assert sequence.shape == episode.shape == geometry.shape == track.shape == source.shape == (22768, 64)
        assert np.all(source == 0) and int(arrays["source_seed"][0]) == run["source_seed"]
        assert bytes(arrays["variant"]).decode("ascii") == run["variant"]
        assert bytes(arrays["study_protocol_sha256"]).decode("ascii") == PROTOCOL_SHA
        for k, (seq_row, eid_row, seed_row, track_row) in enumerate(zip(sequence, episode, geometry, track)):
            update_step = k + 10001
            window = next(index for index, (begin, end) in enumerate(WINDOWS) if begin <= update_step <= end)
            ids = seq_row.astype(np.int64)
            assert np.all(ids >= 0) and np.all(ids < update_step)
            for q, eid, geo, track_id in zip(ids.tolist(), eid_row.tolist(), seed_row.tolist(), track_row.tolist()):
                inserted = rows[q]
                assert (inserted["episode_id"], inserted["seed"], inserted["track"]) == (eid, geo, track_id)
                key = str(window)
                sampled[key][inserted["family"]] += 1
                sampled[key]["total"] += 1
                sampled[key]["source_success_road_geometry"] += int(geo in source_success)
                sampled[key]["source_success_exact_track1"] += int(geo in source_success and track_id == 1)
                sampled[key]["warmup_insertion"] += int(q < 10000)
                sampled[key]["terminal_last_20_completed"] += int(
                    endings[eid]["event"] == "end" and inserted["episode_step"] >= endings[eid]["steps"] - 20)
                sampled[key]["source_success_seed_" + str(geo)] += int(geo in source_success)
                sampled_episodes_by_window[key].add(eid)
                samples_by_episode[eid] += 1
                all_sampled_ids.add(q)
    assert sum(value["total"] for value in sampled.values()) == 22768 * 64

    insert = defaultdict(Counter)
    for row in rows:
        key = str(next(index for index, (start, end) in enumerate(WINDOWS) if start <= row["step"] <= end))
        end = endings[row["episode_id"]]
        insert[key][row["family"]] += 1
        insert[key]["total"] += 1
        insert[key]["source_success_road_geometry"] += int(row["seed"] in source_success)
        insert[key]["source_success_exact_track1"] += int(row["seed"] in source_success and row["track"] == 1)
        insert[key]["warmup_insertion"] += int(row["step"] <= 10000)
        insert[key]["early_middle_late_" + thirds(row["episode_step"], end["steps"])] += 1
        if end["event"] == "end":
            insert[key]["completed_episode_terminal_progress_" + progress_bin(end["progress"])] += 1
            insert[key]["completed_episode_finished"] += int(end["finished"])
            insert[key]["terminal_last_20_completed"] += int(row["episode_step"] >= end["steps"] - 20)
            insert[key]["off_track_episode_last_20"] += int(
                end["retire_reason"] == "off_track" and row["episode_step"] >= end["steps"] - 20)
        else:
            insert[key]["budget_stop_unknown_progress"] += 1

    end_records = [r for r in endings.values() if r["event"] == "end"]
    return {
        "run": run["run_dir"], "source_seed": run["source_seed"], "variant": run["variant"],
        "result_sha256": digest(result_path), "episode_ledger_sha256": digest(episode_path),
        "step_metrics_sha256": digest(metrics_path), "sample_trace_sha256": digest(sampled_path),
        "final_actor_sha256": result["candidates"][-1]["actor_sha256"],
        "final_checkpoint_sha256": result["candidates"][-1]["checkpoint_sha256"],
        "source_checkpoint_sha256": result["source_checkpoint_sha256"],
        "insertion_windows": {k: dict(insert[k]) for k in sorted(insert)},
        "sample_windows": {k: {**{key: count for key, count in sampled[k].items() if not key.startswith("source_success_seed_")},
                               "distinct_source_success_seeds": sum(1 for key, count in sampled[k].items()
                                                                    if key.startswith("source_success_seed_") and count > 0),
                               "distinct_episodes": len(sampled_episodes_by_window[k])}
                           for k in sorted(sampled)},
        "source_success_train_geometry_seeds": sorted(source_success),
        "source_success_geometry_episodes": sum(resets[e]["seed"] in source_success for e in resets),
        "source_success_exact_track1_episodes": sum(resets[e]["seed"] in source_success and resets[e]["track_id"] == 1 for e in resets),
        "distinct_source_success_geometries_encountered": len({r["seed"] for r in resets.values() if r["seed"] in source_success}),
        "source_success_train_geometries_total": len(source_success),
        "episode_count": len(resets), "completed_episodes": len(end_records),
        "completed_finished": sum(r["finished"] for r in end_records),
        "completed_high_terminal_progress": sum(r["progress"] >= 0.9 for r in end_records),
        "completed_terminal_progress_bins": _counter([{"bin": progress_bin(r["progress"])} for r in end_records], "bin"),
        "source_success_geometry_finished": sum(r["finished"] and resets[e]["seed"] in source_success
                                                for e, r in endings.items() if r["event"] == "end"),
        "unique_sampled_sequence_ids": len(all_sampled_ids),
        "unique_sampled_episodes": sum(count > 0 for count in samples_by_episode.values()),
        "sample_slots": 22768 * 64,
    }


def analyze(root: Path) -> dict:
    protocol = json.loads((root / PROTOCOL).read_text())
    assert digest(root / PROTOCOL) == PROTOCOL_SHA
    catalog_path = root / protocol["catalog_path"]
    assert digest(catalog_path) == CATALOG_SHA
    catalog = json.loads(catalog_path.read_text())
    seeds = {row["geometry_seed"]: row["family"] for row in catalog["train"]}
    assert len(seeds) == 120 and set(seeds) == set(protocol["training_pool"]["geometry_seeds"])
    assert not set(seeds) & set(protocol["diagnostics"]["geometry_seeds"])
    summary_path = root / SOURCE_SUMMARY
    summary = json.loads(summary_path.read_text())
    assert summary["catalog_sha256"] == CATALOG_SHA
    original = {row["geometry_seed"]: row for row in summary["geometry"] if row["partition"] == "train"}
    assert set(original) == set(seeds)
    runs = []
    for run in protocol["runs"]:
        source_success = {seed for seed, record in original.items() if record["by_source"][str(run["source_seed"])]["finished"]}
        runs.append(audit_run(root, run, seeds, source_success))
    assert len(runs) == 6
    return {"scope": "TRAIN-only frozen replay; no same-state observation overlap or per-step progress retained",
            "protocol_sha256": PROTOCOL_SHA, "catalog_sha256": CATALOG_SHA,
            "source_train_diagnostic_summary_sha256": digest(summary_path),
            "window_decisions_inclusive": WINDOWS,
            "progress_bins": "terminal completed-episode progress only; [0,.2), [.2,.9), [.9,1] from frozen geometry summary",
            "near_terminal": "last 20 inserted decisions only for fully ended episode; off_track is the wrapper negative-reward counter, not geometric off-road",
            "sample_interpretation": "22768 update rows x64 repeated n-step starting transitions; prefix step-16384 trace is cumulative, not additional; sampled slots are not inserted decisions",
            "source_success_interpretation": "prior deterministic source actor single run on TRAIN road at track1; track1 exact road/obstacle cell, tracks2-4 geometry only; never a same-observation match to noisy changing r6 policy",
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(ROOT)
    output = args.output.resolve()
    output.relative_to((ROOT / RUN).resolve())
    assert not output.exists() and output.parent.is_dir() and not output.parent.is_symlink()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"sha256": digest(output), "runs": len(result["runs"]),
                      "output": str(output), "sample_slots_total": sum(r["sample_slots"] for r in result["runs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
