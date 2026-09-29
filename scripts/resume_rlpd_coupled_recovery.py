"""Source-bound missing-slot continuation, not exact resume, of coupled recovery v1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np

from scripts import collect_rlpd_coupled_recovery as parent

ROOT = parent.ROOT
PROTOCOL = ROOT / "experiments/rlpd-coupled-recovery-r2.json"
OUTPUT = ROOT / "runs/rlpd-coupled-recovery-r2"
PARENT_OUTPUT = parent.OUTPUT
CONTRACT = {"copy_complete": 28, "new_slots": 14, "total_slots": 42,
            "preserve_parent_episode_ids_and_bytes": True, "exact_resume": False,
            "fresh_cells": 0, "threads": 1,
            "reattempt": "rlpd-seed50-seed4272000011-h12",
            "prior_open_branch_cost": "unknown additional 0..2000 decisions; consumed duplicate attempt"}


def key(row):
    return row["policy_id"], int(row["geometry_seed"]), int(row["horizon"])


def schedule(cases, rows):
    wanted = [(case, horizon, index*3 + offset) for index, case in enumerate(cases)
              for offset, horizon in enumerate((0, 12, 25))]
    expected = {(case["policy_id"], case["geometry_seed"], horizon): episode_id
                for case, horizon, episode_id in wanted}
    seen = set()
    for row in rows:
        identity = key(row)
        if identity in seen or identity not in expected or row["episode_id"] != expected[identity]:
            raise parent.diagnosis.DiagnosisError("duplicate, unexpected, or renumbered completed episode")
        seen.add(identity)
    return [(case, horizon, episode_id) for case, horizon, episode_id in wanted
            if (case["policy_id"], case["geometry_seed"], horizon) not in seen]


def freeze():
    old = parent.preflight()
    rows = parent.diagnosis._read_jsonl(PARENT_OUTPUT / "episodes.jsonl")
    intents = parent.diagnosis._read_jsonl(PARENT_OUTPUT / "reset-intents.jsonl")
    receipt = parent.diagnosis._read_json(PARENT_OUTPUT / "external-timeout-receipt.json")
    missing = schedule(old["cases"], rows)
    if len(rows) != 28 or len(intents) != 29 or len(missing) != 14:
        raise parent.diagnosis.DiagnosisError("parent partial census differs from authorized continuation")
    if receipt["completed_episodes"] != 28 or receipt["reset_intents"] != 29 or receipt["dataset_manifest_available"]:
        raise parent.diagnosis.DiagnosisError("parent external interruption receipt changed")
    hashes = dict(old["source_sha256"])
    paths = [parent.PROTOCOL, Path(__file__).resolve(), ROOT / "tests/test_resume_rlpd_coupled_recovery.py"]
    paths += [PARENT_OUTPUT / name for name in ("metadata.json", "episodes.jsonl", "reset-intents.jsonl", "external-timeout-receipt.json")]
    for row in rows:
        path = PARENT_OUTPUT / row["path"]
        if path.parent != PARENT_OUTPUT or parent.branch._sha(path) != row["sha256"]:
            raise parent.diagnosis.DiagnosisError("completed parent trace mismatch")
        paths.append(path)
    hashes.update({p.relative_to(ROOT).as_posix(): parent.branch._sha(p) for p in paths})
    protocol = {"format": "haic-rlpd-coupled-recovery-continuation-v2", "status": "frozen",
                "created_at": parent.utc(), "contract": CONTRACT, "parent_contract": parent.CONTRACT,
                "parent_protocol_sha256": parent.branch._sha(parent.PROTOCOL), "runtime": old["runtime"],
                "source_sha256": hashes, "copied_episodes": rows, "cases": old["cases"],
                "missing_slots": [{"policy_id": case["policy_id"], "geometry_seed": case["geometry_seed"],
                                   "horizon": horizon, "episode_id": episode_id}
                                  for case, horizon, episode_id in missing], "parent_reset_intents": intents,
                "parent_interruption_receipt": receipt}
    parent.branch._write(PROTOCOL, protocol, exclusive=True)
    return {"status": "frozen", "copied_episodes": 28, "new_slots": 14, "protocol_sha256": parent.branch._sha(PROTOCOL)}


def preflight():
    p = parent.diagnosis._read_json(PROTOCOL)
    if p["status"] != "frozen" or p["contract"] != CONTRACT or p["parent_contract"] != parent.CONTRACT:
        raise parent.diagnosis.DiagnosisError("continuation contract drift")
    for relative, expected in p["source_sha256"].items():
        if parent.branch._sha(ROOT / relative) != expected:
            raise parent.diagnosis.DiagnosisError(f"continuation source drift: {relative}")
    old = parent.preflight()
    rows = parent.diagnosis._read_jsonl(PARENT_OUTPUT / "episodes.jsonl")
    missing = schedule(old["cases"], rows)
    actual = [{"policy_id": case["policy_id"], "geometry_seed": case["geometry_seed"],
               "horizon": horizon, "episode_id": episode_id} for case, horizon, episode_id in missing]
    if rows != p["copied_episodes"] or actual != p["missing_slots"] or old["cases"] != p["cases"] or old["runtime"] != p["runtime"]:
        raise parent.diagnosis.DiagnosisError("continuation cohort/runtime drift")
    return p


def copy_completed(p):
    for row in p["copied_episodes"]:
        source, target = PARENT_OUTPUT / row["path"], OUTPUT / row["path"]
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        if parent.branch._sha(target) != row["sha256"]:
            raise parent.diagnosis.DiagnosisError("copied episode bytes changed")
        parent.branch._append(OUTPUT / "episodes.jsonl", row)


def arrays_at(row):
    path = OUTPUT / row["path"]
    if parent.branch._sha(path) != row["sha256"]:
        raise parent.diagnosis.DiagnosisError("continuation trace changed")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def aggregate(p, rows, elapsed):
    if len(rows) != 42 or schedule(p["cases"], rows):
        raise parent.diagnosis.DiagnosisError("incomplete final census")
    rows = sorted(rows, key=lambda row: row["episode_id"])
    by_key = {key(row): row for row in rows}
    unique = {"failure": set(), "finish-control": set()}
    geometries = {s: set() for s in unique}
    pairs = []
    for row in rows:
        arrays = arrays_at(row)
        if int(arrays["recovery_mask"].sum()) != row["accepted_transitions"]:
            raise parent.diagnosis.DiagnosisError("accepted transition count mismatch")
        for index in np.flatnonzero(arrays["recovery_mask"]):
            identity = (str(arrays["observation_sha256"][index]), arrays["executed_action"][index].tobytes())
            unique[row["stratum"]].add(identity)
            geometries[row["stratum"]].add(row["geometry_seed"])
    for case in p["cases"]:
        control = by_key[(case["policy_id"], case["geometry_seed"], 0)]
        for horizon in (12, 25):
            treatment = by_key[(case["policy_id"], case["geometry_seed"], horizon)]
            known = not control["censored"] and not treatment["censored"]
            pairs.append({"policy_id": case["policy_id"], "geometry_seed": case["geometry_seed"], "stratum": case["stratum"],
                          "horizon": horizon, "control_finished": control["finished"], "treatment_finished": treatment["finished"],
                          "finish_pair_known": known, "rescued": bool(known and not control["finished"] and treatment["finished"]),
                          "harmed": bool(known and control["finished"] and not treatment["finished"]),
                          "local_recovery_qualified": treatment["local_recovery_qualified"],
                          "baseline_local_qualified": control["local_recovery_qualified"],
                          "current_baseline_anchor_overspeed": control["current_anchor_curve_entry_overspeed"]})
    accepted = sum(row["accepted_transitions"] for row in rows)
    counts = {"anchors": 14, "episodes": 42, "copied_episodes": 28, "new_episodes": 14,
              "reset_intents": 14, "parent_reset_intents": 29, "total_reset_intents_including_interrupted_attempt": 43,
              "decisions": sum(row["steps"] for row in rows), "new_decisions": sum(row["steps"] for row in rows[28:]),
              "prefix_decisions": sum(row["prefix_decisions"] for row in rows),
              "accepted_transitions": accepted, "unique_accepted_transitions": len(set.union(*unique.values())),
              "accepted_geometries": len(set.union(*geometries.values())),
              "support_by_stratum": {s: {"accepted_transitions": sum(row["accepted_transitions"] for row in rows if row["stratum"] == s),
                                         "unique_accepted_transitions": len(unique[s]), "accepted_geometries": len(geometries[s])} for s in unique},
              "unknown_finish_pairs": sum(not pair["finish_pair_known"] for pair in pairs),
              "rejected_oracle_transitions": sum(min(row["horizon"], max(0, row["steps"]-row["anchor_step"])) for row in rows) - accepted,
              "qualified_branches": sum(row["local_recovery_qualified"] for row in rows if row["horizon"]),
              "rescued": sum(pair["rescued"] for pair in pairs), "harmed": sum(pair["harmed"] for pair in pairs)}
    duration = {}
    for horizon in (12, 25):
        duration[str(horizon)] = {}
        for stratum in unique:
            selected = [pair for pair in pairs if pair["horizon"] == horizon and pair["stratum"] == stratum]
            duration[str(horizon)][stratum] = {"pairs": len(selected), "known_pairs": sum(pair["finish_pair_known"] for pair in selected),
                                              "rescued": sum(pair["rescued"] for pair in selected), "harmed": sum(pair["harmed"] for pair in selected),
                                              "qualified_rescues": sum(pair["rescued"] and pair["local_recovery_qualified"] for pair in selected),
                                              "local_qualified": sum(pair["local_recovery_qualified"] for pair in selected)}
    return {"format": "haic-rlpd-recovery-dataset-v1", "status": "complete", "protocol_sha256": parent.branch._sha(PROTOCOL),
            "parent_protocol_sha256": p["parent_protocol_sha256"], "episodes": rows, "pairs": pairs, "counts": counts,
            "duration_stratum_outcomes": duration, "accepted_geometry_seeds": sorted(set.union(*geometries.values())),
            "current_baselines": [row for row in rows if not row["horizon"]], "elapsed_wall_seconds": elapsed,
            "finished_at": parent.utc(), "exact_resume": False, "deployable_policy": False, "fresh_generalization": False,
            "parent_interruption_receipt": p["parent_interruption_receipt"],
            "interaction_cost": "42 completed trajectories plus unknown 0..2000 decisions from interrupted v1 duplicate attempt",
            "limitation": "Privileged consumed-TRAIN branch upper bound. Local recovery is separate from causal improvement and full finish."}


def interrupted(signum, frame):
    raise RuntimeError(f"continuation interrupted by signal {signum}; partial trace retained, no exact resume")


def run():
    p = preflight()
    if OUTPUT.exists():
        raise parent.diagnosis.DiagnosisError("continuation output exists; refuse overwrite")
    import torch
    from agent import Agent
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    OUTPUT.mkdir()
    started = time.monotonic()
    rows = []
    current = None
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        parent.branch._write(OUTPUT / "metadata.json", {"protocol_sha256": parent.branch._sha(PROTOCOL), "started_at": parent.utc(),
                                                       "runtime": p["runtime"], "threads": 1, "exact_resume": False,
                                                       "copied_episodes": 28, "new_slots": 14, "learner_updates": 0}, exclusive=True)
        parent.branch._write(OUTPUT / "parent-interruption.json", p["parent_interruption_receipt"], exclusive=True)
        copy_completed(p)
        rows = list(p["copied_episodes"])
        actors = {name: Agent(model_path=str(parent.diagnosis._actor_path(name)[0])) for name in parent.diagnosis.RLPD_ACTORS}
        for case, horizon, episode_id in schedule(p["cases"], rows):
            preflight()
            stem = f"{case['policy_id']}-seed{case['geometry_seed']}-h{horizon}"
            current = {"stem": stem, "episode_id": episode_id, "timestamp": parent.utc(), "geometry_seed": case["geometry_seed"],
                       "track_id": 1, "horizon": horizon, "duplicate_interrupted_v1_attempt": stem == CONTRACT["reattempt"]}
            parent.branch._append(OUTPUT / "reset-intents.jsonl", current)
            reference = None
            if horizon:
                baseline = next(row for row in rows if key(row) == (case["policy_id"], case["geometry_seed"], 0))
                reference = arrays_at(baseline)
            summary, arrays = parent.collect(case, actors[case["policy_id"]], horizon, reference, OUTPUT / f"{stem}-partial.npz")
            path = OUTPUT / f"{stem}.npz"
            parent.diagnosis._write_trace(path, arrays)
            old = parent.diagnosis._read_json(parent.PROTOCOL)
            row = dict(summary, episode_id=episode_id, path=path.name, sha256=parent.branch._sha(path),
                       mode="actor" if not horizon else f"oracle-{horizon}", policy_id=case["policy_id"], geometry_seed=case["geometry_seed"],
                       track_id=1, stratum=case["stratum"], actor_sha256=case["actor_sha256"],
                       parent_trace_sha256=old["source_sha256"][f"runs/oracle-policy-diagnosis-v1/policies/{case['policy_id']}-track1-seed{case['geometry_seed']}.npz"],
                       duplicate_interrupted_v1_attempt=current["duplicate_interrupted_v1_attempt"])
            rows.append(row)
            parent.branch._append(OUTPUT / "episodes.jsonl", row)
            print(f"r2 episode {episode_id+1}/42 {stem} steps={row['steps']} qualified={row['local_recovery_qualified']} finish={row['finished']}", flush=True)
        preflight()
        manifest = aggregate(p, rows, time.monotonic()-started)
        parent.branch._write(OUTPUT / "manifest.json", manifest, exclusive=True)
        parent.branch._write(OUTPUT / "result.json", manifest, exclusive=True)
        return manifest["counts"]
    except Exception as error:
        parent.branch._write(OUTPUT / "failure.json", {"timestamp": parent.utc(), "error": repr(error), "last_reset_intent": current,
                                                      "retained_complete_episodes": len(rows), "retained_complete_decisions": sum(row["steps"] for row in rows),
                                                      "exact_resume": False, "partial_trace": f"{current['stem']}-partial.npz" if current else None}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = freeze() if args.freeze else ({"status": "preflight_passed_no_resets", "new_slots": len(preflight()["missing_slots"])} if args.preflight_only else run())
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"continuation failed closed: {error!r}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
