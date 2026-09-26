"""Create hash-linked one-actor predecessor receipts from an entropy-ablation screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, read_protocol, sha256_file, write_json
from scripts.rlpd_entropy_common import STUDY_NAME, read_entropy_protocol
from scripts.run_rlpd_pilot import _rank_for_run, _summarize_screen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def project(protocol_path: Path, run_root: Path):
    protocol_path, run_root = Path(protocol_path).resolve(), Path(run_root).resolve()
    protocol = read_entropy_protocol(protocol_path)
    if protocol["name"] != STUDY_NAME:
        raise ValueError("wrong study for entropy screen receipt projection")
    out_dir = run_root / "screen-selection-projections"
    output = run_root / "screen-selection-projections.json"
    if out_dir.exists() or output.exists():
        raise FileExistsError("screen selection projections already exist; do not overwrite")

    parent_pointer = run_root / "screen-evaluation.json"
    pointer_bytes = parent_pointer.read_bytes()
    parent_pointer_record = json.loads(pointer_bytes)
    parent_dir = Path(parent_pointer_record["evaluation_dir"]).resolve()
    summary_path, manifest_path, candidates_path = (
        parent_dir / "summary.json",
        parent_dir / "manifest.json",
        parent_dir / "candidates.json",
    )
    summary_bytes, manifest_bytes, candidate_bytes = (
        summary_path.read_bytes(), manifest_path.read_bytes(), candidates_path.read_bytes()
    )
    parent_summaries = json.loads(summary_bytes)
    if (
        parent_pointer_record.get("protocol_sha256") != sha256_file(protocol_path)
        or parent_pointer_record.get("partition") != "screen"
        or parent_pointer_record.get("ranked") != parent_summaries
        or len(parent_summaries) != 8
    ):
        raise ValueError("parent screen receipt doesn't match this frozen screen/protocol")

    candidate_by_hash = {}
    for arm in protocol["student_training"]["arms"]:
        for seed in protocol["student_training"]["learner_seeds"]:
            run_dir = run_root / f"{arm}-seed{seed}"
            records = json.loads((run_dir / "frozen_candidates.json").read_text())['candidates']
            for record in records:
                candidate_by_hash[record["actor_sha256"]] = {
                    **record,
                    "arm": arm,
                    "seed": seed,
                    "actor_path": str((run_dir / record["actor_path"]).resolve()),
                }
    candidates, summaries, finish_counts = _summarize_screen(parent_dir)
    selected = {}
    expected_canonical = len(protocol["partitions"]["screen"]["track_ids"]) * len(protocol["partitions"]["screen"]["seeds"])
    for arm in protocol["student_training"]["arms"]:
        for seed in protocol["student_training"]["learner_seeds"]:
            selected[f"{arm}-seed{seed}"] = _rank_for_run(
                candidates,
                summaries,
                finish_counts,
                {"arm": arm, "seed": seed},
                candidate_by_hash,
                expected_canonical_episodes=expected_canonical,
            )
    screen_gates = {}
    for arm in protocol["student_training"]["arms"]:
        screen_gates[arm] = {
            str(seed): bool(
                selected[f"{arm}-seed{seed}"] is not None
                and selected[f"{arm}-seed{seed}"]["canonical_finishes"] >= 1
            )
            for seed in protocol["student_training"]["learner_seeds"]
        }
    if not all(pass_by_seed and all(pass_by_seed.values()) for pass_by_seed in screen_gates.values()):
        raise ValueError("one or more entropy-treatment screen gates failed; no confirmation receipts")

    summary_by_id = {record["candidate_id"]: record for record in parent_summaries}
    candidate_by_id = {record["candidate_id"]: record for record in candidates}
    out_dir.mkdir()
    projections = {}
    protocol_sha = sha256_file(protocol_path)
    for key, record in selected.items():
        candidate_id = record["candidate_id"]
        summary_row = summary_by_id[candidate_id]
        screen_candidate = candidate_by_id[candidate_id]
        actor_sha = record["actor_sha256"]
        if (
            screen_candidate.get("archive_sha256") != actor_sha
            or summary_row.get("archive_sha256") != actor_sha
            or summary_row["summary"].get("finish_rate", 0.0) <= 0.0
        ):
            raise ValueError(f"candidate summary or actor hash is inconsistent for {key}")
        relative_archive = Path(screen_candidate["evaluation_archive_path"])
        source_archive = (parent_dir / relative_archive).resolve()
        if not source_archive.is_relative_to(parent_dir) or sha256_file(source_archive) != actor_sha:
            raise ValueError(f"parent screen archive hash mismatch for {key}")
        destination = out_dir / key
        destination.mkdir()
        projected_archive = destination / relative_archive
        projected_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_archive, projected_archive)
        if sha256_file(projected_archive) != actor_sha:
            raise RuntimeError(f"projected screen actor archive changed for {key}")
        config_rel = screen_candidate.get("evaluation_run_config_path")
        if config_rel:
            source_config = (parent_dir / config_rel).resolve()
            projected_config = destination / config_rel
            projected_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_config, projected_config)
            if sha256_file(projected_config) != sha256_file(source_config):
                raise RuntimeError(f"projected run config changed for {key}")

        manifest = json.loads(manifest_bytes)
        manifest["selection_projection"] = {
            "format": "haic-screen-row-selection-projection-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "parent_screen_pointer": str(parent_pointer.resolve()),
            "parent_screen_pointer_sha256": sha256_file(parent_pointer),
            "parent_screen_summary_sha256": sha256_file(summary_path),
            "parent_screen_manifest_sha256": sha256_file(manifest_path),
            "parent_candidates_sha256": sha256_file(candidates_path),
            "candidate_id": candidate_id,
            "actor_sha256": actor_sha,
            "selection_order": protocol["evaluation"]["selection_order"],
            "no_environment_replay": True,
        }
        write_json(destination / "summary.json", [summary_row], exclusive=True)
        write_json(destination / "manifest.json", manifest, exclusive=True)
        pointer_path = destination / "previous-evaluation.json"
        projected_pointer = {
            "format": "haic-rlpd-derived-screen-selection-v1",
            "protocol_name": protocol["name"],
            "protocol_sha256": protocol_sha,
            "partition": "screen",
            "diagnostic_only": False,
            "evaluation_dir": str(destination.resolve()),
            "ranked": [summary_row],
            "parent_screen_pointer": str(parent_pointer.resolve()),
            "parent_screen_pointer_sha256": sha256_file(parent_pointer),
            "selection_rule": protocol["evaluation"]["selection_order"],
        }
        write_json(pointer_path, projected_pointer, exclusive=True)
        actor = candidate_metadata(Path(candidate_by_hash[actor_sha]["actor_path"]), run_dir=run_root)
        previous = previous_evaluation_metadata(
            pointer_path,
            "confirmation",
            protocol_sha,
            actor,
        )
        projections[key] = {
            "actor_sha256": actor_sha,
            "candidate_id": candidate_id,
            "previous_screen_pointer": str(pointer_path.resolve()),
            "previous_screen_pointer_sha256": sha256_file(pointer_path),
            "projection_manifest_sha256": sha256_file(destination / "manifest.json"),
            "previous_evaluation_preflight": previous,
        }
    result = {
        "format": "haic-rlpd-entropy-screen-selection-v1",
        "study_id": protocol["name"],
        "protocol_sha256": protocol_sha,
        "parent_screen_pointer": str(parent_pointer.resolve()),
        "parent_screen_pointer_sha256": sha256_file(parent_pointer),
        "parent_screen_summary_sha256": sha256_file(summary_path),
        "parent_screen_manifest_sha256": sha256_file(manifest_path),
        "screen_gates": screen_gates,
        "selected": selected,
        "confirmation_receipts": projections,
        "cells_replayed": False,
        "environment_interactions": 0,
        "projector_sha256": sha256_file(Path(__file__)),
    }
    write_json(output, result, exclusive=True)
    return output


def main():
    args = parse_args()
    output = project(args.protocol, args.run_root)
    print(json.dumps({"selection_path": str(output), "sha256": sha256_file(output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
