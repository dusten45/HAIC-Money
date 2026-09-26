"""Project frozen screen rows to actor-specific confirmation lineage receipts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from evaluate_policy import candidate_metadata, previous_evaluation_metadata
from scripts.rlpd_common import ROOT, read_protocol, sha256_file, write_json
from scripts.run_rlpd_followup import _finalist_screen_rank
from scripts.run_rlpd_pilot import _rank_for_run, _summarize_screen


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def project(protocol_path: Path, run_root: Path):
    protocol_path = protocol_path.resolve()
    run_root = run_root.resolve()
    protocol = read_protocol(protocol_path)
    if protocol["name"] != "pixel-rlpd-long-horizon-followup-v1":
        raise ValueError("screen receipt projection is scoped to the frozen long-horizon follow-up")
    if run_root != (ROOT / "runs/20260924-pixel-rlpd-long-horizon-followup-v1").resolve():
        raise ValueError("unexpected V3 run root")
    projection_root = run_root / "screen-selection-projections"
    selection_path = run_root / "screen-selection-projections.json"
    preflight_path = run_root / "screen-selection-preflight.json"
    if projection_root.exists() or selection_path.exists() or preflight_path.exists():
        raise FileExistsError("screen selection projection has already been created; do not overwrite")

    parent_pointer = run_root / "screen-evaluation.json"
    parent_pointer_bytes = parent_pointer.read_bytes()
    parent = json.loads(parent_pointer_bytes)
    parent_dir = Path(parent["evaluation_dir"]).resolve()
    parent_summary_path = parent_dir / "summary.json"
    parent_manifest_path = parent_dir / "manifest.json"
    parent_candidates_path = parent_dir / "candidates.json"
    parent_summary_bytes = parent_summary_path.read_bytes()
    parent_manifest_bytes = parent_manifest_path.read_bytes()
    parent_candidates_bytes = parent_candidates_path.read_bytes()
    if (
        parent.get("protocol_sha256") != sha256_file(protocol_path)
        or parent.get("partition") != "screen"
        or json.loads(parent_summary_bytes) != parent.get("ranked")
    ):
        raise ValueError("parent screen pointer/summary/protocol lineage is inconsistent")
    parent_summaries = json.loads(parent_summary_bytes)
    if len(parent_summaries) != 8 or any(
        row.get("canonical_episodes") != 24
        or row.get("eligible") is not True
        or row.get("determinism_audited") is not True
        or row.get("cpu_reload_matches") is not True
        or row.get("operational_failures") != 0
        for row in parent_summaries
    ):
        raise ValueError("parent screen does not establish all eight candidate operational receipts")

    data_root = run_root
    candidate_identity = {}
    for arm in ("rlpd", "sac"):
        for seed in protocol["student_training"]["learner_seeds"]:
            run_dir = data_root / f"{arm}-seed{seed}"
            records = json.loads((run_dir / "frozen_candidates.json").read_text())["candidates"]
            for record in records:
                candidate_identity[record["actor_sha256"]] = {
                    "arm": arm,
                    "seed": seed,
                    "environment_steps": record["environment_steps"],
                    "checkpoint_sha256": record["checkpoint_sha256"],
                    "actor_path": str((run_dir / record["actor_path"]).resolve()),
                }
    candidates, summaries, finish_counts = _summarize_screen(parent_dir)
    selected_by_run = {
        f"{arm}-seed{seed}": _rank_for_run(
            candidates,
            summaries,
            finish_counts,
            {"arm": arm, "seed": seed},
            candidate_identity,
            expected_canonical_episodes=24,
        )
        for arm in ("rlpd", "sac")
        for seed in protocol["student_training"]["learner_seeds"]
    }
    for seed in protocol["student_training"]["learner_seeds"]:
        treatment = selected_by_run[f"rlpd-seed{seed}"]
        control = selected_by_run[f"sac-seed{seed}"]
        if (
            treatment is None
            or control is None
            or treatment["canonical_finishes"] < 1
            or treatment["canonical_finishes"] < control["canonical_finishes"]
        ):
            raise ValueError("frozen screen promotion gate failed; no confirmation receipt may be projected")
    blind_finalist = _finalist_screen_rank(
        selected_by_run,
        protocol["student_training"]["learner_seeds"],
    )

    summaries_by_id = {record["candidate_id"]: record for record in parent_summaries}
    candidates_by_id = {record["candidate_id"]: record for record in candidates}
    projection_root.mkdir()
    projections = {}
    try:
        for key, selected in selected_by_run.items():
            candidate_id = selected["candidate_id"]
            summary_row = summaries_by_id[candidate_id]
            screen_candidate = candidates_by_id[candidate_id]
            actor_sha = selected["actor_sha256"]
            if (
                screen_candidate.get("archive_sha256") != actor_sha
                or summary_row.get("archive_sha256") != actor_sha
                or summary_row.get("finish_rate", summary_row.get("summary", {}).get("finish_rate")) == 0.0
            ):
                raise ValueError(f"selected screen actor is not exactly bound to its nonzero result: {key}")
            archive_relative = Path(screen_candidate["evaluation_archive_path"])
            parent_archive = (parent_dir / archive_relative).resolve()
            if not parent_archive.is_relative_to(parent_dir) or sha256_file(parent_archive) != actor_sha:
                raise ValueError(f"screen actor archive integrity mismatch: {key}")
            destination = projection_root / key
            destination.mkdir()
            projected_archive = destination / archive_relative
            projected_archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(parent_archive, projected_archive)
            if sha256_file(projected_archive) != actor_sha:
                raise RuntimeError(f"projected actor archive hash changed: {key}")
            config_relative = screen_candidate.get("evaluation_run_config_path")
            if config_relative:
                parent_config = (parent_dir / config_relative).resolve()
                projected_config = destination / config_relative
                projected_config.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(parent_config, projected_config)
                if sha256_file(projected_config) != sha256_file(parent_config):
                    raise RuntimeError(f"projected run-config hash changed: {key}")

            manifest = json.loads(parent_manifest_bytes)
            manifest["selection_projection"] = {
                "format": "haic-screen-row-selection-projection-v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_screen_pointer": str(parent_pointer.relative_to(ROOT)),
                "parent_screen_pointer_sha256": sha256_file(parent_pointer),
                "parent_screen_summary_sha256": sha256_file(parent_summary_path),
                "parent_screen_manifest_sha256": sha256_file(parent_manifest_path),
                "parent_candidates_sha256": sha256_file(parent_candidates_path),
                "parent_canonical_episode_count": 24,
                "projected_candidate_id": candidate_id,
                "projected_actor_sha256": actor_sha,
                "reused_screen_cells_for_execution": False,
                "meaning": "a projection of one exact previously evaluated screen actor row solely for the evaluator's one-actor confirmation lineage; no observations were rerun and parent receipt remains unchanged",
            }
            write_json(destination / "summary.json", [summary_row], exclusive=True)
            write_json(destination / "manifest.json", manifest, exclusive=True)
            pointer = {
                "format": "haic-rlpd-derived-screen-selection-v1",
                "protocol_name": protocol["name"],
                "protocol_sha256": sha256_file(protocol_path),
                "partition": "screen",
                "diagnostic_only": False,
                "evaluation_dir": str(destination.resolve()),
                "ranked": [summary_row],
                "parent_screen_pointer": str(parent_pointer.resolve()),
                "parent_screen_pointer_sha256": sha256_file(parent_pointer),
                "selection_rule": protocol["evaluation"]["selection_order"],
                "selected_checkpoint": {
                    "arm": selected["arm"],
                    "training_seed": selected["training_seed"],
                    "environment_steps": selected["environment_steps"],
                    "actor_sha256": actor_sha,
                    "learner_checkpoint_sha256": selected["learner_checkpoint_sha256"],
                    "canonical_finishes": selected["canonical_finishes"],
                    "avg_progress": selected["avg_progress"],
                    "avg_lap_time_ms": selected["avg_lap_time_ms"],
                },
            }
            pointer_path = destination / "previous-evaluation.json"
            write_json(pointer_path, pointer, exclusive=True)

            actor = candidate_metadata(
                Path(candidate_identity[actor_sha]["actor_path"]),
                run_dir=run_root,
            )
            previous = previous_evaluation_metadata(
                pointer_path,
                "confirmation",
                sha256_file(protocol_path),
                actor,
            )
            projections[key] = {
                "screen_pointer": str(pointer_path.resolve()),
                "screen_pointer_sha256": sha256_file(pointer_path),
                "derived_directory": str(destination.resolve()),
                "derived_manifest_sha256": sha256_file(destination / "manifest.json"),
                "derived_summary_sha256": sha256_file(destination / "summary.json"),
                "actor_sha256": actor_sha,
                "confirmation_previous_evaluation_preflight": previous,
            }
    except BaseException:
        raise

    selected_file = {
        "format": "haic-rlpd-screen-selection-v1",
        "study_id": protocol["name"],
        "protocol_sha256": sha256_file(protocol_path),
        "selection_order": protocol["evaluation"]["selection_order"],
        "parent_screen_pointer": str(parent_pointer.resolve()),
        "parent_screen_pointer_sha256": sha256_file(parent_pointer),
        "parent_screen_manifest_sha256": sha256_file(parent_manifest_path),
        "parent_screen_summary_sha256": sha256_file(parent_summary_path),
        "selected_by_run": selected_by_run,
        "blind_finalist_preselected_from_screen": {
            "arm": blind_finalist["arm"],
            "training_seed": blind_finalist["training_seed"],
            "environment_steps": blind_finalist["environment_steps"],
            "actor_sha256": blind_finalist["actor_sha256"],
        },
        "confirm_receipts": projections,
        "projector_sha256": sha256_file(Path(__file__)),
        "new_environment_interactions": 0,
        "cell_replay": False,
        "confirmation_gate_preflight": "all four exact candidate lineages pass previous_evaluation_metadata without executing cells",
    }
    write_json(selection_path := run_root / "screen-selection-projections.json", selected_file, exclusive=True)
    return selection_path


def main():
    args = parse_args()
    path = project(args.protocol, args.run_root)
    print(json.dumps({"selection_projections": str(path), "sha256": sha256_file(path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
