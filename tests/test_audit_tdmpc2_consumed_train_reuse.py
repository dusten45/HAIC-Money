"""Synthetic-only consumed TRAIN reuse inventory tests; no environment imports."""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import audit_tdmpc2_consumed_train_reuse as audit


def _rows(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


@pytest.fixture
def study(tmp_path: Path, monkeypatch):
    root = tmp_path

    def write(path: str, value: dict | bytes) -> None:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value if isinstance(value, bytes) else
                                (json.dumps(value, sort_keys=True) + "\n").encode())

    def sha(path: str) -> str:
        return hashlib.sha256((root / path).read_bytes()).hexdigest()

    names = ("opening-short-entry-left-turn", "opening-delayed-high-turn", "easy-curvature-anchor",
             "mid-road-left-right-reversal", "mid-road-sustained-or-same-turn", "finish-approach-turn")
    reserved = list(range(3910800001, 3910800513))
    selected = [3910800001, 3910800004, 3910800034, 3910800085]
    for seed in reserved:
        if len(selected) == 120:
            break
        if seed not in selected:
            selected.append(seed)
    # Assign the four old TD roads to four distinct measured families.
    groups = [[selected[i]] for i in range(4)] + [[], []]
    available = [s for s in selected if s not in audit.TD4]
    for group in groups:
        needed = 20 - len(group)
        group.extend(available[:needed])
        available = available[needed:]
    assert len({seed for group in groups for seed in group}) == 120
    train = [{"geometry_seed": seed, "track_id": 1, "family": name,
              "stage": "representative" if index < 10 else "variant",
              "road_coordinate_sha256": f"{seed:064x}",
              "verification": {"raw_reset": True, "one_valid_raw_step": True,
                               "regenerated_coordinate_hash_match": True}}
             for name, group in zip(names, groups) for index, seed in enumerate(group)]
    diagnostic = [{"geometry_seed": seed, "track_id": 1, "family": names[index % 6],
                   "stage": "diagnostic", "road_coordinate_sha256": f"{seed:064x}",
                   "verification": {"raw_reset": True, "one_valid_raw_step": True,
                                    "regenerated_coordinate_hash_match": True}}
                  for index, seed in enumerate(s for s in reserved if s not in selected)][:16]
    candidate = [seed for group in groups for seed in group[:4]]
    assert len(candidate) == 24 and audit.TD4 <= set(candidate)
    manifest = "experiments/tdmpc2-reuse-proposal.json"

    def manifest_write(seeds: list[int] = candidate, *, track: int = 1, obstacles: bool = True) -> None:
        write(manifest, {"format": audit.FORMAT, "purpose": "consumed-TRAIN-development",
                         "cells": [{"partition": "TRAIN", "track_id": track, "geometry_seed": seed,
                                    "obstacles": obstacles} for seed in seeds]})

    manifest_write()
    sources = {}
    for source in sorted(audit._GENERATION_SOURCES):
        write(source, source.encode())
        sources[source] = sha(source)
    write(audit.CATALOG_PROTOCOL, {"format": "haic-drq-training-geometry-protocol-v1",
                                   "candidate_seeds": reserved, "family_rules": [{"name": name} for name in names],
                                   "exclusion_seed_ids": {"reserved": [99991], "heldout": [99992],
                                                          "blind": [99993]},
                                   "source_sha256": sources})
    protocol_sha = sha(audit.CATALOG_PROTOCOL)
    write(audit.CATALOG, {"format": "haic-drq-training-geometry-catalog-v1",
                          "protocol_sha256": protocol_sha, "seed_audit": {
                              "passed": True, "proposed_seeds": reserved},
                          "source_sha256": sources, "train": train, "train_diagnostic": diagnostic})
    catalog_sha = sha(audit.CATALOG)
    write(audit.CATALOG_RESULT, {"status": "success", "catalog_path": audit.CATALOG,
                                 "catalog_sha256": catalog_sha, "protocol_sha256": protocol_sha,
                                 "train_count": 120, "train_diagnostic_count": 16})
    monkeypatch.setattr(audit, "CATALOG_SHA256", catalog_sha)
    monkeypatch.setattr(audit, "CATALOG_PROTOCOL_SHA256", protocol_sha)
    monkeypatch.setattr(audit, "COMPLETE_DECISIONS", 40)
    write(audit.R6_PROTOCOL, {"catalog": {"path": audit.CATALOG, "sha256": catalog_sha,
                                          "protocol_path": audit.CATALOG_PROTOCOL,
                                          "protocol_sha256": protocol_sha},
                              "training_pool": {"geometry_seeds": selected, "track_ids": [1, 2, 3, 4]},
                              "diagnostic_pool": {"geometry_seeds": [row["geometry_seed"] for row in diagnostic]},
                              "environment": {"obstacles": True, "track_ids": [1, 2, 3, 4]}})
    write(audit.ACTOR_RECEIPT, {
        "format": "haic-drq-training-geometry-final-set-v1", "catalog_path": audit.CATALOG,
        "catalog_sha256": catalog_sha, "catalog_protocol_sha256": protocol_sha,
        "roads": [{"geometry_seed": row["geometry_seed"], "track_id": 1,
                   "partition": "TRAIN" if row in train else "TRAIN-DIAGNOSTIC",
                   "family": row["family"], "frozen_source_actor_results": {
                       str(actor): {"cell_id": f"train-geometry:train:seed-{row['geometry_seed']}:track-1:source-{actor}",
                                    "steps": 1} for actor in (0, 1)}}
                  for row in train + diagnostic]})

    cells = [{"track_id": 1, "geometry_seed": seed} for seed in audit.TD_ORDER]
    environment = {"track_id": 1, "obstacles": True, "frame_skip": 4, "reward": "raw"}
    for final, protocol, run in audit.COMPLETE_TD:
        producer = "scripts/train_tdmpc2_damage.py" if "damage" in run else "scripts/train_tdmpc2_long.py"
        write(producer, producer.encode())
        source = {producer: sha(producer)}
        write(protocol, {"cells": cells, "episode_schedule": [0, 1, 2, 3],
                         "environment": environment, "run_dir": run, "source_sha256": source})
        protocol_sha = sha(protocol)
        rows = [{"event": "start", "protocol_sha256": protocol_sha}]
        for i, seed in enumerate(audit.TD_ORDER):
            for event in ("reset_intent", "reset", "episode"):
                row = {"event": event, "episode": i, "track_id": 1, "geometry_seed": seed}
                if event == "episode":
                    row.update(length=10, decisions=10 * (i + 1))
                rows.append(row)
        journal = f"{run}/training.jsonl"
        steps = f"{run}/steps.jsonl"
        write(journal, _rows(rows))
        write(steps, b"synthetic step data\n")
        write(f"{run}/result.json", {"status": "completed_boundary_at_least_100k",
                                      "protocol_sha256": protocol_sha, "source_sha256": source,
                                      "episodes": 4, "decisions": 40, "updates": 40,
                                      "training_ledger_sha256": sha(journal),
                                      "step_ledger_sha256": sha(steps)})
        write(final, {"training_protocol": protocol, "training_protocol_sha256": protocol_sha,
                      "run_status": "completed_boundary_at_least_100k", "decisions": 40,
                      "run_result": f"{run}/result.json", "run_result_sha256": sha(f"{run}/result.json"),
                      "complete_training_ledger_sha256": sha(journal),
                      "training_ledger_sha256": sha(journal),
                      "complete_step_ledger_sha256": sha(steps), "step_ledger_sha256": sha(steps)})
    for path in audit.TD_JOURNALS[:3]:
        write(path, _rows([{"event": "start", "protocol_sha256": "a" * 64},
                           {"event": "reset_intent", "episode": 0,
                            "track_id": 1, "geometry_seed": audit.TD_ORDER[0]},
                           {"event": "reset", "episode": 0,
                            "track_id": 1, "geometry_seed": audit.TD_ORDER[0]},
                           {"event": "episode", "episode": 0, "track_id": 1,
                            "geometry_seed": audit.TD_ORDER[0], "length": 10, "decisions": 10},
                           {"event": "partial", "decisions": 10}]))
    write("experiments/train-seed-claims/.gitkeep", b"")
    write("runs/rlpd-g0-claims/.lock", b"")
    return root, candidate, diagnostic, manifest, write, sha, manifest_write


def _reasons(report: dict) -> str:
    return " ".join(row["reason"] for row in report["blockers"])


def test_clean_fixture_still_blocks_and_never_selects_or_claims(study):
    root, candidate, diagnostic, manifest, _, sha, _ = study
    report = audit.audit(manifest, repo_root=root)
    assert report["status"] == "BLOCKED"
    assert report["cells"] and len(report["cells"]) == 24
    assert report["candidate_sha256"] == sha(manifest)
    assert report["families"] == {name: 4 for name in report["families"]}
    assert report["excluded_diagnostic_geometry_seed_count"] == 16
    assert "historical_exposure_coverage" in str(report["blockers"])
    assert report["protected_outcome_reads"] == 0
    assert report["reservation_or_claim"] is False
    assert len(list((root / "experiments/train-seed-claims").glob("seed-*.json"))) == 0
    assert not set(candidate) & {row["geometry_seed"] for row in diagnostic}


@pytest.mark.parametrize("kind", ["duplicate", "family", "diagnostic", "outside", "missing_td", "track", "obstacles"])
def test_candidate_partition_and_metadata_falsifications(study, kind):
    root, candidate, diagnostic, manifest, _, _, update = study
    changed = candidate.copy()
    if kind == "duplicate":
        changed[1] = changed[0]
    elif kind == "family":
        changed[-1] = json.loads((root / audit.CATALOG).read_text())["train"][4]["geometry_seed"]
    elif kind == "diagnostic":
        changed[-1] = diagnostic[0]["geometry_seed"]
    elif kind == "outside":
        changed[-1] = 3910800400
    elif kind == "missing_td":
        changed[0] = 3910800400
    update(changed, track=2 if kind == "track" else 1, obstacles=kind != "obstacles")
    report = audit.audit(manifest, repo_root=root)
    assert report["status"] == "BLOCKED"
    assert not report.get("families") or kind not in ("track", "obstacles")
    assert report["blockers"]


def test_catalog_source_and_td_receipt_sha_drift_block(study):
    root, _, _, manifest, write, _, _ = study
    write("scripts/generate_drq_training_geometry.py", b"changed generator\n")
    assert "source pin mismatch" in _reasons(audit.audit(manifest, repo_root=root))


def test_actor_interaction_or_r6_pool_falsification_blocks(study):
    root, candidate, _, manifest, write, _, _ = study
    receipt = json.loads((root / audit.ACTOR_RECEIPT).read_text())
    road = next(row for row in receipt["roads"] if row["geometry_seed"] == candidate[5])
    road["frozen_source_actor_results"]["1"]["steps"] = 0
    write(audit.ACTOR_RECEIPT, receipt)
    assert "positive typed interaction" in _reasons(audit.audit(manifest, repo_root=root))
    r6 = json.loads((root / audit.R6_PROTOCOL).read_text())
    r6["diagnostic_pool"]["geometry_seeds"][0] = candidate[5]
    write(audit.R6_PROTOCOL, r6)
    assert "TRAIN/diagnostic pool" in _reasons(audit.audit(manifest, repo_root=root))


def test_raw_and_damage_full_ledger_changes_block(study):
    root, _, _, manifest, write, _, _ = study
    path = "runs/tdmpc2-damage-20260928-v1/steps.jsonl"
    write(path, b"changed step data\n")
    assert "full training receipt/ledger mismatch" in _reasons(audit.audit(manifest, repo_root=root))


def test_protected_alias_detected_from_id_metadata_without_outcome_reads(study):
    root, candidate, _, manifest, write, _, _ = study
    write("experiments/foreign-protocol.json", {"partitions": {"blind": {
        "seeds": [f"track-4:seed-{candidate[5]}"]}}})
    write("evaluations/blind/episodes.jsonl", b"DO NOT READ: protected outcome\n")
    report = audit.audit(manifest, repo_root=root)
    assert "foreign-protocol.json" in str(report["blockers"])
    assert report["protected_outcome_reads"] == 0


def test_foreign_claim_and_candidate_specific_ambiguous_record_block(study):
    root, candidate, _, manifest, write, _, _ = study
    write(f"experiments/train-seed-claims/seed-{candidate[5]}.json", {"geometry_seed": candidate[5]})
    write("runs/foreign/collection.jsonl", _rows([{"event": "reset", "track_id": 1,
                                                       "geometry_seed": candidate[6]}]))
    report = audit.audit(manifest, repo_root=root)
    assert "foreign TRAIN claim" in _reasons(report)
    assert "unknown cross-lane TRAIN ledger" in _reasons(report)


def test_drq_and_dreamer_typed_exposures_are_not_fresh_clearance(study):
    root, candidate, _, manifest, write, sha, _ = study
    family = next(row["family"] for row in json.loads((root / audit.CATALOG).read_text())["train"]
                  if row["geometry_seed"] == candidate[5])
    path = "runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform/episodes.jsonl"
    write(path, _rows([{"event": "reset", "geometry_seed": candidate[5], "seed": candidate[5],
                        "track_id": 2, "geometry_family": family,
                        "catalog_sha256": audit.CATALOG_SHA256}]))
    write("experiments/dreamerv3-reused-train-diversity-v1.json", {
        "purpose": "reused-TRAIN-engineering-diagnostic",
        "r6_protocol": {"path": audit.R6_PROTOCOL, "sha256": sha(audit.R6_PROTOCOL)},
        "cells": [{"track_id": 1, "geometry_seed": candidate[6]}]})
    report = audit.audit(manifest, repo_root=root)
    assert report["typed_exposures"] == [{"path": path, "line": 1,
                                          "geometry_seed": candidate[5], "track_id": 2,
                                          "exact_track1_cell": False}]
    assert "typed cross-lane consumed Dreamer TRAIN source" in str(report["provenance_warnings"])
    assert [b["path"] for b in report["blockers"]] == ["historical_exposure_coverage"]


def test_conflicting_cross_lane_source_pin_blocks(study):
    root, candidate, _, manifest, write, _, _ = study
    write("runs/20260925-drqv2-geometry-mix-v1-r6/learner-0-uniform/episodes.jsonl",
          _rows([{"event": "reset", "geometry_seed": candidate[5], "track_id": 1,
                  "catalog_sha256": "0" * 64}]))
    assert "contradictory road identity" in _reasons(audit.audit(manifest, repo_root=root))


def test_candidate_in_seed_interval_without_literal_id_blocks(study):
    root, candidate, _, manifest, write, _, _ = study
    write("runs/foreign/episodes.jsonl", _rows([{"event": "reset_intent",
                                                      "seedFrom": candidate[6] - 1,
                                                      "seedTo": candidate[6] + 1}]))
    report = audit.audit(manifest, repo_root=root)
    assert "unknown cross-lane TRAIN ledger" in _reasons(report)


def test_torn_td_reset_intent_and_ambiguous_run_record_block(study):
    root, candidate, _, manifest, write, _, _ = study
    path = audit.TD_JOURNALS[0]
    write(path, _rows([{"event": "start", "protocol_sha256": "a" * 64},
                       {"event": "reset_intent", "episode": 0, "track_id": 1,
                        "geometry_seed": audit.TD_ORDER[0]}]) + b'{"event":')
    write("runs/unknown/result.json", {"decisions": 42, "road": candidate[7]})
    report = audit.audit(manifest, repo_root=root)
    assert "malformed JSON" in _reasons(report)
    assert "unbound run-local or result-only" in _reasons(report)


def test_unrelated_lane_drift_is_not_candidate_collision(study):
    root, _, _, manifest, write, _, _ = study
    write("runs/foreign/result.json", {"decisions": 300, "geometry_seed": 990001})
    write("experiments/foreign-protocol.json", {"partitions": {"train": {"seeds": [990002]}}})
    (root / "runs/foreign/episodes.jsonl").symlink_to("result.json")
    report = audit.audit(manifest, repo_root=root)
    assert report["status"] == "BLOCKED"
    assert len(report["blockers"]) == 1
    assert report["blockers"][0]["path"] == "historical_exposure_coverage"
    assert "unreviewed TRAIN run symlink" in str(report["provenance_warnings"])


def test_cli_no_reserve_and_always_nonzero(study, monkeypatch):
    root, _, _, manifest, _, _, _ = study
    monkeypatch.setattr("sys.argv", ["auditor", "--candidate-path", manifest, "--repo-root", str(root)])
    out = io.StringIO()
    with redirect_stdout(out):
        assert audit.main() == 2
    report = json.loads(out.getvalue())
    assert report["status"] == "BLOCKED" and not report["reservation_or_claim"]
    with patch("sys.argv", ["auditor", "--candidate-path", manifest, "--reserve"]):
        with pytest.raises(SystemExit):
            audit.main()
