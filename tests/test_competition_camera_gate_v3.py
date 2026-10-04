"""Fixed, receipt-level decision contract for the fresh v3 camera study."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tools.competition_camera_gate_v3 import combined_decision, compare_pairs


PHASE_GAIN = {"screen": 3, "confirmation": 6, "blind": 3}
PHASE_LOSSES = {"screen": 2, "confirmation": 4, "blind": 2}
PHASE_SEEDS = {"screen": 8, "confirmation": 16, "blind": 8}
PHASE_REPEATS = {"screen": 2, "confirmation": 4, "blind": 2}


def _row(track: int, seed: int, repeat: int, arm: str) -> dict:
    return {
        "partition": "screen", "track_id": track, "seed": seed,
        "repeat": repeat, "arm": arm, "finished": False, "progress": 0.5,
        "lap_time_ms": None, "retire_reason": "off_track", "collision_count": 0,
        "damage": 0.0, "offtrack_samples": 0, "partial_offtrack_samples": 0,
        "initialization_ms": 10.0, "reset_ms": 1.0,
        "action_latency_max_ms": 1.0, "peak_worker_rss_mib": 220.0,
        "steps": 100,
        "action_trace_sha256": ("1" if arm == "control" else "2") * 64,
        "error": None,
    }


def _case(phase: str) -> tuple[list[dict], list[tuple[int, int, int]]]:
    cells = [(track, 2000 + index, 0)
             for track in (1, 2, 3, 4) for index in range(PHASE_SEEDS[phase])]
    rows = []
    for track, seed, repeat in cells:
        for arm in ("control", "candidate"):
            row = _row(track, seed, repeat, arm)
            row["partition"] = phase
            rows.append(row)
    for track, seed, _ in cells[-PHASE_REPEATS[phase]:]:
        canonical_index = next(index for index, cell in enumerate(cells)
                               if cell[:2] == (track, seed))
        cells.append((track, seed, 1))
        rows.extend({**deepcopy(row), "repeat": 1}
                    for row in rows[2 * canonical_index:2 * canonical_index + 2])
    return rows, cells


def _finish(rows: list[dict], index: int, arm: str, lap_time: int = 20_000) -> None:
    row = rows[2 * index + (arm == "candidate")]
    row.update(finished=True, progress=1.0, lap_time_ms=lap_time,
               retire_reason=None)


def _gains(phase: str) -> tuple[list[dict], list[tuple[int, int, int]]]:
    rows, cells = _case(phase)
    for index in range(PHASE_GAIN[phase]):
        _finish(rows, index, "candidate")
    return rows, cells


@pytest.mark.parametrize("phase", ("screen", "confirmation", "blind"))
def test_exact_net_finish_threshold_retains_and_one_less_rejects(phase: str) -> None:
    rows, cells = _gains(phase)
    exact = compare_pairs(rows, cells, phase)
    assert exact["decision"] == "RETAIN"
    assert exact["net_finish_gain"] == PHASE_GAIN[phase]
    assert exact["shared_completed_cells"] == 0
    rows[2 * (PHASE_GAIN[phase] - 1) + 1].update(
        finished=False, progress=0.5, lap_time_ms=None,
        retire_reason="off_track")
    below = compare_pairs(rows, cells, phase)
    assert below["decision"] == "REJECT"
    assert below["net_finish_gain"] == PHASE_GAIN[phase] - 1
    assert any("net finish" in reason for reason in below["reasons"])


@pytest.mark.parametrize("phase", ("screen", "confirmation", "blind"))
def test_exact_lost_control_finish_budget_retains_and_one_more_rejects(phase: str) -> None:
    allowed = PHASE_LOSSES[phase]
    rows, cells = _case(phase)
    for index in range(allowed + 1):
        _finish(rows, index, "control")
    for index in range(allowed + 1, PHASE_GAIN[phase] + 2 * (allowed + 1)):
        _finish(rows, index, "candidate")
    # Remove the final control finish first: the gain is then above its floor.
    rows[2 * allowed].update(finished=False, progress=0.5,
                             lap_time_ms=None, retire_reason="off_track")
    exact = compare_pairs(rows, cells, phase)
    assert exact["decision"] == "RETAIN"
    assert exact["lost_control_finishes"] == allowed
    _finish(rows, allowed, "control")
    excess = compare_pairs(rows, cells, phase)
    assert excess["decision"] == "REJECT"
    assert excess["lost_control_finishes"] == allowed + 1
    assert excess["net_finish_gain"] >= PHASE_GAIN[phase]


def test_aggregate_crash_budget_allows_one_new_crash_when_offset_by_control() -> None:
    rows, cells = _gains("screen")
    rows[2 * 3]["retire_reason"] = "crash"
    rows[2 * 4 + 1]["retire_reason"] = "crash"
    equal = compare_pairs(rows, cells, "screen")
    assert equal["decision"] == "RETAIN"
    assert (equal["control_crashes"], equal["candidate_crashes"],
            equal["new_candidate_crashes"]) == (1, 1, 1)
    assert "new candidate crash" in equal["paired_cells"][4]["flags"]
    assert equal["paired_cells"][4]["candidate"]["progress"] == 0.5
    rows[2 * 5 + 1]["retire_reason"] = "crash"
    excess = compare_pairs(rows, cells, "screen")
    assert excess["decision"] == "REJECT"
    assert any("aggregate crashes" in reason for reason in excess["reasons"])


def test_aggregate_contacts_and_damage_allow_local_spikes_only_when_offset() -> None:
    rows, cells = _gains("screen")
    rows[2 * 3].update(collision_count=4, damage=0.4)
    rows[2 * 4 + 1].update(collision_count=4, damage=0.4)
    equal = compare_pairs(rows, cells, "screen")
    assert equal["decision"] == "RETAIN"
    assert "contact increase" in equal["paired_cells"][4]["flags"]
    assert "damage increase" in equal["paired_cells"][4]["flags"]
    rows[2 * 4 + 1]["collision_count"] = 5
    contacts = compare_pairs(rows, cells, "screen")
    assert contacts["decision"] == "REJECT"
    assert any("aggregate contacts" in reason for reason in contacts["reasons"])
    rows[2 * 4 + 1].update(collision_count=4, damage=0.41)
    damage = compare_pairs(rows, cells, "screen")
    assert damage["decision"] == "REJECT"
    assert any("aggregate damage" in reason for reason in damage["reasons"])


def test_mean_progress_equality_passes_and_lower_mean_fails() -> None:
    rows, cells = _gains("screen")
    for index in (3, 4, 5):
        rows[2 * index + 1]["progress"] = 0.0
    equal = compare_pairs(rows, cells, "screen")
    assert equal["decision"] == "RETAIN"
    assert equal["candidate_mean_progress"] == equal["control_mean_progress"]
    assert equal["both_dnf_progress_losses"] == 3
    assert "both-DNF progress loss" in equal["paired_cells"][3]["flags"]
    rows[2 * 3 + 1]["progress"] = 0.0
    rows[2 * 3]["progress"] = 0.51
    lower = compare_pairs(rows, cells, "screen")
    assert lower["decision"] == "REJECT"
    assert any("mean progress" in reason for reason in lower["reasons"])


def test_shared_finish_total_time_exact_ten_percent_passes() -> None:
    rows, cells = _gains("screen")
    for index in (3, 4):
        _finish(rows, index, "control")
        _finish(rows, index, "candidate")
    rows[2 * 3 + 1]["lap_time_ms"] = 23_000
    rows[2 * 4 + 1]["lap_time_ms"] = 21_000
    exact = compare_pairs(rows, cells, "screen")
    assert exact["decision"] == "RETAIN"
    assert exact["shared_completed_cells"] == 2
    assert exact["control_shared_completed_time_ms"] == 40_000
    assert exact["candidate_shared_completed_time_ms"] == 44_000
    rows[2 * 4 + 1]["lap_time_ms"] = 21_001
    excess = compare_pairs(rows, cells, "screen")
    assert excess["decision"] == "REJECT"
    assert any("shared-finish time" in reason for reason in excess["reasons"])


def test_aggregate_boundaries_do_not_allow_a_small_positive_excess() -> None:
    rows, cells = _gains("screen")
    rows[2 * 3]["damage"] = 0.4
    rows[2 * 4 + 1]["damage"] = 0.4000000005
    assert compare_pairs(rows, cells, "screen")["decision"] == "REJECT"

    rows, cells = _gains("screen")
    for index in (3, 4, 5):
        rows[2 * index + 1]["progress"] = 0.0
    rows[2 * 3]["progress"] = 0.5000000005
    assert compare_pairs(rows, cells, "screen")["decision"] == "REJECT"

    rows, cells = _gains("screen")
    _finish(rows, 3, "control")
    _finish(rows, 3, "candidate", 22_000.0000005)
    assert compare_pairs(rows, cells, "screen")["decision"] == "REJECT"


def test_missing_canonical_and_repeat_receipts_are_incomplete() -> None:
    rows, cells = _gains("screen")
    missing_rows = deepcopy(rows)
    missing_rows.pop(2 * 2 + 1)
    missing = compare_pairs(missing_rows, cells, "screen")
    assert missing["decision"] == "INCOMPLETE"
    assert missing["missing_cells"] == [[1, 2002, 0, "candidate"]]
    missing_repeat = compare_pairs(rows[:-1], cells, "screen")
    assert missing_repeat["decision"] == "INCOMPLETE"
    assert missing_repeat["missing_cells"] == [[4, 2007, 1, "candidate"]]


def test_expected_grid_cannot_omit_a_canonical_or_repeat_cell() -> None:
    rows, cells = _gains("screen")
    omitted_canonical = compare_pairs(rows[2:], cells[1:], "screen")
    assert omitted_canonical["decision"] == "REJECT"
    assert any("canonical geometry grid" in reason
               for reason in omitted_canonical["reasons"])
    omitted_repeat = compare_pairs(rows[:-2], cells[:-1], "screen")
    assert omitted_repeat["decision"] == "REJECT"
    assert any("repeat grid" in reason for reason in omitted_repeat["reasons"])


def test_duplicate_unexpected_operational_and_invalid_receipts_reject() -> None:
    base_rows, cells = _gains("screen")
    cases = []
    duplicate = deepcopy(base_rows) + [deepcopy(base_rows[0])]
    cases.append((duplicate, "duplicate"))
    unexpected = deepcopy(base_rows) + [{**deepcopy(base_rows[0]), "seed": 9999}]
    cases.append((unexpected, "unexpected"))
    operational = deepcopy(base_rows)
    operational[0]["error"] = "worker failed"
    cases.append((operational, "operational"))
    invalid = deepcopy(base_rows)
    invalid[0]["finished"] = 1
    cases.append((invalid, "finished"))
    invalid_time = deepcopy(base_rows)
    invalid_time[0]["initialization_ms"] = 10_000.001
    cases.append((invalid_time, "initialization_ms"))
    invalid_rss = deepcopy(base_rows)
    invalid_rss[0]["peak_worker_rss_mib"] = 1024.001
    cases.append((invalid_rss, "peak_worker_rss_mib"))
    for rows, reason in cases:
        summary = compare_pairs(rows, cells, "screen")
        assert summary["decision"] == "REJECT", reason
        assert reason in " ".join(summary["reasons"])


def test_runtime_limits_are_inclusive_and_nonfinite_metrics_reject() -> None:
    rows, cells = _gains("screen")
    rows[0].update(initialization_ms=10_000, reset_ms=5_000,
                   action_latency_max_ms=5_000, peak_worker_rss_mib=1_024)
    assert compare_pairs(rows, cells, "screen")["decision"] == "RETAIN"
    for field in ("reset_ms", "action_latency_max_ms", "progress", "damage"):
        invalid = deepcopy(rows)
        invalid[0][field] = float("nan")
        summary = compare_pairs(invalid, cells, "screen")
        assert summary["decision"] == "REJECT", field
        assert field in " ".join(summary["reasons"])


def test_exact_repeats_validate_but_never_count_as_new_observations() -> None:
    rows, cells = _gains("screen")
    baseline = compare_pairs(rows, cells, "screen")
    repeated = compare_pairs(rows, cells, "screen")
    assert repeated["decision"] == "RETAIN"
    assert repeated["canonical_cells"] == baseline["canonical_cells"] == 32
    assert repeated["net_finish_gain"] == baseline["net_finish_gain"] == 3
    assert repeated["repeat_pairs_checked"] == 2
    rows[-1]["action_trace_sha256"] = "3" * 64
    mismatch = compare_pairs(rows, cells, "screen")
    assert mismatch["decision"] == "REJECT"
    assert any("repeat mismatch" in reason for reason in mismatch["reasons"])
    rows[-1]["action_trace_sha256"] = "2" * 64
    rows[-1]["progress"] = 0.99
    assert compare_pairs(rows, cells, "screen")["decision"] == "REJECT"


def test_repeat_receipt_with_invalid_runtime_rejects_without_counting_it() -> None:
    rows, cells = _gains("screen")
    rows[-1]["reset_ms"] = 5_000.01
    result = compare_pairs(rows, cells, "screen")
    assert result["decision"] == "REJECT"
    assert result["net_finish_gain"] == 3
    assert any("reset_ms" in reason for reason in result["reasons"])


def test_combined_gate_requires_every_phase_and_eighteen_net_finishes() -> None:
    summaries = {
        "screen": {"decision": "RETAIN", "canonical_cells": 32,
                   "control_finishes": 0, "candidate_finishes": 3, "net_finish_gain": 3},
        "confirmation": {"decision": "RETAIN", "canonical_cells": 64,
                         "control_finishes": 0, "candidate_finishes": 12, "net_finish_gain": 12},
        "blind": {"decision": "RETAIN", "canonical_cells": 32,
                  "control_finishes": 0, "candidate_finishes": 3, "net_finish_gain": 3},
    }
    for summary in summaries.values():
        summary.update(protocol_sha256="a" * 64, candidate_agent_sha256="b" * 64,
                       runner_sha256="c" * 64, decision_engine_sha256="d" * 64)
    exact = combined_decision(summaries)
    assert exact["decision"] == "RETAIN"
    assert exact["canonical_cells"] == 128
    assert exact["net_finish_gain"] == 18
    summaries["confirmation"]["net_finish_gain"] = 11
    summaries["confirmation"]["candidate_finishes"] = 11
    below = combined_decision(summaries)
    assert below["decision"] == "REJECT"
    assert below["net_finish_gain"] == 17
    assert any("18" in reason for reason in below["reasons"])
    summaries["confirmation"]["net_finish_gain"] = 12
    summaries["confirmation"]["candidate_finishes"] = 12
    summaries["blind"]["decision"] = "REJECT"
    assert combined_decision(summaries)["decision"] == "REJECT"
    summaries["blind"]["decision"] = "RETAIN"
    summaries["blind"]["canonical_cells"] = 31
    assert combined_decision(summaries)["decision"] == "INCOMPLETE"
    del summaries["blind"]
    assert combined_decision(summaries)["decision"] == "INCOMPLETE"


def test_combined_gate_rejects_extra_phase_and_identity_drift() -> None:
    summaries = {
        "screen": {"decision": "RETAIN", "canonical_cells": 32,
                   "control_finishes": 0, "candidate_finishes": 3, "net_finish_gain": 3},
        "confirmation": {"decision": "RETAIN", "canonical_cells": 64,
                         "control_finishes": 0, "candidate_finishes": 12, "net_finish_gain": 12},
        "blind": {"decision": "RETAIN", "canonical_cells": 32,
                  "control_finishes": 0, "candidate_finishes": 3, "net_finish_gain": 3},
    }
    for summary in summaries.values():
        summary.update(protocol_sha256="a" * 64, control_agent_sha256="b" * 64,
                       candidate_agent_sha256="c" * 64, model_sha256="d" * 64,
                       harness_sha256="e" * 64, runner_sha256="f" * 64,
                       decision_engine_sha256="0" * 64)
    assert combined_decision(summaries)["decision"] == "RETAIN"
    for field in ("protocol_sha256", "control_agent_sha256", "candidate_agent_sha256",
                  "model_sha256", "harness_sha256", "runner_sha256",
                  "decision_engine_sha256"):
        changed = deepcopy(summaries)
        changed["blind"][field] = "1" * 64
        result = combined_decision(changed)
        assert result["decision"] == "REJECT", field
        assert field in " ".join(result["reasons"])
    missing = deepcopy(summaries)
    del missing["blind"]["decision_engine_sha256"]
    assert combined_decision(missing)["decision"] == "REJECT"
    no_identity = deepcopy(summaries)
    for summary in no_identity.values():
        for field in ("protocol_sha256", "candidate_agent_sha256", "runner_sha256",
                      "decision_engine_sha256"):
            del summary[field]
    assert combined_decision(no_identity)["decision"] == "REJECT"
    inconsistent = deepcopy(summaries)
    inconsistent["blind"]["net_finish_gain"] = 4
    assert combined_decision(inconsistent)["decision"] == "REJECT"
    extra = deepcopy(summaries)
    extra["development"] = deepcopy(summaries["screen"])
    assert combined_decision(extra)["decision"] == "REJECT"
