import numpy as np
import pytest

from agents.apex_2026.evaluate import (
    MANDATORY_CELLS, assess, end_reason, pace_profile, rejection_streak, validate_action,
)


def row(track, seed, *, finished=True, seconds=12.0):
    return {"track_id": track, "seed": seed, "finished": finished,
            "lap_time_ms": round(seconds * 1000) if finished else None,
            "progress": 1.0 if finished else 0.99, "collision_count": 0}


def cells():
    return [row(*cell) for cell in MANDATORY_CELLS], [
        row(track, seed) for track in range(1, 5) for seed in (90001, 90002, 90003, 90004)
    ]


def test_dnf_is_never_treated_as_a_fast_lap():
    mandatory, extra = cells()
    mandatory[0] = row(*MANDATORY_CELLS[0], finished=False)
    verdict = assess(mandatory, extra, limit_seconds=13)
    assert verdict["passed"] is False
    assert verdict["original_target_met"] is False


def test_missing_mandatory_and_duplicate_cells_are_rejected():
    mandatory, extra = cells()
    with pytest.raises(ValueError, match="mandatory"):
        assess(mandatory[:-1], extra, limit_seconds=13)
    with pytest.raises(ValueError, match="duplicate"):
        assess(mandatory, extra + [extra[0]], limit_seconds=13)


def test_relaxation_does_not_relabel_original_target():
    mandatory, extra = cells()
    mandatory[0] = row(*MANDATORY_CELLS[0], seconds=14)
    assert not assess(mandatory, extra, limit_seconds=13)["passed"]
    verdict = assess(mandatory, extra, limit_seconds=15)
    assert verdict["passed"]
    assert not verdict["original_target_met"]
    assert [pace_profile(n) for n in (0, 2, 3, 5, 6, 100)] == [13, 13, 15, 15, 18, 18]


def test_generalization_failure_and_per_track_floor_fail():
    mandatory, extra = cells()
    extra[0] = row(1, 90001, finished=False)
    assert assess(mandatory, extra, limit_seconds=13)["passed"]
    extra[1] = row(1, 90002, finished=False)
    assert not assess(mandatory, extra, limit_seconds=13)["passed"]


@pytest.mark.parametrize("action", ([0, 1.1, 0], [2, 0, 0], [0, 0, -0.1], [0, np.nan, 0], [0, 0]))
def test_invalid_actions_are_rejected_without_clipping(action):
    with pytest.raises(ValueError):
        validate_action(action)


def test_valid_action_is_float32_and_unmodified():
    action = validate_action([-0.4, 0.6, 0.2])
    np.testing.assert_array_equal(action, np.asarray([-0.4, 0.6, 0.2], np.float32))


def test_only_distinct_real_rejections_relax_the_next_trial():
    mandatory, extra = cells()
    mandatory[0] = row(*MANDATORY_CELLS[0], finished=False)
    reports = [{"freeze": {"source_sha256": f"candidate-{i}", "parameters": {},
                           "selected_profile_seconds": 13}, "rows": mandatory} for i in range(3)]
    assert rejection_streak(reports) == 3
    assert pace_profile(rejection_streak(reports)) == 15
    with pytest.raises(ValueError, match="repeat"):
        rejection_streak(reports + [reports[0]])
    reports[-1]["freeze"]["selected_profile_seconds"] = 18
    with pytest.raises(ValueError, match="prospective"):
        rejection_streak(reports)


def test_playfield_termination_is_not_a_time_limit():
    assert end_reason({}, finished=False, terminated=True, truncated=False) == "terminated"
    assert end_reason({}, finished=False, terminated=False, truncated=True) == "time_limit"
    assert end_reason({"retire_reason": "crash"}, finished=False, terminated=True, truncated=False) == "crash"
    assert end_reason({}, finished=True, terminated=False, truncated=True) is None
