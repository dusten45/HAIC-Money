import numpy as np
import pytest

from scripts.diagnose_rlpd_recovery_handoff import events


def telemetry(n=200):
    action = np.tile([.5, .9, 0], (n, 1))
    action[:100, 1] = 0
    return {"applied_action": action, "speed": np.linspace(10, 30, n + 1),
            "center_error": np.ones(n + 1), "curvature": np.full(n + 1, .03),
            "oracle_action": np.tile([.5, 0, .5], (n + 1, 1)),
            "damage": np.zeros(n + 1), "progress": np.linspace(0, 1, n + 1)}


def test_late_event_is_not_an_initial_recovery_failure():
    result = events(telemetry(), 10, 25)
    first = result["events_after_handoff"]["curve-entry-overspeed"]
    assert first["decision"] == 100
    assert first["seconds_after_handoff"] == pytest.approx(5.2)
    assert first["beyond_five_second_followup"] is True
    assert result["causal"] is False
    assert result["first_five_seconds_max_speed_m_s"] == pytest.approx(19.7)


def test_sustained_lateral_event_and_opposition():
    arrays = telemetry()
    arrays["center_error"][60:62] = 7
    arrays["center_error"][40] = 7
    arrays["applied_action"][35, 0] = -.7
    result = events(arrays, 10, 25)
    assert result["events_after_handoff"]["lateral-excursion"]["decision"] == 60
    assert result["events_after_handoff"]["steering-opposition"]["decision"] == 35
    assert result["first_type"] == "steering-opposition"


def test_prefix_damage_does_not_become_added_damage():
    arrays = telemetry()
    arrays["damage"][:] = .2
    result = events(arrays, 10, 25)
    assert result["events_after_handoff"]["added-damage"] is None


def test_terminal_collision_is_not_lost():
    arrays = telemetry()
    arrays["damage"][-1] = .8
    result = events(arrays, 10, 25)
    assert result["events_after_handoff"]["added-damage"]["decision"] == 200
    assert result["events_after_handoff"]["added-damage"]["damage"] == .8
    assert result["terminal_damage_observation_included"] is True


def test_invalid_or_nonfinite_handoff():
    with pytest.raises(ValueError, match="bounds"):
        events(telemetry(), -1, 25)
    arrays = telemetry()
    arrays["speed"][10] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        events(arrays, 10, 25)
