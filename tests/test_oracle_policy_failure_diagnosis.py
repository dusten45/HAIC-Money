import numpy as np

from scripts.diagnose_oracle_policy_failures import (
    first_failure_precursor,
    first_material_divergence,
)


def _trajectory(count=8):
    return {
        "post_position": np.zeros((count, 2), dtype=np.float64),
        "post_heading": np.zeros(count, dtype=np.float64),
        "post_speed": np.ones(count, dtype=np.float64),
    }


def test_material_divergence_requires_one_physical_signal_to_persist():
    policy = _trajectory()
    oracle = _trajectory()
    policy["post_position"][2:, 0] = 2.5
    policy["post_speed"][3] = 4.0
    result = first_material_divergence(policy, oracle, sustained=3)
    assert result is not None
    assert result["step"] == 2
    assert result["criteria"] == ["position"]


def test_alternating_small_signals_do_not_create_a_false_divergence():
    policy = _trajectory()
    oracle = _trajectory()
    policy["post_position"][::2, 0] = 2.1
    policy["post_speed"][1::2] = 3.2
    assert first_material_divergence(policy, oracle, sustained=3) is None


def test_failure_taxonomy_keeps_first_heading_precursor_ahead_of_contact():
    count = 6
    trace = {
        "step": np.arange(count),
        "pre_speed": np.full(count, 5.0),
        "curvature": np.zeros(count),
        "policy_or_oracle_action": np.zeros((count, 3)),
        "oracle_action_at_state": np.zeros((count, 3)),
        "heading_error": np.asarray([0.0, 0.6, 0.6, 0.6, 0.8, 0.9]),
        "center_error": np.zeros(count),
        "damage": np.asarray([0.0, 0.0, 0.0, 0.2, 0.2, 0.2]),
        "progress": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
    }
    result = first_failure_precursor(trace, finished=False)
    assert result["step"] == 1
    assert result["mode"] == "heading-error"
    assert result["causal"] is False
