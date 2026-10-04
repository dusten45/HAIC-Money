"""Exact small interval problems verify the geometric relaxation, not driving."""
import itertools
import math

import numpy as np
import pytest

from agents.apex_2026.research.speed_20261005.tile_time_lower_bound import (
    interval_coverage_width, projection_integral_certificate, qualification_count,
)


def brute_width(a, b, k, start=None):
    values = []
    for indices in itertools.combinations(range(len(a)), k):
        high = max(a[i] for i in indices)
        low = min(b[i] for i in indices)
        if start is None:
            values.append(max(0.0, high-low))
        else:
            values.append(max(high, start)-min(low, start))
    return min(values)


def test_qualification_rounds_up_unique_tiles():
    assert qualification_count(21) == 20
    assert qualification_count(20) == 19
    assert qualification_count(299) == 285


def test_allowed_omission_does_not_require_remote_outlier():
    assert interval_coverage_width([0, 1, 2, 3, 100], [0, 1, 2, 3, 100], 4) == 3


def test_wheel_contact_intervals_can_overlap_without_center_spanning_vertices():
    assert interval_coverage_width([0, 1, 2], [5, 6, 7], 3) == 0


def test_required_start_cannot_be_excluded_from_optimal_span():
    assert interval_coverage_width([0, 1, 2, 100], [0, 1, 2, 100], 3, start=50) == 50


def test_interval_algorithm_matches_exhaustive_subsets_with_ties():
    rng = np.random.default_rng(2371)
    for _ in range(30):
        a = rng.integers(-4, 5, 6).astype(float)
        b = a + rng.integers(0, 5, 6)
        for k in (1, 4, 6):
            for start in (None, -8., 0., 8.):
                actual = interval_coverage_width(a, b, k, start=start)
                assert actual == pytest.approx(brute_width(a, b, k, start))


def test_midpoint_certificate_encloses_rectangle_perimeter():
    vertices = np.array([[[-2, -1]], [[2, -1]], [[2, 1]], [[-2, 1]]], dtype=float)
    result = projection_integral_certificate(vertices, 4, 0., 1024)
    assert result['lower_m'] <= 12 <= result['upper_m']
    assert result['lower_m'] > 11.97


def test_contact_radius_is_a_relaxation_and_start_can_only_strengthen_bound():
    vertices = np.array([[[-2, -1]], [[2, -1]], [[2, 1]], [[-2, 1]]], dtype=float)
    plain = projection_integral_certificate(vertices, 4, 0., 256)
    inflated = projection_integral_certificate(vertices, 4, 1., 256)
    anchored = projection_integral_certificate(vertices, 4, 1., 256, start=np.array([0., 5.]))
    assert inflated['lower_m'] < plain['lower_m']
    assert anchored['lower_m'] >= inflated['lower_m']


def test_invalid_projection_inputs_rejected():
    with pytest.raises(ValueError):
        interval_coverage_width([2], [1], 1)
    with pytest.raises(ValueError):
        interval_coverage_width([0], [1], 2)
    with pytest.raises(ValueError):
        projection_integral_certificate(np.zeros((2, 1, 2)), 2, -1, 128)
