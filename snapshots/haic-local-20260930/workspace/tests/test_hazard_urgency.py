import pytest

from research.hazard_urgency import rescale_quadratic_hazard_risk


def test_linear_urgency_restores_early_signal_from_quadratic_risk():
    square_risk = 0.0064
    urgency = 0.08

    assert rescale_quadratic_hazard_risk(square_risk, urgency, 1.0) == pytest.approx(0.08)
    assert rescale_quadratic_hazard_risk(square_risk, urgency, 2.0) == pytest.approx(0.0064)


def test_zero_urgency_stays_zero_for_linear_risk():
    assert rescale_quadratic_hazard_risk(0.0, 0.0, 1.0) == 0.0


@pytest.mark.parametrize("exponent", [0.0, -1.0, float("nan"), float("inf")])
def test_hazard_risk_rejects_non_positive_or_non_finite_exponents(exponent):
    with pytest.raises(ValueError):
        rescale_quadratic_hazard_risk(0.25, 0.5, exponent)
