"""Analytic C2 endpoints and coupled braking checks for an offline study."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SOURCE=Path(__file__).resolve().parents[1]/"research/speed_20261005/quintic_preview_study.py"


def _module():
    spec=importlib.util.spec_from_file_location("quintic_preview_math_test",SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quintic_matches_both_positions_headings_and_curvatures():
    fit=getattr(_module(),"quintic_coefficients",None)
    assert fit is not None
    length=20.
    c=fit(length,0.,0.,-.015,-7.,-.30,.012)
    d=np.polynomial.polynomial.polyder(c)
    dd=np.polynomial.polynomial.polyder(d)
    for at,position,slope,second in [(0.,0.,0.,-.015),(1.,-7.,-.30,.012)]:
        assert np.polynomial.polynomial.polyval(at,c)==pytest.approx(position,abs=1e-10)
        assert np.polynomial.polynomial.polyval(at,d)/length==pytest.approx(slope,abs=1e-10)
        assert np.polynomial.polynomial.polyval(at,dd)/length**2==pytest.approx(second,abs=1e-10)


def test_c2_quintic_keeps_a_constant_curvature_quadratic_exact():
    fit=getattr(_module(),"quintic_coefficients",None)
    assert fit is not None
    length=12.
    c=fit(length,1.,.10,.02,1.+.10*length+.01*length**2,.10+.02*length,.02)
    at=np.linspace(0.,1.,21)
    np.testing.assert_allclose(np.polynomial.polynomial.polyval(at,c),
        1.+.10*at*length+.01*(at*length)**2,atol=1e-10)


def test_lateral_force_consumption_reduces_available_braking():
    capacity=getattr(_module(),"braking_capacity",None)
    assert capacity is not None
    assert capacity(80.,0.)==pytest.approx(100.)
    assert capacity(80.,.028)==pytest.approx(np.sqrt(190.**2-(80.**2*.028)**2))
    assert capacity(80.,.04)==0.


def test_reaction_distance_cannot_be_used_for_braking():
    envelope=getattr(_module(),"speed_envelope",None)
    assert envelope is not None
    arc=np.linspace(0.,20.,81)
    curvature=np.where(arc<5.,0.,.08)
    result=envelope(arc,curvature,80.,0.)
    assert result["reaction_distance_m"]==pytest.approx(6.4)
    assert not result["reaction_lateral_ok"]
    assert not result["feasible_under_model"]


def test_straight_camera_reference_preserves_current_speed_under_model():
    envelope=getattr(_module(),"speed_envelope",None)
    assert envelope is not None
    arc=np.linspace(0.,40.,161)
    result=envelope(arc,np.zeros_like(arc),80.,0.)
    assert result["feasible_under_model"]
    assert result["initial_lateral_ok"]
