"""Known image warps exercise only public-pixel registration, without resets."""
import importlib.util
from pathlib import Path
import cv2
import numpy as np
import pytest


def module():
    p=Path(__file__).parent/'diagnostics/motion_registration.py'
    assert p.exists(), 'yaw-constrained registration is not implemented'
    spec=importlib.util.spec_from_file_location('motion_registration_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def textured():
    rng=np.random.default_rng(912)
    f=cv2.GaussianBlur(rng.uniform(0,1,(84,84)).astype(np.float32),(3,3),0)
    f[74:]=0
    return f


def warp(frame,yaw,dx,dy):
    # Independent metric-to-image construction; positive yaw is a right turn.
    theta=-yaw*.08;c,s=np.cos(theta),np.sin(theta)
    linear=np.array([[c,-s*1.3608/1.701],[s*1.701/1.3608,c]])
    center=np.array([42.,63.]);offset=center-linear@center+np.array([dx,dy])
    return cv2.warpAffine(frame,np.column_stack((linear,offset)).astype(np.float32),(84,84))


@pytest.mark.parametrize('yaw,dx,dy',[(0.,0.,7.),(2.,-1.3,8.2),(-3.,2.4,11.1)])
def test_recovers_signed_translation_after_anisotropic_yaw(yaw,dx,dy):
    f=textured();g=warp(f,yaw,dx,dy)
    result=module().estimate_motion(f,g,yaw,yaw)
    assert result['valid'],result
    np.testing.assert_allclose(result['translation_pixels'],[dx,dy],atol=.25)
    np.testing.assert_allclose([result['lateral_m_s'],result['forward_m_s']],[-dx/1.3608/.08,dy/1.701/.08],atol=2.4)


def test_blank_frames_explicitly_return_unknown_not_zero_slip():
    z=np.zeros((84,84),np.float32)
    result=module().estimate_motion(z,z,0.,0.)
    assert not result['valid']
    assert result['slip_rad'] is None and result['lateral_m_s'] is None


def test_one_dimensional_texture_cannot_observe_both_translation_components():
    f=np.tile(np.linspace(.2,.8,84,dtype=np.float32),(84,1))
    result=module().estimate_motion(f,f,0.,0.)
    assert not result['valid']
    assert result['slip_rad'] is None


def test_unrelated_frames_fail_confidence_gate():
    f=textured();g=np.random.default_rng(415).uniform(0,1,(84,84)).astype(np.float32)
    result=module().estimate_motion(f,g,0.,0.)
    assert not result['valid']


def test_invalid_inputs_fail_closed():
    z=np.zeros((84,84),np.float32)
    for bad in (z[:40],z+np.nan):
        assert not module().estimate_motion(bad,z,0.,0.)['valid']


def test_interval_truth_expresses_displacement_in_current_body_axes():
    m=module()
    assert hasattr(m,'interval_truth'), 'diagnostic coordinate alignment missing'
    before={'t':1.,'position':[0.,0.]}
    after={'t':1.08,'position':[.8,4.], 'angle':np.pi/2, 'velocity':[10.,50.]}
    truth=m.interval_truth(before,after)
    assert truth['lateral_m_s']==pytest.approx(50.)
    assert truth['forward_m_s']==pytest.approx(-10.)


def test_four_tick_yaw_quadrature_uses_discrete_endpoint_samples():
    f=textured()
    # Linearly varying raw-tick yaw: .5,1,1.5,2 => integrated mean1.25.
    g=warp(f,1.25,-.5,8.)
    result=module().estimate_motion(f,g,0.,2.)
    assert result['valid'],result
    np.testing.assert_allclose(result['translation_pixels'],[-.5,8.],atol=.25)


def test_pixel_speed_consistency_rejects_spurious_near_stationary_match():
    f=textured();g=warp(f,0.,0.,.4)
    result=module().estimate_motion(f,g,0.,0.,speed_hint=60.)
    assert not result['valid']
    assert result['slip_rad'] is None
