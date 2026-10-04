"""HUD calibration uses synthetic indicators only; never resets a simulator."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest


def module():
    p=Path(__file__).parent/'diagnostics/hud_dynamics_calibration.py'
    assert p.exists(), 'HUD decoder missing'
    spec=importlib.util.spec_from_file_location('hud_calibration_test',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_blank_hud_decodes_exact_zero():
    m=module()
    assert m.decode_hud(np.zeros((84,84),np.float32)) == {'yaw_rate':0.,'wheel_angle':0.}


@pytest.mark.parametrize('yaw,steer',[(6,.4),(-6,-.4),(1,.1),(-1,-.1),(.1,.01),(-.1,-.01),(0,0)])
def test_synthetic_hud_decodes_signed_dynamics(yaw,steer):
    m=module(); frame=m.render_hud(yaw,steer,speed=100,rpm=200)
    decoded=m.decode_hud(frame)
    assert abs(decoded['yaw_rate']-yaw)<.10
    assert abs(decoded['wheel_angle']-steer)<.005


def test_speed_rpm_and_other_bar_do_not_contaminate_decoder():
    m=module()
    baseline=m.decode_hud(m.render_hud(6,-.4,0,0))
    for speed,rpm in [(0,400),(50,200),(100,0),(100,400)]:
        assert m.decode_hud(m.render_hud(6,-.4,speed,rpm))==baseline


def test_yaw_decoder_mirror_bias_below_point01_radians_per_second():
    m=module()
    a=m.decode_hud(m.render_hud(2.134,0))['yaw_rate']
    b=m.decode_hud(m.render_hud(-2.134,0))['yaw_rate']
    assert abs(a+b)<.01
