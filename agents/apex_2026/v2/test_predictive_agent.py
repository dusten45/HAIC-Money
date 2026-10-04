import numpy as np
from agents.apex_2026.v2.predictive_agent import Agent


def scene(end=4, obstacle=False):
    f=np.full((84,84),.65,np.float32)
    f[end:74,29:56]=.4
    f[74:]=0
    if obstacle: f[35:40,40:45]=.68
    return f


def test_rolling_calibration_and_throttle_memory():
    a=Agent()
    v,g=a._longitudinal(np.array([50.]),np.array([0.]),np.array([.3]),np.array([0.]),np.array([0.]),ticks=4)
    assert abs(v[0]-50.97358)<.05
    assert g[0]==.3
    v,g=a._longitudinal(np.array([50.]),np.array([0.]),np.array([0.]),np.array([.1]),np.array([.8]),ticks=16)
    assert abs(v[0]-40.1752)<.2
    assert g[0]==0


def test_unknown_motion_retains_state_and_increases_uncertainty():
    a=Agent(); a.slip_estimate=.15; a.slip_uncertainty=.03
    a._update_slip(50.,0.,False,0.)
    assert a.slip_estimate>.10
    assert a.slip_uncertainty>.03
    a._update_slip(50.,.1,True,1.)
    assert abs(a.slip_estimate-.14)<1e-6
    a.reset()
    assert a.internal_throttle==0 and a.prediction is None


def test_pose_anchored_motion_and_actuator_slew():
    a=Agent()
    trajectory=a._rollouts(30.,0.,0.,0.,np.array([.4]),np.array([0.]),np.array([0.]),np.array([100]),np.array([0]))
    assert trajectory['x'][0,0]==0
    assert trajectory['y'][0,0]==0
    assert abs(trajectory['wheel'][0,1]-.06)<1e-7
    assert trajectory['x'][0,-1]>0
    assert np.isfinite(trajectory['y']).all()


def test_obstacle_distance_and_footprint():
    a=Agent(); f=scene(obstacle=True)
    field=a._metric_clearance(f)
    clear=a._sample_clearance(field,np.array([0.]),np.array([(63-37)/1.701]))
    assert clear[0]<.2
    assert a._sample_clearance(field,np.array([0.]),np.array([5.]))[0]>5


def test_straight_accelerates_and_close_deadend_brakes():
    a=Agent()
    path=np.array([[42.,y] for y in range(58,7,-2)],np.float32)
    straight=a._select(scene(),path,45.,0.,0.)
    assert abs(straight[0])<.1 and straight[1]>0 and straight[2]==0
    a.reset()
    path=path[:3]
    deadend=a._select(scene(end=48),path,65.,0.,0.)
    assert deadend[1]==0 and deadend[2]>=.4


def test_invalid_pixels_finite_and_reset():
    a=Agent(); a.internal_throttle=.8
    assert np.isfinite(a.act(np.full((4,84,84),np.nan,np.float32))).all()
    a.reset()
    assert a.internal_throttle==0
