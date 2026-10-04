import numpy as np
from agents.apex_2026.v2.predictive_r1 import Agent


def test_objective_compares_only_executed_prefix_when_stops_safe():
    # Equal first actions must have equal utility regardless of where the
    # hypothetical emergency tail stops. Future stopping distance is a gate.
    a=Agent()
    assert a._utility(4.2,48.,.2)>a._utility(4.,45.,.2)
    assert a._choose([2.,9.],[True,False],[0.,.01])==0
    assert a._choose([2.,9.],[False,False],[3.,1.])==1


def test_brake_tail_reaches_stop_and_keeps_pose_anchor():
    a=Agent(); path=np.array([[0.,y] for y in np.arange(0,41,2)])
    f=np.zeros((84,84),np.float32);f[:74,20:65]=.4
    field=a._metric_clearance(f)
    r=a._certificate([0.,0.,.7],45.,0.,0.,0.,np.ones(4)*45/.54,path,field)
    assert r['stopped'] and r['safe']
    assert abs(r['first'][0])<1e-6 and r['first'][1]>0
    assert r['first'][3]<45
    assert r['stop_y']<10


def test_physics_preserves_spin_instead_of_clipping_yaw():
    a=Agent();a.shadow.reset(40,5.,.1,lateral_speed=10.,omegas=[80,80,190,190])
    r=a.shadow.step([.4,1,0],4)
    assert abs(r[4])>2.


def test_invalid_pixels_and_reset():
    a=Agent(); a.internal_throttle=.7;a.prediction={'speed':50.,'yaw':1.}
    assert np.isfinite(a.act(np.full((4,84,84),np.nan,np.float32))).all()
    a.reset();assert a.internal_throttle==0 and a.prediction is None


def test_tail_geometry_does_not_change_identical_executed_prefix_reward():
    a=Agent();frame=np.full((84,84),.4,np.float32);frame[74:]=0
    field=a._metric_clearance(frame)
    p=np.array([[0.,y] for y in np.arange(0,41,2)])
    turn=p.copy();turn[:,0]=np.maximum(turn[:,1]-10,0)*.4
    state=([0.,1.,0.],70.,0.,0.,0.,np.ones(4)*70/.54)
    r1=a._certificate(*state,p,field)
    r2=a._certificate(*state,turn,field)
    assert np.allclose(r1['first'],r2['first'],atol=1e-6)
    assert abs(r1['utility']-r2['utility'])<1e-6
    assert abs(r1['stop_y']-r2['stop_y'])>.01
