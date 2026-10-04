import numpy as np
from agents.apex_2026.v2.curvature_memory_agent import Agent


def plan(x=0.):
    s=np.arange(1.,21.)
    return np.c_[np.full(len(s),x),s],s,np.full(len(s),100.)


def memory(point=(0.,10.),remaining=10.,age=0):
    return dict(point=np.array(point),tangent=np.array([0.,1.]),cap=30.,remaining=remaining,age=age)


def test_transport_translation_rotation_and_measured_slip():
    p,t=Agent._transport(np.array([0.,10.]),np.array([0.,1.]),50.,0.,0.,False)
    assert np.allclose(p,[0.,6.]) and np.allclose(t,[0.,1.])
    p,t=Agent._transport(np.array([0.,10.]),np.array([0.,1.]),0.,2.5,0.,False)
    assert np.allclose(p,[-10*np.sin(.2),10*np.cos(.2)])
    p,_=Agent._transport(np.array([0.,10.]),np.array([0.,1.]),50.,0.,.1,True)
    assert np.allclose(p,[-4*np.sin(.1),10-4*np.cos(.1)])


def test_retains_bend_on_aligned_new_straight_plan():
    a=Agent();a.curvature_memory=[memory()]
    cap=a._memory_cap(*plan(),50.,0.,0.,True)
    assert abs(cap-np.sqrt(30**2+130*6))<1e-6
    assert a.memory_diagnostics['retained']==1


def test_passed_point_and_changed_branch_release():
    a=Agent();a.curvature_memory=[memory(point=(0.,2.),remaining=2.)]
    assert a._memory_cap(*plan(),50.,0.,0.,False)==100.
    a.curvature_memory=[memory()]
    assert a._memory_cap(*plan(x=8.),0.,0.,0.,False)==100.


def test_opposite_direction_branch_and_ttl_release():
    a=Agent();m=memory();m['tangent']=np.array([0.,-1.]);a.curvature_memory=[m]
    assert a._memory_cap(*plan(),0.,0.,0.,False)==100.
    a.curvature_memory=[memory()]
    for _ in range(3):assert a._memory_cap(*plan(),0.,0.,0.,False)<100.
    assert a._memory_cap(*plan(),0.,0.,0.,False)==100.


def test_reset_clears_memory_and_invalid_observation():
    a=Agent();a.curvature_memory=[memory()];a.reset();assert not a.curvature_memory
    assert np.isfinite(a.act(np.zeros((4,84,84),np.float32))).all()


def test_unmodified_planner_steering_and_allocator_when_memory_inactive():
    import importlib.util
    from pathlib import Path
    from agents.apex_2026.diagnostics.hud_dynamics_calibration import render_hud
    source=Path(__file__).parent/'results/geodesic_sources/b182dc495c41.py'
    spec=importlib.util.spec_from_file_location('frozen_geodesic_r6_parity',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    baseline=module.Agent();candidate=Agent()
    frame=render_hud(0.,0.,speed=50.)
    frame[:74]=.65;frame[:74,28:57]=.4
    obs=np.repeat(frame[None],4,axis=0).astype(np.float32)
    for _ in range(3):
        expected=baseline.act(obs);actual=candidate.act(obs)
        assert np.array_equal(actual,expected)
        assert candidate.diagnostics['path']==baseline.diagnostics['path']
        assert candidate.diagnostics['unmodified_target']==baseline.diagnostics['target_speed']
        assert candidate.diagnostics['retained']==0
