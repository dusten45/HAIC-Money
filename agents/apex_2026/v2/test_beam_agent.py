import importlib.util
import numpy as np
from agents.apex_2026.v2.shadow_physics import ShadowCar


def module():
    spec=importlib.util.find_spec('agents.apex_2026.v2.beam_agent')
    assert spec is not None,'sequential exact-physics beam planner is missing'
    return importlib.import_module(spec.name)


def test_internal_snapshot_restores_identical_counterfactual():
    cls=module().Agent
    shadow=ShadowCar();shadow.reset(35,.3,-.1,throttle=.3,omegas=[65,65,70,70])
    shadow.step((.2,.6,0),4)
    saved=cls.snapshot(shadow)
    expected=shadow.step((-.25,0,.3),8)
    cls.restore(shadow,saved)
    shadow.step((.5,1,0),12)
    cls.restore(shadow,saved)
    observed=shadow.step((-.25,0,.3),8)
    assert np.allclose(observed,expected,atol=2e-5)


def test_objective_rewards_safe_progress_and_does_not_reward_waiting():
    cls=module().Agent
    stopped=cls.stage_cost(0,0,0,0,30)
    moving=cls.stage_cost(2,0,0,25,30)
    assert moving<stopped
    assert cls.stage_cost(4,0,0,25,30)<moving


def test_objective_progress_is_additive_across_prefixes():
    cls=module().Agent
    a=cls.stage_cost(1,0,0,30,30)+cls.stage_cost(2,0,0,30,30)
    b=cls.stage_cost(3,0,0,30,30)+cls.stage_cost(0,0,0,30,30)
    assert np.isclose(a,b)


def test_invalid_observation_brakes():
    agent=module().Agent()
    assert np.array_equal(agent.act(np.full((4,84,84),np.nan)),[0,0,.5])


def test_clear_road_from_rest_accelerates():
    from agents.apex_2026.v2.diagnostics.shadow_rpm import render
    frame=render([0,0,0,0],speed=0)
    frame[:74]=.6;frame[:74,25:59]=.4
    obs=np.repeat(frame[None],4,axis=0)
    agent=module().Agent();action=agent.act(obs)
    assert action[1]>0 and action[2]==0
    assert len(agent.diagnostics['beam_sequence'])==8
    assert agent.diagnostics['collision_checked_raw_ticks']>100


def test_beam_represents_opposite_steering_and_pedal_changes_in_one_sequence():
    agent=module().Agent()
    y=np.linspace(0,40,81)
    reference=np.c_[3*np.sin(y/5),y]
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
    agent.shadow.reset(28,0,0,throttle=.1)
    best,_=agent._search(reference,arc,np.full(len(arc),35.),np.ones((84,84),np.uint8),agent.snapshot(agent.shadow))
    actions=np.asarray(best['sequence'])
    assert np.any(actions[:,0]>.1) and np.any(actions[:,0]<-.1)
    assert actions[0,1]>0 and np.any(actions[1:,1]<actions[0,1])
    # Coasting also changes pedals; braking itself is required separately by
    # the high-speed visibility-boundary counterexample below.
    assert best['progress']>10


def test_textured_visual_motion_path_has_complete_metric_calibration():
    road=module()._Road()
    rng=np.random.default_rng(51)
    frame=rng.uniform(.25,.7,(84,84)).astype(np.float32)
    yaw,slip,valid=road._motion(frame,frame)
    assert valid
    assert abs(yaw)<1e-5 and abs(slip)<1e-5


def test_beam_retains_early_braking_when_late_braking_crosses_visible_boundary():
    agent=module().Agent()
    y=np.linspace(0,37,75)
    reference=np.c_[np.zeros(len(y)),y]
    targets=np.sqrt(24**2+130*(37-y))
    free=np.zeros((84,84),np.uint8);free[:74,25:59]=1
    agent.shadow.reset(80,0,0,omegas=[80/.54]*4)
    initial=agent.snapshot(agent.shadow)
    best,_=agent._search(reference,y,targets,free,initial)
    agent.restore(agent.shadow,initial)
    violations=0
    for action in best['sequence']:
        for _ in range(4):
            x,forward,h,_,_=agent.shadow.step(action)
            violations+=agent._footprint_cost(free,x,forward,h)
    assert violations==0, 'a zero-collision braking sequence exists but beam discarded it'
    assert best['cost']<46, 'must retain a sequence at least as good as constant braking'
