"""Retain the transported pass constraint across real raster detection misses."""
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_memory_corridor_agent import Agent
from agents.apex_2026.fast_early_circle_agent import Agent as Reference
from agents.apex_2026.research.speed_20261005.graph_geometry import distance_to_path


def image(step):
    return np.load(Path(__file__).parent/'fixtures'/f'memory-circle-track2-step{step}.npz')['frame']


def prepare(cls,step):
    a=cls();a.pass_side=1;a.pass_missing=0
    if step==95:a.pass_x=0.;a.pass_y=30.570252792475014
    else:a.pass_x=-1.5513750081651339;a.pass_y=17.83264746227709
    return a


def route(a,frame):
    road=a._road(frame);circles=a._circles(frame,road)
    path,imminent,distance=a._route(road[0],road[1],circles,a._speed(frame),a._yaw(frame))
    return road,path,circles


def test_saved_misses_retain_collision_avoidance_corridor():
    for step in [95,97]:
        a=prepare(Agent,step);road,path,circles=route(a,image(step));assert circles==[]
        remembered=np.array([a.pass_x,a.pass_y]);gap=distance_to_path(remembered[None],np.column_stack((path,road[0])))[0]
        assert gap>=3.3, 'Camera reference must retain margin beyond body+circle2.8m during a brief detection miss'


def test_transport_occurs_once_on_saved_miss():
    frame=image(97);a=prepare(Agent,97);x,y=a.pass_x,a.pass_y-.08*a._speed(frame);angle=.08*a._yaw(frame)
    expected=np.array([x*np.cos(angle)-y*np.sin(angle),x*np.sin(angle)+y*np.cos(angle)])
    route(a,frame);np.testing.assert_allclose([a.pass_x,a.pass_y],expected,atol=1e-12);assert a.pass_missing==1


def test_visible_circle_route_remains_exact():
    frame=image(96);a,b=prepare(Agent,97),prepare(Reference,97)
    ra,pa,ca=route(a,frame);rb,pb,cb=route(b,frame)
    assert len(ca)==1 and ca==cb
    np.testing.assert_array_equal(pa,pb)


def test_expired_memory_and_reset_do_not_add_constraints():
    frame=image(97);a,b=prepare(Agent,97),prepare(Reference,97);a.pass_missing=b.pass_missing=4
    _,pa,_=route(a,frame);_,pb,_=route(b,frame);np.testing.assert_array_equal(pa,pb);assert a.pass_side==0 and a.pass_y is None
    a.reset();assert a.pass_side==0 and a.pass_missing==0 and a.pass_y is None


def test_clear_and_invalid_actions_stay_exact():
    frame=np.full((84,84),.67,np.float32);frame[:73,33:52]=.4;frame[73:]=0.
    for observation in [np.stack([frame]*4),np.full((4,84,84),np.nan)]:
        np.testing.assert_array_equal(Agent().act(observation),Reference().act(observation))
