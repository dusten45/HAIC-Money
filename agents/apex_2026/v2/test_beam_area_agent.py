import json
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.beam_area_agent import Agent


def test_recorded_collision_tick3_is_detected():
    ticks=json.loads((Path(__file__).parent/'results/beam-r3-failure-selected_ticks.json').read_text())
    tick=ticks[1]['ticks'][2];a=Agent()
    for body,state in zip([a.shadow.hull]+a.shadow.wheels,tick['bodies']):
        body.position=state['position'];body.angle=state['angle']
    free=np.ones((84,84),np.uint8);free[53,39]=0
    assert a._footprint_cost(free,*tick['pose'])>0


def test_single_interior_pixel_is_detected():
    a=Agent();a.shadow.reset(0,0,0)
    free=np.ones((84,84),np.uint8);free[64,42]=0
    assert a._footprint_cost(free,0,0,0)>0


def test_maximum_steering_outer_wheel_cell_is_detected():
    a=Agent();a.shadow.reset(0,0,.4)
    # Right front wheel reaches x1.568m, beyond former1.4m rectangle.
    for body in [a.shadow.hull]+a.shadow.wheels:body.position=(body.position.x+.3,body.position.y)
    free=np.ones((84,84),np.uint8);free[61,45]=0
    assert a._footprint_cost(free,.3,0,0)>0


def test_clean_road_has_no_false_collision():
    a=Agent();a.shadow.reset(40,.3,.2)
    free=np.ones((84,84),np.uint8)
    for _ in range(4):
        x,y,h,_,_=a.shadow.step((.3,.7,0))
        assert a._footprint_cost(free,x,y,h)==0


def test_overlapping_fixtures_count_pixel_only_once():
    a=Agent();a.shadow.reset(0,0,0)
    free=np.ones((84,84),np.uint8);free[64,42]=0
    assert a._footprint_cost(free,0,0,0)==1


def test_map_boundary_is_unsafe():
    a=Agent();a.shadow.reset(0,0,0)
    for body in [a.shadow.hull]+a.shadow.wheels:body.position=(body.position.x+32,body.position.y)
    assert a._footprint_cost(np.ones((84,84),np.uint8),32,0,0)>0
