import ast,json,math
from pathlib import Path
import numpy as np
from agents.apex_2026.v2.beam_robust_agent import Agent


def test_collision_mask_has_one_pixel_clearance_and_preserves_input():
    free=np.ones((84,84),np.uint8);free[40,40]=0;before=free.copy()
    robust=Agent._collision_mask(free)
    assert np.array_equal(free,before)
    assert np.all(robust[39:42,39:42]==0)
    assert robust[38,40]==1 and robust[42,40]==1
    assert robust[0,40]==0


def test_previously_safe_zero_clearance_sequence_is_rejected():
    case=json.loads((Path(__file__).parent/'results/beam-r5-margin-counterexample-input.json').read_text());a=Agent();free=np.array([[int(x) for x in row] for row in case['free_rows']],np.uint8)
    costs=[]
    for mask in [free,a._collision_mask(free)]:
        a.shadow.reset(case['speed']*math.cos(case['slip']),case['yaw'],case['wheel'],lateral_speed=case['speed']*math.sin(case['slip']),throttle=case['throttle'],omegas=case['omegas']);cost=0
        for action in case['sequence']:
            for _ in range(4):
                x,y,h,_,_=a.shadow.step(action);cost+=a._footprint_cost(mask,x,y,h)
        costs.append(cost)
    assert costs[0]==0
    assert costs[1]>0


def test_wide_clean_road_remains_clear():
    a=Agent();a.shadow.reset(40,0,0)
    mask=a._collision_mask(np.ones((84,84),np.uint8))
    for _ in range(4):
        x,y,h,_,_=a.shadow.step((0,.7,0));assert a._footprint_cost(mask,x,y,h)==0


def test_source_changes_are_isolated_to_collision_mask_and_act():
    base=Path(__file__).parent
    def methods(name):
        tree=ast.parse((base/name).read_text())
        for node in tree.body:
            if isinstance(node,ast.ClassDef) and node.name=='Agent':
                node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name not in ('act','_collision_mask')]
        return ast.dump(tree)
    assert methods('beam_area_agent.py')==methods('beam_robust_agent.py')
