"""Single-axis Frenet refinement must preserve r1 control behavior."""
import ast
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import cv2


def agent():
    p=Path(__file__).with_name('geodesic_frenet_agent.py')
    assert p.exists(), 'Frenet-only ablation is not implemented'
    s=importlib.util.spec_from_file_location('geodesic_frenet',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


class FrenetOnlyTests(unittest.TestCase):
    def test_wide_corner_reduces_bending_without_losing_clearance(self):
        a=agent();mask=np.zeros((84,84),np.uint8)
        reference=np.array([[42.,58.],[42,50],[42,42],[40,34],[32,30],[24,30],[16,30],[5,30]])
        cv2.polylines(mask,[reference.astype(np.int32)],False,1,19)
        line=a._racing_line(reference,mask);reference=a._resample(reference)
        measure=lambda p:np.max(np.abs(a._geometry((p-[42,63])/[1.3608,-1.701])[1][3:-3]))
        self.assertLess(measure(line),.95*measure(reference))
        distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        x,y=np.rint(line).astype(int).T
        self.assertGreaterEqual(distance[y,x].min(),3.)

    def test_r1_narrow_gap_stays_blocked_in_this_isolated_ablation(self):
        a=agent();frame=np.full((84,84),.15,np.float32);frame[:74,33:52]=.4
        cv2.circle(frame,(42,35),3,.9,-1)
        free,_,_=a._free_space(frame);path=a._geodesic(free)
        self.assertGreater(path[-1,1],35.)

    def test_straight_controller_and_reset(self):
        a=agent();f=np.full((84,84),.15,np.float32);f[:74,28:57]=.4;f[74:]=0.;obs=np.stack([f]*4)
        action=a.act(obs);self.assertLess(abs(action[0]),.05);self.assertGreater(action[1],.1)
        a.reset();np.testing.assert_array_equal(action,a.act(obs))

if __name__=='__main__':unittest.main()
