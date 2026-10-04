"""Interaction: retain r5 line/control and fix only duplicated margin."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np


def agent():
    p=Path(__file__).with_name('geodesic_frenet_margin_agent.py')
    assert p.exists(), 'Single-margin Frenet interaction is not implemented'
    s=importlib.util.spec_from_file_location('frenet_margin',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


class MarginInteractionTests(unittest.TestCase):
    def test_narrow_obstacle_bypass_remains_physically_clear_and_connected(self):
        a=agent();frame=np.full((84,84),.15,np.float32);frame[:74,33:52]=.4
        cv2.circle(frame,(42,35),3,.9,-1)
        free,_,_=a._free_space(frame);path=a._racing_line(a._geodesic(free),free)
        self.assertLess(path[-1,1],6.)
        dense=np.concatenate([np.linspace(p,q,12) for p,q in zip(path[:-1],path[1:])])
        x,y=np.rint(dense).astype(int).T
        distance=cv2.distanceTransform(free,cv2.DIST_L2,5)
        # Pixel-center distance tests have <=sqrt(.5)px rounding error;
        # this is not an exact continuous3px footprint certificate.
        self.assertTrue(np.all(free[y,x]))
        self.assertGreaterEqual(distance[y,x].min(),3.-np.sqrt(.5))
        vx,vy=np.rint(path).astype(int).T
        self.assertGreaterEqual(distance[vy,vx].min(),3.)

    def test_straight_controller_and_reset(self):
        a=agent();f=np.full((84,84),.15,np.float32);f[:74,28:57]=.4;f[74:]=0.;obs=np.stack([f]*4)
        action=a.act(obs);self.assertLess(abs(action[0]),.05);self.assertGreater(action[1],.1)
        a.reset();np.testing.assert_array_equal(action,a.act(obs))

if __name__=='__main__':unittest.main()
