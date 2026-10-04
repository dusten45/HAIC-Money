"""Yaw-oriented body geometry must catch collisions missed by center clearance."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np


def agent():
    p=Path(__file__).with_name('geodesic_footprint_agent.py')
    assert p.exists(), 'Yaw-oriented footprint agent is not implemented'
    s=importlib.util.spec_from_file_location('footprint',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


class FootprintTests(unittest.TestCase):
    def test_front_wing_collision_detected_when_center_is_clear(self):
        a=agent();free=np.ones((84,84),np.uint8);free[41:43,41:44]=0
        center=np.array([[[0.,10.]]]);theta=np.array([[0.]])
        pixel=np.rint(center[0,0]*[1.3608,-1.701]+[42,63]).astype(int)
        self.assertGreater(cv2.distanceTransform(free,cv2.DIST_L2,5)[pixel[1],pixel[0]],3.)
        self.assertTrue(a._footprint_collisions(center,theta,free)[0,0])

    def test_narrow_road_allows_aligned_body_but_not_broadside_body(self):
        a=agent();free=np.zeros((84,84),np.uint8);free[:74,40:45]=1
        center=np.array([[[0.,15.]]])
        self.assertFalse(a._footprint_collisions(center,np.array([[0.]]),free)[0,0])
        self.assertTrue(a._footprint_collisions(center,np.array([[np.pi/4]]),free)[0,0])

    def test_straight_controller_and_reset(self):
        a=agent();f=np.full((84,84),.15,np.float32);f[:74,28:57]=.4;f[74:]=0.;obs=np.stack([f]*4)
        action=a.act(obs);self.assertLess(abs(action[0]),.05);self.assertGreater(action[1],.1)
        a.reset();np.testing.assert_array_equal(action,a.act(obs))

if __name__=='__main__':unittest.main()
