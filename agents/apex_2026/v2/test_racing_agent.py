"""Road-aligned line optimization and transient connector specifications."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np


def agent():
    p=Path(__file__).with_name('racing_agent.py')
    assert p.exists(), 'Racing-line controller is not implemented'
    spec=importlib.util.spec_from_file_location('racing',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.Agent()


class RacingTests(unittest.TestCase):
    def test_road_width_reduces_corner_curvature(self):
        a=agent()
        free=np.zeros((84,84),np.uint8)
        reference=np.array([[42.,58.],[42,50],[42,42],[40,34],[32,30],[24,30],[16,30],[5,30]])
        cv2.polylines(free,[reference.astype(np.int32)],False,1,19)
        refined=a._racing_line(reference,free)
        def bending(path):
            metric=(path-[42,63])/[1.3608,-1.701]
            _,k=a._geometry(metric)
            return np.max(np.abs(k[3:-3]))
        # The reference must be compared at identical uniform arclength density.
        reference=a._resample(reference)
        self.assertLess(bending(refined),.95*bending(reference))
        d=cv2.distanceTransform(free,cv2.DIST_L2,5)
        x,y=np.rint(refined).astype(int).T
        self.assertGreaterEqual(d[y,x].min(),3.)

    def test_narrow_obstacle_passage_keeps_connected_route(self):
        a=agent();frame=np.full((84,84),.15,np.float32);frame[:74,33:52]=.4
        cv2.circle(frame,(42,35),3,.9,-1)
        free,_,_=a._free_space(frame)
        path=a._racing_line(a._geodesic(free),free)
        self.assertLess(path[-1,1],6.)
        dense=np.concatenate([np.linspace(p,q,12) for p,q in zip(path[:-1],path[1:])])
        x,y=np.rint(dense).astype(int).T
        self.assertTrue(np.all(free[y,x]))

    def test_connector_starts_with_measured_turn_before_reversing(self):
        a=agent()
        poses,wheels,yaws=a._rollout_connector(50.,.2,3.,-.2)
        self.assertAlmostEqual(wheels[0],.14,places=6)
        self.assertGreater(yaws[0],0.)
        self.assertGreater(poses[0,0],0.)
        self.assertLess(wheels[-1],0.)
        # The initial state matters even for the same future command.
        other,_,_=a._rollout_connector(50.,-.2,-3.,-.2)
        self.assertGreater(poses[5,0],other[5,0])

    def test_tracking_uncertainty_slows_narrow_passage(self):
        a=agent()
        self.assertLess(a._clearance_speed(3.),a._clearance_speed(7.))
        self.assertGreaterEqual(a._clearance_speed(3.),a.config['min_speed'])

    def test_straight_observation_drives_and_reset_repeats(self):
        a=agent();f=np.full((84,84),.15,np.float32);f[:74,28:57]=.4;obs=np.stack([f]*4)
        action=a.act(obs)
        self.assertLess(abs(action[0]),.05);self.assertGreater(action[1],.1)
        a.reset();np.testing.assert_array_equal(a.act(obs),action)

if __name__=='__main__':unittest.main()
