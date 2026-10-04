"""Behavior specifications for the independent 2-D path representation."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np


def agent():
    path = Path(__file__).with_name('geodesic_agent.py')
    assert path.exists(), '2-D geodesic controller is not implemented'
    spec = importlib.util.spec_from_file_location('geodesic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Agent()


class GeodesicTests(unittest.TestCase):
    def test_hairpin_can_continue_down_image_after_apex(self):
        a = agent()
        mask = np.zeros((84,84), np.uint8)
        cv2.polylines(mask, [np.array([[42,63],[42,28],[62,28],[62,70]])], False, 1, 13)
        path = a._geodesic(mask, (42,60), (62,68))
        self.assertGreater(len(path), 20)
        self.assertGreater(np.max(np.diff(path[:,1])), .1)
        self.assertLess(np.min(path[:,1]), 35)
        self.assertLess(np.linalg.norm(path[-1]-[62,68]), 1.)

    def test_dense_smoothed_path_stays_clear_of_obstacle(self):
        a = agent()
        mask = np.zeros((84,84), np.uint8)
        mask[4:73, 20:65] = 1
        cv2.circle(mask, (42,35), 8, 0, -1)
        path = a._geodesic(mask, (42,60), (42,7))
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dense = np.concatenate([np.linspace(p,q,20) for p,q in zip(path[:-1],path[1:])])
        x,y = np.rint(dense).astype(int).T
        self.assertGreaterEqual(distance[y,x].min(), 2.5)

    def test_curvature_uses_arclength_at_horizontal_tangent(self):
        a = agent()
        t = np.linspace(0,np.pi,60)
        metric = np.column_stack([10*(1-np.cos(t)), 10*np.sin(t)])
        arc, k = a._geometry(metric)
        self.assertGreater(arc[-1], 30)
        self.assertAlmostEqual(float(np.median(np.abs(k[5:-5]))), .1, delta=.015)

    def test_straight_observation_drives_forward_and_reset_is_repeatable(self):
        a = agent()
        frame = np.full((84,84), .15, np.float32)
        frame[:74,28:57] = .4
        obs = np.stack([frame]*4)
        action = a.act(obs)
        self.assertLess(abs(action[0]), .05)
        self.assertGreater(action[1], .1)
        a.reset()
        np.testing.assert_array_equal(a.act(obs), action)

    def test_tracking_arc_respects_near_obstacle_before_distant_target(self):
        a = agent()
        free = np.ones((84,84), np.uint8)
        cv2.circle(free, (42,43), 5, 0, -1)
        metric = np.array([[0.,3.],[-3.,7.],[-7.,12.],[-6.,18.],[2.,25.]])
        curvature, feasible = a._tracking_arc(metric, free, 35., .006)
        self.assertTrue(feasible)
        self.assertLess(curvature, 0.)
        distance = np.linspace(3.,18.,80)
        points = np.column_stack([(1-np.cos(curvature*distance))/curvature,
                                  np.sin(curvature*distance)/curvature])
        pixels = np.rint(points*np.array([1.3608,-1.701])+[42,63]).astype(int)
        self.assertTrue(np.all(free[pixels[:,1],pixels[:,0]]))

    def test_single_footprint_margin_preserves_narrow_obstacle_bypass(self):
        a = agent()
        frame = np.full((84,84), .15, np.float32)
        frame[:74,33:52] = .4
        cv2.circle(frame, (42,35), 3, .9, -1)
        free, obstacles, _ = a._free_space(frame)
        path = a._geodesic(free)
        self.assertLess(path[-1,1], 6., 'Single 3px clearance fits this road; double inflation blocks it')
        raw_distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        x,y = np.rint(path).astype(int).T
        self.assertGreaterEqual(raw_distance[y,x].min(), 3.)

    def test_invalid_observation_brakes(self):
        np.testing.assert_array_equal(agent().act(np.zeros((2,3))),np.array([0,0,.3],np.float32))

if __name__ == '__main__':
    unittest.main()
