"""Bounded shoulder candidates preserve road goals and use per-wheel friction."""
import importlib.util
from pathlib import Path
import unittest
import cv2
import numpy as np


def agent():
    p=Path(__file__).with_name('terrain_agent.py')
    assert p.exists(), 'Terrain agent is not implemented'
    s=importlib.util.spec_from_file_location('terrain',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


class TerrainTests(unittest.TestCase):
    def test_shoulder_is_local_and_obstacle_stays_blocked(self):
        a=agent();f=np.full((84,84),.65,np.float32);f[:74,33:52]=.4
        cv2.circle(f,(42,30),3,.9,-1)
        road,free,obstacles=a._terrain_masks(f)
        self.assertEqual(free[30,42],0)
        self.assertEqual(free[20,10],0)
        outside=cv2.distanceTransform(1-road,cv2.DIST_L2,5)
        self.assertLessEqual(outside[free>0].max()/a.PIXELS_X,2.)

    def test_wheel_contacts_use_individual_road_grip(self):
        a=agent();road=np.zeros((84,84),np.uint8);road[:74,:42]=1
        grip=a._wheel_grip(np.array([[42.,60.],[42.,50.]]),road)
        np.testing.assert_allclose(np.sort(grip[0]),[.6,.6,1.,1.])

    def test_goal_is_fixed_to_normal_road_frontier(self):
        a=agent();f=np.full((84,84),.65,np.float32)
        cv2.polylines(f,[np.array([[42,73],[42,35],[79,35]],np.int32)],False,.4,15)
        base,_,_=a._free_space(f);normal=a._geodesic(base)
        path,_,_,_,meta=a._plan_terrain(f)
        np.testing.assert_allclose(path[-1],normal[-1])
        np.testing.assert_allclose(meta['road_goal'],normal[-1])

    def test_obstacle_occlusion_does_not_move_the_progress_goal_backwards(self):
        a=agent();f=np.full((84,84),.65,np.float32);f[:74,34:51]=.4
        cv2.circle(f,(42,35),4,.9,-1)
        _,_,_,_,meta=a._plan_terrain(f)
        self.assertLess(meta['road_goal'][1],6.)

    def test_grass_pedals_reduce_rear_drive_and_braking_budget(self):
        a=agent();grip=np.array([.6,.6,.6,.6])
        gas,brake=a._terrain_pedals(40.,70.,0.,1.,0.,grip)
        self.assertLessEqual(gas,.18+1e-9);self.assertEqual(brake,0.)
        gas,brake=a._terrain_pedals(70.,30.,0.,1.,0.,grip)
        self.assertEqual(gas,0.)
        self.assertLessEqual(brake,65*.6/303.8957799087198+1e-9)

    def test_straight_action_is_finite(self):
        a=agent();f=np.full((84,84),.65,np.float32);f[:74,28:57]=.4;f[74:]=0.
        action=a.act(np.stack([f]*4))
        self.assertTrue(np.isfinite(action).all());self.assertLess(abs(action[0]),.1)

if __name__=='__main__':unittest.main()
