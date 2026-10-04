import unittest
import numpy as np
from agents.apex_2026.v2.recovery_agent import Agent as Recovery

class ObstacleShieldTests(unittest.TestCase):
 def make(self):
  from agents.apex_2026.v2.obstacle_shield_agent import Agent
  return Agent()
 def test_empty_scene_nominal_identity(self):
  a=self.make();b=Recovery();obs=np.full((4,84,84),.35,np.float32);obs[:,74:]=0
  for _ in range(3):np.testing.assert_array_equal(a.act(obs),b.act(obs))
 def test_road_edges_are_not_obstacles(self):
  a=self.make();f=np.full((84,84),.65,np.float32);f[:,30:54]=.35
  self.assertFalse(a._obstacles(f).any())
 def test_obstacle_mask_has_no_planner_padding(self):
  a=self.make();f=np.full((84,84),.35,np.float32);f[30:34,40:44]=.7
  self.assertEqual(int(a._obstacles(f).sum()),16)
 def test_full_footprint_not_center_only(self):
  a=self.make();a.shadow.reset(0,0,0);o=np.zeros((84,84),np.uint8);o[61,44]=1
  self.assertTrue(a._hits(o));o[:]=0;o[61,50]=1;self.assertFalse(a._hits(o))
 def test_observed_obstacle_triggers_braking(self):
  a=self.make();o=np.zeros((84,84),np.uint8);o[43:47,40:45]=1
  state=dict(speed=40.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=.5)
  action,diag=a._shield(o,np.array([0.,.5,0.]),state,np.array([[0.,5.],[0.,30.]]))
  self.assertTrue(diag['active']);self.assertEqual(action[0],0.);self.assertEqual(action[1],0.);self.assertGreater(action[2],0.)
 def test_outside_trajectory_obstacle_does_not_intervene(self):
  a=self.make();o=np.zeros((84,84),np.uint8);o[43:47,10:15]=1
  state=dict(speed=40.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=.5);nom=np.array([0.,.5,0.])
  action,d=a._shield(o,nom,state,np.array([[0.,5.],[0.,30.]]));np.testing.assert_array_equal(action,nom);self.assertFalse(d['active'])
if __name__=='__main__':unittest.main()
