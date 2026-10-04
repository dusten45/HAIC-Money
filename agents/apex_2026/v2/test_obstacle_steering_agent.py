import unittest
import numpy as np
from agents.apex_2026.v2.obstacle_shield_agent import Agent as Prior

class ObstacleSteeringTests(unittest.TestCase):
 def make(self):
  from agents.apex_2026.v2.obstacle_steering_agent import Agent
  return Agent()
 def test_safe_nominal_preserved(self):
  a=self.make();o=np.zeros((84,84),np.uint8)
  s=dict(speed=40.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=.5);nom=np.array([0.,.5,0.])
  action,d=a._shield(o,nom,s,np.array([[0.,3.],[0.,30.]]));np.testing.assert_array_equal(action,nom);self.assertFalse(d['active'])
 def test_hazard_selects_moving_steering_branch(self):
  a=self.make();o=np.zeros((84,84),np.uint8);o[43:47,40:45]=1
  s=dict(speed=40.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=.5);nom=np.array([0.,.5,0.])
  action,d=a._shield(o,nom,s,np.array([[0.,3.],[0.,30.]]))
  self.assertTrue(d['active']);self.assertTrue(d['backup_found']);self.assertGreater(d['backup_advance_m'],.3)
  self.assertNotEqual(float(action[0]),0.)
 def test_path_projection_has_signed_progress(self):
  a=self.make();path=np.array([[0.,0.],[0.,10.],[10.,10.]])
  s,e,h=a._path_state(path,np.array([3.,10.]),-1.57079632679)
  self.assertAlmostEqual(s,13.);self.assertAlmostEqual(e,0.);self.assertAlmostEqual(h,0.,places=5)
 def test_runtime_empty_obstacle_identity(self):
  a=self.make();b=Prior();obs=np.full((4,84,84),.35,np.float32);obs[:,74:]=0
  np.testing.assert_array_equal(a.act(obs),b.act(obs))
if __name__=='__main__':unittest.main()
