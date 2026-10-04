import unittest
import numpy as np
from agents.apex_2026.candidate.agent import Agent as P1

class RecoveryTests(unittest.TestCase):
 def make(self):
  from agents.apex_2026.v2.recovery_agent import Agent
  return Agent()
 def test_nominal_identity(self):
  a=self.make();b=P1();obs=np.full((4,84,84),.35,np.float32);obs[:,74:]=0
  for _ in range(3):np.testing.assert_array_equal(a.act(obs),b.act(obs))
 def test_memory_egotransform(self):
  a=self.make();a.memory=np.array([[0.,10.],[0.,20.]])
  a._move_memory(10.,0.);np.testing.assert_allclose(a.memory,[[0,9.2],[0,19.2]])
 def test_component_behind_left_remains_candidate(self):
  a=self.make();frame=np.full((84,84),.65,np.float32);frame[63:70,:32]=.35
  a.memory=np.array([[-10.,-2.],[-20.,3.]])
  target=a._entry(frame);self.assertIsNotNone(target);self.assertLess(target[0],0)
 def test_hud_and_own_car_are_not_road_entry(self):
  a=self.make();frame=np.full((84,84),.65,np.float32);frame[71:]=.35;frame[58:68,39:46]=.35
  self.assertIsNone(a._entry(frame))
 def test_recovery_brakes_before_gas(self):
  a=self.make();a.memory=np.array([[-10.,0.],[-20.,5.]])
  frame=np.full((84,84),.65,np.float32);frame[60:70,:32]=.35
  action=a._recover(frame,40.,False)
  self.assertEqual(action[1],0.);self.assertGreater(action[2],0.)
 def test_low_speed_gas_and_escape_are_bounded(self):
  a=self.make();a.memory=np.array([[-10.,0.],[-20.,5.]])
  frame=np.full((84,84),.35,np.float32)
  for _ in range(90):
   action=a._recover(frame,0.,True);self.assertLessEqual(action[1],.200001);self.assertLessEqual(abs(action[0]),.650001)
  self.assertGreater(action[1],0.);self.assertNotEqual(a.recovery_mode,'exhausted')
 def test_near_target_brake_is_gradual(self):
  a=self.make();frame=np.full((84,84),.35,np.float32)
  action=a._recover(frame,8.5,True)
  self.assertGreater(action[2],0.)
  self.assertLess(action[2],.05)
if __name__=='__main__':unittest.main()
