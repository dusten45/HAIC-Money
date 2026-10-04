import json,pathlib,unittest
import numpy as np
from agents.apex_2026.v2.obstacle_steering_agent import Agent as Prior
from agents.apex_2026.v2.shadow_physics import decode_wheel_omega

class ObstacleRoadTests(unittest.TestCase):
 def make(self):
  from agents.apex_2026.v2.obstacle_road_agent import Agent
  return Agent()
 def test_obstacle_free_roadedge_never_triggers(self):
  a=self.make();a.current_frame=np.full((84,84),.65,np.float32);a.current_frame[:,38:46]=.35
  s=dict(speed=40.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=.5);nom=np.array([.4,.5,0.])
  chosen,d=a._shield(np.zeros((84,84),np.uint8),nom,s,np.array([[0.,3.],[0.,30.]]))
  np.testing.assert_array_equal(chosen,nom);self.assertFalse(d['active'])
 def test_current_car_occlusion_exempt(self):
  a=self.make();a.current_frame=np.full((84,84),.35,np.float32);a.current_frame[58:69,39:46]=0.
  s=dict(speed=0.,yaw=0.,wheel=0.,slip=0.,omega=None,throttle=0.)
  r=a._preview(np.zeros((84,84),np.uint8),np.zeros(3),s,np.array([[0.,3.],[0.,30.]]))
  self.assertTrue(r['road_eligible'])
 def test_recorded_step68_candidate_rejected(self):
  record=np.load(pathlib.Path(__file__).parent/'results/obstacle_road_fixtures/step0068.npz');frame=record['frame']
  a=self.make();a.current_frame=frame;s=dict(zip(['speed','yaw','wheel','slip','throttle'],map(float,record['state'])));s['omega']=decode_wheel_omega(frame)
  p=np.array(record['path']);path=np.column_stack(((p[:,0]-42)/a.PIXELS_X,(63-p[:,1])/a.PIXELS_Y))
  obstacle=a._obstacles(frame)
  bad=a._preview(obstacle,record['action'],s,path)
  self.assertFalse(bad['road_eligible']);self.assertGreater(bad['road_outside_max'],.1)
  chosen,d=a._shield(obstacle,np.array(record['nominal']),s,path)
  self.assertTrue(d['active']);self.assertTrue(d['road_backup_eligible'])
  self.assertFalse(np.allclose(chosen,record['action']))
 def test_empty_scene_nominal_identity(self):
  a=self.make();b=Prior();obs=np.full((4,84,84),.35,np.float32);obs[:,74:]=0
  np.testing.assert_array_equal(a.act(obs),b.act(obs))
if __name__=='__main__':unittest.main()
