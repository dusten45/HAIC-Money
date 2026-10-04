import unittest
from pathlib import Path
import numpy as np
from agents.apex_2026.research.speed_20261005.camera_mpc_reference import (
    CameraGeometry,body_points,project_progress,road_slack,obstacle_slack,terminal_speed_limit)
from agents.apex_2026.fast_rear_clear_agent import Agent


class CameraMPCReferenceTests(unittest.TestCase):
    def test_lateral_offset_does_not_force_center_rejoin(self):
        path=np.column_stack((np.full(15,5.),np.arange(3.,33.,2.)))
        positions=np.array([[0.,0.],[0.,1.],[0.,2.]])
        result=project_progress(path,positions,np.tile([0.,10.],(3,1)),.1)
        np.testing.assert_allclose(result['projections'][:,0],[0.,1.,2.])
        np.testing.assert_allclose(result['projections'][:,1],-5.)
        self.assertAlmostEqual(result['reference_origin_arc_m'],-3.)

    def test_hairpin_nearest_return_branch_cannot_jump_progress(self):
        path=np.array([[0.,3.],[0.,10.],[5.,10.],[5.,0.]])
        result=project_progress(path,np.array([[0.,0.],[4.9,1.]]),np.tile([0.,10.],(2,1)),.1)
        self.assertAlmostEqual(result['projections'][1,0],1.)
        self.assertGreater(result['supported_remaining_m'],20.)

    def test_ccw_body_rotation_uses_actual_predicted_yaw(self):
        body=body_points(np.zeros((1,2)),np.array([-np.pi/2]))[0]
        np.testing.assert_allclose([body[:,0].min(),body[:,0].max()],[-2.4,2.6])
        np.testing.assert_allclose([body[:,1].min(),body[:,1].max()],[-1.6,1.6])

    def test_full_body_grass_and_unknown_image_are_rejected(self):
        field=np.full((73,84),10.);grass=np.zeros((73,84));grass[62:65,41:44]=1.
        self.assertLess(road_slack(field,np.zeros((1,2)),np.zeros(1),grass_mask=grass)[0],0.)
        self.assertLess(road_slack(field,np.array([[0.,40.]]),np.zeros(1))[0],0.)

    def test_center_disk_threshold_does_not_hide_front_body_threat(self):
        # Center separation passes3.7, but front rectangle remains too close.
        slack=obstacle_slack(np.zeros((1,2)),np.zeros(1),[[0.,4.]])
        self.assertLess(slack[0],0.)
        self.assertGreater(obstacle_slack(np.zeros((1,2)),np.zeros(1),[[5.,0.]])[0],0.)
        self.assertLess(obstacle_slack(np.zeros((1,2)),np.zeros(1),[[5.,0.]],uncertainty_m=2.)[0],0.)

    def test_terminal_budget_reserves_front_delay_and_uncertainty(self):
        result=terminal_speed_limit({'supported_remaining_m':10.},0.,[0.,1.],10.,100.,.1,1.)
        self.assertAlmostEqual(result,np.sqrt(2.*100.*5.4))
        self.assertEqual(terminal_speed_limit({'supported_remaining_m':2.},0.,[0.,1.],10.,100.,.1),0.)

    def test_saved149_emitted_arc_has_unsupported_body(self):
        frame=np.load(Path(__file__).parent/'fixtures/control-guard-normal-track2.npz')['frame149']
        agent=Agent();field=agent._distance_field(frame)
        distance=np.linspace(0.,8.,33);k=np.tan(-.1385689)/agent.WHEELBASE
        angle=k*distance
        positions=np.column_stack(((1.-np.cos(angle))/k,np.sin(angle)/k))
        self.assertLess(np.min(road_slack(field,positions,-angle)),0.)

    def test_frozen_geometry_wrapper_aligns_all_tick_costs(self):
        path=np.column_stack((np.zeros(12),np.arange(3.,27.,2.)))
        geometry=CameraGeometry(path,np.full((73,84),10.))
        path[:]=1000.  # Caller mutations cannot move a frozen plan reference.
        result=geometry.evaluate([[0.,0.],[0.,1.]],np.zeros(2),[[0.,10.],[0.,10.]],.1)
        np.testing.assert_allclose(result['progress'],[0.,1.])
        self.assertFalse(result['violation'].any())
        with self.assertRaises(ValueError):geometry.evaluate([[np.nan,0.]],np.zeros(1),[[0.,0.]])

    def test_supported_prefix_end_cannot_invent_asphalt_route(self):
        path=np.array([[0.,3.],[0.,5.]])
        geometry=CameraGeometry(path,np.full((73,84),10.))
        result=geometry.evaluate([[0.,0.],[0.,4.]],np.zeros(2),[[0.,40.],[0.,40.]],.1)
        self.assertGreater(result['road_slack_m'][-1],0.)
        self.assertLess(result['reference_front_slack_m'][-1],0.)
        self.assertTrue(result['violation'][-1])

    def test_post_velocity_acceleration_exposes_prefix_end(self):
        # Semi-implicit integration updates position with POST-step speed.
        # Prior-speed reachability would pin projection at0 and hide the end.
        geometry=CameraGeometry(np.array([[0.,3.],[0.,4.]]),np.full((73,84),10.))
        result=geometry.evaluate([[0.,0.],[0.,2.]],np.zeros(2),[[0.,0.],[0.,20.]],.1)
        self.assertAlmostEqual(result['progress_m'][-1],2.)
        self.assertLess(result['reference_front_slack_m'][-1],0.)
        self.assertTrue(result['violation'][-1])


if __name__=='__main__':unittest.main()
