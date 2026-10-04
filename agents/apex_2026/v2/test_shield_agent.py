"""Safety-filter behavior: changing visible barriers must change braking."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import cv2


def agent():
    path=Path(__file__).with_name('shield_agent.py')
    assert path.exists(), 'Physical pixel shield is not implemented'
    spec=importlib.util.spec_from_file_location('shield',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Agent()


class ShieldTests(unittest.TestCase):
    def test_backup_can_stop_inside_visible_bending_corridor(self):
        a=agent();self.assertTrue(hasattr(a,'_backup_feasible'),'Selective backup model missing')
        theta=np.linspace(0,1.3,50)
        metric=np.column_stack([20*(1-np.cos(theta)),20*np.sin(theta)])
        path=np.column_stack([42+metric[:,0]*1.3608,63-metric[:,1]*1.701])
        free=np.zeros((84,84),np.uint8)
        cv2.polylines(free,[np.rint(path).astype(np.int32)],False,1,19)
        safe,diag=a._backup_feasible(free,path,np.array([.16,.4,0]),50.,2.5,.16,0.,True)
        self.assertTrue(safe,diag)

    def test_nominal_action_cannot_spend_invisible_stopping_distance(self):
        a=agent();self.assertTrue(hasattr(a,'_backup_feasible'),'Selective backup model missing')
        free=np.ones((84,84),np.uint8)
        path=np.array([[42.,y] for y in range(58,7,-2)])
        low,_=a._backup_feasible(free,path,np.array([0.,.4,0]),40.,0.,0.,0.,True)
        high,_=a._backup_feasible(free,path,np.array([0.,1.,0]),100.,0.,0.,0.,True)
        self.assertTrue(low);self.assertFalse(high)

    def test_selective_shield_overrides_pedals_when_backup_is_blocked(self):
        a=agent();free=np.ones((84,84),np.uint8);free[:35]=0
        path=np.array([[42.,y] for y in range(58,35,-2)])
        result,diag=a._shield(free,np.array([0.,.5,0.]),80.,0.,0.,0.,True,path)
        self.assertTrue(diag['active']);self.assertEqual(result[0],0.)
        self.assertEqual(result[1],0.);self.assertGreater(result[2]*303.8958,65.)

    def test_selective_shield_preserves_nominal_with_visible_backup(self):
        a=agent();free=np.ones((84,84),np.uint8)
        path=np.array([[42.,y] for y in range(58,7,-2)])
        original=np.array([0.,.4,0.],np.float32)
        result,diag=a._shield(free,original,40.,0.,0.,0.,True,path)
        np.testing.assert_array_equal(result,original)
        self.assertFalse(diag['active'])

    def test_clear_wide_road_preserves_feasible_action(self):
        a=agent();free=np.ones((84,84),np.uint8)
        action=np.array([0,.4,0],np.float32)
        out,diag=a._shield(free,action,40.,0.,0.,0.,True)
        np.testing.assert_array_equal(out,action)
        self.assertFalse(diag['active'])

    def test_visible_wall_requires_more_than_old65_deceleration(self):
        a=agent();free=np.ones((84,84),np.uint8);free[:35]=0
        out,diag=a._shield(free,np.array([0,.5,0],np.float32),80.,0.,0.,0.,True)
        self.assertTrue(diag['active']);self.assertEqual(out[1],0)
        self.assertGreater(out[2]*303.8958,65)
        self.assertLessEqual(out[2],.7)

    def test_yaw_footprint_detects_side_boundary(self):
        a=agent();free=np.zeros((84,84),np.uint8);free[:,28:57]=1
        straight,d0=a._shield(free,np.array([0,.4,0],np.float32),60.,0.,0.,0.,True)
        curved,d1=a._shield(free,np.array([.2,.4,0],np.float32),60.,2.,.2,0.,True)
        self.assertLess(d1['free_distance_m'],d0['free_distance_m'])
        self.assertGreater(curved[2],straight[2])

    def test_unknown_flow_is_reported_and_not_assumed_zero_slip(self):
        a=agent();free=np.zeros((84,84),np.uint8);free[:,28:57]=1
        action=np.array([.08,.5,0],np.float32)
        _,known=a._shield(free,action,60.,.8,.08,0.,True)
        result,unknown=a._shield(free,action,60.,.8,.08,0.,False)
        self.assertTrue(unknown['slip_uncertain'])
        self.assertLessEqual(unknown['free_distance_m'],known['free_distance_m'])
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_observation_preserves_braking_contract(self):
        np.testing.assert_array_equal(agent().act(np.zeros((2,3))),np.array([0,0,.3],np.float32))

    def test_realistic_footprint_near_wall_is_not_centerpoint_safe(self):
        a=agent();free=np.ones((84,84),np.uint8);free[:45]=0
        d=a._free_distance(free,40.,0.,0.,0.,0.,True)
        # Center reaches wall at10.6m; front hull must be stopped sooner.
        self.assertLess(d,9.)


if __name__=='__main__':unittest.main()
