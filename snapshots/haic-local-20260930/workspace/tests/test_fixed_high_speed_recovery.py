import unittest
from unittest.mock import patch
import numpy as np
from haic_agent.fixed_high_speed_recovery import FixedHighSpeedRecovery, extrapolate, wide_center


class RecoveryTests(unittest.TestCase):
    def test_missing_row_keeps_visible_left_bend(self):
        centers={54:29.5,50:21.5,46:15.5}
        far=extrapolate(centers,42)
        self.assertLess(far,15.5)
        self.assertLess(.022*(far-42)+.018*(far-centers[54]),0)

    def test_wide_search_reacquires_edge_run(self):
        frame=np.zeros((84,84),dtype=np.float32)
        frame[42,0:8]=.4
        self.assertEqual(wide_center(frame,42,4),3.5)

    @staticmethod
    def fake(driver, obs):
        driver.last=dict(road_centers={},pixel_speed=30.)
        return np.array([.7,.6,0],dtype=np.float32)

    def test_memory_expires_and_pedals_do_not_change(self):
        driver=FixedHighSpeedRecovery('road_memory'); driver.steps=20
        driver.remembered_road=-.4
        with patch('haic_agent.fixed_high_speed_runtime.FixedHighSpeedAgent.act',self.fake):
            action=driver.act(None)
            self.assertAlmostEqual(float(action[0]),-.4,places=6)
            self.assertAlmostEqual(float(action[1]),.6,places=6)
            driver.missing_steps=30
            action=driver.act(None)
            self.assertAlmostEqual(float(action[0]),.7,places=6)

    def test_impact_slew_preserves_acceleration(self):
        driver=FixedHighSpeedRecovery('impact_slew'); driver.steps=20
        driver.speed_history=[60.]; driver.previous_steer=-.5
        with patch('haic_agent.fixed_high_speed_runtime.FixedHighSpeedAgent.act',self.fake):
            action=driver.act(None)
            self.assertAlmostEqual(float(action[0]),-.32,places=6)
            self.assertAlmostEqual(float(action[1]),.6,places=6)
            self.assertTrue(driver.last['impact_proxy_trigger'])


if __name__=='__main__':
    unittest.main()
