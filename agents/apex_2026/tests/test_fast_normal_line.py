import unittest
from pathlib import Path
import numpy as np
from agents.apex_2026.fast_rear_clear_agent import Agent as Parent
from agents.apex_2026.fast_normal_line_agent import Agent
from agents.apex_2026.research.speed_20261005.rear_normal_line import footprint_depth


class NormalLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = np.load(Path(__file__).parent/'fixtures/normal-line-track2.npz')

    def test_saved_bend_reduces_cap_without_losing_body_clearance(self):
        frame=self.frames['frame150']; parent=Parent(); agent=Agent()
        base=parent._ridge(frame); result=agent._ridge(frame)
        self.assertGreater(agent._ridge_target(result,83.), parent._ridge_target(base,83.)+8.)
        self.assertGreaterEqual(footprint_depth(agent,agent._distance_field(frame),result),1.9)
        np.testing.assert_array_equal(result[0], [0.,0.])

    def test_unknown_image_end_retains_exact_parent_path(self):
        frame=self.frames['frame83']
        np.testing.assert_array_equal(Agent()._ridge(frame),Parent()._ridge(frame))

    def test_straight_line_is_not_bent(self):
        path=np.column_stack((np.zeros(12),np.arange(3.,27.,2.)))
        result=Agent._normal_line(path)
        np.testing.assert_allclose(result[:,0],0.,atol=1e-12)
        self.assertTrue(np.all(np.diff(result[:,1])>0.))

    def test_reset_and_nonfinite_observation_match_parent(self):
        parent=Parent(); agent=Agent(); frame=self.frames['frame150']
        agent.act(frame);parent.act(frame);agent.reset();parent.reset()
        invalid=np.full((4,84,84),np.nan)
        np.testing.assert_array_equal(agent.act(invalid),parent.act(invalid))
        self.assertEqual(agent.lost_frames,parent.lost_frames)
        self.assertEqual(agent.pass_side,parent.pass_side)


if __name__=='__main__': unittest.main()
