"""First-action rollout must permit the downstream road to change curvature."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np


def agent():
    p=Path(__file__).with_name('racing_feedback_agent.py')
    assert p.exists(), 'Feedback rollout agent is not implemented'
    s=importlib.util.spec_from_file_location('racing_feedback',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m.Agent()


class FeedbackTests(unittest.TestCase):
    def test_first_action_preserved_then_feedback_can_countersteer(self):
        a=agent()
        path=np.array([[0.,0.],[0.,3.],[2.,6.],[3.,10.],[1.,15.],[-2.,20.],[-4.,25.]])
        _,constant,_=a._rollout_connector(35.,.1,1.,.1)
        poses,wheels,_=a._rollout_connector(35.,.1,1.,.1,path=path)
        np.testing.assert_array_equal(wheels[:4],constant[:4])
        self.assertLess(wheels[-1],0.)
        self.assertGreater(np.max(poses[:,0]),1.)

    def test_straight_future_feedback_does_not_keep_initial_turn_forever(self):
        a=agent();path=np.column_stack([np.zeros(40),np.arange(40.)])
        const,_,_=a._rollout_connector(40.,0.,0.,.15)
        feedback,_,_=a._rollout_connector(40.,0.,0.,.15,path=path)
        self.assertLess(abs(feedback[-1,0]),abs(const[-1,0]))

    def test_vector_commands_keep_independent_feedback_states(self):
        a=agent();path=np.column_stack([np.zeros(40),np.arange(40.)])
        batch,_,_=a._rollout_connector(40.,0.,0.,np.array([-.1,.1]),path=path)
        single,_,_=a._rollout_connector(40.,0.,0.,.1,path=path)
        np.testing.assert_allclose(batch[:,1],single,atol=1e-9)

    def test_straight_observation_drives_and_reset_repeats(self):
        a=agent();f=np.full((84,84),.15,np.float32);f[:74,28:57]=.4;f[74:]=0.;obs=np.stack([f]*4)
        action=a.act(obs)
        self.assertLess(abs(action[0]),.05)
        self.assertGreater(action[1],.1)
        a.reset();np.testing.assert_array_equal(a.act(obs),action)

if __name__=='__main__':unittest.main()
