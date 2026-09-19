import unittest

import torch

from agent import Baseline1Actor
from export_policy import ACTOR_STATE_KEYS, extract_actor_state


class TestSubmissionPolicy(unittest.TestCase):
    def test_predict_action_has_submission_bounds(self):
        actor = Baseline1Actor()
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias.copy_(torch.tensor([2.0, -1.0, 3.0]))

        actions = actor.predict_action(torch.zeros((2, 4, 84, 84)))

        self.assertTrue(torch.equal(actions, torch.tensor([[1.0, 0.0, 1.0]]).repeat(2, 1)))

    def test_extract_actor_state_ignores_critic_weights(self):
        actor = Baseline1Actor()
        source_state = dict(actor.state_dict())
        source_state["value_net.weight"] = torch.zeros((1, 512))
        source_state["value_net.bias"] = torch.zeros(1)

        exported = extract_actor_state(source_state)

        self.assertEqual(tuple(exported), ACTOR_STATE_KEYS)
        self.assertEqual(set(exported), set(actor.state_dict()))


if __name__ == "__main__":
    unittest.main()
