from haic_agent.contact_continuity_runtime import ContactContinuityAgent
from haic_agent.collision_shield_runtime import CollisionShieldAgent

class Agent:
    def __init__(self):
        self.driver = CollisionShieldAgent(ContactContinuityAgent('crossing_projection'))

    def reset(self, observation):
        self.driver.reset(observation)

    def act(self, observation):
        return self.driver.act(observation)

    def last_step_diagnostics(self):
        return self.driver.last_step_diagnostics()
