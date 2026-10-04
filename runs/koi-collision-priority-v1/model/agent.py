from haic_agent.steering_release_runtime import SteeringReleaseAgent
class Agent:
    def __init__(self): self.driver = SteeringReleaseAgent(stabilize_ambiguous_flank=True)
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
