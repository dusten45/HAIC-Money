from haic_agent.far_hazard_runtime import FarHazardAgent
class Agent:
    def __init__(self): self.driver = FarHazardAgent('arrival_speed')
    def reset(self, observation): self.driver.reset(observation)
    def act(self, observation): return self.driver.act(observation)
    def last_step_diagnostics(self): return self.driver.last_step_diagnostics()
