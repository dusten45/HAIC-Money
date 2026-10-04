"""Stateful reacquisition and stale-avoidance clearing at the same high target."""
import numpy as np
from haic_agent.fixed_high_speed_runtime import FixedHighSpeedAgent

MODES = ("memory_reacquire", "impact_clear")


class FixedHighSpeedRecoveryV2(FixedHighSpeedAgent):
    def __init__(self, mechanism="memory_reacquire"):
        if mechanism not in MODES:
            raise ValueError("unknown recovery mode")
        self.recovery_mode = mechanism
        super().__init__("damping")

    def reset(self, observation=None):
        super().reset(observation)
        self.remembered_road = 0.0
        self.latched_turn = 0.0
        self.recovery_left = 0
        self.recovery_armed = True
        self.centered_streak = 0
        self.impact_left = 0
        self.speed_history = []

    def act(self, observation):
        action = super().act(observation)
        original = float(action[0])
        centers = self.last["road_centers"]
        speed = self.last["pixel_speed"]
        complete = 42 in centers and 54 in centers
        centered = len(centers)==7 and abs(centers[42]-42)<=8 and abs(centers[54]-42)<=8
        self.centered_streak = self.centered_streak+1 if centered else 0
        if self.centered_streak >= 3:
            self.recovery_left = 0
            self.recovery_armed = True
        trigger = False
        if (self.steps>10 and self.impact_left==0 and self.speed_history and speed<50
                and max(self.speed_history)-speed>10):
            self.impact_left = 12
            trigger = True
        if self.steps>10 and self.recovery_mode=="memory_reacquire":
            if (42 not in centers and self.recovery_armed and abs(self.remembered_road)>.01):
                self.recovery_left = 80
                self.recovery_armed = False
                self.latched_turn = np.sign(self.remembered_road)*np.clip(abs(self.remembered_road),.35,.7)
            if self.recovery_left>0:
                action[0] = self.latched_turn
                self.recovery_left -= 1
        elif self.steps>10 and self.recovery_mode=="impact_clear" and self.impact_left>0 and complete:
            road = .022*(centers[42]-42)+.018*(centers[42]-centers[54])
            action[0] = np.clip(road+self.last["correction"],-.7,.7)
        if complete and self.recovery_left==0:
            self.remembered_road = .022*(centers[42]-42)+.018*(centers[42]-centers[54])
        if self.impact_left:
            self.impact_left -= 1
        self.speed_history = (self.speed_history+[speed])[-4:]
        self.last.update(recovery_mode=self.recovery_mode,damping_steer=original,
                         recovery_changed=abs(float(action[0])-original)>1e-6,
                         recovery_steps_remaining=self.recovery_left,centered_streak=self.centered_streak,
                         impact_proxy_trigger=trigger,impact_steps_remaining=self.impact_left)
        return action
