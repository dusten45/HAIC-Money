"""Bounded pixel recovery on frozen P1, leaving nominal actions untouched.

Experimental: visible road entry is not a reachability certificate. Forward-only
turning may fail against contact obstacles or exceed the no-progress budget.
"""
import math
import cv2
import numpy as np
from agents.apex_2026.candidate.agent import Agent as P1

class Agent(P1):
    def reset(self, observation=None):
        super().reset(observation)
        self.memory=None
        self.recovering=False
        self.recovery_ticks=0
        self.stalled=0
        self.recovery_mode='nominal'
        self.recovery_target=None
        self.last_command=np.zeros(3)
        self.release_count=0

    def _move_memory(self, speed, yaw):
        if self.memory is not None:
            # P1 HUD yaw is right-positive. Translate then express in new axes.
            self.memory[:,1]-=.08*speed
            theta=.08*yaw;c,s=math.cos(theta),math.sin(theta)
            self.memory=self.memory@np.array([[c,s],[-s,c]])

    def _entry(self, frame):
        free,_,_=self._free_space(frame)
        free[71:]=0
        free[58:69,39:46]=0 # exclude the car, not observed road
        n,labels,stats,_=cv2.connectedComponentsWithStats(free,8)
        candidates=[]
        for i in range(1,n):
            if stats[i,4]<50:continue
            y,x=np.where(labels==i)
            points=np.column_stack(((x-42)/self.PIXELS_X,(63-y)/self.PIXELS_Y))
            distance=np.linalg.norm(points,axis=1)
            # Entry first, with a small preference for the remembered forward
            # branch. Memory cannot create unseen drivable pixels.
            score=distance.copy()
            if self.memory is not None and len(self.memory)>1:
                direction=self.memory[-1]-self.memory[0]
                direction/=max(np.linalg.norm(direction),1e-6)
                score-=.3*np.clip(points@direction,-10,10)
                proximity=np.linalg.norm(points[:,None]-self.memory[None],axis=2).min(axis=1)
                score+=.2*proximity
            j=int(np.argmin(score));entry=points[j]
            # If already next to road, aim farther along its remembered branch.
            if distance[j]<3. and self.memory is not None and len(self.memory)>1:
                goal=entry+5.*direction
                j=int(np.argmin(np.linalg.norm(points-goal,axis=1)))
                entry=points[j]
            candidates.append((float(score.min()),entry))
        return min(candidates,key=lambda z:z[0])[1] if candidates else None

    def _recover(self, frame, speed, valid):
        del valid
        self.recovery_ticks+=1
        target=self._entry(frame)
        self.recovery_target=None if target is None else target.tolist()
        if self.recovery_ticks>80:
            self.recovery_mode='exhausted'
            return np.array([0.,0.,.3],np.float32)
        if speed>8.:
            self.recovery_mode='decelerate'
            return np.array([float(np.clip(self.last_steer,-.4,.4)),0.,.4],np.float32)
        if target is None:
            self.recovery_mode='no_visible_entry'
            return np.array([0.,0.,.3],np.float32)
        bearing=math.atan2(float(target[0]),float(target[1]))
        steer=float(np.clip(1.2*bearing,-.65,.65))
        self.recovery_mode='entry'
        # Static contact can survive full nominal lock. Try one straight and
        # one opposite-lock escape, measured by continuing pixel/HUD stagnation.
        if self.stalled>=16:
            phase=(self.stalled-16)//10
            if phase==0:steer=0.
            elif phase==1:steer=-steer
            self.recovery_mode='contact_escape' if phase<2 else 'entry'
        gas=float(np.clip((8.-speed)*.04,0.,.2))
        brake=.1 if speed>9. else 0.
        return np.array([steer,gas,brake],np.float32)

    def act(self, observation):
        obs=np.asarray(observation)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all():
            return super().act(observation)
        frame=obs[-1]
        speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
        yaw,_=self._hud_dynamics(frame)
        self._move_memory(speed,yaw)
        stagnant=(np.mean(np.abs(obs[-1,:70]-obs[-2,:70]))<.004)
        self.stalled=self.stalled+1 if speed<2. and self.last_command[1]>.1 and stagnant else 0
        old_throttle=self.internal_throttle
        nominal=super().act(observation)
        valid=bool(self.diagnostics.get('valid'))
        if not self.recovering and (not valid or self.stalled>=10):
            self.recovering=True;self.recovery_ticks=0;self.release_count=0
        if not self.recovering:
            path=np.asarray(self.diagnostics.get('path',[]))
            if len(path)>2:
                self.memory=np.column_stack(((path[:,0]-42)/self.PIXELS_X,(63-path[:,1])/self.PIXELS_Y))
            self.last_command=nominal.copy()
            self.diagnostics.update(recovery_mode='nominal',recovery_active=False)
            return nominal
        action=self._recover(frame,speed,valid)
        target=self.recovery_target
        aligned=target is not None and abs(math.atan2(target[0],target[1]))<.5
        self.release_count=self.release_count+1 if valid and speed>3. and aligned else 0
        if self.release_count>=3:
            self.recovering=False
        # Super.act advanced nominal throttle; replace with actual action history.
        self.internal_throttle=old_throttle
        for _ in range(4):self.internal_throttle+=min(float(action[1])-self.internal_throttle,.1)
        self.last_steer=float(action[0]);self.previous_steer=float(action[0])
        self.last_command=action.copy()
        self.diagnostics.update(recovery_mode=self.recovery_mode,recovery_active=True,
            recovery_ticks=self.recovery_ticks,recovery_target=target,
            recovery_stalled=self.stalled,recovery_speed=speed,
            recovery_nominal=nominal.tolist(),recovery_action=action.tolist())
        return action
