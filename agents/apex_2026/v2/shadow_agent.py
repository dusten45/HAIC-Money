"""Pixel-initialized exact car dynamics with short footprint-aware shooting.

Experimental v2 lane: public pixels and past actions only. Internal physics is
separate from the real simulator. Predictions assume road grip and no damage;
unknown tire/visual state prevents a formal safety guarantee.
"""
import math
import numpy as np
import cv2
from agents.apex_2026.candidate.agent import Agent as PathAgent
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega


class Agent(PathAgent):
    def __init__(self,config=None,*,project_root=None):
        config=dict(config or {})
        self.shadow_horizon=int(config.pop('shadow_horizon',20))
        self.shadow_weight=float(config.pop('shadow_weight',1.))
        super().__init__(config,project_root=project_root)
        self.shadow=ShadowCar()

    def reset(self,observation=None):
        super().reset(observation)
        self.shadow_slip=0.
        self.shadow_throttle=0.

    def act(self,observation):
        initial_throttle=self.shadow_throttle
        baseline=super().act(observation)
        if not self.diagnostics.get('valid'):
            self.shadow_throttle=0.
            return baseline
        frame=np.asarray(observation)[-1]
        d=self.diagnostics
        speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
        yaw,wheel=self._hud_dynamics(frame)
        if d.get('flow_valid'):
            slip=float(d['slip'])
        else:
            slip=self.shadow_slip
        slip=float(np.clip(slip,-.7,.7))
        vx,vy=speed*math.sin(slip),speed*math.cos(slip)
        omega=decode_wheel_omega(frame)
        path=np.asarray(d['path'])
        path=np.column_stack(((path[:,0]-42)/self.PIXELS_X,(63-path[:,1])/self.PIXELS_Y))
        free,_,_=self._free_space(frame)
        # The car occludes its own road footprint; observation supplies no road
        # evidence in that small region. Current pose is an initial condition.
        free[59:69,39:46]=1
        clearance=cv2.distanceTransform(free,cv2.DIST_L2,5)
        hull=np.array([[-1.25,-2.4],[1.25,-2.4],[-1.25,2.6],[1.25,2.6],[0,2.6],[0,0]])
        steers=np.unique(np.clip(np.r_[baseline[0]+np.array([-.24,-.12,0,.12,.24]),-.4,0,.4],-.85,.85))
        pedals=[(float(baseline[1]),float(baseline[2])),(0.,0.),(0.,.6)]
        best=None; candidates=0
        target=float(d['target_speed'])
        for steer in steers:
            for gas,brake in pedals:
                action=np.array([steer,gas,brake],np.float32)
                self.shadow.reset(vy,-yaw,-wheel,lateral_speed=vx,throttle=initial_throttle,omegas=omega)
                cost=0.; first_prediction=None
                for step in range(0,self.shadow_horizon,4):
                    pred=self.shadow.step(action,4)
                    if first_prediction is None:
                        angle=pred[2];vel=self.shadow.hull.linearVelocity
                        lat=math.cos(angle)*vel.x+math.sin(angle)*vel.y
                        fwd=-math.sin(angle)*vel.x+math.cos(angle)*vel.y
                        first_prediction=(math.atan2(lat,max(fwd,.1)),self.shadow.wheels[2].gas)
                    x,y,theta,v,rate=pred
                    ct,st=math.cos(theta),math.sin(theta)
                    points=hull@np.array([[ct,st],[-st,ct]])+np.array([x,y])
                    px=np.rint(42+points[:,0]*self.PIXELS_X).astype(int)
                    py=np.rint(63-points[:,1]*self.PIXELS_Y).astype(int)
                    valid=(px>=0)&(px<84)&(py>=0)&(py<74)
                    clear=np.zeros(len(px));clear[valid]=clearance[py[valid],px[valid]]
                    cost+=120.*float(np.maximum(1.-clear,0).sum())
                    desired_x=float(np.interp(y,path[:,1],path[:,0]))
                    cost+=self.shadow_weight*(x-desired_x)**2
                    cost+=.0015*(v-target)**2
                cost+=.15*(float(steer)-float(baseline[0]))**2
                candidates+=1
                if best is None or cost<best[0]:
                    best=(cost,action,first_prediction)
        _,chosen,prediction=best
        self.shadow_slip,self.shadow_throttle=prediction
        self.internal_throttle=self.shadow_throttle
        self.last_steer=float(chosen[0])
        d.update(shadow_candidates=candidates,shadow_cost=float(best[0]),shadow_slip=slip,
                 shadow_omega=omega.tolist(),shadow_baseline=baseline.tolist(),
                 shadow_action=chosen.tolist(),steer=float(chosen[0]))
        return chosen
