"""Observed-obstacle-only short-horizon braking over frozen P1+recovery.

Independent public physics predicts hull AND wheel fixtures. Road edges never
trigger this filter. Pixel state/grip/continuation assumptions prevent a formal
safety claim; this is a causal experiment, not a certified collision barrier.
"""
import math
import cv2
import numpy as np
from agents.apex_2026.v2.recovery_agent import Agent as Recovery
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega

class Agent(Recovery):
    def __init__(self,config=None,*,project_root=None):
        super().__init__(config,project_root=project_root)
        self.shadow=ShadowCar()

    def reset(self,observation=None):
        super().reset(observation)
        self.model_slip=0.
        self.observed_slip=0.
        self.observed_flow=False

    def _motion(self,previous,current):
        result=super()._motion(previous,current)
        self.observed_flow=bool(result[2])
        if self.observed_flow:self.observed_slip=float(result[1])
        return result

    def _obstacles(self,frame):
        road=((frame>=self.config['road_low'])&(frame<=self.config['road_high'])).astype(np.uint8)
        road=cv2.morphologyEx(road,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        bright=(frame>=.54).astype(np.uint8);bright[:5]=0;bright[61:]=0
        n,labels,stats,_=cv2.connectedComponentsWithStats(bright,8)
        mask=np.zeros_like(bright)
        for i in range(1,n):
            x,y,w,h,area=stats[i]
            if 4<=area<=90 and 2<=w<=11 and 2<=h<=12:
                if road[max(0,y-2):min(74,y+h+2),max(0,x-2):min(84,x+w+2)].mean()>.2:
                    mask[labels==i]=1
        return mask

    def _initialize(self,state):
        s=state['slip'];v=state['speed']
        self.shadow.reset(v*math.cos(s),-state['yaw'],-state['wheel'],
            lateral_speed=v*math.sin(s),throttle=state['throttle'],omegas=state['omega'])

    def _hits(self,obstacles):
        footprint=np.zeros((84,84),np.uint8)
        for body in [self.shadow.hull]+self.shadow.wheels:
            for fixture in body.fixtures:
                vertices=np.array([body.GetWorldPoint(v) for v in fixture.shape.vertices])
                pixels=np.rint(np.column_stack((42+vertices[:,0]*self.PIXELS_X,
                                                63-vertices[:,1]*self.PIXELS_Y))).astype(np.int32)
                cv2.fillConvexPoly(footprint,pixels,1)
        return bool(np.any(footprint & obstacles))

    def _continuation(self,path,steer):
        if len(path)<3:return steer
        theta=self.shadow.hull.angle;c,s=math.cos(theta),math.sin(theta)
        local=(path-np.asarray(self.shadow.hull.position))@np.array([[c,-s],[s,c]])
        visible=local[:,1]>1.
        if not np.any(visible):return steer
        points=local[visible];look=(26.+.16*self.shadow.hull.linearVelocity.length)/self.PIXELS_Y
        point=points[np.argmin(abs(points[:,1]-look))]
        curvature=2*point[0]/max(float(point@point),1.)
        desired=float(np.clip(4.5*curvature,-.85,.85))
        return .35*steer+.65*desired

    def _rollout(self,obstacles,action,state,path):
        self._initialize(state)
        steer=float(action[0])
        for tick in range(1,17):
            if tick>4 and (tick-1)%4==0:
                steer=self._continuation(path,steer)
            self.shadow.step((steer,float(action[1]),float(action[2])),1)
            if self._hits(obstacles):return tick
        return None

    def _shield(self,obstacles,nominal,state,path):
        if not obstacles.any():return nominal,dict(active=False,nominal_hit_tick=None,trials=0)
        hit=self._rollout(obstacles,nominal,state,path)
        if hit is None:return nominal,dict(active=False,nominal_hit_tick=None,trials=1)
        trials=1;best=nominal;best_hit=hit
        for brake in (0.,.15,.3,.4,.7):
            if brake<float(nominal[2]):continue
            candidate=np.array([nominal[0],0.,brake],np.float32)
            collision=self._rollout(obstacles,candidate,state,path);trials+=1
            if collision is None:
                return candidate,dict(active=True,nominal_hit_tick=hit,backup_hit_tick=None,trials=trials,backup_found=True)
            if collision>=best_hit:best,best_hit=candidate,collision
        return best,dict(active=True,nominal_hit_tick=hit,backup_hit_tick=best_hit,trials=trials,backup_found=False)

    def act(self,observation):
        obs=np.asarray(observation);initial=self.internal_throttle
        self.observed_flow=False
        nominal=super().act(observation)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all():return nominal
        frame=obs[-1];yaw,wheel=self._hud_dynamics(frame)
        speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0,100))
        slip=self.observed_slip if self.observed_flow else self.model_slip
        omega=decode_wheel_omega(frame)
        state=dict(speed=speed,yaw=yaw,wheel=wheel,slip=slip,omega=omega,throttle=initial)
        path=np.asarray(self.diagnostics.get('path',[]))
        metric=np.column_stack(((path[:,0]-42)/self.PIXELS_X,(63-path[:,1])/self.PIXELS_Y)) if len(path) else np.empty((0,2))
        obstacles=self._obstacles(frame)
        if self.diagnostics.get('recovery_active') or not self.diagnostics.get('valid'):
            chosen,diag=nominal,dict(active=False,nominal_hit_tick=None,trials=0,bypass='recovery')
        else:
            chosen,diag=self._shield(obstacles,nominal,state,metric)
        self._initialize(state);self.shadow.step(chosen,4)
        angle=self.shadow.hull.angle;v=self.shadow.hull.linearVelocity
        self.model_slip=math.atan2(math.cos(angle)*v.x+math.sin(angle)*v.y,
                                 max(-math.sin(angle)*v.x+math.cos(angle)*v.y,.1))
        if diag['active']:
            self.internal_throttle=initial
            for _ in range(4):self.internal_throttle+=min(float(chosen[1])-self.internal_throttle,.1)
            self.last_command=chosen.copy()
        self.diagnostics.update(obstacle_shield=diag,obstacle_shield_nominal=nominal.tolist(),
            obstacle_shield_action=chosen.tolist(),obstacle_shield_pixels=int(obstacles.sum()),
            obstacle_shield_slip_source='flow' if self.observed_flow else 'propagated_unknown',
            obstacle_shield_speed=speed,obstacle_shield_wheel=wheel,
            obstacle_shield_rpm_ambiguous=bool(np.max(omega)>=350))
        return chosen
