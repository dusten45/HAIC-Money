"""Single-axis candidate road eligibility; obstacle-only activation unchanged.

Road pixels constrain escape candidates, never independently trigger the shield.
The current projected car footprint is occluded initial space. Outside-view or
outside-road candidate footprints are rejected; no formal safety claim follows
from a pixel mask, estimated state and finite physics horizon.
"""
import cv2
import numpy as np
from agents.apex_2026.v2.obstacle_steering_agent import Agent as JointShield

class Agent(JointShield):
    def reset(self,observation=None):
        super().reset(observation)
        self.current_frame=None
        self._road_rejected=0

    def act(self,observation):
        obs=np.asarray(observation)
        self.current_frame=obs[-1].copy() if obs.shape==(4,84,84) else None
        return super().act(observation)

    def _footprint(self):
        mask=np.zeros((84,84),np.uint8);outside=False
        for body in [self.shadow.hull]+self.shadow.wheels:
            for fixture in body.fixtures:
                vertices=np.array([body.GetWorldPoint(v) for v in fixture.shape.vertices])
                pixels=np.rint(np.column_stack((42+vertices[:,0]*self.PIXELS_X,
                                               63-vertices[:,1]*self.PIXELS_Y))).astype(np.int32)
                outside|=bool(np.any((pixels[:,0]<0)|(pixels[:,0]>=84)|(pixels[:,1]<0)|(pixels[:,1]>=71)))
                cv2.fillConvexPoly(mask,pixels,1)
        return mask,outside

    def _preview(self,obstacles,action,state,path):
        self._initialize(state)
        if self.current_frame is None:
            road=np.ones((84,84),np.uint8)
        else:
            frame=self.current_frame
            road=((frame>=self.config['road_low'])&(frame<=self.config['road_high'])).astype(np.uint8)
            road=cv2.morphologyEx(road,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        road[71:]=0
        initial,_=self._footprint();road[initial>0]=1
        hit=None;road_tick=None;outside_max=0.
        for tick in range(1,17):
            self.shadow.step(action,1)
            footprint,outside=self._footprint()
            fraction=float(np.sum((footprint>0)&(road==0))/max(int(footprint.sum()),1))
            outside_max=max(outside_max,fraction,1. if outside else 0.)
            if road_tick is None and (outside or fraction>0.):road_tick=tick
            if hit is None and np.any(footprint&obstacles):hit=tick
        position=np.array(self.shadow.hull.position)
        along,lateral,heading=self._path_state(path,position,self.shadow.hull.angle)
        origin=self._path_state(path,np.zeros(2),0.)[0]
        if road_tick is not None:self._road_rejected+=1
        # Parent's candidate filter accepts only hit=None. Preserve the separate
        # causes for diagnosis; a road rejection is not an activation trigger.
        return dict(hit=hit if hit is not None else road_tick,obstacle_hit=hit,
            road_eligible=road_tick is None,road_outside_max=outside_max,
            road_first_invalid_tick=road_tick,advance=along-origin,lateral=lateral,
            heading=heading,speed=float(self.shadow.hull.linearVelocity.length),
            position=position.tolist())

    def _shield(self,obstacles,nominal,state,path):
        self._road_rejected=0
        chosen,diag=super()._shield(obstacles,nominal,state,path)
        if diag['active']:
            prediction=self._preview(obstacles,chosen,state,path)
            if not prediction['road_eligible']:diag['backup_found']=False
            diag.update(road_backup_eligible=prediction['road_eligible'],
                road_outside_max=prediction['road_outside_max'],
                road_rejected_previews=self._road_rejected,
                road_fallback_unverified=not prediction['road_eligible'])
        return chosen,diag
