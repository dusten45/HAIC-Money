"""Joint steering/braking escape branches at the first observed-obstacle hazard.

Extends the frozen obstacle-only shield. No road-edge trigger or simulator state.
Progress is measured on the currently visible path, not official race progress.
"""
import math
import numpy as np
from agents.apex_2026.v2.obstacle_shield_agent import Agent as BrakeShield

class Agent(BrakeShield):
    @staticmethod
    def _path_state(path,position,heading):
        if len(path)<2:return float(position[1]),abs(float(position[0])),abs(float(heading))
        delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
        u=np.clip(np.sum((position-path[:-1])*delta,axis=1)/np.maximum(length**2,1e-8),0,1)
        projected=path[:-1]+u[:,None]*delta
        distance=np.linalg.norm(projected-position,axis=1);i=int(np.argmin(distance))
        along=float(np.r_[0.,np.cumsum(length)][i]+u[i]*length[i])
        desired=-math.atan2(float(delta[i,0]),float(delta[i,1]))
        error=math.atan2(math.sin(heading-desired),math.cos(heading-desired))
        return along,float(distance[i]),abs(error)

    def _preview(self,obstacles,action,state,path):
        self._initialize(state);hit=None
        for tick in range(1,17):
            self.shadow.step(action,1)
            if hit is None and self._hits(obstacles):hit=tick
        position=np.array(self.shadow.hull.position)
        along,lateral,heading=self._path_state(path,position,self.shadow.hull.angle)
        origin=self._path_state(path,np.zeros(2),0.)[0]
        return dict(hit=hit,advance=along-origin,lateral=lateral,heading=heading,
                    speed=float(self.shadow.hull.linearVelocity.length),position=position.tolist())

    def _shield(self,obstacles,nominal,state,path):
        if not obstacles.any():return nominal,dict(active=False,nominal_hit_tick=None,trials=0)
        hit=self._rollout(obstacles,nominal,state,path)
        if hit is None:return nominal,dict(active=False,nominal_hit_tick=None,trials=1)
        reference=self._preview(obstacles,nominal,state,path)
        goal=max(reference['advance'],1.)
        steers=np.unique(np.clip(np.r_[float(nominal[0])+np.array([-.36,-.18,0,.18,.36]),
                                       0.,state['wheel']],-.85,.85))
        pedals={(float(nominal[1]),float(nominal[2])),(0.,0.),(0.,.15),(0.,.3),(0.,.4),(0.,.7)}
        candidates=[];trials=2
        for steer in steers:
            for gas,brake in sorted(pedals):
                action=np.array([steer,gas,brake],np.float32)
                prediction=self._preview(obstacles,action,state,path);trials+=1
                if prediction['hit'] is not None:continue
                # Meters normalized by the local corridor scale, nominal travel
                # and speed. Heading and action deviation preserve P1 preference.
                cost=(prediction['lateral']/3.)**2+.5*prediction['heading']**2
                cost+=((goal-prediction['advance'])/goal)**2
                cost+=((reference['speed']-prediction['speed'])/max(state['speed'],8.))**2
                cost+=.15*((float(steer)-float(nominal[0]))/.4)**2
                cost+=.03*((gas-float(nominal[1]))**2+(brake-float(nominal[2]))**2)
                candidates.append((cost,action,prediction))
        moving=[c for c in candidates if c[2]['advance']>.3 and c[2]['speed']>1.]
        eligible=moving or candidates
        if not eligible:
            chosen,diag=super()._shield(obstacles,nominal,state,path)
            diag.update(joint_branch=True,joint_trials=trials,moving_backups=0,backup_advance_m=None)
            return chosen,diag
        cost,chosen,prediction=min(eligible,key=lambda c:c[0])
        return chosen,dict(active=True,nominal_hit_tick=hit,backup_hit_tick=None,trials=trials,
            backup_found=True,joint_branch=True,moving_backups=len(moving),
            backup_advance_m=prediction['advance'],backup_lateral_m=prediction['lateral'],
            backup_terminal_speed=prediction['speed'],backup_cost=float(cost),
            backup_position=prediction['position'])

    def act(self,observation):
        action=super().act(observation)
        if self.diagnostics.get('obstacle_shield',{}).get('active'):
            self.last_steer=float(action[0]);self.previous_steer=float(action[0])
            self.last_command=action.copy()
        return action
