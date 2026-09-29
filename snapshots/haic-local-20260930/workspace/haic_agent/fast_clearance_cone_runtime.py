"""Pixel-only passing-clearance steering constraint with unchanged high acceleration."""
import numpy as np
from haic_agent.fast_road_commit_runtime import FastRoadCommitAgent
from haic_agent.geometric_passing_runtime import PX_X, PX_Y, WHEELBASE, GeometricPassingAgent
from haic_agent.pixel_features import current_frame

PARENTS = {'control':'control', 'current':'commit_side', 'cone_commit':'commit_side',
           'cone_bearing':'passing_bearing', 'cone_shock':'shock_recovery', 'cone_rate':'continuous_avoidance'}


class FastClearanceConeAgent(FastRoadCommitAgent):
    def __init__(self, mechanism='cone_commit'):
        if mechanism not in PARENTS:
            raise ValueError('unknown clearance mechanism')
        self.cone_mode = mechanism
        super().__init__(PARENTS[mechanism])

    def act(self, observation):
        action = super().act(observation)
        before = action.copy()
        obstacle = self.base._last_obstacle
        bearing = None
        point = None
        road_bounds = None
        outside_road = None
        if self.cone_mode.startswith('cone_') and obstacle is not None and self.committed_side:
            ox, oy = (obstacle[1]-42)/PX_X, (63-obstacle[0])/PX_Y
            px = ox+3*self.committed_side
            point = (px,oy)
            corridor = GeometricPassingAgent._corridor(self,current_frame(observation))
            if len(corridor)>=2:
                forward,center,left,right = corridor.T
                road_bounds = (float(np.interp(oy,forward,left)),float(np.interp(oy,forward,right)))
                outside_road = not road_bounds[0]<=px<=road_bounds[1]
            bearing = float(np.arctan(2*WHEELBASE*px/max(oy*oy+px*px,1e-6)))
            if self.committed_side>0:
                action[0] = max(float(action[0]),bearing)
            else:
                action[0] = min(float(action[0]),bearing)
            action[0] = np.clip(action[0],-.7,.7)
        changed = bool(abs(float(action[0]-before[0]))>1e-6)
        self.last.update(cone_mode=self.cone_mode,cone_changed=changed,cone_point=point,
                         cone_road_bounds=road_bounds,cone_point_outside_road=outside_road,
                         cone_bearing=bearing,cone_physically_saturated=bool(bearing is not None and abs(bearing)>.4),
                         cone_parent_steer=float(before[0]),cone_final_steer=float(action[0]),
                         steering_delta_from_baseline=self.last['steering_delta_from_baseline']+float(action[0]-before[0]),
                         completion_changed=bool(np.max(np.abs(action-np.asarray(self.last['baseline_action'])))>1e-6))
        return action
