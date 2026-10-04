"""Pose-anchored joint MPC with visible-space brake-tail trajectories.

Perception is copied from the frozen P1 pixel controller; no runtime controller
imports. The rolling model is calibrated in its unsaturated moving domain.
Tire saturation and uncertain visual slip remain approximate model limitations.
"""
from __future__ import annotations
import numpy as np
import cv2

DEFAULTS = dict(road_low=.24, road_high=.52, obstacle_margin=0,
 clearance=3., center_cost=2., previous_cost=.015, bend_cost=.8,
 slope_cost=.04, path_smoothing=200., max_speed=100., horizon_ticks=48)

class Agent:
    PIXELS_X, PIXELS_Y = 1.3608, 1.701
    def __init__(self, config=None, *, project_root=None):
        self.config=dict(DEFAULTS)
        if config:
            unknown=set(config)-set(self.config)
            if unknown: raise ValueError(f'unknown configuration: {sorted(unknown)}')
            self.config.update(config)
        self.reset()

    def reset(self, observation=None):
        self.internal_throttle=0.
        self.previous_path=None
        self._smooth_cache={}
        self.slip_estimate=0.
        self.slip_uncertainty=.04
        self.prediction=None
        self.speed_residual=0.
        self.yaw_residual=0.
        self.diagnostics={}

    def _free_space(self, frame):
            c = self.config
            road = ((frame >= c['road_low']) & (frame <= c['road_high'])).astype(np.uint8)
            # Small image holes arise from aliasing/paint, not only obstacles. Recover
            # the road surface first, then subtract separately identified hazards.
            road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            bright = (frame >= .54).astype(np.uint8)
            bright[:5] = 0
            bright[61:] = 0
            count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
            obstacles = np.zeros_like(road)
            for i in range(1, count):
                x, y, w, h, area = stats[i]
                if 4 <= area <= 90 and 2 <= w <= 11 and 2 <= h <= 12:
                    # Component must be surrounded by road, excluding grass islands.
                    region = road[max(0,y-2):min(74,y+h+2), max(0,x-2):min(84,x+w+2)]
                    if region.mean() > .2:
                        obstacles[labels == i] = 1
            radius = int(c['obstacle_margin'])
            obstacles = cv2.dilate(obstacles, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1,2*radius+1)))
            road[74:] = 0
            free = road.copy()
            free[obstacles > 0] = 0
            clearance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
            return free, obstacles, clearance

    def _path(self, frame):
            free, obstacles, clearance = self._free_space(frame)
            c = self.config
            rows = np.arange(58, 7, -2)
            xs = np.arange(84)
            slopes = np.arange(-4, 5)
            costs = np.full((84, 9), np.inf)
            # Starting pose is known from the fixed renderer, not simulator state.
            costs[:, 4] = .35*(xs-42.)**2
            costs[np.abs(xs-42) > 12] = np.inf
            parents = []
            accepted = []
            last_costs = costs
            for row in rows:
                d = clearance[row]
                node = c['center_cost']/(d+.5) + 3.*np.maximum(c['clearance']-d, 0.)**2
                node += 70.*(free[row] == 0) + 300.*obstacles[row]
                if self.previous_path is not None:
                    px = float(np.interp(row, self.previous_path[::-1,1], self.previous_path[::-1,0]))
                    node += c['previous_cost']*(xs-px)**2
                next_costs = np.full_like(costs, np.inf)
                parent = np.zeros((84,9), np.int16)
                for j, delta in enumerate(slopes):
                    prevx = xs-delta
                    valid = (prevx >= 0) & (prevx < 84)
                    indices = np.flatnonzero(valid)
                    options = costs[prevx[valid]] + c['bend_cost']*(slopes-delta)**2
                    which = np.argmin(options, axis=1)
                    next_costs[indices,j] = options[np.arange(len(indices)),which] + node[valid] + c['slope_cost']*delta**2
                    parent[indices,j] = which
                # Stop at the camera's useful road horizon rather than hallucinate
                # a route over grass beyond an almost-horizontal visible bend.
                best = np.unravel_index(np.argmin(next_costs), next_costs.shape)
                if accepted and (not free[row,best[0]] or np.min(next_costs) > np.min(costs)+100.):
                    break
                accepted.append(row)
                parents.append(parent)
                costs = next_costs
                last_costs = costs
            x, j = np.unravel_index(np.argmin(last_costs), last_costs.shape)
            result = []
            for k in range(len(accepted)-1, -1, -1):
                result.append((float(x),float(accepted[k])))
                oldj = int(parents[k][x,j])
                x -= int(slopes[j])
                j = oldj
            path = np.asarray(result[::-1], np.float32).reshape(-1,2)
            path = self._smooth_path(path, free)
            return path, free, obstacles

    def _smooth_path(self, path, free):
            """Refine the lattice route continuously inside its chosen free intervals.
    
            Integer x steps are search artifacts, not physical corners. A convex
            curvature penalty removes those artifacts without crossing an obstacle
            or changing which side of an obstacle the discrete search selected.
            """
            weight = float(self.config['path_smoothing'])
            if weight <= 0 or len(path) < 4:
                return path
            lower, upper = [], []
            for x, y in path:
                xi, yi = int(round(float(x))), int(round(float(y)))
                if not free[yi, xi]:
                    return path
                left = right = xi
                while left > 0 and free[yi, left-1]:
                    left -= 1
                while right < 83 and free[yi, right+1]:
                    right += 1
                margin = min(float(self.config['clearance']), (right-left)*.45)
                lower.append(left+margin)
                upper.append(right-margin)
            lower, upper = np.array(lower), np.array(upper)
            n = len(path)
            key = (n, weight)
            if key not in self._smooth_cache:
                difference = np.diff(np.eye(n), n=2, axis=0)
                self._smooth_cache[key] = np.eye(n)+weight*(difference.T@difference)
            hessian = self._smooth_cache[key]
            data = path[:,0].astype(np.float64)
            unconstrained = np.linalg.solve(hessian, data)
            x = np.clip(unconstrained, lower, upper)
            if np.max(np.abs(x-unconstrained)) > 1e-8:
                # Accelerated projected gradient solves the small box-constrained QP.
                extrapolated, momentum = x.copy(), 1.
                step = 1./(1.+16.*weight)
                for _ in range(120):
                    new = np.clip(extrapolated-step*(hessian@extrapolated-data), lower, upper)
                    next_momentum = (1.+np.sqrt(1.+4.*momentum*momentum))*.5
                    extrapolated = new+(momentum-1.)/next_momentum*(new-x)
                    x, momentum = new, next_momentum
            return np.column_stack((x, path[:,1])).astype(np.float32)

    def _motion(self, previous, current):
            """Shared measurement concept independently validated by the rollout lane."""
            old = np.clip(previous * 255, 0, 255).astype(np.uint8)
            new = np.clip(current * 255, 0, 255).astype(np.uint8)
            mask = np.zeros((84, 84), np.uint8)
            mask[4:70, 4:80] = 255
            mask[56:70, 36:48] = 0
            points = cv2.goodFeaturesToTrack(old, 100, 0.015, 4, mask=mask)
            if points is None or len(points) < 6:
                return 0.0, 0.0, False
            moved, status, error = cv2.calcOpticalFlowPyrLK(old, new, points, None,
                                                         winSize=(15, 15), maxLevel=2)
            if moved is None:
                return 0.0, 0.0, False
            keep = (status.ravel() > 0) & (error.ravel() < 30)
            if np.count_nonzero(keep) < 6:
                return 0.0, 0.0, False
            scale = np.array([self.PIXELS_X, self.PIXELS_Y], np.float32)
            before = points.reshape(-1, 2)[keep] / scale
            after = moved.reshape(-1, 2)[keep] / scale
            transform, inliers = cv2.estimateAffinePartial2D(before, after,
                        method=cv2.RANSAC, ransacReprojThreshold=0.8, maxIters=500)
            if transform is None or np.count_nonzero(inliers) < 6:
                return 0.0, 0.0, False
            center = np.array([42.0, 63.0]) / scale
            delta = transform[:, :2] @ center + transform[:, 2] - center
            yaw_rate = -np.arctan2(transform[1, 0], transform[0, 0]) / 0.08
            velocity = np.array([-delta[0], delta[1]]) / 0.08
            slip = np.arctan2(velocity[0], max(0.1, velocity[1]))
            zoom = np.sqrt(np.linalg.det(transform[:, :2]))
            if not (0.96 < zoom < 1.04) or abs(yaw_rate) > 8 or np.linalg.norm(velocity) > 140:
                return 0.0, 0.0, False
            return float(yaw_rate), float(np.clip(slip, -0.8, 0.8)), True

    @staticmethod
    def _hud_dynamics(frame):
            """Decode instantaneous HUD state, with positive values turning right.
    
            Pure observation-only calibration from diagnostics/hud_dynamics_calibration;
            no renderer, simulator, or calibration dependency is imported at runtime.
            """
            yaw_crop = frame[74:82,52:75]
            wheel_crop = frame[74:82,31:52]
            yaw_mass = float(yaw_crop[:,:11].sum()-yaw_crop[:,11:].sum())
            wheel_mass = float(wheel_crop[:,:11].sum()-wheel_crop[:,11:].sum())
            yaw = .4580150260806972*yaw_mass+.024360133436811916 if yaw_crop.sum() else 0.
            wheel = .018757917750222175*wheel_mass+.0019537197954649153 if wheel_crop.sum() else 0.
            return -float(yaw), -float(wheel)

    @staticmethod
    def _longitudinal(speed,wheel,gas,brake,throttle,ticks=1):
        v=np.array(speed,dtype=float,copy=True)
        g=np.array(throttle,dtype=float,copy=True)
        for _ in range(ticks):
            g+=np.minimum(gas-g,.1)
            drive=2735.0620191784787*g/(v+2.7)
            # Rolling fit does not model wheelspin. Force caps avoid extending
            # its inverse-speed drive prediction into unphysical accelerations.
            drive=np.minimum(drive,np.where(v<20.,65.,43.8))
            drag=.00031570330627447987*v**3*np.tan(wheel)**2
            v=np.clip(v+.02*(drive-303.8957799087198*brake-drag),0.,100.)
        return v,g

    def _update_slip(self,speed,slip,valid,yaw):
        if speed<8:
            self.slip_estimate=0.
            self.slip_uncertainty=.04
        elif valid:
            self.slip_estimate=float(np.clip(slip+.04*yaw,-.6,.6))
            self.slip_uncertainty=.035
        else:
            # Missing optical flow is not a zero-slip measurement.
            self.slip_estimate*=.95
            self.slip_uncertainty=min(.18,self.slip_uncertainty+.012)

    def _metric_clearance(self,frame):
        free,_,_=self._free_space(frame)
        # Fixed car sprite occludes road underneath the body only.
        free[59:68,39:45]=1
        metric=cv2.resize(free[:74],(84,59),interpolation=cv2.INTER_NEAREST)
        metric=np.pad(metric,((1,1),(1,1)))
        return cv2.distanceTransform(metric,cv2.DIST_L2,5)/self.PIXELS_X

    def _sample_clearance(self,field,x,y):
        xx=np.asarray(43.+self.PIXELS_X*x,np.float32)
        yy=np.asarray(1.+(63.-self.PIXELS_Y*y)*59./74.,np.float32)
        shape=xx.shape
        return cv2.remap(field,xx.reshape((-1,1) if xx.ndim==1 else xx.shape),yy.reshape((-1,1) if yy.ndim==1 else yy.shape),
            cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0).reshape(shape)

    def _rollouts(self,speed,yaw,wheel,slip,steers,gases,brakes,delays,profiles):
        n=len(steers); ticks=int(self.config['horizon_ticks'])
        result={k:np.zeros((n,ticks+1),np.float32) for k in ('x','y','v','yaw','wheel','heading','throttle','load')}
        result['v'][:,0]=speed; result['yaw'][:,0]=yaw
        result['wheel'][:,0]=wheel; result['throttle'][:,0]=self.internal_throttle
        beta=np.full(n,slip,dtype=float)
        for t in range(ticks):
            w=result['wheel'][:,t]; v=result['v'][:,t]; r=result['yaw'][:,t]
            # Pulse candidates allow an S-bend correction without assuming
            # that today's steering command persists for the whole horizon.
            target=np.where((profiles>0)&(t>=profiles),-steers,steers)
            target=np.where((profiles>0)&(t>=2*profiles),0.,target)
            w=w+np.clip(target-w,-.06,.06)
            g=np.where(t>=delays,0.,gases)
            b=np.where(t>=delays,.7,brakes)
            nominal=v*np.tan(w)/3.24
            r=r+.2*(np.clip(nominal,-180./np.maximum(v,1.),180./np.maximum(v,1.))-r)
            lateral=np.abs(v*r)
            # Coupled rear-force risk: large lateral demand lowers admissible
            # throttle; this is a conservative cost, not an exact slip model.
            drive=2735.0620191784787*np.minimum(result['throttle'][:,t]+.1,g)/(v+2.7)
            result['load'][:,t+1]=np.maximum(0.,lateral/180.+drive/85.-1.)
            vn,gn=self._longitudinal(v,w,g,b,result['throttle'][:,t])
            h=result['heading'][:,t]+.02*r
            beta+=.2*(np.arctan(.5*np.tan(w))-beta)
            angle=h+beta
            result['x'][:,t+1]=result['x'][:,t]+.02*.5*(v+vn)*np.sin(angle)
            result['y'][:,t+1]=result['y'][:,t]+.02*.5*(v+vn)*np.cos(angle)
            result['heading'][:,t+1]=h; result['yaw'][:,t+1]=r
            result['wheel'][:,t+1]=w; result['v'][:,t+1]=vn
            result['throttle'][:,t+1]=gn
        return result

    def _select(self,frame,path,speed,yaw,wheel):
        # Fine central steering resolution matters at high speed; the wide
        # range still contains the full physical front-wheel joint limits.
        steering=np.unique(np.r_[np.linspace(-.4,.4,17),np.linspace(-.10,.10,17)])
        pedal=[(g,0.,d) for g in (0.,.3,.6,1.) for d in (4,12,100)]
        pedal += [(0.,b,100) for b in (.2,.4,.7)]
        indices=np.indices((len(steering),len(pedal),3)).reshape(3,-1)
        steers=steering[indices[0]]
        p=np.asarray(pedal)[indices[1]]; profiles=np.array([0,8,16])[indices[2]]
        traj=self._rollouts(speed,yaw,wheel,self.slip_estimate,steers,p[:,0],p[:,1],p[:,2],profiles)
        field=self._metric_clearance(frame)
        x,y,h=traj['x'],traj['y'],traj['heading']
        clearance=np.full_like(x,100.)
        for offset in (-1.15,0.,1.3):
            clearance=np.minimum(clearance,self._sample_clearance(field,x+offset*np.sin(h),y+offset*np.cos(h)))
        times=np.arange(x.shape[1])*.02
        tube=.12+speed*self.slip_uncertainty*.18*(1.-np.exp(-times/.18))
        tube+=min(1.0,self.yaw_residual*speed*.1)*times
        violation=np.maximum(1.35+tube[None,:]-clearance,0.)
        # Ignore known initial occupancy, but penalize collision speed heavily.
        collision=np.mean(violation[:,1:]**2*(100.+.2*traj['v'][:,1:]**2),axis=1)
        pts=np.asarray(path,float).reshape(-1,2)
        if len(pts):
            pts=np.c_[(pts[:,0]-42.)/self.PIXELS_X,(63.-pts[:,1])/self.PIXELS_Y]
            pts=np.vstack(([0.,0.],pts))
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
            # Nearest path point is a soft tracking objective. All safety
            # samples remain anchored at the actual observed body pose.
            d2=(x[:,:,None]-pts[None,None,:,0])**2+(y[:,:,None]-pts[None,None,:,1])**2
            closest=np.argmin(d2,axis=2)
            lateral=np.take_along_axis(d2,closest[:,:,None],axis=2)[:,:,0]
            progress=arc[closest[:,-1]]
        else:
            lateral=x*x; progress=y[:,-1]
        cost=collision + .15*np.mean(lateral,axis=1)-progress
        cost+=8.*np.mean(traj['load']**2,axis=1)
        cost+=.025*(steers-wheel)**2 + .04*p[:,1]
        cost-=.08*traj['v'][:,4]
        best=int(np.argmin(cost))
        self.prediction={'speed':float(traj['v'][best,4]),'yaw':float(traj['yaw'][best,4])}
        self.diagnostics.update(min_clearance=float(np.min(clearance[best,1:])),
          predicted_violation=float(np.max(violation[best,1:])),
          predicted_cost=float(cost[best]),brake_tail_tick=int(p[best,2]),
          prediction_next_speed=self.prediction['speed'],prediction_next_yaw=self.prediction['yaw'],
          predicted_progress=float(progress[best]),candidate_count=len(steers))
        return np.array([steers[best],p[best,0],p[best,1]],np.float32)

    def act(self,observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all():
            return np.array([0.,0.,.7],np.float32)
        frame=obs[-1]
        speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0.,100.))
        yaw,wheel=self._hud_dynamics(frame)
        _,slip,valid=self._motion(obs[-2],frame)
        if self.prediction is not None:
            self.speed_residual=.9*self.speed_residual+.1*abs(speed-self.prediction['speed'])
            self.yaw_residual=.9*self.yaw_residual+.1*abs(yaw-self.prediction['yaw'])
        self._update_slip(speed,slip,valid,yaw)
        path,_,_=self._path(frame)
        self.previous_path=path
        self.diagnostics=dict(speed=speed,yaw_rate=yaw,wheel=wheel,slip=self.slip_estimate,
          slip_uncertainty=self.slip_uncertainty,flow_valid=bool(valid),
          speed_residual=self.speed_residual,yaw_residual=self.yaw_residual,path_nodes=len(path))
        action=self._select(frame,path,speed,yaw,wheel)
        for _ in range(4): self.internal_throttle+=min(float(action[1])-self.internal_throttle,.1)
        return action

    def last_step_diagnostics(self):
        return dict(self.diagnostics)
