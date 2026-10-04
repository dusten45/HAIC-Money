"""V2 full-image geodesic path controller; pixel-only independent runtime.

Unlike row-monotone planning, its path may turn horizontally or back down the
image. Calibrated pixel dynamics and P1 pedal primitives are reused verbatim.
"""
from __future__ import annotations
import heapq
import cv2
import numpy as np

DEFAULTS = dict(max_speed=100., min_speed=24., lateral_accel=100.,
                braking_accel=65., lookahead=18., speed_lookahead=.1,
                pursuit_gain=4.5, steer_smoothing=.35, max_steer=.85,
                clearance=3., obstacle_margin=3, road_low=.24, road_high=.52,
                motion_preview=.12, traction_accel=180., brake_gain=.025)

class _GeodesicR1:
    PIXELS_X, PIXELS_Y = 1.3608, 1.701

    def __init__(self, config=None, *, project_root=None):
        self.config = dict(DEFAULTS)
        if config:
            unknown = set(config)-set(self.config)
            if unknown:
                raise ValueError(f'unknown configuration keys: {sorted(unknown)}')
            self.config.update(config)
        self.reset()

    def reset(self, observation=None):
        self.internal_throttle = 0.
        self.last_steer = 0.
        self.diagnostics = {}
        self.pedal_diagnostics = {}

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

    def _traction_gas_limit(self, speed, steer):
        """Reserve tire force for turning instead of saturating the rear wheels.

        Independent uniform-road dynamics identification supports half throttle
        under moderate lateral load, tapering to coast near 180 m/s² demand.
        This is a runtime image-speed/action calculation, not environment access.
        """
        limit = float(self.config['traction_accel'])
        if limit <= 0:
            return 1.
        demand = speed*speed*abs(np.tan(steer))/3.24
        turning_cap = .5+.5*float(np.clip(1.-demand/20., 0., 1.))
        reserve = float(np.clip((limit-demand)/(limit*4./9.), 0., 1.))
        return turning_cap*reserve

    @staticmethod
    def _predict_pedals(speed, steer, gas, brake, internal_throttle):
        """Four public20ms rolling-model ticks; no saturation/slip guarantee."""
        velocity = float(speed)
        throttle = float(internal_throttle)
        turning_drag = 0.00031570330627447987 * float(np.tan(steer)) ** 2
        for _ in range(4):
            throttle += min(float(gas) - throttle, 0.1)
            acceleration = (2735.0620191784787 * throttle / (velocity + 2.7)
                            - 303.8957799087198 * brake
                            - turning_drag * velocity ** 3)
            velocity = max(0.0, velocity + 0.02 * acceleration)
        return float(velocity), float(throttle)

    def _allocate_pedals(self, speed, target, steer, gas_cap, slip):
        initial_throttle = self.internal_throttle
        mode = 'rolling'
        if speed < 20.0 or speed > 90.0 or abs(slip) > 0.08726646259971647:
            # Preserve frozen HUD100 behavior outside the identified domain.
            excess = speed - target
            gas = float(np.clip((target-speed)*.10+.25,0.,1.)) if excess <= 1 else 0.
            brake = float(np.clip(excess*self.config['brake_gain'],.03,.7)) if excess > 1 else 0.
            gas = min(gas, gas_cap)
            mode = 'legacy_outside_rolling'
            predicted = None
        else:
            coast_speed, _ = self._predict_pedals(speed, steer, 0.0, 0.0, initial_throttle)
            # One m/s matches the uncertainty of this HUD speed measurement.
            if target - 1.0 <= coast_speed <= target + 1.0:
                gas, brake, predicted = 0.0, 0.0, coast_speed
                mode = 'coast_deadband'
            else:
                desired_acceleration = float(np.clip((target-speed)/0.16,
                                        -self.config['braking_accel'], 45.0))
                desired_speed = max(0.0, speed + 0.08*desired_acceleration)
                gas = brake = 0.0
                if coast_speed > desired_speed:
                    low, high = 0.0, 0.4
                    for _ in range(14):
                        middle = (low+high)*0.5
                        predicted, _ = self._predict_pedals(speed, steer, 0.0, middle, initial_throttle)
                        if predicted > desired_speed:
                            low = middle
                        else:
                            high = middle
                    brake = high
                else:
                    low, high = 0.0, float(gas_cap)
                    for _ in range(14):
                        middle = (low+high)*0.5
                        predicted, _ = self._predict_pedals(speed, steer, middle, 0.0, initial_throttle)
                        if predicted < desired_speed:
                            low = middle
                        else:
                            high = middle
                    gas = high
                predicted, _ = self._predict_pedals(speed, steer, gas, brake, initial_throttle)
        # Throttle state follows commands even while the speed model is unused.
        for _ in range(4):
            self.internal_throttle += min(gas-self.internal_throttle, 0.1)
        self.pedal_diagnostics = dict(allocator_mode=mode,
                internal_throttle=float(self.internal_throttle), predicted_next_speed=predicted)
        return float(gas), float(brake)

    def _geodesic(self, free, start=(42,58), goal=None):
        """Clearance-weighted 8-connected search with unrestricted later turns."""
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        allowed = distance >= self.config['clearance']
        sy, sx = int(start[1]), int(start[0])
        # Suppress the rearward road branch locally, without preventing a
        # hairpin from returning below the car away from its departure gate.
        allowed[sy+1:, max(0,sx-18):min(84,sx+19)] = False
        yy,xx = np.nonzero(allowed)
        if not len(xx):
            return np.empty((0,2))
        nearest = np.argmin((xx-sx)**2+(yy-sy)**2)
        sx,sy=int(xx[nearest]),int(yy[nearest])
        costs=np.full((84,84),np.inf); costs[sy,sx]=0.
        parent=np.full((84,84,2),-1,np.int16)
        length=np.zeros((84,84))
        queue=[(0.,sx,sy)]
        moves=[(dx,dy,float(np.hypot(dx,dy))) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
        while queue:
            cost,x,y=heapq.heappop(queue)
            if cost>costs[y,x]+1e-9:
                continue
            for dx,dy,step in moves:
                nx,ny=x+dx,y+dy
                if not (0<=nx<84 and 0<=ny<74 and allowed[ny,nx]):
                    continue
                if dx and dy and not (allowed[y,nx] and allowed[ny,x]):
                    continue
                new=cost+step*(1.+12./(float(distance[ny,nx])+.5)**2)
                if new<costs[ny,nx]:
                    costs[ny,nx]=new;parent[ny,nx]=[x,y]
                    length[ny,nx]=length[y,x]+step
                    heapq.heappush(queue,(new,nx,ny))
        reached=np.isfinite(costs)
        yy,xx=np.nonzero(reached)
        if goal is not None:
            i=np.argmin((xx-goal[0])**2+(yy-goal[1])**2)
        else:
            frontier=(yy<=5)|(xx<=4)|(xx>=79)|(yy>=70)
            # Prefer a visible outgoing road frontier; if it is enclosed by
            # image grass, use the farthest reachable centerline continuation.
            candidates=np.where(frontier & (length[yy,xx]>18))[0]
            if len(candidates):
                # Central exit pixel wins within one opening. Shortest exit
                # avoids tracing unnecessarily around a visible road loop.
                score=costs[yy[candidates],xx[candidates]]-2*distance[yy[candidates],xx[candidates]]
                i=candidates[np.argmin(score)]
            else:
                i=np.argmax(length[yy,xx]+2*distance[yy,xx])
        x,y=int(xx[i]),int(yy[i]);path=[]
        while x>=0:
            path.append([x,y]);x,y=map(int,parent[y,x])
        path=np.array(path[::-1],float)
        if len(path)<3:return path
        original=path.copy()
        # Local constrained elastic smoothing. Every updated pair is densely
        # checked in the image; neither endpoints nor swept segments may cut a
        # corner across the configured sampled-distance clearance threshold.
        for _ in range(35):
            proposed=path.copy()
            proposed[1:-1]=.49*(path[:-2]+path[2:])+.02*original[1:-1]
            points=(proposed[:-1,None,:]*(1-np.linspace(0,1,5)[None,:,None])+
                    proposed[1:,None,:]*np.linspace(0,1,5)[None,:,None])
            px,py=np.rint(points).astype(int).transpose(2,0,1)
            safe=(distance[py,px]>=self.config['clearance']).all(axis=1)
            accept=np.r_[True,safe]&np.r_[safe,True]
            path[accept]=proposed[accept]
        return path

    @staticmethod
    def _geometry(metric):
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(metric,axis=0),axis=1))]
        curvature=np.zeros(len(metric))
        for i in range(len(metric)):
            lo,hi=max(0,i-6),min(len(metric),i+7)
            if hi-lo<5:continue
            ss=arc[lo:hi]-arc[i]
            px=np.polyfit(ss,metric[lo:hi,0],3)
            py=np.polyfit(ss,metric[lo:hi,1],3)
            dx,dy=px[-2],py[-2]
            curvature[i]=(dy*2*px[-3]-dx*2*py[-3])/max((dx*dx+dy*dy)**1.5,.001)
        return arc,curvature

    def _tracking_arc(self, metric, free, speed, nominal):
        """Select a curvature-bounded connector whose sampled arc is drivable.

        The geodesic gives topology; this finite family imposes forward vehicle
        connectivity. It does not model transient tire slip or certify footprint.
        """
        horizon = min(max(8., speed*.35+5.), 22.)
        curvature = np.unique(np.r_[np.linspace(-.115,.115,69), np.clip(nominal,-.115,.115)])
        distance = np.linspace(3., horizon, 38)
        angle = curvature[:,None]*distance[None,:]
        denominator = np.where(abs(curvature)>1e-7,curvature,1.)[:,None]
        px = (1-np.cos(angle))/denominator
        py = np.sin(angle)/denominator
        py[abs(curvature)<1e-7] = distance
        arcs = np.stack([px,py],axis=-1)
        image = np.rint(arcs*np.array([1.3608,-1.701])+[42,63]).astype(int)
        inside=(image[:,:,0]>=0)&(image[:,:,0]<84)&(image[:,:,1]>=0)&(image[:,:,1]<74)
        x=np.clip(image[:,:,0],0,83);y=np.clip(image[:,:,1],0,83)
        clearance=cv2.distanceTransform(free,cv2.DIST_L2,5)
        visible = image[:,:,1]<=58
        collisions=(~inside | (clearance[y,x]<self.config['clearance'])) & visible
        mismatch=np.min(np.sum((arcs[:,:,None,:]-metric[None,None,:,:])**2,axis=-1),axis=-1)
        score=np.mean(mismatch,axis=1)+8*(curvature-nominal)**2
        score+=10000*np.sum(collisions,axis=1)
        nominal_index=int(np.argmin(abs(curvature-np.clip(nominal,-.115,.115))))
        index=nominal_index if not collisions[nominal_index].any() else int(np.argmin(score))
        return float(curvature[index]), bool(not collisions[index].any())

    def act(self, observation):
        obs=np.asarray(observation)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all() or obs.min()<0 or obs.max()>1:
            self.diagnostics={'valid':False}
            return np.array([0,0,.3],np.float32)
        c=self.config;frame=obs[-1].astype(np.float32)
        free,obstacles,clearance=self._free_space(frame)
        path=self._geodesic(free)
        if len(path)<5:
            self.diagnostics={'valid':False,'path':path.tolist()}
            self.internal_throttle=0.
            return np.array([self.last_steer*.8,0,.3],np.float32)
        metric=(path-np.array([42.,63.]))/np.array([self.PIXELS_X,-self.PIXELS_Y])
        arc,kappas=self._geometry(metric)
        arc+=np.linalg.norm(metric[0])
        mass=obs[-2:,77:83,10:13].sum(axis=(1,2)).mean()
        speed=float(np.clip((mass-.27)/.085,0,100))
        lookahead=min(c['lookahead']+c['speed_lookahead']*speed,arc[-1])
        aim=np.array([np.interp(lookahead,arc,metric[:,j]) for j in range(2)])
        curvature=2*aim[0]/max(float(aim@aim),4.)
        yaw,wheel=self._hud_dynamics(frame)
        _,slip,valid=self._motion(obs[-2],obs[-1])
        curvature+=2*c['motion_preview']*(speed*curvature-yaw)/max(lookahead,4.)
        curvature-=2*speed*np.sin(slip)*c['motion_preview']/max(lookahead**2,16.)
        curvature, connector_feasible = self._tracking_arc(metric,free,speed,curvature)
        steer=float(np.clip(c['pursuit_gain']*curvature,-c['max_steer'],c['max_steer']))
        steer=(1-c['steer_smoothing'])*steer+c['steer_smoothing']*self.last_steer
        caps=np.clip(np.sqrt(c['lateral_accel']/np.maximum(abs(kappas),.001)),c['min_speed'],c['max_speed'])
        target=min(c['max_speed'],float(np.min(np.sqrt(caps*caps+2*c['braking_accel']*arc))),float(np.sqrt(c['lateral_accel']/max(abs(curvature),.001))))
        # Visible frontier carries uncertainty: retain enough distance to
        # decelerate to a corner speed just beyond the current image horizon.
        target=min(target,float(np.sqrt(c['min_speed']**2+2*c['braking_accel']*arc[-1])))
        target=max(c['min_speed'],target)
        if not connector_feasible:
            target=min(target,c['min_speed'])
        cap=self._traction_gas_limit(speed,steer)
        gas,brake=self._allocate_pedals(speed,target,steer,cap,slip)
        self.last_steer=steer
        self.diagnostics=dict(valid=True,path=path.tolist(),speed=speed,target_speed=target,
                              curvature=curvature,yaw_rate=yaw,wheel_angle=wheel,slip=slip,
                              flow_valid=valid,horizon=float(arc[-1]),gas_cap=cap,connector_feasible=connector_feasible,
                              path_curvature=kappas.tolist(),path_arc=arc.tolist())
        self.diagnostics.update(self.pedal_diagnostics)
        return np.array([steer,gas,brake],np.float32)

    def last_step_diagnostics(self):
        return dict(self.diagnostics)


# GeodesicR1 above copied verbatim except class name from frozen archive SHA256
# 36c7a4cbbb5d930708cf639aea9d55e0273490ee962e1994218d12a140296218.
from agents.apex_2026.candidate.agent import Agent as _NominalP1


class Agent:
    """P1 default, temporary 2D fallback on pixel geometry warning.

    The sampled arc test is a heuristic warning, not a formal safety certificate.
    Both controllers consume every frame and share executed-action memory.
    """
    def __init__(self,config=None,*,project_root=None):
        if config:
            raise ValueError('fallback r1 has no external tuning configuration')
        self.nominal=_NominalP1(project_root=project_root)
        self.alternative=_GeodesicR1(project_root=project_root)
        self.reset()

    def reset(self,observation=None):
        self.nominal.reset(observation)
        self.alternative.reset(observation)
        self.active=False
        self.dwell=0
        self.clear_count=0
        self.last_steer=0.
        self.throttle=0.
        self.diagnostics={}
        self.total_steps=0
        self.fallback_steps=0
        self.switches=0

    @staticmethod
    def certificate(path,free,speed,curvature):
        """State-only road/connectivity warnings in the current camera frame."""
        path=np.asarray(path,float)
        if len(path)<3:
            return dict(certified=False,prefix_unreachable=True,arc_unsafe=True,
                        lateral_continuation=False,min_arc_clearance=0.)
        metric=(path-[42.,63.])/[1.3608,-1.701]
        x,y=metric[0]
        required=abs(2*x/max(x*x+y*y,.01))
        prefix=required>np.tan(.4)/3.24 and abs(x)>1.5
        # Bound lookahead to local reaction/turn-in geometry, not the entire
        # visible route: a constant arc cannot represent an ordinary S bend.
        horizon=float(np.clip(4.+.2*speed,7.,13.))
        distance=np.linspace(3.,horizon,30)
        k=float(curvature)
        px=(1-np.cos(k*distance))/k if abs(k)>1e-7 else np.zeros_like(distance)
        py=np.sin(k*distance)/k if abs(k)>1e-7 else distance
        xi=np.rint(42+1.3608*px).astype(int)
        yi=np.rint(63-1.701*py).astype(int)
        inside=(xi>=0)&(xi<84)&(yi>=0)&(yi<74)
        field=cv2.distanceTransform(free,cv2.DIST_L2,5)
        clearance=np.zeros(len(xi));clearance[inside]=field[yi[inside],xi[inside]]
        unsafe=bool(np.count_nonzero(clearance<1.8)>=2)
        # An interior row endpoint with connected road extending sideways below
        # it indicates that a monotonically upward path may have truncated.
        endpoint=path[-1]
        lateral=False
        if endpoint[1]>18 and len(path)<16:
            _,labels=cv2.connectedComponents(free.astype(np.uint8),8)
            ex,ey=np.rint(endpoint).astype(int)
            label=labels[np.clip(ey,0,83),np.clip(ex,0,83)]
            if label:
                yy,xx=np.nonzero(labels==label)
                lateral=bool(np.any((abs(xx-42)>18)&(yy>ey+4)&(yy<59)))
        return dict(certified=not(prefix or unsafe or lateral),prefix_unreachable=bool(prefix),
                    prefix_curvature=float(required),arc_unsafe=unsafe,
                    lateral_continuation=lateral,min_arc_clearance=float(clearance.min()),
                    warning_horizon_m=horizon)

    def _select_mode(self,certified):
        old=self.active
        if not certified:
            self.active=True
            self.clear_count=0
        elif self.active:
            self.clear_count+=1
            if self.dwell>=4 and self.clear_count>=3:
                self.active=False
        if self.active:
            self.dwell=self.dwell+1 if old else 1
        else:
            self.dwell=0
        if old!=self.active:self.switches+=1
        return self.active

    def _record_action(self,action,previous_steer,prior_throttle,exact_steer=None,exact_throttle=None):
        throttle=float(prior_throttle)
        for _ in range(4):throttle+=min(float(action[1])-throttle,.1)
        self.throttle=throttle if exact_throttle is None else float(exact_throttle)
        self.last_steer=float(action[0]) if exact_steer is None else float(exact_steer)
        for policy in (self.nominal,self.alternative):
            policy.last_steer=self.last_steer
            policy.internal_throttle=self.throttle
        self.nominal.previous_steer=float(previous_steer)

    def act(self,observation):
        prior_steer,prior_throttle=self.last_steer,self.throttle
        nominal_action=self.nominal.act(observation)
        alternative_action=self.alternative.act(observation)
        nd=self.nominal.diagnostics
        obs=np.asarray(observation)
        if nd.get('valid'):
            free,_,_=self.nominal._free_space(obs[-1])
            cert=self.certificate(nd['path'],free,nd['speed'],nd['curvature'])
        else:
            cert=dict(certified=False,prefix_unreachable=True,arc_unsafe=True,
                      lateral_continuation=False,min_arc_clearance=0.)
        active=self._select_mode(cert['certified'])
        selected=self.alternative if active else self.nominal
        action=alternative_action if active else nominal_action
        exact_steer=selected.last_steer
        exact_throttle=selected.internal_throttle
        self._record_action(action,prior_steer,prior_throttle,exact_steer,exact_throttle)
        self.total_steps+=1;self.fallback_steps+=int(active)
        self.diagnostics=dict(selected.diagnostics)
        self.diagnostics.update(fallback_active=active,fallback_dwell=self.dwell,
                                fallback_clear_count=self.clear_count,
                                fallback_fraction=self.fallback_steps/self.total_steps,
                                fallback_switches=self.switches,geometry_warning=cert,
                                nominal_action=nominal_action.tolist(),alternative_action=alternative_action.tolist())
        return action

    def last_step_diagnostics(self):
        return dict(self.diagnostics)
