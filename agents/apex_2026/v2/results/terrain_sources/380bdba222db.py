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

class Agent:
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
        # Apply only obstacle margin exceeding the geometric footprint.
        radius = max(0, int(c['obstacle_margin']-c['clearance']))
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

    def _geodesic(self, free, start=(42,58), goal=None, terrain=None):
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
        surface_cost=np.ones_like(distance) if terrain is None else 1./np.sqrt(.6+.4*terrain)
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
                new=cost+step*(1.+12./(float(distance[ny,nx])+.5)**2)*surface_cost[ny,nx]
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

    def _footprint_collisions(self, arcs, angles, free):
        """Sample the rotated public vehicle envelope instead of only its center.

        Hull reaches2.6m forward/2.4m rearward;1.58m wheel half-width
        conservatively covers front-wheel steering across its public joint
        range. Raster sampling is approximate, not a continuous certificate.
        """
        if not hasattr(self,'_footprint_samples'):
            outline=np.array([[-1.2,2.6],[1.2,2.6],[1.58,2.21],
                              [1.58,1.0],[1.38,-2.18],[1.,-2.4],
                              [-1.,-2.4],[-1.38,-2.18],[-1.58,1.0],[-1.58,2.21]],np.float32)
            samples=[]
            for x in np.arange(-1.58,1.59,.35):
                for y in np.arange(-2.4,2.61,.35):
                    if cv2.pointPolygonTest(outline,(float(x),float(y)),False)>=0:
                        samples.append([x,y])
            for p,q in zip(outline,np.roll(outline,-1,axis=0)):
                samples.extend(np.linspace(p,q,max(2,int(np.linalg.norm(q-p)/.25)+1)).tolist())
            self._footprint_samples=np.asarray(samples)
        body=self._footprint_samples
        cosine=np.cos(angles)[...,None];sine=np.sin(angles)[...,None]
        x=arcs[...,0,None]+cosine*body[:,0]+sine*body[:,1]
        y=arcs[...,1,None]-sine*body[:,0]+cosine*body[:,1]
        ix=np.rint(x*self.PIXELS_X+42).astype(int)
        iy=np.rint(63-y*self.PIXELS_Y).astype(int)
        inside=(ix>=0)&(ix<84)&(iy>=0)&(iy<74)
        occupied=~inside|(free[np.clip(iy,0,83),np.clip(ix,0,83)]==0)
        return np.any(occupied,axis=-1)

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
        collisions |= self._footprint_collisions(arcs,angle,free) & visible
        mismatch=np.min(np.sum((arcs[:,:,None,:]-metric[None,None,:,:])**2,axis=-1),axis=-1)
        score=np.mean(mismatch,axis=1)+8*(curvature-nominal)**2
        score+=10000*np.sum(collisions,axis=1)
        nominal_index=int(np.argmin(abs(curvature-np.clip(nominal,-.115,.115))))
        index=nominal_index if not collisions[nominal_index].any() else int(np.argmin(score))
        return float(curvature[index]), bool(not collisions[index].any())

    def _resample(self, path):
        metric=(np.asarray(path)-[42,63])/[self.PIXELS_X,-self.PIXELS_Y]
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(metric,axis=0),axis=1))]
        if arc[-1]<2.:return np.asarray(path).copy()
        knots=np.linspace(0,arc[-1],max(5,int(arc[-1])+1))
        return np.column_stack([np.interp(knots,arc,path[:,j]) for j in range(2)])

    def _racing_line(self, path, free):
        """Convex bending surrogate in road-aligned lateral coordinates.

        Connected normal intervals retain the chosen obstacle side. The final
        curve is accepted only after dense physical-clearance verification.
        """
        reference=self._resample(path)
        if len(reference)<5:return reference
        scale=np.array([self.PIXELS_X,-self.PIXELS_Y])
        metric=(reference-[42,63])/scale
        tangent=np.gradient(metric,axis=0)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),.001)
        normal=np.column_stack([tangent[:,1],-tangent[:,0]])
        distance=cv2.distanceTransform(free,cv2.DIST_L2,5)
        offsets=np.arange(-7.,7.01,.25)
        probes=(metric[:,None,:]+normal[:,None,:]*offsets[None,:,None])*scale+[42,63]
        pix=np.rint(probes).astype(int)
        inside=(pix[:,:,0]>=0)&(pix[:,:,0]<84)&(pix[:,:,1]>=0)&(pix[:,:,1]<74)
        valid=inside&(distance[np.clip(pix[:,:,1],0,83),np.clip(pix[:,:,0],0,83)]>=self.config['clearance'])
        zero=len(offsets)//2;lo=np.zeros(len(metric));hi=lo.copy()
        for i,row in enumerate(valid):
            if not row[zero]:continue
            left=right=zero
            while left>0 and row[left-1]:left-=1
            while right+1<len(row) and row[right+1]:right+=1
            lo[i],hi[i]=offsets[left],offsets[right]
        # Initial position/tangent and frontier location remain tied to the
        # topological route; transient pose anchoring is handled below.
        lo[:2]=hi[:2]=0.;lo[-1]=hi[-1]=0.
        n=len(metric);d2=np.diff(np.eye(n),n=2,axis=0)
        ax=d2*normal[:,0];ay=d2*normal[:,1]
        h=ax.T@ax+ay.T@ay+.002*np.eye(n)
        gradient=ax.T@(d2@metric[:,0])+ay.T@(d2@metric[:,1])
        step=1/max(np.max(np.sum(abs(h),axis=1)),.001)
        offset=np.zeros(n)
        for _ in range(200):
            offset=np.clip(offset-step*(h@offset+gradient),lo,hi)
        proposed=(metric+normal*offset[:,None])*scale+[42,63]
        # A single normal interval can be safe while its connecting segment
        # clips a boundary. Backtrack the whole update instead of claiming a
        # row/normal box constraint alone certifies clearance.
        for blend in (1.,.75,.5,.25,0.):
            candidate=reference+blend*(proposed-reference)
            dense=candidate[:-1,None,:]*(1-np.linspace(0,1,8)[None,:,None])+candidate[1:,None,:]*np.linspace(0,1,8)[None,:,None]
            x,y=np.rint(dense).astype(int).transpose(2,0,1)
            if np.all(distance[y,x]>=self.config['clearance']):return candidate
        return path.copy()

    def _terrain_masks(self, frame):
        base,obstacles,_=self._free_space(frame)
        road=((frame>=self.config['road_low'])&(frame<=self.config['road_high'])).astype(np.uint8)
        road=cv2.morphologyEx(road,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        road[74:]=0
        # Dividing by the smaller pixel scale overestimates metric distance,
        # so this pixel band never intentionally exceeds2m from visible road.
        outside=cv2.distanceTransform(1-road,cv2.DIST_L2,5)/self.PIXELS_X
        free=(outside<=2.).astype(np.uint8)
        free[74:]=0;free[obstacles>0]=0
        return road,free,obstacles

    def _wheel_grip(self, path, road):
        metric=(np.asarray(path)-[42,63])/[self.PIXELS_X,-self.PIXELS_Y]
        tangent=np.gradient(metric,axis=0) if len(metric)>1 else np.array([[0.,1.]])
        theta=np.arctan2(tangent[:,0],tangent[:,1])
        wheel=np.array([[-1.1,1.6],[1.1,1.6],[-1.1,-1.64],[1.1,-1.64]])
        x=metric[:,0,None]+np.cos(theta[:,None])*wheel[:,0]+np.sin(theta[:,None])*wheel[:,1]
        y=metric[:,1,None]-np.sin(theta[:,None])*wheel[:,0]+np.cos(theta[:,None])*wheel[:,1]
        ix=np.rint(x*self.PIXELS_X+42).astype(int);iy=np.rint(63-y*self.PIXELS_Y).astype(int)
        inside=(ix>=0)&(ix<84)&(iy>=0)&(iy<74)
        onroad=inside&(road[np.clip(iy,0,83),np.clip(ix,0,83)]>0)
        return .6+.4*onroad.astype(float)

    def _route_cost(self, path, road):
        if len(path)<5:return float('inf')
        metric=(path-[42,63])/[self.PIXELS_X,-self.PIXELS_Y]
        arc,k=self._geometry(metric);grip=self._wheel_grip(path,road)
        # Contact is a conservative pixel-center proxy, not authoritative tile
        # coverage. Normal95% qualification must still be observed at finish.
        if np.any(np.sum(grip>.9,axis=1)<1):return float('inf')
        speed=np.clip(np.sqrt(self.config['lateral_accel']*grip.mean(axis=1)/np.maximum(abs(k),.001)),self.config['min_speed'],self.config['max_speed'])
        return float(np.sum(np.diff(arc)/np.maximum((speed[:-1]+speed[1:])*.5,1.)))

    def _plan_terrain(self, frame):
        road,expanded,obstacles=self._terrain_masks(frame)
        normal=self._geodesic(road)
        if len(normal)<5:return normal,expanded,obstacles,road,dict(terrain_selected=False,road_goal=None)
        goal=normal[-1].copy()
        base=road.copy();base[obstacles>0]=0
        normal=self._geodesic(base,goal=goal)
        if len(normal)>=5:normal=self._racing_line(normal,base)
        candidate=self._geodesic(expanded,goal=goal,terrain=road)
        if len(candidate)>=5:candidate=self._racing_line(candidate,expanded)
        base_cost=self._route_cost(normal,road)
        candidate_cost=self._route_cost(candidate,road)
        if len(normal)<1 or np.linalg.norm(normal[-1]-goal)>1.:base_cost=float('inf')
        if len(candidate)<1 or np.linalg.norm(candidate[-1]-goal)>1.:candidate_cost=float('inf')
        use=candidate_cost<base_cost
        path=candidate if use else normal
        meta=dict(terrain_selected=bool(use),road_goal=goal.tolist(),
                  road_time_proxy=base_cost if np.isfinite(base_cost) else None,
                  terrain_time_proxy=candidate_cost if np.isfinite(candidate_cost) else None)
        return path,expanded,obstacles,road,meta

    def _terrain_pedals(self, speed, target, steer, gas_cap, slip, grip):
        if np.all(grip>.99):return self._allocate_pedals(speed,target,steer,gas_cap,slip)
        # Road-only rolling prediction is not trusted while grass contact is
        # inferred. Reserve drive friction on each rear tire and reduce the
        # comfortable braking budget by mean contact friction.
        rear=float(np.min(grip[2:]));mean=float(np.mean(grip));excess=speed-target
        gas=min(float(np.clip((target-speed)*.1+.25,0,1)),gas_cap,.3*rear) if excess<=1 else 0.
        brake=min(float(np.clip(excess*self.config['brake_gain'],.03,.4)),self.config['braking_accel']*mean/303.8957799087198) if excess>1 else 0.
        for _ in range(4):self.internal_throttle+=min(gas-self.internal_throttle,.1)
        self.pedal_diagnostics=dict(allocator_mode='terrain_friction_bound',internal_throttle=float(self.internal_throttle),predicted_next_speed=None)
        return gas,brake

    def act(self, observation):
        obs=np.asarray(observation)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all() or obs.min()<0 or obs.max()>1:
            self.diagnostics={'valid':False}
            return np.array([0,0,.3],np.float32)
        c=self.config;frame=obs[-1].astype(np.float32)
        path,free,obstacles,road,terrain_meta=self._plan_terrain(frame)
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
        path_grip=self._wheel_grip(path,road).mean(axis=1)
        current_grip=self._wheel_grip(np.array([[42.,63.],[42.,62.]]),road)[0]
        caps=np.clip(np.sqrt(c['lateral_accel']*path_grip/np.maximum(abs(kappas),.001)),c['min_speed'],c['max_speed'])
        brake_distance=np.cumsum(np.diff(np.r_[0.,arc])*path_grip)
        target=min(c['max_speed'],float(np.min(np.sqrt(caps*caps+2*c['braking_accel']*brake_distance))),float(np.sqrt(c['lateral_accel']*float(current_grip.mean())/max(abs(curvature),.001))))
        # Visible frontier carries uncertainty: retain enough distance to
        # decelerate to a corner speed just beyond the current image horizon.
        target=min(target,float(np.sqrt(c['min_speed']**2+2*c['braking_accel']*brake_distance[-1])))
        target=max(c['min_speed'],target)
        if not connector_feasible:
            target=min(target,c['min_speed'])
        cap=self._traction_gas_limit(speed,steer)
        gas,brake=self._terrain_pedals(speed,target,steer,cap,slip,current_grip)
        self.last_steer=steer
        self.diagnostics=dict(valid=True,path=path.tolist(),speed=speed,target_speed=target,
                              curvature=curvature,yaw_rate=yaw,wheel_angle=wheel,slip=slip,
                              flow_valid=valid,horizon=float(arc[-1]),gas_cap=cap,connector_feasible=connector_feasible,
                              path_curvature=kappas.tolist(),path_arc=arc.tolist())
        self.diagnostics.update(self.pedal_diagnostics)
        self.diagnostics.update(terrain_meta)
        self.diagnostics.update(current_wheel_grip=current_grip.tolist(),planned_grass_fraction=float(np.mean(path_grip<.99)))
        return np.array([steer,gas,brake],np.float32)

    def last_step_diagnostics(self):
        return dict(self.diagnostics)
