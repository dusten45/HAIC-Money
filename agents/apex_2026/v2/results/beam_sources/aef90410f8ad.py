"""Exact-physics short sequence beam MPC. Pixel observations/action memory only.

Road primitives copied from geodesic r1 archive SHA256
36c7a4cbbb5d930708cf639aea9d55e0273490ee962e1994218d12a140296218.
Internal shadow snapshots are predicted states, never real environment state.
"""
import math
import heapq
import cv2
import numpy as np
from agents.apex_2026.v2.shadow_physics import ShadowCar, decode_wheel_omega


class _Road:
    PIXELS_X, PIXELS_Y = 1.3608, 1.701
    def __init__(self):
        self.config=dict(road_low=.24,road_high=.52,obstacle_margin=0,clearance=3.)

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


class Agent:
    DEPTH=8
    WIDTH=8
    # Dense rectangular footprint includes hull and wheels. No obstacle dilation
    # is applied; checking this footprint would otherwise double the margin.
    FOOTPRINT=np.array([(x,y) for x in (-1.4,-.7,0,.7,1.4)
                        for y in (-2.4,-1.15,.1,1.35,2.6)],float)

    def __init__(self,config=None,*,project_root=None):
        if config:raise ValueError('beam r1 uses fixed architecture defaults')
        self.road=_Road()
        self.shadow=ShadowCar()
        self.reset()

    def reset(self,observation=None):
        self.throttle=0.
        self.last_steer=0.
        self.slip=0.
        self.diagnostics={}

    @staticmethod
    def snapshot(shadow):
        bodies=[]
        for b in [shadow.hull]+shadow.wheels:
            bodies.append((tuple(b.position),b.angle,tuple(b.linearVelocity),b.angularVelocity))
        tires=[(w.omega,w.gas,w.brake,w.steer) for w in shadow.wheels]
        return bodies,tires

    @staticmethod
    def restore(shadow,state):
        bodies,tires=state
        for b,s in zip([shadow.hull]+shadow.wheels,bodies):
            b.position=s[0];b.angle=s[1];b.linearVelocity=s[2];b.angularVelocity=s[3];b.awake=True
        for w,s in zip(shadow.wheels,tires):
            w.omega,w.gas,w.brake,w.steer=s
        shadow.world.ClearForces()

    @staticmethod
    def stage_cost(delta_progress,cross_track,heading_error,speed,target_speed):
        # Progress is accumulated for every executed prefix, so waiting cannot
        # earn the same score as a trajectory that traverses the visible route.
        return (-4.*delta_progress+.08*(2.*cross_track**2+3.*heading_error**2
                 +.025*max(speed-target_speed,0.)**2+.003*max(target_speed-speed,0.)**2))

    @staticmethod
    def _decode_speed(frame):
        mass=float(frame[74:83,10:13].sum())
        return float(np.clip(11.426880974069949*mass-2.8567432419448333,0.,140.))

    @staticmethod
    def _projection(position,reference,arc,previous_progress,speed):
        start=reference[:-1];delta=np.diff(reference,axis=0)
        fraction=np.clip(np.sum((position-start)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-8),0,1)
        projected=start+fraction[:,None]*delta
        distance=np.sum((position-projected)**2,axis=1)
        progress=arc[:-1]+fraction*np.diff(arc)
        valid=(progress>=max(0,previous_progress-3.))&(progress<=previous_progress+.08*speed+4.)
        distance=np.where(valid,distance,np.inf)
        index=int(np.argmin(distance))
        heading=math.atan2(delta[index,0],delta[index,1])
        return float(progress[index]),float(np.sqrt(distance[index])),heading

    def _footprint_cost(self,free,x,y,heading):
        c,s=math.cos(heading),math.sin(heading)
        points=self.FOOTPRINT@np.array([[c,s],[-s,c]])+[x,y]
        xi=np.rint(42+1.3608*points[:,0]).astype(int)
        yi=np.rint(63-1.701*points[:,1]).astype(int)
        inside=(xi>=0)&(xi<84)&(yi>=0)&(yi<74)
        safe=np.zeros(len(xi),bool)
        safe[inside]=free[yi[inside],xi[inside]]>0
        return int(np.count_nonzero(~safe))

    @staticmethod
    def _pursuit(node,reference,arc):
        body=node['state'][0][0]
        position=np.array(body[0]);heading=body[1]
        speed=float(np.linalg.norm(body[2]))
        aim_s=min(arc[-1],node['progress']+float(np.clip(4.+.18*speed,5.,12.)))
        aim=np.array([np.interp(aim_s,arc,reference[:,j]) for j in range(2)])-position
        c,s=math.cos(heading),math.sin(heading)
        lateral=c*aim[0]+s*aim[1]
        return math.atan(3.24*2*lateral/max(float(aim@aim),1.)),speed

    def _advance(self,node,action,reference,arc,targets,free,terminal):
        self.restore(self.shadow,node['state'])
        collision=0
        for _ in range(4):
            x,y,h,v,yaw=self.shadow.step(action)
            collision+=self._footprint_cost(free,x,y,h)
        progress,cross,ref_heading=self._projection(np.array([x,y]),reference,arc,node['progress'],v)
        error=math.atan2(math.sin(-h-ref_heading),math.cos(-h-ref_heading))
        target=float(np.interp(progress,arc,targets))
        cost=node['cost']+self.stage_cost(progress-node['progress'],cross,error,v,target)
        cost+=200.*collision+.03*(action[0]-node['last'])**2
        if terminal:cost+=4.*cross**2+5.*error**2+.03*(v-target)**2
        return dict(cost=cost,progress=progress,state=self.snapshot(self.shadow),
                    sequence=node['sequence']+[action],last=action[0])

    def _completion(self,node,reference,arc,targets,free,depth):
        # Compare every prefix at the SAME full horizon. This feedback rollout
        # is a ranking estimate, not extra accumulated cost on the prefix.
        tail=node
        for future in range(depth+1,self.DEPTH):
            steer,speed=self._pursuit(tail,reference,arc)
            steer=float(np.clip(steer,-.4,.4))
            target=float(np.interp(tail['progress'],arc,targets))
            if speed>target+1.:gas,brake=0.,.65
            elif speed<target-3.:gas,brake=.7,0.
            else:gas,brake=0.,0.
            tail=self._advance(tail,(steer,gas,brake),reference,arc,targets,free,future==self.DEPTH-1)
        return tail['cost']

    def _search(self,reference,arc,targets,free,initial):
        beam=[dict(cost=0.,progress=0.,state=initial,sequence=[],last=self.last_steer)]
        checked=0
        for depth in range(self.DEPTH):
            children=[]
            for node in beam:
                pursuit,_=self._pursuit(node,reference,arc)
                steers=np.unique(np.clip(pursuit+np.array([-.14,0,.14]),-.4,.4))
                for steer in steers:
                    for gas,brake in ((.7,0.),(0.,0.),(0.,.65)):
                        action=(float(steer),gas,brake)
                        child=self._advance(node,action,reference,arc,targets,free,depth==self.DEPTH-1)
                        child['priority']=self._completion(child,reference,arc,targets,free,depth)
                        checked+=4*(self.DEPTH-depth)
                        children.append(child)
            children.sort(key=lambda n:n['priority'])
            beam=children[:self.WIDTH]
        return beam[0],checked

    def act(self,observation):
        obs=np.asarray(observation)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all() or obs.min()<0 or obs.max()>1:
            self.throttle=0.;self.diagnostics=dict(valid=False)
            return np.array([0.,0.,.5],np.float32)
        frame=obs[-1].astype(np.float32)
        free,obstacles,clearance=self.road._free_space(frame)
        path=self.road._geodesic(free)
        if len(path)<5:
            self.throttle=0.;self.diagnostics=dict(valid=False,path=path.tolist())
            return np.array([self.last_steer*.8,0.,.65],np.float32)
        reference=(path-[42.,63.])/[1.3608,-1.701]
        reference=np.vstack(([0.,0.],reference))
        arc,kappas=self.road._geometry(reference)
        targets=np.clip(np.sqrt(100./np.maximum(abs(kappas),.001)),24.,100.)
        # Unknown continuation beyond the visible route retains corner speed.
        targets[-1]=min(targets[-1],24.)
        for i in range(len(targets)-2,-1,-1):
            targets[i]=min(targets[i],math.sqrt(targets[i+1]**2+130.*(arc[i+1]-arc[i])))
        speed=self._decode_speed(frame)
        yaw,wheel=self.road._hud_dynamics(frame)
        _,measured_slip,flow_valid=self.road._motion(obs[-2],obs[-1])
        if flow_valid:self.slip=float(measured_slip)
        self.slip=float(np.clip(self.slip,-.7,.7))
        omegas=decode_wheel_omega(frame)
        ambiguous=bool(np.any(omegas>340) or np.any(frame[82:84,14:24]>.02))
        self.shadow.reset(speed*math.cos(self.slip),-yaw,float(np.clip(-wheel,-.4,.4)),
                          lateral_speed=speed*math.sin(self.slip),throttle=self.throttle,omegas=omegas)
        initial=self.snapshot(self.shadow)
        # Only the current car-occluded footprint is filled; grass elsewhere is
        # never declared drivable. Future footprint samples must remain on road.
        free[59:68,40:45]=1
        best,checked=self._search(reference,arc,targets,free,initial)
        action=np.array(best['sequence'][0],np.float32)
        self.restore(self.shadow,initial)
        self.shadow.step(action,4)
        h=self.shadow.hull.angle;vel=self.shadow.hull.linearVelocity
        lat=math.cos(h)*vel.x+math.sin(h)*vel.y
        forward=-math.sin(h)*vel.x+math.cos(h)*vel.y
        self.slip=math.atan2(lat,max(forward,.1))
        self.throttle=self.shadow.wheels[2].gas
        self.last_steer=float(action[0])
        self.diagnostics=dict(valid=True,path=path.tolist(),speed=speed,yaw_rate=yaw,wheel_angle=wheel,
                              flow_valid=flow_valid,rpm_ambiguous=ambiguous,omegas=omegas.tolist(),
                              beam_sequence=best['sequence'],beam_cost=float(best['cost']),
                              beam_progress=float(best['progress']),collision_checked_raw_ticks=checked,
                              predicted_slip=self.slip,target_speed=float(targets[0]))
        return action

    def last_step_diagnostics(self):
        return dict(self.diagnostics)
