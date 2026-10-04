"""Internal public-mechanics predictor, initialized from estimated observations only.

Adapted from core/vendor/car_dynamics.py; rendering and telemetry access omitted.
Prediction world contains only the reconstructed car; no real environment state.

MIT License

Copyright (c) 2026 2026-HAIC

Portions derived from Gymnasium:
Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import numpy as np
import Box2D
from Box2D.b2 import fixtureDef, polygonShape, revoluteJointDef

SIZE = 0.02
ENGINE_POWER = 100000000 * SIZE * SIZE
WHEEL_MOMENT_OF_INERTIA = 4000 * SIZE * SIZE
FRICTION_LIMIT = (
    1000000 * SIZE * SIZE
)
WHEEL_R = 27
WHEEL_W = 14
WHEELPOS = [(-55, +80), (+55, +80), (-55, -82), (+55, -82)]
HULL_POLY1 = [(-60, +130), (+60, +130), (+60, +110), (-60, +110)]
HULL_POLY2 = [(-15, +120), (+15, +120), (+20, +20), (-20, 20)]
HULL_POLY3 = [
    (+25, +20),
    (+50, -10),
    (+50, -40),
    (+20, -90),
    (-20, -90),
    (-50, -40),
    (-50, -10),
    (-25, +20),
]
HULL_POLY4 = [(-50, -120), (+50, -120), (+50, -90), (-50, -90)]
WHEEL_COLOR = (0, 0, 0)
WHEEL_WHITE = (77, 77, 77)
MUD_COLOR = (102, 102, 0)


class _Mechanics:
    def __init__(self, world, init_angle, init_x, init_y, grass_friction_multiplier=0.6):
        self.world: Box2D.b2World = world
        self.grass_friction_multiplier = grass_friction_multiplier
        self.grip_multiplier = 1.0
        self.engine_multiplier = 1.0
        self.steering_multiplier = 1.0
        self.hull: Box2D.b2Body = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            angle=init_angle,
            fixtures=[
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY1]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY2]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY3]
                    ),
                    density=1.0,
                ),
                fixtureDef(
                    shape=polygonShape(
                        vertices=[(x * SIZE, y * SIZE) for x, y in HULL_POLY4]
                    ),
                    density=1.0,
                ),
            ],
        )
        self.hull.color = (0.8, 0.0, 0.0)
        self.wheels = []
        self.fuel_spent = 0.0
        WHEEL_POLY = [
            (-WHEEL_W, +WHEEL_R),
            (+WHEEL_W, +WHEEL_R),
            (+WHEEL_W, -WHEEL_R),
            (-WHEEL_W, -WHEEL_R),
        ]
        for wx, wy in WHEELPOS:
            front_k = 1.0 if wy > 0 else 1.0
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE),
                angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=[
                            (x * front_k * SIZE, y * front_k * SIZE)
                            for x, y in WHEEL_POLY
                        ]
                    ),
                    density=0.1,
                    categoryBits=0x0020,
                    maskBits=0x001,
                    restitution=0.0,
                ),
            )
            w.wheel_rad = front_k * WHEEL_R * SIZE
            w.color = WHEEL_COLOR
            w.gas = 0.0
            w.brake = 0.0
            w.steer = 0.0
            w.phase = 0.0
            w.omega = 0.0
            w.skid_start = None
            w.skid_particle = None
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=w,
                localAnchorA=(wx * SIZE, wy * SIZE),
                localAnchorB=(0, 0),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=180 * 900 * SIZE * SIZE,
                motorSpeed=0,
                lowerAngle=-0.4,
                upperAngle=+0.4,
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            w.userData = w
            self.wheels.append(w)
        self.drawlist = self.wheels + [self.hull]
        self.particles = []

    def gas(self, gas):
        gas = np.clip(gas, 0, 1)
        for w in self.wheels[2:4]:
            diff = gas - w.gas
            if diff > 0.1:
                diff = 0.1
            w.gas += diff

    def brake(self, b):
        for w in self.wheels:
            w.brake = b

    def steer(self, s):
        self.wheels[0].steer = s
        self.wheels[1].steer = s

    def step(self, dt):
        for w in self.wheels:
            dir = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = (
                dir * min(50.0 * val, 3.0) * self.steering_multiplier
            )

            grass = True
            friction_limit = FRICTION_LIMIT * self.grass_friction_multiplier
            for tile in w.tiles:
                friction_limit = max(
                    friction_limit, FRICTION_LIMIT * tile.road_friction
                )
                grass = False
            friction_limit *= self.grip_multiplier

            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]
            vs = side[0] * v[0] + side[1] * v[1]


            w.omega += (
                dt
                * (ENGINE_POWER * self.engine_multiplier)
                * w.gas
                / WHEEL_MOMENT_OF_INERTIA
                / (abs(w.omega) + 5.0)
            )
            self.fuel_spent += dt * ENGINE_POWER * w.gas

            if w.brake >= 0.9:
                w.omega = 0
            elif w.brake > 0:
                BRAKE_FORCE = 15
                dir = -np.sign(w.omega)
                val = BRAKE_FORCE * w.brake
                if abs(val) > abs(w.omega):
                    val = abs(w.omega)
                w.omega += dir * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad
            f_force = -vf + vr
            p_force = -vs


            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force
                p_force /= force
                force = friction_limit
                f_force *= force
                p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / WHEEL_MOMENT_OF_INERTIA

            w.ApplyForceToCenter(
                (
                    p_force * side[0] + f_force * forw[0],
                    p_force * side[1] + f_force * forw[1],
                ),
                True,
            )



class _Road:
    road_friction = 1.0


class ShadowCar(_Mechanics):
    """Local coordinates: x right, y forward; positive yaw turns left.

    Reset inputs are estimates from public pixels or previous actions. Scalar
    forward/lateral speeds describe hull center-of-mass velocity. RPM is wheel
    angular velocity in rad/s, not revolutions/minute. Omitting RPM assumes
    rolling tires; that assumption is intentionally exposed for diagnostics.
    """
    def __init__(self):
        super().__init__(Box2D.b2World(gravity=(0, 0)), 0, 0, 0)
        self.world.warmStarting = False
        road = _Road()
        for w in self.wheels:
            w.tiles.add(road)

    def reset(self, forward_speed, yaw_rate, wheel_angle, *, lateral_speed=0.,
              throttle=0., omegas=None, friction=1.):
        self.hull.position = (0, 0)
        self.hull.angle = 0
        self.hull.linearVelocity = (float(lateral_speed), float(forward_speed))
        self.hull.angularVelocity = float(yaw_rate)
        self.grip_multiplier = float(friction)
        center = self.hull.worldCenter
        for i, (w, anchor) in enumerate(zip(self.wheels, WHEELPOS)):
            x, y = anchor[0] * SIZE, anchor[1] * SIZE
            w.position = (x, y)
            w.angle = float(wheel_angle) if i < 2 else 0.
            w.linearVelocity = (lateral_speed-yaw_rate*(y-center.y),
                                forward_speed+yaw_rate*(x-center.x))
            w.angularVelocity = float(yaw_rate)
            w.gas = float(throttle) if i >= 2 else 0.
            w.brake = 0.
            w.steer = float(wheel_angle) if i < 2 else 0.
            fw = w.GetWorldVector((0, 1))
            w.omega = (float(omegas[i]) if omegas is not None else
                       (fw.x*w.linearVelocity.x+fw.y*w.linearVelocity.y)/w.wheel_rad)
        self.world.ClearForces()

    def step(self, action, steps=1):
        """Advance exact 20ms public steps. Action steering is right-positive."""
        for _ in range(steps):
            self.steer(-float(action[0]))
            self.gas(float(action[1]))
            self.brake(float(action[2]))
            super().step(.02)
            self.world.Step(.02, 180, 60)
        return (float(self.hull.position.x), float(self.hull.position.y),
                float(self.hull.angle), float(self.hull.linearVelocity.length),
                float(self.hull.angularVelocity))


_RPM_COEFFICIENTS = np.array([[21.152569617136912, 132.7205723146679, 40.577256872053574, -3.8915732080655503], [409.8502510704804, 155.40306218906187, 5.847482644730661, -1.2010219155385045], [5.263221102880834, -342.0675489187128, -27.2886226519503, 3.53973730551796], [-15.22031740651515, 514.2683844475864, 35.55255705310872, -7.328977307556208], [19.468778831046947, 77.27707932268008, -47.33604417449891, 10.256369146374226], [-24.314646046543288, -21.973883856310074, 92.25889004474195, -14.249423214963766], [16.76276824546332, 1.6472272359718474, 212.09328081734725, 10.497468787938889], [4.198395629218766, -15.211312324356314, -27.27101120554686, -19.11109394976468], [28.899671023775568, 2.9292676374208213, 42.51667551832618, 351.2375509642337], [-93.55929957024023, 37.14380159749405, -38.13105523760195, -56.2784593310019], [0.0, 0.0, 0.0, 0.0], [1.34390369764688, 0.7004224456057386, 2.179635438930667, 2.0041029485355537]])

def decode_wheel_omega(frame):
    """Four public positive wheel angular velocities; negative/saturated bars ambiguous."""
    crop = np.asarray(frame)[74:82,14:25]
    if not np.any(crop):
        return np.zeros(4)
    return np.maximum(0., np.r_[crop.sum(axis=0),1.] @ _RPM_COEFFICIENTS)


import cv2

DEFAULTS =  dict(road_low=.24, road_high=.52, obstacle_margin=0,
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
            self.shadow=ShadowCar()
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

    def last_step_diagnostics(self):
            return dict(self.diagnostics)

    @staticmethod
    def _utility(progress,speed,cross_track):
        # Only the actually executed first80ms is rewarded. A tail's endpoint
        # does not make coasting more attractive than a planned emergency stop.
        return float(progress+.035*speed-.3*cross_track*cross_track)

    @staticmethod
    def _choose(utilities,safe,violations):
        safe=np.asarray(safe,bool)
        if np.any(safe):
            return int(np.argmax(np.where(safe,utilities,-np.inf)))
        # When no certificate is feasible, minimize predicted collision energy.
        return int(np.argmin(violations))

    @staticmethod
    def _pursuit(path,x,y,heading,speed):
        delta=path-np.array([x,y])
        ct,st=math.cos(heading),math.sin(heading)
        local=np.c_[ct*delta[:,0]+st*delta[:,1],-st*delta[:,0]+ct*delta[:,1]]
        ahead=np.flatnonzero(local[:,1]>max(3.,min(15.,speed*.16+3.)))
        index=int(ahead[0]) if len(ahead) else int(np.argmax(local[:,1]))
        px,py=local[index]
        return float(np.clip(math.atan2(6.48*px,max(px*px+py*py,4.)),-.4,.4))

    def _certificate(self,action,speed,yaw,wheel,slip,omega,path,field):
        self.shadow.reset(speed*math.cos(slip),-yaw,-wheel,
            lateral_speed=speed*math.sin(slip),throttle=self.internal_throttle,omegas=omega)
        first=None;clearances=[];energy=0.;stopped=False;sample=None
        # All candidates have the same continuation: immediate emergency brake
        # after the executed80ms, using geometry-based steering toward the path.
        for tick in range(48):
            if tick<4:
                control=action
            else:
                x,y,h,v,_=sample
                steer=self._pursuit(path,x,y,h,v)
                control=(steer,0.,.7)
            sample=self.shadow.step(control,1)
            x,y,h,v,rate=sample
            if tick==3:
                first=sample
                first_throttle=float(self.shadow.wheels[2].gas)
                vel=self.shadow.hull.linearVelocity
                next_slip=math.atan2(math.cos(h)*vel.x+math.sin(h)*vel.y,
                                    max(.1,-math.sin(h)*vel.x+math.cos(h)*vel.y))
            # Capsule around the full body; sample every20ms, including early
            # braking transients. Pixels outside the camera are unknown/unsafe.
            offsets=np.array([-1.15,0.,1.3])
            clearance=float(np.min(self._sample_clearance(field,x-offsets*math.sin(h),y+offsets*math.cos(h))))
            uncertainty=.1+speed*self.slip_uncertainty*.12*(1.-math.exp(-.02*(tick+1)/.12))
            penetration=max(0.,1.35+uncertainty-clearance)
            energy+=penetration*penetration*(1.+v*v)
            clearances.append(clearance-1.35-uncertainty)
            if tick>=3 and v<1.5:
                stopped=True
                break
        x,y,h,v,_=first
        distance=np.sum((path-np.array([x,y]))**2,axis=1)
        # Prefix tracking uses a continuous nearest-segment projection; nearest
        # integer path nodes would quantize rewards at low speed.
        starts=path[:-1];segments=np.diff(path,axis=0)
        u=np.clip(np.sum((np.array([x,y])-starts)*segments,axis=1)/np.maximum(np.sum(segments**2,axis=1),1e-6),0.,1.)
        q=starts+u[:,None]*segments
        index=int(np.argmin(np.sum((q-[x,y])**2,axis=1)))
        lengths=np.linalg.norm(segments,axis=1)
        arc=float(np.sum(lengths[:index])+u[index]*lengths[index])
        cross=float(np.linalg.norm(q[index]-[x,y]))
        return dict(first=first,utility=self._utility(arc,v,cross),
            stopped=stopped,safe=stopped and min(clearances)>=0.,
            violation=energy+(0. if stopped else sample[3]**2+100.),
            margin=min(clearances),stop_y=sample[1],
            throttle=first_throttle,next_slip=next_slip)

    def act(self,observation):
        obs=np.asarray(observation,np.float32)
        if obs.shape!=(4,84,84) or not np.isfinite(obs).all():
            return np.array([0.,0.,.7],np.float32)
        frame=obs[-1]
        speed=float(np.clip((frame[77:83,10:13].sum()-.27)/.085,0.,100.))
        yaw,wheel=self._hud_dynamics(frame)
        _,slip,valid=self._motion(obs[-2],frame)
        self._update_slip(speed,slip,valid,yaw)
        if self.prediction:
            self.speed_residual=.9*self.speed_residual+.1*abs(speed-self.prediction['speed'])
            self.yaw_residual=.9*self.yaw_residual+.1*abs(yaw-self.prediction['yaw'])
        path,_,_=self._path(frame);self.previous_path=path
        points=np.c_[(path[:,0]-42.)/self.PIXELS_X,(63.-path[:,1])/self.PIXELS_Y]
        points=np.vstack(([0.,0.],points))
        if len(points)<2:points=np.array([[0.,0.],[0.,2.]])
        field=self._metric_clearance(frame)
        omega=decode_wheel_omega(frame)
        nominal=self._pursuit(points,0.,0.,0.,speed)
        steers=np.unique(np.clip(np.r_[nominal+np.array([-.12,-.04,0.,.04,.12]),wheel,0.],-.4,.4))
        actions=[np.array([s,g,b],np.float32) for s in steers for g,b in ((1.,0.),(.4,0.),(0.,0.),(0.,.35),(0.,.7))]
        certs=[self._certificate(a,speed,yaw,wheel,self.slip_estimate,omega,points,field) for a in actions]
        chosen=self._choose([c['utility'] for c in certs],[c['safe'] for c in certs],[c['violation'] for c in certs])
        c=certs[chosen]; action=actions[chosen]
        self.internal_throttle=c['throttle']
        # Model slip prediction is retained when visual motion is unavailable;
        # uncertainty still grows until a valid visual measurement arrives.
        self.slip_estimate=float(np.clip(c['next_slip'],-.7,.7))
        self.prediction=dict(speed=c['first'][3],yaw=-c['first'][4])
        self.diagnostics=dict(speed=speed,yaw_rate=yaw,wheel=wheel,flow_valid=bool(valid),
            slip=self.slip_estimate,slip_uncertainty=self.slip_uncertainty,
            omega=omega.tolist(),path_nodes=len(path),nominal_steer=nominal,
            candidate_count=len(actions),feasible_count=sum(c['safe'] for c in certs),
            predicted_margin=c['margin'],predicted_violation=c['violation'],
            prefix_utility=c['utility'],prediction_next_speed=self.prediction['speed'],
            prediction_next_yaw=self.prediction['yaw'],speed_residual=self.speed_residual,
            yaw_residual=self.yaw_residual)
        return action
