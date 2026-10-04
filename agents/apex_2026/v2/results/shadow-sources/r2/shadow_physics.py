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
