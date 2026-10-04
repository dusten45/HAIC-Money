"""Shadow dynamics must match independently executed public mechanics."""
import importlib.util
import numpy as np
import pytest
import Box2D
from core.vendor.car_dynamics import Car


def module():
    spec = importlib.util.find_spec('agents.apex_2026.v2.shadow_physics')
    assert spec is not None, 'pixel-initialized shadow physics is not implemented'
    return importlib.import_module(spec.name)


class Tile:
    road_friction = 1.


def real_car():
    world = Box2D.b2World(gravity=(0, 0))
    car = Car(world, 0, 0, 0)
    tile = Tile()
    for wheel in car.wheels:
        wheel.tiles.add(tile)
    return world, car


def tick(world, car, action):
    car.steer(-action[0]); car.gas(action[1]); car.brake(action[2])
    car.step(.02); world.Step(.02, 180, 60)


def test_straight_public_physics_matches_from_rest():
    shadow = module().ShadowCar()
    world, real = real_car()
    shadow.reset(0, 0, 0)
    for action in [(0, 1, 0)] * 45 + [(0, 0, .4)] * 10:
        shadow.step(action)
        tick(world, real, action)
        assert np.allclose(shadow.hull.position, real.hull.position, atol=2e-5)
        assert np.allclose(shadow.hull.linearVelocity, real.hull.linearVelocity, atol=2e-5)


def test_observable_initialization_reconstructs_rigid_wheels():
    shadow = module().ShadowCar()
    shadow.reset(40, .7, -.2, lateral_speed=3, throttle=.3, omegas=[70,71,80,81])
    assert np.allclose(shadow.hull.linearVelocity, [3,40])
    assert shadow.hull.angularVelocity == pytest.approx(.7)
    for i, wheel in enumerate(shadow.wheels):
        x,y = wheel.position - shadow.hull.worldCenter
        assert np.allclose(wheel.linearVelocity, [3-.7*y,40+.7*x])
        assert wheel.joint.angle == pytest.approx(-.2 if i < 2 else 0)
        assert wheel.omega == [70,71,80,81][i]


def test_motor_and_throttle_slew_and_angle_limit():
    shadow = module().ShadowCar()
    shadow.reset(0,0,0)
    shadow.step((-1,1,0))
    assert shadow.wheels[2].gas == pytest.approx(.1)
    assert shadow.wheels[0].joint.motorSpeed == pytest.approx(3)
    for _ in range(100):
        shadow.step((-1,1,0))
    assert abs(shadow.wheels[0].joint.angle) < .44
    shadow.step((0,0,0))
    assert shadow.wheels[2].gas == 0


def test_rpm_decoder_observes_four_independent_bars():
    from agents.apex_2026.v2.diagnostics.shadow_rpm import render
    decode = getattr(module(), 'decode_wheel_omega', None)
    assert decode is not None, 'four-wheel public RPM decoder is missing'
    for truth in ([40,70,120,160], [150,100,60,200], [0,0,0,0]):
        assert np.allclose(decode(render(truth)), truth, atol=10)


def test_rpm_decoder_ignores_scene_above_hud():
    from agents.apex_2026.v2.diagnostics.shadow_rpm import render
    frame=render([40,70,120,160])
    baseline=module().decode_wheel_omega(frame)
    frame[:74]=.7
    assert np.allclose(module().decode_wheel_omega(frame),baseline,atol=1e-8)
