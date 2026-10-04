"""PRIVILEGED DIAGNOSTIC: official car response on ideal uniform asphalt.

This reads body/wheel state and supplies uniform asphalt contacts in an empty
Box2D world. It is neither a legal camera Agent nor a lap/coverage benchmark.
No official method, source file, road, seed, or competition episode is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import Box2D
import numpy as np

from core.vendor.car_dynamics import Car, FRICTION_LIMIT

ROOT = Path(__file__).resolve().parents[4]
DT = .02
OFFICIAL_FILES = ("core/vendor/car_dynamics.py", "core/vendor/car_racing.py",
                  "env_wrapper.py", "damage.py", "core/track_variables.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Asphalt:
    road_friction = 1.0


def make_car(speed):
    world = Box2D.b2World(gravity=(0, 0))
    car = Car(world, 0, 0, 0)
    tile = Asphalt()
    for body in [car.hull, *car.wheels]:
        body.linearVelocity = (0, speed)
        body.angularVelocity = 0
    for wheel in car.wheels:
        wheel.tiles = {tile}
        wheel.omega = speed / wheel.wheel_rad
    return world, car


def sample(car, previous_velocity):
    velocity = np.asarray(car.hull.linearVelocity, float)
    speed = float(np.linalg.norm(velocity))
    forward = np.asarray(car.hull.GetWorldVector((0, 1)), float)
    side = np.asarray(car.hull.GetWorldVector((1, 0)), float)
    body_slip = math.atan2(float(velocity @ side), float(velocity @ forward))
    old_speed = float(np.linalg.norm(previous_velocity))
    cross = previous_velocity[0] * velocity[1] - previous_velocity[1] * velocity[0]
    trajectory_yaw = math.atan2(cross, float(previous_velocity @ velocity)) / DT
    tire_slips, saturation = [], []
    for wheel in car.wheels:
        wheel_v = np.asarray(wheel.linearVelocity, float)
        wheel_side = np.asarray(wheel.GetWorldVector((1, 0)), float)
        wheel_forward = np.asarray(wheel.GetWorldVector((0, 1)), float)
        lateral = float(wheel_v @ wheel_side)
        longitudinal = float(wheel_v @ wheel_forward) - wheel.omega * wheel.wheel_rad
        tire_slips.append(lateral)
        saturation.append(math.hypot(lateral, longitudinal) * 82 / FRICTION_LIMIT)
    return {"speed_mps": speed, "body_yaw_radps": float(car.hull.angularVelocity),
            "trajectory_yaw_radps": trajectory_yaw,
            "trajectory_curvature_per_m": trajectory_yaw / max(.01, .5 * (speed + old_speed)),
            "lateral_accel_mps2": trajectory_yaw * .5 * (speed + old_speed),
            "longitudinal_accel_mps2": (speed - old_speed) / DT,
            "body_sideslip_deg": math.degrees(body_slip),
            "front_joint_angle_rad": .5 * sum(w.joint.angle for w in car.wheels[:2]),
            "front_tire_slip_mps": float(np.mean(tire_slips[:2])),
            "rear_tire_slip_mps": float(np.mean(tire_slips[2:])),
            "tire_unsaturated_demand_ratio_max": max(saturation)}


def tick(world, car, steer, gas, brake):
    # Official input mapping uses -action[0] as the front-wheel target.
    previous = np.asarray(car.hull.linearVelocity, float)
    car.steer(steer)
    car.gas(gas)
    car.brake(brake)
    car.step(DT)
    world.Step(DT, 6 * 30, 2 * 30)
    return sample(car, previous)


def response(initial_speed, steer, throttle, seconds=3.):
    world, car = make_car(initial_speed)
    samples, integral = [], 0.
    reached = None
    for i in range(round(seconds / DT)):
        if throttle == "privileged_speed_servo":
            speed = float(np.linalg.norm(car.hull.linearVelocity))
            error = initial_speed - speed
            integral = float(np.clip(integral + .10 * error * DT, 0, 1))
            gas = float(np.clip(.2 * error + integral, 0, 1))
        else:
            gas = float(throttle)
        row = tick(world, car, steer, gas, 0.)
        row.update(time_s=(i + 1) * DT, gas=gas)
        samples.append(row)
        if reached is None and abs(row["front_joint_angle_rad"] - np.clip(steer, -.4, .4)) < .005:
            reached = (i + 1) * DT
    window = samples[-30:]
    keys = [k for k in window[0] if k != "time_s"]
    means = {k: float(np.mean([row[k] for row in window])) for k in keys}
    kinematic = math.tan(float(np.clip(steer, -.4, .4))) / 3.24
    return {"initial_speed_mps": initial_speed, "front_target_rad": steer,
            "equivalent_action_steer": -steer, "throttle_mode": throttle,
            "duration_s": seconds, "joint_005rad_settle_s": reached,
            "kinematic_curvature_per_m": kinematic,
            "measured_vs_kinematic_curvature_ratio": means["trajectory_curvature_per_m"] / kinematic if steer else None,
            "last_06s_means": means,
            "transient_per_action": [samples[i] for i in range(3, len(samples), 4)]}


def braking(initial_speed, steer, brake, seconds=.8):
    world, car = make_car(initial_speed)
    distance = 0.
    previous_xy = np.asarray(car.hull.position, float)
    samples, thresholds = [], {}
    for i in range(round(seconds / DT)):
        row = tick(world, car, steer, 0., brake)
        xy = np.asarray(car.hull.position, float)
        distance += float(np.linalg.norm(xy - previous_xy))
        previous_xy = xy
        row.update(time_s=(i + 1) * DT, distance_m=distance)
        samples.append(row)
        for target in (75., 50., 25.):
            if row["speed_mps"] <= target and str(target) not in thresholds:
                thresholds[str(target)] = {"time_s": row["time_s"], "distance_m": distance}
    return {"initial_speed_mps": initial_speed, "front_target_rad": steer,
            "brake": brake, "duration_s": seconds, "thresholds_mps": thresholds,
            "first_02s_mean_decel_mps2": -float(np.mean([s["longitudinal_accel_mps2"] for s in samples[:10]])),
            "first_02s_mean_lateral_accel_mps2": float(np.mean([s["lateral_accel_mps2"] for s in samples[:10]])),
            "final_speed_mps": samples[-1]["speed_mps"],
            "transient_per_action": [samples[i] for i in range(3, len(samples), 4)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior measurements")
    source_hash = digest(__file__)
    before = {p: digest(ROOT / p) for p in OFFICIAL_FILES}
    world, car = make_car(0)
    mass = sum(body.mass for body in [car.hull, *car.wheels])
    constants = {"total_mass_kg": mass, "hull_inertia_kg_m2": car.hull.inertia,
                 "wheel_mass_kg": car.wheels[0].mass, "dt_s": DT,
                 "per_wheel_friction_N": FRICTION_LIMIT,
                 "theoretical_total_accel_mps2": 4 * FRICTION_LIMIT / mass,
                 "wheelbase_m": 3.24, "joint_target_limit_rad": .4,
                 "joint_motor_speed_limit_radps": 3., "tire_stiffness_N_per_mps": 82.}
    del world, car
    if args.smoke:
        rows = [response(100., 0., 0.), response(85., .10, 0.), response(85., -.10, 0.)]
        assert abs(rows[0]["last_06s_means"]["speed_mps"] - 100) < .01
        positive, negative = (r["last_06s_means"] for r in rows[1:])
        assert abs(positive["speed_mps"] - negative["speed_mps"]) < .05
        assert abs(positive["trajectory_curvature_per_m"] + negative["trajectory_curvature_per_m"]) < 1e-4
        brakes = [braking(100., 0., .1), braking(100., 0., .9)]
    else:
        rows = [response(speed, steer, gas) for speed in (50., 70., 85., 100.)
                for steer in (0., .03, .06, .10, .15, .20, .30, .40)
                for gas in (0., .16, 1., "privileged_speed_servo")]
        brakes = [braking(100., steer, brake) for steer in (0., .1, .2)
                  for brake in (.05, .1, .2, .35, .5, .9, 1.)]
    assert source_hash == digest(__file__)
    assert before == {p: digest(ROOT / p) for p in OFFICIAL_FILES}
    report = {"classification": "privileged_uniform_asphalt_physics_diagnostic",
              "is_legal_camera_candidate": False, "is_lap_benchmark": False,
              "source_sha256": source_hash, "official_file_sha256": before,
              "python_version": sys.version, "numpy_version": np.__version__,
              "box2d_version": Box2D.__version__, "constants": constants,
              "limitations": ["Uniform synthetic asphalt contact; no real road edges or obstacles.",
                              "Uses raw vehicle state unavailable to legal camera inference.",
                              "Body positions obey official Box2D maximum translation.",
                              "Fixed-throttle rows change actual speed; initial speed is not a maintained speed.",
                              "Speed servo is a privileged diagnostic and may fail to hold strong turns."],
              "steering_rows": rows, "braking_rows": brakes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
    print(json.dumps({"output": str(args.output), "source_sha256": source_hash,
                      "steering_rows": len(rows), "braking_rows": len(brakes),
                      "constants": constants}, indent=2))


if __name__ == "__main__":
    main()
