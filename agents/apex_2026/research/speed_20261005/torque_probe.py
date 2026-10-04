"""Privileged uniform-asphalt rear-wheel torque feedback diagnosis.

No timed laps, maps, holdout or legal inference are evaluated here. Raw wheel
rotation substitutes for the HUD to test whether a proposed feedback model
can retain grip before camera quantization is introduced.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import (
    DT, OFFICIAL_FILES, ROOT, digest, make_car, tick,
)


def run_case(speed, steer, mode):
    world, car = make_car(speed)
    rows = []
    gas = 0.
    for i in range(150):
        current = float(np.linalg.norm(car.hull.linearVelocity))
        rear_omega = float(np.mean([w.omega for w in car.wheels[2:]]))
        yaw = abs(float(car.hull.angularVelocity))
        if i % 4 == 0:
            if mode == "instant_budget":
                lateral = current**2*abs(math.tan(steer))/3.24
                gas = float(np.clip(.008*current*math.sqrt(max(0., 1.-(lateral/210.)**2)), 0., 1.))
            else:
                lateral = max(current*yaw, current**2*abs(math.tan(steer))/3.24)
                rear_lateral_force = 7.302*lateral/4.
                available = math.sqrt(max(0., 400.**2-rear_lateral_force**2))
                slip_target = .8*available/82.
                feedforward = .54*82.*slip_target*(abs(rear_omega)+5.)/40000.
                error = current+slip_target-.54*rear_omega
                gas = float(np.clip(feedforward+.04*error, 0., 1.))
        row = tick(world, car, steer, gas, 0.)
        row.update(time_s=(i+1)*DT, gas=gas,
                   rear_rolling_speed_mps=.54*rear_omega)
        rows.append(row)
    return {"initial_speed_mps": speed, "front_target_rad": steer,
            "mode": mode, "control_period_s": .08,
            "max_body_slip_deg": max(abs(x["body_sideslip_deg"]) for x in rows),
            "last_06s_mean_speed_mps": float(np.mean([x["speed_mps"] for x in rows[-30:]])),
            "last_06s_mean_body_slip_deg": float(np.mean([abs(x["body_sideslip_deg"]) for x in rows[-30:]])),
            "sampled_checkpoints": [rows[i] for i in (3, 11, 23, 47, 99, 147)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous measurements")
    hashes = {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    source_hash = digest(__file__)
    rows = [run_case(speed, steer, mode) for speed in (70., 100.)
            for steer in (.03, .06) for mode in ("instant_budget", "wheel_feedback")]
    assert source_hash == digest(__file__)
    assert hashes == {p: digest(ROOT/p) for p in OFFICIAL_FILES}
    report = {"classification": "privileged_uniform_asphalt_torque_feedback_diagnostic",
              "is_legal_camera_candidate": False, "is_lap_benchmark": False,
              "new_timed_laps": 0, "source_sha256": source_hash,
              "helper_sha256": digest(ROOT/"agents/apex_2026/research/speed_20261005/physics_dynamics.py"),
              "official_sha256": hashes, "rows": rows,
              "limits": ["Uniform asphalt only, no edges, obstacles or damage.",
                         "Raw wheel rotation is an oracle; HUD error is not included.",
                         "Rear lateral-force allocation is approximate.",
                         "Three seconds is not long-term stability certification."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
    for row in rows:
        print(json.dumps({k: row[k] for k in ("initial_speed_mps", "front_target_rad", "mode", "max_body_slip_deg", "last_06s_mean_speed_mps")}))


if __name__ == "__main__":
    main()
