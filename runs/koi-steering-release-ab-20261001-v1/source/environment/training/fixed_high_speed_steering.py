"""Registered fixed-target, road-constrained TRAIN steering diagnosis."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from haic_agent.fixed_high_speed_runtime import ARMS, FixedHighSpeedAgent, TARGET_SPEED
from training.env_factory import create_training_environment
from training.evaluate_closed_loop import run_episode


def qualify(row, ticks):
    trace = row.get("decision_trace") or []
    evaluated = trace[25:]
    pixel_fraction = float(np.mean([r["controller"]["pixel_speed"] >= .9 * TARGET_SPEED
                                   for r in evaluated])) if evaluated else 0.0
    physical_fraction = float(np.mean([r["speed"] >= .9 * TARGET_SPEED
                                      for r in evaluated])) if evaluated else 0.0
    outside = [not tick["center_on_road"] for tick in ticks]
    longest = current = 0
    for value in outside:
        current = current + 1 if value else 0
        longest = max(longest, current)
    fraction = float(np.mean(outside)) if outside else 1.0
    road_pass = bool(ticks) and fraction <= .01 and longest <= 8
    speed_pass = bool(evaluated) and min(pixel_fraction, physical_fraction) >= .9
    qualified = (bool(row["completed"]) and row["invalid_actions"] == 0
                 and row["collisions"] == 0 and road_pass and speed_pass)
    return dict(qualified=qualified, road_pass=road_pass, speed_pass=speed_pass,
                pixel_speed_maintenance=pixel_fraction, physical_speed_maintenance=physical_fraction,
                center_offroad_fraction=fraction, max_center_offroad_ticks=longest,
                first_center_offroad_tick=next((i + 1 for i, off in enumerate(outside) if off), None),
                changed_steering=sum(r["controller"]["changed"] for r in trace),
                geometry_available=sum(r["controller"]["available"] for r in trace),
                pixel_physical_speed_mae=float(np.mean([abs(r["controller"]["pixel_speed"] - r["speed"])
                                                        for r in evaluated])) if evaluated else None,
                prefix_hash=hashlib.sha256(json.dumps([[r[k] for k in ("steer", "gas", "brake", "car_x", "car_y")]
                                                      for r in trace[:10]]).encode()).hexdigest())


def run(output):
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    for track in (1, 2, 3):
        for seed in (38200, 38201):
            # Rotate execution order without changing any cell or controller.
            offset = (track + seed) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                ticks = []

                def factory(**kwargs):
                    environment = create_training_environment(**kwargs)
                    original_reset = environment.reset

                    def reset(**reset_kwargs):
                        result = original_reset(**reset_kwargs)
                        raw = environment.unwrapped
                        fixtures = [fixture for tile in raw.road for fixture in tile.fixtures]
                        bounds = np.asarray([[min(v[0] for v in f.shape.vertices),
                                              min(v[1] for v in f.shape.vertices),
                                              max(v[0] for v in f.shape.vertices),
                                              max(v[1] for v in f.shape.vertices)] for f in fixtures])
                        # Road fixture vertices are local. Cache world-space bounding boxes.
                        for i, f in enumerate(fixtures):
                            vertices = [f.body.transform * v for v in f.shape.vertices]
                            bounds[i] = [min(v.x for v in vertices), min(v.y for v in vertices),
                                         max(v.x for v in vertices), max(v.y for v in vertices)]
                        original_step = raw.step

                        def step(action):
                            transition = original_step(action)
                            position = raw.car.hull.position
                            x, y = float(position.x), float(position.y)
                            possible = np.flatnonzero((bounds[:, 0] <= x) & (x <= bounds[:, 2]) &
                                                      (bounds[:, 1] <= y) & (y <= bounds[:, 3]))
                            inside = any(fixtures[int(i)].TestPoint((x, y)) for i in possible)
                            ticks.append(dict(center_on_road=bool(inside),
                                              wheels_on_road=sum(bool(w.tiles) for w in raw.car.wheels),
                                              x=x, y=y, speed=float(raw.car.hull.linearVelocity.length)))
                            return transition

                        raw.step = step
                        return result

                    environment.reset = reset
                    return environment

                creation_started = time.monotonic()
                driver = FixedHighSpeedAgent(arm)
                create_seconds = time.monotonic() - creation_started
                reset_times = []
                original_driver_reset = driver.reset

                def timed_reset(observation):
                    reset_started = time.monotonic()
                    original_driver_reset(observation)
                    reset_times.append(time.monotonic() - reset_started)

                driver.reset = timed_reset
                row = run_episode(mode="fixed_high_speed_" + arm, track_id=track, seed=seed,
                                  agent=driver, max_decisions=1200, plan_budget_seconds=4.5,
                                  capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
                row.update(arm=arm, raw_road_ticks=ticks, qualification=qualify(row, ticks),
                           create_seconds=create_seconds, reset_seconds=reset_times,
                           peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
                path = output.parent / f"{track}-{seed}-{arm}.json"
                with path.open("x") as stream:
                    json.dump(row, stream)
                if row.get("error"):
                    raise RuntimeError(f"Infrastructure/agent error preserved at {path}: {row['error']}")
                rows.append(dict(arm=arm, track=track, seed=seed, evidence=path.name,
                                 completed=row["completed"], lap_time_ms=row["lapTimeMs"],
                                 retire_reason=row.get("retire_reason"), qualification=row["qualification"],
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                print(json.dumps(rows[-1]), flush=True)
    summary = {arm: dict(completed=sum(r["completed"] for r in rows if r["arm"] == arm),
                         qualified=sum(r["qualification"]["qualified"] for r in rows if r["arm"] == arm),
                         denominator=6) for arm in ARMS}
    prefix_valid = all(len({r["qualification"]["prefix_hash"] for r in rows
                           if r["track"] == track and r["seed"] == seed}) == 1
                       for track in (1, 2, 3) for seed in (38200, 38201))
    report = dict(diagnostic_only=True, target_speed=TARGET_SPEED, split="train", rows=rows,
                  summary=summary, common_prefix_valid=prefix_valid,
                  duration_seconds=time.monotonic() - started,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
