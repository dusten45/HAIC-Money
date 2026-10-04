"""Official mandatory-track geometry and honest finish-time bound audit.

This diagnostic performs only each map's required reset/warmup. It sends no
Agent actions or timed driving steps and opens no development or holdout cells.
The wheel-reach calculation assumes Box2D's two-meter velocity-integration
limit also bounds total per-step body movement. Position/joint/contact solver
corrections can move bodies afterward, so this assumption is NOT certified.
The unconditional lower bound instead comes from the finish tracker state
machine, and is too weak to decide whether a ten-second lap is possible.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
from pathlib import Path

import Box2D
import numpy as np

from agents.apex_2026.research.speed_20261005.physics_dynamics import ROOT, digest
from core.finish_line import LAP_QUALIFICATION_RATIO
from core.vendor.car_dynamics import SIZE, WHEEL_R, WHEEL_W
from core.vendor.car_racing import FPS, TRACK_DETAIL_STEP, TRACK_WIDTH
from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


MANDATORY = ((1, 516237), (2, 644062), (3, 1007), (4, 18800))
SOURCES = (
    "core/vendor/car_dynamics.py", "core/vendor/car_racing.py",
    "core/finish_line.py", "core/track_variables.py", "env_wrapper.py",
    "local_simulator/environment.py", "local_simulator/schema.py",
)
WHEEL_FIXTURE_RADIUS_M = math.hypot(WHEEL_W * SIZE, WHEEL_R * SIZE)


def point_to_polygon_distance(point, vertices):
    """Exact Euclidean point-to-convex-quad distance in the world plane."""
    quad = np.asarray(vertices, dtype=float)
    edges = np.roll(quad, -1, axis=0) - quad
    offsets = np.asarray(point, dtype=float) - quad
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    if np.all(cross >= -1e-9) or np.all(cross <= 1e-9):
        return 0.
    fraction = np.clip(np.sum(offsets * edges, axis=1) /
                       np.sum(edges * edges, axis=1), 0., 1.)
    nearest = quad + fraction[:, None] * edges
    return float(np.min(np.linalg.norm(nearest - point, axis=1)))


def sample_cell(track, seed):
    spec = MapSpec(track, seed, "official", (), 2000, 4)
    environment, raw = create_environment(spec, render_mode=None)
    try:
        reset_environment(environment, spec)
        n = len(raw.track)
        required = math.ceil(LAP_QUALIFICATION_RATIO * n - 1e-12)
        visited = sum(bool(tile.road_visited) for tile in raw.road)
        assert visited == raw.tile_visited_count
        tracker = raw.finish_line_tracker
        assert tracker is not None and tracker.finish_time_s is None
        centers = np.asarray([(point[2], point[3]) for point in raw.track])
        centerline_length = float(np.linalg.norm(centers - np.roll(centers, 1, axis=0),
                                                 axis=1).sum())
        wheel_starts = np.asarray([tuple(wheel.position) for wheel in raw.car.wheels])
        distances = []
        for tile in raw.road:
            if tile.road_visited:
                continue
            polygon = [tuple(tile.GetWorldPoint(vertex))
                       for vertex in tile.fixtures[0].shape.vertices]
            nearest_center_distance = min(point_to_polygon_distance(point, polygon)
                                          for point in wheel_starts)
            distances.append(max(0., nearest_center_distance - WHEEL_FIXTURE_RADIUS_M))
        need_after_warmup = max(0, required - visited)
        assert need_after_warmup <= len(distances)
        threshold = sorted(distances)[need_after_warmup - 1] if need_after_warmup else 0.
        max_translation = float(Box2D.b2_maxTranslation)
        # Subtract a tiny tolerance before ceil so floating-point rounding
        # cannot make the reported lower bound one physics tick too large.
        conditional_steps = max(0, math.ceil((threshold - 1e-9) / max_translation))
        # With no prior departure, the first future update returns before a
        # crossing can be recognized; the second is the earliest timestamp.
        state_machine_steps = 1 if tracker.departed_start_area else 2
        hub_velocity = tuple(raw.car.hull.linearVelocity)
        return {
            "track_id": track, "seed": seed,
            "tile_count": n, "tiles_required_for_95_percent": required,
            "tiles_visited_during_untimed_warmup": visited,
            "new_tiles_required_after_clock_start": need_after_warmup,
            "clock_start_physics_time_s": float(raw.t),
            "clock_start_hull_speed_mps": float(math.hypot(*hub_velocity)),
            "clock_start_tracker_departed_start_area": bool(tracker.departed_start_area),
            "clock_start_tracker_qualified": tracker.qualified_time_s is not None,
            "clock_start_wheel_positions_m": wheel_starts.tolist(),
            "road_half_width_m": TRACK_WIDTH,
            "finish_half_depth_m": TRACK_DETAIL_STEP * .25,
            "official_obstacles": [{"center_m": [float(v) for v in body.position],
                                    "radius_m": float(body.fixtures[0].shape.radius)}
                                   for body in raw.obstacles],
            "centerline_closed_length_m": centerline_length,
            "centerline_length_divided_by_100_s_estimate_not_bound": centerline_length / 100.,
            "required_new_tile_reach_order_statistic_m": threshold,
            "conditional_velocity_translation_bound_steps": conditional_steps,
            "conditional_velocity_translation_bound_s": conditional_steps / FPS,
            "unconditional_tracker_state_lower_bound_steps": state_machine_steps,
            "unconditional_tracker_state_lower_bound_s": state_machine_steps / FPS,
        }
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; preserve earlier measurement")
    own_hash = digest(__file__)
    source_hashes = {path: digest(ROOT / path) for path in SOURCES}
    cells = [sample_cell(track, seed) for track, seed in MANDATORY]
    assert own_hash == digest(__file__)
    assert source_hashes == {path: digest(ROOT / path) for path in SOURCES}
    report = {
        "classification": "privileged_official_reset_geometry_lower_bound_diagnostic",
        "is_agent_trial": False, "is_full_episode": False,
        "is_holdout": False, "cells": cells,
        "source_sha256": own_hash, "official_and_wrapper_sha256": source_hashes,
        "box2d_distribution_version": metadata.version("box2d-py"),
        "box2d_runtime_version": Box2D.__version__,
        "box2d_max_velocity_integrated_translation_per_step_m": float(Box2D.b2_maxTranslation),
        "physics_step_s": 1. / FPS,
        "wheel_fixture_circumscribed_radius_m": WHEEL_FIXTURE_RADIUS_M,
        "unconditional_proof": (
            "At reset/warmup completion the finish tracker has not departed "
            "the start area on these four maps. Its first future update exits "
            "before recognizing a crossing. Its earliest possible finish-time "
            "timestamp is therefore the second future .02s physics update: "
            "0.04s. This does not use a speed or route assumption."
        ),
        "conditional_wheel_reach_calculation": (
            "IF every wheel-body center moves at most Box2D.b2_maxTranslation "
            "(2m) in total per physics update, the kth-smallest distance from "
            "a warmup wheel center to an unvisited tile polygon, minus the "
            "wheel fixture's circumscribed radius, yields a coverage-time "
            "bound for k=ceil(.95N)-warmup_visits. Box2D caps velocity-integrated "
            "translation, but joint/contact position corrections occur later; "
            "no upper bound on their total effect was established here. Thus "
            "the calculated 2.8–3.46s figures are conditional, not certified "
            "physical minimum times."
        ),
        "limits": [
            "Only four reset/warmups; no Agent act, timed episode, development trial or holdout.",
            "The conditional calculation lets all four wheels independently reach tiles through anything, so it is intentionally weak even if its assumption holds.",
            "Centerline distance divided by 100 is an illustrative route estimate, not a lower bound: legal paths may cut corners and need only 95% tile coverage.",
            "Road and obstacles can delay an optimal route, but no additional minimum time is certified here.",
            "Warmup time is excluded from official lap_time_ms; its tile visits count toward qualification.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    for row in cells:
        print(json.dumps({key: row[key] for key in (
            "track_id", "seed", "tile_count", "tiles_visited_during_untimed_warmup",
            "centerline_closed_length_m", "conditional_velocity_translation_bound_s",
            "unconditional_tracker_state_lower_bound_s"
        )}))


if __name__ == "__main__":
    main()
