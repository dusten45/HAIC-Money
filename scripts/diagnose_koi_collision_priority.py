"""Zero-reset diagnosis of the 47 consumed KOI TRAIN pairs, not policy evaluation.

Only the named generalization archive is read. No environment/runtime is imported.
Run: python -B -m scripts.diagnose_koi_collision_priority
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from haic.algorithms.koi.steering_terms import reconstruct_steering, near_scaled_terms


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/koi-steering-generalization-v1"
OUTPUT = ROOT / "experiments/koi-collision-priority-diagnosis-v1.json"
ARMS = ("crossing_projection", "steering_release_v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def association(obj, state, obstacles):
    if obj is None:
        return None
    matches = []
    c, s = math.cos(state["yaw"]), math.sin(state["yaw"])
    zoom = .6 * max(1 - state["t"], 0) + 16.2 * min(state["t"], 1)
    for index, obstacle in enumerate(obstacles):
        dx, dy = obstacle["x"] - state["x"], obstacle["y"] - state["y"]
        x = 84 * (500 + zoom * (dx * c + dy * s)) / 1000 - .0625
        y = 84 * (799 - (200 + zoom * (-dx * s + dy * c))) / 800 - .0625
        if math.hypot(x - obj[1], y - obj[0]) <= 4:
            matches.append(index)
    return matches[0] if len(matches) == 1 else None


def nominal_projection(steer, obj):
    # Same nominal kinematic constants as frozen release_safe; NOT simulator replay.
    curvature = math.tan(float(np.clip(steer, -.4, .4))) / 3.24
    forward = (63. - obj[0]) / 1.701
    if forward < 0 or abs(curvature * forward) >= 1:
        return None
    lateral = (1 - math.sqrt(1 - (curvature * forward)**2)) / curvature if abs(curvature) > 1e-10 else 0.
    x = 42 + lateral * 1.3608
    return dict(road_only_center_x_px=x, obstacle_center_x_px=obj[1],
                center_gap_px=abs(x - obj[1]), six_px_band_overlap=abs(x - obj[1]) < 6,
                kinematic_curvature_per_m=curvature, forward_m=forward)


def box_sweep(steer, box, speed):
    if box is None or speed <= 0:
        return None
    x, y, w, h = box
    corners = np.asarray([[(a - 42) / 1.3608, (63 - b) / 1.701]
                          for a in (x - 1, x + w) for b in (y - 1, y + h)])
    distance = np.linspace(0., speed * .48, max(2, math.ceil(speed * .48 / .2) + 1))
    curvature = math.tan(float(np.clip(steer, -.4, .4))) / 3.24
    angle = curvature * distance
    c, s = np.cos(angle), np.sin(angle)
    px = (1 - c) / curvature if abs(curvature) > 1e-10 else np.zeros_like(distance)
    py = s / curvature if abs(curvature) > 1e-10 else distance
    dx, dy = corners[:, 0, None] - px, corners[:, 1, None] - py
    u, v = dx * c - dy * s, dx * s + dy * c
    half_width = 1.1 + .28 * math.cos(.4) + .54 * math.sin(.4) + .01
    overlap = ((u.min(axis=0) <= half_width) & (u.max(axis=0) >= -half_width)
               & (v.min(axis=0) <= 2.61) & (v.max(axis=0) >= -2.41))
    indices = np.flatnonzero(overlap)
    first = None if not len(indices) else float(distance[indices[0]] / speed)
    return dict(box_xywh=box, first_overlap_s=first, overlap_within_0_32s=first is not None and first <= .32,
                overlap_within_0_48s=first is not None,
                method="nominal instantaneous constant-command bicycle; padded detected box AABB in moving car frame vs enclosing car rectangle; conservative proxy, not dynamics")


def main():
    report_path = RUN / "episode-report.json"
    report = json.loads(report_path.read_text())
    protocol = json.loads((RUN / "protocol.json").read_text())
    assert sha(RUN / "protocol.json") == report["protocol_sha256"]
    helper = ROOT / "haic/algorithms/koi/steering_terms.py"
    assert sha(helper) == sha(RUN / "source/project/haic/algorithms/koi/steering_terms.py")
    completed = {(r["track_id"], r["seed"], r["mode"]): r for r in report["rows"] if r["status"] == "completed"}
    cells = sorted({(t, s) for t, s, _ in completed if all((t, s, a) in completed for a in ARMS)})
    assert len(cells) == 47
    assert all(t in (1, 2, 3) and 3184000001 <= s <= 3184000016 for t, s in cells)
    inventory, summaries, rows, hits = {}, {}, [], []
    maximum_residual = 0.
    for track, seed in cells:
        for arm in ARMS:
            receipt = completed[track, seed, arm]
            path = RUN / receipt["file"]
            assert path.parent == RUN and sha(path) == receipt["sha256"]
            episode = json.loads(path.read_text())
            raw_path = RUN / episode["raw_trace_file"]
            assert raw_path.parent == RUN and sha(raw_path) == episode["raw_trace_sha256"]
            inventory[str(path.relative_to(ROOT))] = receipt["sha256"]
            inventory[str(raw_path.relative_to(ROOT))] = episode["raw_trace_sha256"]
            states = [episode["initial_state"], *[json.loads(line) for line in raw_path.read_text().splitlines()]]
            obstacles = episode["catalog"]["obstacles"]
            road = np.asarray(episode["catalog"]["track"], dtype=float)
            xy = road[:, 2:4]
            edges = np.roll(xy, -1, axis=0) - xy
            lengths = np.linalg.norm(edges, axis=1)
            angles = np.arctan2(-edges[:, 0], edges[:, 1])
            object_outcomes = []
            for i, obstacle in enumerate(obstacles):
                touched = [s for s in states if obstacle["id"] in s["contacts"] or s["clearance"][i] <= 0]
                outcome = dict(obstacle_id=obstacle["id"], minimum_clearance_m=min(s["clearance"][i] for s in states),
                               hit=bool(touched), first_hit_t=touched[0]["t"] if touched else None)
                object_outcomes.append(outcome)
                if touched:
                    hits.append(dict(track_id=track, seed=seed, arm=arm, **outcome))
            summary = dict(completed=episode["completed"], retire_reason=episode["retire_reason"],
                           damage=episode["damage"], collisions=episode["collisions"],
                           decisions=len(episode["decision_trace"]), raw_ticks=len(states) - 1,
                           objects=object_outcomes)
            summaries[f"{track}:{seed}:{arm}"] = summary
            for d in episode["decision_trace"]:
                c = d["controller"]
                pre, post = c["evaluation_only"]["pre"], c["evaluation_only"]["post"]
                original = reconstruct_steering(c, c.get("baseline_steer", d["steer"]), state={"steps": d["step"]})
                terms = near_scaled_terms(original, c.get("steering_release_alpha", 1.), final_steer=d["steer"])
                assert terms is not None and terms["reconstruction_valid"] and terms["residual_valid"]
                maximum_residual = max(maximum_residual, abs(terms["residual"]), abs(terms["final_sum_residual"]))
                obj = c.get("near_object")
                if obj is None:
                    continue
                index = association(obj, pre, obstacles)
                if index is None:
                    continue
                raw = terms["raw_terms"]
                p, l, damping = (raw[k] for k in ("selected_road_position", "selected_lookahead", "damping"))
                avoid = terms["effective_avoidance_raw"]
                nominal = d["steer"] - terms["avoidance_component"]
                projection = nominal_projection(nominal, obj)
                ri = pre["road_index"]
                lo, hi = (ri - 2) % len(road), (ri + 2) % len(road)
                turn = (angles[hi] - angles[lo] + math.pi) % (2 * math.pi) - math.pi
                distance = sum(lengths[(ri + offset) % len(road)] for offset in range(-2, 2))
                curvature = turn / distance
                centers = {int(k): v for k, v in c["road_centers"].items()}
                curve_px = (centers[30] - 2 * centers[42] + centers[54]) if all(k in centers for k in (30, 42, 54)) else None
                hold = [s for s in states if pre["t"] <= s["t"] <= post["t"]]
                outcome = object_outcomes[index]
                row = dict(track_id=track, seed=seed, arm=arm, decision=d["step"], obstacle_id=obstacles[index]["id"],
                           episode_file=str(path.relative_to(ROOT)), t=pre["t"],
                           completed=episode["completed"], retire_reason=episode["retire_reason"],
                           road_position=p, lookahead=l, damping=damping, road_raw=p+l+damping,
                           road_only_pipeline_steer=nominal, effective_obstacle_raw=avoid,
                           obstacle_afterclip_component=terms["avoidance_component"], actual_steer=d["steer"],
                           gas=d["gas"], brake=d["brake"], active_path=terms["active_path"],
                           road_obstacle_opposed=nominal * avoid < 0,
                           net_steer_opposes_obstacle=d["steer"] * avoid < 0,
                           near_object=obj, selected_flank=c["actual_obstacle_side"],
                           motion_projected_obstacle_x=c.get("projected_obstacle_x"), contact_active=c["contact_active"],
                           nominal_projection=projection, road_curvature_per_m=curvature,
                           nominal_detected_box_sweep=box_sweep(nominal, c.get("steering_release_box"), c.get("pixel_speed", 0.)),
                           road_turn_over_four_segments_rad=turn, pixel_road_second_difference_px=curve_px,
                           heading_error_rad=pre["heading_error"], lateral_m=pre["lateral"], speed_m_s=pre["speed"],
                           release_reason=c.get("steering_release_reason"), release_changed=c.get("steering_release_changed", False),
                           pre_clearance_m=pre["clearance"][index],
                           hold_minimum_clearance_m=min(s["clearance"][index] for s in hold),
                           hold_contact=any(obstacles[index]["id"] in s["contacts"] for s in hold),
                           object_outcome=outcome,
                           time_until_first_hit_s=None if not outcome["hit"] else outcome["first_hit_t"]-pre["t"])
                rows.append(row)
    eligible = [r for r in rows if r["road_obstacle_opposed"] and r["nominal_projection"] and r["nominal_projection"]["six_px_band_overlap"]]
    corner = [r for r in eligible if abs(r["road_curvature_per_m"]) >= .02]
    box_corner = [r for r in rows if r["road_obstacle_opposed"] and abs(r["road_curvature_per_m"]) >= .02
                  and r["nominal_detected_box_sweep"] and r["nominal_detected_box_sweep"]["overlap_within_0_48s"]]
    def ranked(items):
        return sorted(items, key=lambda r: (not (r["object_outcome"]["hit"] and 0 <= (r["time_until_first_hit_s"] or -1) <= 1),
                                            not r["net_steer_opposes_obstacle"], -abs(r["road_curvature_per_m"])))
    result = dict(schema="koi-collision-priority-diagnosis-v1", scope="consumed TRAIN archive only; no old24 reruns or reads",
                  matched_pairs=len(cells), geometry_seeds=len({s for _, s in cells}),
                  environment_resets=0, protected_observations_read=0, official_track4_geometry_read=False,
                  model_hashes=protocol["model_hashes"], episode_report_sha256=sha(report_path),
                  diagnostic_source_sha256=sha(Path(__file__)), steering_terms_sha256=sha(helper),
                  inventory=inventory, maximum_source_reconstruction_residual=maximum_residual,
                  definitions={
                      "road_only": "exact current source steering pipeline with effective obstacle term zeroed; same observed state, not counterfactual trajectory",
                      "nominal_overlap": "constant road-only bicycle center arc at detected centroid forward coordinate lies within unchanged +/-6px band; NOT measured bbox or swept-hull collision",
                      "curvature": "signed tangent change across four catalog road segments / their summed lengths; >=.02/m is descriptive corner stratification, not tuned runtime threshold",
                      "association": "unique <=4px physical rendered-centroid match, never nearest fallback",
                      "hit": "raw contact-list membership OR raw full-fixture signed clearance <=0, whole-episode all six objects",
                      "causality": "command accounting is source-exact; nominal arc is kinematic proxy; observed clearances are actual policy only; post-divergence B/C is not same-state causal comparison",
                  },
                  counts_by_arm={a: dict(decisions=sum(s["decisions"] for k,s in summaries.items() if k.endswith(":"+a)),
                                        associated_object_decisions=sum(r["arm"]==a for r in rows),
                                        opposed_nominal_overlap_decisions=sum(r["arm"]==a for r in eligible),
                                        corner_opposed_nominal_overlap_decisions=sum(r["arm"]==a for r in corner),
                                        corner_overlap_net_road_dominant=sum(r["arm"]==a and r["net_steer_opposes_obstacle"] for r in corner),
                                        archived_box_corner_opposed_swept_overlap=sum(r["arm"]==a for r in box_corner),
                                        hit_objects=sum(r["arm"]==a for r in hits)) for a in ARMS},
                  hit_objects=hits, top_corner_conflicts=ranked(corner)[:24],
                  archived_box_corner_conflicts=ranked(box_corner),
                  all_opposed_overlap_decisions=ranked(eligible),
                  collision_precursors=[r for r in rows if r["object_outcome"]["hit"] and 0 <= r["time_until_first_hit_s"] <= .6],
                  clean_corner_negatives=ranked([r for r in corner if not r["object_outcome"]["hit"]])[:12],
                  episode_summaries=summaries,
                  limitations=["Detected bounding boxes/pixel images are not archived at every near decision; centroid six-pixel overlap is not object-bound intersection.",
                               "Steering actuator lag, slip, moving viewpoint and future road curvature are absent from nominal projection.",
                               "No action intervention or reset; stronger avoidance and immediate recovery remain untested hypotheses.",
                               "High curvature plus steering opposition is also present in clean finishes; it is not sufficient for collision prediction."])
    with OUTPUT.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("matched_pairs", "geometry_seeds", "maximum_source_reconstruction_residual", "counts_by_arm", "hit_objects")}, indent=2))
    for row in ranked(box_corner)[:8]:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
