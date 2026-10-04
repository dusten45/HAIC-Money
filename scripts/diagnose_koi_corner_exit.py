"""Passive corner-exit diagnosis on already-consumed frozen-shield TRAIN logs."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def state(row):
    c = row["controller"]
    p = c["evaluation_only"]["pre"]
    centers = c["road_centers"]
    return {
        "step": row["step"],
        "t": p["t"],
        "station": p["station"],
        "road_index": p["road_index"],
        "duration": c["evaluation_only"]["post"]["t"] - p["t"],
        "steer": row["action"][0],
        "gas": row["action"][1],
        "brake": row["action"][2],
        "target": c["target_speed"],
        "pixel_speed": c["pixel_speed"],
        "speed": p["speed"],
        "heading": p["heading_error"],
        "lateral": p["lateral"],
        "road_centers": centers,
        "pixel_heading": math.atan2(centers["38"] - centers["50"], 12)
        if "38" in centers and "50" in centers else None,
        "onroad": all(v > 0 for v in p["wheel_road_contacts"]),
        "obstacle": bool(c.get("near_object") or c.get("far_active")
                         or c.get("contact_active") or c["shield"]["active"]),
        "recovery": c.get("impact_steps_remaining", 0) > 0
        or c.get("recovery_steps_remaining", 0) > 0,
        "acceleration_target": c.get("acceleration_target"),
        "mechanism_active": c.get("mechanism_active"),
        "arrival_cap": c.get("arrival_cap"),
        "free_distance": c.get("free_distance"),
        "free_samples": c.get("free_samples"),
        "effective_target": c.get("acceleration_target") or c["target_speed"],
    }


def analyze(path):
    rows = [state(json.loads(line)) for line in path.read_text().splitlines()]
    primary_path = path.with_name(path.name.replace(".decisions.jsonl", ".json"))
    primary = json.loads(primary_path.read_text())
    track = primary["catalog"]["track"]
    lengths, headings, stations = [], [], [0.0]
    for a, b in zip(track, track[1:] + track[:1]):
        dx, dy = b[2] - a[2], b[3] - a[3]
        lengths.append(math.hypot(dx, dy))
        headings.append(math.atan2(-dx, dy))
        stations.append(stations[-1] + lengths[-1])
    for r in rows:
        index = r["road_index"]
        traveled = stations[index + 1] - r["station"]
        bends = [0.0]
        j = (index + 1) % len(track)
        while traveled < 28.8 and j != index:
            bends.append(abs((headings[j] - headings[index] + math.pi) % (2 * math.pi) - math.pi))
            traveled += lengths[j]
            j = (j + 1) % len(track)
        r["ahead_28_8m_heading_change_rad"] = max(bends)
        r["sprint_blockers"] = []
        if r["free_samples"] < 4 or r["free_distance"] < max(18, max(r["pixel_speed"], 72) * .4):
            r["sprint_blockers"].append("projected_free_space")
        if r["pixel_speed"] >= 72:
            r["sprint_blockers"].append("speed_at_sprint_cap")
        if abs(r["steer"]) >= .18:
            r["sprint_blockers"].append("steering")
        if ("42" not in r["road_centers"] or "54" not in r["road_centers"]
                or abs(r["road_centers"].get("54", 0) - 42) >= 3):
            r["sprint_blockers"].append("near_road_geometry")
        if r["obstacle"]:
            r["sprint_blockers"].append("obstacle_or_shield")
    episodes = []
    armed = False
    peak_steer = peak_heading = 0.0
    stable = 0
    for i, r in enumerate(rows):
        # Exclude obstacle/recovery actions from the proposed corner mechanism.
        if r["obstacle"] or r["recovery"]:
            armed = False
            stable = 0
            peak_steer = peak_heading = 0.0
            continue
        if abs(r["steer"]) >= 0.25 and abs(r["heading"]) >= 0.15:
            armed = True
            peak_steer = max(peak_steer, abs(r["steer"]))
            peak_heading = max(peak_heading, abs(r["heading"]))
        aligned = (abs(r["steer"]) <= 0.12 and abs(r["heading"]) <= 0.10
                   and r["onroad"])
        stable = stable + 1 if aligned else 0
        if not armed or stable < 2:
            continue
        start = i - 1
        end = start
        while end + 1 < len(rows):
            n = rows[end + 1]
            if (n["obstacle"] or n["recovery"] or not n["onroad"]
                    or abs(n["steer"]) > 0.12 or abs(n["heading"]) > 0.10):
                break
            end += 1
        window = rows[start:end + 1]
        low = [x for x in window if x["gas"] < 0.5]
        target_recovered = next((x for x in window if x["effective_target"] >= 58), None)
        gas_recovered = next((x for x in window if x["gas"] >= .5), None)
        episodes.append({
            "start_step": rows[start]["step"],
            "end_step": rows[end]["step"],
            "peak_steer": peak_steer,
            "peak_heading": peak_heading,
            "aligned_seconds": round(sum(x["duration"] for x in window), 6),
            "low_gas_seconds": round(sum(x["duration"] for x in low), 6),
            "target_recovery_seconds": None if target_recovered is None else round(target_recovered["t"] - window[0]["t"], 6),
            "gas_recovery_seconds": None if gas_recovered is None else round(gas_recovered["t"] - window[0]["t"], 6),
            "onset": window[0],
            "context": rows[max(0, start - 2):min(len(rows), end + 3)] if window[0]["gas"] < .5 else [],
        })
        armed = False
        peak_steer = peak_heading = 0.0
    target_residual = []
    pedal_residual = []
    aligned_low = []
    for r in rows:
        if r["obstacle"] or r["recovery"] or r["step"] <= 10:
            continue
        centers = r["road_centers"].values()
        sweep = max(centers) - min(centers) if len(centers) > 1 else 28
        target_residual.append(abs(r["target"] - max(38, 60 - .8 * sweep)))
        if not r["mechanism_active"] and r["arrival_cap"] is None:
            gas = min(.6, max(0, .12 + .04 * (r["target"] - r["pixel_speed"])))
            brake = min(.28 if r["target"] < 60 else .15,
                        max(0, .02 * (r["pixel_speed"] - r["target"] - 2)))
            pedal_residual.append(max(abs(r["gas"] - gas), abs(r["brake"] - brake)))
        if abs(r["steer"]) <= .12 and abs(r["heading"]) <= .10 and r["gas"] < .5 and r["onroad"]:
            aligned_low.append(r)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "primary_path": str(primary_path),
            "primary_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
            "track_id": primary["track_id"], "seed": primary["seed"],
            "completed": primary["completed"], "lap_ms": primary["lapTimeMs"],
            "damage": primary["damage"], "collision_decisions": primary["collisions"],
            "retire_reason": primary["retire_reason"],
            "max_current_geometry_target_residual": max(target_residual, default=0),
            "current_geometry_target_rows": len(target_residual),
            "max_current_speed_pedal_residual": max(pedal_residual, default=0),
            "current_speed_pedal_rows": len(pedal_residual),
            "decisions": len(rows), "exits": episodes,
            "aligned_low_gas_count": len(aligned_low),
            "aligned_low_gas_below_target_count": sum(r["pixel_speed"] < r["target"] for r in aligned_low)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/koi-nominal-trajectory-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze(p) for p in sorted(args.run.glob("*-frozen_shield.decisions.jsonl"))]
    events = [e for r in results for e in r["exits"]]
    low = [e for e in events if e["onset"]["gas"] < .5]
    summary = {
        "episodes": len(results), "roads": len({r["seed"] for r in results}),
        "decisions": sum(r["decisions"] for r in results), "exit_candidates": len(events),
        "onset_gas_at_least_half": len(events) - len(low),
        "onset_low_gas": len(low),
        "onset_low_gas_at_or_above_target": sum(e["onset"]["pixel_speed"] >= e["onset"]["target"] for e in low),
        "target_recovery_seconds_counts": dict(Counter(str(e["target_recovery_seconds"]) for e in events)),
        "gas_recovery_seconds_counts": dict(Counter(str(e["gas_recovery_seconds"]) for e in events)),
        "max_current_geometry_target_residual": max(r["max_current_geometry_target_residual"] for r in results),
        "current_geometry_target_rows": sum(r["current_geometry_target_rows"] for r in results),
        "max_current_speed_pedal_residual": max(r["max_current_speed_pedal_residual"] for r in results),
        "current_speed_pedal_rows": sum(r["current_speed_pedal_rows"] for r in results),
        "archived_finishes": sum(r["completed"] for r in results),
        "archived_damage": sum(r["damage"] for r in results),
        "archived_collision_decisions": sum(r["collision_decisions"] for r in results),
    }
    bundle = Path("submissions/20261002-crossing-projection-collision-shield-v1-baseline")
    manifest = json.loads((bundle / "manifest.json").read_text())
    preservation = {}
    for name, expected in manifest["files_sha256"].items():
        if name == "submission.zip" or name.startswith("source/"):
            actual = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            assert actual == expected, name
            preservation[name] = actual
    shield = Path("haic/algorithms/koi/collision_shield.py")
    shield_sha = hashlib.sha256(shield.read_bytes()).hexdigest()
    assert shield_sha == manifest["policy_source_sha256"]
    assert summary["max_current_geometry_target_residual"] < 1e-9
    assert summary["max_current_speed_pedal_residual"] < 1e-6
    result = {"scope": "passive consumed TRAIN; no simulator or policy execution",
              "status": "HYPOTHESIS_NOT_CONFIRMED_DIRECTION_CLOSED",
              "decision": "No isolated unnecessary post-exit throttle hold established. Do not implement a candidate or run A/B. This does not prove all corner-exit optimization impossible.",
              "new_simulator_resets": 0, "candidate_created": False,
              "ab_evaluation": "NOT_RUN_HYPOTHESIS_GATE_FAILED",
              "ab_metrics": {"finish_damage_collision_preservation": None,
                             "target_recovery_delta_seconds": None,
                             "fixed_station_exit_time_delta_seconds": None,
                             "lap_delta_ms": None, "new_offroad_or_overspeed_failure": None},
              "source_observation": "Active target=max(38,60-.8*current road-center spread); current-speed proportional pedals, then observed-space sprint to72. Corridor EMA pedals are overwritten; active path has no corner-exit holding timer. Two-HUD speed averaging exists but does not demonstrate an unnecessary exit delay here.",
              "evidence": "30/36 putative alignment onsets already issue gas>=.5. Four of six low-gas onsets exceed their current target; all six fail the projected free-space gate and have substantial real upcoming road heading change within28.8m. Low gas or instantaneous alignment alone is not sufficient evidence to override preview braking.",
              "definition": "Prior obstacle-free |steer|>=.25 and |physical heading|>=.15rad; then two consecutive .08s decisions with |steer|<=.12, |heading|<=.10rad, all wheels onroad; obstacle/recovery invalidates corner history. Low gas means gas<.5, not by itself proof of unnecessary delay.",
              "recovery_definition": "From first aligned decision to effective target>=58 (within2 of base straight target60) or gas>=.5 while alignment persists; effective target uses sprint override72 when active. None is censored, not zero. Alignment-run duration is not a fixed-station exit time or A/B metric.",
              "frozen_bundle_verified": preservation,
              "worktree_shield_sha256": shield_sha,
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": summary,
              "episodes": results}
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    for episode in results:
        print(Path(episode["path"]).name, "decisions", episode["decisions"],
              "exits", len(episode["exits"]),
              "aligned low-gas", episode["aligned_low_gas_count"])
        for event in episode["exits"]:
            r = event["onset"]
            if r["gas"] < .5:
                print(" low onset", r["step"], "gas", round(r["gas"], 3),
                      "target/pixel-speed", round(r["target"], 2), round(r["pixel_speed"], 2),
                      "ahead bend degrees", round(math.degrees(r["ahead_28_8m_heading_change_rad"]), 2),
                      "free m", round(r["free_distance"], 2), r["sprint_blockers"])


if __name__ == "__main__":
    main()
