"""Passive corner target-speed diagnosis; no policy or simulator imports."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


BUNDLE = Path("submissions/20261002-crossing-projection-collision-shield-v1-baseline")
SWEEP_EDGES = (0, 5, 10, 15, 20, 27.5, math.inf)
HEADING_EDGES = (0, 15, 30, 45, 60, math.inf)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def stats(values):
    return dict(n=len(values), min=min(values), mean=mean(values), max=max(values)) if values else dict(n=0)


def geometry(track):
    points = [(r[2], r[3]) for r in track]
    lengths, headings, stations = [], [], [0.0]
    for a, b in zip(points, points[1:] + points[:1]):
        lengths.append(math.hypot(b[0] - a[0], b[1] - a[1]))
        headings.append(math.atan2(-(b[0] - a[0]), b[1] - a[1]))
        stations.append(stations[-1] + lengths[-1])
    turns = [wrap(headings[i] - headings[i - 1]) for i in range(len(points))]
    curvature = [turns[i] / ((lengths[i - 1] + lengths[i]) / 2) for i in range(len(points))]
    # Geometric boundaries are independent of speed, finish, or policy output.
    groups = []
    for i in range(1, len(points) - 1):
        if abs(curvature[i]) < .01:
            continue
        sign = 1 if curvature[i] > 0 else -1
        if (groups and sign == groups[-1][0]
                and stations[i] - stations[groups[-1][1][-1]] <= 10.5
                and all(sign * turns[j] >= -.005 for j in range(groups[-1][1][-1] + 1, i))):
            groups[-1][1].append(i)
        else:
            groups.append((sign, [i]))
    corners = []
    for sign, indices in groups:
        a, b = indices[0], indices[-1]
        total = abs(sum(turns[a:b + 1]))
        if total < math.radians(15):
            continue
        peak = max(indices, key=lambda j: abs(curvature[j]))
        corners.append(dict(start=stations[a], end=stations[b + 1],
                            follow_end=min(stations[b + 1] + 10, stations[-1] - .1),
                            apex=stations[peak], turn_deg=math.degrees(total),
                            max_curvature=max(abs(k) for k in curvature[a:b + 1]),
                            direction=sign, road_indices=[a, b]))
    return lengths, headings, stations, corners


def margin(state):
    # Local straight-strip projection, NOT exact curved-road fixture clearance.
    h = state["heading_error"]
    return 40 / 6 - abs(state["lateral"]) - 1.6 * abs(math.cos(h)) - 2.61 * abs(math.sin(h))


def decision(row, headings, stations):
    c = row["controller"]
    p, q = c["evaluation_only"]["pre"], c["evaluation_only"]["post"]
    centers = list(c["road_centers"].values())
    sweep = max(centers) - min(centers) if len(centers) > 1 else 28.
    expected = max(38., 60. - .8 * sweep)
    blocked = bool(c.get("near_object") or c.get("far_active") or c.get("contact_active")
                   or c["shield"]["active"] or c.get("impact_steps_remaining", 0)
                   or c.get("impact_proxy_trigger") or c.get("recovery_changed")
                   or c.get("recovery_steps_remaining", 0))
    eligible = (row["step"] > 10 and not blocked and not c.get("mechanism_active")
                and c.get("arrival_cap") is None and abs(expected - c["target_speed"]) < 1e-7)
    i = p["road_index"]
    bends = [0.]
    for j in range(i + 1, i + len(headings)):
        station = stations[j % len(headings)] + (stations[-1] if j >= len(headings) else 0)
        if station - p["station"] > 28.8:
            break
        bends.append(abs(wrap(headings[j % len(headings)] - headings[i])))
    target, speed = c["target_speed"], c["pixel_speed"]
    gas = min(.6, max(0., .12 + .04 * (target - speed)))
    brake = min(.28 if target < 60 else .15, max(0., .02 * (speed - target - 2)))
    return dict(step=row["step"], t=p["t"], end_t=q["t"], station=p["station"],
                target=target, speed=p["speed"], pixel_speed=speed, sweep=sweep, observed_rows=len(centers),
                ahead_deg=math.degrees(max(bends)), eligible=eligible, blocked=blocked,
                formula_error=max(abs(gas - row["action"][1]), abs(brake - row["action"][2])),
                brake=row["action"][2], gas=row["action"][1],
                margin=margin(p), offroad=not all(p["wheel_road_contacts"]),
                damage_before=p["environment_state"]["damage"],
                damage_after=q["environment_state"]["damage"])


def analyze(path):
    base = path.name.replace(".decisions.jsonl", "")
    primary_path = path.with_name(base + ".json")
    raw_path = path.with_name(base + ".raw.jsonl")
    primary = json.loads(primary_path.read_text())
    source = [json.loads(line) for line in path.read_text().splitlines()]
    raw = [source[0]["controller"]["evaluation_only"]["pre"]]
    raw.extend(json.loads(line) for line in raw_path.read_text().splitlines())
    lengths, headings, stations, corners = geometry(primary["catalog"]["track"])
    rows = [decision(row, headings, stations) for row in source]
    events = []
    for corner in corners:
        entry_i = next((i for i in range(1, len(raw))
                        if raw[i - 1]["station"] < corner["start"] <= raw[i]["station"]
                        and raw[i]["station"] - raw[i - 1]["station"] < 10), None)
        event = dict(corner, track_id=primary["track_id"], seed=primary["seed"],
                     id=f"{primary['track_id']}/{primary['seed']}/{corner['road_indices'][0]}")
        if entry_i is None:
            event["status"] = "NOT_REACHED"
            events.append(event)
            continue
        exit_i = next((i for i in range(entry_i, len(raw))
                       if raw[i]["station"] >= corner["follow_end"]), None)
        exit_i = exit_i if exit_i is not None else len(raw) - 1
        window = raw[entry_i:exit_i + 1]
        observed_end = window[-1]["station"] >= corner["follow_end"]
        monotone = all(-1. <= b["station"] - a["station"] < 10 for a, b in zip(window, window[1:]))
        event["status"] = "COMPLETE" if observed_end and monotone else "CENSORED_OR_NONFORWARD"
        entry = max((r for r in rows if r["t"] <= window[0]["t"]), key=lambda r: r["t"])
        control = [r for r in rows if r["end_t"] > window[0]["t"] and r["t"] <= window[-1]["t"]]
        approach = [r for r in rows if corner["start"] - 40 <= r["station"] <= corner["apex"]
                    and r["t"] <= window[-1]["t"] and r["eligible"]]
        onset = next((r for r in approach if r["brake"] > .001), None)
        apex_state = min(window, key=lambda r: abs(r["station"] - corner["apex"]))
        eligible = [r for r in control if r["eligible"]]
        lo, hi = next((lo, hi) for lo, hi in zip(SWEEP_EDGES, SWEEP_EDGES[1:])
                      if lo <= entry["sweep"] < hi)
        relevant = [r for r in eligible if lo <= r["sweep"] < hi and r["station"] <= corner["end"]]
        overspeed = [r for r in relevant if r["pixel_speed"] >= r["target"] + 2]
        offroad = sum(not all(r["wheel_road_contacts"]) for r in window)
        all_offroad = sum(not any(r["wheel_road_contacts"]) for r in window)
        collision = sum(bool(r.get("collision") or r["contacts"]) for r in window)
        damage = max((r["damage_after"] for r in control), default=entry["damage_after"]) - entry["damage_before"]
        min_margin = min(margin(r) for r in window)
        clean = event["status"] == "COMPLETE" and not (offroad or collision or damage > 1e-8)
        isolated = entry["eligible"] and not any(r["blocked"] for r in control)
        above = entry["pixel_speed"] >= entry["target"] + 2
        sustained = len(relevant) >= 3 and len(overspeed) / len(relevant) >= .8
        event.update(entry_t=window[0]["t"], exit_t=window[-1]["t"], entry_target=entry["target"],
                     entry_speed=window[0]["speed"], entry_pixel_speed=entry["pixel_speed"],
                     entry_sweep=entry["sweep"], entry_ahead_deg=entry["ahead_deg"],
                     entry_target_speed_state=entry["speed"], entry_decision_dt=entry["t"] - window[0]["t"],
                     apex_speed=apex_state["speed"], min_speed=min(r["speed"] for r in window),
                     target_stats=stats([r["target"] for r in eligible]),
                     minimum_hud_target_gap=min((r["pixel_speed"] - r["target"] for r in eligible), default=None),
                     min_road_margin_proxy_m=min_margin, any_wheel_offroad_ticks=offroad,
                     all_wheels_offroad_ticks=all_offroad, collision_ticks=collision, damage_delta=damage,
                     clean_passage=clean, unconfounded=isolated, entry_above_target_2=above,
                     eligible_decisions=len(eligible), above_target_decisions=len(overspeed),
                     same_band_corner_decisions=len(relevant),
                     sustained_above_target=sustained,
                     braking_decisions=sum(r["brake"] > .001 for r in eligible),
                     near_limit=bool(offroad or collision or damage > 1e-8 or min_margin < 1.),
                     headroom_observation=bool(clean and isolated and above and sustained and min_margin >= 1.),
                     brake_onset=None if onset is None else dict(
                         t=onset["t"], lead_seconds=window[0]["t"] - onset["t"],
                         lead_m=corner["start"] - onset["station"], target=onset["target"],
                         speed=onset["speed"], pixel_speed=onset["pixel_speed"]))
        events.append(event)
    return dict(track_id=primary["track_id"], seed=primary["seed"], completed=primary["completed"],
                lap_ms=primary["lapTimeMs"], damage=primary["damage"], decisions=len(rows),
                raw_states=len(raw), events=events,
                inputs={str(p): digest(p) for p in (path, primary_path, raw_path)}), rows


def summarize(rows, events) -> dict:
    eligible = [r for r in rows if r["eligible"]]
    tables = {}
    for name, field, edges in (("road_center_spread_px", "sweep", SWEEP_EDGES),
                               ("ahead_heading_change_deg", "ahead_deg", HEADING_EDGES)):
        table = []
        for lo, hi in zip(edges, edges[1:]):
            group = [r for r in eligible if lo <= r[field] < hi]
            if not group:
                continue
            table.append(dict(band=f"{lo:g}-{hi:g}", decisions=len(group),
                              target=stats([r["target"] for r in group]),
                              actual_speed=stats([r["speed"] for r in group]),
                              pixel_speed=stats([r["pixel_speed"] for r in group]),
                              braking=sum(r["brake"] > .001 for r in group),
                              reduced_gas=sum(r["gas"] < .5 for r in group),
                              offroad=sum(r["offroad"] for r in group),
                              min_road_margin_proxy_m=min(r["margin"] for r in group)))
        tables[name] = table
    reached = [e for e in events if e["status"] != "NOT_REACHED"]
    bands = []
    for field, lo, hi in [(field, lo, hi)
                         for field, edges in (("entry_ahead_deg", HEADING_EDGES), ("entry_sweep", SWEEP_EDGES))
                         for lo, hi in zip(edges, edges[1:])]:
        group = [e for e in reached if lo <= e[field] < hi]
        isolated = [e for e in group if e["unconfounded"]]
        support = [e for e in isolated if e["headroom_observation"]]
        contradict = [e for e in isolated if e["near_limit"]]
        roads = sorted({e["seed"] for e in support})
        bands.append(dict(dimension=field, band=f"{lo:g}-{hi:g}", reached=len(group), unconfounded=len(isolated),
                          complete=sum(e["status"] == "COMPLETE" for e in isolated),
                          entry_target=stats([e["entry_target"] for e in isolated]),
                          entry_speed=stats([e["entry_speed"] for e in isolated]),
                          min_margin_proxy_m=min((e["min_road_margin_proxy_m"] for e in isolated), default=None),
                          clean_above_target=sum(e["clean_passage"] and e["entry_above_target_2"] for e in isolated),
                          any_offroad=sum(e["any_wheel_offroad_ticks"] > 0 for e in isolated),
                          damage_positive=sum(e["damage_delta"] > 1e-8 for e in isolated),
                          headroom_ids=[e["id"] for e in support], support_roads=roads,
                          near_limit_ids=[e["id"] for e in contradict],
                          candidate_gate=len(support) >= 3 and len(roads) >= 2 and not contradict))
    comparisons = []
    for a in reached:
        if not a["unconfounded"]:
            continue
        peers = [b for b in reached if b["id"] != a["id"] and b["unconfounded"]
                 and b["clean_passage"] and b["max_curvature"] >= a["max_curvature"]
                 and b["turn_deg"] >= a["turn_deg"] and b["entry_speed"] >= a["entry_speed"] + 2]
        if peers:
            b = min(peers, key=lambda e: e["entry_speed"])
            comparisons.append(dict(reference=a["id"], faster_clean=b["id"],
                                    entry_delta=b["entry_speed"] - a["entry_speed"],
                                    faster_apex=b["apex_speed"], faster_entry=b["entry_speed"],
                                    faster_min_margin=b["min_road_margin_proxy_m"],
                                    matched=False))
    natural = [e for e in reached if e["unconfounded"] and e["clean_passage"] and e["entry_above_target_2"]]
    onsets = [e["brake_onset"] for e in reached if e["unconfounded"] and e["brake_onset"] is not None]
    return dict(eligible_decisions=len(eligible), max_pedal_formula_error=max(
        (r["formula_error"] for r in eligible), default=0.), decision_tables=tables,
        event_status=dict(Counter(e["status"] for e in events)), corner_bands=bands,
        braking_onset_lead_m=stats([e["lead_m"] for e in onsets]),
        braking_onset_lead_seconds=stats([e["lead_seconds"] for e in onsets]),
        natural_clean_above_target=[{k: e[k] for k in (
            "id", "entry_target", "entry_pixel_speed", "entry_speed", "apex_speed", "min_speed",
            "entry_sweep", "min_road_margin_proxy_m", "braking_decisions", "sustained_above_target",
            "minimum_hud_target_gap", "headroom_observation")} for e in natural],
        hud_minus_physical_speed=stats([r["pixel_speed"] - r["speed"] for r in eligible]),
        faster_equal_or_sharper_natural_comparisons=comparisons,
        candidate_gate=any(b["candidate_gate"] for b in bands if b["dimension"] == "entry_sweep"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/koi-nominal-trajectory-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    preservation = {}
    for name, expected in manifest["files_sha256"].items():
        if name == "submission.zip" or name.startswith("source/"):
            actual = digest(BUNDLE / name)
            if actual != expected:
                raise ValueError(f"Frozen champion mismatch: {name}")
            preservation[name] = actual
    shield = Path("haic/algorithms/koi/collision_shield.py")
    if digest(shield) != manifest["policy_source_sha256"]:
        raise ValueError("Worktree shield differs from frozen champion")
    episodes, rows = [], []
    for path in sorted(args.run.glob("*-frozen_shield.decisions.jsonl")):
        episode, decisions = analyze(path)
        episodes.append(episode)
        rows.extend(decisions)
    if len(episodes) != 8:
        raise ValueError("Expected the eight existing champion episodes")
    events = [e for ep in episodes for e in ep["events"]]
    summary = summarize(rows, events)
    if summary["max_pedal_formula_error"] > 1e-6:
        raise ValueError("Eligible issued pedals differ from nominal target formula")
    result = dict(scope="Passive consumed TRAIN only; no policy execution or simulator resets",
                  status="HEADROOM_GATE_MET" if summary["candidate_gate"] else "NO_CLEAR_HEADROOM_DIRECTION_CLOSED",
                  candidate_created=False, ab_evaluation="NOT_RUN", new_simulator_resets=0,
                  source_sha256=digest(Path(__file__)), frozen_champion=preservation,
                  definitions=dict(
                      corners="Same-sign geometric curvature >=.01 rad/m; bridge <=10.5m without opposite turn; total >=15deg; exclude track seam; observe through corner end+10m",
                      speed="Physical speed in simulator distance units/s; target and above-target checks use HUD scale ONLY, not direct physical-speed minus target",
                      margin="40/6 - abs(lateral) -1.6*abs(cos(heading_error))-2.61*abs(sin(heading_error)); conservative local-strip footprint projection, NOT exact curved-road clearance",
                      safety="Every archived raw tick: any/all wheel contact loss, obstacle contact/collision; cumulative damage from decision telemetry; retain noncompletion/censoring",
                      gate="At least three complete, unconfounded, all-wheel-onroad passages on >=2 geometries in one road-spread band; HUD entry and >=80% of >=3 eligible SAME-BAND decisions before corner end >=target+2; margin proxy >=1m; no unconfounded near-limit counterexample in band. Post-corner target recovery is not counted against speed support.",
                      interpretation="Entry overspeed followed by braking is not evidence for maintaining that speed; observational comparisons are not matched causal or fresh-road evidence",
                      onset="First eligible issued brake>0.001 from 40m before geometric corner start through apex; signed distance/time before entry, negative means after entry"),
                  summary=summary, episodes=episodes)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], episodes=len(episodes),
                          roads=len({e["seed"] for e in episodes}), decisions=len(rows),
                          eligible_decisions=summary["eligible_decisions"],
                          event_status=summary["event_status"],
                          natural_clean_above_target=len(summary["natural_clean_above_target"]),
                          headroom_observations=[e["id"] for e in events if e.get("headroom_observation")],
                          corner_bands=summary["corner_bands"],
                          braking_onset_lead_m=summary["braking_onset_lead_m"],
                          braking_onset_lead_seconds=summary["braking_onset_lead_seconds"]), indent=2))


if __name__ == "__main__":
    main()
