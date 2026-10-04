"""Focused, zero-reset post-passage diagnosis of six consumed TRAIN cells."""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs/koi-steering-generalization-v1"
NEW = ROOT / "runs/koi-collision-priority-v1"
CELLS = ((1, 3184000002), (2, 3184000006), (2, 3184000001),
         (3, 3184000002), (1, 3184000013), (1, 3184000015))


def snapshot(state):
    return {k: state[k] for k in ("t", "heading_error", "lateral", "speed", "wheel_road_contacts", "station")}


def decision(d):
    c = d["controller"]
    centers = {int(k): v for k, v in c.get("road_centers", {}).items()}
    angle = math.atan((centers[42] - centers[54]) * 1.25 / 12) if 42 in centers and 54 in centers else None
    return dict(step=d["step"], action=[d[k] for k in ("steer", "gas", "brake")],
                pre=snapshot(c["evaluation_only"]["pre"]), post=snapshot(c["evaluation_only"]["post"]),
                road_centers=centers, pixel_secant_heading_rad=angle,
                raw_terms=c.get("steering_terms", {}).get("raw_terms"),
                near_object=c.get("near_object"), release_reason=c.get("steering_release_reason"),
                priority={k:v for k,v in c.items() if k.startswith("priority_")})


def read_episode(path):
    e = json.loads(path.read_text())
    raw_path = path.parent / e["raw_trace_file"]
    data = raw_path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == e["raw_trace_sha256"]
    states = [e["initial_state"], *[json.loads(line) for line in data.splitlines()]]
    return e, states


def offroad_events(states):
    events, start = [], None
    for i, state in enumerate(states):
        off = not any(state["wheel_road_contacts"])
        if off and start is None:
            start = i
        if start is not None and (not off or i == len(states)-1):
            last = i-1 if not off else i
            events.append(dict(start=snapshot(states[start]), end=snapshot(states[last]),
                               duration_s=states[last]["t"]-states[start]["t"]+.02,
                               first_any_contact=None if off else snapshot(state), censored=off))
            start = None
    return events


def passage_windows(e, states):
    windows = []
    trace = e["decision_trace"]
    for index, obj in enumerate(e["catalog"]["obstacles"]):
        passed = None
        for j in range(1, len(states)):
            before, now = states[j-1], states[j]
            if (abs(now["station"]-obj["station"]) < 40
                    and abs(now["road_index"]-obj["anchor_index"]) < 15
                    and now["clearance"][index] < 25
                    and before["rear"][index] <= obj["radius"] < now["rear"][index]):
                passed = j
                break
        if passed is None:
            windows.append(dict(obstacle_id=obj["id"], passed=False))
            continue
        t = states[passed]["t"]
        window = [s for s in states if t-.4 <= s["t"] <= t+2.000001]
        after = [s for s in window if s["t"] >= t]
        samples = {}
        for offset in (-.4, 0, .32, .64, 1., 2.):
            target = t + offset
            samples[str(offset)] = snapshot(min(states, key=lambda s: abs(s["t"]-target))) if target <= states[-1]["t"] else None
        all_four = next((j for j,s in enumerate(after) if all(s["wheel_road_contacts"])
                         and j+11 < len(after) and all(all(x["wheel_road_contacts"]) for x in after[j:j+12])), None)
        nearby = [d for d in trace if t-.4 <= d["controller"]["evaluation_only"]["pre"]["t"] <= t+1.]
        peak = max(after, key=lambda s: abs(s["heading_error"]))
        windows.append(dict(obstacle_id=obj["id"], passed=True, rear_clear_t=t,
                            samples=samples, max_abs_heading_first2s=max(abs(s["heading_error"]) for s in after),
                            max_abs_lateral_first2s=max(abs(s["lateral"]) for s in after),
                            first_all4_contact_sustained_12ticks_after_passage_s=None if all_four is None else after[all_four]["t"]-t,
                            recovery_censored=all_four is None, followup_s=after[-1]["t"]-t,
                            zero_contact_events=offroad_events(window),
                            min_object_clearance_m=min(s["clearance"][index] for s in states),
                            peak_heading_state=snapshot(peak), decisions=[decision(d) for d in nearby]))
    return windows


def summarize(path):
    e, states = read_episode(path)
    windows = passage_windows(e, states)
    return dict(file=str(path.relative_to(ROOT)), track_id=e["track_id"], seed=e["seed"], arm=e["mode"],
                completed=e["completed"], retire_reason=e["retire_reason"], damage=e["damage"],
                terminal=snapshot(states[-1]), offroad_events=offroad_events(states),
                windows=windows,
                max_abs_heading=max(abs(s["heading_error"]) for s in states),
                max_abs_lateral=max(abs(s["lateral"]) for s in states))


def main():
    episodes = [summarize(OLD / f"{t}-{s}-{a}.json") for t,s in CELLS
                for a in ("crossing_projection", "steering_release_v1")]
    regression = []
    for arm in ("steering_release_v2", "collision_priority"):
        path = NEW / f"3-3184000002-r0-{arm}.json"
        e, states = read_episode(path)
        item = summarize(path)
        trace = e["decision_trace"]
        changed = [d for d in trace if d["controller"].get("priority_changed")]
        item["changed_decisions"] = [decision(d) for d in changed]
        if changed:
            first_t = changed[0]["controller"]["evaluation_only"]["pre"]["t"]
            item["first_change_followup"] = [decision(d) for d in trace if first_t-.16 <= d["controller"]["evaluation_only"]["pre"]["t"] <= first_t+2.]
        regression.append(item)
    result = dict(schema="koi-collision-recovery-diagnosis-v1", cells=CELLS, episodes=episodes,
                  new_priority_regression=regression, environment_resets=0,
                  limits=["Six outcome-selected consumed TRAIN conditions; crossing/v2 comparison descriptive, not causal.",
                          "Rear-clear measured on all fixtures relative to fixed obstacle tangent, local geometry and clearance<25m gated.",
                          "Post-passage 2s window may include next object; no causal attribution to previous object.",
                          "All-four recovery is twelve consecutive 50Hz samples (0.24s nominal exposure); zero means already road-supported.",
                          "Wheel-road contacts are not official reward-based off_track_counter.",
                          "New priority archive omitted road centers and steering terms; no reconstruction invented."])
    output = ROOT / "experiments/koi-collision-recovery-diagnosis-v1.json"
    with output.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    for e in episodes + regression:
        candidates = [w for w in e["windows"] if w["passed"]]
        target = {(1,3184000002):4, (2,3184000006):5, (2,3184000001):1,
                  (3,3184000002):0, (1,3184000013):3, (1,3184000015):1}[e["track_id"], e["seed"]]
        top = [w for w in candidates if w["obstacle_id"] == target]
        print(json.dumps(dict(cell=[e["track_id"],e["seed"]], arm=e["arm"], completed=e["completed"],
                              terminal=e["terminal"], offroad_events=e["offroad_events"],
                              top_windows=[{k:v for k,v in w.items() if k not in ("decisions", "zero_contact_events")} for w in top])))
    changed = regression[-1]["changed_decisions"]
    print("FIRST_PRIORITY_CHANGES", json.dumps(changed[:2]))


if __name__ == "__main__":
    main()
