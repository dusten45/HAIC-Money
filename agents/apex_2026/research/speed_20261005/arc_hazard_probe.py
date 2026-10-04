"""Capture two exact, already-open early-circle hazard regression cameras."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from local_simulator.environment import create_environment, reset_environment
from local_simulator.schema import MapSpec


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("agents/apex_2026/results/speed-20261005")
    output = root/"arc-clear-v2-hazard-fixtures.json"
    fixture = root/"arc-clear-v2-hazard-fixtures.npz"
    assert not output.exists() and not fixture.exists()
    source = Path("agents/apex_2026/fast_early_circle_agent.py")
    source_sha = digest(source)
    assert source_sha == "5d9958d142970fcc0556af850ccd83895fbea08c8e1f81921a2f2d1085c7dad5"
    spec = importlib.util.spec_from_file_location("arc_hazard_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arrays, cases = {}, []
    for track, seed, last in ((3, 1007, 54), (4, 18800, 62)):
        trace_path = Path(f".haic-artifacts/apex-speed-20261005/early-circle-v1-traces/track-{track}-seed-{seed}.json")
        trace = json.loads(trace_path.read_text())
        agent = module.Agent()
        map_spec = MapSpec(track, seed, "official", (), 700, 4)
        environment, raw = create_environment(map_spec, render_mode=None)
        action_hash = hashlib.sha256()
        try:
            observation, _ = reset_environment(environment, map_spec)
            agent.reset(observation)
            for step in range(last+1):
                if step == last:
                    state = {name: value for name, value in agent.__dict__.items()
                             if isinstance(value, (int, float, bool, str, type(None)))}
                    arrays[f"track{track}_observation"] = observation.copy()
                    arrays[f"track{track}_prestate"] = np.asarray(json.dumps(state))
                action = agent.act(observation)
                np.testing.assert_array_equal(action, np.asarray(trace[step]["action"], np.float32))
                action_hash.update(action.tobytes())
                observation, _, terminated, truncated, _ = environment.step(action)
                np.testing.assert_allclose(np.linalg.norm(raw.car.hull.linearVelocity), trace[step]["speed"], atol=1e-7, rtol=0.)
                assert not terminated and not truncated
            cases.append({"track_id": track, "seed": seed, "camera_action": last,
                          "prefix_actions": last+1, "action_prefix_sha256": action_hash.hexdigest(),
                          "input_trace_sha256": digest(trace_path), "reference_action": action.tolist(),
                          "pre_state": state})
        finally:
            environment.close()
    assert digest(source) == source_sha
    np.savez_compressed(fixture, **arrays)
    payload = {"classification": "exact_open_mandatory_prefix_camera_fixture_not_lap_validation",
               "source_sha256": source_sha, "script_sha256": digest(__file__),
               "all_actions_and_post_step_speeds_exact": True,
               "fixture_sha256": digest(fixture), "cases": cases, "new_holdout_opened": False}
    output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
