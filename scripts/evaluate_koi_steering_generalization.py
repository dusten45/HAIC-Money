"""Freeze/preflight unchanged v2 on fresh nonprotected TRAIN; --run alone resets."""

import argparse
import fcntl
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any
import zipfile

from scripts import evaluate_koi_adaptive_ab as original
from scripts import evaluate_koi_minimum_clearance_ab as frozen
from scripts.package_koi_adaptive_avoidance import ROOT, SOURCE_COMMIT
from scripts.analyze_koi_steering_generalization import (ANALYSIS_SPEC, CELLS, SEEDS)
from scripts.analyze_koi_steering_release_ab import (ARMS, BASELINE_SHA,
                                                    read_json, read_jsonl)
from scripts import audit_koi_steering_generalization as freshness

SNAPSHOT, PYTHON, ENV = original.SNAPSHOT, original.PYTHON, original.ENV
sha, save = original.sha, original.save
resource_measurements, check_resources = frozen.resource_measurements, frozen.check_resources
footprint_measurements = frozen.footprint_measurements
PROTECTED = (49300, 49301, 49302, 49303, 51300, 51301, 51302, 51303)
STUDY = "koi-steering-generalization-v1"
RUN_PATH = Path("runs") / STUDY
PROTOCOL_PATH = Path("experiments") / f"{STUDY}.json"
R2_PATH = Path("runs/koi-steering-release-ab-20261001-r2")
R2_PROTOCOL_SHA = "9c0a9e4f11c3a1884bfc1bdd1be8841d76c81ca1dc3fa851096f23617396ee96"
CANDIDATE_SHA = "b1911d7dddf05d7c3f405b900f40c8c5696607130d2ef5e1cf473a82166147ce"
ROOT_AGENT_SHA = "673b06f338f7ca34e23e856291eea54123b1ec87e207e175f9abc97945f9b7ee"
EPISODES = len(CELLS) * len(ARMS)


def helper_paths():
    return [Path(__file__), ROOT / "scripts/analyze_koi_steering_generalization.py",
            ROOT / "scripts/audit_koi_steering_generalization.py",
            ROOT / "haic/train_seed_reservations.py",
            ROOT / "scripts/audit_rlpd_gate_unseen_train.py", ROOT / "scripts/audit_rlpd_g0_seeds.py",
            ROOT / "scripts/audit_rlpd_g1_coverage_seeds.py", ROOT / "scripts/audit_tdmpc2_train_diag_seeds.py",
            ROOT / "scripts/__init__.py",
            ROOT / "scripts/analyze_koi_steering_release_ab.py", ROOT / "scripts/evaluate_koi_steering_release_ab.py",
            ROOT / "scripts/evaluate_koi_minimum_clearance_ab.py",
            ROOT / "scripts/analyze_koi_minimum_clearance_ab.py", ROOT / "scripts/evaluate_koi_adaptive_ab.py",
            ROOT / "scripts/analyze_koi_adaptive_ab.py", ROOT / "scripts/package_koi_adaptive_avoidance.py",
            ROOT / "haic/algorithms/koi/steering_terms.py", ROOT / "haic/algorithms/koi/steering_release.py",
            ROOT / "scripts/package_koi_steering_release.py", ROOT / "haic/__init__.py",
            ROOT / "haic/algorithms/__init__.py", ROOT / "haic/algorithms/koi/__init__.py",
            ROOT / "scripts/diagnose_koi_steering_release.py"]


def slot_id(track, seed, arm):
    return f"{track}:{seed}:{arm}"


def controller_state(driver):
    """Read-only original controller state, including a last recovery hold pre-act."""
    while hasattr(driver, "driver"):
        driver = driver.driver
    return driver, dict(obstacle_side=float(driver.base._obstacle_side),
                        steps=int(driver.steps), impact_left=int(driver.impact_left))


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = list(ARMS) if (track + seed) % 2 else list(reversed(ARMS))
        rows.extend(dict(mode=arm, track_id=track, seed=seed, status="unrun") for arm in arms)
    return rows


def check_claims(receipt, protocol_sha256=None, cell=None):
    freshness.verify_claims(receipt, root=ROOT)
    assert not set(SEEDS) & set(PROTECTED)
    if cell is None:
        report = freshness.audit(root=ROOT, authenticated_receipt=receipt, self_protocol_sha256=protocol_sha256)
    else:
        report = freshness.audit_cell(receipt, cell, root=ROOT, self_protocol_sha256=protocol_sha256)
    assert report["status"] == "clear", "fresh TRAIN exposure audit failed"
    registry = ROOT / "experiments/train-seed-claims"
    return {str(Path(registry) / f"seed-{s}.json"): sha(Path(registry) / f"seed-{s}.json")
             for s in PROTECTED if (Path(registry) / f"seed-{s}.json").exists()}


def reference_evidence():
    """Read/re-hash r2 only: historical measurements never interact with consumed cells."""
    run = ROOT / R2_PATH
    assert sha(run / "protocol.json") == R2_PROTOCOL_SHA
    protocol = read_json(run / "protocol.json")
    report = read_json(run / "episode-report.json")
    assert report["protocol_sha256"] == R2_PROTOCOL_SHA and not report["operator_error"]
    assert protocol["model_hashes"] == {ARMS[0]: BASELINE_SHA, ARMS[1]: CANDIDATE_SHA}
    evidence: dict[str, Any] = dict(source_sha256={}, reference_episodes=len(report["rows"]))
    peaks, walls, sizes, roads = [], [], [], set()
    expected = {(r["track_id"], r["geometry_seed"], a) for r in protocol["cells"] for a in ARMS}
    assert len(report["rows"]) == len(expected) == protocol["episodes"]
    seen = set()
    for row in report["rows"]:
        key = row["track_id"], row["seed"], row["mode"]
        assert key in expected and key not in seen and row["status"] == "completed"
        seen.add(key)
        path = (run / row["file"]).resolve()
        assert path.is_relative_to(run.resolve()) and sha(path) == row["sha256"]
        episode = read_json(path)
        roads.add(hashlib.sha256(json.dumps(episode["catalog"]["track"], sort_keys=True).encode()).hexdigest())
        assert (episode["track_id"], episode["seed"], episode["mode"]) == key
        raw = (run / episode["raw_trace_file"]).resolve()
        assert raw.is_relative_to(run.resolve()) and sha(raw) == episode["raw_trace_sha256"]
        ticks, _ = read_jsonl(raw)
        assert len(ticks) == episode["raw_ticks"] and not episode["error"] and not episode["invalid_actions"]
        process = (run / row["process_file"]).resolve()
        assert process.is_relative_to(run.resolve()) and sha(process) == row["process_sha256"]
        receipt = read_json(process)
        assert receipt["status"] == "completed" and not receipt["error"]
        original_process = run / receipt["original_process_file"]
        assert original_process.resolve().is_relative_to(run.resolve()) and sha(original_process) == receipt["original_process_sha256"]
        for item in (path, raw, process, original_process):
            evidence["source_sha256"][str(item)] = sha(item)
        peaks.append(episode["peak_rss_bytes"])
        walls.append(receipt["wall_time_s"])
        sizes.append(path.stat().st_size + raw.stat().st_size)
    assert seen == expected
    for copy in protocol["source_copies"]:
        path = (run / copy["file"]).resolve()
        assert path.is_relative_to(run.resolve()) and sha(path) == copy["sha256"]
        evidence["source_sha256"][str(path)] = sha(path)
    for group in ("environment_source_sha256", "helper_source_sha256"):
        for path, expected_hash in protocol[group].items():
            assert sha(path) == expected_hash, "r2 source changed: " + path
            evidence["source_sha256"][path] = expected_hash
    ledger = run / "reset-ledger.jsonl"
    assert sha(ledger) == report["reset_ledger_sha256"]
    ledger_rows, _ = read_jsonl(ledger)
    assert len(ledger_rows) == 2 * len(expected)
    for path in (run / "protocol.json", run / "episode-report.json", ledger,
                 run / "candidate.zip", run / "candidate.manifest.json",
                 run / "crossing-projection-source-reconstruction.zip"):
        evidence["source_sha256"][str(path)] = sha(path)
    evidence.update(prior_peak_rss_bytes=max(peaks), prior_total_child_wall_s=sum(walls),
                     prior_max_child_wall_s=max(walls), prior_max_episode_bytes=max(sizes),
                     prior_road_sha256=sorted(roads),
                     scope="r2 read-only physical/resource evidence; no consumed cell interaction or freshness claim")
    return evidence


def resource_forecast(evidence):
    forecast = frozen.resource_forecast(evidence)
    per_slot = evidence["prior_total_child_wall_s"] / evidence["reference_episodes"]
    forecast.update(wall_budget_s=per_slot * EPISODES * 1.6, episodes=EPISODES,
                    measured_serial_wall_s=evidence["prior_total_child_wall_s"],
                    approximate_reference_s_per_episode=per_slot,
                    decision_budget_s=4.5, empirical_reference_peak_decision_s=.045,
                    rationale="Serial CPU21:1.5x measured r2 RSS+256MiB reserve;2x bytes; r2 per-slot wall * scheduled episodes *1.6, no old48 floor; no environment benchmark/reset. Slow new-road episodes/audits may censor this fixed budget; preserve failures, never retry/select roads.")
    return forecast


def validate_frozen(protocol_sha256):
    destination = ROOT / RUN_PATH
    assert sha(ROOT / PROTOCOL_PATH) == sha(destination / "protocol.json") == protocol_sha256
    protocol = read_json(destination / "protocol.json")
    assert protocol["study"] == STUDY and protocol["episodes"] == EPISODES == 144
    assert protocol["cells"] == freshness.proposed_cells()
    assert protocol["schedule"] == scheduled_slots() and protocol["analysis_spec"] == ANALYSIS_SPEC
    assert protocol["model_hashes"] == {ARMS[0]: BASELINE_SHA, ARMS[1]: CANDIDATE_SHA}
    assert protocol["operator_sha256"] == sha(__file__)
    assert protocol["analyzer_sha256"] == sha(ROOT / "scripts/analyze_koi_steering_generalization.py")
    assert protocol["root_agent_sha256"] == sha(ROOT / "agent.py") == ROOT_AGENT_SHA
    assert protocol["root_agent_source_sha256"] == {str(ROOT / "agent.py"): ROOT_AGENT_SHA}
    assert protocol["max_decisions"] == 1200 and protocol["frame_skip"] == 4
    assert protocol["warmup_ticks"] == 50 and protocol["raw_fps"] == 50
    assert protocol["python"] == str(PYTHON) and not protocol["official_action"]
    assert len(protocol["environment_source_sha256"]) == 142
    assert set(protocol["helper_source_sha256"]) == {str(p) for p in helper_paths()}
    for group in ("environment_source_sha256", "helper_source_sha256"):
        assert all(sha(p) == h for p, h in protocol[group].items()), "source drift"
    assert all(sha(p) == h for p, h in protocol["reference_evidence"]["source_sha256"].items()), "reference drift"
    for copy in protocol["source_copies"]:
        path = (destination / copy["file"]).resolve()
        assert path.is_relative_to(destination.resolve()) and sha(path) == copy["sha256"], "source copy drift"
    r2 = read_json(ROOT / R2_PATH / "protocol.json")
    assert protocol["model_source_sha256"] == r2["model_source_sha256"]
    assert protocol["candidate_manifest_sha256"] == r2["candidate_manifest_sha256"]
    for arm, name in ((ARMS[0], "crossing-projection-source-reconstruction.zip"), (ARMS[1], "candidate.zip")):
        assert sha(destination / name) == protocol["model_hashes"][arm]
        with zipfile.ZipFile(destination / name) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names))
            assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
            assert {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names} == protocol["model_source_sha256"][arm]
    assert sha(destination / "candidate.manifest.json") == protocol["candidate_manifest_sha256"]
    audit_pin = protocol["freshness"]["audit_receipt"]
    receipt = freshness.load_audit(ROOT / audit_pin["path"], audit_pin["sha256"], root=ROOT)
    return protocol, receipt


def validate_review(protocol, protocol_sha256, path, expected_sha256):
    path = Path(path).resolve()
    assert sha(path) == expected_sha256, "independent review receipt hash mismatch"
    review = read_json(path)
    assert review["status"] == "passed" and review["environment_resets"] == 0
    assert isinstance(review["reviewer"], str) and review["reviewer"].strip()
    assert review["protocol_sha256"] == protocol_sha256
    assert review["operator_sha256"] == protocol["operator_sha256"]
    assert review["analyzer_sha256"] == protocol["analyzer_sha256"]
    assert review["audit_sha256"] == protocol["freshness"]["audit_receipt"]["sha256"]
    return review


def runtime_versions():
    result = subprocess.run([str(PYTHON), "-I", "-c", "import sys,json,numpy,cv2,torch,gymnasium,Box2D,pygame;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,cv2=cv2.__version__,torch=torch.__version__,gymnasium=gymnasium.__version__,Box2D=Box2D.__version__,pygame=pygame.version.ver)))"], capture_output=True, text=True, env=ENV, check=True)
    versions = json.loads(result.stdout)
    assert versions["torch"].startswith("2.1.") and "+cpu" in versions["torch"]
    return versions


def preflight(protocol_sha256):
    """Read-only, zero-reset gate; partial runs are never presented as fresh."""
    destination = ROOT / RUN_PATH
    assert not (destination / "reset-ledger.jsonl").exists(), "run consumed or partial; no resume/freshness relabel"
    assert not list(destination.glob("*.raw.jsonl")), "partial raw evidence exists"
    assert not (destination / "episode-report.json").exists(), "attempt already recorded"
    assert not (destination / "execution-review.json").exists(), "execution already claimed"
    protocol, receipt = validate_frozen(protocol_sha256)
    assert runtime_versions() == protocol["runtime_versions"], "CPU21 runtime drift"
    measurement = resource_measurements(destination)
    check_resources(measurement, protocol["resource_forecast"], EPISODES)
    registry = ROOT / "experiments/train-seed-claims"
    descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        assert check_claims(receipt, protocol_sha256) == protocol["protected_claims_observed_sha256"]
    finally:
        os.close(descriptor)
    return dict(status="passed", protocol_sha256=protocol_sha256, environment_resets=0,
                episodes=EPISODES, resource_measurement=measurement,
                scope="freshness is an authenticated pre-run observation, not a partial-run result")


def lateral_separation(polygons, skins, centers, tangents, radii):
    """Observer-only whole-fixture transverse interval separation from circles."""
    import numpy as np
    rights = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    lo, hi = [], []
    for polygon, skin in zip(polygons, skins):
        projections = np.sum((polygon[None] - centers[:, None]) * rights[:, None], axis=2)
        lo.append(np.min(projections, axis=1) - skin)
        hi.append(np.max(projections, axis=1) + skin)
    return np.maximum(np.min(lo, axis=0) - radii, -np.max(hi, axis=0) - radii)


def worker(arm, track, seed, output):
    """Agent receives original pixels only; physical state belongs solely to observer."""
    import numpy as np
    from haic.algorithms.koi import steering_terms
    assert Path(output).resolve().parent == (ROOT / RUN_PATH).resolve()
    protocol_sha256 = sha(ROOT / PROTOCOL_PATH)
    protocol, exposure = validate_frozen(protocol_sha256)
    assert (track, seed) in CELLS and arm in ARMS
    assert sha(__file__) == protocol["operator_sha256"]
    assert all(sha(p) == h for p, h in protocol["helper_source_sha256"].items())
    assert all(sha(Path.cwd() / n) == h for n, h in protocol["model_source_sha256"][arm].items())
    sys.path.insert(0, str(Path.cwd()))
    model = import_module("agent").Agent()
    modules = {n: str(m.__file__) for n, m in tuple(sys.modules.items())
               if (n == "agent" or n.startswith("haic_agent.")) and getattr(m, "__file__", None)}
    assert all(Path(p).resolve().is_relative_to(Path.cwd()) for p in modules.values())
    import haic_agent
    haic_agent.__path__.append(str(SNAPSHOT / "haic_agent"))
    sys.path.insert(0, str(SNAPSHOT))
    import cv2
    import torch
    import gymnasium
    import Box2D
    import pygame
    from training.env_factory import create_training_environment
    run_episode = import_module("training.evaluate_closed_loop").run_episode
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().is_relative_to(SNAPSHOT):
            assert sha(path) == protocol["environment_source_sha256"][str(Path(path))]
    output = Path(output)
    raw_path = output.with_suffix(".raw.jsonl")
    holder = {}

    class Observer:
        def __init__(self, environment, stream):
            self.inner, self.stream = environment, stream
            self.environment = environment.environment
            self.active = False
            self.step_number = self.raw_count = 0
            self.last = {}
            original_step = self.unwrapped.step

            def observed_step(action):
                result = original_step(action)
                if self.active:
                    state = self.state()
                    state.update(step=self.step_number, collision=bool(result[4].get("collision", False)))
                    self.stream.write(json.dumps(state, allow_nan=False) + "\n")
                    self.raw_count += 1
                return result

            self.unwrapped.step = observed_step

        @property
        def unwrapped(self):
            return self.inner.unwrapped

        def reset(self):
            registry = ROOT / "experiments/train-seed-claims"
            descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                validate_frozen(protocol_sha256)
                assert check_claims(exposure, protocol_sha256,
                                    dict(partition="TRAIN", track_id=track, geometry_seed=seed, obstacles=True)) == protocol["protected_claims_observed_sha256"]
                for group in ("environment_source_sha256", "helper_source_sha256"):
                    assert all(sha(p) == h for p, h in protocol[group].items())
                assert all(sha(p) == h for p, h in protocol["reference_evidence"]["source_sha256"].items())
                assert sha(ROOT / PROTOCOL_PATH) == sha(output.parent / "protocol.json") == protocol_sha256
                assert sha(ROOT / "agent.py") == protocol["root_agent_sha256"]
                review_pin = read_json(output.parent / "execution-review.json")
                validate_review(protocol, protocol_sha256, review_pin["path"], review_pin["sha256"])
                assert all(sha(Path.cwd() / n) == h for n, h in protocol["model_source_sha256"][arm].items())
                check_resources(resource_measurements(output.parent), protocol["resource_forecast"], 1)
                result = self.inner.reset()
            finally:
                os.close(descriptor)
            raw = self.unwrapped
            self.track = np.asarray(raw.track, dtype=np.float64)
            xy = self.track[:, 2:4]
            self.stations = np.r_[0., np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=1))]
            self.road_edges = np.roll(xy, -1, axis=0) - xy
            self.road_lengths = np.linalg.norm(self.road_edges, axis=1)
            self.obstacles = []
            for index, body in enumerate(raw.obstacles):
                position = np.asarray(tuple(body.position))
                anchor = int(np.argmin(np.sum((xy - position)**2, axis=1)))
                beta = float(self.track[anchor, 1])
                self.obstacles.append(dict(id=index, x=float(position[0]), y=float(position[1]),
                                           radius=float(body.fixtures[0].shape.radius), anchor_index=anchor,
                                           station=float(self.stations[anchor]), tangent=[-float(np.sin(beta)), float(np.cos(beta))]))
            self.centers = np.asarray([[o["x"], o["y"]] for o in self.obstacles])
            self.normals = np.asarray([o["tangent"] for o in self.obstacles])
            self.radii = np.asarray([o["radius"] for o in self.obstacles])
            self.catalog = dict(track=raw.track, obstacles=self.obstacles)
            self.geometry_sha256 = hashlib.sha256(json.dumps(self.catalog, sort_keys=True).encode()).hexdigest()
            self.track_geometry_sha256 = hashlib.sha256(json.dumps(raw.track, sort_keys=True).encode()).hexdigest()
            self.initial = self.state()
            self.initial_observation_sha256 = hashlib.sha256(result[0].tobytes()).hexdigest()
            self.active = True
            return result

        def state(self):
            raw = self.unwrapped
            hull = raw.car.hull
            bodies = [hull, *raw.car.wheels]
            polygons, skins = [], []
            for body in bodies:
                for fixture in body.fixtures:
                    polygons.append(np.asarray([tuple(body.GetWorldPoint(v)) for v in fixture.shape.vertices]))
                    skins.append(float(fixture.shape.radius))
            clearance, rear, front = footprint_measurements(polygons, skins, self.centers, self.normals, self.radii)
            separation = lateral_separation(polygons, skins, self.centers, self.normals, self.radii)
            touched = set()
            for contact in raw.world.contacts:
                if contact.touching:
                    a, b = contact.fixtureA.body, contact.fixtureB.body
                    for index, obstacle in enumerate(raw.obstacles):
                        if (a == obstacle and any(b == body for body in bodies)) or (b == obstacle and any(a == body for body in bodies)):
                            touched.add(index)
            position = np.asarray(tuple(hull.position))
            amounts = np.clip(np.sum((position - self.track[:, 2:4]) * self.road_edges, axis=1) / np.maximum(self.road_lengths**2, 1e-12), 0, 1)
            points = self.track[:, 2:4] + amounts[:, None] * self.road_edges
            road_index = int(np.argmin(np.sum((points - position)**2, axis=1)))
            tangent = self.road_edges[road_index] / self.road_lengths[road_index]
            beta = float(np.arctan2(-tangent[0], tangent[1]))
            lateral = float(np.dot(position - points[road_index], [np.cos(beta), np.sin(beta)]))
            return dict(t=float(raw.t), x=float(position[0]), y=float(position[1]), yaw=float(hull.angle),
                        speed=float(np.hypot(*hull.linearVelocity)), road_index=road_index,
                        station=float(self.stations[road_index] + amounts[road_index] * self.road_lengths[road_index]),
                        lateral=lateral, heading_error=float((hull.angle - beta + np.pi) % (2 * np.pi) - np.pi),
                        rear=rear.tolist(), front=front.tolist(), clearance=clearance.tolist(), lateral_separation=separation.tolist(),
                        contacts=sorted(touched), wheel_road_contacts=[len(w.tiles) for w in raw.car.wheels])

        def step(self, action):
            self.step_number += 1
            before = self.state()
            result = self.inner.step(action)
            self.last = dict(pre=before, post=self.state(), **{k: bool(result[4].get(k, False)) for k in ("simulator_terminated", "wrapper_off_track", "wrapper_crashed")})
            return result

        def close(self):
            self.inner.close()

    class MeasuredAgent:
        def reset(self, observation):
            return model.reset(observation)

        def act(self, observation):
            _, before = controller_state(model.driver)
            partial.write(json.dumps(dict(status="act_intent", baseline_pre_act_state=before)) + "\n")
            action = model.act(observation)
            diagnostics = dict(model.last_step_diagnostics())
            driver, after = controller_state(model.driver)
            diagnostics.update(baseline_pre_act_state=before, baseline_post_act_state=after,
                               impact_left_before_act=before["impact_left"])
            baseline_action = np.asarray([diagnostics.get("baseline_steer", action[0]), action[1], action[2]], dtype=np.float32)
            terms = steering_terms.decompose(driver, observation, baseline_action, diagnostics)
            diagnostics["baseline_steering_terms"] = terms
            if arm == ARMS[0]:
                diagnostics.update(steering_terms=terms, steering_release_changed=False,
                                   steering_release_reason="frozen_baseline", steering_release_gate=False,
                                   steering_release_safe=None, baseline_steer=float(action[0]))
            else:
                assert "steering_terms" in diagnostics, "candidate must expose post-release steering components"
                assert diagnostics["steering_terms"]["schema"] == protocol["steering_terms_schema"]
            diagnostics.update(actual_obstacle_side=driver.base._obstacle_side, actual_impact_left=driver.impact_left)
            self.diagnostics = diagnostics
            partial.write(json.dumps(dict(status="act_returned", action=[float(x) for x in action],
                                          baseline_pre_act_state=before, baseline_post_act_state=after,
                                          steering_terms=diagnostics["steering_terms"],
                                          steering_release_reason=diagnostics.get("steering_release_reason"),
                                          steering_generation_ambiguous_motion_ignored=bool(diagnostics.get("steering_generation_ambiguous_motion_ignored", False))),
                                     allow_nan=False) + "\n")
            return action

        def last_step_diagnostics(self):
            return dict(self.diagnostics, evaluation_only=holder["environment"].last)

    partial_path = output.with_suffix(".decisions.jsonl")
    with raw_path.open("x", buffering=1) as stream, partial_path.open("x", buffering=1) as partial:
        def factory(**kwargs):
            holder["environment"] = Observer(create_training_environment(**kwargs), stream)
            return holder["environment"]
        result = run_episode(mode=arm, track_id=track, seed=seed, agent=MeasuredAgent(), max_decisions=1200,
                             plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    result.update(**original.observer_measurements(holder.get("environment")), raw_trace_file=raw_path.name,
                  track_geometry_sha256=getattr(holder.get("environment"), "track_geometry_sha256", None),
                  partial_decisions_file=partial_path.name, partial_decisions_sha256=sha(partial_path),
                  raw_trace_sha256=sha(raw_path), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  versions=dict(numpy=np.__version__, cv2=getattr(cv2, "__version__"), gymnasium=gymnasium.__version__, torch=torch.__version__,
                                Box2D=Box2D.__version__, pygame=pygame.version.ver, python=sys.version), runtime_module_paths=modules)
    save(output, result)


def run_child(command, directory, output, timeout, identity, operator_hash):
    try:
        original.run_child(command, directory, output, timeout)
    finally:
        path = Path(output).with_suffix(".process.json")
        if path.exists():
            receipt = read_json(path)
            # Old helper writes once; add the source/event binding in a separate
            # new receipt rather than edit that frozen helper or original output.
            save(Path(output).with_suffix(".bound-process.json"), dict(receipt, slot_id=identity, operator_sha256=operator_hash,
                                                                      original_process_file=path.name, original_process_sha256=sha(path)))


def freeze_protocol(audit_path, audit_sha256):
    """Freeze sources and copy the already-built r2 models, without loading an Agent."""
    from haic.algorithms.koi.steering_terms import SCHEMA
    destination = (ROOT / RUN_PATH).resolve()
    assert not destination.exists() and not (ROOT / PROTOCOL_PATH).exists(), "never overwrite a frozen run"
    audit_path = Path(audit_path).resolve()
    assert audit_path == (ROOT / freshness.AUDIT_PATH).resolve(), "use canonical authenticated exposure receipt"
    exposure = freshness.load_audit(audit_path, audit_sha256, root=ROOT)
    evidence = reference_evidence()
    r2 = ROOT / R2_PATH
    parent = read_json(r2 / "protocol.json")
    candidate, manifest = r2 / "candidate.zip", r2 / "candidate.manifest.json"
    receipt = read_json(manifest)
    assert receipt["candidate"] == "koi-steering-release-v2"
    assert receipt["candidate_zip_sha256"] == sha(candidate) == CANDIDATE_SHA
    assert sha(manifest) == parent["candidate_manifest_sha256"]
    assert receipt["baseline_source_sha256"] == parent["model_source_sha256"][ARMS[0]]
    assert sha(ROOT / "agent.py") == ROOT_AGENT_SHA, "root Agent changed since user freeze"
    sources = [SNAPSHOT / "env_wrapper.py", SNAPSHOT / "damage.py"]
    for name in ("training", "core", "haic_agent"):
        sources.extend(sorted((SNAPSHOT / name).rglob("*.py")))
    assert len(sources) == 142
    assert {str(p): sha(p) for p in sources} == parent["environment_source_sha256"]
    helpers = helper_paths()
    assert len(helpers) == len(set(helpers))
    assert parent["model_source_sha256"][ARMS[1]]["haic_agent/steering_terms.py"] == sha(ROOT / "haic/algorithms/koi/steering_terms.py")
    assert parent["model_source_sha256"][ARMS[1]]["haic_agent/steering_release_runtime.py"] == sha(ROOT / "haic/algorithms/koi/steering_release.py")
    forecast, versions = resource_forecast(evidence), runtime_versions()
    registry = ROOT / "experiments/train-seed-claims"
    descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        protected_pins = check_claims(exposure)
        destination.mkdir(exist_ok=False)
        copies = []
        for path in [*helpers, ROOT / "agent.py", *sources]:
            relative = Path("source/environment") / path.relative_to(SNAPSHOT) if path in sources else Path("source/project") / path.relative_to(ROOT)
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copies.append(dict(file=str(relative), source=str(path), sha256=sha(copied)))
        for name, path in (("crossing-projection-source-reconstruction.zip", r2 / "crossing-projection-source-reconstruction.zip"),
                           ("candidate.zip", candidate), ("candidate.manifest.json", manifest),
                           ("audit-receipt.json", audit_path)):
            (destination / name).write_bytes(path.read_bytes())
            copies.append(dict(file=name, source=str(path), sha256=sha(destination / name)))
        measurement = resource_measurements(destination)
        check_resources(measurement, forecast, EPISODES)
        protocol = dict(study=STUDY, scope="fresh-to-frozen-candidate nonprotected TRAIN generalization; no protected/official action", source_commit=SOURCE_COMMIT,
                         baseline="immutable c4e224d source reconstruction, not restoration of missing original crossing ZIP",
                         model_hashes=parent["model_hashes"], model_source_sha256=parent["model_source_sha256"], candidate_manifest_sha256=sha(manifest),
                         cells=freshness.proposed_cells(), excludes=list(PROTECTED),
                         episodes=EPISODES, max_decisions=1200, python=str(PYTHON), frame_skip=4, warmup_ticks=50, raw_fps=50,
                         steering_terms_schema=SCHEMA, analysis_spec=ANALYSIS_SPEC, schedule=scheduled_slots(),
                         runtime_versions=versions, environment_source_sha256={str(p): sha(p) for p in sources},
                         helper_source_sha256={str(p): sha(p) for p in helpers}, source_copies=copies,
                         operator_sha256=sha(__file__), analyzer_sha256=sha(ROOT / "scripts/analyze_koi_steering_generalization.py"),
                         root_agent_sha256=ROOT_AGENT_SHA, root_agent_source_sha256={str(ROOT / "agent.py"): ROOT_AGENT_SHA},
                         r2_protocol_sha256=R2_PROTOCOL_SHA,
                         resource_forecast=forecast, resource_measurement=measurement, reference_evidence=evidence,
                         freshness=dict(audit_receipt=dict(path=str(freshness.AUDIT_PATH), sha256=audit_sha256)),
                         protected_claims_observed_sha256=protected_pins,
                         geometry_clusters=len(SEEDS), cells_per_geometry=3,
                         track_geometry_digest="SHA256 json.dumps(raw.track, sort_keys=True); layout digest additionally includes obstacles",
                         telemetry_passive_not_agent_input=True, fresh_claim_registry_modified=False, official_action=False)
        save(destination / "protocol.json", protocol)
        with (ROOT / PROTOCOL_PATH).open("xb") as stream:
            stream.write((destination / "protocol.json").read_bytes())
    finally:
        os.close(descriptor)
    protocol_sha256 = sha(destination / "protocol.json")
    validate_frozen(protocol_sha256)
    return dict(status="frozen", protocol_sha256=protocol_sha256, environment_resets=0,
                protocol_path=str(ROOT / PROTOCOL_PATH), run=str(destination), episodes=EPISODES)


def run(protocol_sha256, review_path, review_sha256):
    preflight(protocol_sha256)
    protocol, exposure = validate_frozen(protocol_sha256)
    review_path = Path(review_path).resolve()
    validate_review(protocol, protocol_sha256, review_path, review_sha256)
    destination = (ROOT / RUN_PATH).resolve()
    models = {ARMS[0]: destination / "crossing-projection-source-reconstruction.zip", ARMS[1]: destination / "candidate.zip"}
    forecast, versions = protocol["resource_forecast"], protocol["runtime_versions"]
    registry = ROOT / "experiments/train-seed-claims"
    rows, current, error = scheduled_slots(), None, None
    ledger_path = destination / "reset-ledger.jsonl"
    # Create exclusively: another runner cannot overwrite review or reset evidence.
    with (destination / "execution-review.json").open("x") as stream:
        json.dump(dict(path=str(review_path), sha256=review_sha256), stream)
    started, pairs, roads = time.monotonic(), {}, {}
    try:
        with ledger_path.open("x", buffering=1) as ledger, tempfile.TemporaryDirectory(prefix="koi-steering-generalization-", dir="/tmp/kilo") as temporary:
            directories = {}
            for arm, path in models.items():
                directory = Path(temporary) / arm
                directory.mkdir()
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(directory)
                directories[arm] = directory
            for current in rows:
                arm, track, seed = current["mode"], current["track_id"], current["seed"]
                check_resources(resource_measurements(destination), forecast, sum(r["status"] == "unrun" for r in rows))
                remaining = forecast["wall_budget_s"] - (time.monotonic() - started)
                assert remaining > 0
                validate_frozen(protocol_sha256)
                validate_review(protocol, protocol_sha256, review_path, review_sha256)
                identity = slot_id(track, seed, arm)
                descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    assert check_claims(exposure, protocol_sha256,
                                        dict(partition="TRAIN", track_id=track, geometry_seed=seed, obstacles=True)) == protocol["protected_claims_observed_sha256"]
                    ledger.write(json.dumps(dict(status="reset_intent", arm=arm, track=track, seed=seed, slot_id=identity, time=time.time())) + "\n")
                    current["status"] = "reset_intent"
                finally:
                    os.close(descriptor)
                output = destination / f"{track}-{seed}-{arm}.json"
                command = f"import sys;sys.path.insert(0,{str(ROOT)!r});from scripts.evaluate_koi_steering_generalization import worker;worker({arm!r},{track},{seed},{str(output)!r})"
                run_child([str(PYTHON), "-I", "-c", command], directories[arm], output, min(forecast["child_timeout_s"], remaining), identity, protocol["operator_sha256"])
                record = read_json(output)
                process = output.with_suffix(".bound-process.json")
                compact = {k: record[k] for k in ("mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage", "collisions", "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes")}
                compact.update(status="completed", file=output.name, sha256=sha(output), slot_id=identity,
                               process_file=process.name, process_sha256=sha(process))
                current.update(compact)
                ledger.write(json.dumps(compact, allow_nan=False) + "\n")
                print(json.dumps(compact, allow_nan=False), flush=True)
                assert not record["error"] and not record["invalid_actions"]
                assert record["retire_reason"] not in {"act_timeout", "reset_timeout", "agent_reset_error", "agent_setup_error", "evaluation_error", "invalid_action", "unknown", "max_steps"}
                assert record["versions"] == versions and record["damage_telemetry_valid"] and record["collision_telemetry_valid"]
                road_digest = record["track_geometry_sha256"]
                assert road_digest not in protocol["reference_evidence"]["prior_road_sha256"], "exact old consumed road replay"
                assert seed not in roads or roads[seed] == road_digest, "same-seed cross-track road mismatch"
                assert all(s == seed or h != road_digest for s, h in roads.items()), "distinct seeds did not produce distinct roads"
                roads[seed] = road_digest
                previous = pairs.pop((track, seed), None)
                if previous is not None:
                    assert previous["geometry_sha256"] == record["geometry_sha256"]
                    assert previous["initial_state"] == record["initial_state"] and previous["initial_observation_sha256"] == record["initial_observation_sha256"]
                    keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
                    assert min(len(previous["decision_trace"]), len(record["decision_trace"])) >= 10
                    assert [[d[k] for k in keys] for d in previous["decision_trace"][:10]] == [[d[k] for k in keys] for d in record["decision_trace"][:10]]
                else:
                    pairs[(track, seed)] = record
            assert not pairs and len(roads) == len(SEEDS)
            validate_frozen(protocol_sha256)
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        if current is not None and current["status"] != "completed":
            current.update(status="operator_error", error=error)
            failed_output = destination / f"{current['track_id']}-{current['seed']}-{current['mode']}.json"
            current["partial_artifacts"] = [dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size)
                                            for suffix in (".json", ".raw.jsonl", ".decisions.jsonl", ".process.json", ".bound-process.json", ".stdout.txt", ".stderr.txt")
                                            if (path := failed_output.with_suffix(suffix)).exists()]
        save(destination / "operator-failure.json", dict(error=error, traceback=traceback.format_exc()))
        raise
    finally:
        save(destination / "episode-report.json", dict(protocol_sha256=protocol_sha256,
                                                       rows=rows, operator_error=error, reset_ledger_sha256=sha(ledger_path) if ledger_path.exists() else None,
                                                       review_receipt=dict(path=str(review_path), sha256=review_sha256),
                                                       complete=error is None and all(r["status"] == "completed" for r in rows),
                                                       freshness_scope="authenticated pre-run fresh-to-candidate TRAIN; interacted/partial cells remain consumed",
                                                       track_geometry_sha256={str(s): h for s, h in roads.items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-protocol", action="store_true", help="zero-reset immutable protocol/models/source copy")
    action.add_argument("--preflight", action="store_true", help="zero-reset read-only frozen validation")
    action.add_argument("--run", action="store_true", help="serial CPU21 interaction; requires independent source-bound review")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--review", type=Path)
    parser.add_argument("--review-sha256")
    args = parser.parse_args()
    if args.freeze_protocol:
        if not args.audit or not args.audit_sha256:
            parser.error("--freeze-protocol requires --audit and --audit-sha256")
        print(json.dumps(freeze_protocol(args.audit, args.audit_sha256), allow_nan=False))
    else:
        if not args.protocol_sha256:
            parser.error("--preflight/--run requires --protocol-sha256")
        if args.preflight:
            print(json.dumps(preflight(args.protocol_sha256), allow_nan=False))
        else:
            if not args.review or not args.review_sha256:
                parser.error("--run requires --review and --review-sha256")
            run(args.protocol_sha256, args.review, args.review_sha256)


if __name__ == "__main__":
    main()
