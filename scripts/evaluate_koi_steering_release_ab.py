"""Source-frozen serial CPU21 steering-release A/B; operator invocation alone resets."""

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
import zipfile

from scripts import evaluate_koi_adaptive_ab as original
from scripts import evaluate_koi_minimum_clearance_ab as frozen
from scripts.package_koi_adaptive_avoidance import ROOT, SOURCE_COMMIT, frozen_baseline_files
from scripts.analyze_koi_steering_release_ab import (ANALYSIS_SPEC, ARMS, BASELINE_SHA,
                                                    EXTRA_MEMBERS, read_json, read_jsonl)

SNAPSHOT, PYTHON, ENV = original.SNAPSHOT, original.PYTHON, original.ENV
CELLS, SEEDS = original.CELLS, original.SEEDS
sha, save = original.sha, original.save
resource_measurements, check_resources = frozen.resource_measurements, frozen.check_resources
footprint_measurements = frozen.footprint_measurements
PROTECTED = (49300, 49301, 49302, 49303, 51300, 51301, 51302, 51303)


def helper_paths():
    return [Path(__file__), ROOT / "scripts/analyze_koi_steering_release_ab.py", ROOT / "scripts/evaluate_koi_minimum_clearance_ab.py",
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


def check_claims(registry):
    frozen.check_claims(Path(registry))
    assert not set(SEEDS) & set(PROTECTED)
    # Existing protected claims do not collide with consumed reuse. Their bytes
    # are never changed; the schedule must never contain those geometry seeds.
    return {str(Path(registry) / f"seed-{s}.json"): sha(Path(registry) / f"seed-{s}.json")
            for s in PROTECTED if (Path(registry) / f"seed-{s}.json").exists()}


def consumed_evidence():
    evidence = frozen.consumed_evidence()
    for path in tuple(evidence["source_sha256"]):
        if path.endswith(".json"):
            read_json(path)
        elif path.endswith(".jsonl"):
            read_jsonl(path)
    peaks, walls, sizes = [], [], []
    for run_name in ("koi-minimum-clearance-ab-20260930-v1", "koi-minimum-clearance-ab-20260930-r2", "koi-minimum-clearance-ab-20260930-r3"):
        run = ROOT / "runs" / run_name
        protocol = read_json(run / "protocol.json")
        report = read_json(run / "episode-report.json")
        assert report["protocol_sha256"] == sha(run / "protocol.json") and not report["operator_error"]
        assert len(report["rows"]) == 48
        expected = {(t, s, a) for t, s in CELLS for a in ("crossing_projection", "minimum_clearance_v1")}
        seen = set()
        for row in report["rows"]:
            key = row["track_id"], row["seed"], row["mode"]
            assert key in expected and key not in seen and row["status"] == "completed"
            seen.add(key)
            path = (run / row["file"]).resolve()
            assert path.is_relative_to(run.resolve()) and sha(path) == row["sha256"]
            episode = read_json(path)
            assert (episode["track_id"], episode["seed"], episode["mode"]) == key
            raw = (run / episode["raw_trace_file"]).resolve()
            assert raw.is_relative_to(run.resolve()) and sha(raw) == episode["raw_trace_sha256"]
            ticks, _ = read_jsonl(raw)
            assert len(ticks) == episode["raw_ticks"]
            assert not episode["error"] and not episode["invalid_actions"]
            process = path.with_suffix(".process.json")
            receipt = read_json(process)
            assert receipt["status"] == "completed" and not receipt["error"]
            for item in (path, raw, process):
                evidence["source_sha256"][str(item)] = sha(item)
            if run_name.endswith("r3"):
                peaks.append(episode["peak_rss_bytes"])
                walls.append(receipt["wall_time_s"])
                sizes.append(path.stat().st_size + raw.stat().st_size)
        assert seen == expected
        for copy in protocol["source_copies"]:
            path = (run / copy["file"]).resolve()
            assert path.is_relative_to(run.resolve()) and sha(path) == copy["sha256"]
            evidence["source_sha256"][str(path)] = sha(path)
        ledger = run / "reset-ledger.jsonl"
        assert sha(ledger) == report["reset_ledger_sha256"]
        read_jsonl(ledger)
        for path in (run / "protocol.json", run / "episode-report.json", ledger):
            evidence["source_sha256"][str(path)] = sha(path)
    audits = [ROOT / "experiments" / f"koi-minimum-clearance-ab-{version}-audit.json" for version in ("v1", "r2", "r3")]
    audits.extend((ROOT / "experiments/koi-minimum-clearance-consumed-audit-v1.json",
                   ROOT / "experiments/koi-steering-release-consumed-audit-v1.json"))
    for path in audits:
        read_json(path)
        evidence["source_sha256"][str(path)] = sha(path)
    diagnosis_path = ROOT / "experiments/koi-steering-release-baseline-diagnosis-v1.json"
    diagnosis = read_json(diagnosis_path)
    assert diagnosis["study"] == "koi-steering-release-baseline-diagnosis-v1"
    assert all(diagnosis[k] == 0 for k in ("environment_imports", "environment_constructions", "environment_resets", "model_constructions"))
    for path, expected in diagnosis["output_sha256"].items():
        assert sha(path) == expected
        evidence["source_sha256"][path] = expected
    evidence["source_sha256"][str(diagnosis_path)] = sha(diagnosis_path)
    evidence.update(prior_peak_rss_bytes=max(peaks), prior_total_child_wall_s=sum(walls),
                    prior_max_child_wall_s=max(walls), prior_max_episode_bytes=max(sizes),
                    old_process_hash_caveat="Old process receipts lacked original pins; new hashes bind present bytes, not retroactive authentication.",
                    baseline_causal_diagnosis_sha256=sha(diagnosis_path),
                    closure="Adaptive speed-target and all minimum-clearance/margin hypotheses closed; copied evidence only, no candidate imports or logic reuse.")
    return evidence


def resource_forecast(evidence):
    forecast = frozen.resource_forecast(evidence)
    assert evidence["prior_total_child_wall_s"] * 1.6 <= 1800, "empirical wall forecast exceeds frozen1800s budget"
    forecast.update(wall_budget_s=1800, measured_serial_wall_s=evidence["prior_total_child_wall_s"],
                    approximate_reference_s_per_episode=evidence["prior_total_child_wall_s"] / 48,
                    decision_budget_s=4.5, empirical_reference_peak_decision_s=.045,
                    rationale="Completed CPU21 peak ~285MB and ~20s/episode (~960s/48),1.5x RSS+256MiB reserve,1.6x wall screen below fixed1800s cap;45ms reference decision <<4.5s; no environment benchmark/reset for forecast.")
    return forecast


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
    protocol = read_json(Path(output).parent / "protocol.json")
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
                check_claims(registry)
                for group in ("environment_source_sha256", "helper_source_sha256"):
                    assert all(sha(p) == h for p, h in protocol[group].items())
                assert all(sha(p) == h for p, h in protocol["consumed_reuse_evidence"]["source_sha256"].items())
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
            return action

        def last_step_diagnostics(self):
            return dict(self.diagnostics, evaluation_only=holder["environment"].last)

    with raw_path.open("x", buffering=1) as stream:
        def factory(**kwargs):
            holder["environment"] = Observer(create_training_environment(**kwargs), stream)
            return holder["environment"]
        result = run_episode(mode=arm, track_id=track, seed=seed, agent=MeasuredAgent(), max_decisions=1200,
                             plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    result.update(**original.observer_measurements(holder.get("environment")), raw_trace_file=raw_path.name,
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


def run(destination, candidate, manifest=None, study="koi-steering-release-ab-v1"):
    from haic.algorithms.koi.steering_terms import SCHEMA
    destination, candidate = Path(destination).resolve(), Path(candidate).resolve()
    manifest = Path(manifest).resolve() if manifest else candidate.with_suffix(".manifest.json")
    evidence = consumed_evidence()
    receipt = read_json(manifest)
    assert receipt["candidate"].startswith("koi-steering-release-") and receipt["candidate_zip_sha256"] == sha(candidate)
    baseline_files = frozen_baseline_files()
    baseline_members = {n: hashlib.sha256(d).hexdigest() for n, d in baseline_files.items()}
    assert receipt["baseline_source_sha256"] == baseline_members
    destination.mkdir(exist_ok=False)
    rows, current, protocol, error = scheduled_slots(), None, None, None
    ledger_path = destination / "reset-ledger.jsonl"
    try:
        baseline = destination / "crossing-projection-source-reconstruction.zip"
        with zipfile.ZipFile(baseline, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(baseline_files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
        assert sha(baseline) == BASELINE_SHA
        sources = [SNAPSHOT / "env_wrapper.py", SNAPSHOT / "damage.py"]
        for name in ("training", "core", "haic_agent"):
            sources.extend(sorted((SNAPSHOT / name).rglob("*.py")))
        assert len(sources) == 142
        for path in sources:
            blob = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:snapshots/haic-local-20260930/workspace/{path.relative_to(SNAPSHOT)}"], cwd=ROOT, check=True, capture_output=True).stdout
            assert hashlib.sha256(blob).hexdigest() == sha(path)
        helpers = helper_paths()
        copies = []
        for path in [*helpers, *sources]:
            relative = Path("source/environment") / path.relative_to(SNAPSHOT) if path in sources else Path("source/project") / path.relative_to(ROOT)
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copies.append(dict(file=str(relative), source=str(path), sha256=sha(copied)))
        models = {ARMS[0]: baseline, ARMS[1]: candidate}
        members = {}
        for arm, path in models.items():
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                assert len(names) == len(set(names)) and all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
                members[arm] = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names}
        inventory = {r["path"]: r["sha256"] for r in receipt["files"]}
        assert len(inventory) == len(receipt["files"]) and members[ARMS[1]] == inventory
        assert set(inventory) == set(baseline_files) | EXTRA_MEMBERS
        assert all(inventory[n] == h for n, h in baseline_members.items() if n != "agent.py")
        assert inventory["haic_agent/steering_terms.py"] == sha(ROOT / "haic/algorithms/koi/steering_terms.py")
        assert inventory["haic_agent/steering_release_runtime.py"] == sha(ROOT / "haic/algorithms/koi/steering_release.py")
        for name, path in (("candidate.zip", candidate), ("candidate.manifest.json", manifest)):
            (destination / name).write_bytes(path.read_bytes())
            copies.append(dict(file=name, source=str(path), sha256=sha(destination / name)))
        forecast, measurement = resource_forecast(evidence), resource_measurements(destination)
        check_resources(measurement, forecast, 48)
        runtime = subprocess.run([str(PYTHON), "-I", "-c", "import sys,json,numpy,cv2,torch,gymnasium,Box2D,pygame;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,cv2=cv2.__version__,torch=torch.__version__,gymnasium=gymnasium.__version__,Box2D=Box2D.__version__,pygame=pygame.version.ver)))"], capture_output=True, text=True, env=ENV, check=True)
        versions = json.loads(runtime.stdout)
        assert versions["torch"].startswith("2.1.") and "+cpu" in versions["torch"]
        protocol = dict(study=study, scope="explicit consumed TRAIN reuse; no fresh/protected/official action", source_commit=SOURCE_COMMIT,
                        baseline="immutable c4e224d source reconstruction, not restoration of missing original crossing ZIP",
                        model_hashes={a: sha(p) for a, p in models.items()}, model_source_sha256=members, candidate_manifest_sha256=sha(manifest),
                        cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in CELLS], excludes=list(PROTECTED),
                        episodes=48, max_decisions=1200, python=str(PYTHON), frame_skip=4, warmup_ticks=50, raw_fps=50,
                        steering_terms_schema=SCHEMA, analysis_spec=ANALYSIS_SPEC, schedule=scheduled_slots(),
                        runtime_versions=versions, environment_source_sha256={str(p): sha(p) for p in sources},
                        helper_source_sha256={str(p): sha(p) for p in helpers}, source_copies=copies,
                        operator_sha256=sha(__file__), analyzer_sha256=sha(ROOT / "scripts/analyze_koi_steering_release_ab.py"),
                        resource_forecast=forecast, resource_measurement=measurement, consumed_reuse_evidence=evidence,
                        telemetry_passive_not_agent_input=True, fresh_claim_registry_modified=False, official_action=False)
        registry = ROOT / "experiments/train-seed-claims"
        descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            protocol["protected_claims_observed_sha256"] = check_claims(registry)
            save(destination / "protocol.json", protocol)
        finally:
            os.close(descriptor)
        started, pairs = time.monotonic(), {}
        with ledger_path.open("x", buffering=1) as ledger, tempfile.TemporaryDirectory(prefix="koi-steering-release-", dir="/tmp/kilo") as temporary:
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
                for group in ("environment_source_sha256", "helper_source_sha256"):
                    assert all(sha(p) == h for p, h in protocol[group].items())
                assert all(sha(p) == h for p, h in evidence["source_sha256"].items())
                assert sha(models[arm]) == protocol["model_hashes"][arm] and sha(manifest) == protocol["candidate_manifest_sha256"]
                identity = slot_id(track, seed, arm)
                descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    check_claims(registry)
                    ledger.write(json.dumps(dict(status="reset_intent", arm=arm, track=track, seed=seed, slot_id=identity, time=time.time())) + "\n")
                    current["status"] = "reset_intent"
                finally:
                    os.close(descriptor)
                output = destination / f"{track}-{seed}-{arm}.json"
                command = f"import sys;sys.path.insert(0,{str(ROOT)!r});from scripts.evaluate_koi_steering_release_ab import worker;worker({arm!r},{track},{seed},{str(output)!r})"
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
                previous = pairs.pop((track, seed), None)
                if previous is not None:
                    assert previous["geometry_sha256"] == record["geometry_sha256"]
                    assert previous["initial_state"] == record["initial_state"] and previous["initial_observation_sha256"] == record["initial_observation_sha256"]
                    keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
                    assert min(len(previous["decision_trace"]), len(record["decision_trace"])) >= 10
                    assert [[d[k] for k in keys] for d in previous["decision_trace"][:10]] == [[d[k] for k in keys] for d in record["decision_trace"][:10]]
                else:
                    pairs[(track, seed)] = record
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        if current is not None and current["status"] != "completed":
            current.update(status="operator_error", error=error)
        save(destination / "operator-failure.json", dict(error=error, traceback=traceback.format_exc()))
        raise
    finally:
        save(destination / "episode-report.json", dict(protocol_sha256=sha(destination / "protocol.json") if protocol and (destination / "protocol.json").exists() else None,
                                                       rows=rows, operator_error=error, reset_ledger_sha256=sha(ledger_path) if ledger_path.exists() else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--study", default="koi-steering-release-ab-v1")
    args = parser.parse_args()
    run(args.output, args.candidate, args.manifest, args.study)


if __name__ == "__main__":
    main()
