"""Source-frozen serial CPU21 A/B on exactly24 already-consumed TRAIN cells."""

import argparse
import fcntl
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any
import zipfile

from scripts import evaluate_koi_adaptive_ab as original
from scripts.package_koi_adaptive_avoidance import ROOT, SOURCE_COMMIT, frozen_baseline_files


SNAPSHOT = original.SNAPSHOT
PYTHON = original.PYTHON
ENV = original.ENV
CELLS = original.CELLS
SEEDS = original.SEEDS
ARMS = ("crossing_projection", "minimum_clearance_v1")
PRIOR = ROOT / "runs/koi-adaptive-ab-20260930-v1"
sha = original.sha
save = original.save


def scheduled_slots():
    rows = []
    for track, seed in CELLS:
        arms = list(ARMS)
        if (track + seed) % 2 == 0:
            arms.reverse()
        rows.extend(dict(mode=a, track_id=track, seed=seed, status="unrun") for a in arms)
    return rows


def consumed_evidence() -> dict[str, Any]:
    report = json.loads((PRIOR / "episode-report.json").read_text())
    protocol = json.loads((PRIOR / "protocol.json").read_text())
    assert report["protocol_sha256"] == sha(PRIOR / "protocol.json")
    assert report["operator_error"] is None
    assert [(r["track_id"], r["geometry_seed"]) for r in protocol["cells"]] == CELLS
    assert all(r["partition"] == "TRAIN" and r["obstacles"] for r in protocol["cells"])
    expected = {(t, s, a) for t, s in CELLS for a in ("crossing_projection", "adaptive_v1")}
    assert len(report["rows"]) == 48
    seen = set()
    files = {str(PRIOR / "protocol.json"): sha(PRIOR / "protocol.json"),
             str(PRIOR / "episode-report.json"): sha(PRIOR / "episode-report.json")}
    peaks, durations, sizes = [], [], []
    for row in report["rows"]:
        key = (row["track_id"], row["seed"], row["mode"])
        assert key in expected and key not in seen and row["status"] == "completed"
        seen.add(key)
        path = (PRIOR / row["file"]).resolve()
        assert path.is_relative_to(PRIOR.resolve()) and sha(path) == row["sha256"]
        episode = json.loads(path.read_text())
        assert (episode["track_id"], episode["seed"], episode["mode"]) == key
        assert not episode["error"] and not episode["invalid_actions"]
        raw = (PRIOR / episode["raw_trace_file"]).resolve()
        assert raw.is_relative_to(PRIOR.resolve()) and sha(raw) == episode["raw_trace_sha256"]
        assert sum(bool(line.strip()) for line in raw.read_text().splitlines()) == episode["raw_ticks"]
        process = path.with_suffix(".process.json")
        receipt = json.loads(process.read_text())
        assert receipt["status"] == "completed" and not receipt["error"]
        for item in (path, raw, process):
            files[str(item)] = sha(item)
        peaks.append(episode["peak_rss_bytes"])
        durations.append(receipt["wall_time_s"])
        sizes.append(path.stat().st_size + raw.stat().st_size)
    assert seen == expected
    audit = ROOT / "experiments/koi-adaptive-ab-v1-audit.json"
    files[str(audit)] = sha(audit)
    return dict(scope="explicit previously consumed TRAIN reuse, not freshness clearance", source_sha256=files,
                cells=24, episodes=48, prior_peak_rss_bytes=max(peaks), prior_total_child_wall_s=sum(durations),
                prior_max_child_wall_s=max(durations), prior_max_episode_bytes=max(sizes))


def resource_measurements(destination):
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.split()[0]) * 1024
    membership = Path("/proc/self/cgroup").read_text().strip()
    relative = next(line.split("::", 1)[1] for line in membership.splitlines() if line.startswith("0::"))
    leaf = Path("/sys/fs/cgroup") / relative.lstrip("/")
    ancestors = []
    for directory in (leaf, *leaf.parents):
        if not directory.is_relative_to(Path("/sys/fs/cgroup")):
            break
        limit = (directory / "memory.max").read_text().strip()
        current = int((directory / "memory.current").read_text())
        ancestors.append(dict(path=str(directory), memory_max=limit, memory_current=current,
                              raw_headroom_bytes=None if limit == "max" else int(limit) - current,
                              memory_events=(directory / "memory.events").read_text(),
                              memory_stat=(directory / "memory.stat").read_text(),
                              memory_pressure=(directory / "memory.pressure").read_text()))
    assert ancestors
    return dict(host_mem_available_bytes=meminfo["MemAvailable"], host_dirty_bytes=meminfo["Dirty"],
                host_writeback_bytes=meminfo["Writeback"], cgroup=membership, visible_cgroup_ancestors=ancestors,
                disk=[dict(path=str(p), device=p.stat().st_dev, free_bytes=shutil.disk_usage(p).free) for p in (Path(destination), Path("/tmp/kilo"))])


def resource_forecast(evidence) -> dict[str, Any]:
    return dict(peak_child_bytes=int(evidence["prior_peak_rss_bytes"] * 1.5),
                memory_reserve_bytes=256 * 1024**2, serial_children=1, gpu=False,
                per_episode_disk_bytes=max(8 * 1024**2, evidence["prior_max_episode_bytes"] * 2),
                disk_reserve_bytes=128 * 1024**2, temp_bytes=16 * 1024**2,
                wall_budget_s=max(1800, evidence["prior_total_child_wall_s"] * 1.6),
                child_timeout_s=max(90, evidence["prior_max_child_wall_s"] * 2),
                rationale="Same CPU21 serial telemetry path;1.5x measured prior peak+256MiB growth reserve,2x per-episode bytes,1.6x measured total duration; not official limits")


def check_resources(measurement, forecast, remaining):
    required = forecast["peak_child_bytes"] + forecast["memory_reserve_bytes"]
    assert measurement["host_mem_available_bytes"] > required, "host memory forecast does not fit"
    for group in measurement["visible_cgroup_ancestors"]:
        assert group["raw_headroom_bytes"] is None or group["raw_headroom_bytes"] > required, "cgroup memory forecast does not fit"
    out, temporary = measurement["disk"]
    output_need = remaining * forecast["per_episode_disk_bytes"] + forecast["disk_reserve_bytes"]
    temp_need = forecast["temp_bytes"] + forecast["disk_reserve_bytes"]
    if out["device"] == temporary["device"]:
        assert out["free_bytes"] > output_need + forecast["temp_bytes"], "shared filesystem forecast does not fit"
    else:
        assert out["free_bytes"] > output_need and temporary["free_bytes"] > temp_need, "filesystem forecast does not fit"


def check_claims(registry):
    collisions = [str(registry / f"seed-{s}.json") for s in SEEDS if (registry / f"seed-{s}.json").exists()]
    assert not collisions, "active claim collision: " + repr(collisions)


def footprint_measurements(polygons, skins, centers, normals, radii):
    """Signed circle separation from the union of polygon fixtures and their skins."""
    import numpy as np
    clearances = []
    rears, fronts = [], []
    for polygon, skin in zip(polygons, skins):
        edges = np.roll(polygon, -1, axis=0) - polygon
        offset = centers[:, None, :] - polygon[None]
        amount = np.clip(np.sum(offset * edges[None], axis=2) / np.maximum(np.sum(edges**2, axis=1), 1e-12), 0, 1)
        distance = np.min(np.linalg.norm(offset - amount[:, :, None] * edges[None], axis=2), axis=1)
        cross = edges[None, :, 0] * offset[:, :, 1] - edges[None, :, 1] * offset[:, :, 0]
        inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
        distance[inside] *= -1
        clearances.append(distance - skin - radii)
        projections = np.sum((polygon[None] - centers[:, None]) * normals[:, None], axis=2)
        rears.append(np.min(projections - skin, axis=1))
        fronts.append(np.max(projections + skin, axis=1))
    return np.min(clearances, axis=0), np.min(rears, axis=0), np.max(fronts, axis=0)


def worker(arm, track, seed, output):
    """Passive observer owns telemetry; the Agent receives only original pixels."""
    import resource
    import numpy as np
    protocol = json.loads(Path(output).parent.joinpath("protocol.json").read_text())
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
    raw_log = raw_path.open("x", buffering=1)
    holder = {}

    class Observer:
        def __init__(self, environment):
            self.inner = environment
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
                    raw_log.write(json.dumps(state, allow_nan=False) + "\n")
                    self.raw_count += 1
                return result

            self.unwrapped.step = observed_step

        @property
        def unwrapped(self):
            return self.inner.unwrapped

        def reset(self):
            descriptor = os.open(ROOT / "experiments/train-seed-claims", os.O_RDONLY | os.O_DIRECTORY)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                check_claims(ROOT / "experiments/train-seed-claims")
                assert all(sha(p) == h for p, h in protocol["environment_source_sha256"].items())
                result = self.inner.reset()
            finally:
                os.close(descriptor)
            raw = self.unwrapped
            self.track = np.asarray(raw.track, dtype=np.float64)
            xy = self.track[:, 2:4]
            self.stations = np.r_[0.0, np.cumsum(np.linalg.norm(xy[1:] - xy[:-1], axis=1))]
            self.road_edges = np.roll(xy, -1, axis=0) - xy
            self.road_lengths = np.linalg.norm(self.road_edges, axis=1)
            self.obstacles = []
            for i, body in enumerate(raw.obstacles):
                position = np.asarray(tuple(body.position), dtype=np.float64)
                anchor = int(np.argmin(np.sum((xy - position)**2, axis=1)))
                beta = float(self.track[anchor, 1])
                self.obstacles.append(dict(id=i, x=float(position[0]), y=float(position[1]),
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
            touched = set()
            for contact in raw.world.contacts:
                if contact.touching:
                    a, b = contact.fixtureA.body, contact.fixtureB.body
                    for i, obstacle in enumerate(raw.obstacles):
                        if (a == obstacle and any(b == body for body in bodies)) or (b == obstacle and any(a == body for body in bodies)):
                            touched.add(i)
            position = np.asarray(tuple(hull.position))
            road_amount = np.clip(np.sum((position - self.track[:, 2:4]) * self.road_edges, axis=1) / np.maximum(self.road_lengths**2, 1e-12), 0, 1)
            road_points = self.track[:, 2:4] + road_amount[:, None] * self.road_edges
            road_index = int(np.argmin(np.sum((road_points - position)**2, axis=1)))
            tangent = self.road_edges[road_index] / self.road_lengths[road_index]
            beta = float(np.arctan2(-tangent[0], tangent[1]))
            lateral = float(np.dot(position - road_points[road_index], [np.cos(beta), np.sin(beta)]))
            heading = float((hull.angle - beta + np.pi) % (2 * np.pi) - np.pi)
            return dict(t=float(raw.t), x=float(position[0]), y=float(position[1]), yaw=float(hull.angle),
                        speed=float(np.hypot(*hull.linearVelocity)), road_index=road_index,
                        station=float(self.stations[road_index] + road_amount[road_index] * self.road_lengths[road_index]),
                        lateral=lateral, heading_error=heading, rear=rear.tolist(), front=front.tolist(), clearance=clearance.tolist(),
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
            action = model.act(observation)
            self.last_steer = float(action[0])
            return action

        def last_step_diagnostics(self):
            result = dict(model.last_step_diagnostics())
            driver = model.driver
            if hasattr(driver, "driver"):
                driver = driver.driver
            side = driver.base._obstacle_side
            impact = driver.impact_left
            result.update(evaluation_only=holder["environment"].last, actual_obstacle_side=side, actual_impact_left=impact)
            if arm == ARMS[0]:
                result.update(clearance_projection_active=bool(result.get("contact_active", False)),
                              clearance_avoidance_steering_active=bool(driver.steps > 10 and result.get("near_object") is not None and side != 0 and impact == 0),
                              baseline_steer=self.last_steer, minimum_clearance_changed=False,
                              minimum_clearance_reason="frozen_baseline", clearance_recovery_active=None)
            else:
                result.update(clearance_projection_active=result.get("baseline_projection_active"),
                              clearance_avoidance_steering_active=result.get("avoidance_active"),
                              clearance_recovery_active=result.get("recovery_active"))
            result["clearance_projection_available"] = result.get("projected_obstacle_x") is not None
            return result

    def factory(**kwargs):
        holder["environment"] = Observer(create_training_environment(**kwargs))
        return holder["environment"]

    try:
        result = run_episode(mode=arm, track_id=track, seed=seed, agent=MeasuredAgent(), max_decisions=1200,
                             plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True, environment_factory=factory)
    finally:
        raw_log.close()
    result.update(**original.observer_measurements(holder.get("environment")), raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  versions=dict(numpy=np.__version__, cv2=getattr(cv2, "__version__"), gymnasium=gymnasium.__version__, torch=torch.__version__,
                                Box2D=Box2D.__version__, pygame=pygame.version.ver, python=sys.version), runtime_module_paths=modules)
    save(output, result)


def run(destination, candidate, manifest=None, study="koi-minimum-clearance-ab-v1"):
    from scripts.analyze_koi_minimum_clearance_ab import ANALYSIS_SPEC
    destination, candidate = Path(destination).resolve(), Path(candidate).resolve()
    manifest = Path(manifest).resolve() if manifest else candidate.with_suffix(".manifest.json")
    evidence = consumed_evidence()
    baseline_files = frozen_baseline_files()
    receipt = json.loads(manifest.read_text())
    assert receipt["candidate"].startswith("koi-minimum-clearance")
    assert receipt["candidate_zip_sha256"] == sha(candidate)
    assert receipt["baseline_source_sha256"] == {n: hashlib.sha256(d).hexdigest() for n, d in baseline_files.items()}
    destination.mkdir(exist_ok=False)
    rows = scheduled_slots()
    current = None
    error = None
    protocol = None
    ledger_path = destination / "reset-ledger.jsonl"
    try:
        baseline = destination / "crossing-projection-source-reconstruction.zip"
        with zipfile.ZipFile(baseline, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(baseline_files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, data)
        assert sha(baseline) == "a4b35c56659555f5e60a372261df722628060e4afcd73c88ed9e9688cdb82bb8"
        sources = [SNAPSHOT / "env_wrapper.py", SNAPSHOT / "damage.py"]
        for directory in ("training", "core", "haic_agent"):
            sources.extend(sorted((SNAPSHOT / directory).rglob("*.py")))
        for path in sources:
            blob = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:snapshots/haic-local-20260930/workspace/{path.relative_to(SNAPSHOT)}"], cwd=ROOT, check=True, capture_output=True).stdout
            assert hashlib.sha256(blob).hexdigest() == sha(path)
        helpers = [Path(__file__), ROOT / "scripts/analyze_koi_minimum_clearance_ab.py", ROOT / "scripts/evaluate_koi_adaptive_ab.py",
                   ROOT / "scripts/analyze_koi_adaptive_ab.py", ROOT / "scripts/package_koi_adaptive_avoidance.py"]
        package_source = ROOT / "scripts/package_koi_minimum_clearance.py"
        if package_source.exists():
            helpers.append(package_source)
        copies = []
        for path in [*helpers, *sources]:
            relative = Path("source/environment") / path.relative_to(SNAPSHOT) if path in sources else Path("source/helpers") / path.name
            copied = destination / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(path.read_bytes())
            copies.append(dict(file=str(relative), source=str(path), sha256=sha(copied)))
        models = {ARMS[0]: baseline, ARMS[1]: candidate}
        members = {}
        for arm, path in models.items():
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                assert len(set(names)) == len(names)
                assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names)
                members[arm] = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in names}
        assert members[ARMS[1]] == {r["path"]: r["sha256"] for r in receipt["files"]}
        assert set(members[ARMS[1]]) == set(baseline_files) | {"haic_agent/minimum_clearance_runtime.py"}
        assert all(members[ARMS[1]][n] == hashlib.sha256(d).hexdigest() for n, d in baseline_files.items() if n != "agent.py")
        (destination / "candidate.zip").write_bytes(candidate.read_bytes())
        (destination / "candidate.manifest.json").write_bytes(manifest.read_bytes())
        copies.extend(dict(file=name, source=str(path), sha256=sha(destination / name))
                      for name, path in (("candidate.zip", candidate), ("candidate.manifest.json", manifest)))
        forecast = resource_forecast(evidence)
        measurement = resource_measurements(destination)
        check_resources(measurement, forecast, 48)
        runtime = subprocess.run([str(PYTHON), "-I", "-c", "import sys,json,numpy,cv2,torch,gymnasium,Box2D,pygame;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,cv2=cv2.__version__,torch=torch.__version__,gymnasium=gymnasium.__version__,Box2D=Box2D.__version__,pygame=pygame.version.ver)))"], capture_output=True, text=True, env=ENV, check=True)
        versions = json.loads(runtime.stdout)
        assert versions["torch"].startswith("2.1.") and "+cpu" in versions["torch"]
        consumed_audit = ROOT / "experiments/koi-minimum-clearance-consumed-audit-v1.json"
        if consumed_audit.exists():
            evidence["source_sha256"][str(consumed_audit)] = sha(consumed_audit)
        protocol = dict(study=study, scope="explicit consumed TRAIN reuse; no fresh/protected/official action", source_commit=SOURCE_COMMIT,
                        baseline="frozen_baseline_files source reconstruction, not missing original ZIP restoration",
                        model_hashes={a: sha(p) for a, p in models.items()}, model_source_sha256=members, candidate_manifest_sha256=sha(manifest),
                        cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in CELLS],
                        excludes=[49300, 49301, 49302, 49303, 51300, 51301, 51302, 51303], protected_cells_allowed=False,
                        episodes=48, max_decisions=1200, python=str(PYTHON), frame_skip=4, warmup_ticks=50, raw_fps=50,
                        runtime_versions=versions, environment_source_sha256={str(p): sha(p) for p in sources},
                        helper_source_sha256={str(p): sha(p) for p in helpers}, source_copies=copies,
                        operator_sha256=sha(__file__), analyzer_sha256=sha(ROOT / "scripts/analyze_koi_minimum_clearance_ab.py"),
                        analysis_spec=ANALYSIS_SPEC, schedule=rows, resource_forecast=forecast, resource_measurement=measurement,
                        consumed_reuse_evidence=evidence, telemetry_passive_not_agent_input=True,
                        fresh_claim_registry_modified=False, official_action=False)
        registry = ROOT / "experiments/train-seed-claims"
        descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            check_claims(registry)
            save(destination / "protocol.json", protocol)
        finally:
            os.close(descriptor)
        started = time.monotonic()
        with ledger_path.open("x", buffering=1) as ledger, tempfile.TemporaryDirectory(prefix="koi-clearance-ab-", dir="/tmp/kilo") as temporary:
            directories = {}
            for arm, path in models.items():
                directory = Path(temporary) / arm
                directory.mkdir()
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(directory)
                directories[arm] = directory
            pairs = {}
            for current in rows:
                arm, track, seed = current["mode"], current["track_id"], current["seed"]
                check_resources(resource_measurements(destination), forecast, sum(r["status"] == "unrun" for r in rows))
                remaining = forecast["wall_budget_s"] - (time.monotonic() - started)
                assert remaining > 0
                for group in ("environment_source_sha256", "helper_source_sha256"):
                    assert all(sha(p) == h for p, h in protocol[group].items())
                assert all(sha(p) == h for p, h in evidence["source_sha256"].items())
                assert sha(models[arm]) == protocol["model_hashes"][arm]
                assert sha(manifest) == protocol["candidate_manifest_sha256"]
                descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    check_claims(registry)
                    ledger.write(json.dumps(dict(status="reset_intent", arm=arm, track=track, seed=seed, time=time.time())) + "\n")
                    current["status"] = "reset_intent"
                finally:
                    os.close(descriptor)
                output = destination / f"{track}-{seed}-{arm}.json"
                command = f"import sys;sys.path.insert(0,{str(ROOT)!r});from scripts.evaluate_koi_minimum_clearance_ab import worker;worker({arm!r},{track},{seed},{str(output)!r})"
                original.run_child([str(PYTHON), "-I", "-c", command], directories[arm], output, min(forecast["child_timeout_s"], remaining))
                record = json.loads(output.read_text())
                compact = {k: record[k] for k in ("mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage", "collisions", "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes")}
                compact.update(status="completed", file=output.name, sha256=sha(output))
                current.update(compact)
                ledger.write(json.dumps(compact) + "\n")
                print(json.dumps(compact), flush=True)
                assert not record["error"] and not record["invalid_actions"]
                assert record["retire_reason"] not in {"act_timeout", "reset_timeout", "agent_reset_error", "agent_setup_error", "evaluation_error", "invalid_action", "unknown"}
                assert record["versions"] == versions and record["damage_telemetry_valid"] and record["collision_telemetry_valid"]
                previous = pairs.pop((track, seed), None)
                if previous is not None:
                    assert previous["geometry_sha256"] == record["geometry_sha256"]
                    assert previous["initial_observation_sha256"] == record["initial_observation_sha256"] and previous["initial_state"] == record["initial_state"]
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--study", default="koi-minimum-clearance-ab-v1")
    args = parser.parse_args()
    run(args.output, args.candidate, args.manifest, args.study)


if __name__ == "__main__":
    main()
