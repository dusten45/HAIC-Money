"""Frozen consumed-TRAIN A/B with passive decision and 50 Hz world telemetry."""

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
from typing import Any
import traceback
import zipfile

from scripts.package_koi_adaptive_avoidance import ROOT, SOURCE_COMMIT, frozen_baseline_files


SNAPSHOT = Path("/tmp/kilo/koi-baseline-c4e224d/snapshots/haic-local-20260930/workspace")
PYTHON = Path("/tmp/kilo/haic-cpu21/bin/python")
CANDIDATE = ROOT / "submissions/koi-adaptive-avoidance-v1.zip"
CANDIDATE_SHA = "5f7057a432074d2e7215f5d2aeb863aafc4d557d9ba93c9fb3d34ee2b46e679c"
SEEDS = (38300, 38301, 38302, 38303, 50300, 50301, 50302, 50303)
CELLS = [(track, seed) for seed in SEEDS for track in (1, 2, 3)]
ENV = {
    **os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
    "PYGAME_HIDE_SUPPORT_PROMPT": "1", "PYTHONDONTWRITEBYTECODE": "1",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def scheduled_slots():
    slots = []
    for track, seed in CELLS:
        arms = ["crossing_projection", "adaptive_v1"]
        if (track + seed) % 2 == 0:
            arms.reverse()
        slots.extend(dict(mode=arm, track_id=track, seed=seed, status="unrun") for arm in arms)
    return slots


def observer_measurements(observer):
    """Construction/reset/agent-reset errors can leave telemetry uninitialized."""
    return dict(
        catalog=getattr(observer, "catalog", None),
        geometry_sha256=getattr(observer, "geometry_sha256", None),
        initial_state=getattr(observer, "initial", None),
        initial_observation_sha256=getattr(observer, "initial_observation_sha256", None),
        raw_ticks=getattr(observer, "raw_count", 0),
        simulation_start=getattr(observer, "initial", {}).get("t"),
        simulation_end=getattr(observer, "last", {}).get("post", {}).get("t"),
    )


def run_child(command, directory, output, timeout):
    """Retain timeout/nonzero-exit diagnostics without deleting partial telemetry."""
    started = time.monotonic()
    try:
        child = subprocess.run(command, cwd=directory, env=ENV,
                               capture_output=True, text=True, timeout=timeout)
        stdout, stderr = child.stdout, child.stderr
        failure = None if child.returncode == 0 else f"child_exit_{child.returncode}"
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        failure = "child_timeout"
    for suffix, text in ((".stdout.txt", stdout), (".stderr.txt", stderr)):
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        Path(output).with_suffix(suffix).write_text(text)
    receipt = dict(status="completed" if failure is None else "operator_error",
                   error=failure, wall_time_s=time.monotonic() - started)
    save(Path(output).with_suffix(".process.json"), receipt)
    if failure:
        raise RuntimeError(failure)


def worker(arm, track, seed, output):
    """Import the package first; evaluator state is never supplied to the Agent."""
    import resource
    import numpy as np
    protocol = json.loads(Path(output).parent.joinpath("protocol.json").read_text())
    assert (track, seed) in CELLS and arm in protocol["model_hashes"]
    assert sha(__file__) == protocol["operator_sha256"]
    assert all(sha(Path.cwd() / name) == expected
               for name, expected in protocol["model_source_sha256"][arm].items())
    sys.path.insert(0, str(Path.cwd()))
    agent: Any = import_module("agent")
    model = agent.Agent()
    runtime_modules = {
        name: str(module.__file__) for name, module in list(sys.modules.items())
        if (name == "agent" or name.startswith("haic_agent.")) and getattr(module, "__file__", None)
    }
    assert all(Path(path).resolve().is_relative_to(Path.cwd()) for path in runtime_modules.values())
    import haic_agent
    haic_agent.__path__.append(str(SNAPSHOT / "haic_agent"))
    sys.path.insert(0, str(SNAPSHOT))
    import cv2
    import torch
    import gymnasium
    import Box2D
    import pygame
    from training.env_factory import create_training_environment
    run_episode: Any = import_module("training.evaluate_closed_loop").run_episode
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
            self.step_number = 0
            self.last = {}
            self.raw_count = 0
            raw = environment.unwrapped
            original_step = raw.step

            def observed_step(action):
                result = original_step(action)
                if self.active:
                    state = self.state()
                    state.update(step=self.step_number, collision=bool(result[4].get("collision", False)))
                    raw_log.write(json.dumps(state, allow_nan=False) + "\n")
                    self.raw_count += 1
                return result

            raw.step = observed_step

        @property
        def unwrapped(self):
            return self.inner.unwrapped

        def reset(self):
            result = self.inner.reset()
            raw = self.unwrapped
            self.track = np.asarray(raw.track, dtype=np.float64)
            xy = self.track[:, 2:4]
            lengths = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
            self.stations = np.r_[0.0, np.cumsum(lengths)]
            self.road_edges = np.roll(xy, -1, axis=0) - xy
            self.road_lengths = np.linalg.norm(self.road_edges, axis=1)
            self.obstacles = []
            for i, body in enumerate(raw.obstacles):
                position = np.asarray(tuple(body.position), dtype=np.float64)
                anchor = int(np.argmin(np.sum((xy - position)**2, axis=1)))
                beta = float(self.track[anchor, 1])
                self.obstacles.append(dict(
                    id=i, x=float(position[0]), y=float(position[1]),
                    radius=float(body.fixtures[0].shape.radius), anchor_index=anchor,
                    station=float(self.stations[anchor]), tangent=[-float(np.sin(beta)), float(np.cos(beta))],
                ))
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
            starts = np.concatenate(polygons)
            ends = np.concatenate([np.roll(p, -1, axis=0) for p in polygons])
            edge_skins = np.concatenate([np.full(len(p), r) for p, r in zip(polygons, skins)])
            edges = ends - starts
            offset = self.centers[:, None, :] - starts[None]
            amount = np.clip(np.sum(offset * edges[None], axis=2) / np.maximum(np.sum(edges**2, axis=1), 1e-12), 0, 1)
            distance = np.linalg.norm(offset - amount[:, :, None] * edges[None], axis=2)
            clearance = np.min(distance - edge_skins[None], axis=1) - self.radii
            for polygon in polygons:
                edge = np.roll(polygon, -1, axis=0) - polygon
                q = self.centers[:, None] - polygon[None]
                cross = edge[None, :, 0] * q[:, :, 1] - edge[None, :, 1] * q[:, :, 0]
                inside = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
                clearance[inside] = -self.radii[inside]
            projections = np.sum((starts[None] - self.centers[:, None]) * self.normals[:, None], axis=2)
            rear = np.min(projections - edge_skins[None], axis=1)
            front = np.max(projections + edge_skins[None], axis=1)
            touched = set()
            for contact in raw.world.contacts:
                if not contact.touching:
                    continue
                a, b = contact.fixtureA.body, contact.fixtureB.body
                for i, obstacle in enumerate(raw.obstacles):
                    if (a == obstacle and any(b == body for body in bodies)) or (b == obstacle and any(a == body for body in bodies)):
                        touched.add(i)
            position = np.asarray(tuple(hull.position))
            road_amount = np.clip(np.sum((position - self.track[:, 2:4]) * self.road_edges, axis=1)
                                  / np.maximum(self.road_lengths**2, 1e-12), 0, 1)
            road_points = self.track[:, 2:4] + road_amount[:, None] * self.road_edges
            road_index = int(np.argmin(np.sum((road_points - position)**2, axis=1)))
            tangent = self.road_edges[road_index] / self.road_lengths[road_index]
            beta = float(np.arctan2(-tangent[0], tangent[1]))
            lateral = float(np.dot(position - road_points[road_index], [np.cos(beta), np.sin(beta)]))
            heading = float((hull.angle - beta + np.pi) % (2 * np.pi) - np.pi)
            return dict(
                t=float(raw.t), x=float(position[0]), y=float(position[1]), yaw=float(hull.angle),
                speed=float(np.hypot(*hull.linearVelocity)),
                road_index=road_index,
                station=float(self.stations[road_index] + road_amount[road_index] * self.road_lengths[road_index]),
                lateral=lateral, heading_error=heading,
                rear=rear.tolist(), front=front.tolist(), clearance=clearance.tolist(),
                contacts=sorted(touched), wheel_road_contacts=[len(w.tiles) for w in raw.car.wheels],
            )

        def step(self, action):
            self.step_number += 1
            before = self.state()
            result = self.inner.step(action)
            self.last = dict(
                pre=before, post=self.state(),
                simulator_terminated=bool(result[4].get("simulator_terminated", False)),
                wrapper_off_track=bool(result[4].get("wrapper_off_track", False)),
                wrapper_crashed=bool(result[4].get("wrapper_crashed", False)),
            )
            return result

        def close(self):
            self.inner.close()

    class MeasuredAgent:
        def reset(self, observation):
            return model.reset(observation)

        def act(self, observation):
            return model.act(observation)

        def last_step_diagnostics(self):
            result = model.last_step_diagnostics()
            driver = model.driver
            if hasattr(driver, "driver"):
                driver = driver.driver
            result.update(
                evaluation_only=holder["environment"].last,
                actual_obstacle_side=driver.base._obstacle_side,
                actual_impact_left=driver.impact_left,
            )
            return result

    def factory(**kwargs):
        holder["environment"] = Observer(create_training_environment(**kwargs))
        return holder["environment"]

    result = run_episode(
        mode=arm, track_id=track, seed=seed, agent=MeasuredAgent(), max_decisions=1200,
        plan_budget_seconds=4.5, capture_trace=True, fail_on_invalid_action=True,
        environment_factory=factory,
    )
    raw_log.close()
    observer = holder.get("environment")
    result.update(
        **observer_measurements(observer),
        raw_trace_file=raw_path.name, raw_trace_sha256=sha(raw_path),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        versions=dict(numpy=np.__version__, cv2=getattr(cv2, "__version__"), gymnasium=gymnasium.__version__,
                      torch=torch.__version__, Box2D=Box2D.__version__, pygame=pygame.version.ver,
                      python=sys.version),
        runtime_module_paths=runtime_modules,
    )
    save(output, result)
    print(json.dumps({k: result[k] for k in (
        "mode", "track_id", "seed", "completed", "lapTimeMs", "progress", "damage", "collisions",
        "retire_reason", "error", "invalid_actions", "steps", "raw_ticks", "peak_rss_bytes",
    )}), flush=True)


def run(destination):
    from scripts.analyze_koi_adaptive_ab import ANALYSIS_SPEC

    destination = Path(destination).resolve()
    destination.mkdir(exist_ok=False)
    for name, path in (("operator-source.py", Path(__file__)),
                       ("analyzer-source.py", ROOT / "scripts/analyze_koi_adaptive_ab.py")):
        (destination / name).write_bytes(path.read_bytes())
    assert sha(CANDIDATE) == CANDIDATE_SHA
    baseline = destination / "crossing-projection-source-reconstruction.zip"
    with zipfile.ZipFile(baseline, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(frozen_baseline_files().items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    # Pin the complete local import trees, including transitive evaluator imports.
    sources = [SNAPSHOT / "env_wrapper.py", SNAPSHOT / "damage.py"]
    for directory in ("training", "core", "haic_agent"):
        sources.extend(sorted((SNAPSHOT / directory).rglob("*.py")))
    for path in sources:
        blob = subprocess.run(["git", "show", f"{SOURCE_COMMIT}:snapshots/haic-local-20260930/workspace/{path.relative_to(SNAPSHOT)}"],
                              cwd=ROOT, check=True, capture_output=True).stdout
        assert hashlib.sha256(blob).hexdigest() == sha(path)
    models = {"crossing_projection": baseline, "adaptive_v1": CANDIDATE}
    hashes = {name: sha(path) for name, path in models.items()}
    model_sources = {}
    for arm, path in models.items():
        with zipfile.ZipFile(path) as archive:
            model_sources[arm] = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    resource_measurement = dict(
        cgroup=str(Path("/proc/self/cgroup").read_text()).strip(),
        memory_max=Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        memory_current=int(Path("/sys/fs/cgroup/memory.current").read_text()),
        memory_events=Path("/sys/fs/cgroup/memory.events").read_text(),
        disk_free=shutil.disk_usage(destination).free,
        historical_peak_rss_bytes=284217344,
    )
    assert resource_measurement["cgroup"] == "0::/"
    runtime = subprocess.run(
        [str(PYTHON), "-I", "-c", "import sys,json,numpy,cv2,torch,gymnasium,Box2D,pygame;print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,cv2=cv2.__version__,torch=torch.__version__,gymnasium=gymnasium.__version__,Box2D=Box2D.__version__,pygame=pygame.version.ver)))"],
        capture_output=True, text=True, env=ENV, check=True,
    )
    protocol: dict[str, Any] = dict(
        study="koi-adaptive-ab-v1", scope="explicitly consumed TRAIN, no freshness claim",
        source_commit=SOURCE_COMMIT, model_hashes=hashes, model_source_sha256=model_sources,
        baseline="pinned source reconstruction, not original crossing ZIP restoration",
        cells=[dict(track_id=t, geometry_seed=s, partition="TRAIN", obstacles=True) for t, s in CELLS],
        excludes=[49300,49301,49302,49303,51300,51301,51302,51303],
        episodes=48, max_decisions=1200, child_timeout_s=90,
        python=str(PYTHON), environment_source_sha256={str(p): sha(p) for p in sources},
        runtime_versions=json.loads(runtime.stdout),
        operator_sha256=sha(__file__), frame_skip=4, warmup_ticks=50, raw_fps=50,
        analysis_spec=ANALYSIS_SPEC,
        analyzer_sha256=sha(ROOT / "scripts/analyze_koi_adaptive_ab.py"),
        frozen_source_files={name: sha(destination / name)
                             for name in ("operator-source.py", "analyzer-source.py")},
        schedule=scheduled_slots(),
        detection_match="unique projected circle center within 4 pixels; ambiguous/unmatched retained",
        passage="fixed road-tangent front overlap to all hull/wheel fixtures rear clearance",
        common_entry="continuous local centerline station 25 units before frozen obstacle anchor",
        gates=dict(lost_finishes=0, per_cell_damage_nonincreasing=True,
                   per_cell_collision_decisions_nonincreasing=True, no_new_clean_obstacle_hit=True,
                   matched_segment_time_improvement=True, actual_clean_higher_than_44_geometry_seeds=2),
        corrections="one full reevaluation only after explicit trace-supported defect",
        resource_forecast=dict(peak_per_child_bytes=1073741824, total_disk_bytes=536870912,
                               serial_children=1, gpu=False, wall_budget_s=1200,
                               memory_reserve_bytes=1073741824, per_episode_disk_bytes=8388608),
        resource_measurement=resource_measurement,
        historical_resource_evidence_sha256=sha(SNAPSHOT.parents[2] / "releases/arrival-speed-20260930/evidence/batch_audit.json"),
        exposure_evidence=dict(
            primary_38300=sha(SNAPSHOT.parents[2] / "releases/arrival-speed-20260930/evidence/report.json"),
            declaration_50300=sha(SNAPSHOT / "docs/plans/active/contact-continuity-20260930-manifest.json"),
        ),
        fresh_claim_registry_modified=False, official_action=False,
    )
    registry = ROOT / "experiments/train-seed-claims"
    assert protocol["frozen_source_files"]["operator-source.py"] == protocol["operator_sha256"]
    assert protocol["frozen_source_files"]["analyzer-source.py"] == protocol["analyzer_sha256"]
    descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        assert not any((registry / f"seed-{seed}.json").exists() for seed in SEEDS)
        save(destination / "protocol.json", protocol)
    finally:
        os.close(descriptor)
    rows = scheduled_slots()
    current_slot = None
    operator_error = None
    started = time.monotonic()
    try:
        with (destination / "reset-ledger.jsonl").open("x", buffering=1) as ledger, \
                tempfile.TemporaryDirectory(prefix="koi-ab-", dir="/tmp/kilo") as temporary:
            directories = {}
            for arm, path in models.items():
                directory = Path(temporary) / arm
                directory.mkdir()
                with zipfile.ZipFile(path) as archive:
                    archive.extractall(directory)
                directories[arm] = directory
            for track, seed in CELLS:
                arms = list(models) if (track + seed) % 2 else list(reversed(models))
                pair = []
                for arm in arms:
                    current_slot = next(row for row in rows if row["mode"] == arm and row["track_id"] == track and row["seed"] == seed)
                    memory_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                    memory_current = int(Path("/sys/fs/cgroup/memory.current").read_text())
                    assert memory_max == "max" or int(memory_max) - memory_current > 2 * 1024**3
                    assert shutil.disk_usage(destination).free > sum(row["status"] == "unrun" for row in rows) * 8388608 + 134217728
                    remaining_seconds = 1200 - (time.monotonic() - started)
                    assert remaining_seconds > 0
                    assert sha(models[arm]) == hashes[arm]
                    assert sha(__file__) == protocol["operator_sha256"]
                    assert all(sha(path) == protocol["environment_source_sha256"][str(path)] for path in sources)
                    descriptor = os.open(registry, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX)
                        assert not any((registry / f"seed-{s}.json").exists() for s in SEEDS)
                        ledger.write(json.dumps(dict(status="reset_intent", arm=arm, track=track, seed=seed, time=time.time())) + "\n")
                        current_slot["status"] = "reset_intent"
                    finally:
                        os.close(descriptor)
                    output = destination / f"{track}-{seed}-{arm}.json"
                    command = f"import sys;sys.path.insert(0,{str(ROOT)!r});from scripts.evaluate_koi_adaptive_ab import worker;worker({arm!r},{track},{seed},{str(output)!r})"
                    run_child([str(PYTHON), "-I", "-c", command], directories[arm], output, min(90, remaining_seconds))
                    record = json.loads(output.read_text())
                    compact = {key: record[key] for key in (
                        "mode","track_id","seed","completed","lapTimeMs","progress","damage","collisions",
                        "retire_reason","error","invalid_actions","steps","raw_ticks","peak_rss_bytes",
                    )}
                    compact.update(status="completed", file=output.name, sha256=sha(output))
                    current_slot.update(compact)
                    pair.append(record)
                    ledger.write(json.dumps(compact) + "\n")
                    print(json.dumps(compact), flush=True)
                    assert not record["error"] and not record["invalid_actions"]
                    assert record["retire_reason"] not in {
                        "act_timeout", "reset_timeout", "agent_reset_error", "evaluation_error", "invalid_action",
                    }
                    if "versions" in record:
                        assert record["versions"] == protocol["runtime_versions"]
                assert pair[0]["geometry_sha256"] == pair[1]["geometry_sha256"]
                assert pair[0]["initial_observation_sha256"] == pair[1]["initial_observation_sha256"]
                if "initial_state" in pair[0]:
                    assert pair[0]["initial_state"] == pair[1]["initial_state"]
                keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw", "progress")
                assert [[r[k] for k in keys] for r in pair[0]["decision_trace"][:10]] == [[r[k] for k in keys] for r in pair[1]["decision_trace"][:10]]
    except Exception as error:
        operator_error = f"{type(error).__name__}: {error}"
        if current_slot is not None and current_slot["status"] != "completed":
            current_slot.update(status="operator_error", error=operator_error)
        save(destination / "operator-failure.json", dict(error=operator_error, traceback=traceback.format_exc()))
        raise
    finally:
        save(destination / "episode-report.json", dict(protocol_sha256=sha(destination / "protocol.json"),
                                                      rows=rows, operator_error=operator_error))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
