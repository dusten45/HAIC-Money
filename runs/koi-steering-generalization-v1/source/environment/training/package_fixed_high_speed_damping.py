"""Deliver the exact user-requested damping controller with packaged replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from training.confirm_stable_package import check_archive


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = "fixed-high-speed-steering-train-20260929"
SOURCES = ("haic_agent/fixed_high_speed_runtime.py", "haic_agent/corridor_agent.py",
           "haic_agent/pixel_features.py")
ENTRY = '''"""Fixed high target speed with temporal steering correction."""
from haic_agent.fixed_high_speed_runtime import FixedHighSpeedAgent


class Agent:
    def __init__(self):
        self._controller = FixedHighSpeedAgent("damping")

    def reset(self, observation):
        self._controller.reset(observation)

    def act(self, observation):
        return self._controller.act(observation)
'''
CHILD = r'''
import json, resource, sys, time
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); track=int(sys.argv[2]); seed=int(sys.argv[3])
started=time.monotonic()
import agent
driver=agent.Agent()
import_create_s=time.monotonic()-started
assert Path(agent.__file__).resolve().parent == Path.cwd()
started=time.monotonic(); driver.reset(np.zeros((4,84,84),dtype=np.float32))
reset_s=time.monotonic()-started
# Evaluator dependencies only: runtime modules were already imported from ZIP.
import haic_agent
for name in ('fixed_high_speed_runtime','corridor_agent','pixel_features'):
    assert Path(sys.modules['haic_agent.'+name].__file__).resolve().parent == Path.cwd()/'haic_agent'
haic_agent.__path__.append(str(root/'haic_agent'))
sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode
row=run_episode(mode='packaged_fixed_high_speed_damping',track_id=track,seed=seed,
                agent=driver,max_decisions=1200,plan_budget_seconds=4.5,
                capture_trace=True,fail_on_invalid_action=True)
row.update(import_create_s=import_create_s,reset_s=reset_s,
           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
print(json.dumps(row),flush=True)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(output):
    original = json.loads((ROOT / "runs/haic-research-v2" / REFERENCE_RUN / "run_manifest.json").read_text())
    files = {"agent.py": ENTRY.encode(), "haic_agent/__init__.py": b'"""Pixel controller runtime."""\n'}
    for name in SOURCES:
        data = (ROOT / name).read_bytes()
        if digest(data) != original["source_hashes"][name]:
            raise ValueError(f"Source no longer matches tested damping: {name}")
        files[name] = data
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    archive_hash = digest(output.read_bytes())
    checks = check_archive(output)
    reference_root = ROOT / "artifacts/haic-research-v2" / REFERENCE_RUN
    reference = json.loads((reference_root / "report.json").read_text())
    originals = {(r["track"], r["seed"]): r for r in reference["rows"] if r["arm"] == "damping"}
    results = []
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(output) as archive:
            archive.extractall(directory)
        for track in (1, 2, 3):
            for seed in (38200, 38201):
                completed = subprocess.run([sys.executable, "-c", CHILD, str(ROOT), str(track), str(seed)],
                                           cwd=directory, capture_output=True, text=True, timeout=90)
                if completed.returncode:
                    raise RuntimeError(completed.stderr[-3000:])
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                source = originals[track, seed]
                source_bytes = (reference_root / source["evidence"]).read_bytes()
                if digest(source_bytes) != source["sha256"]:
                    raise ValueError("Original episode hash mismatch")
                before = json.loads(source_bytes)
                keys = ("steer", "gas", "brake", "car_x", "car_y", "car_yaw")
                old_trace, new_trace = before["decision_trace"], row["decision_trace"]
                same = len(old_trace) == len(new_trace) and all(
                    abs(a[k] - b[k]) < 1e-6 for a, b in zip(old_trace, new_trace) for k in keys)
                same = same and all(before[k] == row[k] for k in ("completed", "lapTimeMs", "collisions", "damage", "retire_reason"))
                row["same_as_original_damping"] = same
                path = output.parent / f"packaged-{track}-{seed}.json"
                with path.open("x") as stream:
                    json.dump(row, stream)
                if (not same or row["error"] or row["invalid_actions"] or row["import_create_s"] >= 10
                        or row["reset_s"] >= 5 or row["act_max_ms"] >= 5000 or row["peak_rss_bytes"] >= 1024**3):
                    raise RuntimeError(f"Package replay contract failed: {track}:{seed}")
                results.append({k: row[k] for k in ("track_id", "seed", "completed", "lapTimeMs", "progress",
                                                   "retire_reason", "collisions", "damage", "import_create_s",
                                                   "reset_s", "act_p50_ms", "act_p95_ms", "act_max_ms",
                                                   "peak_rss_bytes", "same_as_original_damping")})
                print(json.dumps(results[-1]), flush=True)
    if digest(output.read_bytes()) != archive_hash:
        raise RuntimeError("ZIP changed during replay")
    report = dict(archive=output.name, archive_sha256=archive_hash, checks=checks,
                  files={name: digest(data) for name, data in files.items()},
                  episodes=results, completed=sum(r["completed"] for r in results), denominator=6,
                  package_identity_verified=True, consumed_train_replay=True,
                  user_accepts_observed_excursions=True, official_submission=False)
    with (output.parent / "package_report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    # A readable entry beside the ZIP; dependencies are all contained inside ZIP.
    (output.parent / "agent.py").write_text(ENTRY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
