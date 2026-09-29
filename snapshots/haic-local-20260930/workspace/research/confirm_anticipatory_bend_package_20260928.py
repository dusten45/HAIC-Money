"""Compare two immutable extracted ZIPs on reserved local confirmation cells."""

import ast
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plans/2026-09-28-anticipatory-bend-package-confirmation.md"
RUN = ROOT / "runs/haic-research-v2/anticipatory-bend-package-confirmation-20260928"
PACKAGES = {
    "control": ROOT / "artifacts/haic-research-v2/full-road-guard-candidate-20260928/submission-full-road-guard.zip",
    "candidate": ROOT / "artifacts/haic-research-v2/anticipatory-bend-candidate-20260928/submission-anticipatory-bend.zip",
}
EXPECTED = {
    "control": "5c55670ab5aef4bf64115909792650a5e76bc3576a64f3aaeb3780efabc18e40",
    "candidate": "ce28ab744956bc4898d102c41ff6184ee4ac5a819a8559a656a6aedda4008f66",
}
PLAN_SHA = "3c5ad5469f7992b02b5d753b915314e21946d2d806eeef4b7e5a8d7f7a3502d8"
CELLS = tuple((track, seed) for track in (1, 2, 3) for seed in (308, 309, 310, 311))
BANNED_IMPORTS = {"ctypes", "importlib", "multiprocessing", "os", "pathlib", "resource", "shutil", "signal", "socket", "subprocess", "sys"}
BANNED_CALLS = {"compile", "eval", "exec", "__import__"}
RESULT_KEYS = ("track_id", "seed", "completed", "lapTimeMs", "progress", "collisions", "damage", "retire_reason", "error", "invalid_actions", "steps", "act_p50_ms", "act_p95_ms", "act_max_ms", "wall_time_s")

CHILD = r'''
import json
import sys
import time
import resource
from pathlib import Path

root = Path(sys.argv[1])
track = int(sys.argv[2])
seed = int(sys.argv[3])
started = time.monotonic()
import agent
driver = agent.Agent()
import_s = time.monotonic() - started
if "haic_agent" not in sys.modules or not str(Path(sys.modules["haic_agent"].__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("packaged haic_agent was not imported")
if not str(Path(agent.__file__).resolve()).startswith(str(Path.cwd())):
    raise RuntimeError("packaged agent was not imported")
sys.path.append(str(root))
from training.evaluate_closed_loop import run_episode
raw = run_episode(mode="packaged_confirmation", track_id=track, seed=seed,
                  agent=driver, max_decisions=2000, plan_budget_seconds=4.5,
                  fail_on_invalid_action=True)
keys = json.loads(sys.argv[4])
row = {key: raw[key] for key in keys}
row["preview_count"] = getattr(driver._controller, "preview_count", None)
row["import_create_s"] = import_s
row["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
print(json.dumps(row), flush=True)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def static_check(path):
    total = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if "agent.py" not in names or len(names) != len(set(names)) or len(names) > 1000:
            raise RuntimeError("invalid ZIP root or file count")
        for info in archive.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts or "\\" in name or info.is_dir():
                raise RuntimeError("invalid ZIP member path")
            if info.file_size > 500_000_000 or info.file_size / max(info.compress_size, 1) > 100:
                raise RuntimeError("ZIP member exceeds official size/ratio limit")
            if Path(name).suffix.lower() in {".com", ".dll", ".dylib", ".exe", ".msi", ".scr", ".so"}:
                raise RuntimeError("forbidden executable member")
            total += info.file_size
            if name.endswith(".py"):
                tree = ast.parse(archive.read(name).decode("utf-8"), filename=name)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        if any(alias.name.split(".")[0] in BANNED_IMPORTS for alias in node.names):
                            raise RuntimeError(f"forbidden import in {name}")
                    elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in BANNED_IMPORTS:
                        raise RuntimeError(f"forbidden import in {name}")
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                        raise RuntimeError(f"forbidden dynamic call in {name}")
        if total > 2_000_000_000 or path.stat().st_size > 500_000_000:
            raise RuntimeError("ZIP exceeds official total size limit")
        return {"files": len(names), "uncompressed_bytes": total, "archive_bytes": path.stat().st_size}


def summary(rows):
    finished = sorted(row["lapTimeMs"] for row in rows if row["completed"])
    incomplete = [row["progress"] for row in rows if not row["completed"]]
    return {
        "completed": len(finished), "denominator": len(rows),
        "median_finished_lap_ms": statistics.median(finished) if finished else None,
        "mean_incomplete_progress": statistics.mean(incomplete) if incomplete else None,
        "p90_finished_lap_ms": finished[min(len(finished) - 1, int(0.9 * (len(finished) - 1)))] if finished else None,
        "collisions": sum(row["collisions"] or 0 for row in rows),
        "damage": sum(row["damage"] or 0 for row in rows),
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "preview_count": sum(row["preview_count"] or 0 for row in rows),
        "max_import_create_s": max(row["import_create_s"] for row in rows),
        "max_peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        "max_act_p95_ms": max(row["act_p95_ms"] for row in rows),
        "max_act_ms": max(row["act_max_ms"] for row in rows),
    }


def main():
    if sha(PLAN) != PLAN_SHA:
        raise RuntimeError("approved plan changed")
    for arm, path in PACKAGES.items():
        if sha(path) != EXPECTED[arm]:
            raise RuntimeError(f"{arm} package changed")
    checks = {arm: static_check(path) for arm, path in PACKAGES.items()}
    RUN.mkdir(parents=True, exist_ok=False)
    sources = ("training/evaluate_closed_loop.py", "training/env_factory.py", "env_wrapper.py", "core/vendor/car_racing.py", "research/confirm_anticipatory_bend_package_20260928.py")
    source_hashes = {name: sha(ROOT / name) for name in sources}
    manifest = {
        "run_id": RUN.name, "split": "confirmation", "approved_plan": str(PLAN.relative_to(ROOT)),
        "approved_plan_sha256": PLAN_SHA, "cells": [{"track_id": t, "seed": s} for t, s in CELLS],
        "max_decisions": 2000, "packages": {arm: {"path": str(path.relative_to(ROOT)), "sha256": EXPECTED[arm], **checks[arm]} for arm, path in PACKAGES.items()},
        "source_sha256": source_hashes, "created_unix": time.time(),
    }
    (RUN / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = {"control": [], "candidate": []}
    with tempfile.TemporaryDirectory() as directory:
        extracted = {}
        for arm, path in PACKAGES.items():
            target = Path(directory) / arm
            target.mkdir()
            with zipfile.ZipFile(path) as archive:
                archive.extractall(target)
            extracted[arm] = target
        with (RUN / "events.jsonl").open("x", encoding="utf-8") as events:
            for index, (track, seed) in enumerate(CELLS):
                order = ("control", "candidate") if index % 2 == 0 else ("candidate", "control")
                for arm in order:
                    completed = subprocess.run(
                        [sys.executable, "-c", CHILD, str(ROOT), str(track), str(seed), json.dumps(RESULT_KEYS)],
                        cwd=extracted[arm], text=True, capture_output=True, timeout=120, check=False,
                    )
                    if completed.returncode:
                        events.write(json.dumps({"type": "infrastructure_error", "arm": arm, "track_id": track, "seed": seed, "stderr": completed.stderr[-2000:]}) + "\n")
                        events.flush()
                        raise RuntimeError(f"{arm} {track}/{seed}: {completed.stderr[-2000:]}")
                    row = json.loads(completed.stdout.strip().splitlines()[-1])
                    rows[arm].append(row)
                    events.write(json.dumps({"type": "episode", "arm": arm, **row}) + "\n")
                    events.flush()
                    print(arm, track, seed, row["completed"], row["lapTimeMs"], row["progress"], row["preview_count"], flush=True)
    drift = {name: sha(ROOT / name) for name in sources} != source_hashes or any(sha(path) != EXPECTED[arm] for arm, path in PACKAGES.items())
    sums = {arm: summary(values) for arm, values in rows.items()}
    control = sums["control"]
    candidate = sums["candidate"]
    candidate_only_failures = [list(cell) for cell, c, b in zip(CELLS, rows["candidate"], rows["control"]) if not c["completed"] and b["completed"]]
    decision = "REJECT" if candidate["completed"] < control["completed"] or candidate_only_failures else "ADVANCE" if candidate["completed"] > control["completed"] or (candidate["median_finished_lap_ms"] is not None and candidate["median_finished_lap_ms"] < control["median_finished_lap_ms"]) else "REVISE"
    report = {
        "run_id": RUN.name, "decision": decision, "infrastructure_valid": not drift,
        "candidate_only_failures": candidate_only_failures, "summaries": sums,
        "gates": {
            "rule_compliance": "PASS" if not drift and all(s["invalid_actions"] == 0 and s["max_import_create_s"] <= 10 and s["max_peak_rss_bytes"] <= 1024 * 1024 * 1024 and s["max_act_ms"] <= 5000 for s in sums.values()) else "FAIL",
            "mechanism_activation": "PASS" if candidate["preview_count"] > 0 else "FAIL",
            "competitive_or_product_outcome": "PASS" if decision == "ADVANCE" else "FAIL",
        },
    }
    (RUN / "integration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
