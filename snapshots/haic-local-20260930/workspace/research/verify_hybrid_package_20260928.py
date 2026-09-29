"""Verify packaged runtime in a clean directory and replay one tune cell."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "artifacts/haic-research-v2/score-confirm-hybrid-20260928/submission-obstacle-hybrid.zip"
CHILD = r'''
import ctypes
from ctypes import wintypes
import json
import sys
import time
import numpy as np
import agent

class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t)]

started = time.monotonic()
driver = agent.Agent()
init_s = time.monotonic() - started
obs = np.zeros((4, 84, 84), dtype=np.float32)
started = time.monotonic(); driver.reset(obs); reset_s = time.monotonic() - started
started = time.monotonic(); action = driver.act(obs); act_s = time.monotonic() - started
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
psapi = ctypes.WinDLL('psapi', use_last_error=True)
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
counter = Counters(); counter.cb = ctypes.sizeof(Counters)
if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
    raise RuntimeError('RSS measurement failed')
sys.path.append(sys.argv[1])
from training.evaluate_closed_loop import run_episode
row = run_episode(mode='package_smoke', track_id=2, seed=132, agent=driver,
                  max_decisions=2000, plan_budget_seconds=4.5)
print(json.dumps({'agent_file': agent.__file__, 'init_s': init_s, 'reset_s': reset_s,
    'act_s': act_s, 'rss_bytes': int(counter.WorkingSetSize),
    'finite_action': bool(np.isfinite(action).all() and action.shape == (3,)),
    'episode': {k: row[k] for k in ('completed', 'lapTimeMs', 'collisions',
               'damage', 'invalid_actions', 'act_max_ms')}}))
'''


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(destination)
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(ROOT)], cwd=destination,
            text=True, capture_output=True, timeout=60, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr + completed.stdout)
        result = json.loads(completed.stdout.splitlines()[-1])
        if Path(result["agent_file"]).resolve() != destination / "agent.py":
            raise RuntimeError("smoke imported an agent outside the archive")
    if not (result["init_s"] < 10 and result["reset_s"] < 5 and result["act_s"] < 5):
        raise RuntimeError("runtime deadline exceeded")
    if result["rss_bytes"] > 1_024 * 1024 * 1024 or not result["finite_action"]:
        raise RuntimeError("memory or action contract violated")
    if result["episode"]["lapTimeMs"] != 26000 or not result["episode"]["completed"]:
        raise RuntimeError("packaged policy differs from frozen tune behavior")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
