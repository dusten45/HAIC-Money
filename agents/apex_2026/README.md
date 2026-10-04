# Apex 2026 independent agent research

This folder preserves the independent-agent work requested on 2026-10-04.
The active root agent and official physics are unchanged. This is a research
checkpoint: the requested 10–13 second performance and strong generalization
have **not** been achieved. No candidate is certified as competition-winning.

## Preserved candidates

- `agent.py`: small metric camera prototype; incomplete mandatory coverage.
- `robust_agent.py`: standalone conservative camera controller with tested
  68/.65/.22 defaults; all mandatory tracks finish, but pace misses the target.
- `hybrid_agent.py`: road-only metric steering with independent robust obstacle
  routing. Its metric controller cannot keep a competing obstacle pass latch.
- `mpc_agent.py`: vectorized short-horizon camera trajectory planner, version 2.
  Passing-lane rollouts, rotated hull clearance, and bounded slow motion fixed
  three earlier artificial stops; track 4 still stalls. Experimental only.

All candidates infer from images and internal memory. They require NumPy and
contain no map seed lookup, simulator import, root-agent import, or checkpoint.
The evaluator and packager are development tools and must not be submitted.

## Historical checkpoint mandatory laps

These are the earlier checkpoint measurements, not the new Linux cloud runs.
Times are actual simulated finish times excluding warmup. DNF is not a lap.

| Track / seed | Existing agent | Robust | Hybrid | MPC v2 |
|---|---:|---:|---:|---:|
| 1 / 516237 | 23.98 s | 22.30 s | 19.62 s | 20.28 s |
| 2 / 644062 | 31.00 s | 27.88 s | 26.68 s | 30.36 s |
| 3 / 1007 | 27.84 s | 26.72 s | 22.74 s | 31.68 s |
| 4 / 18800 | 24.82 s | 24.30 s | 21.78 s | DNF |

Robust and hybrid mandatory runs had zero obstacle contacts. Hybrid reduces
the four-lap total by 15.63%, but its additional development coverage is only
10/16 finishes, versus 13/16 for the existing and robust agents. Hybrid has
not been promoted. All hybrid development retirements occurred before its
700-step budget, so none needs a longer timeout resolution. Holdout geometries
were unopened at that checkpoint. Source-bound summaries are in `benchmark.json`; large camera
traces and immutable original receipts remain in the ignored local directory
`.haic-artifacts/independent-racer-20261004/`.
The robust timings bind its frozen version-2 source with explicit 68/.65/.22
parameters. The preserved robust file also has optional disabled corner
controls; it still needs exact cold artifact validation before promotion.

The fixed prospective pace profiles are 13, then 15, then 18 seconds after
three and six distinct development rejections. Those relaxations never change
past results or completion/runtime/safety rules. Every candidate above still
fails the original target and the latest 18-second pace profile.

## Cloud continuation from 14bb967

The user resumed development on `codex/apex-2026-independent-agent`. New Python
3.11.16 Linux measurements, source-bound receipts, rejected variants and
reversible patches are in `results/cloud-20261004/`. The fresh hybrid required
laps are **19.62 / 26.20 / 22.90 / 21.82 seconds**, with **10/16** extra finishes;
historical MPC completion results do not reproduce on this platform.

Seven complete prospective development candidates failed: safety, pace, recovery,
preview, envelope, corridor and guarded preview. R1–R3 used 13 seconds, R4–R6
used 15, and R7 used 18 under the predefined rules. Each tested source is
preserved; none is silently substituted for an earlier receipt. Both the
original 10–13 second goal and the relaxed 18 second profile remain unmet.
No candidate is promoted.

`guarded_preview_agent.py` is the final reviewed research snapshot, with `{}`
parameters, frozen and pushed in `b43ef98` before opening holdout. Its required
laps are **21.92 / 27.64 / 26.14 / 23.74 seconds**, all with zero contacts.
Development extras finish **12/16** (per-track **2/4, 4/4, 3/4, 3/4**).
Preview V4 remains the balanced development reference; the guarded revision
is slower on common finishes and has more development contacts.

Holdout is now consumed: **15/16** extra finishes in **20.00–25.18 seconds**,
with no subsequent tuning or selection change. Cold source and extracted ZIP
required runs each reproduce all four action hashes and outcomes. See the
[final report](results/cloud-20261004/FINAL.md), [immutable freeze](results/cloud-20261004/final/freeze.json)
and [summary](results/cloud-20261004/final/summary.json).

Fresh current Apex and new behavior tests pass **213/213**. The existing full
baseline has 1170 passes, 10 skips and 15 old dependency /
provenance failures; see `results/cloud-20261004/BASELINE.md`. Cloud inference
uses NumPy only. The official repository source was rechecked byte for byte;
website-only announcements were inaccessible from this environment.

## Reproduce the frozen research snapshot on Linux

From the repository root using the Python 3.11 virtual environment:

```bash
env MPLCONFIGDIR=/tmp/haic-mpl XDG_CACHE_HOME=/tmp/haic-cache SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m agents.apex_2026.evaluate --source agents/apex_2026/guarded_preview_agent.py --suite mandatory --role benchmark --output .haic-artifacts/apex-new-required-repeat.json
.venv/bin/python -m pytest -q tests/test_apex_evaluation.py tests/test_apex_package.py tests/test_apex_vision.py tests/test_apex_robust.py tests/test_apex_hybrid.py tests/test_apex_mpc.py agents/apex_2026/tests
.venv/bin/python -m agents.apex_2026.package --source agents/apex_2026/guarded_preview_agent.py --output .haic-artifacts/apex-new-research.zip
```

Use a new output path for every study or package. Performance failures are in
the receipt verdict; a zero process exit code means the run had no operational
error. These local cloud checks do not establish official container certification.
Previously opened holdout seeds must be treated as consumed development data.

## Historical Windows reproduction

From the repository root with its existing Python environment:

```powershell
.\.venv\Scripts\python.exe -m agents.apex_2026.evaluate --source agents/apex_2026/hybrid_agent.py --suite mandatory --output .haic-artifacts/apex-new-mandatory.json
.\.venv\Scripts\python.exe -m pytest -q tests/test_apex_evaluation.py tests/test_apex_package.py tests/test_apex_vision.py tests/test_apex_robust.py tests/test_apex_hybrid.py tests/test_apex_mpc.py
```

`--suite development` repeats the four mandatory cells and adds sixteen
development cells. A holdout set that has not been evaluated may be opened only after
freezing the final source and parameters. See `DESIGN.md` for the full protocol
and `RULES.md` for the checked official contract.
