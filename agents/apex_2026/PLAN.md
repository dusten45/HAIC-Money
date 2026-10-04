# Apex cloud continuation implementation plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` for the independent experiment lanes and review each completed change.

**Goal:** Improve the standalone camera agent toward four 10–13 second official laps and comparable completion and pace on extra geometries.

**Architecture:** Preserve the root agent, official simulator, and historical candidates. Test clear-road propulsion and obstacle safety separately before combining measured improvements in a new standalone candidate. Bind fresh results to source, parameters, environment, and action hashes.

**Tech stack:** Python 3.11.16, NumPy 1.26.0 inference; the existing pinned official simulator and evaluator.

**Spec:** `agents/apex_2026/DESIGN.md` and the user's cloud continuation instruction.

## Global constraints

- Work from `14bb967` on `codex/apex-2026-independent-agent`; commit and push coherent validated checkpoints.
- Root `agent.py`, `core/`, `env_wrapper.py`, and `damage.py` stay unchanged.
- Inference receives only four 84×84 grayscale frames; no seed lookup, simulator access, root imports, or runtime downloads.
- Measure the official finish crossing after warmup; DNFs remain DNFs.
- Required cells: `(1,516237)`, `(2,644062)`, `(3,1007)`, `(4,18800)`.
- Original maximum is 13 seconds. New prospective profiles follow three distinct completed development rejections per change: 13 → 15 → 18 seconds. Benchmark screens do not advance this count.
- A profile needs all required cells within its limit, at least 15/16 extra cells within it, and at least 3/4 on each track. Runtime and completion limits never relax.
- Open no holdout simulation before final source and parameters are frozen. Preserve old receipts; describe all new measurements as new runs.

## Review focus

- Low-speed turns should accelerate when lateral demand leaves grip available.
- Hazard routing must retain necessary braking and bounded speed.
- Mode changes must synchronize steering memory and avoid sudden unsafe commands.
- Missing/invalid camera data must produce finite bounded recovery actions and reset cleanly.
- Receipt integrity and full development coverage must govern prospective relaxation.

## Tasks

### 1. Reproduce and preserve the starting checkpoint

- [x] Read the six requested context/design/result files and inspect Python/dependencies.
- [x] Independently recheck the official repository and local physics hashes.
- [x] Run the 56 existing Apex tests and fresh hybrid required laps.
- [x] Run the other existing required-lap baselines and full existing test suite.
- [x] Save compact fresh receipts and provenance notes; commit and push this checkpoint.

### 2. Independent implementation and probes

**Files:** new `pace_agent.py`, new `safety_agent.py`, focused tests in `agents/apex_2026/tests/`; preserve `hybrid_agent.py`.

**Interface:** each standalone file exposes `Agent(**parameters)`, `reset(observation=None)`, and `act(observation) -> float32[3]`.

- [x] Write and run a failing low-speed turn/near-slip test before replacing the blanket throttle cap with a lateral-demand gate in `pace_agent.py`.
- [x] Write and run a failing hazard braking/target test before protecting guided pedals in `safety_agent.py`.
- [x] Investigate camera speed calibration and steering/physics independently; implement a correction only when measurements justify it.
- [x] Run bounded benchmark screens; freeze each tested source and parameters while its simulation runs. Corridor V1's operational error is preserved separately from its three valid screens.
- [x] Test and independently review completed source changes, then commit and push them with honest measured results. All guarded integrations are reviewed and measured; none qualifies for adoption.

### 3. Development selection and protocol

**Files:** selected new standalone candidate, focused tests, `evaluate.py` and its tests only if an identified protocol defect needs correction; `results/cloud-20261004/` for compact receipts.

- [x] Test that a mandatory-only screen cannot count as a completed development rejection; correct the identified evaluator mismatch after current runs finish.
- [x] Evaluate seven distinct prospective candidates on all 20 development cells. Maintain the ordered receipt chain and stop unsuccessful parameter directions after repeated clear regressions.
- [x] Apply 15/18 second profiles only after the required newly completed rejections; report the 13 second verdict separately. R1–R3 used 13, R4–R6 used 15, and R7 used 18. Every trial failed both its profile and the original goal.
- [x] Compare completion and pace over all cells; commit and push the checkpoint. No qualifying candidate exists. Preview V4 remains the balanced reference; guarded preview is frozen only as a reviewed research snapshot and is slower with more development contacts.

### 4. Final freeze, verification, and report

- [x] Freeze the research source, parameters, development receipt, and environment hashes in a reviewable manifest; commit and push `b43ef98` before holdout starts.
- [x] Only then run the separate holdout, exact cold required repeats, package checks, and extracted ZIP driving for the frozen artifact. Holdout extras finish 15/16, but no finish meets 13 or 18 seconds; holdout is now consumed.
- [x] Independently review evidence and verify protected source hashes, exact repeat traces, archive contents and freeze chronology.
- [x] Update README, benchmark continuation evidence, and concise CONTEXT; preserve them in the final report commit.
- [x] Record all four lap times, extra/holdout completion and pace, runtime/package limits, original and relaxed verdicts, and remaining gaps in `results/cloud-20261004/FINAL.md`. The requested performance goal remains unmet; no failed candidate is adopted.

## Active high-speed continuation

The preceding completed checklist is the preserved earlier research snapshot.
The user restarted implementation toward four early-10-second laps, minimizing
curve slowdown; no current source meets that goal. The subsequently imposed
04:00 KST cutoff was explicitly cancelled; no clock deadline is active.

- [x] Diagnose true speed, steering, rear grip and HUD cropping on fresh evidence.
- [x] Preserve each frozen rejected screen and source lineage; commit and push.
- [ ] Improve camera path tracking, smooth constrained routes and observe slip.
- [ ] Verify any improved source on mandatory and additional development cells.
- [ ] Continue until the requested goal is reached or weekly Codex quota is
      exhausted; preserve honest source-bound results throughout.

No new holdout opens before a later final source/parameter freeze. The current
old holdout is consumed; mandatory screens do not advance fallback rejection
counts. Fast mode cannot be changed by the tools exposed to this session.
